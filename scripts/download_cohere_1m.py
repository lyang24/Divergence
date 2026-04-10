#!/usr/bin/env python3
"""Download Cohere 1M embeddings from HuggingFace and convert to Divergence binary format.

Usage:
    pip install datasets numpy
    python3 scripts/download_cohere_1m.py [--n-vectors 1000000] [--k 100]

Downloads from: Cohere/wikipedia-22-12-en-embeddings (HuggingFace)
Writes to:      data/cohere_1000k/
    vectors.bin   - f32 flat array, shape (n, 768), L2-normalized
    queries.bin   - f32 flat array, shape (nq, 768), L2-normalized
    gt.bin        - u32 flat array, shape (nq, k)
    meta.txt      - n, nq, dim, k on separate lines
"""

import argparse
import os
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-vectors", type=int, default=1_000_000)
    parser.add_argument("--n-queries", type=int, default=10_000)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--dst", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    n = args.n_vectors
    nq = args.n_queries
    k = args.k
    dst = args.dst or f"data/cohere_{n // 1000}k"
    os.makedirs(dst, exist_ok=True)

    # Check if already converted
    if all(os.path.exists(os.path.join(dst, f)) for f in ["vectors.bin", "queries.bin", "gt.bin", "meta.txt"]):
        print(f"Already exists: {dst}/ — skipping")
        return

    from datasets import load_dataset

    print(f"Loading Cohere wikipedia embeddings from HuggingFace (first {n + nq} rows)...")
    t0 = time.time()
    ds = load_dataset(
        "Cohere/wikipedia-22-12-en-embeddings",
        split=f"train[:{n + nq}]",
        columns=["emb"],
    )
    print(f"  Loaded {len(ds)} rows, {time.time()-t0:.1f}s")

    # Extract embeddings
    print("Extracting embeddings...")
    t0 = time.time()
    all_embs = np.array(ds["emb"], dtype=np.float32)
    dim = all_embs.shape[1]
    print(f"  Shape: {all_embs.shape}, dim={dim}, {time.time()-t0:.1f}s")

    # Split into vectors and queries
    # Use last nq rows as queries (deterministic split)
    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(len(all_embs))
    query_indices = indices[:nq]
    vector_indices = indices[nq:nq + n]

    queries = all_embs[query_indices]
    vectors = all_embs[vector_indices]
    print(f"  Vectors: {vectors.shape}, Queries: {queries.shape}")

    # L2-normalize for cosine metric
    print("L2-normalizing...")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors /= norms
    qnorms = np.linalg.norm(queries, axis=1, keepdims=True)
    qnorms[qnorms == 0] = 1.0
    queries /= qnorms

    # Compute brute-force ground truth
    print(f"Computing brute-force top-{k} (cosine via dot on normalized)...")
    t0 = time.time()
    gt = np.empty((nq, k), dtype=np.uint32)
    batch_size = 100
    for i in range(0, nq, batch_size):
        end = min(i + batch_size, nq)
        sims = queries[i:end] @ vectors.T  # (batch, n)
        topk_idx = np.argpartition(-sims, k, axis=1)[:, :k]
        for j in range(end - i):
            order = np.argsort(-sims[j, topk_idx[j]])
            gt[i + j] = topk_idx[j][order].astype(np.uint32)
        if end % 1000 == 0 or end == nq:
            print(f"  {end}/{nq} queries done, {time.time()-t0:.1f}s")
    print(f"  Ground truth computed, {time.time()-t0:.1f}s")

    # Write binary files
    print(f"Writing to {dst}/")
    vectors.tofile(os.path.join(dst, "vectors.bin"))
    print(f"  vectors.bin: {vectors.nbytes / 1e9:.2f} GB")
    queries.tofile(os.path.join(dst, "queries.bin"))
    print(f"  queries.bin: {queries.nbytes / 1e6:.1f} MB")
    gt.tofile(os.path.join(dst, "gt.bin"))
    print(f"  gt.bin: {gt.nbytes / 1e6:.1f} MB")

    with open(os.path.join(dst, "meta.txt"), "w") as f:
        f.write(f"{n}\n{nq}\n{dim}\n{k}\n")

    # Sanity check
    print(f"\n--- Sanity Check ---")
    print(f"Vector norms (first 5): {np.linalg.norm(vectors[:5], axis=1)}")
    print(f"GT[0] first 5: {gt[0, :5]}")
    sims0 = queries[0] @ vectors[gt[0, :5]].T
    print(f"GT[0] first 5 sims: {sims0}")
    print(f"\nDone. {dst}/ ready.")


if __name__ == "__main__":
    main()
