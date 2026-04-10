#!/usr/bin/env python3
"""Convert SIFT1M dataset from fvecs/ivecs format to Divergence binary format.

Downloads SIFT1M from corpus-texmex.irisa.fr if not already present.

Usage:
    python3 scripts/convert_sift1m.py [--n-vectors 1000000] [--k 100] [--src /path/to/sift]

Writes to: data/sift_1000k/ (or data/sift_{n/1000}k/ if --n-vectors is set)
    vectors.bin   - f32 flat array, shape (n, 128)
    queries.bin   - f32 flat array, shape (nq, 128)
    gt.bin        - u32 flat array, shape (nq, k)
    meta.txt      - n, nq, dim, k on separate lines
"""

import argparse
import os
import struct
import sys
import tarfile
import time
import urllib.request

import numpy as np


SIFT1M_URL = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
SIFT1M_URL_HTTP = "http://corpus-texmex.irisa.fr/sift.tar.gz"


def load_fvecs(path: str) -> np.ndarray:
    """Load fvecs file: [dim (i32)] [f32 * dim] repeated."""
    data = np.fromfile(path, dtype=np.float32)
    dim = int(data[0])  # first 4 bytes = dimension (as float reinterpret of i32)
    # Re-read with proper handling: each record is (1 + dim) floats
    raw = np.fromfile(path, dtype=np.uint8)
    # Read dimension from first 4 bytes as int32
    dim = struct.unpack('<i', raw[:4].tobytes())[0]
    record_bytes = 4 + dim * 4  # 4 bytes dim + dim*4 bytes data
    n = len(raw) // record_bytes
    assert len(raw) == n * record_bytes, f"File size mismatch: {len(raw)} not divisible by {record_bytes}"
    # Reshape and extract float data (skip first 4 bytes of each record)
    records = raw.reshape(n, record_bytes)
    vectors = np.frombuffer(records[:, 4:].tobytes(), dtype=np.float32).reshape(n, dim)
    return vectors


def load_ivecs(path: str) -> np.ndarray:
    """Load ivecs file: [dim (i32)] [i32 * dim] repeated."""
    raw = np.fromfile(path, dtype=np.uint8)
    dim = struct.unpack('<i', raw[:4].tobytes())[0]
    record_bytes = 4 + dim * 4
    n = len(raw) // record_bytes
    assert len(raw) == n * record_bytes, f"File size mismatch: {len(raw)} not divisible by {record_bytes}"
    records = raw.reshape(n, record_bytes)
    indices = np.frombuffer(records[:, 4:].tobytes(), dtype=np.int32).reshape(n, dim)
    return indices


def download_sift1m(dst_dir: str):
    """Download and extract SIFT1M dataset."""
    tar_path = os.path.join(dst_dir, "sift.tar.gz")
    if os.path.exists(os.path.join(dst_dir, "sift")):
        print("SIFT1M already extracted, skipping download")
        return

    os.makedirs(dst_dir, exist_ok=True)

    if not os.path.exists(tar_path):
        print(f"Downloading SIFT1M from {SIFT1M_URL_HTTP} ...")
        t0 = time.time()
        try:
            urllib.request.urlretrieve(SIFT1M_URL_HTTP, tar_path)
        except Exception as e:
            print(f"HTTP download failed ({e}), trying FTP...")
            urllib.request.urlretrieve(SIFT1M_URL, tar_path)
        print(f"  Downloaded in {time.time()-t0:.1f}s")

    print("Extracting sift.tar.gz ...")
    t0 = time.time()
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(dst_dir)
    print(f"  Extracted in {time.time()-t0:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-vectors", type=int, default=1_000_000,
                        help="Number of base vectors to use (default 1000000)")
    parser.add_argument("--k", type=int, default=100,
                        help="Number of ground-truth neighbors (default 100)")
    parser.add_argument("--src", type=str, default=None,
                        help="Source directory containing sift/ subfolder")
    parser.add_argument("--dst", type=str, default=None,
                        help="Output directory (default: data/sift_{n/1000}k/)")
    parser.add_argument("--metric", type=str, default="l2", choices=["l2", "cosine"],
                        help="Distance metric. SIFT is natively L2. Use 'cosine' to L2-normalize.")
    args = parser.parse_args()

    n = args.n_vectors
    k = args.k

    # Source directory
    if args.src:
        src_dir = args.src
    else:
        src_dir = "data"
        download_sift1m(src_dir)

    sift_dir = os.path.join(src_dir, "sift")
    if not os.path.isdir(sift_dir):
        print(f"ERROR: {sift_dir} not found. Provide --src pointing to parent of sift/ dir.")
        sys.exit(1)

    suffix = f"_{args.metric}" if args.metric != "l2" else ""
    dst = args.dst or f"data/sift_{n // 1000}k{suffix}"
    os.makedirs(dst, exist_ok=True)

    # Load base vectors
    print(f"Loading base vectors from {sift_dir}/sift_base.fvecs ...")
    t0 = time.time()
    vectors = load_fvecs(os.path.join(sift_dir, "sift_base.fvecs"))
    dim = vectors.shape[1]
    print(f"  {vectors.shape[0]} vectors, dim={dim}, {time.time()-t0:.1f}s")

    if n < vectors.shape[0]:
        vectors = vectors[:n]
        print(f"  Subset to {n} vectors")
    else:
        n = vectors.shape[0]

    # Load queries
    print(f"Loading queries from {sift_dir}/sift_query.fvecs ...")
    queries = load_fvecs(os.path.join(sift_dir, "sift_query.fvecs"))
    nq = queries.shape[0]
    print(f"  {nq} queries, dim={queries.shape[1]}")

    # Load ground truth
    print(f"Loading ground truth from {sift_dir}/sift_groundtruth.ivecs ...")
    gt_raw = load_ivecs(os.path.join(sift_dir, "sift_groundtruth.ivecs"))
    gt_k = gt_raw.shape[1]
    print(f"  {gt_raw.shape[0]} queries, {gt_k} neighbors each")

    # If subsetting vectors, recompute GT for the subset
    if n < 1_000_000:
        print(f"Recomputing ground truth for {n}-vector subset (L2) ...")
        t0 = time.time()
        gt = np.empty((nq, k), dtype=np.uint32)
        for i in range(nq):
            dists = np.sum((vectors - queries[i]) ** 2, axis=1)
            topk = np.argpartition(dists, k)[:k]
            order = np.argsort(dists[topk])
            gt[i] = topk[order].astype(np.uint32)
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{nq} queries done")
        print(f"  GT recomputed in {time.time()-t0:.1f}s")
    else:
        # Use provided GT, truncate to k
        gt = gt_raw[:, :k].astype(np.uint32)
        if gt_k < k:
            print(f"WARNING: GT has only {gt_k} neighbors, padding with zeros to {k}")
            gt_padded = np.zeros((nq, k), dtype=np.uint32)
            gt_padded[:, :gt_k] = gt
            gt = gt_padded

    # Optional: L2-normalize for cosine metric
    if args.metric == "cosine":
        print("L2-normalizing vectors and queries for cosine metric ...")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors /= norms
        qnorms = np.linalg.norm(queries, axis=1, keepdims=True)
        qnorms[qnorms == 0] = 1.0
        queries /= qnorms
        # Recompute GT with dot product (cosine on normalized = dot)
        print(f"Recomputing ground truth for cosine metric ...")
        t0 = time.time()
        gt = np.empty((nq, k), dtype=np.uint32)
        batch_size = 100
        for i in range(0, nq, batch_size):
            end = min(i + batch_size, nq)
            sims = queries[i:end] @ vectors.T
            topk_idx = np.argpartition(-sims, k, axis=1)[:, :k]
            for j in range(end - i):
                order = np.argsort(-sims[j, topk_idx[j]])
                gt[i + j] = topk_idx[j][order].astype(np.uint32)
        print(f"  Cosine GT computed in {time.time()-t0:.1f}s")

    # Write binary files
    vec_path = os.path.join(dst, "vectors.bin")
    print(f"Writing {vec_path} ({vectors.nbytes / 1e6:.1f} MB) ...")
    vectors.astype(np.float32).tofile(vec_path)

    q_path = os.path.join(dst, "queries.bin")
    print(f"Writing {q_path} ({queries.nbytes / 1e6:.1f} MB) ...")
    queries.astype(np.float32).tofile(q_path)

    gt_path = os.path.join(dst, "gt.bin")
    print(f"Writing {gt_path} ({gt.nbytes / 1e6:.1f} MB) ...")
    gt.tofile(gt_path)

    meta_path = os.path.join(dst, "meta.txt")
    metric_str = args.metric
    with open(meta_path, "w") as f:
        f.write(f"{n}\n{nq}\n{dim}\n{k}\n")
    print(f"Written {meta_path}")

    # Sanity check
    print("\n--- Sanity Check ---")
    print(f"n={n}, nq={nq}, dim={dim}, k={k}, metric={metric_str}")
    print(f"Vector norms (first 5): {np.linalg.norm(vectors[:5], axis=1)}")
    print(f"Query norms (first 5): {np.linalg.norm(queries[:5], axis=1)}")
    print(f"GT[0] first 5 neighbors: {gt[0, :5]}")
    if metric_str == "l2":
        dists0 = np.sum((vectors[gt[0, :5]] - queries[0]) ** 2, axis=1)
        print(f"GT[0] first 5 L2 distances: {dists0}")
    else:
        sims0 = queries[0] @ vectors[gt[0, :5]].T
        print(f"GT[0] first 5 cosine sims: {sims0}")
    print(f"\nDone. Files in {dst}/")


if __name__ == "__main__":
    main()
