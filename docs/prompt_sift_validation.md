# Prompt: SIFT1M Cross-Dataset Validation

## Goal

Download SIFT1M dataset, convert to Divergence format, build NSW index + heavy_edge layout ONCE (persistent), and run Phase 1 cache-aware beam search sweep. This validates that our Phase 1 findings (co-resident caching is implicit, pivoting is redundant given prefetch) generalize beyond Cohere 100K.

**SIFT1M**: 1M vectors, dim=128, L2 metric, 10K queries, standard ANN benchmark.

## Important Context

- SIFT uses **L2 distance** (not cosine). The codebase supports `MetricType::L2`.
- SIFT is low-dimensional (128d) vs Cohere (768d). This means:
  - Adjacency blocks fit more neighbors (same 4KB pages, smaller vector dimension doesn't affect adjacency — adjacency stores VIDs not vectors)
  - But FP32 vectors are much smaller: 128×4 = 512 bytes vs 768×4 = 3072 bytes
  - More records fit per page in v3 layout (~31 per page for deg=32, independent of dim)
- Ground truth from SIFT1M has 100 neighbors (k=100), same as our Cohere setup
- SIFT1M provides 10,000 queries (vs Cohere's 1,000)

## Step 0: Download and Convert SIFT1M

On EC2:

```bash
cd /mnt/nvme/divergence
python3 scripts/convert_sift1m.py --n-vectors 1000000 --k 100 --src /mnt/nvme/divergence/data --dst /mnt/nvme/divergence/data/sift_1000k
```

This downloads SIFT1M from corpus-texmex.irisa.fr (~160MB compressed), extracts, converts fvecs/ivecs to our binary format (vectors.bin, queries.bin, gt.bin, meta.txt). L2 metric, no normalization.

Verify:
```bash
ls -lh /mnt/nvme/divergence/data/sift_1000k/
# Should see: vectors.bin (~488MB), queries.bin (~4.9MB), gt.bin (~3.8MB), meta.txt
cat /mnt/nvme/divergence/data/sift_1000k/meta.txt
# Should show: 1000000 / 10000 / 128 / 100
```

## Step 1: Add generic dataset loader to disk_search.rs

Currently `load_cohere_dataset` is hardcoded. Add a generic loader that works with any dataset in our binary format:

```rust
/// Load dataset in Divergence binary format (vectors.bin, queries.bin, gt.bin, meta.txt).
/// Returns (vectors_flat, queries_flat, ground_truth, n, nq, dim, k).
fn load_dataset(dataset_dir: &str, max_n: usize) -> Option<(Vec<f32>, Vec<f32>, Vec<Vec<u32>>, usize, usize, usize, usize)> {
    let meta_path = format!("{}/meta.txt", dataset_dir);
    let meta = match std::fs::read_to_string(&meta_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Skipping: cannot read {}: {}", meta_path, e);
            return None;
        }
    };
    let nums: Vec<usize> = meta.lines()
        .filter_map(|l| l.trim().parse().ok())
        .collect();
    assert!(nums.len() >= 4, "meta.txt must have 4 lines: n, nq, dim, k");
    let (n_total, nq, dim, k) = (nums[0], nums[1], nums[2], nums[3]);
    let n = n_total.min(max_n);

    eprintln!("Loading dataset from {} (n={}/{}, nq={}, dim={}, k={}) ...",
              dataset_dir, n, n_total, nq, dim, k);

    // Load vectors
    let vec_bytes = std::fs::read(format!("{}/vectors.bin", dataset_dir)).expect("vectors.bin");
    let all_floats: &[f32] = unsafe {
        std::slice::from_raw_parts(vec_bytes.as_ptr() as *const f32, vec_bytes.len() / 4)
    };
    let vectors: Vec<f32> = all_floats[..n * dim].to_vec();

    // Load queries
    let q_bytes = std::fs::read(format!("{}/queries.bin", dataset_dir)).expect("queries.bin");
    let q_floats: &[f32] = unsafe {
        std::slice::from_raw_parts(q_bytes.as_ptr() as *const f32, q_bytes.len() / 4)
    };
    let queries_flat: Vec<f32> = q_floats[..nq * dim].to_vec();

    // Load ground truth
    let gt_bytes = std::fs::read(format!("{}/gt.bin", dataset_dir)).expect("gt.bin");
    let gt_u32: &[u32] = unsafe {
        std::slice::from_raw_parts(gt_bytes.as_ptr() as *const u32, gt_bytes.len() / 4)
    };
    let mut ground_truth = Vec::with_capacity(nq);
    for qi in 0..nq {
        let row = &gt_u32[qi * k..(qi + 1) * k];
        // If subset: filter GT to only include IDs < n
        let filtered: Vec<u32> = if n < n_total {
            row.iter().copied().filter(|&id| (id as usize) < n).collect()
        } else {
            row.to_vec()
        };
        ground_truth.push(filtered);
    }

    eprintln!("  Loaded: {} vectors, {} queries, dim={}, k={}", n, nq, dim, k);
    Some((vectors, queries_flat, ground_truth, n, nq, dim, k))
}
```

This is the same logic as `load_cohere_dataset` but reads `meta.txt` for parameters instead of hardcoding them. Put it near the existing `load_cohere_dataset` function.

## Step 2: Add SIFT build + sweep tests

Add two new test functions to `crates/engine/tests/disk_search.rs`. They follow the exact same pattern as `exp_veloann_phase1_build` and `exp_veloann_phase1_sweep` but use:
- `SIFT_DIR` env var (default: `data/sift_1000k`)
- `SIFT_N` env var (default: 1_000_000)
- `MetricType::L2` instead of `MetricType::Cosine`
- FP32SimdVectorBank with L2 metric
- Different directory under BENCH_DIR: `veloann_phase1_sift/heavy_edge`
- NSW build with m_max=32, ef_construction=64 (not 200 — 1M vectors at ef_c=200 is too slow)

### exp_sift_phase1_build

```rust
/// Build NSW index + heavy_edge layout for SIFT1M. Run once, reuse for sweeps.
/// Env: SIFT_DIR, SIFT_N, BENCH_DIR (required).
#[test]
#[ignore]
fn exp_sift_phase1_build() {
    let max_n: usize = std::env::var("SIFT_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);

    let dataset_dir = std::env::var("SIFT_DIR").unwrap_or_else(|_| {
        let manifest = env!("CARGO_MANIFEST_DIR");
        format!("{}/../../data/sift_1000k", manifest)
    });

    let (vectors, _queries_flat, _ground_truth, n, _nq, dim, _k) =
        match load_dataset(&dataset_dir, max_n) {
            Some(d) => d,
            None => return,
        };

    let bench_dir = std::env::var("BENCH_DIR").expect("BENCH_DIR required for build");
    let base_dir = std::path::PathBuf::from(&bench_dir).join("veloann_phase1_sift");
    let layout_dir = base_dir.join("heavy_edge");

    // Check if already built
    if layout_dir.join("adj_index.dat").exists() && layout_dir.join("vectors.dat").exists() {
        let meta = IndexMeta::load_from(&layout_dir.join("meta.json")).unwrap();
        if meta.num_vectors as usize == n {
            eprintln!("Already built: SIFT {}K heavy_edge {} pages, skipping rebuild",
                      n / 1000, meta.num_pages.unwrap_or(0));
            return;
        }
    }

    let m_max = 32;
    let ef_construction = 64;  // 1M at ef_c=200 is too slow

    eprintln!("=== BUILD: SIFT {}K, dim={}, m_max={}, ef_c={} ===",
              n / 1000, dim, m_max, ef_construction);

    // Build NSW index
    eprintln!("Building NSW index ...");
    let t0 = std::time::Instant::now();
    let config = NswConfig::new(m_max, ef_construction);
    let builder = NswBuilder::new(config, dim, MetricType::L2, n);
    for (i, v) in vectors.chunks_exact(dim).enumerate() {
        builder.insert(VectorId(i as u32), v);
    }
    let index = builder.build();
    eprintln!("  Index built in {:.1}s", t0.elapsed().as_secs_f64());

    let entry_ids: Vec<u32> = index.entry_set().iter().map(|v| v.0).collect();

    // Write base index (vectors.dat + meta.json + adjacency.dat)
    std::fs::create_dir_all(&base_dir).unwrap();
    let writer_base = IndexWriter::new(&base_dir);
    writer_base.write(
        n as u32, dim, "l2", index.max_degree(), ef_construction,
        &entry_ids, index.vectors_raw(), |vid| index.neighbors(vid),
    ).unwrap();

    // Build heavy_edge layout
    eprintln!("Building heavy_edge layout ...");
    let t0 = std::time::Instant::now();
    let he_reorder = heavy_edge_reorder_graph(n, |vid| index.neighbors(vid));
    eprintln!("  Reorder computed in {:.1}s", t0.elapsed().as_secs_f64());

    std::fs::create_dir_all(&layout_dir).unwrap();
    let w = IndexWriter::new(&layout_dir);
    w.write_v3(
        n as u32, dim, "l2", index.max_degree(), ef_construction,
        &entry_ids, index.vectors_raw(), |vid| index.neighbors(vid),
        &he_reorder, "heavy_edge",
    ).unwrap();
    std::fs::copy(base_dir.join("vectors.dat"), layout_dir.join("vectors.dat")).unwrap();

    let meta = IndexMeta::load_from(&layout_dir.join("meta.json")).unwrap();
    eprintln!("  heavy_edge: {} pages — DONE", meta.num_pages.unwrap_or(0));
}
```

### exp_sift_phase1_sweep

```rust
/// Sweep page_sched_b on pre-built SIFT heavy_edge layout. Fast (~2 min).
/// Env: SIFT_DIR, SIFT_N, BENCH_DIR (required).
#[test]
#[ignore]
fn exp_sift_phase1_sweep() {
    let max_n: usize = std::env::var("SIFT_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);

    let dataset_dir = std::env::var("SIFT_DIR").unwrap_or_else(|_| {
        let manifest = env!("CARGO_MANIFEST_DIR");
        format!("{}/../../data/sift_1000k", manifest)
    });

    let (_vectors, queries_flat, ground_truth, n, nq, dim, k) =
        match load_dataset(&dataset_dir, max_n) {
            Some(d) => d,
            None => return,
        };

    let ef = 200;
    let prefetch_width = 4;
    let num_bench_queries = 200;   // SIFT has 10K queries, use more
    let warmup_queries = 50;
    let cache_pct = 5usize;
    let sched_b_values: &[usize] = &[0, 2, 4, 8];

    let total_queries_needed = num_bench_queries + warmup_queries;
    assert!(nq >= total_queries_needed, "Need {} queries but dataset has {}", total_queries_needed, nq);

    // Load pre-built artifacts
    let bench_dir = std::env::var("BENCH_DIR").expect("BENCH_DIR required");
    let layout_dir = std::path::PathBuf::from(&bench_dir)
        .join("veloann_phase1_sift").join("heavy_edge");

    assert!(
        layout_dir.join("adj_index.dat").exists(),
        "Run exp_sift_phase1_build first! Missing: {}/adj_index.dat",
        layout_dir.display()
    );

    let disk_vectors = load_vectors(&layout_dir.join("vectors.dat"), n, dim).unwrap();
    let meta = IndexMeta::load_from(&layout_dir.join("meta.json")).unwrap();
    let num_pages = meta.num_pages.unwrap_or(0) as usize;
    let adj_index = load_adj_index(&layout_dir.join("adj_index.dat"), n).unwrap();
    let entry_set: Vec<VectorId> = meta.entry_set.iter().map(|&v| VectorId(v)).collect();

    eprintln!(
        "=== SWEEP: SIFT {}K, dim={}, k={}, ef={}, W={}, cache={}%, heavy_edge {} pages ===",
        n / 1000, dim, k, ef, prefetch_width, cache_pct, num_pages
    );

    // NOTE: SIFT uses L2 metric
    let bench_queries: Vec<Vec<f32>> = queries_flat
        .chunks_exact(dim).take(num_bench_queries).map(|c| c.to_vec()).collect();
    let bench_gt: Vec<&Vec<u32>> = ground_truth.iter().take(num_bench_queries).collect();
    let warmup_vecs: Vec<Vec<f32>> = queries_flat
        .chunks_exact(dim).skip(num_bench_queries).take(warmup_queries).map(|c| c.to_vec()).collect();

    let dir_str = layout_dir.to_str().unwrap().to_owned();
    let direct_io = true;

    // --- The sweep loop is identical to Cohere Phase 1, except MetricType::L2 ---
    // Copy the exact pattern from exp_veloann_phase1_sweep (lines 10901-11037+)
    // replacing MetricType::Cosine with MetricType::L2 in FP32SimdVectorBank::new

    if !with_runtime(|rt| {
        rt.block_on(async {
            let fp32_bank = FP32SimdVectorBank::new(&disk_vectors, dim, MetricType::L2);

            eprintln!(
                "\n{:>8} {:>6} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>12}",
                "mode", "sched_b", "recall", "p50ms", "p99ms", "QPS",
                "exp/q", "blk/q", "mis/q", "hit/q", "phy/q", "sched_hits/q"
            );

            for &sched_b in sched_b_values {
                // === COLD mode ===
                // [Copy cold-mode block from exp_veloann_phase1_sweep verbatim]
                // Use disk_graph_search_pipe_v3_pagesched with the same parameters

                // === WARM mode ===
                // [Copy warm-mode block from exp_veloann_phase1_sweep verbatim]
            }
        })
    }) {
        eprintln!("WARNING: no monoio runtime");
    }
}
```

**Copy the cold/warm blocks exactly from `exp_veloann_phase1_sweep`** (lines 10911-11037). The only change is `MetricType::L2` in the `FP32SimdVectorBank::new` call.

## Step 3: Run on EC2

```bash
# SSH to EC2
ssh -i ~/Downloads/ubuntu.pem ubuntu@54.183.93.241

# Setup
cd /mnt/nvme/divergence
source ~/.cargo/env

# 1. Convert SIFT1M (one time, ~2 min)
python3 scripts/convert_sift1m.py --n-vectors 1000000 --k 100 \
    --src /mnt/nvme/divergence/data --dst /mnt/nvme/divergence/data/sift_1000k

# 2. Build index (one time, ~5-10 min at ef_c=64 for 1M)
SIFT_DIR=/mnt/nvme/divergence/data/sift_1000k SIFT_N=1000000 \
  BENCH_DIR=/mnt/nvme/bench \
  cargo test --release -p divergence-engine --test disk_search exp_sift_phase1_build \
  -- --nocapture --ignored

# 3. Sweep (fast, ~2 min — reuses pre-built artifacts)
SIFT_DIR=/mnt/nvme/divergence/data/sift_1000k SIFT_N=1000000 \
  BENCH_DIR=/mnt/nvme/bench \
  cargo test --release -p divergence-engine --test disk_search exp_sift_phase1_sweep \
  -- --nocapture --ignored
```

## Expected Results

### What we expect to see (if Phase 1 findings generalize):

1. **High cold cache hit rate** (>85%) at sched_b=0, from heavy_edge co-resident caching
2. **mis/q drops with sched_b** (20→1-2 at B=4)
3. **p50 unchanged** despite miss reduction (prefetch already masks misses)
4. **Recall stable** across all sched_b values

### What would be interesting/surprising:

- If hit rate is LOWER at sched_b=0 → SIFT's graph structure places neighbors differently
- If p50 DOES improve with sched_b → SIFT 1M has more misses that prefetch can't cover (more data = more cache misses, prefetch window may be insufficient)
- If recall degrades with sched_b → pivoting causes suboptimal beam exploration on L2

### Key differences from Cohere:
- **10× more vectors** (1M vs 100K) → more pages, lower cache hit rate expected
- **L2 metric** (not cosine) → different distance distribution, different graph structure
- **Lower dimension** (128 vs 768) → cheaper compute, IO becomes even more dominant
- **5% cache = 50K pages** vs 5K for Cohere → still small relative to 1M vectors

## Verification Checklist

- [ ] `cargo check -p divergence-engine --tests` compiles
- [ ] Existing tests pass (non-ignored)
- [ ] SIFT data files exist with correct sizes
- [ ] Build creates `veloann_phase1_sift/heavy_edge/{adj_index.dat, adjacency_pages.dat, vectors.dat, meta.json}`
- [ ] Build detects existing artifacts and skips rebuild on re-run
- [ ] Sweep loads pre-built artifacts (doesn't rebuild)
- [ ] Recall is reasonable (>0.90 at ef=200 — may differ from Cohere due to ef_c=64 and L2 metric)
- [ ] Output table matches Cohere Phase 1 format for easy comparison

## Notes

- Do NOT build SAQ codes for SIFT in this step. SAQ is only needed for the two-stage pipeline, and Phase 1 tests the graph/caching behavior which uses FP32 scoring.
- If 1M NSW build at ef_c=64 is still too slow (>15 min), try with SIFT_N=100000 first for a quick validation, then 1M.
- The `load_dataset` function replaces per-dataset loaders. You can optionally refactor existing Cohere tests to use it, but it's not required for this task.
