//! Integration test: build NSW → write to disk → async search → verify results match.
//!
//! These tests require io_uring support (Linux 5.1+, not inside unprivileged containers).
//! They are automatically skipped if io_uring is unavailable.


use std::rc::Rc;

use divergence_core::distance::{
    FP32SimdVectorBank, VectorBank,
};
use divergence_core::{MetricType, VectorId};
use divergence_engine::{
    disk_graph_search, disk_graph_search_pipe, disk_graph_search_pipe_v3,
    build_page_to_vids,
    disk_graph_search_pipe_v3_freeexp,
    disk_graph_search_pipe_v3_pagesched,
    disk_graph_search_pipe_v3_traced,
    AdaEfParams, AdaEfStats, AdaEfTable, estimate_ada_ef,
    AdjacencyPool, IoDriver,
    PerfLevel, SearchPerfContext,
    TraceRecorder,
};
use divergence_index::{NswBuilder, NswConfig};
use divergence_storage::{
    load_vectors, IndexMeta, IndexWriter,
    AdjIndexEntry, bfs_reorder_graph,
    heavy_edge_reorder_graph, load_adj_index, neighbor_run_reorder_graph,
};

use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;

/// Try to build a monoio io_uring runtime. Returns false if io_uring is not
/// available (e.g. unprivileged container, old kernel), and runs the closure
/// on success.
fn with_runtime(
    f: impl FnOnce(&mut monoio::Runtime<monoio::time::TimeDriver<monoio::IoUringDriver>>),
) -> bool {
    match monoio::RuntimeBuilder::<monoio::IoUringDriver>::new()
        .enable_all()
        .build()
    {
        Ok(mut rt) => {
            f(&mut rt);
            true
        }
        Err(_) => false,
    }
}

fn generate_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..dim).map(|_| rng.r#gen::<f32>()).collect())
        .collect()
}

#[test]
fn disk_search_matches_memory() {
    let n = 500;
    let dim = 32;
    let k = 10;
    let ef = 64;
    let m_max = 32;
    let ef_construction = 200;

    // 1. Generate vectors and build NSW in memory
    let vectors = generate_vectors(n, dim, 42);
    let config = NswConfig::new(m_max, ef_construction);
    let builder = NswBuilder::new(config, dim, MetricType::L2, n);
    for (i, v) in vectors.iter().enumerate() {
        builder.insert(VectorId(i as u32), v);
    }
    let index = builder.build();

    // 2. Search in memory for ground truth
    let query: Vec<f32> = generate_vectors(1, dim, 999)[0].clone();
    let memory_results = index.search(&query, k, ef);

    // 3. Write to disk
    let dir = tempfile::tempdir().unwrap();
    let dir_str = dir.path().to_str().unwrap().to_owned();
    let writer = IndexWriter::new(dir.path());
    writer
        .write(
            n as u32,
            dim,
            "l2",
            index.max_degree(),
            ef_construction,
            &index
                .entry_set()
                .iter()
                .map(|v| v.0)
                .collect::<Vec<_>>(),
            index.vectors_raw(),
            |vid| index.neighbors(vid),
        )
        .unwrap();

    // 4. Load meta + vectors for disk search
    let meta = IndexMeta::load_from(&dir.path().join("meta.json")).unwrap();
    let disk_vectors = load_vectors(&dir.path().join("vectors.dat"), n, dim).unwrap();

    let entry_set: Vec<VectorId> = meta.entry_set.iter().map(|&v| VectorId(v)).collect();

    // 5. Run async disk search inside monoio runtime
    if !with_runtime(|rt| {
        let disk_results = rt.block_on(async {
            let io = IoDriver::open(&dir_str, dim, 64, false)
                .await
                .expect("failed to open IO driver");

            let pool = AdjacencyPool::new(64 * 1024);
            let bank = FP32SimdVectorBank::new(&disk_vectors, dim, MetricType::L2);
            let mut perf = SearchPerfContext::default();

            disk_graph_search(
                &query, &entry_set, k, ef, &pool, &io, &bank, &mut perf,
                PerfLevel::CountOnly,
            )
            .await
        });

        // 6. Verify: disk results should match memory results exactly
        assert_eq!(
            disk_results.len(),
            memory_results.len(),
            "result count mismatch"
        );

        for (i, (disk, mem)) in disk_results.iter().zip(memory_results.iter()).enumerate() {
            assert_eq!(
                disk.id, mem.id,
                "VID mismatch at position {}: disk={:?} mem={:?}",
                i, disk.id, mem.id
            );
            assert!(
                (disk.distance - mem.distance).abs() < 1e-6,
                "distance mismatch at position {}: disk={} mem={}",
                i,
                disk.distance,
                mem.distance
            );
        }
    }) {
        eprintln!("SKIPPED: io_uring not available");
    }
}

#[test]
fn io_driver_reads_single_block() {
    let n = 3u32;
    let dim = 4;

    // Write a small adjacency file
    let dir = tempfile::tempdir().unwrap();
    let dir_str = dir.path().to_str().unwrap().to_owned();

    let adj: Vec<Vec<u32>> = vec![vec![1, 2], vec![0, 2], vec![0, 1]];
    let vectors: Vec<f32> = vec![0.0; n as usize * dim];

    let writer = IndexWriter::new(dir.path());
    writer
        .write(n, dim, "l2", 32, 200, &[0], &vectors, |vid| {
            &adj[vid as usize]
        })
        .unwrap();

    // Read back with IoDriver
    if !with_runtime(|rt| {
        rt.block_on(async {
            let io = IoDriver::open(&dir_str, dim, 64, false)
                .await
                .expect("failed to open IO driver");

            for vid in 0..n {
                let buf = io.read_adj_block(vid).await.expect("read failed");
                let neighbors = divergence_storage::decode_adj_block(buf.as_slice());
                assert_eq!(
                    neighbors, adj[vid as usize],
                    "mismatch at vid {}",
                    vid
                );
            }
        });
    }) {
        eprintln!("SKIPPED: io_uring not available");
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Compute recall@k: |approx_ids ∩ exact_ids| / k
fn recall_at_k(approx_ids: &[u32], exact_ids: &[u32]) -> f64 {
    let k = exact_ids.len();
    if k == 0 {
        return 1.0;
    }
    let exact_set: std::collections::HashSet<u32> = exact_ids.iter().copied().collect();
    let hits = approx_ids.iter().filter(|id| exact_set.contains(id)).count();
    hits as f64 / k as f64
}

// ---------------------------------------------------------------------------
// REAL DATA: dataset loading helpers
// ---------------------------------------------------------------------------

/// Load any dataset in Divergence binary format (vectors.bin, queries.bin, gt.bin, meta.txt).
fn load_dataset(
    dir: &str,
    max_vectors: usize,
) -> Option<(Vec<f32>, Vec<f32>, Vec<Vec<u32>>, usize, usize, usize, usize)> {
    use std::fs;
    use std::io::Read as _;

    let meta_path = format!("{}/meta.txt", dir);
    let meta = match fs::read_to_string(&meta_path) {
        Ok(s) => s,
        Err(_) => {
            eprintln!("SKIPPED: dataset not found at {}", dir);
            return None;
        }
    };
    let nums: Vec<usize> = meta.lines().filter_map(|l| l.trim().parse().ok()).collect();
    if nums.len() < 4 {
        eprintln!("SKIPPED: Invalid meta.txt format");
        return None;
    }
    let (n_total, nq, dim, k) = (nums[0], nums[1], nums[2], nums[3]);
    let n = n_total.min(max_vectors);

    let mut vbuf = vec![0u8; n * dim * 4];
    let mut f = fs::File::open(format!("{}/vectors.bin", dir)).ok()?;
    f.read_exact(&mut vbuf).ok()?;
    let vectors: Vec<f32> = vbuf
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let mut qbuf = vec![0u8; nq * dim * 4];
    let mut f = fs::File::open(format!("{}/queries.bin", dir)).ok()?;
    f.read_exact(&mut qbuf).ok()?;
    let queries: Vec<f32> = qbuf
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let mut gbuf = vec![0u8; nq * k * 4];
    let mut f = fs::File::open(format!("{}/gt.bin", dir)).ok()?;
    f.read_exact(&mut gbuf).ok()?;
    let gt_flat: Vec<u32> = gbuf
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let ground_truth: Vec<Vec<u32>> = if n < n_total {
        gt_flat
            .chunks_exact(k)
            .map(|row| row.iter().copied().filter(|&id| (id as usize) < n).collect())
            .collect()
    } else {
        gt_flat.chunks_exact(k).map(|row| row.to_vec()).collect()
    };

    eprintln!("Loaded dataset: {} vectors, {} queries, dim={}, k={}", n, nq, dim, k);
    Some((vectors, queries, ground_truth, n, nq, dim, k))
}

/// Build NSW index with parallel insertion using rayon.
fn build_nsw_parallel(vectors: &[f32], n: usize, dim: usize, metric: MetricType, m_max: usize, ef_construction: usize) -> NswBuilder {
    use rayon::prelude::*;

    eprintln!("Building NSW index (m_max={}, ef_c={}, n={}, parallel) ...", m_max, ef_construction, n);
    let t0 = std::time::Instant::now();
    let config = NswConfig::new(m_max, ef_construction);
    let builder = NswBuilder::new(config, dim, metric, n);

    // Insert first vector sequentially to set entry point
    builder.insert(VectorId(0), &vectors[..dim]);

    // Insert remaining vectors in parallel
    (1..n).into_par_iter().for_each(|i| {
        builder.insert(VectorId(i as u32), &vectors[i * dim..(i + 1) * dim]);
    });

    let index = builder;
    eprintln!("  Index built in {:.1}s", t0.elapsed().as_secs_f64());
    index
}

/// Compute percentile from a sorted slice. p in [0, 100].
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() { return 0.0; }
    let idx = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

// ---------------------------------------------------------------------------
// Stable Benchmark Runner
// ---------------------------------------------------------------------------

#[allow(dead_code)]
struct BenchConfig {
    label: String,
    ef: usize,        // used as max_ef when ada_ef=true
    k: usize,
    prefetch_width: usize,
    stall_limit: u32,  // used as default when ada_ef=false
    drain_budget: u32,
    adj_inflight: usize,
    cache_pct: usize,
    num_queries: usize,
    warmup_queries: usize,
    ada_ef: bool,      // if true, per-query (ef, S, D) from Ada-ef scoring
    /// If true, clear pool before each query (true cold per-query measurement).
    clear_per_query: bool,
}

struct BenchResult {
    recall: f64,
    lat_p50_ms: f64,
    lat_p99_ms: f64,
    qps: f64,
    avg_expansions: f64,
    avg_useful: f64,
    avg_wasted: f64,
    avg_blk_q: f64,
    avg_miss_q: f64,
    avg_hit_q: f64,
    avg_singleflight: f64,
    avg_pf_issued: f64,
    avg_pf_consumed: f64,
    avg_best_at: f64,
    avg_first_topk: f64,
    early_stop_pct: f64,
    waste_ratio: f64,
    hit_rate: f64,
    /// Physical NVMe IO reads per query (miss loads + prefetch loads + bypasses).
    avg_phys_reads_q: f64,
    // Timing breakdown (avg per query, ms)
    avg_io_wait_ms: f64,
    avg_compute_ms: f64,
    avg_dist_ms: f64,
    // Derived: avg ms per cache miss (io_ms / mis_q)
    ms_per_miss: f64,
    // Cache health: total bypasses and evict failures across all queries
    total_bypasses: u64,
    total_evict_fail: u64,
    // Refine stats (two-stage pipeline only)
    avg_refine_count: f64,
    avg_refine_ms: f64,
    /// Total IO requests per query: adj_phy + refine vector reads.
    /// Note: adj reads are 4KB pages, refine reads are dim*4 bytes each.
    avg_total_io_q: f64,
    /// Actual refine IO bytes submitted to kernel (from read_at return, not estimated).
    /// 0 for non-int8 paths (where we don't track bytes yet).
    avg_refine_bytes: f64,
}

async fn run_bench(
    cfg: &BenchConfig,
    entry_set: &[VectorId],
    pool: &Rc<AdjacencyPool>,
    io: &Rc<IoDriver>,
    bank: &dyn VectorBank,
    query_vecs: &[Vec<f32>],
    ground_truth: &[Vec<u32>],
    ada: Option<(&AdaEfStats, &AdaEfTable)>,
    query_scores: &[f64],
) -> BenchResult {
    let nq = cfg.num_queries.min(query_vecs.len());

    // Warmup pass
    for q in query_vecs.iter().take(cfg.warmup_queries) {
        let mut perf = SearchPerfContext::default();
        disk_graph_search_pipe(
            q, entry_set, cfg.k, cfg.ef, cfg.prefetch_width,
            cfg.stall_limit, cfg.drain_budget,
            pool, io, bank, &mut perf, PerfLevel::CountOnly,
        ).await;
    }

    // Per-query Ada-ef tracking (only populated when ada_ef=true)
    struct AdaQueryInfo { score: f64, ef_used: usize, recall: f64, blk: u64 }
    let mut ada_info: Vec<AdaQueryInfo> = Vec::new();

    let mut recalls = Vec::with_capacity(nq);
    let mut latencies_ms = Vec::with_capacity(nq);
    let mut sum_exp = 0u64;
    let mut sum_useful = 0u64;
    let mut sum_wasted = 0u64;
    let mut sum_blk = 0u64;
    let mut sum_miss = 0u64;
    let mut sum_hit = 0u64;
    let mut sum_phys_reads = 0u64;
    let mut sum_sf = 0u64;
    let mut sum_pf_issued = 0u64;
    let mut sum_pf_consumed = 0u64;
    let mut sum_best_at = 0u64;
    let mut sum_first_topk = 0u64;
    let mut early_count = 0u64;
    let mut sum_io_wait_ns = 0u64;
    let mut sum_compute_ns = 0u64;
    let mut sum_dist_ns = 0u64;

    let cache_stats_before = pool.stats();
    let wall_start = std::time::Instant::now();

    for i in 0..nq {
        let q = &query_vecs[i];

        // Determine per-query params: Ada-ef or fixed
        let (ef, sl, db, ada_score) = if cfg.ada_ef {
            if let Some((stats, table)) = ada {
                // Compute seed distances (same as search seeding — pure DRAM)
                let seed_dists: Vec<f32> = entry_set
                    .iter()
                    .map(|&ep| bank.distance(q, ep.0 as usize))
                    .collect();
                // Compute score for diagnostics
                let (mu, sigma) = stats.estimate_fdl_params(q);
                let mut thresholds = [0.0f64; 5];
                for b in 0..5 {
                    thresholds[b] = mu + sigma * divergence_engine::ada_ef::inv_normal_cdf(0.001 * (b + 1) as f64);
                }
                let mut counts = [0u32; 5];
                for &d in &seed_dists {
                    let d = d as f64;
                    for (bin, &thresh) in thresholds.iter().enumerate() {
                        if d <= thresh { counts[bin] += 1; break; }
                    }
                }
                let weights = [100.0, 36.788, 13.534, 4.979, 1.832];
                let score: f64 = counts.iter().zip(weights.iter())
                    .map(|(&c, &w)| w * c as f64 / seed_dists.len() as f64).sum();

                let p = estimate_ada_ef(&seed_dists, stats, q, table);
                // No ef cap — let hard queries get ef > cfg.ef if table says so
                (p.ef, p.stall_limit, p.drain_budget, Some(score))
            } else {
                (cfg.ef, cfg.stall_limit, cfg.drain_budget, None)
            }
        } else {
            (cfg.ef, cfg.stall_limit, cfg.drain_budget, None)
        };

        let mut perf = SearchPerfContext::default();
        let t0 = std::time::Instant::now();
        let results = disk_graph_search_pipe(
            q, entry_set, cfg.k, ef, cfg.prefetch_width,
            sl, db,
            pool, io, bank, &mut perf, PerfLevel::EnableTime,
        ).await;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1_000.0;
        latencies_ms.push(elapsed_ms);

        let ids: Vec<u32> = results.iter().map(|s| s.id.0).collect();
        let q_recall = recall_at_k(&ids, &ground_truth[i]);
        recalls.push(q_recall);

        if let Some(score) = ada_score {
            ada_info.push(AdaQueryInfo { score, ef_used: ef, recall: q_recall, blk: perf.blocks_read });
        }

        sum_exp += perf.expansions;
        sum_useful += perf.useful_expansions;
        sum_wasted += perf.wasted_expansions;
        sum_blk += perf.blocks_read;
        sum_miss += perf.blocks_miss;
        sum_hit += perf.blocks_hit;
        sum_phys_reads += perf.phys_reads;
        sum_sf += perf.singleflight_waits;
        sum_pf_issued += perf.prefetch_issued;
        sum_pf_consumed += perf.prefetch_consumed;
        sum_best_at += perf.best_result_at_expansion;
        sum_first_topk += perf.first_topk_at_expansion;
        sum_io_wait_ns += perf.io_wait_ns;
        sum_compute_ns += perf.compute_ns;
        sum_dist_ns += perf.dist_ns;
        if perf.stopped_early {
            early_count += 1;
        }
    }

    let wall_secs = wall_start.elapsed().as_secs_f64();
    let nf = nq as f64;

    let mean_recall = recalls.iter().sum::<f64>() / nf;
    let qps = nf / wall_secs;

    let mut sorted_lat = latencies_ms.clone();
    sorted_lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = percentile(&sorted_lat, 50.0);
    let p99 = percentile(&sorted_lat, 99.0);

    let total_exp = sum_useful + sum_wasted;
    let waste_ratio = if total_exp > 0 { sum_wasted as f64 / total_exp as f64 * 100.0 } else { 0.0 };
    let hit_rate = if sum_blk > 0 { sum_hit as f64 / sum_blk as f64 * 100.0 } else { 0.0 };
    let ns_to_ms = 1.0 / 1_000_000.0;

    // Print per-bucket diagnostics for ALL configs (using precomputed query_scores)
    {
        let buckets: &[(f64, &str)] = &[
            (20.0, ">=20"), (16.0, ">=16"), (12.0, ">=12"), (8.0, ">=8"), (0.0, "<8"),
        ];
        eprintln!("    {:>8} {:>5} {:>7} {:>6} {:>6} {:>7} {:>7}", "bucket", "n", "recall", "blk/q", "avg_ef", "p99ms", "maxms");
        for (bi, &(thresh, label)) in buckets.iter().enumerate() {
            let indices: Vec<usize> = (0..nq).filter(|&i| {
                let s = query_scores[i];
                if bi == 0 { s >= thresh }
                else { s >= thresh && s < buckets[bi - 1].0 }
            }).collect();
            if indices.is_empty() { continue; }
            let n = indices.len();
            let avg_recall = indices.iter().map(|&i| recalls[i]).sum::<f64>() / n as f64;
            // Per-bucket latency stats
            let mut bucket_lats: Vec<f64> = indices.iter().map(|&i| latencies_ms[i]).collect();
            bucket_lats.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let bucket_p99 = percentile(&bucket_lats, 99.0);
            let bucket_max = bucket_lats.last().copied().unwrap_or(0.0);
            // blk from ada_info if available, else from per-query perf (not tracked for non-ada)
            let (avg_blk, avg_ef) = if !ada_info.is_empty() {
                let blk: f64 = indices.iter().map(|&i| ada_info[i].blk as f64).sum::<f64>() / n as f64;
                let ef_avg: f64 = indices.iter().map(|&i| ada_info[i].ef_used as f64).sum::<f64>() / n as f64;
                (blk, ef_avg)
            } else {
                (0.0, cfg.ef as f64)
            };
            if avg_blk > 0.0 {
                eprintln!("    {:>8} {:>5} {:>7.3} {:>6.1} {:>6.0} {:>7.1} {:>7.1}", label, n, avg_recall, avg_blk, avg_ef, bucket_p99, bucket_max);
            } else {
                eprintln!("    {:>8} {:>5} {:>7.3} {:>6} {:>6.0} {:>7.1} {:>7.1}", label, n, avg_recall, "-", avg_ef, bucket_p99, bucket_max);
            }
        }
    }

    BenchResult {
        recall: mean_recall,
        lat_p50_ms: p50,
        lat_p99_ms: p99,
        qps,
        avg_expansions: sum_exp as f64 / nf,
        avg_useful: sum_useful as f64 / nf,
        avg_wasted: sum_wasted as f64 / nf,
        avg_blk_q: sum_blk as f64 / nf,
        avg_miss_q: sum_miss as f64 / nf,
        avg_hit_q: sum_hit as f64 / nf,
        avg_singleflight: sum_sf as f64 / nf,
        avg_pf_issued: sum_pf_issued as f64 / nf,
        avg_pf_consumed: sum_pf_consumed as f64 / nf,
        avg_best_at: sum_best_at as f64 / nf,
        avg_first_topk: sum_first_topk as f64 / nf,
        early_stop_pct: early_count as f64 / nf * 100.0,
        waste_ratio,
        hit_rate,
        avg_phys_reads_q: sum_phys_reads as f64 / nf,
        avg_io_wait_ms: sum_io_wait_ns as f64 / nf * ns_to_ms,
        avg_compute_ms: sum_compute_ns as f64 / nf * ns_to_ms,
        avg_dist_ms: sum_dist_ns as f64 / nf * ns_to_ms,
        ms_per_miss: if sum_miss > 0 {
            (sum_io_wait_ns as f64 * ns_to_ms) / sum_miss as f64
        } else {
            0.0
        },
        total_bypasses: pool.stats().bypasses - cache_stats_before.bypasses,
        total_evict_fail: pool.stats().evict_fail_all_pinned - cache_stats_before.evict_fail_all_pinned,
        avg_refine_count: 0.0,
        avg_refine_ms: 0.0,
        avg_total_io_q: sum_phys_reads as f64 / nf,
        avg_refine_bytes: 0.0,
    }
}

/// Run benchmark using v3 page-packed adjacency (key = page_id).
///
/// Caller must have a running prefetch worker. For `clear_per_query`, this
/// function uses pause/drain/wait/clear/unpause — the worker stays alive.
async fn run_bench_v3(
    cfg: &BenchConfig,
    entry_set: &[VectorId],
    pool: &Rc<AdjacencyPool>,
    io: &Rc<IoDriver>,
    bank: &dyn VectorBank,
    adj_index: &[AdjIndexEntry],
    query_vecs: &[Vec<f32>],
    ground_truth: &[Vec<u32>],
) -> BenchResult {
    let nq = cfg.num_queries.min(query_vecs.len());

    // Warmup pass
    for q in query_vecs.iter().take(cfg.warmup_queries) {
        let mut perf = SearchPerfContext::default();
        disk_graph_search_pipe_v3(
            q, entry_set, cfg.k, cfg.ef, cfg.prefetch_width,
            cfg.stall_limit, cfg.drain_budget,
            pool, io, bank, adj_index, &mut perf, PerfLevel::CountOnly,
        ).await;
    }

    let mut recalls = Vec::with_capacity(nq);
    let mut latencies_ms = Vec::with_capacity(nq);
    let mut sum_exp = 0u64;
    let mut sum_useful = 0u64;
    let mut sum_wasted = 0u64;
    let mut sum_blk = 0u64;
    let mut sum_miss = 0u64;
    let mut sum_hit = 0u64;
    let mut sum_phys_reads = 0u64;
    let mut sum_sf = 0u64;
    let mut sum_pf_issued = 0u64;
    let mut sum_pf_consumed = 0u64;
    let mut sum_best_at = 0u64;
    let mut sum_first_topk = 0u64;
    let mut early_count = 0u64;
    let mut sum_io_wait_ns = 0u64;
    let mut sum_compute_ns = 0u64;
    let mut sum_dist_ns = 0u64;

    let cache_stats_before = pool.stats();
    let wall_start = std::time::Instant::now();

    for i in 0..nq {
        if cfg.clear_per_query {
            // Quiesce: pause hints → drain channel → yield → wait LOADING → clear → unpause
            pool.pause_prefetch(true);
            pool.drain_prefetch();
            // Yield once to let already-spawned IO tasks set their LOADING flags
            monoio::time::sleep(std::time::Duration::from_micros(50)).await;
            while pool.has_loading() {
                monoio::time::sleep(std::time::Duration::from_micros(100)).await;
            }
            pool.clear();
            pool.pause_prefetch(false);
        }
        let q = &query_vecs[i];
        let mut perf = SearchPerfContext::default();
        let t0 = std::time::Instant::now();
        let results = disk_graph_search_pipe_v3(
            q, entry_set, cfg.k, cfg.ef, cfg.prefetch_width,
            cfg.stall_limit, cfg.drain_budget,
            pool, io, bank, adj_index, &mut perf, PerfLevel::EnableTime,
        ).await;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1_000.0;
        latencies_ms.push(elapsed_ms);

        let ids: Vec<u32> = results.iter().map(|s| s.id.0).collect();
        let q_recall = recall_at_k(&ids, &ground_truth[i]);
        recalls.push(q_recall);

        sum_exp += perf.expansions;
        sum_useful += perf.useful_expansions;
        sum_wasted += perf.wasted_expansions;
        sum_blk += perf.blocks_read;
        sum_miss += perf.blocks_miss;
        sum_hit += perf.blocks_hit;
        sum_phys_reads += perf.phys_reads;
        sum_sf += perf.singleflight_waits;
        sum_pf_issued += perf.prefetch_issued;
        sum_pf_consumed += perf.prefetch_consumed;
        sum_best_at += perf.best_result_at_expansion;
        sum_first_topk += perf.first_topk_at_expansion;
        sum_io_wait_ns += perf.io_wait_ns;
        sum_compute_ns += perf.compute_ns;
        sum_dist_ns += perf.dist_ns;
        if perf.stopped_early {
            early_count += 1;
        }
    }

    let wall_secs = wall_start.elapsed().as_secs_f64();
    let nf = nq as f64;

    let mean_recall = recalls.iter().sum::<f64>() / nf;
    let qps = nf / wall_secs;

    let mut sorted_lat = latencies_ms.clone();
    sorted_lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = percentile(&sorted_lat, 50.0);
    let p99 = percentile(&sorted_lat, 99.0);

    let total_exp = sum_useful + sum_wasted;
    let waste_ratio = if total_exp > 0 { sum_wasted as f64 / total_exp as f64 * 100.0 } else { 0.0 };
    let hit_rate = if sum_blk > 0 { sum_hit as f64 / sum_blk as f64 * 100.0 } else { 0.0 };
    let ns_to_ms = 1.0 / 1_000_000.0;

    BenchResult {
        recall: mean_recall,
        lat_p50_ms: p50,
        lat_p99_ms: p99,
        qps,
        avg_expansions: sum_exp as f64 / nf,
        avg_useful: sum_useful as f64 / nf,
        avg_wasted: sum_wasted as f64 / nf,
        avg_blk_q: sum_blk as f64 / nf,
        avg_miss_q: sum_miss as f64 / nf,
        avg_hit_q: sum_hit as f64 / nf,
        avg_singleflight: sum_sf as f64 / nf,
        avg_pf_issued: sum_pf_issued as f64 / nf,
        avg_pf_consumed: sum_pf_consumed as f64 / nf,
        avg_best_at: sum_best_at as f64 / nf,
        avg_first_topk: sum_first_topk as f64 / nf,
        early_stop_pct: early_count as f64 / nf * 100.0,
        waste_ratio,
        hit_rate,
        avg_phys_reads_q: sum_phys_reads as f64 / nf,
        avg_io_wait_ms: sum_io_wait_ns as f64 / nf * ns_to_ms,
        avg_compute_ms: sum_compute_ns as f64 / nf * ns_to_ms,
        avg_dist_ms: sum_dist_ns as f64 / nf * ns_to_ms,
        ms_per_miss: if sum_miss > 0 {
            (sum_io_wait_ns as f64 * ns_to_ms) / sum_miss as f64
        } else {
            0.0
        },
        total_bypasses: pool.stats().bypasses - cache_stats_before.bypasses,
        total_evict_fail: pool.stats().evict_fail_all_pinned - cache_stats_before.evict_fail_all_pinned,
        avg_refine_count: 0.0,
        avg_refine_ms: 0.0,
        avg_total_io_q: sum_phys_reads as f64 / nf,
        avg_refine_bytes: 0.0,
    }
}

fn print_bench_header(n: usize, dim: usize, num_queries: usize, warmup_queries: usize) {
    print_bench_header_layout(n, dim, num_queries, warmup_queries, None);
}

fn print_bench_header_layout(n: usize, dim: usize, num_queries: usize, warmup_queries: usize, layout: Option<&str>) {
    let layout_str = layout.map(|l| format!(", layout={}", l)).unwrap_or_default();
    eprintln!("\n=== BENCH: Cohere {}K, dim={}, GT=brute-force, seed=42, nq={}, warmup={}{} ===",
        n / 1000, dim, num_queries, warmup_queries, layout_str);
    eprintln!(
        "{:<14} {:>4} {:>4} {:>2} {:>3} {:>3} {:>4} {:>7} {:>7} {:>7} {:>7} {:>5} {:>5} {:>5} {:>6} {:>6} {:>6} {:>6} {:>5} {:>5} {:>5} {:>6} {:>6} {:>7} {:>7} {:>5} {:>7} {:>7} {:>7} {:>6} {:>6} {:>6} {:>6} {:>7} {:>7}",
        "label", "ef", "k", "W", "S", "D", "c%",
        "recall", "p50ms", "p99ms", "qps",
        "exp", "use", "wst", "blk/q", "mis/q", "phy/q", "hit/q", "sf/q",
        "pf_i", "pf_c", "best@", "1stk@",
        "early%", "waste%", "hit%",
        "io_ms", "cmp_ms", "dst_ms", "ms/mis",
        "byp", "evfail",
        "ref/q", "ref_ms", "io/q"
    );
}

fn print_bench_row(cfg: &BenchConfig, r: &BenchResult) {
    eprintln!(
        "{:<14} {:>4} {:>4} {:>2} {:>3} {:>3} {:>4} {:>7.3} {:>7.1} {:>7.1} {:>7.1} {:>5.0} {:>5.0} {:>5.0} {:>6.1} {:>6.1} {:>6.1} {:>6.1} {:>5.1} {:>5.1} {:>5.1} {:>6.1} {:>6.1} {:>7.1} {:>7.1} {:>5.1} {:>7.2} {:>7.2} {:>7.2} {:>6.3} {:>6} {:>6} {:>6.0} {:>7.2} {:>7.1}",
        cfg.label, cfg.ef, cfg.k, cfg.prefetch_width, cfg.stall_limit, cfg.drain_budget, cfg.cache_pct,
        r.recall, r.lat_p50_ms, r.lat_p99_ms, r.qps,
        r.avg_expansions, r.avg_useful, r.avg_wasted,
        r.avg_blk_q, r.avg_miss_q, r.avg_phys_reads_q, r.avg_hit_q, r.avg_singleflight,
        r.avg_pf_issued, r.avg_pf_consumed,
        r.avg_best_at, r.avg_first_topk,
        r.early_stop_pct, r.waste_ratio, r.hit_rate,
        r.avg_io_wait_ms, r.avg_compute_ms, r.avg_dist_ms, r.ms_per_miss,
        r.total_bypasses, r.total_evict_fail,
        r.avg_refine_count, r.avg_refine_ms, r.avg_total_io_q
    );
}

#[test]
#[ignore] // EC2-only: BENCH_DIR + COHERE_N required
fn exp_bench_stable() {
    let max_n: usize = std::env::var("COHERE_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    let dataset_dir = std::env::var("COHERE_DIR").unwrap_or_else(|_| {
        let manifest = env!("CARGO_MANIFEST_DIR");
        format!("{}/../../data/cohere_100k", manifest)
    });

    let (vectors, queries_flat, ground_truth, n, nq, dim, k) =
        match load_dataset(&dataset_dir, max_n) {
            Some(d) => d,
            None => return,
        };

    let m_max = 32;
    let ef_construction = 200;
    let prefetch_budget = 4;

    eprintln!("Building NSW index (n={}, dim={}, m_max={}, ef_c={}) ...", n, dim, m_max, ef_construction);
    let t0 = std::time::Instant::now();
    let config = NswConfig::new(m_max, ef_construction);
    let builder = NswBuilder::new(config, dim, MetricType::Cosine, n);
    for (i, v) in vectors.chunks_exact(dim).enumerate() {
        builder.insert(VectorId(i as u32), v);
    }
    let index = builder.build();
    eprintln!("  Index built in {:.1}s", t0.elapsed().as_secs_f64());

    // Write to disk
    let bench_dir = std::env::var("BENCH_DIR").ok();
    let direct_io = bench_dir.is_some();
    let _tmpdir;
    let dir_path: std::path::PathBuf;
    if let Some(ref bd) = bench_dir {
        dir_path = std::path::PathBuf::from(bd);
        std::fs::create_dir_all(&dir_path).unwrap();
    } else {
        _tmpdir = tempfile::tempdir().unwrap();
        dir_path = _tmpdir.path().to_path_buf();
    }
    let dir_str = dir_path.to_str().unwrap().to_owned();
    let writer = IndexWriter::new(&dir_path);
    writer
        .write(
            n as u32, dim, "cosine", index.max_degree(), ef_construction,
            &index.entry_set().iter().map(|v| v.0).collect::<Vec<_>>(),
            index.vectors_raw(), |vid| index.neighbors(vid),
        )
        .unwrap();
    eprintln!("  v1 index written to {} (direct_io={})", dir_str, direct_io);

    // Write v3 page-packed adjacency into a subdirectory.
    // V3_LAYOUT env var selects reorder strategy (default: heavy_edge).
    let v3_layout = std::env::var("V3_LAYOUT").unwrap_or_else(|_| "heavy_edge".to_string());
    let v3_dir_path = dir_path.join("v3");
    std::fs::create_dir_all(&v3_dir_path).unwrap();
    let v3_dir_str = v3_dir_path.to_str().unwrap().to_owned();
    let entry_ids: Vec<u32> = index.entry_set().iter().map(|v| v.0).collect();
    let t0_v3 = std::time::Instant::now();
    let reorder = match v3_layout.as_str() {
        "bfs" => bfs_reorder_graph(n, &entry_ids, |vid| index.neighbors(vid)),
        "heavy_edge" => heavy_edge_reorder_graph(n, |vid| index.neighbors(vid)),
        other => panic!("unknown V3_LAYOUT '{}' (expected 'bfs' or 'heavy_edge')", other),
    };
    let v3_writer = IndexWriter::new(&v3_dir_path);
    v3_writer
        .write_v3(
            n as u32, dim, "cosine", index.max_degree(), ef_construction,
            &entry_ids,
            index.vectors_raw(), |vid| index.neighbors(vid),
            &reorder,
            &v3_layout,
        )
        .unwrap();
    // Copy vectors.dat to v3 dir (IoDriver + VectorBank need same dir)
    std::fs::copy(dir_path.join("vectors.dat"), v3_dir_path.join("vectors.dat")).unwrap();
    let v3_meta = IndexMeta::load_from(&v3_dir_path.join("meta.json")).unwrap();
    eprintln!("  v3 index written to {} ({} pages, {} reorder) in {:.1}s",
        v3_dir_str, v3_meta.num_pages.unwrap_or(0), v3_layout, t0_v3.elapsed().as_secs_f64());

    let disk_vectors = load_vectors(&dir_path.join("vectors.dat"), n, dim).unwrap();
    let entry_set: Vec<VectorId> = {
        let meta = IndexMeta::load_from(&dir_path.join("meta.json")).unwrap();
        meta.entry_set.iter().map(|&v| VectorId(v)).collect()
    };

    // Load v3 adj_index into DRAM — use meta.num_vectors, not dataset n
    let adj_index = load_adj_index(
        &v3_dir_path.join("adj_index.dat"),
        v3_meta.num_vectors as usize,
    ).unwrap();
    eprintln!("  adj_index loaded: {} entries ({:.1} KB DRAM)",
        adj_index.len(), adj_index.len() as f64 * 8.0 / 1024.0);

    let num_queries = nq.min(100);
    let query_vecs: Vec<Vec<f32>> = queries_flat
        .chunks_exact(dim).take(num_queries).map(|c| c.to_vec()).collect();

    let warmup_queries = 10;

    // Compute Ada-ef stats (diagonal-only v0, O(n*d))
    // Vectors must be L2-normalized for cosine FDL theory.
    // Cohere vectors are NOT pre-normalized — use from_raw_vectors_cosine().
    eprintln!("Computing Ada-ef stats (diagonal variance + normalize, n={}, dim={}) ...", n, dim);
    let ada_stats = AdaEfStats::from_raw_vectors_cosine(&vectors, n, dim);

    // Build Ada-ef table: score thresholds calibrated to observed Cohere 100K distribution.
    // Score range: ~5-28, median ~14, p25 ~11.
    // v0.2: ef-only calibration (S=0, D=0 everywhere) to isolate ef effect.
    // Hard queries get ef>200 to test if more budget recovers recall.
    let ada_table = AdaEfTable::new(
        &[
            (20.0, 150, 0, 0),   // top ~20%: clearly easy
            (16.0, 170, 0, 0),   // above median
            (12.0, 190, 0, 0),   // below median
            (8.0,  200, 0, 0),   // hard
        ],
        AdaEfParams { ef: 250, stall_limit: 0, drain_budget: 0 },  // hardest: EXTRA budget
    );

    // Define benchmark configs: baselines + Ada-ef, both warm and cold
    let configs = vec![
        // --- Warm baselines (5% cache) ---
        BenchConfig {
            label: "warm".to_string(),
            ef: 200, k, prefetch_width: 4,
            stall_limit: 0, drain_budget: 0,
            adj_inflight: 64, cache_pct: 5,
            num_queries, warmup_queries,
            ada_ef: false,
            clear_per_query: false,
        },
        BenchConfig {
            label: "warm-S4D16".to_string(),
            ef: 200, k, prefetch_width: 4,
            stall_limit: 4, drain_budget: 16,
            adj_inflight: 64, cache_pct: 5,
            num_queries, warmup_queries,
            ada_ef: false,
            clear_per_query: false,
        },
        BenchConfig {
            label: "warm-ada".to_string(),
            ef: 200, k, prefetch_width: 4,
            stall_limit: 0, drain_budget: 0,  // overridden per-query
            adj_inflight: 64, cache_pct: 5,
            num_queries, warmup_queries,
            ada_ef: true,
            clear_per_query: false,
        },
        // --- ef sweep (warm): diagnose >=8 bucket ---
        BenchConfig {
            label: "warm-ef225".to_string(),
            ef: 225, k, prefetch_width: 4,
            stall_limit: 0, drain_budget: 0,
            adj_inflight: 64, cache_pct: 5,
            num_queries, warmup_queries,
            ada_ef: false,
            clear_per_query: false,
        },
        BenchConfig {
            label: "warm-ef250".to_string(),
            ef: 250, k, prefetch_width: 4,
            stall_limit: 0, drain_budget: 0,
            adj_inflight: 64, cache_pct: 5,
            num_queries, warmup_queries,
            ada_ef: false,
            clear_per_query: false,
        },
        BenchConfig {
            label: "warm-ef300".to_string(),
            ef: 300, k, prefetch_width: 4,
            stall_limit: 0, drain_budget: 0,
            adj_inflight: 64, cache_pct: 5,
            num_queries, warmup_queries,
            ada_ef: false,
            clear_per_query: false,
        },
        // --- Cold baselines (0% cache, IO-bound) ---
        BenchConfig {
            label: "cold".to_string(),
            ef: 200, k, prefetch_width: 4,
            stall_limit: 0, drain_budget: 0,
            adj_inflight: 64, cache_pct: 0,
            num_queries, warmup_queries: 0,
            ada_ef: false,
            clear_per_query: false,
        },
        BenchConfig {
            label: "cold-S4D16".to_string(),
            ef: 200, k, prefetch_width: 4,
            stall_limit: 4, drain_budget: 16,
            adj_inflight: 64, cache_pct: 0,
            num_queries, warmup_queries: 0,
            ada_ef: false,
            clear_per_query: false,
        },
        BenchConfig {
            label: "cold-ada".to_string(),
            ef: 200, k, prefetch_width: 4,
            stall_limit: 0, drain_budget: 0,  // overridden per-query
            adj_inflight: 64, cache_pct: 0,
            num_queries, warmup_queries: 0,
            ada_ef: true,
            clear_per_query: false,
        },
    ];

    if !with_runtime(|rt| {
        rt.block_on(async {
            let io = Rc::new(
                IoDriver::open(&dir_str, dim, 64, direct_io)
                    .await
                    .expect("failed to open IO driver"),
            );
            let bank = FP32SimdVectorBank::new(&disk_vectors, dim, MetricType::Cosine);

            // Precompute per-query Ada-ef scores (used for both diagnostics and bucket breakdown)
            let query_scores: Vec<f64> = {
                let mut scores = Vec::with_capacity(num_queries);
                for q in &query_vecs {
                    let seed_dists: Vec<f32> = entry_set
                        .iter()
                        .map(|&ep| bank.distance(q, ep.0 as usize))
                        .collect();
                    let (mu, sigma) = ada_stats.estimate_fdl_params(q);
                    let mut thresholds = [0.0f64; 5];
                    for i in 0..5 {
                        thresholds[i] = mu + sigma * divergence_engine::ada_ef::inv_normal_cdf(0.001 * (i + 1) as f64);
                    }
                    let mut counts = [0u32; 5];
                    for &d in &seed_dists {
                        let d = d as f64;
                        for (bin, &thresh) in thresholds.iter().enumerate() {
                            if d <= thresh { counts[bin] += 1; break; }
                        }
                    }
                    let n_seeds = seed_dists.len() as f64;
                    let weights = [100.0, 36.788, 13.534, 4.979, 1.832];
                    let score: f64 = counts.iter().zip(weights.iter())
                        .map(|(&c, &w)| w * c as f64 / n_seeds).sum();
                    scores.push(score);
                }
                scores
            };

            // Print score distribution
            {
                let mut sorted = query_scores.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                eprintln!(
                    "\nAda-ef score distribution (nq={}): min={:.2} p25={:.2} p50={:.2} mean={:.2} p75={:.2} max={:.2}",
                    sorted.len(), sorted[0], sorted[sorted.len()/4], sorted[sorted.len()/2],
                    sorted.iter().sum::<f64>() / sorted.len() as f64,
                    sorted[3*sorted.len()/4], sorted[sorted.len()-1]
                );
                let buckets = [20.0, 16.0, 12.0, 8.0, 0.0];
                let labels = [">=20 (ef=150)", ">=16 (ef=170)", ">=12 (ef=190)", ">=8 (ef=200)", "<8 (floor=250)"];
                for (i, &thresh) in buckets.iter().enumerate() {
                    let count = if i == 0 {
                        query_scores.iter().filter(|&&s| s >= thresh).count()
                    } else {
                        query_scores.iter().filter(|&&s| s >= thresh && s < buckets[i-1]).count()
                    };
                    eprintln!("  {}: {} queries ({:.0}%)", labels[i], count, count as f64 / query_scores.len() as f64 * 100.0);
                }
            }

            print_bench_header(n, dim, num_queries, warmup_queries);

            for cfg in &configs {
                // cache_pct=0 → single 8-way set (32KB), truly cold
                let pool_bytes = if cfg.cache_pct > 0 {
                    n * 4096 * cfg.cache_pct / 100
                } else {
                    8 * 4096 // one set
                };
                let pool = Rc::new(AdjacencyPool::new(pool_bytes));
                let handle = AdjacencyPool::spawn_prefetch_worker(
                    Rc::clone(&pool), Rc::clone(&io), prefetch_budget,
                );

                let ada = if cfg.ada_ef {
                    Some((&ada_stats, &ada_table))
                } else {
                    None
                };

                let result = run_bench(
                    cfg, &entry_set, &pool, &io, &bank,
                    &query_vecs, &ground_truth, ada, &query_scores,
                ).await;

                print_bench_row(cfg, &result);

                pool.stop_prefetch();
                handle.await;
            }

            // =================================================================
            // v3 page-packed adjacency benchmarks
            // =================================================================
            eprintln!("\n--- v3 page-packed adjacency ({} reorder) ---", v3_layout);

            let io_v3 = Rc::new(
                IoDriver::open_pages(&v3_dir_str, dim, 64, direct_io)
                    .await
                    .expect("failed to open v3 IO driver"),
            );

            let v3_configs = vec![
                // --- Steady-state (cross-query cache warms up) ---
                BenchConfig {
                    label: "v3-warm".to_string(),
                    ef: 200, k, prefetch_width: 4,
                    stall_limit: 0, drain_budget: 0,
                    adj_inflight: 64, cache_pct: 5,
                    num_queries, warmup_queries,
                    ada_ef: false,
                    clear_per_query: false,
                },
                BenchConfig {
                    label: "v3-warm-S4D16".to_string(),
                    ef: 200, k, prefetch_width: 4,
                    stall_limit: 4, drain_budget: 16,
                    adj_inflight: 64, cache_pct: 5,
                    num_queries, warmup_queries,
                    ada_ef: false,
                    clear_per_query: false,
                },
                // Cold steady-state (starts cold, warms across queries)
                BenchConfig {
                    label: "v3-cold".to_string(),
                    ef: 200, k, prefetch_width: 4,
                    stall_limit: 0, drain_budget: 0,
                    adj_inflight: 64, cache_pct: 0,
                    num_queries, warmup_queries: 0,
                    ada_ef: false,
                    clear_per_query: false,
                },
                // --- Per-query cold: pool.clear() before each query ---
                // Isolates intra-query page reuse (no cross-query warming)
                // Should approach simulation's ~135 phys_reads/q
                BenchConfig {
                    label: "v3-perq-cold".to_string(),
                    ef: 200, k, prefetch_width: 4,
                    stall_limit: 0, drain_budget: 0,
                    adj_inflight: 64, cache_pct: 0,
                    num_queries, warmup_queries: 0,
                    ada_ef: false,
                    clear_per_query: true,
                },
                BenchConfig {
                    label: "v3-perq-S4D16".to_string(),
                    ef: 200, k, prefetch_width: 4,
                    stall_limit: 4, drain_budget: 16,
                    adj_inflight: 64, cache_pct: 0,
                    num_queries, warmup_queries: 0,
                    ada_ef: false,
                    clear_per_query: true,
                },
                // --- ef sweep (warm): find iso-recall points for DiskANN comparison ---
                BenchConfig {
                    label: "v3-warm-ef225".to_string(),
                    ef: 225, k, prefetch_width: 4,
                    stall_limit: 0, drain_budget: 0,
                    adj_inflight: 64, cache_pct: 5,
                    num_queries, warmup_queries,
                    ada_ef: false,
                    clear_per_query: false,
                },
                BenchConfig {
                    label: "v3-warm-ef250".to_string(),
                    ef: 250, k, prefetch_width: 4,
                    stall_limit: 0, drain_budget: 0,
                    adj_inflight: 64, cache_pct: 5,
                    num_queries, warmup_queries,
                    ada_ef: false,
                    clear_per_query: false,
                },
                BenchConfig {
                    label: "v3-warm-ef300".to_string(),
                    ef: 300, k, prefetch_width: 4,
                    stall_limit: 0, drain_budget: 0,
                    adj_inflight: 64, cache_pct: 5,
                    num_queries, warmup_queries,
                    ada_ef: false,
                    clear_per_query: false,
                },
                // --- ef sweep (perq-cold): strictest comparison ---
                BenchConfig {
                    label: "v3-perq-ef225".to_string(),
                    ef: 225, k, prefetch_width: 4,
                    stall_limit: 0, drain_budget: 0,
                    adj_inflight: 64, cache_pct: 0,
                    num_queries, warmup_queries: 0,
                    ada_ef: false,
                    clear_per_query: true,
                },
                BenchConfig {
                    label: "v3-perq-ef250".to_string(),
                    ef: 250, k, prefetch_width: 4,
                    stall_limit: 0, drain_budget: 0,
                    adj_inflight: 64, cache_pct: 0,
                    num_queries, warmup_queries: 0,
                    ada_ef: false,
                    clear_per_query: true,
                },
            ];

            let v3_num_pages = v3_meta.num_pages.unwrap_or(0) as usize;

            for cfg in &v3_configs {
                // v3 pool sizing: key is page_id, not vid.
                // cache_pct=0 → 256 pages (1MB) to allow intra-query page reuse
                //   (p99 = ~164 unique pages/q; below this, eviction kills reuse)
                // cache_pct>0 → pct of total pages, min 256
                let pool_pages = if cfg.cache_pct > 0 {
                    (v3_num_pages * cfg.cache_pct / 100).max(256)
                } else {
                    256 // ~1MB, enough for intra-query reuse
                };
                let pool_bytes = pool_pages * 4096;
                let pool = Rc::new(AdjacencyPool::new(pool_bytes));
                let handle = AdjacencyPool::spawn_prefetch_worker(
                    Rc::clone(&pool), Rc::clone(&io_v3), prefetch_budget,
                );

                let result = run_bench_v3(
                    cfg, &entry_set, &pool, &io_v3, &bank,
                    &adj_index, &query_vecs, &ground_truth,
                ).await;
                print_bench_row(cfg, &result);

                pool.stop_prefetch();
                handle.await;
            }

            // =================================================================
            // Hub pinning benchmark: pin first N pages, measure benefit
            // =================================================================
            eprintln!("\n--- Hub pinning (v3, {}-reordered pages) ---", v3_layout);
            print_bench_header(n, dim, num_queries, 0);

            for &pin_count in &[64u32, 128u32] {
                let pin_pages: Vec<u32> = (0..pin_count).collect();

                // v3-perq-pinN: per-query cold with pinned hub pages
                {
                    let pool_pages = 256usize; // 1MB
                    let pool_bytes = pool_pages * 4096;
                    let pool = Rc::new(AdjacencyPool::new(pool_bytes));
                    // Pin before starting worker
                    let actually_pinned = pool.pin_pages(&pin_pages, &io_v3).await
                        .expect("pin_pages failed");
                    eprintln!("  pin{}: requested={}, actually_pinned={}", pin_count, pin_count, actually_pinned);
                    let handle = AdjacencyPool::spawn_prefetch_worker(
                        Rc::clone(&pool), Rc::clone(&io_v3), prefetch_budget,
                    );

                    let cfg = BenchConfig {
                        label: format!("v3-perq-pin{}", pin_count),
                        ef: 200, k, prefetch_width: 4,
                        stall_limit: 0, drain_budget: 0,
                        adj_inflight: 64, cache_pct: 0,
                        num_queries, warmup_queries: 0,
                        ada_ef: false,
                        clear_per_query: true,
                    };
                    let result = run_bench_v3(
                        &cfg, &entry_set, &pool, &io_v3, &bank,
                        &adj_index, &query_vecs, &ground_truth,
                    ).await;
                    print_bench_row(&cfg, &result);
                    pool.stop_prefetch();
                    handle.await;
                }

                // v3-warm-pinN: warm cache with pinned hub pages
                {
                    let pool_pages = (v3_num_pages * 5 / 100).max(256);
                    let pool_bytes = pool_pages * 4096;
                    let pool = Rc::new(AdjacencyPool::new(pool_bytes));
                    // Pin BEFORE starting prefetch worker
                    let actually_pinned = pool.pin_pages(&pin_pages, &io_v3).await
                        .expect("pin_pages failed");
                    eprintln!("  pin{}: requested={}, actually_pinned={}", pin_count, pin_count, actually_pinned);
                    let handle = AdjacencyPool::spawn_prefetch_worker(
                        Rc::clone(&pool), Rc::clone(&io_v3), prefetch_budget,
                    );

                    let cfg = BenchConfig {
                        label: format!("v3-warm-pin{}", pin_count),
                        ef: 200, k, prefetch_width: 4,
                        stall_limit: 0, drain_budget: 0,
                        adj_inflight: 64, cache_pct: 5,
                        num_queries, warmup_queries,
                        ada_ef: false,
                        clear_per_query: false,
                    };
                    let result = run_bench_v3(
                        &cfg, &entry_set, &pool, &io_v3, &bank,
                        &adj_index, &query_vecs, &ground_truth,
                    ).await;
                    print_bench_row(&cfg, &result);
                    pool.stop_prefetch();
                    handle.await;
                }
            }

            // =================================================================
            // Equal-budget comparison: same pool_bytes for v1 vs v3
            // Separates "page packing wins" from "pool sizing wins"
            // All per-query cold (pool.clear() before each query)
            // =================================================================
            eprintln!("\n--- Equal-budget comparison (v1 vs v3, same pool_bytes, perq-cold) ---");
            print_bench_header(n, dim, num_queries, 0);

            let budget_sizes: Vec<(usize, &str)> = vec![
                (32 * 1024, "32KB"),
                (256 * 1024, "256KB"),
                (1024 * 1024, "1MB"),
                (4 * 1024 * 1024, "4MB"),
            ];

            for &(budget_bytes, budget_label) in &budget_sizes {
                // v1 with this budget, per-query cold
                {
                    let label = format!("v1-{}", budget_label);
                    let pool = Rc::new(AdjacencyPool::new(budget_bytes));
                    let handle = AdjacencyPool::spawn_prefetch_worker(
                        Rc::clone(&pool), Rc::clone(&io), prefetch_budget,
                    );
                    let cfg = BenchConfig {
                        label, ef: 200, k, prefetch_width: 4,
                        stall_limit: 0, drain_budget: 0,
                        adj_inflight: 64, cache_pct: 0,
                        num_queries, warmup_queries: 0,
                        ada_ef: false,
                        clear_per_query: true,
                    };
                    // run_bench doesn't support clear_per_query, so inline the loop
                    let mut recalls = Vec::with_capacity(num_queries);
                    let mut latencies_ms = Vec::with_capacity(num_queries);
                    let mut sum_exp = 0u64;
                    let mut sum_useful = 0u64;
                    let mut sum_wasted = 0u64;
                    let mut sum_blk = 0u64;
                    let mut sum_miss = 0u64;
                    let mut sum_hit = 0u64;
                    let mut sum_phys_reads = 0u64;
                    let mut sum_sf = 0u64;
                    let mut sum_pf_issued = 0u64;
                    let mut sum_pf_consumed = 0u64;
                    let mut sum_io_wait_ns = 0u64;
                    let mut sum_compute_ns = 0u64;
                    let mut sum_dist_ns = 0u64;
                    let wall_start = std::time::Instant::now();

                    for i in 0..num_queries {
                        // Quiesce: pause → drain → yield → wait LOADING → clear → unpause
                        pool.pause_prefetch(true);
                        pool.drain_prefetch();
                        monoio::time::sleep(std::time::Duration::from_micros(50)).await;
                        while pool.has_loading() {
                            monoio::time::sleep(std::time::Duration::from_micros(100)).await;
                        }
                        pool.clear();
                        pool.pause_prefetch(false);
                        let q = &query_vecs[i];
                        let mut perf = SearchPerfContext::default();
                        let t0 = std::time::Instant::now();
                        let results = disk_graph_search_pipe(
                            q, &entry_set, k, 200, 4, 0, 0,
                            &pool, &io, &bank, &mut perf, PerfLevel::EnableTime,
                        ).await;
                        latencies_ms.push(t0.elapsed().as_secs_f64() * 1_000.0);
                        let ids: Vec<u32> = results.iter().map(|s| s.id.0).collect();
                        recalls.push(recall_at_k(&ids, &ground_truth[i]));
                        sum_exp += perf.expansions;
                        sum_useful += perf.useful_expansions;
                        sum_wasted += perf.wasted_expansions;
                        sum_blk += perf.blocks_read;
                        sum_miss += perf.blocks_miss;
                        sum_hit += perf.blocks_hit;
                        sum_phys_reads += perf.phys_reads;
                        sum_sf += perf.singleflight_waits;
                        sum_pf_issued += perf.prefetch_issued;
                        sum_pf_consumed += perf.prefetch_consumed;
                        sum_io_wait_ns += perf.io_wait_ns;
                        sum_compute_ns += perf.compute_ns;
                        sum_dist_ns += perf.dist_ns;
                    }

                    let nf = num_queries as f64;
                    let wall_secs = wall_start.elapsed().as_secs_f64();
                    let mut sorted_lat = latencies_ms.clone();
                    sorted_lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let total_exp = sum_useful + sum_wasted;
                    let ns_to_ms = 1.0 / 1_000_000.0;
                    let result = BenchResult {
                        recall: recalls.iter().sum::<f64>() / nf,
                        lat_p50_ms: percentile(&sorted_lat, 50.0),
                        lat_p99_ms: percentile(&sorted_lat, 99.0),
                        qps: nf / wall_secs,
                        avg_expansions: sum_exp as f64 / nf,
                        avg_useful: sum_useful as f64 / nf,
                        avg_wasted: sum_wasted as f64 / nf,
                        avg_blk_q: sum_blk as f64 / nf,
                        avg_miss_q: sum_miss as f64 / nf,
                        avg_hit_q: sum_hit as f64 / nf,
                        avg_singleflight: sum_sf as f64 / nf,
                        avg_pf_issued: sum_pf_issued as f64 / nf,
                        avg_pf_consumed: sum_pf_consumed as f64 / nf,
                        avg_best_at: 0.0, avg_first_topk: 0.0,
                        early_stop_pct: 0.0,
                        waste_ratio: if total_exp > 0 { sum_wasted as f64 / total_exp as f64 * 100.0 } else { 0.0 },
                        hit_rate: if sum_blk > 0 { sum_hit as f64 / sum_blk as f64 * 100.0 } else { 0.0 },
                        avg_phys_reads_q: sum_phys_reads as f64 / nf,
                        avg_io_wait_ms: sum_io_wait_ns as f64 / nf * ns_to_ms,
                        avg_compute_ms: sum_compute_ns as f64 / nf * ns_to_ms,
                        avg_dist_ms: sum_dist_ns as f64 / nf * ns_to_ms,
                        ms_per_miss: if sum_miss > 0 { (sum_io_wait_ns as f64 * ns_to_ms) / sum_miss as f64 } else { 0.0 },
                        total_bypasses: pool.stats().bypasses,
                        total_evict_fail: pool.stats().evict_fail_all_pinned,
                        avg_refine_count: 0.0,
                        avg_refine_ms: 0.0,
                        avg_total_io_q: sum_phys_reads as f64 / nf,
                        avg_refine_bytes: 0.0,
                    };
                    print_bench_row(&cfg, &result);
                    pool.stop_prefetch();
                    handle.await;
                }

                // v3 with same budget, per-query cold
                {
                    let label = format!("v3-{}", budget_label);
                    let pool = Rc::new(AdjacencyPool::new(budget_bytes));
                    let handle = AdjacencyPool::spawn_prefetch_worker(
                        Rc::clone(&pool), Rc::clone(&io_v3), prefetch_budget,
                    );
                    let cfg = BenchConfig {
                        label, ef: 200, k, prefetch_width: 4,
                        stall_limit: 0, drain_budget: 0,
                        adj_inflight: 64, cache_pct: 0,
                        num_queries, warmup_queries: 0,
                        ada_ef: false,
                        clear_per_query: true,
                    };
                    let result = run_bench_v3(
                        &cfg, &entry_set, &pool, &io_v3, &bank,
                        &adj_index, &query_vecs, &ground_truth,
                    ).await;
                    print_bench_row(&cfg, &result);
                    pool.stop_prefetch();
                    handle.await;
                }
            }

            // PQ-only scoring section removed — PQ codebook training is too slow
            // for the stable benchmark. Use exp_pq_gate / exp_pq_gate_v2 instead.
            // (PQ oracle + PQ graph-traversal code removed from here)
        });
    }) {
        eprintln!("SKIPPED: io_uring not available");
    }
}

// ===== TWPP: Trace-Weighted Page Packing Experiment =====

/// Helper: compute unique pages touched for a set of VID expansions using adj_index.
fn unique_pages_for_query(expanded_vids: &[u32], adj_index: &[AdjIndexEntry]) -> usize {
    let mut pages = std::collections::HashSet::new();
    for &vid in expanded_vids {
        pages.insert(adj_index[vid as usize].page_id);
    }
    pages.len()
}


/// Topology-based page packing experiments.
///
/// Three experiments in one test:
/// 1. Layout comparison (cold): sequential / BFS / neighbor-run / heavy-edge
///    - Measure: recall, latency, unique_pages/q, phys/q, miss/q, hit/q
/// 2. Warm benchmark: same 4 layouts with pre-warmed cache
/// 3. Page-aware scheduling: B-sweep on BFS layout (B=1,4,8,16)
///    - disk_graph_search_pipe_v3_pagesched with pop_preferred
///
/// Holdout protocol: bench [200..300), warmup [300..320), disjoint.
#[test]
#[ignore] // EC2-only: BENCH_DIR + COHERE_DIR required
fn exp_topology_packing() {
    let max_n: usize = std::env::var("COHERE_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    let dataset_dir = std::env::var("COHERE_DIR").unwrap_or_else(|_| {
        let manifest = env!("CARGO_MANIFEST_DIR");
        format!("{}/../../data/cohere_100k", manifest)
    });

    let (vectors, queries_flat, ground_truth, n, nq, dim, k) =
        match load_dataset(&dataset_dir, max_n) {
            Some(d) => d,
            None => return,
        };

    let m_max = 32;
    let ef_construction = 200;
    let ef = 200;
    let prefetch_width = 4;
    let num_bench_queries = 100;
    let warmup_queries = 20;
    let cache_pct = 5usize;
    let bench_start = 200;
    let bench_end = bench_start + num_bench_queries;
    let warmup_start = bench_end;
    let warmup_end = warmup_start + warmup_queries;

    assert!(nq >= warmup_end,
        "Need {} queries but dataset has {}", warmup_end, nq);

    eprintln!("=== EXP-TOPOLOGY: Cohere {}K, dim={}, k={} ===", n / 1000, dim, k);
    eprintln!("  bench=[{}..{}), warmup=[{}..{})", bench_start, bench_end, warmup_start, warmup_end);

    // Build NSW index
    eprintln!("Building NSW index (n={}, m_max={}, ef_c={}) ...", n, m_max, ef_construction);
    let t0 = std::time::Instant::now();
    let config = NswConfig::new(m_max, ef_construction);
    let builder = NswBuilder::new(config, dim, MetricType::Cosine, n);
    for (i, v) in vectors.chunks_exact(dim).enumerate() {
        builder.insert(VectorId(i as u32), v);
    }
    let index = builder.build();
    eprintln!("  Index built in {:.1}s", t0.elapsed().as_secs_f64());

    let entry_ids: Vec<u32> = index.entry_set().iter().map(|v| v.0).collect();
    let entry_set: Vec<VectorId> = index.entry_set().to_vec();

    // Base dir for all layouts
    let bench_dir = std::env::var("BENCH_DIR").ok();
    let direct_io = bench_dir.is_some();
    let _tmpdir;
    let base_dir: std::path::PathBuf;
    if let Some(ref bd) = bench_dir {
        base_dir = std::path::PathBuf::from(bd).join("topology");
        std::fs::create_dir_all(&base_dir).unwrap();
    } else {
        _tmpdir = tempfile::tempdir().unwrap();
        base_dir = _tmpdir.path().to_path_buf();
    }

    // Write vectors.dat (shared)
    let vec_dir = base_dir.clone();
    let writer_base = IndexWriter::new(&vec_dir);
    writer_base.write(
        n as u32, dim, "cosine", index.max_degree(), ef_construction,
        &entry_ids, index.vectors_raw(), |vid| index.neighbors(vid),
    ).unwrap();
    let disk_vectors = load_vectors(&vec_dir.join("vectors.dat"), n, dim).unwrap();

    // Compute 4 reorders
    eprintln!("Computing reorders...");
    let t0 = std::time::Instant::now();

    // Sequential (identity)
    let seq_reorder: Vec<u32> = (0..n as u32).collect();

    // BFS
    let bfs_reorder_map = bfs_reorder_graph(n, &entry_ids, |vid| index.neighbors(vid));

    // Neighbor-run BFS
    let nbr_run_reorder_map = neighbor_run_reorder_graph(n, &entry_ids, |vid| index.neighbors(vid));

    // Heavy-edge (MARGO-style)
    let heavy_edge_reorder_map = heavy_edge_reorder_graph(n, |vid| index.neighbors(vid));

    eprintln!("  Reorders computed in {:.1}s", t0.elapsed().as_secs_f64());

    // Write all 4 layouts
    struct LayoutInfo {
        label: &'static str,
        dir: std::path::PathBuf,
        num_pages: usize,
        adj_index: Vec<AdjIndexEntry>,
    }

    let reorders: Vec<(&str, Vec<u32>)> = vec![
        ("sequential", seq_reorder),
        ("bfs", bfs_reorder_map),
        ("nbr_run", nbr_run_reorder_map),
        ("heavy_edge", heavy_edge_reorder_map),
    ];

    let mut layouts: Vec<LayoutInfo> = Vec::new();
    for (label, reorder) in &reorders {
        let layout_dir = base_dir.join(label);
        std::fs::create_dir_all(&layout_dir).unwrap();
        let writer = IndexWriter::new(&layout_dir);
        writer.write_v3(
            n as u32, dim, "cosine", index.max_degree(), ef_construction,
            &entry_ids, index.vectors_raw(), |vid| index.neighbors(vid),
            reorder,
            label,
        ).unwrap();
        std::fs::copy(vec_dir.join("vectors.dat"), layout_dir.join("vectors.dat")).unwrap();
        let meta = IndexMeta::load_from(&layout_dir.join("meta.json")).unwrap();
        let num_pages = meta.num_pages.unwrap_or(0) as usize;
        let adj_index = load_adj_index(
            &layout_dir.join("adj_index.dat"),
            meta.num_vectors as usize,
        ).unwrap();
        eprintln!("  {}: {} pages, written to {:?}", label, num_pages, layout_dir);
        layouts.push(LayoutInfo {
            label,
            dir: layout_dir,
            num_pages,
            adj_index,
        });
    }

    // Prepare queries
    let all_query_vecs: Vec<Vec<f32>> = queries_flat
        .chunks_exact(dim).take(warmup_end).map(|c| c.to_vec()).collect();
    let bench_queries = &all_query_vecs[bench_start..bench_end];
    let bench_gt = &ground_truth[bench_start..bench_end];
    let warmup_query_slice = &all_query_vecs[warmup_start..warmup_end];

    if !with_runtime(|rt| {
        rt.block_on(async {
            let fp32_bank = FP32SimdVectorBank::new(&disk_vectors, dim, MetricType::Cosine);

            // ===== Phase 1: Cold layout comparison =====
            eprintln!("\n--- Phase 1: Cold benchmark ({} queries [{}..{})) ---",
                num_bench_queries, bench_start, bench_end);
            eprintln!("{:>12} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7}",
                "layout", "recall", "p50ms", "p99ms", "QPS",
                "exp/q", "blk/q", "miss/q", "hit/q", "phys/q", "upg/q");

            for layout in &layouts {
                let dir_str = layout.dir.to_str().unwrap();
                let io = Rc::new(
                    IoDriver::open_pages(dir_str, dim, 64, direct_io)
                        .await
                        .expect("failed to open IO driver"),
                );
                let pool_pages = (layout.num_pages * cache_pct / 100).max(256);
                let pool_bytes = pool_pages * 4096;
                let pool = Rc::new(AdjacencyPool::new(pool_bytes));
                let handle = AdjacencyPool::spawn_prefetch_worker(
                    Rc::clone(&pool), Rc::clone(&io), prefetch_width,
                );

                let mut recalls = Vec::with_capacity(num_bench_queries);
                let mut latencies_ms = Vec::with_capacity(num_bench_queries);
                let mut sum_exp = 0u64;
                let mut sum_blk = 0u64;
                let mut sum_miss = 0u64;
                let mut sum_hit = 0u64;
                let mut sum_phys = 0u64;
                let mut sum_upg = 0u64;

                let wall_start = std::time::Instant::now();
                for qi in 0..num_bench_queries {
                    // Cold: clear cache
                    pool.pause_prefetch(true);
                    pool.drain_prefetch();
                    monoio::time::sleep(std::time::Duration::from_micros(50)).await;
                    while pool.has_loading() {
                        monoio::time::sleep(std::time::Duration::from_micros(100)).await;
                    }
                    pool.clear();
                    pool.pause_prefetch(false);

                    let mut perf = SearchPerfContext::default();
                    let mut query_trace = TraceRecorder::new();
                    let t0 = std::time::Instant::now();
                    let results = disk_graph_search_pipe_v3_traced(
                        &bench_queries[qi], &entry_set, k, ef, prefetch_width, 0, 0,
                        &pool, &io, &fp32_bank, &layout.adj_index,
                        &mut perf, PerfLevel::EnableTime, &mut query_trace,
                    ).await;
                    latencies_ms.push(t0.elapsed().as_secs_f64() * 1_000.0);

                    let ids: Vec<u32> = results.iter().map(|s| s.id.0).collect();
                    recalls.push(recall_at_k(&ids, &bench_gt[qi]));

                    sum_exp += perf.expansions;
                    sum_blk += perf.blocks_read;
                    sum_miss += perf.blocks_miss;
                    sum_hit += perf.blocks_hit;
                    sum_phys += perf.phys_reads;

                    let expanded: Vec<u32> = query_trace.node_counts.keys().copied().collect();
                    sum_upg += unique_pages_for_query(&expanded, &layout.adj_index) as u64;
                    query_trace = TraceRecorder::new();
                }
                let wall_secs = wall_start.elapsed().as_secs_f64();
                let nf = num_bench_queries as f64;

                let mean_recall = recalls.iter().sum::<f64>() / nf;
                let qps = nf / wall_secs;
                let mut sorted_lat = latencies_ms.clone();
                sorted_lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let p50 = percentile(&sorted_lat, 50.0);
                let p99 = percentile(&sorted_lat, 99.0);

                eprintln!("{:>12} {:>7.3} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1}",
                    layout.label, mean_recall, p50, p99, qps,
                    sum_exp as f64 / nf,
                    sum_blk as f64 / nf,
                    sum_miss as f64 / nf,
                    sum_hit as f64 / nf,
                    sum_phys as f64 / nf,
                    sum_upg as f64 / nf);

                pool.stop_prefetch();
                handle.await;
            }

            // ===== Phase 2: Warm layout comparison =====
            eprintln!("\n--- Phase 2: Warm benchmark (warmup [{}..{}), measure [{}..{})) ---",
                warmup_start, warmup_end, bench_start, bench_end);
            eprintln!("{:>12} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7}",
                "layout", "recall", "p50ms", "p99ms", "QPS",
                "exp/q", "blk/q", "miss/q", "hit/q", "phys/q");

            for layout in &layouts {
                let dir_str = layout.dir.to_str().unwrap();
                let io = Rc::new(
                    IoDriver::open_pages(dir_str, dim, 64, direct_io)
                        .await
                        .expect("failed to open IO driver"),
                );
                let pool_pages = (layout.num_pages * cache_pct / 100).max(256);
                let pool_bytes = pool_pages * 4096;
                let pool = Rc::new(AdjacencyPool::new(pool_bytes));
                let handle = AdjacencyPool::spawn_prefetch_worker(
                    Rc::clone(&pool), Rc::clone(&io), prefetch_width,
                );

                // Warmup on disjoint queries
                for q in warmup_query_slice {
                    let mut perf = SearchPerfContext::default();
                    disk_graph_search_pipe_v3(
                        q, &entry_set, k, ef, prefetch_width, 0, 0,
                        &pool, &io, &fp32_bank, &layout.adj_index, &mut perf, PerfLevel::CountOnly,
                    ).await;
                }

                // Measure
                let mut recalls = Vec::with_capacity(num_bench_queries);
                let mut latencies_ms = Vec::with_capacity(num_bench_queries);
                let mut sum_exp = 0u64;
                let mut sum_blk = 0u64;
                let mut sum_miss = 0u64;
                let mut sum_hit = 0u64;
                let mut sum_phys = 0u64;

                let wall_start = std::time::Instant::now();
                for qi in 0..num_bench_queries {
                    let mut perf = SearchPerfContext::default();
                    let t0 = std::time::Instant::now();
                    let results = disk_graph_search_pipe_v3(
                        &bench_queries[qi], &entry_set, k, ef, prefetch_width, 0, 0,
                        &pool, &io, &fp32_bank, &layout.adj_index, &mut perf, PerfLevel::EnableTime,
                    ).await;
                    latencies_ms.push(t0.elapsed().as_secs_f64() * 1_000.0);

                    let ids: Vec<u32> = results.iter().map(|s| s.id.0).collect();
                    recalls.push(recall_at_k(&ids, &bench_gt[qi]));

                    sum_exp += perf.expansions;
                    sum_blk += perf.blocks_read;
                    sum_miss += perf.blocks_miss;
                    sum_hit += perf.blocks_hit;
                    sum_phys += perf.phys_reads;
                }
                let wall_secs = wall_start.elapsed().as_secs_f64();
                let nf = num_bench_queries as f64;

                let mean_recall = recalls.iter().sum::<f64>() / nf;
                let qps = nf / wall_secs;
                let mut sorted_lat = latencies_ms.clone();
                sorted_lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let p50 = percentile(&sorted_lat, 50.0);
                let p99 = percentile(&sorted_lat, 99.0);

                eprintln!("{:>12} {:>7.3} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1}",
                    layout.label, mean_recall, p50, p99, qps,
                    sum_exp as f64 / nf,
                    sum_blk as f64 / nf,
                    sum_miss as f64 / nf,
                    sum_hit as f64 / nf,
                    sum_phys as f64 / nf);

                let cache_snap = pool.stats();
                eprintln!("    cache: hits={} misses={} bypasses={} evictions={} pf_hits={} phys={}",
                    cache_snap.hits, cache_snap.misses, cache_snap.bypasses,
                    cache_snap.evictions, cache_snap.prefetch_hits, cache_snap.phys_reads);

                pool.stop_prefetch();
                handle.await;
            }

            // ===== Phase 3: Page-aware scheduling B-sweep on BFS layout =====
            eprintln!("\n--- Phase 3: Page-aware scheduling B-sweep (BFS layout, warm) ---");
            eprintln!("{:>8} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7}",
                "B", "recall", "p50ms", "p99ms", "QPS",
                "exp/q", "blk/q", "miss/q", "hit/q", "phys/q", "sched/q");

            let bfs_layout = &layouts[1]; // BFS is index 1
            for b in [1usize, 4, 8, 16] {
                let dir_str = bfs_layout.dir.to_str().unwrap();
                let io = Rc::new(
                    IoDriver::open_pages(dir_str, dim, 64, direct_io)
                        .await
                        .expect("failed to open IO driver"),
                );
                let pool_pages = (bfs_layout.num_pages * cache_pct / 100).max(256);
                let pool_bytes = pool_pages * 4096;
                let pool = Rc::new(AdjacencyPool::new(pool_bytes));
                let handle = AdjacencyPool::spawn_prefetch_worker(
                    Rc::clone(&pool), Rc::clone(&io), prefetch_width,
                );

                // Warmup
                for q in warmup_query_slice {
                    let mut perf = SearchPerfContext::default();
                    disk_graph_search_pipe_v3_pagesched(
                        q, &entry_set, k, ef, prefetch_width, 0, 0,
                        &pool, &io, &fp32_bank, &bfs_layout.adj_index,
                        &mut perf, PerfLevel::CountOnly, b,
                    ).await;
                }

                // Measure
                let mut recalls = Vec::with_capacity(num_bench_queries);
                let mut latencies_ms = Vec::with_capacity(num_bench_queries);
                let mut sum_exp = 0u64;
                let mut sum_blk = 0u64;
                let mut sum_miss = 0u64;
                let mut sum_hit = 0u64;
                let mut sum_phys = 0u64;
                let mut sum_sched = 0u64;

                let wall_start = std::time::Instant::now();
                for qi in 0..num_bench_queries {
                    let mut perf = SearchPerfContext::default();
                    let t0 = std::time::Instant::now();
                    let results = disk_graph_search_pipe_v3_pagesched(
                        &bench_queries[qi], &entry_set, k, ef, prefetch_width, 0, 0,
                        &pool, &io, &fp32_bank, &bfs_layout.adj_index,
                        &mut perf, PerfLevel::EnableTime, b,
                    ).await;
                    latencies_ms.push(t0.elapsed().as_secs_f64() * 1_000.0);

                    let ids: Vec<u32> = results.iter().map(|s| s.id.0).collect();
                    recalls.push(recall_at_k(&ids, &bench_gt[qi]));

                    sum_exp += perf.expansions;
                    sum_blk += perf.blocks_read;
                    sum_miss += perf.blocks_miss;
                    sum_hit += perf.blocks_hit;
                    sum_phys += perf.phys_reads;
                    sum_sched += perf.page_sched_hits;
                }
                let wall_secs = wall_start.elapsed().as_secs_f64();
                let nf = num_bench_queries as f64;

                let mean_recall = recalls.iter().sum::<f64>() / nf;
                let qps = nf / wall_secs;
                let mut sorted_lat = latencies_ms.clone();
                sorted_lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let p50 = percentile(&sorted_lat, 50.0);
                let p99 = percentile(&sorted_lat, 99.0);

                eprintln!("{:>8} {:>7.3} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1}",
                    format!("B={}", b), mean_recall, p50, p99, qps,
                    sum_exp as f64 / nf,
                    sum_blk as f64 / nf,
                    sum_miss as f64 / nf,
                    sum_hit as f64 / nf,
                    sum_phys as f64 / nf,
                    sum_sched as f64 / nf);

                pool.stop_prefetch();
                handle.await;
            }

            // ===== Phase 4: Combined best-layout + B=8 scheduling =====
            eprintln!("\n--- Phase 4: Combined (each layout + B=8 scheduling, warm) ---");
            eprintln!("{:>16} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7}",
                "layout+B", "recall", "p50ms", "p99ms", "QPS",
                "exp/q", "blk/q", "miss/q", "hit/q", "phys/q", "sched/q");

            for layout in &layouts {
                let dir_str = layout.dir.to_str().unwrap();
                let io = Rc::new(
                    IoDriver::open_pages(dir_str, dim, 64, direct_io)
                        .await
                        .expect("failed to open IO driver"),
                );
                let pool_pages = (layout.num_pages * cache_pct / 100).max(256);
                let pool_bytes = pool_pages * 4096;
                let pool = Rc::new(AdjacencyPool::new(pool_bytes));
                let handle = AdjacencyPool::spawn_prefetch_worker(
                    Rc::clone(&pool), Rc::clone(&io), prefetch_width,
                );

                // Warmup
                for q in warmup_query_slice {
                    let mut perf = SearchPerfContext::default();
                    disk_graph_search_pipe_v3_pagesched(
                        q, &entry_set, k, ef, prefetch_width, 0, 0,
                        &pool, &io, &fp32_bank, &layout.adj_index,
                        &mut perf, PerfLevel::CountOnly, 8,
                    ).await;
                }

                // Measure
                let mut recalls = Vec::with_capacity(num_bench_queries);
                let mut latencies_ms = Vec::with_capacity(num_bench_queries);
                let mut sum_exp = 0u64;
                let mut sum_blk = 0u64;
                let mut sum_miss = 0u64;
                let mut sum_hit = 0u64;
                let mut sum_phys = 0u64;
                let mut sum_sched = 0u64;

                let wall_start = std::time::Instant::now();
                for qi in 0..num_bench_queries {
                    let mut perf = SearchPerfContext::default();
                    let t0 = std::time::Instant::now();
                    let results = disk_graph_search_pipe_v3_pagesched(
                        &bench_queries[qi], &entry_set, k, ef, prefetch_width, 0, 0,
                        &pool, &io, &fp32_bank, &layout.adj_index,
                        &mut perf, PerfLevel::EnableTime, 8,
                    ).await;
                    latencies_ms.push(t0.elapsed().as_secs_f64() * 1_000.0);

                    let ids: Vec<u32> = results.iter().map(|s| s.id.0).collect();
                    recalls.push(recall_at_k(&ids, &bench_gt[qi]));

                    sum_exp += perf.expansions;
                    sum_blk += perf.blocks_read;
                    sum_miss += perf.blocks_miss;
                    sum_hit += perf.blocks_hit;
                    sum_phys += perf.phys_reads;
                    sum_sched += perf.page_sched_hits;
                }
                let wall_secs = wall_start.elapsed().as_secs_f64();
                let nf = num_bench_queries as f64;

                let mean_recall = recalls.iter().sum::<f64>() / nf;
                let qps = nf / wall_secs;
                let mut sorted_lat = latencies_ms.clone();
                sorted_lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let p50 = percentile(&sorted_lat, 50.0);
                let p99 = percentile(&sorted_lat, 99.0);

                eprintln!("{:>16} {:>7.3} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1}",
                    format!("{}+B8", layout.label), mean_recall, p50, p99, qps,
                    sum_exp as f64 / nf,
                    sum_blk as f64 / nf,
                    sum_miss as f64 / nf,
                    sum_hit as f64 / nf,
                    sum_phys as f64 / nf,
                    sum_sched as f64 / nf);

                pool.stop_prefetch();
                handle.await;
            }
        });
    }) {
        eprintln!("Skipped: io_uring not available");
    }
}

/// Sensitivity sweep: heavy_edge vs BFS across pool_pages, W, ef.
///
/// Two passes per grid point:
///   - perq-cold: pool.clear() each query (layout truth — phys/q tracks upg/q)
///   - warm: disjoint warmup [300..320), then measure without clearing
///
/// W=0 is real no-prefetch (no worker spawned, prefetch_window=0).
/// Pool sizes are explicit page counts (not %, which clamps to 256 at 100K).
#[test]
#[ignore] // EC2-only: BENCH_DIR + COHERE_DIR required
fn exp_heavy_edge_sweep() {
    let max_n: usize = std::env::var("COHERE_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    let dataset_dir = std::env::var("COHERE_DIR").unwrap_or_else(|_| {
        let manifest = env!("CARGO_MANIFEST_DIR");
        format!("{}/../../data/cohere_100k", manifest)
    });

    let (vectors, queries_flat, ground_truth, n, nq, dim, k) =
        match load_dataset(&dataset_dir, max_n) {
            Some(d) => d,
            None => return,
        };

    let m_max = 32;
    let ef_construction = 200;
    let num_bench_queries = 100;
    let warmup_queries = 20;
    let bench_start = 200;
    let bench_end = bench_start + num_bench_queries;
    let warmup_start = bench_end;
    let warmup_end = warmup_start + warmup_queries;

    assert!(nq >= warmup_end, "Need {} queries but dataset has {}", warmup_end, nq);

    eprintln!("=== EXP-HEAVY-EDGE-SWEEP v2: Cohere {}K, dim={}, k={} ===", n / 1000, dim, k);

    // Build NSW index
    eprintln!("Building NSW index (n={}, m_max={}, ef_c={}) ...", n, m_max, ef_construction);
    let t0 = std::time::Instant::now();
    let config = NswConfig::new(m_max, ef_construction);
    let builder = NswBuilder::new(config, dim, MetricType::Cosine, n);
    for (i, v) in vectors.chunks_exact(dim).enumerate() {
        builder.insert(VectorId(i as u32), v);
    }
    let index = builder.build();
    eprintln!("  Index built in {:.1}s", t0.elapsed().as_secs_f64());

    let entry_ids: Vec<u32> = index.entry_set().iter().map(|v| v.0).collect();
    let entry_set: Vec<VectorId> = index.entry_set().to_vec();

    // Base dir
    let bench_dir = std::env::var("BENCH_DIR").ok();
    let direct_io = bench_dir.is_some();
    let _tmpdir;
    let base_dir: std::path::PathBuf;
    if let Some(ref bd) = bench_dir {
        base_dir = std::path::PathBuf::from(bd).join("he_sweep_v2");
        std::fs::create_dir_all(&base_dir).unwrap();
    } else {
        _tmpdir = tempfile::tempdir().unwrap();
        base_dir = _tmpdir.path().to_path_buf();
    }

    // Write vectors.dat (shared)
    let vec_dir = base_dir.clone();
    let writer_base = IndexWriter::new(&vec_dir);
    writer_base.write(
        n as u32, dim, "cosine", index.max_degree(), ef_construction,
        &entry_ids, index.vectors_raw(), |vid| index.neighbors(vid),
    ).unwrap();
    let disk_vectors = load_vectors(&vec_dir.join("vectors.dat"), n, dim).unwrap();

    // BFS layout
    eprintln!("Computing BFS reorder...");
    let bfs_reorder_map = bfs_reorder_graph(n, &entry_ids, |vid| index.neighbors(vid));
    let bfs_dir = base_dir.join("bfs");
    std::fs::create_dir_all(&bfs_dir).unwrap();
    let bfs_writer = IndexWriter::new(&bfs_dir);
    bfs_writer.write_v3(
        n as u32, dim, "cosine", index.max_degree(), ef_construction,
        &entry_ids, index.vectors_raw(), |vid| index.neighbors(vid),
        &bfs_reorder_map,
        "bfs",
    ).unwrap();
    std::fs::copy(vec_dir.join("vectors.dat"), bfs_dir.join("vectors.dat")).unwrap();
    let bfs_meta = IndexMeta::load_from(&bfs_dir.join("meta.json")).unwrap();
    let bfs_num_pages = bfs_meta.num_pages.unwrap_or(0) as usize;
    let bfs_adj_index = load_adj_index(&bfs_dir.join("adj_index.dat"), n).unwrap();
    eprintln!("  BFS: {} pages", bfs_num_pages);

    // Heavy-edge layout
    eprintln!("Computing heavy_edge reorder...");
    let t0 = std::time::Instant::now();
    let he_reorder_map = heavy_edge_reorder_graph(n, |vid| index.neighbors(vid));
    eprintln!("  heavy_edge reorder computed in {:.1}s", t0.elapsed().as_secs_f64());
    let he_dir = base_dir.join("heavy_edge");
    std::fs::create_dir_all(&he_dir).unwrap();
    let he_writer = IndexWriter::new(&he_dir);
    he_writer.write_v3(
        n as u32, dim, "cosine", index.max_degree(), ef_construction,
        &entry_ids, index.vectors_raw(), |vid| index.neighbors(vid),
        &he_reorder_map,
        "heavy_edge",
    ).unwrap();
    std::fs::copy(vec_dir.join("vectors.dat"), he_dir.join("vectors.dat")).unwrap();
    let he_meta = IndexMeta::load_from(&he_dir.join("meta.json")).unwrap();
    let he_num_pages = he_meta.num_pages.unwrap_or(0) as usize;
    let he_adj_index = load_adj_index(&he_dir.join("adj_index.dat"), n).unwrap();
    eprintln!("  heavy_edge: {} pages", he_num_pages);

    // Queries
    let all_query_vecs: Vec<Vec<f32>> = queries_flat
        .chunks_exact(dim).take(warmup_end).map(|c| c.to_vec()).collect();
    let bench_queries = &all_query_vecs[bench_start..bench_end];
    let bench_gt = &ground_truth[bench_start..bench_end];
    let warmup_query_slice = &all_query_vecs[warmup_start..warmup_end];

    let bfs_dir_str = bfs_dir.to_str().unwrap().to_owned();
    let he_dir_str = he_dir.to_str().unwrap().to_owned();

    if !with_runtime(|rt| {
        rt.block_on(async {
            let fp32_bank = FP32SimdVectorBank::new(&disk_vectors, dim, MetricType::Cosine);

            // Sweep grid — explicit pool sizes (not %, which clamps at 100K)
            let pool_page_counts = [256usize, 512, 1024];
            let ws = [0usize, 2, 4, 8];
            let efs = [150usize, 200, 250];

            // ===== Pass 1: Per-query cold (layout truth) =====
            eprintln!("\n--- Per-query cold (pool.clear() each query) ---");
            eprintln!("{:>6} {:>3} {:>2} {:>4} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7}",
                "pool", "ef", "W", "lay", "recall", "p50ms", "p99ms", "QPS",
                "exp/q", "miss/q", "hit/q", "phys/q");
            eprintln!("{}", "-".repeat(100));

            for &pool_pg in &pool_page_counts {
                for &ef in &efs {
                    for &w in &ws {
                        for (label, adj_index_ref, dir_str) in [
                            ("BFS", &bfs_adj_index[..], bfs_dir_str.as_str()),
                            ("HE", &he_adj_index[..], he_dir_str.as_str()),
                        ] {
                            let io = Rc::new(
                                IoDriver::open_pages(dir_str, dim, 64, direct_io)
                                    .await
                                    .expect("failed to open IO driver"),
                            );
                            let pool_bytes = pool_pg * 4096;
                            let pool = Rc::new(AdjacencyPool::new(pool_bytes));

                            // W=0: no prefetch worker, prefetch_window=0
                            let handle = if w > 0 {
                                Some(AdjacencyPool::spawn_prefetch_worker(
                                    Rc::clone(&pool), Rc::clone(&io), w,
                                ))
                            } else {
                                None
                            };

                            let mut recalls = Vec::with_capacity(num_bench_queries);
                            let mut latencies_ms = Vec::with_capacity(num_bench_queries);
                            let mut sum_exp = 0u64;
                            let mut sum_miss = 0u64;
                            let mut sum_hit = 0u64;
                            let mut sum_phys = 0u64;

                            let wall_start = std::time::Instant::now();
                            for qi in 0..num_bench_queries {
                                // Clear cache for cold measurement
                                if w > 0 {
                                    pool.pause_prefetch(true);
                                    pool.drain_prefetch();
                                    monoio::time::sleep(std::time::Duration::from_micros(50)).await;
                                    while pool.has_loading() {
                                        monoio::time::sleep(std::time::Duration::from_micros(100)).await;
                                    }
                                }
                                pool.clear();
                                if w > 0 {
                                    pool.pause_prefetch(false);
                                }

                                let mut perf = SearchPerfContext::default();
                                let t0 = std::time::Instant::now();
                                let results = disk_graph_search_pipe_v3(
                                    &bench_queries[qi], &entry_set, k, ef, w, 0, 0,
                                    &pool, &io, &fp32_bank, adj_index_ref,
                                    &mut perf, PerfLevel::EnableTime,
                                ).await;
                                latencies_ms.push(t0.elapsed().as_secs_f64() * 1_000.0);

                                let ids: Vec<u32> = results.iter().map(|s| s.id.0).collect();
                                recalls.push(recall_at_k(&ids, &bench_gt[qi]));

                                sum_exp += perf.expansions;
                                sum_miss += perf.blocks_miss;
                                sum_hit += perf.blocks_hit;
                                sum_phys += perf.phys_reads;
                            }
                            let wall_secs = wall_start.elapsed().as_secs_f64();
                            let nf = num_bench_queries as f64;

                            let mean_recall = recalls.iter().sum::<f64>() / nf;
                            let qps = nf / wall_secs;
                            let mut sorted_lat = latencies_ms.clone();
                            sorted_lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
                            let p50 = percentile(&sorted_lat, 50.0);
                            let p99 = percentile(&sorted_lat, 99.0);

                            eprintln!("{:>6} {:>3} {:>2} {:>4} {:>7.3} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1}",
                                pool_pg, ef, w, label, mean_recall, p50, p99, qps,
                                sum_exp as f64 / nf,
                                sum_miss as f64 / nf,
                                sum_hit as f64 / nf,
                                sum_phys as f64 / nf);

                            pool.stop_prefetch();
                            if let Some(h) = handle {
                                h.await;
                            }
                        }
                    }
                }
            }

            // ===== Pass 2: Warm (disjoint warmup, then measure) =====
            eprintln!("\n--- Warm (warmup [{}..{}), measure [{}..{})) ---",
                warmup_start, warmup_end, bench_start, bench_end);
            eprintln!("{:>6} {:>3} {:>2} {:>4} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7}",
                "pool", "ef", "W", "lay", "recall", "p50ms", "p99ms", "QPS",
                "exp/q", "miss/q", "hit/q", "phys/q");
            eprintln!("{}", "-".repeat(100));

            for &pool_pg in &pool_page_counts {
                for &ef in &efs {
                    for &w in &ws {
                        for (label, adj_index_ref, dir_str) in [
                            ("BFS", &bfs_adj_index[..], bfs_dir_str.as_str()),
                            ("HE", &he_adj_index[..], he_dir_str.as_str()),
                        ] {
                            let io = Rc::new(
                                IoDriver::open_pages(dir_str, dim, 64, direct_io)
                                    .await
                                    .expect("failed to open IO driver"),
                            );
                            let pool_bytes = pool_pg * 4096;
                            let pool = Rc::new(AdjacencyPool::new(pool_bytes));

                            let handle = if w > 0 {
                                Some(AdjacencyPool::spawn_prefetch_worker(
                                    Rc::clone(&pool), Rc::clone(&io), w,
                                ))
                            } else {
                                None
                            };

                            // Warmup on disjoint queries
                            for q in warmup_query_slice {
                                let mut perf = SearchPerfContext::default();
                                disk_graph_search_pipe_v3(
                                    q, &entry_set, k, ef, w, 0, 0,
                                    &pool, &io, &fp32_bank, adj_index_ref,
                                    &mut perf, PerfLevel::CountOnly,
                                ).await;
                            }

                            // Measure (no clear between queries)
                            let mut recalls = Vec::with_capacity(num_bench_queries);
                            let mut latencies_ms = Vec::with_capacity(num_bench_queries);
                            let mut sum_exp = 0u64;
                            let mut sum_miss = 0u64;
                            let mut sum_hit = 0u64;
                            let mut sum_phys = 0u64;

                            let wall_start = std::time::Instant::now();
                            for qi in 0..num_bench_queries {
                                let mut perf = SearchPerfContext::default();
                                let t0 = std::time::Instant::now();
                                let results = disk_graph_search_pipe_v3(
                                    &bench_queries[qi], &entry_set, k, ef, w, 0, 0,
                                    &pool, &io, &fp32_bank, adj_index_ref,
                                    &mut perf, PerfLevel::EnableTime,
                                ).await;
                                latencies_ms.push(t0.elapsed().as_secs_f64() * 1_000.0);

                                let ids: Vec<u32> = results.iter().map(|s| s.id.0).collect();
                                recalls.push(recall_at_k(&ids, &bench_gt[qi]));

                                sum_exp += perf.expansions;
                                sum_miss += perf.blocks_miss;
                                sum_hit += perf.blocks_hit;
                                sum_phys += perf.phys_reads;
                            }
                            let wall_secs = wall_start.elapsed().as_secs_f64();
                            let nf = num_bench_queries as f64;

                            let mean_recall = recalls.iter().sum::<f64>() / nf;
                            let qps = nf / wall_secs;
                            let mut sorted_lat = latencies_ms.clone();
                            sorted_lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
                            let p50 = percentile(&sorted_lat, 50.0);
                            let p99 = percentile(&sorted_lat, 99.0);

                            eprintln!("{:>6} {:>3} {:>2} {:>4} {:>7.3} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1}",
                                pool_pg, ef, w, label, mean_recall, p50, p99, qps,
                                sum_exp as f64 / nf,
                                sum_miss as f64 / nf,
                                sum_hit as f64 / nf,
                                sum_phys as f64 / nf);

                            pool.stop_prefetch();
                            if let Some(h) = handle {
                                h.await;
                            }
                        }
                    }
                }
            }
        });
    }) {
        eprintln!("Skipped: io_uring not available");
    }
}

// =============================================================================
// EXP-VELOANN-PHASE1: Co-Resident Caching + Cache-Aware Beam Search
// =============================================================================
//
// Two-phase experiment:
// 1. exp_veloann_phase1_build — builds NSW + heavy_edge layout to BENCH_DIR (once)
// 2. exp_veloann_phase1_sweep — loads pre-built artifacts, sweeps sched_b (fast)
//
// Co-resident record caching is IMPLICIT (AdjacencyPool caches by page_id).
// Heavy_edge layout ensures co-located records are graph neighbors.

/// Build NSW index + heavy_edge layout to BENCH_DIR. Run once, reuse for sweeps.
/// Env: COHERE_DIR, COHERE_N, BENCH_DIR (required).
#[test]
#[ignore]
fn exp_veloann_phase1_build() {
    let max_n: usize = std::env::var("COHERE_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);

    let dataset_dir = std::env::var("COHERE_DIR").unwrap_or_else(|_| {
        let manifest = env!("CARGO_MANIFEST_DIR");
        format!("{}/../../data/cohere_100k", manifest)
    });

    let (vectors, _queries_flat, _ground_truth, n, _nq, dim, _k) =
        match load_dataset(&dataset_dir, max_n) {
            Some(d) => d,
            None => return,
        };

    let bench_dir = std::env::var("BENCH_DIR").expect("BENCH_DIR required for build");
    let base_dir = std::path::PathBuf::from(&bench_dir).join("veloann_phase1");
    let layout_dir = base_dir.join("heavy_edge");

    // Check if already built
    if layout_dir.join("adj_index.dat").exists() && layout_dir.join("vectors.dat").exists() {
        let meta = IndexMeta::load_from(&layout_dir.join("meta.json")).unwrap();
        eprintln!("Already built: heavy_edge {} pages, skipping rebuild", meta.num_pages.unwrap_or(0));
        return;
    }

    let m_max = 32;
    let ef_construction = 200;

    eprintln!("=== BUILD: Cohere {}K, dim={}, m_max={}, ef_c={} ===", n / 1000, dim, m_max, ef_construction);

    // Build NSW index
    eprintln!("Building NSW index ...");
    let t0 = std::time::Instant::now();
    let config = NswConfig::new(m_max, ef_construction);
    let builder = NswBuilder::new(config, dim, MetricType::Cosine, n);
    for (i, v) in vectors.chunks_exact(dim).enumerate() {
        builder.insert(VectorId(i as u32), v);
    }
    let index = builder.build();
    eprintln!("  Index built in {:.1}s", t0.elapsed().as_secs_f64());

    let entry_ids: Vec<u32> = index.entry_set().iter().map(|v| v.0).collect();

    // Write base index (vectors.dat + meta.json)
    std::fs::create_dir_all(&base_dir).unwrap();
    let writer_base = IndexWriter::new(&base_dir);
    writer_base.write(
        n as u32, dim, "cosine", index.max_degree(), ef_construction,
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
        n as u32, dim, "cosine", index.max_degree(), ef_construction,
        &entry_ids, index.vectors_raw(), |vid| index.neighbors(vid),
        &he_reorder, "heavy_edge",
    ).unwrap();
    std::fs::copy(base_dir.join("vectors.dat"), layout_dir.join("vectors.dat")).unwrap();

    let meta = IndexMeta::load_from(&layout_dir.join("meta.json")).unwrap();
    eprintln!("  heavy_edge: {} pages — DONE", meta.num_pages.unwrap_or(0));
}

/// Sweep page_sched_b on pre-built heavy_edge layout. Fast (~1 min).
/// Env: COHERE_DIR, COHERE_N, BENCH_DIR (required).
#[test]
#[ignore]
fn exp_veloann_phase1_sweep() {
    let max_n: usize = std::env::var("COHERE_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);

    let dataset_dir = std::env::var("COHERE_DIR").unwrap_or_else(|_| {
        let manifest = env!("CARGO_MANIFEST_DIR");
        format!("{}/../../data/cohere_100k", manifest)
    });

    let (_vectors, queries_flat, ground_truth, n, nq, dim, k) =
        match load_dataset(&dataset_dir, max_n) {
            Some(d) => d,
            None => return,
        };

    let ef = 200;
    let prefetch_width = 4;
    let num_bench_queries = 100;
    let warmup_queries = 20;
    let cache_pct = 5usize;
    let sched_b_values: &[usize] = &[0, 2, 4, 8];

    let total_queries_needed = num_bench_queries + warmup_queries;
    assert!(nq >= total_queries_needed, "Need {} queries but dataset has {}", total_queries_needed, nq);

    // Load pre-built artifacts
    let bench_dir = std::env::var("BENCH_DIR").expect("BENCH_DIR required");
    let base_dir = std::path::PathBuf::from(&bench_dir).join("veloann_phase1");
    let layout_dir = base_dir.join("heavy_edge");

    assert!(
        layout_dir.join("adj_index.dat").exists(),
        "Run exp_veloann_phase1_build first! Missing: {}/adj_index.dat",
        layout_dir.display()
    );

    let disk_vectors = load_vectors(&layout_dir.join("vectors.dat"), n, dim).unwrap();
    let meta = IndexMeta::load_from(&layout_dir.join("meta.json")).unwrap();
    let num_pages = meta.num_pages.unwrap_or(0) as usize;
    let adj_index = load_adj_index(&layout_dir.join("adj_index.dat"), n).unwrap();
    let entry_set: Vec<VectorId> = meta.entry_set.iter().map(|&v| VectorId(v)).collect();

    eprintln!(
        "=== SWEEP: Cohere {}K, dim={}, k={}, ef={}, W={}, cache={}%, heavy_edge {} pages ===",
        n / 1000, dim, k, ef, prefetch_width, cache_pct, num_pages
    );

    // Query slices
    let bench_queries: Vec<Vec<f32>> = queries_flat
        .chunks_exact(dim).take(num_bench_queries).map(|c| c.to_vec()).collect();
    let bench_gt: Vec<&Vec<u32>> = ground_truth.iter().take(num_bench_queries).collect();
    let warmup_vecs: Vec<Vec<f32>> = queries_flat
        .chunks_exact(dim).skip(num_bench_queries).take(warmup_queries).map(|c| c.to_vec()).collect();

    let dir_str = layout_dir.to_str().unwrap().to_owned();
    let direct_io = true;

    if !with_runtime(|rt| {
        rt.block_on(async {
            let fp32_bank = FP32SimdVectorBank::new(&disk_vectors, dim, MetricType::Cosine);

            eprintln!(
                "\n{:>8} {:>6} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>12}",
                "mode", "sched_b", "recall", "p50ms", "p99ms", "QPS",
                "exp/q", "blk/q", "mis/q", "hit/q", "phy/q", "sched_hits/q"
            );

            for &sched_b in sched_b_values {
                // === COLD mode ===
                {
                    let io = Rc::new(
                        IoDriver::open_pages(&dir_str, dim, 64, direct_io)
                            .await.expect("failed to open IO driver"),
                    );
                    let pool_pages = (num_pages * cache_pct / 100).max(256);
                    let pool = Rc::new(AdjacencyPool::new(pool_pages * 4096));
                    let handle = AdjacencyPool::spawn_prefetch_worker(
                        Rc::clone(&pool), Rc::clone(&io), prefetch_width,
                    );

                    let mut recalls = Vec::with_capacity(num_bench_queries);
                    let mut latencies_ms = Vec::with_capacity(num_bench_queries);
                    let mut sum_exp = 0u64;
                    let mut sum_blk = 0u64;
                    let mut sum_miss = 0u64;
                    let mut sum_hit = 0u64;
                    let mut sum_phys = 0u64;
                    let mut sum_sched = 0u64;

                    let wall_start = std::time::Instant::now();
                    for qi in 0..num_bench_queries {
                        pool.pause_prefetch(true);
                        pool.drain_prefetch();
                        monoio::time::sleep(std::time::Duration::from_micros(50)).await;
                        while pool.has_loading() {
                            monoio::time::sleep(std::time::Duration::from_micros(100)).await;
                        }
                        pool.clear();
                        pool.pause_prefetch(false);

                        let mut perf = SearchPerfContext::default();
                        let t0 = std::time::Instant::now();
                        let results = disk_graph_search_pipe_v3_pagesched(
                            &bench_queries[qi], &entry_set, k, ef, prefetch_width, 0, 0,
                            &pool, &io, &fp32_bank, &adj_index,
                            &mut perf, PerfLevel::EnableTime, sched_b,
                        ).await;
                        latencies_ms.push(t0.elapsed().as_secs_f64() * 1_000.0);

                        let ids: Vec<u32> = results.iter().map(|s| s.id.0).collect();
                        recalls.push(recall_at_k(&ids, bench_gt[qi]));

                        sum_exp += perf.expansions;
                        sum_blk += perf.blocks_read;
                        sum_miss += perf.blocks_miss;
                        sum_hit += perf.blocks_hit;
                        sum_phys += perf.phys_reads;
                        sum_sched += perf.page_sched_hits;
                    }
                    let wall_secs = wall_start.elapsed().as_secs_f64();
                    let nq = num_bench_queries as f64;
                    let avg_recall: f64 = recalls.iter().sum::<f64>() / nq;
                    latencies_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let p50 = latencies_ms[(nq * 0.50) as usize];
                    let p99 = latencies_ms[((nq * 0.99) as usize).min(num_bench_queries - 1)];

                    eprintln!(
                        "{:>8} {:>6} {:>7.3} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>12.1}",
                        "cold", sched_b, avg_recall, p50, p99, nq / wall_secs,
                        sum_exp as f64 / nq, sum_blk as f64 / nq, sum_miss as f64 / nq,
                        sum_hit as f64 / nq, sum_phys as f64 / nq, sum_sched as f64 / nq,
                    );

                    pool.stop_prefetch();
                    handle.await;
                }

                // === WARM mode ===
                {
                    let io = Rc::new(
                        IoDriver::open_pages(&dir_str, dim, 64, direct_io)
                            .await.expect("failed to open IO driver"),
                    );
                    let pool_pages = (num_pages * cache_pct / 100).max(256);
                    let pool = Rc::new(AdjacencyPool::new(pool_pages * 4096));
                    let handle = AdjacencyPool::spawn_prefetch_worker(
                        Rc::clone(&pool), Rc::clone(&io), prefetch_width,
                    );

                    // Warmup
                    for qi in 0..warmup_queries {
                        let mut perf = SearchPerfContext::default();
                        disk_graph_search_pipe_v3_pagesched(
                            &warmup_vecs[qi], &entry_set, k, ef, prefetch_width, 0, 0,
                            &pool, &io, &fp32_bank, &adj_index,
                            &mut perf, PerfLevel::CountOnly, sched_b,
                        ).await;
                    }

                    let mut recalls = Vec::with_capacity(num_bench_queries);
                    let mut latencies_ms = Vec::with_capacity(num_bench_queries);
                    let mut sum_exp = 0u64;
                    let mut sum_blk = 0u64;
                    let mut sum_miss = 0u64;
                    let mut sum_hit = 0u64;
                    let mut sum_phys = 0u64;
                    let mut sum_sched = 0u64;

                    let wall_start = std::time::Instant::now();
                    for qi in 0..num_bench_queries {
                        let mut perf = SearchPerfContext::default();
                        let t0 = std::time::Instant::now();
                        let results = disk_graph_search_pipe_v3_pagesched(
                            &bench_queries[qi], &entry_set, k, ef, prefetch_width, 0, 0,
                            &pool, &io, &fp32_bank, &adj_index,
                            &mut perf, PerfLevel::EnableTime, sched_b,
                        ).await;
                        latencies_ms.push(t0.elapsed().as_secs_f64() * 1_000.0);

                        let ids: Vec<u32> = results.iter().map(|s| s.id.0).collect();
                        recalls.push(recall_at_k(&ids, bench_gt[qi]));

                        sum_exp += perf.expansions;
                        sum_blk += perf.blocks_read;
                        sum_miss += perf.blocks_miss;
                        sum_hit += perf.blocks_hit;
                        sum_phys += perf.phys_reads;
                        sum_sched += perf.page_sched_hits;
                    }
                    let wall_secs = wall_start.elapsed().as_secs_f64();
                    let nq = num_bench_queries as f64;
                    let avg_recall: f64 = recalls.iter().sum::<f64>() / nq;
                    latencies_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let p50 = latencies_ms[(nq * 0.50) as usize];
                    let p99 = latencies_ms[((nq * 0.99) as usize).min(num_bench_queries - 1)];

                    eprintln!(
                        "{:>8} {:>6} {:>7.3} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>12.1}",
                        "warm", sched_b, avg_recall, p50, p99, nq / wall_secs,
                        sum_exp as f64 / nq, sum_blk as f64 / nq, sum_miss as f64 / nq,
                        sum_hit as f64 / nq, sum_phys as f64 / nq, sum_sched as f64 / nq,
                    );

                    pool.stop_prefetch();
                    handle.await;
                }
            }

            eprintln!("\n--- Co-resident caching analysis ---");
            eprintln!("AdjacencyPool caches by page_id (not VID). With heavy_edge layout:");
            eprintln!("  - Each 4KB page holds ~31 adjacency records (130 bytes each at deg=32)");
            eprintln!("  - Loading any VID on page P caches ALL records on P");
            eprintln!("  - Co-resident caching is IMPLICIT — no extra code needed");
            eprintln!("  - Look at hit/q: higher = more co-resident hits");
            eprintln!("  - sched_hits/q shows how often pivoting chose a cached candidate");
        });
    }) {
        eprintln!("Skipped: io_uring not available");
    }
}

// =============================================================================
// SIFT1M cross-dataset validation of VeloANN Phase 1
// =============================================================================

/// Build NSW index + heavy_edge layout for SIFT. Run once, reuse for sweeps.
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

    // Check if already built with matching n
    if layout_dir.join("adj_index.dat").exists() && layout_dir.join("vectors.dat").exists() {
        let meta = IndexMeta::load_from(&layout_dir.join("meta.json")).unwrap();
        if meta.num_vectors as usize == n {
            eprintln!("Already built: SIFT {}K heavy_edge {} pages, skipping rebuild",
                      n / 1000, meta.num_pages.unwrap_or(0));
            return;
        }
    }

    let m_max = 32;
    let ef_construction = 64; // ef_c=200 too slow at 1M

    eprintln!("=== BUILD: SIFT {}K, dim={}, m_max={}, ef_c={} ===",
              n / 1000, dim, m_max, ef_construction);

    let builder = build_nsw_parallel(&vectors, n, dim, MetricType::L2, m_max, ef_construction);
    let index = builder.build();

    let entry_ids: Vec<u32> = index.entry_set().iter().map(|v| v.0).collect();

    // Write base index
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

/// Sweep page_sched_b on pre-built SIFT heavy_edge layout.
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
    let num_bench_queries = 200;
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

    let bench_queries: Vec<Vec<f32>> = queries_flat
        .chunks_exact(dim).take(num_bench_queries).map(|c| c.to_vec()).collect();
    let bench_gt: Vec<&Vec<u32>> = ground_truth.iter().take(num_bench_queries).collect();
    let warmup_vecs: Vec<Vec<f32>> = queries_flat
        .chunks_exact(dim).skip(num_bench_queries).take(warmup_queries).map(|c| c.to_vec()).collect();

    let dir_str = layout_dir.to_str().unwrap().to_owned();
    let direct_io = true;

    if !with_runtime(|rt| {
        rt.block_on(async {
            // SIFT uses L2 metric
            let fp32_bank = FP32SimdVectorBank::new(&disk_vectors, dim, MetricType::L2);

            eprintln!(
                "\n{:>8} {:>6} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>12}",
                "mode", "sched_b", "recall", "p50ms", "p99ms", "QPS",
                "exp/q", "blk/q", "mis/q", "hit/q", "phy/q", "sched_hits/q"
            );

            for &sched_b in sched_b_values {
                // === COLD mode ===
                {
                    let io = Rc::new(
                        IoDriver::open_pages(&dir_str, dim, 64, direct_io)
                            .await.expect("failed to open IO driver"),
                    );
                    let pool_pages = (num_pages * cache_pct / 100).max(256);
                    let pool = Rc::new(AdjacencyPool::new(pool_pages * 4096));
                    let handle = AdjacencyPool::spawn_prefetch_worker(
                        Rc::clone(&pool), Rc::clone(&io), prefetch_width,
                    );

                    let mut recalls = Vec::with_capacity(num_bench_queries);
                    let mut latencies_ms = Vec::with_capacity(num_bench_queries);
                    let mut sum_exp = 0u64;
                    let mut sum_blk = 0u64;
                    let mut sum_miss = 0u64;
                    let mut sum_hit = 0u64;
                    let mut sum_phys = 0u64;
                    let mut sum_sched = 0u64;

                    let wall_start = std::time::Instant::now();
                    for qi in 0..num_bench_queries {
                        pool.pause_prefetch(true);
                        pool.drain_prefetch();
                        monoio::time::sleep(std::time::Duration::from_micros(50)).await;
                        while pool.has_loading() {
                            monoio::time::sleep(std::time::Duration::from_micros(100)).await;
                        }
                        pool.clear();
                        pool.pause_prefetch(false);

                        let mut perf = SearchPerfContext::default();
                        let t0 = std::time::Instant::now();
                        let results = disk_graph_search_pipe_v3_pagesched(
                            &bench_queries[qi], &entry_set, k, ef, prefetch_width, 0, 0,
                            &pool, &io, &fp32_bank, &adj_index,
                            &mut perf, PerfLevel::EnableTime, sched_b,
                        ).await;
                        latencies_ms.push(t0.elapsed().as_secs_f64() * 1_000.0);

                        let ids: Vec<u32> = results.iter().map(|s| s.id.0).collect();
                        recalls.push(recall_at_k(&ids, bench_gt[qi]));

                        sum_exp += perf.expansions;
                        sum_blk += perf.blocks_read;
                        sum_miss += perf.blocks_miss;
                        sum_hit += perf.blocks_hit;
                        sum_phys += perf.phys_reads;
                        sum_sched += perf.page_sched_hits;
                    }
                    let wall_secs = wall_start.elapsed().as_secs_f64();
                    let nq = num_bench_queries as f64;
                    let avg_recall: f64 = recalls.iter().sum::<f64>() / nq;
                    latencies_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let p50 = latencies_ms[(nq * 0.50) as usize];
                    let p99 = latencies_ms[((nq * 0.99) as usize).min(num_bench_queries - 1)];

                    eprintln!(
                        "{:>8} {:>6} {:>7.3} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>12.1}",
                        "cold", sched_b, avg_recall, p50, p99, nq / wall_secs,
                        sum_exp as f64 / nq, sum_blk as f64 / nq, sum_miss as f64 / nq,
                        sum_hit as f64 / nq, sum_phys as f64 / nq, sum_sched as f64 / nq,
                    );

                    pool.stop_prefetch();
                    handle.await;
                }

                // === WARM mode ===
                {
                    let io = Rc::new(
                        IoDriver::open_pages(&dir_str, dim, 64, direct_io)
                            .await.expect("failed to open IO driver"),
                    );
                    let pool_pages = (num_pages * cache_pct / 100).max(256);
                    let pool = Rc::new(AdjacencyPool::new(pool_pages * 4096));
                    let handle = AdjacencyPool::spawn_prefetch_worker(
                        Rc::clone(&pool), Rc::clone(&io), prefetch_width,
                    );

                    // Warmup
                    for qi in 0..warmup_queries {
                        let mut perf = SearchPerfContext::default();
                        disk_graph_search_pipe_v3_pagesched(
                            &warmup_vecs[qi], &entry_set, k, ef, prefetch_width, 0, 0,
                            &pool, &io, &fp32_bank, &adj_index,
                            &mut perf, PerfLevel::CountOnly, sched_b,
                        ).await;
                    }

                    let mut recalls = Vec::with_capacity(num_bench_queries);
                    let mut latencies_ms = Vec::with_capacity(num_bench_queries);
                    let mut sum_exp = 0u64;
                    let mut sum_blk = 0u64;
                    let mut sum_miss = 0u64;
                    let mut sum_hit = 0u64;
                    let mut sum_phys = 0u64;
                    let mut sum_sched = 0u64;

                    let wall_start = std::time::Instant::now();
                    for qi in 0..num_bench_queries {
                        let mut perf = SearchPerfContext::default();
                        let t0 = std::time::Instant::now();
                        let results = disk_graph_search_pipe_v3_pagesched(
                            &bench_queries[qi], &entry_set, k, ef, prefetch_width, 0, 0,
                            &pool, &io, &fp32_bank, &adj_index,
                            &mut perf, PerfLevel::EnableTime, sched_b,
                        ).await;
                        latencies_ms.push(t0.elapsed().as_secs_f64() * 1_000.0);

                        let ids: Vec<u32> = results.iter().map(|s| s.id.0).collect();
                        recalls.push(recall_at_k(&ids, bench_gt[qi]));

                        sum_exp += perf.expansions;
                        sum_blk += perf.blocks_read;
                        sum_miss += perf.blocks_miss;
                        sum_hit += perf.blocks_hit;
                        sum_phys += perf.phys_reads;
                        sum_sched += perf.page_sched_hits;
                    }
                    let wall_secs = wall_start.elapsed().as_secs_f64();
                    let nq = num_bench_queries as f64;
                    let avg_recall: f64 = recalls.iter().sum::<f64>() / nq;
                    latencies_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let p50 = latencies_ms[(nq * 0.50) as usize];
                    let p99 = latencies_ms[((nq * 0.99) as usize).min(num_bench_queries - 1)];

                    eprintln!(
                        "{:>8} {:>6} {:>7.3} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>12.1}",
                        "warm", sched_b, avg_recall, p50, p99, nq / wall_secs,
                        sum_exp as f64 / nq, sum_blk as f64 / nq, sum_miss as f64 / nq,
                        sum_hit as f64 / nq, sum_phys as f64 / nq, sum_sched as f64 / nq,
                    );

                    pool.stop_prefetch();
                    handle.await;
                }
            }
        });
    }) {
        eprintln!("Skipped: io_uring not available");
    }
}

// =============================================================================
// EXP-SIFT-PHASE2: Free Expansions from Co-Located Records
// =============================================================================

/// Phase 2: free expansions on pre-built SIFT heavy_edge layout.
/// Env: SIFT_DIR, SIFT_N, BENCH_DIR (required).
#[test]
#[ignore]
fn exp_sift_phase2_freeexp() {
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

    let prefetch_width = 4;
    let num_bench_queries = 200;
    let cache_pct = 5usize;
    let sched_b = 4usize;

    assert!(nq >= num_bench_queries, "Need {} queries but dataset has {}", num_bench_queries, nq);

    // Load pre-built artifacts
    let bench_dir = std::env::var("BENCH_DIR").expect("BENCH_DIR required");
    let layout_dir = std::path::PathBuf::from(&bench_dir)
        .join("veloann_phase1_sift")
        .join("heavy_edge");

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

    // Build page-to-VIDs inverted index
    let page_to_vids = build_page_to_vids(&adj_index, n);

    eprintln!(
        "=== PHASE2-FREEEXP: SIFT {}K, dim={}, k={}, W={}, cache={}%, {} pages ===",
        n / 1000, dim, k, prefetch_width, cache_pct, num_pages
    );

    let bench_queries: Vec<Vec<f32>> = queries_flat
        .chunks_exact(dim).take(num_bench_queries).map(|c| c.to_vec()).collect();
    let bench_gt: Vec<&Vec<u32>> = ground_truth.iter().take(num_bench_queries).collect();
    let dir_str = layout_dir.to_str().unwrap().to_owned();

    if !with_runtime(|rt| {
        rt.block_on(async {
            let fp32_bank = FP32SimdVectorBank::new(&disk_vectors, dim, MetricType::L2);

            // --- Sweep 1: free expansion cap at ef=200 ---
            eprintln!("\n--- Sweep 1: Free expansion cap sweep (ef=200, cold, sched_b={}) ---", sched_b);
            eprintln!(
                "{:>14} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>12} {:>12} {:>7}",
                "config", "recall", "p50ms", "p99ms", "QPS",
                "exp/q", "blk/q", "mis/q", "hit/q", "phy/q",
                "bonus_sc/q", "bonus_pu/q", "dist/q"
            );

            let configs: &[(&str, bool, u64)] = &[
                ("baseline",     false, 0),
                ("freeexp_500",  true,  500),
                ("freeexp_2000", true,  2000),
                ("freeexp_inf",  true,  u64::MAX),
            ];
            let ef = 200;

            for &(label, use_freeexp, max_bonus) in configs {
                let io = Rc::new(
                    IoDriver::open_pages(&dir_str, dim, 64, true)
                        .await.expect("failed to open IO driver"),
                );
                let pool_pages = (num_pages * cache_pct / 100).max(256);
                let pool = Rc::new(AdjacencyPool::new(pool_pages * 4096));
                let handle = AdjacencyPool::spawn_prefetch_worker(
                    Rc::clone(&pool), Rc::clone(&io), prefetch_width,
                );

                let mut recalls = Vec::with_capacity(num_bench_queries);
                let mut latencies_ms = Vec::with_capacity(num_bench_queries);
                let mut sum_exp = 0u64;
                let mut sum_blk = 0u64;
                let mut sum_miss = 0u64;
                let mut sum_hit = 0u64;
                let mut sum_phys = 0u64;
                let mut sum_bonus_sc = 0u64;
                let mut sum_bonus_pu = 0u64;
                let mut sum_dist = 0u64;

                let wall_start = std::time::Instant::now();
                for qi in 0..num_bench_queries {
                    pool.pause_prefetch(true);
                    pool.drain_prefetch();
                    monoio::time::sleep(std::time::Duration::from_micros(50)).await;
                    while pool.has_loading() {
                        monoio::time::sleep(std::time::Duration::from_micros(100)).await;
                    }
                    pool.clear();
                    pool.pause_prefetch(false);

                    let mut perf = SearchPerfContext::default();
                    let t0 = std::time::Instant::now();
                    let results = if use_freeexp {
                        disk_graph_search_pipe_v3_freeexp(
                            &bench_queries[qi], &entry_set, k, ef, prefetch_width, 0, 0,
                            &pool, &io, &fp32_bank, &adj_index, &page_to_vids,
                            max_bonus, &mut perf, PerfLevel::EnableTime, sched_b,
                        ).await
                    } else {
                        disk_graph_search_pipe_v3_pagesched(
                            &bench_queries[qi], &entry_set, k, ef, prefetch_width, 0, 0,
                            &pool, &io, &fp32_bank, &adj_index,
                            &mut perf, PerfLevel::EnableTime, sched_b,
                        ).await
                    };
                    latencies_ms.push(t0.elapsed().as_secs_f64() * 1_000.0);

                    let ids: Vec<u32> = results.iter().map(|s| s.id.0).collect();
                    recalls.push(recall_at_k(&ids, bench_gt[qi]));

                    sum_exp += perf.expansions;
                    sum_blk += perf.blocks_read;
                    sum_miss += perf.blocks_miss;
                    sum_hit += perf.blocks_hit;
                    sum_phys += perf.phys_reads;
                    sum_bonus_sc += perf.bonus_scored;
                    sum_bonus_pu += perf.bonus_pushed;
                    sum_dist += perf.distance_computes;
                }
                let wall_secs = wall_start.elapsed().as_secs_f64();
                let nq = num_bench_queries as f64;
                let avg_recall: f64 = recalls.iter().sum::<f64>() / nq;
                latencies_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let p50 = latencies_ms[(nq * 0.50) as usize];
                let p99 = latencies_ms[((nq * 0.99) as usize).min(num_bench_queries - 1)];

                eprintln!(
                    "{:>14} {:>7.3} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>12.1} {:>12.1} {:>7.1}",
                    label, avg_recall, p50, p99, nq / wall_secs,
                    sum_exp as f64 / nq, sum_blk as f64 / nq, sum_miss as f64 / nq,
                    sum_hit as f64 / nq, sum_phys as f64 / nq,
                    sum_bonus_sc as f64 / nq, sum_bonus_pu as f64 / nq,
                    sum_dist as f64 / nq,
                );

                pool.stop_prefetch();
                handle.await;
            }

            // --- Sweep 2: ef reduction with free expansion ---
            eprintln!("\n--- Sweep 2: ef reduction with freeexp_2000 (cold, sched_b={}) ---", sched_b);
            eprintln!(
                "{:>5} {:>10} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>12} {:>12} {:>7}",
                "ef", "freeexp", "recall", "p50ms", "p99ms", "QPS",
                "exp/q", "blk/q", "mis/q", "hit/q",
                "bonus_sc/q", "bonus_pu/q", "dist/q"
            );

            let ef_configs: &[(usize, bool, u64)] = &[
                (100, false, 0),
                (100, true,  2000),
                (150, false, 0),
                (150, true,  2000),
                (200, false, 0),
                (200, true,  2000),
            ];

            for &(ef, use_freeexp, max_bonus) in ef_configs {
                let io = Rc::new(
                    IoDriver::open_pages(&dir_str, dim, 64, true)
                        .await.expect("failed to open IO driver"),
                );
                let pool_pages = (num_pages * cache_pct / 100).max(256);
                let pool = Rc::new(AdjacencyPool::new(pool_pages * 4096));
                let handle = AdjacencyPool::spawn_prefetch_worker(
                    Rc::clone(&pool), Rc::clone(&io), prefetch_width,
                );

                let mut recalls = Vec::with_capacity(num_bench_queries);
                let mut latencies_ms = Vec::with_capacity(num_bench_queries);
                let mut sum_exp = 0u64;
                let mut sum_blk = 0u64;
                let mut sum_miss = 0u64;
                let mut sum_hit = 0u64;
                let mut sum_bonus_sc = 0u64;
                let mut sum_bonus_pu = 0u64;
                let mut sum_dist = 0u64;

                let wall_start = std::time::Instant::now();
                for qi in 0..num_bench_queries {
                    pool.pause_prefetch(true);
                    pool.drain_prefetch();
                    monoio::time::sleep(std::time::Duration::from_micros(50)).await;
                    while pool.has_loading() {
                        monoio::time::sleep(std::time::Duration::from_micros(100)).await;
                    }
                    pool.clear();
                    pool.pause_prefetch(false);

                    let mut perf = SearchPerfContext::default();
                    let t0 = std::time::Instant::now();
                    let results = if use_freeexp {
                        disk_graph_search_pipe_v3_freeexp(
                            &bench_queries[qi], &entry_set, k, ef, prefetch_width, 0, 0,
                            &pool, &io, &fp32_bank, &adj_index, &page_to_vids,
                            max_bonus, &mut perf, PerfLevel::EnableTime, sched_b,
                        ).await
                    } else {
                        disk_graph_search_pipe_v3_pagesched(
                            &bench_queries[qi], &entry_set, k, ef, prefetch_width, 0, 0,
                            &pool, &io, &fp32_bank, &adj_index,
                            &mut perf, PerfLevel::EnableTime, sched_b,
                        ).await
                    };
                    latencies_ms.push(t0.elapsed().as_secs_f64() * 1_000.0);

                    let ids: Vec<u32> = results.iter().map(|s| s.id.0).collect();
                    recalls.push(recall_at_k(&ids, bench_gt[qi]));

                    sum_exp += perf.expansions;
                    sum_blk += perf.blocks_read;
                    sum_miss += perf.blocks_miss;
                    sum_hit += perf.blocks_hit;
                    sum_bonus_sc += perf.bonus_scored;
                    sum_bonus_pu += perf.bonus_pushed;
                    sum_dist += perf.distance_computes;
                }
                let wall_secs = wall_start.elapsed().as_secs_f64();
                let nq = num_bench_queries as f64;
                let avg_recall: f64 = recalls.iter().sum::<f64>() / nq;
                latencies_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let p50 = latencies_ms[(nq * 0.50) as usize];
                let p99 = latencies_ms[((nq * 0.99) as usize).min(num_bench_queries - 1)];

                eprintln!(
                    "{:>5} {:>10} {:>7.3} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>12.1} {:>12.1} {:>7.1}",
                    ef, if use_freeexp { "ON" } else { "OFF" },
                    avg_recall, p50, p99, nq / wall_secs,
                    sum_exp as f64 / nq, sum_blk as f64 / nq, sum_miss as f64 / nq,
                    sum_hit as f64 / nq,
                    sum_bonus_sc as f64 / nq, sum_bonus_pu as f64 / nq,
                    sum_dist as f64 / nq,
                );

                pool.stop_prefetch();
                handle.await;
            }
        });
    }) {
        eprintln!("Skipped: io_uring not available");
    }
}

// =============================================================================
// Cohere 100K Phase 2: Free expansions cross-validation
// =============================================================================

/// Phase 2 free expansion sweep on Cohere 100K (dim=768, cosine).
/// Uses pre-built artifacts from exp_veloann_phase1_build.
/// Env: COHERE_DIR, COHERE_N, BENCH_DIR (required).
#[test]
#[ignore]
fn exp_cohere_phase2_freeexp() {
    let max_n: usize = std::env::var("COHERE_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);

    let dataset_dir = std::env::var("COHERE_DIR").unwrap_or_else(|_| {
        let manifest = env!("CARGO_MANIFEST_DIR");
        format!("{}/../../data/cohere_100k", manifest)
    });

    let (_vectors, queries_flat, ground_truth, n, nq, dim, k) =
        match load_dataset(&dataset_dir, max_n) {
            Some(d) => d,
            None => return,
        };

    let prefetch_width = 4;
    let num_bench_queries = 100;
    let cache_pct = 5usize;
    let sched_b = 4usize;

    assert!(nq >= num_bench_queries, "Need {} queries but dataset has {}", num_bench_queries, nq);

    // Load pre-built artifacts
    let bench_dir = std::env::var("BENCH_DIR").expect("BENCH_DIR required");
    let layout_dir = std::path::PathBuf::from(&bench_dir)
        .join("veloann_phase1")
        .join("heavy_edge");

    assert!(
        layout_dir.join("adj_index.dat").exists(),
        "Run exp_veloann_phase1_build first! Missing: {}/adj_index.dat",
        layout_dir.display()
    );

    let disk_vectors = load_vectors(&layout_dir.join("vectors.dat"), n, dim).unwrap();
    let meta = IndexMeta::load_from(&layout_dir.join("meta.json")).unwrap();
    let num_pages = meta.num_pages.unwrap_or(0) as usize;
    let adj_index = load_adj_index(&layout_dir.join("adj_index.dat"), n).unwrap();
    let entry_set: Vec<VectorId> = meta.entry_set.iter().map(|&v| VectorId(v)).collect();

    // Build page-to-VIDs inverted index
    let page_to_vids = build_page_to_vids(&adj_index, n);

    eprintln!(
        "=== PHASE2-FREEEXP: Cohere {}K, dim={}, k={}, W={}, cache={}%, {} pages ===",
        n / 1000, dim, k, prefetch_width, cache_pct, num_pages
    );

    let bench_queries: Vec<Vec<f32>> = queries_flat
        .chunks_exact(dim).take(num_bench_queries).map(|c| c.to_vec()).collect();
    let bench_gt: Vec<&Vec<u32>> = ground_truth.iter().take(num_bench_queries).collect();
    let dir_str = layout_dir.to_str().unwrap().to_owned();

    if !with_runtime(|rt| {
        rt.block_on(async {
            let fp32_bank = FP32SimdVectorBank::new(&disk_vectors, dim, MetricType::Cosine);

            // --- Sweep 1: free expansion cap at ef=200 ---
            eprintln!("\n--- Sweep 1: Free expansion cap sweep (ef=200, cold, sched_b={}) ---", sched_b);
            eprintln!(
                "{:>14} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>12} {:>12} {:>7}",
                "config", "recall", "p50ms", "p99ms", "QPS",
                "exp/q", "blk/q", "mis/q", "hit/q", "phy/q",
                "bonus_sc/q", "bonus_pu/q", "dist/q"
            );

            let configs: &[(&str, bool, u64)] = &[
                ("baseline",     false, 0),
                ("freeexp_500",  true,  500),
                ("freeexp_2000", true,  2000),
                ("freeexp_inf",  true,  u64::MAX),
            ];
            let ef = 200;

            for &(label, use_freeexp, max_bonus) in configs {
                let io = Rc::new(
                    IoDriver::open_pages(&dir_str, dim, 64, true)
                        .await.expect("failed to open IO driver"),
                );
                let pool_pages = (num_pages * cache_pct / 100).max(256);
                let pool = Rc::new(AdjacencyPool::new(pool_pages * 4096));
                let handle = AdjacencyPool::spawn_prefetch_worker(
                    Rc::clone(&pool), Rc::clone(&io), prefetch_width,
                );

                let mut recalls = Vec::with_capacity(num_bench_queries);
                let mut latencies_ms = Vec::with_capacity(num_bench_queries);
                let mut sum_exp = 0u64;
                let mut sum_blk = 0u64;
                let mut sum_miss = 0u64;
                let mut sum_hit = 0u64;
                let mut sum_phys = 0u64;
                let mut sum_bonus_sc = 0u64;
                let mut sum_bonus_pu = 0u64;
                let mut sum_dist = 0u64;

                let wall_start = std::time::Instant::now();
                for qi in 0..num_bench_queries {
                    pool.pause_prefetch(true);
                    pool.drain_prefetch();
                    monoio::time::sleep(std::time::Duration::from_micros(50)).await;
                    while pool.has_loading() {
                        monoio::time::sleep(std::time::Duration::from_micros(100)).await;
                    }
                    pool.clear();
                    pool.pause_prefetch(false);

                    let mut perf = SearchPerfContext::default();
                    let t0 = std::time::Instant::now();
                    let results = if use_freeexp {
                        disk_graph_search_pipe_v3_freeexp(
                            &bench_queries[qi], &entry_set, k, ef, prefetch_width, 0, 0,
                            &pool, &io, &fp32_bank, &adj_index, &page_to_vids,
                            max_bonus, &mut perf, PerfLevel::EnableTime, sched_b,
                        ).await
                    } else {
                        disk_graph_search_pipe_v3_pagesched(
                            &bench_queries[qi], &entry_set, k, ef, prefetch_width, 0, 0,
                            &pool, &io, &fp32_bank, &adj_index,
                            &mut perf, PerfLevel::EnableTime, sched_b,
                        ).await
                    };
                    latencies_ms.push(t0.elapsed().as_secs_f64() * 1_000.0);

                    let ids: Vec<u32> = results.iter().map(|s| s.id.0).collect();
                    recalls.push(recall_at_k(&ids, bench_gt[qi]));

                    sum_exp += perf.expansions;
                    sum_blk += perf.blocks_read;
                    sum_miss += perf.blocks_miss;
                    sum_hit += perf.blocks_hit;
                    sum_phys += perf.phys_reads;
                    sum_bonus_sc += perf.bonus_scored;
                    sum_bonus_pu += perf.bonus_pushed;
                    sum_dist += perf.distance_computes;
                }
                let wall_secs = wall_start.elapsed().as_secs_f64();
                let nq = num_bench_queries as f64;
                let avg_recall: f64 = recalls.iter().sum::<f64>() / nq;
                latencies_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let p50 = latencies_ms[(nq * 0.50) as usize];
                let p99 = latencies_ms[((nq * 0.99) as usize).min(num_bench_queries - 1)];

                eprintln!(
                    "{:>14} {:>7.3} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>12.1} {:>12.1} {:>7.1}",
                    label, avg_recall, p50, p99, nq / wall_secs,
                    sum_exp as f64 / nq, sum_blk as f64 / nq, sum_miss as f64 / nq,
                    sum_hit as f64 / nq, sum_phys as f64 / nq,
                    sum_bonus_sc as f64 / nq, sum_bonus_pu as f64 / nq,
                    sum_dist as f64 / nq,
                );

                pool.stop_prefetch();
                handle.await;
            }

            // --- Sweep 2: ef reduction with free expansion ---
            eprintln!("\n--- Sweep 2: ef reduction with freeexp_2000 (cold, sched_b={}) ---", sched_b);
            eprintln!(
                "{:>5} {:>10} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>12} {:>12} {:>7}",
                "ef", "freeexp", "recall", "p50ms", "p99ms", "QPS",
                "exp/q", "blk/q", "mis/q", "hit/q",
                "bonus_sc/q", "bonus_pu/q", "dist/q"
            );

            let ef_configs: &[(usize, bool, u64)] = &[
                (100, false, 0),
                (100, true,  2000),
                (150, false, 0),
                (150, true,  2000),
                (200, false, 0),
                (200, true,  2000),
            ];

            for &(ef, use_freeexp, max_bonus) in ef_configs {
                let io = Rc::new(
                    IoDriver::open_pages(&dir_str, dim, 64, true)
                        .await.expect("failed to open IO driver"),
                );
                let pool_pages = (num_pages * cache_pct / 100).max(256);
                let pool = Rc::new(AdjacencyPool::new(pool_pages * 4096));
                let handle = AdjacencyPool::spawn_prefetch_worker(
                    Rc::clone(&pool), Rc::clone(&io), prefetch_width,
                );

                let mut recalls = Vec::with_capacity(num_bench_queries);
                let mut latencies_ms = Vec::with_capacity(num_bench_queries);
                let mut sum_exp = 0u64;
                let mut sum_blk = 0u64;
                let mut sum_miss = 0u64;
                let mut sum_hit = 0u64;
                let mut sum_bonus_sc = 0u64;
                let mut sum_bonus_pu = 0u64;
                let mut sum_dist = 0u64;

                let wall_start = std::time::Instant::now();
                for qi in 0..num_bench_queries {
                    pool.pause_prefetch(true);
                    pool.drain_prefetch();
                    monoio::time::sleep(std::time::Duration::from_micros(50)).await;
                    while pool.has_loading() {
                        monoio::time::sleep(std::time::Duration::from_micros(100)).await;
                    }
                    pool.clear();
                    pool.pause_prefetch(false);

                    let mut perf = SearchPerfContext::default();
                    let t0 = std::time::Instant::now();
                    let results = if use_freeexp {
                        disk_graph_search_pipe_v3_freeexp(
                            &bench_queries[qi], &entry_set, k, ef, prefetch_width, 0, 0,
                            &pool, &io, &fp32_bank, &adj_index, &page_to_vids,
                            max_bonus, &mut perf, PerfLevel::EnableTime, sched_b,
                        ).await
                    } else {
                        disk_graph_search_pipe_v3_pagesched(
                            &bench_queries[qi], &entry_set, k, ef, prefetch_width, 0, 0,
                            &pool, &io, &fp32_bank, &adj_index,
                            &mut perf, PerfLevel::EnableTime, sched_b,
                        ).await
                    };
                    latencies_ms.push(t0.elapsed().as_secs_f64() * 1_000.0);

                    let ids: Vec<u32> = results.iter().map(|s| s.id.0).collect();
                    recalls.push(recall_at_k(&ids, bench_gt[qi]));

                    sum_exp += perf.expansions;
                    sum_blk += perf.blocks_read;
                    sum_miss += perf.blocks_miss;
                    sum_hit += perf.blocks_hit;
                    sum_bonus_sc += perf.bonus_scored;
                    sum_bonus_pu += perf.bonus_pushed;
                    sum_dist += perf.distance_computes;
                }
                let wall_secs = wall_start.elapsed().as_secs_f64();
                let nq = num_bench_queries as f64;
                let avg_recall: f64 = recalls.iter().sum::<f64>() / nq;
                latencies_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let p50 = latencies_ms[(nq * 0.50) as usize];
                let p99 = latencies_ms[((nq * 0.99) as usize).min(num_bench_queries - 1)];

                eprintln!(
                    "{:>5} {:>10} {:>7.3} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>12.1} {:>12.1} {:>7.1}",
                    ef, if use_freeexp { "ON" } else { "OFF" },
                    avg_recall, p50, p99, nq / wall_secs,
                    sum_exp as f64 / nq, sum_blk as f64 / nq, sum_miss as f64 / nq,
                    sum_hit as f64 / nq,
                    sum_bonus_sc as f64 / nq, sum_bonus_pu as f64 / nq,
                    sum_dist as f64 / nq,
                );

                pool.stop_prefetch();
                handle.await;
            }
        });
    }) {
        eprintln!("Skipped: io_uring not available");
    }
}

// =============================================================================
// Phase 3: Multi-Query Coroutine Scheduler
// =============================================================================
//
// Sweep B (concurrent queries per core) to fill CPU idle time during IO waits.
// Uses shared Rc<AdjacencyPool> + Rc<IoDriver> + monoio::spawn for cooperative
// multi-query scheduling on a single core.

/// Run Phase 3 multi-query scheduler sweep on a pre-built dataset.
///
/// Generic over dataset: caller provides loaded vectors, queries, ground truth,
/// and the path to pre-built heavy_edge artifacts.
fn run_phase3_sweep(
    dataset_name: &str,
    metric: MetricType,
    disk_vectors: &[f32],
    bench_queries: &[Vec<f32>],
    bench_gt: &[&Vec<u32>],
    warmup_queries: &[Vec<f32>],
    n: usize,
    dim: usize,
    k: usize,
    adj_index: &[AdjIndexEntry],
    page_to_vids: &[Vec<u32>],
    entry_set: &[VectorId],
    num_pages: usize,
    dir_str: &str,
    cache_pct: usize,
) {
    let num_bench_queries = bench_queries.len();
    let num_warmup = warmup_queries.len();
    let ef = 200;
    let sched_b = 4usize;
    let b_values: &[usize] = &[1, 2, 4, 8];
    let total_prefetch_budget = 4usize;

    eprintln!(
        "=== PHASE3: Multi-Query Scheduler, {} {}K, dim={}, ef={}, cache={}%, {} pages ===",
        dataset_name, n / 1000, dim, ef, cache_pct, num_pages
    );

    if !with_runtime(|rt| {
        rt.block_on(async {
            // Rc-wrap shared data for spawned tasks
            let vecs_rc: Rc<[f32]> = Rc::from(disk_vectors);
            let entry_set_rc: Rc<[VectorId]> = Rc::from(entry_set);
            let adj_index_rc: Rc<[AdjIndexEntry]> = Rc::from(adj_index);
            let page_to_vids_rc: Rc<[Vec<u32>]> = Rc::from(page_to_vids);

            // --- Warm mode ---
            eprintln!("\n--- Warm mode (warmup {} queries, then benchmark) ---", num_warmup);
            eprintln!(
                "{:>2} {:>4} {:>7} {:>8} {:>8} {:>10} {:>10} {:>7} {:>7} {:>7} {:>7} {:>7} {:>8} {:>7}",
                "B", "W/q", "recall", "q_p50ms", "q_p99ms", "bat_p50ms", "bat_p99ms",
                "QPS", "exp/q", "blk/q", "mis/q", "hit/q", "bonus/q", "dist/q"
            );

            for &b in b_values {
                let per_query_w = (total_prefetch_budget / b).max(1);
                let pool_pages = (num_pages * cache_pct / 100).max(256);

                let io = Rc::new(
                    IoDriver::open_pages(dir_str, dim, 64, true)
                        .await.expect("failed to open IO driver"),
                );
                let pool = Rc::new(AdjacencyPool::new(pool_pages * 4096));
                let handle = AdjacencyPool::spawn_prefetch_worker(
                    Rc::clone(&pool), Rc::clone(&io), total_prefetch_budget,
                );

                // Warmup: sequential queries to fill cache
                {
                    let warmup_bank = FP32SimdVectorBank::new(&vecs_rc, dim, metric);
                    for wq in warmup_queries {
                        let mut perf = SearchPerfContext::default();
                        disk_graph_search_pipe_v3_freeexp(
                            wq, entry_set, k, ef, 4, 0, 0,
                            &pool, &io, &warmup_bank, adj_index, page_to_vids,
                            2000, &mut perf, PerfLevel::CountOnly, sched_b,
                        ).await;
                    }
                }

                // Batched benchmark
                let num_batches = (num_bench_queries + b - 1) / b;
                let mut query_latencies_ms: Vec<f64> = Vec::with_capacity(num_bench_queries);
                let mut batch_latencies_ms: Vec<f64> = Vec::with_capacity(num_batches);
                let mut recalls: Vec<f64> = Vec::with_capacity(num_bench_queries);
                let mut sum_exp = 0u64;
                let mut sum_blk = 0u64;
                let mut sum_miss = 0u64;
                let mut sum_hit = 0u64;
                let mut sum_bonus = 0u64;
                let mut sum_dist = 0u64;

                let wall_start = std::time::Instant::now();
                for batch_idx in 0..num_batches {
                    let batch_start = batch_idx * b;
                    let batch_end = (batch_start + b).min(num_bench_queries);

                    let batch_t0 = std::time::Instant::now();
                    let mut handles = Vec::with_capacity(b);
                    for qi in batch_start..batch_end {
                        let pool_c = Rc::clone(&pool);
                        let io_c = Rc::clone(&io);
                        let vecs_c = Rc::clone(&vecs_rc);
                        let es_c = Rc::clone(&entry_set_rc);
                        let adj_c = Rc::clone(&adj_index_rc);
                        let p2v_c = Rc::clone(&page_to_vids_rc);
                        let q = bench_queries[qi].clone();

                        handles.push(monoio::spawn(async move {
                            let bank = FP32SimdVectorBank::new(&vecs_c, dim, metric);
                            let mut perf = SearchPerfContext::default();
                            let t = std::time::Instant::now();
                            let results = disk_graph_search_pipe_v3_freeexp(
                                &q, &es_c, k, ef, per_query_w, 0, 0,
                                &pool_c, &io_c, &bank, &adj_c, &p2v_c,
                                2000, &mut perf, PerfLevel::EnableTime, sched_b,
                            ).await;
                            let elapsed_ms = t.elapsed().as_secs_f64() * 1_000.0;
                            (results, perf, elapsed_ms)
                        }));
                    }

                    for (j, h) in handles.into_iter().enumerate() {
                        let (results, perf, elapsed_ms) = h.await;
                        query_latencies_ms.push(elapsed_ms);

                        let qi = batch_start + j;
                        let ids: Vec<u32> = results.iter().map(|s| s.id.0).collect();
                        recalls.push(recall_at_k(&ids, bench_gt[qi]));

                        sum_exp += perf.expansions;
                        sum_blk += perf.blocks_read;
                        sum_miss += perf.blocks_miss;
                        sum_hit += perf.blocks_hit;
                        sum_bonus += perf.bonus_scored;
                        sum_dist += perf.distance_computes;
                    }
                    batch_latencies_ms.push(batch_t0.elapsed().as_secs_f64() * 1_000.0);
                }
                let wall_secs = wall_start.elapsed().as_secs_f64();
                let nq = num_bench_queries as f64;
                let avg_recall = recalls.iter().sum::<f64>() / nq;

                query_latencies_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let q_p50 = query_latencies_ms[(nq * 0.50) as usize];
                let q_p99 = query_latencies_ms[((nq * 0.99) as usize).min(num_bench_queries - 1)];

                batch_latencies_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let nb = num_batches as f64;
                let bat_p50 = batch_latencies_ms[(nb * 0.50) as usize];
                let bat_p99 = batch_latencies_ms[((nb * 0.99) as usize).min(num_batches - 1)];

                eprintln!(
                    "{:>2} {:>4} {:>7.3} {:>8.1} {:>8.1} {:>10.1} {:>10.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>8.1} {:>7.1}",
                    b, per_query_w, avg_recall, q_p50, q_p99, bat_p50, bat_p99,
                    nq / wall_secs,
                    sum_exp as f64 / nq, sum_blk as f64 / nq, sum_miss as f64 / nq,
                    sum_hit as f64 / nq, sum_bonus as f64 / nq, sum_dist as f64 / nq,
                );

                pool.stop_prefetch();
                handle.await;
            }

            // --- Cold mode (clear cache per batch) ---
            eprintln!("\n--- Cold mode (clear cache before each batch) ---");
            eprintln!(
                "{:>2} {:>4} {:>7} {:>8} {:>8} {:>10} {:>10} {:>7} {:>7} {:>7} {:>7} {:>7} {:>8} {:>7}",
                "B", "W/q", "recall", "q_p50ms", "q_p99ms", "bat_p50ms", "bat_p99ms",
                "QPS", "exp/q", "blk/q", "mis/q", "hit/q", "bonus/q", "dist/q"
            );

            for &b in b_values {
                let per_query_w = (total_prefetch_budget / b).max(1);
                let pool_pages = (num_pages * cache_pct / 100).max(256);

                let io = Rc::new(
                    IoDriver::open_pages(dir_str, dim, 64, true)
                        .await.expect("failed to open IO driver"),
                );
                let pool = Rc::new(AdjacencyPool::new(pool_pages * 4096));
                let handle = AdjacencyPool::spawn_prefetch_worker(
                    Rc::clone(&pool), Rc::clone(&io), total_prefetch_budget,
                );

                let num_batches = (num_bench_queries + b - 1) / b;
                let mut query_latencies_ms: Vec<f64> = Vec::with_capacity(num_bench_queries);
                let mut batch_latencies_ms: Vec<f64> = Vec::with_capacity(num_batches);
                let mut recalls: Vec<f64> = Vec::with_capacity(num_bench_queries);
                let mut sum_exp = 0u64;
                let mut sum_blk = 0u64;
                let mut sum_miss = 0u64;
                let mut sum_hit = 0u64;
                let mut sum_bonus = 0u64;
                let mut sum_dist = 0u64;

                let wall_start = std::time::Instant::now();
                for batch_idx in 0..num_batches {
                    let batch_start = batch_idx * b;
                    let batch_end = (batch_start + b).min(num_bench_queries);

                    // Cold: clear cache before each batch
                    pool.pause_prefetch(true);
                    pool.drain_prefetch();
                    monoio::time::sleep(std::time::Duration::from_micros(50)).await;
                    while pool.has_loading() {
                        monoio::time::sleep(std::time::Duration::from_micros(100)).await;
                    }
                    pool.clear();
                    pool.pause_prefetch(false);

                    let batch_t0 = std::time::Instant::now();
                    let mut handles = Vec::with_capacity(b);
                    for qi in batch_start..batch_end {
                        let pool_c = Rc::clone(&pool);
                        let io_c = Rc::clone(&io);
                        let vecs_c = Rc::clone(&vecs_rc);
                        let es_c = Rc::clone(&entry_set_rc);
                        let adj_c = Rc::clone(&adj_index_rc);
                        let p2v_c = Rc::clone(&page_to_vids_rc);
                        let q = bench_queries[qi].clone();

                        handles.push(monoio::spawn(async move {
                            let bank = FP32SimdVectorBank::new(&vecs_c, dim, metric);
                            let mut perf = SearchPerfContext::default();
                            let t = std::time::Instant::now();
                            let results = disk_graph_search_pipe_v3_freeexp(
                                &q, &es_c, k, ef, per_query_w, 0, 0,
                                &pool_c, &io_c, &bank, &adj_c, &p2v_c,
                                2000, &mut perf, PerfLevel::EnableTime, sched_b,
                            ).await;
                            let elapsed_ms = t.elapsed().as_secs_f64() * 1_000.0;
                            (results, perf, elapsed_ms)
                        }));
                    }

                    for (j, h) in handles.into_iter().enumerate() {
                        let (results, perf, elapsed_ms) = h.await;
                        query_latencies_ms.push(elapsed_ms);

                        let qi = batch_start + j;
                        let ids: Vec<u32> = results.iter().map(|s| s.id.0).collect();
                        recalls.push(recall_at_k(&ids, bench_gt[qi]));

                        sum_exp += perf.expansions;
                        sum_blk += perf.blocks_read;
                        sum_miss += perf.blocks_miss;
                        sum_hit += perf.blocks_hit;
                        sum_bonus += perf.bonus_scored;
                        sum_dist += perf.distance_computes;
                    }
                    batch_latencies_ms.push(batch_t0.elapsed().as_secs_f64() * 1_000.0);
                }
                let wall_secs = wall_start.elapsed().as_secs_f64();
                let nq = num_bench_queries as f64;
                let avg_recall = recalls.iter().sum::<f64>() / nq;

                query_latencies_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let q_p50 = query_latencies_ms[(nq * 0.50) as usize];
                let q_p99 = query_latencies_ms[((nq * 0.99) as usize).min(num_bench_queries - 1)];

                batch_latencies_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let nb = num_batches as f64;
                let bat_p50 = batch_latencies_ms[(nb * 0.50) as usize];
                let bat_p99 = batch_latencies_ms[((nb * 0.99) as usize).min(num_batches - 1)];

                eprintln!(
                    "{:>2} {:>4} {:>7.3} {:>8.1} {:>8.1} {:>10.1} {:>10.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>8.1} {:>7.1}",
                    b, per_query_w, avg_recall, q_p50, q_p99, bat_p50, bat_p99,
                    nq / wall_secs,
                    sum_exp as f64 / nq, sum_blk as f64 / nq, sum_miss as f64 / nq,
                    sum_hit as f64 / nq, sum_bonus as f64 / nq, sum_dist as f64 / nq,
                );

                pool.stop_prefetch();
                handle.await;
            }
        });
    }) {
        eprintln!("Skipped: io_uring not available");
    }
}

/// Phase 3: Multi-query coroutine scheduler on SIFT 1M.
/// Uses pre-built artifacts from exp_sift_phase1_build.
/// Env: SIFT_DIR, SIFT_N, BENCH_DIR (required).
#[test]
#[ignore]
fn exp_multi_query_scheduler_sift() {
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

    let num_bench_queries = 200;
    let num_warmup = 50;
    let cache_pct = 5usize;

    assert!(nq >= num_bench_queries + num_warmup,
        "Need {} queries but dataset has {}", num_bench_queries + num_warmup, nq);

    let bench_dir = std::env::var("BENCH_DIR").expect("BENCH_DIR required");
    let layout_dir = std::path::PathBuf::from(&bench_dir)
        .join("veloann_phase1_sift")
        .join("heavy_edge");

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
    let page_to_vids = build_page_to_vids(&adj_index, n);

    let bench_queries: Vec<Vec<f32>> = queries_flat
        .chunks_exact(dim).take(num_bench_queries).map(|c| c.to_vec()).collect();
    let bench_gt: Vec<&Vec<u32>> = ground_truth.iter().take(num_bench_queries).collect();
    let warmup_vecs: Vec<Vec<f32>> = queries_flat
        .chunks_exact(dim).skip(num_bench_queries).take(num_warmup).map(|c| c.to_vec()).collect();

    let dir_str = layout_dir.to_str().unwrap();

    run_phase3_sweep(
        "SIFT", MetricType::L2, &disk_vectors,
        &bench_queries, &bench_gt, &warmup_vecs,
        n, dim, k, &adj_index, &page_to_vids, &entry_set,
        num_pages, dir_str, cache_pct,
    );
}

/// Phase 3: Multi-query coroutine scheduler on Cohere 100K.
/// Uses pre-built artifacts from exp_veloann_phase1_build.
/// Env: COHERE_DIR, COHERE_N, BENCH_DIR (required).
#[test]
#[ignore]
fn exp_multi_query_scheduler_cohere() {
    let max_n: usize = std::env::var("COHERE_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);

    let dataset_dir = std::env::var("COHERE_DIR").unwrap_or_else(|_| {
        let manifest = env!("CARGO_MANIFEST_DIR");
        format!("{}/../../data/cohere_100k", manifest)
    });

    let (_vectors, queries_flat, ground_truth, n, nq, dim, k) =
        match load_dataset(&dataset_dir, max_n) {
            Some(d) => d,
            None => return,
        };

    let num_bench_queries = 100;
    let num_warmup = 50;
    let cache_pct = 5usize;

    assert!(nq >= num_bench_queries + num_warmup,
        "Need {} queries but dataset has {}", num_bench_queries + num_warmup, nq);

    let bench_dir = std::env::var("BENCH_DIR").expect("BENCH_DIR required");
    let layout_dir = std::path::PathBuf::from(&bench_dir)
        .join("veloann_phase1")
        .join("heavy_edge");

    assert!(
        layout_dir.join("adj_index.dat").exists(),
        "Run exp_veloann_phase1_build first! Missing: {}/adj_index.dat",
        layout_dir.display()
    );

    let disk_vectors = load_vectors(&layout_dir.join("vectors.dat"), n, dim).unwrap();
    let meta = IndexMeta::load_from(&layout_dir.join("meta.json")).unwrap();
    let num_pages = meta.num_pages.unwrap_or(0) as usize;
    let adj_index = load_adj_index(&layout_dir.join("adj_index.dat"), n).unwrap();
    let entry_set: Vec<VectorId> = meta.entry_set.iter().map(|&v| VectorId(v)).collect();
    let page_to_vids = build_page_to_vids(&adj_index, n);

    let bench_queries: Vec<Vec<f32>> = queries_flat
        .chunks_exact(dim).take(num_bench_queries).map(|c| c.to_vec()).collect();
    let bench_gt: Vec<&Vec<u32>> = ground_truth.iter().take(num_bench_queries).collect();
    let warmup_vecs: Vec<Vec<f32>> = queries_flat
        .chunks_exact(dim).skip(num_bench_queries).take(num_warmup).map(|c| c.to_vec()).collect();

    let dir_str = layout_dir.to_str().unwrap();

    run_phase3_sweep(
        "Cohere", MetricType::Cosine, &disk_vectors,
        &bench_queries, &bench_gt, &warmup_vecs,
        n, dim, k, &adj_index, &page_to_vids, &entry_set,
        num_pages, dir_str, cache_pct,
    );
}
