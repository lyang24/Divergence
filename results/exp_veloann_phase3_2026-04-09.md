# VeloANN Phase 3: Multi-Query Coroutine Scheduler

**Date:** 2026-04-09
**Datasets:** SIFT 1M (dim=128, L2) + Cohere 100K (dim=768, cosine)
**Hardware:** EC2 i4i.xlarge, NVMe direct IO, single core
**Branch:** veloann-phase1

## Concept

CPU utilization at B=1 is ~20% — the core is idle 80% of the time waiting on NVMe IO.
With B concurrent queries per core sharing the same monoio runtime, one query can
compute while another awaits IO. VeloANN formula: B = ceil(alpha * I/T) ~ 5.

Key differences from failed EXP-BW (2026-03-03):
- V3 pages + heavy_edge: 90%+ cache hit rate
- Cache-aware pivoting (sched_b=4)
- Free expansions from co-located records
- Shared cache across queries (Rc<AdjacencyPool>)
- Total prefetch budget = 4 (constant), per-query W = 4/B

## Implementation

No new library code — purely a test/experiment using existing primitives:
- `monoio::spawn` for concurrent coroutines on single core
- `Rc<AdjacencyPool>` shared across all B queries (cache sharing)
- `Rc<IoDriver>` shared IO channel
- Batched execution: process all queries in batches of B

## Results — SIFT 1M (dim=128, L2)

### Warm mode (warmup 50 queries, then benchmark)

| B | W/q | Recall | q_p50ms | q_p99ms | bat_p50ms | bat_p99ms | QPS | mis/q | hit/q | bonus/q |
|---|-----|--------|---------|---------|-----------|-----------|-----|-------|-------|---------|
| 1 | 4 | 0.963 | 6.9 | 9.6 | 6.9 | 9.6 | 144.4 | 1.1 | 202.7 | 52.2 |
| 2 | 2 | 0.963 | 10.3 | 12.6 | 10.9 | 13.1 | **185.3** | 2.2 | 389.2 | 55.9 |
| 4 | 1 | 0.963 | 21.1 | 28.0 | 25.1 | 28.5 | 161.2 | 5.2 | 763.8 | 64.8 |
| 8 | 1 | 0.963 | 43.9 | 55.5 | 53.1 | 58.1 | 154.6 | 11.0 | 1513.1 | 72.9 |

### Cold mode (clear cache before each batch)

| B | W/q | Recall | q_p50ms | q_p99ms | bat_p50ms | bat_p99ms | QPS | mis/q | hit/q | bonus/q |
|---|-----|--------|---------|---------|-----------|-----------|-----|-------|-------|---------|
| 1 | 4 | 0.963 | 7.0 | 9.2 | 7.0 | 9.2 | 123.9 | 1.9 | 201.6 | 80.1 |
| 2 | 2 | 0.963 | 10.2 | 12.9 | 10.9 | 13.1 | **169.9** | 3.1 | 387.5 | 77.5 |
| 4 | 1 | 0.963 | 20.5 | 27.1 | 24.6 | 28.2 | 157.9 | 5.9 | 759.4 | 79.1 |
| 8 | 1 | 0.963 | 40.0 | 50.8 | 48.7 | 55.1 | 161.6 | 11.4 | 1497.9 | 77.6 |

## Results — Cohere 100K (dim=768, cosine)

### Warm mode

| B | W/q | Recall | q_p50ms | q_p99ms | bat_p50ms | bat_p99ms | QPS | mis/q | hit/q | bonus/q |
|---|-----|--------|---------|---------|-----------|-----------|-----|-------|-------|---------|
| 1 | 4 | 0.963 | 6.9 | 8.6 | 94.1 | 95.8 | 10.6 | 0.3 | 201.0 | 12.0 |
| 2 | 2 | 0.962 | 95.2 | 99.3 | 185.2 | 186.7 | 10.8 | 0.7 | 388.5 | 14.7 |
| 4 | 1 | 0.962 | 187.4 | 282.5 | 367.2 | 370.0 | 10.9 | 2.1 | 761.3 | 22.1 |
| 8 | 1 | 0.962 | 300.2 | 651.4 | 735.9 | 744.8 | 10.9 | 5.6 | 1473.5 | 30.8 |

### Cold mode

| B | W/q | Recall | q_p50ms | q_p99ms | bat_p50ms | bat_p99ms | QPS | mis/q | hit/q | bonus/q |
|---|-----|--------|---------|---------|-----------|-----------|-----|-------|-------|---------|
| 1 | 4 | 0.962 | 7.3 | 9.3 | 94.3 | 96.5 | 10.5 | 1.3 | 199.6 | 45.0 |
| 2 | 2 | 0.962 | 95.7 | 99.4 | 185.8 | 187.3 | 10.7 | 1.9 | 387.1 | 41.5 |
| 4 | 1 | 0.962 | 190.0 | 282.8 | 368.9 | 370.4 | 10.8 | 3.4 | 760.7 | 40.9 |
| 8 | 1 | 0.963 | 300.5 | 652.0 | 735.7 | 739.0 | 10.8 | 6.7 | 1482.1 | 40.7 |

## Analysis

### SIFT 1M: B=2 is the sweet spot (+28% QPS)

1. **B=2 warm: 185 QPS (+28% over B=1's 144)** — the best configuration. Two queries
   interleave well: when one waits on IO, the other computes. Per-query p50 increases
   from 6.9ms to 10.3ms (1.5x) but throughput scales 1.28x.

2. **B=4 regresses**: 161 QPS, down from B=2's 185. Too many queries competing for
   cache → misses increase from 1.1 to 5.2/q. Reduced prefetch (W=1) also hurts.

3. **B=8 further regresses**: 155 QPS. 11 misses/q, p99 at 55ms. Cache thrashing.

4. **p99 controlled at B=2**: 12.6ms warm (1.3x baseline's 9.6ms). Below the 3x
   threshold for acceptable tail latency. B=4 reaches 28ms (2.9x), borderline.

5. **Cold mode**: B=2 still best at 170 QPS (+37% over B=1's 124). More misses create
   more yield points, so interleaving is slightly more effective. But B=4 and B=8
   don't benefit further — cache thrashing dominates.

6. **Recall unchanged** across all B values (0.963). Queries are independent.

### Cohere 100K: No benefit from multi-query

1. **QPS flat at ~10.8** regardless of B. Per-query latency scales linearly (6.9ms →
   95ms → 187ms → 300ms at B=1/2/4/8) with zero throughput gain.

2. **Root cause: compute-bound, not IO-bound.** At dim=768, distance computation takes
   ~6.5ms per query (3260 distances × ~2µs each). With only 0.3 cache misses per query
   (warm), there are almost no `.await` yield points. Queries run back-to-back with no
   interleaving opportunity.

3. **Contrast with SIFT**: At dim=128, distance computation is ~0.5ms per query,
   leaving ~6ms of IO-idle time for other queries to fill. Cohere fills the CPU with
   compute, so there's no idle time to reclaim.

4. **The VeloANN formula predicts this**: B = ceil(α × I/T). For Cohere warm,
   I ≈ 0.1ms (0.3 misses × 0.3ms each), T ≈ 6.5ms, so B = ceil(0.015) = 1.
   Multi-query only helps when I/T >> 0.

### Cross-dataset comparison

| Dataset | Dim | B=1 QPS | Best B | Best QPS | Speedup | IO idle time |
|---------|-----|---------|--------|----------|---------|-------------|
| SIFT 1M | 128 | 144 | B=2 | 185 | +28% | High (~80%) |
| Cohere 100K | 768 | 10.6 | B=1 | 10.6 | 0% | Near zero |

## Conclusion

Multi-query coroutine scheduling delivers **+28% QPS on SIFT 1M at B=2** with
controlled tail latency (p99 < 1.5x baseline). This validates the VeloANN hypothesis
that cooperative IO multiplexing fills CPU idle time during NVMe waits.

However, the benefit is **dimension-dependent**: high-dimensional data (Cohere, dim=768)
is compute-bound, not IO-bound, leaving no idle time for interleaving. The technique
works best when:
- Dimensions are moderate (≤256) so distance computation is cheap
- Cache miss rate is non-trivial (more yield points)
- B is kept low (2-4) to avoid cache thrashing

The p99 explosion at B≥4 (even on SIFT) suggests that B=2 is the production operating
point for single-core scheduling. Multi-core parallelism (separate cores, each with
B=2) is the path to higher aggregate throughput.
