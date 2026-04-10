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

No new search library code. Test uses existing monoio::spawn + shared Rc primitives.
Added `FP32SimdVectorBank::compute_norms` and `with_norms` to avoid per-spawn norm
recomputation (cosine norm precompute is O(N×dim), ~87ms for 100K×768).

## Results — SIFT 1M (dim=128, L2)

### Warm mode (warmup 50 queries, then benchmark)

| B | W/q | Recall | q_p50ms | q_p99ms | bat_p50ms | bat_p99ms | QPS | mis/q | hit/q | bonus/q |
|---|-----|--------|---------|---------|-----------|-----------|-----|-------|-------|---------|
| 1 | 4 | 0.963 | 7.1 | 10.2 | 7.1 | 10.3 | 141.5 | 1.2 | 202.6 | 56.0 |
| 2 | 2 | 0.963 | 10.4 | 14.1 | 10.9 | 14.5 | **183.6** | 2.2 | 389.4 | 57.0 |
| 4 | 1 | 0.963 | 18.2 | 23.3 | 19.7 | 24.6 | **203.1** | 5.2 | 764.5 | 68.6 |
| 8 | 1 | 0.963 | 45.2 | 54.5 | 54.2 | 58.1 | 150.5 | 10.5 | 1521.9 | 67.0 |

### Cold mode (clear cache before each batch)

| B | W/q | Recall | q_p50ms | q_p99ms | bat_p50ms | bat_p99ms | QPS | mis/q | hit/q | bonus/q |
|---|-----|--------|---------|---------|-----------|-----------|-----|-------|-------|---------|
| 1 | 4 | 0.963 | 7.1 | 10.6 | 7.1 | 10.6 | 122.2 | 1.9 | 201.6 | 80.1 |
| 2 | 2 | 0.963 | 10.3 | 13.2 | 11.0 | 13.6 | **167.5** | 3.1 | 387.6 | 77.5 |
| 4 | 1 | 0.963 | 17.1 | 22.0 | 18.5 | 22.8 | **203.1** | 6.0 | 758.3 | 79.3 |
| 8 | 1 | 0.963 | 41.8 | 54.4 | 50.5 | 55.0 | 157.0 | 11.4 | 1503.8 | 76.4 |

## Results — Cohere 100K (dim=768, cosine)

### Warm mode

| B | W/q | Recall | q_p50ms | q_p99ms | bat_p50ms | bat_p99ms | QPS | mis/q | hit/q | bonus/q |
|---|-----|--------|---------|---------|-----------|-----------|-----|-------|-------|---------|
| 1 | 4 | 0.963 | 6.4 | 8.0 | 6.4 | 8.1 | 154.9 | 0.3 | 200.9 | 13.8 |
| 2 | 2 | 0.963 | 10.1 | 11.7 | 10.6 | 11.8 | **191.4** | 0.7 | 388.8 | 11.9 |
| 4 | 1 | 0.962 | 17.8 | 21.1 | 19.1 | 21.2 | **209.6** | 1.8 | 762.4 | 17.1 |
| 8 | 1 | 0.962 | 37.0 | 44.2 | 40.6 | 45.3 | 193.8 | 5.3 | 1476.6 | 25.4 |

### Cold mode

| B | W/q | Recall | q_p50ms | q_p99ms | bat_p50ms | bat_p99ms | QPS | mis/q | hit/q | bonus/q |
|---|-----|--------|---------|---------|-----------|-----------|-----|-------|-------|---------|
| 1 | 4 | 0.962 | 7.1 | 9.1 | 7.2 | 9.1 | 120.7 | 1.3 | 199.6 | 45.0 |
| 2 | 2 | 0.962 | 10.7 | 13.9 | 11.2 | 14.1 | **163.3** | 1.9 | 387.8 | 41.4 |
| 4 | 1 | 0.963 | 18.0 | 21.1 | 18.9 | 21.2 | **198.7** | 3.3 | 763.3 | 40.5 |
| 8 | 1 | 0.962 | 36.6 | 41.6 | 39.4 | 41.9 | 197.6 | 6.7 | 1479.7 | 39.9 |

## Analysis

### Both datasets show B=4 as the optimal throughput point

| Dataset | B=1 QPS | B=2 QPS | B=4 QPS | B=8 QPS | Best speedup |
|---------|---------|---------|---------|---------|-------------|
| SIFT warm | 141.5 | 183.6 | **203.1** | 150.5 | **+44%** (B=4) |
| SIFT cold | 122.2 | 167.5 | **203.1** | 157.0 | **+66%** (B=4) |
| Cohere warm | 154.9 | 191.4 | **209.6** | 193.8 | **+35%** (B=4) |
| Cohere cold | 120.7 | 163.3 | **198.7** | 197.6 | **+65%** (B=4) |

### Key findings

1. **B=4 is the sweet spot on both datasets.** +44% QPS on SIFT warm, +35% on Cohere
   warm. Cold mode gains are even larger (+65-66%) because more cache misses create more
   yield points for interleaving.

2. **B=8 regresses.** Cache thrashing (10-11 misses/q vs 1-5 at B=4) and reduced
   prefetch budget cause p99 explosion and QPS drop.

3. **Recall unchanged** at 0.962-0.963 across all B values. Queries are independent —
   interleaving doesn't affect search quality.

4. **Per-query p50 scales ~linearly**: B=1: 7ms, B=2: 10ms, B=4: 18ms, B=8: 37-45ms.
   This is expected — queries share the core, so individual latency increases
   proportionally to B.

5. **Batch p99 controlled at B=4**: ~22-25ms (SIFT), ~21ms (Cohere). B=8 reaches
   55ms (SIFT), 45ms (Cohere).

6. **Cache sharing amplifies with B**: hit/q at B=4 is ~760 (vs ~201 at B=1).
   4 concurrent queries generate cache hits for each other via the shared
   AdjacencyPool. This is the "co-resident benefit multiplied by B."

7. **Cohere benefits despite being compute-heavy**: Unlike the failed previous run
   (which had per-spawn norm recomputation overhead), with precomputed norms Cohere
   shows similar scaling to SIFT. The 768-dim distance computation is ~6ms per query
   but still leaves enough IO-idle time at yield points for interleaving.

### Why B=4 works now (vs EXP-BW failure at B=2)

The old EXP-BW experiment (2026-03-03) saw p99 explode to 644ms at B=8 with zero QPS
gain. The difference:

| Factor | EXP-BW (old) | Phase 3 (now) |
|--------|-------------|---------------|
| Layout | V1 (1 block/VID) | V3 pages + heavy_edge |
| Cache hit rate | ~0% | 90%+ |
| Misses/q at B=1 | ~200 | 1-2 |
| Prefetch | B×W contention | Fixed W=4 budget |
| Pivoting | None | sched_b=4 |
| Free expansion | None | bonus from co-located VIDs |

## Conclusion

Multi-query coroutine scheduling delivers **+35-66% single-core QPS at B=4** across
both datasets with controlled tail latency. This validates the VeloANN cooperative
IO multiplexing hypothesis. The technique works because:

1. V3 page layout + heavy_edge keeps cache hit rate high even at B=4
2. Shared AdjacencyPool means queries warm the cache for each other
3. Fixed prefetch budget (W=4 total) prevents IO contention
4. Cache-aware pivoting ensures queries expand cached candidates first

Production recommendation: **B=4 per core**, with multi-core parallelism via separate
monoio runtimes per core (each with B=4), for aggregate throughput scaling.
