# EXP-VELOANN-PHASE1: Co-Resident Caching + Cache-Aware Beam Search

**Date**: 2026-04-09
**Instance**: i4i.xlarge (NVMe), EC2 us-west-1
**Dataset**: Cohere 100K, dim=768, k=100, ef=200, W=4, 5% cache
**Layout**: heavy_edge only
**Paper**: VeloANN (PVLDB 2026, Zhao et al.)

## Verdict: co-resident caching is implicit and already effective; pivoting eliminates misses but prefetch already masks latency

### Finding 1: Co-resident caching is FREE

AdjacencyPool caches by `page_id`, not VID. Loading any VID on page P caches
the entire 4KB page under key P. All co-located VIDs (~31 per page at deg=32)
become instant cache hits. **No code changes needed.**

Evidence: at `sched_b=0` (no pivoting), cold `hit/q=180.9` out of 201.1 total
block accesses — **90% cache hit rate** even with cold-per-query pool clearing.
This is entirely due to within-query co-resident hits from heavy_edge layout.

### Finding 2: Pivoting eliminates cache misses but doesn't help latency

| mode | sched_b | recall | p50ms | p99ms | QPS   | mis/q | hit/q | phy/q | sched_hits/q |
|------|---------|--------|-------|-------|-------|-------|-------|-------|--------------|
| cold | 0       | 0.963  | 7.3   | 9.5   | 119.5 | 20.2  | 180.9 | 108.2 | 0.0          |
| cold | 2       | 0.962  | 7.2   | 9.0   | 121.2 | 4.0   | 197.1 | 108.1 | 15.0         |
| cold | 4       | 0.962  | 7.2   | 9.1   | 120.4 | 1.3   | 199.9 | 107.8 | 17.3         |
| cold | 8       | 0.963  | 7.4   | 9.2   | 118.0 | 1.0   | 200.4 | 107.7 | 17.5         |
| warm | 0       | 0.963  | 6.8   | 8.5   | 148.0 | 15.7  | 185.4 | 89.0  | 0.0          |
| warm | 2       | 0.962  | 6.6   | 8.4   | 151.6 | 2.0   | 199.2 | 89.5  | 13.2         |
| warm | 4       | 0.962  | 6.7   | 8.3   | 148.6 | 0.3   | 201.1 | 88.8  | 14.8         |
| warm | 8       | 0.962  | 6.7   | 8.2   | 147.8 | 0.1   | 201.3 | 88.5  | 14.7         |

`mis/q` drops from 20.2 → 1.3 at B=4 (93.6% reduction). But p50 is essentially
unchanged (7.3 → 7.2ms cold, 6.8 → 6.7ms warm).

### Why no latency improvement?

**Prefetch already masks the misses.** `phy/q` stays at ~108 regardless of
`sched_b`. The ~20 cold cache misses at B=0 were already being served by
async prefetch (prefetch_window=4) — the beam rarely stalled on them. Pivoting
converts those misses to hits (eliminating the prefetch→hit transition), but
the latency was already hidden.

In other words: prefetch + co-resident caching already achieve near-optimal
latency. Pivoting further reduces the miss count to near-zero but there's
nothing left to save — the IO pipeline was already keeping up.

### Recall stability

Recall = 0.962-0.963 across all B values — pivoting does not degrade recall.

### B sensitivity

B=2 captures most of the miss reduction (20.2 → 4.0). B=4 reduces to 1.3.
B=8 offers marginal further reduction (1.0) but adds slight overhead.
B=4 is optimal, matching VeloANN's recommendation.

## Analysis

The VeloANN paper's cache-aware beam search was designed for systems without
prefetch pipelines. Divergence already has an effective prefetch mechanism
(lookahead=4, async io_uring) that covers the same ground. The combination
of heavy_edge layout + page_id-based caching + prefetch makes the system
"accidentally VeloANN-complete" — co-resident caching and latency hiding
are already achieved.

**Where pivoting WILL help:** When we add multi-query scheduling (Phase 3),
prefetch budget per query will be reduced (W_total/B). In that regime,
pivoting becomes essential to avoid stalls that prefetch can no longer cover.

## Conclusion

- Co-resident caching: **already implicit** via page_id keying. No code needed.
- Cache-aware pivoting: **correct but redundant** given current prefetch pipeline.
- Recommendation: Enable B=4 by default (zero cost, eliminates misses, prepares
  for multi-query scheduling). The `disk_graph_search_pipe_v3_cacheaware` wrapper
  is available for this purpose.
- Next: Phase 2 (free expansions) or Phase 3 (multi-query scheduler) where
  pivoting becomes load-bearing.
