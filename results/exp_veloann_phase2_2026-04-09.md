# VeloANN Phase 2: Free Expansions from Co-Located Records

**Date:** 2026-04-09
**Dataset:** SIFT 1M (dim=128, L2)
**Hardware:** EC2 i4i.xlarge, NVMe direct IO
**Branch:** veloann-phase1

## Concept

When a cache-miss page is loaded for VID X, the page contains ~31 co-located VIDs
(packed via heavy_edge layout). Phase 2 scores all unvisited co-located VIDs on that
page for "free" — zero additional IO cost. Non-dominated scores are pushed into the
beam as bonus candidates.

## Implementation

- `build_page_to_vids()`: precomputes page → VID[] inverted index from AdjIndexEntry
- `disk_graph_search_pipe_v3_freeexp()`: full search function
  - Checks `was_resident = pool.is_resident(page_id)` BEFORE loading
  - On cache miss: iterates page_to_vids, scores unvisited co-located VIDs
  - Respects `max_bonus_per_query` cap
- Perf counters: `bonus_scored`, `bonus_pushed` in SearchPerfContext

## Results

### Sweep 1: Cap sweep at ef=200

| Config | Recall | p50ms | p99ms | QPS | exp/q | mis/q | hit/q | bonus_sc/q | bonus_pu/q | dist/q |
|--------|--------|-------|-------|-----|-------|-------|-------|------------|------------|--------|
| baseline | 0.962 | 7.2 | 10.0 | 121.0 | 203.7 | 2.3 | 201.5 | 0.0 | 0.0 | 2719.3 |
| freeexp_500 | 0.963 | 6.8 | 9.7 | 127.9 | 203.5 | 1.9 | 201.6 | 80.1 | 71.4 | 2785.5 |
| freeexp_2000 | 0.963 | 6.8 | 9.6 | 128.3 | 203.5 | 1.9 | 201.6 | 80.1 | 71.4 | 2785.5 |
| freeexp_inf | 0.963 | 6.7 | 8.9 | 129.3 | 203.5 | 1.9 | 201.6 | 80.1 | 71.4 | 2785.5 |

### Sweep 2: ef reduction × freeexp

| ef | freeexp | Recall | p50ms | p99ms | QPS | exp/q | mis/q | hit/q | bonus_sc/q | bonus_pu/q | dist/q |
|----|---------|--------|-------|-------|-----|-------|-------|-------|------------|------------|--------|
| 100 | OFF | 0.900 | 4.2 | 6.0 | 190.1 | 104.5 | 2.2 | 102.2 | 0.0 | 0.0 | 1587.8 |
| 100 | ON | 0.900 | 3.8 | 6.8 | 200.5 | 104.2 | 1.9 | 102.3 | 79.0 | 55.7 | 1654.7 |
| 150 | OFF | 0.943 | 5.8 | 8.1 | 146.4 | 154.0 | 2.3 | 151.8 | 0.0 | 0.0 | 2172.5 |
| 150 | ON | 0.942 | 5.4 | 8.0 | 154.9 | 153.8 | 1.9 | 151.8 | 80.1 | 65.1 | 2238.8 |
| 200 | OFF | 0.962 | 7.4 | 10.6 | 119.5 | 203.7 | 2.3 | 201.5 | 0.0 | 0.0 | 2719.3 |
| 200 | ON | 0.963 | 6.7 | 9.4 | 127.5 | 203.5 | 1.9 | 201.6 | 80.1 | 71.4 | 2785.5 |

## Analysis

1. **Modest bonus volume:** Only ~80 bonus scores per query (not thousands). The high
   cache hit rate (99%+) means most pages are already resident, so few cache-miss pages
   trigger the free expansion path. Of co-located VIDs on miss pages, most are already
   visited by the beam.

2. **Consistent latency improvement:** Free expansions improve p50 by 0.4–0.7ms across
   all ef levels (5–10% improvement). QPS improves 5–7%.

3. **Cap irrelevant:** All cap values (500, 2000, inf) produce identical results because
   only ~80 bonuses fire per query. No cap needed in practice.

4. **Recall neutral:** Free expansions don't meaningfully change recall (+0.001 at ef=200).
   The bonus candidates are co-located neighbors, which the beam would discover anyway
   through graph traversal.

5. **Not a substitute for ef:** freeexp at ef=100 (0.900 recall) does not approach
   ef=200 baseline (0.962). Free expansions save some IO but don't replace graph
   exploration depth.

6. **Why limited impact:** The heavy_edge layout + high cache hit rate means the system
   is already exploiting page locality well. There are very few cache misses to trigger
   free expansions, and those miss pages contain VIDs the beam has mostly already visited.

## Conclusion

Free expansions provide a modest 5-10% latency improvement at no IO cost. The technique
is complementary to cache-aware pivoting (Phase 1) but limited by the already-high cache
hit rate. The biggest wins would come on datasets/workloads with lower cache hit rates
where more pages are loaded cold.
