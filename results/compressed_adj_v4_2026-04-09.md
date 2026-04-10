# Compressed Adjacency (v4) Experiment Results

**Date**: 2026-04-09
**Dataset**: Cohere 100K, dim=768, k=100, ef=200, cosine
**Layout**: heavy_edge reorder, B=1, W=4, page_sched_b=4, warm cache

## Page Count

| Format | Pages | Reduction |
|--------|-------|-----------|
| v3 (uncompressed) | 1526 | — |
| v4 (delta-varint) | 804 | **47.3%** |

Records per page: v3 ~31, v4 ~65 (avg degree ~15)

## Benchmark Results

| cache% | Format | cache_pages | recall | p50 ms | QPS | mis/q | bonus/q | dist/q |
|--------|--------|-------------|--------|--------|-----|-------|---------|--------|
| 1% | v3 | 16/1526 | 0.960 | 9.4 | 107 | 10.1 | 384 | 3627 |
| 1% | v4 | 16/804 | 0.960 | 9.6 | 105 | 10.2 | 808 | 4042 |
| 5% | v3 | 76/1526 | 0.959 | 7.9 | 126 | 2.0 | 68 | 3350 |
| 5% | v4 | 40/804 | 0.960 | 8.3 | 119 | 4.0 | 310 | 3574 |
| 20% | v3 | 305/1526 | 0.959 | 6.7 | 149 | 0.3 | 15 | 3305 |
| 20% | v4 | 160/804 | 0.959 | 7.0 | 142 | 0.6 | 46 | 3334 |

## Analysis

1. **Compression works**: 47% page reduction, recall unchanged (lossless).
2. **v4 slightly slower at same cache %**: +0.2-0.4ms p50 across all cache ratios.
3. **Root cause: free expansion overhead**. v4 pages hold ~65 VIDs vs ~31.
   Each cache miss triggers bonus scoring for all co-located unvisited VIDs.
   At 5% cache: v4 does 310 bonus scores vs v3's 68 (4.6×), costing CPU.
4. **Unfair comparison**: same cache % gives v4 fewer absolute pages
   (5% of 804=40 vs 5% of 1526=76). Same absolute pages would favor v4.

## Next Steps

- Rerun with free expansions disabled (max_bonus=0) to isolate IO benefit
- Rerun with same absolute cache size for fair comparison
- Cap bonus_per_query more aggressively (e.g., 500 instead of 2000)
- Consider only scoring co-residents within distance threshold
