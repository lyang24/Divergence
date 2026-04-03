# EXP-GRAPHAGO: Activity-Neighborhood Graph Ordering

**Date**: 2026-03-31
**Instance**: i4i.xlarge (NVMe), EC2 us-east-1
**Dataset**: Cohere 100K, dim=768, k=100, ef=200, W=4, 5% cache, perq-cold
**Paper**: GraphAGO (SC'25, Xu et al.)

## Verdict: heavy_edge remains best layout

GraphAGO (both degree-based and trace-based activity) performs roughly equal to BFS
and significantly worse than heavy_edge on all IO metrics.

## Results (FP32 cosine, cold cache)

| layout      | recall | p50ms | p99ms | QPS   | mis/q | phy/q | upg/q |
|-------------|--------|-------|-------|-------|-------|-------|-------|
| bfs         | 0.956  | 8.4   | 11.8  | 103.2 | 24.5  | 136.5 | 134.7 |
| heavy_edge  | 0.956  | 7.6   | 10.2  | 115.2 | 21.4  | 109.0 | 104.0 |
| ago_degree  | 0.956  | 8.4   | 10.7  | 105.2 | 25.7  | 135.0 | 131.8 |
| ago_traced  | 0.956  | 8.4   | 10.8  | 104.6 | 26.2  | 134.4 | 130.0 |

upg/q = unique pages per query (from traced expansion VIDs mapped to adj_index pages).

## Analysis

**GraphAGO algorithm**: Two-phase ordering — (1) pack top hub_count=100 vertices by
activity at the front, (2) iterate remaining vertices by descending activity, eagerly
assigning each vertex's unassigned neighbors to get consecutive IDs (neighbor-runs).

**Why it loses to heavy_edge**: GraphAGO assigns consecutive IDs via neighbor-runs but
is not page-aware. Whether consecutive IDs land on the same 4KB page depends on record
sizes (2 + degree*4 bytes). At degree=32, records are 130 bytes, so ~31 fit per page.
GraphAGO's linear ID assignment creates neighbor-runs that may span page boundaries.

Heavy_edge's greedy page fill explicitly tracks page budget and picks the best-fitting
neighbor for each page. This page-awareness is the key advantage — it guarantees that
co-placed vertices actually share a page, not just consecutive IDs.

**Traced vs degree activity**: Nearly identical results (upg/q 130.0 vs 131.8). The
activity ordering heuristic matters less than the packing strategy. Both GraphAGO
variants produce similar page layouts because the neighbor-run phase dominates — once
you iterate by activity and assign neighbors, the hub phase (only 100 vertices) has
minimal impact.

**Comparison to affinity (VeloANN)**: Also tested in this session — affinity scored
upg/q ~112 vs heavy_edge's 104. All alternative orderings lose to heavy_edge's
page-aware greedy fill.

## Trace Collection Stats

- 200 trace queries (cold, BFS layout)
- 21,528 unique VIDs expanded (21.5% of 100K)
- 40,196 total expansions (~201 expansions/query)
- Reorder time: heavy_edge 2.3s, graphago_degree 0.0s, graphago_traced 0.0s

## Key Takeaway

Page-aware greedy packing (heavy_edge) > linear neighbor-run ordering (GraphAGO) >
BFS traversal order. The critical feature is tracking page capacity during assignment,
not the vertex ordering heuristic.
