# Paper Insights for Project Divergence

This document captures the concrete technical lessons from 4 papers read in full
(Gorgeous, VeloANN, B+ANN, OctopusANN/IO-Design-Space) plus the 14 papers
summarized in `research_summary.md`. The focus is on what to build and why —
distilled into decisions that directly inform Divergence's Rust implementation.

---

## 1. The I/O Bottleneck is Real, but CPU is the Surprise Constraint

The IISWC benchmarking paper (from research_summary.md) found that existing vector
databases utilize only 8.9% of NVMe SSD bandwidth. The intuitive assumption is
that disk is the bottleneck. It is not. **CPU is the bottleneck** — systems spend
most of their time computing distances, managing data structures, and waiting for
synchronous IO to complete, leaving the SSD idle.

VeloANN measured only 57% CPU utilization on Sift1M with DiskANN's synchronous
compute-IO model. The CPU literally idles while waiting for 4KB reads that take
~100μs each.

OctopusANN's latency breakdown across 4 datasets confirms this: IO accounts for
70-90% of total query latency. But this is not because the SSD is slow — it is
because the CPU issues one IO at a time and blocks.

**Lesson for Divergence:** The entire async execution model (coroutines, io_uring,
batched IO) is not a nice-to-have optimization. It is the fundamental architecture
choice that separates <50% throughput from >90% of in-memory performance. VeloANN
proves this is achievable at 10% memory footprint.

---

## 2. Record-Level Caching vs Page-Level Caching: The Definitive Argument

This is the single most validated finding across the literature. Three independent
papers converge on the same conclusion from different angles:

**VeloANN (empirical):** On Sift1M, 47.3% of vertices are never accessed during a
workload, yet only 0.1% of pages go untouched. A page-level buffer pool with LRU
achieves only 13% hit rate at 10% buffer ratio and 54% at 50%. The fundamental
problem: accessing a single hot vertex forces its entire page (including cold
co-resident vertices) into the cache, evicting other useful data.

**Tiered-Indexing (theoretical):** Formalizes this as the "lukewarm page" problem.
Page-level caching creates pages where some records are hot and others are cold.
The page stays in cache because of the hot records, but the cold records waste
memory. Record-level tiering achieves 3.7-4.5x improvement across B+tree, hash,
heap, and LSM-tree workloads under skew. The result generalizes across data
structures.

**Gorgeous (analytical):** Proves mathematically that caching adjacency lists alone
(without exact vectors) is superior when:

    S_a < (1 - σ) / σ * S_v

where S_a is adjacency list size, S_v is exact vector size, σ ≈ 0.5 is the
refinement ratio. For the Wiki dataset (S_v = 1536B, S_a = 200B), the
adjacency-only cache holds 88% of all nodes' graph structure vs only 10% with
the coupled adjacency+vector cache. The disk IO reduction is dramatically better.

**Lesson for Divergence:** Implement a record-level buffer pool from day one. Never
use mmap. The buffer pool should:
- Track individual vertex records, not pages
- Use a compact record mapping array (VectorId → slot/page, 1 bit residency)
- Employ clock-based eviction (sophisticated policies like LRU offer only marginal
  improvement for ANN workloads — VeloANN Table 1)
- On page read, load only the requested record + affinity-group records, discard
  the rest

---

## 3. Adjacency Lists are More Important Than Vectors

Gorgeous's key insight, validated empirically: during graph traversal, every visited
node needs its adjacency list (to find the next hop), but only a small fraction
(σ * D where σ ≈ 0.5) need their exact vectors (for final refinement).

The numbers are striking. With a 10% memory budget on the Wiki dataset:
- Adjacency-only cache: holds adjacency lists for 88% of nodes
- Node cache (adjacency + vector): holds complete records for only 10% of nodes
- The adjacency-only cache achieves significantly better throughput AND continues
  improving with more memory, while the node cache plateaus quickly

This has cascading implications for the storage layout:

1. **Memory allocation priority:** compressed PQ codes (always in memory) → adjacency
   lists (graph cache, high priority) → exact vectors (node cache, low priority,
   fetch on demand)

2. **Disk block design:** each node's disk block should contain its own adjacency list
   + exact vector + neighbors' adjacency lists (graph-replicated layout). This way,
   reading one block for refinement also prefetches graph structure for future hops.

3. **Two-stage search algorithm:** the search stage operates almost entirely on
   in-memory data (compressed codes for approximate distances, cached adjacency
   lists for graph traversal). Only the refinement stage touches disk for exact
   vectors of the top-σ*D candidates.

**Lesson for Divergence:** The object decomposition in the README (routing metadata,
candidate blocks, compressed codes, exact vectors) is validated. Each object type
has a different access frequency and should be independently cacheable. Adjacency
lists (routing metadata / candidate blocks) should always be prioritized for memory
residency over exact vectors.

---

## 4. The Coroutine-Based Async Execution Model

VeloANN provides the most detailed blueprint for what Divergence's data plane
should look like. The key idea: model each query as a coroutine. When the
coroutine hits a cache miss, it suspends and yields to the scheduler. The
scheduler picks up another ready coroutine. When the IO completes, the original
coroutine is resumed.

**VeloANN's scheduler loop (Fig. 3b):**
```
fn schedule(ctx, future):
    // Consume all tasks
    while let Some(t) = ctx.tasks.pop():
        t.run()            // resume the coroutine
    // Submit pending IO and process completions
    ctx.io_uring.submit()
    ctx.io_uring.park()    // block if nothing ready
```

**Batch size formula:** B = ⌈α · I/T⌉ coroutines per thread, where:
- I = average disk IO latency (e.g., 100μs for NVMe random read)
- T = average compute time between IO events
- α ≈ 1.0-2.0, tuned empirically

For high-dimensional data (768d+), compute is expensive enough that B=2 suffices.
For low-dimensional (128d), T is small so B=4-8 needed to keep CPU busy.

**Why coroutines over thread-pool + futures:**
- No cross-thread synchronization for the buffer pool (thread-per-core owns its
  own io_uring instance)
- Stackless coroutines (Rust async/await) have minimal overhead — less than a
  last-level cache miss per context switch
- Predictable tail latency: no thread scheduling jitter from the OS

**In Rust terms:**
- Each query is an `async fn search(query, params) -> Vec<(VectorId, f32)>`
- Cache misses do `let record = buffer_pool.get_or_load(vid).await` which may
  suspend if the record is on disk
- The scheduler is a manual executor (not tokio — we need io_uring integration
  and thread-per-core pinning)
- Consider `monoio` or hand-rolled executor on top of `io-uring` crate

**Lesson for Divergence:** The async runtime is not optional and cannot be bolted on
later. Design the search algorithm from the start as async, with explicit
yield/resume points at every potential IO. Use Rust's native async/await — it maps
perfectly to VeloANN's coroutine model.

---

## 5. Cache-Aware Beam Search: The Critical-Path Optimization

Standard beam search explores candidates in strict distance order. This means if
the best candidate is on disk, the search stalls waiting for IO — even if the
second-best candidate is already in memory and nearly as good.

VeloANN's cache-aware beam search (Algorithm 2) fixes this:

```
v ← top candidate from P (closest to query)
C ← top-B candidates from P  // look-ahead set

if v is on disk:
    for c in C:
        if c is in memory:
            v ← c; break     // pivot to in-memory candidate
        else:
            prefetch(c)       // issue async IO for on-disk candidates in C

// Process v (compute distances to neighbors, update candidate list)
record = read(v).await       // suspend if on disk
```

This has two effects:
1. **Amplifies effective cache hit rate** by a factor of B — the probability of
   finding ANY in-memory candidate in the top-B is much higher than just the top-1
2. **Hides IO latency** — on-disk candidates in the look-ahead set get prefetched
   asynchronously while the in-memory candidate is being processed

VeloANN's ablation study shows each optimization layer contributes:
- +Async (coroutines): 1.8x throughput over synchronous baseline
- +Record (record-level buffer): 1.23x over async-only
- +Prefetch: 2.15x over async-only
- +CBS (cache-aware beam search): 1.5x over prefetch, lowest latency overall

The full stack achieves 2.2x throughput and 1.5x latency improvement over the
baseline at 95% recall.

**Lesson for Divergence:** Implement beam search with the cache-aware pivot from
the start. The look-ahead set size B (beam width) is the key tuning knob — W=4
is optimal for VeloANN on high-dim data, with diminishing returns beyond that.

---

## 6. On-Disk Layout: Variable-Size Records Beat Fixed-Size Pages

Traditional systems (DiskANN, Starling) use fixed-size records padded to fill
pages. This causes severe internal fragmentation, especially at high dimensions:

- Sift1M (128d): ~0% fragmentation (records are small, many fit per page)
- Gist1M (960d): **52% fragmentation** (one record nearly fills a page, rest wasted)
- For 768d float32: a single vector is 3KB, adjacency list ~256B → 3.25KB record,
  wastes 0.75KB per 4KB page (19% fragmentation)

VeloANN's solution: **variable-size compressed records** in a slotted page layout.
- Vectors compressed to 4 bits/dimension with ExtRaBitQ (768d → 384 bytes vs 3072)
- Adjacency lists compressed with delta encoding + Elias-Fano
- Result: a 768d record shrinks from ~3.25KB to ~500-600 bytes
- Multiple records per 4KB page → dramatically less fragmentation
- Disk consumption: up to **10x smaller** than DiskANN (VeloANN Table 3)

The slotted page layout (B-tree inspired):
- Page header (5 bytes): metadata
- Slot array: grows from the front, fixed 9 bytes per slot (VID, Color, Length, Offset)
- Data heap: grows from the back, variable-size records
- Sorted by VID for binary search lookups

**Lesson for Divergence:** Use compressed, variable-size records from the start.
The slotted page layout is well-understood (standard in relational databases) and
handles variable-size records cleanly. The space savings compound — smaller records
mean more records per page, which means better co-placement, which means fewer IOs.

---

## 7. Affinity-Based Record Co-Placement

VeloANN's key insight about co-placement: don't use adjacency-based co-location
(like Starling's graph reordering). Use **distance-based affinity**.

Why? Modern proximity graphs (Vamana, HNSW) optimize for navigational efficiency,
not geometric proximity. They include long-range edges for fast convergence, which
means graph neighbors are often geometrically distant. Co-locating graph neighbors
on the same page puts geometrically unrelated vectors together — polluting cache
with irrelevant data.

Instead, VeloANN defines affinity by distance: vectors v_i and v_j are affine iff
d(v_i, v_j) ≤ τ. The threshold τ is set to the average 5th-percentile
distance-to-centroid across all clusters (computed during graph construction).

The beauty of this approach: affinity identification is **free** — during graph
construction, you already compute distances to candidate neighbors. Just record
those within threshold τ:

```
for each vertex p during graph construction:
    [V, D] = GreedySearch(G, p, l_b)
    A_p = {v : (v, d) in (V, D) where d ≤ τ and |A_p| < k}
    S[p] = A_p
```

No extra distance computations needed. k is set relative to page capacity to
prevent affinity groups from spanning multiple pages.

Co-located records are tagged with a Color byte. On a cache miss for any record
on a page, all same-Color records are proactively loaded into the buffer pool.
This turns a single 4KB read into multiple useful cache entries.

VeloANN's evaluation: τ = 5% (of dataset diameter) strikes the optimal balance.
τ = 0% (no co-placement, single-record loading) has 1.28x worse latency.
τ = 10% (too relaxed) degrades because irrelevant records pollute the buffer.

**Lesson for Divergence:** Integrate affinity identification into graph construction.
Use distance-based affinity, not graph-adjacency-based. The Color tag mechanism
for proactive loading is simple and effective.

---

## 8. The Navigation Graph (MemGraph) is the Highest-Value Memory Investment

OctopusANN's systematic evaluation of individual optimizations reveals a clear
hierarchy of value:

| Optimization     | Standalone QPS Impact | IO Reduction |
|------------------|----------------------|--------------|
| MemGraph         | **+54.2%**           | -32.5%       |
| PageShuffle+PSe  | +28.9%               | -28.3%       |
| DynamicWidth     | +12.5%               | -25.2%       |
| Cache (SSSP)     | modest               | modest       |
| Pipeline         | **counterproductive** | increases IO |

MemGraph: sample 0.1% of vertices, build a small in-memory Vamana graph. Use it
as a fast entry point selector. This dramatically shortens search paths by
starting from high-quality entry points instead of random/centroid-based ones.

The overhead is minimal: ~30-50MB of memory for 100M vectors. The MemGraph
sampling and construction add negligible time to index building.

Gorgeous similarly uses a "navigation index" built from 0.5% sampled vectors.
If the nav index doesn't improve performance (determined by profiling on sample
queries), it's simply disabled (e.g., Text2Image dataset).

**Pipeline search is counterproductive** under high concurrency — a critical
finding from OctopusANN. Pipeline search issues speculative IO before confirming
that a candidate is worth exploring. Under high concurrency, this creates
contention and wastes SSD bandwidth. VeloANN's approach (coroutine-based async
with cache-aware beam search) is strictly superior because it doesn't speculate
— it makes informed decisions about what to prefetch based on the look-ahead set.

**Lesson for Divergence:** Build the MemGraph early (Phase 1 of search pipeline).
It provides the single biggest performance gain for the least engineering effort.
Do NOT implement pipeline/speculative IO — use VeloANN's async coroutine model
instead.

---

## 9. Dynamic Beam Width and the Approach/Converge Phases

OctopusANN (building on PipeANN's insight) identifies two distinct phases in
graph traversal:

1. **Approach phase:** search is far from the target region, making large hops.
   Each hop's destination is far from the previous — broad exploration.
   Wider beam = more speculative reads = wasted IO.

2. **Converge phase:** search is near the target, making small refinement hops.
   Candidates are close together — higher locality, higher cache hit potential.
   Wider beam = more useful exploration = better recall.

DynamicWidth starts with a narrow beam ω (approach phase, avoid wasted reads)
and gradually widens to ω_max as the search converges. This alone provides ~25%
IO reduction and ~50% QPS improvement at relaxed recall targets.

However, DynamicWidth exhibits **severe degradation** at high recall on
high-dimensional datasets. When recall targets are stringent (>95%), the number
of search iterations increases significantly, and the widening beam causes
massive IO. OctopusANN recommends combining DynamicWidth with MemGraph (which
shortens search paths, reducing the opportunity for DynamicWidth to blow up).

**Lesson for Divergence:** Implement dynamic beam width as a configurable strategy.
Default: start at ω=1, max ω=4-8. Always combine with MemGraph for path
shortening. For the query planner (future), this is a concrete lever: adjust
ω based on current resource state (queue depth, cache hit rate).

---

## 10. Graph-Replicated Disk Blocks vs Separated Layout

Gorgeous evaluates three disk layouts:

1. **DiskANN-style (co-located):** each block = adjacency list + exact vector for
   one node. Starling adds graph reordering for neighbor co-location.

2. **Separation layout:** graph blocks (adjacency lists only) + vector blocks
   (vectors only). Hypothesis: two separate reads are better because you can
   read graph structure without loading vectors.

3. **Graph-replicated layout:** each block = node's adjacency list + exact vector +
   R nearest neighbors' adjacency lists. Trades disk space for IO savings.

Result: **graph-replicated wins decisively** (Gorgeous Fig. 8, 17). The separation
layout is actually worse than DiskANN's co-located layout because during
refinement, ALL top-ranking vectors need to be loaded from disk — and in the
separated layout, none of them were fetched during the search stage (which only
read graph blocks). This creates additional IO that negates the graph-block
savings.

The graph-replicated layout works because adjacency lists are small (200-256
bytes) relative to 4KB pages. After placing one node's full record (~500-3000
bytes depending on dimension), there's room for 4-15 neighbor adjacency lists.
When these neighbors are traversed in subsequent hops, their adjacency lists are
already in the buffer pool — saving one disk read per neighbor.

The disk space overhead is moderate: below 2x for high-dimensional vectors
(because adjacency lists are much smaller than vectors). Given the low cost of
SSD capacity, this is a favorable trade.

**Lesson for Divergence:** Use graph-replicated disk blocks. The R parameter
(number of packed neighbor adjacency lists) should be configured based on vector
dimensionality: higher R for lower dimensions (more room per page), lower R for
higher dimensions. Constrain each node's block to at most one 4KB page.

---

## 11. B+ANN: An Alternative Index Architecture Worth Understanding

B+ANN (2511.15557) proposes a fundamentally different approach from graph-based
methods: a B+Tree built from hierarchical k-means clustering, with skip-edge
connections between leaf nodes for fine-grained graph traversal.

Key features:
- **Hierarchical clustering** partitions the space into leaves containing similar
  vectors. Each leaf is a cluster with a centroid key.
- **B+Tree structure** stores centroids as keys, vectors in leaves. Inner nodes
  stay in memory, leaves can be on disk.
- **Skip-edge connections** between leaf nodes enable HNSW-style greedy search
  across cluster boundaries (intra-connections within a leaf, inter-connections
  between neighboring leaves).
- **Batch-friendly:** leaf nodes contain many vectors → distance computations
  become matrix-matrix operations → SIMD/GPU friendly.
- **Semantic views:** for temporally correlated queries (RAG conversations), extract
  a subtree as a cached view that serves subsequent related queries.

Performance: 10x speedup over HNSW at Recall@10 on ARM64. 24x faster index
construction than DiskANN. B+ANN can reach 99.8% Recall@10 on SIFT-1B in <9ms.

**Relevance to Divergence:**
- B+ANN validates the index-agnostic execution model — the 5-stage pipeline
  (Router → Candidate Producer → Scorer → Pruner → Refiner) works for tree-based
  indexes too, not just graph-based ones.
- The B+Tree Router stage is naturally hierarchical: top-down tree traversal
  narrows to promising leaf clusters.
- Leaf-level batch distance computation (matrix multiplication) could enable
  future GPU acceleration more cleanly than graph-based pointer chasing.
- The semantic view concept aligns with Divergence's future query planning: for
  workloads with temporal correlation, the planner could maintain hot views.

**Lesson for Divergence:** Design the execution pipeline to be index-agnostic from
the start. The Router, Candidate Producer, Scorer, Pruner, and Refiner stages
should be trait-based so that both graph-based (Vamana/HNSW) and partition-based
(IVF/B+ANN) index backends can plug in. Start with Vamana (best understood,
most validated), but don't hardcode graph assumptions into the pipeline.

---

## 12. What Combinations Work (OctopusANN's Composition Analysis)

OctopusANN is the only paper that systematically tests combinations of
optimizations. This is invaluable because individual technique evaluations can
be misleading — some techniques are complementary, others interfere.

**Best combination (C5 = OctopusANN):**
MemGraph + PageShuffle + PageSearch + DynamicWidth

Achieves:
- 4.1-37.9% higher QPS than Starling
- 87.5-149.5% higher QPS than DiskANN
- at Recall@10 = 90%

**Key synergy findings:**

1. **PageShuffle + PageSearch are complementary:** PS improves data locality
   (neighbors on same page), PSe exploits that locality (compute distances for
   all co-resident vectors, not just the target). Neither is effective alone,
   but together they provide 28.9% QPS improvement.

2. **Pipeline + DynamicWidth are complementary but Pipeline is dominated:**
   Pipeline alone is counterproductive (speculative reads waste bandwidth).
   Combined with DynamicWidth, Pipeline recovers but still underperforms
   DynamicWidth alone. Verdict: skip Pipeline, use DynamicWidth.

3. **MemGraph dominates as a standalone:** provides the biggest single-factor
   improvement and compounds well with everything else. It's also the cheapest
   to implement (small sampling + graph construction, ~30MB memory overhead).

4. **For high-dimensional data, layout optimizations become less effective:**
   When each 4KB page holds only 1 record (e.g., 960d vectors), PageShuffle
   and PageSearch provide no benefit. The IO amplification problem for
   high-dimensional data remains an open research question.

**Practical recommendation from OctopusANN (Fig. 24 decision tree):**
- If enough memory → use in-memory indexes (HNSW, Vamana)
- If memory constrained but need high recall + high concurrency → OctopusANN
  (MemGraph + PS + PSe + DW)
- If memory constrained + low concurrency → PipeANN
- If memory constrained + low recall acceptable → SPANN (IVF-based)

**Lesson for Divergence:** Implement in this priority order:
1. MemGraph (biggest bang for buck)
2. Record co-placement / affinity (Divergence's version of PageShuffle)
3. Cache-aware beam search (Divergence's version of DW + PSe combined)
4. Skip Pipeline — coroutine-based async with prefetching is strictly better

---

## 13. Quantization: ExtRaBitQ is the State of the Art

VeloANN uses ExtRaBitQ with a two-level hierarchy:
- **1-bit binary codes** (1 bit per dimension): kept in memory for ultra-fast
  initial distance estimation. For 768d, that's only 96 bytes per vector.
- **4-bit extended codes** (4 bits per dimension): stored on disk alongside
  adjacency lists for more accurate approximate distances. For 768d, that's
  384 bytes per vector.

This achieves a ~4.5x compression ratio over the original vectors while
maintaining accuracy sufficient for high recall.

Gorgeous uses standard PQ (Product Quantization) as a black box, with the
compression ratio determined by profiling on a 1% sample. The optimal ratio
varies by dataset and modality (cross-modal datasets like Text2Image need less
compression to maintain accuracy).

OctopusANN's evaluation confirms: PQ is the baseline memory layout optimization
that all other techniques build on top of. Without PQ, nothing else works well
because the memory footprint is too large.

**Lesson for Divergence:** Start with PQ (simpler, well-understood, widely
implemented). Add RaBitQ/ExtRaBitQ as an advanced option once the pipeline is
working. The quantizer should be trait-based so different methods can be swapped.
Always profile the optimal compression ratio on a sample of the target dataset.

---

## 14. Thread Scaling and Concurrency

VeloANN achieves near-linear scaling to 32 threads (1.39x over PipeANN, 1.82x
over DiskANN/Starling at 32 threads on Wiki). The key enablers:

1. **Thread-per-core:** each thread owns its io_uring instance, its own local
   ready queue, and operates on a partition of the query stream. No shared
   mutable state for the hot path.

2. **Record-level buffer pool:** the record mapping array is read-mostly (lookups)
   with CAS-based updates (state transitions). No global locks.

3. **Lock-free eviction:** clock-based sweep with CAS transitions. The eviction
   coroutine runs concurrently with query processing on the same thread,
   minimizing contention.

Gorgeous achieves 83% average throughput improvement over DiskANN and 60% over
Starling across different thread counts, with the gap widening at higher thread
counts. This is because Gorgeous requires fewer disk accesses per query, making
it less dependent on per-thread disk bandwidth.

**Lesson for Divergence:** The concurrency model described in the README
(thread-per-core, no global locks, lock-sharded structures, optimistic reads)
is well-validated. Implement it exactly as described. Use `core_affinity` crate
for thread pinning. The io_uring instance must be per-thread (not shared).

---

## Summary of What to Build First

Based on the papers' measured impact (from highest to lowest):

| Priority | Component                        | Expected Impact | Paper Source        |
|----------|----------------------------------|-----------------|---------------------|
| 1        | Async coroutine runtime + io_uring | 1.8x throughput | VeloANN            |
| 2        | Record-level buffer pool          | 1.23x on top    | VeloANN, Tiered-Idx|
| 3        | MemGraph (in-memory nav graph)    | +54% QPS        | OctopusANN         |
| 4        | Prefetching                       | 2.15x on top    | VeloANN            |
| 5        | Cache-aware beam search           | 1.5x on top     | VeloANN            |
| 6        | Graph-replicated disk blocks      | +60% throughput  | Gorgeous            |
| 7        | Affinity co-placement             | -28% IO latency  | VeloANN            |
| 8        | Dynamic beam width                | +50% QPS (relax) | OctopusANN         |
| 9        | Two-stage search (σ refinement)   | -50% exact reads | Gorgeous            |
| 10       | Variable-size compressed records  | 10x disk saving  | VeloANN            |

Note: impacts are not independent — they compound. VeloANN's full stack (items
1-5 + 7 + 10) achieves 5.8x throughput over DiskANN. Gorgeous's stack (items
6 + 9 + graph-prioritized cache) achieves 1.6x over Starling. OctopusANN's
best combination (items 3 + 7-variant + 8) achieves 1.37x over Starling.

---

## 15. Drop the Hierarchy: Flat NSW Matches HNSW for d >= 32

**Source:** "Down with the Hierarchy" (FlatNav), Munyampirwa et al., Feb 2025.

Rigorous benchmarking across 13 datasets (1M to 100M vectors, d=96 to d=1536)
shows that a flat navigable small world graph (single-layer, no hierarchy) matches
HNSW on both median and P99 latency at all recall levels for high-dimensional data.

**Key findings:**
- For d >= 32, hierarchy provides zero measurable benefit on latency or recall.
- FlatNav saves 38% peak memory on BigANN-100M and 39% on Yandex-DEEP-100M.
- The "Hub Highway Hypothesis" explains why: in high-dimensional spaces, hub nodes
  (a small subset that appears disproportionately in neighbor lists) naturally form
  a well-connected routing structure. These hubs serve the same function as
  hierarchical layers — routing queries from far away to the target neighborhood.
  The hierarchy is redundant because the flat graph already has fast highways.
- For d < 32, HNSW does provide a speedup. Hierarchy matters only in low dimensions.
- Hub connectivity increases with dimensionality (skewness of node access
  distribution increases with d for L2 distance).

**Action for Divergence:** Drop upper layers entirely. Our target is 768d+ embeddings.
Build a flat NSW: same beam search, same heuristic pruning, same M/ef parameters,
just one layer. This eliminates:
- `upper_links: Vec<RwLock<Option<Vec<Vec<u32>>>>>` (allocation-heavy, lock-heavy)
- Greedy descent code path and level assignment RNG
- The flatten-to-offset-table logic in `build()`
- ~40% code complexity reduction, ~35% memory reduction, zero recall loss.

---

## 16. Accelerating HNSW Construction with Compact Coding (Flash)

**Source:** "Accelerating Graph Indexing for ANNS on Modern CPUs", Wang et al.,
SIGMOD 2025.

**Construction-only optimization.** Does not change search at all. Profiles HNSW
construction with `perf`: distance computation is 90% of indexing time (48-49%
memory accesses, 42-48% arithmetic). Flash replaces FP32 distances during
insertion with PCA+PQ compact codes in a SIMD-friendly columnar layout, achieving
10-22x faster construction with same graph quality.

**One useful data point for us:** the 90% figure confirms SIMD distance is the #1
priority for construction speed. Everything else in this paper is deferred until
build time on 100M+ datasets becomes a bottleneck.

---

## 17. Multi-Stage Search with Density-Aware Quantization (AQR-HNSW)

**Source:** "AQR-HNSW: Accelerating ANN Search via Density-aware Quantization and
Multi-stage Re-ranking", Tewary et al., Feb 2026.

**Key observation:** 93% of evaluated nodes during HNSW search are ultimately
rejected — they don't influence the final result. Only 7% are "decision nodes."
Yet all nodes receive full FP32 distance computation. This is massively wasteful.

**Three high-value techniques:**

### 17.1 Density-Aware Adaptive 8-bit Quantization
Per-dimension adaptive encoding based on local data density:
- Compute local density ρ_i for each point (inverse of avg k-NN distance).
- Global heterogeneity δ = (max(ρ) - min(ρ)) / (max(ρ) + ε).
- Per-dimension percentile bounds [P_l, P_h] adapt to δ:
  - Dense regions → tighter bounds (more precision where discrimination matters)
  - Sparse regions → wider bounds (less precision needed)
- Scale factor per dimension: scale_j = 255 / (max_j - min_j + ε)
- Distance preservation via per-dimension weights w_j and global s_dist factor.
- Result: 4x compression with better distance preservation than naive min/max SQ.
- Also adjusts M_i and ef_construction per density level — denser regions get more
  connections and deeper search during construction.

### 17.2 Three-Stage Progressive Refinement Search
```
Stage 1: Coarse Quantized Search
  - Quantize query to 8-bit
  - Search graph using quantized distances (uint8 arithmetic, SIMD-friendly)
  - Retrieve top-N_c coarse candidates with ef_search = N_c * m_ef

Stage 2: Asymmetric Distance Refinement
  - For each candidate: decode quantized vector → reconstructed float vector
  - Compute asymmetric distance: full-precision query vs reconstructed vector
  - Re-rank candidates (no exact vector load needed)
  - Early termination: if gap between k-th and (k+1)-th candidate > τ_gap
    OR ratio > τ_ratio, stop here — top-k is confident.

Stage 3: Exact Reranking (only when needed)
  - Load exact vectors for top-N_rerank candidates
  - Compute exact distances, final re-rank
```
- Stage 2 is the key innovation: asymmetric distance avoids loading exact vectors
  while being much more accurate than quantized distance. Catches most ranking
  errors from Stage 1 without any IO.
- Early termination saves 35% of exact computations. Thresholds: τ_gap ∈ [0.01,0.03],
  τ_ratio ∈ [1.01,1.05] work well across datasets.
- Recommended params for 95%+ recall: N_c=55, N_rerank=20, τ_gap=0.012,
  τ_ratio=1.010, m_ef=3.

### 17.3 SIMD Acceleration Across All Three Stages
- Stage 1 (uint8): load 16 uint8 pairs into 128-bit registers, compute absolute
  differences, widen to 16-bit, square, accumulate with parallel multiply-add.
- Stage 2 (float query encoding): vectorize float→uint8 conversion across dims.
- Stage 3 (exact): FMA instructions: (q[j] - x[j])^2 + sum in single operation.
- Architecture detection at compile time: AVX-512 > AVX2 > SSE2 > NEON > scalar.

**Result:** 2.5-3.3x higher QPS than baseline HNSW at 95%+ recall. 5x faster index
construction. 75% memory reduction. 3.2x lower P99 on GIST-1M.

**Action for Divergence:** The 3-stage search pipeline maps directly to Phase 6:
- Stage 1 = Router + Candidate Producer (quantized graph traversal)
- Stage 2 = Scorer (asymmetric distance, no IO)
- Stage 3 = Refiner (exact vector load, selective)
Early termination is trivial to implement and high payoff. Density-aware SQ is
a better quantizer than naive min/max — implement when adding quantization.

---

## 18. SSD-Optimized HNSW with io_uring + Colocation (Turbocharging VectorDB)

**Source:** "Turbocharging Vector Databases using Modern SSDs", Shim et al.,
PVLDB 2025. Implemented in pgvector (PostgreSQL).

Comprehensive study of I/O optimization for disk-resident HNSW. Three orthogonal
techniques, each independently validated:

### 18.1 io_uring Async Pipelining for Neighbor Scan
The core problem: pgvector's neighbor scan is sequential — read page, compute
distance, read next page. SSD utilization is 1.98%.

Solution (`pgv-async-iou`):
```
function EvaluateNeighbors(Q, N):
    cached ← {n ∈ N | n ∈ buffer_cache}
    uncached ← N \ cached
    Submit uncached to io_uring           // async batch read
    for c ∈ cached do
        Compute Distance(Q, c)            // compute while IO in flight
    while uncached ≠ ∅ do
        r ← wait for any IO completion    // poll, not block-all
        Compute Distance(Q, r)
        Remove r from uncached
```
- Don't wait for all IOs — process each completion immediately.
- Poll io_uring completion queue (`min_complete=6` optimal) to avoid both
  busy-wait (CPU waste) and block-too-long (latency spikes).
- Result: 8.55x QPS improvement on Samsung PM1743 (highest parallelism SSD).
  Even on older SSDs: 3.82x (SSD-D) to 6.4x (SSD-C).
- Cache miss penalty reduced by 80.5-94.1% via parallel IO.
- Index construction also benefits: 85% faster insertion of 1% incremental data
  on pre-built 99% index (383s → 2566s for pgv-async-iou vs pgv-orig on DBpedia).

### 18.2 Spatially-Aware Insertion Reordering
Before building the index, sort vectors by spatial locality. Similar vectors
inserted together share graph neighbors → cache hit ratio jumps dramatically.

Two strategies tested:
- **PCA-based**: project to 1D via first principal component, sort. Fast (10.39s
  for 1M vectors), moderate hit ratio improvement (74.35%).
- **K-means clustering**: cluster into groups, insert cluster-by-cluster. Slower
  (510.97s for 1M) but higher hit ratio (86.58%).

Combined with io_uring: PCA reordering + pgv-async-iou → 90.3% faster index build.
K-means + pgv-async-iou → 84.9% faster.

- Total write volume also drops (46.4% reduction with PCA) because higher cache
  hit ratio means fewer buffer evictions → fewer dirty page flushes.
- Reordering does NOT affect graph structure or query performance — the HNSW
  algorithm produces statistically identical graphs regardless of insertion order.
- For incremental indexing: only reorder the new batch (e.g., 5% new data), insert
  into existing 95% index. Still 67% build time reduction.

### 18.3 BNF (Block Neighbor Frequency) Colocation
During disk layout, assign each node to the page containing the most of its
graph neighbors. This is locality-preserving colocation for HNSW:

```
for each node n:
    C ← ∅
    for each partition p ∈ P:
        count ← count_neighbor_overlap(n, p)
        C ← C ∪ {(p, count)}
    Sort C by count descending
    for (P, count) in C:
        if P ∈ insert_pages:
            page ← find_available_page(P)
            if is_page_full(page): page ← create_extended_page(P)
            insert_into_page(n, page)
            return
    insert_into_page(n, fallback_page)   // no good partition found
```

- Saturates at ~64 nodes per partition for most datasets.
- Hit ratio: 15.84% → 51.19% on Deep dataset (3.23x improvement).
- For locality-aware incremental insertion: new nodes placed near their graph
  neighbors' existing pages, maintaining locality over time.
- Combined effect of all three techniques: 7.6x higher insertion throughput,
  hit ratio from 43.5% to 79.1%.

### 18.4 Comparison with DiskANN
- pgv-ours (all three techniques) beats DiskANN on search under cache-friendly
  workloads (MMLU queries) and matches on random queries.
- pgv-ours supports dynamic updates (insert/delete); DiskANN is static.
- End-to-end index build: pgv-ours is 10x faster than DiskANN on C4-100K (1536d)
  because DiskANN's PQ codebook computation is expensive for high-dim data.
- At larger scales (Deep-100M): pgv-ours 3.6x faster build time.

**Key numbers:**
| Metric | pgv-orig | pgv-async-iou | pgv-ours (all) |
|--------|----------|---------------|----------------|
| QPS (DBpedia-1M, 10% buffer) | 13.02 | 105.81 | 111.4 |
| Cache hit ratio (GloVe, 50% buffer) | 59.24% | ~65% | 79.1% |
| SSD utilization | 1.98% | >40% | >40% |
| Build time (Deep-1M) | 787s | - | 513s |

**Action for Divergence:**
1. io_uring async pipelining is Phase 5 — already planned. This paper provides
   the concrete poll-based algorithm and min_complete tuning (6 is optimal).
2. Insertion reordering is FREE — sort by PCA first component before building.
   Add to Phase 2 index construction as a preprocessing step.
3. BNF colocation is Phase 3 disk layout — refines our existing affinity-based
   co-placement plan with a concrete algorithm and measured saturation point (64
   nodes per partition).

---

## Prioritized Action Items from Papers 15-18

| Priority | Action | Source | Impact |
|----------|--------|--------|--------|
| **1** | **Drop hierarchy → flat NSW** | FlatNav (§15) | -35% memory, -40% code complexity, zero recall loss at d≥128 |
| **2** | **SIMD distance** (already identified) | Flash confirms 90% of build time is distance (§16) | 8-16x on hot path |
| **3** | **3-stage search: quantized → asymmetric → exact** | AQR-HNSW (§17) | 2.5x QPS, 35% fewer exact vector loads |
| **4** | **io_uring async neighbor scan with pipelining** | Turbocharging (§18.1) | 8.5x QPS on NVMe |
| **5** | **Insertion reordering before build** (PCA sort) | Turbocharging (§18.2) | 3x cache hit ratio, essentially free |
| **6** | **BNF colocation on disk pages** | Turbocharging (§18.3) | 3.2x cache hit ratio improvement |
| **7** | **Early termination in search** | AQR-HNSW (§17.2) | 35% fewer exact computations, trivial to implement |
| **8** | **Density-aware 8-bit SQ for build + search** | AQR-HNSW (§17.1) | 4x compression, 5x faster build |

Items 1-2 are immediate (current Phase 2 work). Items 3-7 map to Phases 3-6.
Item 8 is a build-time optimization for when scale demands it.
