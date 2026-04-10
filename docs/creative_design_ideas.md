# Creative Design Ideas: Beyond Paper Implementations

**Date**: 2026-03-31
**Context**: Divergence is io_uring + thread-per-core + SAQ-in-DRAM. Papers (PageANN, VeloANN, DiskANN, PipeANN) provide useful techniques but don't fully exploit our architecture. This doc captures original ideas that go beyond paper re-implementation.

---

## The Fundamental Constraint

Every query does ~201 **serial** page reads (blk/q ≈ ef+1). Prefetch hides latency but doesn't reduce the count. The serial dependency: you can't know which node to expand next until you've scored the current node's neighbors. Every optimization attacks this from a different angle.

---

## Idea 1: Free Expansions from Co-Located Records ★★★

### Problem

With heavy-edge layout, a 4KB page holds ~10-20 adjacency records. When we read a page for node X, we pay for 4KB but only use ~100 bytes (X's record). The other ~3.9KB of co-located records sit in our cache buffer, untouched.

### Design

When we read a page for node X, eagerly decode ALL records on that page. For every co-located node Y that hasn't been visited, compute SAQ distance (free — DRAM, zero IO) and push into the beam if non-dominated. These are "bonus expansions" at zero IO cost.

**Why this is different from PageANN**: PageANN restructures the graph into a page-node graph. We don't need to. Our BFS/heavy-edge layout already co-locates graph neighbors. We just need to exploit the data we already paid to read.

### Implementation

Need an inverted index: page_id → list of (vid, offset, degree). Build once at startup from adj_index, ~100KB for 100K vectors. On every page read, iterate the inverted index for that page, decode all co-located records, SAQ-score unvisited nodes.

### Expected Impact

If each page has ~15 records and 30% of bonus expansions are useful, that's ~4-5 free useful expansions per IO. The beam converges faster with more candidates per IO → needs fewer IOs to reach the same recall. Target: 20-40% blk/q reduction.

### Risk

SAQ distance computations are cheap (~1μs for 768d) but not free. 15 extra distances per expansion × 201 expansions = 3015 extra SAQ distances. At ~1μs each that's ~3ms of CPU — may eat IO savings. Mitigate by only scoring co-located nodes that are graph neighbors of any beam member (higher utility probability).

### Verification

- blk/q must decrease ≥ 20% at iso-recall
- bonus_scored/q, bonus_pushed/q, bonus_useful% metrics
- CPU overhead (dst_ms increase) vs IO savings (io_wait_ms decrease)

---

## Idea 2: Two-Hop-Ahead Speculative Prefetch ★★

### Problem

Current prefetch: look at top-W candidates in the beam → prefetch their pages. This is 1-hop-ahead. By the time we pop the candidate, the page may or may not have arrived.

### Design

After scoring expansion X's neighbors, the top-1 scored neighbor Z is the most likely next expansion. Z's page_id is in adj_index (DRAM). Prefetch Z's page immediately — don't wait for Z to be popped from the beam. This is effectively 2-hop-ahead prefetch.

**Why io_uring makes this special**: Submitting an extra SQE costs near-zero (especially with SQPOLL). If the speculation is wrong, the page still enters cache (useful for future expansions).

### Implementation

After the neighbor scoring loop, find the best-scored unvisited neighbor. Look up its page_id in adj_index. If not already resident, issue prefetch_hint. ~5 lines of code.

### Expected Impact

Hit rate should be >50% (best neighbor often becomes next expansion). When it hits, the page is already in cache when popped — saving one IO wait. At 50% hit rate over 201 expansions, ~100 zero-wait expansions. Target: 10-20% p50 improvement.

### Risk

Low. Worst case: wasted prefetch (page evicted before use). With 5% cache, the cache is large enough to absorb speculative pages.

### Verification

- spec_prefetch_issued/q, spec_prefetch_hit% metrics
- p50 improvement without blk/q change (same IOs, less waiting)

---

## Idea 3: SAQ Neighbor Gating ★★

### Problem

All non-dominated neighbors enter the candidate heap, generating future page reads. 54% of expansions are wasted (add 0 neighbors). No limit on pushes per expansion.

### Design

After SAQ-scoring all neighbors, only push the Top-T best into the candidate heap (T = ceil(degree × gate_ratio)). Fewer candidates → fewer expansions → fewer page reads.

**Pure search-side change. Zero infrastructure cost.** SAQ distances are already computed.

### Implementation

See `docs/impl_saq_gating.md` for full specification.

### Expected Impact

20-40% blk/q reduction at iso-recall. Conservative gate_ratio=0.75 should be safe; aggressive gate_ratio=0.33 may lose 1-2% recall.

### Risk

Medium. SAQ proxy has σ ≈ 0.5 relative error. If ranking is too noisy, gating filters good neighbors. gate_min=4 floor prevents total starvation.

### Verification

- gate_ratio=1.0 must match baseline exactly
- blk/q vs recall Pareto curve at different gate_ratios
- waste% should decrease

---

## Idea 4: Query-Batched Page Dedup ★★ (throughput)

### Problem

Thread-per-core with coroutine multiplexing means one core processes many queries. Each query independently walks the graph, but spatially-similar queries visit overlapping pages.

### Design

On each core, maintain a small query queue. Before dispatching, cluster queries by entry-set page neighborhood (cheap: compare entry-set page_ids or SAQ distances). Process spatially-similar queries back-to-back so the AdjacencyPool is warm for the second query.

**Stronger version**: Interleave expansions from two spatially-similar queries. Both need many of the same pages. AdjacencyPool hit rate goes from ~30% (cold) to ~60%+ (warm from sibling query).

### Why Thread-Per-Core Enables This

No lock contention. The query scheduler is local to the core. Reordering without cross-thread coordination.

### Expected Impact

Throughput optimization, not single-query latency. 30-50% QPS improvement under load by amortizing cache across similar queries.

### Risk

Adds scheduling complexity. Reordering introduces fairness concerns (some queries wait longer). Only helps under concurrent load.

### Verification

- QPS improvement at fixed concurrency (4, 8, 16 queries/core)
- p99 must not degrade (fairness)
- Cache hit rate improvement metric

---

## Idea 5: Compressed Adjacency with Surplus Space ★★

### Problem

Current v3 record: 2 + degree×4 bytes. At degree=32, that's 130 bytes. No space for inline data.

### Design

Delta + varint encode sorted neighbor VIDs. Average ~2 bytes per neighbor (vs 4). Record size drops from 130 to ~66 bytes. This creates surplus space options:

**Option A**: Pack 2× more records per page → better co-location → amplifies Idea 1.

**Option B**: Use surplus for inline PQ16 codes (16 bytes per neighbor). Enables score-before-expand gating with a different error profile than SAQ. Useful for Idea 1 (scoring co-located nodes without full SAQ computation).

**Option C**: Use surplus for "neighborhood fingerprint" — 32-byte SimHash of each node's neighborhood. Zero-IO filter for wasted expansions.

### Expected Impact

Depends on which option. Option A compounds with Idea 1 for potentially 40-60% blk/q reduction. Option B enables inline codes without block format expansion.

### Risk

Varint decoding on hot path adds ~50ns per expansion. Negligible vs IO cost.

### Verification

- Records per page metric (before/after compression)
- blk/q at iso-recall

---

## Idea 6: Adaptive IO Depth Per Query ★

### Problem

Easy queries converge in ~50 expansions, hard queries need all 200. All get the same prefetch budget (W=4) and QD allocation.

### Design

Track convergence rate (distance improvement per expansion). Fast-converging queries: reduce prefetch, save QD for other queries. Slow-converging queries: increase prefetch and lookahead aggressively.

**With io_uring**: Dynamically adjust SQE submission count per expansion. Fast queries: 1 SQE. Slow queries: 4-6 SQEs. Global QD budget rebalances in real-time across queries on the same core.

### Expected Impact

Marginal for single-query latency. Helps throughput under multi-query load by redistributing QD from easy to hard queries.

### Risk

Low. Worst case: reverts to current uniform allocation.

### Verification

- QD utilization distribution across query difficulty bins
- p50 and p99 at different concurrency levels

---

## Idea 7: DRAM Navigation Shortcut Layer ★★★ (at scale)

### Problem

First ~20 expansions navigate from entry point to target neighborhood. Low-value expansions (don't contribute to top-k) but each costs 1 page read.

### Design

Maintain a small in-DRAM graph (~1% of nodes, uniformly sampled) with full SAQ distances. Run 10-20 expansions in DRAM (zero IO) to find a better entry point, then switch to the full NVMe graph.

**For 100K vectors**: 1K DRAM nodes, ~32 neighbors each, SAQ codes: 1K × 912B ≈ 1MB.
**For 10M vectors**: 100K DRAM nodes, ~1GB. Still fits.

### Expected Impact

At 100K: ~20 saved IO expansions (~10% of total). Modest.
At 10M+: ~50-100 saved IO expansions. Critical — approach phase grows logarithmically with dataset size.

### Risk

Building the DRAM navigation layer requires careful node selection (must cover the space uniformly). Poor coverage → bad entry points → more NVMe expansions, not fewer.

### Verification

- entry_quality: SAQ distance of best DRAM candidate vs best NVMe entry point
- approach_phase_io: number of expansions before first top-k candidate enters beam

---

## Priority and Sequencing

```
Phase 1: SAQ Gating (Idea 3)                    [easy, do first]
  → Validates whether gating reduces blk/q at all
  → Zero infrastructure cost

Phase 2: Free Expansions (Idea 1)               [the big bet]
  → Requires: inverted page index, modified v3_inner loop
  → Compounds with gating (gated free expansions = even better filtering)
  → If both gating + free expansions work: potentially 40-60% blk/q reduction

Phase 3: Two-Hop Speculative Prefetch (Idea 2)  [cheap, orthogonal]
  → ~5 lines of code
  → Reduces p50 without changing blk/q
  → Works independently of gating/free-expansions

Phase 4: Compressed Adjacency (Idea 5)          [infrastructure]
  → Enables more records per page → amplifies Idea 1
  → Prerequisite for inline codes at scale

Phase 5: DRAM Navigation (Idea 7)               [scale preparation]
  → Only matters at >1M vectors
  → Build when we move beyond Cohere 100K benchmarks
```

Ideas 3, 1, and 2 are complementary and compound. Together they attack blk/q (gating + free expansions) and IO latency (speculative prefetch) simultaneously. This combination is unique to our architecture — no paper proposes all three because no paper has io_uring + thread-per-core + SAQ-in-DRAM.
