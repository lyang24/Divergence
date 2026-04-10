# VeloANN Adoption Plan for Divergence

**Date**: 2026-04-08
**Source**: VeloANN (PVLDB 2026, Zhao et al., arXiv:2602.22805v1)
**Goal**: Adopt VeloANN's proven techniques to improve both latency and throughput.

---

## Current Divergence Baseline (Cohere 100K, 768d, k=100, ef=200)

| Metric | Value |
|--------|-------|
| Recall | 0.963 |
| p50 (cold, 1 core) | 8-9ms |
| QPS (1 core) | ~100 |
| QPS (8 cores) | ~320 (2.95× scaling) |
| mis/q (cold) | ~135 of 201 |
| mis/q (warm, 5% cache) | ~21 of 201 |
| Cache hit rate (cold) | ~33% |
| CPU utilization (B=1) | ~20% (idle during I/O waits) |

---

## Change 1: Co-Resident Record Caching

### What
When we read a 4KB page for VID X, also cache ALL other records on that page in the AdjacencyPool. Currently we discard ~30 co-located records per page read.

### Why
With heavy_edge layout, co-located records are graph neighbors — high probability of future access. Each page holds ~31 records (130 bytes each at degree 32). Caching all of them turns 1 cache miss into ~30 future cache hits.

### VeloANN Reference
Section 3.4: "Upon accessing any record with a non-zero Color tag, all co-tagged records on the page are proactively fetched into the buffer pool." We don't need Color tags — heavy_edge already ensures co-located records are useful graph neighbors.

### Implementation

**File: `crates/engine/src/cache.rs`**

Add a method to AdjacencyPool:

```rust
/// After loading a page for VID X, also insert all other records
/// on this page into the cache. Requires the adj_index to identify
/// which VIDs live on which page and their offsets.
pub fn cache_co_residents(
    &self,
    page_id: u32,
    page_data: &[u8],
    adj_index: &[AdjIndexEntry],
    page_vids: &[u32],  // inverted index: which VIDs are on this page
)
```

**File: `crates/engine/src/search.rs`**

Build a page→VIDs inverted index at startup (once per index load):

```rust
/// Inverted index: page_id → list of VIDs on that page.
/// Built once from adj_index. ~100KB for 100K vectors.
fn build_page_to_vids(adj_index: &[AdjIndexEntry], n: usize) -> Vec<Vec<u32>> {
    let num_pages = adj_index.iter().map(|e| e.page_id).max().unwrap_or(0) as usize + 1;
    let mut page_to_vids = vec![Vec::new(); num_pages];
    for vid in 0..n {
        page_to_vids[adj_index[vid].page_id as usize].push(vid as u32);
    }
    page_to_vids
}
```

After `pool.get_or_load(page_id, io).await` succeeds in the search loop, call `cache_co_residents` to insert all other records from that page.

**Challenge**: AdjacencyPool uses set-associative slots. Inserting ~30 records at once may cause eviction churn. Mitigate by:
- Only insert co-residents that aren't already resident
- Use a lower priority (don't mark as second-chance on first insert)
- Cap co-resident inserts per page read (e.g., max 16)

### Expected Impact
- Cache hit rate: cold ~33% → ~70-80% (each miss warms ~30 neighbors)
- mis/q: ~135 → ~60-80 (fewer physical reads)
- p50: ~8-9ms → ~5-6ms (fewer I/O stalls)

### Verification
- mis/q must decrease ≥30% at iso-recall
- recall must be unchanged (0.963 ± 0.001)
- New perf counters: `co_resident_cached/q`, `co_resident_hit/q`
- Compare with and without co-resident caching at same cache size

---

## Change 2: Cache-Aware Beam Search

### What
When the top candidate is on-disk, look at the next B candidates. If any are in-memory, expand that one instead (zero I/O, instant). Prefetch the on-disk candidates in the background.

### Why
At cold cache, ~67% of expansions hit disk. With B=4 lookahead, probability that at least 1 of 4 is cached ≈ 1 - 0.67⁴ ≈ 80%. Dramatically reduces critical-path stalls.

### VeloANN Reference
Section 4.2, Algorithm 2: Cache-aware Beam Search. The exact algorithm:
1. `v ← top-1 nearest from P \ E`
2. `C ← top-B nearest from P \ E` (look-ahead set)
3. If v is on-disk: iterate C, find first in-memory c → expand c instead; prefetch on-disk candidates in C
4. If no in-memory candidate found: expand v (may stall on I/O)
5. After expansion: add neighbors to P, mark explored in E, trim P to size L

### Implementation

**File: `crates/engine/src/search.rs`**

We ALREADY have proto-cache-aware search via `pop_preferred` with `page_sched_b` (line 847-864). This is exactly VeloANN's Algorithm 2. The current implementation:

```rust
let (candidate, was_preferred) = if page_sched_b > 1 {
    match candidates.pop_preferred(page_sched_b, |vid| {
        pool.is_resident(adj_index[vid as usize].page_id)
    }) {
        Some(pair) => pair,
        None => break,
    }
} else {
    match candidates.pop() {
        Some(c) => (c, false),
        None => break,
    }
};
```

**What needs to change:**

1. **Always enable cache-aware pivoting** (not gated behind `page_sched_b > 1`). Make it the default behavior. The `page_sched_b` parameter becomes the look-ahead window B.

2. **Add prefetch for non-pivoted on-disk candidates.** Currently `pop_preferred` just picks the best in-memory candidate. VeloANN also issues prefetch for the on-disk candidates it skipped during the pivot scan. Add this inside `pop_preferred` or after it returns:

```rust
// After pop_preferred, prefetch the on-disk candidates we looked at
if was_preferred {
    // The original top-1 (and other on-disk candidates in lookahead)
    // should be prefetched since we'll need them soon
    let lookahead = candidates.peek_nearest(&mut buf[..B]);
    for cand in lookahead {
        let pid = adj_index[cand.id.0 as usize].page_id;
        if !pool.is_resident(pid) {
            pool.prefetch_hint(pid);
        }
    }
}
```

3. **`is_resident` check on VID, not just page_id.** With co-resident caching (Change 1), a VID's record may be cached even if the page_id-based check says otherwise (because we cached the record when loading a neighboring VID's page). Need `pool.is_resident_vid(vid)` in addition to page-based check.

**Actually**: our current `pool.is_resident(page_id)` checks the cache by page_id key. Since we cache co-residents under the SAME page_id, this already works. When any record on a page is loaded, the whole page is cached under that page_id. So `is_resident(page_id)` returning true means all records on that page are accessible. No change needed here.

4. **Default B=4.** VeloANN finds B=4 optimal for high-dimensional datasets. Expose as parameter, default to 4.

### Expected Impact
- Stall rate: ~67% (cold) → ~13% (with co-resident caching amplifying cache, plus pivoting)
- p50: further 1-2ms reduction on top of Change 1
- Throughput: fewer stalls = more useful work per unit time

### Verification
- Sweep B in {1, 2, 4, 8}: measure recall, p50, p99, QPS
- B=1 must match baseline exactly
- `page_sched_hits/q` counter already exists — verify it increases with B
- recall must be unchanged (pivoting doesn't lose recall, just changes expansion order)

---

## Change 3: Free Expansions from Co-Located Records

### What
When we read a page for VID X, decode ALL records on that page. For every co-located VID Y that hasn't been visited, compute SAQ distance (free — DRAM, no I/O) and push into the beam if good enough.

### Why
With heavy_edge layout, a 4KB page holds ~31 adjacency records. We pay for 4KB but only use ~130 bytes (1 record). The other ~30 records sit in the cached page, decoded for free. SAQ scoring is ~1μs per vector — 30 extra scores cost ~30μs, trivial vs 50-80μs I/O saved.

### VeloANN Reference
Not directly in VeloANN — this is from our creative_design_ideas.md (Idea 1). But it's the natural next step after co-resident caching: instead of just caching co-located records, actively SCORE them to discover good neighbors without any additional I/O.

### Implementation

**File: `crates/engine/src/search.rs`**

After loading a page in the expansion loop, iterate all records on the page (using the page→VIDs inverted index from Change 1):

```rust
// After expanding VID X's neighbors normally:
// Bonus: score all co-located records on this page
if let Some(page_vids) = page_to_vids.get(page_id as usize) {
    for &co_vid in page_vids {
        let ci = co_vid as usize;
        if ci >= num_vectors || visited[ci] || ci == vid { continue; }
        visited[ci] = true;
        let d = bank.distance(query, ci);  // SAQ distance, DRAM only
        perf.bonus_scored += 1;
        let dominated = nearest.len() >= ef && d >= nearest.furthest().unwrap().distance;
        if !dominated {
            let scored = ScoredId { distance: d, id: VectorId(co_vid) };
            candidates.push(scored);
            nearest.push(scored);
            perf.bonus_pushed += 1;
        }
    }
}
```

### Risk
SAQ scoring ~30 extra vectors per expansion × 201 expansions = ~6000 extra SAQ distances. At ~1μs each = ~6ms CPU. This could eat the I/O savings. Mitigate:
- Only score co-located nodes that are graph neighbors of ANY beam member (not all)
- Or cap bonus scoring to first N expansions where beam is still forming
- Or only score if the page was a cache miss (don't waste CPU on already-explored neighborhoods)

### Expected Impact
- blk/q reduction of 20-40% (more candidates discovered per I/O)
- Beam converges faster with more candidates per I/O
- Risk of CPU overhead eating savings — need to measure

### Verification
- New counters: `bonus_scored/q`, `bonus_pushed/q`, `bonus_useful_pct`
- blk/q must decrease ≥20% at iso-recall
- CPU overhead: measure `dst_ms` increase vs `io_wait_ms` decrease
- Compare with/without free expansions at same ef

---

## Change 4: Multi-Query Coroutine Scheduler

### What
Process B=2-4 queries concurrently on each core via coroutine interleaving. When one query stalls on I/O, switch to another query that has data ready.

### Why
At B=1, CPU utilization is ~20% (idle during I/O waits). With B=4, CPU can work on other queries during stalls. VeloANN's formula: B = ceil(α × I/T) where I=I/O latency (~70μs), T=compute time (~10μs), α=I/O frequency per compute (~0.67 for cold). B ≈ 5 for cold cache.

### VeloANN Reference
Section 3.1: Coroutine-based ANNS Execution. The I/O-Aware scheduler runs a loop per core: pick ready coroutine → resume → submit I/O → poll completions → wake suspended coroutines.

### Implementation

**New file: `crates/engine/src/scheduler.rs`**

```rust
/// Per-core multi-query scheduler.
/// Manages B concurrent search coroutines on a single monoio runtime.
pub struct QueryScheduler {
    batch_size: usize,  // B
    pool: Rc<AdjacencyPool>,
    io: Rc<IoDriver>,
}

impl QueryScheduler {
    /// Process a batch of queries. Returns results in order.
    pub async fn execute_batch(
        &self,
        queries: Vec<QueryRequest>,
    ) -> Vec<QueryResult> {
        // Spawn each query as a monoio task
        let handles: Vec<_> = queries.into_iter().map(|req| {
            let pool = Rc::clone(&self.pool);
            let io = Rc::clone(&self.io);
            monoio::spawn(async move {
                // Existing search function — await points let scheduler switch
                disk_graph_search_pipe_v3(
                    &req.query, &req.entry_set, req.k, req.ef, req.prefetch_window,
                    req.stall_limit, req.drain_budget,
                    &pool, &io, &req.bank, &req.adj_index,
                    &mut req.perf, req.level,
                ).await
            })
        }).collect();

        // Await all — monoio interleaves them at await points
        let mut results = Vec::with_capacity(handles.len());
        for h in handles {
            results.push(h.await);
        }
        results
    }
}
```

**Key considerations:**
- monoio's scheduler already interleaves tasks at `.await` points (I/O waits)
- The AdjacencyPool is `Rc<RefCell>` — shared across queries on same core, no atomics needed (cooperative scheduling, non-preemptive)
- IoDriver is also `Rc` — shared io_uring instance, batched submissions
- Prefetch budget should be W_total/B per query to keep total QD constant
- GlobalIoBudget contention: with B queries on one core, they still share the same global budget. Since they're cooperative (not preemptive), only one query acquires at a time — no real contention

**Difference from our failed B>1 experiment:**
- Previous: spawned independently, uncoordinated, each with W=4 prefetch
- New: coordinated via shared pool+io, prefetch budget = W/B per query, cache-aware pivoting active

### Expected Impact
- QPS per core: ~100 → ~300-400 (3-4× from filling CPU idle time)
- p50 may increase slightly (shared core, context switching overhead)
- p99 should stay bounded (unlike our previous B>1 experiment, because cache-aware pivoting reduces stalls)

### Verification
- Sweep B in {1, 2, 4, 8} at fixed gQD
- B=1 must match baseline QPS and latency exactly
- QPS scaling: expect near-linear up to B=4, diminishing after
- p99 must not explode (critical — this is what killed our previous B>1)
- CPU utilization measurement: expect ~60-80% at B=4

---

## Change 5: Compressed Adjacency Lists

### What
Delta + varint encode sorted neighbor VID lists. Current: 4 bytes per neighbor × 32 = 128 bytes + 2 byte header = 130 bytes. Compressed: ~2 bytes avg per neighbor → 66 bytes per record.

### Why
2× more records per page → amplifies co-resident caching (Change 1) and free expansions (Change 3). More records per page = more useful data per 4KB read.

### VeloANN Reference
Section 3.3: "Adjacency lists are sorted and integer-compressed (e.g., delta encoding or Partitioned Elias-Fano coding) to reduce space consumption."

### Implementation

**File: `crates/storage/src/adjacency.rs`**

Add v4 packed format alongside v3:

```rust
/// v4 record: [degree: u16][delta-varint encoded neighbor VIDs]
/// Neighbors are sorted, then delta-encoded (store differences),
/// then varint-encoded (1-5 bytes per delta, typically 1-2).
fn encode_adj_v4(neighbors: &[u32]) -> Vec<u8> {
    let mut sorted = neighbors.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let mut buf = Vec::with_capacity(2 + sorted.len() * 2);
    buf.extend_from_slice(&(sorted.len() as u16).to_le_bytes());
    let mut prev = 0u32;
    for &vid in &sorted {
        let delta = vid - prev;
        encode_varint(delta, &mut buf);
        prev = vid;
    }
    buf
}
```

**File: `crates/engine/src/search.rs`**

Add v4 record decoder — decode varint deltas to reconstruct neighbor list. Called in the inner loop instead of `page_record_vid`.

### Expected Impact
- Records per page: ~31 → ~62 (2× denser)
- Co-resident caching effectiveness: 2× more records cached per page read
- Free expansions: 2× more candidates per page
- Decode overhead: ~50ns per record (varint decode) — negligible vs I/O cost

### Risk
Low. Varint decoding is well-understood and fast. Backwards compatibility: keep v3 format, add v4 as opt-in.

### Verification
- Records-per-page metric (before vs after)
- blk/q at iso-recall (should improve with Changes 1+3)
- Decode latency microbenchmark

---

## Change 6: Residency Bitset for O(1) Cache Checks

### What
Add a flat bitset to AdjacencyPool for O(1) `is_resident_page(page_id)` checks. Currently uses 8-way set scan.

### Why
Cache-aware beam search (Change 2) checks residency for B candidates per expansion. With B=4 and 201 expansions, that's ~800 checks per query. Current 8-way scan is fast but a bitset is faster and enables larger B without overhead concern.

### Implementation

**File: `crates/engine/src/cache.rs`**

```rust
struct AdjacencyPool {
    // ... existing fields ...
    /// Bitset: 1 bit per page_id. Set on insert, cleared on evict.
    /// Size: num_pages / 8 bytes (~400 bytes for 3200 pages).
    resident_pages: RefCell<Vec<u64>>,
}

pub fn is_resident_fast(&self, page_id: u32) -> bool {
    let bits = self.resident_pages.borrow();
    let word = page_id as usize / 64;
    let bit = page_id as usize % 64;
    word < bits.len() && (bits[word] & (1u64 << bit)) != 0
}
```

Update on every insert/evict to keep bitset in sync.

### Expected Impact
- Negligible latency improvement (current scan is already fast)
- Enables larger B values without concern
- Cleaner API for cache-aware search

---

## Implementation Order

```
Phase 1: Co-Resident Caching + Cache-Aware Search       [Changes 1+2]
  ├─ Build page→VIDs inverted index
  ├─ Modify AdjacencyPool: cache_co_residents on page load
  ├─ Enable page_sched_b=4 by default in search
  ├─ Add prefetch for skipped on-disk candidates during pivot
  └─ Experiment: sweep B={1,2,4,8}, measure recall/p50/mis_q

Phase 2: Free Expansions                                [Change 3]
  ├─ Score co-located VIDs with SAQ on each page load
  ├─ Push non-dominated into beam
  ├─ Add bonus_scored/bonus_pushed perf counters
  └─ Experiment: measure blk/q reduction, CPU overhead

Phase 3: Multi-Query Scheduler                          [Change 4]
  ├─ QueryScheduler struct with batch execution
  ├─ Coordinated prefetch budget (W_total / B)
  ├─ Throughput benchmark: sweep B={1,2,4} at concurrency
  └─ Verify p99 does not explode

Phase 4: Compressed Adjacency                           [Change 5]
  ├─ v4 format: delta+varint encoding
  ├─ Decoder in search loop
  ├─ Re-run Phase 1+2 experiments with v4 layout
  └─ Measure records-per-page improvement

Phase 5: Residency Bitset                               [Change 6]
  ├─ Only if profiling shows is_resident as bottleneck
  └─ Low priority — current 8-way scan is fast enough
```

**Phase 1 is the priority for tomorrow.** It's the highest-ROI change: two complementary techniques (co-resident caching + cache-aware pivoting) that compound with heavy_edge layout. Pure search-side changes, no architectural overhaul.

---

## Experiment Design for Phase 1

### Test: `exp_veloann_cache_aware`

**Setup**: Cohere 100K, dim=768, k=100, ef=200, W=4, 5% cache, heavy_edge layout, cold per-query.

**Configurations to test:**

| Config | Co-Resident Cache | page_sched_b | Description |
|--------|-------------------|--------------|-------------|
| baseline | OFF | 1 | Current behavior |
| coresid_only | ON | 1 | Co-resident caching, no pivoting |
| pivot_only | OFF | 4 | Cache-aware pivoting, no co-resident |
| both_b2 | ON | 2 | Combined, B=2 |
| both_b4 | ON | 4 | Combined, B=4 |
| both_b8 | ON | 8 | Combined, B=8 |

**Metrics per config:**

| Metric | Source |
|--------|--------|
| recall | standard |
| p50, p99 | wall-clock |
| QPS | wall-clock |
| exp/q | perf.expansions |
| blk/q | perf.blocks_read |
| mis/q | perf.blocks_miss |
| hit/q | perf.blocks_hit |
| phy/q | perf.phys_reads |
| page_sched_hits/q | perf.page_sched_hits |
| co_resident_cached/q | NEW counter |
| co_resident_hit/q | NEW counter |

**Pass criteria:**
- recall unchanged (0.963 ± 0.001) across all configs
- `both_b4` mis/q ≤ 70% of baseline mis/q
- `both_b4` p50 ≤ 80% of baseline p50
- No config should have p99 > 2× baseline p99

---

## Files Changed Summary

| File | Changes | ~LOC |
|------|---------|------|
| `crates/engine/src/cache.rs` | `cache_co_residents()`, co-resident counters, optional residency bitset | ~60 |
| `crates/engine/src/search.rs` | Enable page_sched_b default=4, add prefetch-on-pivot, page_to_vids inverted index, free expansions loop | ~80 |
| `crates/engine/src/perf.rs` | New counters: co_resident_cached, co_resident_hit, bonus_scored, bonus_pushed | ~8 |
| `crates/engine/tests/disk_search.rs` | `exp_veloann_cache_aware` test | ~200 |
| `crates/engine/src/scheduler.rs` | QueryScheduler (Phase 3 only) | ~100 |
| `crates/storage/src/adjacency.rs` | v4 compressed format (Phase 4 only) | ~80 |

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Co-resident caching causes eviction churn | Higher eviction rate, lower effective cache | Cap co-resident inserts per page, lower priority for co-residents |
| Cache-aware pivoting degrades recall | Expanding sub-optimal candidates | Paper shows recall unchanged; verify experimentally |
| Free expansions CPU overhead eats IO savings | Net slower despite fewer IOs | Cap bonus scoring, only on cache-miss pages |
| Multi-query scheduler increases p99 | Latency regression at B>1 | Start with B=2, compare to VeloANN formula B=ceil(α×I/T) |
| Compressed adjacency decode overhead | Slower inner loop | Benchmark: expect ~50ns decode vs ~50-80μs I/O |

---

## Dead Ends (Do NOT Revisit)

- **VeloANN's distance-based affinity co-placement**: heavy_edge already beats it on our data (mis/q 20.3 vs 21.6)
- **GraphAGO activity-neighborhood ordering**: tested, loses to heavy_edge (upg/q 131 vs 104)
- **Pipelined refine**: net slower under constant QD (adj_pf 4→2 costs more than overlap saves)
- **SAQ gating**: blk/q unchanged (201.1) — gated-out neighbors were never expanded anyway
- **TWPP trace-based co-placement**: query-specific, doesn't generalize
- **Slotted page layout**: low priority — our degrees are near-uniform, fixed-record packing works
