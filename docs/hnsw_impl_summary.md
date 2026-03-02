# HNSW Implementation Summary

This document captures every decision for Divergence's HNSW implementation:
the algorithms, data structures, memory layouts, concurrency model, and
NVMe-aware design choices. It draws from the original HNSW paper, VeloANN,
OctopusANN, and battle-tested code from qdrant, USearch, and infinity.

---

## 1. Graph Layout (In-Memory)

### Level 0: Contiguous fixed-size array

Level 0 contains every vertex. Store it in a single contiguous allocation
for cache-friendly sequential access and O(1) lookup by VectorId.

```
graph_l0: Vec<u8>    // capacity = num_vectors * L0_SLOT_SIZE

L0 slot layout (per vertex):
┌──────────────────────────────────┐
│ level: u8                        │  which layers this vertex appears in
│ neighbor_count: u16              │  current number of L0 neighbors
│ _pad: u8                         │  alignment
│ neighbors: [u32; Mmax0]          │  fixed-capacity neighbor array
└──────────────────────────────────┘
L0_SLOT_SIZE = 4 + Mmax0 * 4 bytes
```

Offset for vertex `v`: `v.0 as usize * L0_SLOT_SIZE`.

Why fixed-size slots with Mmax0 capacity:
- No per-vertex allocation. One big `Vec<u8>` allocated once.
- O(1) random access. No indirection, no pointer chasing.
- Matches infinity's `VertexL0` pattern, proven in production.
- Mmax0 = 2*M. With M=32, Mmax0=64, L0_SLOT_SIZE = 260 bytes.
  At 100M vectors = ~24 GB. This is a **construction-only** cost.
  The builder requires a machine with sufficient DRAM to hold the full L0
  array during construction. After build, the reader loads L0 from disk
  on demand via the buffer pool — only the working set resides in DRAM.
  For targets that cannot afford 24 GB during construction, reduce M or
  build in batches with graph merging (post-MVP).

### Higher levels: Flattened offset table + packed neighbor array (reader only)

Vertices above level 0 are rare (~1/M fraction per level). The **reader** (HnswIndex)
stores all upper-level neighbor data in a single flat `Vec<u32>`, indexed by an
offset table. The **builder** (HnswBuilder) uses `Vec<RwLock<Option<Vec<Vec<u32>>>>>`
during construction because neighbor lists need per-node mutable access under locks.
On `build()`, the builder flattens this into the packed format below.

```
upper_offsets: Vec<u32>   // [vid] -> start offset into upper_data, or u32::MAX if level 0 only
upper_data: Vec<u32>      // packed: [level_count, L1_count, L1_neighbors..., L2_count, L2_neighbors..., ...]
```

Layout for a vertex with levels 1..=L:
```
upper_data[offset..]:
  level_count: u32           // L (number of upper levels)
  L1_count: u32              // number of L1 neighbors
  L1_neighbors: [u32; L1_count]
  L2_count: u32
  L2_neighbors: [u32; L2_count]
  ...
```

Why flattened instead of `Vec<Option<Vec<Vec<u32>>>>`:
- Triple indirection (3 pointer chases per access) destroys cache locality.
- Each inner `Vec` is a heap allocation — at 100M vertices with 3% having upper
  levels, that's 3M allocations with 72 bytes overhead each = 216 MB wasted.
- Flat layout: one allocation for offsets, one for data. Zero per-vertex heap alloc.
- Random access: `upper_offsets[vid]` → `&upper_data[offset..]`. One indirection.

Why not contiguous fixed-size for upper levels:
- Upper levels hold <1% of total neighbor data.
- Contiguous would waste `N * num_levels * Mmax * 4` bytes for mostly-empty slots.
- Sequential access on upper levels is not performance-critical (greedy ef=1 search).

### Vector storage during construction

```
vectors: Vec<f32>    // flat: vectors[vid * dim .. (vid+1) * dim]
```

Contiguous flat array. Accessed by `vid * dim` offset. No indirection.

---

## 2. Core Data Structures

### 2.1 VisitedPool (improved from qdrant)

Pool of reusable visited-set handles with generation counting. Each handle
is a bitset, not a byte-per-element array. Thread borrows a handle, uses it
for one search/insert, returns it on drop.

**Why bitset instead of qdrant's u8-per-element scheme:**
- u8 per element = 1 byte per VectorId. At 100M vectors = 100 MB per handle.
  With 16 threads each holding a handle during construction = 1.6 GB just for
  visited tracking. Unacceptable.
- u8 wraps at 255. When it wraps, you must `memset` the entire 100 MB array
  to zero. This is a tail-latency spike: ~25 μs on modern CPUs, and it happens
  unpredictably every 255 searches.
- Bitset = 1 bit per VectorId. At 100M vectors = 12.5 MB per handle.
  16 handles = 200 MB. 8x reduction.

**Design: bitset + u64 generation tag:**

```rust
struct VisitedList {
    generation: u64,
    words: Vec<u64>,       // capacity = ceil(num_vectors / 64)
    tags: Vec<u64>,        // one generation tag per 64-bit word
}

struct VisitedListHandle<'a> {
    pool: &'a VisitedPool,
    inner: VisitedList,
}

impl VisitedListHandle<'_> {
    fn check_and_mark(&mut self, id: VectorId) -> bool {
        let idx = id.0 as usize;
        let word_idx = idx / 64;
        let bit_idx = idx % 64;
        let mask = 1u64 << bit_idx;

        // If this word's tag doesn't match current generation, it's stale — clear it
        if self.inner.tags[word_idx] != self.inner.generation {
            self.inner.tags[word_idx] = self.inner.generation;
            self.inner.words[word_idx] = 0;
        }

        let was_visited = (self.inner.words[word_idx] & mask) != 0;
        self.inner.words[word_idx] |= mask;
        was_visited
    }

    fn next_iteration(&mut self) {
        // Just bump generation. Stale words are lazily cleared on access.
        // No bulk memset ever needed.
        self.inner.generation += 1;
    }
}

struct VisitedPool {
    pool: Mutex<Vec<VisitedList>>,    // parking_lot::Mutex
}
```

**Key properties:**
- `next_iteration()` is O(1). No memset. No tail-latency spike.
- Each word is lazily cleared on first access in a new generation.
  During a typical search (ef=128, touching ~1000 vertices), only ~16 words
  are actually accessed and cleared — not the full array.
- u64 generation never wraps in practice (2^64 searches ≈ heat death of universe).
- Memory: `(N/64 * 8) + (N/64 * 8)` = `N/4` bytes per handle.
  At 100M vectors = 25 MB per handle (words + tags). 16 handles = 400 MB.
  Still 4x better than u8 scheme. Can optimize further by sharing a single
  tags array per handle if memory is tight.

### 2.2 FixedCapacityHeap

Two heaps used in HNSW search:
- **candidates**: min-heap (pop nearest). Drives the frontier expansion.
- **nearest**: max-heap with fixed capacity (pop furthest when full). Tracks results.

```rust
/// Max-heap that evicts the worst element when full.
/// Used for tracking the result set (W in the paper).
struct FixedCapacityHeap {
    data: Vec<ScoredId>,    // pre-allocated to capacity
    capacity: usize,
}

impl FixedCapacityHeap {
    fn push(&mut self, item: ScoredId) -> Option<ScoredId>;  // returns evicted if full
    fn furthest(&self) -> Option<&ScoredId>;                  // peek worst
    fn into_sorted(self) -> Vec<ScoredId>;                    // drain sorted by distance
}

/// Entry in both heaps.
#[derive(Clone, Copy)]
struct ScoredId {
    distance: f32,
    id: VectorId,
}
```

The candidates heap is a standard `BinaryHeap` (or custom min-heap).
The nearest heap wraps `BinaryHeap<Reverse<ScoredId>>` with a size limit.

Pre-allocated to `ef` or `ef_construction` at the start of each search.
Cleared, not reallocated, between searches.

### 2.3 SearchContext

Bundles all per-search temporary state. Allocated once per thread,
reused across searches.

```rust
struct SearchContext {
    nearest: FixedCapacityHeap,           // result set (W)
    candidates: BinaryHeap<ScoredId>,     // frontier (C), min-ordered
    neighbor_buf: Vec<VectorId>,          // scratch for decoded neighbors
}

impl SearchContext {
    fn clear(&mut self, ef: usize) {
        self.nearest.clear_with_capacity(ef);
        self.candidates.clear();
        self.neighbor_buf.clear();
    }
}
```

Modeled after USearch's `context_t`. One per worker thread, never
crosses thread boundaries, no synchronization needed.

---

## 3. Algorithms

### 3.1 Level Assignment

```rust
fn random_level(ml: f64, rng: &mut impl Rng) -> usize {
    let r: f64 = rng.gen::<f64>();  // uniform in (0, 1)
    (-r.ln() * ml).floor() as usize
}
```

Where `ml = 1.0 / (M as f64).ln()`.

At M=32: ml ≈ 0.289. Average level ≈ 0.289. ~97% of vertices are level 0 only.

### 3.2 Search Layer (HNSW Paper Algorithm 2)

```rust
fn search_layer(
    &self,
    query: &[f32],
    entry_points: &[VectorId],
    ef: usize,
    level: usize,
    visited: &mut VisitedListHandle,
    ctx: &mut SearchContext,
    distance: &dyn DistanceComputer,
) {
    ctx.clear(ef);

    for &ep in entry_points {
        let d = distance.distance(query, self.get_vector(ep));
        ctx.nearest.push(ScoredId { distance: d, id: ep });
        ctx.candidates.push(ScoredId { distance: d, id: ep });  // min-heap: negate or use Reverse
        visited.check_and_mark(ep);
    }

    while let Some(candidate) = ctx.candidates.pop() {  // nearest unprocessed
        if let Some(furthest) = ctx.nearest.furthest() {
            if candidate.distance > furthest.distance {
                break;  // all candidates are worse than worst result
            }
        }

        let neighbors = self.get_neighbors(candidate.id, level);
        for &nbr in neighbors {
            if visited.check_and_mark(nbr) {
                continue;
            }
            let d = distance.distance(query, self.get_vector(nbr));
            let dominated = ctx.nearest.len() >= ef
                && d >= ctx.nearest.furthest().unwrap().distance;
            if !dominated {
                ctx.candidates.push(ScoredId { distance: d, id: nbr });
                ctx.nearest.push(ScoredId { distance: d, id: nbr });
            }
        }
    }
}
```

### 3.3 Greedy Search (Upper Layers)

For layers above the insertion level, ef=1. Just find the single nearest:

```rust
fn search_entry(
    &self,
    query: &[f32],
    mut current: VectorId,
    level: usize,
    distance: &dyn DistanceComputer,
) -> VectorId {
    let mut cur_dist = distance.distance(query, self.get_vector(current));
    loop {
        let mut changed = false;
        for &nbr in self.get_neighbors(current, level) {
            let d = distance.distance(query, self.get_vector(nbr));
            if d < cur_dist {
                cur_dist = d;
                current = nbr;
                changed = true;
            }
        }
        if !changed { break; }
    }
    current
}
```

### 3.4 Heuristic Neighbor Selection (HNSW Paper Algorithm 4)

The key differentiator from simple nearest-M selection. Ensures diverse
connections across cluster boundaries.

```rust
fn select_neighbors_heuristic(
    &self,
    target: VectorId,
    candidates: &mut Vec<ScoredId>,  // sorted by distance ascending
    m: usize,
    distance: &dyn DistanceComputer,
) -> Vec<VectorId> {
    let mut selected: Vec<VectorId> = Vec::with_capacity(m);

    for &ScoredId { id: candidate, distance: cand_dist } in candidates.iter() {
        if selected.len() >= m {
            break;
        }
        // Check: is candidate closer to target than to any already-selected neighbor?
        let mut good = true;
        for &existing in &selected {
            let dist_to_existing = distance.distance(
                self.get_vector(candidate),
                self.get_vector(existing),
            );
            if dist_to_existing < cand_dist {
                good = false;
                break;
            }
        }
        if good {
            selected.push(candidate);
        }
    }
    selected
}
```

No `extendCandidates` (default false, only for extreme clustering).
`keepPrunedConnections` optionally fills remaining slots from discarded
candidates if |selected| < m. Start without it — add only if recall
on real data shows sparse neighborhoods.

### 3.5 Insertion (HNSW Paper Algorithm 1)

```rust
fn insert(&mut self, id: VectorId, vector: &[f32]) {
    let level = random_level(self.ml, &mut self.rng);
    self.store_vector(id, vector);
    self.set_level(id, level);

    if self.num_vectors() == 0 {
        self.entry_point = id;
        self.max_level = level;
        return;
    }

    let mut ep = self.entry_point;

    // Phase 1: greedy descent from top to level+1
    for lc in (level + 1..=self.max_level).rev() {
        ep = self.search_entry(vector, ep, lc, &self.distance);
    }

    // Phase 2: beam search + connect at levels min(level, max_level)..0
    let mut visited = self.visited_pool.get(self.num_vectors());
    let mut ctx = self.get_search_context();

    for lc in (0..=level.min(self.max_level)).rev() {
        self.search_layer(vector, &[ep], self.ef_construction, lc, &mut visited, &mut ctx, &self.distance);

        // select neighbors via heuristic
        let mut scored: Vec<ScoredId> = ctx.nearest.drain_sorted();
        let neighbors = self.select_neighbors_heuristic(id, &mut scored, self.m_for_level(lc), &self.distance);

        // connect id -> neighbors
        self.set_neighbors(id, lc, &neighbors);

        // connect neighbors -> id (bidirectional), shrink if over Mmax
        for &nbr in &neighbors {
            let mut nbr_neighbors = self.get_neighbors_mut(nbr, lc);
            nbr_neighbors.push(id);
            let mmax = self.mmax_for_level(lc);
            if nbr_neighbors.len() > mmax {
                // re-prune nbr's connections
                let mut candidates: Vec<ScoredId> = nbr_neighbors.iter()
                    .map(|&n| ScoredId {
                        id: n,
                        distance: self.distance.distance(self.get_vector(nbr), self.get_vector(n)),
                    })
                    .collect();
                candidates.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
                let pruned = self.select_neighbors_heuristic(nbr, &mut candidates, mmax, &self.distance);
                self.set_neighbors(nbr, lc, &pruned);
            }
        }

        // entry point for next layer = nearest found
        ep = scored[0].id;
        visited.next_iteration();
    }

    // update global entry point if new node is higher
    if level > self.max_level {
        self.entry_point = id;
        self.max_level = level;
    }
}
```

### 3.6 KNN Search (HNSW Paper Algorithm 5)

```rust
fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<ScoredId> {
    assert!(ef >= k);
    let mut ep = self.entry_point;

    // greedy descent on upper layers
    for lc in (1..=self.max_level).rev() {
        ep = self.search_entry(query, ep, lc, &self.distance);
    }

    // full beam search on layer 0
    let mut visited = self.visited_pool.get(self.num_vectors());
    let mut ctx = self.get_search_context();
    self.search_layer(query, &[ep], ef, 0, &mut visited, &mut ctx, &self.distance);

    let mut results = ctx.nearest.into_sorted();
    results.truncate(k);
    results
}
```

---

## 4. Concurrency (Construction)

### Model: insert-parallel, search-after-build

For MVP, construction is parallel (multiple threads inserting), search
is only available after construction completes. Online insertion (search
during build) is a post-MVP concern.

### Synchronization (from qdrant + USearch)

1. **Per-node RwLock for neighbor lists.** During insertion of vertex q,
   we read neighbors of the current candidate (read lock) and write to
   the neighbor lists of q and its selected neighbors (write lock).
   Use `parking_lot::RwLock` — faster than std.

2. **Global lock for entry point + max level.** Only contended when a
   new vertex has a higher level than the current max (rare, ~1/N).
   Use `parking_lot::Mutex`.

3. **VisitedPool provides per-thread isolation.** Each thread borrows
   its own VisitedList handle. No contention.

4. **Level 0 array is pre-allocated.** All slots exist before any
   insertions. Only the neighbor data within each slot is mutated.
   No reallocation races.

```rust
struct HnswBuilder {
    // read-only after init
    config: HnswConfig,
    dimension: usize,
    vectors: Vec<f32>,         // flat, pre-allocated

    // per-node mutable state (construction uses Vec<Vec> per node, flattened on freeze)
    links_l0: Vec<RwLock<L0Neighbors>>,      // one per vertex, fixed-size inline array
    links_upper: Vec<RwLock<Option<Vec<Vec<VectorId>>>>>,  // only Some for level >= 1

    // global mutable state
    entry_point: Mutex<(VectorId, usize)>,   // (id, level)

    // thread-local resources
    visited_pool: VisitedPool,

    // immutable config
    distance: Box<dyn DistanceComputer>,
}
```

### Thread-local context

Each construction thread owns a `SearchContext`. No sharing. Allocated once
at thread spawn, reused for every insertion on that thread.

---

## 5. Builder → Reader Transition

After construction, convert the mutable builder into an immutable reader.
Strip all locks, compact the data.

```rust
struct HnswIndex {
    config: HnswConfig,
    dimension: usize,
    vectors: Vec<f32>,           // flat
    graph_l0: Vec<u8>,           // contiguous L0 slots, no locks
    upper_offsets: Vec<u32>,     // [vid] -> offset into upper_data, u32::MAX if level 0 only
    upper_data: Vec<u32>,        // packed: [level_count, L1_count, L1_neighbors..., ...]
    entry_point: VectorId,
    max_level: usize,
    visited_pool: VisitedPool,
    distance: Box<dyn DistanceComputer>,
}
```

`HnswIndex` implements `GraphStore`. Lock-free reads, safe for concurrent
search from multiple threads (each with own `VisitedListHandle`).

---

## 6. NVMe-Aware Design Hooks

The in-memory HNSW is Phase 2. But the design accounts for Phase 3+
(disk storage, buffer pool, async IO) from day one:

### 6.1 GraphStore trait is IO-agnostic

```rust
pub trait GraphStore: Send + Sync {
    fn neighbors(&self, id: VectorId, level: usize) -> &[VectorId];
    fn vector(&self, id: VectorId) -> &[f32];
    fn entry_point(&self) -> VectorId;
    fn max_level(&self) -> usize;
    fn num_vectors(&self) -> usize;
}
```

In-memory `HnswIndex` implements this with direct array access.

**Disk-resident search does NOT use this trait.** The disk search path
(`cache_aware_beam_search` in `engine/search.rs`) is a separate async
function that reads from the buffer pool and suspends on cache miss.
It is not parameterized over `GraphStore` — it's a different code path
entirely. Two implementations, not one abstracted interface. The sync
`GraphStore` trait is for in-memory construction and in-memory search only.

### 6.2 Serialization format matches disk layout

When `HnswIndex` serializes to disk, it writes:

- `adjacency.dat`: Level 0 neighbor lists in fixed-size 4KB blocks.
  Block `vid` starts at offset `vid * 4096`. One NVMe page per vertex.
  Contains: `[vid: u32, num_neighbors: u16, pad: u16, neighbors: [u32; Mmax0]]`.
  Padded to 4096 bytes. Direct 1:1 mapping — no record_map needed.

- `vectors.dat`: Exact vectors, contiguous `[f32; dim]` per vertex.
  Offset: `vid * dim * 4`. Read during refinement via O_DIRECT.

- `pq_codes.dat`: PQ-compressed codes, contiguous. Loaded into DRAM at
  startup. Not on disk access path during search.
  **DRAM constraint:** code_size = num_subspaces * ceil(bits_per_code / 8).
  With 8 subspaces, 8 bits/code: 8 bytes/vector. At 100M vectors = 800 MB.
  With 32 subspaces: 32 bytes/vector = 3.2 GB. Must budget this against the
  target DRAM envelope. If PQ codes exceed budget, fall back to on-disk PQ
  with buffer pool caching (loses the "all PQ in DRAM" fast path).

- `nav_graph.dat`: Small in-memory navigation graph (MemGraph). 0.1% of
  vertices, built as separate small HNSW. Loaded at startup.

- `meta.json`: Index metadata.

### 6.3 Search is structured for coroutine suspension

The search loop's hot path — `get_neighbors(vid)` — is the exact point
where a disk-resident implementation will suspend. VeloANN's cache-aware
beam search modifies the standard HNSW search loop:

1. Pop nearest candidate `v` from candidates heap.
2. If `v` is not in the buffer pool (cache miss):
   - Scan the look-ahead set (next W candidates).
   - If any are cached, pivot to the cached one instead.
   - Fire async prefetch for the on-disk candidates.
3. Load the adjacency block (may suspend the coroutine on IO).
4. Expand neighbors as normal.

This is not implemented in Phase 2 but the `search_layer` function is
structured so that replacing `self.get_neighbors(vid)` with
`self.adj_pool.get_or_load(vid, io).await` is a mechanical change.

### 6.4 Level 0 block layout = NVMe page

The L0 slot at construction uses Mmax0*4 + 4 bytes. The on-disk block is
4096 bytes. This means:

- M=32 → Mmax0=64 → neighbor data = 256 bytes + 8 byte header = 264 bytes.
  Fits in 4KB with 3832 bytes to spare (available for graph-replicated
  neighbor adjacency in Opt-B).

- M=64 → Mmax0=128 → neighbor data = 512 bytes + 8 byte header = 520 bytes.
  Still fits in 4KB with 3576 bytes spare.

The construction L0 slot is smaller than 4KB (no padding). On serialization,
we expand each slot to a full 4KB block aligned for O_DIRECT.

**Disk space tradeoff (explicit MVP decision):**
1:1 VID→page mapping wastes ~3.5 KB per vertex (264 bytes used out of 4096).
At 100M vectors = 400 GB on disk. This is intentional for Phase 1:
- Eliminates the record_map indirection (VID is the page offset).
- Guarantees one IO per adjacency fetch — no cross-page records.
- NVMe SSDs are cheap (1 TB = ~$50). Disk space is not the bottleneck.
- The spare space is not truly wasted if graph-replicated blocks (Opt-B)
  pack neighbor adjacency lists into it, amortizing 3-4 IO hops into 1.

Phase 2 optimization: slotted pages with multiple records per page (VeloANN
layout) for datasets where 400 GB is unacceptable. This requires a
record_map but improves IO amplification.

---

## 7. Parameters

### Defaults

| Parameter | Value | Notes |
|-----------|-------|-------|
| M | 32 | Connectivity per layer. Good balance for 128-768d. |
| Mmax0 | 64 | Level 0 capacity = 2*M |
| Mmax | 32 | Upper level capacity = M |
| ef_construction | 200 | Construction beam width |
| ef_search | 64 | Default search beam width (tunable per query) |
| ml | 1/ln(M) ≈ 0.289 | Level multiplier |
| Neighbor selection | Heuristic (Alg 4) | Always. No simple selection. |

### Tuning knobs exposed

- `ef_search`: per-query, controls recall/speed tradeoff.
- `M`: set at index creation, immutable after.
- `ef_construction`: set at index creation.

---

## 8. Verification

### Correctness
- Bidirectional links: after insertion, for every edge (a, b) at level L,
  both a's and b's neighbor list at level L contain the other.
- Neighbor count: no vertex exceeds Mmax0 (level 0) or Mmax (upper levels).
- Entry point: always the vertex with the highest level.
- Level distribution: empirically matches geometric distribution.

### Recall
- Sift1M (128d, L2): recall@10 >= 95% at ef=128, >= 99% at ef=256.
- GloVe-200 (200d, cosine): recall@10 >= 90% at ef=128.

### Performance baselines
- Construction: measure inserts/second, compare against hnswlib.
- Search: measure QPS at target recall, compare against hnswlib.
- Memory: measure bytes/vector, compare against theoretical minimum.
  Theoretical: Mmax0*4 + avg_upper_layers*Mmax*4 + dim*4 per vector.

---

## 9. File Map

```
crates/core/src/
├── types.rs         VectorId, MetricType (done)
├── distance.rs      DistanceComputer trait + L2/Cosine/IP (done)
├── quantization.rs  Quantizer trait (done, PQ impl later)
└── encoding.rs      AdjacencyList types (done)

crates/index/src/
├── graph.rs         GraphStore + GraphBuilder traits (done)
├── hnsw.rs          HnswBuilder, HnswIndex, insert, search
├── visited.rs       VisitedPool, VisitedList, VisitedListHandle
├── heap.rs          FixedCapacityHeap, ScoredId
└── search.rs        SearchContext, search_layer, search_entry
```
