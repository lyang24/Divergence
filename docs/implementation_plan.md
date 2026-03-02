# Project Divergence — Rust Implementation Plan

## Context

Project Divergence is an NVMe-native single-node vector search engine. The goal is to
fully utilize modern NVMe SSDs and CPU cores without mmap or OS page cache, achieving
predictable tail latency under memory constraints. Research from 18 papers validates
the core architecture: thread-per-core async execution, record-level buffer pool,
two-stage scoring, and object-based tiered storage.

This plan covers the full implementation from empty repo to working search engine,
organized in 7 phases. Benchmarking uses VectorDBBench. Async runtime uses monoio.
Starting index is HNSW behind a `GraphStore` trait; disk-friendly bounded-degree
graph (Vamana-style) planned as second backend.

### Design Principles

1. **Zero mmap on query path.** mmap is permitted only during offline index
   construction. All hot-path memory is explicitly managed.
2. **MVP first, optimize second.** Simple flat-file layouts before slotted pages.
   Graph-replicated blocks and affinity co-placement are post-baseline optimizations.
3. **Allocation-free queries.** Per-query arena allocator, bitset-based visited
   tracking, fixed-capacity heaps. No `BinaryHeap` / `HashSet` on the hot path.
4. **Separate IO budgets.** Adjacency reads and refine reads get independent
   in-flight quotas to prevent refinement from starving traversal.

---

## Phase 1: Project Skeleton & Core Primitives

**Goal:** Cargo workspace, core types, distance functions, quantization.

### 1.1 Cargo workspace structure

```
divergence/
├── Cargo.toml              (workspace root)
├── crates/
│   ├── core/               (shared types, distance, quantization)
│   ├── storage/            (on-disk layout, page management, IO)
│   ├── buffer/             (record-level buffer pool)
│   ├── index/              (graph construction, graph store trait)
│   ├── engine/             (async runtime, query pipeline, scheduler)
│   └── server/             (API layer — later phase)
├── docs/
│   ├── paper_insights.md   (research synthesis)
│   └── implementation_plan.md (this file)
├── benches/                (criterion micro-benchmarks)
├── tests/                  (integration tests)
└── tools/                  (dataset loaders, index builders)
```

### 1.2 Core types (`crates/core/`)

```rust
// Identifiers
pub struct VectorId(pub u32);        // 4 bytes, supports up to 4B vectors
pub struct BlockId(pub u32);         // adjacency block identifier

// Distance
pub enum MetricType { L2, Cosine, InnerProduct }

pub trait DistanceComputer: Send + Sync {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32;
    fn distance_batch(&self, query: &[f32], vectors: &[&[f32]], results: &mut [f32]);
}

// SIMD-accelerated implementations for L2, Cosine, IP
// Use std::simd (nightly) or pulp crate for portable SIMD
```

### 1.3 Vector quantization (`crates/core/quantization/`)

Start with Product Quantization (PQ) — widely used, well-understood:

```rust
pub trait Quantizer: Send + Sync {
    type Code: AsRef<[u8]>;
    fn train(&mut self, vectors: &[&[f32]]);
    fn encode(&self, vector: &[f32]) -> Self::Code;
    fn distance(&self, query: &[f32], code: &Self::Code) -> f32;
    fn distance_batch(&self, query: &[f32], codes: &[Self::Code], results: &mut [f32]);
}

pub struct ProductQuantizer {
    num_subspaces: usize,       // M (e.g., 8, 16, 32)
    bits_per_code: usize,       // typically 8 (256 centroids per subspace)
    codebooks: Vec<Vec<f32>>,   // M codebooks, each with 256 centroids
    dimension: usize,
}
```

Future: add RaBitQ/ExtRaBitQ as advanced quantizer.

### 1.4 Adjacency list encoding (`crates/core/encoding/`)

```rust
pub struct AdjacencyList {
    neighbors: Vec<VectorId>,   // in-memory: plain sorted vec
}

pub struct CompressedAdjList {
    data: Vec<u8>,              // on-disk: delta-encoded + varint
}

// Delta encoding for sorted neighbor IDs
// Elias-Fano or simple varint for compact representation
// Decode is fast — sequential scan with running sum
```

### 1.5 Per-query arena and allocation-free data structures

```rust
/// Fixed-capacity min-heap for candidate tracking. No dynamic allocation.
pub struct BoundedHeap<T: Ord> {
    data: Vec<T>,       // pre-allocated to max capacity
    len: usize,
    capacity: usize,
}

/// Bitset-based visited tracking. VectorId must be dense/contiguous.
pub struct VisitedSet {
    bits: Vec<u64>,     // ceil(num_vectors / 64) words
    generation: u64,    // bump generation to "clear" without zeroing
    tags: Vec<u64>,     // per-word generation tag
}

/// Per-query scratch space, reused across queries on the same thread.
pub struct QueryArena {
    candidates: BoundedHeap<Candidate>,
    visited: VisitedSet,
    neighbor_buf: Vec<VectorId>,  // reusable buffer for decoded neighbors
}
```

The `VisitedSet` uses generation counting: instead of zeroing the bitset between
queries, bump a generation counter. A bit is "set" only if its tag matches the
current generation. This gives O(1) clear.

---

## Phase 2: Graph Index Construction

**Goal:** Build a graph index in memory behind a generic `GraphStore` trait.
HNSW is the first backend; Vamana planned as second.

### 2.1 GraphStore trait (`crates/index/`)

```rust
/// Abstract graph storage — decoupled from specific index algorithm.
pub trait GraphStore: Send + Sync {
    fn neighbors(&self, id: VectorId, level: usize) -> &[VectorId];
    fn vector(&self, id: VectorId) -> &[f32];
    fn entry_point(&self) -> VectorId;
    fn max_level(&self) -> usize;
    fn num_vectors(&self) -> usize;
    fn max_degree(&self, level: usize) -> usize;
}

/// Abstract builder — different algorithms implement this.
pub trait GraphBuilder: Send + Sync {
    fn add(&mut self, id: VectorId, vector: &[f32]);
    fn build(self) -> Box<dyn GraphStore>;
}
```

### 2.2 HNSW builder (`crates/index/hnsw.rs`)

```rust
pub struct HnswBuilder {
    max_connections: usize,     // M (e.g., 16, 32, 64)
    max_connections_0: usize,   // M0 = 2*M for level 0
    ef_construction: usize,     // construction beam width (e.g., 200)
    ml: f64,                    // level multiplier = 1/ln(M)
    dimension: usize,
    metric: MetricType,
    // storage
    vectors: Vec<Vec<f32>>,
    levels: Vec<Vec<Vec<VectorId>>>,  // [node][level] -> neighbors
    entry_point: VectorId,
    max_level: usize,
}
```

Standard HNSW insertion (Malkov & Yashunin 2018):
1. Random level assignment: l = floor(-ln(uniform()) * ml)
2. Greedy search from entry point down to level l+1 (find closest)
3. At levels l..0: insert with beam search, prune neighbors (simple or heuristic)
4. Update entry point if new node has higher level

### 2.3 Future: Vamana builder (`crates/index/vamana.rs`)

Planned as Phase 2.5 or 3 — bounded-degree graph with more disk-friendly
access patterns (fixed out-degree, no multi-level structure). Implements
the same `GraphBuilder` / `GraphStore` traits.

### 2.4 Graph serialization to MVP disk format

Write the constructed graph to Phase 3's flat-file layout:
- Adjacency lists serialized into fixed-size blocks
- PQ codes written as a contiguous array
- Exact vectors written sequentially

---

## Phase 3: On-Disk Storage Engine (MVP — Flat Files)

**Goal:** Simple, correct, NVMe-aligned storage. No slotted pages yet.
Three separate files, each with a trivial layout.

### 3.1 File layout (`crates/storage/`)

```
index_dir/
├── meta.json               # index metadata (dim, metric, M, ef, quantizer config,
│                           #   num_vectors, adj_block_size, code_size, vec_size)
├── adjacency.dat           # fixed-size adjacency blocks, O_DIRECT aligned
├── pq_codes.dat            # contiguous PQ code array, loaded into DRAM at startup
├── vectors.dat             # exact vectors, O_DIRECT aligned, read during refine
├── nav_graph.dat           # small navigation graph (0.1-0.5% sampled), loaded at startup
└── record_map.dat          # only needed if block_id != vid (optional in MVP)
```

### 3.2 Adjacency blocks (`adjacency.dat`)

Each vector gets one fixed-size block:

```
Block layout (4096 bytes):
┌──────────────────────────────┐
│ VectorId          (4 bytes)  │
│ Num neighbors     (2 bytes)  │
│ Padding           (2 bytes)  │
│ Neighbor IDs      (R * 4 B)  │  R = max_degree (e.g., 64 → 256 bytes)
│ Reserved / zero-padded       │
└──────────────────────────────┘
```

- `block_id = vid` — implicit mapping, no separate record_map needed
- Offset in file: `vid * BLOCK_SIZE`
- BLOCK_SIZE = 4096 bytes (one NVMe page)
- For graphs with small adjacency lists (< 4KB), this wastes space. That is
  acceptable for MVP — simple and correct beats dense and fragile.
- If adjacency list + header < 4KB, remaining space is zero-padded.
- If adjacency list > 4KB (unlikely with bounded degree), overflow to a second
  block and store a link.

### 3.3 PQ codes (`pq_codes.dat`)

**Loaded into DRAM at startup — no mmap.**

```rust
pub struct PqCodeStore {
    codes: Vec<u8>,         // contiguous: codes[vid * code_size .. (vid+1) * code_size]
    code_size: usize,       // bytes per code (e.g., 32 for 8-subspace PQ with 8-bit codes)
    num_vectors: usize,
}

impl PqCodeStore {
    /// Read entire file into a Vec<u8> at startup.
    pub fn load(path: &Path) -> io::Result<Self> { ... }

    /// O(1) lookup by VectorId — no IO, no cache, pure pointer arithmetic.
    pub fn get(&self, vid: VectorId) -> &[u8] {
        let offset = vid.0 as usize * self.code_size;
        &self.codes[offset..offset + self.code_size]
    }
}
```

Why not mmap:
- PQ codes are small (e.g., 32 bytes/vector × 100M vectors = 3.2 GB) and
  accessed uniformly during every query. They belong fully in DRAM.
- Under multi-tenant memory pressure, mmap'd pages can be evicted by the OS
  unpredictably, causing latency spikes. Explicit allocation is stable.
- If the code store is too large for DRAM, a future `CodeBlockPool` can cache
  blocks — but that's post-MVP.

### 3.4 Exact vectors (`vectors.dat`)

Contiguous array, read on demand during refinement:

```
vectors.dat layout:
  [vec_0: dim * 4 bytes] [vec_1: dim * 4 bytes] ... [vec_N: dim * 4 bytes]
```

- Offset: `vid * dim * sizeof(f32)`
- Read size: `dim * sizeof(f32)` (may span page boundaries — align reads to
  4KB pages and extract the vector from the buffer)
- All reads via O_DIRECT + io_uring

### 3.5 Navigation graph (`nav_graph.dat`)

Small in-memory graph (0.1-0.5% of dataset), loaded at startup:

```rust
pub struct NavGraph {
    graph: HnswIndex,              // tiny HNSW built from sampled vectors
    sample_to_full: Vec<VectorId>, // map sampled IDs → full dataset IDs
}
```

Serialized with serde (bincode or custom). Loaded into DRAM at startup.
Typically ~30-50 MB for 100M vectors.

---

## Phase 4: Buffer Pool — Split by Object Type

**Goal:** Separate pools for adjacency blocks and exact vectors. No single
"universal slot" that wastes memory on worst-case record sizes.

### 4.1 AdjacencyPool (`crates/buffer/adjacency_pool.rs`)

Caches adjacency blocks (fixed 4KB each in MVP layout).

```rust
pub struct AdjacencyPool {
    /// Pre-allocated buffer: capacity * BLOCK_SIZE bytes.
    data: AlignedBuffer,
    /// VectorId → slot state. Indexed by vid.
    /// Packed u32: bits [31] = resident, bits [0..30] = slot_index.
    mapping: Vec<AtomicU32>,
    /// Slot metadata (state machine).
    slot_states: Vec<AtomicU8>,   // Free | Locked | Occupied | Marked
    /// Lock-free free list.
    free_list: SegQueue<u32>,
    /// Clock hand for eviction.
    clock_hand: AtomicU32,
    /// Number of slots.
    capacity: u32,
}
```

Because adjacency blocks are fixed-size (4KB), every slot is the same size.
No waste from variable-size records. The pool is sized as:

    capacity = (memory_budget * adj_pool_fraction) / BLOCK_SIZE

Default `adj_pool_fraction` = 0.70 (adjacency lists get 70% of cache budget,
per Gorgeous's finding that adjacency caching is more valuable than vector caching).

### 4.2 VectorPool (`crates/buffer/vector_pool.rs`) — Optional

Caches exact vectors for hot nodes. Slot size = `dim * sizeof(f32)`.
Only used if memory budget allows after adjacency pool allocation.

```rust
pub struct VectorPool {
    data: AlignedBuffer,
    mapping: Vec<AtomicU32>,      // vid → slot
    slot_states: Vec<AtomicU8>,
    free_list: SegQueue<u32>,
    clock_hand: AtomicU32,
    capacity: u32,
    vector_size: usize,           // dim * 4
}
```

Default `vector_pool_fraction` = 0.30 of remaining cache budget (after PQ codes
in DRAM and adjacency pool).

### 4.3 Slot state machine (same for both pools)

```
    ┌──────┐  loading   ┌────────┐
    │ Free │ ──────────→ │ Locked │
    └──────┘             └────────┘
       ↑                    │ load complete
       │ evicting           ↓
    ┌────────┐  mark     ┌──────────┐
    │ Locked │ ←──────── │ Occupied │
    └────────┘           └──────────┘
       ↑                    │ clock sweep
       │ evict              ↓
       │                ┌────────┐
       └────────────────│ Marked │
                        └────────┘
                           │ access (second chance)
                           ↓
                        ┌──────────┐
                        │ Occupied │ (back to occupied)
                        └──────────┘
```

All transitions via CAS. No locks on the hot path.

### 4.4 Memory budget allocation

```
Total cache budget (e.g., 20% of dataset size)
├── PQ codes: loaded into DRAM at startup (not part of the pool)
├── Nav graph: loaded into DRAM at startup (not part of the pool)
├── AdjacencyPool: 70% of remaining budget
└── VectorPool: 30% of remaining budget (optional, can be 0)
```

---

## Phase 5: Async Execution Runtime (monoio + io_uring)

**Goal:** Thread-per-core scheduler that overlaps IO and compute via coroutines.

### 5.1 Runtime architecture (`crates/engine/runtime.rs`)

Using monoio as the async runtime with io_uring backend:

```rust
pub struct WorkerThread {
    core_id: usize,
    adj_pool: Arc<AdjacencyPool>,
    vec_pool: Arc<VectorPool>,
    pq_codes: Arc<PqCodeStore>,
    nav_graph: Arc<NavGraph>,
    adj_file: monoio::fs::File,     // adjacency.dat, O_DIRECT
    vec_file: monoio::fs::File,     // vectors.dat, O_DIRECT
    query_rx: Receiver<QueryTask>,
}

impl WorkerThread {
    pub fn run(self) {
        core_affinity::set_for_current(CoreId { id: self.core_id });
        monoio::RuntimeBuilder::new()
            .enable_all()
            .build()
            .unwrap()
            .block_on(self.main_loop());
    }

    async fn main_loop(&self) {
        loop {
            let queries = self.query_rx.recv_batch().await;
            let futures: Vec<_> = queries
                .into_iter()
                .map(|q| monoio::spawn(self.handle_query(q)))
                .collect();
            join_all(futures).await;
        }
    }
}
```

### 5.2 IO operations with separate budgets (`crates/engine/io.rs`)

```rust
pub struct IoDriver {
    adj_file: monoio::fs::File,
    vec_file: monoio::fs::File,
    /// Semaphore: max adjacency reads in-flight per thread.
    adj_inflight: Semaphore,     // e.g., capacity = 64
    /// Semaphore: max refine reads in-flight per thread.
    vec_inflight: Semaphore,     // e.g., capacity = 32
}

impl IoDriver {
    /// Read an adjacency block. Acquires adj_inflight permit.
    pub async fn read_adj_block(&self, vid: VectorId) -> AlignedBuffer {
        let _permit = self.adj_inflight.acquire().await;
        let offset = vid.0 as u64 * BLOCK_SIZE as u64;
        let buf = AlignedBuffer::new(BLOCK_SIZE);
        let (result, buf) = self.adj_file.read_exact_at(buf, offset).await;
        result.unwrap();
        buf
    }

    /// Read an exact vector. Acquires vec_inflight permit.
    pub async fn read_vector(&self, vid: VectorId, dim: usize) -> AlignedBuffer {
        let _permit = self.vec_inflight.acquire().await;
        let vec_size = dim * std::mem::size_of::<f32>();
        let page_offset = (vid.0 as u64 * vec_size as u64) & !(4095);
        let buf = AlignedBuffer::new(4096);  // read aligned page
        let (result, buf) = self.vec_file.read_exact_at(buf, page_offset).await;
        result.unwrap();
        buf
    }
}
```

Why separate inflight budgets:
- Adjacency reads are on the critical path (graph traversal stalls without them).
- Refine reads are batched at the end and less latency-sensitive.
- Without separation, a burst of refine reads can exhaust the io_uring SQ and
  starve adjacency reads, causing traversal stalls and p99 spikes.

### 5.3 Batch size tuning

`B = ⌈α · I/T⌉` concurrent coroutines per thread:
- I = ~100μs (NVMe 4KB random read latency)
- T = compute time per search step (dataset-dependent)
- α = 1.0-2.0 (start with 1.5)
- Default B = 2 for high-dim (768d+), B = 4-8 for low-dim (128d)

Hard limits per thread:
- `max_adj_inflight` = 64 (default)
- `max_vec_inflight` = 32 (default)
- Total `max_inflight` = `max_adj_inflight + max_vec_inflight` ≤ io_uring SQ depth

Configurable via `SearchConfig`.

### 5.4 Dispatcher (`crates/engine/dispatcher.rs`)

Round-robin query distribution across worker threads:

```rust
pub struct Dispatcher {
    workers: Vec<Sender<QueryTask>>,
    next_worker: AtomicUsize,
}

impl Dispatcher {
    pub fn submit(&self, query: QueryTask) {
        let idx = self.next_worker.fetch_add(1, Relaxed) % self.workers.len();
        self.workers[idx].send(query).unwrap();
    }
}
```

---

## Phase 6: Two-Stage Search Pipeline

**Goal:** The 5-stage query pipeline with cache-aware beam search.
All hot-path data structures are allocation-free (use QueryArena from Phase 1.5).

### 6.1 Pipeline trait design (`crates/engine/pipeline.rs`)

```rust
pub trait Router: Send + Sync {
    fn route(&self, query: &[f32]) -> Vec<VectorId>;
}

pub trait CandidateProducer: Send + Sync {
    async fn produce(&self, query: &[f32], entry_points: &[VectorId],
                     params: &SearchParams, arena: &mut QueryArena) -> &[Candidate];
}

pub trait Scorer: Send + Sync {
    fn score(&self, query: &[f32], candidates: &[VectorId], results: &mut [f32]);
}

pub trait Pruner: Send + Sync {
    fn prune(&self, scored: &[(VectorId, f32)], params: &SearchParams,
             out: &mut Vec<VectorId>);
}

pub trait Refiner: Send + Sync {
    async fn refine(&self, query: &[f32], candidates: &[VectorId])
        -> Vec<(VectorId, f32)>;
}
```

### 6.2 Router (MemGraph)

Uses the nav graph loaded at startup:

```rust
pub struct MemGraphRouter {
    nav_graph: NavGraph,
}

impl Router for MemGraphRouter {
    fn route(&self, query: &[f32]) -> Vec<VectorId> {
        let nav_results = self.nav_graph.graph.search(query, 1, 10);
        nav_results.iter()
            .map(|(id, _)| self.nav_graph.sample_to_full[id.0 as usize])
            .collect()
    }
}
```

### 6.3 Cache-aware beam search (`crates/engine/search.rs`)

Core search using allocation-free data structures:

```rust
pub async fn cache_aware_beam_search(
    query: &[f32],
    entry_points: &[VectorId],
    params: &SearchParams,
    arena: &mut QueryArena,       // reusable per-query scratch space
    adj_pool: &AdjacencyPool,
    io: &IoDriver,
    pq_codes: &PqCodeStore,
    quantizer: &dyn Quantizer,
) -> &[Candidate] {

    let candidates = &mut arena.candidates;  // BoundedHeap, pre-allocated
    let visited = &mut arena.visited;         // VisitedSet with generation counting
    visited.clear();                          // O(1): just bump generation

    // Initialize with entry points
    for &ep in entry_points {
        let code = pq_codes.get(ep);
        let dist = quantizer.distance(query, code);
        candidates.push(Candidate { vid: ep, dist });
        visited.set(ep);
    }

    while let Some(v) = candidates.pop_nearest_unvisited() {
        // Cache-aware pivot: if v is on disk, check look-ahead for in-memory alt
        let (target, prefetch_list) = if !adj_pool.is_resident(v.vid) {
            let mut pivot = v;
            let look_ahead = candidates.peek_top_n(params.beam_width);
            for c in look_ahead {
                if adj_pool.is_resident(c.vid) {
                    pivot = c;
                    break;
                } else {
                    io.prefetch_adj_block(c.vid);  // fire-and-forget async
                }
            }
            pivot
        } else {
            v
        };

        // Load adjacency block (may suspend if on disk)
        let adj_block = adj_pool.get_or_load(target.vid, io).await;

        // Expand: compute approximate distances to neighbors
        let neighbors = adj_block.decode_neighbors(&mut arena.neighbor_buf);
        for &nbr in neighbors {
            if !visited.test_and_set(nbr) {
                let code = pq_codes.get(nbr);
                let dist = quantizer.distance(query, code);
                candidates.push(Candidate { vid: nbr, dist });
            }
        }
    }

    candidates.as_sorted_slice()
}
```

### 6.4 Refinement stage

```rust
pub async fn refine(
    query: &[f32],
    candidates: &[Candidate],       // from beam search, sorted by approx dist
    sigma: f32,                      // refinement ratio, default 0.5
    k: usize,
    vec_pool: &VectorPool,
    io: &IoDriver,
    distance: &dyn DistanceComputer,
    dim: usize,
) -> Vec<(VectorId, f32)> {
    let refine_count = (candidates.len() as f32 * sigma) as usize;
    let to_refine = &candidates[..refine_count.min(candidates.len())];

    // Batch IO: submit all exact vector reads concurrently
    let mut exact_results = Vec::with_capacity(to_refine.len());
    for &Candidate { vid, .. } in to_refine {
        let vec_data = vec_pool.get_or_load(vid, io, dim).await;
        let exact_vec = vec_data.as_f32_slice(dim);
        let exact_dist = distance.distance(query, exact_vec);
        exact_results.push((vid, exact_dist));
    }

    exact_results.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    exact_results.truncate(k);
    exact_results
}
```

### 6.5 Dynamic beam width (post-MVP optimization)

Start narrow, widen as search converges:

```rust
pub struct DynamicBeamWidth {
    omega_min: usize,      // approach phase width (e.g., 1)
    omega_max: usize,      // converge phase width (e.g., 8)
    expansion_rate: f32,   // how quickly to widen
}
```

Track distance improvement between iterations. When improvement slows (converge
phase), increase beam width. Defer to post-baseline — fixed beam width is fine
for MVP.

---

## Phase 7: Control Plane

**Goal:** Background processes for heat tracking, tier migration, block reorg.

### 7.1 Heat tracker (`crates/engine/heat.rs`)

```rust
pub struct HeatTracker {
    access_counts: Vec<AtomicU32>,      // per-VectorId access counter
    decay_interval: Duration,            // e.g., every 60 seconds
    decay_factor: f32,                   // e.g., 0.5 (exponential decay)
}

impl HeatTracker {
    pub fn record_access(&self, vid: VectorId) {
        self.access_counts[vid.0 as usize].fetch_add(1, Relaxed);
    }

    pub async fn decay_loop(&self) {
        loop {
            sleep(self.decay_interval).await;
            for count in &self.access_counts {
                let old = count.load(Relaxed);
                count.store((old as f32 * self.decay_factor) as u32, Relaxed);
            }
        }
    }
}
```

### 7.2 Tier migration

Promotion/demotion decisions based on heat:
- Promotion threshold: if heat > threshold AND pool has space, proactively load
- Adaptive threshold: hill-climbing on cache miss rate gradient
- Track heat separately per object type (adjacency vs vectors)

### 7.3 Block reorganization

Background process to re-place records when access patterns shift:
- Monitor cache hit rate per region
- Incremental page rewrite when patterns shift
- No global index rebuild required

---

## Post-MVP Optimization Phases

These are deferred until the baseline pipeline is running and profiled:

### Opt-A: Slotted page layout (replaces flat adjacency blocks)

Variable-size compressed records in B-tree-inspired pages. Replaces the simple
4KB-per-node layout with dense packing:

```
Page (4096 bytes):
  Header (6 bytes): Tag(1) | Count(1) | HeapStart(2) | HeapUsed(2)
  Slot Array (grows →): [VID(4) | Color(1) | Length(2) | Offset(2)] = 9 bytes/slot
  Free Space
  Data Heap (← grows): variable-size compressed records
```

Requires record-level buffer pool changes (variable slot sizes or slab allocator).

### Opt-B: Graph-replicated disk blocks

Pack R neighbors' adjacency lists into each node's disk block. Saves one IO per
neighbor during subsequent traversal hops. Complexity: build-time bin-packing,
record_map update, overflow handling.

Only worth it after profiling confirms "next-hop adjacency miss" is the dominant
IO cost.

### Opt-C: Affinity co-placement

Distance-based affinity grouping with Color tags. Proactive loading of
same-affinity records on cache miss. Start with simple insertion-order locality
or BFS-based grouping; graduate to learned τ thresholds after baseline profiling.

### Opt-D: Vamana graph backend

Bounded-degree graph with fixed out-degree — more IO-predictable than HNSW's
multi-level structure. Implements `GraphBuilder` / `GraphStore` traits.

---

## Build Order

```
Phase 1: Core types, distance functions, PQ quantizer, QueryArena
         ~1-2 weeks
         Deliverable: crates/core with working PQ, SIMD distances,
         BoundedHeap, VisitedSet

Phase 2: HNSW graph construction (in-memory) behind GraphStore trait
         ~2-3 weeks
         Deliverable: crates/index with HNSW build + in-memory search,
         verify recall on Sift1M

Phase 3: Flat-file on-disk storage (adjacency.dat, pq_codes.dat, vectors.dat)
         ~1-2 weeks
         Deliverable: crates/storage with O_DIRECT reads, PQ code loading,
         graph serialization

Phase 4: Split buffer pools (AdjacencyPool + VectorPool)
         ~2 weeks
         Deliverable: crates/buffer with CAS state machine, clock eviction,
         separate pools with configurable budget split

Phase 5: Async runtime + io_uring (monoio) with separate IO budgets
         ~2-3 weeks
         Deliverable: crates/engine with thread-per-core workers, async reads,
         adj/vec inflight semaphores

Phase 6: Two-stage search pipeline (cache-aware beam search + refinement)
         ~3-4 weeks
         Deliverable: Full pipeline with MemGraph router, allocation-free
         beam search, end-to-end disk-based search

Phase 7: Control plane (heat tracking, tier migration)
         ~2 weeks
         Deliverable: Background heat tracking, adaptive eviction

Post-MVP: Opt-A through Opt-D based on profiling results
```

## Key Rust Dependencies

| Crate | Purpose |
|-------|---------|
| `monoio` | Thread-per-core async runtime with io_uring |
| `crossbeam` | Lock-free data structures (SegQueue for free list) |
| `core_affinity` | CPU thread pinning |
| `memmap2` | Memory mapping for offline construction only (never on query path) |
| `serde` + `serde_json` | Metadata serialization |
| `rand` + `rand_xoshiro` | Fast RNG for HNSW level assignment |
| `criterion` | Micro-benchmarks |
| `pulp` | Portable SIMD (alternative: `std::simd` on nightly) |
| `parking_lot` | Fast mutexes/rwlocks for construction only |

## Verification

- **Phase 1:** Unit tests for distance functions (vs naive), PQ round-trip,
  BoundedHeap correctness, VisitedSet generation counting
- **Phase 2:** Recall@10 on Sift1M ≥ 95% at ef=128
- **Phase 3:** Write-then-read verification with O_DIRECT, PQ codes load correctly
- **Phase 4:** Concurrent CAS stress test, eviction under pressure, budget split
- **Phase 5:** io_uring correctness, inflight semaphore fairness, throughput vs fio
- **Phase 6:** End-to-end recall matches in-memory; QPS at 10%/20%/50% buffer ratio
- **Phase 7:** Heat-based eviction improves hit rate under skew
- **Integration:** VectorDBBench suite
