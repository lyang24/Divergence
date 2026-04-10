# Paper Distillation 6.0: PageANN + VeloANN

**Date**: 2026-03-31
**Sources**: PageANN (arXiv 2509.25487, Tsinghua), VeloANN (PVLDB 2026, Zhejiang)

---

## 1. PageANN: Page-Centric Graph for SSD-Based ANN Search

**Paper**: "PageANN: A Page-Centric Graph Index for SSD-Based Approximate Nearest Neighbor Search"

### 1.1 Core Idea

Restructures the vector index so that each SSD page IS a graph node. Traditional disk ANN stores one vector's adjacency per page and pays one IO per graph hop. PageANN packs multiple vectors onto each page along with their compressed representations, then builds a *page-level* graph where edges connect pages rather than individual vectors.

Result: one page read gives you both graph structure AND approximate distances for scoring neighbors on the next hop. Eliminates the two-stage penalty (graph traversal → separate vector reads).

### 1.2 Page-Node Graph Construction

1. **Partition vectors into pages** using topology-aware packing (graph neighbor adjacency, not query traces)
2. **Build page-level graph**: edge between page A and page B if any vector on A has a vector neighbor on B
3. **Merge redundant edges**: if multiple vectors on page A point to vectors on page B, collapse into one page-to-page edge
4. **Store inline compressed codes**: each page contains compressed (PQ/SQ) representations of all its vectors

### 1.3 Search Procedure

1. Pop page-node from candidate heap
2. Read entire page (one IO): get all vectors + their compressed codes + adjacency to other pages
3. Score ALL vectors on the page using inline compressed codes (zero extra IO)
4. For each neighboring page: estimate distance using the best vector on current page that links to it
5. Push neighboring pages into candidate heap

### 1.4 Key Results

- **46% fewer IOs** at iso-recall vs DiskANN (their primary claim)
- Page dedup: visiting a page scores ALL its vectors at once, avoiding redundant reads
- Compressed codes inline = zero-IO approximate scoring per hop
- Lightweight hash routing index: maps vector IDs to pages in DRAM (replaces full adjacency index)

### 1.5 Relevance to Divergence

**High relevance, partial adoption path:**

| PageANN Idea | Divergence Mapping | Adoption Cost |
|---|---|---|
| Inline compressed codes | Store PQ/SAQ codes in adjacency blocks → zero-IO proxy scoring | Medium (block format change) |
| Topology-aware page packing | Pack graph neighbors onto same page → fewer unique pages/q | Medium (build-side only) |
| Page-node graph | Full restructure to page-level search | High (too invasive) |
| Merge redundant edges | Collapse same-page neighbor edges | Low-Medium |
| Hash routing index | Not needed (we already have adj_index in DRAM) | N/A |

**Key insight**: We don't need the full page-node graph restructure. The highest-value idea — inline compressed codes — can be adopted incrementally by expanding our existing adjacency block format. This is exactly our existing Opt-A plan.

**What PageANN validates**: inline codes work. Their 46% IO reduction confirms that score-before-expand gating is the right next step. Our existing `inline_pq_design.md` is on the right track.

---

## 2. VeloANN: Cache-Optimized Disk ANN with io_uring

**Paper**: "VeloANN: An Efficient and Robust Disk-Based ANN Index via Coroutine-Based Asynchronous I/O" (PVLDB 2026)

### 2.1 Architecture

Rust + monoio + io_uring, thread-per-core, coroutine-based async. **Nearly identical to Divergence.** This validates our core architectural choices.

### 2.2 Key Ideas

#### 2.2a Record-Level Buffer Pool
- Cache individual vector records (not full pages)
- LRU eviction, pin/unpin semantics
- **Divergence status**: ✓ Already implemented (AdjacencyPool, 8-way set-associative, clock eviction)

#### 2.2b Affinity-Based Co-Placement (Record Coloring)
- During graph construction, assign colors to vertices so spatially close vectors get the same color
- Vectors with the same color are packed onto the same SSD page
- Uses **static graph properties** (neighbor adjacency from the graph structure itself)
- This is fundamentally different from our failed TWPP experiment:
  - TWPP: query-specific co-expansion traces → don't generalize across queries
  - VeloANN: graph neighbor adjacency → universal, query-independent
  - Our TWPP postmortem concluded "static graph properties are more robust than query traces" — VeloANN confirms this

**Algorithm**: greedy coloring where each vertex inherits the most common color among its already-colored neighbors. Colors map to page assignments. Simple, O(N × degree).

#### 2.2c Compressed Variable-Size Records
- **ExtRaBitQ**: 4-bit quantized vectors (similar quality range to our SAQ)
- **Elias-Fano compressed adjacency lists**: variable-length encoding for neighbor IDs, much more compact than fixed u32 arrays
- **Slotted page layout**: variable-size records packed into pages with a slot directory
  - Each page has a header with slot count + offsets
  - Records vary in size (different degrees, different vector dimensions)
  - Better page utilization than fixed 4KB blocks (our current format wastes 3960/4096 bytes at degree=32)

#### 2.2d Cache-Aware Beam Search
- Prioritize expanding candidates whose data is already in the buffer pool
- Split candidates into "in-memory" (expand immediately, no IO) and "on-disk" (need IO)
- **Divergence status**: ✓ Partially implemented (page_sched variant already does priority-based scheduling)

### 2.3 Key Results

- 3-10× faster than DiskANN on their benchmarks
- Affinity co-placement reduces unique pages accessed by 20-40%
- Variable-size records improve page utilization from ~15% to ~80%
- Cache-aware scheduling reduces effective IO by prioritizing cached data

### 2.4 Relevance to Divergence

| VeloANN Idea | Divergence Status | Priority |
|---|---|---|
| Rust + io_uring + thread-per-core | ✓ Already our architecture | — |
| Record-level buffer pool | ✓ AdjacencyPool | — |
| Cache-aware beam search | ✓ page_sched variant | — |
| Affinity co-placement | **Not implemented** | ★★ High (static graph property, confirmed robust) |
| Compressed variable-size records | **Not implemented** (Opt-A) | ★★ High (prerequisite for inline codes) |
| Slotted page layout | **Not implemented** (Opt-A) | ★★ High (enables variable-size) |
| Elias-Fano adjacency | Not implemented | ★ Medium (nice compression, not critical) |

---

## 3. Synthesis: What Changes for Divergence

### 3.1 Confirmed Correct
- Thread-per-core + monoio + io_uring (both papers use similar arch)
- Record-level caching (VeloANN validates)
- SAQ/PQ as proxy distance (both papers use compressed codes for approximate scoring)
- Static graph properties for layout > query traces (VeloANN confirms our TWPP postmortem)

### 3.2 New Priorities (Updated)

Previous priority list:
1. PageANN-style inline neighbor codes
2. SAQ ef sweep
3. Vamana alpha>1

**Updated priority list:**
1. **Slotted page layout + variable-size records** — infrastructure for everything below
2. **Inline PQ/SAQ codes** — zero-IO proxy scoring (PageANN's 46% IO reduction)
3. **Affinity co-placement** — pack graph neighbors onto same page (VeloANN's 20-40% IO reduction)
4. Vamana alpha>1 — fewer hops per query (orthogonal, can be done anytime)

### 3.3 Dead Ends (Confirmed)
- Query-trace-based page packing (TWPP) — VeloANN's success with static graph affinity confirms traces are the wrong signal
- Pipelined refine (overlapping vec IO with traversal) — device QD contention makes this a net loss
- Page-node graph (full PageANN restructure) — too invasive, most of the value comes from inline codes alone
