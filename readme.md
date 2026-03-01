# Project Divergence

Project Divergence is a hardware-native, resource-scheduled retrieval engine designed for large-scale vector and hybrid search workloads.

It is built around a single idea:

Queries are not executed against a fixed index structure.
They are planned and scheduled across a graph of heterogeneous hardware resources.

Project Divergence treats storage, memory, compute, and network as first-class schedulable resource clouds. It provides predictable latency, multi-tenant isolation, and a continuous performance-to-cost spectrum without binding the system to a single ANN algorithm.

---

## Motivation

Modern vector databases hit the same structural walls:

1. **CPU underutilization is the real bottleneck, not SSD bandwidth.** Benchmarks show NVMe SSDs at under 9% read bandwidth utilization during search -- the CPU stalls on distance computation and graph traversal, not on IO. Systems that treat SSD as the bottleneck leave hardware on the table.

2. **mmap and page cache produce unpredictable tail latency.** OS-managed paging creates "lukewarm" pages: a 4KB page may contain one hot record and dozens of cold ones, wasting DRAM. Page-level caching cannot match record-level access patterns, and eviction decisions are invisible to the application.

3. **Per-tenant indexing causes memory bloat; metadata filtering degrades search quality.** Building a separate index per tenant uses 10x more memory than a shared index. The alternative -- metadata filtering on a shared index -- skips graph edges and degrades recall. Neither scales.

4. **Static indexes cannot adapt to distribution shifts or heterogeneous query patterns.** A global HNSW graph built for yesterday's data distribution routes poorly after ingestion skew changes. Entry point selection, partition boundaries, and cache placement all go stale.

5. **Tight coupling between index structure and execution strategy.** Systems optimized for one ANN algorithm cannot switch approaches based on workload characteristics, hardware availability, or cost constraints.

As workloads become multi-tenant, distribution shifts become frequent, and hardware becomes heterogeneous, these assumptions break down. Project Divergence is built for this environment.

---

## Core Architecture

### 1. Hardware as Resource Clouds

Every hardware component is modeled as a resource cloud with measurable, queryable properties. The planner's job is to identify which resource is the bottleneck for a given query right now and plan around it.

Per-resource properties that matter:

| Resource | Key Properties |
|----------|---------------|
| **CPU** | Core count, SIMD width, cycles per distance computation |
| **DRAM** | Capacity, bandwidth (GB/s), access latency (ns) |
| **NVMe SSD** | 4KB random read IOPS, sequential bandwidth, queue depth limits |
| **RDMA NIC** | IOPS limits, round-trip latency, bandwidth |
| **GPU** | Compute throughput, device memory capacity, transfer overhead |
| **Object Storage** | GET latency (ms), throughput (req/s), cost per operation |

The bottleneck shifts dynamically: a cold query on a lightly loaded system is IO-bound; the same query under heavy concurrency is CPU-bound. The planner observes current resource state -- queue lengths, cache occupancy, IO pipeline depth -- and routes accordingly.

Execution decisions are made against this resource graph rather than against hardcoded tier assumptions.

---

### 2. Object-Based Tiered Architecture

All index and data structures are decomposed into versioned objects with an explicit compression hierarchy:

| Tier | Representation | Typical Placement | Purpose |
|------|---------------|-------------------|---------|
| **L1** | 1-bit binary codes | L3 cache / DRAM | Fast coarse filtering, Hamming distance |
| **L2** | 4-bit quantized codes (e.g., RaBitQ) | DRAM / NVMe | Approximate scoring |
| **L3** | Exact vectors | NVMe / Object store | Final refinement of top-k subset |

Routing metadata (graph edges, centroids, entry points) has higher access frequency than any vector representation and is prioritized for the fastest available tier. Caching graph structure alone allows far more nodes' topology to reside in memory than caching full vectors.

Key properties:

- Object storage is the source of truth
- Higher-performance tiers act as adaptive working sets
- Object-level eviction and promotion are independent per object type -- routing metadata, compressed codes, and exact vectors are promoted/evicted on separate schedules
- Objects may be rebuilt, migrated, or evicted without affecting correctness

---

### 3. Query as a Resource Allocation Plan

A query is transformed into an execution plan under constraints (latency budget, resource budget, tenant quota, hardware availability, data residency). The planner determines:

- Which partitions or blocks to touch
- Which storage tiers to access
- Whether to use CPU, GPU, or remote resources
- How much parallelism to apply
- When to degrade or terminate early

This turns retrieval into a scheduling problem rather than a fixed algorithm invocation.

**Batch-level planning.** Queries arriving in a batch share cluster access patterns due to embedding model structure. The planner groups similar queries to maximize cache reuse and prefetch efficiency, reducing cache miss rates from ~60% to over 90%.

**Learned routing.** Lightweight models handle entry point selection and partition pruning. A contrastive two-tower model projects query and hub-node representations into a shared latent space for entry point recommendation. A gradient-boosted model predicts the minimum shards to visit per query. Both run in microseconds.

**Cache-aware decisions.** The planner prefers in-memory candidates over slightly-closer cold candidates to avoid IO stalls. When the best candidate requires a disk read but a nearly-as-good candidate is already in the buffer pool, the planner takes the cached path.

**Resource-state awareness.** The planner reads current resource state before planning: NVMe queue depths, buffer pool contents, CPU utilization, outstanding IO count. Profiles like "ultra-performance" (full DRAM residency, aggressive prefetch, GPU rerank) or "cost-optimized" (small DRAM hotset, strict budgets, cold-tier reads allowed) are constraint presets applied to this same planning process.

---

### 4. Index-Agnostic Execution Model

Retrieval is decomposed into composable stages:

1. **Router** -- Determines candidate partitions or entry points. Learned routing models operate here.
2. **Candidate Producer** -- Generates candidate identifiers via graph traversal or posting list scan.
3. **Pruner** -- Eliminates unnecessary blocks or partitions before heavy computation.
4. **Scorer** -- Computes approximate similarity using compressed codes and CPU SIMD or GPU.
5. **Refiner** -- Performs high-precision reranking from exact vectors on slower tiers.

Each stage can run on different hardware. The Router may run on CPU using a lightweight model while the Scorer runs on GPU.

**Graph-based indexing (Vamana/HNSW) is the primary method** -- it dominates for latency-critical, high-accuracy workloads. IVF serves as a secondary method for update-heavy or append-heavy workloads where graph maintenance cost is prohibitive.

Any ANN method can be implemented as a specific combination of these stages. The system is not bound to one algorithm.

---

### 5. Multi-Tenant Isolation

Project Divergence uses shared index structures with per-tenant lightweight metadata rather than per-tenant index copies.

- A shared global clustering tree provides the base index structure
- Per-tenant routing is encoded via Bloom filters and shortlists attached to tree nodes, capturing each tenant's data distribution with minimal overhead
- Tenants grow deeper in dense regions of the shared tree without affecting other tenants

Each tenant receives resource quotas expressed in unified resource units:

- Bytes read from DRAM
- Bytes read from NVMe
- Object storage operations
- CPU cycles
- GPU time
- Network bandwidth

The scheduler enforces no noisy-neighbor interference, predictable tail latency, controlled cost exposure, and graceful degradation when limits are reached.

---

## Architecture

### Data Plane

The data plane executes query plans with minimal overhead and deterministic behavior.

**Thread-per-core with stackless coroutines.** Each core runs a single thread. Queries are multiplexed as stackless coroutines within that thread. No cross-thread synchronization on the hot path.

**io_uring with O_DIRECT.** All SSD reads bypass the OS page cache. io_uring provides kernel-level async IO with minimal syscall overhead. The application controls exactly what is cached and when.

**Record-level buffer pool.** Unlike page-level caching (where a 4KB page may hold one hot record and many cold ones), the buffer pool tracks individual records. Clock-based eviction captures skewed access patterns that page-level schemes miss.

**Controlled concurrency.** Each thread runs B = ceil(alpha * I/T) coroutines, where I is IO latency and T is per-query compute time. This formula maximizes IO-compute overlap without over-subscribing the SSD queue.

**Cache-aware beam search.** During graph traversal, the search algorithm checks the buffer pool before issuing IO. If a neighbor is already cached, it is visited immediately regardless of distance rank. This reorders traversal to minimize IO stalls while preserving result quality.

**Execution flow:** the data plane builds an execution DAG from the query plan, eliminates unnecessary blocks and tiers before heavy computation, performs batched controlled IO from appropriate tiers, computes similarities, retrieves high-precision data for final ranking, and checks resource budgets at each stage for early termination.

### Control Plane

The control plane manages placement, migration, and system adaptation.

**Heat tracking with adaptive migration.** Per-object access frequency is monitored. Records are promoted or demoted between tiers based on access heat, using hill-climbing on cache miss rate to find optimal migration thresholds. Exponential sampling reduces monitoring overhead.

**Distribution shift detection.** The control plane monitors query and data distribution changes. When routing structures become stale (entry points no longer representative, partition boundaries misaligned), it triggers incremental re-routing -- rebuilding lightweight routing metadata without touching the base index.

**Cache segmentation monitoring.** In distributed deployments, the control plane tracks cache segmentation penalty (CSP = 1 - CHR/CHR_max) across compute nodes. Logical index partitioning via balanced clustering and adaptive query routing reduce redundant caching, keeping CSP low.

**Background operations.** Compaction, version management, affinity-based block reorganization, and tier rebalancing run as background tasks with bounded resource budgets that do not interfere with query serving.

---

## Storage Model

**Compression hierarchy.** Data exists at three fidelity levels (1-bit binary codes, 4-bit quantized codes, exact vectors), each stored as independent versioned objects. The control plane decides which levels are materialized in which tiers based on access patterns and memory budget.

**SSD page alignment.** Each logical block maps to exactly one physical SSD page read (4KB). No logical operation requires reading across page boundaries.

**Co-located routing hints.** Each on-disk block contains not only its own data but also the adjacency lists (neighbor IDs) of its contained records. Reading one block gives the traversal algorithm everything it needs to decide the next hop without an additional IO.

**Affinity-based co-placement.** Spatially proximate vectors are co-located on the same SSD page. This is achieved through balanced clustering and local reorganization without requiring a global index rebuild.

**Versioned manifests.** Object storage is authoritative. Tiered caches are ephemeral. Index structures are locally evolvable. Block reorganization can occur without global rebuilds. This enables elastic scaling, hotset reshaping, online reconfiguration, and rapid rollback.

---

## Implementation Principles

- **Rust-native implementation** -- validated by existing high-performance vector search systems
- **io_uring + O_DIRECT** -- bypass OS page cache entirely; application-controlled caching
- **Thread-per-core, stackless coroutines** -- no cross-thread synchronization on hot path
- **Record-level buffer pool with clock eviction** -- not mmap, not page-level
- **Controlled queue depth** -- B = ceil(alpha * I/T) coroutines per thread
- **Graph-based primary index (Vamana)** with IVF as secondary for update-heavy workloads
- **4-bit RaBitQ + delta/Elias-Fano adjacency encoding** for on-disk compression
- **Learned routing** at the Router stage (two-tower entry point selection, GBDT partition pruning)
- **Lock-sharded internal structures** with optimistic versioned concurrency

---

## Status

Early architecture stage.

Upcoming milestones:

- Core resource graph implementation
- Object store abstraction
- Execution planner prototype
- Single-node NVMe tier prototype
- Multi-tenant quota enforcement
- Optional GPU integration

---

## License

TBD
