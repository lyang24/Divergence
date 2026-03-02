# Research Paper Summary for Project Divergence

This document summarizes 14 research papers and maps their findings to Divergence's five core ideas:
1. Hardware as Resource Clouds
2. Object-Based Tiered Architecture
3. Query as a Resource Allocation Plan
4. Index-Agnostic Execution Model
5. Multi-Tenant Isolation via Resource Units

---

## Paper Summaries

### 1. Storage-Based ANNS: Performance, Cost, and I/O Characteristics
**File:** `2025-iiswc-vectordb.pdf`
**Authors:** Ren, Doekemeijer, Apparao, Trivedi (IISWC 2025)

**Core finding:** First systematic benchmarking of storage-based vector databases (Milvus, Qdrant, Weaviate, LanceDB) on NVMe SSDs.

**Key insights:**
- Storage-based setups do NOT necessarily perform worse than memory-based ones. Milvus-DiskANN outperforms Milvus-IVF (in-memory) by 3.2x throughput.
- Modern NVMe SSDs are severely underutilized -- max read bandwidth achieved was only 8.9% of Samsung 990 Pro capacity.
- **CPU is the bottleneck, not SSD.** Over 99.99% of I/O requests are 4KB random reads.
- The choice of vector database system matters as much as index algorithm -- up to 7.1x throughput variation with the same index type.
- `search_list` parameter increase from 10 to 100 yields only 6.5% accuracy gain but 60.9% throughput decrease.

**Relevance to Divergence:**
- Strongly validates the "Hardware as Resource Clouds" idea -- CPU vs SSD resource balance is critical and current systems don't model it.
- Validates "Query as Resource Allocation Plan" -- tuning knobs like `search_list` create a continuous accuracy-cost spectrum that could be planner-controlled.
- The massive SSD underutilization suggests IO scheduling is a first-order concern for the data plane.

---

### 2. Curator: Efficient Indexing for Multi-Tenant Vector Databases
**File:** `2401.07119v1.pdf`
**Authors:** Jin, Wu, Hu, Maggs, Zhang, Zhuo (Duke/Yale, 2024)

**Core finding:** Purpose-built multi-tenant vector index that achieves both high search performance and low memory footprint, breaking the per-tenant-index vs. metadata-filtering trade-off.

**Key insights:**
- Per-tenant indexing uses 10.1x more memory than metadata filtering on YFCC100M.
- Curator's Tenant Clustering Trees (TCTs) are implicit sub-trees of a shared Global Clustering Tree (GCT), encoded via Bloom filters and shortlists.
- 32.9x faster than metadata-filtered IVF, 7.9-8.7x less memory than per-tenant HNSW.
- Each tenant's data distribution is captured with minimal overhead by growing deeper in dense regions.

**Relevance to Divergence:**
- Directly validates "Multi-Tenant Isolation via Resource Units" -- shows that per-tenant resource accounting is essential but doesn't need per-tenant index replication.
- The TCT concept maps to Divergence's "Router" stage -- tenant-aware routing without dedicated structures per tenant.
- Bloom filters + shortlists are a form of "Candidate Producer" with built-in access control.

---

### 3. ZipCache: DRAM/SSD Cache with Built-in Transparent Compression
**File:** `2411.03174v3.pdf`
**Authors:** Xie, Ma, Zhong, Chen, Zhang (MEMSYS '24)

**Core finding:** Hybrid DRAM/SSD key-value cache that treats compression as a first-class design component using computational SSDs (ScaleFlux CSD 3000).

**Key insights:**
- B+tree index preserves key order for 2x higher compression ratio than hash-based layouts.
- "Super-leaf" pages decouple B+tree leaf page size from 4KB SSD I/O unit.
- Decompression early termination: hash-based sub-page mapping allows decompressing only 1/16 of a block (0.10us vs 1.48us).
- Adaptive compression bypassing: hot pages kept uncompressed in DRAM sub-tier.
- 72.4% higher throughput, 42.4% lower p90 latency, 26.2x less SSD write amplification vs CacheLib.

**Relevance to Divergence:**
- Validates "Hardware as Resource Clouds" -- computational SSD as a distinct resource with its own latency/throughput profile.
- The adaptive compression bypassing is a form of resource-aware data placement that Divergence's control plane could implement.
- Super-leaf pages with sub-page mapping could inform Divergence's "Compressed codes" object design.
- The inclusive caching model (data in both DRAM and SSD tiers) maps to Divergence's tiered cache philosophy.

---

### 4. d-HNSW: Efficient Vector Search on Disaggregated Memory
**File:** `2505.11783v1.pdf`
**Authors:** Liu, Fang, Qian (UC Santa Cruz, 2025)

**Core finding:** First vector search engine for RDMA-based disaggregated memory, achieving 117x latency reduction over naive approaches.

**Key insights:**
- Representative Index Caching: tiny meta-HNSW (~0.4MB) from 500 sampled vectors cached on compute node, serves as cluster classifier.
- RDMA-friendly graph layout: sub-HNSW clusters serialized contiguously with shared overflow memory.
- Batched query-aware data loading: doorbell batching merges multiple RDMA reads into single network round trip.
- Each sub-HNSW loaded from memory pool only once per query batch.

**Relevance to Divergence:**
- Directly maps to "Hardware as Resource Clouds" -- compute pool (limited DRAM, many cores) vs memory pool (large memory, near-zero compute) as distinct resource types.
- Meta-HNSW is precisely Divergence's "Router" concept -- lightweight routing metadata guiding access to heavier structures.
- Batch-aware loading validates "Query as Resource Allocation Plan" -- analyzing batch requirements to minimize resource usage.

---

### 5. GATE: Graph with Adaptive Topology and Query Awareness
**File:** `2506.15986v1.pdf`
**Authors:** Ruan, Chen, Yang, Ke, Gao (Zhejiang/HKBU, KDD 2025)

**Core finding:** Learned high-tier navigation graph that selects optimal entry points for graph-based ANNS, achieving 1.2-2.0x speedup.

**Key insights:**
- Hub nodes selected via Hierarchical Balanced K-Means; topological features extracted via Graph2Vec.
- Contrastive two-tower model projects hub node and query representations into shared latent space for "recommending" the optimal entry point.
- 30-40% reduction in search path length; only 1.2% gap between in-distribution and out-of-distribution queries.
- Plug-and-play: works on top of any existing graph index (NSG, HNSW) without modification.

**Relevance to Divergence:**
- Maps to "Index-Agnostic Execution Model" -- GATE is a composable "Router" stage that can be layered on any graph-based "Candidate Producer."
- Query-awareness through learned models aligns with Divergence's planner concept -- routing decisions adapted to query distribution.
- The two-tier index (hub graph + full graph) validates Divergence's decomposition into routing metadata + candidate blocks.

---

### 6. CoTra: Efficient and Scalable Distributed Vector Search with RDMA
**File:** `2507.06653v1.pdf`
**Authors:** Zhi, Chen, Yan, Lu, Li, Zhang, Chen, Cheng (CUHK/Fudan/Microsoft, 2025)

**Core finding:** Collaborative traversal algorithm for distributed vector search that achieves near-linear scalability (0.8x of ideal) across 16 machines.

**Key insights:**
- Independent sharding causes 4x redundant computation due to O(M * log(N/M)) > O(log N).
- Global index reduces redundancy but requires expensive remote memory accesses.
- CoTra: primary partitions maintain local candidate queues with periodic sync; secondary partitions serve on-demand via RDMA pull-data or task-push.
- Coroutine-based scheduling overlaps computation and communication.
- 9.8-13.4x throughput over single machine; 8.7-33.3x over Milvus.

**Relevance to Divergence:**
- Validates "Hardware as Resource Clouds" with heterogeneous execution across machines -- primary/secondary distinction maps to resource-aware scheduling.
- Pull-Data vs Task-Push is a query-time decision about which resource (bandwidth vs remote compute) to spend -- exactly "Query as Resource Allocation Plan."
- The navigation index (1% sample replicated everywhere) is another instance of Divergence's routing metadata as lightweight hot tier.
- Coroutine-based scheduling maps directly to Divergence's data plane: controlled concurrency with async IO.

---

### 7. SHINE: Scalable HNSW Index in Disaggregated Memory
**File:** `2507.17647v1.pdf`
**Authors:** Widmoser, Kocher, Augsten (Univ. Salzburg, 2025)

**Core finding:** First graph-preserving distributed HNSW in disaggregated memory, with formal metric for cache segmentation penalty.

**Key insights:**
- Global HNSW index preserves all edges (no accuracy loss from sharding) across memory nodes via RDMA.
- Cache Segmentation Penalty (CSP) = 1 - CHR/CHR_max quantifies wasted cache capacity from overlapping caches.
- Logical index partitioning via balanced k-means + adaptive query routing reduces CSP from 71% to 32%.
- Relaxed LRU with cooling table and selective admission (1% of base-level nodes) tuned for small local-to-remote latency gap.
- 1.3-1.7x throughput improvement; dramatically better under skewed workloads.

**Relevance to Divergence:**
- CSP metric is directly useful for Divergence's resource accounting -- quantifying cache efficiency across distributed compute nodes.
- Adaptive query routing (monitoring queue lengths, dynamic load balancing) maps to the "Query as Resource Allocation Plan" planner.
- Logical partitioning without data movement validates Divergence's separation of placement decisions from physical layout.
- The disaggregated model (many small-memory compute nodes + large-memory nodes) is a concrete instance of Divergence's resource cloud graph.

---

### 8. Gorgeous: Revisiting Data Layout for Disk-Resident Vector Search
**File:** `2508.15290v1.pdf`
**Authors:** Yin, Yan, Zhou, Li, Li, Zhang, Wang, Yao, Cheng (CUHK/Wuhan/Huawei, 2025)

**Core finding:** Graph-prioritized memory cache and graph-replicated disk blocks improve disk-based ANNS by 78% throughput and 41% lower latency over DiskANN.

**Key insights:**
- **Adjacency lists are more important than vectors** for caching. Caching graph structure alone allows far more nodes' topology to reside in memory.
- Graph-replicated disk block: each node's 4KB block also contains nearest neighbors' adjacency lists, pre-fetching graph structure for subsequent hops.
- Two-stage search: approximate scoring first (PQ in memory), then refinement of only top sigma*D candidates with exact vectors from disk.
- Async block prefetch pipeline with loading/ready queues via libaio.

**Relevance to Divergence:**
- Directly validates the "Object-Based Tiered Architecture" -- routing metadata (adjacency lists) and refine payload (exact vectors) have different access frequencies and should be placed differently.
- The graph-replicated block is a concrete implementation of Divergence's "candidate block" object with co-located routing hints.
- Two-stage search maps exactly to Divergence's Scorer (compressed) -> Refiner (exact) pipeline.
- Memory cache planning (offline profiling to determine optimal PQ compression ratio, cache allocation) is a precursor to Divergence's control plane heat tracking.

---

### 9. CALL: Context-Aware Low-Latency Retrieval in Disk-Based Vector Databases
**File:** `2509.18670v1.pdf`
**Authors:** Jeong, Cho, Park, Kim, Park (Sogang Univ., 2025)

**Core finding:** Query reordering, group-aware prefetching, and load-balanced cluster loading reduce tail latency by 33% and end-to-end latency by 84%.

**Key insights:**
- Queries in a batch share cluster access patterns due to embedding model structure -- exploitable inter-query locality.
- Bitmap-based vectorized Jaccard similarity for efficient query grouping (47% fewer CPU branch misses).
- Group-aware prefetch: at group boundary, preload clusters needed by next group's first query.
- Latency-aware cluster loading: greedy bin-packing assigns variable-size cluster files to threads, avoiding stragglers.
- Cache hit rate from 60% (FIFO) to 92% (CALL) on fever dataset.

**Relevance to Divergence:**
- Validates "Query as Resource Allocation Plan" -- query reordering is a scheduling decision that dramatically affects resource utilization.
- Group-aware prefetch maps to Divergence's planner anticipating data needs across query batches.
- Load-balanced cluster loading is exactly what Divergence's data plane should do with "batched asynchronous IO" and "controlled concurrency."
- The inter-query locality insight informs Divergence's control plane heat tracking.

---

### 10. PageANN: Scalable Disk-Based ANNS with Page-Aligned Graph
**File:** `2509.25487v2.pdf`
**Authors:** Kang, Jiang, Yang, Liu, Li (UT Dallas/Rutgers, 2025)

**Core finding:** Page-node graph aligns logical graph hops with physical SSD page reads, achieving 1.85-10.83x higher throughput with minimal memory (0.05GB for 100M vectors).

**Key insights:**
- Each graph node = one SSD page containing multiple similar vectors + neighbor page IDs + compressed neighbor vectors.
- Self-contained pages: reading one page gives you everything needed for the next hop decision.
- Dynamic memory-disk coordination: adaptively decides what to store in each tier based on memory budget.
- LSH-based lightweight routing index (constant-time entry point computation).
- Near-linear throughput scaling from 1 to 16 threads.

**Relevance to Divergence:**
- The page-node concept is a concrete implementation of Divergence's "Candidate Block" -- a unit of data aligned to hardware IO granularity.
- Memory-disk coordination strategy is exactly Divergence's control plane tier placement decisions.
- Self-contained pages validate Divergence's versioned object model -- each block is independently meaningful.
- The graceful degradation from 30% to 0% memory validates Divergence's "continuous performance-to-cost spectrum."

---

### 11. Vector Search for the Future: From Memory to Cloud-Native
**File:** `2601.01937v1.pdf`
**Authors:** Song, Zhou, Jensen, Xu (HKBU/SJTU/Aalborg, Tutorial 2026)

**Core finding:** Comprehensive survey of vector search evolution across three architectural eras: memory-resident, memory-SSD heterogeneous, and elastic multi-tiered cloud-native.

**Key insights:**
- Unified technique architecture: algorithm layer, computation layer (BLAS/SIMD), I/O layer (batched/async IO, compute-IO overlap).
- Five open challenges for elastic multi-tiered VS:
  1. Tier-aware index co-design
  2. Adaptive and predictive caching
  3. Efficient querying from object storage
  4. Elasticity and auto-scaling
  5. Cost optimization
- Industrial systems: Zilliz Cloud (memory-SSD-object tiering) and TurboPuffer (hot/warm/cold with speculative prefetching).
- Graph-based methods dominate for latency-critical high-accuracy ANN; IVF-based better for dynamic workloads.

**Relevance to Divergence:**
- The five open challenges are precisely what Divergence's architecture aims to solve.
- Validates all five core ideas -- the survey identifies the same gaps Divergence addresses.
- The unified technique architecture (algorithm/computation/IO layers) maps to Divergence's control plane/data plane split.
- Zilliz Cloud and TurboPuffer are the closest existing systems to Divergence's vision, but lack the unified resource scheduling framework.

---

### 12. VeloANN: Optimizing SSD-Resident Graph Indexing for High-Throughput Vector Search
**File:** `2602.22805v1.pdf`
**Authors:** Zhao, Lu, Tian, Zhang, Li, Zhao, Li, Qian (ECNU/ByteDance, PVLDB 2026)

**Core finding:** Holistic co-design of compute-IO scheduling, on-disk layout, and buffer management achieves 5.8x throughput over DiskANN while approaching in-memory performance.

**Key insights:**
- **Implemented in Rust** (~13K lines) -- validates Divergence's language choice.
- Coroutine-based thread-per-core async execution: B = ceil(alpha * I/T) coroutines per thread optimally overlap IO and compute.
- Uses Linux io_uring for high-performance async IO with O_DIRECT bypass.
- Record-level buffer pool (not page-level) with clock-based eviction captures skewed record access patterns.
- ExtRaBitQ 4-bit compression + delta/Elias-Fano adjacency encoding achieves 10x disk space reduction.
- Affinity-based record co-placement: spatially proximate vectors co-located on same page.
- Cache-aware beam search: prioritizes in-memory candidates over slightly-closer on-disk candidates to reduce IO stalls.
- At 10% memory footprint, achieves 92% of fully in-memory throughput.

**Relevance to Divergence:**
- **Closest system to Divergence's data plane vision.** Thread-per-core, coroutines, io_uring, controlled queue depth, Rust implementation.
- Record-level buffer pool vs page-level validates Divergence's "explicit buffer management" over mmap.
- Cache-aware beam search is a form of query-time resource-aware decision making -- choosing to spend less optimal compute to avoid expensive IO.
- The B = ceil(alpha * I/T) formula is a concrete implementation of Divergence's resource-aware planner.
- Affinity-based co-placement maps to Divergence's control plane "block reorganization without global rebuilds."

---

### 13. Tiered-Indexing: Optimizing Access Methods for Skew
**File:** `778_2025_Article_928.pdf`
**Authors:** Zhou, Hao, Yu, Stonebraker (MIT/Wisconsin, VLDB Journal 2025)

**Core finding:** General framework for decomposing any index structure into hot/cold tiers with record-level migration, achieving 3.7-4.5x improvement for B+tree, hash, heap, and LSM-tree under skew.

**Key insights:**
- Granularity mismatch: page-level caching vs record-level access creates "lukewarm" pages wasting memory.
- 2-Tier designs with shared structure: 2-Hash uses shared hash function so eviction produces sequential IO; 2B+tree walks hot tree in key order.
- BiLSM-Tree: bidirectional migration with exponential sampling rate, early migration during flush, adaptive control via hill-climbing.
- Optimistic locking with version numbers scales for read-heavy workloads (1.78x over pessimistic at 32 threads).
- Inclusive caching (data in both tiers) outperforms exclusive by 70% for read-only workloads.

**Relevance to Divergence:**
- Validates "Object-Based Tiered Architecture" -- record-level (object-level) tiering is fundamentally superior to page-level tiering under skew.
- Adaptive migration control (hill-climbing on cache miss rate) maps to Divergence's control plane heat tracking and tier placement decisions.
- The "lukewarm page" problem directly motivates Divergence's object-based decomposition -- routing metadata, candidate blocks, compressed codes, and refine payload should be independently tierable.
- Optimistic locking with version numbers is a concrete concurrency control technique for Divergence's lock-sharded structures.

---

### 14. SmartANNS: Scalable Billion-point ANNS Using SmartSSDs
**File:** `atc24-tian.pdf`
**Authors:** Tian, Liu, Duan, Liao, Jin, Zhang (HUST, ATC 2024)

**Core finding:** Host CPU + SmartSSD cooperative architecture achieves 10.7x QPS improvement over naive SmartSSD approaches for billion-scale ANNS.

**Key insights:**
- Hierarchical indexing: host CPU maintains shard centroids in memory, narrows search space, then dispatches to SmartSSD FPGAs.
- Learning-based shard pruning: GBDT model predicts minimum shards per query (~1MB model, microsecond inference).
- Hotness-aware data layout + dynamic task scheduling considering data locality and load balance.
- FPGA search engine with data pool/kernel pool mechanism overlapping shard loading with searching.
- Near-linear scalability with number of SmartSSDs (3.76x with 4 devices).
- SmartSSDs free host CPU/memory/PCIe bandwidth for co-located workloads.

**Relevance to Divergence:**
- SmartSSDs are a concrete instance of Divergence's "Resource Cloud" -- storage with compute capability, measurable latency/bandwidth/cost.
- Host-device cooperative processing is exactly Divergence's "execution plan across heterogeneous hardware resources."
- The GBDT shard pruner is a learned "Pruner" stage in Divergence's pipeline.
- Task scheduling (locality + load balance) maps directly to Divergence's planner deciding "which storage tiers to access."

---

## Thematic Analysis: Mapping to Divergence's Core Ideas

### Idea 1: Hardware as Resource Clouds

**Strong validation.** The literature consistently shows that treating hardware as interchangeable commodities fails:

| Paper | Key Evidence |
|-------|-------------|
| IISWC Benchmark | CPU is bottleneck, not SSD; SSDs at 8.9% utilization |
| d-HNSW | Compute pool vs memory pool as distinct resources |
| CoTra | Memory bandwidth as fundamental bottleneck; RDMA NIC IOPS limits |
| SHINE | Local-remote memory latency gap (10-100x) demands cache design tuned to hardware |
| SmartANNS | SmartSSD = storage + compute resource with its own latency/throughput profile |
| VeloANN | io_uring, O_DIRECT, thread-per-core -- all hardware-aware choices |

**Implications for Divergence:**
- Resource graphs must include: CPU cores, DRAM capacity/bandwidth, NVMe SSD IOPS/bandwidth, RDMA NIC IOPS/bandwidth, GPU compute/memory, computational storage processing power.
- Each resource type has distinct concurrency limits, latency characteristics, and cost profiles.
- The IISWC finding that CPU (not SSD) is the bottleneck is critical -- Divergence's planner must model CPU cost per IO operation.

### Idea 2: Object-Based Tiered Architecture

**Strong validation.** Every paper dealing with disk-based search confirms the value of decomposing index structures into independently tierable objects:

| Paper | Object Decomposition Evidence |
|-------|------------------------------|
| Gorgeous | Adjacency lists vs vectors have different access frequencies; caching graph structure alone is optimal |
| PageANN | Self-contained page-nodes with vectors + neighbor IDs + compressed neighbor vectors |
| VeloANN | Record-level (not page-level) buffer pool captures skewed access patterns |
| Tiered-Indexing | Granularity mismatch (page vs record) wastes memory; record-level tiering is superior |
| ZipCache | Different object sizes (tiny/medium/large) need different handling strategies |
| Survey | Object storage as source of truth with SSD/memory as adaptive working sets |

**Implications for Divergence:**
- Routing metadata should always be in the fastest available tier (DRAM or L3 cache).
- Compressed codes (PQ/RaBitQ) belong in DRAM as a compact representation for approximate scoring.
- Candidate blocks (adjacency lists, posting lists) have higher access frequency than vectors and should be prioritized for DRAM caching.
- Refine payload (exact vectors) needed only for final reranking of a small subset.
- Object-level eviction/promotion decisions must be independent per object type.

### Idea 3: Query as a Resource Allocation Plan

**Strong validation.** Multiple papers demonstrate that treating queries as scheduling problems yields dramatic improvements:

| Paper | Scheduling Evidence |
|-------|-------------------|
| CALL | Query reordering + group-aware prefetch: 84% latency reduction |
| CoTra | Primary/secondary partition classification per query |
| VeloANN | Cache-aware beam search: choose in-memory candidate over slightly-closer on-disk candidate |
| Gorgeous | Two-stage search: defer exact computation to refinement of small subset |
| SHINE | Adaptive query routing: monitor queue lengths, dynamically balance load |
| SmartANNS | GBDT predicts minimum shards per query; task scheduler considers locality + load balance |

**Implications for Divergence:**
- Query planner should decide: (1) which partitions to touch, (2) which tier to read from, (3) how much parallelism, (4) when to terminate early, (5) which candidates to refine.
- Batch-level planning (CALL, d-HNSW) offers additional optimization opportunities.
- Learned models (GATE's two-tower, SmartANNS's GBDT) can make routing/pruning decisions.
- The planner should account for current resource state (queue lengths, cache contents, IO pipeline depth).

### Idea 4: Index-Agnostic Execution Model

**Strong validation.** The literature shows that retrieval is decomposable into generic stages regardless of index type:

| Stage | IVF-Based | Graph-Based | Evidence Paper |
|-------|-----------|-------------|----------------|
| Router | Centroid search | Entry point selection | GATE, d-HNSW, SmartANNS |
| Candidate Producer | Posting list scan | Graph traversal | PageANN, VeloANN, Gorgeous |
| Pruner | Adaptive nprobe | Early termination | SmartANNS (GBDT), CALL |
| Scorer | PQ distance | Compressed distance | Gorgeous (PQ), VeloANN (ExtRaBitQ) |
| Refiner | Exact rerank | Exact rerank | Gorgeous (two-stage), VeloANN |

**Implications for Divergence:**
- The five-stage pipeline (Router -> Candidate Producer -> Pruner -> Scorer -> Refiner) is validated across all index families.
- Each stage can be backed by different implementations: learned (GATE), hierarchical (d-HNSW), hash-based (PageANN).
- Stages can be independently placed on different hardware resources (Router on CPU, Scorer on GPU, etc.).
- Composability enables mixing: e.g., IVF routing + graph-based candidate production + GPU scoring.

### Idea 5: Multi-Tenant Isolation via Resource Units

**Moderate validation.** Curator is the only paper directly addressing multi-tenancy in vector search, but resource isolation themes appear throughout:

| Paper | Isolation/Fairness Evidence |
|-------|---------------------------|
| Curator | Per-tenant TCTs with Bloom filters; stable latency regardless of other tenants' data growth |
| SHINE | Adaptive query routing prevents individual CNs from being overloaded |
| CALL | Fair cache utilization across query groups |
| CoTra | Per-query resource allocation via primary/secondary classification |
| IISWC Benchmark | Different databases handle concurrency scaling very differently (multi-tenant implications) |

**Implications for Divergence:**
- Unified resource unit model (bytes read, CPU cycles, GPU time, network bandwidth) allows per-tenant quotas.
- TCT-like structures can provide tenant-aware routing without per-tenant index replication.
- Adaptive routing and queue-length monitoring enable noisy-neighbor prevention.
- The continuous performance-to-cost spectrum (ultra/balanced/cost-optimized) can be enforced per-tenant.

---

## Key Architectural Takeaways for Divergence

### Data Plane Design (informed by VeloANN, CoTra, Gorgeous)
1. **Thread-per-core with coroutines** -- validated by VeloANN (Rust, io_uring) and CoTra (C++ coroutines).
2. **Record-level buffer management** over page-level or mmap -- validated by VeloANN and Tiered-Indexing.
3. **Batch-size formula**: B = ceil(alpha * I/T) coroutines per thread to optimally overlap IO and compute.
4. **Cache-aware execution**: prefer in-memory candidates over slightly-better on-disk candidates.
5. **Two-stage scoring**: compressed approximate distances first, then exact refinement of top-k subset.

### Control Plane Design (informed by CALL, SHINE, SmartANNS, Tiered-Indexing)
1. **Heat tracking**: monitor per-object access frequency for tier placement.
2. **Adaptive migration**: record-level promotion/demotion with hill-climbing or exponential sampling.
3. **CSP monitoring**: track cache segmentation penalty across distributed nodes.
4. **Block reorganization**: affinity-based co-placement without global index rebuilds (VeloANN).
5. **Distribution shift detection**: adaptive control mechanisms (BiLSM-Tree's linear regression).

### Storage Model (informed by PageANN, Gorgeous, Survey, ZipCache)
1. **Object-aligned storage**: routing metadata, compressed codes, candidate blocks, refine payload as independent versioned objects.
2. **SSD page alignment**: each logical block maps to one physical page read.
3. **Graph-replicated blocks**: co-locate neighbor routing hints within each block for prefetching.
4. **Compression hierarchy**: 1-bit binary codes (in memory) -> 4-bit extended codes (on SSD) -> exact vectors (cold tier).

### Query Planning (informed by CoTra, CALL, GATE, SmartANNS)
1. **Per-query resource budgets**: planner determines partitions, tiers, parallelism, and termination conditions.
2. **Batch-level optimization**: group similar queries for cache/prefetch efficiency.
3. **Learned routing**: lightweight models (GATE two-tower, SmartANNS GBDT) for entry point selection and shard pruning.
4. **Primary/secondary execution modes**: spend local compute for primary partitions, minimal network for secondary.

---

## Gaps and Open Questions

1. **No paper combines all five ideas.** Each paper addresses 1-3 of Divergence's core concepts. The integration is novel.
2. **GPU integration in tiered systems** is largely unexplored. Only the survey mentions it as future work.
3. **Object storage as source of truth** is discussed only by the survey (Zilliz Cloud, TurboPuffer) but with limited technical depth.
4. **Cross-workload resource sharing** (vector search + scalar queries + writes) within a unified resource model is unstudied.
5. **Dynamic index reconfiguration** (switching between IVF and graph-based on the fly based on workload) has no existing solution.
6. **Cost modeling** for the continuous performance-to-cost spectrum lacks formal treatment.
