# Project Divergence

Project Divergence is a single-node, NVMe-native vector search engine.

Modern NVMe devices are fast.  
Most vector databases fail to utilize them.

Divergence explicitly controls I/O scheduling, buffer management, and traversal execution to achieve predictable tail latency under limited memory.

---

## What It Is

- Graph-based ANN engine
- NVMe-backed adjacency blocks
- Coroutine-based execution
- Thread-per-core runtime
- Record-level caching (no `mmap`)
- Two-stage search (compressed → refine)

---

## What Makes It Different

- No reliance on OS page cache
- Explicit `io_uring` scheduling
- Adjacency prioritized over vector caching
- Graceful performance scaling with memory
- Predictable p99 under load

---

## Design Contracts

Five invariants that define the system. Each is bounded, measurable, and has a fallback.

**1. Flat NSW with bounded entry routing**
Single-layer graph. No hierarchy. A small entry set (32–256 nodes, selected offline by coarse centroid routing) bounds the expansion before graph traversal begins. Fallback: random entry still works, just slower. Measurable: entry set size and average path length to target neighborhood.

**2. Budgeted async IO**
All NVMe IO goes through io_uring with strict per-core `max_in_flight_reads` and per-query `max_nvme_blocks` budgets. Adjacency reads and refine reads draw from separate quotas — refine IO cannot starve traversal. Backpressure: when budget is exhausted, the coroutine yields until a slot opens. Measurable: per-query IO count, per-core queue depth, SSD utilization %.

**3. Object-typed buffer pool with pinned regions**
Three logical pools: adjacency blocks (highest priority), compressed codes, exact vectors (lowest priority). A pinned region guarantees navigation data stays resident under any load. The adaptive region uses clock eviction. Cold traffic cannot evict hot navigation. Measurable: per-pool hit ratio, pinned region residency %, eviction rate.

**4. Graph-replicated disk blocks — immutable, bounded, with fallback**
Each 4KB block contains one node's adjacency + vector + up to K neighbor adjacency lists. K is a fixed budget per block (not a packing optimization). Replication exists only in immutable segments built offline. The system runs without replication (plain adjacency + vector blocks) at reduced throughput. Measurable: replication factor K, bytes wasted per block, fallback throughput ratio.

**5. Early termination with recall guardrail**
Search stops early only when a margin-based bound is satisfied: the distance gap between the k-th and (k+1)-th candidate exceeds a threshold for multiple consecutive iterations. Default: off. Enabled only when a target recall level is configured, and the threshold is calibrated against a held-out validation set. Measurable: recall@k with and without termination, % of queries terminated early, threshold stability across datasets.

---

## Current Scope

Single-node only.

Focused on:
- NVMe utilization
- CPU–I/O overlap
- Stable execution under memory pressure

Not focused on:
- Distributed execution
- Embedding generation
- Learned routing
- Cloud storage integration

---

## Roadmap

**Phase 1**  
NVMe-backed graph traversal with coroutine scheduler

**Phase 2**  
Cache-aware beam search and I/O batching

**Phase 3**  
Block reorganization and layout optimization

---

Early development.