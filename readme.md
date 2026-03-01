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