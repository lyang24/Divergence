# Project Divergence

Project Divergence is a hardware-native, resource-scheduled retrieval engine designed for large-scale vector and hybrid search workloads.

It is built around a single idea:

Queries are not executed against a fixed index structure.  
They are planned and scheduled across a graph of heterogeneous hardware resources.

Project Divergence treats storage, memory, compute, and network as first-class schedulable resource clouds. It provides predictable latency, multi-tenant isolation, and a continuous performance-to-cost spectrum without binding the system to a single ANN algorithm.

---

## Motivation

Modern vector databases tend to share several structural limitations:

1. Static global index structures that are expensive to rebuild when data distribution shifts.
2. Heavy reliance on mmap and OS page cache, leading to unpredictable tail latency and weak multi-tenant isolation.
3. Architecture optimized for one dominant storage or locality pattern.
4. Limited control over cost versus performance trade-offs.
5. Tight coupling between index structure and execution strategy.

As workloads become multi-tenant, distribution shifts become frequent, and hardware becomes heterogeneous, these assumptions break down.

Project Divergence is built for:

- Dynamic data distributions
- Tiered storage environments
- Heterogeneous hardware fleets
- Strict latency targets
- Cost-sensitive deployments
- Multi-tenant isolation

---

## Core Philosophy

### 1. Hardware as Resource Clouds

Every hardware component is modeled as a resource cloud with measurable properties:

- Latency
- Bandwidth
- Concurrency capacity
- Cost characteristics
- Sharing and isolation capabilities

Examples include:

- CPU Cloud
- DRAM Cloud
- NVMe Cloud
- Remote Block Storage Cloud
- Object Storage Cloud
- GPU Cloud
- Network Cloud

Execution decisions are made against this resource graph rather than against hardcoded tier assumptions.

---

### 2. Object-Based Tiered Architecture

All index and data structures are decomposed into versioned objects:

- Routing metadata
- Candidate blocks
- Compressed codes
- Refine payload
- Delta segments
- Manifest metadata

Object storage is the source of truth.

Higher-performance tiers act as adaptive working sets and may be rebuilt, migrated, or evicted without affecting correctness.

---

### 3. Query as a Resource Allocation Plan

A query is transformed into an execution plan under constraints:

- Latency budget
- Resource budget
- Tenant quota
- Hardware availability
- Data residency

The planner determines:

- Which partitions or blocks to touch
- Which storage tiers to access
- Whether to use CPU, GPU, or remote resources
- How much parallelism to apply
- When to degrade or terminate early

This turns retrieval into a scheduling problem rather than a fixed algorithm invocation.

---

### 4. Index-Agnostic Execution Model

Project Divergence is not bound to HNSW, IVF, DiskANN, or any specific ANN structure.

Instead, it decomposes retrieval into composable stages:

1. Router  
   Determines candidate partitions or blocks to inspect.

2. Candidate Producer  
   Generates candidate identifiers or blocks.

3. Pruner  
   Applies block-level or code-level pruning to eliminate unnecessary work.

4. Scorer  
   Computes similarity using CPU SIMD or GPU acceleration.

5. Refiner  
   Performs high-precision reranking using slower tiers when necessary.

Any ANN method can be implemented as a specific combination of these stages.

---

### 5. Multi-Tenant Isolation via Resource Units

Project Divergence introduces a unified retrieval resource unit model.

Resource consumption includes:

- Bytes read from DRAM
- Bytes read from NVMe
- Object storage operations
- CPU cycles
- GPU time
- Network bandwidth

Each tenant receives resource quotas and rate limits.

The scheduler enforces:

- No noisy neighbor interference
- Predictable tail latency
- Controlled cost exposure
- Graceful degradation when limits are reached

---

## Architecture Overview

### Control Plane

Responsible for:

- Statistics collection
- Heat tracking
- Resource accounting
- Tier placement decisions
- Background compaction
- Version management

### Data Plane

Responsible for:

- Execution plan evaluation
- Batched asynchronous IO
- Controlled concurrency
- SIMD scoring
- Optional GPU offload
- Early termination

The data plane is intentionally minimal and deterministic.

---

## Storage Model

Project Divergence uses a versioned manifest system.

- Object storage is authoritative.
- Tiered caches are ephemeral.
- Index structures are locally evolvable.
- Block reorganization can occur without global rebuilds.

This enables:

- Elastic scaling
- Hotset reshaping
- Online reconfiguration
- Rapid rollback

---

## Execution Flow

1. Plan  
   Build an execution DAG based on resource state and constraints.

2. Prune  
   Eliminate unnecessary blocks and tiers before heavy computation.

3. Fetch  
   Perform batched, controlled IO from appropriate tiers.

4. Score  
   Compute similarities using CPU or GPU.

5. Refine  
   Retrieve high-precision data for final ranking.

6. Enforce  
   Check resource budgets and apply degradation policies if required.

---

## Performance and Cost Profiles

Project Divergence supports a continuous performance-to-cost spectrum.

Ultra Performance:
- Full DRAM residency
- Aggressive prefetch
- GPU-enabled rerank
- No cold-tier reads

Balanced:
- Hot data in DRAM
- Warm data in NVMe
- Selective GPU use

Cost Optimized:
- Small DRAM hotset
- Limited NVMe
- Cold-tier allowed
- Strict resource budgets

All profiles share the same execution engine.

---

## Implementation Principles

- Rust-native implementation
- Explicit buffer management
- Asynchronous IO runtime
- Controlled queue depth
- Lock-sharded internal structures
- Pluggable index components
- Resource-aware planner

---

## Long-Term Vision

Project Divergence is not a faster vector database.

It is a hardware-native retrieval execution foundation.

Its goal is to unify:

- Representation
- Indexing
- Storage
- Compute
- Network
- Multi-tenancy

into a single schedulable resource system.

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
