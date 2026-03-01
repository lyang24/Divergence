# Project Divergence

Project Divergence is a hardware-native, resource-scheduled retrieval engine for vector and hybrid search workloads.

It is built on a single principle:

Retrieval is not a fixed index operation.  
It is a planned execution over a graph of heterogeneous hardware resources.

Divergence treats storage, memory, compute, and network as schedulable resource clouds.  
It provides predictable latency, multi-tenant isolation, and a continuous performance-to-cost spectrum without binding the system to a single ANN structure.

---

## Motivation

Modern vector databases typically assume:

- A static global index structure
- A dominant storage tier
- OS page cache as implicit caching policy
- One primary execution strategy

These assumptions break under:

- Shifting data distributions
- Heterogeneous hardware fleets
- Multi-tenant interference
- Strict latency targets
- Cost-sensitive deployments

Project Divergence is designed for dynamic environments where:

- Data layout evolves
- Hardware capacity changes
- Tenants have distinct resource budgets
- Performance and cost must be explicitly controlled

---

## Core Principles

### 1. Hardware as Resource Clouds

Each hardware type is modeled as a resource cloud defined by measurable properties:

- Latency
- Bandwidth
- Concurrency capacity
- Cost characteristics
- Isolation capability

Examples include:

- CPU Cloud
- DRAM Cloud
- NVMe Cloud
- Remote Block Storage Cloud
- Object Storage Cloud
- GPU Cloud
- Network Cloud

Execution decisions are made against this resource graph, not against hardcoded tier assumptions.

---

### 2. Object-Based Tiered Architecture

All retrieval structures are decomposed into versioned objects:

- Routing metadata
- Candidate blocks
- Compressed representations
- Refine payload
- Delta segments
- Manifest metadata

Object storage acts as the source of truth.

Higher-performance tiers function as adaptive working sets and may be rebuilt, migrated, or evicted without affecting correctness.

---

### 3. Query as an Execution Plan

Each query is compiled into an execution plan under explicit constraints:

- Latency budget
- Resource budget
- Tenant quota
- Hardware state
- Data residency

The planner determines:

- Which partitions or blocks to inspect
- Which tiers to access
- How much parallelism to apply
- Whether to use CPU, GPU, or remote resources
- When to degrade or terminate early

Retrieval becomes a scheduling problem rather than a fixed algorithm call.

---

### 4. Index-Agnostic Execution Model

Divergence does not bind itself to HNSW, IVF, DiskANN, or any specific ANN method.

Instead, retrieval is decomposed into composable stages:

1. Router  
   Identifies relevant partitions or blocks.

2. Candidate Producer  
   Generates candidate identifiers or block references.

3. Pruner  
   Eliminates unnecessary work using block-level or representation-level bounds.

4. Scorer  
   Computes similarity using CPU SIMD or GPU acceleration.

5. Refiner  
   Performs high-precision reranking using slower tiers when required.

Different ANN techniques become specific implementations of these stages.

---

### 5. Search RU: Unified Resource Accounting

Project Divergence introduces a unified retrieval resource unit model called Search RU.

Search RU accounts for:

- DRAM bytes accessed
- NVMe bytes read
- Object storage operations
- CPU cycles
- GPU time
- Network bandwidth

Each tenant is allocated RU quotas and rate limits.

The planner enforces RU budgets as hard constraints and applies controlled degradation policies when limits are reached.

This enables:

- Predictable multi-tenant isolation
- Explicit cost control
- Stable tail latency under load

---

## Architecture Overview

### Control Plane

Responsible for:

- Resource accounting
- Heat tracking
- Tier placement policies
- Statistics collection
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

The data plane is designed to remain minimal and deterministic.

---

## Storage Model

Divergence uses a versioned manifest system.

- Object storage is authoritative.
- Higher tiers are adaptive caches.
- Index blocks are locally evolvable.
- Reorganization can occur without global rebuilds.

This enables:

- Elastic scaling
- Online reconfiguration
- Hotset reshaping
- Safe rollback

---

## Performance and Cost Profiles

Divergence supports a continuous performance-to-cost spectrum.

High Performance:
- Large DRAM residency
- Aggressive prefetch
- GPU-enabled scoring
- No cold-tier access

Balanced:
- Hot data in DRAM
- Warm data in NVMe
- Selective GPU usage

Cost Optimized:
- Small hotset
- Limited warm tier
- Cold-tier access allowed
- Strict RU budgets

All profiles share the same execution engine.

---

## Implementation Principles

- Rust-native implementation
- Explicit buffer management
- Asynchronous IO runtime
- Controlled queue depth
- Lock-sharded internal structures
- Pluggable retrieval stages
- Resource-aware planning

---

## Roadmap

Phase 1:
- Resource graph abstraction
- Object store interface
- Execution planner prototype
- Single-node NVMe tier prototype

Phase 2:
- Search RU enforcement
- Multi-tenant quota management
- Hybrid retrieval support

Phase 3:
- Optional GPU integration
- Distributed resource graph
- Remote block access

---

## Non-Goals

- Embedding generation
- End-to-end RAG application stack
- Training infrastructure

Project Divergence focuses exclusively on the retrieval execution layer.

---

## Status

Early architecture stage.

Contributions and design discussions are welcome.
