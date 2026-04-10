# EXP-PIPELINED-REFINE: Overlap Vector IO with Graph Traversal

**Date**: 2026-03-11
**Status**: NEGATIVE RESULT — modest overlap gain, but net slower than baseline

## Hypothesis

During two-stage search (SAQ traversal → FP16 disk refine), all 160 vector reads happen AFTER traversal completes. At R=160 FP16 (dim=768), that's 160 × 1.5KB = 240KB of demand-waited IO taking ~3-5ms. If we speculatively issue vector reads for VIDs entering top-R during traversal, most reads complete before traversal ends, eliminating the idle gap between stages.

Expected payoff: ~1-2ms p50 improvement with total QD held constant.

## Design

### Architecture: callback hook in existing traversal loop

Added `Option<&mut RefinePrefetcher>` parameter to `disk_graph_search_pipe_v3_inner`. After each `nearest.push(scored)`, the prefetcher is notified. Zero overhead when `None`.

### RefinePrefetcher

- `HashMap<u32, JoinHandle<Result<(f32, usize)>>>` for inflight reads
- `HashMap<u32, (f32, usize)>` for completed results
- `FixedCapacityHeap` (capacity=R) mirrors top-R to decide what to prefetch
- `max_prefetches = 1.5 * refine_r` — hard cap on total spawns (bounds waste from top-R churn)
- `expansion_warmup` — skip prefetching during first N expansions

### IO budget: total QD constant

Vec prefetch tasks acquire `GlobalIoBudget` tokens (same pool as adjacency reads) via RAII drop guard. Plus `vec_pf_sem` (LocalSemaphore) caps concurrent vec prefetches within a core. For the experiment, adj_pf reduced from 4→2 when pipelined, so total device QD is constant.

### Two-pass refine

1. For each candidate in final top-R: check `ready` → await `inflight` → else fresh read
2. Batch remaining misses (same as existing refine pattern)
3. Quiesce: await ALL remaining inflight handles (even wasted ones) before next query

### No loop duplication

The full traversal loop is NOT copied. A single `on_candidate_pushed()` callback is added behind `Option`, keeping the existing v3_inner as the single source of truth.

## Setup

- **Dataset**: Cohere 100K, dim=768, cosine, k=100
- **Proxy**: SAQ (eqseg16, unpacked), ef=200
- **Refine**: FP16 disk reads, R=160, refine_inflight=16
- **Cache**: 5% (256 pages), per-query cold (cache cleared between queries)
- **Hardware**: i4i.2xlarge NVMe, direct_io=true
- **Configs swept**: adj_pf ∈ {2,4}, vec_pf_budget ∈ {4,8}, warmup ∈ {0,100}

## Results

| Config | recall | p50ms | p99ms | QPS | ref_ms | vpf_issued/q | vpf_hits/q | vpf_wasted/q | vpf_hit% |
|--------|--------|-------|-------|-----|--------|-------------|-----------|-------------|----------|
| baseline-adjpf4 | 0.963 | 11.2 | 13.3 | 81.8 | 3.78 | 0 | 0 | 0 | - |
| baseline-adjpf2 | 0.963 | 14.6 | 17.3 | 63.8 | 5.21 | 0 | 0 | 0 | - |
| pipe-vpf4-w0 | 0.963 | 13.6 | 16.6 | 68.0 | 3.29 | 240 | 54 | 186 | 33.6% |
| pipe-vpf4-w100 | 0.963 | 14.3 | 16.2 | 65.7 | 4.69 | 16 | 14 | 1 | 8.9% |
| pipe-vpf8-w0 | 0.963 | 13.4 | 16.4 | 68.8 | 3.20 | 240 | 54 | 186 | 33.6% |
| pipe-vpf8-w100 | 0.963 | 14.3 | 16.1 | 65.9 | 4.68 | 16 | 14 | 1 | 8.9% |

**vpf_inflight_at_end**: 240.0 (w=0) and 15.7 (w=100) — nearly all prefetches still pending when traversal ends.

## Analysis

### Recall unchanged (0.963) — correctness verified

### Pipelining wins vs iso-adj-budget baseline (adjpf2)

Best pipelined (vpf8-w0): 14.6→13.4ms p50 (-8.2%), refine_ms 5.21→3.20ms (-38%). This is real overlap benefit under constant adj budget.

### But loses vs adjpf4 baseline

Best pipelined (13.4ms) is still 2.2ms slower than adjpf4 baseline (11.2ms). Reducing adj_pf from 4→2 to make QD room for vec prefetch costs more than the overlap saves. The adj_pf=4→2 penalty (3.4ms) exceeds the refine overlap gain (1.2ms).

### Low hit rate (33.6%)

Only 54/160 refine reads were prefetched and used. Top-R churns heavily during 200 expansions — 240 prefetches issued but 186 wasted (VIDs evicted from top-R before traversal ended).

### All prefetches still inflight at traversal end (vpf_inflight_at_end=240)

Vector IO didn't actually complete during traversal. The NVMe device was saturated by adjacency reads; vec prefetches got queued behind them. The "overlap" is mostly just reordering when IO happens, not true parallelism.

### warmup=100 barely helps

Only 16 prefetches issued (late in traversal), 88% hit rate but tiny absolute count (14 hits). Not enough to move the needle.

### vpf=4 vs vpf=8: no difference

The bottleneck is device QD contention, not local concurrency.

## Root Cause

Under constant total QD, vec prefetch competes with adjacency reads for the same NVMe bandwidth. With ~200 serial adjacency reads per query consuming most device capacity, adding speculative vector reads mostly just delays adjacency reads without completing before traversal ends.

The fundamental constraint: adjacency reads are **serial** (each expansion depends on the previous hop's neighbors), while vector reads are **bulk-parallelizable** but only useful after traversal identifies which VIDs to refine. Moving vector reads earlier doesn't help when the device is already saturated.

## Verdict

**Dead end for this workload.** Pipelined refine adds complexity without net benefit because:
1. Reducing adj_pf to make QD room costs more than the overlap saves
2. Device is saturated by adj reads — vec prefetches queue behind them
3. Top-R churn wastes 78% of prefetches (186/240)

Would only help if: (a) device has QD headroom (multi-device, higher QD sweet spot), or (b) traversal takes many more hops giving vec prefetches time to complete, or (c) adj reads become significantly fewer (better graph, fewer hops).

## Code Map

| File | What |
|------|------|
| `crates/engine/src/search.rs` | `RefinePrefetcher` struct, `on_candidate_pushed()`, `disk_graph_search_pipe_v3_refine_fp16_pipelined()` |
| `crates/engine/src/perf.rs` | `vec_prefetch_issued/hits/wasted/inflight_at_end` fields |
| `crates/engine/src/io.rs` | `IoDriver::global_budget()` getter |
| `crates/engine/tests/disk_search.rs` | `exp_pipelined_refine` test |
