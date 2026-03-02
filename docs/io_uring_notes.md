# io_uring + Thread-Per-Core: Research & Implementation Notes

Distilled from research papers (~/Documents/divergence2/) and reference
implementations (monoio, liburing, io-uring, glommio). Only information
directly relevant to building an NVMe-native vector search engine.

**Status: DRAFT — more references pending**

---

## Part I: Research Papers

### 1. io_uring Fundamentals

Two shared-memory ring buffers between user and kernel:
- Submission Queue (SQ): app enqueues IO requests
- Completion Queue (CQ): kernel posts results
- Single `io_uring_enter()` syscall can submit a batch and reap completions

Three execution paths:
1. Inline completion — immediate for fast/cached reads
2. Async via poll set — event handler for non-blocking ops
3. Worker thread fallback — for blocking ops (~7.3µs overhead)

**Critical: naive io_uring = only 1.06x over libaio.** Gains require tuning:

| Optimization | Effect | Kernel |
|---|---|---|
| Registered/fixed buffers | Eliminates per-request page pinning, +4-6% | 5.1+ |
| Batch ≥16 SQ submissions | 3-6x CPU cost reduction per op | any |
| SQPoll mode | Kernel thread polls SQ, eliminates submission syscalls | 5.4+ |
| IOPoll (completion polling) | 1.7x throughput at low QD | 5.1+ |
| DEFER_TASKRUN | Processes completions only on io_uring_enter() | 6.1+ |
| SINGLE_ISSUER hint | Enables single-threaded optimizations | 6.0+ |
| COOP_TASKRUN | Avoids IRQ for task interrupts | 5.19+ |

With all optimizations: **2.05-5x speedup** (buffer manager: 2.05x, network: 5x).

### 2. NVMe Hardware Numbers

Reference points — actual numbers depend on device, queue depth, block size, kernel version, and workload mix.

| Metric | Value |
|---|---|
| Kioxia CM7-R IOPS | 2.45M |
| Optane 4KB random read | ~7.3µs device, ~12.8µs total (vanilla Linux) |
| Samsung 980 Pro throughput | 7 GB/s over 4 PCIe 4.0 lanes |
| NVMe density vs DRAM | 30x cheaper per byte |

Latency breakdown (Optane, 4KB random read):
- Device IO: ~7.3µs
- Kernel stack overhead: ~5.5µs (vanilla) → ~2.8µs (optimized AIOS)
- Total: ~12.8µs (vanilla) → ~10.1µs (21% reduction)

### 3. AIOS: Async IO Stack Optimization (ATC 2019)

Kernel IO stack overhead is non-negligible at sub-10µs device latency.

Lightweight Block IO Layer (LBIO):
- Single memory object per request (vs bio + request + iod in vanilla)
- Overlaps page allocation with device IO
- Lazy page cache indexing: delay insertion until after IO completes

Results: 15-33% latency reduction, 11-44% IOPS increase on RocksDB.

### 4. Coroutine-Based Async IO (ADMS 2022)

Coroutines achieve optimal SSD throughput with **16x fewer threads** than sync IO.

| Configuration | Throughput | Threads needed |
|---|---|---|
| DRAM (sync) | 152 GB/s | ≥25 |
| SSD (sync) | 50 GB/s | ≥4 |
| SSD (async + coroutines) | 50 GB/s | ~1 effective |

- IO depth 64 per thread for best throughput
- Stackless coroutines — minimal overhead per suspension
- Out-of-core index joins: **up to 60% improvement**, still 25% faster at 80% cache hit

### 5. CPU Cycle Accounting

Per-operation costs (3.7 GHz CPU):

| Operation | Cycles |
|---|---|
| Single SQ submission + spinlock | ~1,000 |
| Single CQ poll + error check | ~500 |
| Memory copy (unfixed buffers, 4KB) | ~2,000 |
| **Total single read** | **~3,500-5,500** |

Batch read (16 ops):
- Submission: ~2,000 total → ~125/op
- Reaping: ~1,000 total → ~63/op
- **Per-op overhead: ~188 cycles** (3-6x reduction vs single)

Throughput ceilings (theoretical, pure-IO with batched submission, no compute):
- Per core: ~800k IOPS (CPU-bound, assumes batch≥16 and no distance computation)
- Hardware: ~2.5M IOPS (7GB/s SSD ÷ 4KB)
- 4 async threads: 2-4M IOPS

### 6. Graph Traversal IO Patterns

Queue depth analysis:
- QD=1: IO-bound, thread blocks on single read
- QD=4-8: modest parallelism
- QD=16-32: effective for 2-4 threads
- QD=64: optimal for single high-concurrency thread

Graph traversal:
- Random 4KB reads to sparse graph nodes
- Multiple candidates = natural batching opportunity
- Queue 4-16 adjacency reads before blocking
- Process distances while IO in flight

---

## Part II: Reference Implementations

### 7. monoio Internals (bytedance/monoio 0.2.4)

Key source files:
- `monoio/src/runtime.rs` — executor loop
- `monoio/src/driver/uring/mod.rs` — io_uring driver
- `monoio/src/buf/io_buf.rs` — IoBuf/IoBufMut traits
- `monoio/src/fs/open_options.rs` — file opening
- `monoio/src/driver/op/read.rs` — read_at opcode generation

#### 7.1 Runtime Loop

The executor loop in `runtime.rs` (lines 154-183):
```
loop {
    1. Pop tasks from local TaskQueue (VecDeque, capacity 4096)
    2. Run each task (poll future once)
    3. Anti-starvation: max_round = tasks.len() * 2
    4. If tasks remain: driver.submit() to flush accumulated SQEs
    5. If no tasks: driver.park() → submit_and_wait(1) → tick() to harvest CQEs
}
```

SQE submission is **lazy**: `submit_with_data()` just pushes to the SQ ring
(no syscall). Actual `io_uring_enter()` only happens on `park()` or when SQ is full.

CQE harvesting in `UringInner::tick()` (lines 376-396):
```rust
let cq = self.uring.completion();
for cqe in cq {  // drains entire CQ in one loop
    let index = cqe.user_data();
    self.ops.complete(index, resultify(&cqe), cqe.flags());
}
```

Single-loop drain — efficient. No batch-peek API, but iterates until empty.

#### 7.2 What monoio supports / doesn't support

| Feature | Status |
|---|---|
| O_DIRECT via custom_flags() | YES |
| 4KB-aligned user buffers | YES (user provides, no hidden alloc) |
| Zero-copy (buffer ownership transfer) | YES — `read_at(buf, pos)` moves buf in, returns it |
| Registered/fixed buffers | NO — not exposed |
| SQPOLL | NO — not configured by default |
| DEFER_TASKRUN | NO |
| SINGLE_ISSUER | NO |
| buf_ring / buffer pool | NO — user manages buffers |
| Work stealing | NO — strictly per-thread (TaskQueue is !Send) |
| Task migration | Only with `sync` feature via waker channels |

#### 7.3 IoBuf / IoBufMut Traits (exact requirements)

```rust
// Bounds: Unpin + 'static
pub unsafe trait IoBuf: Unpin + 'static {
    fn read_ptr(&self) -> *const u8;
    fn bytes_init(&self) -> usize;
}

pub unsafe trait IoBufMut: Unpin + 'static {
    fn write_ptr(&mut self) -> *mut u8;
    fn bytes_total(&mut self) -> usize;
    unsafe fn set_init(&mut self, pos: usize);
}
```

Built-in implementations: `Vec<u8>`, `Box<[u8]>`, `Box<[u8; N]>`,
`&'static mut [u8; N]`, `bytes::BytesMut` (feature-gated).
Also `Rc<T: IoBuf>`, `Arc<T: IoBuf>`, `ManuallyDrop<T>`.

**Buffer ownership model:** `file.read_at(buf, offset)` consumes `buf`,
returns `(Result<usize>, buf)`. Kernel writes directly to `write_ptr()`.
After completion, monoio calls `set_init(bytes_read)`.

#### 7.4 File IO Path

```rust
// Opening with O_DIRECT:
OpenOptions::new()
    .read(true)
    .custom_flags(libc::O_DIRECT)  // line 456 in open_options.rs
    .open("adjacency.dat").await

// Read generates this SQE (read.rs lines 117-125):
opcode::Read::new(
    types::Fd(self.fd.raw_fd()),
    self.buf.write_ptr(),      // direct pointer, no copy
    self.buf.bytes_total(),
).offset(self.offset).build()
```

No intermediate buffers or copies. Kernel DMA directly to user buffer.

### 8. io-uring Crate Capabilities (tokio-rs/io-uring)

Key source: `/home/lanqing/repos/io-uring/src/`

This is the low-level Rust binding monoio builds on. Capabilities monoio
doesn't expose that we may need:

#### 8.1 Registered Buffers API (submit.rs lines 231-350)

```rust
// Register fixed buffers (eliminates per-request page pinning)
unsafe fn register_buffers(&self, bufs: &[libc::iovec]) -> io::Result<()>
// Sparse registration (5.13+) — allocate table, populate later
fn register_buffers_sparse(&self, nr: u32) -> io::Result<()>
// Update range within registered set
unsafe fn register_buffers_update(&self, offset: u32, bufs: &[libc::iovec], tags: Option<&[u64]>) -> io::Result<()>
// Unregister
fn unregister_buffers(&self) -> io::Result<()>
```

With registered buffers, use `ReadFixed`/`WriteFixed` opcodes referencing
buffer by index instead of pointer. Saves ~2000 cycles per 4KB read.

#### 8.2 Buffer Ring (Provided Buffers)

```rust
// Register a ring of buffers the kernel can select from (5.19+)
unsafe fn register_buf_ring_with_flags(
    &self, ring_addr: u64, ring_entries: u16, bgid: u16, flags: u16
) -> io::Result<()>
```

Kernel picks a free buffer from the ring for each read completion.
CQE flags contain the selected buffer ID. Max 32,768 entries per ring.

#### 8.3 Setup Flags We Care About

| Method | Flag | Use case |
|---|---|---|
| `setup_sqpoll(idle_ms)` | IORING_SETUP_SQPOLL | Kernel polls SQ |
| `setup_sqpoll_cpu(cpu)` | IORING_SETUP_SQ_AFF | Pin SQPOLL to core |
| `setup_iopoll()` | IORING_SETUP_IOPOLL | Busy-wait completions |
| `setup_defer_taskrun()` | IORING_SETUP_DEFER_TASKRUN | Batch task processing |
| `setup_single_issuer()` | IORING_SETUP_SINGLE_ISSUER | Single-thread hint |
| `setup_coop_taskrun()` | IORING_SETUP_COOP_TASKRUN | Avoid IRQ preemption |
| `setup_cqsize(n)` | IORING_SETUP_CQSIZE | Custom CQ ring size |

#### 8.4 SQ Batching (squeue.rs lines 268-314)

```rust
// Push single SQE
unsafe fn push(&mut self, entry: &E) -> Result<(), PushError>
// Push batch — checks capacity upfront, all-or-nothing
unsafe fn push_multiple(&mut self, entries: &[E]) -> Result<(), PushError>
// Flush to kernel (atomic Release on tail)
fn sync(&mut self)
```

`need_wakeup()` checks if SQPOLL thread is sleeping (SeqCst fence).

#### 8.5 NVMe Passthrough

`UringCmd16` / `UringCmd80` opcodes for direct NVMe command submission.
Supports `IORING_URING_CMD_FIXED` flag for registered buffers.
No high-level wrapper — raw command bytes.

### 9. liburing Patterns to Steal

Key source: `/home/lanqing/repos/liburing/`

#### 9.1 Batch Submission (io_uring-test.c)

```c
// O_DIRECT open
fd = open(fname, O_RDONLY | O_DIRECT);
// Aligned buffer allocation
posix_memalign(&buf, 4096, 4096);
// Prep multiple SQEs before single submit
for (i = 0; i < QD; i++) {
    sqe = io_uring_get_sqe(&ring);
    io_uring_prep_readv(sqe, fd, &iovecs[i], 1, offset);
    io_uring_sqe_set_data(sqe, &data[i]);
}
io_uring_submit(&ring);  // ONE syscall for QD reads
```

#### 9.2 Peek-Before-Wait CQE Harvesting (queue.c)

```c
// Fast path: read CQ ring without syscall
io_uring_peek_batch_cqe(ring, cqes, count);
// Only enter kernel if CQ empty AND need to wait
if (no_events) io_uring_wait_cqe(ring, &cqe);
```

This avoids syscall overhead when completions are already available.

#### 9.3 Inflight Tracking (io_uring-cp.c)

```c
// QD=64, track inflight count
int inflight = 0;
while (work_remaining || inflight) {
    while (inflight < QD && work_remaining) {
        submit_read(ring);
        inflight++;
    }
    io_uring_submit(&ring);
    // Harvest completions
    while (peek_cqe(&cqe)) {
        process(cqe);
        inflight--;
    }
}
```

This is the pattern for our `adj_inflight` semaphore.

### 10. glommio Patterns to Steal

Key source: `/home/lanqing/repos/glommio/glommio/src/sys/`

#### 10.1 Registered Buffer Pool with Buddy Allocator (dma_buffer.rs)

```rust
enum BufferStorage {
    Sys(SysAlloc),        // Regular malloc, 4096-aligned
    Uring(UringBuffer),   // From registered buddy allocator
}

UringBufferAllocator {
    data: NonNull<u8>,
    allocator: BuddyAlloc,            // Sub-allocates from large registered chunk
    uring_buffer_id: Option<u32>,     // Index for ReadFixed/WriteFixed
}
```

Pattern: register one large chunk with kernel, sub-allocate with buddy.
Automatic fallback to unregistered path when pool exhausted:

```rust
match buf.uring_buffer_id() {
    Some(idx) => sqe.prep_read_fixed(fd, buf, pos, idx),
    None => sqe.prep_read(fd, buf, pos),  // fallback
}
```

**Steal this for Divergence:** pre-register adjacency buffer pool as
one large allocation. Sub-allocate 4KB blocks. Use ReadFixed when
available, fall back to regular Read.

#### 10.2 Thread-Per-Core Executor (reactor.rs)

```rust
// Single Reactor per LocalExecutor (one-to-one)
// CQ preemption detection via cached khead/ktail pointers
preempt_ptr_head / preempt_ptr_tail  // cached at init
// Compare atomically in hot path
```

---

## Part III: Divergence Architecture Decisions

### 11. Three Non-Negotiable Contracts

**Write these into the engine crate's module doc.**

**Contract 1: Per-core rings, no sharing.**
One io_uring instance per core. One monoio executor per core.
Never share submission queues across cores. No cross-core sync on IO path.

**Contract 2: Bounded in-flight IO.**
Every core has `max_adj_inflight` (adjacency reads) and `max_vec_inflight`
(vector reads). Every query has `max_blocks_per_query`. These are hard limits,
not suggestions. Backpressure via async semaphore when limits reached.

**Contract 3: No allocator / HashMap in hot path.**
visited, heaps, scratch buffers are all thread-local + arena / fixed-capacity.
No `Box::new()`, no `HashMap`, no `BTreeMap` during graph traversal.
FixedCapacityHeap, CandidateHeap — all exist.

Visited tracking has two regimes:
- **Build / small in-memory index:** generation-count VisitedPool is fine
  (O(1) clear, array sized to N, acceptable when N fits in DRAM).
- **NVMe / large N:** visited MUST be bounded. A full-size visited[N] array
  (whether generation-count or bitset) blows DRAM and causes cache/TLB thrashing
  under thread-per-core concurrency. Use a fixed-capacity open-addressing table
  (thread-local arena, capacity ≈ `max_expansions`) instead. No HashSet (allocates).

### 12. Per-Core Worker Design

Each worker thread = one complete search engine instance.

Per-core owned state (no sharing):
- monoio runtime (owns io_uring instance)
- adjacency cache shard (buffer pool partition)
- SearchContext (visited bitset handle pool, heaps, scratch buffers)
- in-flight IO counters (adj_inflight, vec_inflight)
- pending query queue

Cross-core communication (minimal):
- Query dispatch: `hash(query) → core` or round-robin
- No shared caches. Each core's cache is independent.
- Statistics collection via relaxed atomics only.

### 13. Two IO Queues (Mandatory)

```
adjacency_reads  →  adj_inflight semaphore (default 64)
    ↑ latency-critical: stalls traversal, is p99 killer

refine_reads     →  vec_inflight semaphore (default 32)
    ↑ throughput-sensitive: batched, can degrade gracefully
```

Refine reads must NEVER starve adjacency reads. Separate semaphores enforce this.
Total inflight per core: `adj_inflight + vec_inflight ≤ io_uring SQ depth`.

**Per-query budget exhaustion:**
- `max_blocks_per_query` hit → stop expansion, return best-so-far
- Refine budget hit → skip or cap refinement, return approximate top-k
- Both cases are normal operation, not errors — record as counters for tuning

### 14. Single Yield Point in Traversal

The traversal loop has exactly ONE place that can suspend:

```
while candidate = candidates.pop():
    if candidate.dist > furthest: break

    adj_block = adj_pool.get_or_load(vid).await  // ← ONLY yield point

    neighbors = decode(adj_block)          // sync, no yield
    for nbr in neighbors:
        dist = distance(query, vec[nbr])   // sync, DRAM
        push_if_not_dominated(nbr, dist)   // sync
```

Everything else is synchronous compute. This minimizes scheduling jitter
and makes latency predictable.

**Future optimization:** prefetch window — issue N adjacency reads concurrently
instead of one at a time. Deferred until we have latency numbers from the MVP
to justify the added complexity.

### 15. Monoio: What to Use, What to Build On Top

**USE from monoio (no changes needed):**
- Per-core executor loop (runtime.rs)
- Lazy SQE batching (push now, submit on park — this IS natural batching)
- File IO with buffer ownership transfer
- Task spawning (`monoio::spawn` for concurrent queries per core)
- Custom setup flags via `RuntimeBuilder::uring_builder()` (see below)

**BUILD on top of monoio (no patching required for these):**
- Buffer pool — monoio has none, we build our own
- Semaphore — monoio has no local semaphore, hand-roll with RefCell + Waker

**Setup flags** work out of the box via `RuntimeBuilder::uring_builder()`:
```rust
let mut uring_builder = io_uring::IoUring::builder();
uring_builder.setup_single_issuer();
uring_builder.setup_coop_taskrun();
// uring_builder.setup_sqpoll(2000);  // optional, experiment
// uring_builder.setup_iopoll();       // optional, needs O_DIRECT

monoio::RuntimeBuilder::<IoUringDriver>::new()
    .uring_builder(uring_builder)
    .with_entries(1024)
    .enable_all()
    .build()
```

**Known gap: registered buffers (ReadFixed/WriteFixed).**
monoio does not expose registered buffer APIs. This is the only significant
capability gap. Three options, in order of preference:

1. **Grab uring fd, register directly** — monoio's uring driver holds the
   `IoUring` instance. If we can access the raw fd (or the inner `IoUring`),
   call `register_buffers()` via the io-uring crate. Cheapest experiment.
2. **Patch monoio** — add `register_buffers()` and `ReadFixed` opcode to
   monoio's driver. Upstream-friendly if done right.
3. **Fork monoio** — last resort if upstream won't accept changes.

**Action:** try option (1) first in the microbench (§17). Only escalate if
it doesn't work or the perf gain from ReadFixed is confirmed worth the effort.

### 16. Experimental Defaults (Tunable)

```
max_adj_inflight:     64      (per core)
max_vec_inflight:     32      (per core)
max_blocks_per_query: 256     (per query, prevents runaway)
ring_entries:         1024    (SQ depth)
cq_size:             2048    (2x SQ for headroom)
block_size:          4096    (adjacency block)
buffer_alignment:    4096    (for O_DIRECT)
sqpoll:              OFF     (enable if microbench shows submit syscall >15% of CPU)
iopoll:              OFF     (enable with O_DIRECT on NVMe if p99 latency matters more than throughput)
cqe_harvest:         drain   (iterate until CQ empty)
```

These are starting points. The microbench (see §17) determines real values.

### 17. First Milestone: NVMe Reader Microbench

**Do NOT write graph search until this works.**

Build a minimal monoio program that:
1. Opens a file with O_DIRECT
2. Uses 4KB-aligned buffer pool (pre-allocated, reusable)
3. Each core submits N random 4KB reads (simulating adjacency reads)
4. Controls `max_inflight_reads` per core
5. Measures:
   - QPS (4KB reads/sec per core and total)
   - Average and p99 latency
   - CPU utilization (rough)
   - Device bandwidth utilization vs `fio` baseline

This validates the IO path before any graph logic touches it.

---

## Part IV: Key Takeaways

1. **Naive io_uring is useless.** Must tune: registered buffers, batching,
   setup flags. Each optimization is 10-30% individually, 2-5x combined.

2. **monoio gives us 80% of what we need.** Per-core executor, buffer
   ownership, O_DIRECT support. We add: registered buffers, setup flags,
   buffer pool, semaphores.

3. **Separate IO budgets are load-bearing.** Adjacency reads stall traversal
   (p99 killer). Vector reads are batched and degradable. Never mix them.

4. **One yield point in the hot loop.** `get_or_load(vid).await` is the
   only suspension point. Everything else is sync compute on DRAM data.

5. **Microbench before graph search.** Prove the IO path works and measure
   baseline throughput. Then layer graph traversal on top.

---

*Sources: VLDB 2026 io_uring study, ATC 2019 AIOS, ADMS 2022 coroutines,
CHEOPS 2023 polling, Pestka et al. async architecture patterns,
monoio source, io-uring crate source, liburing examples, glommio source*
