# Reference Codebase Notes

Source code analysis of monoio, io-uring, liburing, and glommio.
Focused on implementation details relevant to Divergence.

---

## 1. monoio (bytedance/monoio) — ~/repos/monoio

Thread-per-core async runtime with io_uring backend. This is our primary runtime.

### 1.1 Source Map

| File | What it does |
|---|---|
| `monoio/src/runtime.rs` | Executor loop, Context (TLS), task driving |
| `monoio/src/driver/uring/mod.rs` | io_uring driver: submit, park, tick (CQE harvest) |
| `monoio/src/driver/op/read.rs` | Read/ReadAt opcode generation |
| `monoio/src/driver/op/write.rs` | Write/WriteAt opcode generation |
| `monoio/src/buf/io_buf.rs` | IoBuf/IoBufMut trait definitions + impls |
| `monoio/src/buf/mod.rs` | Buffer module exports |
| `monoio/src/fs/open_options.rs` | File open with custom_flags (O_DIRECT) |
| `monoio/src/fs/file/mod.rs` | File struct, read_at/write_at methods |
| `monoio/src/scheduler.rs` | TaskQueue (VecDeque, !Send, capacity 4096) |
| `monoio/src/builder.rs` | RuntimeBuilder, uring_builder() passthrough |
| `monoio/src/task/mod.rs` | Task creation, JoinHandle |
| `monoio/src/task/core.rs` | Task internals, polling |

### 1.2 Executor Loop (runtime.rs lines 154-183)

```
scoped_thread_local!(pub(crate) static CURRENT: Context);

Context {
    tasks: TaskQueue,       // VecDeque<Task>, !Send !Sync, capacity 4096
    thread_id: usize,
    time_handle: Option<TimeHandle>,
    blocking_handle: BlockingHandle,  // only with `sync` feature
}
```

Main loop:
1. Pop tasks from `TaskQueue` (FIFO VecDeque)
2. `t.run()` — poll the future once
3. Anti-starvation: `max_round = tasks.len() * 2`, break if exceeded
4. If tasks still queued: `driver.submit()` flushes accumulated SQEs
5. If no tasks: `driver.park()` → `submit_and_wait(1)` → `tick()` harvests CQEs

Key: the loop alternates between running ready tasks and parking to wait
for IO completions. No work-stealing. Strictly single-threaded.

### 1.3 io_uring Driver (driver/uring/mod.rs)

```rust
struct UringInner {
    ops: Ops,                         // Slab of in-flight operations
    uring: ManuallyDrop<IoUring>,     // from io-uring crate
    timespec: Timespec,               // reusable timeout buffer
    ext_arg: bool,                    // kernel 5.11+ feature detection
}
```

**SQE Submission — Lazy (lines 430-469):**
`submit_with_data()` just pushes the SQE to the ring via `sq.push(&sqe)`.
No syscall. Only flushes when SQ is full (line 439) or on park.

**CQE Harvesting — Drain loop (lines 376-396):**
```rust
fn tick(&mut self) {
    let cq = self.uring.completion();
    for cqe in cq {                           // drains entire CQ
        let index = cqe.user_data();
        self.ops.complete(index, resultify(&cqe), cqe.flags());
    }
}
```
Single pass, iterates until CQ is empty. Each CQE's user_data maps back
to the operation's slab index, which wakes the corresponding task.

**Park (lines 216-318):**
- Pre-5.11: installs a Timeout SQE, then `submit_and_wait(1)`
- 5.11+: uses `submit_with_args()` passing timeout as ext_arg (lower overhead)
- After submit returns: calls `tick()` to process completions

**Flush space (lines 399-428):**
When SQ is full, calls `self.uring.submit()` to flush to kernel,
then retries the push. This is the only non-park submission path.

### 1.4 File IO Path

Opening a file:
```rust
// open_options.rs line 456
pub fn custom_flags(&mut self, flags: i32) -> &mut Self {
    self.custom_flags = flags as libc::c_int;
    self
}
```

Read operation creates this SQE (driver/op/read.rs lines 117-125):
```rust
opcode::Read::new(
    types::Fd(self.fd.raw_fd()),
    self.buf.write_ptr(),       // direct pointer to user buffer
    self.buf.bytes_total(),     // capacity
).offset(self.offset).build()
// tagged with .user_data(op.index)
```

Returns `BufResult<usize, T>` = `(io::Result<usize>, T)`.
Buffer ownership: moved into Op, returned after completion.
No intermediate copies. Kernel DMA writes directly to user pointer.

### 1.5 IoBuf / IoBufMut (buf/io_buf.rs)

```rust
pub unsafe trait IoBuf: Unpin + 'static {
    fn read_ptr(&self) -> *const u8;     // pointer to initialized data
    fn bytes_init(&self) -> usize;       // how many bytes are initialized
    // provided: as_slice(), slice(), slice_unchecked()
}

pub unsafe trait IoBufMut: Unpin + 'static {
    fn write_ptr(&mut self) -> *mut u8;  // pointer where kernel can write
    fn bytes_total(&mut self) -> usize;  // total writable capacity
    unsafe fn set_init(&mut self, pos: usize);  // kernel wrote this many bytes
    // provided: slice_mut(), slice_mut_unchecked()
}
```

Implementations for: `Vec<u8>`, `Box<[u8]>`, `Box<[u8; N]>`,
`&'static mut [u8; N]`, `bytes::BytesMut` (feature-gated),
`Rc<T: IoBuf>`, `Arc<T: IoBuf>`, `ManuallyDrop<T>`.

Contract: `read_ptr()` / `write_ptr()` must return stable pointers
(no realloc during IO). This is why ownership is transferred.

### 1.6 RuntimeBuilder (builder.rs)

```rust
RuntimeBuilder::<IoUringDriver>::new()
    .with_entries(1024)           // SQ ring size
    .uring_builder(custom_builder) // pass io_uring::Builder for setup flags
    .enable_all()                 // enable timer
    .build()                      // returns Runtime<TimeDriver<IoUringDriver>>
```

`uring_builder()` (line 166) accepts `io_uring::Builder` — this is how
we pass SQPOLL, IOPOLL, SINGLE_ISSUER, etc.

### 1.7 What monoio does NOT have

- No registered/fixed buffer support (no `opcode::ReadFixed`)
- No SQPOLL configuration in defaults
- No DEFER_TASKRUN, SINGLE_ISSUER, COOP_TASKRUN
- No buffer pool or buf_ring
- No local async semaphore
- No file registration (`register_files`)
- No batch CQE peek (uses drain loop, which is fine)

To add these, either patch monoio or drop to the io-uring crate directly
for buffer registration, while keeping monoio for the executor loop.

---

## 2. io-uring (tokio-rs/io-uring) — ~/repos/io-uring

Low-level Rust bindings for io_uring. monoio wraps this crate.
Understanding this tells us what's possible at the syscall level.

### 2.1 Source Map

| File | What it does |
|---|---|
| `src/lib.rs` | IoUring struct, Builder, Parameters, feature detection |
| `src/opcode.rs` | 60+ opcodes (Read, Write, ReadFixed, etc.) |
| `src/submit.rs` | Submitter: submit(), register_buffers(), register_buf_ring() |
| `src/squeue.rs` | SubmissionQueue: push(), push_multiple(), sync() |
| `src/cqueue.rs` | CompletionQueue: iteration, overflow detection |
| `src/types.rs` | Fd, Fixed, DestinationSlot, BufRingEntry |
| `src/sys.rs` | Raw syscall wrappers |

### 2.2 Builder Setup Flags (lib.rs lines 323-465)

```rust
Builder::new()
    .setup_sqpoll(2000)        // SQPOLL with 2s idle timeout
    .setup_sqpoll_cpu(0)       // pin SQPOLL thread to core 0
    .setup_iopoll()            // busy-wait for completions
    .setup_cqsize(2048)        // explicit CQ size
    .setup_defer_taskrun()     // delay task_work to io_uring_enter() (6.1+)
    .setup_single_issuer()     // single-thread optimization hint (6.0+)
    .setup_coop_taskrun()      // avoid IRQ preemption (5.19+)
    .setup_submit_all()        // continue after individual SQE errors (5.18+)
    .setup_clamp()             // clamp ring sizes to max
    .build(entries)            // returns IoUring<SQE, CQE>
```

Feature detection after build:
```rust
let params = ring.params();
params.is_setup_sqpoll()
params.is_feature_nodrop()          // CQ never silently drops
params.is_feature_submit_stable()   // data consumed on push
params.is_feature_fast_poll()
params.is_feature_native_workers()
```

### 2.3 Registered Buffers (submit.rs lines 231-350)

```rust
let submitter = ring.submitter();

// Register fixed buffers (kernel pins pages, creates DMA mappings)
unsafe { submitter.register_buffers(&iovecs)? };

// Sparse registration: allocate table first, populate later (5.13+)
submitter.register_buffers_sparse(num_buffers)?;
unsafe { submitter.register_buffers_update(offset, &iovecs, tags)? };

// Unregister
submitter.unregister_buffers()?;
```

After registration, use `ReadFixed` / `WriteFixed` opcodes:
```rust
opcode::ReadFixed::new(
    types::Fd(fd),
    buf_ptr,
    buf_len,
    buf_index,     // index into registered array
).offset(file_offset).build()
```

Saves ~2000 cycles per read (eliminates page pinning per request).

### 2.4 Buffer Ring / Provided Buffers (submit.rs lines 605-687)

```rust
// Register a ring of buffers (5.19+)
unsafe { submitter.register_buf_ring_with_flags(
    ring_addr,       // pointer to ring memory
    ring_entries,    // max 32768
    bgid,           // buffer group ID
    flags,
)? };

// Use with BUFFER_SELECT flag on read operations
opcode::Read::new(fd, ptr::null_mut(), buf_len)
    .buf_group(bgid)
    .build()
    .flags(squeue::Flags::BUFFER_SELECT)
```

Kernel picks a buffer from the group. CQE flags contain buffer ID.
Application reclaims buffer after processing.

### 2.5 Submission Queue Batching (squeue.rs lines 268-314)

```rust
let mut sq = ring.submission();

// Single push
unsafe { sq.push(&entry)? };

// Batch push (all-or-nothing: fails if not enough space for all)
unsafe { sq.push_multiple(&entries)? };

// Flush to kernel (atomic Release on tail pointer)
sq.sync();
// Drop also calls sync()

// Check SQPOLL thread state
sq.need_wakeup()  // uses SeqCst fence
```

### 2.6 Completion Queue (cqueue.rs)

```rust
let cq = ring.completion();

// Iterate (drains available CQEs)
for cqe in cq {
    let user_data = cqe.user_data();
    let result = cqe.result();
    let flags = cqe.flags();
    // Extract buffer ID if BUFFER_SELECT was used:
    let buf_id = cqueue::buffer_select(flags);
}

// Overflow detection
cq.overflow()  // number of dropped CQEs if CQ was full
```

### 2.7 NVMe Passthrough (opcode.rs lines 1637-1678)

```rust
// 16-byte command passthrough
opcode::UringCmd16::new(types::Fd(nvme_fd), cmd_op)
    .cmd(cmd_bytes)          // 16 bytes of NVMe command
    .buf_index(buf_idx)      // optional: registered buffer
    .build()

// 80-byte command passthrough
opcode::UringCmd80::new(types::Fd(nvme_fd), cmd_op)
    .cmd(cmd_bytes)          // 80 bytes
    .build()                 // returns Entry128 (128-byte SQE)
```

Supports `IORING_URING_CMD_FIXED` flag for registered buffers.

### 2.8 Key Types

```rust
// SQE entry: 64 bytes
pub struct Entry(sys::io_uring_sqe);
// Extended SQE: 128 bytes (for UringCmd80, etc.)
pub struct Entry128(Entry, [u8; 64]);

// SQE flags (squeue.rs lines 57-116)
Flags::IO_DRAIN      // drain previous entries first
Flags::IO_LINK       // link to next entry
Flags::IO_HARDLINK   // hard link (no sever on error)
Flags::ASYNC         // force async execution
Flags::BUFFER_SELECT // select from buffer group
Flags::SKIP_SUCCESS  // skip CQE on success (5.17+)
```

---

## 3. liburing (axboe/liburing) — ~/repos/liburing

C reference implementation. The "ground truth" for io_uring patterns.

### 3.1 Source Map

| File | What it does |
|---|---|
| `src/queue.c` | SQ/CQ management: flush, peek, submit |
| `src/register.c` | Buffer/file registration |
| `src/setup.c` | Ring initialization |
| `src/syscall.c` | Raw io_uring_enter() wrapper |
| `src/include/liburing.h` | Inline helpers: prep_read, prep_write, etc. |
| `examples/io_uring-test.c` | O_DIRECT + batch reads |
| `examples/io_uring-cp.c` | File copy with QD=64 inflight tracking |
| `examples/link-cp.c` | SQE linking (read→write chains) |
| `examples/proxy.c` | SQPOLL configuration |
| `examples/reg-wait.c` | Registered wait regions (6.0+) |

### 3.2 Batch Submission Pattern (io_uring-test.c)

```c
// Setup
fd = open(fname, O_RDONLY | O_DIRECT);
posix_memalign(&buf, 4096, 4096);  // mandatory for O_DIRECT

// Prep multiple SQEs
for (i = 0; i < QD; i++) {
    sqe = io_uring_get_sqe(&ring);
    io_uring_prep_readv(sqe, fd, &iovecs[i], 1, offset);
    io_uring_sqe_set_data(sqe, &data[i]);
}
// Single submit syscall for all QD reads
io_uring_submit(&ring);
```

### 3.3 CQE Harvest — Peek Before Wait (queue.c lines 154-197)

```c
// Fast path: no syscall
unsigned io_uring_peek_batch_cqe(ring, cqes, count) {
    unsigned ready = io_uring_cq_ready(ring);
    if (ready) {
        // Read directly from CQ ring memory
        unsigned head = *ring->cq.khead;
        for (i = 0; i < min(ready, count); i++) {
            cqes[i] = &ring->cq.cqes[(head + i) & mask];
        }
        return i;
    }
    // Slow path: flush SQ, then peek again
    __io_uring_get_events(ring);
    return __io_uring_peek_cqe(ring, ...);
}
```

`io_uring_cq_ready()` is pure pointer arithmetic — no syscall.
Only falls back to kernel if CQ is empty.

### 3.4 SQ Flush Logic (queue.c lines 203-238)

```c
unsigned __io_uring_flush_sq(ring) {
    // Copy from local tail to kernel-visible tail
    smp_store_release(ring->sq.ktail, tail);
    // Read kernel head to know how many were consumed
    head = smp_load_acquire(ring->sq.khead);
    return tail - head;  // pending count
}
```

Memory ordering: Release on tail write, Acquire on head read.
Critical for SQPOLL correctness.

### 3.5 Inflight Tracking Pattern (io_uring-cp.c)

```c
#define QD 64

struct io_data {
    int read;           // 1=read, 0=write
    off_t offset;
    struct iovec iov;
};

int inflight = 0;

while (insize || inflight) {
    // Submit reads up to QD
    while (inflight < QD && insize) {
        sqe = io_uring_get_sqe(ring);
        io_uring_prep_readv(sqe, infd, &data->iov, 1, offset);
        io_uring_sqe_set_data(sqe, data);
        inflight++;
        insize -= this_size;
    }
    io_uring_submit(ring);

    // Harvest completions
    ret = io_uring_wait_cqe(ring, &cqe);  // block for at least 1
    // Then peek remaining non-blocking
    while (io_uring_peek_cqe(ring, &cqe) == 0) {
        process(cqe);
        io_uring_cqe_seen(ring, cqe);
        inflight--;
    }
}
```

This wait-then-peek pattern is optimal: one blocking wait, then
drain all available without syscalls.

### 3.6 SQE Linking (link-cp.c)

```c
// Chain read → write as linked operations
sqe_read = io_uring_get_sqe(ring);
io_uring_prep_readv(sqe_read, infd, &data->iov, 1, data->offset);
sqe_read->flags |= IOSQE_IO_LINK;  // link to next SQE

sqe_write = io_uring_get_sqe(ring);
io_uring_prep_writev(sqe_write, outfd, &data->iov, 1, data->offset);
// write only executes after read completes
```

On linked SQE failure, subsequent linked entries get `-ECANCELED`.

### 3.7 Buffer Registration (register.c)

```c
// Register fixed buffers (kernel pins pages)
struct iovec iovs[N];
for (i = 0; i < N; i++) {
    iovs[i].iov_base = aligned_bufs[i];
    iovs[i].iov_len = BUF_SIZE;
}
io_uring_register_buffers(ring, iovs, N);

// Use with prep_read_fixed (references buffer by index)
io_uring_prep_read_fixed(sqe, fd, buf, len, offset, buf_index);
```

### 3.8 SQPOLL Configuration (proxy.c)

```c
struct io_uring_params params = {
    .flags = IORING_SETUP_SQPOLL,
    .sq_thread_idle = 1000,  // ms before kernel thread sleeps
};
io_uring_queue_init_params(1024, &ring, &params);

// Check if SQPOLL thread needs waking
if (IO_URING_READ_ONCE(*ring.sq.kflags) & IORING_SQ_NEED_WAKEUP)
    io_uring_enter(ring.fd, 0, 0, IORING_ENTER_SQ_WAKEUP, NULL);
```

---

## 4. glommio (DataDog/glommio) — ~/repos/glommio

Thread-per-core async runtime. Alternative to monoio. Worth studying
for its buffer pool design.

### 4.1 Source Map

| File | What it does |
|---|---|
| `glommio/src/reactor.rs` | Core reactor, one per LocalExecutor |
| `glommio/src/sys/uring.rs` | io_uring integration, SQE fill, buffer mgmt |
| `glommio/src/sys/dma_buffer.rs` | DmaBuffer types, buddy allocator |
| `glommio/src/executor/mod.rs` | LocalExecutor, thread-per-core |
| `glommio/src/io/dma_file.rs` | O_DIRECT file abstraction |
| `glommio/src/io/buffered_file.rs` | Buffered file abstraction |

### 4.2 Architecture

Thread-per-core confirmed:
- `LocalExecutor` — single-threaded executor (scoped_thread_local)
- `LocalExecutorBuilder` — per-core setup with `Placement` for affinity
- `LocalExecutorPoolBuilder` — multi-core pool
- One `Reactor` per `LocalExecutor` (one-to-one mapping)

Reactor (reactor.rs):
- Caches CQ `khead`/`ktail` pointers at init (lines 174-175)
- Preemption detection: compare head/tail atomically in hot path
- Single Reactor drives all IO for its executor

### 4.3 Buffer Management — The Good Part (sys/dma_buffer.rs)

```rust
// Three buffer storage backends
enum BufferStorage {
    Sys(SysAlloc),        // malloc with 4096-byte alignment
    Uring(UringBuffer),   // from registered buddy allocator
    EventFd(*mut u8),     // special-purpose
    ReadResult(ReadResult),
}

// Buddy allocator for registered buffer sub-allocation
struct UringBufferAllocator {
    data: NonNull<u8>,                // large pre-registered chunk
    allocator: BuddyAlloc,           // buddy allocator for sub-blocks
    uring_buffer_id: Option<u32>,     // index when registered with kernel
}
```

Registration flow:
1. Allocate large chunk (e.g., 64MB aligned)
2. Register with kernel: `io_uring_register_buffers()`
3. Sub-allocate 4KB blocks via buddy allocator
4. Each sub-allocation knows its `uring_buffer_id`

Usage in SQE fill (sys/uring.rs lines 267-400):
```rust
match buf.uring_buffer_id() {
    Some(idx) => sqe.prep_read_fixed(fd, buf, pos, idx),  // fast path
    None => sqe.prep_read(fd, buf, pos),                    // fallback
}
```

Automatic fallback when buddy pool is exhausted (line 157).
Never fails — just slower when using unregistered path.

### 4.4 SQE Fill Pattern (sys/uring.rs)

Handles read/write with automatic buffer allocation via closure pattern.
`ReadFixed` variant attempted first, falls back to dynamic.
Tracks buffers in `IoBuffer::DmaSink(buf)` slots.

### 4.5 Key Differences from monoio

| Aspect | glommio | monoio |
|---|---|---|
| Maturity | More features, larger codebase | Simpler, lighter |
| Buffer pool | Buddy allocator + registered | None |
| Registered buffers | YES, with fallback | NO |
| File abstraction | DmaFile (O_DIRECT native) | File + custom_flags |
| Timeout | Native via SQE (5.8+) | Via SQE or ext_arg |
| Task queue | Multiple levels (latency rings) | Single VecDeque |
| CQ preemption | Detects via cached khead/ktail | No preemption detection |
| Ecosystem | DataDog production use | ByteDance production use |

### 4.6 What to Steal for Divergence

**YES — steal these patterns:**
- Buddy allocator for adjacency buffer pool (better than flat free-list)
- Registered buffer + fallback pattern (transparent performance gain)
- DmaBuffer type with storage backend enum (clean abstraction)
- Preemption detection via cached CQ pointers

**NO — skip these:**
- Multi-level task queues (over-engineered for our use case)
- DmaFile abstraction (too opinionated, we want raw control)
- Buffered file mode (we never want buffered IO on hot path)

---

## 5. Cross-Cutting Observations

### 5.1 Everyone Does Lazy SQ Submission

Both monoio and glommio push SQEs without syscall, then batch-flush.
liburing's pattern is the same: prep N SQEs, one submit().
This is the universal pattern — never submit single SQEs.

### 5.2 CQE Drain Is Always a Simple Loop

No fancy batching needed. Just iterate until CQ is empty.
liburing's peek_batch is an optimization for avoiding the loop overhead
when you only want N completions, but draining all is simpler and correct.

### 5.3 Buffer Ownership Is the Hard Part

All three Rust runtimes (monoio, glommio, tokio-uring) use ownership
transfer: buffer moves into the operation, comes back in the result.
This is forced by io_uring's completion model — the kernel holds the
buffer pointer between submit and complete.

For Divergence: our AlignedBuf must be movable (Unpin + 'static),
own its allocation (no borrows), and return to a pool after use.

### 5.4 Registered Buffers Are Worth ~4-6% Free

glommio does it, monoio doesn't. The io-uring crate supports it.
For Divergence: register the adjacency buffer pool at startup.
Use ReadFixed for adjacency reads (fixed 4KB). Fall back to Read
for anything else. This is a clean 4-6% win with minimal complexity.

### 5.5 SQPOLL Is Controversial

liburing shows how to set it up. Neither monoio nor glommio enable
it by default. For our use case (latency-sensitive, single-issuer
per core), SQPOLL may help but adds a kernel thread per core.
Benchmark before committing.
