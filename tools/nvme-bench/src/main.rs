//! NVMe reader microbench — validates the io_uring IO path.
//!
//! Fires random 4KB reads at a file with O_DIRECT via monoio/io_uring.
//! Measures IOPS and latency per core. Compare output against:
//!
//!   fio --name=test --filename=<file> --rw=randread --bs=4k \
//!       --ioengine=io_uring --direct=1 --iodepth=64 --numjobs=<cores> \
//!       --runtime=10 --time_based --group_reporting
//!
//! Usage:
//!   # First create a test file (1GB):
//!   dd if=/dev/urandom of=/tmp/bench.dat bs=1M count=1024 oflag=direct
//!
//!   # Run the bench:
//!   cargo run --release -p nvme-bench -- --file /tmp/bench.dat --cores 1 --inflight 64 --seconds 5

use std::alloc::{self, Layout};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;

const ALIGNMENT: usize = 4096;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "nvme-bench", about = "NVMe random read microbench via io_uring")]
struct Args {
    /// Path to test file (must exist, use dd to create)
    #[arg(long)]
    file: String,

    /// Number of CPU cores to use
    #[arg(long, default_value = "1")]
    cores: usize,

    /// Max inflight reads per core (queue depth)
    #[arg(long, default_value = "64")]
    inflight: usize,

    /// Test duration in seconds
    #[arg(long, default_value = "5")]
    seconds: u64,

    /// Read block size in bytes
    #[arg(long, default_value = "4096")]
    block_size: usize,

    /// io_uring SQ entries
    #[arg(long, default_value = "1024")]
    ring_entries: u32,
}

// ---------------------------------------------------------------------------
// Aligned buffer (same concept as engine's AlignedBuf, standalone here)
// ---------------------------------------------------------------------------

struct AlignedBuf {
    ptr: NonNull<u8>,
    len: usize,
    capacity: usize,
}

impl AlignedBuf {
    fn new(size: usize) -> Self {
        let capacity = (size.max(ALIGNMENT) + ALIGNMENT - 1) & !(ALIGNMENT - 1);
        let layout = Layout::from_size_align(capacity, ALIGNMENT).unwrap();
        let ptr = unsafe { alloc::alloc_zeroed(layout) };
        let ptr = NonNull::new(ptr).expect("alloc failed");
        Self {
            ptr,
            len: 0,
            capacity,
        }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.capacity, ALIGNMENT).unwrap();
        unsafe { alloc::dealloc(self.ptr.as_ptr(), layout) };
    }
}

unsafe impl monoio::buf::IoBuf for AlignedBuf {
    fn read_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }
    fn bytes_init(&self) -> usize {
        self.len
    }
}

unsafe impl monoio::buf::IoBufMut for AlignedBuf {
    fn write_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }
    fn bytes_total(&mut self) -> usize {
        self.capacity
    }
    unsafe fn set_init(&mut self, pos: usize) {
        self.len = pos;
    }
}

// ---------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------

fn run_worker(
    core_id: usize,
    args: &Args,
    stop: Arc<AtomicBool>,
) -> (u64, u64, u64, Vec<u64>) {
    // Pin to core
    core_affinity::set_for_current(core_affinity::CoreId { id: core_id });

    // Build monoio runtime with tuned io_uring
    let mut uring_builder = io_uring::IoUring::builder();
    uring_builder.setup_single_issuer();
    uring_builder.setup_coop_taskrun();

    let mut rt = monoio::RuntimeBuilder::<monoio::IoUringDriver>::new()
        .uring_builder(uring_builder)
        .with_entries(args.ring_entries)
        .enable_all()
        .build()
        .expect("failed to build monoio runtime");

    let file_path = args.file.clone();
    let inflight = args.inflight;
    let block_size = args.block_size;

    // Get file size
    let file_meta = std::fs::metadata(&file_path).expect("cannot stat file");
    let file_size = file_meta.len();
    let max_block = file_size / block_size as u64;
    assert!(max_block > 0, "file too small for block_size={}", block_size);

    rt.block_on(async move {
        use std::os::unix::fs::OpenOptionsExt as _;
        use std::rc::Rc;

        let file = monoio::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT)
            .open(&file_path)
            .await
            .expect("failed to open file with O_DIRECT");

        // monoio is single-threaded, so Rc is fine for sharing the file
        let file = Rc::new(file);

        // Spawn `inflight` concurrent read tasks
        let mut handles = Vec::new();
        for task_id in 0..inflight {
            let file = Rc::clone(&file);
            let stop = stop.clone();
            let seed = (core_id as u64) * 1000 + task_id as u64;

            handles.push(monoio::spawn(async move {
                let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
                let mut local_count = 0u64;
                let mut local_lat = Vec::with_capacity(100_000);

                while !stop.load(Ordering::Relaxed) {
                    let block_idx = rng.gen_range(0..max_block);
                    let offset = block_idx * block_size as u64;
                    let buf = AlignedBuf::new(block_size);

                    let t0 = Instant::now();
                    let (result, _buf) = file.read_at(buf, offset).await;
                    let elapsed_ns = t0.elapsed().as_nanos() as u64;

                    match result {
                        Ok(n) if n == block_size => {
                            local_count += 1;
                            local_lat.push(elapsed_ns);
                        }
                        Ok(n) => {
                            eprintln!("short read: {} bytes at offset {}", n, offset);
                        }
                        Err(e) => {
                            eprintln!("read error at offset {}: {}", offset, e);
                        }
                    }
                }

                (local_count, local_lat)
            }));
        }

        // Collect results from all tasks
        let mut total_reads = 0u64;
        let mut total_lat_ns = 0u64;
        let mut max_lat_ns = 0u64;
        let mut all_latencies = Vec::new();

        for handle in handles {
            let (count, lats) = handle.await;
            total_reads += count;
            for &lat in &lats {
                total_lat_ns += lat;
                if lat > max_lat_ns {
                    max_lat_ns = lat;
                }
            }
            all_latencies.extend(lats);
        }

        (total_reads, total_lat_ns, max_lat_ns, all_latencies)
    })
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let args = Args::parse();

    // Validate file exists
    let file_meta = std::fs::metadata(&args.file)
        .unwrap_or_else(|e| panic!("cannot access {}: {}", args.file, e));
    let file_size_mb = file_meta.len() as f64 / (1024.0 * 1024.0);

    println!("=== NVMe Reader Microbench ===");
    println!("File:       {} ({:.1} MB)", args.file, file_size_mb);
    println!("Cores:      {}", args.cores);
    println!("Inflight:   {} per core", args.inflight);
    println!("Block size: {} bytes", args.block_size);
    println!("Duration:   {} seconds", args.seconds);
    println!("Ring entries: {}", args.ring_entries);
    println!();

    let stop = Arc::new(AtomicBool::new(false));

    // Spawn worker threads
    let mut handles = Vec::new();

    // Get available cores
    let core_ids = core_affinity::get_core_ids().unwrap_or_default();
    let num_cores = args.cores.min(core_ids.len());
    if num_cores == 0 {
        eprintln!("no CPU cores available");
        std::process::exit(1);
    }
    if num_cores < args.cores {
        eprintln!(
            "warning: requested {} cores but only {} available, using {}",
            args.cores, core_ids.len(), num_cores
        );
    }

    let start = Instant::now();

    for i in 0..num_cores {
        let stop = stop.clone();
        let core_id = core_ids[i].id;
        let file = args.file.clone();
        let inflight = args.inflight;
        let seconds = args.seconds;
        let block_size = args.block_size;
        let ring_entries = args.ring_entries;

        handles.push(std::thread::spawn(move || {
            let worker_args = Args {
                file,
                cores: 1,
                inflight,
                seconds,
                block_size,
                ring_entries,
            };
            run_worker(core_id, &worker_args, stop)
        }));
    }

    // Timer thread
    {
        let stop = stop.clone();
        let seconds = args.seconds;
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_secs(seconds));
            stop.store(true, Ordering::Relaxed);
        });
    }

    // Collect results
    let mut grand_total_reads = 0u64;
    let mut all_latencies: Vec<u64> = Vec::new();

    for (i, handle) in handles.into_iter().enumerate() {
        let (reads, total_lat_ns, max_lat_ns, mut latencies) = handle.join().unwrap();
        let elapsed = start.elapsed().as_secs_f64();
        let iops = reads as f64 / elapsed;
        let avg_lat_us = if reads > 0 {
            total_lat_ns as f64 / reads as f64 / 1000.0
        } else {
            0.0
        };

        latencies.sort_unstable();
        let p50_us = percentile_us(&latencies, 50.0);
        let p99_us = percentile_us(&latencies, 99.0);
        let p999_us = percentile_us(&latencies, 99.9);

        println!(
            "Core {:2}: {:>8.0} IOPS | avg {:>7.1}µs | p50 {:>7.1}µs | p99 {:>7.1}µs | p99.9 {:>7.1}µs | max {:>7.1}µs | {} reads",
            i, iops, avg_lat_us, p50_us, p99_us, p999_us,
            max_lat_ns as f64 / 1000.0, reads
        );

        grand_total_reads += reads;
        all_latencies.append(&mut latencies);
    }

    let elapsed = start.elapsed().as_secs_f64();
    let total_iops = grand_total_reads as f64 / elapsed;
    let bw_mbs = total_iops * args.block_size as f64 / (1024.0 * 1024.0);

    all_latencies.sort_unstable();
    let total_avg_us = if grand_total_reads > 0 {
        all_latencies.iter().sum::<u64>() as f64 / grand_total_reads as f64 / 1000.0
    } else {
        0.0
    };
    let total_p50 = percentile_us(&all_latencies, 50.0);
    let total_p99 = percentile_us(&all_latencies, 99.0);
    let total_p999 = percentile_us(&all_latencies, 99.9);

    println!();
    println!("=== Total ===");
    println!("IOPS:      {:.0}", total_iops);
    println!("Bandwidth: {:.1} MB/s", bw_mbs);
    println!("Latency:   avg {:.1}µs | p50 {:.1}µs | p99 {:.1}µs | p99.9 {:.1}µs",
        total_avg_us, total_p50, total_p99, total_p999);
    println!("Reads:     {} in {:.2}s", grand_total_reads, elapsed);
    println!();
    println!("Compare with fio:");
    println!("  fio --name=test --filename={} --rw=randread --bs={}",
        args.file, args.block_size);
    println!("      --ioengine=io_uring --direct=1 --iodepth={} --numjobs={}",
        args.inflight, num_cores);
    println!("      --runtime={} --time_based --group_reporting", args.seconds);
}

fn percentile_us(sorted: &[u64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 * pct / 100.0) as usize).min(sorted.len() - 1);
    sorted[idx] as f64 / 1000.0
}
