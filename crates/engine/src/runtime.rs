//! Thread-per-core async runtime using monoio + io_uring.
//!
//! Each worker thread gets:
//! - One CPU core (pinned via core_affinity)
//! - One monoio executor (owns one io_uring instance)
//! - No cross-core sharing on the IO path

use std::future::Future;
use std::thread::JoinHandle;

pub struct WorkerConfig {
    pub core_id: usize,
    pub uring_entries: u32, // default 1024
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            core_id: 0,
            uring_entries: 1024,
        }
    }
}

/// Spawn a worker thread pinned to a core, running a monoio io_uring executor.
pub fn spawn_worker<F, Fut>(config: WorkerConfig, f: F) -> JoinHandle<()>
where
    F: FnOnce() -> Fut + Send + 'static,
    Fut: Future<Output = ()> + 'static,
{
    std::thread::spawn(move || {
        // Pin to core
        core_affinity::set_for_current(core_affinity::CoreId {
            id: config.core_id,
        });

        // Build monoio runtime with io_uring
        let mut uring_builder = io_uring::IoUring::builder();
        uring_builder.setup_single_issuer();
        uring_builder.setup_coop_taskrun();

        let mut rt = monoio::RuntimeBuilder::<monoio::IoUringDriver>::new()
            .uring_builder(uring_builder)
            .with_entries(config.uring_entries)
            .enable_all()
            .build()
            .expect("failed to build monoio runtime");

        rt.block_on(f());
    })
}
