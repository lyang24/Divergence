//! Integration test: build NSW → write to disk → async search → verify results match.
//!
//! These tests require io_uring support (Linux 5.1+, not inside unprivileged containers).
//! They are automatically skipped if io_uring is unavailable.

use divergence_core::distance::create_distance_computer;
use divergence_core::{MetricType, VectorId};
use divergence_engine::{disk_graph_search, IoDriver};
use divergence_index::{NswBuilder, NswConfig};
use divergence_storage::{load_vectors, IndexMeta, IndexWriter};

use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;

/// Try to build a monoio io_uring runtime. Returns false if io_uring is not
/// available (e.g. unprivileged container, old kernel), and runs the closure
/// on success.
fn with_runtime(f: impl FnOnce(&mut monoio::Runtime<monoio::time::TimeDriver<monoio::IoUringDriver>>)) -> bool {
    match monoio::RuntimeBuilder::<monoio::IoUringDriver>::new()
        .enable_all()
        .build()
    {
        Ok(mut rt) => {
            f(&mut rt);
            true
        }
        Err(_) => false,
    }
}

fn generate_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..dim).map(|_| rng.r#gen::<f32>()).collect())
        .collect()
}

#[test]
fn disk_search_matches_memory() {
    let n = 500;
    let dim = 32;
    let k = 10;
    let ef = 64;
    let m_max = 32;
    let ef_construction = 200;

    // 1. Generate vectors and build NSW in memory
    let vectors = generate_vectors(n, dim, 42);
    let config = NswConfig::new(m_max, ef_construction);
    let builder = NswBuilder::new(config, dim, MetricType::L2, n);
    for (i, v) in vectors.iter().enumerate() {
        builder.insert(VectorId(i as u32), v);
    }
    let index = builder.build();

    // 2. Search in memory for ground truth
    let query: Vec<f32> = generate_vectors(1, dim, 999)[0].clone();
    let memory_results = index.search(&query, k, ef);

    // 3. Write to disk
    let dir = tempfile::tempdir().unwrap();
    let dir_str = dir.path().to_str().unwrap().to_owned();
    let writer = IndexWriter::new(dir.path());
    writer
        .write(
            n as u32,
            dim,
            "l2",
            index.max_degree(),
            ef_construction,
            &index
                .entry_set()
                .iter()
                .map(|v| v.0)
                .collect::<Vec<_>>(),
            index.vectors_raw(),
            |vid| index.neighbors(vid),
        )
        .unwrap();

    // 4. Load meta + vectors for disk search
    let meta = IndexMeta::load_from(&dir.path().join("meta.json")).unwrap();
    let disk_vectors = load_vectors(&dir.path().join("vectors.dat"), n, dim).unwrap();

    let entry_set: Vec<VectorId> = meta.entry_set.iter().map(|&v| VectorId(v)).collect();
    let distance = create_distance_computer(MetricType::L2);

    // 5. Run async disk search inside monoio runtime
    if !with_runtime(|rt| {
        let disk_results = rt.block_on(async {
            // direct_io = false because tmpfs doesn't support O_DIRECT
            let io = IoDriver::open(&dir_str, dim, 64, false)
                .await
                .expect("failed to open IO driver");

            disk_graph_search(&query, &entry_set, k, ef, &io, &disk_vectors, dim, &*distance).await
        });

        // 6. Verify: disk results should match memory results exactly
        assert_eq!(
            disk_results.len(),
            memory_results.len(),
            "result count mismatch"
        );

        for (i, (disk, mem)) in disk_results.iter().zip(memory_results.iter()).enumerate() {
            assert_eq!(
                disk.id, mem.id,
                "VID mismatch at position {}: disk={:?} mem={:?}",
                i, disk.id, mem.id
            );
            assert!(
                (disk.distance - mem.distance).abs() < 1e-6,
                "distance mismatch at position {}: disk={} mem={}",
                i,
                disk.distance,
                mem.distance
            );
        }
    }) {
        eprintln!("SKIPPED: io_uring not available");
    }
}

#[test]
fn io_driver_reads_single_block() {
    let n = 3u32;
    let dim = 4;

    // Write a small adjacency file
    let dir = tempfile::tempdir().unwrap();
    let dir_str = dir.path().to_str().unwrap().to_owned();

    let adj: Vec<Vec<u32>> = vec![vec![1, 2], vec![0, 2], vec![0, 1]];
    let vectors: Vec<f32> = vec![0.0; n as usize * dim];

    let writer = IndexWriter::new(dir.path());
    writer
        .write(n, dim, "l2", 32, 200, &[0], &vectors, |vid| {
            &adj[vid as usize]
        })
        .unwrap();

    // Read back with IoDriver
    if !with_runtime(|rt| {
        rt.block_on(async {
            let io = IoDriver::open(&dir_str, dim, 64, false)
                .await
                .expect("failed to open IO driver");

            for vid in 0..n {
                let buf = io.read_adj_block(vid).await.expect("read failed");
                let neighbors = divergence_storage::decode_adj_block(buf.as_slice());
                assert_eq!(
                    neighbors, adj[vid as usize],
                    "mismatch at vid {}",
                    vid
                );
            }
        });
    }) {
        eprintln!("SKIPPED: io_uring not available");
    }
}
