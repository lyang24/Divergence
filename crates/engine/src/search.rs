//! Async beam search over on-disk adjacency graph.
//!
//! Vectors stay in DRAM. Adjacency blocks are read from NVMe via io_uring.
//! Each `.await` on `io.read_adj_block()` is a coroutine suspension point —
//! monoio can service other queries while the NVMe read is in flight.

use divergence_core::distance::DistanceComputer;
use divergence_core::VectorId;
use divergence_index::{CandidateHeap, FixedCapacityHeap, ScoredId};
use divergence_storage::decode_adj_block;

use crate::io::IoDriver;

/// Async beam search on disk-resident adjacency graph.
///
/// - `query`: the query vector
/// - `entry_set`: hub entry point VIDs
/// - `k`: number of results to return
/// - `ef`: beam width (ef >= k)
/// - `io`: async IO driver for adjacency reads
/// - `vectors`: all vectors in DRAM (flat f32 array)
/// - `dimension`: vector dimensionality
/// - `distance`: distance function
pub async fn disk_graph_search(
    query: &[f32],
    entry_set: &[VectorId],
    k: usize,
    ef: usize,
    io: &IoDriver,
    vectors: &[f32],
    dimension: usize,
    distance: &dyn DistanceComputer,
) -> Vec<ScoredId> {
    let mut nearest = FixedCapacityHeap::new(ef);
    let mut candidates = CandidateHeap::new();

    // Bounded visited set: capacity tied to max expansions.
    // For MVP, use a simple Vec<bool> sized to num_vectors.
    // TODO: replace with bounded open-addressing table for large N.
    let num_vectors = vectors.len() / dimension;
    let mut visited = vec![false; num_vectors];

    // Seed from entry set (DRAM only, no IO)
    for &ep in entry_set {
        let vid = ep.0 as usize;
        if vid < num_vectors {
            visited[vid] = true;
            let vec_offset = vid * dimension;
            let d = distance.distance(query, &vectors[vec_offset..vec_offset + dimension]);
            let scored = ScoredId { distance: d, id: ep };
            nearest.push(scored);
            candidates.push(scored);
        }
    }

    // Beam search: expand candidates by reading adjacency blocks from disk
    while let Some(candidate) = candidates.pop() {
        if let Some(furthest) = nearest.furthest() {
            if candidate.distance > furthest.distance {
                break;
            }
        }

        // ← ONLY yield point: read adjacency block from NVMe via io_uring
        let adj_buf = match io.read_adj_block(candidate.id.0).await {
            Ok(buf) => buf,
            Err(_) => continue, // skip on IO error
        };

        let neighbors = decode_adj_block(adj_buf.as_slice());

        // Process neighbors (all sync, DRAM only)
        for nbr_raw in neighbors {
            let nbr_idx = nbr_raw as usize;
            if nbr_idx >= num_vectors || visited[nbr_idx] {
                continue;
            }
            visited[nbr_idx] = true;

            let vec_offset = nbr_idx * dimension;
            let d = distance.distance(query, &vectors[vec_offset..vec_offset + dimension]);

            let dominated = nearest.len() >= ef && d >= nearest.furthest().unwrap().distance;
            if !dominated {
                let scored = ScoredId {
                    distance: d,
                    id: VectorId(nbr_raw),
                };
                candidates.push(scored);
                nearest.push(scored);
            }
        }
    }

    let mut results = nearest.into_sorted_vec();
    results.truncate(k);
    results
}
