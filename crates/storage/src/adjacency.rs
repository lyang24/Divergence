//! Fixed-size adjacency block layout for MVP.
//!
//! Each vector gets one 4096-byte block in adjacency.dat.
//! Block offset = vid * BLOCK_SIZE.
//!
//! Block layout (little-endian):
//!   [num_neighbors: u16][padding: 6 bytes][neighbor_0: u32]...[zero-pad to 4096]
//!
//! Header is 8 bytes, leaving room for (4096 - 8) / 4 = 1022 neighbors max.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

pub const BLOCK_SIZE: usize = 4096;
const HEADER_SIZE: usize = 8;
const MAX_NEIGHBORS: usize = (BLOCK_SIZE - HEADER_SIZE) / 4;

/// Encode a neighbor list into a 4096-byte block.
pub fn encode_adj_block(neighbors: &[u32], buf: &mut [u8; BLOCK_SIZE]) {
    assert!(
        neighbors.len() <= MAX_NEIGHBORS,
        "too many neighbors: {} > {}",
        neighbors.len(),
        MAX_NEIGHBORS
    );

    buf.fill(0);

    // Header: u16 neighbor count (LE) + 6 bytes padding
    let count = neighbors.len() as u16;
    buf[0..2].copy_from_slice(&count.to_le_bytes());

    // Neighbors: u32 LE starting at offset 8
    for (i, &nbr) in neighbors.iter().enumerate() {
        let off = HEADER_SIZE + i * 4;
        buf[off..off + 4].copy_from_slice(&nbr.to_le_bytes());
    }
}

/// Decode a neighbor list from a 4096-byte block.
pub fn decode_adj_block(buf: &[u8]) -> Vec<u32> {
    assert!(buf.len() >= HEADER_SIZE);

    let count = u16::from_le_bytes([buf[0], buf[1]]) as usize;
    assert!(count <= MAX_NEIGHBORS);

    let mut neighbors = Vec::with_capacity(count);
    for i in 0..count {
        let off = HEADER_SIZE + i * 4;
        let nbr = u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]);
        neighbors.push(nbr);
    }
    neighbors
}

/// Write adjacency.dat: one BLOCK_SIZE block per vector, sequentially.
pub fn write_adjacency_file<'a>(
    path: &Path,
    num_vectors: u32,
    neighbors_fn: impl Fn(u32) -> &'a [u32],
) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    let mut block = [0u8; BLOCK_SIZE];

    for vid in 0..num_vectors {
        let nbrs = neighbors_fn(vid);
        encode_adj_block(nbrs, &mut block);
        writer.write_all(&block)?;
    }

    writer.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_empty() {
        let mut buf = [0u8; BLOCK_SIZE];
        encode_adj_block(&[], &mut buf);
        let decoded = decode_adj_block(&buf);
        assert!(decoded.is_empty());
    }

    #[test]
    fn roundtrip_small() {
        let neighbors = vec![10, 20, 30, 42];
        let mut buf = [0u8; BLOCK_SIZE];
        encode_adj_block(&neighbors, &mut buf);
        let decoded = decode_adj_block(&buf);
        assert_eq!(decoded, neighbors);
    }

    #[test]
    fn roundtrip_max() {
        let neighbors: Vec<u32> = (0..MAX_NEIGHBORS as u32).collect();
        let mut buf = [0u8; BLOCK_SIZE];
        encode_adj_block(&neighbors, &mut buf);
        let decoded = decode_adj_block(&buf);
        assert_eq!(decoded, neighbors);
    }

    #[test]
    fn file_offsets_correct() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("adjacency.dat");

        let data: Vec<Vec<u32>> = vec![vec![1, 2, 3], vec![10, 20], vec![]];
        write_adjacency_file(&path, 3, |vid| &data[vid as usize]).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes.len(), 3 * BLOCK_SIZE);

        // Verify each block
        for (vid, expected) in data.iter().enumerate() {
            let offset = vid * BLOCK_SIZE;
            let decoded = decode_adj_block(&bytes[offset..offset + BLOCK_SIZE]);
            assert_eq!(&decoded, expected, "mismatch at vid {}", vid);
        }
    }
}
