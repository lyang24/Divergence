//! IndexWriter — serializes an in-memory graph to disk files.
//!
//! Produces three files in the output directory:
//!   - adjacency.dat: one 4096-byte block per vector
//!   - vectors.dat: contiguous f32 array
//!   - meta.json: index metadata + entry set

use std::io;
use std::path::PathBuf;

use crate::adjacency::{self, BLOCK_SIZE};
use crate::meta::IndexMeta;
use crate::vectors;

pub struct IndexWriter {
    dir: PathBuf,
}

impl IndexWriter {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    /// Write all index files. Takes raw data — no dependency on index crate.
    pub fn write<'a>(
        &self,
        num_vectors: u32,
        dimension: usize,
        metric: &str,
        max_degree: usize,
        ef_construction: usize,
        entry_set: &[u32],
        vectors_data: &[f32],
        neighbors_fn: impl Fn(u32) -> &'a [u32],
    ) -> io::Result<()> {
        std::fs::create_dir_all(&self.dir)?;

        // Write adjacency.dat
        adjacency::write_adjacency_file(
            &self.adj_path(),
            num_vectors,
            &neighbors_fn,
        )?;

        // Write vectors.dat
        vectors::write_vectors_file(&self.vec_path(), vectors_data)?;

        // Write meta.json
        let meta = IndexMeta {
            dimension,
            metric: metric.to_string(),
            num_vectors,
            max_degree,
            ef_construction,
            adj_block_size: BLOCK_SIZE,
            entry_set: entry_set.iter().map(|&v| v).collect(),
        };
        meta.write_to(&self.meta_path())?;

        Ok(())
    }

    pub fn adj_path(&self) -> PathBuf {
        self.dir.join("adjacency.dat")
    }

    pub fn vec_path(&self) -> PathBuf {
        self.dir.join("vectors.dat")
    }

    pub fn meta_path(&self) -> PathBuf {
        self.dir.join("meta.json")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_small_index() {
        let dir = tempfile::tempdir().unwrap();
        let writer = IndexWriter::new(dir.path());

        let num_vectors = 5u32;
        let dim = 4;
        let vectors: Vec<f32> = (0..num_vectors as usize * dim).map(|i| i as f32).collect();
        let adj: Vec<Vec<u32>> = vec![
            vec![1, 2],
            vec![0, 2, 3],
            vec![0, 1],
            vec![1, 4],
            vec![3],
        ];
        let entry_set = vec![1u32, 0];

        writer
            .write(
                num_vectors,
                dim,
                "l2",
                32,
                200,
                &entry_set,
                &vectors,
                |vid| &adj[vid as usize],
            )
            .unwrap();

        // Verify files exist with correct sizes
        assert!(writer.adj_path().exists());
        assert!(writer.vec_path().exists());
        assert!(writer.meta_path().exists());

        let adj_size = std::fs::metadata(writer.adj_path()).unwrap().len();
        assert_eq!(adj_size, num_vectors as u64 * BLOCK_SIZE as u64);

        let vec_size = std::fs::metadata(writer.vec_path()).unwrap().len();
        assert_eq!(vec_size, (num_vectors as usize * dim * 4) as u64);

        // Verify meta roundtrips
        let meta = IndexMeta::load_from(&writer.meta_path()).unwrap();
        assert_eq!(meta.num_vectors, num_vectors);
        assert_eq!(meta.dimension, dim);
        assert_eq!(meta.entry_set, entry_set);
    }
}
