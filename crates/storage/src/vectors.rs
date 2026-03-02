//! Exact vector storage — contiguous f32 array on disk.
//!
//! Vector N starts at byte offset N * dim * 4.
//! Written with std::fs (sync). Read back with io_uring during refinement.

use std::fs::File;
use std::io::{self, BufWriter, Read, Write};
use std::path::Path;

/// Write vectors.dat: contiguous f32 values in native (LE) byte order.
pub fn write_vectors_file(path: &Path, vectors: &[f32]) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Safe: f32 slice → u8 slice, same memory layout (LE on x86)
    let bytes = unsafe {
        std::slice::from_raw_parts(vectors.as_ptr() as *const u8, vectors.len() * 4)
    };
    writer.write_all(bytes)?;
    writer.flush()
}

/// Load vectors.dat back into memory.
pub fn load_vectors(path: &Path, num_vectors: usize, dim: usize) -> io::Result<Vec<f32>> {
    let expected_bytes = num_vectors * dim * 4;
    let mut file = File::open(path)?;
    let mut bytes = vec![0u8; expected_bytes];
    file.read_exact(&mut bytes)?;

    // Safe: u8 vec → f32 vec, properly aligned (Vec guarantees alignment ≥ 4)
    let mut floats = Vec::with_capacity(num_vectors * dim);
    for chunk in bytes.chunks_exact(4) {
        floats.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(floats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_vectors() {
        let dim = 8;
        let n = 100;
        let vectors: Vec<f32> = (0..n * dim).map(|i| i as f32 * 0.1).collect();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vectors.dat");

        write_vectors_file(&path, &vectors).unwrap();
        let loaded = load_vectors(&path, n, dim).unwrap();
        assert_eq!(loaded, vectors);
    }

    #[test]
    fn file_size_correct() {
        let dim = 32;
        let n = 50;
        let vectors: Vec<f32> = vec![1.0; n * dim];

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vectors.dat");

        write_vectors_file(&path, &vectors).unwrap();
        let meta = std::fs::metadata(&path).unwrap();
        assert_eq!(meta.len(), (n * dim * 4) as u64);
    }
}
