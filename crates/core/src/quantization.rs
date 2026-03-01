/// Trait for vector quantization (PQ, RaBitQ, etc.).
pub trait Quantizer: Send + Sync {
    /// Train the quantizer on a set of vectors.
    fn train(&mut self, vectors: &[&[f32]]);

    /// Encode a vector into a compact code.
    fn encode(&self, vector: &[f32]) -> Vec<u8>;

    /// Compute approximate distance between a query and a compressed code.
    fn distance(&self, query: &[f32], code: &[u8]) -> f32;

    /// Batch distance computation.
    fn distance_batch(&self, query: &[f32], codes: &[&[u8]], results: &mut [f32]);

    /// Size of a single code in bytes.
    fn code_size(&self) -> usize;
}
