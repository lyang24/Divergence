use divergence_core::VectorId;

/// Abstract graph storage — decoupled from specific index algorithm.
pub trait GraphStore: Send + Sync {
    fn neighbors(&self, id: VectorId, level: usize) -> &[VectorId];
    fn vector(&self, id: VectorId) -> &[f32];
    fn entry_point(&self) -> VectorId;
    fn max_level(&self) -> usize;
    fn num_vectors(&self) -> usize;
    fn max_degree(&self, level: usize) -> usize;
}

/// Abstract builder — different algorithms implement this.
pub trait GraphBuilder: Send + Sync {
    fn add(&mut self, id: VectorId, vector: &[f32]);
    fn build(self: Box<Self>) -> Box<dyn GraphStore>;
}
