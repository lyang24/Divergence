use divergence_core::VectorId;

/// Abstract graph storage — decoupled from specific index algorithm.
/// Flat (single-layer) graph. No hierarchy.
pub trait GraphStore: Send + Sync {
    fn neighbors(&self, id: VectorId) -> &[u32];
    fn vector(&self, id: VectorId) -> &[f32];
    fn entry_points(&self) -> &[VectorId]; // 32–256 hub nodes
    fn num_vectors(&self) -> usize;
    fn max_neighbors(&self) -> usize;
}

/// Abstract builder — different algorithms implement this.
pub trait GraphBuilder: Send + Sync {
    fn insert(&self, id: VectorId, vector: &[f32]);
    fn build(self: Box<Self>) -> Box<dyn GraphStore>;
}
