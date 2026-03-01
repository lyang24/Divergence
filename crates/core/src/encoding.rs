use crate::VectorId;

/// In-memory adjacency list — plain sorted neighbor IDs.
#[derive(Debug, Clone)]
pub struct AdjacencyList {
    pub neighbors: Vec<VectorId>,
}

/// On-disk adjacency list — delta-encoded + varint compressed.
#[derive(Debug, Clone)]
pub struct CompressedAdjList {
    pub data: Vec<u8>,
}
