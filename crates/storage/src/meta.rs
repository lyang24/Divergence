use divergence_core::MetricType;
use serde::{Deserialize, Serialize};

/// Index metadata, serialized as meta.json in the index directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMeta {
    pub dimension: usize,
    pub metric: String,
    pub num_vectors: u32,
    pub max_degree: usize,
    pub ef_construction: usize,
    pub pq_subspaces: usize,
    pub pq_bits: usize,
    pub adj_block_size: usize,
}

impl IndexMeta {
    pub fn metric_type(&self) -> MetricType {
        match self.metric.as_str() {
            "l2" => MetricType::L2,
            "cosine" => MetricType::Cosine,
            "ip" | "inner_product" => MetricType::InnerProduct,
            _ => panic!("unknown metric: {}", self.metric),
        }
    }
}
