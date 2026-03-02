use std::fs::File;
use std::io::{self, BufReader, BufWriter};
use std::path::Path;

use serde::{Deserialize, Serialize};

use divergence_core::MetricType;

/// Index metadata, serialized as meta.json in the index directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMeta {
    pub dimension: usize,
    pub metric: String,
    pub num_vectors: u32,
    pub max_degree: usize,
    pub ef_construction: usize,
    pub adj_block_size: usize,
    pub entry_set: Vec<u32>,
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

    pub fn write_to(&self, path: &Path) -> io::Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))
    }

    pub fn load_from(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        serde_json::from_reader(reader).map_err(|e| io::Error::new(io::ErrorKind::Other, e))
    }
}
