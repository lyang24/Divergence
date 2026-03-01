use crate::MetricType;

/// Computes distances between vectors.
pub trait DistanceComputer: Send + Sync {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32;
    fn distance_batch(&self, query: &[f32], vectors: &[&[f32]], results: &mut [f32]);
}

/// Create a distance computer for the given metric.
pub fn create_distance_computer(metric: MetricType) -> Box<dyn DistanceComputer> {
    match metric {
        MetricType::L2 => Box::new(L2Distance),
        MetricType::Cosine => Box::new(CosineDistance),
        MetricType::InnerProduct => Box::new(InnerProductDistance),
    }
}

pub struct L2Distance;
pub struct CosineDistance;
pub struct InnerProductDistance;

impl DistanceComputer for L2Distance {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum()
    }

    fn distance_batch(&self, query: &[f32], vectors: &[&[f32]], results: &mut [f32]) {
        for (i, v) in vectors.iter().enumerate() {
            results[i] = self.distance(query, v);
        }
    }
}

impl DistanceComputer for CosineDistance {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            dot += x * y;
            norm_a += x * x;
            norm_b += y * y;
        }
        let denom = (norm_a * norm_b).sqrt();
        if denom == 0.0 { 1.0 } else { 1.0 - dot / denom }
    }

    fn distance_batch(&self, query: &[f32], vectors: &[&[f32]], results: &mut [f32]) {
        for (i, v) in vectors.iter().enumerate() {
            results[i] = self.distance(query, v);
        }
    }
}

impl DistanceComputer for InnerProductDistance {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        -dot // negate so smaller = more similar
    }

    fn distance_batch(&self, query: &[f32], vectors: &[&[f32]], results: &mut [f32]) {
        for (i, v) in vectors.iter().enumerate() {
            results[i] = self.distance(query, v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_identical_vectors() {
        let d = L2Distance;
        let v = vec![1.0, 2.0, 3.0];
        assert_eq!(d.distance(&v, &v), 0.0);
    }

    #[test]
    fn l2_known_distance() {
        let d = L2Distance;
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert_eq!(d.distance(&a, &b), 25.0); // squared L2
    }

    #[test]
    fn cosine_identical_vectors() {
        let d = CosineDistance;
        let v = vec![1.0, 2.0, 3.0];
        assert!((d.distance(&v, &v)).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        let d = CosineDistance;
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((d.distance(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn ip_known_value() {
        let d = InnerProductDistance;
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert_eq!(d.distance(&a, &b), -32.0); // -(1*4 + 2*5 + 3*6)
    }
}
