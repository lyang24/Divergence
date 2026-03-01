//! AdjacencyPool — caches fixed-size 4KB adjacency blocks.
//! 70% of cache budget by default.
//! CAS-based slot state machine with clock eviction.
