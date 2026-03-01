use std::sync::Mutex;

/// Pool of reusable visited-set handles.
/// Each thread borrows a handle for one search/insert, returns it on drop.
pub struct VisitedPool {
    pool: Mutex<Vec<VisitedList>>,
    num_elements: usize,
}

impl VisitedPool {
    pub fn new(num_elements: usize) -> Self {
        Self {
            pool: Mutex::new(Vec::new()),
            num_elements,
        }
    }

    /// Borrow a handle. Reuses from pool or creates new.
    /// Generation is bumped on acquisition so stale state is invalidated.
    pub fn get(&self) -> VisitedListHandle<'_> {
        let mut inner = self
            .pool
            .lock()
            .unwrap()
            .pop()
            .unwrap_or_else(|| VisitedList::new(self.num_elements));
        // Bump generation to invalidate any stale visited bits from
        // a previous use of this handle. Without this, reused handles
        // would see vertices from a prior search as "already visited".
        inner.generation += 1;
        VisitedListHandle { pool: self, inner }
    }

    fn return_list(&self, list: VisitedList) {
        self.pool.lock().unwrap().push(list);
    }
}

/// Bitset with lazy per-word generation clearing.
/// No bulk memset ever needed — next_iteration() is O(1).
struct VisitedList {
    generation: u64,
    words: Vec<u64>,
    tags: Vec<u64>,
}

impl VisitedList {
    fn new(num_elements: usize) -> Self {
        let num_words = (num_elements + 63) / 64;
        Self {
            generation: 1,
            words: vec![0u64; num_words],
            tags: vec![0u64; num_words],
        }
    }
}

/// RAII handle — returned to pool on drop.
pub struct VisitedListHandle<'a> {
    pool: &'a VisitedPool,
    inner: VisitedList,
}

impl VisitedListHandle<'_> {
    /// Returns true if already visited. Marks as visited either way.
    #[inline]
    pub fn check_and_mark(&mut self, id: u32) -> bool {
        let idx = id as usize;
        let word_idx = idx / 64;
        let bit_idx = idx % 64;
        let mask = 1u64 << bit_idx;

        // Lazy clear: if this word's tag is stale, reset it
        if self.inner.tags[word_idx] != self.inner.generation {
            self.inner.tags[word_idx] = self.inner.generation;
            self.inner.words[word_idx] = 0;
        }

        let was_visited = (self.inner.words[word_idx] & mask) != 0;
        self.inner.words[word_idx] |= mask;
        was_visited
    }

    /// Advance to next iteration. O(1) — no clearing.
    pub fn next_iteration(&mut self) {
        self.inner.generation += 1;
    }
}

impl Drop for VisitedListHandle<'_> {
    fn drop(&mut self) {
        // Swap out inner and return it. Replace with dummy.
        let list = std::mem::replace(
            &mut self.inner,
            VisitedList {
                generation: 0,
                words: Vec::new(),
                tags: Vec::new(),
            },
        );
        self.pool.return_list(list);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_and_mark_basic() {
        let pool = VisitedPool::new(100);
        let mut handle = pool.get();
        assert!(!handle.check_and_mark(5));
        assert!(handle.check_and_mark(5));
        assert!(!handle.check_and_mark(6));
    }

    #[test]
    fn next_iteration_clears_lazily() {
        let pool = VisitedPool::new(100);
        let mut handle = pool.get();
        handle.check_and_mark(10);
        assert!(handle.check_and_mark(10));
        handle.next_iteration();
        // After new generation, previously visited IDs are no longer marked
        assert!(!handle.check_and_mark(10));
    }

    #[test]
    fn handle_returned_to_pool() {
        let pool = VisitedPool::new(100);
        {
            let _handle = pool.get();
        }
        // Should reuse the returned handle
        assert_eq!(pool.pool.lock().unwrap().len(), 1);
        let _handle2 = pool.get();
        assert_eq!(pool.pool.lock().unwrap().len(), 0);
    }

    #[test]
    fn boundary_ids() {
        let pool = VisitedPool::new(128);
        let mut handle = pool.get();
        // Test word boundaries: 0, 63, 64, 127
        for &id in &[0u32, 63, 64, 127] {
            assert!(!handle.check_and_mark(id));
            assert!(handle.check_and_mark(id));
        }
    }
}
