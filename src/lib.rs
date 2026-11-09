//! Get mutable references to the values of a [`HashMap`] for a
//! **runtime-length** list of keys at the same time.
//!
//! [`HashMap::get_disjoint_mut`] in std only accepts a fixed number of keys
//! (`[&Q; N]`, where `N` must be known at compile time). This crate adds
//! [`GetManyMut::get_many_mut`], which takes a slice of keys whose length is
//! decided at runtime.
//!
//! ```
//! use std::collections::HashMap;
//! use hashmap_get_many_mut::GetManyMut;
//!
//! let mut stock = HashMap::from([
//!     ("apple".to_string(), 10),
//!     ("banana".to_string(), 20),
//!     ("orange".to_string(), 30),
//! ]);
//!
//! // The number of keys is only known at runtime.
//! let keys: Vec<&str> = "banana apple missing".split(' ').collect();
//!
//! let mut values = stock.get_many_mut(&keys);
//! for value in values.iter_mut().flatten() {
//!     **value += 100;
//! }
//! assert!(values[2].is_none()); // "missing" is not in the map
//!
//! assert_eq!(stock["banana"], 120);
//! assert_eq!(stock["apple"], 110);
//! assert_eq!(stock["orange"], 30);
//! ```

#![deny(missing_docs)]

use std::borrow::Borrow;
use std::collections::HashMap;
use std::hash::{BuildHasher, Hash};
use std::ptr::NonNull;

mod sealed {
    pub trait Sealed {}

    impl<K, V, S> Sealed for std::collections::HashMap<K, V, S> {}
}

/// Extension trait that adds [`get_many_mut`](GetManyMut::get_many_mut) to
/// [`HashMap`].
///
/// This trait is sealed and cannot be implemented outside this crate.
pub trait GetManyMut<K, V>: sealed::Sealed {
    /// Returns mutable references to the values for all `keys` at once.
    ///
    /// The `i`-th element of the returned `Vec` corresponds to `keys[i]`, and
    /// is `None` if that key is not in the map.
    ///
    /// Unlike [`HashMap::get_disjoint_mut`], the number of keys does not have
    /// to be known at compile time.
    ///
    /// # Panics
    ///
    /// Panics if two or more of `keys` refer to the same entry in the map.
    /// Repeating a key that is *not* in the map does not panic (every
    /// occurrence just yields `None`), which matches
    /// [`HashMap::get_disjoint_mut`].
    ///
    /// ```should_panic
    /// use std::collections::HashMap;
    /// use hashmap_get_many_mut::GetManyMut;
    ///
    /// let mut map = HashMap::from([("a", 1), ("b", 2)]);
    /// let _ = map.get_many_mut(&["a", "b", "a"]); // panics: "a" twice
    /// ```
    ///
    /// # Complexity
    ///
    /// In addition to one lookup per key, the duplicate check sorts the
    /// found entries' addresses: `O(n log n)` time and one extra allocation
    /// of `n` pointers, where `n = keys.len()`. (std's `get_disjoint_mut`
    /// uses an `O(n²)` check, which is fine for its intended small, fixed `n`.)
    fn get_many_mut<Q>(&mut self, keys: &[&Q]) -> Vec<Option<&mut V>>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized;
}

impl<K, V, S> GetManyMut<K, V> for HashMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    fn get_many_mut<Q>(&mut self, keys: &[&Q]) -> Vec<Option<&mut V>>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        // Collect everything first, holding only raw pointers while looking up.
        // If we created `&mut V` while continuing to look up, a later `get_mut`
        // could read an earlier entry through a shared reference while probing,
        // invalidating the `&mut V` handed out before (UB under Stacked Borrows;
        // `tests::with_hash_collisions` makes Miri detect this).
        //
        // This relies on `HashMap::get_mut` not moving or writing to entries
        // other than the one it returns. std does not document this, but its
        // implementation (hashbrown) never relocates entries on lookup.
        let ptrs: Vec<Option<NonNull<V>>> = keys
            .iter()
            .map(|&key| self.get_mut(key).map(NonNull::from))
            .collect();

        if ptrs.len() >= 2 {
            // Detect duplicates by address, not by key (`Q: Eq`).
            // `Borrow`, `Eq` and `Hash` are safe traits and may be implemented
            // incorrectly; relying on them could hand out aliasing `&mut V`.
            let mut sorted: Vec<NonNull<V>> = ptrs.iter().flatten().copied().collect();
            sorted.sort_unstable();
            assert!(
                sorted.windows(2).all(|w| w[0] != w[1]),
                "get_many_mut: duplicate keys"
            );
        }

        // Option<NonNull<V>> and Option<&mut V> have the same layout, so this
        // collect reuses `ptrs`' allocation (std's in-place collect
        // optimization; not guaranteed, but no new allocation happens today).
        ptrs.into_iter()
            .map(|p| {
                p.map(|mut p| {
                    // SAFETY: every pointer refers to a distinct entry (checked
                    // by address above), each came from `get_mut` on `self`,
                    // which is borrowed mutably for the returned lifetime, and
                    // `self` is not accessed again here, so no `&mut V`
                    // created below can be invalidated or aliased.
                    unsafe { p.as_mut() }
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::Hasher;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    fn sample() -> HashMap<String, i32> {
        HashMap::from([
            ("a".to_string(), 1),
            ("b".to_string(), 2),
            ("c".to_string(), 3),
        ])
    }

    #[test]
    fn success() {
        let mut map = sample();
        let mut values = map.get_many_mut(&["a", "b"]);

        if let [Some(a), Some(b)] = values.as_mut_slice() {
            **a += 10;
            **b += 20;
        } else {
            panic!("unexpected result");
        }

        assert_eq!(map["a"], 11);
        assert_eq!(map["b"], 22);
        assert_eq!(map["c"], 3);
    }

    #[test]
    fn missing_keys_are_none() {
        let mut map = sample();
        let values = map.get_many_mut(&["x", "b", "y"]);
        assert!(values[0].is_none());
        assert_eq!(values[1].as_deref(), Some(&2));
        assert!(values[2].is_none());
    }

    #[test]
    fn empty_and_single() {
        let mut map = sample();
        assert!(map.get_many_mut::<str>(&[]).is_empty());

        let mut values = map.get_many_mut(&["c"]);
        **values[0].as_mut().unwrap() = 30;
        assert_eq!(map["c"], 30);
    }

    #[test]
    #[should_panic(expected = "duplicate keys")]
    fn duplicate_keys_panic() {
        let mut map = sample();
        let _ = map.get_many_mut(&["a", "a"]);
    }

    #[test]
    fn duplicate_missing_keys_do_not_panic() {
        let mut map = sample();
        let values = map.get_many_mut(&["x", "a", "x"]);
        assert!(values[0].is_none());
        assert!(values[2].is_none());
    }

    #[test]
    fn many_keys() {
        let mut map: HashMap<u32, u32> = (0..200).map(|i| (i, i)).collect();
        let keys: Vec<u32> = (0..200).rev().step_by(3).collect();
        let key_refs: Vec<&u32> = keys.iter().collect();

        for value in map.get_many_mut(&key_refs).into_iter().flatten() {
            *value += 1000;
        }

        for i in 0..200 {
            let expected = if keys.contains(&i) { i + 1000 } else { i };
            assert_eq!(map[&i], expected);
        }
    }

    /// A key type whose `Eq` and `Hash` disagree with its `Borrow` target:
    /// the map compares `Key`s case-insensitively, but lookups go through
    /// `str`'s exact comparison. This breaks the `Borrow` contract; the result
    /// may be wrong, but it must never produce aliasing `&mut`.
    #[derive(Debug)]
    struct Key(String);

    impl PartialEq for Key {
        fn eq(&self, other: &Self) -> bool {
            self.0.eq_ignore_ascii_case(&other.0)
        }
    }
    impl Eq for Key {}
    impl Hash for Key {
        fn hash<H: Hasher>(&self, state: &mut H) {
            self.0.to_ascii_lowercase().hash(state);
        }
    }
    impl Borrow<str> for Key {
        fn borrow(&self) -> &str {
            &self.0
        }
    }

    #[test]
    fn broken_borrow_impl_never_aliases() {
        let mut map: HashMap<Key, i32, ConstState> = HashMap::with_hasher(ConstState);
        map.insert(Key("a".to_string()), 1);
        map.insert(Key("b".to_string()), 2);

        // Whatever happens (a result or a panic), it must not be UB;
        // Miri checks this.
        let _ = catch_unwind(AssertUnwindSafe(|| {
            let mut values = map.get_many_mut(&["a", "b", "A"]);
            for value in values.iter_mut().flatten() {
                **value += 1;
            }
        }));
    }

    /// A hasher that maps every key to the same hash value.
    /// It forces collisions so that every `get_mut` compares keys of other
    /// entries while probing.
    #[derive(Default, Clone)]
    struct ConstHasher;

    impl Hasher for ConstHasher {
        fn finish(&self) -> u64 {
            0
        }
        fn write(&mut self, _: &[u8]) {}
    }

    #[derive(Default, Clone)]
    struct ConstState;

    impl BuildHasher for ConstState {
        type Hasher = ConstHasher;
        fn build_hasher(&self) -> ConstHasher {
            ConstHasher
        }
    }

    /// Regression test: returned `&mut V` must not be invalidated by later
    /// lookups.
    ///
    /// An implementation that creates `&mut V` while it keeps looking up keys
    /// (e.g. a single iterator chain without the intermediate `collect`) lets a
    /// later `get_mut` read an earlier entry through a shared reference and
    /// invalidate the `&mut V` handed out before. With `RandomState` that
    /// almost never happens because collisions are rare, so this test forces
    /// collisions and relies on Miri (Stacked Borrows) to catch it.
    #[test]
    fn with_hash_collisions() {
        // Try several orders so the result does not depend on probe order.
        // "x" is not in the map.
        let orders = [
            ["a", "b", "c", "x"],
            ["c", "b", "a", "x"],
            ["b", "x", "c", "a"],
        ];

        for keys in orders {
            let mut map: HashMap<String, i32, ConstState> = HashMap::with_hasher(ConstState);
            map.insert("a".to_string(), 0);
            map.insert("b".to_string(), 1);
            map.insert("c".to_string(), 2);

            let mut values = map.get_many_mut(&keys);

            for (key, value) in keys.iter().zip(&values) {
                assert_eq!(value.is_some(), *key != "x");
            }

            // Write only after all references were obtained; Miri reports
            // here if any of them was invalidated.
            for value in values.iter_mut().flatten() {
                **value += 10;
            }

            assert_eq!(map["a"], 10);
            assert_eq!(map["b"], 11);
            assert_eq!(map["c"], 12);
        }
    }

    #[test]
    fn duplicate_keys_with_hash_collisions_panic() {
        let mut map: HashMap<String, i32, ConstState> = HashMap::with_hasher(ConstState);
        map.insert("a".to_string(), 0);
        map.insert("b".to_string(), 1);

        let result = catch_unwind(AssertUnwindSafe(|| {
            let _ = map.get_many_mut(&["b", "a", "b"]);
        }));
        assert!(result.is_err());
    }
}
