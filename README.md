# hashmap_get_many_mut

Get mutable references to the values of a `std::collections::HashMap` for a
**runtime-length** list of keys at the same time.

std's [`HashMap::get_disjoint_mut`](https://doc.rust-lang.org/std/collections/struct.HashMap.html#method.get_disjoint_mut)
takes `[&Q; N]`, so the number of keys must be known at compile time.
This crate adds `get_many_mut`, which takes a slice `&[&Q]`.

```rust
use std::collections::HashMap;
use hashmap_get_many_mut::GetManyMut;

let mut stock = HashMap::from([
    ("apple".to_string(), 10),
    ("banana".to_string(), 20),
    ("orange".to_string(), 30),
]);

// The number of keys is only known at runtime.
let keys: Vec<&str> = "banana apple missing".split(' ').collect();

let mut values = stock.get_many_mut(&keys);
for value in values.iter_mut().flatten() {
    **value += 100;
}
assert!(values[2].is_none()); // "missing" is not in the map

assert_eq!(stock["banana"], 120);
assert_eq!(stock["apple"], 110);
```

## Behavior

- The `i`-th element of the result corresponds to `keys[i]`; missing keys
  yield `None`.
- **Panics** if two or more keys refer to the same entry. Repeating a key that
  is not in the map does not panic.
- Duplicates are detected by the entries' **addresses**, not by comparing
  keys, so incorrect `Eq` / `Hash` / `Borrow` implementations can give wrong
  results but never aliasing `&mut`.
- The duplicate check is `O(n log n)` (std's fixed-size version is `O(n²)`).

### `get_many_unchecked_mut`

`unsafe fn get_many_unchecked_mut` skips the duplicate check and its extra
allocation, like std's `get_disjoint_unchecked_mut`. The caller must make sure
that no two keys refer to the same entry; otherwise it is undefined behavior,
even if the returned references are not used.

```rust
use std::collections::HashMap;
use hashmap_get_many_mut::GetManyMut;

let mut map = HashMap::from([("a", 1), ("b", 2), ("c", 3)]);

// SAFETY: "a" and "c" are distinct, so they refer to different entries.
let values = unsafe { map.get_many_unchecked_mut(&["a", "c"]) };
```

## Implementation note

The implementation looks up every key with `HashMap::get_mut`, keeps only raw
pointers until all lookups are done, checks them for duplicates, and only then
turns them into `&mut V`.

This relies on `HashMap::get_mut` not moving or writing to entries other than
the one it returns. std does not document this guarantee, but its
implementation (hashbrown) never relocates entries on lookup. The test suite
is run under Miri (Stacked Borrows and Tree Borrows), including a test that
forces hash collisions.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.
