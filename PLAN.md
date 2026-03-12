# Plan: Generic oblivious_shuffle for TFHE-rs

## Public API

```rust
/// Shuffles encrypted data into a uniformly random permutation.
///
/// key_precision: bit-width of random sort keys (8, 16, 32, 64, or 128).
/// Higher precision = lower collision probability but slower comparisons.
pub fn oblivious_shuffle<T: Shuffleable>(data: Vec<T>, key_precision: u32) -> Vec<T>
```

## File Structure

```
src/
├── lib.rs          # Library root: pub mod + re-exports
├── network.rs      # Bitonic network topology (unchanged logic, minor cleanup)
├── shuffle.rs      # Generic shuffle: traits, impls, oblivious_shuffle()
└── main.rs         # Benchmarks for n=8, 16, 32 using public API
```

## Traits

### `Shuffleable` (public) — for data types that can be shuffled
```rust
pub trait Shuffleable: Sized + Send + Sync {
    fn trivial_zero() -> Self;
    fn conditional_swap(cond: &FheBool, a: &Self, b: &Self) -> (Self, Self);
}
```
Implemented for: FheUint{8,16,32,64,128}, FheInt{8,16,32,64,128}

### `SortKey` (private) — for sort key types
```rust
trait SortKey: Sized + Send + Sync {
    fn generate_random(seed: Seed) -> Self;
    fn trivial_max() -> Self;
    fn cmp_gt(&self, other: &Self) -> FheBool;
    fn cmp_lt(&self, other: &Self) -> FheBool;
    fn conditional_swap(cond: &FheBool, a: &Self, b: &Self) -> (Self, Self);
}
```
Implemented for: FheUint{8,16,32,64,128}

Both implemented via macros to avoid repetition.

## Core Implementation

```rust
// Public entry point — dispatches on key_precision
pub fn oblivious_shuffle<T: Shuffleable>(data: Vec<T>, key_precision: u32) -> Vec<T> {
    match key_precision {
        8 => shuffle_with_keys::<T, FheUint8>(data),
        16 => shuffle_with_keys::<T, FheUint16>(data),
        32 => shuffle_with_keys::<T, FheUint32>(data),
        64 => shuffle_with_keys::<T, FheUint64>(data),
        128 => shuffle_with_keys::<T, FheUint128>(data),
        _ => panic!("Unsupported key_precision: {}. Use 8, 16, 32, 64, or 128.", key_precision),
    }
}

// Generic core — works with any Shuffleable data + SortKey keys
fn shuffle_with_keys<D: Shuffleable, K: SortKey>(data: Vec<D>) -> Vec<D> {
    // 1. Generate n random sort keys via OPRF
    // 2. Pad to next power of 2 (trivial_max keys, trivial_zero data)
    // 3. Run bitonic network stages in parallel (rayon)
    // 4. Return first n elements
}
```

## Changes Per File

### 1. NEW: `src/lib.rs`
- `pub mod network;`
- `pub mod shuffle;`
- `pub use shuffle::{oblivious_shuffle, Shuffleable};`

### 2. REWRITE: `src/shuffle.rs`
- Remove all FheUint64-specific code
- Define `Shuffleable` trait (public)
- Define `SortKey` trait (crate-private)
- Implement both via macros for all supported types
- `oblivious_shuffle()` — public API, dispatches on key_precision
- `shuffle_with_keys()` — generic core, no stdout output (library-clean)
- Unit tests:
  - `test_shuffle_2_elements` — encrypted n=2, FheUint8 data, key_precision=8 (~10s)
  - `test_shuffle_4_elements` — encrypted n=4, FheUint64 data, key_precision=64 (#[ignore], ~30s)
  - `test_shuffle_different_key_precisions` — encrypted n=2, test 8/16/32/64 key widths
  - `test_shuffle_uint16_data` — encrypted n=2 with FheUint16 data

### 3. CLEANUP: `src/network.rs`
- Keep all existing functions and tests
- Add: `test_bitonic_sorts_8_elements`, `test_bitonic_sorts_32_elements`
- Add: `test_all_permutations_reachable_n4` — exhaustive check that all 24 permutations of 4 elements are reachable

### 4. UPDATE: `src/main.rs`
- Import from library: `use fhe_shuffle::{oblivious_shuffle, ...}`
- Benchmark `oblivious_shuffle` for n=8, 16, 32 with key_precision=64
- No `mod` declarations — uses library crate
- Keep the results table format

### 5. UPDATE: `Cargo.toml`
- No changes needed (Cargo auto-detects lib.rs + main.rs)
