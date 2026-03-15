# FHE Shuffle

Oblivious permutation of encrypted data using [TFHE-rs](https://github.com/zama-ai/tfhe-rs). Shuffles `n` encrypted ciphertexts into a uniformly random permutation — without the server ever learning the permutation applied. Works with all TFHE-rs encrypted integer types (`FheUint8` through `FheUint128`, `FheInt8` through `FheInt128`). Supports any `n >= 2` (non-power-of-2 sizes are handled automatically via padding).

## Quick Start

```rust
use tfhe::{prelude::*, set_server_key, ConfigBuilder, FheUint64};
use fhe_shuffle::oblivious_shuffle;

let config = ConfigBuilder::default().build();
let (client_key, server_key) = tfhe::generate_keys(config);
set_server_key(server_key);

let values: Vec<FheUint64> = (0..16u64)
    .map(|i| FheUint64::encrypt(i, &client_key))
    .collect();

// Shuffle with 32-bit sort keys (collision probability < 2^-25 for n=16)
let shuffled = oblivious_shuffle(values, 32);
```

## Key Precision

The `key_precision` parameter controls the bit-width of the random sort keys used internally. Higher precision = lower collision probability but slower comparisons:

| key_precision | Key type    | Collision bound (n=16) | Relative speed |
|---------------|-------------|------------------------|----------------|
| 8             | FheUint8    | ~2^-1 (NOT recommended)| ~4x faster     |
| 16            | FheUint16   | ~2^-9                  | ~2x faster     |
| 32            | FheUint32   | ~2^-25                 | baseline       |
| 64            | FheUint64   | < 2^-57                | ~1x (same)     |
| 128           | FheUint128  | < 2^-121               | slower         |

**Recommendation:** Use **32-bit keys** for most applications. Negligible bias for N ≤ 10,000 with ~2x speedup over 64-bit. Use 16-bit keys only if N ≤ 32 and performance is critical.

## Algorithm

The core idea: **reduce shuffling to sorting by random keys**. If you assign each element a uniformly random sort key and sort the (key, element) pairs, the resulting order of elements is a uniformly random permutation (provided all keys are distinct). We implement this entirely in FHE using a bitonic sorting network.

### Step 1 — Generate Oblivious Random Sort Keys

The server generates `n` encrypted random values using TFHE's oblivious pseudo-random function (OPRF):

```
for i in 0..n:
    sort_keys[i] = FheUintXX::generate_oblivious_pseudo_random(Seed(random_u128))
```

Each seed is a public random `u128`, but the OPRF evaluates a PRF under the FHE secret key — the resulting ciphertext encrypts a pseudorandom value that **nobody knows** (not the server, not the client) until the client decrypts. This is what makes the shuffle oblivious: the server cannot learn which permutation was applied.

### Step 2 — Pair Keys with Data

Each data element gets associated with its random sort key:

```
pairs = [(sort_keys[0], data[0]), (sort_keys[1], data[1]), ..., (sort_keys[n-1], data[n-1])]
```

### Step 3 — Sort via Bitonic Network

We sort the `n` pairs by their encrypted keys using a **bitonic sorting network**. A bitonic network is a fixed sequence of compare-and-swap operations whose wiring is independent of the data — making it ideal for FHE where branching on encrypted values is impossible.

#### Non-Power-of-2 Support

The bitonic network requires a power-of-2 input size. When `n` is not a power of 2, the shuffle automatically pads to the next power of 2:

- **Padding sort keys**: Trivially encrypted maximum values. Since the bitonic network sorts in ascending order, these maximum keys are guaranteed to sort to the end of the array.
- **Padding data**: Trivially encrypted zeros (value is irrelevant — padding is discarded after sorting).

After sorting, only the first `n` positions are returned. **n=17 costs the same as n=32** — choose power-of-2 sizes when possible.

#### What is a Bitonic Network?

A bitonic sorting network sorts by recursively building **bitonic sequences** (sequences that first increase then decrease, or vice versa) and then merging them. For `n = 2^k` elements, the network consists of `k(k+1)/2` stages, each containing `n/2` independent comparators.

For `n = 16` (`k = 4`): **10 stages**, each with **8 comparators** = **80 comparators** total.

The network is organized into 4 phases:

```
Phase 0 (k=1): 1 stage   — sorts pairs of 2 into bitonic sequences of 4
Phase 1 (k=2): 2 stages  — merges into bitonic sequences of 8
Phase 2 (k=3): 3 stages  — merges into bitonic sequences of 16
Phase 3 (k=4): 4 stages  — final merge, produces fully sorted output
                           ──────
                           10 stages total
```

Each stage consists of 8 comparators that operate on disjoint pairs. The pair indices are determined by XOR masks — in stage `(phase, step)`, element `i` is compared with element `j = i XOR (1 << step)`:

```
Stage  1: (0,1) (2,3) (4,5) (6,7) (8,9) (10,11) (12,13) (14,15)   step=1
Stage  2: (0,2) (1,3) (4,6) (5,7) (8,10) (9,11) (12,14) (13,15)   step=2
Stage  3: (0,1) (2,3) (4,5) (6,7) (8,9) (10,11) (12,13) (14,15)   step=1
Stage  4: (0,4) (1,5) (2,6) (3,7) (8,12) (9,13) (10,14) (11,15)   step=4
Stage  5: (0,2) (1,3) (4,6) (5,7) (8,10) (9,11) (12,14) (13,15)   step=2
Stage  6: (0,1) (2,3) (4,5) (6,7) (8,9) (10,11) (12,13) (14,15)   step=1
Stage  7: (0,8) (1,9) (2,10) (3,11) (4,12) (5,13) (6,14) (7,15)   step=8
Stage  8: (0,4) (1,5) (2,6) (3,7) (8,12) (9,13) (10,14) (11,15)   step=4
Stage  9: (0,2) (1,3) (4,6) (5,7) (8,10) (9,11) (12,14) (13,15)   step=2
Stage 10: (0,1) (2,3) (4,5) (6,7) (8,9) (10,11) (12,13) (14,15)   step=1
```

Each comparator also has a **direction** (ascending or descending) determined by `(i >> (phase + 1)) & 1`. This alternating direction is what creates the bitonic merge structure.

#### What Each Comparator Does (in FHE)

Each comparator takes two (key, data) pairs at positions `i` and `j` and conditionally swaps them. In plaintext this would be a simple `if key[i] > key[j] then swap`, but in FHE we cannot branch on encrypted values. Instead:

```
// 1. Compare the encrypted keys (produces an encrypted boolean)
cmp = key[i].gt(&key[j])       // or .lt() depending on direction

// 2. Conditional swap using flip — both flips run in parallel:
(new_key[i], new_key[j]) = cmp.flip(key[i], key[j])    // swap keys if cmp is true
(new_data[i], new_data[j]) = cmp.flip(data[i], data[j])  // swap data if cmp is true
```

The `gt` comparison is the most expensive operation — it requires a chain of programmable bootstrapping (PBS) operations across all blocks.

The `flip` operator performs a conditional swap of two encrypted values: `flip(true, a, b) = (b, a)` and `flip(false, a, b) = (a, b)`. The two `flip` calls (keys and data) are independent and run in parallel via `rayon::join`.

### Step 4 — Extract Shuffled Data

After all stages complete, the pairs are sorted by key. We discard the keys and any padding, returning the first `n` data elements, which are now in a uniformly random order:

```
result = [pairs[0].data, pairs[1].data, ..., pairs[n-1].data]
```

### Parallelism

The algorithm maximizes parallelism at two levels:

1. **Inter-comparator** (stage-level): All comparators in each stage operate on disjoint pairs and execute in parallel via `rayon::par_iter()`.

2. **Intra-comparator**: Within each comparator, the 2 `flip` calls (keys and data) are independent and run concurrently via `rayon::join()`.

The overall depth is the number of sequential stages. The width is `n/2` parallel comparators per stage.

### Why Not Benes Network?

A Benes permutation network has optimal depth (`2k-1 = 7` stages for `n=16`) and only needs binary control bits (swap or don't swap), making each comparator cheaper (2 × `if_then_else`, no `gt`). However, it **cannot produce uniform permutations from random control bits**.

The problem: a Benes network for `n=16` has 56 control bits, giving `2^56` possible configurations. These map onto `16! ≈ 2^44.25` permutations. Since `2^56` does not divide `16!`, the mapping is necessarily non-uniform — some permutations are reached by more bit patterns than others.

The bitonic sort approach avoids this entirely. With distinct sort keys, each of the `16!` possible orderings of keys maps to exactly one permutation. The only source of non-uniformity is key collisions, which occur with negligible probability.

## Uniformity Analysis

The shuffle is a **uniformly random permutation** conditioned on all sort keys being distinct.

**Collision probability** (birthday bound) for n elements with k-bit keys:

```
P(at least one collision) ≈ n² / (2 × 2^k)
```

| n | 16-bit keys | 32-bit keys | 64-bit keys |
|---|-------------|-------------|-------------|
| 8 | 0.04% | ~0% | ~0% |
| 16 | 0.2% | ~0% | ~0% |
| 32 | 0.8% | ~0% | ~0% |
| 256 | **39%** | 0.0008% | ~0% |
| 1024 | **~100%** | 0.012% | ~0% |
| 65536 | broken | **39%** | ~0% |

Even in the (astronomically unlikely) event of a collision, the output is still a valid permutation — just not drawn from the perfectly uniform distribution.

### Scaling

| n | Padded to | Stages | Comparators | Collision bound (32-bit) |
|---|-----------|--------|-------------|--------------------------|
| 4 | 4 | 3 | 6 | ~0% |
| 8 | 8 | 6 | 24 | ~0% |
| 16 | 16 | 10 | 80 | ~0% |
| 32 | 32 | 15 | 240 | ~0% |
| 64 | 64 | 21 | 672 | ~0% |
| 128 | 128 | 28 | 1792 | 0.0002% |
| 256 | 256 | 36 | 4608 | 0.0008% |

## Running Benchmarks

### Prerequisites

- Rust 1.70+
- For GPU: NVIDIA GPU with CUDA toolkit installed

### CPU

```bash
source environment/dev.env
cargo run --release
```

### GPU (CUDA)

```bash
source environment/dev.env
cargo run --release --features gpu
```

### Tests

```bash
source environment/dev.env
cargo test --release                  # fast tests (~20s)
cargo test --release -- --ignored     # includes slow n=8 and n=3 tests
```

The `environment/dev.env` file sets `RUSTFLAGS="-C target-cpu=native"` to enable AVX2/AVX-512 optimizations on the host CPU.

## Project Structure

```
fhe_shuffle/
├── Cargo.toml              # Dependencies: tfhe 1.5.3, rayon, rand
├── environment/
│   └── dev.env             # RUSTFLAGS for native CPU features
├── README.md
└── src/
    ├── lib.rs              # Library root: re-exports oblivious_shuffle + Shuffleable
    ├── shuffle.rs          # Generic shuffle: traits, impls, oblivious_shuffle()
    ├── network.rs          # Bitonic sorting network topology
    └── main.rs             # Multi-precision benchmark runner (CPU + GPU)
```

## Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `tfhe` | 1.5.3 | FHE operations (OPRF, `flip`, `gt`, encrypted integers) |
| `rayon` | 1.10 | Parallel comparators within each stage |
| `rand` | 0.8 | Seeds for OPRF generation |
