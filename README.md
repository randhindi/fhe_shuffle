# FHE Shuffle

Oblivious permutation of encrypted data using [TFHE-rs](https://github.com/zama-ai/tfhe-rs). Shuffles `n` `FheUint64` ciphertexts into a uniformly random permutation — without the server ever learning the permutation applied. Supports any `n >= 2` (non-power-of-2 sizes are handled automatically via padding).

## Algorithm

The core idea: **reduce shuffling to sorting by random keys**. If you assign each element a uniformly random sort key and sort the (key, element) pairs, the resulting order of elements is a uniformly random permutation (provided all keys are distinct). We implement this entirely in FHE using a bitonic sorting network.

### Step 1 — Generate Oblivious Random Sort Keys

The server generates `n` encrypted random `FheUint64` values using TFHE's oblivious pseudo-random function (OPRF):

```
for i in 0..n:
    sort_keys[i] = FheUint64::generate_oblivious_pseudo_random(Seed(random_u128))
```

Each seed is a public random `u128`, but the OPRF evaluates a PRF under the FHE secret key — the resulting ciphertext encrypts a pseudorandom 64-bit value that **nobody knows** (not the server, not the client) until the client decrypts. This is what makes the shuffle oblivious: the server cannot learn which permutation was applied.

### Step 2 — Pair Keys with Data

Each data element gets associated with its random sort key:

```
pairs = [(sort_keys[0], data[0]), (sort_keys[1], data[1]), ..., (sort_keys[n-1], data[n-1])]
```

### Step 3 — Sort via Bitonic Network

We sort the `n` pairs by their encrypted keys using a **bitonic sorting network**. A bitonic network is a fixed sequence of compare-and-swap operations whose wiring is independent of the data — making it ideal for FHE where branching on encrypted values is impossible.

#### Non-Power-of-2 Support

The bitonic network requires a power-of-2 input size. When `n` is not a power of 2, the shuffle automatically pads to the next power of 2:

- **Padding sort keys**: Trivially encrypted `u64::MAX` values. Since the bitonic network sorts in ascending order, these maximum keys are guaranteed to sort to the end of the array.
- **Padding data**: Trivially encrypted zeros (value is irrelevant — padding is discarded after sorting).

After sorting, only the first `n` positions are returned. The real elements occupy these positions (sorted by their random OPRF keys = uniform permutation), while padding sits at the end.

For example, shuffling 10 elements uses a 16-element bitonic network (10 stages), with 6 padding slots. The cost is the same as shuffling 16 elements — the overhead comes from rounding up to the next power of 2.

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

// 2. Conditional swap — all 4 if_then_else run in parallel:
new_key[i] = cmp.if_then_else(key[j], key[i])     // if cmp: take j's key, else keep i's
new_key[j] = cmp.if_then_else(key[i], key[j])     // if cmp: take i's key, else keep j's
new_data[i] = cmp.if_then_else(data[j], data[i])   // if cmp: take j's data, else keep i's
new_data[j] = cmp.if_then_else(data[i], data[j])   // if cmp: take i's data, else keep j's
```

The `gt` comparison is the most expensive operation. Each `FheUint64` consists of 32 blocks of 2 encrypted bits each, and comparing two values requires a chain of programmable bootstrapping (PBS) operations across all 32 blocks.

The 4 `if_then_else` operations are cheaper (one PBS per block each) and are fully independent, so they run in parallel via nested `rayon::join`.

### Step 4 — Extract Shuffled Data

After all stages complete, the pairs are sorted by key. We discard the keys and any padding, returning the first `n` data elements, which are now in a uniformly random order:

```
result = [pairs[0].data, pairs[1].data, ..., pairs[n-1].data]
```

### Parallelism

The algorithm maximizes parallelism at two levels:

1. **Inter-comparator** (stage-level): All 8 comparators in each stage operate on disjoint pairs and execute in parallel via `rayon::par_iter()`.

2. **Intra-comparator**: Within each comparator, the 4 `if_then_else` calls are independent and run concurrently via nested `rayon::join()` (2 keys + 2 data values, structured as two pairs of two).

The overall depth is **10 sequential stages**. The width is **8 parallel comparators × 4 parallel if_then_else = 32-way parallelism** within each stage (plus the `gt` comparison which must complete first).

### Why Not Benes Network?

A Benes permutation network has optimal depth (`2k-1 = 7` stages for `n=16`) and only needs binary control bits (swap or don't swap), making each comparator cheaper (2 × `if_then_else`, no `gt`). However, it **cannot produce uniform permutations from random control bits**.

The problem: a Benes network for `n=16` has 56 control bits, giving `2^56` possible configurations. These map onto `16! ≈ 2^44.25` permutations. Since `2^56` does not divide `16!`, the mapping is necessarily non-uniform — some permutations are reached by more bit patterns than others.

The bitonic sort approach avoids this entirely. With distinct sort keys, each of the `16!` possible orderings of keys maps to exactly one permutation. The only source of non-uniformity is key collisions, which occur with negligible probability.

## Uniformity Analysis

The shuffle is a **uniformly random permutation** conditioned on all 16 sort keys being distinct.

**Collision probability** (birthday bound):

```
P(at least one collision) = 1 - Product(i=0..15) of (2^64 - i) / 2^64
                          ≤ C(16, 2) / 2^64
                          = 120 / 2^64
                          ≈ 6.5 × 10^-18
                          < 2^-57
```

This means:
- **Statistical distance from uniform**: `< 2^-57`
- **Bias per permutation**: for any specific permutation π, `|Pr[output = π] - 1/16!| < 2^-57 / 16!`
- This is well within the `< 2^-40` security bound

In practice, you would need to run the shuffle roughly `2^57` times before expecting a single key collision. Even in the (astronomically unlikely) event of a collision, the output is still a valid permutation of the input — just not drawn from the perfectly uniform distribution.

### FHE Operations Per Shuffle

For `n=16`:

| Operation | Count | Notes |
|-----------|-------|-------|
| OPRF (key generation) | 16 | One `FheUint64` per element |
| `gt` / `lt` comparisons | 80 | 10 stages × 8 comparators |
| `if_then_else` | 320 | 80 comparators × 4 each |
| **Total PBS** | **~12,800** | Each `FheUint64` = 32 blocks of 2 bits |

### Scaling to Other Sizes

| n | Padded to | Stages | Comparators | `gt` | `if_then_else` | Collision bound |
|---|-----------|--------|-------------|------|----------------|-----------------|
| 4 | 4 | 3 | 6 | 6 | 24 | < 2^-61 |
| 8 | 8 | 6 | 24 | 24 | 96 | < 2^-59 |
| 10 | 16 | 10 | 80 | 80 | 320 | < 2^-58 |
| 16 | 16 | 10 | 80 | 80 | 320 | < 2^-57 |
| 20 | 32 | 15 | 240 | 240 | 960 | < 2^-56 |
| 32 | 32 | 15 | 240 | 240 | 960 | < 2^-55 |
| 52 | 64 | 21 | 672 | 672 | 2688 | < 2^-54 |
| 64 | 64 | 21 | 672 | 672 | 2688 | < 2^-53 |

Non-power-of-2 sizes are padded to the next power of 2, so `n=10` has the same cost as `n=16`. The collision bound degrades gracefully: even at `n=1024`, `P(collision) < 2^-44`, still well within `2^-40`.

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
cargo test --release
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
    ├── main.rs             # Benchmark runner (CPU + GPU)
    ├── network.rs          # Bitonic sorting network topology
    └── shuffle.rs          # FHE shuffle using bitonic sort
```

## Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `tfhe` | 1.5.3 | FHE operations (`FheUint64`, OPRF, `if_then_else`, `gt`) |
| `rayon` | 1.10 | Parallel comparators within each stage |
| `rand` | 0.8 | Seeds for OPRF generation |
