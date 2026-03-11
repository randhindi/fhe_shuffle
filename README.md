# FHE Shuffle

Oblivious permutation of encrypted data using [TFHE-rs](https://github.com/zama-ai/tfhe-rs). Shuffles 16 `FheUint64` ciphertexts into a uniformly random permutation — without the server ever learning the permutation applied.

## How It Works

The shuffle uses a **bitonic sorting network** with **oblivious pseudo-random sort keys**:

1. **Generate random keys**: The server generates 16 encrypted random `FheUint64` values using TFHE's oblivious pseudo-random function (OPRF). Each value is encrypted — nobody (not even the server) knows the plaintext keys until the client decrypts.

2. **Sort by random keys**: The 16 (key, data) pairs are sorted using a fixed bitonic sorting network. The network topology is data-independent: the same sequence of compare-and-swap operations runs regardless of the encrypted values, making the entire computation oblivious.

3. **Output**: After sorting, the data elements are in a uniformly random order determined by the random keys.

### Bitonic Sorting Network

A bitonic network for `n = 2^k` elements has `k(k+1)/2` stages, each containing `n/2` independent comparators:

| n | Stages | Comparators/stage | Total comparators |
|---|--------|-------------------|-------------------|
| 2 | 1 | 1 | 1 |
| 4 | 3 | 2 | 6 |
| 8 | 6 | 4 | 24 |
| 16 | 10 | 8 | 80 |

Each comparator performs:
- 1 × `gt` (or `lt`) — compare the two sort keys
- 4 × `if_then_else` — conditionally swap both the key and the data element

All 8 comparators within a stage are independent and run in parallel via [rayon](https://github.com/rayon-rs/rayon). The 4 `if_then_else` calls within each comparator also run in parallel using nested `rayon::join`.

### Why Not Benes Network?

A Benes permutation network has optimal depth (`2k-1 = 7` stages for `n=16`) but **cannot produce uniform permutations from random control bits**. The issue: `2^56` random bit patterns map non-uniformly onto `16! ≈ 2^44.25` permutations. Some permutations are reached by more bit patterns than others, creating measurable bias.

The bitonic sort approach avoids this entirely — any distinct set of sort keys maps to exactly one permutation.

## Uniformity Analysis

The shuffle is a **uniformly random permutation** conditioned on all 16 sort keys being distinct.

**Collision probability** (birthday bound):

```
P(collision) = C(16, 2) / 2^64 = 120 / 2^64 ≈ 6.5 × 10^-18 < 2^-57
```

This means:
- **Statistical distance from uniform**: `< 2^-57`
- **Bias**: for any specific permutation π, `|Pr[output = π] - 1/16!| < 2^-57 / 16!`
- This is well within the `< 2^-40` security bound

In practice, you would need to run the shuffle roughly `2^57` times before observing a single key collision (and even then, the output is still a valid permutation — just not drawn from the perfectly uniform distribution).

### FHE Operations Per Shuffle

For `n=16`:

| Operation | Count | Notes |
|-----------|-------|-------|
| OPRF (key generation) | 16 | One `FheUint64` per element |
| `gt` / `lt` comparisons | 80 | 10 stages × 8 comparators |
| `if_then_else` | 320 | 80 comparators × 4 each |
| **Total PBS** | **~12,800** | Each `FheUint64` = 32 blocks of 2 bits |

## Running Benchmarks

### Prerequisites

- Rust 1.70+
- For GPU: NVIDIA GPU with CUDA toolkit installed

### CPU

```bash
cd fhe_shuffle
source environment/dev.env
cargo run --release
```

### GPU (CUDA)

```bash
cd fhe_shuffle
source environment/dev.env
cargo run --release --features gpu
```

### Tests

```bash
cd fhe_shuffle
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
