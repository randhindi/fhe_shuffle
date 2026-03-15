# TODOs

## Add `oblivious_shuffle_with_seeds()` API for verifiable randomness

**Problem**: `shuffle_with_keys()` currently uses `rand::thread_rng()` to generate
OPRF seeds (shuffle.rs:184). In an adversarial setting (e.g., Solana smart contract),
a malicious server can choose seeds that produce colliding OPRF outputs, breaking
shuffle uniformity. This is the critical security gap for production deployment.

**Proposed API**:
```rust
pub fn oblivious_shuffle_with_seeds<T: Shuffleable>(
    data: Vec<T>,
    seeds: Vec<Seed>,
    key_precision: u32,
) -> Vec<T>
```

The caller provides externally-verified seeds (e.g., from a VRF or on-chain randomness
beacon). The existing `oblivious_shuffle()` remains as a convenience wrapper that
generates seeds internally.

**Blocked by**: Solana integration design (need to know how on-chain randomness will
be sourced — VRF, commit-reveal, randomness beacon, etc.)
