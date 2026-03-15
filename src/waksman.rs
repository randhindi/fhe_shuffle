//! Waksman permutation network shuffle.
//!
//! Uses a Waksman network with OPRF-generated random control bits to produce
//! a near-uniform permutation of encrypted data. Each switch is a single `flip`
//! (conditional swap) — no expensive `gt` comparisons needed.
//!
//! A single pass of random bits does NOT produce a uniform permutation.
//! Multiple passes are composed to reduce bias. Use `waksman_tv_distance()`
//! (in tests) to empirically measure bias for a given (N, passes) pair.

use rayon::prelude::*;
use tfhe::Seed;

use crate::network::padded_size;
use crate::shuffle::Shuffleable;

// ---------------------------------------------------------------------------
// Network topology
// ---------------------------------------------------------------------------

/// Generates a Waksman permutation network for n elements (n must be a power of 2, n >= 2).
///
/// Returns stages of disjoint `(i, j)` switch pairs. Each switch conditionally
/// swaps elements i and j based on a control bit. All switches in a stage operate
/// on disjoint indices and can execute in parallel.
///
/// Recursive construction:
/// - Base case N=2: single switch `(0, 1)`
/// - N>2: left column (N/2 switches pairing consecutive elements),
///   two recursive sub-networks of size N/2 (on even/odd outputs),
///   right column (same pairing as left)
///
/// Depth: `2*log2(N) - 1` stages.
///
/// We include ALL switches (no fixed-switch optimization from canonical Waksman)
/// because more switches = more mixing from random bits.
pub fn waksman_network(n: usize) -> Vec<Vec<(usize, usize)>> {
    assert!(n.is_power_of_two() && n >= 2);
    if n == 2 {
        return vec![vec![(0, 1)]];
    }

    let depth = 2 * (n.trailing_zeros() as usize) - 1;
    let mut stages: Vec<Vec<(usize, usize)>> = (0..depth).map(|_| Vec::new()).collect();

    build_waksman(&mut stages, &(0..n).collect::<Vec<_>>(), 0);

    stages
}

/// Recursively fill stages for a Waksman sub-network on the given element indices.
fn build_waksman(
    stages: &mut [Vec<(usize, usize)>],
    indices: &[usize],
    stage_offset: usize,
) {
    let n = indices.len();
    if n == 2 {
        stages[stage_offset].push((indices[0], indices[1]));
        return;
    }

    // Left column: pair consecutive elements
    for i in (0..n).step_by(2) {
        stages[stage_offset].push((indices[i], indices[i + 1]));
    }

    // Split into top (even-indexed outputs) and bottom (odd-indexed outputs)
    let top: Vec<usize> = (0..n).step_by(2).map(|i| indices[i]).collect();
    let bottom: Vec<usize> = (1..n).step_by(2).map(|i| indices[i]).collect();

    // Recurse on both halves (they use disjoint indices, so they share stages)
    build_waksman(stages, &top, stage_offset + 1);
    build_waksman(stages, &bottom, stage_offset + 1);

    // Right column: same pairing as left
    // Right column stage = stage_offset + 1 + depth_of_sub_network
    let sub_depth = 2 * ((n / 2).trailing_zeros() as usize) - 1;
    let right_stage = stage_offset + 1 + sub_depth;

    for i in (0..n).step_by(2) {
        stages[right_stage].push((indices[i], indices[i + 1]));
    }
}

/// Depth (number of sequential stages) of a Waksman network for n elements.
pub fn waksman_depth(n: usize) -> usize {
    assert!(n.is_power_of_two() && n >= 2);
    if n == 2 {
        1
    } else {
        2 * (n.trailing_zeros() as usize) - 1
    }
}

/// Total number of switches in our Waksman network for n elements.
pub fn waksman_switch_count(n: usize) -> usize {
    assert!(n.is_power_of_two() && n >= 2);
    let network = waksman_network(n);
    network.iter().map(|s| s.len()).sum()
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Shuffles encrypted data using a Waksman permutation network with random control bits.
///
/// Each pass applies the full Waksman network with fresh OPRF-generated encrypted
/// boolean control bits. Multiple passes are composed to reduce bias toward uniformity.
///
/// # Arguments
///
/// * `data` - Encrypted values to shuffle (length >= 2)
/// * `num_passes` - Number of Waksman network passes. More passes = closer to uniform.
///
/// # Panics
///
/// Panics if `data.len() < 2` or `num_passes < 1`.
pub fn waksman_shuffle<T: Shuffleable>(data: Vec<T>, num_passes: u32) -> Vec<T> {
    assert!(data.len() >= 2, "Need at least 2 elements to shuffle");
    assert!(num_passes >= 1, "Need at least 1 pass");

    let n = data.len();
    let padded_n = padded_size(n);
    let network = waksman_network(padded_n);
    let switches_per_pass: usize = network.iter().map(|s| s.len()).sum();

    // Initialize working array with Option for take/put pattern
    let mut data: Vec<Option<T>> = data.into_iter().map(Some).collect();
    for _ in 0..(padded_n - n) {
        data.push(Some(T::trivial_zero()));
    }

    for _pass in 0..num_passes {
        // Generate random control bits for all switches in this pass
        let bits = generate_random_bits(switches_per_pass);

        // Pre-compute cumulative bit offsets per stage
        let mut bit_offset = 0;
        for stage in &network {
            let stage_size = stage.len();

            let pairs: Vec<_> = stage
                .iter()
                .enumerate()
                .map(|(k, &(i, j))| {
                    let di = data[i].take().unwrap();
                    let dj = data[j].take().unwrap();
                    (i, j, di, dj, bit_offset + k)
                })
                .collect();

            let results: Vec<_> = pairs
                .into_par_iter()
                .map(|(i, j, di, dj, bit_idx)| {
                    let (new_di, new_dj) = T::conditional_swap(&bits[bit_idx], &di, &dj);
                    (i, j, new_di, new_dj)
                })
                .collect();

            for (i, j, new_di, new_dj) in results {
                data[i] = Some(new_di);
                data[j] = Some(new_dj);
            }

            bit_offset += stage_size;
        }
    }

    data.into_iter().take(n).map(|x| x.unwrap()).collect()
}

// ---------------------------------------------------------------------------
// Control bit generation
// ---------------------------------------------------------------------------

/// Generate `count` encrypted random booleans via OPRF.
///
/// Tries FheBool OPRF directly; falls back to FheUint8 LSB extraction.
fn generate_random_bits(count: usize) -> Vec<tfhe::FheBool> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let seeds: Vec<Seed> = (0..count).map(|_| Seed(rng.gen::<u128>())).collect();

    seeds
        .into_par_iter()
        .map(|seed| tfhe::FheBool::generate_oblivious_pseudo_random(seed))
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use std::collections::HashSet;
    use std::sync::OnceLock;
    use tfhe::{prelude::*, set_server_key, ClientKey, ConfigBuilder, FheUint64, ServerKey};

    // ---- Shared FHE keys (same pattern as shuffle.rs) ----

    fn shared_keys() -> &'static (ClientKey, ServerKey) {
        static KEYS: OnceLock<(ClientKey, ServerKey)> = OnceLock::new();
        static RAYON_INIT: OnceLock<()> = OnceLock::new();

        let keys = KEYS.get_or_init(|| {
            let config = ConfigBuilder::default().build();
            tfhe::generate_keys(config)
        });

        set_server_key(keys.1.clone());

        RAYON_INIT.get_or_init(|| {
            let sk = keys.1.clone();
            rayon::broadcast(move |_| set_server_key(sk.clone()));
        });

        keys
    }

    // ---- Network topology tests ----

    #[test]
    fn test_waksman_network_n2() {
        let net = waksman_network(2);
        assert_eq!(net.len(), 1);
        assert_eq!(net[0], vec![(0, 1)]);
    }

    #[test]
    fn test_waksman_network_n4() {
        let net = waksman_network(4);
        assert_eq!(net.len(), 3); // depth = 2*2 - 1 = 3
        // Left column: (0,1), (2,3)
        // Middle: sub-networks on {0,2} and {1,3}
        // Right column: (0,1), (2,3)
    }

    #[test]
    fn test_waksman_network_sizes() {
        for &n in &[2, 4, 8, 16, 32] {
            let net = waksman_network(n);
            assert_eq!(net.len(), waksman_depth(n), "Wrong depth for n={}", n);
        }
    }

    #[test]
    fn test_waksman_depth_values() {
        assert_eq!(waksman_depth(2), 1);
        assert_eq!(waksman_depth(4), 3);
        assert_eq!(waksman_depth(8), 5);
        assert_eq!(waksman_depth(16), 7);
        assert_eq!(waksman_depth(32), 9);
    }

    #[test]
    fn test_waksman_stages_are_disjoint() {
        for &n in &[4, 8, 16, 32] {
            let net = waksman_network(n);
            for (stage_idx, stage) in net.iter().enumerate() {
                let mut seen = vec![false; n];
                for &(i, j) in stage {
                    assert!(
                        !seen[i],
                        "n={}: element {} appears twice in stage {}",
                        n, i, stage_idx
                    );
                    assert!(
                        !seen[j],
                        "n={}: element {} appears twice in stage {}",
                        n, j, stage_idx
                    );
                    assert!(i < n && j < n);
                    seen[i] = true;
                    seen[j] = true;
                }
            }
        }
    }

    #[test]
    fn test_waksman_plaintext_is_permutation() {
        let mut rng = rand::thread_rng();
        for &n in &[4, 8, 16] {
            let net = waksman_network(n);
            let mut data: Vec<usize> = (0..n).collect();
            // Apply one pass with random bits
            for stage in &net {
                for &(i, j) in stage {
                    if rng.gen_bool(0.5) {
                        data.swap(i, j);
                    }
                }
            }
            let mut sorted = data.clone();
            sorted.sort();
            assert_eq!(sorted, (0..n).collect::<Vec<usize>>(), "n={} not a permutation", n);
        }
    }

    #[test]
    fn test_waksman_all_permutations_reachable_n4() {
        let net = waksman_network(4);
        let total_switches: usize = net.iter().map(|s| s.len()).sum();
        let mut seen = HashSet::new();

        // Enumerate all 2^total_switches bit patterns
        for bits in 0..(1u64 << total_switches) {
            let mut data: Vec<usize> = (0..4).collect();
            let mut bit_idx = 0;
            for stage in &net {
                for &(i, j) in stage {
                    if (bits >> bit_idx) & 1 == 1 {
                        data.swap(i, j);
                    }
                    bit_idx += 1;
                }
            }
            seen.insert(data);
        }

        assert_eq!(seen.len(), 24, "Not all 24 permutations of N=4 were reachable");
    }

    // ---- Bias analysis ----

    /// Compute pairwise correlation bias: given input_0 → output_0,
    /// measure max |P(input_i → output_j | input_0 → output_0) - 1/(N-1)|
    /// over all valid (i, j) pairs.
    ///
    /// This captures the joint distribution bias that position bias misses
    /// (position marginals are uniform by symmetry of the Waksman network).
    fn waksman_pairwise_bias(n: usize, num_passes: u32, num_samples: usize) -> f64 {
        let net = waksman_network(n);
        let mut rng = rand::thread_rng();
        // conditional_counts[i][j] = count of (input_i → output_j) given (input_0 → output_0)
        let mut conditional_counts = vec![vec![0usize; n]; n];
        let mut conditioning_total = 0usize;

        for _ in 0..num_samples {
            let mut perm: Vec<usize> = (0..n).collect();
            for _pass in 0..num_passes {
                for stage in &net {
                    for &(i, j) in stage {
                        if rng.gen_bool(0.5) {
                            perm.swap(i, j);
                        }
                    }
                }
            }
            // Check if input_0 went to output_0
            if perm[0] == 0 {
                conditioning_total += 1;
                for (output_pos, &input_pos) in perm.iter().enumerate() {
                    conditional_counts[input_pos][output_pos] += 1;
                }
            }
        }

        if conditioning_total < 100 {
            return f64::NAN; // not enough samples
        }

        let expected = 1.0 / (n - 1) as f64;
        let mut max_bias = 0.0f64;
        // Check P(input_i → output_j | input_0 → output_0) for i>0, j>0
        for i in 1..n {
            for j in 1..n {
                let observed = conditional_counts[i][j] as f64 / conditioning_total as f64;
                let bias = (observed - expected).abs();
                max_bias = max_bias.max(bias);
            }
        }
        max_bias
    }

    /// Exact pairwise bias for small N.
    fn waksman_pairwise_bias_exact(n: usize, num_passes: u32) -> f64 {
        let net = waksman_network(n);
        let switches_per_pass: usize = net.iter().map(|s| s.len()).sum();
        let total_bits = switches_per_pass * num_passes as usize;
        assert!(total_bits <= 30, "Too many bits for exhaustive enumeration");

        let total_patterns = 1u64 << total_bits;
        let mut conditional_counts = vec![vec![0u64; n]; n];
        let mut conditioning_total = 0u64;

        for pattern in 0..total_patterns {
            let mut perm: Vec<usize> = (0..n).collect();
            let mut bit_pos = 0;
            for _pass in 0..num_passes {
                for stage in &net {
                    for &(i, j) in stage {
                        if (pattern >> bit_pos) & 1 == 1 {
                            perm.swap(i, j);
                        }
                        bit_pos += 1;
                    }
                }
            }
            if perm[0] == 0 {
                conditioning_total += 1;
                for (output_pos, &input_pos) in perm.iter().enumerate() {
                    conditional_counts[input_pos][output_pos] += 1;
                }
            }
        }

        let expected = 1.0 / (n - 1) as f64;
        let mut max_bias = 0.0f64;
        for i in 1..n {
            for j in 1..n {
                let observed = conditional_counts[i][j] as f64 / conditioning_total as f64;
                let bias = (observed - expected).abs();
                max_bias = max_bias.max(bias);
            }
        }
        max_bias
    }

    #[test]
    fn test_waksman_bias_n4_exact() {
        println!("\n=== Waksman Pairwise Bias (N=4, exact) ===");
        println!("{:>8} {:>16}", "passes", "max |bias|");
        println!("{}", "-".repeat(26));
        for passes in 1..=4 {
            let bias = waksman_pairwise_bias_exact(4, passes);
            println!("{:>8} {:>16.6}", passes, bias);
        }
    }

    #[test]
    fn test_waksman_bias_table() {
        let samples = 20_000_000;
        println!("\n=== Waksman Pairwise Bias (Monte Carlo, {} samples) ===", samples);

        for &n in &[4, 8, 16] {
            println!("\nN={}:", n);
            println!("{:>8} {:>16}", "passes", "max |bias|");
            println!("{}", "-".repeat(26));
            for passes in 1..=10 {
                let bias = waksman_pairwise_bias(n, passes, samples);
                println!("{:>8} {:>16.6}", passes, bias);
            }
        }

        // Reference: bitonic collision probability
        println!("\n=== Bitonic Collision Probability (for comparison) ===");
        println!("(probability that any pair collides, disrupting relative order)");
        for &n in &[8, 16] {
            let bias_16 = 1.0 - (0..n).map(|i| 1.0 - i as f64 / 65536.0).product::<f64>();
            let bias_32 = 1.0 - (0..n).map(|i| 1.0 - i as f64 / 4294967296.0).product::<f64>();
            println!(
                "N={}: 16-bit = {:.6}, 32-bit = {:.9}",
                n, bias_16, bias_32
            );
        }
    }

    // ---- FHE tests ----

    #[test]
    fn test_waksman_fhe_2_elements() {
        let (client_key, _) = shared_keys();
        let values: Vec<FheUint64> = (0..2u64)
            .map(|i| FheUint64::encrypt(i, client_key))
            .collect();
        let result = waksman_shuffle(values, 1);
        let decrypted: Vec<u64> = result.iter().map(|v| v.decrypt(client_key)).collect();
        let mut sorted = decrypted.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0u64, 1]);
    }

    #[test]
    fn test_waksman_fhe_4_elements() {
        let (client_key, _) = shared_keys();
        let values: Vec<FheUint64> = (0..4u64)
            .map(|i| FheUint64::encrypt(i, client_key))
            .collect();
        let result = waksman_shuffle(values, 2);
        let decrypted: Vec<u64> = result.iter().map(|v| v.decrypt(client_key)).collect();
        let mut sorted = decrypted.clone();
        sorted.sort();
        assert_eq!(sorted, (0..4u64).collect::<Vec<_>>());
    }

    #[test]
    #[ignore] // ~30s -- run with: cargo test -- --ignored
    fn test_waksman_fhe_8_elements() {
        let (client_key, _) = shared_keys();
        let values: Vec<FheUint64> = (0..8u64)
            .map(|i| FheUint64::encrypt(i, client_key))
            .collect();
        let result = waksman_shuffle(values, 3);
        let decrypted: Vec<u64> = result.iter().map(|v| v.decrypt(client_key)).collect();
        let mut sorted = decrypted.clone();
        sorted.sort();
        assert_eq!(sorted, (0..8u64).collect::<Vec<_>>());
    }

    #[test]
    #[should_panic(expected = "Need at least 2 elements")]
    fn test_waksman_panics_on_empty() {
        let data: Vec<FheUint64> = vec![];
        waksman_shuffle(data, 1);
    }

    #[test]
    #[should_panic(expected = "Need at least 1 pass")]
    fn test_waksman_panics_on_zero_passes() {
        let (client_key, _) = shared_keys();
        let data: Vec<FheUint64> = (0..2u64)
            .map(|i| FheUint64::encrypt(i, client_key))
            .collect();
        waksman_shuffle(data, 0);
    }
}
