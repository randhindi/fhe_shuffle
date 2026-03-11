/// Bitonic sorting network generator.
///
/// A bitonic sorting network for n=2^k elements has k*(k+1)/2 stages,
/// each with n/2 comparators. It sorts any input sequence.
///
/// For n=16: 10 stages × 8 comparators = 80 comparisons, depth = 10.
///
/// When used with random sort keys, it produces a uniformly random permutation
/// (conditioned on no key collisions). For 64-bit keys and n=16, the collision
/// probability is C(16,2)/2^64 < 2^-57, giving statistical distance < 2^-57
/// from the uniform distribution over permutations.

/// Generates a bitonic sorting network for n elements (n must be a power of 2).
///
/// Returns a list of stages, where each stage contains disjoint (i, j, ascending) triples.
/// Each triple represents a compare-and-swap: if ascending, put the smaller element at i;
/// if descending, put the larger element at i.
pub fn bitonic_network(n: usize) -> Vec<Vec<(usize, usize, bool)>> {
    assert!(n.is_power_of_two() && n >= 2);
    let log_n = n.trailing_zeros() as usize;
    let mut stages = Vec::new();

    for phase in 0..log_n {
        for step in (0..=phase).rev() {
            let mut comparators = Vec::new();
            for i in 0..n {
                let j = i ^ (1 << step);
                if j > i {
                    let ascending = (i >> (phase + 1)) & 1 == 0;
                    comparators.push((i, j, ascending));
                }
            }
            stages.push(comparators);
        }
    }

    stages
}

/// Count total comparators in a bitonic network of size n.
pub fn bitonic_comparator_count(n: usize) -> usize {
    let network = bitonic_network(n);
    network.iter().map(|stage| stage.len()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitonic_network_sizes() {
        // n=2: 1 stage, 1 comparator
        let net = bitonic_network(2);
        assert_eq!(net.len(), 1);
        assert_eq!(net[0].len(), 1);

        // n=4: 3 stages, 2 comparators each
        let net = bitonic_network(4);
        assert_eq!(net.len(), 3);
        for stage in &net {
            assert_eq!(stage.len(), 2);
        }

        // n=8: 6 stages, 4 comparators each
        let net = bitonic_network(8);
        assert_eq!(net.len(), 6);
        for stage in &net {
            assert_eq!(stage.len(), 4);
        }

        // n=16: 10 stages, 8 comparators each
        let net = bitonic_network(16);
        assert_eq!(net.len(), 10);
        for stage in &net {
            assert_eq!(stage.len(), 8);
        }
    }

    #[test]
    fn test_bitonic_stages_are_disjoint() {
        let net = bitonic_network(16);
        for (stage_idx, stage) in net.iter().enumerate() {
            let mut seen = vec![false; 16];
            for &(i, j, _) in stage {
                assert!(
                    !seen[i],
                    "Element {} appears twice in stage {}",
                    i, stage_idx
                );
                assert!(
                    !seen[j],
                    "Element {} appears twice in stage {}",
                    j, stage_idx
                );
                assert!(i < 16 && j < 16, "Index out of bounds in stage {}", stage_idx);
                seen[i] = true;
                seen[j] = true;
            }
        }
    }

    #[test]
    fn test_bitonic_comparator_count() {
        assert_eq!(bitonic_comparator_count(2), 1);
        assert_eq!(bitonic_comparator_count(4), 6);
        assert_eq!(bitonic_comparator_count(8), 24);
        assert_eq!(bitonic_comparator_count(16), 80);
    }

    #[test]
    fn test_bitonic_sorts_correctly() {
        let network = bitonic_network(16);
        let mut data: Vec<u64> = vec![15, 3, 7, 11, 0, 14, 2, 10, 8, 4, 12, 1, 6, 13, 9, 5];

        for stage in &network {
            for &(i, j, ascending) in stage {
                if ascending {
                    if data[i] > data[j] {
                        data.swap(i, j);
                    }
                } else {
                    if data[i] < data[j] {
                        data.swap(i, j);
                    }
                }
            }
        }

        assert_eq!(data, (0..16).collect::<Vec<u64>>());
    }

    #[test]
    fn test_bitonic_shuffle_is_permutation() {
        // Simulate a shuffle: assign random keys and sort by them
        let network = bitonic_network(16);
        let keys: Vec<u64> = vec![
            9823, 1234, 5678, 42, 99999, 0, 7777, 3141,
            2718, 65535, 11111, 8080, 404, 12345, 6789, 55555,
        ];
        let mut data: Vec<u64> = (0..16).collect();

        // Sort data by keys using bitonic network
        let mut key_data: Vec<(u64, u64)> = keys.iter().copied().zip(data.iter().copied()).collect();

        for stage in &network {
            for &(i, j, ascending) in stage {
                let should_swap = if ascending {
                    key_data[i].0 > key_data[j].0
                } else {
                    key_data[i].0 < key_data[j].0
                };
                if should_swap {
                    key_data.swap(i, j);
                }
            }
        }

        data = key_data.iter().map(|&(_, v)| v).collect();

        // Must be a valid permutation of 0..16
        let mut sorted = data.clone();
        sorted.sort();
        assert_eq!(sorted, (0..16).collect::<Vec<u64>>());
    }
}
