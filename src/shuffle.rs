use rayon::prelude::*;
use tfhe::{prelude::*, FheUint64};

use crate::network::{bitonic_network, padded_size};

/// Uniform shuffle using a bitonic sorting network with random sort keys.
///
/// Assigns each data element a random FheUint64 key, then sorts (key, data) pairs
/// using a bitonic network. Since the keys are uniformly random 64-bit values,
/// the resulting permutation is uniformly random conditioned on no key collisions.
///
/// Collision probability: C(n,2)/2^64 < 2^-57 for n=16, well within < 2^-40.
///
/// The bitonic network for n=16 has 10 stages, each with 8 parallel comparators.
/// Each comparator: 1 × gt + 2 × flip (conditional swap of both key and data).
///
/// **Non-power-of-2 support**: If n is not a power of 2, the input is internally
/// padded to the next power of 2. Padding elements use trivially encrypted u64::MAX
/// sort keys, which guarantees they sort to the end of the array. After sorting, only
/// the first n elements (the real data) are returned.
///
/// Arguments:
/// - `data`: encrypted values to shuffle (length >= 2)
/// - `sort_keys`: encrypted random keys (one per element, same length as data)
///
/// Returns: shuffled encrypted values (a uniformly random permutation of the input)
pub fn bitonic_shuffle(data: Vec<FheUint64>, sort_keys: Vec<FheUint64>) -> Vec<FheUint64> {
    let n = data.len();
    assert!(n >= 2, "Need at least 2 elements to shuffle");
    assert_eq!(
        sort_keys.len(),
        n,
        "Need exactly {} sort keys, got {}",
        n,
        sort_keys.len()
    );

    let padded_n = padded_size(n);
    let needs_padding = padded_n > n;

    // Pad to next power of 2 if needed
    let mut data: Vec<Option<FheUint64>> = data.into_iter().map(Some).collect();
    let mut keys: Vec<Option<FheUint64>> = sort_keys.into_iter().map(Some).collect();

    if needs_padding {
        let pad_count = padded_n - n;
        println!(
            "  Padding {} elements to {} (next power of 2, +{} padding)",
            n, padded_n, pad_count
        );
        for _ in 0..pad_count {
            // Trivially encrypted u64::MAX keys sort to the end
            keys.push(Some(FheUint64::encrypt_trivial(u64::MAX)));
            // Padding data — value doesn't matter, will be discarded
            data.push(Some(FheUint64::encrypt_trivial(0u64)));
        }
    }

    let network = bitonic_network(padded_n);

    for (stage_num, stage) in network.iter().enumerate() {
        let stage_start = std::time::Instant::now();

        // Extract elements and their keys for this stage
        let pairs: Vec<(usize, usize, bool, FheUint64, FheUint64, FheUint64, FheUint64)> = stage
            .iter()
            .map(|&(i, j, ascending)| {
                let ki = keys[i].take().expect("key already consumed");
                let kj = keys[j].take().expect("key already consumed");
                let di = data[i].take().expect("data already consumed");
                let dj = data[j].take().expect("data already consumed");
                (i, j, ascending, ki, kj, di, dj)
            })
            .collect();

        // All comparators in a stage are independent — run in parallel
        let results: Vec<(usize, usize, FheUint64, FheUint64, FheUint64, FheUint64)> = pairs
            .into_par_iter()
            .map(|(i, j, ascending, ki, kj, di, dj)| {
                // Compare keys: should_swap = (ascending && ki > kj) || (!ascending && ki < kj)
                // For ascending: swap if ki > kj → put smaller key at i
                // For descending: swap if ki < kj → put larger key at i
                let cmp = if ascending {
                    ki.gt(&kj)
                } else {
                    ki.lt(&kj)
                };

                // Conditional swap both keys and data in parallel
                // flip(true, a, b) = (b, a); flip(false, a, b) = (a, b)
                let ((new_ki, new_kj), (new_di, new_dj)) = rayon::join(
                    || cmp.flip(&ki, &kj),
                    || cmp.flip(&di, &dj),
                );

                (i, j, new_ki, new_kj, new_di, new_dj)
            })
            .collect();

        for (i, j, new_ki, new_kj, new_di, new_dj) in results {
            keys[i] = Some(new_ki);
            keys[j] = Some(new_kj);
            data[i] = Some(new_di);
            data[j] = Some(new_dj);
        }

        println!(
            "  Stage {}/{}: {:?}",
            stage_num + 1,
            network.len(),
            stage_start.elapsed()
        );
    }

    // Return only the first n elements (discard padding)
    data.into_iter()
        .take(n)
        .map(|x| x.unwrap())
        .collect()
}
