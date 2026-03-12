//! Core oblivious shuffle implementation.
//!
//! Uses a bitonic sorting network with oblivious pseudo-random sort keys
//! to produce a uniformly random permutation of encrypted data.

use rayon::prelude::*;
use tfhe::prelude::*;
use tfhe::{FheBool, FheInt8, FheInt16, FheInt32, FheInt64, FheInt128,
           FheUint8, FheUint16, FheUint32, FheUint64, FheUint128, Seed};

use crate::network::{bitonic_network, padded_size};

// ---------------------------------------------------------------------------
// Public trait: types that can be obliviously shuffled
// ---------------------------------------------------------------------------

/// Trait for encrypted types that can be shuffled.
///
/// Implemented for all standard TFHE-rs encrypted integer types:
/// `FheUint{8,16,32,64,128}` and `FheInt{8,16,32,64,128}`.
pub trait Shuffleable: Sized + Send + Sync {
    /// Create a trivially encrypted zero (used for padding).
    fn trivial_zero() -> Self;

    /// Conditionally swap two values based on an encrypted boolean.
    /// If `cond` encrypts `true`, returns `(b, a)`; otherwise `(a, b)`.
    fn conditional_swap(cond: &FheBool, a: &Self, b: &Self) -> (Self, Self);
}

// ---------------------------------------------------------------------------
// Private trait: types that can serve as random sort keys
// ---------------------------------------------------------------------------

/// Trait for encrypted unsigned integer types used as sort keys.
trait SortKey: Sized + Send + Sync {
    /// Generate an oblivious pseudo-random encrypted value.
    fn generate_random(seed: Seed) -> Self;

    /// Create a trivially encrypted maximum value (used for padding keys).
    fn trivial_max() -> Self;

    /// Encrypted greater-than comparison.
    fn cmp_gt(&self, other: &Self) -> FheBool;

    /// Encrypted less-than comparison.
    fn cmp_lt(&self, other: &Self) -> FheBool;

    /// Conditionally swap two keys.
    fn conditional_swap(cond: &FheBool, a: &Self, b: &Self) -> (Self, Self);
}

// ---------------------------------------------------------------------------
// Macro implementations
// ---------------------------------------------------------------------------

macro_rules! impl_shuffleable_uint {
    ($fhe_type:ty, $clear_type:ty) => {
        impl Shuffleable for $fhe_type {
            fn trivial_zero() -> Self {
                <$fhe_type>::encrypt_trivial(0 as $clear_type)
            }
            fn conditional_swap(cond: &FheBool, a: &Self, b: &Self) -> (Self, Self) {
                cond.flip(a, b)
            }
        }
    };
}

macro_rules! impl_shuffleable_int {
    ($fhe_type:ty, $clear_type:ty) => {
        impl Shuffleable for $fhe_type {
            fn trivial_zero() -> Self {
                <$fhe_type>::encrypt_trivial(0 as $clear_type)
            }
            fn conditional_swap(cond: &FheBool, a: &Self, b: &Self) -> (Self, Self) {
                cond.flip(a, b)
            }
        }
    };
}

macro_rules! impl_sort_key {
    ($fhe_type:ty, $clear_type:ty) => {
        impl SortKey for $fhe_type {
            fn generate_random(seed: Seed) -> Self {
                <$fhe_type>::generate_oblivious_pseudo_random(seed)
            }
            fn trivial_max() -> Self {
                <$fhe_type>::encrypt_trivial(<$clear_type>::MAX)
            }
            fn cmp_gt(&self, other: &Self) -> FheBool {
                FheOrd::gt(self, other)
            }
            fn cmp_lt(&self, other: &Self) -> FheBool {
                FheOrd::lt(self, other)
            }
            fn conditional_swap(cond: &FheBool, a: &Self, b: &Self) -> (Self, Self) {
                cond.flip(a, b)
            }
        }
    };
}

// Shuffleable: unsigned integer types
impl_shuffleable_uint!(FheUint8, u8);
impl_shuffleable_uint!(FheUint16, u16);
impl_shuffleable_uint!(FheUint32, u32);
impl_shuffleable_uint!(FheUint64, u64);
impl_shuffleable_uint!(FheUint128, u128);

// Shuffleable: signed integer types
impl_shuffleable_int!(FheInt8, i8);
impl_shuffleable_int!(FheInt16, i16);
impl_shuffleable_int!(FheInt32, i32);
impl_shuffleable_int!(FheInt64, i64);
impl_shuffleable_int!(FheInt128, i128);

// SortKey: unsigned integer types only (keys must be unsigned for uniform distribution)
impl_sort_key!(FheUint8, u8);
impl_sort_key!(FheUint16, u16);
impl_sort_key!(FheUint32, u32);
impl_sort_key!(FheUint64, u64);
impl_sort_key!(FheUint128, u128);

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Shuffles encrypted data into a uniformly random permutation.
///
/// Generates oblivious pseudo-random sort keys internally, then sorts the
/// data using a bitonic sorting network. The resulting permutation is
/// uniformly random with negligible bias (determined by `key_precision`).
///
/// # Arguments
///
/// * `data` - Encrypted values to shuffle (length >= 2)
/// * `key_precision` - Bit-width of random sort keys: 8, 16, 32, 64, or 128.
///   Higher precision = lower collision probability but slower comparisons.
///   Use 64 for a good balance (collision bound < 2^-57 for n <= 16).
///
/// # Panics
///
/// Panics if `data.len() < 2` or `key_precision` is not one of {8, 16, 32, 64, 128}.
///
/// # Example
///
/// ```no_run
/// use tfhe::{prelude::*, set_server_key, ConfigBuilder, FheUint32};
/// use fhe_shuffle::oblivious_shuffle;
///
/// let config = ConfigBuilder::default().build();
/// let (client_key, server_key) = tfhe::generate_keys(config);
/// set_server_key(server_key);
///
/// let values: Vec<FheUint32> = (0..16u32)
///     .map(|i| FheUint32::encrypt(i, &client_key))
///     .collect();
///
/// // Shuffle with 64-bit sort keys (collision probability < 2^-57)
/// let shuffled = oblivious_shuffle(values, 64);
/// ```
pub fn oblivious_shuffle<T: Shuffleable>(data: Vec<T>, key_precision: u32) -> Vec<T> {
    assert!(data.len() >= 2, "Need at least 2 elements to shuffle");
    match key_precision {
        8 => shuffle_with_keys::<T, FheUint8>(data),
        16 => shuffle_with_keys::<T, FheUint16>(data),
        32 => shuffle_with_keys::<T, FheUint32>(data),
        64 => shuffle_with_keys::<T, FheUint64>(data),
        128 => shuffle_with_keys::<T, FheUint128>(data),
        _ => panic!(
            "Unsupported key_precision: {}. Use 8, 16, 32, 64, or 128.",
            key_precision
        ),
    }
}

// ---------------------------------------------------------------------------
// Generic core
// ---------------------------------------------------------------------------

/// Shuffles data using sort keys of type K.
fn shuffle_with_keys<D: Shuffleable, K: SortKey>(data: Vec<D>) -> Vec<D> {
    let n = data.len();
    let padded_n = padded_size(n);
    let network = bitonic_network(padded_n);

    // Generate n random sort keys via OPRF
    let mut rng = rand::thread_rng();
    let sort_keys: Vec<K> = (0..n)
        .map(|_| {
            use rand::Rng;
            K::generate_random(Seed(rng.gen::<u128>()))
        })
        .collect();

    // Initialize working arrays with Option for take/put pattern
    let mut data: Vec<Option<D>> = data.into_iter().map(Some).collect();
    let mut keys: Vec<Option<K>> = sort_keys.into_iter().map(Some).collect();

    // Pad to next power of 2 if needed
    for _ in 0..(padded_n - n) {
        keys.push(Some(K::trivial_max()));
        data.push(Some(D::trivial_zero()));
    }

    // Execute bitonic sorting network
    for stage in &network {
        let pairs: Vec<_> = stage
            .iter()
            .map(|&(i, j, ascending)| {
                let ki = keys[i].take().unwrap();
                let kj = keys[j].take().unwrap();
                let di = data[i].take().unwrap();
                let dj = data[j].take().unwrap();
                (i, j, ascending, ki, kj, di, dj)
            })
            .collect();

        let results: Vec<_> = pairs
            .into_par_iter()
            .map(|(i, j, ascending, ki, kj, di, dj)| {
                let cmp = if ascending {
                    ki.cmp_gt(&kj)
                } else {
                    ki.cmp_lt(&kj)
                };

                let ((new_ki, new_kj), (new_di, new_dj)) = rayon::join(
                    || K::conditional_swap(&cmp, &ki, &kj),
                    || D::conditional_swap(&cmp, &di, &dj),
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
    }

    // Return first n elements, discard padding
    data.into_iter().take(n).map(|x| x.unwrap()).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tfhe::{set_server_key, ConfigBuilder, ClientKey, ServerKey};

    fn setup_keys() -> (ClientKey, ServerKey) {
        let config = ConfigBuilder::default().build();
        tfhe::generate_keys(config)
    }

    fn setup_server_key(server_key: ServerKey) {
        set_server_key(server_key.clone());
        rayon::broadcast(|_| set_server_key(server_key.clone()));
    }

    fn verify_permutation<T: FheDecrypt<u64>>(
        result: &[T],
        n: usize,
        client_key: &ClientKey,
    ) {
        let decrypted: Vec<u64> = result.iter().map(|v| v.decrypt(client_key)).collect();
        assert_eq!(decrypted.len(), n);
        let mut sorted = decrypted.clone();
        sorted.sort();
        assert_eq!(
            sorted,
            (0..n as u64).collect::<Vec<u64>>(),
            "Not a valid permutation! Got: {:?}",
            decrypted
        );
    }

    #[test]
    fn test_shuffle_2_elements_uint8() {
        let (client_key, server_key) = setup_keys();
        setup_server_key(server_key);

        let values: Vec<FheUint8> = (0..2u8)
            .map(|i| FheUint8::encrypt(i, &client_key))
            .collect();

        let result = oblivious_shuffle(values, 8);

        let decrypted: Vec<u8> = result.iter().map(|v| v.decrypt(&client_key)).collect();
        assert_eq!(decrypted.len(), 2);
        let mut sorted = decrypted.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0u8, 1u8]);
    }

    #[test]
    fn test_shuffle_2_elements_key_precision_16() {
        let (client_key, server_key) = setup_keys();
        setup_server_key(server_key);

        let values: Vec<FheUint64> = (0..2u64)
            .map(|i| FheUint64::encrypt(i, &client_key))
            .collect();

        let result = oblivious_shuffle(values, 16);
        verify_permutation(&result, 2, &client_key);
    }

    #[test]
    fn test_shuffle_2_elements_key_precision_32() {
        let (client_key, server_key) = setup_keys();
        setup_server_key(server_key);

        let values: Vec<FheUint64> = (0..2u64)
            .map(|i| FheUint64::encrypt(i, &client_key))
            .collect();

        let result = oblivious_shuffle(values, 32);
        verify_permutation(&result, 2, &client_key);
    }

    #[test]
    fn test_shuffle_2_elements_key_precision_64() {
        let (client_key, server_key) = setup_keys();
        setup_server_key(server_key);

        let values: Vec<FheUint64> = (0..2u64)
            .map(|i| FheUint64::encrypt(i, &client_key))
            .collect();

        let result = oblivious_shuffle(values, 64);
        verify_permutation(&result, 2, &client_key);
    }

    #[test]
    fn test_shuffle_uint16_data() {
        let (client_key, server_key) = setup_keys();
        setup_server_key(server_key);

        let values: Vec<FheUint16> = (0..2u16)
            .map(|i| FheUint16::encrypt(i, &client_key))
            .collect();

        let result = oblivious_shuffle(values, 8);

        let decrypted: Vec<u16> = result.iter().map(|v| v.decrypt(&client_key)).collect();
        assert_eq!(decrypted.len(), 2);
        let mut sorted = decrypted.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0u16, 1u16]);
    }

    #[test]
    fn test_shuffle_int32_data() {
        let (client_key, server_key) = setup_keys();
        setup_server_key(server_key);

        let values: Vec<FheInt32> = (0..2i32)
            .map(|i| FheInt32::encrypt(i, &client_key))
            .collect();

        let result = oblivious_shuffle(values, 8);

        let decrypted: Vec<i32> = result.iter().map(|v| v.decrypt(&client_key)).collect();
        assert_eq!(decrypted.len(), 2);
        let mut sorted = decrypted.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0i32, 1i32]);
    }

    #[test]
    #[ignore] // ~30s — run with: cargo test -- --ignored
    fn test_shuffle_4_elements() {
        let (client_key, server_key) = setup_keys();
        setup_server_key(server_key);

        let values: Vec<FheUint64> = (0..4u64)
            .map(|i| FheUint64::encrypt(i, &client_key))
            .collect();

        let result = oblivious_shuffle(values, 64);
        verify_permutation(&result, 4, &client_key);
    }

    #[test]
    #[ignore] // ~30s — run with: cargo test -- --ignored
    fn test_shuffle_non_power_of_2() {
        let (client_key, server_key) = setup_keys();
        setup_server_key(server_key);

        let values: Vec<FheUint64> = (0..3u64)
            .map(|i| FheUint64::encrypt(i, &client_key))
            .collect();

        let result = oblivious_shuffle(values, 64);
        verify_permutation(&result, 3, &client_key);
    }

    #[test]
    #[should_panic(expected = "Need at least 2 elements")]
    fn test_shuffle_panics_on_empty() {
        let data: Vec<FheUint64> = vec![];
        oblivious_shuffle(data, 64);
    }

    #[test]
    #[should_panic(expected = "Need at least 2 elements")]
    fn test_shuffle_panics_on_single() {
        let config = ConfigBuilder::default().build();
        let (client_key, _) = tfhe::generate_keys(config);
        let data = vec![FheUint64::encrypt(0u64, &client_key)];
        oblivious_shuffle(data, 64);
    }

    #[test]
    #[should_panic(expected = "Unsupported key_precision")]
    fn test_shuffle_panics_on_invalid_precision() {
        let config = ConfigBuilder::default().build();
        let (client_key, server_key) = tfhe::generate_keys(config);
        setup_server_key(server_key);
        let data: Vec<FheUint64> = (0..2u64)
            .map(|i| FheUint64::encrypt(i, &client_key))
            .collect();
        oblivious_shuffle(data, 48);
    }
}
