//! # FHE Shuffle
//!
//! Oblivious permutation of encrypted data using TFHE-rs.
//!
//! Shuffles a list of encrypted values into a uniformly random permutation,
//! without the server learning which permutation was applied. Works with all
//! TFHE-rs encrypted integer types (`FheUint8` through `FheUint128`,
//! `FheInt8` through `FheInt128`).
//!
//! # Example
//!
//! ```no_run
//! use tfhe::{prelude::*, set_server_key, ConfigBuilder, FheUint64};
//! use fhe_shuffle::oblivious_shuffle;
//!
//! let config = ConfigBuilder::default().build();
//! let (client_key, server_key) = tfhe::generate_keys(config);
//! set_server_key(server_key);
//!
//! let values: Vec<FheUint64> = (0..8u64)
//!     .map(|i| FheUint64::encrypt(i, &client_key))
//!     .collect();
//!
//! let shuffled = oblivious_shuffle(values, 64);
//! ```
//!
//! # Key Precision
//!
//! The `key_precision` parameter controls the bit-width of the random sort keys
//! used internally. Higher precision means lower collision probability (more uniform)
//! but slower comparisons:
//!
//! | key_precision | Key type   | Collision bound (n=16) |
//! |---------------|------------|------------------------|
//! | 8             | FheUint8   | ~2^-1 (NOT recommended)|
//! | 16            | FheUint16  | ~2^-9                  |
//! | 32            | FheUint32  | ~2^-25                 |
//! | 64            | FheUint64  | < 2^-57                |
//! | 128           | FheUint128 | < 2^-121               |

pub mod network;
pub mod shuffle;
pub mod waksman;

pub use shuffle::{oblivious_shuffle, Shuffleable};
pub use waksman::waksman_shuffle;
