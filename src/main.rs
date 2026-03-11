// FHE Shuffle — Uniformly random oblivious permutation using TFHE-rs
//
// Uses a bitonic sorting network with server-generated oblivious pseudo-random sort keys.
// Each element gets a random FheUint64 key; sorting by these keys produces a uniform
// random permutation (statistical distance < 2^-57 from uniform for n=16).
//
// For n=16: 10 stages of 8 parallel comparators.
// Each comparator = 1 × gt + 4 × if_then_else.
// Depth: O(log²n).
//
// Supports both CPU and GPU backends:
//   CPU: cargo run --release
//   GPU: cargo run --release --features gpu   (requires CUDA)

mod network;
mod shuffle;

use std::time::Instant;

use rand::Rng;
use tfhe::{prelude::*, set_server_key, ConfigBuilder, FheUint64, Seed};

#[cfg(feature = "gpu")]
use tfhe::CompressedServerKey;

use network::{bitonic_comparator_count, bitonic_network};
use shuffle::bitonic_shuffle;

const N: usize = 16;

fn main() {
    println!("=== FHE Shuffle Benchmark ===");
    #[cfg(feature = "gpu")]
    println!("Backend: GPU (CUDA)");
    #[cfg(not(feature = "gpu"))]
    println!("Backend: CPU");
    println!(
        "Shuffling {} encrypted FheUint64 values using bitonic sort network",
        N
    );
    println!("Uniformity: statistical distance < 2^-57 from uniform permutation");
    println!();

    // Print network topology
    let network = bitonic_network(N);
    let num_comparators = bitonic_comparator_count(N);
    println!(
        "Bitonic network (n={}): {} stages × {} comparators/stage = {} total comparators",
        N,
        network.len(),
        N / 2,
        num_comparators
    );
    println!();

    // Key generation
    println!("[1] Generating TFHE keys...");
    let keygen_start = Instant::now();
    let config = ConfigBuilder::default().build();
    let (client_key, server_key) = tfhe::generate_keys(config);
    println!("  Key generation: {:?}", keygen_start.elapsed());

    // Set up backend (CPU or GPU)
    #[cfg(not(feature = "gpu"))]
    {
        set_server_key(server_key.clone());
        rayon::broadcast(|_| set_server_key(server_key.clone()));
    }

    #[cfg(feature = "gpu")]
    {
        drop(server_key);
        let gpu_start = Instant::now();
        let compressed = CompressedServerKey::new(&client_key);
        set_server_key(compressed.decompress_to_gpu());
        rayon::broadcast(|_| set_server_key(compressed.decompress_to_gpu()));
        println!("  GPU key transfer: {:?}", gpu_start.elapsed());
    }
    println!();

    // Encrypt data values (0..15)
    println!("[2] Encrypting {} values...", N);
    let enc_start = Instant::now();
    let values: Vec<FheUint64> = (0..N as u64)
        .map(|i| FheUint64::encrypt(i, &client_key))
        .collect();
    let enc_time = enc_start.elapsed();
    println!(
        "  {} × FheUint64 in {:?} ({:?}/value)",
        N,
        enc_time,
        enc_time / N as u32
    );
    println!();

    // Generate oblivious pseudo-random sort keys
    println!(
        "[3] Generating {} oblivious random sort keys (FheUint64)...",
        N
    );
    let mut rng = rand::thread_rng();
    let rng_start = Instant::now();
    let sort_keys: Vec<FheUint64> = (0..N)
        .map(|_| FheUint64::generate_oblivious_pseudo_random(Seed(rng.gen::<u128>())))
        .collect();
    let rng_time = rng_start.elapsed();
    println!(
        "  {} × FheUint64 in {:?} ({:?}/key)",
        N,
        rng_time,
        rng_time / N as u32
    );
    println!();

    // Run the bitonic shuffle
    println!(
        "[4] Running bitonic shuffle ({} stages)...",
        network.len()
    );
    let shuffle_start = Instant::now();
    let result = bitonic_shuffle(values, sort_keys);
    let shuffle_time = shuffle_start.elapsed();
    println!("  Total shuffle: {:?}", shuffle_time);
    println!(
        "  Per-stage avg: {:?}",
        shuffle_time / network.len() as u32
    );
    println!();

    // Decrypt and verify
    println!("Decrypting and verifying...");
    let dec_start = Instant::now();
    let decrypted: Vec<u64> = result
        .iter()
        .map(|v| {
            let val: u64 = v.decrypt(&client_key);
            val
        })
        .collect();
    let dec_time = dec_start.elapsed();

    let mut sorted = decrypted.clone();
    sorted.sort();
    assert_eq!(
        sorted,
        (0..N as u64).collect::<Vec<u64>>(),
        "Not a valid permutation!"
    );
    println!("  Decryption: {:?}", dec_time);
    println!("  Result: {:?}", decrypted);
    println!("  Valid permutation confirmed");
    println!();

    // Summary
    println!("========================================");
    println!("Summary");
    println!("========================================");
    #[cfg(feature = "gpu")]
    println!("  Backend:          GPU (CUDA)");
    #[cfg(not(feature = "gpu"))]
    println!("  Backend:          CPU");
    println!("  OPRF generation:  {:?}", rng_time);
    println!("  Shuffle:          {:?}", shuffle_time);
    println!("  Total server-side: {:?}", rng_time + shuffle_time);
}
