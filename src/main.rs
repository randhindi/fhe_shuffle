// FHE Shuffle — Uniformly random oblivious permutation using TFHE-rs
//
// Uses a bitonic sorting network with server-generated oblivious pseudo-random sort keys.
// Each element gets a random FheUint64 key; sorting by these keys produces a uniform
// random permutation (statistical distance < 2^-57 from uniform for n=16).
//
// Supports non-power-of-2 sizes (padded internally to next power of 2).
//
// Supports both CPU and GPU backends:
//   CPU: cargo run --release
//   GPU: cargo run --release --features gpu   (requires CUDA)

mod network;
mod shuffle;

use std::time::{Duration, Instant};

use rand::Rng;
use tfhe::{prelude::*, set_server_key, ConfigBuilder, ClientKey, FheUint64, Seed};

#[cfg(feature = "gpu")]
use tfhe::CompressedServerKey;

use network::{bitonic_comparator_count, bitonic_network, padded_size};
use shuffle::bitonic_shuffle;

const SIZES: &[usize] = &[8, 16, 32];

struct BenchResult {
    n: usize,
    padded_n: usize,
    stages: usize,
    comparators: usize,
    oprf_time: Duration,
    shuffle_time: Duration,
    total_time: Duration,
}

fn bench_shuffle(n: usize, client_key: &ClientKey) -> BenchResult {
    let padded_n = padded_size(n);
    let network = bitonic_network(padded_n);
    let num_comparators = bitonic_comparator_count(padded_n);
    let num_stages = network.len();

    println!("--- n={} (padded to {}, {} stages) ---", n, padded_n, num_stages);

    // Encrypt data values
    let enc_start = Instant::now();
    let values: Vec<FheUint64> = (0..n as u64)
        .map(|i| FheUint64::encrypt(i, client_key))
        .collect();
    println!("  Encrypt: {:?}", enc_start.elapsed());

    // Generate oblivious pseudo-random sort keys
    let mut rng = rand::thread_rng();
    let rng_start = Instant::now();
    let sort_keys: Vec<FheUint64> = (0..n)
        .map(|_| FheUint64::generate_oblivious_pseudo_random(Seed(rng.gen::<u128>())))
        .collect();
    let oprf_time = rng_start.elapsed();
    println!("  OPRF:    {:?}", oprf_time);

    // Run the bitonic shuffle
    let shuffle_start = Instant::now();
    let result = bitonic_shuffle(values, sort_keys);
    let shuffle_time = shuffle_start.elapsed();
    println!("  Shuffle: {:?}", shuffle_time);

    // Decrypt and verify
    let decrypted: Vec<u64> = result
        .iter()
        .map(|v| {
            let val: u64 = v.decrypt(client_key);
            val
        })
        .collect();

    assert_eq!(decrypted.len(), n);
    let mut sorted = decrypted.clone();
    sorted.sort();
    assert_eq!(
        sorted,
        (0..n as u64).collect::<Vec<u64>>(),
        "n={}: Not a valid permutation!",
        n
    );
    println!("  Verified: valid permutation");

    let total_time = oprf_time + shuffle_time;
    println!("  Total:   {:?}", total_time);
    println!();

    BenchResult {
        n,
        padded_n,
        stages: num_stages,
        comparators: num_comparators,
        oprf_time,
        shuffle_time,
        total_time,
    }
}

fn format_duration(d: Duration) -> String {
    let secs = d.as_secs_f64();
    if secs < 60.0 {
        format!("{:.1}s", secs)
    } else if secs < 3600.0 {
        format!("{:.1}m", secs / 60.0)
    } else {
        format!("{:.1}h", secs / 3600.0)
    }
}

fn main() {
    println!("=== FHE Shuffle Benchmark Suite ===");
    #[cfg(feature = "gpu")]
    println!("Backend: GPU (CUDA)");
    #[cfg(not(feature = "gpu"))]
    println!("Backend: CPU");
    println!("Sizes: {:?}", SIZES);
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

    // Run benchmarks
    println!("[2] Running benchmarks...");
    println!();

    let results: Vec<BenchResult> = SIZES
        .iter()
        .map(|&n| bench_shuffle(n, &client_key))
        .collect();

    // Print results table
    println!("========================================================================");
    println!("                         BENCHMARK RESULTS");
    println!("========================================================================");
    #[cfg(feature = "gpu")]
    println!("Backend: GPU (CUDA)");
    #[cfg(not(feature = "gpu"))]
    println!("Backend: CPU");
    println!();
    println!(
        "{:>6} {:>8} {:>8} {:>6} {:>12} {:>12} {:>12} {:>12}",
        "n", "padded", "stages", "comps", "OPRF", "Shuffle", "Total", "/element"
    );
    println!("{}", "-".repeat(82));

    for r in &results {
        let per_element = r.total_time / r.n as u32;
        println!(
            "{:>6} {:>8} {:>8} {:>6} {:>12} {:>12} {:>12} {:>12}",
            r.n,
            r.padded_n,
            r.stages,
            r.comparators,
            format_duration(r.oprf_time),
            format_duration(r.shuffle_time),
            format_duration(r.total_time),
            format_duration(per_element),
        );
    }
    println!("{}", "-".repeat(82));
}
