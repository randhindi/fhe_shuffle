// FHE Shuffle Benchmark
//
// Benchmarks oblivious_shuffle for n = 8, 16, 32 elements.
//
// CPU:  cargo run --release
// GPU:  cargo run --release --features gpu

use std::time::{Duration, Instant};

use tfhe::{prelude::*, set_server_key, ConfigBuilder, ClientKey, FheUint64};

#[cfg(feature = "gpu")]
use tfhe::CompressedServerKey;

use fhe_shuffle::network::{bitonic_comparator_count, bitonic_network, padded_size};
use fhe_shuffle::oblivious_shuffle;

const SIZES: &[usize] = &[8, 16, 32];
const KEY_PRECISION: u32 = 64;

struct BenchResult {
    n: usize,
    padded_n: usize,
    stages: usize,
    comparators: usize,
    shuffle_time: Duration,
}

fn bench_shuffle(n: usize, client_key: &ClientKey) -> BenchResult {
    let padded_n = padded_size(n);
    let network = bitonic_network(padded_n);

    println!("--- n={} (padded to {}, {} stages) ---", n, padded_n, network.len());

    // Encrypt data values
    let enc_start = Instant::now();
    let values: Vec<FheUint64> = (0..n as u64)
        .map(|i| FheUint64::encrypt(i, client_key))
        .collect();
    println!("  Encrypt: {:?}", enc_start.elapsed());

    // Run oblivious shuffle (OPRF + sort happen inside)
    let shuffle_start = Instant::now();
    let result = oblivious_shuffle(values, KEY_PRECISION);
    let shuffle_time = shuffle_start.elapsed();
    println!("  Shuffle: {:?}", shuffle_time);

    // Decrypt and verify
    let decrypted: Vec<u64> = result.iter().map(|v| v.decrypt(client_key)).collect();
    assert_eq!(decrypted.len(), n);
    let mut sorted = decrypted.clone();
    sorted.sort();
    assert_eq!(sorted, (0..n as u64).collect::<Vec<u64>>(), "Not a valid permutation!");
    println!("  Verified: valid permutation");
    println!();

    BenchResult {
        n,
        padded_n,
        stages: network.len(),
        comparators: bitonic_comparator_count(padded_n),
        shuffle_time,
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
    println!("=== FHE Shuffle Benchmark ===");
    #[cfg(feature = "gpu")]
    println!("Backend: GPU (CUDA)");
    #[cfg(not(feature = "gpu"))]
    println!("Backend: CPU");
    println!("Key precision: {} bits", KEY_PRECISION);
    println!("Sizes: {:?}", SIZES);
    println!();

    // Key generation
    println!("[1] Generating TFHE keys...");
    let keygen_start = Instant::now();
    let config = ConfigBuilder::default().build();
    let (client_key, server_key) = tfhe::generate_keys(config);
    println!("  Key generation: {:?}", keygen_start.elapsed());

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
    println!("Key precision: {} bits", KEY_PRECISION);
    println!();
    println!(
        "{:>6} {:>8} {:>8} {:>6} {:>12} {:>12}",
        "n", "padded", "stages", "comps", "Total", "/element"
    );
    println!("{}", "-".repeat(58));

    for r in &results {
        let per_element = r.shuffle_time / r.n as u32;
        println!(
            "{:>6} {:>8} {:>8} {:>6} {:>12} {:>12}",
            r.n,
            r.padded_n,
            r.stages,
            r.comparators,
            format_duration(r.shuffle_time),
            format_duration(per_element),
        );
    }
    println!("{}", "-".repeat(58));
}
