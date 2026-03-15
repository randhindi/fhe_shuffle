// FHE Shuffle Benchmark
//
// Benchmarks bitonic sort shuffle and Waksman network shuffle.
//
// CPU:  cargo run --release
// GPU:  cargo run --release --features gpu

use std::time::{Duration, Instant};

use tfhe::{prelude::*, set_server_key, ConfigBuilder, ClientKey, FheUint64};

#[cfg(feature = "gpu")]
use tfhe::CompressedServerKey;

use fhe_shuffle::network::{bitonic_comparator_count, bitonic_network, padded_size};
use fhe_shuffle::oblivious_shuffle;
use fhe_shuffle::waksman::{waksman_depth, waksman_shuffle, waksman_switch_count};

const SIZES: &[usize] = &[8, 16];
const KEY_PRECISIONS: &[u32] = &[16, 32];
const WAKSMAN_PASSES: &[u32] = &[3];

struct BitonicResult {
    n: usize,
    key_bits: u32,
    padded_n: usize,
    stages: usize,
    comparators: usize,
    shuffle_time: Duration,
}

struct WaksmanResult {
    n: usize,
    num_passes: u32,
    padded_n: usize,
    stages_per_pass: usize,
    switches_per_pass: usize,
    shuffle_time: Duration,
}

fn bench_bitonic(n: usize, key_bits: u32, client_key: &ClientKey) -> BitonicResult {
    let padded_n = padded_size(n);
    let network = bitonic_network(padded_n);

    println!(
        "--- Bitonic: n={}, {}-bit keys (padded to {}, {} stages) ---",
        n, key_bits, padded_n, network.len()
    );

    let enc_start = Instant::now();
    let values: Vec<FheUint64> = (0..n as u64)
        .map(|i| FheUint64::encrypt(i, client_key))
        .collect();
    println!("  Encrypt: {:?}", enc_start.elapsed());

    let shuffle_start = Instant::now();
    let result = oblivious_shuffle(values, key_bits);
    let shuffle_time = shuffle_start.elapsed();
    println!("  Shuffle: {:?}", shuffle_time);

    let decrypted: Vec<u64> = result.iter().map(|v| v.decrypt(client_key)).collect();
    assert_eq!(decrypted.len(), n);
    let mut sorted = decrypted.clone();
    sorted.sort();
    assert_eq!(sorted, (0..n as u64).collect::<Vec<u64>>(), "Not a valid permutation!");
    println!("  Verified: valid permutation\n");

    BitonicResult {
        n,
        key_bits,
        padded_n,
        stages: network.len(),
        comparators: bitonic_comparator_count(padded_n),
        shuffle_time,
    }
}

fn bench_waksman(n: usize, num_passes: u32, client_key: &ClientKey) -> WaksmanResult {
    let padded_n = padded_size(n);

    println!(
        "--- Waksman: n={}, {} passes (padded to {}, {} stages/pass) ---",
        n, num_passes, padded_n, waksman_depth(padded_n)
    );

    let enc_start = Instant::now();
    let values: Vec<FheUint64> = (0..n as u64)
        .map(|i| FheUint64::encrypt(i, client_key))
        .collect();
    println!("  Encrypt: {:?}", enc_start.elapsed());

    let shuffle_start = Instant::now();
    let result = waksman_shuffle(values, num_passes);
    let shuffle_time = shuffle_start.elapsed();
    println!("  Shuffle: {:?}", shuffle_time);

    let decrypted: Vec<u64> = result.iter().map(|v| v.decrypt(client_key)).collect();
    assert_eq!(decrypted.len(), n);
    let mut sorted = decrypted.clone();
    sorted.sort();
    assert_eq!(sorted, (0..n as u64).collect::<Vec<u64>>(), "Not a valid permutation!");
    println!("  Verified: valid permutation\n");

    WaksmanResult {
        n,
        num_passes,
        padded_n,
        stages_per_pass: waksman_depth(padded_n),
        switches_per_pass: waksman_switch_count(padded_n),
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
    println!("Sizes: {:?}", SIZES);
    println!("Bitonic key precisions: {:?} bits", KEY_PRECISIONS);
    println!("Waksman passes: {:?}", WAKSMAN_PASSES);
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

    // Run bitonic benchmarks
    println!("[2] Running bitonic shuffle benchmarks...\n");
    let mut bitonic_results: Vec<BitonicResult> = Vec::new();
    for &key_bits in KEY_PRECISIONS {
        for &n in SIZES {
            bitonic_results.push(bench_bitonic(n, key_bits, &client_key));
        }
    }

    // Run Waksman benchmarks
    println!("[3] Running Waksman shuffle benchmarks...\n");
    let mut waksman_results: Vec<WaksmanResult> = Vec::new();
    for &passes in WAKSMAN_PASSES {
        for &n in SIZES {
            waksman_results.push(bench_waksman(n, passes, &client_key));
        }
    }

    // Print results
    println!("========================================================================");
    println!("                         BENCHMARK RESULTS");
    println!("========================================================================");
    #[cfg(feature = "gpu")]
    println!("Backend: GPU (CUDA)");
    #[cfg(not(feature = "gpu"))]
    println!("Backend: CPU");
    println!();

    // Bitonic results
    for &key_bits in KEY_PRECISIONS {
        println!("--- Bitonic, {}-bit sort keys ---", key_bits);
        println!(
            "{:>6} {:>8} {:>8} {:>6} {:>12} {:>12}",
            "n", "padded", "stages", "comps", "Total", "/element"
        );
        println!("{}", "-".repeat(58));
        for r in bitonic_results.iter().filter(|r| r.key_bits == key_bits) {
            let per_element = r.shuffle_time / r.n as u32;
            println!(
                "{:>6} {:>8} {:>8} {:>6} {:>12} {:>12}",
                r.n, r.padded_n, r.stages, r.comparators,
                format_duration(r.shuffle_time), format_duration(per_element),
            );
        }
        println!("{}", "-".repeat(58));
        println!();
    }

    // Waksman results
    for &passes in WAKSMAN_PASSES {
        println!("--- Waksman, {} passes ---", passes);
        println!(
            "{:>6} {:>8} {:>8} {:>8} {:>12} {:>12}",
            "n", "padded", "stages", "switches", "Total", "/element"
        );
        println!("{}", "-".repeat(62));
        for r in waksman_results.iter().filter(|r| r.num_passes == passes) {
            let per_element = r.shuffle_time / r.n as u32;
            println!(
                "{:>6} {:>8} {:>8} {:>8} {:>12} {:>12}",
                r.n, r.padded_n,
                r.stages_per_pass * passes as usize,
                r.switches_per_pass * passes as usize,
                format_duration(r.shuffle_time), format_duration(per_element),
            );
        }
        println!("{}", "-".repeat(62));
        println!();
    }

    // Comparison table
    println!("========================================================================");
    println!("                          COMPARISON");
    println!("========================================================================");
    println!(
        "{:>6} {:>14} {:>14} {:>14} {:>10} {:>10}",
        "n", "Bitonic-16", "Bitonic-32", "Waksman-3p", "W/B16", "W/B32"
    );
    println!("{}", "-".repeat(72));
    for &n in SIZES {
        let b16 = bitonic_results.iter().find(|r| r.n == n && r.key_bits == 16);
        let b32 = bitonic_results.iter().find(|r| r.n == n && r.key_bits == 32);
        let w3 = waksman_results.iter().find(|r| r.n == n && r.num_passes == 3);

        let b16_str = b16.map_or("N/A".into(), |r| format_duration(r.shuffle_time));
        let b32_str = b32.map_or("N/A".into(), |r| format_duration(r.shuffle_time));
        let w3_str = w3.map_or("N/A".into(), |r| format_duration(r.shuffle_time));

        let speedup_16 = match (b16, w3) {
            (Some(b), Some(w)) => format!("{:.1}x", b.shuffle_time.as_secs_f64() / w.shuffle_time.as_secs_f64()),
            _ => "N/A".into(),
        };
        let speedup_32 = match (b32, w3) {
            (Some(b), Some(w)) => format!("{:.1}x", b.shuffle_time.as_secs_f64() / w.shuffle_time.as_secs_f64()),
            _ => "N/A".into(),
        };

        println!(
            "{:>6} {:>14} {:>14} {:>14} {:>10} {:>10}",
            n, b16_str, b32_str, w3_str, speedup_16, speedup_32,
        );
    }
    println!("{}", "-".repeat(72));
    println!("W/B16 = Bitonic-16 time / Waksman time (higher = Waksman faster)");
    println!("W/B32 = Bitonic-32 time / Waksman time (higher = Waksman faster)");
}
