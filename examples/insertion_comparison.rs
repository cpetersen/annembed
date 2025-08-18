//! Comparison of serial vs parallel insertion performance and reproducibility
//! 
//! This example demonstrates:
//! 1. Performance difference between serial and parallel insertion
//! 2. Reproducibility implications of each approach
//! 3. How to choose the right method for your use case

use annembed::fromhnsw::kgraph_from_hnsw_all;
use hnsw_rs::prelude::*;
use rand::distr::{Distribution, Uniform};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::Instant;

fn build_hnsw_serial(data: &[Vec<f32>], seed: Option<u64>) -> Hnsw<f32, DistL2> {
    let ef_c = 200;
    let max_nb_connection = 16;
    let nb_layer = 16.min((data.len() as f32).ln().trunc() as usize);
    
    let hnsw = match seed {
        Some(s) => Hnsw::<f32, DistL2>::new_with_seed(
            max_nb_connection,
            data.len(),
            nb_layer,
            ef_c,
            DistL2 {},
            s,
        ),
        None => Hnsw::<f32, DistL2>::new(
            max_nb_connection,
            data.len(),
            nb_layer,
            ef_c,
            DistL2 {},
        ),
    };
    
    // Serial insertion
    for (i, v) in data.iter().enumerate() {
        hnsw.insert((v, i));
    }
    
    hnsw
}

fn build_hnsw_parallel(data: &[Vec<f32>], seed: Option<u64>) -> Hnsw<f32, DistL2> {
    let ef_c = 200;
    let max_nb_connection = 16;
    let nb_layer = 16.min((data.len() as f32).ln().trunc() as usize);
    
    let mut hnsw = match seed {
        Some(s) => {
            let mut h = Hnsw::<f32, DistL2>::new_with_seed(
                max_nb_connection,
                data.len(),
                nb_layer,
                ef_c,
                DistL2 {},
                s,
            );
            // Force parallel mode even with seed to demonstrate the difference
            h.set_deterministic_mode(false);
            h
        },
        None => Hnsw::<f32, DistL2>::new(
            max_nb_connection,
            data.len(),
            nb_layer,
            ef_c,
            DistL2 {},
        ),
    };
    
    // Parallel insertion (will actually be parallel now)
    let data_with_id: Vec<(&Vec<f32>, usize)> = 
        data.iter().enumerate().map(|(i, v)| (v, i)).collect();
    hnsw.parallel_insert(&data_with_id);
    
    hnsw
}

fn main() {
    env_logger::init();
    
    println!("\n=== Serial vs Parallel Insertion Comparison ===\n");
    
    // Generate test data
    let mut rng = StdRng::seed_from_u64(42);
    let unif = Uniform::<f32>::new(0., 1.).unwrap();
    
    // Test with different dataset sizes
    let sizes = vec![100, 500, 1000, 5000];
    
    for &n_points in &sizes {
        println!("\n--- Testing with {} points ---", n_points);
        
        let dim = 50;
        let mut data = Vec::new();
        for _ in 0..n_points {
            let mut point = Vec::new();
            for _ in 0..dim {
                point.push(unif.sample(&mut rng));
            }
            data.push(point);
        }
        
        // Test 1: Performance comparison without seed
        println!("\n1. Performance (no seed):");
        
        let start = Instant::now();
        let _hnsw_serial = build_hnsw_serial(&data, None);
        let serial_time = start.elapsed();
        println!("   Serial insertion:   {:?}", serial_time);
        
        let start = Instant::now();
        let _hnsw_parallel = build_hnsw_parallel(&data, None);
        let parallel_time = start.elapsed();
        println!("   Parallel insertion: {:?}", parallel_time);
        
        let speedup = serial_time.as_secs_f64() / parallel_time.as_secs_f64();
        println!("   Speedup: {:.2}x", speedup);
        
        // Test 2: Reproducibility with seed
        println!("\n2. Reproducibility (with seed = 12345):");
        
        // Serial with seed - should be reproducible
        let seed = 12345;
        let hnsw1 = build_hnsw_serial(&data, Some(seed));
        let hnsw2 = build_hnsw_serial(&data, Some(seed));
        
        // Convert to graphs for comparison
        let kgraph1 = kgraph_from_hnsw_all::<f32, DistL2, f32>(&hnsw1, 10).unwrap();
        let kgraph2 = kgraph_from_hnsw_all::<f32, DistL2, f32>(&hnsw2, 10).unwrap();
        
        // Check if graphs are identical (simple check on edge counts)
        let edges1: usize = kgraph1.get_neighbours().iter()
            .map(|n| n.len())
            .sum();
        let edges2: usize = kgraph2.get_neighbours().iter()
            .map(|n| n.len())
            .sum();
        
        println!("   Serial (run 1):   {} edges", edges1);
        println!("   Serial (run 2):   {} edges", edges2);
        println!("   Serial reproducible: {}", edges1 == edges2);
        
        // Parallel with seed - forcing parallel mode to show non-reproducibility
        // Note: With smart defaults, this would normally use serial insertion
        let hnsw3 = build_hnsw_parallel(&data, Some(seed));
        let hnsw4 = build_hnsw_parallel(&data, Some(seed));
        
        let kgraph3 = kgraph_from_hnsw_all::<f32, DistL2, f32>(&hnsw3, 10).unwrap();
        let kgraph4 = kgraph_from_hnsw_all::<f32, DistL2, f32>(&hnsw4, 10).unwrap();
        
        let edges3: usize = kgraph3.get_neighbours().iter()
            .map(|n| n.len())
            .sum();
        let edges4: usize = kgraph4.get_neighbours().iter()
            .map(|n| n.len())
            .sum();
        
        println!("   Parallel (run 1): {} edges", edges3);
        println!("   Parallel (run 2): {} edges", edges4);
        println!("   Parallel reproducible: {}", edges3 == edges4);
    }
    
    println!("\n=== Summary ===");
    println!("\n1. Performance:");
    println!("   - Parallel insertion is typically 2-8x faster on multi-core systems");
    println!("   - Speedup increases with dataset size");
    println!("   - For small datasets (<100 points), overhead may reduce benefits");
    
    println!("\n2. Reproducibility:");
    println!("   - Serial insertion: ALWAYS reproducible with seed");
    println!("   - Parallel insertion: NOT reproducible even with seed");
    println!("     (due to non-deterministic thread scheduling)");
    println!("   - Smart defaults: When created with seed, HNSW automatically");
    println!("     uses deterministic mode (serial insertion via parallel_insert)");
    
    println!("\n3. Recommendations:");
    println!("   - Use parallel insertion when:");
    println!("     * Dataset is large (>1000 points)");
    println!("     * Reproducibility is not required");
    println!("     * Performance is critical");
    println!("   - Use serial insertion when:");
    println!("     * Reproducibility is required");
    println!("     * Dataset is small");
    println!("     * Debugging or testing");
    
    println!("\n4. Implemented Smart Defaults:");
    println!("   ```rust");
    println!("   // Automatic behavior based on construction:");
    println!("   let hnsw_seeded = Hnsw::new_with_seed(..., seed);");
    println!("   hnsw_seeded.parallel_insert(&data);  // Uses serial internally");
    println!("   ");
    println!("   let hnsw_unseeded = Hnsw::new(...);");
    println!("   hnsw_unseeded.parallel_insert(&data);  // Uses true parallel");
    println!("   ");
    println!("   // Manual override when needed:");
    println!("   let mut hnsw = Hnsw::new_with_seed(..., seed);");
    println!("   hnsw.set_deterministic_mode(false);  // Force parallel mode");
    println!("   hnsw.parallel_insert(&data);  // Now uses true parallel");
    println!("   ```");
}