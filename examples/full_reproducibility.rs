//! Example demonstrating full reproducibility with seeded HNSW and embedder
//! 
//! This example shows how to achieve complete reproducibility by:
//! 1. Using a seed for HNSW graph construction (layer assignments)
//! 2. Using a seed for the embedder (random initialization and sampling)
//! 3. Using sequential insertion to avoid parallel non-determinism

use annembed::prelude::*;
use annembed::fromhnsw::kgraph_from_hnsw_all;
use hnsw_rs::prelude::*;
use rand::distr::{Distribution, Uniform};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn main() {
    // Initialize logging
    env_logger::init();
    
    println!("\n=== Full Reproducibility Example ===\n");
    
    // Generate synthetic data with fixed seed
    let mut rng = StdRng::seed_from_u64(42);
    let unif = Uniform::<f32>::new(0., 1.).unwrap();
    let n_points = 100;
    let dim = 20;
    
    let mut data = Vec::new();
    for _ in 0..n_points {
        let mut point = Vec::new();
        for _ in 0..dim {
            point.push(unif.sample(&mut rng));
        }
        data.push(point);
    }
    
    println!("Generated {} points in {} dimensions", n_points, dim);
    
    // Seeds for reproducibility
    let hnsw_seed = 12345;
    let embedder_seed = 67890;
    
    println!("\nSeeds:");
    println!("  HNSW seed: {}", hnsw_seed);
    println!("  Embedder seed: {}", embedder_seed);
    
    // Run embedding multiple times to verify reproducibility
    let mut embeddings = Vec::new();
    
    for run in 0..3 {
        println!("\n--- Run {} ---", run + 1);
        
        // Step 1: Build HNSW with seed for reproducible layer assignments
        let max_nb_connection = 16;
        let ef_construction = 200;
        let nb_layer = 16.min((n_points as f32).ln().trunc() as usize);
        
        let hnsw = Hnsw::<f32, DistL2>::new_with_seed(
            max_nb_connection,
            n_points,
            nb_layer,
            ef_construction,
            DistL2 {},
            hnsw_seed,
        );
        
        // Step 2: Insert data sequentially for reproducibility
        // (parallel_insert has non-deterministic ordering)
        for (i, point) in data.iter().enumerate() {
            hnsw.insert((point, i));
        }
        
        println!("Built HNSW graph with {} points", hnsw.get_nb_point());
        
        // Step 3: Convert HNSW to KGraph for embedder
        let knbn = 15;
        let kgraph = kgraph_from_hnsw_all::<f32, DistL2, f32>(&hnsw, knbn).unwrap();
        
        // Step 4: Configure embedder with seed
        let mut embed_params = EmbedderParams::default();
        embed_params.asked_dim = 2;
        embed_params.nb_grad_batch = 200;
        embed_params.scale_rho = 1.;
        embed_params.beta = 1.;
        embed_params.grad_step = 1.;
        embed_params.nb_sampling_by_edge = 10;
        embed_params.dmap_init = true;
        embed_params.random_seed = Some(embedder_seed);
        
        // Step 5: Run embedding
        let mut embedder = Embedder::new(&kgraph, embed_params);
        let _embed_res = embedder.embed().unwrap();
        
        println!("Embedding completed");
        
        let embedding = embedder.get_embedded().unwrap();
        embeddings.push(embedding.clone());
    }
    
    // Verify reproducibility
    println!("\n=== Reproducibility Check ===");
    
    let mut all_identical = true;
    for i in 1..embeddings.len() {
        let diff = (&embeddings[0] - &embeddings[i])
            .mapv(|x| x.abs())
            .sum();
        
        println!("Run 1 vs Run {}: total difference = {:.2e}", i + 1, diff);
        
        if diff > 1e-10 {
            all_identical = false;
            println!("  ⚠️  Runs are not exactly identical!");
        } else {
            println!("  ✅ Runs are identical!");
        }
    }
    
    if all_identical {
        println!("\n🎉 SUCCESS: Full reproducibility achieved!");
        println!("All runs produced EXACTLY identical embeddings.");
    } else {
        println!("\n⚠️  WARNING: Some differences detected between runs.");
        println!("This may be due to floating-point rounding in parallel operations.");
    }
    
    // Display some statistics
    let final_embedding = &embeddings[0];
    let mean_x = final_embedding.column(0).mean().unwrap();
    let mean_y = final_embedding.column(1).mean().unwrap();
    let std_x = final_embedding.column(0).std(0.);
    let std_y = final_embedding.column(1).std(0.);
    
    println!("\n=== Embedding Statistics ===");
    println!("Dimension 1: mean = {:.4}, std = {:.4}", mean_x, std_x);
    println!("Dimension 2: mean = {:.4}, std = {:.4}", mean_y, std_y);
    
    println!("\n=== Summary ===");
    println!("When using:");
    println!("1. Hnsw::new_with_seed() for deterministic layer assignments");
    println!("2. Sequential insertion (not parallel_insert)");
    println!("3. EmbedderParams::random_seed for deterministic initialization");
    println!("\nYou achieve COMPLETE reproducibility across runs!");
}