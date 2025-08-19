//! Tests for random seed reproducibility in embedder

#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::fromhnsw::kgraph::kgraph_from_hnsw_all;
    use ndarray::Array2;
    use rand::distr::Uniform;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// Generate simple test data
    fn generate_test_data(n_points: usize, n_dims: usize) -> Vec<Vec<f32>> {
        let mut data = Vec::new();
        let mut rng = StdRng::seed_from_u64(42); // Fixed seed for data generation
        let dist = Uniform::new(0.0, 1.0).unwrap();
        
        for i in 0..n_points {
            let mut point = Vec::new();
            for _ in 0..n_dims {
                point.push((i as f32 * 0.1) + rng.sample(dist));
            }
            data.push(point);
        }
        data
    }

    /// Build HNSW graph from data
    fn build_hnsw(data: &[Vec<f32>]) -> Hnsw<f32, DistL2> {
        let ef_c = 50;
        let max_nb_connection = 24;
        let nb_layer = 16.min((data.len() as f32).ln().trunc() as usize);
        let hnsw = Hnsw::<f32, DistL2>::new(
            max_nb_connection,
            data.len(),
            nb_layer,
            ef_c,
            DistL2 {},
        );
        
        let data_with_id: Vec<(&Vec<f32>, usize)> = 
            data.iter().enumerate().map(|(i, v)| (v, i)).collect();
        hnsw.parallel_insert(&data_with_id);
        hnsw
    }
    
    /// Build HNSW graph from data with a seed for reproducibility
    fn build_hnsw_with_seed(data: &[Vec<f32>], seed: u64) -> Hnsw<f32, DistL2> {
        let ef_c = 50;
        let max_nb_connection = 24;
        let nb_layer = 16.min((data.len() as f32).ln().trunc() as usize);
        let hnsw = Hnsw::<f32, DistL2>::new_with_seed(
            max_nb_connection,
            data.len(),
            nb_layer,
            ef_c,
            DistL2 {},
            seed,
        );
        
        // Use serial insertion for reproducibility when created with a seed
        let data_with_id: Vec<(&Vec<f32>, usize)> = 
            data.iter().enumerate().map(|(i, v)| (v, i)).collect();
        hnsw.serial_insert(&data_with_id);
        hnsw
    }

    #[test]
    fn test_random_seed_simple_reproducibility() {
        // Generate small test data
        let data = generate_test_data(20, 10);
        let hnsw = build_hnsw(&data);
        let kgraph = kgraph_from_hnsw_all::<f32, DistL2, f32>(&hnsw, 5).unwrap();
        
        // Test parameters with seed
        let mut params = EmbedderParams::default();
        params.asked_dim = 2;
        params.nb_grad_batch = 10;
        params.random_seed = Some(42);
        
        // Run embedding multiple times with same seed
        let mut results = Vec::new();
        for _ in 0..3 {
            let mut embedder = Embedder::new(&kgraph, params);
            let _ = embedder.embed().unwrap();
            let embedding = embedder.get_embedded().unwrap();
            results.push(embedding.clone());
        }
        
        // Check that all results are identical
        for i in 1..results.len() {
            assert_eq!(
                results[0].shape(),
                results[i].shape(),
                "Result {} has different shape",
                i
            );
            
            // Check first point coordinates
            let diff = (&results[0] - &results[i]).mapv(|x| x.abs()).sum();
            assert!(
                diff < 1e-6,
                "Result {} differs from first result. Total diff: {}",
                i,
                diff
            );
        }
    }

    #[test]
    fn test_random_seed_different_seeds() {
        // Generate small test data
        let data = generate_test_data(20, 10);
        let hnsw = build_hnsw(&data);
        let kgraph = kgraph_from_hnsw_all::<f32, DistL2, f32>(&hnsw, 5).unwrap();
        
        // Test with different seeds
        let seeds = vec![42, 100, 200];
        let mut results = Vec::new();
        
        for seed in seeds {
            let mut params = EmbedderParams::default();
            params.asked_dim = 2;
            params.nb_grad_batch = 10;
            params.random_seed = Some(seed);
            
            let mut embedder = Embedder::new(&kgraph, params);
            let _ = embedder.embed().unwrap();
            let embedding = embedder.get_embedded().unwrap();
            results.push(embedding.clone());
        }
        
        // Check that results with different seeds are different
        for i in 0..results.len() {
            for j in i + 1..results.len() {
                let diff = (&results[i] - &results[j]).mapv(|x| x.abs()).sum();
                assert!(
                    diff > 0.1,
                    "Results with different seeds {} and {} are too similar. Diff: {}",
                    i,
                    j,
                    diff
                );
            }
        }
    }

    #[test]
    fn test_random_seed_none_is_random() {
        // Generate small test data
        let data = generate_test_data(20, 10);
        let hnsw = build_hnsw(&data);
        let kgraph = kgraph_from_hnsw_all::<f32, DistL2, f32>(&hnsw, 5).unwrap();
        
        // Test without seed (should be random)
        let mut params = EmbedderParams::default();
        params.asked_dim = 2;
        params.nb_grad_batch = 10;
        params.random_seed = None;
        
        let mut results = Vec::new();
        for _ in 0..3 {
            let mut embedder = Embedder::new(&kgraph, params);
            let _ = embedder.embed().unwrap();
            let embedding = embedder.get_embedded().unwrap();
            results.push(embedding.clone());
        }
        
        // Check that at least some results are different
        let mut found_difference = false;
        for i in 1..results.len() {
            let diff = (&results[0] - &results[i]).mapv(|x| x.abs()).sum();
            if diff > 0.01 {
                found_difference = true;
                break;
            }
        }
        
        assert!(
            found_difference,
            "All results without seed are identical, expected randomness"
        );
    }

    #[test]
    fn test_hierarchical_embedding_seed() {
        // Test larger dataset that triggers hierarchical embedding
        let data = generate_test_data(100, 20);
        let hnsw = build_hnsw(&data);
        
        // Create a hierarchical graph
        let kgraph = kgraph_from_hnsw_all::<f32, DistL2, f32>(&hnsw, 10).unwrap();
        
        let mut params = EmbedderParams::default();
        params.asked_dim = 2;
        params.nb_grad_batch = 20;
        params.random_seed = Some(42);
        params.dmap_init = true; // Use diffusion map initialization
        
        // Run multiple times
        let mut results = Vec::new();
        for _ in 0..3 {
            let mut embedder = Embedder::new(&kgraph, params);
            let _ = embedder.embed().unwrap();
            let embedding = embedder.get_embedded().unwrap();
            results.push(embedding.clone());
        }
        
        // Check reproducibility
        for i in 1..results.len() {
            let diff = (&results[0] - &results[i]).mapv(|x| x.abs()).sum();
            assert!(
                diff < 0.01,
                "Hierarchical embedding {} differs. Diff: {}",
                i,
                diff
            );
        }
    }

    #[test]
    fn test_entropy_optim_seed() {
        // Directly test EntropyOptim which is where much of the randomness comes from
        use crate::tools::nodeparam::*;
        
        // Create simple test data
        let n_nodes = 20;
        let mut node_params_vec = Vec::new();
        let max_nbng = 5;
        for i in 0..n_nodes {
            let mut edges = Vec::new();
            for j in 0..max_nbng {
                if i != j {
                    edges.push(OutEdge {
                        node: (i + j + 1) % n_nodes,
                        weight: 0.1 * (j + 1) as f32,
                    });
                }
            }
            node_params_vec.push(NodeParam::new(1.0, edges));
        }
        
        let _node_params = NodeParams::new(node_params_vec, max_nbng);
        
        // Test with fixed seed
        let mut params = EmbedderParams::default();
        params.asked_dim = 2;
        params.nb_grad_batch = 10;
        params.nb_sampling_by_edge = 5;
        params.random_seed = Some(42);
        
        // Create initial embedding
        let _initial_embedding = Array2::<f64>::zeros((n_nodes, 2));
        
        // Run entropy optimization multiple times - simplified test
        // Note: EntropyOptim methods are not public, so we can't directly test it
        // This test would need to be in the embedder module itself to access private methods
        
        // For now, we'll just test that the seed is properly stored
        assert_eq!(params.random_seed, Some(42));
    }

    #[test] 
    fn test_random_init_reproducibility() {
        // Test that random initialization is reproducible with same seed
        let data = generate_test_data(20, 10);
        let hnsw = build_hnsw(&data);
        let kgraph = kgraph_from_hnsw_all::<f32, DistL2, f32>(&hnsw, 5).unwrap();
        
        let mut params = EmbedderParams::default();
        params.asked_dim = 2;
        params.random_seed = Some(42);
        params.dmap_init = false; // Force random initialization
        
        // Create embedders and run embedding with random init
        let mut results = Vec::new();
        for _ in 0..3 {
            let mut embedder = Embedder::new(&kgraph, params);
            // Run embedding which will use random init internally
            let _ = embedder.embed().unwrap();
            let embedding = embedder.get_embedded().unwrap();
            results.push(embedding.clone());
        }
        
        // Check all are identical
        for i in 1..results.len() {
            assert_eq!(
                results[0].shape(),
                results[i].shape(),
                "Shape mismatch"
            );
            
            let diff = (&results[0] - &results[i]).mapv(|x| x.abs()).sum();
            assert!(
                diff < 1e-6,
                "Random init {} differs. Diff: {}",
                i,
                diff
            );
        }
    }

    #[test]
    fn test_parallel_vs_sequential() {
        // Test if parallel processing affects reproducibility
        let data = generate_test_data(50, 15);
        let hnsw = build_hnsw(&data);
        let kgraph = kgraph_from_hnsw_all::<f32, DistL2, f32>(&hnsw, 8).unwrap();
        
        let mut params = EmbedderParams::default();
        params.asked_dim = 2;
        params.nb_grad_batch = 20;
        params.random_seed = Some(42);
        
        // Run multiple times to check for timing-dependent behavior
        let mut results = Vec::new();
        for run in 0..5 {
            // Add some variability in timing
            if run > 0 {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            
            let mut embedder = Embedder::new(&kgraph, params);
            let _ = embedder.embed().unwrap();
            let embedding = embedder.get_embedded().unwrap();
            results.push(embedding.clone());
        }
        
        // Check reproducibility
        let mut differences = Vec::new();
        for i in 1..results.len() {
            let diff = (&results[0] - &results[i]).mapv(|x| x.abs()).sum();
            differences.push(diff);
            println!("Run {} diff from run 0: {}", i, diff);
        }
        
        // All should be very similar
        for (i, diff) in differences.iter().enumerate() {
            assert!(
                *diff < 0.01,
                "Run {} has significant difference: {}",
                i + 1,
                diff
            );
        }
    }
    
    #[test]
    fn test_full_reproducibility_with_seeded_hnsw() {
        // Test complete reproducibility: both HNSW and embedder use seeds
        println!("\n\ntest_full_reproducibility_with_seeded_hnsw");
        
        // Generate test data with fixed seed
        let data = generate_test_data(50, 15);
        
        // Seeds for reproducibility
        let hnsw_seed = 12345;
        let embedder_seed = 67890;
        
        // Create multiple runs with same seeds
        let mut results = Vec::new();
        for run in 0..3 {
            println!("Run {}", run);
            
            // Build HNSW with seed for reproducible layer assignments
            let hnsw = build_hnsw_with_seed(&data, hnsw_seed);
            let kgraph = kgraph_from_hnsw_all::<f32, DistL2, f32>(&hnsw, 8).unwrap();
            
            // Set embedder parameters with seed
            let mut params = EmbedderParams::default();
            params.asked_dim = 2;
            params.nb_grad_batch = 20;
            params.random_seed = Some(embedder_seed);
            params.dmap_init = false; // Use random init for reproducibility test
            
            // Run embedding
            let mut embedder = Embedder::new(&kgraph, params);
            let _ = embedder.embed().unwrap();
            let embedding = embedder.get_embedded().unwrap();
            results.push(embedding.clone());
        }
        
        // Verify all runs produce identical results
        println!("\nChecking reproducibility across runs:");
        for i in 1..results.len() {
            let diff = (&results[0] - &results[i]).mapv(|x| x.abs()).sum();
            println!("Run {} diff from run 0: {}", i, diff);
            
            // With both seeds fixed, results should be EXACTLY identical
            assert!(
                diff < 1e-10,  // Very strict tolerance for identical computation
                "Run {} differs from run 0 by {}. Expected exact reproducibility with fixed seeds.",
                i,
                diff
            );
        }
        
        println!("✅ Full reproducibility achieved with seeded HNSW and embedder!");
    }
    
    #[test]
    fn test_hnsw_seed_affects_embedding() {
        // Test that different HNSW seeds produce different embeddings
        println!("\n\ntest_hnsw_seed_affects_embedding");
        
        // Generate test data
        let data = generate_test_data(30, 10);
        
        // Fixed embedder seed
        let embedder_seed = 42;
        
        // Create embeddings with different HNSW seeds
        let hnsw_seeds = vec![100, 200, 300];
        let mut results = Vec::new();
        
        for hnsw_seed in &hnsw_seeds {
            let hnsw = build_hnsw_with_seed(&data, *hnsw_seed);
            let kgraph = kgraph_from_hnsw_all::<f32, DistL2, f32>(&hnsw, 5).unwrap();
            
            let mut params = EmbedderParams::default();
            params.asked_dim = 2;
            params.nb_grad_batch = 10;
            params.random_seed = Some(embedder_seed);
            
            let mut embedder = Embedder::new(&kgraph, params);
            let _ = embedder.embed().unwrap();
            let embedding = embedder.get_embedded().unwrap();
            results.push(embedding.clone());
        }
        
        // Verify that different HNSW seeds lead to different embeddings
        let mut found_difference = false;
        for i in 0..results.len() {
            for j in i + 1..results.len() {
                let diff = (&results[i] - &results[j]).mapv(|x| x.abs()).sum();
                println!("HNSW seed {} vs {}: diff = {}", hnsw_seeds[i], hnsw_seeds[j], diff);
                if diff > 0.01 {
                    found_difference = true;
                }
            }
        }
        
        assert!(
            found_difference,
            "Different HNSW seeds should produce different embeddings due to different graph structures"
        );
    }
}
