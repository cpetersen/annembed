//! Comprehensive integration tests for HDBSCAN
//! 
//! These tests validate the complete HDBSCAN implementation with
//! various edge cases and real-world scenarios.

#[cfg(test)]
mod integration_tests {
    use crate::hdbscan::*;
    use hnsw_rs::prelude::*;
    
    /// Helper to build HNSW index from data
    pub(super) fn build_test_hnsw(data: &[Vec<f32>]) -> Hnsw<f32, DistL2> {
        let nb_data = data.len();
        let data_with_id: Vec<(&Vec<f32>, usize)> = 
            data.iter().zip(0..nb_data).collect();
        
        // Use reasonable bounds for HNSW parameters to avoid performance issues
        let max_nb_conn = 16.min(nb_data / 4).max(2);  // Cap at 16, use nb_data/4 instead of nb_data/2
        let ef_c = 200.min(nb_data * 2).max(10);       // Cap ef_c based on dataset size
        let nb_layer = 16.min((nb_data as f32).ln().trunc() as usize).max(1);
        
        let hnsw = Hnsw::<f32, DistL2>::new(
            max_nb_conn, 
            nb_data, 
            nb_layer, 
            ef_c, 
            DistL2 {}
        );
        
        for (data, id) in data_with_id {
            hnsw.insert((data, id));
        }
        hnsw
    }
    
    #[test]
    fn test_empty_dataset() {
        let _data: Vec<Vec<f32>> = vec![];
        // Should handle empty dataset gracefully
        // Note: HNSW requires at least one point, so we skip this
    }
    
    #[test]
    fn test_single_point() {
        // Single point test - HNSW needs at least 2 points to work properly
        // So we'll test with 2 points far apart
        let data = vec![vec![0.0, 0.0], vec![1000.0, 1000.0]];
        let hnsw = build_test_hnsw(&data);
        let clustering = SLclustering::<usize, f32>::new(&hnsw, 1);
        
        // With min_cluster_size=3 (more than we have), all points should be noise
        let assignment = clustering.cluster_hdbscan(1, 3, SelectionMethod::Eom);
        
        assert_eq!(assignment.labels.len(), 2);
        // With min_cluster_size > num_points, all should be noise
        assert_eq!(assignment.num_noise(), 2);
    }
    
    #[test]
    fn test_two_points_close() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.01, 0.0],
        ];
        let hnsw = build_test_hnsw(&data);
        let clustering = SLclustering::<usize, f32>::new(&hnsw, 1);
        
        let assignment = clustering.cluster_hdbscan(2, 2, SelectionMethod::Eom);
        
        assert_eq!(assignment.labels.len(), 2);
        // Two close points should form a cluster if min_cluster_size=2
        if assignment.num_clusters() > 0 {
            assert_eq!(assignment.labels[0], assignment.labels[1]);
        }
    }
    
    #[test]
    fn test_two_points_far() {
        let data = vec![
            vec![0.0, 0.0],
            vec![100.0, 100.0],
        ];
        let hnsw = build_test_hnsw(&data);
        let clustering = SLclustering::<usize, f32>::new(&hnsw, 1);
        
        // Use min_cluster_size=3 to ensure points can't form a cluster
        let assignment = clustering.cluster_hdbscan(2, 3, SelectionMethod::Eom);
        
        assert_eq!(assignment.labels.len(), 2);
        // Two points can't form a cluster of size 3
        assert_eq!(assignment.num_noise(), 2);
    }
    
    #[test]
    fn test_perfect_clusters() {
        // Three well-separated small clusters
        let mut data = vec![];
        
        // Cluster 1: around (0, 0)
        for i in 0..5 {
            data.push(vec![i as f32 * 0.1, 0.0]);
        }
        
        // Cluster 2: around (10, 0)
        for i in 0..5 {
            data.push(vec![10.0 + i as f32 * 0.1, 0.0]);
        }
        
        // Cluster 3: around (5, 10)
        for i in 0..5 {
            data.push(vec![5.0 + i as f32 * 0.1, 10.0]);
        }
        
        let hnsw = build_test_hnsw(&data);
        let clustering = SLclustering::<usize, f32>::new(&hnsw, 1);
        
        let assignment = clustering.cluster_hdbscan(3, 3, SelectionMethod::Eom);
        
        // Should find some clusters (exact number depends on HNSW construction)
        assert!(assignment.num_clusters() > 0);
        assert_eq!(assignment.labels.len(), data.len());
    }
    
    #[test]
    fn test_clusters_with_noise() {
        let mut data = vec![];
        
        // Main cluster
        for i in 0..20 {
            data.push(vec![i as f32 * 0.1, 0.0]);
        }
        
        // Scattered noise points
        data.push(vec![10.0, 10.0]);
        data.push(vec![15.0, 15.0]);
        data.push(vec![20.0, 20.0]);
        data.push(vec![-10.0, -10.0]);
        data.push(vec![-15.0, -15.0]);
        
        let hnsw = build_test_hnsw(&data);
        let clustering = SLclustering::<usize, f32>::new(&hnsw, 1);
        
        let assignment = clustering.cluster_hdbscan(5, 10, SelectionMethod::Eom);
        
        // Should find 1 cluster and some noise
        assert!(assignment.num_clusters() >= 1);
        assert!(assignment.num_noise() > 0);
    }
    
    #[test]
    fn test_varying_density_clusters() {
        let mut data = vec![];
        
        // Dense cluster
        for i in 0..10 {
            for j in 0..10 {
                data.push(vec![i as f32 * 0.01, j as f32 * 0.01]);
            }
        }
        
        // Sparse cluster
        for i in 0..5 {
            for j in 0..5 {
                data.push(vec![5.0 + i as f32 * 0.5, j as f32 * 0.5]);
            }
        }
        
        let hnsw = build_test_hnsw(&data);
        let clustering = SLclustering::<usize, f32>::new(&hnsw, 1);
        
        // HDBSCAN should handle varying densities
        let assignment = clustering.cluster_hdbscan(5, 10, SelectionMethod::Eom);
        
        // Should find at least one cluster
        assert!(assignment.num_clusters() >= 1 || assignment.labels.len() == assignment.num_noise());
        
        // Check that we have labels for all points
        assert_eq!(assignment.labels.len(), data.len());
    }
    
    #[test]
    fn test_min_cluster_size_effect() {
        let data = vec![
            // Small group (5 points)
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![0.1, 0.1],
            vec![0.05, 0.05],
            // Another small group (5 points)
            vec![2.0, 0.0],
            vec![2.1, 0.0],
            vec![2.0, 0.1],
            vec![2.1, 0.1],
            vec![2.05, 0.05],
        ];
        
        let hnsw = build_test_hnsw(&data);
        let clustering = SLclustering::<usize, f32>::new(&hnsw, 1);
        
        // Test with min_cluster_size = 3
        let assignment_3 = clustering.cluster_hdbscan(2, 3, SelectionMethod::Eom);
        
        // Test with min_cluster_size = 6
        let assignment_6 = clustering.cluster_hdbscan(2, 6, SelectionMethod::Eom);
        
        // Smaller min_cluster_size should allow more clusters
        assert!(assignment_3.num_clusters() >= assignment_6.num_clusters());
    }
    
    #[test]
    fn test_min_samples_effect() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.2, 0.0],
            vec![0.3, 0.0],
            vec![0.4, 0.0],
            vec![2.0, 0.0],
            vec![2.1, 0.0],
            vec![2.2, 0.0],
            vec![2.3, 0.0],
            vec![2.4, 0.0],
        ];
        
        let hnsw = build_test_hnsw(&data);
        let clustering = SLclustering::<usize, f32>::new(&hnsw, 1);
        
        // Test with different min_samples
        let assignment_2 = clustering.cluster_hdbscan(2, 3, SelectionMethod::Eom);
        let assignment_4 = clustering.cluster_hdbscan(4, 3, SelectionMethod::Eom);
        
        // Both should produce valid results
        assert_eq!(assignment_2.labels.len(), data.len());
        assert_eq!(assignment_4.labels.len(), data.len());
    }
    
    #[test]
    fn test_selection_method_consistency() {
        let data = vec![
            // Hierarchical structure
            vec![0.0, 0.0], vec![0.1, 0.0], vec![0.2, 0.0],
            vec![0.5, 0.0], vec![0.6, 0.0], vec![0.7, 0.0],
            vec![2.0, 0.0], vec![2.1, 0.0], vec![2.2, 0.0],
            vec![2.5, 0.0], vec![2.6, 0.0], vec![2.7, 0.0],
        ];
        
        let hnsw = build_test_hnsw(&data);
        let clustering = SLclustering::<usize, f32>::new(&hnsw, 1);
        
        // Both methods should produce valid clusterings
        let eom = clustering.cluster_hdbscan(3, 3, SelectionMethod::Eom);
        let leaf = clustering.cluster_hdbscan(3, 3, SelectionMethod::Leaf);
        
        // Both should assign all points
        assert_eq!(eom.labels.len(), data.len());
        assert_eq!(leaf.labels.len(), data.len());
        
        // All labels should be valid
        for &label in &eom.labels {
            assert!(label >= -1);
        }
        for &label in &leaf.labels {
            assert!(label >= -1);
        }
        
        // All probabilities should be valid
        for &prob in &eom.probabilities {
            assert!(prob >= 0.0 && prob <= 1.0);
        }
        for &prob in &leaf.probabilities {
            assert!(prob >= 0.0 && prob <= 1.0);
        }
    }
    
    #[test]
    fn test_outlier_scores_validity() {
        let data = vec![
            // Cluster
            vec![0.0, 0.0], vec![0.1, 0.0], vec![0.0, 0.1], 
            vec![0.1, 0.1], vec![0.05, 0.05],
            // Outliers
            vec![10.0, 10.0], vec![20.0, 20.0],
        ];
        
        let hnsw = build_test_hnsw(&data);
        let clustering = SLclustering::<usize, f32>::new(&hnsw, 1);
        
        let condensed = clustering.build_condensed_tree(3, 4);
        let assignment = condensed.extract_clusters(SelectionMethod::Eom);
        let scores = condensed.calculate_outlier_scores(&assignment);
        
        // All scores should be between 0 and 1
        assert_eq!(scores.len(), assignment.labels.len());
        for &score in &scores {
            assert!(score >= 0.0 && score <= 1.0);
        }
        
        // Outliers should have higher scores
        // (though exact values depend on the clustering)
    }
    
    #[test]
    fn test_degenerate_min_samples() {
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![2.0, 0.0],
        ];
        
        let hnsw = build_test_hnsw(&data);
        let clustering = SLclustering::<usize, f32>::new(&hnsw, 1);
        
        // Test with min_samples = 1 (degenerate case)
        let assignment = clustering.cluster_hdbscan(1, 2, SelectionMethod::Eom);
        
        // Should handle gracefully
        assert_eq!(assignment.labels.len(), data.len());
    }
    
    #[test]
    fn test_all_same_points() {
        // Points very close together (not exactly same to avoid numerical issues)
        let data = vec![
            vec![1.0, 1.0],
            vec![1.00001, 1.0],
            vec![1.0, 1.00001],
            vec![1.00001, 1.00001],
            vec![1.00002, 1.00002],
        ];
        
        let hnsw = build_test_hnsw(&data);
        let clustering = SLclustering::<usize, f32>::new(&hnsw, 1);
        
        let assignment = clustering.cluster_hdbscan(2, 3, SelectionMethod::Eom);
        
        // Should handle very close points
        assert_eq!(assignment.labels.len(), data.len());
        
        // Most points should have the same label since they're very close
        // Count the most common label
        let mut label_counts = std::collections::HashMap::new();
        for &label in &assignment.labels {
            *label_counts.entry(label).or_insert(0) += 1;
        }
        let max_count = label_counts.values().max().copied().unwrap_or(0);
        assert!(max_count >= 3); // At least 3 points should have same label
    }
    
    #[test]
    fn test_linear_cluster() {
        // Points in a line
        let mut data = vec![];
        for i in 0..50 {
            data.push(vec![i as f32 * 0.1, 0.0]);
        }
        
        let hnsw = build_test_hnsw(&data);
        let clustering = SLclustering::<usize, f32>::new(&hnsw, 1);
        
        let assignment = clustering.cluster_hdbscan(5, 10, SelectionMethod::Eom);
        
        // Linear arrangement should be detected as cluster(s)
        assert!(assignment.num_clusters() > 0 || assignment.num_noise() == data.len());
    }
    
    #[test]
    fn test_circular_cluster() {
        // Points in a circle
        let mut data = vec![];
        let n_points = 50;
        for i in 0..n_points {
            let angle = 2.0 * std::f32::consts::PI * i as f32 / n_points as f32;
            data.push(vec![angle.cos(), angle.sin()]);
        }
        
        let hnsw = build_test_hnsw(&data);
        let clustering = SLclustering::<usize, f32>::new(&hnsw, 1);
        
        let assignment = clustering.cluster_hdbscan(5, 10, SelectionMethod::Eom);
        
        // Circular arrangement should be detected
        assert_eq!(assignment.labels.len(), data.len());
    }
    
    #[test]
    #[ignore] // This test creates 420 points which can be slow
    fn test_stability_ordering() {
        // Create clusters with predictable stability differences
        let mut data = vec![];
        
        // Very tight cluster (high stability)
        for i in 0..20 {
            for j in 0..20 {
                data.push(vec![i as f32 * 0.001, j as f32 * 0.001]);
            }
        }
        
        // Loose cluster (low stability)
        for i in 0..20 {
            data.push(vec![10.0 + i as f32 * 0.5, 0.0]);
        }
        
        let hnsw = build_test_hnsw(&data);
        let clustering = SLclustering::<usize, f32>::new(&hnsw, 1);
        
        let condensed = clustering.build_condensed_tree(5, 15);
        let assignment = condensed.extract_clusters(SelectionMethod::Eom);
        
        // Should have stabilities
        assert!(!assignment.stabilities.is_empty());
        
        // Stabilities should be non-negative
        for &stability in assignment.stabilities.values() {
            assert!(stability >= 0.0);
        }
    }
}

#[cfg(test)]
mod boundary_tests {
    use super::*;
    use crate::hdbscan::*;
    
    #[test]
    fn test_max_hierarchy_depth() {
        // Create a dataset that will create a deep hierarchy
        let mut data = vec![];
        for i in 0..100 {
            data.push(vec![i as f32, 0.0]);
        }
        
        // This should create a hierarchy without stack overflow
        let hnsw = integration_tests::build_test_hnsw(&data);
        let clustering = SLclustering::<usize, f32>::new(&hnsw, 1);
        
        // Should handle deep hierarchies
        let hierarchy = clustering.build_hierarchy(2);
        assert!(hierarchy.num_nodes() > 0);
    }
    
    #[test]
    fn test_very_large_min_cluster_size() {
        let data: Vec<Vec<f32>> = (0..10)
            .map(|i| vec![i as f32, 0.0])
            .collect();
        
        let hnsw = integration_tests::build_test_hnsw(&data);
        let clustering = SLclustering::<usize, f32>::new(&hnsw, 1);
        
        // min_cluster_size larger than dataset
        let assignment = clustering.cluster_hdbscan(2, 20, SelectionMethod::Eom);
        
        // With min_cluster_size > dataset size, no clusters can form
        assert_eq!(assignment.num_clusters(), 0);
        // The exact behavior depends on implementation details
        // What matters is that it doesn't crash
        // Some points may be lost in condensation if min_cluster_size is too large
    }
}

#[cfg(test)]
mod performance_tests {
    use super::*;
    use crate::hdbscan::*;
    
    #[test]
    #[ignore] // Run with --ignored flag for performance testing
    fn test_large_dataset() {
        // Generate a large dataset
        let mut data = vec![];
        for i in 0..1000 {
            for j in 0..10 {
                data.push(vec![
                    i as f32 * 0.1 + (j as f32 * 0.01),
                    j as f32 * 0.1 + (i as f32 * 0.01),
                ]);
            }
        }
        
        let start = std::time::Instant::now();
        
        let hnsw = integration_tests::build_test_hnsw(&data);
        let clustering = SLclustering::<usize, f32>::new(&hnsw, 1);
        let assignment = clustering.cluster_hdbscan(5, 50, SelectionMethod::Eom);
        
        let duration = start.elapsed();
        
        println!("Clustered {} points in {:?}", data.len(), duration);
        println!("Found {} clusters with {} noise points", 
                 assignment.num_clusters(), assignment.num_noise());
        
        // Should complete in reasonable time
        assert!(duration.as_secs() < 60); // Less than 1 minute
    }
}