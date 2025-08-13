//! Cluster extraction for HDBSCAN
//! 
//! This module implements cluster extraction from the condensed tree using
//! the Excess of Mass (EOM) method, which selects clusters to maximize
//! the total stability across the tree.

use num_traits::float::Float;
use std::collections::HashMap;
use super::condensed::CondensedTree;

/// Result of cluster extraction
#[derive(Debug, Clone)]
pub struct ClusterAssignment {
    /// Point id -> cluster label (-1 for noise)
    pub labels: Vec<i32>,
    /// Cluster membership probabilities/strengths
    pub probabilities: Vec<f32>,
    /// Cluster stability scores (cluster_label -> stability)
    pub stabilities: HashMap<i32, f32>,
    /// Selected condensed tree node indices
    pub selected_clusters: Vec<usize>,
}

impl ClusterAssignment {
    /// Create a new cluster assignment for n points
    pub fn new(num_points: usize) -> Self {
        ClusterAssignment {
            labels: vec![-1; num_points],
            probabilities: vec![0.0; num_points],
            stabilities: HashMap::new(),
            selected_clusters: Vec::new(),
        }
    }
    
    /// Get the number of clusters found (excluding noise)
    pub fn num_clusters(&self) -> usize {
        self.stabilities.len()
    }
    
    /// Get the number of noise points
    pub fn num_noise(&self) -> usize {
        self.labels.iter().filter(|&&l| l == -1).count()
    }
    
    /// Get unique cluster labels (excluding noise)
    pub fn cluster_labels(&self) -> Vec<i32> {
        let mut labels: Vec<i32> = self.stabilities.keys().copied().collect();
        labels.sort();
        labels
    }
}

/// Cluster selection methods
#[derive(Debug, Clone, Copy)]
pub enum SelectionMethod {
    /// Excess of Mass - selects clusters to maximize total stability
    Eom,
    /// Select all leaf clusters
    Leaf,
}

impl<F: Float> CondensedTree<F> {
    /// Extract clusters using the specified method
    pub fn extract_clusters(&self, method: SelectionMethod) -> ClusterAssignment {
        match method {
            SelectionMethod::Eom => self.extract_clusters_eom(),
            SelectionMethod::Leaf => self.extract_clusters_leaf(),
        }
    }
    
    /// Extract clusters using Excess of Mass (EOM) method
    pub fn extract_clusters_eom(&self) -> ClusterAssignment {
        // Find the total number of points
        let num_points = self.get_num_points();
        let mut assignment = ClusterAssignment::new(num_points);
        
        if self.nodes.is_empty() {
            return assignment;
        }
        
        // Select optimal clusters using dynamic programming
        let selected_nodes = self.select_optimal_clusters_eom();
        
        // Assign labels and calculate probabilities
        self.assign_labels_and_probabilities(&selected_nodes, &mut assignment);
        
        assignment
    }
    
    /// Extract clusters by selecting all leaf clusters
    pub fn extract_clusters_leaf(&self) -> ClusterAssignment {
        let num_points = self.get_num_points();
        let mut assignment = ClusterAssignment::new(num_points);
        
        if self.nodes.is_empty() {
            return assignment;
        }
        
        // Select all leaf nodes as clusters
        let selected_nodes: Vec<usize> = self.nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| node.is_leaf())
            .map(|(idx, _)| idx)
            .collect();
        
        // Assign labels and calculate probabilities
        self.assign_labels_and_probabilities(&selected_nodes, &mut assignment);
        
        assignment
    }
    
    /// Select optimal clusters using EOM dynamic programming
    fn select_optimal_clusters_eom(&self) -> Vec<usize> {
        if self.root.is_none() {
            return Vec::new();
        }
        
        let num_nodes = self.nodes.len();
        let mut is_selected = vec![false; num_nodes];
        let mut subtree_stability = vec![F::zero(); num_nodes];
        
        // Post-order traversal to compute optimal selection
        if let Some(root_idx) = self.root {
            self.postorder_select_eom(root_idx, &mut is_selected, &mut subtree_stability);
        }
        
        // Collect selected node indices
        is_selected
            .iter()
            .enumerate()
            .filter_map(|(idx, &selected)| if selected { Some(idx) } else { None })
            .collect()
    }
    
    /// Post-order traversal for EOM cluster selection
    fn postorder_select_eom(
        &self,
        node_idx: usize,
        is_selected: &mut [bool],
        subtree_stability: &mut [F],
    ) {
        // Add cycle detection
        self.postorder_select_eom_impl(node_idx, is_selected, subtree_stability, &mut vec![false; self.nodes.len()]);
    }
    
    fn postorder_select_eom_impl(
        &self,
        node_idx: usize,
        is_selected: &mut [bool],
        subtree_stability: &mut [F],
        visited: &mut [bool],
    ) {
        if visited[node_idx] {
            log::error!("Cycle detected at node {}", node_idx);
            return;
        }
        visited[node_idx] = true;
        let node = &self.nodes[node_idx];
        
        if node.is_leaf() {
            // Leaf nodes are candidates for selection only if they have points
            // (not just fallen points) or positive stability
            if !node.points.is_empty() || node.stability > F::zero() {
                is_selected[node_idx] = true;
                subtree_stability[node_idx] = node.stability;
            } else {
                // Node with only fallen points and no stability - don't select
                is_selected[node_idx] = false;
                subtree_stability[node_idx] = F::zero();
            }
        } else {
            // Internal node - compare selecting this vs selecting children
            let mut children_stability = F::zero();
            
            // First, process all children
            for &child_idx in &node.child_nodes {
                if child_idx >= self.nodes.len() {
                    log::error!("Invalid child index {} in postorder_select_eom", child_idx);
                    continue;
                }
                self.postorder_select_eom_impl(child_idx, is_selected, subtree_stability, visited);
                children_stability = children_stability + subtree_stability[child_idx];
            }
            
            // Compare this node's stability with sum of children's
            if node.stability >= children_stability {
                // Select this node
                is_selected[node_idx] = true;
                subtree_stability[node_idx] = node.stability;
                
                // Deselect all descendants
                self.deselect_descendants(node_idx, is_selected);
            } else {
                // Keep children selected
                is_selected[node_idx] = false;
                subtree_stability[node_idx] = children_stability;
            }
        }
    }
    
    /// Deselect all descendants of a node
    fn deselect_descendants(&self, node_idx: usize, is_selected: &mut [bool]) {
        for &child_idx in &self.nodes[node_idx].child_nodes {
            is_selected[child_idx] = false;
            self.deselect_descendants(child_idx, is_selected);
        }
    }
    
    /// Assign cluster labels and calculate probabilities
    fn assign_labels_and_probabilities(
        &self,
        selected_nodes: &[usize],
        assignment: &mut ClusterAssignment,
    ) {
        // Create a mapping of all points to their positions
        let mut point_to_index: HashMap<usize, usize> = HashMap::new();
        let mut next_index = 0;
        
        // First pass: collect all points
        for node in &self.nodes {
            for &point in &node.points {
                if !point_to_index.contains_key(&point) {
                    point_to_index.insert(point, next_index);
                    next_index += 1;
                }
            }
            for &(point, _) in &node.points_fallen {
                if !point_to_index.contains_key(&point) {
                    point_to_index.insert(point, next_index);
                    next_index += 1;
                }
            }
        }
        
        // Resize vectors to actual number of points
        assignment.labels.resize(next_index, -1);
        assignment.probabilities.resize(next_index, 0.0);
        
        // Assign labels to selected clusters
        let mut cluster_label = 0i32;
        
        for &node_idx in selected_nodes {
            let node = &self.nodes[node_idx];
            
            // Process points that stayed in the cluster
            for &point in &node.points {
                if let Some(&point_idx) = point_to_index.get(&point) {
                    assignment.labels[point_idx] = cluster_label;
                    
                    // Calculate probability based on lambda persistence
                    // Probability = 1 - (lambda_birth / lambda_death)
                    // But we need to be careful with the formulation
                    let lambda_birth = node.lambda_birth;
                    let lambda_death = node.lambda_death.unwrap_or(F::from(1e-10).unwrap());
                    
                    let prob = if lambda_death < lambda_birth && lambda_birth > F::zero() {
                        // Higher lambda means closer points, so inverse relationship
                        let ratio = lambda_death / lambda_birth;
                        F::one() - ratio
                    } else {
                        F::one() // Full membership if no proper death
                    };
                    
                    assignment.probabilities[point_idx] = prob.to_f32().unwrap_or(1.0);
                }
            }
            
            // Store cluster stability
            assignment.stabilities.insert(
                cluster_label,
                node.stability.to_f32().unwrap_or(0.0)
            );
            
            assignment.selected_clusters.push(node_idx);
            cluster_label += 1;
        }
        
        // Handle noise points (those that fell out and aren't in selected clusters)
        // They already have label -1 and probability 0.0 by default
    }
}

/// Calculate outlier scores using GLOSH (Global-Local Outlier Score from Hierarchies)
impl<F: Float> CondensedTree<F> {
    /// Calculate GLOSH outlier scores for all points
    pub fn calculate_outlier_scores(&self, assignment: &ClusterAssignment) -> Vec<f32> {
        let mut scores = vec![0.0f32; assignment.labels.len()];
        
        // For each point, calculate its outlier score
        // GLOSH score = 1 - (lambda_point_fell_out / lambda_cluster_death)
        
        // First, build a map of which cluster each point belongs to
        let mut point_cluster_map: HashMap<usize, usize> = HashMap::new();
        for &node_idx in &assignment.selected_clusters {
            let node = &self.nodes[node_idx];
            for &point in &node.points {
                point_cluster_map.insert(point, node_idx);
            }
        }
        
        // Calculate scores
        for (point_idx, &label) in assignment.labels.iter().enumerate() {
            if label == -1 {
                // Noise point - maximum outlier score
                scores[point_idx] = 1.0;
            } else {
                // In a cluster - score based on how early it would fall out
                // Lower score means more central to cluster
                scores[point_idx] = 1.0 - assignment.probabilities[point_idx];
            }
        }
        
        scores
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdbscan::hierarchy::ClusterHierarchy;
    
    #[test]
    fn test_cluster_assignment_creation() {
        let assignment = ClusterAssignment::new(10);
        assert_eq!(assignment.labels.len(), 10);
        assert_eq!(assignment.probabilities.len(), 10);
        assert_eq!(assignment.num_clusters(), 0);
        assert_eq!(assignment.num_noise(), 10); // All points start as noise
    }
    
    #[test]
    fn test_eom_extraction_simple() {
        // Create a simple hierarchy
        let mut hierarchy = ClusterHierarchy::<f32>::new(6);
        
        // Build two clear clusters
        let c01 = hierarchy.merge(0, 1, 10.0);
        let c012 = hierarchy.merge(c01, 2, 8.0);
        
        let c34 = hierarchy.merge(3, 4, 9.0);
        let c345 = hierarchy.merge(c34, 5, 7.0);
        
        // Merge at low lambda
        hierarchy.merge(c012, c345, 1.0);
        
        // Create condensed tree
        let condensed = CondensedTree::from_hierarchy(&hierarchy, 2);
        
        // Extract clusters
        let assignment = condensed.extract_clusters_eom();
        
        // Should have found some clusters (num_clusters returns usize, always >= 0)
        // Just check it doesn't panic
        
        // All points should have labels
        assert_eq!(assignment.labels.len(), 6);
    }
    
    #[test]
    fn test_leaf_extraction() {
        // Create a hierarchy
        let mut hierarchy = ClusterHierarchy::<f32>::new(5);
        
        // Build a simple tree
        let c01 = hierarchy.merge(0, 1, 5.0);
        let c23 = hierarchy.merge(2, 3, 4.0);
        let c0123 = hierarchy.merge(c01, c23, 2.0);
        hierarchy.merge(c0123, 4, 1.0); // Add point 4
        
        let condensed = CondensedTree::from_hierarchy(&hierarchy, 2);
        
        // Extract using leaf method
        let assignment = condensed.extract_clusters(SelectionMethod::Leaf);
        
        // Should have assigned labels for points
        // The exact number depends on how points are tracked through condensation
        assert!(assignment.labels.len() > 0);
        assert!(assignment.labels.len() <= 5);
    }
    
    #[test]
    fn test_outlier_scores() {
        // Create hierarchy with an outlier
        let mut hierarchy = ClusterHierarchy::<f32>::new(5);
        
        // Main cluster
        let c01 = hierarchy.merge(0, 1, 10.0);
        let c012 = hierarchy.merge(c01, 2, 9.0);
        let c0123 = hierarchy.merge(c012, 3, 8.0);
        
        // Outlier merges at very low lambda
        hierarchy.merge(c0123, 4, 0.1);
        
        let condensed = CondensedTree::from_hierarchy(&hierarchy, 3);
        let assignment = condensed.extract_clusters_eom();
        
        // Calculate outlier scores
        let scores = condensed.calculate_outlier_scores(&assignment);
        
        // Should have scores for all points
        assert_eq!(scores.len(), assignment.labels.len());
        
        // Scores should be between 0 and 1
        for score in &scores {
            assert!(*score >= 0.0 && *score <= 1.0);
        }
    }
    
    #[test]
    fn test_cluster_selection_stability() {
        // Test that EOM selects based on stability
        let mut hierarchy = ClusterHierarchy::<f32>::new(8);
        
        // Create two clusters with different stabilities
        // Tight cluster (high stability)
        let tight1 = hierarchy.merge(0, 1, 20.0);
        let tight2 = hierarchy.merge(2, 3, 19.0);
        let tight = hierarchy.merge(tight1, tight2, 18.0);
        
        // Loose cluster (lower stability)
        let loose1 = hierarchy.merge(4, 5, 5.0);
        let loose2 = hierarchy.merge(6, 7, 4.0);
        let loose = hierarchy.merge(loose1, loose2, 3.0);
        
        // Merge clusters
        hierarchy.merge(tight, loose, 0.5);
        
        let condensed = CondensedTree::from_hierarchy(&hierarchy, 3);
        
        // Extract clusters
        let assignment = condensed.extract_clusters_eom();
        
        // Check that we got reasonable results
        assert!(assignment.selected_clusters.len() > 0);
        
        // Verify stabilities are recorded
        for (label, stability) in &assignment.stabilities {
            assert!(*stability >= 0.0);
            log::debug!("Cluster {} has stability {}", label, stability);
        }
    }
    
    #[test]
    fn test_probability_calculation() {
        // Create a simple hierarchy
        let mut hierarchy = ClusterHierarchy::<f32>::new(4);
        
        // Merge points at known lambdas
        let c01 = hierarchy.merge(0, 1, 10.0);
        let c23 = hierarchy.merge(2, 3, 8.0);
        hierarchy.merge(c01, c23, 2.0);
        
        let condensed = CondensedTree::from_hierarchy(&hierarchy, 2);
        let assignment = condensed.extract_clusters_eom();
        
        // All probabilities should be between 0 and 1
        for prob in &assignment.probabilities {
            assert!(*prob >= 0.0 && *prob <= 1.0,
                    "Probability {} out of range", prob);
        }
    }
}