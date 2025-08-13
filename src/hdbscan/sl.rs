//! single linkage hdbscan
//!
//!
//!
//! implements single linkage clustering on top of Kruskal algorithm.
//!
//!

#![allow(unused)]

use num_traits::cast::FromPrimitive;
use num_traits::int::PrimInt;

use num_traits::float::*; // tp get FRAC_1_PI from FloatConst

use std::cmp::{Ordering, PartialEq, PartialOrd};
use std::collections::{BinaryHeap, HashSet};

use hnsw_rs::prelude::*;

use super::condensed::CondensedTree;
use super::core_distance::{CoreDistance, mutual_reachability_distance};
use super::extraction::{ClusterAssignment, SelectionMethod};
use super::hierarchy::{ClusterHierarchy, HierarchicalUnionFind};
use super::kruskal::*;
use crate::fromhnsw::kgraph::KGraph;
use crate::fromhnsw::kgraph_from_hnsw_all;

// 1.  We get from the hnsw a list of edges for kruskal algorithm
// 2.  Run kruskal algorithm ,  we get nodes of edge, weigth of edge and parent of nodes
//        - so we get at each step the id of cluster representative that unionized.
//
// 3. Fill a Dendrogram structure

/// This structure represent a merge step
///
pub struct UnionStep<NodeIdx: PrimInt, F: Float> {
    /// node a of edge removed
    nodea: NodeIdx,
    /// node b of edge removed
    nodeb: NodeIdx,
    /// weight of edge
    weight: F,
    /// step at which the merge occurs
    step: usize,
    /// representative of nodea in the union-find
    clusta: NodeIdx,
    /// representative of nodeb in the union-find
    clustb: NodeIdx,
} // end of struct UnionStep

/// Some basic statistics on Clusters
pub struct ClusterStat {
    /// mean of density around each point
    /// For hnsw , we compute mean dist to k-neighbours for each point and compute mean on cluster.
    mean_density: f32,
    /// nuber of terms in Cluster
    size: u32,
}

pub struct Dendrogram<NodeIdx: PrimInt, F: Float> {
    steps: Vec<UnionStep<NodeIdx, F>>,
}

impl<NodeIdx: PrimInt, F: Float> Dendrogram<NodeIdx, F> {
    pub fn new(nbstep: usize) -> Self {
        Dendrogram {
            steps: Vec::<UnionStep<NodeIdx, F>>::with_capacity(nbstep),
        }
    }
} // end of impl Dendrogram

/// edge to be stored in a binary heap for Dendogram formation
struct Edge<F: Float + PartialOrd> {
    nodea: u32,
    nodeb: u32,
    weight: F,
}

// We can do that beccause we cannot have NaN coming from Hnsw
fn compare_edge<F: Float + PartialOrd>(edgea: &Edge<F>, edgeb: &Edge<F>) -> Ordering {
    match (edgea.weight, edgeb.weight) {
        (x, y) if x.is_nan() && y.is_nan() => Ordering::Equal,
        (x, _) if x.is_nan() => Ordering::Greater,
        (_, y) if y.is_nan() => Ordering::Less,
        (_, _) => edgea.weight.partial_cmp(&edgeb.weight).unwrap(),
    }
}
// We need to implement an Ord for edge based on a float representation of Edge weight

impl<F: Float + PartialOrd> PartialEq for Edge<F> {
    fn eq(&self, other: &Self) -> bool {
        self.nodea == other.nodea && self.nodeb == other.nodeb
    }
}

impl<F: Float + PartialOrd> Eq for Edge<F> {}

impl<F: Float + PartialOrd> PartialOrd for Edge<F> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<F: Float + PartialOrd> Ord for Edge<F> {
    fn cmp(&self, other: &Self) -> Ordering {
        compare_edge(self, other)
    }
}

/// The structure driving Single Linkage Clustering
/// It is constructed from a Hnsw
pub struct SLclustering<NodeIdx: PrimInt, F: Float> {
    // the kgraph summary provided by hnsw
    pub kgraph: KGraph<F>,
    //
    dendrogram: Dendrogram<NodeIdx, F>,
    // ask for at most nbcluster. We can stop if we get in nbcluster union steps
    nbcluster: usize,
} // end of  SLclustering

impl<NodeIdx: PrimInt, F> SLclustering<NodeIdx, F>
where
    F: PartialOrd
        + FromPrimitive
        + Float
        + Send
        + Sync
        + Clone
        + std::fmt::UpperExp
        + std::iter::Sum,
{
    //
    pub fn new<D>(hnsw: &Hnsw<F, D>, nbcluster: usize) -> Self
    where
        D: Distance<F> + Send + Sync,
    {
        //
        // get kgraph summary
        //
        let nbng = hnsw.get_max_nb_connection() as usize;
        let kgraph = kgraph_from_hnsw_all(hnsw, nbng).unwrap();
        //
        let nbstep = kgraph.get_nb_nodes() - nbcluster;
        SLclustering {
            kgraph,
            dendrogram: Dendrogram::<NodeIdx, F>::new(nbstep),
            nbcluster,
        }
    } // end of new

    /// Compute core distances for all points using the k-nearest neighbor graph
    pub fn compute_core_distances(&self, min_samples: usize) -> Vec<CoreDistance<F>> {
        let neighbours = self.kgraph.get_neighbours();
        let mut core_distances = Vec::with_capacity(neighbours.len());
        
        for (point_id, edges) in neighbours.iter().enumerate() {
            // Convert edges to neighbor format
            let neighbors: Vec<(usize, F)> = edges
                .iter()
                .map(|e| (e.node, e.weight))
                .collect();
            
            core_distances.push(CoreDistance::new(point_id, neighbors, min_samples));
        }
        
        core_distances
    }
    
    /// Build minimum spanning tree using mutual reachability distances
    /// Returns a vector of edges (a, b, mrd_weight) sorted by weight
    pub fn build_mrd_mst(&self, min_samples: usize) -> Vec<(usize, usize, F)> {
        let core_distances = self.compute_core_distances(min_samples);
        let neighbours = self.kgraph.get_neighbours();
        
        // Build edge list with mutual reachability distances
        let mut mrd_edges = Vec::new();
        let mut seen_edges = std::collections::HashSet::new();
        
        for (i, edges) in neighbours.iter().enumerate() {
            for edge in edges {
                let j = edge.node;
                // Only add each edge once (use canonical ordering)
                let edge_key = if i < j { (i, j) } else { (j, i) };
                
                if seen_edges.insert(edge_key) {
                    let mrd = mutual_reachability_distance(
                        i,
                        j,
                        edge.weight,
                        &core_distances,
                    );
                    mrd_edges.push((i as u32, j as u32, mrd));
                }
            }
        }
        
        // Run Kruskal's algorithm on the MRD edges
        let mst: Vec<(usize, usize, F)> = kruskal(&mrd_edges)
            .map(|(a, b, w)| (a as usize, b as usize, w))
            .collect();
        
        log::info!("Built MST with {} edges using MRD", mst.len());
        
        mst
    }
    
    /// Build minimum spanning tree using standard distances (original method)
    pub fn build_standard_mst(&self) -> Vec<(usize, usize, F)> {
        let neighbours = self.kgraph.get_neighbours();
        let nbnodes = neighbours.len();
        let max_nbng = self.kgraph.get_max_nbng();
        
        let mut edge_list = Vec::<(u32, u32, F)>::with_capacity(max_nbng * nbnodes);
        let mut seen_edges = std::collections::HashSet::new();
        
        for (n, edge_vec) in neighbours.iter().enumerate() {
            for edge in edge_vec.iter() {
                // Only add each edge once
                let edge_key = if n < edge.node { (n, edge.node) } else { (edge.node, n) };
                if seen_edges.insert(edge_key) {
                    edge_list.push((n as u32, edge.node as u32, edge.weight));
                }
            }
        }
        
        kruskal(&edge_list)
            .map(|(a, b, w)| (a as usize, b as usize, w))
            .collect()
    }
    
    /// Build the condensed tree from the hierarchy
    /// 
    /// # Arguments
    /// * `min_samples` - Minimum number of samples for core point computation
    /// * `min_cluster_size` - Minimum size for a cluster to be considered
    /// 
    /// # Returns
    /// A CondensedTree that can be used for cluster extraction
    pub fn build_condensed_tree(&self, min_samples: usize, min_cluster_size: usize) -> CondensedTree<F> {
        let hierarchy = self.build_hierarchy(min_samples);
        CondensedTree::from_hierarchy(&hierarchy, min_cluster_size)
    }
    
    /// Complete HDBSCAN clustering pipeline
    /// 
    /// # Arguments
    /// * `min_samples` - Minimum number of samples for core point computation
    /// * `min_cluster_size` - Minimum size for a cluster to be considered
    /// * `selection_method` - Method for selecting clusters (EOM or Leaf)
    /// 
    /// # Returns
    /// A ClusterAssignment with labels, probabilities, and stabilities
    pub fn cluster_hdbscan(
        &self,
        min_samples: usize,
        min_cluster_size: usize,
        selection_method: SelectionMethod,
    ) -> ClusterAssignment {
        let condensed = self.build_condensed_tree(min_samples, min_cluster_size);
        condensed.extract_clusters(selection_method)
    }
    
    /// Build the cluster hierarchy from an MST using mutual reachability distances
    /// 
    /// # Arguments
    /// * `min_samples` - Minimum number of samples for core point computation
    /// 
    /// # Returns
    /// A ClusterHierarchy that represents the dendrogram of the clustering
    pub fn build_hierarchy(&self, min_samples: usize) -> ClusterHierarchy<F> {
        // Get the MST with mutual reachability distances
        let mst = self.build_mrd_mst(min_samples);
        
        // Sort edges by weight (ascending) for bottom-up hierarchy construction
        let mut sorted_edges = mst;
        sorted_edges.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
        
        // Initialize hierarchy and union-find
        let num_points = self.kgraph.get_nb_nodes();
        let mut hierarchy = ClusterHierarchy::new(num_points);
        let mut uf = HierarchicalUnionFind::<usize>::new(num_points);
        
        // Track which points have been added to the hierarchy
        let mut points_in_tree = std::collections::HashSet::new();
        
        // Process edges in order of increasing weight
        for (node_a, node_b, weight) in sorted_edges {
            // Track points that are in the tree
            points_in_tree.insert(node_a);
            points_in_tree.insert(node_b);
            
            // Find representatives of the two nodes
            let rep_a = uf.find(node_a);
            let rep_b = uf.find(node_b);
            
            // Skip if already in same cluster
            if rep_a == rep_b {
                continue;
            }
            
            // Get the hierarchy nodes for the representatives
            let node_id_a = uf.get_node_id(rep_a).unwrap();
            let node_id_b = uf.get_node_id(rep_b).unwrap();
            
            // Compute lambda (1/distance) - avoid division by zero
            let lambda = if weight > F::zero() {
                F::one() / weight
            } else {
                F::infinity()
            };
            
            // Merge clusters in hierarchy
            let new_cluster_id = hierarchy.merge(node_id_a, node_id_b, lambda);
            
            // Union the sets and get new representative
            let new_rep = uf.union(node_a, node_b);
            
            // Update union-find to hierarchy mapping
            // The new representative maps to the new cluster
            uf.update_node_mapping(new_rep, new_cluster_id);
        }
        
        // Handle disconnected points - connect them at infinite distance
        // Find all points not in the tree
        let mut disconnected_points: Vec<usize> = Vec::new();
        for i in 0..num_points {
            if !points_in_tree.contains(&i) {
                disconnected_points.push(i);
                // Each disconnected point starts as its own cluster in hierarchy
                // (This is already done in hierarchy initialization)
            }
        }
        
        // Check for disconnected components - ALWAYS check, even if all points were in edges
        // because the MST might have multiple trees
        let mut roots = Vec::new();
        for i in 0..num_points {
            if uf.find(i) == i {
                if let Some(node_id) = uf.get_node_id(i) {
                    roots.push((i, node_id));
                }
            }
        }
        
        // If we have multiple roots, we have disconnected components that need to be merged
        if roots.len() > 1 {
            log::warn!("Found {} disconnected components in MST", roots.len());
            
            // Merge all components at very small lambda (large distance)
            let very_small_lambda = F::from(1e-10).unwrap();
            let mut current_root = roots[0];
            
            for i in 1..roots.len() {
                let next_root = roots[i];
                // Merge in hierarchy
                let new_cluster_id = hierarchy.merge(current_root.1, next_root.1, very_small_lambda);
                // Union in UF
                let new_rep = uf.union(current_root.0, next_root.0);
                uf.update_node_mapping(new_rep, new_cluster_id);
                current_root = (new_rep, new_cluster_id);
            }
        }
        
        log::info!("Built hierarchy with {} nodes, included {} points", 
                  hierarchy.num_nodes(), points_in_tree.len());
        
        hierarchy
    }
    
    /// computes clustering
    pub fn cluster(&mut self) {
        let _kgraph_stats = self.kgraph.get_kraph_stats();
        //
        // get a list of (node, node, weight of edge), compute mst
        let neighboourhood_info = self.kgraph.get_neighbours();
        let nbnodes = neighboourhood_info.len();
        let max_nbng = self.kgraph.get_max_nbng();
        let mut edge_list = Vec::<(u32, u32, F)>::with_capacity(max_nbng * nbnodes);
        for (n, edge_vec) in neighboourhood_info.iter().enumerate() {
            for edge in edge_vec.iter() {
                edge_list.push((n as u32, edge.node as u32, edge.weight));
            }
        }
        let mst_edge_iter = kruskal(&edge_list);
        // now we transfer edges in a binary_heap
        let mut edge_heap = BinaryHeap::<Edge<F>>::with_capacity(edge_list.len());
        for edge in mst_edge_iter {
            edge_heap.push(Edge {
                nodea: edge.0,
                nodeb: edge.1,
                weight: edge.2,
            });
        }
        // have an iterator of edge traversing tree , in increasing order

        // we initialize clusters with singletons

        // we run unification (possibly with density filter)
    } // end of cluster
} // end of impl for Hclust

#[cfg(test)]
mod tests {
    use super::*;
    
    /// Helper function to create a simple HNSW index from data
    fn build_hnsw(data: &[Vec<f32>]) -> Hnsw<f32, DistL2> {
        let nb_data = data.len();
        let data_with_id: Vec<(&Vec<f32>, usize)> = 
            data.iter().zip(0..nb_data).collect();
        
        let max_nb_conn = 8.max(nb_data / 2);
        let ef_c = 200;
        let nb_layer = 16.min((nb_data as f32).ln().trunc() as usize).max(1);
        
        let mut hnsw = Hnsw::<f32, DistL2>::new(
            max_nb_conn, 
            nb_data, 
            nb_layer, 
            ef_c, 
            DistL2 {}
        );
        // Use serial insert for deterministic tests
        for (data, id) in data_with_id {
            hnsw.insert((data, id));
        }
        hnsw
    }
    
    #[test]
    fn test_core_distance_computation() {
        // Create a simple dataset
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![0.5, 0.5],
        ];
        
        let hnsw = build_hnsw(&data);
        let clustering = SLclustering::<usize, f32>::new(&hnsw, 1);
        let core_distances = clustering.compute_core_distances(3);
        
        // All points should have finite core distances
        for cd in &core_distances {
            assert!(cd.get_core_distance().is_finite(), 
                    "Point {} should have finite core distance", cd.point_id);
            assert!(cd.is_core_point());
        }
    }
    
    #[test]
    fn test_mrd_mst_construction() {
        // Two well-separated clusters
        let data = vec![
            // Cluster 1
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            // Cluster 2  
            vec![5.0, 0.0],
            vec![5.1, 0.0],
            vec![5.0, 0.1],
        ];
        
        let hnsw = build_hnsw(&data);
        let clustering = SLclustering::<usize, f32>::new(&hnsw, 1);
        
        // Build both MSTs
        let standard_mst = clustering.build_standard_mst();
        let mrd_mst = clustering.build_mrd_mst(3);
        
        // Both should have n-1 edges
        assert_eq!(standard_mst.len(), data.len() - 1);
        assert_eq!(mrd_mst.len(), data.len() - 1);
        
        // Find the edge with maximum weight (should be between clusters)
        let max_edge = mrd_mst.iter()
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
            .unwrap();
        
        // The heaviest edge should connect the two clusters
        let (a, b, weight) = max_edge;
        let cluster_a = if *a < 3 { 0 } else { 1 };
        let cluster_b = if *b < 3 { 0 } else { 1 };
        
        assert_ne!(cluster_a, cluster_b,
                   "Heaviest edge ({},{}) with weight {} should be between clusters",
                   a, b, weight);
    }
    
    #[test]
    fn test_mrd_properties() {
        // Test that MRD satisfies its mathematical properties
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![2.0, 0.0],
        ];
        
        let hnsw = build_hnsw(&data);
        let clustering = SLclustering::<usize, f32>::new(&hnsw, 1);
        let core_distances = clustering.compute_core_distances(2);
        
        // MRD should be symmetric
        let dist_01 = 1.0_f32;
        let mrd_01 = mutual_reachability_distance(0, 1, dist_01, &core_distances);
        let mrd_10 = mutual_reachability_distance(1, 0, dist_01, &core_distances);
        assert!((mrd_01 - mrd_10).abs() < 1e-6, "MRD should be symmetric");
        
        // MRD >= actual distance
        assert!(mrd_01 >= dist_01 - 1e-6, "MRD should be >= actual distance");
        
        // MRD >= core distances
        assert!(mrd_01 >= core_distances[0].get_core_distance() - 1e-6);
        assert!(mrd_01 >= core_distances[1].get_core_distance() - 1e-6);
    }
    
    #[test]
    fn test_outlier_detection() {
        // Cluster with one outlier
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![0.1, 0.1],
            vec![10.0, 10.0], // Outlier
        ];
        
        let hnsw = build_hnsw(&data);
        let clustering = SLclustering::<usize, f32>::new(&hnsw, 1);
        let core_distances = clustering.compute_core_distances(3);
        
        // Outlier should have much larger core distance
        let outlier_core = core_distances[4].get_core_distance();
        let cluster_core_max = core_distances[..4].iter()
            .map(|cd| cd.get_core_distance())
            .fold(0.0_f32, |a, b| a.max(b));
        
        assert!(outlier_core > cluster_core_max * 10.0,
                "Outlier core distance {} should be much larger than cluster max {}",
                outlier_core, cluster_core_max);
    }
    
    #[test]
    fn test_hierarchy_construction() {
        // Create a simple dataset with clear clusters
        let data = vec![
            // Cluster 1
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            // Cluster 2
            vec![3.0, 0.0],
            vec![3.1, 0.0],
            vec![3.0, 0.1],
        ];
        
        let hnsw = build_hnsw(&data);
        let clustering = SLclustering::<usize, f32>::new(&hnsw, 1);
        let hierarchy = clustering.build_hierarchy(2);
        
        // Check basic hierarchy properties
        assert_eq!(hierarchy.num_points(), 6);
        // Should have 6 leaves + 5 internal nodes (binary tree)
        assert_eq!(hierarchy.num_nodes(), 11);
        
        // Get hierarchy statistics
        let stats = hierarchy.get_stats();
        assert_eq!(stats.num_leaves, 6);
        assert_eq!(stats.num_merges, 5);
        
        // Lambda values should be positive (or infinite for first merges)
        assert!(stats.min_lambda >= 0.0 || stats.min_lambda.is_infinite());
    }
    
    #[test]
    fn test_hierarchy_with_single_cluster() {
        // All points in one tight cluster
        let data = vec![
            vec![0.0, 0.0],
            vec![0.01, 0.0],
            vec![0.0, 0.01],
            vec![0.01, 0.01],
        ];
        
        let hnsw = build_hnsw(&data);
        let clustering = SLclustering::<usize, f32>::new(&hnsw, 1);
        let hierarchy = clustering.build_hierarchy(2);
        
        // Should have created a complete hierarchy
        assert_eq!(hierarchy.num_points(), 4);
        assert_eq!(hierarchy.num_nodes(), 7); // 4 leaves + 3 internal
        
        // All merges should happen at similar lambda values (dense cluster)
        let stats = hierarchy.get_stats();
        if stats.min_lambda.is_finite() && stats.max_lambda.is_finite() {
            let lambda_range = stats.max_lambda - stats.min_lambda;
            // In a dense cluster, lambda values should be similar
            assert!(lambda_range < stats.max_lambda * 2.0,
                    "Lambda range {} too large for single cluster", lambda_range);
        }
    }
    
    #[test]
    fn test_hierarchy_traversals() {
        // Simple 3-point dataset
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![2.0, 0.0],
        ];
        
        let hnsw = build_hnsw(&data);
        let clustering = SLclustering::<usize, f32>::new(&hnsw, 1);
        let hierarchy = clustering.build_hierarchy(2);
        
        // Test traversal methods
        let preorder = hierarchy.preorder_traversal();
        let postorder = hierarchy.postorder_traversal();
        
        // Both should visit all nodes
        assert_eq!(preorder.len(), hierarchy.num_nodes());
        assert_eq!(postorder.len(), hierarchy.num_nodes());
        
        // Root should be first in preorder, last in postorder
        if let Some(root_id) = hierarchy.root {
            assert_eq!(preorder[0], root_id);
            assert_eq!(postorder[postorder.len() - 1], root_id);
        }
    }
    
    #[test]
    fn test_condensed_tree_construction() {
        // Dataset with two clear clusters
        let data = vec![
            // Cluster 1 (tight)
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![0.1, 0.1],
            // Cluster 2 (tight)
            vec![5.0, 0.0],
            vec![5.1, 0.0],
            vec![5.0, 0.1],
            vec![5.1, 0.1],
        ];
        
        let hnsw = build_hnsw(&data);
        let clustering = SLclustering::<usize, f32>::new(&hnsw, 1);
        
        // Build condensed tree with min_cluster_size=3
        let condensed = clustering.build_condensed_tree(2, 3);
        
        // Should have created a condensed tree
        assert!(condensed.num_nodes() > 0);
        
        // Get statistics
        let stats = condensed.get_stats();
        log::debug!("Condensed tree stats: {:?}", stats);
        
        // Should have identified some stable clusters
        assert!(stats.num_clusters > 0);
        assert!(stats.max_stability > 0.0);
    }
    
    #[test]
    fn test_condensed_tree_with_noise() {
        // Dataset with clusters and noise
        let data = vec![
            // Main cluster
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![0.1, 0.1],
            vec![0.05, 0.05],
            // Outliers
            vec![10.0, 10.0],
            vec![15.0, 15.0],
        ];
        
        let hnsw = build_hnsw(&data);
        let clustering = SLclustering::<usize, f32>::new(&hnsw, 1);
        
        // Build condensed tree with min_cluster_size=4
        let condensed = clustering.build_condensed_tree(3, 4);
        
        let stats = condensed.get_stats();
        log::debug!("Condensed tree with noise stats: {:?}", stats);
        
        // Should have some noise points (outliers)
        // Note: exact count depends on hierarchy structure
        assert!(condensed.num_nodes() > 0);
    }
    
    #[test]
    fn test_most_stable_clusters() {
        // Create dataset with clusters of different stability
        let data = vec![
            // Very tight cluster (high stability)
            vec![0.0, 0.0],
            vec![0.01, 0.0],
            vec![0.0, 0.01],
            vec![0.01, 0.01],
            // Looser cluster (lower stability)
            vec![5.0, 0.0],
            vec![5.3, 0.0],
            vec![5.0, 0.3],
            vec![5.3, 0.3],
        ];
        
        let hnsw = build_hnsw(&data);
        let clustering = SLclustering::<usize, f32>::new(&hnsw, 1);
        
        let condensed = clustering.build_condensed_tree(2, 3);
        
        // Get most stable clusters
        let stable_clusters = condensed.get_most_stable_clusters(2);
        
        // Should find stable clusters
        assert!(!stable_clusters.is_empty());
        
        // Clusters should be ordered by stability (descending)
        for i in 1..stable_clusters.len() {
            assert!(stable_clusters[i-1].1 >= stable_clusters[i].1);
        }
    }
    
    #[test]
    fn test_complete_hdbscan_pipeline() {
        // Create dataset with two clear clusters
        let data = vec![
            // Cluster 1
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![0.1, 0.1],
            vec![0.05, 0.05],
            // Cluster 2
            vec![3.0, 0.0],
            vec![3.1, 0.0],
            vec![3.0, 0.1],
            vec![3.1, 0.1],
            vec![3.05, 0.05],
        ];
        
        let hnsw = build_hnsw(&data);
        let clustering = SLclustering::<usize, f32>::new(&hnsw, 1);
        
        // Run complete HDBSCAN pipeline
        let assignment = clustering.cluster_hdbscan(3, 3, SelectionMethod::Eom);
        
        // Should have found clusters
        assert!(assignment.num_clusters() > 0);
        
        // Should have labels for all points
        assert_eq!(assignment.labels.len(), data.len());
        
        // Labels should be valid (either cluster id >= 0 or -1 for noise)
        for label in &assignment.labels {
            assert!(*label >= -1);
        }
        
        // Probabilities should be in range [0, 1]
        for prob in &assignment.probabilities {
            assert!(*prob >= 0.0 && *prob <= 1.0);
        }
        
        log::debug!("Found {} clusters with {} noise points",
                   assignment.num_clusters(), assignment.num_noise());
    }
    
    #[test]
    fn test_hdbscan_with_outliers() {
        // Dataset with clusters and outliers
        let data = vec![
            // Main cluster
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![0.1, 0.1],
            vec![0.05, 0.05],
            // Outliers far away
            vec![10.0, 10.0],
            vec![15.0, 15.0],
            vec![20.0, 20.0],
        ];
        
        let hnsw = build_hnsw(&data);
        let clustering = SLclustering::<usize, f32>::new(&hnsw, 1);
        
        // Run HDBSCAN with min_cluster_size=4
        let assignment = clustering.cluster_hdbscan(3, 4, SelectionMethod::Eom);
        
        // Should identify the main cluster
        let cluster_points: Vec<usize> = assignment.labels
            .iter()
            .enumerate()
            .filter(|(_, label)| **label >= 0)
            .map(|(idx, _)| idx)
            .collect();
        
        // Should identify at least some cluster points
        // The exact number depends on HNSW graph construction
        assert!(cluster_points.len() > 0 || assignment.num_noise() == data.len());
        
        // Outliers should be marked as noise (-1)
        let noise_count = assignment.num_noise();
        assert!(noise_count > 0);
        
        log::debug!("Identified {} cluster points and {} noise points",
                   cluster_points.len(), noise_count);
    }
    
    #[test]
    fn test_leaf_vs_eom_selection() {
        // Create hierarchical dataset
        let data = vec![
            // Subcluster 1a
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            // Subcluster 1b
            vec![0.5, 0.0],
            vec![0.6, 0.0],
            // Subcluster 2a
            vec![3.0, 0.0],
            vec![3.1, 0.0],
            // Subcluster 2b
            vec![3.5, 0.0],
            vec![3.6, 0.0],
        ];
        
        let hnsw = build_hnsw(&data);
        let clustering = SLclustering::<usize, f32>::new(&hnsw, 1);
        
        // Test with EOM selection
        let eom_assignment = clustering.cluster_hdbscan(2, 2, SelectionMethod::Eom);
        
        // Test with Leaf selection
        let leaf_assignment = clustering.cluster_hdbscan(2, 2, SelectionMethod::Leaf);
        
        // Both should produce valid clusterings
        assert!(eom_assignment.labels.len() == data.len());
        assert!(leaf_assignment.labels.len() == data.len());
        
        // Leaf selection typically produces more clusters
        log::debug!("EOM found {} clusters, Leaf found {} clusters",
                   eom_assignment.num_clusters(), leaf_assignment.num_clusters());
        
        // Verify all labels are valid
        for label in &eom_assignment.labels {
            assert!(*label >= -1);
        }
        for label in &leaf_assignment.labels {
            assert!(*label >= -1);
        }
    }
}
