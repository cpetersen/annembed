//! Hierarchy construction for HDBSCAN
//! 
//! This module implements the hierarchical clustering tree structure that forms
//! the basis of HDBSCAN. The hierarchy is built from the minimum spanning tree
//! using mutual reachability distances.

use num_traits::float::Float;
use num_traits::int::PrimInt;
use std::collections::HashMap;

/// Represents a node in the cluster hierarchy
#[derive(Debug, Clone)]
pub struct HierarchyNode<F: Float> {
    /// Unique identifier for this node
    pub id: usize,
    /// Points contained in this cluster (for leaf nodes, just one point)
    pub points: Vec<usize>,
    /// Lambda value (1/distance) at which this cluster forms
    pub lambda_birth: F,
    /// Lambda value at which this cluster splits (None if still active)
    pub lambda_death: Option<F>,
    /// Parent cluster id (None for root)
    pub parent: Option<usize>,
    /// Child cluster ids
    pub children: Vec<usize>,
    /// Number of points in this cluster
    pub size: usize,
}

impl<F: Float> HierarchyNode<F> {
    /// Create a new leaf node for a single point
    pub fn new_leaf(id: usize, point_id: usize) -> Self {
        HierarchyNode {
            id,
            points: vec![point_id],
            lambda_birth: F::infinity(),
            lambda_death: None,
            parent: None,
            children: vec![],
            size: 1,
        }
    }
    
    /// Create a new internal node by merging two clusters
    pub fn new_merge(id: usize, child_a: &Self, child_b: &Self, lambda: F) -> Self {
        let mut points = child_a.points.clone();
        points.extend(&child_b.points);
        
        HierarchyNode {
            id,
            points: points.clone(),
            lambda_birth: lambda,
            lambda_death: None,
            parent: None,
            children: vec![child_a.id, child_b.id],
            size: points.len(),
        }
    }
    
    /// Check if this is a leaf node
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
    
    /// Get the stability of this cluster
    /// Stability = sum over all points of (lambda_death - lambda_birth)
    pub fn stability(&self) -> F {
        if let Some(lambda_death) = self.lambda_death {
            F::from(self.size).unwrap() * (lambda_death - self.lambda_birth)
        } else {
            F::zero() // Still active, no stability yet
        }
    }
}

/// The complete cluster hierarchy
#[derive(Debug)]
pub struct ClusterHierarchy<F: Float> {
    /// All nodes in the hierarchy
    pub nodes: Vec<HierarchyNode<F>>,
    /// Map from point_id to leaf node_id
    point_to_leaf: HashMap<usize, usize>,
    /// Root node id (last node created)
    pub root: Option<usize>,
    /// Next available node id
    next_id: usize,
}

impl<F: Float> ClusterHierarchy<F> {
    /// Create a new hierarchy with singleton clusters for each point
    pub fn new(num_points: usize) -> Self {
        let mut hierarchy = ClusterHierarchy {
            nodes: Vec::with_capacity(2 * num_points - 1), // Binary tree has 2n-1 nodes
            point_to_leaf: HashMap::new(),
            root: None,
            next_id: 0,
        };
        
        // Initialize with singleton clusters (leaf nodes)
        for point_id in 0..num_points {
            let node = HierarchyNode::new_leaf(hierarchy.next_id, point_id);
            hierarchy.point_to_leaf.insert(point_id, node.id);
            hierarchy.nodes.push(node);
            hierarchy.next_id += 1;
        }
        
        hierarchy
    }
    
    /// Merge two clusters at the given lambda value
    /// Returns the id of the new parent cluster
    pub fn merge(&mut self, cluster_a_id: usize, cluster_b_id: usize, lambda: F) -> usize {
        // Mark children as dying at this lambda
        self.nodes[cluster_a_id].lambda_death = Some(lambda);
        self.nodes[cluster_b_id].lambda_death = Some(lambda);
        
        // Create new parent cluster
        let cluster_a = &self.nodes[cluster_a_id];
        let cluster_b = &self.nodes[cluster_b_id];
        let new_node = HierarchyNode::new_merge(self.next_id, cluster_a, cluster_b, lambda);
        
        let new_id = new_node.id;
        
        // Update parent pointers of children
        self.nodes[cluster_a_id].parent = Some(new_id);
        self.nodes[cluster_b_id].parent = Some(new_id);
        
        // Add the new node
        self.nodes.push(new_node);
        self.next_id += 1;
        
        // Update root (the last merge will be the root)
        self.root = Some(new_id);
        
        new_id
    }
    
    /// Get the number of nodes in the hierarchy
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }
    
    /// Get the number of points (leaf nodes)
    pub fn num_points(&self) -> usize {
        self.point_to_leaf.len()
    }
    
    /// Get the leaf node for a given point
    pub fn get_leaf_node(&self, point_id: usize) -> Option<&HierarchyNode<F>> {
        self.point_to_leaf.get(&point_id)
            .and_then(|&node_id| self.nodes.get(node_id))
    }
    
    /// Get a node by its id
    pub fn get_node(&self, node_id: usize) -> Option<&HierarchyNode<F>> {
        self.nodes.get(node_id)
    }
    
    /// Get all nodes at a specific level (distance from leaves)
    pub fn get_nodes_at_level(&self, level: usize) -> Vec<&HierarchyNode<F>> {
        self.nodes.iter()
            .filter(|node| self.get_node_level(node.id) == level)
            .collect()
    }
    
    /// Calculate the level (height) of a node in the tree
    fn get_node_level(&self, node_id: usize) -> usize {
        let node = &self.nodes[node_id];
        if node.is_leaf() {
            0
        } else {
            // Level is 1 + max level of children
            node.children.iter()
                .map(|&child_id| self.get_node_level(child_id))
                .max()
                .unwrap_or(0) + 1
        }
    }
    
    /// Get statistics about the hierarchy
    pub fn get_stats(&self) -> HierarchyStats<F> {
        let mut min_lambda = F::infinity();
        let mut max_lambda = F::neg_infinity();
        let mut total_merges = 0;
        
        for node in &self.nodes {
            if !node.is_leaf() {
                total_merges += 1;
                if node.lambda_birth < min_lambda {
                    min_lambda = node.lambda_birth;
                }
                if node.lambda_birth > max_lambda {
                    max_lambda = node.lambda_birth;
                }
            }
        }
        
        HierarchyStats {
            num_nodes: self.nodes.len(),
            num_leaves: self.point_to_leaf.len(),
            num_merges: total_merges,
            min_lambda,
            max_lambda,
            tree_height: self.root.map(|r| self.get_node_level(r)).unwrap_or(0),
        }
    }
    
    /// Traverse the hierarchy in pre-order (parent before children)
    pub fn preorder_traversal(&self) -> Vec<usize> {
        let mut result = Vec::new();
        if let Some(root) = self.root {
            self.preorder_recursive(root, &mut result);
        }
        result
    }
    
    fn preorder_recursive(&self, node_id: usize, result: &mut Vec<usize>) {
        result.push(node_id);
        let node = &self.nodes[node_id];
        for &child_id in &node.children {
            self.preorder_recursive(child_id, result);
        }
    }
    
    /// Traverse the hierarchy in post-order (children before parent)
    pub fn postorder_traversal(&self) -> Vec<usize> {
        let mut result = Vec::new();
        if let Some(root) = self.root {
            self.postorder_recursive(root, &mut result);
        }
        result
    }
    
    fn postorder_recursive(&self, node_id: usize, result: &mut Vec<usize>) {
        let node = &self.nodes[node_id];
        for &child_id in &node.children {
            self.postorder_recursive(child_id, result);
        }
        result.push(node_id);
    }
}

/// Statistics about a cluster hierarchy
#[derive(Debug)]
pub struct HierarchyStats<F: Float> {
    pub num_nodes: usize,
    pub num_leaves: usize,
    pub num_merges: usize,
    pub min_lambda: F,
    pub max_lambda: F,
    pub tree_height: usize,
}

/// Extended Union-Find structure that tracks cluster representatives
/// and allows us to map between Union-Find representatives and hierarchy nodes
pub struct HierarchicalUnionFind<Ix: PrimInt> {
    parent: Vec<Ix>,
    rank: Vec<usize>,
    /// Map from UF representative to hierarchy node id
    rep_to_node: HashMap<usize, usize>,
}

impl<Ix: PrimInt> HierarchicalUnionFind<Ix> {
    /// Create a new Union-Find with n elements
    pub fn new(n: usize) -> Self {
        let parent = (0..n).map(|i| Ix::from(i).unwrap()).collect();
        let rank = vec![1; n];
        let mut rep_to_node = HashMap::new();
        
        // Initially, each element maps to itself (leaf nodes)
        for i in 0..n {
            rep_to_node.insert(i, i);
        }
        
        HierarchicalUnionFind {
            parent,
            rank,
            rep_to_node,
        }
    }
    
    /// Find the representative of an element with path compression
    pub fn find(&mut self, mut x: Ix) -> Ix {
        let x_idx = x.to_usize().unwrap();
        while self.parent[x_idx] != x {
            let parent_idx = self.parent[x_idx].to_usize().unwrap();
            self.parent[x_idx] = self.parent[parent_idx];
            x = self.parent[x_idx];
        }
        x
    }
    
    /// Union two sets and return the new representative
    pub fn union(&mut self, x: Ix, y: Ix) -> Ix {
        let root_x = self.find(x);
        let root_y = self.find(y);
        
        if root_x == root_y {
            return root_x;
        }
        
        let rx_idx = root_x.to_usize().unwrap();
        let ry_idx = root_y.to_usize().unwrap();
        
        // Union by rank
        if self.rank[rx_idx] < self.rank[ry_idx] {
            self.parent[rx_idx] = root_y;
            root_y
        } else if self.rank[rx_idx] > self.rank[ry_idx] {
            self.parent[ry_idx] = root_x;
            root_x
        } else {
            self.parent[ry_idx] = root_x;
            self.rank[rx_idx] += 1;
            root_x
        }
    }
    
    /// Get the hierarchy node id for a UF representative
    pub fn get_node_id(&self, rep: Ix) -> Option<usize> {
        self.rep_to_node.get(&rep.to_usize().unwrap()).copied()
    }
    
    /// Update the mapping from UF representative to hierarchy node
    pub fn update_node_mapping(&mut self, old_rep: Ix, new_rep: Ix, new_node_id: usize) {
        let old_idx = old_rep.to_usize().unwrap();
        let new_idx = new_rep.to_usize().unwrap();
        
        self.rep_to_node.remove(&old_idx);
        self.rep_to_node.insert(new_idx, new_node_id);
    }
    
    /// Check if two elements are in the same set
    pub fn connected(&mut self, x: Ix, y: Ix) -> bool {
        self.find(x) == self.find(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hierarchy_creation() {
        let hierarchy = ClusterHierarchy::<f32>::new(5);
        
        assert_eq!(hierarchy.num_nodes(), 5);
        assert_eq!(hierarchy.num_points(), 5);
        
        // All initial nodes should be leaves
        for i in 0..5 {
            let node = hierarchy.get_node(i).unwrap();
            assert!(node.is_leaf());
            assert_eq!(node.size, 1);
            assert_eq!(node.points.len(), 1);
            assert_eq!(node.points[0], i);
        }
    }
    
    #[test]
    fn test_hierarchy_merge() {
        let mut hierarchy = ClusterHierarchy::<f32>::new(3);
        
        // Merge nodes 0 and 1 at lambda = 2.0
        let parent_id = hierarchy.merge(0, 1, 2.0);
        
        assert_eq!(parent_id, 3); // Should be the next id
        assert_eq!(hierarchy.num_nodes(), 4);
        
        let parent_node = hierarchy.get_node(parent_id).unwrap();
        assert!(!parent_node.is_leaf());
        assert_eq!(parent_node.size, 2);
        assert_eq!(parent_node.lambda_birth, 2.0);
        assert_eq!(parent_node.children, vec![0, 1]);
        
        // Check that children have been updated
        let child_0 = hierarchy.get_node(0).unwrap();
        assert_eq!(child_0.parent, Some(parent_id));
        assert_eq!(child_0.lambda_death, Some(2.0));
    }
    
    #[test]
    fn test_complete_hierarchy() {
        let mut hierarchy = ClusterHierarchy::<f32>::new(4);
        
        // Build a complete hierarchy
        // Merge 0 and 1 at lambda = 3.0
        let cluster_01 = hierarchy.merge(0, 1, 3.0);
        
        // Merge 2 and 3 at lambda = 2.5
        let cluster_23 = hierarchy.merge(2, 3, 2.5);
        
        // Merge the two clusters at lambda = 1.0
        let root = hierarchy.merge(cluster_01, cluster_23, 1.0);
        
        assert_eq!(hierarchy.root, Some(root));
        assert_eq!(hierarchy.num_nodes(), 7); // 4 leaves + 3 internal
        
        // Check tree structure
        let stats = hierarchy.get_stats();
        assert_eq!(stats.num_leaves, 4);
        assert_eq!(stats.num_merges, 3);
        assert_eq!(stats.tree_height, 2);
        assert_eq!(stats.min_lambda, 1.0);
        assert_eq!(stats.max_lambda, 3.0);
    }
    
    #[test]
    fn test_hierarchical_union_find() {
        let mut uf = HierarchicalUnionFind::<usize>::new(5);
        
        // Initially all elements are their own representative
        for i in 0..5 {
            assert_eq!(uf.find(i), i);
            assert_eq!(uf.get_node_id(i), Some(i));
        }
        
        // Union 0 and 1
        let rep_01 = uf.union(0, 1);
        assert!(uf.connected(0, 1));
        assert!(!uf.connected(0, 2));
        
        // Update node mapping
        uf.update_node_mapping(1, rep_01, 10); // Map to new hierarchy node 10
        assert_eq!(uf.get_node_id(rep_01), Some(10));
    }
    
    #[test]
    fn test_traversals() {
        let mut hierarchy = ClusterHierarchy::<f32>::new(3);
        
        // Build hierarchy: (0,1) merge to 3, then (3,2) merge to 4
        hierarchy.merge(0, 1, 2.0);
        hierarchy.merge(3, 2, 1.0);
        
        // Preorder should visit parent before children
        let preorder = hierarchy.preorder_traversal();
        assert_eq!(preorder, vec![4, 3, 0, 1, 2]);
        
        // Postorder should visit children before parent
        let postorder = hierarchy.postorder_traversal();
        assert_eq!(postorder, vec![0, 1, 3, 2, 4]);
    }
}