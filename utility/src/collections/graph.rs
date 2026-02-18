use std::collections::{HashMap, HashSet, VecDeque};

use crate::error::{AetherError, AetherResult, ErrorDomain};

// Define a Node in the graph
#[derive(Debug, Clone)]
pub struct GraphNode<T> {
  pub id: usize,
  pub data: T,
}

// Define an Edge in the graph
#[derive(Debug, Clone)]
pub struct GraphEdge {
  pub source: usize,
  pub target: usize,
}

// Define the Graph structure
#[derive(Debug, Clone)]
pub struct Graph<T> {
  nodes: HashMap<usize, GraphNode<T>>,
  edges: HashMap<usize, Vec<GraphEdge>>,
  next_node_id: usize,
}

impl<T> Default for Graph<T> {
  fn default() -> Self {
    Graph {
      nodes: HashMap::new(),
      edges: HashMap::new(),
      next_node_id: 0usize,
    }
  }
}

impl<T> Graph<T> {
  /// Creates a new, empty graph
  pub fn new() -> Self {
    Graph::default()
  }

  pub fn next_node_id(&self) -> usize {
    self.next_node_id
  }

  /// Adds a new node to the graph and returns its unique ID
  pub fn add_node(&mut self, data: T) -> usize {
    let node_id = self.next_node_id;
    self.next_node_id += 1;

    let node = GraphNode { id: node_id, data };
    self.nodes.insert(node_id, node);
    self.edges.entry(node_id).or_default();

    node_id
  }

  /// Removes a node and all associated edges from the graph
  pub fn remove_node(&mut self, node_id: usize) -> Option<GraphNode<T>> {
    // Remove the node
    let removed_node = self.nodes.remove(&node_id);

    // Remove all outgoing edges
    self.edges.remove(&node_id);

    // Remove all incoming edges
    for edge_list in self.edges.values_mut() {
      edge_list.retain(|edge| edge.target != node_id);
    }

    removed_node
  }

  /// Adds a directed edge from `source` to `target` with associated data
  pub fn add_edge(&mut self, source: usize, target: usize) -> AetherResult<()> {
    if !self.nodes.contains_key(&source) {
      return Err(
        AetherError::new(ErrorKind::InvalidNode)
          .context(format!("source node {} does not exist", source)),
      );
    }
    if !self.nodes.contains_key(&target) {
      return Err(
        AetherError::new(ErrorKind::InvalidNode)
          .context(format!("target node {} does not exits", target)),
      );
    }

    let edge = GraphEdge { source, target };
    self.edges.entry(source).or_default().push(edge);

    Ok(())
  }

  /// Removes a directed edge from `source` to `target`
  pub fn remove_edge(&mut self, source: usize, target: usize) -> bool {
    if let Some(edge_list) = self.edges.get_mut(&source) {
      let original_len = edge_list.len();
      edge_list.retain(|edge| edge.target != target);
      return edge_list.len() < original_len;
    }
    false
  }

  /// Returns a reference to a node by ID
  pub fn get_node(&self, node_id: usize) -> Option<&GraphNode<T>> {
    self.nodes.get(&node_id)
  }

  /// Returns a mutable reference to a node by ID
  pub fn get_node_mut(&mut self, node_id: usize) -> Option<&mut GraphNode<T>> {
    self.nodes.get_mut(&node_id)
  }

  /// Returns all nodes in the graph
  pub fn nodes(&self) -> impl Iterator<Item = &GraphNode<T>> {
    self.nodes.values()
  }

  /// Returns all nodes in the graph
  pub fn nodes_mut(&mut self) -> impl Iterator<Item = &mut GraphNode<T>> {
    self.nodes.values_mut()
  }

  /// Returns all nodes in the graph
  pub fn into_nodes(self) -> impl Iterator<Item = GraphNode<T>> {
    self.nodes.into_values()
  }

  /// Returns all edges in the graph
  pub fn edges(&self) -> impl Iterator<Item = &GraphEdge> {
    self.edges.values().flat_map(|edge_list| edge_list.iter())
  }

  /// Returns outgoing edges from a given node
  pub fn outgoing_edges(&self, node_id: usize) -> Option<&Vec<GraphEdge>> {
    self.edges.get(&node_id)
  }

  /// Returns incoming edges to a given node
  pub fn incoming_edges(&self, node_id: usize) -> Vec<&GraphEdge> {
    self
      .edges
      .values()
      .flat_map(|edge_list| edge_list.iter())
      .filter(|edge| edge.target == node_id)
      .collect()
  }

  /// Performs Depth-First Search (DFS) starting from `start_id`
  pub fn dfs<F>(&self, start_id: usize, mut visit: F)
  where
    F: FnMut(&GraphNode<T>),
  {
    let mut visited = HashSet::new();
    self.dfs_recurse(start_id, &mut visited, &mut visit);
  }

  fn dfs_recurse<F>(
    &self,
    node_id: usize,
    visited: &mut HashSet<usize>,
    visit: &mut F,
  ) where
    F: FnMut(&GraphNode<T>),
  {
    if visited.contains(&node_id) {
      return;
    }
    visited.insert(node_id);

    if let Some(node) = self.nodes.get(&node_id) {
      visit(node);
    }

    if let Some(edges) = self.edges.get(&node_id) {
      for edge in edges {
        self.dfs_recurse(edge.target, visited, visit);
      }
    }
  }

  /// Performs Breadth-First Search (BFS) starting from `start_id`
  pub fn bfs<F>(&self, start_id: usize, mut visit: F)
  where
    F: FnMut(&GraphNode<T>),
  {
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();

    visited.insert(start_id);
    queue.push_back(start_id);

    while let Some(current_id) = queue.pop_front() {
      if let Some(node) = self.nodes.get(&current_id) {
        visit(node);
      }

      if let Some(edges) = self.edges.get(&current_id) {
        for edge in edges {
          if !visited.contains(&edge.target) {
            visited.insert(edge.target);
            queue.push_back(edge.target);
          }
        }
      }
    }
  }

  /// Performs Topological Sort on the graph, Khan's algorithm
  /// Returns `Ok` with a sorted list if the graph is a DAG,
  /// or `Err` with a message if a cycle is detected
  pub fn topological_sort(&self) -> AetherResult<Vec<usize>> {
    // Compute in-degree of each node
    let mut in_degree: HashMap<usize, usize> =
      self.nodes.keys().map(|&id| (id, 0)).collect();

    for edges in self.edges.values() {
      for edge in edges {
        *in_degree.entry(edge.target).or_insert(0) += 1;
      }
    }

    // Collect nodes with in-degree 0
    // Use an ordered set to keep processing deterministic and roughly match insertion order.
    let mut queue: std::collections::BTreeSet<usize> = in_degree
      .iter()
      .filter_map(|(&id, &deg)| if deg == 0 { Some(id) } else { None })
      .collect();

    let mut sorted = Vec::new();

    while let Some(&node_id) = queue.iter().next() {
      queue.remove(&node_id);
      sorted.push(node_id);

      if let Some(edges) = self.edges.get(&node_id) {
        for edge in edges {
          if let Some(degree) = in_degree.get_mut(&edge.target) {
            *degree -= 1;
            if *degree == 0 {
              // No other dependents need to be resolved for this target
              queue.insert(edge.target);
            }
          }
        }
      }
    }

    if sorted.len() == self.nodes.len() {
      Ok(sorted)
    } else {
      Err(
        AetherError::new(ErrorKind::GraphCycle)
          .context("graph has at least one cycle"),
      )
    }
  }
}

impl<T: Clone> Graph<T> {
  pub fn extend(&mut self, other: &Graph<T>) -> AetherResult<()> {
    // Determine the offset based on the current next_node_id
    let offset = self.next_node_id;

    // Create a mapping from the other graph's node IDs to the new node IDs
    let mut id_map = HashMap::new();

    // Add all nodes from the other graph with new IDs
    for (id, node) in &other.nodes {
      let new_id = id + offset;
      if self.nodes.contains_key(&new_id) {
        return Err(
          AetherError::new(ErrorKind::NodeCollision)
            .context(format!("node collision detected for {}", new_id)),
        );
      }
      id_map.insert(*id, new_id);
      let new_node = GraphNode {
        id: new_id,
        data: node.data.clone(),
      };
      self.nodes.insert(new_id, new_node);
      self.edges.entry(new_id).or_default();
    }

    // Add all edges from the other graph with updated source and target IDs
    for (source, edge_list) in &other.edges {
      let new_source = *id_map.get(source).ok_or_else(|| {
        AetherError::new(ErrorKind::InvalidNode)
          .context(format!("source node {} not found in id map", source))
      })?;
      for edge in edge_list {
        let new_target = *id_map.get(&edge.target).ok_or_else(|| {
          AetherError::new(ErrorKind::InvalidNode)
            .context(format!("target node {} not found in id map", edge.target))
        })?;
        let new_edge = GraphEdge {
          source: new_source,
          target: new_target,
        };
        self.edges.entry(new_source).or_default().push(new_edge);
      }
    }

    // Update the next_node_id to reflect the addition of new nodes
    self.next_node_id += other.next_node_id;

    Ok(())
  }
}

pub enum ErrorKind {
  InvalidNode,
  GraphCycle,
  NodeCollision,
}

impl ErrorDomain for ErrorKind {
  fn domain(&self) -> &str {
    "graph"
  }
}

impl std::fmt::Display for ErrorKind {
  fn fmt(
    &self,
    f: &mut std::fmt::Formatter<'_>,
  ) -> Result<(), std::fmt::Error> {
    let string = match self {
      ErrorKind::InvalidNode => "requested a non existant node",
      ErrorKind::GraphCycle => {
        "a node within the graph depends on a node which also depends on the same node"
      }
      ErrorKind::NodeCollision => {
        "a collision was found between 2 distinct nodes and their ids"
      }
    };

    write!(f, "{}", string)?;
    Ok(())
  }
}
