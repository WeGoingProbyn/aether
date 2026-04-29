use crate::{
  debugger::IcsError,
  maths::{matrix::Matrix, transformation::Transformation},
  ICS_ERROR,
};
use std::{
  borrow::Borrow,
  fmt,
  hash::{Hash, Hasher},
  ops::Deref,
};

/// Strongly typed identifier for hierarchy nodes to avoid stringly-typed maps.
#[derive(Clone, Eq, PartialEq, Hash, PartialOrd, Ord, Debug)]
pub struct NodeId(String);

impl NodeId {
  pub fn new<S: Into<String>>(name: S) -> Self {
    NodeId(name.into())
  }

  pub fn as_str(&self) -> &str {
    &self.0
  }
}

impl From<&str> for NodeId {
  fn from(value: &str) -> Self {
    NodeId::new(value)
  }
}

impl From<String> for NodeId {
  fn from(value: String) -> Self {
    NodeId::new(value)
  }
}

impl Borrow<str> for NodeId {
  fn borrow(&self) -> &str {
    &self.0
  }
}

impl AsRef<str> for NodeId {
  fn as_ref(&self) -> &str {
    &self.0
  }
}

impl Deref for NodeId {
  type Target = str;

  fn deref(&self) -> &Self::Target {
    &self.0
  }
}

impl fmt::Display for NodeId {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{}", self.0)
  }
}

/// Represents a node in a hierarchical structure for scene and object hierarchy management.
#[derive(Debug, Clone)]
pub struct HierarchyNode {
  /// The name of the node.
  pub name: NodeId,
  /// A list of child nodes.
  pub children: Vec<HierarchyNode>,
  /// Optional index referring to an IndicesPart.
  pub indicespart_index: Option<usize>,
  /// Optional transformation relative to the parent node.
  pub relative_transform: Option<Transformation>,
  /// Optional matrix representing the relative transformation.
  pub relative_matrix: Option<Matrix<4, 4, f32>>,
  /// The final propagated transformation matrix.
  pub propagated_transform: Matrix<4, 4, f32>,
}

impl Hash for HierarchyNode {
  fn hash<H: Hasher>(&self, state: &mut H) {
    self.name.hash(state);
    self.indicespart_index.hash(state);
  }
}

impl HierarchyNode {
  /// Creates a new `HierarchyNode` with the given name and optional indices part index.
  ///
  /// # Arguments
  ///
  /// * `name` - The name of the node.
  /// * `indicespart_index` - An optional index referring to an IndicesPart.
  ///
  /// # Examples
  ///
  /// ```
  /// let node = HierarchyNode::new("root", None);
  /// ```
  pub fn new(name: &str, indicespart_index: Option<usize>) -> Self {
    HierarchyNode {
      name: NodeId::new(name),
      children: Vec::new(),
      indicespart_index,
      relative_transform: None,
      relative_matrix: None,
      propagated_transform: Matrix::<4, 4, f32>::identity(),
    }
  }

  /// Returns a reference to the node's relative transformation.
  pub fn transform(&self) -> &Option<Transformation> {
    &self.relative_transform
  }

  /// Returns a mutable reference to the node's relative transformation.
  pub fn transform_mut(&mut self) -> &mut Option<Transformation> {
    &mut self.relative_transform
  }

  /// Returns the name of the node.
  pub fn name(&self) -> &NodeId {
    &self.name
  }

  /// Returns a reference to the child nodes.
  pub fn children(&self) -> &Vec<HierarchyNode> {
    &self.children
  }

  /// Adds a child node to the current node.
  ///
  /// # Arguments
  ///
  /// * `child` - The child node to be added.
  ///
  /// # Examples
  ///
  /// ```
  /// let mut root = HierarchyNode::new("root", None);
  /// let child = HierarchyNode::new("child", None);
  /// root.add_child(child);
  /// ```
  pub fn add_child(&mut self, child: HierarchyNode) {
    self.children.push(child);
  }

  /// Recursively finds a mutable reference to the node with the given name.
  ///
  /// # Arguments
  ///
  /// * `name` - The name of the node to find.
  ///
  /// # Returns
  ///
  /// `Some(&mut HierarchyNode)` if found, otherwise `None`.
  pub fn find_node_mut(&mut self, name: &str) -> Option<&mut HierarchyNode> {
    if self.name.as_str() == name {
      return Some(self);
    }
    for child in &mut self.children {
      if let Some(found) = child.find_node_mut(name) {
        return Some(found);
      }
    }
    None
  }

  /// Recursively finds a reference to the node with the given name.
  ///
  /// # Arguments
  ///
  /// * `name` - The name of the node to find.
  ///
  /// # Returns
  ///
  /// `Some(&HierarchyNode)` if found, otherwise `None`.
  pub fn find_node(&self, name: &str) -> Option<&HierarchyNode> {
    if self.name.as_str() == name {
      return Some(self);
    }
    for child in &self.children {
      if let Some(found) = child.find_node(name) {
        return Some(found);
      }
    }
    None
  }

  /// Recursively adds a child node to the node with the given name.
  ///
  /// # Arguments
  ///
  /// * `parent_name` - The name of the parent node.
  /// * `node` - The child node to add.
  ///
  /// # Returns
  ///
  /// `Ok(())` if the child was successfully added, otherwise an `IcsError`.
  pub fn add_child_node(&mut self, parent_name: &str, node: HierarchyNode) -> Result<(), IcsError> {
    if self.name.as_str() == parent_name {
      self.children.push(node.clone());
      return Ok(());
    }
    for child in &mut self.children {
      if child.add_child_node(parent_name, node.clone()).is_ok() {
        return Ok(());
      }
    }
    Err(ICS_ERROR!(
      why: format!("HierarchyNode: Parent node '{}' not found", parent_name),
      fix: "Ensure the parent node exists before adding a child"
    ))
  }

  /// Recursively counts the number of nodes in the subtree, including itself.
  ///
  /// # Returns
  ///
  /// The total number of nodes in the subtree.
  ///
  /// # Examples
  ///
  /// ```
  /// let mut root = HierarchyNode::new("root", None);
  /// root.add_child(HierarchyNode::new("child", None));
  /// assert_eq!(root.count_nodes(), 2);
  /// ```
  pub fn count_nodes(&self) -> usize {
    1 + self
      .children
      .iter()
      .map(|child| child.count_nodes())
      .sum::<usize>()
  }

  /// Updates the propagated transformations of this node and its children.
  ///
  /// # Arguments
  ///
  /// * `parent_transform` - The transformation matrix of the parent node.
  ///
  /// # Examples
  ///
  /// ```
  /// let mut root = HierarchyNode::new("root", None);
  /// let transform = Matrix::<4, 4, f32>::identity();
  /// root.propagate_transform(transform);
  /// ```
  pub fn propagate_transform(&mut self, parent_transform: Matrix<4, 4, f32>) {
    let current_transform = if let Some(transform) = &self.relative_transform {
      parent_transform * transform.transform()
    } else if let Some(matrix) = self.relative_matrix {
      parent_transform * matrix
    } else {
      parent_transform
    };

    self.propagated_transform = current_transform;

    for child in self.children.iter_mut() {
      child.propagate_transform(self.propagated_transform);
    }
  }
}

/// Represents a hierarchical structure of nodes.
#[derive(Debug, Clone, Hash)]
pub struct Hierarchy {
  /// A list of node names in the hierarchy.
  pub node_names: Vec<NodeId>,
  /// A list of root nodes in the hierarchy.
  pub root_nodes: Vec<HierarchyNode>,
}

impl Hierarchy {
  /// Creates a new, empty `Hierarchy`.
  ///
  /// # Example
  /// ```
  /// let hierarchy = Hierarchy::new();
  /// assert!(hierarchy.root_nodes.is_empty());
  /// ```
  pub fn new() -> Self {
    Hierarchy {
      node_names: Vec::new(),
      root_nodes: Vec::new(),
    }
  }

  /// Adds a root node to the hierarchy.
  ///
  /// # Arguments
  /// * `node` - The root node to be added.
  pub fn add_root_node(&mut self, node: HierarchyNode) {
    if node.indicespart_index.is_some() {
      self.node_names.push(node.name.clone());
    }
    self.root_nodes.push(node);
  }

  /// Adds a child node under a specified parent node.
  ///
  /// # Arguments
  /// * `parent_name` - The name of the parent node.
  /// * `node` - The child node to add.
  ///
  /// # Errors
  /// Returns an error if the parent node is not found.
  pub fn add_child_node(&mut self, parent_name: &str, node: HierarchyNode) -> Result<(), IcsError> {
    for root in &mut self.root_nodes {
      if let Some(_) = root.find_node_mut(parent_name) {
        self.node_names.push(node.name.clone());
        root.add_child_node(parent_name, node)?;
        return Ok(());
      }
    }
    Err(ICS_ERROR!(
      why: format!("Hierarchy: Parent node '{}' not found", parent_name),
      fix: "Ensure the parent node exists before adding a child"
    ))
  }

  /// Recursively searches for a node by name and applies a closure if found.
  ///
  /// # Arguments
  /// * `name` - The name of the node to find.
  /// * `f` - A closure to apply if the node is found.
  ///
  /// # Returns
  /// `true` if the node was found, `false` otherwise.
  pub fn find_node_mut<F>(&mut self, name: &str, f: &mut F) -> bool
  where
    F: FnMut(&mut HierarchyNode),
  {
    for root in &mut self.root_nodes {
      if let Some(node) = root.find_node_mut(name) {
        f(node);
        return true;
      }
    }
    false
  }

  /// Recursively searches for a node by name and applies a closure if found.
  ///
  /// # Arguments
  /// * `name` - The name of the node to find.
  /// * `f` - A closure to apply if the node is found.
  ///
  /// # Returns
  /// `true` if the node was found, `false` otherwise.
  pub fn find_node<F>(&self, name: &str, f: &F) -> bool
  where
    F: Fn(&HierarchyNode),
  {
    for root in &self.root_nodes {
      if let Some(node) = root.find_node(name) {
        f(node);
        return true;
      }
    }
    false
  }

  /// Counts the total number of nodes in the hierarchy.
  pub fn count(&self) -> usize {
    self.root_nodes.iter().map(|node| node.count_nodes()).sum()
  }

  /// Returns an iterator over all `HierarchyNode`s in the hierarchy (depth-first traversal).
  pub fn iter(&self) -> HierarchyIterator<'_> {
    HierarchyIterator::new(&self.root_nodes)
  }

  /// Returns a reference to the list of node names.
  pub fn node_names(&self) -> &Vec<NodeId> {
    &self.node_names
  }
}

impl fmt::Display for HierarchyNode {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    writeln!(f, "HierarchyNode: {}", self.name)?;
    writeln!(f, "  Indices Part: {:?}", self.indicespart_index)?;
    writeln!(f, "  Transform: {:?}", self.transform())?;
    writeln!(f, "  Propogated: {}", self.propagated_transform)?;
    for child in &self.children {
      write!(f, "  ")?;
      child.fmt(f)?;
    }
    Ok(())
  }
}

impl fmt::Display for Hierarchy {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    writeln!(f, "Hierarchy:")?;
    for node in &self.root_nodes {
      write!(f, "  ")?;
      node.fmt(f)?;
    }
    Ok(())
  }
}

pub struct HierarchyIterator<'a> {
  stack: Vec<&'a HierarchyNode>,
}

impl<'a> HierarchyIterator<'a> {
  /// Initializes a new HierarchyIterator with the given root nodes
  fn new(root_nodes: &'a Vec<HierarchyNode>) -> Self {
    HierarchyIterator {
      stack: root_nodes.iter().collect(),
    }
  }
}

impl<'a> Iterator for HierarchyIterator<'a> {
  type Item = &'a HierarchyNode;

  fn next(&mut self) -> Option<Self::Item> {
    // Pop the last node from the stack
    if let Some(node) = self.stack.pop() {
      // Push the children onto the stack in reverse order to maintain left-to-right traversal
      for child in node.children.iter().rev() {
        self.stack.push(child);
      }
      Some(node)
    } else {
      None
    }
  }
}
