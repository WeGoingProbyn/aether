// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! The combinatorial heart of selective refinement — a per-base-cell quadtree
//! forest, independent of any geometry.
//!
//! Each base-mesh cell roots a 4-ary tree (an angular quad-split). The tree's
//! **leaves** are the active cells; a cell that is "refined" becomes an internal
//! node with four leaf children. Refinement is angular only (radial layers are
//! carried by the geometry layer, not here), so a node always has exactly four
//! children.
//!
//! Active cells are enumerated depth-first, base cell by base cell, child index
//! 0→3, which assigns each leaf a dense [`CellId`]. [`RefinementForest::adapt`]
//! applies an (already 2:1-balanced) [`AdaptRequest`] and returns the new forest
//! together with the old→new [`CellRemap`] — the bridge to the field remap and
//! the geometry rebuild. This module is pure bookkeeping: no centroids, no faces,
//! fully testable on its own.

use std::collections::HashMap;

use utility::domain::{CellId, CellRemap, NewCellSource, TopologyEpoch};

use crate::refine::AdaptRequest;

/// One node of a base cell's refinement quadtree.
#[derive(Clone, Debug, PartialEq, Eq)]
enum QuadNode {
  /// An active cell (no further subdivision).
  Leaf,
  /// A refined cell: four angular children, in child-index order.
  Split(Box<[QuadNode; 4]>),
}

impl QuadNode {
  /// Collect this subtree's leaf paths (relative to this node) into `out`,
  /// depth-first in child order, appending to `prefix`.
  fn collect_leaves(&self, prefix: &mut Vec<u8>, out: &mut Vec<Vec<u8>>) {
    match self {
      QuadNode::Leaf => out.push(prefix.clone()),
      QuadNode::Split(children) => {
        for (k, child) in children.iter().enumerate() {
          prefix.push(k as u8);
          child.collect_leaves(prefix, out);
          prefix.pop();
        }
      }
    }
  }

  /// Refine the leaf at `path` into four leaves. No-op if `path` is not a leaf.
  fn refine_at(&mut self, path: &[u8]) {
    match path.split_first() {
      None => {
        if matches!(self, QuadNode::Leaf) {
          *self = QuadNode::Split(Box::new([
            QuadNode::Leaf,
            QuadNode::Leaf,
            QuadNode::Leaf,
            QuadNode::Leaf,
          ]));
        }
      }
      Some((&k, rest)) => {
        if let QuadNode::Split(children) = self {
          children[k as usize].refine_at(rest);
        }
      }
    }
  }

  /// Collapse the node at `path` back to a single leaf (coarsening). No-op if the
  /// node is already a leaf.
  fn coarsen_at(&mut self, path: &[u8]) {
    match path.split_first() {
      None => *self = QuadNode::Leaf,
      Some((&k, rest)) => {
        if let QuadNode::Split(children) = self {
          children[k as usize].coarsen_at(rest);
        }
      }
    }
  }
}

/// A forest of per-base-cell refinement quadtrees. Leaves are the active cells.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RefinementForest {
  roots: Vec<QuadNode>,
}

/// A leaf's location: which base cell, and the child-index path from that base
/// cell's root (empty path ⇒ the base cell itself is a leaf, i.e. unrefined).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LeafLocation {
  pub base_cell: CellId,
  pub path: Vec<u8>,
}

impl LeafLocation {
  /// The refinement level of this leaf (0 = unrefined base cell).
  pub fn level(&self) -> u32 {
    self.path.len() as u32
  }
}

impl RefinementForest {
  /// A forest over `base_cell_count` cells, all unrefined (level 0).
  pub fn new(base_cell_count: usize) -> Self {
    Self {
      roots: vec![QuadNode::Leaf; base_cell_count],
    }
  }

  pub fn base_cell_count(&self) -> usize {
    self.roots.len()
  }

  /// Active cells in dense [`CellId`] order: base cell by base cell, leaves
  /// depth-first in child order. The index into this list is the cell's id.
  pub fn leaves(&self) -> Vec<LeafLocation> {
    let mut out = Vec::new();
    for (base, root) in self.roots.iter().enumerate() {
      let mut paths = Vec::new();
      let mut prefix = Vec::new();
      root.collect_leaves(&mut prefix, &mut paths);
      for path in paths {
        out.push(LeafLocation {
          base_cell: CellId::from(base),
          path,
        });
      }
    }
    out
  }

  pub fn leaf_count(&self) -> usize {
    self.roots.iter().map(QuadNode::leaf_count).sum()
  }

  /// Apply a 2:1-balanced [`AdaptRequest`] (cells are *current* leaf ids) and
  /// return the resulting forest plus the old→new [`CellRemap`].
  ///
  /// Refining a leaf replaces it with four children; coarsening collapses a
  /// refined cell's four leaf children back to one. A refined old leaf has no
  /// single new image (it *died*); each new child is a
  /// [`Child`](NewCellSource::Child) of it. A coarsened cell is a
  /// [`Merge`](NewCellSource::Merge) of its old children. Untouched leaves are
  /// [`Survivor`](NewCellSource::Survivor)s.
  pub fn adapt(
    &self,
    request: &AdaptRequest,
    old_epoch: TopologyEpoch,
    new_epoch: TopologyEpoch,
  ) -> (RefinementForest, CellRemap) {
    let old_leaves = self.leaves();
    let old_count = old_leaves.len();
    // (base, path) → old CellId, for correlating after the rebuild.
    let old_index: HashMap<(usize, Vec<u8>), usize> = old_leaves
      .iter()
      .enumerate()
      .map(|(id, loc)| ((loc.base_cell.index(), loc.path.clone()), id))
      .collect();

    // Build the new forest. Coarsen first (collapse parents), then refine, both
    // addressed by the parent/leaf path in the OLD enumeration. Coarsen targets
    // are deduplicated by the parent node they collapse.
    let mut new_forest = self.clone();
    let mut collapsed_parents: Vec<(usize, Vec<u8>)> = Vec::new();
    for &cell in &request.coarsen {
      let loc = &old_leaves[cell.index()];
      if loc.path.is_empty() {
        continue; // a base-cell-level leaf cannot coarsen further
      }
      let parent_path = loc.path[..loc.path.len() - 1].to_vec();
      let key = (loc.base_cell.index(), parent_path.clone());
      if !collapsed_parents.contains(&key) {
        collapsed_parents.push(key);
        new_forest.roots[loc.base_cell.index()].coarsen_at(&parent_path);
      }
    }
    for &cell in &request.refine {
      let loc = &old_leaves[cell.index()];
      new_forest.roots[loc.base_cell.index()].refine_at(&loc.path);
    }

    // Correlate the new leaves against the old enumeration to build the remap.
    let new_leaves = new_forest.leaves();
    let new_count = new_leaves.len();
    let mut new_sources = Vec::with_capacity(new_count);
    let mut old_to_new = vec![None; old_count];

    for (new_id, loc) in new_leaves.iter().enumerate() {
      let key = (loc.base_cell.index(), loc.path.clone());
      if let Some(&old_id) = old_index.get(&key) {
        // Same (base, path) existed before — a survivor.
        new_sources.push(NewCellSource::Survivor(CellId::from(old_id)));
        old_to_new[old_id] = Some(CellId::from(new_id));
      } else if !loc.path.is_empty()
        && old_index.contains_key(&(
          loc.base_cell.index(),
          loc.path[..loc.path.len() - 1].to_vec(),
        ))
      {
        // Parent was an old leaf that got refined — this is a new child.
        let parent_old = old_index[&(
          loc.base_cell.index(),
          loc.path[..loc.path.len() - 1].to_vec(),
        )];
        new_sources.push(NewCellSource::Child {
          parent: CellId::from(parent_old),
        });
      } else {
        // This (base, path) was an internal node that collapsed to a leaf — a
        // merge of all old leaves that lived beneath it.
        let children: Vec<CellId> = old_index
          .iter()
          .filter(|((b, p), _)| {
            *b == loc.base_cell.index()
              && p.len() > loc.path.len()
              && p[..loc.path.len()] == loc.path[..]
          })
          .map(|(_, &id)| CellId::from(id))
          .collect();
        new_sources.push(NewCellSource::Merge { children });
      }
    }

    let remap = CellRemap::new(old_epoch, new_epoch, old_to_new, new_sources);
    (new_forest, remap)
  }
}

impl QuadNode {
  fn leaf_count(&self) -> usize {
    match self {
      QuadNode::Leaf => 1,
      QuadNode::Split(children) => {
        children.iter().map(QuadNode::leaf_count).sum()
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn refine(cells: &[usize]) -> AdaptRequest {
    AdaptRequest {
      refine: cells.iter().copied().map(CellId::from).collect(),
      coarsen: vec![],
    }
  }

  #[test]
  fn fresh_forest_is_all_level_zero_leaves() {
    let forest = RefinementForest::new(3);
    let leaves = forest.leaves();
    assert_eq!(leaves.len(), 3);
    assert!(leaves.iter().all(|l| l.level() == 0));
    // Leaf id == base cell id when unrefined.
    for (i, l) in leaves.iter().enumerate() {
      assert_eq!(l.base_cell, CellId::from(i));
    }
  }

  #[test]
  fn refining_one_leaf_adds_three_cells_with_child_sources() {
    let forest = RefinementForest::new(2);
    let (new, remap) = forest.adapt(
      &refine(&[0]),
      TopologyEpoch::ZERO,
      TopologyEpoch::ZERO.next(),
    );
    // Base cell 0 → 4 children, base cell 1 untouched ⇒ 5 leaves.
    assert_eq!(new.leaf_count(), 5);
    assert_eq!(remap.new_count(), 5);

    // New cells 0..4 are children of old cell 0; new cell 4 is old cell 1.
    for child in 0..4 {
      assert_eq!(
        remap.source_of(CellId::from(child)),
        &NewCellSource::Child {
          parent: CellId::from(0)
        }
      );
    }
    assert_eq!(
      remap.source_of(CellId::from(4)),
      &NewCellSource::Survivor(CellId::from(1))
    );
    // Old cell 0 died (refined away); old cell 1 survived to new id 4.
    assert_eq!(remap.image_of(CellId::from(0)), None);
    assert_eq!(remap.image_of(CellId::from(1)), Some(CellId::from(4)));
    // The new children's level is 1.
    assert_eq!(new.leaves()[0].level(), 1);
  }

  #[test]
  fn coarsening_four_siblings_merges_them_back() {
    // Refine cell 0 (→ leaves 0..4), then coarsen all four children back.
    let forest = RefinementForest::new(2);
    let (refined, _) = forest.adapt(
      &refine(&[0]),
      TopologyEpoch::ZERO,
      TopologyEpoch::ZERO.next(),
    );
    let coarsen = AdaptRequest {
      refine: vec![],
      coarsen: (0..4).map(CellId::from).collect(),
    };
    let (merged, remap) = refined.adapt(
      &coarsen,
      TopologyEpoch::ZERO.next(),
      TopologyEpoch::ZERO.next().next(),
    );
    assert_eq!(merged.leaf_count(), 2);
    // New cell 0 is the merge of old children 0..4.
    match remap.source_of(CellId::from(0)) {
      NewCellSource::Merge { children } => {
        let mut ids: Vec<usize> = children.iter().map(|c| c.index()).collect();
        ids.sort();
        assert_eq!(ids, vec![0, 1, 2, 3]);
      }
      other => panic!("expected Merge, got {other:?}"),
    }
    // Old surviving cell 4 (the other base cell) maps to new cell 1.
    assert_eq!(remap.image_of(CellId::from(4)), Some(CellId::from(1)));
    assert_eq!(merged.leaves()[1].level(), 0);
  }

  #[test]
  fn two_level_refinement_tracks_levels() {
    let forest = RefinementForest::new(1);
    let e0 = TopologyEpoch::ZERO;
    let (l1, _) = forest.adapt(&refine(&[0]), e0, e0.next());
    assert_eq!(l1.leaf_count(), 4);
    // Refine the first child again.
    let (l2, _) = l1.adapt(&refine(&[0]), e0.next(), e0.next().next());
    assert_eq!(l2.leaf_count(), 7); // one child → 4, plus 3 siblings
    let leaves = l2.leaves();
    // The first four leaves are the level-2 grandchildren.
    assert!(leaves[..4].iter().all(|l| l.level() == 2));
    assert!(leaves[4..].iter().all(|l| l.level() == 1));
  }
}
