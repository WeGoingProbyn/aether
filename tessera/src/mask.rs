// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Per-mesh cell-activity masks.
//!
//! A [`CellMask`] marks which cells of a mesh are *active* — really part of that
//! mesh's domain — versus *inactive* ("this cell isn't really here"). It is the
//! general primitive behind land–sea masking: an ocean shell is a full globe, but
//! the columns under land are not really ocean, so the ocean solver should skip
//! them. The mask is structural geometry-adjacent data, so it lives in `tessera`
//! alongside meshes and couplers (see [`crate::world_mesh::Tessera`]).
//!
//! **CellId space.** A mask is indexed by **global [`CellId`]** — the same space
//! that global `pleroma` fields (`SoaField`) use, *not* any partition-local space.
//! Meshes have a deterministic global cell ordering, so a mask built for a mesh
//! lines up with that mesh's fields on every run, and partitioning (the
//! local↔global remap) never touches it.

use utility::domain::CellId;
use utility::error::ErrorDomain;

/// Errors from building a [`CellMask`] against a [`crate::world_mesh::Tessera`].
#[derive(Debug)]
pub enum MaskError {
  /// `build_geographic_cell_mask` was asked to mask a mesh that is not registered.
  MeshNotRegistered,
}

impl ErrorDomain for MaskError {
  fn domain(&self) -> &str {
    "tessera mask"
  }
}

impl std::fmt::Display for MaskError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      MaskError::MeshNotRegistered => {
        write!(f, "cannot build a cell mask for an unregistered mesh")
      }
    }
  }
}

/// An active/inactive flag per cell of a mesh, indexed by global [`CellId`].
///
/// `is_active` is an O(1) index; the active/inactive counts are tallied once at
/// construction and cached, so they are O(1) too (never recomputed on a hot path).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CellMask {
  active: Vec<bool>,
  active_count: usize,
}

impl CellMask {
  /// Build a mask of `n` cells, marking `CellId(i)` active iff `active(CellId(i))`.
  pub fn from_fn(n: usize, active: impl Fn(CellId) -> bool) -> Self {
    let active: Vec<bool> = (0..n).map(|i| active(CellId::from(i))).collect();
    let active_count = active.iter().filter(|&&a| a).count();
    Self {
      active,
      active_count,
    }
  }

  /// A mask of `n` cells with every cell active (a no-op mask).
  pub fn all_active(n: usize) -> Self {
    Self {
      active: vec![true; n],
      active_count: n,
    }
  }

  /// Whether `cell` is active. Cells outside the mask (`>= len`) are reported
  /// inactive rather than panicking, so a length mismatch fails safe.
  pub fn is_active(&self, cell: CellId) -> bool {
    self.active.get(cell.index()).copied().unwrap_or(false)
  }

  /// Number of cells the mask covers.
  pub fn len(&self) -> usize {
    self.active.len()
  }

  pub fn is_empty(&self) -> bool {
    self.active.is_empty()
  }

  /// Active-cell count (cached; O(1)).
  pub fn active_count(&self) -> usize {
    self.active_count
  }

  /// Inactive-cell count (cached; O(1)).
  pub fn inactive_count(&self) -> usize {
    self.active.len() - self.active_count
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn from_fn_marks_and_counts_cells() {
    // Even cells active, odd inactive.
    let mask = CellMask::from_fn(10, |c| c.index() % 2 == 0);
    assert_eq!(mask.len(), 10);
    assert_eq!(mask.active_count(), 5);
    assert_eq!(mask.inactive_count(), 5);
    assert!(mask.is_active(CellId::from(0)));
    assert!(!mask.is_active(CellId::from(1)));
  }

  #[test]
  fn all_active_is_a_no_op_mask() {
    let mask = CellMask::all_active(4);
    assert_eq!(mask.active_count(), 4);
    assert_eq!(mask.inactive_count(), 0);
    assert!((0..4).all(|i| mask.is_active(CellId::from(i))));
  }

  #[test]
  fn out_of_range_cell_is_inactive_not_a_panic() {
    let mask = CellMask::all_active(2);
    assert!(!mask.is_active(CellId::from(5)));
  }
}
