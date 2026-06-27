// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Mesh-agnostic adaptive-refinement contract — the seam every AMR consumer
//! plugs into.
//!
//! Refinement is split across three responsibilities so the mesh-agnostic logic
//! never leaks into a backend:
//!
//! - The **driver** (the adapt barrier in the runtime) decides *what* to adapt:
//!   a refinement criterion produces a *desired* [`RefineFlags`].
//! - The **balancer** ([`balance_2to1`]) — mesh-agnostic, working only on
//!   [`Topology`] + a cell-level function — turns the desired flags into a
//!   2:1-*balanced* [`AdaptRequest`]. Balance is enforced *here, before any
//!   backend runs*, because conservative hanging-node flux assumes neighbouring
//!   cells differ by at most one refinement level.
//! - The **backend** ([`RefinableMesh`]) owns *how*: only it knows its child
//!   layout, so it rebuilds the (immutable) mesh and computes the [`CellRemap`].
//!   It is also the safety net — it validates the request and returns
//!   [`RefineError::UnbalancedRequest`] rather than silently producing a
//!   greater-than-one level jump.
//!
//! The identity types ([`CellRemap`], [`NewCellSource`], [`TopologyEpoch`]) live
//! in `utility::domain` as shared vocabulary and are re-exported here for
//! convenience.

use std::sync::Arc;

use utility::domain::CellId;
use utility::error::{AetherResult, ErrorDomain};

pub use utility::domain::{CellRemap, NewCellSource, TopologyEpoch};

use crate::mesh::Mesh;
use crate::topology::Topology;

/// Errors a [`RefinableMesh`] backend may raise while validating / realising an
/// [`AdaptRequest`].
#[derive(Debug, PartialEq, Eq)]
pub enum RefineError {
  /// The request is not 2:1-balanced (a refinement-level jump greater than one
  /// across some face). A balanced request is the backend's precondition; run it
  /// through [`balance_2to1`] first.
  UnbalancedRequest,
  /// A flag referenced a cell id outside the mesh.
  InvalidCell,
  /// The backend cannot satisfy the request (e.g. a cell is already at the
  /// maximum refinement level, or the requested axis is not refinable).
  Unsupported,
}

impl ErrorDomain for RefineError {
  fn domain(&self) -> &str {
    "tessera refine"
  }
}

impl std::fmt::Display for RefineError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      RefineError::UnbalancedRequest => {
        write!(f, "adapt request is not 2:1-balanced")
      }
      RefineError::InvalidCell => {
        write!(f, "adapt request references an unknown cell")
      }
      RefineError::Unsupported => {
        write!(f, "backend cannot satisfy the requested refinement")
      }
    }
  }
}

/// The *desired* adaptation, as chosen by a refinement criterion before
/// balancing. Cells listed in `refine` should drop to a finer level and cells in
/// `coarsen` to a coarser one; the lists need not be 2:1-balanced — that is
/// [`balance_2to1`]'s job.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RefineFlags {
  pub refine: Vec<CellId>,
  pub coarsen: Vec<CellId>,
}

/// A 2:1-*balanced* adaptation request: the output of [`balance_2to1`] and the
/// input to [`RefinableMesh::adapt`]. Each listed cell changes by exactly one
/// refinement level.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AdaptRequest {
  pub refine: Vec<CellId>,
  pub coarsen: Vec<CellId>,
}

/// A mesh that can be adapted at runtime. Implemented explicitly by a backend
/// (e.g. the cube-sphere) — *not* blanket-implemented — because realising a
/// request needs backend-specific knowledge of child layout.
pub trait RefinableMesh<const D: usize>: Mesh<D> {
  /// The current refinement level of a cell (0 = base mesh). Drives the
  /// balancer's neighbour-difference check.
  fn cell_level(&self, cell: CellId) -> u32;

  /// How many children a cell produces when refined one level (e.g. 4 for an
  /// angular quad-split). Lets the storage/driver size the remap up front.
  fn refine_fanout(&self) -> usize;

  /// Realise a *balanced* request: rebuild a new immutable mesh and return it
  /// alongside the [`CellRemap`] from the old cell space to the new one. The
  /// backend is authoritative for child layout and **must reject** an
  /// unbalanced or invalid request rather than produce a bad topology.
  fn adapt(
    &self,
    request: &AdaptRequest,
  ) -> AetherResult<(Arc<dyn Mesh<D>>, CellRemap)>;
}

/// Turn a *desired* [`RefineFlags`] into a 2:1-balanced [`AdaptRequest`].
///
/// Mesh-agnostic: it reads only `topo`'s interior-face adjacency and the current
/// `level` of each cell, so it works for any [`Topology`]. `cell_count` sizes the
/// working level array (the [`Topology`] trait does not expose a cell count). The
/// algorithm computes each cell's *target* level after the desired changes, then
/// iterates to a fixpoint raising the lower side of any face whose endpoints
/// differ by more than one level. It emits a single-level request: cells whose
/// target exceeds their current level are refined, cells whose target dropped are
/// coarsened. An already-balanced desire is returned unchanged.
pub fn balance_2to1(
  topo: &dyn Topology,
  cell_count: usize,
  level: impl Fn(CellId) -> u32,
  desired: &RefineFlags,
) -> AdaptRequest {
  let current: Vec<u32> =
    (0..cell_count).map(|i| level(CellId::from(i))).collect();
  let mut target = current.clone();

  // Refinement wins over coarsening on the same cell.
  for &c in &desired.refine {
    let i = c.index();
    target[i] = current[i] + 1;
  }
  for &c in &desired.coarsen {
    let i = c.index();
    if !desired.refine.contains(&c) {
      target[i] = current[i].saturating_sub(1);
    }
  }

  // Fixpoint: no face may span more than one level. Raising the lower side can
  // create new violations elsewhere, so iterate until stable.
  loop {
    let mut changed = false;
    for &(_, a, b) in topo.interior_faces() {
      let (ia, ib) = (a.index(), b.index());
      if target[ia] > target[ib] + 1 {
        target[ib] = target[ia] - 1;
        changed = true;
      } else if target[ib] > target[ia] + 1 {
        target[ia] = target[ib] - 1;
        changed = true;
      }
    }
    if !changed {
      break;
    }
  }

  let mut refine = Vec::new();
  let mut coarsen = Vec::new();
  for i in 0..cell_count {
    let cell = CellId::from(i);
    if target[i] > current[i] {
      refine.push(cell);
    } else if target[i] < current[i] {
      coarsen.push(cell);
    }
  }
  AdaptRequest { refine, coarsen }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::cube_sphere::CubeSphere;
  use crate::geometry::CellGeometry;

  const R_INNER: f64 = 6.371e6;
  const R_OUTER: f64 = 6.391e6;

  #[test]
  fn identity_remap_is_a_no_op() {
    let remap = CellRemap::identity(TopologyEpoch::ZERO, 5);
    assert_eq!(remap.old_count(), 5);
    assert_eq!(remap.new_count(), 5);
    assert_eq!(remap.old_epoch(), remap.new_epoch());
    for i in 0..5 {
      let c = CellId::from(i);
      assert_eq!(remap.image_of(c), Some(c));
      assert!(!remap.is_newborn(c));
    }
    assert_eq!(remap.died().count(), 0);
    assert_eq!(remap.born().count(), 0);
  }

  #[test]
  fn remap_records_birth_and_death_in_both_directions() {
    // Old cell 0 is refined into new children 0,1; old cell 1 survives as new 2.
    let remap = CellRemap::new(
      TopologyEpoch::ZERO,
      TopologyEpoch::ZERO.next(),
      vec![None, Some(CellId::from(2))],
      vec![
        NewCellSource::Child {
          parent: CellId::from(0),
        },
        NewCellSource::Child {
          parent: CellId::from(0),
        },
        NewCellSource::Survivor(CellId::from(1)),
      ],
    );
    // Death of old 0, survival of old 1.
    assert_eq!(remap.image_of(CellId::from(0)), None);
    assert_eq!(remap.image_of(CellId::from(1)), Some(CellId::from(2)));
    assert_eq!(remap.died().collect::<Vec<_>>(), vec![CellId::from(0)]);
    // Birth of new 0 and 1, survival of new 2.
    assert!(remap.is_newborn(CellId::from(0)));
    assert!(remap.is_newborn(CellId::from(1)));
    assert!(!remap.is_newborn(CellId::from(2)));
    assert_eq!(
      remap.born().collect::<Vec<_>>(),
      vec![CellId::from(0), CellId::from(1)]
    );
  }

  #[test]
  fn balance_of_an_already_balanced_request_is_identity() {
    // Uniform level-0 mesh: refining one cell creates only a 0→1 jump (diff 1),
    // which is already balanced, so the request is the desired set unchanged.
    let mesh = CubeSphere::new([8, 8, 1], R_INNER, R_OUTER);
    let desired = RefineFlags {
      refine: vec![CellId::from(0)],
      coarsen: vec![],
    };
    let req = balance_2to1(&mesh, mesh.cell_count(), |_| 0, &desired);
    assert_eq!(req.refine, vec![CellId::from(0)]);
    assert!(req.coarsen.is_empty());
  }

  #[test]
  fn balance_propagates_to_keep_neighbour_levels_within_one() {
    // A pre-existing level-2 cell next to level-0 cells: balancing must refine the
    // intervening level-0 neighbours up to level 1 so no face spans >1 level.
    let mesh = CubeSphere::new([8, 8, 1], R_INNER, R_OUTER);
    // Hand-built level field: cell 0 already at level 2, everything else 0.
    let level = |c: CellId| if c.index() == 0 { 2 } else { 0 };
    let req =
      balance_2to1(&mesh, mesh.cell_count(), level, &RefineFlags::default());
    // Cell 0's interior neighbours (level 0) must be lifted to level 1.
    let mut neighbours = std::collections::HashSet::new();
    for &(_, a, b) in mesh.interior_faces() {
      if a == CellId::from(0) {
        neighbours.insert(b);
      } else if b == CellId::from(0) {
        neighbours.insert(a);
      }
    }
    assert!(!neighbours.is_empty());
    for n in neighbours {
      assert!(req.refine.contains(&n), "neighbour {n:?} not balanced");
    }
  }
}
