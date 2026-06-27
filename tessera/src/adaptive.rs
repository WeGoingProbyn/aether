// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Mesh-agnostic adaptive refinement wrapper.
//!
//! [`AdaptiveMesh`] wraps any base mesh that implements the [`Subdividable`]
//! geometry hook and adds the [`RefinableMesh`] capability on top — so the
//! adaptive machinery lives here, once, instead of inside every mesh type. The
//! base mesh owns only the *geometry* of refinement (how a cell subdivides);
//! `AdaptiveMesh` owns the *bookkeeping* (levels, the [`CellRemap`], and the
//! [`Mesh`] view the rest of the engine sees).
//!
//! **v1 scope (step 2A): uniform refinement.** Every cell refines together, so
//! the refined mesh has no mixed levels and therefore no hanging faces. The
//! public seam ([`RefinableMesh`] + [`CellRemap`] + [`Subdividable`]) is the
//! stable contract; step 2B replaces the *internal* representation with a
//! per-cell refinement forest to support selective refinement and hanging-node
//! topology, without changing that seam.

use std::sync::Arc;

use utility::domain::{
  BoundaryTag, CellId, CellRemap, FaceId, Point, TopologyEpoch,
};
use utility::error::{AetherError, AetherResult};
use utility::maths::vector::Vector;

use crate::geometry::{CellGeometry, CellMetrics, FaceGeometry, FaceMetrics};
use crate::mesh::Mesh;
use crate::refine::{AdaptRequest, RefinableMesh, RefineError};
use crate::topology::{FaceConnection, Topology};

/// A base mesh that knows how to subdivide its own cells — the geometry hook the
/// mesh-agnostic [`AdaptiveMesh`] delegates to. Keeping it on the base mesh is
/// what lets `AdaptiveMesh` stay generic: only the cube-sphere (or any future
/// mesh) knows how its curved cells split.
///
/// v1 exposes *uniform* refinement; selective per-cell subdivision (with the
/// child-geometry queries a hanging-node rebuild needs) is added in step 2B.
pub trait Subdividable: Mesh<3> {
  /// Children produced when one cell refines one level (4 for an angular
  /// quad-split).
  fn refine_fanout(&self) -> usize;

  /// Refine *every* cell one level and return the finer mesh — itself
  /// [`Subdividable`], so it can refine again — together with the old→new
  /// [`CellRemap`]. Each old cell is replaced by its children (it has no single
  /// new image), and every new cell is a [`Child`](utility::domain::NewCellSource::Child)
  /// of the old cell it subdivides. `old_epoch`/`new_epoch` stamp the remap.
  fn refine_uniform_once(
    &self,
    old_epoch: TopologyEpoch,
    new_epoch: TopologyEpoch,
  ) -> AetherResult<(Arc<dyn Subdividable>, CellRemap)>;
}

/// A base mesh plus a refinement state, presented to the engine as a plain
/// [`Mesh`] and to the adapt barrier as a [`RefinableMesh`]. In v1 the state is a
/// single uniform `level`; all geometry/topology is delegated to the (possibly
/// already-refined) inner mesh.
pub struct AdaptiveMesh {
  inner: Arc<dyn Subdividable>,
  level: u32,
  epoch: TopologyEpoch,
}

impl AdaptiveMesh {
  /// Wrap a base mesh at refinement level 0 ([`TopologyEpoch::ZERO`]).
  pub fn new(base: Arc<dyn Subdividable>) -> Self {
    Self {
      inner: base,
      level: 0,
      epoch: TopologyEpoch::ZERO,
    }
  }

  /// The current uniform refinement level (0 = base mesh).
  pub fn level(&self) -> u32 {
    self.level
  }

  /// The current topology epoch (bumped once per adapt).
  pub fn epoch(&self) -> TopologyEpoch {
    self.epoch
  }

  /// The wrapped (possibly already-refined) base mesh.
  pub fn inner(&self) -> &Arc<dyn Subdividable> {
    &self.inner
  }

  /// Whether `request` refines every cell exactly once and coarsens nothing —
  /// the only shape v1 realises.
  fn is_uniform_refine(&self, request: &AdaptRequest) -> bool {
    let n = self.inner.cell_count();
    if !request.coarsen.is_empty() || request.refine.len() != n {
      return false;
    }
    let mut seen = vec![false; n];
    for c in &request.refine {
      match seen.get_mut(c.index()) {
        Some(slot) if !*slot => *slot = true,
        _ => return false, // out of range or duplicate
      }
    }
    seen.into_iter().all(|s| s)
  }
}

impl CellGeometry<3> for AdaptiveMesh {
  fn cell_centroid(&self, cell: CellId) -> &Point<3> {
    self.inner.cell_centroid(cell)
  }
  fn cell_world_centroid(&self, cell: CellId) -> Point<3> {
    self.inner.cell_world_centroid(cell)
  }
  fn cell_volume(&self, cell: CellId) -> f64 {
    self.inner.cell_volume(cell)
  }
  fn cell_metrics(&self, cell: CellId) -> &CellMetrics<3> {
    self.inner.cell_metrics(cell)
  }
  fn cell_count(&self) -> usize {
    self.inner.cell_count()
  }
}

impl FaceGeometry<3> for AdaptiveMesh {
  fn face_centroid(&self, face: FaceId) -> &Point<3> {
    self.inner.face_centroid(face)
  }
  fn face_world_centroid(&self, face: FaceId) -> Point<3> {
    self.inner.face_world_centroid(face)
  }
  fn face_world_vertices(&self, face: FaceId) -> Option<Vec<Point<3>>> {
    self.inner.face_world_vertices(face)
  }
  fn face_area_vector(&self, face: FaceId) -> Vector<f64, 3> {
    self.inner.face_area_vector(face)
  }
  fn face_area(&self, face: FaceId) -> f64 {
    self.inner.face_area(face)
  }
  fn face_metrics(&self, face: FaceId) -> &FaceMetrics<3> {
    self.inner.face_metrics(face)
  }
  fn face_count(&self) -> usize {
    self.inner.face_count()
  }
}

impl Topology for AdaptiveMesh {
  fn face_connection(&self, face: FaceId) -> &FaceConnection {
    self.inner.face_connection(face)
  }
  fn cell_faces(&self, cell: CellId) -> &[FaceId] {
    self.inner.cell_faces(cell)
  }
  fn interior_faces(&self) -> &[(FaceId, CellId, CellId)] {
    self.inner.interior_faces()
  }
  fn boundary_faces(&self, tag: BoundaryTag) -> &[(FaceId, CellId)] {
    self.inner.boundary_faces(tag)
  }
  fn boundary_tags(&self) -> Box<dyn Iterator<Item = BoundaryTag> + '_> {
    self.inner.boundary_tags()
  }
}

impl RefinableMesh<3> for AdaptiveMesh {
  fn cell_level(&self, _cell: CellId) -> u32 {
    // Uniform in v1; step 2B makes this per-cell.
    self.level
  }

  fn refine_fanout(&self) -> usize {
    self.inner.refine_fanout()
  }

  fn adapt(
    &self,
    request: &AdaptRequest,
  ) -> AetherResult<(Arc<dyn Mesh<3>>, CellRemap)> {
    if !self.is_uniform_refine(request) {
      return Err(AetherError::new(RefineError::Unsupported).context(
        "v1 AdaptiveMesh refines uniformly only (every cell, no coarsening)",
      ));
    }
    let new_epoch = self.epoch.next();
    let (finer, remap) =
      self.inner.refine_uniform_once(self.epoch, new_epoch)?;
    let adapted = AdaptiveMesh {
      inner: finer,
      level: self.level + 1,
      epoch: new_epoch,
    };
    Ok((Arc::new(adapted), remap))
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::cube_sphere::CubeSphere;
  use crate::refine::RefineFlags;

  const R_INNER: f64 = 6.371e6;
  const R_OUTER: f64 = 6.391e6;

  fn refine_all(n: usize) -> AdaptRequest {
    AdaptRequest {
      refine: (0..n).map(CellId::from).collect(),
      coarsen: vec![],
    }
  }

  #[test]
  fn uniform_refine_quadruples_cells_and_conserves_volume() {
    let base = Arc::new(CubeSphere::new([4, 4, 2], R_INNER, R_OUTER));
    let mesh = AdaptiveMesh::new(base);
    let old_count = mesh.cell_count();
    let old_volume: f64 = (0..old_count)
      .map(|i| mesh.cell_volume(CellId::from(i)))
      .sum();

    let (refined, remap) = mesh.adapt(&refine_all(old_count)).unwrap();

    // 4 angular children per cell (radial preserved).
    assert_eq!(refined.cell_count(), old_count * 4);
    assert_eq!(remap.new_count(), old_count * 4);
    assert_eq!(remap.old_count(), old_count);
    assert_eq!(remap.new_epoch(), TopologyEpoch::ZERO.next());

    // Total shell volume is unchanged by refining the partition.
    let new_volume: f64 = (0..refined.cell_count())
      .map(|i| refined.cell_volume(CellId::from(i)))
      .sum();
    assert!(
      (old_volume - new_volume).abs() / old_volume < 1e-9,
      "refinement changed total volume: {old_volume} vs {new_volume}"
    );
  }

  #[test]
  fn each_parent_has_fanout_children_whose_volumes_sum_to_it() {
    let base = Arc::new(CubeSphere::new([4, 4, 1], R_INNER, R_OUTER));
    let mesh = AdaptiveMesh::new(base);
    let old_count = mesh.cell_count();
    let (refined, remap) = mesh.adapt(&refine_all(old_count)).unwrap();

    // Group new children by parent and check the count + volume partition.
    let mut child_volume_sum = vec![0.0; old_count];
    let mut child_count = vec![0usize; old_count];
    for new_i in 0..refined.cell_count() {
      let parent = match remap.source_of(CellId::from(new_i)) {
        utility::domain::NewCellSource::Child { parent } => parent.index(),
        other => panic!("expected Child, got {other:?}"),
      };
      child_count[parent] += 1;
      child_volume_sum[parent] += refined.cell_volume(CellId::from(new_i));
    }
    for parent in 0..old_count {
      assert_eq!(child_count[parent], 4, "parent {parent} child count");
      let parent_vol = mesh.cell_volume(CellId::from(parent));
      assert!(
        (child_volume_sum[parent] - parent_vol).abs() / parent_vol < 1e-9,
        "parent {parent} children volume {} != {parent_vol}",
        child_volume_sum[parent]
      );
    }

    // Every old cell is "refined away": no single new image.
    assert_eq!(remap.died().count(), old_count);
  }

  #[test]
  fn selective_refine_is_unsupported_in_v1() {
    let base = Arc::new(CubeSphere::new([4, 4, 1], R_INNER, R_OUTER));
    let mesh = AdaptiveMesh::new(base);
    let one = AdaptRequest {
      refine: vec![CellId::from(0)],
      coarsen: vec![],
    };
    assert!(mesh.adapt(&one).is_err());
    // A balanced single-cell request still isn't uniform → unsupported in 2A.
    let _ = RefineFlags::default();
  }
}
