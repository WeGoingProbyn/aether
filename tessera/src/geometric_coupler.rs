// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Geometric, level-aware coupler for two radially-stacked shells.
//!
//! [`RadialStackCoupler`](crate::radial_stack::RadialStackCoupler) pairs cells by
//! 1:1 `(panel,i,j,k)` index arithmetic, which assumes both shells are structured
//! cube-spheres at matching angular resolution. Under AMR a shell can be
//! adaptively refined and loses that structure, so this coupler instead matches
//! the shells' shared interface boundary faces by **world-space angular
//! footprint** — the lower shell's outer ring (`lower_tag`) against the upper
//! shell's inner ring (`upper_tag`). Matching is angular (radius-independent), so
//! it works even though the two shells touch at slightly different radii.
//!
//! Because coupled shells share the base angular grid, interface faces are
//! *nested*: a coarse face on one side exactly contains the fine faces on the
//! other. Each overlap becomes one [`FacePair`]; a coarse cell therefore appears
//! in several pairs (N:M). The per-pair overlap area (the finer face) is computed
//! by [`CoupledFace::from_pair`](crate::coupling::CoupledFace::from_pair) and
//! normalised into gather/scatter weights by `syzygy`'s `CouplingStencil`.
//!
//! Limitation: coupled shells must share the base angular grid (so interface
//! faces nest exactly rather than partially overlapping).

use utility::domain::{BoundaryTag, CellId, FaceId};
use utility::maths::vector::Vector;

use crate::coupling::{FacePair, MeshCoupler, Side};
use crate::mesh::Mesh;

pub struct GeometricRadialCoupler {
  pairs: Vec<FacePair>,
}

impl GeometricRadialCoupler {
  /// Pair `lower`'s `lower_tag` boundary faces with `upper`'s `upper_tag`
  /// boundary faces by angular-footprint nesting.
  pub fn build(
    lower: &dyn Mesh<3>,
    lower_tag: BoundaryTag,
    upper: &dyn Mesh<3>,
    upper_tag: BoundaryTag,
  ) -> Self {
    let lower_faces = interface_faces(lower, lower_tag);
    let upper_faces = interface_faces(upper, upper_tag);

    // Two-pass nearest-neighbour by angular position. Each face pairs with the
    // single face *containing its centroid* on the other side — which is its
    // angularly-nearest face, since the interface tiles the sphere. Doing it from
    // both sides captures every nesting direction: when the lower side is finer,
    // each fine lower face finds its one coarse upper parent (pass 1); when the
    // upper side is finer, each fine upper face finds its coarse lower parent
    // (pass 2). The union (deduplicated) is the exact N:M overlap set; for a
    // conforming interface it is the 1:1 bijection.
    let mut seen: std::collections::HashSet<(usize, usize)> =
      std::collections::HashSet::new();
    let mut pairs = Vec::new();
    let mut add = |lf: FaceId, uf: FaceId, pairs: &mut Vec<FacePair>| {
      if seen.insert((lf.index(), uf.index())) {
        pairs.push(FacePair::new(lf, uf));
      }
    };
    for (lf, ldir) in &lower_faces {
      if let Some((uf, _)) = nearest(*ldir, &upper_faces) {
        add(*lf, uf, &mut pairs);
      }
    }
    for (uf, udir) in &upper_faces {
      if let Some((lf, _)) = nearest(*udir, &lower_faces) {
        add(lf, *uf, &mut pairs);
      }
    }
    Self { pairs }
  }

  /// Build with the default shell interface tags: the lower shell's outer ring is
  /// [`BoundaryTag::AtmosphereEdge`] and the upper shell's inner ring is
  /// [`BoundaryTag::Ground`] (the cube-sphere shell defaults).
  pub fn between_shells(lower: &dyn Mesh<3>, upper: &dyn Mesh<3>) -> Self {
    Self::build(
      lower,
      BoundaryTag::AtmosphereEdge,
      upper,
      BoundaryTag::Ground,
    )
  }

  pub fn pair_count(&self) -> usize {
    self.pairs.len()
  }
}

impl MeshCoupler for GeometricRadialCoupler {
  fn paired_face(&self, side: Side, face: FaceId) -> Option<(Side, FaceId)> {
    self.pairs.iter().find_map(|p| match side {
      Side::A => (p.a() == face).then(|| (Side::B, p.b())),
      Side::B => (p.b() == face).then(|| (Side::A, p.a())),
    })
  }

  fn paired_cell(&self, _side: Side, _cell: CellId) -> Option<(Side, CellId)> {
    // An N:M coupler has no single cell↔cell pairing; consumers use the weighted
    // face pairs (`pairs()` → `CouplingStencil`).
    None
  }

  fn pairs(&self) -> &[FacePair] {
    &self.pairs
  }
}

/// For each boundary face tagged `tag`: its unit world direction (the radius is
/// irrelevant — matching is angular).
fn interface_faces(
  mesh: &dyn Mesh<3>,
  tag: BoundaryTag,
) -> Vec<(FaceId, [f64; 3])> {
  mesh
    .boundary_faces(tag)
    .iter()
    .filter_map(|(face, _owner)| {
      Some((*face, unit(mesh.face_world_centroid(*face))?))
    })
    .collect()
}

/// The face in `faces` whose direction is angularly nearest to `dir`.
fn nearest(
  dir: [f64; 3],
  faces: &[(FaceId, [f64; 3])],
) -> Option<(FaceId, f64)> {
  faces
    .iter()
    .map(|(f, d)| (*f, angle(dir, *d)))
    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
}

fn unit(v: Vector<f64, 3>) -> Option<[f64; 3]> {
  let m = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
  (m > 0.0).then(|| [v[0] / m, v[1] / m, v[2] / m])
}

fn angle(a: [f64; 3], b: [f64; 3]) -> f64 {
  (a[0] * b[0] + a[1] * b[1] + a[2] * b[2])
    .clamp(-1.0, 1.0)
    .acos()
}

#[cfg(test)]
mod tests {
  use std::sync::Arc;

  use crate::adaptive::AdaptiveMesh;
  use crate::coupling::CoupledFace;
  use crate::cube_sphere::CubeSphere;
  use crate::mesh::Mesh;
  use crate::radial_stack::RadialStackCoupler;
  use crate::refine::AdaptRequest;
  use utility::domain::MeshKey;

  use super::*;

  fn shells() -> (Arc<CubeSphere>, Arc<CubeSphere>) {
    // Lower (surface-like) just below the interface, upper (atmosphere-like)
    // just above; same angular grid, different radial layer counts.
    let lower = Arc::new(CubeSphere::new([4, 4, 1], 0.9, 1.0));
    let upper = Arc::new(CubeSphere::new([4, 4, 2], 1.0, 1.1));
    (lower, upper)
  }

  #[test]
  fn uniform_shells_give_a_one_to_one_bijection() {
    let (lower, upper) = shells();
    let coupler =
      GeometricRadialCoupler::between_shells(lower.as_ref(), upper.as_ref());
    // Same pair count as the index coupler (6 panels × 4×4 angular cells).
    let index = RadialStackCoupler::new([4, 4], 1, 2);
    assert_eq!(coupler.pair_count(), index.pairs().len());

    // Each lower and each upper interface face appears exactly once.
    let mut lower_seen = std::collections::HashMap::new();
    let mut upper_seen = std::collections::HashMap::new();
    for p in coupler.pairs() {
      *lower_seen.entry(p.a().index()).or_insert(0) += 1;
      *upper_seen.entry(p.b().index()).or_insert(0) += 1;
    }
    assert!(lower_seen.values().all(|&c| c == 1));
    assert!(upper_seen.values().all(|&c| c == 1));
  }

  #[test]
  fn refining_one_side_nests_n_to_1_and_overlap_areas_sum() {
    let (lower_base, upper) = shells();
    // Refine a panel-interior surface cell (panel 0, angular (1,1) ⇒ local 5).
    let lower = AdaptiveMesh::new(lower_base);
    let (refined, _) = lower
      .refine(&AdaptRequest {
        refine: vec![CellId::from(5)],
        coarsen: vec![],
      })
      .unwrap();

    let coupler =
      GeometricRadialCoupler::between_shells(&refined, upper.as_ref());
    let uniform = GeometricRadialCoupler::between_shells(
      &AdaptiveMesh::new(Arc::new(CubeSphere::new([4, 4, 1], 0.9, 1.0))),
      upper.as_ref(),
    );
    // Refining one interface cell into four adds three pairs (1 → 4).
    assert_eq!(coupler.pair_count(), uniform.pair_count() + 3);

    // Find the upper face that now gathers from several lower faces, and confirm
    // the overlap areas of its pairs sum to (approximately) its own face area.
    let mut by_upper: std::collections::HashMap<usize, Vec<FacePair>> =
      std::collections::HashMap::new();
    for p in coupler.pairs() {
      by_upper.entry(p.b().index()).or_default().push(*p);
    }
    let (&_upper_idx, group) = by_upper
      .iter()
      .find(|(_, g)| g.len() == 4)
      .expect("a 4:1 group");

    let mut overlap_sum = 0.0;
    let mut coarse_area = 0.0;
    for pair in group {
      let cf = CoupledFace::from_pair(
        MeshKey::SURFACE,
        &refined,
        MeshKey::ATMOSPHERE,
        upper.as_ref(),
        *pair,
      );
      overlap_sum += cf.area; // min(fine lower, coarse upper) = fine lower area
      coarse_area = cf.area_b; // the upper (coarse) face area
    }
    // Sanity that the four pairs really are the coarse face's children: their
    // overlap areas tile it. The match is only to O(cell²) because cube-sphere
    // `phys_area` is a midpoint-rule estimate (a coarse cell's single-sample area
    // ≈ the sum of its four fine sub-cells only up to curvature — the same
    // approximation behind the Phase-3 conservation tolerance). Exact
    // conservation comes from the *normalised* coupling weights, asserted at
    // `CouplingStencil` construction, not from raw areas matching.
    assert!(
      (overlap_sum - coarse_area).abs() / coarse_area < 2e-2,
      "fine overlaps {overlap_sum} should ~tile the coarse face {coarse_area}"
    );
  }
}
