// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Mesh-agnostic adaptive refinement wrapper.
//!
//! [`AdaptiveMesh`] wraps any base mesh that implements the [`Subdividable`]
//! geometry hook and adds the [`RefinableMesh`] capability on top — so the
//! adaptive machinery lives here, once, instead of inside every mesh type. The
//! base mesh owns only the *geometry* of refinement (how a cell subdivides);
//! `AdaptiveMesh` owns the *bookkeeping*: the [`RefinementForest`] of active
//! cells, the rebuilt flat [`Mesh`] the engine sees, and the [`CellRemap`] each
//! adapt produces.
//!
//! # Hanging faces
//!
//! Refinement is angular (radial layers are preserved), so an interface is either
//! *conforming* (both sides at the same level) or *hanging* (the coarse side
//! meets several fine cells). A hanging interface is represented as **N
//! independent fine sub-face records**, each an ordinary
//! [`FaceConnection::Interior`]`{ owner: fine, neighbour: coarse }` carrying the
//! *fine* sub-face's area and outward normal. The coarse cell therefore simply
//! has more faces; the finite-volume kernel needs no special case, and
//! conservation follows from Σ(fine areas) = coarse area. Because the coarse face
//! is simply tiled by however many fine sub-faces meet it, this handles arbitrary
//! level jumps conservatively — 2:1 balance is the adaptation driver's *accuracy*
//! policy (via `balance_2to1`), not a requirement of the geometry.
//!
//! # Geometry oracle
//!
//! Each leaf's geometry is read from a uniformly-refined copy of the base mesh
//! at the leaf's level ([`Subdividable::uniform_at_level`] +
//! [`Subdividable::leaf_fine_cell`]), so no curvilinear maths lives here. Faces
//! are matched across leaves by their world-space footprint, which handles
//! intra-panel, conforming, and hanging interfaces uniformly.
//!
//! **v1 limitation:** refinement that creates a level jump *across a cube-sphere
//! panel seam* is rejected (the seam faces cannot be matched by the current
//! footprint test); keep refined regions inside a panel, or refine seam-adjacent
//! cells together. Interior regional refinement — the common case — is supported.

use std::collections::HashMap;
use std::sync::Arc;

use utility::domain::{
  BoundaryTag, CellId, CellRemap, FaceId, Point, TopologyEpoch,
};
use utility::error::{AetherError, AetherResult};
use utility::maths::vector::Vector;

use crate::geometry::{CellGeometry, CellMetrics, FaceGeometry, FaceMetrics};
use crate::mesh::Mesh;
use crate::refine::{AdaptRequest, RefinableMesh, RefineError};
use crate::refine_forest::RefinementForest;
use crate::topology::{FaceConnection, Topology};

/// A base mesh that knows how to subdivide its own cells — the geometry hook the
/// mesh-agnostic [`AdaptiveMesh`] delegates to. Keeping it on the base mesh is
/// what lets `AdaptiveMesh` stay generic: only the cube-sphere (or any future
/// mesh) knows how its curved cells split.
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

  /// A uniformly-refined copy of this *base* mesh at refinement `levels` (≥ 1).
  /// The mesh-agnostic geometry oracle: `AdaptiveMesh` builds one of these per
  /// distinct level present in the forest and reads each leaf's geometry from it
  /// (via [`leaf_fine_cell`](Subdividable::leaf_fine_cell)), so no per-leaf
  /// curvilinear maths lives in the wrapper. Must be called on the level-0 base.
  fn uniform_at_level(&self, levels: u32) -> Arc<dyn Subdividable>;

  /// The cell id, within [`uniform_at_level`](Subdividable::uniform_at_level)`(path.len())`,
  /// of the leaf reached from base cell `base_cell` by `path` (child indices, the
  /// quad-split convention: child `k` is angular quadrant `(k & 1, (k >> 1) & 1)`).
  /// This is the only place the backend's index layout is exposed; the wrapper
  /// uses it purely to look up geometry.
  fn leaf_fine_cell(&self, base_cell: CellId, path: &[u8]) -> CellId;
}

/// A base mesh plus a [`RefinementForest`], presented to the engine as a plain
/// [`Mesh`] (rebuilt flat geometry + topology) and to the adapt barrier as a
/// [`RefinableMesh`].
pub struct AdaptiveMesh {
  base: Arc<dyn Subdividable>,
  forest: RefinementForest,
  epoch: TopologyEpoch,

  // Per active cell (leaf), in dense CellId order.
  cell_levels: Vec<u32>,
  cell_centroids: Vec<Point<3>>,
  cell_world_centroids: Vec<Point<3>>,
  cell_volumes: Vec<f64>,
  cell_metrics: Vec<CellMetrics<3>>,
  cell_face_adj: Vec<Vec<FaceId>>,

  // Per assembled face.
  face_centroids: Vec<Point<3>>,
  face_world_centroids: Vec<Point<3>>,
  face_world_vertices: Vec<Vec<Point<3>>>,
  face_areas: Vec<f64>,
  face_area_vectors: Vec<Vector<f64, 3>>,
  face_metrics: Vec<FaceMetrics<3>>,
  face_connections: Vec<FaceConnection>,

  interior_face_list: Vec<(FaceId, CellId, CellId)>,
  boundary_face_lists: Vec<(BoundaryTag, Vec<(FaceId, CellId)>)>,
}

/// One face of one leaf, gathered from the geometry oracle before matching.
struct LeafFace {
  owner: usize,
  verts: Vec<Point<3>>,
  world_centroid: Point<3>,
  comp_centroid: Point<3>,
  area: f64,
  /// Outward from `owner`.
  area_vec: Vector<f64, 3>,
  metrics: FaceMetrics<3>,
  /// `Some((tag, out_sign))` if this is a domain boundary in the oracle mesh.
  boundary: Option<(BoundaryTag, f64)>,
}

fn dot3(a: &Vector<f64, 3>, b: &Vector<f64, 3>) -> f64 {
  a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn dist3(a: &Point<3>, b: &Point<3>) -> f64 {
  ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Order-independent world-footprint key: vertices rounded to the millimetre and
/// sorted. Two faces that are the same physical quad (e.g. shared by two leaves
/// at the same level, or the same oracle face seen from both sides) hash equal.
fn footprint_key(verts: &[Point<3>]) -> Vec<[i64; 3]> {
  let mut k: Vec<[i64; 3]> = verts
    .iter()
    .map(|v| {
      [
        (v[0] * 1.0e3).round() as i64,
        (v[1] * 1.0e3).round() as i64,
        (v[2] * 1.0e3).round() as i64,
      ]
    })
    .collect();
  k.sort();
  k
}

impl AdaptiveMesh {
  /// Wrap a base mesh at refinement level 0 ([`TopologyEpoch::ZERO`]). The
  /// assembled level-0 mesh reproduces the base mesh's topology.
  pub fn new(base: Arc<dyn Subdividable>) -> Self {
    let forest = RefinementForest::new(base.cell_count());
    // Level 0 has no level jumps, so assembly cannot fail.
    Self::build(base, forest, TopologyEpoch::ZERO)
      .expect("level-0 adaptive mesh assembly is always valid")
  }

  /// The current topology epoch (bumped once per adapt).
  pub fn epoch(&self) -> TopologyEpoch {
    self.epoch
  }

  /// The wrapped level-0 base mesh.
  pub fn base(&self) -> &Arc<dyn Subdividable> {
    &self.base
  }

  /// The refinement level of a cell (0 = unrefined). Same as the
  /// [`RefinableMesh`] method, available without the trait in scope.
  pub fn level_of(&self, cell: CellId) -> u32 {
    self.cell_levels[cell.index()]
  }

  /// Apply a (balanced) request and return the **concrete** refined mesh plus
  /// the remap — the typed counterpart of [`RefinableMesh::adapt`], so the
  /// adaptation driver can keep refining the same `AdaptiveMesh` across ticks
  /// (the `dyn Mesh` the trait returns would lose the type).
  pub fn refine(
    &self,
    request: &AdaptRequest,
  ) -> AetherResult<(AdaptiveMesh, CellRemap)> {
    let new_epoch = self.epoch.next();
    let (new_forest, remap) = self.forest.adapt(request, self.epoch, new_epoch);
    Ok((
      AdaptiveMesh::build(self.base.clone(), new_forest, new_epoch)?,
      remap,
    ))
  }

  /// Assemble the flat mesh from `base` + `forest`. Errors with
  /// [`RefineError::UnbalancedRequest`] if a leaf face cannot be matched to its
  /// neighbour(s) — in practice, refinement that creates a level jump across a
  /// cube-sphere panel seam (the v1 limitation). Level jumps *within* a panel are
  /// tiled conservatively and do not error.
  fn build(
    base: Arc<dyn Subdividable>,
    forest: RefinementForest,
    epoch: TopologyEpoch,
  ) -> AetherResult<Self> {
    let leaves = forest.leaves();
    let ncells = leaves.len();
    let max_level = leaves.iter().map(|l| l.level()).max().unwrap_or(0);

    // Geometry oracle: one uniform mesh per level present (level 0 = base).
    let oracles: Vec<Option<Arc<dyn Subdividable>>> = (0..=max_level)
      .map(|lvl| (lvl >= 1).then(|| base.uniform_at_level(lvl)))
      .collect();

    let mut cell_levels = Vec::with_capacity(ncells);
    let mut cell_centroids = Vec::with_capacity(ncells);
    let mut cell_world_centroids = Vec::with_capacity(ncells);
    let mut cell_volumes = Vec::with_capacity(ncells);
    let mut cell_metrics = Vec::with_capacity(ncells);
    let mut lfaces: Vec<LeafFace> = Vec::new();

    for (leaf_id, leaf) in leaves.iter().enumerate() {
      let level = leaf.level();
      let (src, src_cell): (&dyn Subdividable, CellId) = if level == 0 {
        (base.as_ref(), leaf.base_cell)
      } else {
        let m = oracles[level as usize].as_ref().unwrap().as_ref();
        (m, base.leaf_fine_cell(leaf.base_cell, &leaf.path))
      };

      cell_levels.push(level);
      cell_centroids.push(*src.cell_centroid(src_cell));
      cell_world_centroids.push(src.cell_world_centroid(src_cell));
      cell_volumes.push(src.cell_volume(src_cell));
      let cm = src.cell_metrics(src_cell);
      cell_metrics.push(CellMetrics {
        sqrt_metric: cm.sqrt_metric,
        comp_volume: cm.comp_volume,
        phys_volume: cm.phys_volume,
      });

      for &face in src.cell_faces(src_cell) {
        let raw_av = src.face_area_vector(face);
        let area = src.face_area(face);
        let (area_vec, boundary) = match src.face_connection(face) {
          FaceConnection::Interior { owner, .. } => {
            // Orient outward from THIS cell, regardless of oracle ownership.
            let av = if *owner == src_cell {
              raw_av
            } else {
              raw_av * -1.0
            };
            (av, None)
          }
          FaceConnection::Boundary { tag, out_sign, .. } => {
            (raw_av, Some((*tag, *out_sign)))
          }
        };
        let m = src.face_metrics(face);
        let inv = if area != 0.0 { 1.0 / area } else { 0.0 };
        let verts = src
          .face_world_vertices(face)
          .unwrap_or_else(|| vec![src.face_world_centroid(face)]);
        lfaces.push(LeafFace {
          owner: leaf_id,
          verts,
          world_centroid: src.face_world_centroid(face),
          comp_centroid: *src.face_centroid(face),
          area,
          area_vec,
          metrics: FaceMetrics {
            normal: area_vec * inv,
            comp_area: m.comp_area,
            phys_area: m.phys_area,
            sqrt_metric: m.sqrt_metric,
          },
          boundary,
        });
      }
    }

    let mut mesh = AdaptiveMesh {
      base,
      forest,
      epoch,
      cell_levels,
      cell_centroids,
      cell_world_centroids,
      cell_volumes,
      cell_metrics,
      cell_face_adj: vec![Vec::new(); ncells],
      face_centroids: Vec::new(),
      face_world_centroids: Vec::new(),
      face_world_vertices: Vec::new(),
      face_areas: Vec::new(),
      face_area_vectors: Vec::new(),
      face_metrics: Vec::new(),
      face_connections: Vec::new(),
      interior_face_list: Vec::new(),
      boundary_face_lists: Vec::new(),
    };

    // Bucket leaf-faces by world footprint.
    let mut buckets: HashMap<Vec<[i64; 3]>, Vec<usize>> = HashMap::new();
    for (i, lf) in lfaces.iter().enumerate() {
      buckets.entry(footprint_key(&lf.verts)).or_default().push(i);
    }

    let mut boundary_map: HashMap<BoundaryTag, Vec<(FaceId, CellId)>> =
      HashMap::new();
    let mut leftover: Vec<usize> = Vec::new();

    for idxs in buckets.values() {
      match idxs.as_slice() {
        [a, b] => {
          // Conforming interior face shared by two leaves. Owner = lower id, use
          // its outward geometry.
          let (o, n) = if lfaces[*a].owner <= lfaces[*b].owner {
            (*a, *b)
          } else {
            (*b, *a)
          };
          if lfaces[o].owner == lfaces[n].owner {
            return Err(
              AetherError::new(RefineError::UnbalancedRequest)
                .context("two coincident faces on the same cell"),
            );
          }
          mesh.push_interior(&lfaces[o], lfaces[n].owner);
        }
        [single] => {
          let lf = &lfaces[*single];
          if let Some((tag, out_sign)) = lf.boundary {
            mesh.push_boundary(lf, tag, out_sign, &mut boundary_map);
          } else {
            leftover.push(*single);
          }
        }
        _ => {
          return Err(
            AetherError::new(RefineError::UnbalancedRequest).context(
              "more than two coincident faces (level jump > 1 unsupported)",
            ),
          );
        }
      }
    }

    // Resolve hanging interfaces: each leftover coarse face meets several smaller
    // fine faces. Process largest-first so a coarse face consumes its children.
    leftover
      .sort_by(|&a, &b| lfaces[b].area.partial_cmp(&lfaces[a].area).unwrap());
    let mut consumed = vec![false; lfaces.len()];
    for ci in 0..leftover.len() {
      let c_idx = leftover[ci];
      if consumed[c_idx] {
        continue;
      }
      let c = &lfaces[c_idx];
      let c_norm = c.area_vec * (1.0 / c.area.max(f64::MIN_POSITIVE));
      let radius = c
        .verts
        .iter()
        .map(|v| dist3(v, &c.world_centroid))
        .fold(0.0, f64::max)
        * 1.001;

      let mut children = Vec::new();
      for &f_idx in &leftover {
        if f_idx == c_idx || consumed[f_idx] {
          continue;
        }
        let f = &lfaces[f_idx];
        if f.owner == c.owner || f.area >= c.area {
          continue;
        }
        let f_norm = f.area_vec * (1.0 / f.area.max(f64::MIN_POSITIVE));
        // Same physical interface ⇒ outward normals antiparallel, and the fine
        // face sits within the coarse face's footprint.
        if dot3(&c_norm, &f_norm) > -0.9 {
          continue;
        }
        if dist3(&f.world_centroid, &c.world_centroid) > radius {
          continue;
        }
        children.push(f_idx);
      }

      if !children.is_empty() {
        for f_idx in children {
          mesh.push_interior(&lfaces[f_idx], c.owner);
          consumed[f_idx] = true;
        }
        consumed[c_idx] = true;
      }
    }

    // Any leftover not consumed is an orphan — an unmatched seam face under mixed
    // refinement (the v1 limitation) or an unbalanced (> 1 level) jump.
    if let Some(&orphan) = leftover.iter().find(|&&i| !consumed[i]) {
      return Err(AetherError::new(RefineError::UnbalancedRequest).context(
        format!(
          "unmatched face on cell {} (refinement across a panel seam is \
           unsupported in v1, or the request was not 2:1-balanced)",
          lfaces[orphan].owner
        ),
      ));
    }

    mesh.boundary_face_lists = boundary_map.into_iter().collect();
    Ok(mesh)
  }

  fn push_interior(&mut self, lf: &LeafFace, neighbour: usize) {
    let fid = FaceId::from(self.face_connections.len());
    self.push_face_geometry(lf);
    self.face_connections.push(FaceConnection::Interior {
      owner: CellId::from(lf.owner),
      neighbour: CellId::from(neighbour),
    });
    self.cell_face_adj[lf.owner].push(fid);
    self.cell_face_adj[neighbour].push(fid);
    self.interior_face_list.push((
      fid,
      CellId::from(lf.owner),
      CellId::from(neighbour),
    ));
  }

  fn push_boundary(
    &mut self,
    lf: &LeafFace,
    tag: BoundaryTag,
    out_sign: f64,
    boundary_map: &mut HashMap<BoundaryTag, Vec<(FaceId, CellId)>>,
  ) {
    let fid = FaceId::from(self.face_connections.len());
    self.push_face_geometry(lf);
    self.face_connections.push(FaceConnection::Boundary {
      owner: CellId::from(lf.owner),
      tag,
      out_sign,
    });
    self.cell_face_adj[lf.owner].push(fid);
    boundary_map
      .entry(tag)
      .or_default()
      .push((fid, CellId::from(lf.owner)));
  }

  fn push_face_geometry(&mut self, lf: &LeafFace) {
    self.face_centroids.push(lf.comp_centroid);
    self.face_world_centroids.push(lf.world_centroid);
    self.face_world_vertices.push(lf.verts.clone());
    self.face_areas.push(lf.area);
    self.face_area_vectors.push(lf.area_vec);
    self.face_metrics.push(FaceMetrics {
      normal: lf.metrics.normal,
      comp_area: lf.metrics.comp_area,
      phys_area: lf.metrics.phys_area,
      sqrt_metric: lf.metrics.sqrt_metric,
    });
  }
}

impl CellGeometry<3> for AdaptiveMesh {
  fn cell_centroid(&self, cell: CellId) -> &Point<3> {
    &self.cell_centroids[cell.index()]
  }
  fn cell_world_centroid(&self, cell: CellId) -> Point<3> {
    self.cell_world_centroids[cell.index()]
  }
  fn cell_volume(&self, cell: CellId) -> f64 {
    self.cell_volumes[cell.index()]
  }
  fn cell_metrics(&self, cell: CellId) -> &CellMetrics<3> {
    &self.cell_metrics[cell.index()]
  }
  fn cell_count(&self) -> usize {
    self.cell_centroids.len()
  }
}

impl FaceGeometry<3> for AdaptiveMesh {
  fn face_centroid(&self, face: FaceId) -> &Point<3> {
    &self.face_centroids[face.index()]
  }
  fn face_world_centroid(&self, face: FaceId) -> Point<3> {
    self.face_world_centroids[face.index()]
  }
  fn face_world_vertices(&self, face: FaceId) -> Option<Vec<Point<3>>> {
    Some(self.face_world_vertices[face.index()].clone())
  }
  fn face_area_vector(&self, face: FaceId) -> Vector<f64, 3> {
    self.face_area_vectors[face.index()]
  }
  fn face_area(&self, face: FaceId) -> f64 {
    self.face_areas[face.index()]
  }
  fn face_metrics(&self, face: FaceId) -> &FaceMetrics<3> {
    &self.face_metrics[face.index()]
  }
  fn face_count(&self) -> usize {
    self.face_connections.len()
  }
}

impl Topology for AdaptiveMesh {
  fn face_connection(&self, face: FaceId) -> &FaceConnection {
    &self.face_connections[face.index()]
  }
  fn cell_faces(&self, cell: CellId) -> &[FaceId] {
    &self.cell_face_adj[cell.index()]
  }
  fn interior_faces(&self) -> &[(FaceId, CellId, CellId)] {
    &self.interior_face_list
  }
  fn boundary_faces(&self, tag: BoundaryTag) -> &[(FaceId, CellId)] {
    self
      .boundary_face_lists
      .iter()
      .find(|(t, _)| *t == tag)
      .map(|(_, l)| l.as_slice())
      .unwrap_or(&[])
  }
  fn boundary_tags(&self) -> Box<dyn Iterator<Item = BoundaryTag> + '_> {
    Box::new(self.boundary_face_lists.iter().map(|(t, _)| *t))
  }
}

impl RefinableMesh<3> for AdaptiveMesh {
  fn cell_level(&self, cell: CellId) -> u32 {
    self.cell_levels[cell.index()]
  }

  fn refine_fanout(&self) -> usize {
    self.base.refine_fanout()
  }

  fn adapt(
    &self,
    request: &AdaptRequest,
  ) -> AetherResult<(Arc<dyn Mesh<3>>, CellRemap)> {
    let (adapted, remap) = self.refine(request)?;
    Ok((Arc::new(adapted), remap))
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::cube_sphere::CubeSphere;
  use utility::domain::NewCellSource;

  const R_INNER: f64 = 6.371e6;
  const R_OUTER: f64 = 6.391e6;

  fn refine_all(n: usize) -> AdaptRequest {
    AdaptRequest {
      refine: (0..n).map(CellId::from).collect(),
      coarsen: vec![],
    }
  }

  /// Net oriented physical area vector `Σ ±(area_vec · sqrt_metric)` over a
  /// cell's faces (outward). The finite-volume invariant: this is unchanged when
  /// a neighbour refines (the replacing fine sub-faces sum to the coarse face),
  /// so it is the free-stream / conservation signature at a hanging interface.
  fn net_area<M: Mesh<3> + ?Sized>(m: &M, cell: usize) -> [f64; 3] {
    let mut s = [0.0f64; 3];
    for &face in m.cell_faces(CellId::from(cell)) {
      let av = m.face_area_vector(face);
      let sm = m.face_metrics(face).sqrt_metric;
      let sign = match m.face_connection(face) {
        FaceConnection::Interior { owner, .. } => {
          if owner.index() == cell {
            1.0
          } else {
            -1.0
          }
        }
        FaceConnection::Boundary { out_sign, .. } => *out_sign,
      };
      for k in 0..3 {
        s[k] += sign * av[k] * sm;
      }
    }
    s
  }

  fn close(a: [f64; 3], b: [f64; 3], tol: f64) -> bool {
    (0..3).all(|k| (a[k] - b[k]).abs() <= tol)
  }

  #[test]
  fn level_zero_reproduces_the_base_geometry() {
    let base = Arc::new(CubeSphere::new([4, 4, 2], R_INNER, R_OUTER));
    let mesh = AdaptiveMesh::new(base.clone());
    assert_eq!(mesh.cell_count(), base.cell_count());
    // Same interior + boundary face counts as the base topology.
    assert_eq!(mesh.interior_faces().len(), base.interior_faces().len());
    // Per-cell net oriented area vector matches the base exactly (assembly
    // reproduces the base geometry cell-for-cell).
    for c in 0..base.cell_count() {
      assert!(
        close(net_area(&mesh, c), net_area(base.as_ref(), c), 1.0),
        "assembled cell {c} net area differs from base"
      );
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

    assert_eq!(refined.cell_count(), old_count * 4);
    assert_eq!(remap.new_count(), old_count * 4);
    let new_volume: f64 = (0..refined.cell_count())
      .map(|i| refined.cell_volume(CellId::from(i)))
      .sum();
    assert!(
      (old_volume - new_volume).abs() / old_volume < 1e-9,
      "refinement changed total volume: {old_volume} vs {new_volume}"
    );
  }

  #[test]
  fn each_parent_has_four_children_under_uniform_refine() {
    let base = Arc::new(CubeSphere::new([4, 4, 1], R_INNER, R_OUTER));
    let mesh = AdaptiveMesh::new(base);
    let old_count = mesh.cell_count();
    let (_refined, remap) = mesh.adapt(&refine_all(old_count)).unwrap();
    let mut child_count = vec![0usize; old_count];
    for new_i in 0..remap.new_count() {
      if let NewCellSource::Child { parent } =
        remap.source_of(CellId::from(new_i))
      {
        child_count[parent.index()] += 1;
      }
    }
    assert!(child_count.iter().all(|&c| c == 4));
    assert_eq!(remap.died().count(), old_count);
  }

  #[test]
  fn selective_interior_refine_builds_hanging_faces() {
    // Refine a single panel-interior cell on [4,4,1]: panel 0, angular (1,1),
    // radial 0 ⇒ local 1 + 1*4 = 5. Its four lateral neighbours (cells 1, 4, 6,
    // 9) are in the same panel, so no seam is involved.
    let base = Arc::new(CubeSphere::new([4, 4, 1], R_INNER, R_OUTER));
    let mesh = AdaptiveMesh::new(base);
    let n0 = mesh.cell_count();
    let req = AdaptRequest {
      refine: vec![CellId::from(5)],
      coarsen: vec![],
    };

    // Rebuild a concrete AdaptiveMesh (adapt returns a dyn Mesh).
    let new_epoch = mesh.epoch.next();
    let (new_forest, remap) = mesh.forest.adapt(&req, mesh.epoch, new_epoch);
    let refined =
      AdaptiveMesh::build(mesh.base().clone(), new_forest, new_epoch).unwrap();

    // One cell → four children: net +3 cells, and a hanging interface exists
    // (more interior faces than the conforming level-0 mesh).
    assert_eq!(refined.cell_count(), n0 + 3);
    assert!(refined.interior_faces().len() > mesh.interior_faces().len());
    let level1 = (0..refined.cell_count())
      .filter(|&c| refined.cell_level(CellId::from(c)) == 1)
      .count();
    assert_eq!(level1, 4);

    // Conservation / free-stream at the hanging interface: each coarse neighbour
    // of the refined cell keeps the *same* net oriented area as before — its one
    // shared face was replaced by fine sub-faces that sum to it.
    for base_neighbour in [1usize, 4, 6, 9] {
      let new_id = remap.image_of(CellId::from(base_neighbour)).unwrap();
      let before = net_area(&mesh, base_neighbour);
      let after = net_area(&refined, new_id.index());
      // Scale: total physical face area of the cell. The hanging sub-faces must
      // sum to the coarse face they replaced to within the cube-sphere's own
      // level-to-level metric tolerance (a small fraction of a face).
      let scale: f64 = mesh
        .cell_faces(CellId::from(base_neighbour))
        .iter()
        .map(|&f| mesh.face_area(f) * mesh.face_metrics(f).sqrt_metric)
        .sum();
      let diff = ((after[0] - before[0]).powi(2)
        + (after[1] - before[1]).powi(2)
        + (after[2] - before[2]).powi(2))
      .sqrt();
      assert!(
        diff < 1e-3 * scale,
        "coarse neighbour {base_neighbour}: hanging sub-faces don't sum to the \
         replaced face — Δ={diff:.3e}, scale={scale:.3e}"
      );
    }
  }

  #[test]
  fn multi_level_jump_is_handled_conservatively() {
    // Refine cell 5 (→ level-1 children at ids 5..=8), then refine one of those
    // children to level 2 while its base-cell neighbours stay level 0. The coarse
    // neighbour's shared face now meets a *mix* of level-1 and level-2 fine faces;
    // the sub-face representation still tiles it, so this builds and conserves —
    // the mesh handles arbitrary level jumps (2:1 is the driver's accuracy policy,
    // not a requirement of the geometry).
    let base = Arc::new(CubeSphere::new([4, 4, 1], R_INNER, R_OUTER));
    let mesh = AdaptiveMesh::new(base);
    let e0 = mesh.epoch;
    let (forest1, _) = mesh.forest.adapt(
      &AdaptRequest {
        refine: vec![CellId::from(5)],
        coarsen: vec![],
      },
      e0,
      e0.next(),
    );
    let level1 =
      AdaptiveMesh::build(mesh.base().clone(), forest1, e0.next()).unwrap();
    assert_eq!(level1.cell_level(CellId::from(5)), 1);

    let e1 = level1.epoch;
    let req = AdaptRequest {
      refine: vec![CellId::from(5)],
      coarsen: vec![],
    };
    let (forest2, remap) = level1.forest.adapt(&req, e1, e1.next());
    let level2 =
      AdaptiveMesh::build(level1.base().clone(), forest2, e1.next()).unwrap();
    // A level-2 cell now exists next to level-0 base neighbours.
    assert!(
      (0..level2.cell_count()).any(|c| level2.cell_level(CellId::from(c)) == 2)
    );

    // Base cell 4 (level 0, a neighbour of base cell 5) still conserves its net
    // oriented area across the mixed-level hanging interface.
    let new4 = remap.image_of(CellId::from(4)).unwrap();
    let before = net_area(&level1, 4);
    let after = net_area(&level2, new4.index());
    let scale: f64 = level1
      .cell_faces(CellId::from(4))
      .iter()
      .map(|&f| level1.face_area(f) * level1.face_metrics(f).sqrt_metric)
      .sum();
    let diff = ((after[0] - before[0]).powi(2)
      + (after[1] - before[1]).powi(2)
      + (after[2] - before[2]).powi(2))
    .sqrt();
    assert!(
      diff < 1e-3 * scale,
      "mixed-level interface not conservative"
    );
  }
}
