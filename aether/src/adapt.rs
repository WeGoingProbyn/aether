// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Adaptive mesh refinement driver — the criterion vocabulary and the per-mesh
//! adapter the [`World`](crate::core::World) runs at the end-of-tick barrier.
//!
//! The barrier itself lives in [`World::tick`](crate::core::World::tick): on a
//! successful tick it asks each [`MeshAdapter`]'s [`RefinementCriterion`] what to
//! adapt, balances the request ([`balance_2to1`]), refines the mesh
//! ([`AdaptiveMesh::refine`]), conservatively remaps every field on that mesh
//! ([`Pleroma::remap_mesh_fields`](pleroma::Pleroma::remap_mesh_fields)), swaps
//! the new mesh into the [`Tessera`](tessera::world_mesh::Tessera) with a bumped
//! [`TopologyEpoch`](utility::domain::TopologyEpoch), and emits
//! [`Event::TopologyChanged`](utility::events::Event::TopologyChanged) for the
//! read-side consumers (query / render / checkpoint).
//!
//! **Direction note.** A criterion reads world state through `&Pleroma`. A physics
//! criterion reads a field; [`FocusLodCriterion`] reads an *input resource* the
//! host writes into pleroma ([`ResourceKey::RefinementFocus`]) — never the
//! outbound [`EventBus`](utility::events::EventBus), which is sim→consumer only.
//!
//! Nothing in this module knows about rendering. A host that wants
//! view-dependent LOD converts its camera into a [`RefinementFocus`] and writes
//! it in; the view it *presents* is its own concern and never round-trips
//! through the simulation.

use std::sync::Arc;

use pleroma::Pleroma;
use pleroma::core::storage::{FieldStorage, SoaField};
use tessera::adaptive::AdaptiveMesh;
use tessera::refine::{RefinableMesh, RefineFlags};
use utility::domain::{CellId, FieldKey, MeshKey};
use utility::error::{AetherError, AetherResult, ErrorDomain};

#[derive(Debug)]
pub enum AdaptError {
  /// The criterion's field is not registered.
  MissingField,
  /// The field length disagrees with the mesh cell count.
  FieldLengthMismatch,
}

impl ErrorDomain for AdaptError {
  fn domain(&self) -> &str {
    "aether adapt"
  }
}

impl std::fmt::Display for AdaptError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      AdaptError::MissingField => {
        write!(f, "refinement criterion field missing")
      }
      AdaptError::FieldLengthMismatch => {
        write!(f, "criterion field length != mesh cell count")
      }
    }
  }
}

/// Decides which cells should refine / coarsen from the world's state. The driver
/// balances and applies the result, so a criterion need not return a
/// 2:1-balanced set. Mesh- and consumer-agnostic: reads state through `&Pleroma`.
pub trait RefinementCriterion: Send + Sync {
  fn evaluate(
    &self,
    mesh: &dyn RefinableMesh<3>,
    pleroma: &Pleroma,
  ) -> AetherResult<RefineFlags>;
}

/// Refine where a scalar component of a field varies sharply between neighbours,
/// coarsen where it is flat. The two thresholds give hysteresis (set
/// `coarsen_below < refine_above`), and `max_level` caps refinement depth.
///
/// `N` is the field's storage width; `component` selects the scalar (e.g. density
/// is component 0 of a `SoaField<6>` Euler state).
pub struct GradientCriterion<const N: usize> {
  field: FieldKey,
  component: usize,
  refine_above: f64,
  coarsen_below: f64,
  max_level: u32,
}

impl<const N: usize> GradientCriterion<N> {
  pub fn new(
    field: FieldKey,
    component: usize,
    refine_above: f64,
    coarsen_below: f64,
    max_level: u32,
  ) -> Self {
    Self {
      field,
      component,
      refine_above,
      coarsen_below,
      max_level,
    }
  }
}

impl<const N: usize> RefinementCriterion for GradientCriterion<N> {
  fn evaluate(
    &self,
    mesh: &dyn RefinableMesh<3>,
    pleroma: &Pleroma,
  ) -> AetherResult<RefineFlags> {
    let field: &SoaField<N> = pleroma.read(self.field).ok_or_else(|| {
      AetherError::new(AdaptError::MissingField)
        .context(format!("{:?}", self.field))
    })?;
    let n = mesh.cell_count();
    if field.len() != n {
      return Err(AetherError::new(AdaptError::FieldLengthMismatch));
    }
    let val = |c: CellId| field.state(c)[self.component];

    // Per-cell indicator: the largest jump to a face neighbour.
    let mut indicator = vec![0.0f64; n];
    for &(_face, a, b) in mesh.interior_faces() {
      let d = (val(a) - val(b)).abs();
      indicator[a.index()] = indicator[a.index()].max(d);
      indicator[b.index()] = indicator[b.index()].max(d);
    }

    let mut flags = RefineFlags::default();
    for c in 0..n {
      let cell = CellId::from(c);
      let level = mesh.cell_level(cell);
      if indicator[c] > self.refine_above && level < self.max_level {
        flags.refine.push(cell);
      } else if indicator[c] < self.coarsen_below && level > 0 {
        flags.coarsen.push(cell);
      }
    }
    Ok(flags)
  }
}

/// Refine cells inside a spherical cap (within `inner_angle` of a world-space
/// direction) up to `max_level`, and coarsen cells outside a slightly wider cap
/// (`outer_angle > inner_angle`, for hysteresis). Field-free — it reads only cell
/// centroids — so it is the simplest way to drive a *localised* refinement, e.g.
/// a region of interest. Keep the cap inside one cube-sphere panel to avoid the
/// v1 seam limitation.
pub struct RegionRefinementCriterion {
  /// World-space cap centre (unit vector).
  center: [f64; 3],
  cos_inner: f64,
  cos_outer: f64,
  max_level: u32,
}

impl RegionRefinementCriterion {
  /// `center` is a world-space direction (normalised internally); `inner_angle`
  /// and `outer_angle` are the cap half-angles in radians (`outer > inner`).
  pub fn new(
    center: [f64; 3],
    inner_angle: f64,
    outer_angle: f64,
    max_level: u32,
  ) -> Self {
    let n = (center[0].powi(2) + center[1].powi(2) + center[2].powi(2)).sqrt();
    let n = if n > 0.0 { n } else { 1.0 };
    Self {
      center: [center[0] / n, center[1] / n, center[2] / n],
      cos_inner: inner_angle.cos(),
      cos_outer: outer_angle.cos(),
      max_level,
    }
  }
}

impl RefinementCriterion for RegionRefinementCriterion {
  fn evaluate(
    &self,
    mesh: &dyn RefinableMesh<3>,
    _pleroma: &Pleroma,
  ) -> AetherResult<RefineFlags> {
    let mut flags = RefineFlags::default();
    for c in 0..mesh.cell_count() {
      let cell = CellId::from(c);
      let p = mesh.cell_world_centroid(cell);
      let n = (p[0].powi(2) + p[1].powi(2) + p[2].powi(2)).sqrt();
      if n == 0.0 {
        continue;
      }
      let cos =
        (p[0] * self.center[0] + p[1] * self.center[1] + p[2] * self.center[2])
          / n;
      let level = mesh.cell_level(cell);
      if cos >= self.cos_inner && level < self.max_level {
        flags.refine.push(cell);
      } else if cos < self.cos_outer && level > 0 {
        flags.coarsen.push(cell);
      }
    }
    Ok(flags)
  }
}

/// The host's region of interest, read by [`FocusLodCriterion`]. Defined in
/// `utility` so any host can write it without depending on aether, and
/// re-exported here for the criterion +
/// [`World::set_refinement_focus`](crate::core::World::set_refinement_focus).
pub use utility::domain::RefinementFocus;

/// Focus-driven level-of-detail: refine cells that subtend a large angle from
/// the [`RefinementFocus`] and coarsen those that subtend a small one, so detail
/// tracks the host's region of interest. The indicator is a projected-size proxy
/// — a cell's characteristic size (`cell_volume^(1/3)`) over its distance to the
/// focus — so refinement is *graded* (closer ⇒ more levels) rather than a single
/// shell.
///
/// `refine_above` / `coarsen_below` are thresholds on that size/distance ratio
/// (`coarsen_below < refine_above` gives hysteresis), and `max_level` caps depth.
/// Reads [`ResourceKey::RefinementFocus`]; if the host hasn't written one yet, it
/// is a no-op (the mesh is left unchanged) — so a run with no focus simply never
/// focus-refines.
///
/// Note this is a *data-plane* criterion: it changes the discretisation the
/// solver integrates. A host driving it from a camera gets view-dependent AMR,
/// but nothing here knows what a camera is.
pub struct FocusLodCriterion {
  refine_above: f64,
  coarsen_below: f64,
  max_level: u32,
}

impl FocusLodCriterion {
  pub fn new(refine_above: f64, coarsen_below: f64, max_level: u32) -> Self {
    Self {
      refine_above,
      coarsen_below,
      max_level,
    }
  }
}

impl RefinementCriterion for FocusLodCriterion {
  fn evaluate(
    &self,
    mesh: &dyn RefinableMesh<3>,
    pleroma: &Pleroma,
  ) -> AetherResult<RefineFlags> {
    // No focus yet ⇒ nothing to do (headless / pre-first-frame).
    let Some(focus) = pleroma.read_resource::<RefinementFocus>(
      utility::domain::ResourceKey::RefinementFocus,
    ) else {
      return Ok(RefineFlags::default());
    };
    let eye = focus.position;

    let mut flags = RefineFlags::default();
    for c in 0..mesh.cell_count() {
      let cell = CellId::from(c);
      let p = mesh.cell_world_centroid(cell);
      let d = ((p[0] - eye[0]).powi(2)
        + (p[1] - eye[1]).powi(2)
        + (p[2] - eye[2]).powi(2))
      .sqrt()
      .max(f64::EPSILON);
      let size = mesh.cell_volume(cell).max(0.0).cbrt();
      let projected = size / d;
      let level = mesh.cell_level(cell);
      if projected > self.refine_above && level < self.max_level {
        flags.refine.push(cell);
      } else if projected < self.coarsen_below && level > 0 {
        flags.coarsen.push(cell);
      }
    }
    Ok(flags)
  }
}

/// Bounds how often and how much the mesh adapts, so the (full) re-mesh + field
/// remap cost cannot dominate: adapt only every `every_n_ticks` ticks, and change
/// at most `max_refine` / `max_coarsen` cells per adapt.
#[derive(Clone, Copy, Debug)]
pub struct AdaptGovernor {
  pub every_n_ticks: u64,
  pub max_refine: usize,
  pub max_coarsen: usize,
}

impl Default for AdaptGovernor {
  fn default() -> Self {
    Self {
      every_n_ticks: 1,
      max_refine: usize::MAX,
      max_coarsen: usize::MAX,
    }
  }
}

impl AdaptGovernor {
  pub fn new(
    every_n_ticks: u64,
    max_refine: usize,
    max_coarsen: usize,
  ) -> Self {
    Self {
      every_n_ticks: every_n_ticks.max(1),
      max_refine,
      max_coarsen,
    }
  }

  /// Whether the barrier should evaluate this adapter on `tick_count`.
  pub fn fires_on(&self, tick_count: u64) -> bool {
    tick_count % self.every_n_ticks == 0
  }

  /// Truncate a request to the per-adapt churn caps.
  pub fn cap(&self, mut flags: RefineFlags) -> RefineFlags {
    flags.refine.truncate(self.max_refine);
    flags.coarsen.truncate(self.max_coarsen);
    flags
  }
}

/// One adaptive mesh under the driver's control: which mesh, the live
/// [`AdaptiveMesh`] (kept typed so it can keep refining), its criterion, and its
/// governor.
pub struct MeshAdapter {
  pub(crate) mesh_key: MeshKey,
  pub(crate) mesh: Arc<AdaptiveMesh>,
  pub(crate) criterion: Box<dyn RefinementCriterion>,
  pub(crate) governor: AdaptGovernor,
}

impl MeshAdapter {
  pub fn new(
    mesh_key: MeshKey,
    mesh: Arc<AdaptiveMesh>,
    criterion: Box<dyn RefinementCriterion>,
    governor: AdaptGovernor,
  ) -> Self {
    Self {
      mesh_key,
      mesh,
      criterion,
      governor,
    }
  }

  pub fn mesh_key(&self) -> MeshKey {
    self.mesh_key
  }

  pub fn mesh(&self) -> &Arc<AdaptiveMesh> {
    &self.mesh
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn governor_cadence_and_churn_caps() {
    let g = AdaptGovernor::new(3, 2, 1);
    // Fires every 3rd tick.
    assert!(g.fires_on(0));
    assert!(!g.fires_on(1));
    assert!(!g.fires_on(2));
    assert!(g.fires_on(3));

    // Churn caps truncate the request.
    let flags = RefineFlags {
      refine: (0..5).map(CellId::from).collect(),
      coarsen: (0..4).map(CellId::from).collect(),
    };
    let capped = g.cap(flags);
    assert_eq!(capped.refine.len(), 2);
    assert_eq!(capped.coarsen.len(), 1);
  }

  #[test]
  fn zero_cadence_is_clamped_to_one() {
    assert_eq!(AdaptGovernor::new(0, 1, 1).every_n_ticks, 1);
  }

  #[test]
  fn focus_lod_refines_the_cells_nearest_the_focus() {
    use tessera::cube_sphere::CubeSphere;
    use tessera::geometry::CellGeometry;
    use utility::domain::ResourceKey;

    let mesh =
      AdaptiveMesh::new(Arc::new(CubeSphere::new([8, 8, 1], 0.99, 1.0)));
    let eye = [0.0, 0.0, 1.5]; // just beyond the +z pole
    let mut pleroma = Pleroma::new();
    pleroma.register_resource(
      ResourceKey::RefinementFocus,
      RefinementFocus { position: eye },
    );

    let dist = |c: CellId| {
      let p = mesh.cell_world_centroid(c);
      ((p[0] - eye[0]).powi(2)
        + (p[1] - eye[1]).powi(2)
        + (p[2] - eye[2]).powi(2))
      .sqrt()
    };
    // Pick a threshold at the 75th percentile of the size/distance indicator so
    // roughly the nearest quarter refine — robust to the cube-sphere's per-cell
    // volume variation.
    let mut projected: Vec<f64> = (0..mesh.cell_count())
      .map(|c| {
        let cell = CellId::from(c);
        mesh.cell_volume(cell).max(0.0).cbrt() / dist(cell)
      })
      .collect();
    projected.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let threshold = projected[projected.len() * 3 / 4];

    let criterion = FocusLodCriterion::new(threshold, threshold * 0.1, 1);
    let flags = criterion.evaluate(&mesh, &pleroma).unwrap();

    assert!(!flags.refine.is_empty(), "some cells should view-refine");
    assert!(
      flags.refine.len() < mesh.cell_count(),
      "not every cell should refine"
    );
    // The refined cells are nearer the eye than the mesh as a whole.
    let mean_all: f64 = (0..mesh.cell_count())
      .map(|c| dist(CellId::from(c)))
      .sum::<f64>()
      / mesh.cell_count() as f64;
    let mean_refined: f64 = flags.refine.iter().map(|&c| dist(c)).sum::<f64>()
      / flags.refine.len() as f64;
    assert!(
      mean_refined < mean_all,
      "refined cells ({mean_refined:.3}) should sit nearer the eye than \
       average ({mean_all:.3})"
    );
  }
}
