// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use std::{any::Any, collections::HashMap, sync::Arc};

use utility::domain::{MeshKey, TopologyEpoch};
use utility::error::{AetherError, AetherResult};

use crate::{
  coupling::{CoupledFace, MeshCoupler},
  geo::GeoCoord,
  mask::{CellMask, MaskError},
  mesh::Mesh,
  partition::Decomposition,
};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DecompositionKey(&'static str);

impl DecompositionKey {
  pub const DEFAULT: DecompositionKey = DecompositionKey("default");

  pub const fn new(name: &'static str) -> Self {
    Self(name)
  }

  pub const fn name(self) -> &'static str {
    self.0
  }
}

struct MeshEntry {
  mesh: Arc<dyn Mesh<3>>,
  decompositions: HashMap<DecompositionKey, Box<dyn Any + Send + Sync>>,
}

impl MeshEntry {
  fn new(mesh: Arc<dyn Mesh<3>>) -> Self {
    Self {
      mesh,
      decompositions: HashMap::new(),
    }
  }
}

pub struct CouplerEntry {
  mesh_a: MeshKey,
  mesh_b: MeshKey,
  coupler: Box<dyn MeshCoupler>,
}

impl CouplerEntry {
  pub fn new(
    mesh_a: MeshKey,
    mesh_b: MeshKey,
    coupler: Box<dyn MeshCoupler>,
  ) -> Self {
    Self {
      mesh_a,
      mesh_b,
      coupler,
    }
  }

  pub fn mesh_a(&self) -> MeshKey {
    self.mesh_a
  }

  pub fn mesh_b(&self) -> MeshKey {
    self.mesh_b
  }

  pub fn coupler(&self) -> &dyn MeshCoupler {
    self.coupler.as_ref()
  }
}

pub struct CouplerView<'a> {
  entry: &'a CouplerEntry,
  mesh_a: &'a dyn Mesh<3>,
  mesh_b: &'a dyn Mesh<3>,
}

impl<'a> CouplerView<'a> {
  pub fn mesh_a(&self) -> MeshKey {
    self.entry.mesh_a
  }

  pub fn mesh_b(&self) -> MeshKey {
    self.entry.mesh_b
  }

  pub fn pair_count(&self) -> usize {
    self.entry.coupler.pairs().len()
  }

  pub fn faces(&self) -> impl Iterator<Item = CoupledFace> + '_ {
    self.entry.coupler.pairs().iter().map(|pair| {
      CoupledFace::from_pair(
        self.entry.mesh_a,
        self.mesh_a,
        self.entry.mesh_b,
        self.mesh_b,
        *pair,
      )
    })
  }
}

#[derive(Default)]
pub struct Tessera {
  meshes: HashMap<MeshKey, MeshEntry>,
  couplers: Vec<CouplerEntry>,
  /// Per-mesh cell-activity masks (which cells are really part of the domain).
  /// Built at setup alongside couplers; absent ⇒ all cells active.
  masks: HashMap<MeshKey, CellMask>,
  /// Per-mesh topology version. Bumped once each time a mesh is adapted
  /// (refined / coarsened); absent ⇒ [`TopologyEpoch::ZERO`] (an unadapted base
  /// mesh). Consumers read this to detect that a mesh's `CellId` space changed.
  epochs: HashMap<MeshKey, TopologyEpoch>,
}

impl Tessera {
  pub fn new() -> Self {
    Self::default()
  }

  pub fn register_mesh(
    &mut self,
    key: MeshKey,
    mesh: Arc<dyn Mesh<3>>,
  ) -> Option<Arc<dyn Mesh<3>>> {
    self
      .meshes
      .insert(key, MeshEntry::new(mesh))
      .map(|entry| entry.mesh)
  }

  pub fn mesh(&self, key: MeshKey) -> Option<&Arc<dyn Mesh<3>>> {
    self.meshes.get(&key).map(|entry| &entry.mesh)
  }

  pub fn contains_mesh(&self, key: MeshKey) -> bool {
    self.meshes.contains_key(&key)
  }

  pub fn meshes(&self) -> impl Iterator<Item = (MeshKey, &Arc<dyn Mesh<3>>)> {
    self.meshes.iter().map(|(&key, entry)| (key, &entry.mesh))
  }

  /// The current topology epoch of `mesh` ([`TopologyEpoch::ZERO`] for an
  /// unadapted base mesh, or one never registered).
  pub fn topology_epoch(&self, mesh: MeshKey) -> TopologyEpoch {
    self
      .epochs
      .get(&mesh)
      .copied()
      .unwrap_or(TopologyEpoch::ZERO)
  }

  /// Record a mesh's topology epoch (set by the adapt barrier after a re-mesh).
  pub fn set_topology_epoch(&mut self, mesh: MeshKey, epoch: TopologyEpoch) {
    self.epochs.insert(mesh, epoch);
  }

  /// Store a [`CellMask`] for `mesh`, replacing any previous one.
  pub fn set_cell_mask(&mut self, mesh: MeshKey, mask: CellMask) {
    self.masks.insert(mesh, mask);
  }

  /// The cell-activity mask for `mesh`, if one was built. `None` means *no
  /// masking* — every cell is active (the backward-compatible default). Consumers
  /// (e.g. the ocean solver) treat absence as all-active by design; the "forgot to
  /// build an intended mask" footgun is guarded at assembly/test time, not here.
  pub fn cell_mask(&self, mesh: MeshKey) -> Option<&CellMask> {
    self.masks.get(&mesh)
  }

  /// Build and store a geographic [`CellMask`] for `mesh`: cell `i` is active iff
  /// `active(geo_of_cell_i)`. Cells are visited in **global [`CellId`] order** and
  /// their geographic coordinate is derived the same way terrain/the geo-index do
  /// (`cell_world_centroid` → [`GeoCoord::from_world`]). tessera stays
  /// domain-agnostic: the predicate is over [`GeoCoord`], so the caller encodes the
  /// classification (e.g. "ocean ⇒ active"). Errors if `mesh` is not registered.
  ///
  /// Call this at setup, after the mesh is registered and before the world is built
  /// for ticking.
  pub fn build_geographic_cell_mask(
    &mut self,
    mesh: MeshKey,
    surface_radius: f64,
    active: impl Fn(GeoCoord) -> bool,
  ) -> AetherResult<()> {
    let geometry = self
      .mesh(mesh)
      .ok_or_else(|| AetherError::new(MaskError::MeshNotRegistered))?;
    let n = geometry.cell_count();
    let mask = CellMask::from_fn(n, |cell| {
      let pos = geometry.cell_world_centroid(cell);
      active(GeoCoord::from_world(&pos, surface_radius))
    });
    self.masks.insert(mesh, mask);
    Ok(())
  }

  pub fn register_decomposition<M>(
    &mut self,
    mesh: MeshKey,
    key: DecompositionKey,
    decomposition: Decomposition<3, M>,
  ) -> Option<Decomposition<3, M>>
  where
    M: Mesh<3> + 'static,
  {
    let entry = self.meshes.get_mut(&mesh)?;
    entry
      .decompositions
      .insert(key, Box::new(decomposition))
      .and_then(|previous| {
        previous
          .downcast::<Decomposition<3, M>>()
          .ok()
          .map(|boxed| *boxed)
      })
  }

  pub fn decomposition<M>(
    &self,
    mesh: MeshKey,
    key: DecompositionKey,
  ) -> Option<&Decomposition<3, M>>
  where
    M: Mesh<3> + 'static,
  {
    self
      .meshes
      .get(&mesh)?
      .decompositions
      .get(&key)?
      .downcast_ref::<Decomposition<3, M>>()
  }

  pub fn contains_decomposition(
    &self,
    mesh: MeshKey,
    key: DecompositionKey,
  ) -> bool {
    self
      .meshes
      .get(&mesh)
      .is_some_and(|entry| entry.decompositions.contains_key(&key))
  }

  pub fn add_coupler(
    &mut self,
    mesh_a: MeshKey,
    mesh_b: MeshKey,
    coupler: impl MeshCoupler + 'static,
  ) -> usize {
    let id = self.couplers.len();
    self
      .couplers
      .push(CouplerEntry::new(mesh_a, mesh_b, Box::new(coupler)));
    id
  }

  pub fn couplers(&self) -> &[CouplerEntry] {
    &self.couplers
  }

  pub fn coupler_view(&self, index: usize) -> Option<CouplerView<'_>> {
    let entry = self.couplers.get(index)?;
    let mesh_a = self.mesh(entry.mesh_a)?.as_ref();
    let mesh_b = self.mesh(entry.mesh_b)?.as_ref();
    Some(CouplerView {
      entry,
      mesh_a,
      mesh_b,
    })
  }

  pub fn couplers_between(
    &self,
    mesh_a: MeshKey,
    mesh_b: MeshKey,
  ) -> impl Iterator<Item = &CouplerEntry> {
    self.couplers.iter().filter(move |entry| {
      (entry.mesh_a == mesh_a && entry.mesh_b == mesh_b)
        || (entry.mesh_a == mesh_b && entry.mesh_b == mesh_a)
    })
  }
}

#[cfg(test)]
mod mask_tests {
  use std::collections::HashMap;

  use utility::domain::{CellId, MeshKey};

  use super::Tessera;
  use crate::cube_sphere::CubeSphere;
  use crate::geo::GeoCoord;

  const R_INNER: f64 = 6.371e6;
  const R_OUTER: f64 = 6.391e6;
  const SURFACE: f64 = R_INNER;

  #[test]
  fn geographic_mask_is_per_cell_correct_and_column_consistent() {
    let mesh = CubeSphere::new([8, 8, 3], R_INNER, R_OUTER);
    let mut tessera = Tessera::new();
    tessera.register_mesh(MeshKey::OCEAN, std::sync::Arc::new(mesh));

    // Northern hemisphere active — an altitude-independent (lat-only) predicate.
    tessera
      .build_geographic_cell_mask(MeshKey::OCEAN, SURFACE, |g| g.lat > 0.0)
      .unwrap();
    let mesh = tessera.mesh(MeshKey::OCEAN).unwrap().clone();
    let mask = tessera.cell_mask(MeshKey::OCEAN).unwrap();

    // Non-trivial split.
    assert!(mask.active_count() > 0 && mask.inactive_count() > 0);
    assert_eq!(mask.len(), mesh.cell_count());

    // Per-cell correctness + column consistency: cells sharing a lat/lon column
    // (different radial layers) must share their mask bit.
    let mut columns: HashMap<(i64, i64), bool> = HashMap::new();
    for i in 0..mesh.cell_count() {
      let cell = CellId::from(i);
      let geo = GeoCoord::from_world(&mesh.cell_world_centroid(cell), SURFACE);
      assert_eq!(mask.is_active(cell), geo.lat > 0.0, "cell {i} mask wrong");

      let key = ((geo.lat * 1e6) as i64, (geo.lon * 1e6) as i64);
      match columns.get(&key) {
        Some(&bit) => {
          assert_eq!(bit, mask.is_active(cell), "column {key:?} mask split")
        }
        None => {
          columns.insert(key, mask.is_active(cell));
        }
      }
    }
    // The mask covered whole columns, so active cells are a multiple of the
    // radial-layer count (3).
    assert_eq!(mask.active_count() % 3, 0);
  }

  #[test]
  fn building_a_mask_for_an_unregistered_mesh_errors() {
    let mut tessera = Tessera::new();
    assert!(
      tessera
        .build_geographic_cell_mask(MeshKey::OCEAN, SURFACE, |_| true)
        .is_err()
    );
  }
}
