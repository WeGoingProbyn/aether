// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use std::{any::Any, collections::HashMap, sync::Arc};

use utility::domain::MeshKey;

use crate::{coupling::MeshCoupler, mesh::Mesh, partition::Decomposition};

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

#[derive(Default)]
pub struct Tessera {
  meshes: HashMap<MeshKey, MeshEntry>,
  couplers: Vec<CouplerEntry>,
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
