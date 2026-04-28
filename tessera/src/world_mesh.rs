// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashMap, sync::Arc};

use utility::domain::MeshKey;

use crate::{coupling::MeshCoupler, mesh::Mesh};

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
  meshes: HashMap<MeshKey, Arc<dyn Mesh<3>>>,
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
    self.meshes.insert(key, mesh)
  }

  pub fn mesh(&self, key: MeshKey) -> Option<&Arc<dyn Mesh<3>>> {
    self.meshes.get(&key)
  }

  pub fn contains_mesh(&self, key: MeshKey) -> bool {
    self.meshes.contains_key(&key)
  }

  pub fn meshes(&self) -> &HashMap<MeshKey, Arc<dyn Mesh<3>>> {
    &self.meshes
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
