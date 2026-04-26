// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashMap, sync::Arc};

use tessera::mesh::Mesh;
use utility::domain::{FieldKey, MeshKey};

use crate::core::access::ScheduleAccess;
use crate::core::storage::FieldStorage;
use crate::runtime::slot::FieldSlot;

pub struct Pleroma {
  meshes: HashMap<MeshKey, Arc<dyn Mesh<3>>>,
  fields: HashMap<FieldKey, FieldSlot>,
}

impl Default for Pleroma {
  fn default() -> Self {
    Self::new()
  }
}

impl Pleroma {
  pub fn new() -> Self {
    Pleroma {
      meshes: HashMap::new(),
      fields: HashMap::new(),
    }
  }

  pub fn register_mesh(
    &mut self,
    key: MeshKey,
    mesh: Arc<dyn Mesh<3>>,
  ) {
    self.meshes.insert(key, mesh);
  }

  pub fn register_field<S, const N: usize>(
    &mut self,
    _key: FieldKey,
    _init: S,
  ) where
    S: FieldStorage<N> + 'static,
  {
    unimplemented!("Pleroma::register_field is awaiting runtime impl")
  }

  // Single-stage / test direct access (safe; takes &mut self).
  pub fn read<S: 'static>(&self, _key: FieldKey) -> Option<&S> {
    unimplemented!("Pleroma::read is awaiting runtime impl")
  }

  pub fn write<S: 'static>(&mut self, _key: FieldKey) -> Option<&mut S> {
    unimplemented!("Pleroma::write is awaiting runtime impl")
  }

  /// Nexus entry-point. Hands out a `ScheduleAccess` for one DAG layer.
  /// Nexus then calls `unsafe ScheduleAccess::view_for` once per parallel
  /// stage with that stage's declared reads/writes. The unsafe split is
  /// sound because the schedule has already verified non-overlap at the
  /// layer level.
  pub fn schedule_access(&mut self) -> ScheduleAccess<'_> {
    unimplemented!("Pleroma::schedule_access is awaiting runtime impl")
  }

  pub fn meshes(&self) -> &HashMap<MeshKey, Arc<dyn Mesh<3>>> {
    &self.meshes
  }
}
