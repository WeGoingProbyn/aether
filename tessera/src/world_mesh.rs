// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashMap, sync::Arc};

use crate::{coupling::MeshCoupler, mesh::Mesh};

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum MeshType {
  Atmosphere,
  Surface,
  Mantle,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct MeshKey(MeshType);

pub struct LayeredMesh {
  meshes: HashMap<MeshKey, Arc<dyn Mesh<3>>>,
  couplers: Vec<(MeshKey, MeshKey, Box<dyn MeshCoupler>)> 
}

