// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashMap, sync::Arc};

use utility::domain::MeshKey;

use crate::{coupling::MeshCoupler, mesh::Mesh};

pub struct LayeredMesh {
  meshes: HashMap<MeshKey, Arc<dyn Mesh<3>>>,
  couplers: Vec<(MeshKey, MeshKey, Box<dyn MeshCoupler>)> 
}

