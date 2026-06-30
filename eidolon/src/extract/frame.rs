// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use tessera::world_mesh::Tessera;
use utility::domain::WorldId;

use crate::{
  extract::{
    coupler_debug::coupler_face_lines,
    mesh::{boundary_surface_triangles, cell_centroid_points},
  },
  ir::{RenderFrame, RenderWorld, Transform},
};

pub fn tessera_debug_frame(
  frame: u64,
  sim_time: f64,
  world: WorldId,
  tessera: &Tessera,
) -> RenderFrame {
  RenderFrame {
    frame,
    sim_time,
    worlds: vec![tessera_debug_world(world, tessera)],
    camera: None,
  }
}

pub fn tessera_debug_world(world: WorldId, tessera: &Tessera) -> RenderWorld {
  let mut meshes = tessera
    .meshes()
    .flat_map(|(mesh_key, mesh)| {
      [
        boundary_surface_triangles(world, mesh_key, mesh.as_ref()),
        cell_centroid_points(world, mesh_key, mesh.as_ref()),
      ]
    })
    .collect::<Vec<_>>();

  for (index, entry) in tessera.couplers().iter().enumerate() {
    if let Some(lines) = coupler_face_lines(world, index, tessera, entry) {
      meshes.push(lines);
    }
  }

  RenderWorld {
    id: world,
    label: format!("world_{}", world.0),
    transform: Transform::IDENTITY,
    transform_epoch: 0,
    meshes,
    layers: Vec::new(),
    diagnostics: Vec::new(),
  }
}
