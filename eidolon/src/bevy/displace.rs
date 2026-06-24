// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! `displace_meshes_system` — applies terrain relief in the reference
//! renderer. When a mesh has a displacement directive
//! ([`crate::ir::Update::SetMeshDisplacement`]), this offsets its vertices
//! radially by the driving layer's per-cell samples (e.g. surface elevation)
//! and writes the result into `Mesh::ATTRIBUTE_POSITION`.
//!
//! Displacement is recomputed from the stored *base* positions, so repeated
//! applies never compound. Only meshes flagged in
//! `RenderRegistry::dirty_displacements` are touched, so a static heightfield
//! costs one apply at registration and nothing thereafter.

use bevy::prelude::*;
use utility::profile;

use crate::ir::{LayerSamples, ScalarSamples};
use crate::render_ops::radial_displaced_positions;

use super::registry::RenderRegistry;

#[profile]
pub fn displace_meshes_system(
  mut registry: ResMut<RenderRegistry>,
  mut meshes: ResMut<Assets<Mesh>>,
) {
  if registry.dirty_displacements.is_empty() {
    return;
  }
  let dirty: Vec<_> = registry.dirty_displacements.drain().collect();
  for mesh_handle in dirty {
    let Some(binding) = registry.displacements.get(&mesh_handle).copied()
    else {
      continue;
    };
    let Some(mesh_entry) = registry.meshes.get(&mesh_handle) else {
      continue;
    };
    if mesh_entry.base_positions.is_empty() {
      continue;
    }
    let Some(layer_entry) = registry.layers.get(&binding.layer) else {
      continue;
    };
    let Some(LayerSamples::Scalar(ScalarSamples::PerCell(samples))) =
      layer_entry.samples.as_ref()
    else {
      continue;
    };

    // When relief is switched off, restore the flat base geometry instead.
    let positions = if registry.displacement_enabled {
      radial_displaced_positions(
        &mesh_entry.base_positions,
        &mesh_entry.vertex_to_cell,
        samples,
        binding.scale,
      )
    } else {
      mesh_entry.base_positions.clone()
    };

    if let Some(asset) = meshes.get_mut(&mesh_entry.mesh_handle) {
      asset.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    }
  }
}
