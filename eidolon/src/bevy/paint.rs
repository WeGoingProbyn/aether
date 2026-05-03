// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! `paint_layers` system — converts the bound layer's per-cell scalar
//! samples into per-vertex colours and writes them into
//! `Mesh::ATTRIBUTE_COLOR`.
//!
//! Only meshes flagged in `RenderRegistry::dirty_meshes` are
//! repainted, so a tick that bumps just a sample buffer doesn't pay
//! the cost of re-baking colours for every visible mesh.

use bevy::prelude::*;
use utility::profile;

use crate::ir::{LayerSamples, ScalarSamples};

use super::{
  palette::colour_for_scalar,
  registry::{LayerKindCache, RenderRegistry},
};

#[profile]
pub fn paint_layers_system(
  mut registry: ResMut<RenderRegistry>,
  mut meshes: ResMut<Assets<Mesh>>,
) {
  if registry.dirty_meshes.is_empty() {
    return;
  }
  let dirty: Vec<_> = registry.dirty_meshes.drain().collect();
  for mesh_handle in dirty {
    let Some(mesh_entry) = registry.meshes.get(&mesh_handle) else {
      continue;
    };
    let Some(layer_handle) = registry.bindings.get(&mesh_handle) else {
      // No active layer for this mesh — leave colours as the default
      // (white) baked at register time.
      continue;
    };
    let Some(layer_entry) = registry.layers.get(layer_handle) else {
      continue;
    };
    let LayerKindCache::Scalar { palette, range } = &layer_entry.kind else {
      // Vector / Mask layers aren't painted as vertex colours yet.
      continue;
    };
    let Some(LayerSamples::Scalar(ScalarSamples::PerCell(samples))) =
      layer_entry.samples.as_ref()
    else {
      continue;
    };

    let palette_handle = match palette {
      Some(p) => p,
      None => continue,
    };
    let Some(palette) = registry.palettes.get(palette_handle) else {
      continue;
    };

    let (min, max) = range
      .map(|r| (r.min as f32, r.max as f32))
      .unwrap_or_else(|| sample_range(samples));

    let asset = match meshes.get_mut(&mesh_entry.mesh_handle) {
      Some(asset) => asset,
      None => continue,
    };

    let mut colours: Vec<[f32; 4]> =
      vec![[1.0, 1.0, 1.0, 1.0]; mesh_entry.vertex_count];
    for (vertex_idx, cell_idx) in mesh_entry.vertex_to_cell.iter().enumerate() {
      let Some(cell_idx) = cell_idx else {
        continue;
      };
      let Some(value) = samples.get(*cell_idx) else {
        continue;
      };
      let rgba = colour_for_scalar(palette, *value as f32, min, max);
      if vertex_idx < colours.len() {
        colours[vertex_idx] = [rgba.r, rgba.g, rgba.b, rgba.a];
      }
    }
    asset.insert_attribute(Mesh::ATTRIBUTE_COLOR, colours);
  }
}

fn sample_range(samples: &[f64]) -> (f32, f32) {
  let mut min = f32::INFINITY;
  let mut max = f32::NEG_INFINITY;
  for v in samples {
    let v = *v as f32;
    if !v.is_finite() {
      continue;
    }
    if v < min {
      min = v;
    }
    if v > max {
      max = v;
    }
  }
  if !min.is_finite() || !max.is_finite() || (max - min).abs() < f32::EPSILON {
    (0.0, 1.0)
  } else {
    (min, max)
  }
}
