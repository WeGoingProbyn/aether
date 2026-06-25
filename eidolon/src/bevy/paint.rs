// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! `paint_layers` system — converts the bound layer's per-cell samples into
//! per-vertex colours and writes them into `Mesh::ATTRIBUTE_COLOR`.
//!
//! Scalar layers map through a palette + range; categorical layers map through
//! the consumer-supplied [`CategoricalStyle`]. Only meshes flagged in
//! `RenderRegistry::dirty_meshes` are repainted, so a tick that bumps just a
//! sample buffer doesn't re-bake colours for every visible mesh.

use bevy::prelude::*;
use utility::profile;

use crate::ir::{CategoricalSamples, LayerSamples, Rgba, ScalarSamples};

use super::{
  categorical::CategoricalStyle,
  palette::colour_for_scalar,
  playback::FrameInterpolatorResource,
  registry::{LayerKindCache, MeshEntry, RenderRegistry},
};

#[profile]
pub fn paint_layers_system(
  mut registry: ResMut<RenderRegistry>,
  mut meshes: ResMut<Assets<Mesh>>,
  interp: Res<FrameInterpolatorResource>,
  style: Res<CategoricalStyle>,
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

    let colours: Option<Vec<[f32; 4]>> = match &layer_entry.kind {
      LayerKindCache::Scalar { palette, range } => {
        // Prefer the render-clock-interpolated samples; fall back to the raw
        // latest samples when the interpolator has nothing for this layer.
        let interpolated = interp.0.samples(*layer_handle);
        let samples: &[f64] = match interpolated.as_deref() {
          Some(values) => values,
          None => match layer_entry.samples.as_ref() {
            Some(LayerSamples::Scalar(ScalarSamples::PerCell(values))) => {
              values
            }
            _ => continue,
          },
        };
        let Some(palette_handle) = palette else {
          continue;
        };
        let Some(palette) = registry.palettes.get(palette_handle) else {
          continue;
        };
        let (min, max) = range
          .map(|r| (r.min as f32, r.max as f32))
          .unwrap_or_else(|| sample_range(samples));
        Some(bake_cell_colours(mesh_entry, |cell| {
          samples.get(cell).map(|v| {
            rgba_array(colour_for_scalar(palette, *v as f32, min, max))
          })
        }))
      }
      LayerKindCache::Categorical => {
        let Some(LayerSamples::Categorical(CategoricalSamples::PerCell(ids))) =
          layer_entry.samples.as_ref()
        else {
          continue;
        };
        Some(bake_cell_colours(mesh_entry, |cell| {
          ids
            .get(cell)
            .map(|id| rgba_array(style.colour_for_class(*id)))
        }))
      }
      // Vector / Mask layers aren't painted as vertex colours.
      _ => None,
    };

    let Some(colours) = colours else {
      continue;
    };
    if let Some(asset) = meshes.get_mut(&mesh_entry.mesh_handle) {
      asset.insert_attribute(Mesh::ATTRIBUTE_COLOR, colours);
    }
  }
}

/// Bake per-vertex colours by resolving each vertex's owning cell through
/// `colour_of`. Vertices with no cell, or whose cell returns `None`, stay white.
fn bake_cell_colours(
  mesh_entry: &MeshEntry,
  colour_of: impl Fn(usize) -> Option<[f32; 4]>,
) -> Vec<[f32; 4]> {
  let mut colours: Vec<[f32; 4]> =
    vec![[1.0, 1.0, 1.0, 1.0]; mesh_entry.vertex_count];
  for (vertex_idx, cell_idx) in mesh_entry.vertex_to_cell.iter().enumerate() {
    let Some(cell_idx) = cell_idx else {
      continue;
    };
    if vertex_idx >= colours.len() {
      continue;
    }
    if let Some(rgba) = colour_of(*cell_idx) {
      colours[vertex_idx] = rgba;
    }
  }
  colours
}

fn rgba_array(c: Rgba) -> [f32; 4] {
  [c.r, c.g, c.b, c.a]
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
