// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Dumb `RenderFrame -> UpdateBatch` adapter.
//!
//! This is the "Phase 2C" hop in the plan: it lets us validate
//! `BackendRegistry` end-to-end against the existing snapshot
//! extractors without writing any diffing logic. The smart
//! [`super::producer`] (Phase 2D) replaces this on the live render
//! path; the adapter keeps living for VTK and tests.
//!
//! `frame_to_initial_batch` registers everything; `frame_to_replace_batch`
//! frees the previous frame and registers the new one (so successive
//! VTK dumps can reuse the same registry without leaking old state).

use std::collections::HashSet;

use crate::ir::{
  LayerHandle, LayerKind, LayerSamples, MeshHandle, PaletteHandle, RenderFrame,
  RenderLayer, Update, UpdateBatch, WorldHandle,
};

/// Build a batch that registers every world / palette / mesh / layer
/// in the frame from scratch, plus a final `SetSimTime`.
pub fn frame_to_initial_batch(frame: &RenderFrame) -> UpdateBatch {
  let mut updates = Vec::new();
  emit_register_palettes(frame, &mut updates);
  emit_register_worlds_and_meshes(frame, &mut updates);
  emit_register_layers(frame, &mut updates);
  updates.push(Update::SetSimTime {
    sim_time: frame.sim_time,
    frame: frame.frame,
  });
  UpdateBatch {
    frame: frame.frame,
    sim_time: frame.sim_time,
    updates,
  }
}

/// Build a batch that frees everything in `prev` and registers `frame`.
/// The prev/new split is a frame-level granularity — every world is
/// torn down and rebuilt. Inefficient on purpose; the smart producer
/// is the optimised path.
pub fn frame_to_replace_batch(
  frame: &RenderFrame,
  prev: &RenderFrame,
) -> UpdateBatch {
  let mut updates = Vec::new();
  emit_free_worlds(prev, &mut updates);
  emit_free_palettes(prev, &mut updates);

  emit_register_palettes(frame, &mut updates);
  emit_register_worlds_and_meshes(frame, &mut updates);
  emit_register_layers(frame, &mut updates);
  updates.push(Update::SetSimTime {
    sim_time: frame.sim_time,
    frame: frame.frame,
  });
  UpdateBatch {
    frame: frame.frame,
    sim_time: frame.sim_time,
    updates,
  }
}

fn emit_register_palettes(frame: &RenderFrame, updates: &mut Vec<Update>) {
  let mut seen = HashSet::new();
  for world in &frame.worlds {
    for layer in &world.layers {
      if let RenderLayer::Scalar(scalar) = layer {
        let handle = PaletteHandle::from_static_name(scalar.palette.name);
        if seen.insert(handle) {
          updates.push(Update::RegisterPalette {
            handle,
            palette: scalar.palette.clone(),
          });
        }
      }
    }
  }
}

fn emit_register_worlds_and_meshes(
  frame: &RenderFrame,
  updates: &mut Vec<Update>,
) {
  for world in &frame.worlds {
    let world_handle = WorldHandle::from_world_id(world.id);
    updates.push(Update::RegisterWorld {
      handle: world_handle,
      world_id: world.id,
      label: world.label.clone(),
      transform: world.transform,
      transform_epoch: world.transform_epoch,
    });
    for mesh in &world.meshes {
      updates.push(Update::RegisterMesh {
        handle: mesh.id.handle(),
        world: world_handle,
        id: mesh.id,
        label: mesh.label.clone(),
        source: mesh.source.clone(),
        geometry: mesh.geometry.clone(),
        transform: mesh.transform,
        geometry_epoch: mesh.epoch,
        transform_epoch: mesh.transform_epoch,
      });
    }
  }
}

fn emit_register_layers(frame: &RenderFrame, updates: &mut Vec<Update>) {
  for world in &frame.worlds {
    for layer in &world.layers {
      register_layer(layer, updates);
    }
  }
}

fn register_layer(layer: &RenderLayer, updates: &mut Vec<Update>) {
  match layer {
    RenderLayer::Scalar(scalar) => {
      let target: MeshHandle = scalar.target.handle();
      let handle = LayerHandle::for_target(scalar.id, target);
      let palette = Some(PaletteHandle::from_static_name(scalar.palette.name));
      updates.push(Update::RegisterLayer {
        handle,
        id: scalar.id,
        label: scalar.label.clone(),
        target,
        source: scalar.source,
        kind: LayerKind::Scalar {
          palette,
          range: scalar.range,
        },
      });
      updates.push(Update::UpdateLayerSamples {
        handle,
        samples: LayerSamples::Scalar(scalar.samples.clone()),
        epoch: 0,
      });
    }
    RenderLayer::Vector(vector) => {
      let target: MeshHandle = vector.target.handle();
      let handle = LayerHandle::for_target(vector.id, target);
      updates.push(Update::RegisterLayer {
        handle,
        id: vector.id,
        label: vector.label.clone(),
        target,
        source: vector.source,
        kind: LayerKind::Vector {
          glyph: vector.glyph,
        },
      });
      updates.push(Update::UpdateLayerSamples {
        handle,
        samples: LayerSamples::Vector(vector.samples.clone()),
        epoch: 0,
      });
    }
    RenderLayer::Mask(mask) => {
      let target: MeshHandle = mask.target.handle();
      let handle = LayerHandle::for_target(mask.id, target);
      updates.push(Update::RegisterLayer {
        handle,
        id: mask.id,
        label: mask.label.clone(),
        target,
        source: mask.source,
        kind: LayerKind::Mask,
      });
      updates.push(Update::UpdateLayerSamples {
        handle,
        samples: LayerSamples::Mask(mask.samples.clone()),
        epoch: 0,
      });
    }
    RenderLayer::Debug(_) => {
      // Debug layers don't have a wire form yet (see LayerKind::Debug
      // doc). Skip them in the dumb adapter; the snapshot path keeps
      // them around for VTK to render directly.
    }
  }
}

fn emit_free_worlds(prev: &RenderFrame, updates: &mut Vec<Update>) {
  for world in &prev.worlds {
    updates.push(Update::FreeWorld {
      handle: WorldHandle::from_world_id(world.id),
    });
  }
}

fn emit_free_palettes(prev: &RenderFrame, updates: &mut Vec<Update>) {
  let mut seen = HashSet::new();
  for world in &prev.worlds {
    for layer in &world.layers {
      if let RenderLayer::Scalar(scalar) = layer {
        let handle = PaletteHandle::from_static_name(scalar.palette.name);
        if seen.insert(handle) {
          updates.push(Update::FreePalette { handle });
        }
      }
    }
  }
}
