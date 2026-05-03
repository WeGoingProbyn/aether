// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Apply-side cache for the bevy backend. Maps eidolon `*Handle`s to
//! the bevy `Entity`s and `Handle<Asset>`s the apply system has
//! spawned/uploaded.
//!
//! Mirrors the [`crate::registry::BackendRegistry`] in spirit: it's
//! the source of truth for "what's currently visible", but instead of
//! reconstructing a snapshot it keeps live ECS handles. The IR
//! registry stays around for VTK; this one is bevy-only.

use std::collections::{HashMap, HashSet};

use bevy::prelude::*;

use crate::ir::{
  LayerHandle, LayerKind, LayerSamples, MeshHandle, Palette, PaletteHandle,
  RenderMeshId, ScalarRange, WorldHandle,
};

/// One world's bevy ECS state.
#[derive(Debug)]
pub struct WorldEntry {
  pub entity: Entity,
}

/// One mesh's bevy ECS + asset state.
#[derive(Debug)]
pub struct MeshEntry {
  pub entity: Entity,
  pub mesh_handle: Handle<Mesh>,
  pub material_handle: Handle<StandardMaterial>,
  pub world: WorldHandle,
  pub render_id: RenderMeshId,
  /// Vertex count of the current geometry. Paint uses this to size
  /// the colour attribute buffer.
  pub vertex_count: usize,
  /// Per-vertex cell index (Some(i) for the cell that owns the
  /// triangle). `None` for vertices that aren't part of a renderable
  /// cell. The paint system uses this to look up a per-cell sample.
  pub vertex_to_cell: Vec<Option<usize>>,
}

/// One layer's last-known sample/palette state.
#[derive(Debug)]
pub struct LayerEntry {
  pub target: MeshHandle,
  pub kind: LayerKindCache,
  pub samples: Option<LayerSamples>,
}

#[derive(Debug, Clone)]
pub enum LayerKindCache {
  Scalar {
    palette: Option<PaletteHandle>,
    range: Option<ScalarRange>,
  },
  Vector,
  Mask,
}

impl From<&LayerKind> for LayerKindCache {
  fn from(kind: &LayerKind) -> Self {
    match kind {
      LayerKind::Scalar { palette, range } => LayerKindCache::Scalar {
        palette: *palette,
        range: *range,
      },
      LayerKind::Vector { .. } => LayerKindCache::Vector,
      LayerKind::Mask => LayerKindCache::Mask,
      LayerKind::Debug => LayerKindCache::Mask,
    }
  }
}

#[derive(Resource, Debug, Default)]
pub struct RenderRegistry {
  pub worlds: HashMap<WorldHandle, WorldEntry>,
  pub meshes: HashMap<MeshHandle, MeshEntry>,
  pub layers: HashMap<LayerHandle, LayerEntry>,
  pub palettes: HashMap<PaletteHandle, Palette>,
  /// Currently bound layer for each mesh. Painting only happens for
  /// the bound layer (or a default-active scalar — the apply system
  /// auto-binds the first scalar layer it sees for a given mesh).
  pub bindings: HashMap<MeshHandle, LayerHandle>,
  /// Meshes whose currently-bound layer's samples or palette changed
  /// this tick. The paint system drains this set every frame.
  pub dirty_meshes: HashSet<MeshHandle>,
}

impl RenderRegistry {
  pub fn mark_mesh_dirty(&mut self, mesh: MeshHandle) {
    self.dirty_meshes.insert(mesh);
  }

  /// Mark every mesh whose binding currently points at `layer` as
  /// dirty. Used when a layer's samples or palette change.
  pub fn mark_layer_dirty(&mut self, layer: LayerHandle) {
    let meshes_to_mark: Vec<MeshHandle> = self
      .bindings
      .iter()
      .filter_map(|(mesh, bound)| (*bound == layer).then_some(*mesh))
      .collect();
    for mesh in meshes_to_mark {
      self.dirty_meshes.insert(mesh);
    }
  }
}
