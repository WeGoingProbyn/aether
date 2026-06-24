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
  /// Undisplaced vertex positions captured when the geometry was built.
  /// Displacement recomputes positions from this base each time so repeated
  /// applies don't compound. Empty for non-triangle geometry.
  pub base_positions: Vec<[f32; 3]>,
}

/// Which layer's samples displace a mesh, and by how much. Set by
/// [`crate::ir::Update::SetMeshDisplacement`].
#[derive(Debug, Clone, Copy)]
pub struct DisplacementBinding {
  pub layer: LayerHandle,
  pub scale: f32,
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
      // The reference renderer has no categorical material mapping yet (that is
      // a consumer / visual-verification concern); cache it like a mask so the
      // layer is tracked without inventing colours.
      LayerKind::Categorical { .. } => LayerKindCache::Mask,
      LayerKind::Debug => LayerKindCache::Mask,
    }
  }
}

#[derive(Resource, Debug)]
pub struct RenderRegistry {
  pub worlds: HashMap<WorldHandle, WorldEntry>,
  pub meshes: HashMap<MeshHandle, MeshEntry>,
  pub layers: HashMap<LayerHandle, LayerEntry>,
  pub palettes: HashMap<PaletteHandle, Palette>,
  /// Currently bound layer for each mesh. Painting only happens for
  /// the bound layer (or a default-active scalar — the apply system
  /// auto-binds the first scalar layer it sees for a given mesh).
  pub bindings: HashMap<MeshHandle, LayerHandle>,
  /// Which layer (and exaggeration) displaces each mesh's geometry.
  pub displacements: HashMap<MeshHandle, DisplacementBinding>,
  /// Meshes whose currently-bound layer's samples or palette changed
  /// this tick. The paint system drains this set every frame.
  pub dirty_meshes: HashSet<MeshHandle>,
  /// Meshes whose displacement needs (re)applying — its directive or its
  /// driving layer's samples changed. The displace system drains this set.
  pub dirty_displacements: HashSet<MeshHandle>,
  /// Master switch for terrain relief. When `false`, displaced meshes are
  /// flattened back to their base geometry — a consumer flips this off for a
  /// debug field view (relief distorts the colours) and on for the rendered
  /// look. Defaults to `true`: a displacement directive means displace.
  pub displacement_enabled: bool,
}

impl Default for RenderRegistry {
  fn default() -> Self {
    Self {
      worlds: HashMap::new(),
      meshes: HashMap::new(),
      layers: HashMap::new(),
      palettes: HashMap::new(),
      bindings: HashMap::new(),
      displacements: HashMap::new(),
      dirty_meshes: HashSet::new(),
      dirty_displacements: HashSet::new(),
      displacement_enabled: true,
    }
  }
}

impl RenderRegistry {
  pub fn mark_mesh_dirty(&mut self, mesh: MeshHandle) {
    self.dirty_meshes.insert(mesh);
  }

  /// Enable or disable terrain relief globally. On a change, every displaced
  /// mesh is re-flagged so the displace system re-applies (displaced) or
  /// restores (flat) on the next frame. A no-op when already in that state.
  pub fn set_displacement_enabled(&mut self, enabled: bool) {
    if self.displacement_enabled == enabled {
      return;
    }
    self.displacement_enabled = enabled;
    let meshes: Vec<MeshHandle> = self.displacements.keys().copied().collect();
    for mesh in meshes {
      self.dirty_displacements.insert(mesh);
    }
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

  /// Mark every mesh displaced by `layer` for re-displacement. Used when the
  /// driving layer's samples change.
  pub fn mark_displacement_dirty(&mut self, layer: LayerHandle) {
    let meshes_to_mark: Vec<MeshHandle> = self
      .displacements
      .iter()
      .filter_map(|(mesh, d)| (d.layer == layer).then_some(*mesh))
      .collect();
    for mesh in meshes_to_mark {
      self.dirty_displacements.insert(mesh);
    }
  }
}

#[cfg(test)]
mod tests {
  use utility::domain::{MeshKey, WorldId};

  use super::*;
  use crate::ir::{LayerId, RenderMeshId};

  fn mesh_handle() -> MeshHandle {
    RenderMeshId {
      world: WorldId(0),
      mesh: MeshKey::SURFACE,
      representation: crate::ir::MeshRepresentation::BoundaryFaces,
    }
    .handle()
  }

  #[test]
  fn toggling_displacement_reflags_displaced_meshes_only_on_change() {
    let mut registry = RenderRegistry::default();
    assert!(registry.displacement_enabled, "defaults to displacing");

    let mesh = mesh_handle();
    let layer = LayerHandle::for_target(LayerId::from_static("elev"), mesh);
    registry
      .displacements
      .insert(mesh, DisplacementBinding { layer, scale: 10.0 });

    // Disabling flips the flag and re-flags the displaced mesh (to flatten).
    registry.set_displacement_enabled(false);
    assert!(!registry.displacement_enabled);
    assert!(registry.dirty_displacements.contains(&mesh));

    // Re-applying the same state is a no-op (no spurious dirtying).
    registry.dirty_displacements.clear();
    registry.set_displacement_enabled(false);
    assert!(registry.dirty_displacements.is_empty());

    // Re-enabling re-flags it again (to re-displace).
    registry.set_displacement_enabled(true);
    assert!(registry.displacement_enabled);
    assert!(registry.dirty_displacements.contains(&mesh));
  }
}
