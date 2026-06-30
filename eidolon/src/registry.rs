// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Apply-side cache for the Update protocol.
//!
//! `BackendRegistry::apply(&UpdateBatch)` walks the wire updates and
//! mutates an in-memory model of the scene. `snapshot()` materialises
//! that model as a `RenderFrame` for non-streaming consumers (VTK,
//! tests). Streaming consumers (bevy) read the registry directly via
//! the public accessors.
//!
//! The registry is *forgiving*: receiving an `Update*` for an unknown
//! handle, or a sample whose kind doesn't match the declared
//! `LayerKind`, returns an error rather than panicking. This is the
//! contract Phase 3's coalescer leans on — adversarial sequences must
//! not be able to corrupt state.

use std::collections::HashMap;

use utility::domain::WorldId;

use crate::ir::{
  CategoricalLayer, LayerHandle, LayerId, LayerKind, LayerSamples, LayerSource,
  MaskLayer, MaskSamples as IrMaskSamples, MeshHandle, MeshSource, Palette,
  PaletteHandle, RenderFrame, RenderGeometry, RenderLayer, RenderMesh,
  RenderMeshId, RenderWorld, ScalarLayer, ScalarRange,
  ScalarSamples as IrScalarSamples, Transform, Update, UpdateBatch,
  VectorGlyph, VectorLayer, VectorSamples as IrVectorSamples, WorldHandle,
};

/// Errors the registry surfaces back to its caller. None of these are
/// panics — the registry stays in a consistent state when an Update is
/// rejected.
#[derive(Clone, Debug, PartialEq)]
pub enum RegistryError {
  UnknownWorld(WorldHandle),
  UnknownMesh(MeshHandle),
  UnknownLayer(LayerHandle),
  UnknownPalette(PaletteHandle),
  /// `LayerSamples` variant didn't match the layer's registered
  /// `LayerKind`.
  SampleKindMismatch {
    layer: LayerHandle,
  },
  /// A `Register*` for a handle we already know about. Treated as a
  /// soft warning — callers can choose to treat it as an error during
  /// development.
  Duplicate(&'static str),
}

#[derive(Clone, Debug)]
struct WorldEntry {
  id: WorldId,
  label: String,
  transform: Transform,
  transform_epoch: u64,
  meshes: Vec<MeshHandle>,
  sun_direction: Option<[f64; 3]>,
}

#[derive(Clone, Debug)]
struct MeshEntry {
  id: RenderMeshId,
  world: WorldHandle,
  label: String,
  source: MeshSource,
  geometry: RenderGeometry,
  transform: Transform,
  geometry_epoch: u64,
  transform_epoch: u64,
}

#[derive(Clone, Debug)]
struct LayerEntry {
  id: LayerId,
  label: String,
  target: MeshHandle,
  source: LayerSource,
  kind: LayerKind,
  samples: Option<LayerSamples>,
}

#[derive(Clone, Debug, Default)]
pub struct BackendRegistry {
  palettes: HashMap<PaletteHandle, Palette>,
  worlds: HashMap<WorldHandle, WorldEntry>,
  /// World registration order, so `snapshot()` is stable across runs.
  world_order: Vec<WorldHandle>,
  meshes: HashMap<MeshHandle, MeshEntry>,
  layers: HashMap<LayerHandle, LayerEntry>,
  /// Currently bound layer for each mesh (UpdateLayerBinding). `None`
  /// or absent means "no binding" — the backend may still render the
  /// mesh untextured.
  bindings: HashMap<MeshHandle, LayerHandle>,
  sim_time: f64,
  frame: u64,
}

impl BackendRegistry {
  pub fn new() -> Self {
    Self::default()
  }

  pub fn sim_time(&self) -> f64 {
    self.sim_time
  }

  pub fn frame(&self) -> u64 {
    self.frame
  }

  pub fn palette(&self, handle: PaletteHandle) -> Option<&Palette> {
    self.palettes.get(&handle)
  }

  pub fn binding_for(&self, mesh: MeshHandle) -> Option<LayerHandle> {
    self.bindings.get(&mesh).copied()
  }

  /// Apply every update in the batch. Errors are accumulated and
  /// returned all together — applying continues past a single bad
  /// update so a malformed `UpdateLayerSamples` doesn't drop a later
  /// `FreeMesh` on the floor.
  pub fn apply(&mut self, batch: &UpdateBatch) -> Vec<RegistryError> {
    let mut errors = Vec::new();
    self.frame = batch.frame;
    for update in &batch.updates {
      if let Err(error) = self.apply_one(update) {
        errors.push(error);
      }
    }
    if !batch.updates.is_empty() {
      // SetSimTime in the batch wins; otherwise carry the batch's
      // declared sim_time forward.
      self.sim_time = batch.sim_time;
    }
    errors
  }

  fn apply_one(&mut self, update: &Update) -> Result<(), RegistryError> {
    match update {
      Update::RegisterPalette { handle, palette } => {
        if self.palettes.insert(*handle, palette.clone()).is_some() {
          return Err(RegistryError::Duplicate("palette"));
        }
      }
      Update::FreePalette { handle } => {
        self.palettes.remove(handle);
      }

      Update::RegisterWorld {
        handle,
        world_id,
        label,
        transform,
        transform_epoch,
      } => {
        if self
          .worlds
          .insert(
            *handle,
            WorldEntry {
              id: *world_id,
              label: label.clone(),
              transform: *transform,
              transform_epoch: *transform_epoch,
              meshes: Vec::new(),
              sun_direction: None,
            },
          )
          .is_some()
        {
          return Err(RegistryError::Duplicate("world"));
        }
        self.world_order.push(*handle);
      }
      Update::UpdateWorldTransform {
        handle,
        transform,
        transform_epoch,
      } => {
        let entry = self
          .worlds
          .get_mut(handle)
          .ok_or(RegistryError::UnknownWorld(*handle))?;
        entry.transform = *transform;
        entry.transform_epoch = *transform_epoch;
      }
      Update::FreeWorld { handle } => {
        let Some(entry) = self.worlds.remove(handle) else {
          return Err(RegistryError::UnknownWorld(*handle));
        };
        // Cascade: drop every mesh that lived inside the world.
        for mesh in entry.meshes {
          self.drop_mesh_entry(mesh);
        }
        self.world_order.retain(|h| h != handle);
      }

      Update::RegisterMesh {
        handle,
        world,
        id,
        label,
        source,
        geometry,
        transform,
        geometry_epoch,
        transform_epoch,
      } => {
        if !self.worlds.contains_key(world) {
          return Err(RegistryError::UnknownWorld(*world));
        }
        if self
          .meshes
          .insert(
            *handle,
            MeshEntry {
              id: *id,
              world: *world,
              label: label.clone(),
              source: source.clone(),
              geometry: geometry.clone(),
              transform: *transform,
              geometry_epoch: *geometry_epoch,
              transform_epoch: *transform_epoch,
            },
          )
          .is_some()
        {
          return Err(RegistryError::Duplicate("mesh"));
        }
        self
          .worlds
          .get_mut(world)
          .expect("world existence checked above")
          .meshes
          .push(*handle);
      }
      Update::UpdateMeshGeometry {
        handle,
        geometry,
        epoch,
      } => {
        let entry = self
          .meshes
          .get_mut(handle)
          .ok_or(RegistryError::UnknownMesh(*handle))?;
        entry.geometry = geometry.clone();
        entry.geometry_epoch = *epoch;
      }
      Update::UpdateMeshTransform {
        handle,
        transform,
        epoch,
      } => {
        let entry = self
          .meshes
          .get_mut(handle)
          .ok_or(RegistryError::UnknownMesh(*handle))?;
        entry.transform = *transform;
        entry.transform_epoch = *epoch;
      }
      Update::FreeMesh { handle } => {
        if !self.meshes.contains_key(handle) {
          return Err(RegistryError::UnknownMesh(*handle));
        }
        self.drop_mesh_entry(*handle);
      }

      Update::RegisterLayer {
        handle,
        id,
        label,
        target,
        source,
        kind,
      } => {
        if !self.meshes.contains_key(target) {
          return Err(RegistryError::UnknownMesh(*target));
        }
        if let LayerKind::Scalar {
          palette: Some(p), ..
        } = kind
        {
          if !self.palettes.contains_key(p) {
            return Err(RegistryError::UnknownPalette(*p));
          }
        }
        if self
          .layers
          .insert(
            *handle,
            LayerEntry {
              id: *id,
              label: label.clone(),
              target: *target,
              source: *source,
              kind: kind.clone(),
              samples: None,
            },
          )
          .is_some()
        {
          return Err(RegistryError::Duplicate("layer"));
        }
      }
      Update::UpdateLayerSamples {
        handle,
        samples,
        epoch: _,
      } => {
        let entry = self
          .layers
          .get_mut(handle)
          .ok_or(RegistryError::UnknownLayer(*handle))?;
        if !samples.matches_kind(&entry.kind) {
          return Err(RegistryError::SampleKindMismatch { layer: *handle });
        }
        entry.samples = Some(samples.clone());
      }
      Update::UpdateLayerPalette { handle, palette } => {
        if let Some(p) = palette {
          if !self.palettes.contains_key(p) {
            return Err(RegistryError::UnknownPalette(*p));
          }
        }
        let entry = self
          .layers
          .get_mut(handle)
          .ok_or(RegistryError::UnknownLayer(*handle))?;
        if let LayerKind::Scalar {
          palette: ref mut p, ..
        } = entry.kind
        {
          *p = *palette;
        }
      }
      Update::UpdateLayerBinding { mesh, layer } => {
        if !self.meshes.contains_key(mesh) {
          return Err(RegistryError::UnknownMesh(*mesh));
        }
        match layer {
          Some(l) => {
            if !self.layers.contains_key(l) {
              return Err(RegistryError::UnknownLayer(*l));
            }
            self.bindings.insert(*mesh, *l);
          }
          None => {
            self.bindings.remove(mesh);
          }
        }
      }
      Update::FreeLayer { handle } => {
        if self.layers.remove(handle).is_none() {
          return Err(RegistryError::UnknownLayer(*handle));
        }
        // Clear any binding that referenced this layer.
        self.bindings.retain(|_, layer| layer != handle);
      }

      Update::SetMeshDisplacement { .. } => {
        // A live-render hint only. The snapshot/VTK path reconstructs the
        // undisplaced IR geometry, so there is nothing to apply here.
      }

      Update::UpdateSunDirection { world, direction } => {
        let entry = self
          .worlds
          .get_mut(world)
          .ok_or(RegistryError::UnknownWorld(*world))?;
        entry.sun_direction = Some(*direction);
      }

      Update::SetCamera { .. } => {
        // A live-render hint (the simulation-owned view). The snapshot / VTK
        // path has no camera, so there is nothing to apply here.
      }

      Update::SetSimTime { sim_time, frame } => {
        self.sim_time = *sim_time;
        self.frame = *frame;
      }
    }
    Ok(())
  }

  fn drop_mesh_entry(&mut self, handle: MeshHandle) {
    if let Some(entry) = self.meshes.remove(&handle) {
      // Detach from its world's mesh list.
      if let Some(world) = self.worlds.get_mut(&entry.world) {
        world.meshes.retain(|h| *h != handle);
      }
    }
    self.bindings.remove(&handle);
    // Drop any layers that targeted this mesh — their owning code
    // would have to re-register them anyway once the mesh is gone.
    let dead_layers: Vec<LayerHandle> = self
      .layers
      .iter()
      .filter_map(|(h, l)| (l.target == handle).then_some(*h))
      .collect();
    for layer in dead_layers {
      self.layers.remove(&layer);
      self.bindings.retain(|_, bound| *bound != layer);
    }
  }

  /// Materialise a `RenderFrame` from current registry state. Useful
  /// for VTK export and tests.
  pub fn snapshot(&self) -> RenderFrame {
    let mut worlds = Vec::with_capacity(self.world_order.len());
    for handle in &self.world_order {
      let Some(world) = self.worlds.get(handle) else {
        continue;
      };
      let mut render_world = RenderWorld {
        id: world.id,
        label: world.label.clone(),
        transform: world.transform,
        transform_epoch: world.transform_epoch,
        meshes: Vec::with_capacity(world.meshes.len()),
        layers: Vec::new(),
        diagnostics: Vec::new(),
      };
      for mesh_handle in &world.meshes {
        if let Some(mesh) = self.meshes.get(mesh_handle) {
          render_world.meshes.push(self.snapshot_mesh(mesh));
        }
      }
      // Layers whose target lives in this world.
      for (_, entry) in self.layers.iter() {
        let Some(target_mesh) = self.meshes.get(&entry.target) else {
          continue;
        };
        if target_mesh.world != *handle {
          continue;
        }
        if let Some(layer) = self.snapshot_layer(target_mesh.id, entry) {
          render_world.layers.push(layer);
        }
      }
      worlds.push(render_world);
    }
    RenderFrame {
      frame: self.frame,
      sim_time: self.sim_time,
      worlds,
      camera: None,
    }
  }

  fn snapshot_mesh(&self, entry: &MeshEntry) -> RenderMesh {
    RenderMesh {
      id: entry.id,
      label: entry.label.clone(),
      source: entry.source.clone(),
      geometry: entry.geometry.clone(),
      transform: entry.transform,
      epoch: entry.geometry_epoch,
      transform_epoch: entry.transform_epoch,
    }
  }

  fn snapshot_layer(
    &self,
    target: RenderMeshId,
    entry: &LayerEntry,
  ) -> Option<RenderLayer> {
    match (&entry.kind, &entry.samples) {
      (
        LayerKind::Scalar { palette, range },
        Some(LayerSamples::Scalar(samples)),
      ) => {
        let palette = palette
          .and_then(|p| self.palettes.get(&p).cloned())
          .unwrap_or_else(Palette::diagnostic);
        Some(RenderLayer::Scalar(ScalarLayer {
          id: entry.id,
          label: entry.label.clone(),
          target,
          source: entry.source,
          samples: samples.clone(),
          range: *range,
          palette,
        }))
      }
      (LayerKind::Vector { glyph }, Some(LayerSamples::Vector(samples))) => {
        Some(RenderLayer::Vector(VectorLayer {
          id: entry.id,
          label: entry.label.clone(),
          target,
          source: entry.source,
          samples: samples.clone(),
          glyph: *glyph,
        }))
      }
      (LayerKind::Mask, Some(LayerSamples::Mask(samples))) => {
        Some(RenderLayer::Mask(MaskLayer {
          id: entry.id,
          label: entry.label.clone(),
          target,
          source: entry.source,
          samples: samples.clone(),
        }))
      }
      (
        LayerKind::Categorical { classes },
        Some(LayerSamples::Categorical(samples)),
      ) => Some(RenderLayer::Categorical(CategoricalLayer {
        id: entry.id,
        label: entry.label.clone(),
        target,
        source: entry.source,
        samples: samples.clone(),
        classes: classes.clone(),
      })),
      _ => None,
    }
  }
}

// Silence unused-warning for traits that may not be needed yet but
// are exported by the type system.
#[allow(dead_code)]
fn _force_imports(
  _: &IrScalarSamples,
  _: &IrVectorSamples,
  _: &IrMaskSamples,
  _: ScalarRange,
  _: VectorGlyph,
) {
}
