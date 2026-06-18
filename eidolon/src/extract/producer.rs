// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Smart, diffing `FrameProducer`.
//!
//! Replaces the dumb [`super::snapshot_adapter`] on the live render
//! path. Maintains an internal cache of last-emitted hashes per
//! resource (geometry, samples, transforms), so the only Updates that
//! reach the backend each tick are the ones that actually changed.
//!
//! Inputs the producer can't read on its own (a planet's position from
//! gravitas's `BodyState`) come in via the `extract()` parameters —
//! eidolon stays free of physics-crate deps. The runner that owns the
//! sim-thread is responsible for fetching the body position from
//! pleroma resources and handing it down.

use std::collections::{HashMap, HashSet};

use pleroma::{Pleroma, core::storage::FieldStorage};
use tessera::world_mesh::Tessera;
use utility::domain::{FieldKey, MeshKey, ResourceKey, WorldId};
use utility::profile;

use crate::{
  extract::mesh::{boundary_surface_triangles, cell_centroid_points},
  ir::{
    LayerHandle, LayerId, LayerKind, LayerSamples, LayerSource, MeshHandle,
    MeshRepresentation, Palette, PaletteHandle, RenderGeometry, RenderMesh,
    RenderMeshId, ScalarSamples, Transform, Update, UpdateBatch, WorldHandle,
  },
};

/// Static configuration the producer was created with — what meshes
/// and layers to track for a given world.
#[derive(Clone, Debug, Default)]
pub struct ExtractConfig {
  pub world_label: String,
  /// Optional uniform scale applied to the world's transform. Useful
  /// once the bevy backend wants local-space geometry. Defaults to
  /// `1.0` (geometry stays in world coordinates).
  pub world_scale: f64,
  pub meshes: Vec<MeshConfig>,
  pub layers: Vec<ScalarLayerConfig>,
  /// Whether to emit `UpdateSunDirection` from `ResourceKey::SunPosition`.
  pub track_sun_direction: bool,
}

#[derive(Clone, Debug)]
pub struct MeshConfig {
  pub mesh_key: MeshKey,
  pub representation: MeshRepresentation,
  pub label: String,
}

#[derive(Clone, Debug)]
pub struct ScalarLayerConfig {
  pub id: LayerId,
  pub label: String,
  pub target_mesh: MeshKey,
  pub target_representation: MeshRepresentation,
  pub field: FieldKey,
  pub component: usize,
  pub palette: Palette,
}

#[derive(Clone, Debug, Default)]
struct ProducerCache {
  registered_palettes: HashSet<PaletteHandle>,
  registered_world: bool,
  world_transform_hash: Option<u64>,
  world_transform_epoch: u64,
  /// Meshes we've already emitted a `RegisterMesh` for. The tessera
  /// mesh is immutable for the lifetime of the producer, so once a
  /// mesh is here we never touch it again — no rebuild, no hash, no
  /// `UpdateMeshGeometry`. Mesh transform is also identity by
  /// construction; tracking it would just be wasted work.
  mesh_geometry_hashes: HashMap<MeshHandle, u64>,
  mesh_geometry_epochs: HashMap<MeshHandle, u64>,
  mesh_transform_hashes: HashMap<MeshHandle, u64>,
  mesh_transform_epochs: HashMap<MeshHandle, u64>,
  registered_meshes: HashSet<MeshHandle>,
  registered_layers: HashSet<LayerHandle>,
  layer_sample_hashes: HashMap<LayerHandle, u64>,
  layer_sample_epochs: HashMap<LayerHandle, u64>,
  last_sun_direction_hash: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct FrameProducer {
  config: ExtractConfig,
  cache: ProducerCache,
}

impl FrameProducer {
  pub fn new(config: ExtractConfig) -> Self {
    Self {
      config,
      cache: ProducerCache::default(),
    }
  }

  pub fn config(&self) -> &ExtractConfig {
    &self.config
  }

  /// Produce one batch of Updates for this tick.
  ///
  /// `body_position` is the world's centre in absolute simulation
  /// coordinates — typically read by the runner from
  /// `BodyState<3>::positions[body_index]`. Pass `None` if the world
  /// has no orbital body (the world stays at the origin).
  #[profile]
  pub fn extract(
    &mut self,
    world_id: WorldId,
    tessera: &Tessera,
    pleroma: &Pleroma,
    body_position: Option<[f64; 3]>,
    sim_time: f64,
    frame: u64,
  ) -> UpdateBatch {
    let mut updates = Vec::new();

    self.emit_palettes(&mut updates);
    self.emit_world(world_id, body_position, &mut updates);
    self.emit_meshes(world_id, tessera, &mut updates);
    self.emit_layers(world_id, pleroma, &mut updates);
    self.emit_sun_direction(world_id, pleroma, &mut updates);

    updates.push(Update::SetSimTime { sim_time, frame });
    UpdateBatch {
      frame,
      sim_time,
      updates,
    }
  }

  fn emit_palettes(&mut self, updates: &mut Vec<Update>) {
    for layer in &self.config.layers {
      let handle = PaletteHandle::from_static_name(layer.palette.name);
      if self.cache.registered_palettes.insert(handle) {
        updates.push(Update::RegisterPalette {
          handle,
          palette: layer.palette.clone(),
        });
      }
    }
  }

  fn emit_world(
    &mut self,
    world_id: WorldId,
    body_position: Option<[f64; 3]>,
    updates: &mut Vec<Update>,
  ) {
    let world_handle = WorldHandle::from_world_id(world_id);
    let scale = if self.config.world_scale > 0.0 {
      self.config.world_scale
    } else {
      1.0
    };
    let transform = match body_position {
      Some(centre) => Transform::translation_scaling(centre, scale),
      None => {
        if scale == 1.0 {
          Transform::IDENTITY
        } else {
          Transform::scaling(scale)
        }
      }
    };
    let transform_hash = hash_transform(&transform);

    if !self.cache.registered_world {
      self.cache.world_transform_epoch = 1;
      updates.push(Update::RegisterWorld {
        handle: world_handle,
        world_id,
        label: self.config.world_label.clone(),
        transform,
        transform_epoch: self.cache.world_transform_epoch,
      });
      self.cache.registered_world = true;
      self.cache.world_transform_hash = Some(transform_hash);
    } else if self.cache.world_transform_hash != Some(transform_hash) {
      self.cache.world_transform_epoch =
        self.cache.world_transform_epoch.wrapping_add(1);
      updates.push(Update::UpdateWorldTransform {
        handle: world_handle,
        transform,
        transform_epoch: self.cache.world_transform_epoch,
      });
      self.cache.world_transform_hash = Some(transform_hash);
    }
  }

  fn emit_meshes(
    &mut self,
    world_id: WorldId,
    tessera: &Tessera,
    updates: &mut Vec<Update>,
  ) {
    let world_handle = WorldHandle::from_world_id(world_id);
    for mesh_cfg in &self.config.meshes {
      let Some(mesh) = tessera.mesh(mesh_cfg.mesh_key) else {
        continue;
      };
      let render_mesh = build_mesh(world_id, mesh_cfg, mesh.as_ref());
      let handle = render_mesh.id.handle();
      let geometry_hash = hash_geometry(&render_mesh.geometry);
      let transform_hash = hash_transform(&render_mesh.transform);

      if self.cache.registered_meshes.insert(handle) {
        self.cache.mesh_geometry_epochs.insert(handle, 1);
        self.cache.mesh_transform_epochs.insert(handle, 1);
        updates.push(Update::RegisterMesh {
          handle,
          world: world_handle,
          id: render_mesh.id,
          label: render_mesh.label.clone(),
          source: render_mesh.source.clone(),
          geometry: render_mesh.geometry.clone(),
          transform: render_mesh.transform,
          geometry_epoch: 1,
          transform_epoch: 1,
        });
        self
          .cache
          .mesh_geometry_hashes
          .insert(handle, geometry_hash);
        self
          .cache
          .mesh_transform_hashes
          .insert(handle, transform_hash);
      } else {
        if self.cache.mesh_geometry_hashes.get(&handle) != Some(&geometry_hash)
        {
          let epoch = self
            .cache
            .mesh_geometry_epochs
            .get(&handle)
            .copied()
            .unwrap_or(0)
            .wrapping_add(1);
          self.cache.mesh_geometry_epochs.insert(handle, epoch);
          updates.push(Update::UpdateMeshGeometry {
            handle,
            geometry: render_mesh.geometry.clone(),
            epoch,
          });
          self
            .cache
            .mesh_geometry_hashes
            .insert(handle, geometry_hash);
        }
        if self.cache.mesh_transform_hashes.get(&handle)
          != Some(&transform_hash)
        {
          let epoch = self
            .cache
            .mesh_transform_epochs
            .get(&handle)
            .copied()
            .unwrap_or(0)
            .wrapping_add(1);
          self.cache.mesh_transform_epochs.insert(handle, epoch);
          updates.push(Update::UpdateMeshTransform {
            handle,
            transform: render_mesh.transform,
            epoch,
          });
          self
            .cache
            .mesh_transform_hashes
            .insert(handle, transform_hash);
        }
      }
    }
  }

  fn emit_layers(
    &mut self,
    world_id: WorldId,
    pleroma: &Pleroma,
    updates: &mut Vec<Update>,
  ) {
    for layer_cfg in &self.config.layers {
      let target_mesh_id = RenderMeshId {
        world: world_id,
        mesh: layer_cfg.target_mesh,
        representation: layer_cfg.target_representation,
      };
      let target = target_mesh_id.handle();
      // Only emit layer updates for meshes we've registered.
      if !self.cache.registered_meshes.contains(&target) {
        continue;
      }
      let handle = LayerHandle::for_target(layer_cfg.id, target);
      let palette =
        Some(PaletteHandle::from_static_name(layer_cfg.palette.name));

      // Read the field. Today we only know how to extract scalars from
      // SoaField<1> through SoaField<5>; loosely match the components
      // we need.
      let Some(samples) =
        read_scalar_component(pleroma, layer_cfg.field, layer_cfg.component)
      else {
        continue;
      };
      let samples_hash = hash_f64_slice(&samples);

      if self.cache.registered_layers.insert(handle) {
        self.cache.layer_sample_epochs.insert(handle, 1);
        updates.push(Update::RegisterLayer {
          handle,
          id: layer_cfg.id,
          label: layer_cfg.label.clone(),
          target,
          source: LayerSource::Field(layer_cfg.field),
          kind: LayerKind::Scalar {
            palette,
            range: None,
          },
        });
        updates.push(Update::UpdateLayerSamples {
          handle,
          samples: LayerSamples::Scalar(ScalarSamples::PerCell(samples)),
          epoch: 1,
        });
        self.cache.layer_sample_hashes.insert(handle, samples_hash);
      } else if self.cache.layer_sample_hashes.get(&handle)
        != Some(&samples_hash)
      {
        let epoch = self
          .cache
          .layer_sample_epochs
          .get(&handle)
          .copied()
          .unwrap_or(0)
          .wrapping_add(1);
        self.cache.layer_sample_epochs.insert(handle, epoch);
        updates.push(Update::UpdateLayerSamples {
          handle,
          samples: LayerSamples::Scalar(ScalarSamples::PerCell(samples)),
          epoch,
        });
        self.cache.layer_sample_hashes.insert(handle, samples_hash);
      }
    }
  }

  fn emit_sun_direction(
    &mut self,
    world_id: WorldId,
    pleroma: &Pleroma,
    updates: &mut Vec<Update>,
  ) {
    if !self.config.track_sun_direction {
      return;
    }
    let Some(direction) = pleroma
      .read_resource::<[f64; 3]>(ResourceKey::SunPosition)
      .copied()
    else {
      return;
    };
    let h = hash_f64_slice(&direction);
    if self.cache.last_sun_direction_hash != Some(h) {
      updates.push(Update::UpdateSunDirection {
        world: WorldHandle::from_world_id(world_id),
        direction,
      });
      self.cache.last_sun_direction_hash = Some(h);
    }
  }
}

fn build_mesh(
  world_id: WorldId,
  cfg: &MeshConfig,
  mesh: &dyn tessera::mesh::Mesh<3>,
) -> RenderMesh {
  let mut render_mesh = match cfg.representation {
    MeshRepresentation::BoundaryFaces => {
      boundary_surface_triangles(world_id, cfg.mesh_key, mesh)
    }
    MeshRepresentation::Cells => {
      cell_centroid_points(world_id, cfg.mesh_key, mesh)
    }
    // For the others, fall back to cell centroids — extractors for
    // them aren't wired into the producer yet. Tests don't exercise
    // these paths.
    _ => cell_centroid_points(world_id, cfg.mesh_key, mesh),
  };
  if !cfg.label.is_empty() {
    render_mesh.label = cfg.label.clone();
  }
  render_mesh
}

fn read_scalar_component(
  pleroma: &Pleroma,
  field: FieldKey,
  component: usize,
) -> Option<Vec<f64>> {
  // Try common arities. The producer only knows scalar layers, so the
  // caller passes the component index; the storage shape just has to
  // contain that many components.
  if let Some(storage) =
    pleroma.read::<pleroma::core::storage::SoaField<1>>(field)
  {
    if component < 1 {
      return Some(storage.component(component).as_ref().to_vec());
    }
  }
  if let Some(storage) =
    pleroma.read::<pleroma::core::storage::SoaField<3>>(field)
  {
    if component < 3 {
      return Some(storage.component(component).as_ref().to_vec());
    }
  }
  if let Some(storage) =
    pleroma.read::<pleroma::core::storage::SoaField<4>>(field)
  {
    if component < 4 {
      return Some(storage.component(component).as_ref().to_vec());
    }
  }
  if let Some(storage) =
    pleroma.read::<pleroma::core::storage::SoaField<5>>(field)
  {
    if component < 5 {
      return Some(storage.component(component).as_ref().to_vec());
    }
  }
  if let Some(storage) =
    pleroma.read::<pleroma::core::storage::SoaField<6>>(field)
  {
    if component < 6 {
      return Some(storage.component(component).as_ref().to_vec());
    }
  }
  None
}

// ---- Hashes (FNV-1a 64) ----

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv_mix_u64(mut h: u64, value: u64) -> u64 {
  let bytes = value.to_le_bytes();
  for byte in bytes {
    h ^= byte as u64;
    h = h.wrapping_mul(FNV_PRIME);
  }
  h
}

fn fnv_mix_f32(h: u64, value: f32) -> u64 {
  fnv_mix_u64(h, value.to_bits() as u64)
}

fn fnv_mix_f64(h: u64, value: f64) -> u64 {
  fnv_mix_u64(h, value.to_bits())
}

fn hash_f64_slice(values: &[f64]) -> u64 {
  let mut h = FNV_OFFSET;
  for v in values {
    h = fnv_mix_f64(h, *v);
  }
  h
}

fn hash_transform(t: &Transform) -> u64 {
  let mut h = FNV_OFFSET;
  for c in t.centre {
    h = fnv_mix_f64(h, c);
  }
  for o in t.orientation {
    h = fnv_mix_f64(h, o);
  }
  fnv_mix_f64(h, t.scale)
}

fn hash_geometry(geometry: &RenderGeometry) -> u64 {
  match geometry {
    RenderGeometry::Triangles(tri) => {
      let mut h = FNV_OFFSET ^ 0xA1;
      for [x, y, z] in &tri.positions {
        h = fnv_mix_f32(h, *x);
        h = fnv_mix_f32(h, *y);
        h = fnv_mix_f32(h, *z);
      }
      for i in &tri.indices {
        h = fnv_mix_u64(h, *i as u64);
      }
      h
    }
    RenderGeometry::Lines(lines) => {
      let mut h = FNV_OFFSET ^ 0xA2;
      for [x, y, z] in &lines.positions {
        h = fnv_mix_f32(h, *x);
        h = fnv_mix_f32(h, *y);
        h = fnv_mix_f32(h, *z);
      }
      for [a, b] in &lines.segments {
        h = fnv_mix_u64(h, *a as u64);
        h = fnv_mix_u64(h, *b as u64);
      }
      h
    }
    RenderGeometry::Points(points) => {
      let mut h = FNV_OFFSET ^ 0xA3;
      for [x, y, z] in &points.positions {
        h = fnv_mix_f32(h, *x);
        h = fnv_mix_f32(h, *y);
        h = fnv_mix_f32(h, *z);
      }
      h
    }
    RenderGeometry::Packed(asset) => {
      // Packed geometry is engine-specific. Hash the byte stream.
      let mut h = FNV_OFFSET ^ 0xA4;
      for byte in asset.vertex_data.iter() {
        h ^= *byte as u64;
        h = h.wrapping_mul(FNV_PRIME);
      }
      for byte in asset.index_data.iter() {
        h ^= *byte as u64;
        h = h.wrapping_mul(FNV_PRIME);
      }
      h
    }
  }
}
