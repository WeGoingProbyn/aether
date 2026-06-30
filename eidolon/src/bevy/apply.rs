// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! `apply_updates` system — drains the channel each `PreUpdate` and
//! mutates the bevy world so subsequent systems (paint, render) see
//! the latest sim state.

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use utility::profile;

use crate::ir::{LayerKind, RenderGeometry, Update, UpdateBatch};

use super::{
  playback::{FrameInterpolatorResource, snapshot_sample_frame},
  plugin::UpdateReceiverResource,
  registry::{
    DisplacementBinding, LayerEntry, LayerKindCache, MeshEntry, RenderRegistry,
    WorldEntry,
  },
  transform::to_bevy_transform,
};

#[profile]
pub fn apply_updates_system(
  mut commands: Commands,
  receiver: Res<UpdateReceiverResource>,
  mut registry: ResMut<RenderRegistry>,
  mut meshes: ResMut<Assets<Mesh>>,
  mut materials: ResMut<Assets<StandardMaterial>>,
  mut interp: ResMut<FrameInterpolatorResource>,
  mut sun: ResMut<super::sun::SunDirection>,
  mut sim_camera: ResMut<super::camera::SimCamera>,
) {
  let Some(batch) = pull_batch(&receiver) else {
    return;
  };

  let UpdateBatch { updates, .. } = batch;
  for update in updates {
    // A frame boundary: the registry now holds this frame's complete samples,
    // so snapshot them into the interpolator before the (no-op) apply.
    if let Update::SetSimTime { sim_time, .. } = &update {
      let frame = snapshot_sample_frame(&registry, *sim_time);
      interp.0.push(frame);
    }
    // Record the sun direction so `orient_sun_light_system` can aim the
    // directional light from it (shading tracks the orbiting sun).
    if let Update::UpdateSunDirection { direction, .. } = &update {
      sun.direction = Some(Vec3::new(
        direction[0] as f32,
        direction[1] as f32,
        direction[2] as f32,
      ));
    }
    // Record the simulation-owned view so `position_camera_from_view_system`
    // can drive a `SimDrivenCamera` from it.
    if let Update::SetCamera { camera } = &update {
      sim_camera.view = Some(*camera);
    }
    apply_one(
      update,
      &mut commands,
      &mut registry,
      &mut meshes,
      &mut materials,
    );
  }
}

fn pull_batch(receiver: &UpdateReceiverResource) -> Option<UpdateBatch> {
  let guard = receiver.receiver.lock().ok()?;
  guard.drain_coalesced()
}

fn apply_one(
  update: Update,
  commands: &mut Commands,
  registry: &mut RenderRegistry,
  meshes: &mut Assets<Mesh>,
  materials: &mut Assets<StandardMaterial>,
) {
  match update {
    Update::RegisterPalette { handle, palette } => {
      registry.palettes.insert(handle, palette);
    }
    Update::FreePalette { handle } => {
      registry.palettes.remove(&handle);
    }

    Update::RegisterWorld {
      handle, transform, ..
    } => {
      let entity = commands
        .spawn((
          to_bevy_transform(&transform),
          GlobalTransform::default(),
          Visibility::default(),
        ))
        .id();
      registry.worlds.insert(handle, WorldEntry { entity });
    }
    Update::UpdateWorldTransform {
      handle, transform, ..
    } => {
      if let Some(entry) = registry.worlds.get(&handle) {
        commands
          .entity(entry.entity)
          .insert(to_bevy_transform(&transform));
      }
    }
    Update::FreeWorld { handle } => {
      if let Some(entry) = registry.worlds.remove(&handle) {
        commands.entity(entry.entity).despawn();
      }
      // Cascade: drop meshes/layers/bindings whose world is gone.
      registry.meshes.retain(|_, mesh_entry| {
        let keep = mesh_entry.world != handle;
        if !keep {
          commands.entity(mesh_entry.entity).despawn();
        }
        keep
      });
      registry.bindings.retain(|mesh, _| {
        registry
          .meshes
          .contains_key(mesh)
          .then_some(true)
          .unwrap_or(false)
      });
      registry
        .layers
        .retain(|_, l| registry.meshes.contains_key(&l.target));
    }

    Update::RegisterMesh {
      handle,
      world,
      id,
      label,
      source: _,
      geometry,
      transform,
      ..
    } => {
      let world_entity = registry.worlds.get(&world).map(|w| w.entity);
      let bevy_mesh = build_bevy_mesh(&geometry);
      let vertex_count = bevy_mesh.count_vertices();
      let vertex_to_cell = vertex_to_cell_for(&geometry);
      let base_positions = base_positions_for(&geometry);
      let mesh_handle = meshes.add(bevy_mesh);
      let material_handle = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        perceptual_roughness: 1.0,
        ..default()
      });

      let mut mesh_cmd = commands.spawn((
        Mesh3d(mesh_handle.clone()),
        MeshMaterial3d(material_handle.clone()),
        to_bevy_transform(&transform),
        GlobalTransform::default(),
        Visibility::default(),
        Name::new(label),
      ));
      if let Some(world_entity) = world_entity {
        mesh_cmd.insert(ChildOf(world_entity));
      }
      let entity = mesh_cmd.id();

      registry.meshes.insert(
        handle,
        MeshEntry {
          entity,
          mesh_handle,
          material_handle,
          world,
          render_id: id,
          vertex_count,
          vertex_to_cell,
          base_positions,
        },
      );
    }
    Update::UpdateMeshGeometry {
      handle,
      geometry,
      epoch: _,
    } => {
      if let Some(entry) = registry.meshes.get_mut(&handle) {
        let bevy_mesh = build_bevy_mesh(&geometry);
        entry.vertex_count = bevy_mesh.count_vertices();
        entry.vertex_to_cell = vertex_to_cell_for(&geometry);
        entry.base_positions = base_positions_for(&geometry);
        if let Some(asset) = meshes.get_mut(&entry.mesh_handle) {
          *asset = bevy_mesh;
        }
        registry.dirty_meshes.insert(handle);
        // New base geometry invalidates any prior displacement.
        if registry.displacements.contains_key(&handle) {
          registry.dirty_displacements.insert(handle);
        }
      }
    }
    Update::UpdateMeshTransform {
      handle, transform, ..
    } => {
      if let Some(entry) = registry.meshes.get(&handle) {
        commands
          .entity(entry.entity)
          .insert(to_bevy_transform(&transform));
      }
    }
    Update::FreeMesh { handle } => {
      if let Some(entry) = registry.meshes.remove(&handle) {
        commands.entity(entry.entity).despawn();
      }
      registry.bindings.remove(&handle);
      registry.dirty_meshes.remove(&handle);
      registry.layers.retain(|_, l| l.target != handle);
    }

    Update::RegisterLayer {
      handle,
      target,
      kind,
      ..
    } => {
      let cache: LayerKindCache = (&kind).into();
      registry.layers.insert(
        handle,
        LayerEntry {
          target,
          kind: cache,
          samples: None,
        },
      );
      // Auto-bind the first scalar layer that lands for a mesh; later
      // updates can re-bind explicitly.
      if matches!(kind, LayerKind::Scalar { .. }) {
        registry.bindings.entry(target).or_insert(handle);
      }
    }
    Update::UpdateLayerSamples {
      handle, samples, ..
    } => {
      if let Some(entry) = registry.layers.get_mut(&handle) {
        entry.samples = Some(samples);
      }
      registry.mark_layer_dirty(handle);
      registry.mark_displacement_dirty(handle);
    }
    Update::UpdateLayerPalette { handle, palette } => {
      if let Some(entry) = registry.layers.get_mut(&handle) {
        if let LayerKindCache::Scalar { palette: slot, .. } = &mut entry.kind {
          *slot = palette;
        }
      }
      registry.mark_layer_dirty(handle);
    }
    Update::UpdateLayerBinding { mesh, layer } => {
      match layer {
        Some(layer) => {
          registry.bindings.insert(mesh, layer);
        }
        None => {
          registry.bindings.remove(&mesh);
        }
      }
      registry.dirty_meshes.insert(mesh);
    }
    Update::FreeLayer { handle } => {
      if let Some(entry) = registry.layers.remove(&handle) {
        // Drop bindings pointing at the freed layer.
        registry.bindings.retain(|_, l| *l != handle);
        registry.displacements.retain(|_, d| d.layer != handle);
        registry.dirty_meshes.insert(entry.target);
      }
    }

    Update::SetMeshDisplacement { mesh, layer, scale } => {
      registry
        .displacements
        .insert(mesh, DisplacementBinding { layer, scale });
      // Apply now (the driving layer's samples may already be present).
      registry.dirty_displacements.insert(mesh);
    }

    Update::UpdateSunDirection { .. } => {
      // Handled in `apply_updates_system` (it owns the `SunDirection`
      // resource); the sun direction does not touch the registry/ECS here.
    }
    Update::SetCamera { .. } => {
      // Handled in `apply_updates_system` (it owns the `SimCamera` resource);
      // the camera is applied to the `SimDrivenCamera` by a dedicated system.
    }
    Update::SetSimTime { .. } => {
      // Wall-clock progression is the driver's concern, not the
      // backend's. Skip.
    }
  }
}

/// Translate an eidolon `RenderGeometry` into a bevy `Mesh`. Per-cell
/// faceted shading: every triangle's three vertices share the cell's
/// colour during paint.
fn build_bevy_mesh(geometry: &RenderGeometry) -> Mesh {
  match geometry {
    RenderGeometry::Triangles(tri) => {
      let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
      );
      let positions: Vec<[f32; 3]> = tri.positions.clone();
      let normals: Vec<[f32; 3]> = if tri.normals.len() == positions.len() {
        tri.normals.clone()
      } else {
        vec![[0.0, 0.0, 1.0]; positions.len()]
      };
      let colours: Vec<[f32; 4]> =
        tri.colours.iter().map(|c| [c.r, c.g, c.b, c.a]).collect();
      let colours = if colours.len() == positions.len() {
        colours
      } else {
        vec![[1.0, 1.0, 1.0, 1.0]; positions.len()]
      };

      mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
      mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
      mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colours);
      mesh.insert_indices(Indices::U32(tri.indices.clone()));
      mesh
    }
    RenderGeometry::Lines(lines) => {
      let mut mesh =
        Mesh::new(PrimitiveTopology::LineList, RenderAssetUsages::default());
      let positions: Vec<[f32; 3]> = lines.positions.clone();
      let colours: Vec<[f32; 4]> =
        lines.colours.iter().map(|c| [c.r, c.g, c.b, c.a]).collect();
      let colours = if colours.len() == positions.len() {
        colours
      } else {
        vec![[1.0, 1.0, 1.0, 1.0]; positions.len()]
      };
      let indices: Vec<u32> =
        lines.segments.iter().flat_map(|[a, b]| [*a, *b]).collect();
      mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
      mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colours);
      mesh.insert_indices(Indices::U32(indices));
      mesh
    }
    RenderGeometry::Points(points) => {
      let mut mesh =
        Mesh::new(PrimitiveTopology::PointList, RenderAssetUsages::default());
      let positions: Vec<[f32; 3]> = points.positions.clone();
      let colours: Vec<[f32; 4]> = points
        .colours
        .iter()
        .map(|c| [c.r, c.g, c.b, c.a])
        .collect();
      let colours = if colours.len() == positions.len() {
        colours
      } else {
        vec![[1.0, 1.0, 1.0, 1.0]; positions.len()]
      };
      mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
      mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colours);
      mesh
    }
    RenderGeometry::Packed(_) => {
      // Packed asset blobs aren't supported by the bevy backend yet —
      // they'd need a deserialiser that knows the bevy mesh format.
      // Emit an empty placeholder so the apply system doesn't crash.
      Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
      )
    }
  }
}

/// Capture the undisplaced vertex positions of a triangle mesh so the displace
/// system can recompute relief from a stable base. Non-triangle geometry isn't
/// displaceable, so it returns empty.
fn base_positions_for(geometry: &RenderGeometry) -> Vec<[f32; 3]> {
  match geometry {
    RenderGeometry::Triangles(tri) => tri.positions.clone(),
    _ => Vec::new(),
  }
}

/// Build a per-vertex cell-id lookup for the geometry. Triangles and
/// lines store one id per primitive; we expand it to per-vertex so
/// the paint system can resolve a sample with a single index.
fn vertex_to_cell_for(geometry: &RenderGeometry) -> Vec<Option<usize>> {
  match geometry {
    RenderGeometry::Triangles(tri) => {
      let triangle_count = tri.indices.len() / 3;
      let mut out = vec![None; tri.positions.len()];
      for triangle_idx in 0..triangle_count {
        let cell: Option<usize> = tri
          .cell_ids
          .get(triangle_idx)
          .and_then(|c| c.map(|c| c.index()));
        for v in 0..3 {
          let vertex_index = tri.indices[triangle_idx * 3 + v] as usize;
          if vertex_index < out.len() {
            out[vertex_index] = cell;
          }
        }
      }
      out
    }
    RenderGeometry::Lines(lines) => {
      // Lines have no per-segment cell association in the IR yet, so
      // every vertex maps to None; paint won't recolour line meshes.
      vec![None; lines.positions.len()]
    }
    RenderGeometry::Points(points) => points
      .cell_ids
      .iter()
      .map(|c: &Option<utility::domain::CellId>| c.map(|cid| cid.index()))
      .collect(),
    RenderGeometry::Packed(_) => Vec::new(),
  }
}
