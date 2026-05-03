// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Phase 4: smoke-test the bevy backend against a hand-built
//! `UpdateBatch`. Builds a `MinimalPlugins`-only `App`, inserts the
//! `AetherBevyPlugin`, pushes a Register/Update sequence, ticks once
//! and asserts the registry shows the expected ECS state.

#![cfg(feature = "bevy")]

use bevy::asset::AssetPlugin;
use bevy::prelude::*;

use eidolon::{
  bevy::{AetherBevyPlugin, RenderRegistry},
  ir::{
    LayerHandle, LayerId, LayerKind, LayerSamples, MeshHandle,
    MeshRepresentation, MeshSource, Palette, PaletteHandle, RenderGeometry,
    RenderMeshId, Rgba, ScalarSamples, Transform, TriangleMesh, Update,
    UpdateBatch, WorldHandle,
  },
  runtime::render_channel,
};
use utility::domain::{CellId, MeshKey, WorldId};

#[test]
fn apply_system_spawns_world_mesh_and_layer_entities() {
  let (tx, rx) = render_channel(4);

  let palette_handle = PaletteHandle::from_static_name("test");
  let world_handle = WorldHandle::from_world_id(WorldId(7));
  let mesh_handle = MeshHandle(0xABCD);
  let layer_handle =
    LayerHandle::for_target(LayerId::from_static("temperature"), mesh_handle);

  let triangle_mesh = TriangleMesh {
    positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    normals: vec![[0.0, 0.0, 1.0]; 3],
    colours: vec![Rgba::WHITE; 3],
    indices: vec![0, 1, 2],
    cell_ids: vec![Some(CellId::from(0))],
    face_ids: vec![None],
  };

  tx.send(UpdateBatch {
    frame: 0,
    sim_time: 0.0,
    updates: vec![
      Update::RegisterPalette {
        handle: palette_handle,
        palette: Palette::diagnostic(),
      },
      Update::RegisterWorld {
        handle: world_handle,
        world_id: WorldId(7),
        label: "earth".into(),
        transform: Transform::IDENTITY,
        transform_epoch: 1,
      },
      Update::RegisterMesh {
        handle: mesh_handle,
        world: world_handle,
        id: RenderMeshId {
          world: WorldId(7),
          mesh: MeshKey::SURFACE,
          representation: MeshRepresentation::BoundaryFaces,
        },
        label: "earth surface".into(),
        source: MeshSource::TesseraMesh(MeshKey::SURFACE),
        geometry: RenderGeometry::Triangles(triangle_mesh),
        transform: Transform::IDENTITY,
        geometry_epoch: 1,
        transform_epoch: 1,
      },
      Update::RegisterLayer {
        handle: layer_handle,
        id: LayerId::from_static("temperature"),
        label: "temperature".into(),
        target: mesh_handle,
        source: eidolon::ir::LayerSource::Derived(0),
        kind: LayerKind::Scalar {
          palette: Some(palette_handle),
          range: None,
        },
      },
      Update::UpdateLayerSamples {
        handle: layer_handle,
        samples: LayerSamples::Scalar(ScalarSamples::PerCell(vec![288.0])),
        epoch: 1,
      },
      Update::SetSimTime {
        sim_time: 0.0,
        frame: 0,
      },
    ],
  })
  .unwrap();

  // Drop the sender so the channel doesn't block the bevy thread when
  // the test exits.
  drop(tx);

  let mut app = App::new();
  app
    .add_plugins(MinimalPlugins)
    .add_plugins(AssetPlugin::default())
    .init_asset::<Mesh>()
    .init_asset::<StandardMaterial>()
    .init_asset::<bevy::image::Image>()
    .add_plugins(AetherBevyPlugin::new(rx));

  app.update();

  let registry = app.world().resource::<RenderRegistry>();
  assert!(
    registry.worlds.contains_key(&world_handle),
    "world entity should be registered"
  );
  assert!(
    registry.meshes.contains_key(&mesh_handle),
    "mesh entity should be registered"
  );
  assert!(
    registry.layers.contains_key(&layer_handle),
    "layer should be cached"
  );
  assert_eq!(
    registry.bindings.get(&mesh_handle).copied(),
    Some(layer_handle),
    "registering a scalar layer should auto-bind it to the mesh"
  );

  // Run paint by ticking once more; the dirty flag was set by
  // UpdateLayerSamples and the paint system runs on Update.
  app.update();

  let registry = app.world().resource::<RenderRegistry>();
  let mesh_entry = &registry.meshes[&mesh_handle];
  let meshes = app.world().resource::<Assets<Mesh>>();
  let mesh = meshes
    .get(&mesh_entry.mesh_handle)
    .expect("mesh asset present");
  let colours = mesh
    .attribute(Mesh::ATTRIBUTE_COLOR)
    .expect("paint should have written colours");
  assert_eq!(colours.len(), mesh_entry.vertex_count);
}
