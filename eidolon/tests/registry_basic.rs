// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Phase 2B: BackendRegistry round-trip — Register/Update/Free a world,
//! mesh, palette and scalar layer; assert the snapshot reconstructs the
//! corresponding RenderFrame.

use eidolon::ir::{
  LayerHandle, LayerId, LayerKind, LayerSamples, LayerSource, MeshHandle,
  MeshRepresentation, MeshSource, Palette, PaletteHandle, RenderGeometry,
  RenderLayer, RenderMeshId, ScalarSamples, Transform, TriangleMesh, Update,
  UpdateBatch, WorldHandle,
};
use eidolon::registry::BackendRegistry;
use utility::domain::{FieldKey, FieldName, MeshKey, WorldId};

fn surface_mesh_id(world: WorldId) -> RenderMeshId {
  RenderMeshId {
    world,
    mesh: MeshKey::SURFACE,
    representation: MeshRepresentation::BoundaryFaces,
  }
}

fn make_batch(updates: Vec<Update>) -> UpdateBatch {
  UpdateBatch {
    frame: 1,
    sim_time: 0.5,
    updates,
  }
}

#[test]
fn lifecycle_round_trip_into_snapshot() {
  let world_id = WorldId(0);
  let world = WorldHandle(0xa1);
  let palette = PaletteHandle(0xb2);
  let mesh_id = surface_mesh_id(world_id);
  let mesh = mesh_id.handle();
  let layer = LayerHandle(0xc3);
  let layer_id = LayerId::from_static("surface_temperature");

  let mut registry = BackendRegistry::new();
  let errors = registry.apply(&make_batch(vec![
    Update::RegisterPalette {
      handle: palette,
      palette: Palette::thermal(),
    },
    Update::RegisterWorld {
      handle: world,
      world_id,
      label: "earth".into(),
      transform: Transform::translation_scaling([0.0, 0.0, 0.0], 1.0e6),
      transform_epoch: 0,
    },
    Update::RegisterMesh {
      handle: mesh,
      world,
      id: mesh_id,
      label: "surface".into(),
      source: MeshSource::TesseraMesh(MeshKey::SURFACE),
      geometry: RenderGeometry::Triangles(TriangleMesh::default()),
      transform: Transform::IDENTITY,
      geometry_epoch: 0,
      transform_epoch: 0,
    },
    Update::RegisterLayer {
      handle: layer,
      id: layer_id,
      label: "surface_temperature".into(),
      target: mesh,
      source: LayerSource::Field(FieldKey::new(
        MeshKey::SURFACE,
        FieldName::Temperature,
      )),
      kind: LayerKind::Scalar {
        palette: Some(palette),
        range: None,
      },
    },
    Update::UpdateLayerSamples {
      handle: layer,
      samples: LayerSamples::Scalar(ScalarSamples::PerCell(vec![
        1.0, 2.0, 3.0,
      ])),
      epoch: 1,
    },
    Update::UpdateLayerBinding {
      mesh,
      layer: Some(layer),
    },
    Update::SetSimTime {
      sim_time: 0.5,
      frame: 1,
    },
  ]));
  assert!(errors.is_empty(), "{:?}", errors);

  assert_eq!(registry.binding_for(mesh), Some(layer));
  assert!(registry.palette(palette).is_some());

  let snapshot = registry.snapshot();
  assert_eq!(snapshot.frame, 1);
  assert_eq!(snapshot.sim_time, 0.5);
  assert_eq!(snapshot.worlds.len(), 1);

  let render_world = &snapshot.worlds[0];
  assert_eq!(render_world.id, world_id);
  assert_eq!(render_world.label, "earth");
  assert_eq!(render_world.transform.scale, 1.0e6);
  assert_eq!(render_world.meshes.len(), 1);
  assert_eq!(render_world.layers.len(), 1);

  match &render_world.layers[0] {
    RenderLayer::Scalar(layer) => {
      assert_eq!(layer.id, layer_id);
      assert_eq!(layer.target, mesh_id);
      assert_eq!(layer.samples, ScalarSamples::PerCell(vec![1.0, 2.0, 3.0]));
      assert_eq!(layer.palette.name, Palette::thermal().name);
    }
    _ => panic!("expected scalar layer in snapshot"),
  }
}

#[test]
fn update_layer_samples_replaces_payload() {
  let world_id = WorldId(0);
  let world = WorldHandle(1);
  let mesh_id = surface_mesh_id(world_id);
  let mesh = mesh_id.handle();
  let layer = LayerHandle(2);
  let layer_id = LayerId::from_static("foo");

  let mut registry = BackendRegistry::new();
  registry.apply(&make_batch(vec![
    Update::RegisterWorld {
      handle: world,
      world_id,
      label: "w".into(),
      transform: Transform::IDENTITY,
      transform_epoch: 0,
    },
    Update::RegisterMesh {
      handle: mesh,
      world,
      id: mesh_id,
      label: "m".into(),
      source: MeshSource::TesseraMesh(MeshKey::SURFACE),
      geometry: RenderGeometry::Triangles(TriangleMesh::default()),
      transform: Transform::IDENTITY,
      geometry_epoch: 0,
      transform_epoch: 0,
    },
    Update::RegisterLayer {
      handle: layer,
      id: layer_id,
      label: "foo".into(),
      target: mesh,
      source: LayerSource::Derived(0),
      kind: LayerKind::Scalar {
        palette: None,
        range: None,
      },
    },
    Update::UpdateLayerSamples {
      handle: layer,
      samples: LayerSamples::Scalar(ScalarSamples::PerCell(vec![10.0])),
      epoch: 1,
    },
  ]));
  // Replace samples in a second batch.
  registry.apply(&UpdateBatch {
    frame: 2,
    sim_time: 1.0,
    updates: vec![Update::UpdateLayerSamples {
      handle: layer,
      samples: LayerSamples::Scalar(ScalarSamples::PerCell(vec![20.0, 30.0])),
      epoch: 2,
    }],
  });

  let snap = registry.snapshot();
  let RenderLayer::Scalar(layer) = &snap.worlds[0].layers[0] else {
    panic!("expected scalar layer")
  };
  assert_eq!(layer.samples, ScalarSamples::PerCell(vec![20.0, 30.0]));
}

#[test]
fn world_transform_update_bumps_epoch_in_snapshot() {
  let world_id = WorldId(0);
  let world = WorldHandle(7);
  let mut registry = BackendRegistry::new();
  registry.apply(&make_batch(vec![Update::RegisterWorld {
    handle: world,
    world_id,
    label: "w".into(),
    transform: Transform::IDENTITY,
    transform_epoch: 0,
  }]));
  registry.apply(&UpdateBatch {
    frame: 5,
    sim_time: 2.5,
    updates: vec![Update::UpdateWorldTransform {
      handle: world,
      transform: Transform::translation([1.0, 2.0, 3.0]),
      transform_epoch: 7,
    }],
  });
  let snap = registry.snapshot();
  let w = &snap.worlds[0];
  assert_eq!(w.transform.centre, [1.0, 2.0, 3.0]);
  assert_eq!(w.transform_epoch, 7);
}
