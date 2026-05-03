// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Phase 2C: replace-batch wipes prev state and reflects only the new
//! frame.

use eidolon::{
  extract::{frame_to_initial_batch, frame_to_replace_batch},
  ir::{
    LayerId, LayerSource, MeshRepresentation, MeshSource, RenderFrame,
    RenderGeometry, RenderLayer, RenderMesh, RenderMeshId, RenderWorld,
    ScalarLayer, ScalarSamples, TriangleMesh,
  },
  registry::BackendRegistry,
};
use utility::domain::{FieldKey, FieldName, MeshKey, WorldId};

fn frame_with_temperature(world_id: WorldId, samples: Vec<f64>) -> RenderFrame {
  let mesh_id = RenderMeshId {
    world: world_id,
    mesh: MeshKey::SURFACE,
    representation: MeshRepresentation::BoundaryFaces,
  };
  let mesh = RenderMesh::new(
    mesh_id,
    "surface",
    MeshSource::TesseraMesh(MeshKey::SURFACE),
    RenderGeometry::Triangles(TriangleMesh::default()),
  );
  let layer = ScalarLayer::new(
    LayerId::from_static("surface_temperature"),
    "surface_temperature",
    mesh_id,
    LayerSource::Field(FieldKey::new(MeshKey::SURFACE, FieldName::Temperature)),
    ScalarSamples::PerCell(samples),
  );
  let mut world = RenderWorld::new(world_id);
  world.meshes = vec![mesh];
  world.layers = vec![RenderLayer::Scalar(layer)];
  RenderFrame {
    frame: 0,
    sim_time: 0.0,
    worlds: vec![world],
  }
}

#[test]
fn replace_batch_drops_prev_world_and_registers_new() {
  let prev_world_id = WorldId(0);
  let new_world_id = WorldId(1);
  let prev = frame_with_temperature(prev_world_id, vec![10.0, 20.0]);
  let new = frame_with_temperature(new_world_id, vec![100.0]);

  let mut registry = BackendRegistry::new();
  registry.apply(&frame_to_initial_batch(&prev));
  registry.apply(&frame_to_replace_batch(&new, &prev));

  let snap = registry.snapshot();
  assert_eq!(snap.worlds.len(), 1, "prev world should have been freed");
  assert_eq!(snap.worlds[0].id, new_world_id);

  let RenderLayer::Scalar(layer) = &snap.worlds[0].layers[0] else {
    panic!("expected scalar layer")
  };
  assert_eq!(layer.samples, ScalarSamples::PerCell(vec![100.0]));
}

#[test]
fn replace_batch_with_same_world_id_overwrites_state() {
  let world_id = WorldId(0);
  let prev = frame_with_temperature(world_id, vec![288.0]);
  let new = frame_with_temperature(world_id, vec![300.0, 305.0, 310.0]);

  let mut registry = BackendRegistry::new();
  registry.apply(&frame_to_initial_batch(&prev));
  let errors = registry.apply(&frame_to_replace_batch(&new, &prev));
  assert!(errors.is_empty(), "{errors:?}");

  let snap = registry.snapshot();
  assert_eq!(snap.worlds.len(), 1);
  let RenderLayer::Scalar(layer) = &snap.worlds[0].layers[0] else {
    panic!("expected scalar layer")
  };
  assert_eq!(
    layer.samples,
    ScalarSamples::PerCell(vec![300.0, 305.0, 310.0])
  );
}
