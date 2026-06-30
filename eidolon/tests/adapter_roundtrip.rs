// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Phase 2C: dumb snapshot -> updates -> registry -> snapshot
//! roundtrip. The adapter is "correct" if feeding its output into a
//! BackendRegistry rebuilds the original RenderFrame's content (set-
//! equality, since registry ordering is per-handle not per-input).

use eidolon::{
  extract::frame_to_initial_batch,
  ir::{
    LayerId, LayerSource, MeshRepresentation, MeshSource, Palette, PointCloud,
    RenderFrame, RenderGeometry, RenderLayer, RenderMesh, RenderMeshId,
    RenderWorld, Rgba, ScalarLayer, ScalarSamples, Transform, TriangleMesh,
  },
  registry::BackendRegistry,
};
use utility::domain::{FieldKey, FieldName, MeshKey, WorldId};

fn earth_frame() -> RenderFrame {
  let world_id = WorldId(0);
  let surface_id = RenderMeshId {
    world: world_id,
    mesh: MeshKey::SURFACE,
    representation: MeshRepresentation::BoundaryFaces,
  };
  let surface_centroids_id = RenderMeshId {
    world: world_id,
    mesh: MeshKey::SURFACE,
    representation: MeshRepresentation::Cells,
  };

  let surface_mesh = RenderMesh::new(
    surface_id,
    "earth surface",
    MeshSource::TesseraMesh(MeshKey::SURFACE),
    RenderGeometry::Triangles(TriangleMesh {
      positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
      normals: vec![[0.0, 0.0, 1.0]; 3],
      colours: vec![Rgba::WHITE; 3],
      indices: vec![0, 1, 2],
      cell_ids: vec![Some(0usize.into())],
      face_ids: vec![Some(0usize.into())],
    }),
  );
  let centroid_mesh = RenderMesh::new(
    surface_centroids_id,
    "centroids",
    MeshSource::TesseraMesh(MeshKey::SURFACE),
    RenderGeometry::Points(PointCloud {
      positions: vec![[0.5, 0.5, 0.0]],
      colours: vec![Rgba::CYAN],
      cell_ids: vec![Some(0usize.into())],
      face_ids: vec![None],
    }),
  );

  let temperature = ScalarLayer::new(
    LayerId::from_static("surface_temperature"),
    "surface_temperature",
    surface_id,
    LayerSource::Field(FieldKey::new(MeshKey::SURFACE, FieldName::Temperature)),
    ScalarSamples::PerCell(vec![288.0, 290.0, 285.0]),
  );
  let pressure = ScalarLayer::new(
    LayerId::from_static("surface_pressure"),
    "surface_pressure",
    surface_id,
    LayerSource::Field(FieldKey::new(MeshKey::SURFACE, FieldName::Pressure)),
    ScalarSamples::PerCell(vec![101_000.0]),
  );

  let mut world = RenderWorld::new(world_id);
  world.label = "earth".into();
  world.transform = Transform::translation_scaling([1.0e7, 0.0, 0.0], 6.371e6);
  world.transform_epoch = 1;
  world.meshes = vec![surface_mesh, centroid_mesh];
  world.layers = vec![
    RenderLayer::Scalar(temperature),
    RenderLayer::Scalar(pressure),
  ];

  RenderFrame {
    frame: 42,
    sim_time: 1.5,
    worlds: vec![world],
    camera: None,
  }
}

#[test]
fn initial_batch_round_trips_through_registry() {
  let frame = earth_frame();
  let batch = frame_to_initial_batch(&frame);

  let mut registry = BackendRegistry::new();
  let errors = registry.apply(&batch);
  assert!(errors.is_empty(), "{:?}", errors);

  let snap = registry.snapshot();
  assert_eq!(snap.frame, frame.frame);
  assert_eq!(snap.sim_time, frame.sim_time);
  assert_eq!(snap.worlds.len(), 1);

  let original = &frame.worlds[0];
  let recovered = &snap.worlds[0];
  assert_eq!(recovered.id, original.id);
  assert_eq!(recovered.label, original.label);
  assert_eq!(recovered.transform, original.transform);
  assert_eq!(recovered.transform_epoch, original.transform_epoch);
  assert_eq!(recovered.meshes.len(), original.meshes.len());
  assert_eq!(recovered.layers.len(), original.layers.len());

  // Mesh content survives byte-for-byte.
  for original_mesh in &original.meshes {
    let recovered_mesh = recovered
      .meshes
      .iter()
      .find(|m| m.id == original_mesh.id)
      .expect("mesh not present in roundtrip snapshot");
    assert_eq!(recovered_mesh.label, original_mesh.label);
    assert_eq!(recovered_mesh.geometry, original_mesh.geometry);
    assert_eq!(recovered_mesh.transform, original_mesh.transform);
    assert_eq!(recovered_mesh.epoch, original_mesh.epoch);
    assert_eq!(
      recovered_mesh.transform_epoch,
      original_mesh.transform_epoch
    );
  }

  // Layer content (id, target, samples, palette name) survives.
  for original_layer in &original.layers {
    let RenderLayer::Scalar(o) = original_layer else {
      panic!("test only emits scalar layers")
    };
    let recovered_layer = recovered
      .layers
      .iter()
      .find_map(|l| match l {
        RenderLayer::Scalar(s) if s.id == o.id => Some(s),
        _ => None,
      })
      .expect("layer missing from roundtrip");
    assert_eq!(recovered_layer.target, o.target);
    assert_eq!(recovered_layer.samples, o.samples);
    assert_eq!(recovered_layer.palette.name, Palette::diagnostic().name);
    // Default palette since `ScalarLayer::new` doesn't set one;
    // pinning that to make sure the adapter doesn't drop the choice.
  }
}

#[test]
fn empty_frame_emits_only_set_sim_time() {
  let frame = RenderFrame {
    frame: 7,
    sim_time: 0.25,
    worlds: Vec::new(),
    camera: None,
  };
  let batch = frame_to_initial_batch(&frame);
  assert_eq!(batch.updates.len(), 1);
  assert!(matches!(
    batch.updates[0],
    eidolon::ir::Update::SetSimTime { .. }
  ));
}
