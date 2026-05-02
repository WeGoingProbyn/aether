// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Phase 2A: Transform + epochs are part of the snapshot IR. Backends
//! built later in the plan rely on these fields; this test pins their
//! presence and round-trips them through `RenderMesh::with_transform`.

use eidolon::ir::{
  MeshRepresentation, MeshSource, RenderGeometry, RenderMesh, RenderMeshId,
  RenderWorld, Transform, TriangleMesh,
};
use utility::domain::{MeshKey, WorldId};

#[test]
fn render_mesh_carries_transform_and_separate_epochs() {
  let id = RenderMeshId {
    world: WorldId(0),
    mesh: MeshKey::SURFACE,
    representation: MeshRepresentation::BoundaryFaces,
  };
  let mut mesh = RenderMesh::new(
    id,
    "earth surface",
    MeshSource::TesseraMesh(MeshKey::SURFACE),
    RenderGeometry::Triangles(TriangleMesh::default()),
  );
  // Default state: identity transform, both epochs at zero.
  assert_eq!(mesh.transform, Transform::IDENTITY);
  assert_eq!(mesh.epoch, 0);
  assert_eq!(mesh.transform_epoch, 0);

  // Move it: only the transform_epoch advances in the producer's
  // logic, but the IR field itself is publicly mutable so a hand-built
  // snapshot can carry whatever values the test needs.
  mesh = mesh
    .with_transform(Transform::translation_scaling([1.0e7, 0.0, 0.0], 6.371e6));
  mesh.transform_epoch = 1;

  assert_eq!(mesh.transform.centre, [1.0e7, 0.0, 0.0]);
  assert_eq!(mesh.transform.scale, 6.371e6);
  assert_eq!(mesh.epoch, 0); // geometry hasn't changed
  assert_eq!(mesh.transform_epoch, 1);
}

#[test]
fn render_world_constructor_starts_at_identity() {
  let w = RenderWorld::new(WorldId(7));
  assert_eq!(w.transform, Transform::IDENTITY);
  assert_eq!(w.transform_epoch, 0);
  assert!(w.meshes.is_empty());
  assert!(w.layers.is_empty());
  assert!(w.diagnostics.is_empty());
}

#[test]
fn translation_and_scaling_helpers_round_trip() {
  let t = Transform::translation([2.0, 4.0, 6.0]);
  assert_eq!(t.centre, [2.0, 4.0, 6.0]);
  assert_eq!(t.scale, 1.0);

  let s = Transform::scaling(0.5);
  assert_eq!(s.centre, [0.0, 0.0, 0.0]);
  assert_eq!(s.scale, 0.5);

  let ts = Transform::translation_scaling([1.0, 2.0, 3.0], 10.0);
  assert_eq!(ts.centre, [1.0, 2.0, 3.0]);
  assert_eq!(ts.scale, 10.0);
}
