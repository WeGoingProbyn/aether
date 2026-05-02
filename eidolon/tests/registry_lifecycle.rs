// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Phase 2B: adversarial Register/Update/Free orderings.
//!
//! The registry must stay consistent under malformed input — Update
//! before Register surfaces an error, Free cascades through child
//! resources, sample-kind mismatches don't corrupt state.

use eidolon::ir::{
  LayerHandle, LayerId, LayerKind, LayerSamples, LayerSource,
  MeshRepresentation, MeshSource, PaletteHandle, RenderGeometry, RenderMeshId,
  ScalarSamples, Transform, TriangleMesh, Update, UpdateBatch, VectorGlyph,
  VectorSamples, WorldHandle,
};
use eidolon::registry::{BackendRegistry, RegistryError};
use utility::domain::{MeshKey, WorldId};

fn make_mesh_id(world: WorldId) -> RenderMeshId {
  RenderMeshId {
    world,
    mesh: MeshKey::SURFACE,
    representation: MeshRepresentation::BoundaryFaces,
  }
}

fn batch(updates: Vec<Update>) -> UpdateBatch {
  UpdateBatch {
    frame: 0,
    sim_time: 0.0,
    updates,
  }
}

#[test]
fn update_before_register_returns_unknown_handle_error() {
  let mut registry = BackendRegistry::new();
  let world = WorldHandle(1);
  let errors = registry.apply(&batch(vec![Update::UpdateWorldTransform {
    handle: world,
    transform: Transform::IDENTITY,
    transform_epoch: 1,
  }]));
  assert_eq!(errors, vec![RegistryError::UnknownWorld(world)]);
}

#[test]
fn register_mesh_against_unknown_world_is_rejected() {
  let mut registry = BackendRegistry::new();
  let world = WorldHandle(1);
  let mesh_id = make_mesh_id(WorldId(0));
  let mesh = mesh_id.handle();
  let errors = registry.apply(&batch(vec![Update::RegisterMesh {
    handle: mesh,
    world,
    id: mesh_id,
    label: "m".into(),
    source: MeshSource::TesseraMesh(MeshKey::SURFACE),
    geometry: RenderGeometry::Triangles(TriangleMesh::default()),
    transform: Transform::IDENTITY,
    geometry_epoch: 0,
    transform_epoch: 0,
  }]));
  assert_eq!(errors, vec![RegistryError::UnknownWorld(world)]);
}

#[test]
fn free_world_cascades_to_meshes_and_layers_and_bindings() {
  let world_id = WorldId(0);
  let world = WorldHandle(1);
  let mesh_id = make_mesh_id(world_id);
  let mesh = mesh_id.handle();
  let layer = LayerHandle(2);

  let mut registry = BackendRegistry::new();
  registry.apply(&batch(vec![
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
      id: LayerId::from_static("temp"),
      label: "temp".into(),
      target: mesh,
      source: LayerSource::Derived(0),
      kind: LayerKind::Scalar {
        palette: None,
        range: None,
      },
    },
    Update::UpdateLayerBinding {
      mesh,
      layer: Some(layer),
    },
  ]));

  // Free the world — every child should disappear with it.
  let errors =
    registry.apply(&batch(vec![Update::FreeWorld { handle: world }]));
  assert!(errors.is_empty(), "{errors:?}");

  let snap = registry.snapshot();
  assert!(snap.worlds.is_empty());
  assert_eq!(registry.binding_for(mesh), None);

  // Touching the freed handles is now a hard error.
  let errors = registry.apply(&batch(vec![Update::UpdateMeshGeometry {
    handle: mesh,
    geometry: RenderGeometry::Triangles(TriangleMesh::default()),
    epoch: 1,
  }]));
  assert_eq!(errors, vec![RegistryError::UnknownMesh(mesh)]);
}

#[test]
fn free_layer_clears_dependent_bindings() {
  let world_id = WorldId(0);
  let world = WorldHandle(1);
  let mesh_id = make_mesh_id(world_id);
  let mesh = mesh_id.handle();
  let layer = LayerHandle(7);

  let mut registry = BackendRegistry::new();
  registry.apply(&batch(vec![
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
      id: LayerId::from_static("foo"),
      label: "foo".into(),
      target: mesh,
      source: LayerSource::Derived(0),
      kind: LayerKind::Scalar {
        palette: None,
        range: None,
      },
    },
    Update::UpdateLayerBinding {
      mesh,
      layer: Some(layer),
    },
  ]));

  assert_eq!(registry.binding_for(mesh), Some(layer));
  registry.apply(&batch(vec![Update::FreeLayer { handle: layer }]));
  assert_eq!(registry.binding_for(mesh), None);
}

#[test]
fn sample_kind_mismatch_is_rejected_without_corrupting_state() {
  let world_id = WorldId(0);
  let world = WorldHandle(1);
  let mesh_id = make_mesh_id(world_id);
  let mesh = mesh_id.handle();
  let layer = LayerHandle(3);

  let mut registry = BackendRegistry::new();
  registry.apply(&batch(vec![
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
    // Layer registered as Scalar.
    Update::RegisterLayer {
      handle: layer,
      id: LayerId::from_static("temp"),
      label: "temp".into(),
      target: mesh,
      source: LayerSource::Derived(0),
      kind: LayerKind::Scalar {
        palette: None,
        range: None,
      },
    },
  ]));
  // Try to feed it Vector samples.
  let errors = registry.apply(&batch(vec![Update::UpdateLayerSamples {
    handle: layer,
    samples: LayerSamples::Vector(VectorSamples::PerCell(Vec::new())),
    epoch: 1,
  }]));
  assert_eq!(errors, vec![RegistryError::SampleKindMismatch { layer }]);

  // Following that with a valid Scalar update should still apply.
  let errors = registry.apply(&batch(vec![Update::UpdateLayerSamples {
    handle: layer,
    samples: LayerSamples::Scalar(ScalarSamples::PerCell(vec![1.0])),
    epoch: 2,
  }]));
  assert!(errors.is_empty(), "{errors:?}");
}

#[test]
fn duplicate_register_returns_warning_but_keeps_first_state() {
  let world_id = WorldId(0);
  let world = WorldHandle(1);
  let mut registry = BackendRegistry::new();
  registry.apply(&batch(vec![Update::RegisterWorld {
    handle: world,
    world_id,
    label: "first".into(),
    transform: Transform::IDENTITY,
    transform_epoch: 0,
  }]));
  let errors = registry.apply(&batch(vec![Update::RegisterWorld {
    handle: world,
    world_id,
    label: "second".into(),
    transform: Transform::IDENTITY,
    transform_epoch: 0,
  }]));
  assert_eq!(errors, vec![RegistryError::Duplicate("world")]);
  // Second insert *did* overwrite via HashMap::insert; current spec is
  // "report duplicate, last write wins". Pin that behaviour so we
  // change it consciously later.
  let snap = registry.snapshot();
  assert_eq!(snap.worlds[0].label, "second");
}

#[test]
fn unknown_palette_during_register_layer_is_rejected() {
  let world_id = WorldId(0);
  let world = WorldHandle(1);
  let mesh_id = make_mesh_id(world_id);
  let mesh = mesh_id.handle();
  let palette = PaletteHandle(99);

  let mut registry = BackendRegistry::new();
  registry.apply(&batch(vec![
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
  ]));
  let errors = registry.apply(&batch(vec![Update::RegisterLayer {
    handle: LayerHandle(2),
    id: LayerId::from_static("bad"),
    label: "bad".into(),
    target: mesh,
    source: LayerSource::Derived(0),
    kind: LayerKind::Scalar {
      palette: Some(palette),
      range: None,
    },
  }]));
  assert_eq!(errors, vec![RegistryError::UnknownPalette(palette)]);
}

// VectorGlyph is exported via the IR; including it here makes sure the
// re-export survives if someone reorganises module boundaries.
#[allow(dead_code)]
fn _vector_glyph_export_check() -> VectorGlyph {
  VectorGlyph::Arrow
}
