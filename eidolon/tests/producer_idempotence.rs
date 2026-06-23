// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Phase 2D: a second `extract()` on unchanged sim state emits only
//! the always-on lifecycle (`SetSimTime`). No re-Register, no
//! UpdateMeshGeometry, no UpdateLayerSamples.

use std::sync::Arc;

use eidolon::extract::{
  ExtractConfig, FrameProducer, MeshConfig, ScalarLayerConfig,
};
use eidolon::ir::{MeshRepresentation, Palette, Update};
use pleroma::{Pleroma, core::storage::SoaField};
use tessera::cube_sphere::{CubeSphere, CubeSphereShellSpec};
use tessera::geometry::CellGeometry;
use tessera::mesh::Mesh;
use tessera::world_mesh::Tessera;
use utility::domain::{BoundaryTag, FieldKey, FieldName, MeshKey, WorldId};

fn build_world() -> (Tessera, Pleroma, usize) {
  let surface = Arc::new(CubeSphere::shell(
    CubeSphereShellSpec::uniform([1, 1, 1], 0.9, 1.0)
      .with_boundaries(BoundaryTag::Ground, BoundaryTag::AtmosphereEdge),
  ));
  let cell_count = surface.cell_count();
  let mut tessera = Tessera::new();
  tessera.register_mesh(MeshKey::SURFACE, surface as Arc<dyn Mesh<3>>);

  let mut pleroma = Pleroma::new();
  let temperature =
    SoaField::<1>::from_fn(cell_count, |cell| [288.0 + cell.index() as f64]);
  pleroma.register_field(
    FieldKey::new(MeshKey::SURFACE, FieldName::Temperature),
    temperature,
  );

  (tessera, pleroma, cell_count)
}

fn config() -> ExtractConfig {
  ExtractConfig {
    world_label: "earth".into(),
    world_scale: 1.0,
    meshes: vec![MeshConfig {
      mesh_key: MeshKey::SURFACE,
      representation: MeshRepresentation::BoundaryFaces,
      label: "earth surface".into(),
    }],
    layers: vec![ScalarLayerConfig {
      id: eidolon::ir::LayerId::from_static("surface_temperature"),
      label: "surface_temperature".into(),
      target_mesh: MeshKey::SURFACE,
      target_representation: MeshRepresentation::BoundaryFaces,
      field: FieldKey::new(MeshKey::SURFACE, FieldName::Temperature),
      component: 0,
      palette: Palette::diagnostic(),
    }],
    categorical_layers: vec![],
    track_sun_direction: false,
  }
}

#[test]
fn second_extract_with_no_changes_only_sets_sim_time() {
  let (tessera, pleroma, _) = build_world();
  let mut producer = FrameProducer::new(config());

  let first = producer.extract(WorldId(0), &tessera, &pleroma, None, 0.0, 0);
  // First batch must include lifecycle Register* updates.
  assert!(
    first
      .updates
      .iter()
      .any(|u| matches!(u, Update::RegisterWorld { .. })),
    "first batch should register the world"
  );
  assert!(
    first
      .updates
      .iter()
      .any(|u| matches!(u, Update::RegisterMesh { .. })),
    "first batch should register the surface mesh"
  );
  assert!(
    first
      .updates
      .iter()
      .any(|u| matches!(u, Update::UpdateLayerSamples { .. })),
    "first batch should ship initial samples"
  );

  let second = producer.extract(WorldId(0), &tessera, &pleroma, None, 1.0, 1);
  assert_eq!(
    second.updates.len(),
    1,
    "expected only SetSimTime, got {:?}",
    second.updates
  );
  assert!(matches!(second.updates[0], Update::SetSimTime { .. }));
}
