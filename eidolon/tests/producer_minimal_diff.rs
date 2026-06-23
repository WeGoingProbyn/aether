// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Phase 2D: when only the temperature samples change between two
//! `extract()` calls, the diff batch contains exactly one
//! `UpdateLayerSamples` and *no* `UpdateMeshGeometry` /
//! `UpdateWorldTransform` / `UpdateMeshTransform`.

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
  pleroma.register_field(
    FieldKey::new(MeshKey::SURFACE, FieldName::Temperature),
    SoaField::<1>::from_fn(cell_count, |cell| [288.0 + cell.index() as f64]),
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
fn samples_change_emits_only_update_layer_samples() {
  let (tessera, mut pleroma, cell_count) = build_world();
  let mut producer = FrameProducer::new(config());

  let _first = producer.extract(WorldId(0), &tessera, &pleroma, None, 0.0, 0);

  // Bump the temperature field — geometry and transform stay put.
  let temperature = pleroma
    .write::<SoaField<1>>(FieldKey::new(
      MeshKey::SURFACE,
      FieldName::Temperature,
    ))
    .expect("temperature field registered");
  *temperature =
    SoaField::<1>::from_fn(cell_count, |cell| [400.0 + cell.index() as f64]);

  let diff = producer.extract(WorldId(0), &tessera, &pleroma, None, 1.0, 1);

  let update_samples = diff
    .updates
    .iter()
    .filter(|u| matches!(u, Update::UpdateLayerSamples { .. }))
    .count();
  assert_eq!(
    update_samples, 1,
    "expected one UpdateLayerSamples, got: {:?}",
    diff.updates
  );

  for update in &diff.updates {
    match update {
      Update::UpdateLayerSamples { .. } | Update::SetSimTime { .. } => {}
      other => panic!(
        "unexpected update in samples-only diff: {:?} (full batch: {:?})",
        other, diff.updates
      ),
    }
  }
}
