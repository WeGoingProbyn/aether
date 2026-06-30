// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Terrain displacement (Pillar 3 render): a scalar layer flagged with a
//! displacement scale makes the producer emit a `SetMeshDisplacement` directive
//! tying the layer to its target mesh. The reference renderer uses it to raise
//! terrain relief; the IR geometry itself stays undisplaced data.

use std::sync::Arc;

use eidolon::extract::{
  ExtractConfig, FrameProducer, MeshConfig, ScalarLayerConfig,
};
use eidolon::ir::{LayerHandle, LayerId, MeshRepresentation, Palette, Update};
use pleroma::{Pleroma, core::storage::SoaField};
use tessera::cube_sphere::{CubeSphere, CubeSphereShellSpec};
use tessera::geometry::CellGeometry;
use tessera::mesh::Mesh;
use tessera::world_mesh::Tessera;
use utility::domain::{BoundaryTag, FieldKey, FieldName, MeshKey, WorldId};

fn build_world() -> (Tessera, Pleroma) {
  let surface = Arc::new(CubeSphere::shell(
    CubeSphereShellSpec::uniform([2, 2, 1], 0.9, 1.0)
      .with_boundaries(BoundaryTag::Ground, BoundaryTag::AtmosphereEdge),
  ));
  let cell_count = surface.cell_count();
  let mut tessera = Tessera::new();
  tessera.register_mesh(MeshKey::SURFACE, surface as Arc<dyn Mesh<3>>);

  let mut pleroma = Pleroma::new();
  pleroma.register_field(
    FieldKey::new(MeshKey::SURFACE, FieldName::SurfaceElevation),
    SoaField::<1>::from_fn(cell_count, |cell| [cell.index() as f64 * 100.0]),
  );
  (tessera, pleroma)
}

fn config(displacement: Option<f32>) -> ExtractConfig {
  ExtractConfig {
    world_label: "terrain".into(),
    world_scale: 1.0,
    meshes: vec![MeshConfig::new(
      MeshKey::SURFACE,
      MeshRepresentation::BoundaryFaces,
      "surface",
    )],
    layers: vec![ScalarLayerConfig {
      id: LayerId::from_static("surface_elevation"),
      label: "surface_elevation".into(),
      target_mesh: MeshKey::SURFACE,
      target_representation: MeshRepresentation::BoundaryFaces,
      field: FieldKey::new(MeshKey::SURFACE, FieldName::SurfaceElevation),
      component: 0,
      palette: Palette::diagnostic(),
      displacement,
    }],
    categorical_layers: vec![],
    track_sun_direction: false,
    track_camera: false,
  }
}

#[test]
fn displacement_layer_emits_directive_tied_to_its_mesh_and_layer() {
  let (tessera, pleroma) = build_world();

  let mut producer = FrameProducer::new(config(Some(40.0)));
  let batch = producer.extract(WorldId(0), &tessera, &pleroma, None, 0.0, 0);

  // The directive points at the surface mesh and the elevation layer.
  let target = eidolon::ir::RenderMeshId {
    world: WorldId(0),
    mesh: MeshKey::SURFACE,
    representation: MeshRepresentation::BoundaryFaces,
  }
  .handle();
  let expected_layer =
    LayerHandle::for_target(LayerId::from_static("surface_elevation"), target);

  let directive = batch
    .updates
    .iter()
    .find_map(|u| match u {
      Update::SetMeshDisplacement { mesh, layer, scale } => {
        Some((*mesh, *layer, *scale))
      }
      _ => None,
    })
    .expect("a displacement directive should be emitted");
  assert_eq!(directive.0, target, "displaces the surface mesh");
  assert_eq!(directive.1, expected_layer, "driven by the elevation layer");
  assert_eq!(directive.2, 40.0, "carries the configured exaggeration");
}

#[test]
fn directive_is_emitted_once_not_every_tick() {
  let (tessera, pleroma) = build_world();

  let mut producer = FrameProducer::new(config(Some(40.0)));
  let first = producer.extract(WorldId(0), &tessera, &pleroma, None, 0.0, 0);
  assert_eq!(
    first
      .updates
      .iter()
      .filter(|u| matches!(u, Update::SetMeshDisplacement { .. }))
      .count(),
    1,
    "first batch declares displacement once"
  );

  // Elevation is static, so the second tick re-emits nothing but the time.
  let second = producer.extract(WorldId(0), &tessera, &pleroma, None, 1.0, 1);
  assert!(
    !second
      .updates
      .iter()
      .any(|u| matches!(u, Update::SetMeshDisplacement { .. })),
    "displacement is not re-declared on a no-change tick"
  );
}

#[test]
fn no_displacement_flag_emits_no_directive() {
  let (tessera, pleroma) = build_world();

  let mut producer = FrameProducer::new(config(None));
  let batch = producer.extract(WorldId(0), &tessera, &pleroma, None, 0.0, 0);
  assert!(
    !batch
      .updates
      .iter()
      .any(|u| matches!(u, Update::SetMeshDisplacement { .. })),
    "a plain scalar layer must not emit displacement"
  );
}
