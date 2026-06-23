// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Terrain render data (Pillar 3): the producer emits the surface type as an
//! art-free **categorical** layer (per-cell class id + the class vocabulary) and
//! the elevation as a plain scalar **data** layer — no colours. Together with
//! the surface mesh geometry this is everything a consumer needs to build a
//! terrain look; eidolon makes no art decision.

use std::sync::Arc;

use eidolon::extract::{
  ExtractConfig, FrameProducer, MeshConfig, ScalarLayerConfig,
  surface_type_categorical_layer,
};
use eidolon::ir::{
  CategoricalSamples, LayerId, LayerKind, LayerSamples, MeshRepresentation,
  Palette, Update,
};
use pleroma::{Pleroma, core::storage::SoaField};
use tessera::cube_sphere::{CubeSphere, CubeSphereShellSpec};
use tessera::geometry::CellGeometry;
use tessera::mesh::Mesh;
use tessera::world_mesh::Tessera;
use utility::domain::{
  BoundaryTag, FieldKey, FieldName, MeshKey, SurfaceClass, WorldId,
};

fn build_world() -> (Tessera, Pleroma, usize) {
  let surface = Arc::new(CubeSphere::shell(
    CubeSphereShellSpec::uniform([2, 2, 1], 0.9, 1.0)
      .with_boundaries(BoundaryTag::Ground, BoundaryTag::AtmosphereEdge),
  ));
  let cell_count = surface.cell_count();
  let mut tessera = Tessera::new();
  tessera.register_mesh(MeshKey::SURFACE, surface as Arc<dyn Mesh<3>>);

  let mut pleroma = Pleroma::new();
  // Surface type: cycle ocean / land / ice across cells.
  let classes = [SurfaceClass::Ocean, SurfaceClass::Land, SurfaceClass::Ice];
  pleroma.register_field(
    FieldKey::new(MeshKey::SURFACE, FieldName::SurfaceType),
    SoaField::<1>::from_fn(cell_count, |cell| {
      [classes[cell.index() % 3].code()]
    }),
  );
  // Elevation data layer (a plain scalar, no palette needed by the consumer).
  pleroma.register_field(
    FieldKey::new(MeshKey::SURFACE, FieldName::SurfaceElevation),
    SoaField::<1>::from_fn(cell_count, |cell| [cell.index() as f64 * 100.0]),
  );
  (tessera, pleroma, cell_count)
}

fn config() -> ExtractConfig {
  ExtractConfig {
    world_label: "terrain".into(),
    world_scale: 1.0,
    meshes: vec![MeshConfig {
      mesh_key: MeshKey::SURFACE,
      representation: MeshRepresentation::BoundaryFaces,
      label: "surface".into(),
    }],
    layers: vec![ScalarLayerConfig {
      id: LayerId::from_static("surface_elevation"),
      label: "surface_elevation".into(),
      target_mesh: MeshKey::SURFACE,
      target_representation: MeshRepresentation::BoundaryFaces,
      field: FieldKey::new(MeshKey::SURFACE, FieldName::SurfaceElevation),
      component: 0,
      palette: Palette::diagnostic(),
    }],
    categorical_layers: vec![surface_type_categorical_layer(
      LayerId::from_static("surface_type"),
      "surface_type",
      MeshKey::SURFACE,
      MeshRepresentation::BoundaryFaces,
    )],
    track_sun_direction: false,
  }
}

#[test]
fn emits_categorical_surface_type_and_scalar_elevation() {
  let (tessera, pleroma, cell_count) = build_world();
  let mut producer = FrameProducer::new(config());
  let batch = producer.extract(WorldId(0), &tessera, &pleroma, None, 0.0, 0);

  // A categorical layer was registered with the ocean/land/ice vocabulary.
  let classes = batch
    .updates
    .iter()
    .find_map(|u| match u {
      Update::RegisterLayer {
        kind: LayerKind::Categorical { classes },
        ..
      } => Some(classes.clone()),
      _ => None,
    })
    .expect("a categorical layer should be registered");
  assert_eq!(
    classes.label_of(SurfaceClass::Ocean.code() as u32),
    Some("Ocean")
  );
  assert_eq!(
    classes.label_of(SurfaceClass::Ice.code() as u32),
    Some("Ice")
  );

  // Its samples are per-cell class ids matching the field.
  let ids = batch
    .updates
    .iter()
    .find_map(|u| match u {
      Update::UpdateLayerSamples {
        samples: LayerSamples::Categorical(CategoricalSamples::PerCell(ids)),
        ..
      } => Some(ids.clone()),
      _ => None,
    })
    .expect("categorical samples should be emitted");
  assert_eq!(ids.len(), cell_count);
  let expected = [
    SurfaceClass::Ocean.code() as u32,
    SurfaceClass::Land.code() as u32,
    SurfaceClass::Ice.code() as u32,
  ];
  for (i, &id) in ids.iter().enumerate() {
    assert_eq!(id, expected[i % 3], "class id mismatch at cell {i}");
  }

  // The elevation scalar data layer is present too (the displacement source).
  let elevation_layers = batch
    .updates
    .iter()
    .filter(|u| {
      matches!(
        u,
        Update::RegisterLayer {
          kind: LayerKind::Scalar { .. },
          ..
        }
      )
    })
    .count();
  assert_eq!(
    elevation_layers, 1,
    "elevation scalar layer should register"
  );
}
