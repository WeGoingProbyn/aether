// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Phase 5 (render-LOD): when a mesh is adaptively refined and swapped into the
//! `Tessera`, the frame producer re-emits its geometry — no special AMR code in
//! eidolon. `build_mesh` rebuilds from whatever mesh `tessera` currently holds
//! each frame, so a changed (refined) mesh hashes differently and produces an
//! `UpdateMeshGeometry`. Render level-of-detail is the topology-change path,
//! observed for free.

use std::sync::Arc;

use eidolon::extract::{
  ExtractConfig, FrameProducer, MeshConfig, ScalarLayerConfig,
};
use eidolon::ir::{MeshRepresentation, Palette, Update};
use pleroma::{Pleroma, core::storage::SoaField};
use tessera::adaptive::AdaptiveMesh;
use tessera::cube_sphere::CubeSphere;
use tessera::geometry::CellGeometry;
use tessera::mesh::Mesh;
use tessera::refine::AdaptRequest;
use tessera::world_mesh::Tessera;
use utility::domain::{CellId, FieldKey, FieldName, MeshKey, WorldId};

const SURFACE_TEMPERATURE: FieldKey =
  FieldKey::new(MeshKey::SURFACE, FieldName::Temperature);

fn config() -> ExtractConfig {
  ExtractConfig {
    world_label: "earth".into(),
    world_scale: 1.0,
    meshes: vec![MeshConfig {
      mesh_key: MeshKey::SURFACE,
      representation: MeshRepresentation::BoundaryFaces,
      label: "surface".into(),
      cell_filter: None,
    }],
    layers: vec![ScalarLayerConfig {
      id: eidolon::ir::LayerId::from_static("surface_temperature"),
      label: "surface_temperature".into(),
      target_mesh: MeshKey::SURFACE,
      target_representation: MeshRepresentation::BoundaryFaces,
      field: SURFACE_TEMPERATURE,
      component: 0,
      palette: Palette::diagnostic(),
      displacement: None,
    }],
    categorical_layers: vec![],
    track_sun_direction: false,
  }
}

fn register_temperature(pleroma: &mut Pleroma, cells: usize) {
  pleroma.register_field(
    SURFACE_TEMPERATURE,
    SoaField::<1>::from_fn(cells, |c| [288.0 + c.index() as f64]),
  );
}

#[test]
fn refining_a_mesh_re_emits_geometry() {
  // Adaptive surface; refine a panel-interior cell on the second extract.
  let base = Arc::new(CubeSphere::new([8, 8, 1], 0.9, 1.0));
  let mesh = AdaptiveMesh::new(base);
  let n0 = mesh.cell_count();

  let mut tessera = Tessera::new();
  tessera.register_mesh(MeshKey::SURFACE, Arc::new(mesh) as Arc<dyn Mesh<3>>);
  let mut pleroma = Pleroma::new();
  register_temperature(&mut pleroma, n0);

  let mut producer = FrameProducer::new(config());

  // First frame registers the mesh.
  let first = producer.extract(WorldId(0), &tessera, &pleroma, None, 0.0, 0);
  assert!(
    first
      .updates
      .iter()
      .any(|u| matches!(u, Update::RegisterMesh { .. })),
    "first frame should register the surface mesh"
  );

  // Refine cell 36 (panel-0 interior on [8,8,1]: 4 + 4*8). Rebuild the field at
  // the new size and swap the refined mesh into the tessera.
  let amesh = AdaptiveMesh::new(Arc::new(CubeSphere::new([8, 8, 1], 0.9, 1.0)));
  let (refined, _remap) = amesh
    .refine(&AdaptRequest {
      refine: vec![CellId::from(36)],
      coarsen: vec![],
    })
    .unwrap();
  let new_count = refined.cell_count();
  assert!(new_count > n0);
  register_temperature(&mut pleroma, new_count);
  tessera
    .register_mesh(MeshKey::SURFACE, Arc::new(refined) as Arc<dyn Mesh<3>>);

  // Second frame must re-emit the geometry for the changed mesh.
  let second = producer.extract(WorldId(0), &tessera, &pleroma, None, 1.0, 1);
  assert!(
    second
      .updates
      .iter()
      .any(|u| matches!(u, Update::UpdateMeshGeometry { .. })),
    "refined mesh should re-emit geometry, got {:?}",
    second.updates
  );
}
