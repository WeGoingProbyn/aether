// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Phase 5 (query): the semantic query view is rebuilt from the world's meshes,
//! so reacting to a topology change is simply `WorldQuery::new(&tessera, r)` on
//! the new mesh — an `AdaptiveMesh` is a `Mesh`, so its `GeoIndex` builds the
//! same way. This checks the rebuilt index is correct over a *refined* mesh: every
//! cell's own centroid locates back to that cell.

use std::sync::Arc;

use eidolon::query::WorldQuery;
use tessera::adaptive::AdaptiveMesh;
use tessera::cube_sphere::CubeSphere;
use tessera::geo::GeoCoord;
use tessera::geometry::CellGeometry;
use tessera::mesh::Mesh;
use tessera::refine::AdaptRequest;
use tessera::world_mesh::Tessera;
use utility::domain::{CellId, MeshKey, MeshType};

const SURFACE_RADIUS: f64 = 1.0;

#[test]
fn query_index_rebuilds_over_a_refined_mesh() {
  // Refine a panel-interior cell, register the refined mesh as the surface, and
  // rebuild the query view from it.
  let amesh = AdaptiveMesh::new(Arc::new(CubeSphere::new([8, 8, 1], 0.9, 1.0)));
  let n0 = amesh.cell_count();
  let (refined, _remap) = amesh
    .refine(&AdaptRequest {
      refine: vec![CellId::from(36)],
      coarsen: vec![],
    })
    .unwrap();
  let new_count = refined.cell_count();
  assert!(new_count > n0);

  let refined: Arc<dyn Mesh<3>> = Arc::new(refined);
  let mut tessera = Tessera::new();
  tessera.register_mesh(MeshKey::SURFACE, refined.clone());

  let query = WorldQuery::new(&tessera, SURFACE_RADIUS);

  // Every cell's own centroid locates back to that cell — the rebuilt index
  // covers the refined cell set exactly (including the new fine children).
  for i in 0..new_count {
    let cell = CellId::from(i);
    let geo =
      GeoCoord::from_world(&refined.cell_world_centroid(cell), SURFACE_RADIUS);
    assert_eq!(
      query.locate(MeshType::Surface, geo),
      Some(cell),
      "cell {i} did not locate back to itself on the refined mesh"
    );
  }
}
