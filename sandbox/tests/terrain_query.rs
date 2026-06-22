// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Pillar 3 inert-terrain end-to-end: register the static heightfield +
//! surface-type fields with `terra::TerrainModel`, extract them into a snapshot
//! through the query producer path, and read them back through the Pillar-1
//! query API — with no physics coupling. Confirms terrain is real, extractable,
//! queryable state before any coupling is wired.

use std::sync::Arc;

use eidolon::extract::{
  default_surface_terrain_quantities, extract_quantity_frame,
};
use eidolon::playback::FrameInterpolator;
use eidolon::query::{ScalarQuantity, WorldQuery};
use nexus::Pleroma;
use terra::{TerrainModel, earthlike_terrain};
use tessera::cube_sphere::CubeSphere;
use tessera::geo::GeoCoord;
use tessera::mesh::Mesh;
use tessera::world_mesh::Tessera;
use utility::domain::{MeshKey, MeshType, SurfaceClass};

#[test]
fn inert_terrain_extracts_and_queries_end_to_end() {
  let r_inner = 6.371e6;
  let r_outer = 6.381e6;
  let mesh = Arc::new(CubeSphere::new([16, 16, 1], r_inner, r_outer));

  let mut tessera = Tessera::new();
  tessera.register_mesh(MeshKey::SURFACE, mesh.clone() as Arc<dyn Mesh<3>>);

  // Register + initialise the inert terrain fields.
  let mut pleroma = Pleroma::new();
  let model = TerrainModel::new(MeshKey::SURFACE, r_inner);
  model
    .register_fields(&mut pleroma, mesh.as_ref(), earthlike_terrain)
    .unwrap();

  // Extract into a snapshot via the query producer path.
  let frame = extract_quantity_frame(
    &pleroma,
    0.0,
    &default_surface_terrain_quantities(),
  );
  let mut snapshot = FrameInterpolator::new();
  snapshot.push(frame);

  let query = WorldQuery::new(&tessera, r_inner);

  // Across several points, elevation is finite and the class is consistent
  // with the generator's land≥0 / ocean≤0 invariant.
  for (lat, lon) in [(10.0, 20.0), (-30.0, 140.0), (55.0, -75.0), (0.0, -90.0)]
  {
    let at = GeoCoord::from_degrees(lat, lon, 0.0);
    let elev = query
      .sample_scalar(&snapshot, ScalarQuantity::SurfaceElevation, at)
      .value()
      .expect("elevation available");
    assert!(
      elev.is_finite(),
      "elevation {elev} not finite at {lat},{lon}"
    );

    let class = query
      .surface_class(&snapshot, at)
      .value()
      .expect("surface class available");
    match class {
      SurfaceClass::Land => assert!(elev >= 0.0, "land below datum: {elev}"),
      SurfaceClass::Ocean => assert!(elev <= 0.0, "ocean above datum: {elev}"),
      SurfaceClass::Ice => {}
    }
  }

  // The generator makes the poles ice.
  let pole = GeoCoord::from_degrees(85.0, 0.0, 0.0);
  assert_eq!(
    query.surface_class(&snapshot, pole).value().unwrap(),
    SurfaceClass::Ice
  );
}
