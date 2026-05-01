// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! End-to-end tests for `RadiativeTransferStep` via nexus + pleroma.
//!
//! Uses a tiny cube-sphere shell so we can register both an "atmosphere"
//! and a "surface" mesh and verify lumen's outputs for known
//! configurations: a sun fixed on the +x axis should heat the +x
//! hemisphere and leave the -x hemisphere dark; a hot surface in
//! daylight should still produce a positive net flux while a hot
//! surface at night should produce a negative one.

use std::sync::Arc;

use lumen::{RadiationModel, RadiationParameters};
use nexus::{
  FieldStorage, MeshKey, Nexus, Pleroma, ResourceKey, SoaField, WorldConstants,
  WorldId,
};
use tessera::{
  cube_sphere::{CubeSphere, CubeSphereShellSpec},
  geometry::CellGeometry,
  mesh::Mesh,
  world_mesh::Tessera,
};
use utility::{
  constants::STEFAN_BOLTZMANN, domain::CellId, thread::pool::Pool,
};

const SURFACE_T: f64 = 290.0;
const ATM_T: f64 = 250.0;

fn build_world() -> (
  Tessera,
  Pleroma,
  Arc<CubeSphere>,
  Arc<CubeSphere>,
  RadiationModel,
) {
  let surface_mesh = Arc::new(CubeSphere::shell(CubeSphereShellSpec::uniform(
    [4, 4, 1],
    0.99,
    1.0,
  )));
  let atm_mesh = Arc::new(CubeSphere::shell(CubeSphereShellSpec::uniform(
    [4, 4, 2],
    1.0,
    1.2,
  )));
  let mut tessera = Tessera::new();
  let surface_mesh_dyn: Arc<dyn Mesh<3>> = surface_mesh.clone();
  let atm_mesh_dyn: Arc<dyn Mesh<3>> = atm_mesh.clone();
  tessera.register_mesh(MeshKey::SURFACE, surface_mesh_dyn);
  tessera.register_mesh(MeshKey::ATMOSPHERE, atm_mesh_dyn);

  let model = RadiationModel::new(MeshKey::ATMOSPHERE, MeshKey::SURFACE);

  let mut pleroma = Pleroma::new();
  // Aer/terra would normally register the temperature fields. For tests we
  // do it directly with simple constant initial conditions.
  let fields = model.fields();
  pleroma.register_field(
    fields.surface_temperature,
    SoaField::<1>::from_fn(surface_mesh.cell_count(), |_| [SURFACE_T]),
  );
  pleroma.register_field(
    fields.atm_temperature,
    SoaField::<1>::from_fn(atm_mesh.cell_count(), |_| [ATM_T]),
  );
  model
    .register_fields(&mut pleroma, atm_mesh.as_ref(), surface_mesh.as_ref())
    .unwrap();

  (tessera, pleroma, surface_mesh, atm_mesh, model)
}

fn run_one_tick(
  tessera: &Tessera,
  pleroma: &mut Pleroma,
  model: &RadiationModel,
  sun_direction: [f64; 3],
) {
  pleroma.register_resource(ResourceKey::SunPosition, sun_direction);

  let mut nexus = Nexus::new();
  model.add_stages(&mut nexus).unwrap();
  let mut compiled = nexus.build(pleroma).unwrap();
  compiled
    .tick(
      WorldId(0),
      tessera,
      &WorldConstants::default(),
      pleroma,
      &Pool::default(),
      1.0,
    )
    .unwrap();
}

#[test]
fn day_side_atmosphere_heats_night_side_does_not() {
  let (tessera, mut pleroma, _surface_mesh, atm_mesh, model) = build_world();
  let fields = model.fields();
  let sun = [1.0, 0.0, 0.0];

  run_one_tick(&tessera, &mut pleroma, &model, sun);

  let heating: &SoaField<1> = pleroma.read(fields.heating_tendency).unwrap();

  let mut day_side_seen = false;
  let mut night_side_seen = false;
  for i in 0..atm_mesh.cell_count() {
    let centroid = atm_mesh.cell_centroid(CellId::from(i));
    let h = heating.state(CellId::from(i))[0];
    if centroid[0] > 0.5 {
      // Day side: solar heating dominates the LW damping at these temps.
      assert!(h > 0.0, "expected positive heating on day side, got {h}");
      day_side_seen = true;
    } else if centroid[0] < -0.5 {
      // Night side: only the longwave damping term contributes (sign
      // depends on T_atm vs reference). We just check it's not the
      // day-side value — i.e. no solar input.
      assert!(
        h <= 0.0 || h.abs() < 1e-6,
        "expected non-positive heating on night side, got {h}"
      );
      night_side_seen = true;
    }
  }
  assert!(day_side_seen);
  assert!(night_side_seen);
}

#[test]
fn surface_net_flux_matches_stefan_boltzmann_balance() {
  let (tessera, mut pleroma, surface_mesh, _atm_mesh, model) = build_world();
  let fields = model.fields();
  let params = RadiationParameters::default();
  let sun = [1.0, 0.0, 0.0];

  run_one_tick(&tessera, &mut pleroma, &model, sun);

  let net_flux: &SoaField<1> = pleroma.read(fields.net_surface_flux).unwrap();

  for i in 0..surface_mesh.cell_count() {
    let centroid = surface_mesh.cell_centroid(CellId::from(i));
    let r =
      (centroid[0].powi(2) + centroid[1].powi(2) + centroid[2].powi(2)).sqrt();
    let mu = (centroid[0] / r).max(0.0);

    let incoming_sw =
      (1.0 - params.atmospheric_absorption) * params.solar_constant * mu;
    let absorbed_sw = (1.0 - params.surface_albedo) * incoming_sw;
    let outgoing_lw =
      params.surface_emissivity * STEFAN_BOLTZMANN * SURFACE_T.powi(4);
    let expected = absorbed_sw - (1.0 - params.greenhouse_factor) * outgoing_lw;
    let actual = net_flux.state(CellId::from(i))[0];
    assert!(
      (actual - expected).abs() < 1e-6,
      "cell {i}: expected {expected}, got {actual}"
    );
  }
}

#[test]
fn night_side_surface_radiates_negative_net_flux() {
  // Hot surface, no incoming sun on the dark side → net flux should be
  // negative (planet losing energy to space).
  let (tessera, mut pleroma, surface_mesh, _atm_mesh, model) = build_world();
  let fields = model.fields();
  let sun = [1.0, 0.0, 0.0];

  run_one_tick(&tessera, &mut pleroma, &model, sun);

  let net_flux: &SoaField<1> = pleroma.read(fields.net_surface_flux).unwrap();
  for i in 0..surface_mesh.cell_count() {
    let centroid = surface_mesh.cell_centroid(CellId::from(i));
    if centroid[0] < -0.5 {
      let f = net_flux.state(CellId::from(i))[0];
      assert!(f < 0.0, "cell {i} on night side expected negative, got {f}");
    }
  }
}

#[test]
fn missing_sun_position_resource_surfaces_as_error() {
  let (tessera, mut pleroma, _surface_mesh, _atm_mesh, model) = build_world();
  // Deliberately do NOT register SunPosition.
  let mut nexus = Nexus::new();
  model.add_stages(&mut nexus).unwrap();
  let mut compiled = nexus.build(&pleroma).unwrap();
  let result = compiled.tick(
    WorldId(0),
    &tessera,
    &WorldConstants::default(),
    &mut pleroma,
    &Pool::default(),
    1.0,
  );
  assert!(result.is_err());
}
