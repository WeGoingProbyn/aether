// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Phase 3: air–sea evaporation. A warm sea surface over dry near-surface
//! air must add water vapour to the atmosphere's bottom layer (and only
//! that layer), driving specific humidity toward saturation at the
//! sea-surface temperature.

use std::sync::Arc;

use aer::{EvaporationStep, ShellColumns, saturation_specific_humidity};
use nexus::{
  AtmosphereConstants, FieldKey, FieldName, FieldStorage, MeshKey, Nexus,
  Pleroma, SoaField, WorldConstants, WorldId,
};
use tessera::{
  cube_sphere::{CubeSphere, CubeSphereShellSpec},
  geometry::CellGeometry,
  world_mesh::Tessera,
};
use utility::domain::CellId;
use utility::thread::pool::Pool;

const ANGULAR: [usize; 2] = [2, 2];
const LAYERS: usize = 3;
const GAMMA: f64 = 1.4;
const GAS_CONSTANT: f64 = 287.0;

const STATE: FieldKey =
  FieldKey::new(MeshKey::ATMOSPHERE, FieldName::EulerState);
const SST: FieldKey =
  FieldKey::new(MeshKey::ATMOSPHERE, FieldName::Temperature);
const EVAP: FieldKey =
  FieldKey::new(MeshKey::ATMOSPHERE, FieldName::EvaporationFlux);
const AVAIL: FieldKey =
  FieldKey::new(MeshKey::ATMOSPHERE, FieldName::MoistureAvailability);

fn constants() -> WorldConstants {
  WorldConstants {
    mass: 1.0,
    radius: 1.0,
    surface_gravity: 0.0,
    atmosphere: Some(AtmosphereConstants {
      reference_temperature: 288.0,
      reference_pressure: 101_325.0,
      gamma: GAMMA,
      gas_constant: GAS_CONSTANT,
      molar_mass: 0.02897,
      albedo: None,
      angular_velocity: 0.0,
      axial_tilt: 0.0,
    }),
    radiation: None,
  }
}

#[test]
fn warm_sea_evaporates_into_bottom_layer_only() {
  let mesh = Arc::new(CubeSphere::shell(CubeSphereShellSpec::uniform(
    [ANGULAR[0], ANGULAR[1], LAYERS],
    1000.0,
    1100.0,
  )));
  let mut tessera = Tessera::new();
  tessera.register_mesh(MeshKey::ATMOSPHERE, mesh.clone());

  // Dry air at rest everywhere (q = 0), warm sea surface at 300 K.
  let p = 101_325.0;
  let t_air = 290.0;
  let rho = p / (GAS_CONSTANT * t_air);
  let energy = p / (GAMMA - 1.0);
  let mut pleroma = Pleroma::new();
  pleroma.register_field(
    STATE,
    SoaField::<6>::from_fn(mesh.cell_count(), |_| {
      [rho, 0.0, 0.0, 0.0, energy, 0.0]
    }),
  );
  pleroma.register_field(
    SST,
    SoaField::<1>::from_fn(mesh.cell_count(), |_| [300.0]),
  );
  pleroma.register_field(EVAP, SoaField::<1>::zeros(mesh.cell_count()));

  let columns = ShellColumns::cube_sphere(ANGULAR, LAYERS);
  let mut nexus = Nexus::new();
  nexus.add(
    EvaporationStep::new(MeshKey::ATMOSPHERE, STATE, SST, EVAP, columns, 1e-3)
      .unwrap(),
  );
  let mut compiled = nexus.build(&pleroma).unwrap();
  compiled
    .tick(
      WorldId(0),
      &tessera,
      &constants(),
      &mut pleroma,
      &Pool::default(),
      100.0,
    )
    .unwrap();

  let state: &SoaField<6> = pleroma.read(STATE).unwrap();
  let stride = columns.radial_stride();

  // A bottom-layer cell (panel 0, column 0, layer 0) gained vapour.
  let bottom = state.state(CellId::from(0));
  assert!(bottom[5] > 0.0, "bottom layer should have gained vapour");
  let q_added = bottom[5] / bottom[0];
  let q_sat_sea = saturation_specific_humidity(300.0, p);
  assert!(
    q_added > 0.0 && q_added < q_sat_sea,
    "humidity {q_added} should move toward but not exceed q_sat {q_sat_sea}"
  );

  // An upper-layer cell (layer 1, same column) is untouched (still dry).
  let upper = state.state(CellId::from(stride));
  assert_eq!(upper[5], 0.0, "only the bottom layer evaporates");
}

/// Run one evaporation step from the same dry-air / warm-sea state and return the
/// bottom cell's vapour (`ρq`). `availability` registers a uniform open-water
/// fraction field and gates the step; `None` leaves the step ungated.
fn bottom_vapour_after_step(availability: Option<f64>) -> f64 {
  let mesh = Arc::new(CubeSphere::shell(CubeSphereShellSpec::uniform(
    [ANGULAR[0], ANGULAR[1], LAYERS],
    1000.0,
    1100.0,
  )));
  let mut tessera = Tessera::new();
  tessera.register_mesh(MeshKey::ATMOSPHERE, mesh.clone());

  let p = 101_325.0;
  let t_air = 290.0;
  let rho = p / (GAS_CONSTANT * t_air);
  let energy = p / (GAMMA - 1.0);
  let mut pleroma = Pleroma::new();
  pleroma.register_field(
    STATE,
    SoaField::<6>::from_fn(mesh.cell_count(), |_| {
      [rho, 0.0, 0.0, 0.0, energy, 0.0]
    }),
  );
  pleroma.register_field(
    SST,
    SoaField::<1>::from_fn(mesh.cell_count(), |_| [300.0]),
  );
  pleroma.register_field(EVAP, SoaField::<1>::zeros(mesh.cell_count()));

  let columns = ShellColumns::cube_sphere(ANGULAR, LAYERS);
  let mut step =
    EvaporationStep::new(MeshKey::ATMOSPHERE, STATE, SST, EVAP, columns, 1e-3)
      .unwrap();
  if let Some(a) = availability {
    pleroma.register_field(
      AVAIL,
      SoaField::<1>::from_fn(mesh.cell_count(), |_| [a]),
    );
    step = step.with_moisture_availability(AVAIL).unwrap();
  }

  let mut nexus = Nexus::new();
  nexus.add(step);
  let mut compiled = nexus.build(&pleroma).unwrap();
  compiled
    .tick(
      WorldId(0),
      &tessera,
      &constants(),
      &mut pleroma,
      &Pool::default(),
      100.0,
    )
    .unwrap();

  let state: &SoaField<6> = pleroma.read(STATE).unwrap();
  state.state(CellId::from(0))[5]
}

#[test]
fn moisture_availability_gates_evaporation() {
  let full = bottom_vapour_after_step(Some(1.0));
  let ungated = bottom_vapour_after_step(None);
  let land = bottom_vapour_after_step(Some(0.0));
  let half = bottom_vapour_after_step(Some(0.5));

  assert!(full > 0.0, "open water (availability 1) evaporates");
  assert_eq!(
    ungated, full,
    "no availability field behaves as availability 1"
  );
  assert_eq!(land, 0.0, "land (availability 0) injects no vapour");
  // The relaxation is linear in availability for one step from a fixed state.
  let ratio = half / full;
  assert!(
    (ratio - 0.5).abs() < 1e-6,
    "half availability gives ~half the vapour, got {ratio}"
  );
}
