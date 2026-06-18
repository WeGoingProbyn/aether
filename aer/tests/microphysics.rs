// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Phase 3: condensation/precipitation microphysics. The saturation
//! adjustment must (a) leave sub-saturated air untouched, and
//! (b) precipitate exactly the super-saturated excess while conserving
//! total water (vapour + precipitation) and releasing the matching latent
//! heat into the energy equation.

use std::sync::Arc;

use aer::{
  LATENT_HEAT_VAPORISATION, SaturationAdjustmentStep,
  saturation_specific_humidity,
};
use nexus::{
  AtmosphereConstants, FieldStorage, MeshKey, Nexus, Pleroma, SoaField,
  WorldConstants, WorldId,
};
use nexus::{FieldKey, FieldName};
use tessera::{
  geometry::IdentityMap,
  mesh::{Mesh, StructuredBlock},
  world_mesh::Tessera,
};
use utility::domain::CellId;
use utility::thread::pool::Pool;

const STATE: FieldKey =
  FieldKey::new(MeshKey::ATMOSPHERE, FieldName::EulerState);
const PRECIP: FieldKey =
  FieldKey::new(MeshKey::ATMOSPHERE, FieldName::PrecipitationFlux);

const GAMMA: f64 = 1.4;
const GAS_CONSTANT: f64 = 287.0;

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
fn saturation_specific_humidity_increases_with_temperature() {
  let p = 101_325.0;
  let cold = saturation_specific_humidity(273.15, p);
  let warm = saturation_specific_humidity(303.15, p);
  assert!(cold > 0.0 && warm > cold, "q_sat must grow with T");
  // Sanity: ~30 °C saturated air holds a few percent water by mass.
  assert!(warm < 0.05 && warm > 0.02, "q_sat(30C) = {warm}");
}

/// Build a single-cell atmosphere Euler state at rest with the given
/// temperature and specific humidity at reference pressure.
fn rest_state(temperature: f64, q: f64) -> [f64; 6] {
  let p = 101_325.0;
  let rho = p / (GAS_CONSTANT * temperature);
  let energy = p / (GAMMA - 1.0); // at rest, E = internal energy
  [rho, 0.0, 0.0, 0.0, energy, rho * q]
}

fn run_adjustment(initial: [f64; 6]) -> ([f64; 6], f64) {
  let mut tessera = Tessera::new();
  let mesh = Arc::new(StructuredBlock::uniform(
    [0.0; 3].into(),
    [1.0; 3],
    [1, 1, 1],
    Box::new(IdentityMap::<3>),
  ));
  let mesh_for_registry: Arc<dyn Mesh<3>> = mesh;
  tessera.register_mesh(MeshKey::ATMOSPHERE, mesh_for_registry);

  let mut pleroma = Pleroma::new();
  pleroma.register_field(STATE, SoaField::<6>::from_fn(1, |_| initial));
  pleroma.register_field(PRECIP, SoaField::<1>::zeros(1));

  let mut nexus = Nexus::new();
  nexus.add(
    SaturationAdjustmentStep::new(MeshKey::ATMOSPHERE, STATE, PRECIP).unwrap(),
  );
  let mut compiled = nexus.build(&pleroma).unwrap();
  compiled
    .tick(
      WorldId(0),
      &tessera,
      &constants(),
      &mut pleroma,
      &Pool::default(),
      10.0,
    )
    .unwrap();

  let state: &SoaField<6> = pleroma.read(STATE).unwrap();
  let precip: &SoaField<1> = pleroma.read(PRECIP).unwrap();
  (
    state.state(CellId::from(0)),
    precip.state(CellId::from(0))[0],
  )
}

#[test]
fn subsaturated_air_is_left_untouched() {
  // q well below saturation at 288 K → no condensation.
  let initial = rest_state(288.0, 0.002);
  let (after, precip) = run_adjustment(initial);
  assert_eq!(precip, 0.0);
  for i in 0..6 {
    assert!(
      (after[i] - initial[i]).abs() < 1e-9,
      "component {i} changed"
    );
  }
}

#[test]
fn supersaturated_excess_precipitates_conserving_water_and_energy() {
  let temperature = 295.0;
  let p = 101_325.0;
  let rho = p / (GAS_CONSTANT * temperature);
  let q_sat = saturation_specific_humidity(temperature, p);
  let q0 = q_sat + 0.01; // 1% supersaturated by mass fraction
  let initial = rest_state(temperature, q0);
  let dt = 10.0;

  let (after, precip_rate) = run_adjustment(initial);

  let vapour_before = initial[5];
  let vapour_after = after[5];
  let precipitated = precip_rate * dt; // kg/m³ removed this step

  // (a) Total water conserved: vapour lost == precipitation produced.
  assert!(
    ((vapour_before - vapour_after) - precipitated).abs() < 1e-9,
    "water not conserved: Δvapour {}, precip {}",
    vapour_before - vapour_after,
    precipitated
  );
  // (b) Vapour pulled back to (approximately) saturation.
  let q_after = after[5] / after[0];
  assert!(
    (q_after - q_sat).abs() < 1e-4,
    "q_after {q_after} should be ~q_sat {q_sat}"
  );
  // (c) Latent heat of the condensed mass added to energy.
  let expected_energy = initial[4] + LATENT_HEAT_VAPORISATION * precipitated;
  assert!(
    (after[4] - expected_energy).abs() < 1e-3 * expected_energy,
    "energy {} should be initial + L·condensed {}",
    after[4],
    expected_energy
  );
  // Density and momentum untouched.
  assert_eq!(after[0], rho);
  assert_eq!(after[1], 0.0);
}
