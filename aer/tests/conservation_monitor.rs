// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! End-to-end coverage for `AtmosphereConservationMonitor` running as a nexus
//! stage: it publishes a per-field report into the `Diagnostics` resource, and
//! the active `DiagnosticsPolicy` decides whether non-finite state warns
//! (tick succeeds) or fails (tick returns `Err`).

use std::sync::Arc;

use aer::AtmosphereConservationMonitor;
use nexus::{
  FieldKey, FieldName, MeshKey, Nexus, Pleroma, SoaField, WorldConstants,
  WorldId,
};
use tessera::{
  geometry::IdentityMap,
  mesh::{Mesh, StructuredBlock},
  world_mesh::Tessera,
};
use utility::{
  diagnostics::{DiagnosticsPolicy, WorldDiagnostics},
  domain::ResourceKey,
  thread::pool::Pool,
};

const STATE: FieldKey =
  FieldKey::new(MeshKey::ATMOSPHERE, FieldName::EulerState);

/// Unit-volume single cell so conserved totals equal the raw state values.
fn single_cell_tessera() -> Tessera {
  let mut tessera = Tessera::new();
  let mesh: Arc<dyn Mesh<3>> = Arc::new(StructuredBlock::uniform(
    [0.0; 3].into(),
    [1.0; 3],
    [1, 1, 1],
    Box::new(IdentityMap::<3>),
  ));
  tessera.register_mesh(MeshKey::ATMOSPHERE, mesh);
  tessera
}

fn constants() -> WorldConstants {
  WorldConstants {
    mass: 1.0,
    radius: 1.0,
    surface_gravity: 0.0,
    atmosphere: None,
    radiation: None,
  }
}

/// `[rho, rho_u, rho_v, rho_w, energy, rho_q]`.
const GOOD_STATE: [f64; 6] = [1.2, 0.0, 0.0, 0.0, 250_000.0, 0.012];

fn world_with_state(
  policy: DiagnosticsPolicy,
  state: [f64; 6],
) -> (Tessera, Pleroma) {
  let mut pleroma = Pleroma::new();
  pleroma.register_field(STATE, SoaField::<6>::from_fn(1, |_| state));
  pleroma.register_resource(
    ResourceKey::Diagnostics,
    WorldDiagnostics::with_policy(policy),
  );
  (single_cell_tessera(), pleroma)
}

fn tick(
  tessera: &Tessera,
  pleroma: &mut Pleroma,
) -> utility::error::AetherResult<()> {
  let mut nexus = Nexus::new();
  nexus.add(AtmosphereConservationMonitor::new(
    MeshKey::ATMOSPHERE,
    STATE,
  ));
  let mut compiled = nexus.build(pleroma).unwrap();
  compiled.tick(
    WorldId(0),
    tessera,
    &constants(),
    pleroma,
    &Pool::default(),
    1.0,
  )
}

#[test]
fn publishes_conserved_totals_and_zero_non_finite_for_healthy_state() {
  let (tessera, mut pleroma) =
    world_with_state(DiagnosticsPolicy::Observe, GOOD_STATE);
  tick(&tessera, &mut pleroma).expect("healthy tick succeeds");

  let diagnostics = pleroma
    .read_resource::<WorldDiagnostics>(ResourceKey::Diagnostics)
    .unwrap();
  let report = diagnostics
    .fields
    .get(&STATE)
    .expect("field report present");

  assert_eq!(report.non_finite_cells, 0);
  // Unit cell volume → each conserved total equals the raw component.
  let names: Vec<&str> = report.conserved.iter().map(|(n, _)| *n).collect();
  assert_eq!(
    names,
    vec![
      "mass",
      "momentum_x",
      "momentum_y",
      "momentum_z",
      "total_energy",
      "water"
    ]
  );
  for ((_, total), expected) in report.conserved.iter().zip(GOOD_STATE.iter()) {
    assert!((total - expected).abs() < 1e-9, "{total} vs {expected}");
  }
  assert!(!diagnostics.has_non_finite());
}

#[test]
fn fail_policy_turns_non_finite_state_into_an_error() {
  let mut bad = GOOD_STATE;
  bad[0] = f64::NAN;
  let (tessera, mut pleroma) = world_with_state(DiagnosticsPolicy::Fail, bad);

  let result = tick(&tessera, &mut pleroma);
  assert!(result.is_err(), "Fail policy must surface non-finite state");
}

#[test]
fn warn_policy_flags_non_finite_but_succeeds() {
  let mut bad = GOOD_STATE;
  bad[4] = f64::INFINITY;
  let (tessera, mut pleroma) = world_with_state(DiagnosticsPolicy::Warn, bad);

  tick(&tessera, &mut pleroma).expect("Warn policy never fails the tick");

  let diagnostics = pleroma
    .read_resource::<WorldDiagnostics>(ResourceKey::Diagnostics)
    .unwrap();
  assert!(diagnostics.has_non_finite());
  assert_eq!(diagnostics.fields[&STATE].non_finite_cells, 1);
}

#[test]
fn off_policy_publishes_nothing() {
  let (tessera, mut pleroma) =
    world_with_state(DiagnosticsPolicy::Off, GOOD_STATE);
  tick(&tessera, &mut pleroma).expect("Off tick succeeds");

  let diagnostics = pleroma
    .read_resource::<WorldDiagnostics>(ResourceKey::Diagnostics)
    .unwrap();
  assert!(
    diagnostics.fields.is_empty(),
    "Off policy must not publish a report"
  );
}
