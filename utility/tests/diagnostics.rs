// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Smoke tests for the diagnostics trait surface and the
//! `#[derive(StateDiagnostics)]` proc-macro. Verifies the three trait
//! impls expand correctly for a fixture struct that mirrors the
//! Euler-style usage pattern.

use utility::StateDiagnostics;
use utility::diagnostics::{
  ConservationQuantities, Diagnostics, ExtraDiagnostics,
};

#[derive(StateDiagnostics)]
#[diagnostics(
  components("rho", "rho_u", "energy"),
  conserved(("mass", 0), ("momentum", 1), ("total_energy", 2)),
  extras(
    ("u", self.velocity(state)),
    ("kinetic", self.kinetic(state)),
    ("pressure", self.pressure(state)),
  ),
)]
struct SimpleEuler1D {
  gamma: f64,
}

impl SimpleEuler1D {
  fn velocity(&self, state: &[f64; 3]) -> f64 {
    state[1] / state[0]
  }
  fn kinetic(&self, state: &[f64; 3]) -> f64 {
    let u = self.velocity(state);
    0.5 * state[0] * u * u
  }
  fn pressure(&self, state: &[f64; 3]) -> f64 {
    (self.gamma - 1.0) * (state[2] - self.kinetic(state))
  }
}

#[test]
fn diagnostics_yields_named_components() {
  let law = SimpleEuler1D { gamma: 1.4 };
  let state = [2.0, 4.0, 50.0];
  let pairs: Vec<(&'static str, f64)> = law.diagnostics(&state).collect();
  assert_eq!(pairs, vec![("rho", 2.0), ("rho_u", 4.0), ("energy", 50.0)]);
}

#[test]
fn extras_yields_named_derived_quantities() {
  let law = SimpleEuler1D { gamma: 1.4 };
  let state = [2.0, 4.0, 50.0];
  let pairs: Vec<(&'static str, f64)> = law.extras(&state).collect();
  // u = 4/2 = 2, kinetic = 0.5 * 2 * 4 = 4, pressure = 0.4 * (50 - 4) = 18.4
  assert_eq!(pairs.len(), 3);
  assert_eq!(pairs[0], ("u", 2.0));
  assert_eq!(pairs[1], ("kinetic", 4.0));
  assert_eq!(pairs[2].0, "pressure");
  assert!((pairs[2].1 - 18.4).abs() < 1e-12);
}

#[test]
fn conserved_quantities_metadata_lists_named_components() {
  let names_and_components: Vec<(&'static str, usize)> =
    SimpleEuler1D::CONSERVED_QUANTITIES
      .iter()
      .map(|q| (q.name, q.component))
      .collect();
  assert_eq!(
    names_and_components,
    vec![("mass", 0), ("momentum", 1), ("total_energy", 2)]
  );
}

#[test]
fn component_names_const_is_directly_accessible() {
  assert_eq!(SimpleEuler1D::COMPONENT_NAMES, ["rho", "rho_u", "energy"]);
}

/// A struct with only `components(...)` exercises the path where neither
/// `extras` nor `conserved` are emitted.
#[derive(StateDiagnostics)]
#[diagnostics(components("a", "b"))]
struct Minimal;

#[test]
fn components_only_produces_only_diagnostics_impl() {
  assert_eq!(Minimal::COMPONENT_NAMES, ["a", "b"]);
  let pairs: Vec<(&'static str, f64)> =
    Minimal.diagnostics(&[1.0, 2.0]).collect();
  assert_eq!(pairs, vec![("a", 1.0), ("b", 2.0)]);
}
