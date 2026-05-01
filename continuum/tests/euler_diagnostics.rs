// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Verifies the `#[derive(StateDiagnostics)]` impls on `Euler2D` /
//! `Euler3D` produce the right component names, conserved-quantity
//! metadata, and derived-quantity values.

use continuum::model::{Euler2D, Euler3D};
use utility::diagnostics::{
  ConservationQuantities, Diagnostics, ExtraDiagnostics,
};

#[test]
fn euler3d_components_match_field_names() {
  assert_eq!(
    Euler3D::COMPONENT_NAMES,
    ["rho", "rho_u", "rho_v", "rho_w", "energy"]
  );
}

#[test]
fn euler3d_conserved_quantities_cover_full_state() {
  let pairs: Vec<(&'static str, usize)> = Euler3D::CONSERVED_QUANTITIES
    .iter()
    .map(|q| (q.name, q.component))
    .collect();
  assert_eq!(
    pairs,
    vec![
      ("mass", 0),
      ("momentum_x", 1),
      ("momentum_y", 2),
      ("momentum_z", 3),
      ("total_energy", 4),
    ]
  );
}

#[test]
fn euler3d_extras_match_law_helpers() {
  let law = Euler3D::new(1.4);
  // Pick an arbitrary state with rho=2, momentum=(4, -2, 6), energy=80.
  let state = [2.0, 4.0, -2.0, 6.0, 80.0];
  let extras: Vec<(&'static str, f64)> = law.extras(&state).collect();

  assert_eq!(extras.len(), 6);
  let by_name: std::collections::HashMap<&'static str, f64> =
    extras.into_iter().collect();

  let u: f64 = 4.0 / 2.0; // 2.0
  let v: f64 = -2.0 / 2.0; // -1.0
  let w: f64 = 6.0 / 2.0; // 3.0
  let speed = (u * u + v * v + w * w).sqrt();
  let ke: f64 = 0.5 / 2.0 * (4.0 * 4.0 + (-2.0_f64) * (-2.0) + 6.0 * 6.0);
  let pressure = (1.4 - 1.0) * (80.0 - ke);

  assert_eq!(by_name["u"], u);
  assert_eq!(by_name["v"], v);
  assert_eq!(by_name["w"], w);
  assert!((by_name["speed"] - speed).abs() < 1e-12);
  assert!((by_name["kinetic_energy_density"] - ke).abs() < 1e-12);
  assert!((by_name["pressure"] - pressure).abs() < 1e-12);
}

#[test]
fn euler2d_derive_emits_all_three_traits() {
  assert_eq!(
    Euler2D::COMPONENT_NAMES,
    ["rho", "rho_u", "rho_v", "energy"]
  );
  assert_eq!(Euler2D::CONSERVED_QUANTITIES.len(), 4);
  assert_eq!(Euler2D::CONSERVED_QUANTITIES[0].name, "mass");

  let law = Euler2D::new(1.4);
  let state = [1.0, 2.0, 3.0, 20.0];
  let pairs: Vec<(&'static str, f64)> = law.diagnostics(&state).collect();
  assert_eq!(
    pairs,
    vec![
      ("rho", 1.0),
      ("rho_u", 2.0),
      ("rho_v", 3.0),
      ("energy", 20.0),
    ]
  );

  let extras: Vec<(&'static str, f64)> = law.extras(&state).collect();
  assert_eq!(extras.len(), 5);
}
