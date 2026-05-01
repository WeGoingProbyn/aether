// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Integration test for `continuum::diagnostics::integrate_conserved`.
//! Builds a small cube-sphere shell, fills it with a known constant state,
//! integrates each declared conserved component, and verifies the totals
//! equal `component_value * sum_of_cell_volumes`.

use continuum::diagnostics::integrate_conserved;
use continuum::model::Euler3D;
use pleroma::core::storage::SoaField;
use tessera::cube_sphere::CubeSphere;
use tessera::geometry::CellGeometry;
use utility::domain::CellId;

#[test]
fn integrates_each_conserved_component_with_volume_weighting() {
  let mesh = CubeSphere::new([4, 4, 2], 1.0, 2.0);
  let n_cells = mesh.cell_count();

  // Constant state across the mesh: rho=2, momentum=(3,5,7), energy=11.
  let state = [2.0, 3.0, 5.0, 7.0, 11.0];
  let field = SoaField::<5>::from_fn(n_cells, |_| state);

  let total_volume: f64 = (0..n_cells)
    .map(|i| mesh.cell_volume(CellId::from(i)))
    .sum();

  let law = Euler3D::new(1.4);
  let totals: std::collections::HashMap<&'static str, f64> =
    integrate_conserved(&law, &mesh, &field)
      .into_iter()
      .collect();

  // Every conserved name should be present.
  assert_eq!(totals.len(), 5);
  for name in [
    "mass",
    "momentum_x",
    "momentum_y",
    "momentum_z",
    "total_energy",
  ] {
    assert!(totals.contains_key(name), "missing total for {name}");
  }

  // For a constant state, total = state[component] * total_volume.
  assert!((totals["mass"] - state[0] * total_volume).abs() < 1e-9);
  assert!((totals["momentum_x"] - state[1] * total_volume).abs() < 1e-9);
  assert!((totals["momentum_y"] - state[2] * total_volume).abs() < 1e-9);
  assert!((totals["momentum_z"] - state[3] * total_volume).abs() < 1e-9);
  assert!((totals["total_energy"] - state[4] * total_volume).abs() < 1e-9);
}

#[test]
fn returns_totals_in_declared_order() {
  let mesh = CubeSphere::new([2, 2, 1], 1.0, 1.5);
  let n_cells = mesh.cell_count();
  let field = SoaField::<5>::from_fn(n_cells, |_| [1.0, 0.0, 0.0, 0.0, 0.0]);

  let law = Euler3D::new(1.4);
  let totals: Vec<&'static str> = integrate_conserved(&law, &mesh, &field)
    .into_iter()
    .map(|(name, _)| name)
    .collect();
  assert_eq!(
    totals,
    vec![
      "mass",
      "momentum_x",
      "momentum_y",
      "momentum_z",
      "total_energy"
    ]
  );
}
