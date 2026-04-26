// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Smoke test: drive `FvmSolver::step` on a `CubeSphere` mesh starting from
//! a constant state at rest. Verifies that the mesh is wired into the solver
//! correctly — mass and energy fluxes vanish identically at every face, so ρ
//! and E should remain *exactly* unchanged on every cell. Momentum should
//! stay near zero globally (pressure × normal sums to zero over the closed
//! sphere by panel-pair symmetry); per-cell momentum drift is allowed because
//! the world-frame area vectors are midpoint approximations on a curved grid.

use continuum::{
  boundary::{BoundaryRegistry, Transmissive},
  cube_sphere::CubeSphere,
  field::{FieldStorage, SoaField},
  geometry::{CellGeometry, CellId},
  model::{Euler3D, RusanovFlux},
  solver::{FvmSolver, SolverConfig, TimeIntegration},
  topology::BoundaryTag,
};

/// Sum component `c` of state weighted by physical cell volume.
fn integrate_component(
  state: &SoaField<5>,
  mesh: &CubeSphere,
  c: usize,
) -> f64 {
  let mut total = 0.0;
  let mut s = [0.0; 5];
  for i in 0..mesh.cell_count() {
    let cell = CellId::from(i);
    state.state_into(cell, &mut s);
    let phys_vol = mesh.cell_volume(cell) * mesh.cell_metrics(cell).sqrt_metric;
    total += s[c] * phys_vol;
  }
  total
}

#[test]
fn cube_sphere_constant_state_at_rest() {
  let dims = [4, 4, 2];
  let mesh = CubeSphere::new(dims, 1.0, 2.0);

  let gamma = 1.4;
  let rho0 = 1.0;
  let p0 = 1.0;
  let e0 = p0 / (gamma - 1.0); // u=v=w=0 → E = p/(γ-1)
  let initial: [f64; 5] = [rho0, 0.0, 0.0, 0.0, e0];

  let n_cells = mesh.cell_count();
  let mut state = SoaField::<5>::from_fn(n_cells, |_| initial);
  let mut residual = SoaField::<5>::zeros(n_cells);

  let mut bcs = BoundaryRegistry::<3, 5>::new();
  bcs.register(BoundaryTag::Ground, Transmissive);
  bcs.register(BoundaryTag::AtmosphereEdge, Transmissive);

  let mut solver = FvmSolver::new(
    SolverConfig::new(0.5, 0.1, TimeIntegration::ForwardEuler),
    Euler3D::new(gamma),
    RusanovFlux,
  );

  let mass_before = integrate_component(&state, &mesh, 0);
  let energy_before = integrate_component(&state, &mesh, 4);

  let dt = solver.step(&mut state, &mut residual, &mesh, &bcs);
  assert!(
    dt > 0.0 && dt.is_finite(),
    "dt should be positive and finite, got {}",
    dt
  );

  // ----- Per-cell invariants: ρ and E exactly unchanged -----
  // Mass flux is ρu·n; with u = 0 everywhere, F_ρ = 0 at every face on every
  // panel. Same for energy flux (E+p)u·n. Per-cell ρ and E should not move
  // by even one ulp.
  let mut s = [0.0; 5];
  for i in 0..n_cells {
    let cell = CellId::from(i);
    state.state_into(cell, &mut s);
    let drho = (s[0] - rho0).abs();
    let de = (s[4] - e0).abs();
    assert!(
      drho < 1e-13,
      "cell {}: ρ drifted by {} (expected exact)",
      i,
      drho
    );
    assert!(
      de < 1e-13,
      "cell {}: E drifted by {} (expected exact)",
      i,
      de
    );
  }

  // ----- Global invariants: total mass and energy unchanged -----
  let mass_after = integrate_component(&state, &mesh, 0);
  let energy_after = integrate_component(&state, &mesh, 4);
  let mass_drift = (mass_after - mass_before).abs() / mass_before;
  let energy_drift = (energy_after - energy_before).abs() / energy_before;
  assert!(mass_drift < 1e-13, "Σρ drift = {}", mass_drift);
  assert!(energy_drift < 1e-13, "ΣE drift = {}", energy_drift);

  // ----- Global momentum: net force on a closed sphere is zero -----
  // Each Ground/AtmosphereEdge face contributes -p·n_out·A·sqrt_metric to
  // the residual. By panel-pair (XP/XN, YP/YN, ZP/ZN) symmetry of the cube
  // sphere, the boundary normals cancel in each component, so the global
  // sum should be ≈ 0. Per-cell momentum can be non-zero (curvature gives a
  // non-vanishing discrete pressure gradient) but the integral should not.
  for c in 1..=3 {
    let p_total = integrate_component(&state, &mesh, c);
    assert!(
      p_total.abs() < 1e-10 * mass_before,
      "Σρu_{} = {} (mass = {})",
      c,
      p_total,
      mass_before
    );
  }
}
