// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Hydrostatic balance test for `Euler3D` with constant gravity. Initialises
//! a flat tall box with an isothermal Earth-like atmosphere in hydrostatic
//! equilibrium — `∂p/∂z = -ρ·g` — and verifies the solver doesn't blow it up
//! over many steps.
//!
//! This is a pre-well-balanced check: we don't expect the discrete pressure
//! gradient and the gravity source to cancel *exactly* (Rusanov + Transmissive
//! ghosts produce O(h²) imbalance at boundary cells), so some spurious
//! vertical motion is expected. The asserts are: density positive, energy
//! positive, no NaN, max |w| bounded well below sound speed, and density
//! doesn't crash by more than a factor of 2 from the analytical profile.

use continuum::boundary::{BoundaryRegistry, ReflectiveWall};
use continuum::model::{Euler3D, RusanovFlux};
use continuum::solver::{FvmSolver, SolverConfig, TimeIntegration};
use pleroma::core::storage::{FieldStorage, SoaField};
use tessera::geometry::{CellGeometry, IdentityMap};
use tessera::mesh::StructuredBlock;
use utility::domain::BoundaryTag;
use utility::domain::CellId;

#[test]
fn flat_atmosphere_remains_hydrostatic_under_gravity() {
  // Earth-like isothermal atmosphere parameters.
  let g = 9.81; // m/s²
  let gamma = 1.4;
  let r_air = 287.0; // J/(kg·K)
  let t0 = 288.0; // K (≈ 15 °C)
  let p0 = 101_325.0; // Pa
  let rho0 = p0 / (r_air * t0); // ≈ 1.225 kg/m³
  let h_scale = r_air * t0 / g; // ≈ 8430 m

  // 10 km tall column with isotropic cells (2.5 km wide). Horizontal extent
  // is physically arbitrary for the 1D hydrostatic test, but `compute_dt`
  // uses `vol^(1/D)` as the characteristic length and would violate CFL on
  // anisotropic cells. Cube-cells avoids that; the actual atmospheric
  // physics is independent of horizontal extent here.
  let z_max = 10_000.0;
  let dims = [8, 8, 32];
  let cell_edge = z_max / dims[2] as f64; // 312.5 m
  let mesh = StructuredBlock::uniform(
    [0.0, 0.0, 0.0].into(),
    [
      cell_edge * dims[0] as f64,
      cell_edge * dims[1] as f64,
      z_max,
    ],
    dims,
    Box::new(IdentityMap::<3>),
  );

  // Hydrostatic isothermal state: ρ(z) = ρ₀·exp(-z/H), p(z) = p₀·exp(-z/H).
  // u = v = w = 0; E = p/(γ-1) since kinetic energy is zero.
  let n_cells = mesh.cell_count();
  let init = |cell: CellId| -> [f64; 5] {
    let z = mesh.cell_centroid(cell)[2];
    let factor = (-z / h_scale).exp();
    let rho = rho0 * factor;
    let p = p0 * factor;
    let e = p / (gamma - 1.0);
    [rho, 0.0, 0.0, 0.0, e]
  };
  let mut state = SoaField::<5>::from_fn(n_cells, init);
  let mut residual = SoaField::<5>::zeros(n_cells);

  // Reflective walls on all 6 sides — atmosphere is a closed box.
  let mut bcs = BoundaryRegistry::<3, 5>::new();
  for tag in [
    BoundaryTag::Left,
    BoundaryTag::Right,
    BoundaryTag::Bottom,
    BoundaryTag::Top,
    BoundaryTag::Front,
    BoundaryTag::Back,
  ] {
    bcs.register(tag, ReflectiveWall);
  }

  let mut solver = FvmSolver::new(
    SolverConfig::new(0.5, 5.0, TimeIntegration::ForwardEuler),
    Euler3D::with_gravity(gamma, [0.0, 0.0, -g]),
    RusanovFlux,
  );

  // 50 steps × dt ~ dz/c ≈ 312/340 ≈ 0.9 s gives ~30–45 s of simulated time.
  // For a non-well-balanced scheme the spurious vertical wind at boundary
  // cells should still stay small (≪ sound speed) over this window.
  let n_steps = 50;
  for step in 0..n_steps {
    let dt = solver.step(&mut state, &mut residual, &mesh, &bcs);
    assert!(dt > 0.0 && dt.is_finite(), "step {}: dt={}", step, dt);
  }

  let mut max_w_abs = 0.0_f64;
  let mut min_rho = f64::INFINITY;
  let mut max_rho = 0.0_f64;
  let mut s = [0.0; 5];
  for i in 0..n_cells {
    let cell = CellId::from(i);
    state.state_into(cell, &mut s);
    let rho = s[0];
    let energy = s[4];
    assert!(rho > 0.0 && rho.is_finite(), "ρ={} at cell {}", rho, i);
    assert!(
      energy > 0.0 && energy.is_finite(),
      "E={} at cell {}",
      energy,
      i
    );
    let w = s[3] / rho;
    max_w_abs = max_w_abs.max(w.abs());
    min_rho = min_rho.min(rho);
    max_rho = max_rho.max(rho);
  }

  eprintln!(
    "hydrostatic test after {} steps: max |w| = {:.3} m/s, ρ ∈ [{:.4}, {:.4}] kg/m³",
    n_steps, max_w_abs, min_rho, max_rho
  );

  // Sound speed for this atmosphere: c = √(γRT) ≈ 340 m/s. Spurious vertical
  // motion should stay well under that.
  assert!(
    max_w_abs < 50.0,
    "max |w| = {:.3} m/s grew too large after {} steps",
    max_w_abs,
    n_steps
  );

  // Density bounds: highest pressure at z=0, lowest at z=z_max.
  let expected_max = rho0; // at z = 0
  let expected_min = rho0 * (-z_max / h_scale).exp(); // ≈ 0.37 kg/m³

  assert!(
    max_rho < 1.5 * expected_max,
    "max ρ = {:.4} grew above 1.5 × ρ₀ ({:.4})",
    max_rho,
    expected_max
  );
  assert!(
    min_rho > 0.5 * expected_min,
    "min ρ = {:.4} fell below 0.5 × expected min ({:.4})",
    min_rho,
    expected_min
  );
}
