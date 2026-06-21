// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Well-balancing: a hydrostatic atmosphere at rest (∇p = ρg) must stay at
//! rest. The ordinary finite-volume scheme spuriously accelerates it (the face
//! pressure from averaging cell centres disagrees with the cell-centred ρg
//! source); hydrostatic reconstruction balances `flux − source = 0` to machine
//! precision. These tests pin both: the residual is ~0 and the state does not
//! drift, where the un-balanced scheme produces a large residual and visible
//! motion.

use continuum::boundary::{BoundaryRegistry, ReflectiveWall};
use continuum::model::{Euler3D, RusanovFlux};
use continuum::solver::{FvmSolver, SolverConfig, TimeIntegration};
use pleroma::core::storage::{CellView, FieldStorage, SoaField};
use tessera::cube_sphere::CubeSphere;
use tessera::geometry::CellGeometry;
use utility::domain::{BoundaryTag, CellId};

const GAMMA: f64 = 1.4;
const R0: f64 = 6.371e6;
const HEIGHT: f64 = 20_000.0;
const G: f64 = 9.81;
const RT: f64 = 287.0 * 288.0; // gas constant × reference temperature
const P0: f64 = 1.0e5;

fn shell(na: usize, nr: usize) -> CubeSphere {
  CubeSphere::new([na, na, nr], R0, R0 + HEIGHT)
}

/// Isothermal hydrostatic state at rest: `p(r) = p0·exp(-(r-r0)/H)`,
/// `H = RT/g`, `ρ = p/RT`, `E = p/(γ-1)` (no kinetic energy).
fn hydrostatic_state(mesh: &CubeSphere) -> SoaField<5> {
  let h = RT / G;
  SoaField::from_fn(mesh.cell_count(), |cell| {
    let c = mesh.cell_world_centroid(cell);
    let r = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
    let p = P0 * (-(r - R0) / h).exp();
    let rho = p / RT;
    [rho, 0.0, 0.0, 0.0, p / (GAMMA - 1.0)]
  })
}

fn law(mesh: &CubeSphere, well_balanced: bool) -> Euler3D {
  Euler3D::with_radial_gravity(GAMMA, G).well_balanced(well_balanced)
}

fn bcs() -> BoundaryRegistry<3, 5> {
  let mut bcs = BoundaryRegistry::new();
  bcs.register(BoundaryTag::Ground, ReflectiveWall);
  bcs.register(BoundaryTag::AtmosphereEdge, ReflectiveWall);
  bcs
}

/// Largest momentum-residual magnitude over all cells (the spurious force per
/// unit volume, kg/(m²·s²)).
fn max_momentum_residual(res: &SoaField<5>) -> f64 {
  (0..res.len())
    .map(|i| {
      let s = res.state(CellId::from(i));
      let s = s.as_state();
      (s[1] * s[1] + s[2] * s[2] + s[3] * s[3]).sqrt()
    })
    .fold(0.0, f64::max)
}

#[test]
fn well_balanced_residual_is_machine_zero() {
  let mesh = shell(8, 8);
  let state = hydrostatic_state(&mesh);
  let bcs = bcs();
  let config = SolverConfig::new(0.5, 1.0, TimeIntegration::ForwardEuler);
  let mut res = SoaField::<5>::zeros(mesh.cell_count());

  let plain = FvmSolver::new(config.clone(), law(&mesh, false), RusanovFlux);
  plain.compute_residual(&state, &mut res, &mesh, &bcs);
  let plain_max = max_momentum_residual(&res);

  let balanced = FvmSolver::new(config, law(&mesh, true), RusanovFlux);
  balanced.compute_residual(&state, &mut res, &mesh, &bcs);
  let balanced_max = max_momentum_residual(&res);

  eprintln!(
    "max |momentum residual|: plain {plain_max:.3e}, balanced {balanced_max:.3e}"
  );
  // The un-balanced scheme leaves a real spurious force.
  assert!(
    plain_max > 1.0,
    "expected a large un-balanced residual, got {plain_max:.3e}"
  );
  // Well-balancing kills it to machine precision (≳10 orders down).
  assert!(
    balanced_max < 1e-6,
    "well-balanced residual not machine-zero: {balanced_max:.3e}"
  );
  assert!(balanced_max < plain_max * 1e-8);
}

#[test]
fn well_balanced_atmosphere_does_not_drift() {
  let mesh = shell(8, 8);
  let bcs = bcs();
  let config = SolverConfig::new(0.4, 1.0, TimeIntegration::ForwardEuler);

  let speed = |state: &SoaField<5>| -> f64 {
    (0..state.len())
      .map(|i| {
        let s = state.state(CellId::from(i));
        let s = s.as_state();
        (s[1] * s[1] + s[2] * s[2] + s[3] * s[3]).sqrt() / s[0]
      })
      .fold(0.0, f64::max)
  };

  // Balanced: step well past the acoustic time and stay essentially at rest.
  let mut balanced =
    FvmSolver::new(config.clone(), law(&mesh, true), RusanovFlux);
  let mut state = hydrostatic_state(&mesh);
  let mut res = SoaField::<5>::zeros(mesh.cell_count());
  for _ in 0..200 {
    balanced.step(&mut state, &mut res, &mesh, &bcs);
  }
  let balanced_speed = speed(&state);

  // Un-balanced: the same run develops a real spurious wind.
  let mut plain = FvmSolver::new(config, law(&mesh, false), RusanovFlux);
  let mut pstate = hydrostatic_state(&mesh);
  for _ in 0..200 {
    plain.step(&mut pstate, &mut res, &mesh, &bcs);
  }
  let plain_speed = speed(&pstate);

  eprintln!(
    "max speed after 200 steps: balanced {balanced_speed:.3e} m/s, \
     plain {plain_speed:.3e} m/s"
  );
  assert!(
    balanced_speed < 1e-6,
    "balanced atmosphere drifted: {balanced_speed:.3e} m/s"
  );
  assert!(
    plain_speed > 1e-2,
    "expected the un-balanced scheme to drift, got {plain_speed:.3e} m/s"
  );
}
