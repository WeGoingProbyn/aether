// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Phase-4 acceptance test for the IMEX split on `MoistEuler3D`: with the
//! acoustic terms (mass flux, pressure gradient, pressure work) integrated
//! implicitly and advection explicitly, the scheme stays stable at a step far
//! beyond the acoustic CFL — where fully-explicit Forward Euler diverges —
//! while conserving mass in a closed box. The implicit solve is over the
//! acoustic-only Jacobian.

use continuum::boundary::{BoundaryRegistry, ReflectiveWall};
use continuum::implicit::backend::ImplicitBackend;
use continuum::implicit::schemes::RosenbrockScheme;
use continuum::model::{MoistEuler3D, RusanovFlux};
use continuum::solver::{FvmSolver, SolverConfig, TimeIntegration};
use pleroma::core::storage::{CellView, FieldStorage, SoaField};
use tessera::geometry::{CellGeometry, IdentityMap};
use tessera::mesh::StructuredBlock;
use utility::domain::{BoundaryTag, CellId};

const GAMMA: f64 = 1.4;

fn initial_state(c: &utility::domain::Point<3>) -> [f64; 6] {
  let bump = 0.2 * (-((c[0] - 0.5) * (c[0] - 0.5)) / 0.02).exp();
  let rho = 1.0 + bump;
  let p = 1.0 + bump;
  [rho, 0.0, 0.0, 0.0, p / (GAMMA - 1.0), 0.01 * rho]
}

fn mesh() -> StructuredBlock<3> {
  StructuredBlock::uniform(
    [0.0, 0.0, 0.0].into(),
    [1.0, 0.25, 0.25],
    [24, 1, 1],
    Box::new(IdentityMap::<3>),
  )
}

fn closed_box_bcs() -> BoundaryRegistry<3, 6> {
  let mut bcs = BoundaryRegistry::new();
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
  bcs
}

fn make_state(mesh: &StructuredBlock<3>) -> SoaField<6> {
  SoaField::from_fn(mesh.cell_count(), |cell| {
    initial_state(mesh.cell_centroid(cell))
  })
}

fn total_mass(state: &SoaField<6>, cells: usize) -> f64 {
  (0..cells)
    .map(|i| state.state(CellId::from(i)).as_state()[0])
    .sum()
}

fn finite_positive(state: &SoaField<6>, cells: usize) -> bool {
  (0..cells).all(|i| {
    let s = state.state(CellId::from(i));
    let s = s.as_state();
    s.iter().all(|v| v.is_finite()) && s[0] > 0.0
  })
}

#[test]
fn imex_stable_beyond_acoustic_cfl_where_explicit_diverges() {
  let mesh = mesh();
  let cells = mesh.cell_count();
  let bcs = closed_box_bcs();

  let acoustic_cfl = FvmSolver::new(
    SolverConfig::new(0.5, 1.0, TimeIntegration::ForwardEuler),
    MoistEuler3D::new(GAMMA),
    RusanovFlux,
  )
  .compute_dt(&make_state(&mesh), &mesh);

  let big_dt = 30.0 * acoustic_cfl;
  let steps = 15;
  let config = SolverConfig::new(0.5, 1.0, TimeIntegration::ForwardEuler);

  // --- Fully explicit at big_dt: diverges. ---
  let mut explicit =
    FvmSolver::new(config.clone(), MoistEuler3D::new(GAMMA), RusanovFlux);
  let mut e_state = make_state(&mesh);
  let mut e_res = SoaField::zeros(cells);
  for _ in 0..steps {
    explicit.step_with_dt(big_dt, &mut e_state, &mut e_res, &mesh, &bcs);
  }
  assert!(
    !finite_positive(&e_state, cells),
    "explicit unexpectedly survived {big_dt} (30x acoustic CFL)"
  );

  // --- IMEX (acoustic implicit, advection explicit) at the same big_dt. ---
  let mut imex = FvmSolver::with_backend(
    config,
    MoistEuler3D::new(GAMMA),
    RusanovFlux,
    ImplicitBackend::<6>::new(
      RosenbrockScheme::LinearlyImplicitEuler.tableau(),
    )
    .with_imex()
    .with_auto_diff()
    .with_block_jacobi(),
  );
  let mut state = make_state(&mesh);
  let mut residual = SoaField::zeros(cells);
  let mass0 = total_mass(&state, cells);

  for _ in 0..steps {
    imex.step_with_dt(big_dt, &mut state, &mut residual, &mesh, &bcs);
    assert!(finite_positive(&state, cells), "IMEX diverged");
    assert!(
      imex.backend().last_report().unwrap().converged,
      "IMEX implicit solve did not converge"
    );
  }

  let mass1 = total_mass(&state, cells);
  assert!(
    (mass1 - mass0).abs() / mass0 < 1e-6,
    "mass drift {mass0} -> {mass1}"
  );
}
