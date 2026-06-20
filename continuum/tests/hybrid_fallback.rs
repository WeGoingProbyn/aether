// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Phase-3 acceptance test for the hybrid implicit/explicit backend: it accepts
//! large implicit steps when valid, and when the implicit attempt is rejected
//! it falls back to explicit sub-steps so the field still advances and stays
//! conservative.

use continuum::boundary::{BoundaryRegistry, ReflectiveWall};
use continuum::implicit::backend::ImplicitBackend;
use continuum::implicit::hybrid::HybridBackend;
use continuum::implicit::schemes::RosenbrockScheme;
use continuum::model::{Euler2D, RusanovFlux};
use continuum::solver::{FvmSolver, SolverConfig, TimeIntegration};
use pleroma::core::storage::{CellView, FieldStorage, SoaField};
use tessera::geometry::{CellGeometry, IdentityMap};
use tessera::mesh::StructuredBlock;
use utility::domain::{BoundaryTag, CellId};

const GAMMA: f64 = 1.4;

fn initial_state(x: f64) -> [f64; 4] {
  let bump = (-((x - 0.5) * (x - 0.5)) / 0.01).exp();
  let rho = 1.0 + 0.2 * bump;
  let p = 1.0 + 0.2 * bump;
  [rho, 0.0, 0.0, p / (GAMMA - 1.0)]
}

fn closed_box_bcs() -> BoundaryRegistry<2, 4> {
  let mut bcs = BoundaryRegistry::new();
  for tag in [
    BoundaryTag::Left,
    BoundaryTag::Right,
    BoundaryTag::Bottom,
    BoundaryTag::Top,
  ] {
    bcs.register(tag, ReflectiveWall);
  }
  bcs
}

fn mesh() -> StructuredBlock<2> {
  StructuredBlock::uniform(
    [0.0, 0.0].into(),
    [1.0, 0.05],
    [32, 1],
    Box::new(IdentityMap::<2>),
  )
}

fn make_state(mesh: &StructuredBlock<2>) -> SoaField<4> {
  SoaField::from_fn(mesh.cell_count(), |cell| {
    initial_state(mesh.cell_centroid(cell)[0])
  })
}

fn total_mass(state: &SoaField<4>, cells: usize) -> f64 {
  (0..cells)
    .map(|i| state.state(CellId::from(i)).as_state()[0])
    .sum()
}

fn finite_positive(state: &SoaField<4>, cells: usize) -> bool {
  (0..cells).all(|i| {
    let s = state.state(CellId::from(i));
    let s = s.as_state();
    s.iter().all(|v| v.is_finite()) && s[0] > 0.0
  })
}

fn changed(a: &SoaField<4>, b: &SoaField<4>, cells: usize) -> f64 {
  let mut m = 0.0_f64;
  for i in 0..cells {
    let x = a.state(CellId::from(i));
    let y = b.state(CellId::from(i));
    for k in 0..4 {
      m = m.max((x.as_state()[k] - y.as_state()[k]).abs());
    }
  }
  m
}

#[test]
fn hybrid_accepts_implicit_when_valid() {
  let mesh = mesh();
  let cells = mesh.cell_count();
  let bcs = closed_box_bcs();
  let cfl_dt = FvmSolver::new(
    SolverConfig::new(0.5, 1.0, TimeIntegration::ForwardEuler),
    Euler2D::new(GAMMA),
    RusanovFlux,
  )
  .compute_dt(&make_state(&mesh), &mesh);
  // A step the implicit solver comfortably converges on (well past explicit
  // CFL, but within GMRES's reach) so the hybrid accepts it every tick.
  let big_dt = 4.0 * cfl_dt;
  let config = SolverConfig::new(0.5, 1.0, TimeIntegration::ForwardEuler);

  let mut solver = FvmSolver::with_backend(
    config,
    Euler2D::new(GAMMA),
    RusanovFlux,
    HybridBackend::<4>::new(
      ImplicitBackend::new(RosenbrockScheme::LinearlyImplicitEuler.tableau())
        .with_auto_diff(),
    ),
  );
  let mut state = make_state(&mesh);
  let mut residual = SoaField::zeros(cells);
  for _ in 0..10 {
    solver.step_with_dt(big_dt, &mut state, &mut residual, &mesh, &bcs);
  }
  assert!(finite_positive(&state, cells), "hybrid diverged");
  assert!(
    solver.backend().report().implicit_accepted >= 9,
    "expected mostly accepted implicit steps, got {:?}",
    solver.backend().report()
  );
  assert_eq!(solver.backend().report().fallbacks, 0);
}

#[test]
fn hybrid_falls_back_and_still_advances_conservatively() {
  let mesh = mesh();
  let cells = mesh.cell_count();
  let bcs = closed_box_bcs();
  let cfl_dt = FvmSolver::new(
    SolverConfig::new(0.5, 1.0, TimeIntegration::ForwardEuler),
    Euler2D::new(GAMMA),
    RusanovFlux,
  )
  .compute_dt(&make_state(&mesh), &mesh);
  let big_dt = 20.0 * cfl_dt;
  let config = SolverConfig::new(0.5, 1.0, TimeIntegration::ForwardEuler);

  // An impossibly tight motion bound rejects every implicit step, forcing the
  // explicit fallback path on every tick.
  let mut solver = FvmSolver::with_backend(
    config,
    Euler2D::new(GAMMA),
    RusanovFlux,
    HybridBackend::<4>::new(ImplicitBackend::new(
      RosenbrockScheme::LinearlyImplicitEuler.tableau(),
    ))
    .with_max_relative_step(1e-12),
  );
  let mut state = make_state(&mesh);
  let mut residual = SoaField::zeros(cells);
  let initial = make_state(&mesh);

  let mass0 = total_mass(&state, cells);
  let steps = 10;
  for _ in 0..steps {
    solver.step_with_dt(big_dt, &mut state, &mut residual, &mesh, &bcs);
  }

  let report = solver.backend().report();
  assert_eq!(report.implicit_accepted, 0, "expected only fallbacks");
  assert_eq!(report.fallbacks, steps);
  assert!(
    report.explicit_substeps > steps,
    "fallback should take several sub-steps per tick"
  );

  // The field still advanced, stayed physical, and conserved mass.
  assert!(finite_positive(&state, cells), "fallback diverged");
  assert!(
    changed(&state, &initial, cells) > 1e-4,
    "field did not advance"
  );
  let mass1 = total_mass(&state, cells);
  assert!((mass1 - mass0).abs() / mass0 < 1e-9, "mass not conserved");
}
