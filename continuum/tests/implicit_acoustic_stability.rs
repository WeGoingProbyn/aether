// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Phase-1 acceptance test for the implicit backend: on a stiff acoustic
//! problem it stays stable at a step far beyond the explicit CFL limit (where
//! Forward Euler diverges), while conserving mass and energy in a closed box.

use continuum::boundary::{BoundaryRegistry, ReflectiveWall};
use continuum::implicit::backend::ImplicitBackend;
use continuum::implicit::schemes::RosenbrockScheme;
use continuum::model::{Euler2D, RusanovFlux};
use continuum::solver::{FvmSolver, SolverConfig, TimeIntegration};
use pleroma::core::storage::{CellView, FieldStorage, SoaField};
use tessera::geometry::{CellGeometry, IdentityMap};
use tessera::mesh::StructuredBlock;
use utility::domain::{BoundaryTag, CellId};

const GAMMA: f64 = 1.4;

/// A smooth pressure / density bump at rest — an acoustic pulse that will ring
/// across the box at the sound speed.
fn initial_state(x: f64) -> [f64; 4] {
  let bump = (-((x - 0.5) * (x - 0.5)) / 0.01).exp();
  let rho = 1.0 + 0.2 * bump;
  let p = 1.0 + 0.2 * bump;
  [rho, 0.0, 0.0, p / (GAMMA - 1.0)]
}

fn closed_box_bcs() -> BoundaryRegistry<2, 4> {
  let mut bcs = BoundaryRegistry::new();
  bcs.register(BoundaryTag::Left, ReflectiveWall);
  bcs.register(BoundaryTag::Right, ReflectiveWall);
  bcs.register(BoundaryTag::Bottom, ReflectiveWall);
  bcs.register(BoundaryTag::Top, ReflectiveWall);
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

fn total_energy(state: &SoaField<4>, cells: usize) -> f64 {
  (0..cells)
    .map(|i| state.state(CellId::from(i)).as_state()[3])
    .sum()
}

fn all_finite_positive(state: &SoaField<4>, cells: usize) -> bool {
  (0..cells).all(|i| {
    let s = state.state(CellId::from(i));
    let s = s.as_state();
    s.iter().all(|v| v.is_finite()) && s[0] > 0.0
  })
}

#[test]
fn implicit_stable_at_large_dt_where_explicit_diverges() {
  let mesh = mesh();
  let cells = mesh.cell_count();
  let bcs = closed_box_bcs();

  // Explicit CFL step for this mesh / state.
  let probe = FvmSolver::new(
    SolverConfig::new(0.5, 1.0, TimeIntegration::ForwardEuler),
    Euler2D::new(GAMMA),
    RusanovFlux,
  );
  let cfl_dt = probe.compute_dt(&make_state(&mesh), &mesh);
  assert!(cfl_dt > 0.0 && cfl_dt.is_finite());

  // Step well beyond the acoustic CFL limit.
  let big_dt = 50.0 * cfl_dt;
  let steps = 20;

  // --- Explicit Forward Euler at big_dt: expected to diverge. ---
  let mut explicit = FvmSolver::new(
    SolverConfig::new(0.5, 1.0, TimeIntegration::ForwardEuler),
    Euler2D::new(GAMMA),
    RusanovFlux,
  );
  let mut e_state = make_state(&mesh);
  let mut e_res = SoaField::zeros(cells);
  for _ in 0..steps {
    explicit.step_with_dt(big_dt, &mut e_state, &mut e_res, &mesh, &bcs);
  }
  assert!(
    !all_finite_positive(&e_state, cells),
    "explicit Forward Euler unexpectedly stayed stable at {big_dt} \
     (50x CFL {cfl_dt}) — the test premise is wrong"
  );

  // --- Implicit (linearly-implicit Euler) at the same big_dt: stable. ---
  let mut implicit = FvmSolver::with_backend(
    SolverConfig::new(0.5, 1.0, TimeIntegration::ForwardEuler),
    Euler2D::new(GAMMA),
    RusanovFlux,
    ImplicitBackend::<4>::new(
      RosenbrockScheme::LinearlyImplicitEuler.tableau(),
    ),
  );
  let mut i_state = make_state(&mesh);
  let mut i_res = SoaField::zeros(cells);

  let mass0 = total_mass(&i_state, cells);
  let energy0 = total_energy(&i_state, cells);

  for _ in 0..steps {
    implicit.step_with_dt(big_dt, &mut i_state, &mut i_res, &mesh, &bcs);
    assert!(
      all_finite_positive(&i_state, cells),
      "implicit backend diverged at {big_dt}"
    );
  }

  // Closed box: mass and energy conserved to the linear-solve tolerance.
  let mass1 = total_mass(&i_state, cells);
  let energy1 = total_energy(&i_state, cells);
  assert!(
    (mass1 - mass0).abs() / mass0 < 1e-6,
    "mass drift {} -> {}",
    mass0,
    mass1
  );
  assert!(
    (energy1 - energy0).abs() / energy0 < 1e-4,
    "energy drift {} -> {}",
    energy0,
    energy1
  );
}
