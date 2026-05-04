// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use continuum::boundary::{BoundaryRegistry, Transmissive};
use continuum::cpu::CpuBackend;
use continuum::model::{Euler2D, RusanovFlux};
use continuum::solver::{FvmSolver, SolverConfig, TimeIntegration};
use pleroma::core::storage::{CellView, FieldStorage, SoaField};
use tessera::geometry::{CellGeometry, IdentityMap};
use tessera::mesh::StructuredBlock;
use utility::domain::{BoundaryTag, CellId};

fn initial_state(x: f64, gamma: f64) -> [f64; 4] {
  if x < 0.5 {
    let rho = 1.0;
    let p = 1.0;
    [rho, 0.0, 0.0, p / (gamma - 1.0)]
  } else {
    let rho = 0.125;
    let p = 0.1;
    [rho, 0.0, 0.0, p / (gamma - 1.0)]
  }
}

fn transmissive_bcs() -> BoundaryRegistry<2, 4> {
  let mut bcs = BoundaryRegistry::new();
  bcs.register(BoundaryTag::Left, Transmissive);
  bcs.register(BoundaryTag::Right, Transmissive);
  bcs.register(BoundaryTag::Bottom, Transmissive);
  bcs.register(BoundaryTag::Top, Transmissive);
  bcs
}

#[test]
fn with_backend_matches_default_cpu_backend_and_updates_solver_state() {
  let gamma = 1.4;
  let dims = [8, 1];
  let mesh = StructuredBlock::uniform(
    [0.0, 0.0].into(),
    [1.0, 0.01],
    dims,
    Box::new(IdentityMap::<2>),
  );

  let mut default_state = SoaField::from_fn(mesh.cell_count(), |cell| {
    let x = mesh.cell_centroid(cell)[0];
    initial_state(x, gamma)
  });
  let mut explicit_state = SoaField::from_fn(mesh.cell_count(), |cell| {
    let x = mesh.cell_centroid(cell)[0];
    initial_state(x, gamma)
  });
  let mut default_residual = SoaField::zeros(mesh.cell_count());
  let mut explicit_residual = SoaField::zeros(mesh.cell_count());

  let bcs = transmissive_bcs();
  let config = SolverConfig::new(0.5, 1e-4, TimeIntegration::Rk2);
  let mut default_solver =
    FvmSolver::new(config.clone(), Euler2D::new(gamma), RusanovFlux);
  let mut explicit_solver = FvmSolver::with_backend(
    config,
    Euler2D::new(gamma),
    RusanovFlux,
    CpuBackend::default(),
  );

  let dt_default =
    default_solver.step(&mut default_state, &mut default_residual, &mesh, &bcs);
  let dt_explicit = explicit_solver.step(
    &mut explicit_state,
    &mut explicit_residual,
    &mesh,
    &bcs,
  );

  assert!((dt_explicit - dt_default).abs() < 1e-14);
  assert_eq!(default_solver.current_step(), 1);
  assert_eq!(explicit_solver.current_step(), 1);
  assert!((default_solver.time() - dt_default).abs() < 1e-14);
  assert!((explicit_solver.time() - dt_explicit).abs() < 1e-14);

  for i in 0..mesh.cell_count() {
    let default = default_state.state(CellId::from(i));
    let explicit = explicit_state.state(CellId::from(i));
    for (k, default_component) in default.as_state().iter().enumerate() {
      assert!(
        (default_component - explicit.as_state()[k]).abs() < 1e-14,
        "cell {} component {} differs: default={} explicit={}",
        i,
        k,
        default.as_state()[k],
        explicit.as_state()[k],
      );
    }
  }
}

#[test]
fn step_with_dt_matches_step_when_given_computed_dt() {
  let gamma = 1.4;
  let dims = [8, 1];
  let mesh = StructuredBlock::uniform(
    [0.0, 0.0].into(),
    [1.0, 0.01],
    dims,
    Box::new(IdentityMap::<2>),
  );

  let mut step_state = SoaField::from_fn(mesh.cell_count(), |cell| {
    let x = mesh.cell_centroid(cell)[0];
    initial_state(x, gamma)
  });
  let mut fixed_dt_state = SoaField::from_fn(mesh.cell_count(), |cell| {
    let x = mesh.cell_centroid(cell)[0];
    initial_state(x, gamma)
  });
  let mut step_residual = SoaField::zeros(mesh.cell_count());
  let mut fixed_dt_residual = SoaField::zeros(mesh.cell_count());

  let bcs = transmissive_bcs();
  let config = SolverConfig::new(0.5, 1e-4, TimeIntegration::Rk2);
  let mut step_solver =
    FvmSolver::new(config.clone(), Euler2D::new(gamma), RusanovFlux);
  let mut fixed_dt_solver =
    FvmSolver::new(config, Euler2D::new(gamma), RusanovFlux);

  let dt = fixed_dt_solver.compute_dt(&fixed_dt_state, &mesh);
  let step_dt =
    step_solver.step(&mut step_state, &mut step_residual, &mesh, &bcs);
  let fixed_dt = fixed_dt_solver.step_with_dt(
    dt,
    &mut fixed_dt_state,
    &mut fixed_dt_residual,
    &mesh,
    &bcs,
  );

  assert!((fixed_dt - step_dt).abs() < 1e-14);
  assert_eq!(fixed_dt_solver.current_step(), step_solver.current_step());
  assert!((fixed_dt_solver.time() - step_solver.time()).abs() < 1e-14);

  for i in 0..mesh.cell_count() {
    let step_cell = step_state.state(CellId::from(i));
    let fixed_dt_cell = fixed_dt_state.state(CellId::from(i));
    for (k, step_component) in step_cell.as_state().iter().enumerate() {
      assert!(
        (step_component - fixed_dt_cell.as_state()[k]).abs() < 1e-14,
        "cell {} component {} differs: step={} fixed_dt={}",
        i,
        k,
        step_cell.as_state()[k],
        fixed_dt_cell.as_state()[k],
      );
    }
  }
}
