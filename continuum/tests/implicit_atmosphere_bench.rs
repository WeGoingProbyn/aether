// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Wall-clock benchmark: explicit vs hybrid implicit on a real cube-sphere
//! atmosphere shell. Answers "does the implicit solver actually beat the
//! explicit sub-stepping?" before committing the live atmosphere to it.
//!
//! Diagnostic, opt-in:
//! `cargo test -p continuum --test implicit_atmosphere_bench --release \
//!   -- --ignored --nocapture`

use std::time::Instant;

use continuum::boundary::{BoundaryRegistry, ReflectiveWall};
use continuum::implicit::backend::ImplicitBackend;
use continuum::implicit::hybrid::HybridBackend;
use continuum::implicit::schemes::RosenbrockScheme;
use continuum::model::{MoistEuler3D, RusanovFlux};
use continuum::solver::{FvmSolver, SolverConfig, TimeIntegration};
use pleroma::core::storage::{CellView, FieldStorage, SoaField};
use tessera::cube_sphere::CubeSphere;
use tessera::geometry::CellGeometry;
use utility::domain::{BoundaryTag, CellId};

const GAMMA: f64 = 1.4;

fn atmosphere_bcs() -> BoundaryRegistry<3, 6> {
  let mut bcs = BoundaryRegistry::new();
  bcs.register(BoundaryTag::Ground, ReflectiveWall);
  bcs.register(BoundaryTag::AtmosphereEdge, ReflectiveWall);
  bcs
}

fn initial_state(mesh: &CubeSphere, cells: usize) -> SoaField<6> {
  SoaField::<6>::from_fn(cells, |cell| {
    let c = mesh.cell_world_centroid(cell);
    // A gentle acoustic perturbation so the shell rings (excites the stiff
    // acoustic modes the implicit solver is meant to leap over).
    let bump = 0.1 * (c[0] * 6.0).sin() * (c[1] * 6.0).cos();
    let rho = 1.0 + bump;
    let p = 1.0 + bump;
    [rho, 0.0, 0.0, 0.0, p / (GAMMA - 1.0), 0.01 * rho]
  })
}

fn finite(state: &SoaField<6>, cells: usize) -> bool {
  (0..cells).all(|i| {
    let s = state.state(CellId::from(i));
    let s = s.as_state();
    s.iter().all(|v| v.is_finite()) && s[0] > 0.0
  })
}

#[test]
#[ignore = "wall-clock benchmark; run with --release --ignored --nocapture"]
fn bench_explicit_vs_hybrid_atmosphere() {
  let mesh = CubeSphere::new([12, 12, 6], 1.0, 1.08);
  let cells = mesh.cell_count();
  let bcs = atmosphere_bcs();
  let gravity = mesh.radial_gravity_field(1.0);
  let law = || MoistEuler3D::with_per_cell_gravity(GAMMA, gravity.clone());
  let config = SolverConfig::new(0.4, 1.0, TimeIntegration::ForwardEuler);

  let cfl_dt = FvmSolver::new(config.clone(), law(), RusanovFlux)
    .compute_dt(&initial_state(&mesh, cells), &mesh);
  // Advance a fixed slice of simulated time.
  let sim_time = 150.0 * cfl_dt;
  eprintln!(
    "cube-sphere atmosphere: {cells} cells, explicit CFL dt = {cfl_dt:.3e}, \
     advancing sim_time = {sim_time:.3e}"
  );

  // --- Explicit reference. ---
  {
    let mut solver = FvmSolver::new(config.clone(), law(), RusanovFlux);
    let mut state = initial_state(&mesh, cells);
    let mut residual = SoaField::<6>::zeros(cells);
    let t0 = Instant::now();
    let mut elapsed = 0.0;
    let mut steps = 0;
    while elapsed < sim_time {
      let dt = solver.step(&mut state, &mut residual, &mesh, &bcs);
      elapsed += dt;
      steps += 1;
    }
    let wall = t0.elapsed().as_secs_f64() * 1e3;
    assert!(finite(&state, cells), "explicit diverged");
    eprintln!("explicit:        {steps:5} steps, {wall:8.1} ms");
  }

  // --- Hybrid (full-implicit) at several step-size multiples. ---
  for &mult in &[5.0_f64, 10.0, 20.0, 40.0] {
    let target_dt = mult * cfl_dt;
    let implicit =
      ImplicitBackend::new(RosenbrockScheme::LinearlyImplicitEuler.tableau())
        .with_auto_diff()
        .with_block_jacobi();
    let mut solver = FvmSolver::with_backend(
      config.clone(),
      law(),
      RusanovFlux,
      HybridBackend::<6>::new(implicit),
    );
    let mut state = initial_state(&mesh, cells);
    let mut residual = SoaField::<6>::zeros(cells);
    let t0 = Instant::now();
    let mut elapsed = 0.0;
    let mut steps = 0;
    while elapsed < sim_time {
      let dt = (sim_time - elapsed).min(target_dt);
      solver.step_with_dt(dt, &mut state, &mut residual, &mesh, &bcs);
      elapsed += dt;
      steps += 1;
    }
    let wall = t0.elapsed().as_secs_f64() * 1e3;
    let r = solver.backend().report();
    assert!(finite(&state, cells), "hybrid diverged at {mult}x");
    eprintln!(
      "hybrid(full) {mult:4.0}x: {steps:5} steps, {wall:8.1} ms  \
       (implicit_ok={}, fallbacks={})",
      r.implicit_accepted, r.fallbacks
    );
  }

  // --- IMEX (acoustic implicit, advection explicit) at the same multiples. ---
  for &mult in &[5.0_f64, 10.0, 20.0, 40.0] {
    let target_dt = mult * cfl_dt;
    let mut solver = FvmSolver::with_backend(
      config.clone(),
      law(),
      RusanovFlux,
      ImplicitBackend::<6>::new(
        RosenbrockScheme::LinearlyImplicitEuler.tableau(),
      )
      .with_imex()
      .with_auto_diff()
      .with_block_jacobi(),
    );
    let mut state = initial_state(&mesh, cells);
    let mut residual = SoaField::<6>::zeros(cells);
    let t0 = Instant::now();
    let mut elapsed = 0.0;
    let mut steps = 0;
    let mut matvecs = 0;
    let mut converged = true;
    while elapsed < sim_time {
      let dt = (sim_time - elapsed).min(target_dt);
      solver.step_with_dt(dt, &mut state, &mut residual, &mesh, &bcs);
      let r = solver.backend().last_report().unwrap();
      matvecs += r.gmres_iters;
      converged &= r.converged;
      elapsed += dt;
      steps += 1;
    }
    let wall = t0.elapsed().as_secs_f64() * 1e3;
    assert!(finite(&state, cells), "IMEX diverged at {mult}x");
    eprintln!(
      "imex         {mult:4.0}x: {steps:5} steps, {wall:8.1} ms  \
       (matvecs={matvecs}, converged={converged})"
    );
  }
}
