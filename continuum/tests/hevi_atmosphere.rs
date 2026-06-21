// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! HEVI on a thin cube-sphere atmosphere shell: with the vertical acoustic
//! terms integrated implicitly per radial column, the scheme stays stable far
//! beyond the vertical acoustic CFL — where fully-explicit stepping diverges —
//! which is exactly the constraint that pins the demo (vertical-bound in 100%
//! of cells, ~83x headroom; see `cfl_anatomy`).

use std::time::Instant;

use continuum::boundary::{BoundaryRegistry, ReflectiveWall};
use continuum::implicit::hevi::{HeviBackend, radial_columns_from_geometry};
use continuum::model::{MoistEuler3D, RusanovFlux};
use continuum::solver::{FvmSolver, SolverConfig, TimeIntegration};
use pleroma::core::storage::{CellView, FieldStorage, SoaField};
use tessera::cube_sphere::CubeSphere;
use tessera::geometry::CellGeometry;
use utility::domain::{BoundaryTag, CellId};

const GAMMA: f64 = 1.4;

fn shell(na: usize, nr: usize) -> CubeSphere {
  // 20 km atmosphere over Earth's radius — the demo's thin-shell geometry.
  CubeSphere::new([na, na, nr], 6.371e6, 6.371e6 + 20_000.0)
}

fn atmosphere_bcs() -> BoundaryRegistry<3, 6> {
  let mut bcs = BoundaryRegistry::new();
  bcs.register(BoundaryTag::Ground, ReflectiveWall);
  bcs.register(BoundaryTag::AtmosphereEdge, ReflectiveWall);
  bcs
}

fn make_state(mesh: &CubeSphere) -> SoaField<6> {
  SoaField::from_fn(mesh.cell_count(), |cell| {
    let c = mesh.cell_world_centroid(cell);
    let r = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
    // A vertical acoustic perturbation (varies with altitude) at rest.
    let frac = (r - 6.371e6) / 20_000.0;
    let rho = 1.2 * (1.0 + 0.1 * (frac * 6.0).sin());
    let p = 1.0e5 * (1.0 + 0.1 * (frac * 6.0).sin());
    [rho, 0.0, 0.0, 0.0, p / (GAMMA - 1.0), 0.01 * rho]
  })
}

fn columns(mesh: &CubeSphere) -> Vec<continuum::implicit::hevi::RadialColumn> {
  radial_columns_from_geometry(
    mesh,
    |c| {
      let p = mesh.cell_world_centroid(c);
      [p[0], p[1], p[2]]
    },
    |_| true,
  )
}

fn finite_positive(state: &SoaField<6>, cells: usize) -> bool {
  (0..cells).all(|i| {
    let s = state.state(CellId::from(i));
    let s = s.as_state();
    s.iter().all(|v| v.is_finite()) && s[0] > 0.0
  })
}

fn total_mass(state: &SoaField<6>, cells: usize) -> f64 {
  (0..cells)
    .map(|i| state.state(CellId::from(i)).as_state()[0])
    .sum()
}

#[test]
fn hevi_columns_cover_partition_owned_cells() {
  use std::sync::Arc;
  use tessera::partition::decompose_cube_sphere_panels;

  let mesh = Arc::new(shell(8, 6));
  let decomposition = decompose_cube_sphere_panels(mesh);
  assert_eq!(decomposition.partitions.len(), 6, "one partition per panel");

  for (i, partition) in decomposition.partitions.iter().enumerate() {
    let num_owned = partition.num_owned();
    let columns = radial_columns_from_geometry(
      partition,
      |c| {
        let p = partition.cell_world_centroid(c);
        [p[0], p[1], p[2]]
      },
      |c| c.index() < num_owned,
    );
    let covered: usize = columns.iter().map(|c| c.cells.len()).sum();
    eprintln!(
      "partition {i}: {num_owned} owned, {} columns covering {covered} cells",
      columns.len()
    );
    assert!(!columns.is_empty(), "partition {i} produced no columns");
    assert_eq!(
      covered, num_owned,
      "partition {i}: columns must cover exactly the owned cells"
    );
    // No column may include a ghost cell.
    for col in &columns {
      for &c in &col.cells {
        assert!(c.index() < num_owned, "column contains a ghost cell");
      }
    }
  }
}

#[test]
fn hevi_stable_beyond_vertical_cfl_where_explicit_diverges() {
  let mesh = shell(8, 8);
  let cells = mesh.cell_count();
  let bcs = atmosphere_bcs();
  let config = SolverConfig::new(0.4, 1.0, TimeIntegration::ForwardEuler);

  let vertical_cfl =
    FvmSolver::new(config.clone(), MoistEuler3D::new(GAMMA), RusanovFlux)
      .compute_dt(&make_state(&mesh), &mesh);
  let big_dt = 15.0 * vertical_cfl;
  let steps = 12;

  // Fully explicit at big_dt: vertical acoustic instability ⇒ diverges.
  let mut explicit =
    FvmSolver::new(config.clone(), MoistEuler3D::new(GAMMA), RusanovFlux);
  let mut e_state = make_state(&mesh);
  let mut e_res = SoaField::zeros(cells);
  for _ in 0..steps {
    explicit.step_with_dt(big_dt, &mut e_state, &mut e_res, &mesh, &bcs);
  }
  assert!(
    !finite_positive(&e_state, cells),
    "explicit unexpectedly survived {big_dt} (15x vertical CFL)"
  );

  // HEVI (vertical implicit) at the same big_dt: stable.
  let mut hevi = FvmSolver::with_backend(
    config,
    MoistEuler3D::new(GAMMA),
    RusanovFlux,
    HeviBackend::<6>::new(columns(&mesh)),
  );
  let mut state = make_state(&mesh);
  let mut residual = SoaField::zeros(cells);
  let mass0 = total_mass(&state, cells);
  for _ in 0..steps {
    hevi.step_with_dt(big_dt, &mut state, &mut residual, &mesh, &bcs);
    assert!(finite_positive(&state, cells), "HEVI diverged");
    assert_eq!(hevi.backend().fallback_columns(), 0, "no singular columns");
  }
  let mass1 = total_mass(&state, cells);
  assert!(
    (mass1 - mass0).abs() / mass0 < 1e-3,
    "mass drift {mass0} -> {mass1}"
  );
}

#[test]
#[ignore = "wall-clock benchmark; run with --release --ignored --nocapture"]
fn bench_hevi_vs_explicit() {
  let mesh = shell(48, 30);
  let cells = mesh.cell_count();
  let bcs = atmosphere_bcs();
  let config = SolverConfig::new(0.25, 1.0, TimeIntegration::ForwardEuler);
  let vertical_cfl =
    FvmSolver::new(config.clone(), MoistEuler3D::new(GAMMA), RusanovFlux)
      .compute_dt(&make_state(&mesh), &mesh);
  let sim_time = 200.0 * vertical_cfl;
  eprintln!(
    "{cells} cells, vertical CFL dt = {vertical_cfl:.4} s, \
     advancing {sim_time:.1} s"
  );

  // Explicit reference.
  let mut explicit =
    FvmSolver::new(config.clone(), MoistEuler3D::new(GAMMA), RusanovFlux);
  let mut s = make_state(&mesh);
  let mut r = SoaField::zeros(cells);
  let t0 = Instant::now();
  let mut elapsed = 0.0;
  let mut steps = 0;
  while elapsed < sim_time {
    let dt = explicit.step(&mut s, &mut r, &mesh, &bcs);
    elapsed += dt;
    steps += 1;
  }
  let explicit_ms = t0.elapsed().as_secs_f64() * 1e3;
  eprintln!("explicit: {steps:5} steps, {explicit_ms:8.1} ms");

  // HEVI at a large multiple of the vertical CFL.
  for &mult in &[20.0_f64, 40.0, 80.0, 160.0, 320.0] {
    let target = mult * vertical_cfl;
    let mut hevi = FvmSolver::with_backend(
      config.clone(),
      MoistEuler3D::new(GAMMA),
      RusanovFlux,
      HeviBackend::<6>::new(columns(&mesh)),
    );
    let mut s = make_state(&mesh);
    let mut r = SoaField::zeros(cells);
    let t0 = Instant::now();
    let mut elapsed = 0.0;
    let mut steps = 0;
    let mut fallbacks = 0;
    while elapsed < sim_time {
      let dt = (sim_time - elapsed).min(target);
      hevi.step_with_dt(dt, &mut s, &mut r, &mesh, &bcs);
      fallbacks += hevi.backend().fallback_columns();
      elapsed += dt;
      steps += 1;
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    let ok = finite_positive(&s, cells);
    eprintln!(
      "hevi {mult:3.0}x: {steps:5} steps, {ms:8.1} ms  \
       ({:.1}x vs explicit, fallbacks={fallbacks}, finite={ok})",
      explicit_ms / ms
    );
  }
}
