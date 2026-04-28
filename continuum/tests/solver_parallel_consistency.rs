// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use continuum::boundary::{BoundaryRegistry, Transmissive};
use continuum::cpu::CpuFvmRunner;
use continuum::model::{Euler2D, RusanovFlux};
use continuum::solver::{FvmSolver, SolverConfig, TimeIntegration};
use pleroma::core::storage::{CellView, FieldStorage, SoaField};
use tessera::geometry::{CellGeometry, CellId, IdentityMap};
use tessera::mesh::StructuredBlock;
use tessera::partition::decompose_structured;
use utility::domain::BoundaryTag;

use utility::thread::pool::Pool;

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
fn parallel_rk2_matches_serial_and_updates_solver_state() {
  let gamma = 1.4;
  let dims = [8, 1];
  let mesh = Arc::new(StructuredBlock::uniform(
    [0.0, 0.0].into(),
    [1.0, 0.01],
    dims,
    Box::new(IdentityMap::<2>),
  ));
  let decomp = decompose_structured(Arc::clone(&mesh), dims, 2, 1);

  let mut serial_state = SoaField::from_fn(mesh.cell_count(), |cell| {
    let x = mesh.cell_centroid(cell)[0];
    initial_state(x, gamma)
  });
  let mut serial_residual = SoaField::zeros(mesh.cell_count());

  let mut partition_states: Vec<SoaField<4>> = decomp
    .partitions
    .iter()
    .map(|partition| {
      SoaField::from_fn(partition.cell_count(), |local| {
        let global = partition.local_to_global(local);
        let x = mesh.cell_centroid(global)[0];
        initial_state(x, gamma)
      })
    })
    .collect();
  let mut partition_residuals: Vec<SoaField<4>> = decomp
    .partitions
    .iter()
    .map(|partition| SoaField::zeros(partition.cell_count()))
    .collect();

  let bcs = transmissive_bcs();
  let config = SolverConfig::new(0.5, 1e-4, TimeIntegration::Rk2);
  let mut serial_solver =
    FvmSolver::new(config.clone(), Euler2D::new(gamma), RusanovFlux);
  let mut parallel_solver =
    FvmSolver::new(config, Euler2D::new(gamma), RusanovFlux);
  let pool = Pool::new(2).unwrap();
  let cpu = CpuFvmRunner::new(&pool);

  let dt_serial = serial_solver.step(
    &mut serial_state,
    &mut serial_residual,
    mesh.as_ref(),
    &bcs,
  );

  let dt_parallel = cpu.step(
    &mut parallel_solver,
    &decomp,
    &mut partition_states,
    &mut partition_residuals,
    &bcs,
  );

  let mut gathered_parallel = vec![[0.0; 4]; mesh.cell_count()];
  for (pi, partition) in decomp.partitions.iter().enumerate() {
    for j in 0..partition.num_owned() {
      let local = CellId::from(j);
      let global = partition.local_to_global(local);
      gathered_parallel[global.index()] =
        *partition_states[pi].state(local).as_state();
    }
  }

  assert!((dt_parallel - dt_serial).abs() < 1e-14);
  assert_eq!(parallel_solver.current_step(), 1);
  assert!((parallel_solver.time() - dt_parallel).abs() < 1e-14);

  for (i, instance) in gathered_parallel.iter().enumerate() {
    let serial = serial_state.state(CellId::from(i));
    for (k, serial_state) in serial.as_state().iter().enumerate() {
      assert!(
        (serial_state - instance[k]).abs() < 1e-11,
        "cell {} component {} differs: serial={} parallel={}",
        i,
        k,
        serial.as_state()[k],
        gathered_parallel[i][k],
      );
    }
  }
}
