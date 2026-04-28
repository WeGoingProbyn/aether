// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use continuum::boundary::{BoundaryRegistry, ReflectiveWall, Transmissive};
use continuum::cpu::CpuFvmRunner;
use continuum::model::{Euler2D, RusanovFlux};
use continuum::output::write_partitioned_vtu;
use continuum::solver::{FvmSolver, SolverConfig, TimeIntegration};
use pleroma::core::storage::SoaField;
use tessera::geometry::{CellGeometry, IdentityMap};
use tessera::mesh::StructuredBlock;
use tessera::partition::decompose_structured;
use utility::domain::BoundaryTag;

use utility::error::AetherResult;
use utility::info;
use utility::logger::{Level, LogWriter, Logger, StdSink};
use utility::profiler::Profiler;
use utility::thread::pool::Pool;

fn main() -> AetherResult<()> {
  Logger::init(
    vec![Box::new(StdSink::new(std::io::stdout()).capacity(1))],
    Level::Trace,
  );

  Profiler::init();

  let pool = Pool::default();
  let cpu = CpuFvmRunner::new(&pool);

  let dims = [1000, 1];
  let mesh = Arc::new(StructuredBlock::uniform(
    [0.0, 0.0].into(),
    [1.0, 0.01],
    dims,
    Box::new(IdentityMap::<2>),
  ));

  let gamma = 1.4;
  let num_partitions = 2;
  let decomp = decompose_structured(Arc::clone(&mesh), dims, num_partitions, 1);

  // Create per-partition fields
  let mut states: Vec<SoaField<4>> = decomp
    .partitions
    .iter()
    .map(|p| {
      SoaField::from_fn(p.cell_count(), |cell| {
        let global = p.local_to_global(cell);
        let x = mesh.cell_centroid(global)[0];
        if x < 0.5 {
          let rho = 1.0;
          let p = 1.0;
          [rho, 0.0, 0.0, p / (gamma - 1.0)]
        } else {
          let rho = 0.125;
          let p = 0.1;
          [rho, 0.0, 0.0, p / (gamma - 1.0)]
        }
      })
    })
    .collect();

  let mut residuals = decomp
    .partitions
    .iter()
    .map(|p| SoaField::zeros(p.cell_count()))
    .collect::<Vec<SoaField<4>>>();

  let mut bcs = BoundaryRegistry::new();
  bcs.register(BoundaryTag::Left, Transmissive);
  bcs.register(BoundaryTag::Right, Transmissive);
  bcs.register(BoundaryTag::Bottom, ReflectiveWall);
  bcs.register(BoundaryTag::Top, ReflectiveWall);

  let config = SolverConfig::new(0.5, 1e-4, TimeIntegration::ForwardEuler);
  let mut solver = FvmSolver::new(config, Euler2D::new(1.4), RusanovFlux);

  let vtk_output_dir = std::env::var("AETHER_SOD_VTK_DIR")
    .unwrap_or_else(|_| "output/sod_shock".to_string());
  let vtk_write_every = std::env::var("AETHER_SOD_VTK_EVERY")
    .ok()
    .and_then(|value| value.parse::<usize>().ok())
    .filter(|&value| value > 0)
    .unwrap_or(100);

  let mut time = 0.0;
  let mut step = 0;

  let initial_base = format!("step_{step:06}");
  let initial_manifest = write_partitioned_vtu(
    &decomp,
    &states,
    solver.law(),
    &vtk_output_dir,
    &initial_base,
  )?;
  info!("wrote vtk snapshot: {}", initial_manifest.to_string_lossy());

  while time < 0.2 {
    let dt = cpu.step(&mut solver, &decomp, &mut states, &mut residuals, &bcs);
    time += dt;
    step += 1;
    info!("step={}, t={:.6}, dt={:.6}", step, time, dt);

    if step % vtk_write_every == 0 || time >= 0.2 {
      let base_name = format!("step_{step:06}");
      let manifest = write_partitioned_vtu(
        &decomp,
        &states,
        solver.law(),
        &vtk_output_dir,
        &base_name,
      )?;
      info!("wrote vtk snapshot: {}", manifest.to_string_lossy());
    }
  }

  // Print density profile — gather from partitions
  // for (i, partition) in decomp.partitions.iter().enumerate() {
  //   for j in 0..partition.num_owned() {
  //     let cell = CellId::from(j);
  //     let global = partition.local_to_global(cell);
  //     let s = states[i].state(cell);
  //     info!(
  //       "{:.4} {:.4}",
  //       mesh.cell_centroid(global)[0],
  //       s.as_state()[0]
  //     );
  //   }
  // }

  pool.flush_profiler();
  Profiler::print(&mut LogWriter::new(Level::Info));
  Ok(())
}
