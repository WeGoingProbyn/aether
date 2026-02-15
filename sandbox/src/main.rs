use std::sync::Arc;

use continuum::field::{SoaField, CellView, FieldStorage};
use continuum::geometry::{CellId, IdentityMap, CellGeometry};
use continuum::mesh::StructuredBlock;
use continuum::boundary::{BoundaryRegistry, ReflectiveWall, Transmissive};
use continuum::model::{Euler2D, RusanovFlux};
use continuum::solver::{FvmSolver, SolverConfig, TimeIntegration};
use continuum::partition::decompose_structured;
use continuum::topology::BoundaryTag;

use utility::error::AetherResult;
use utility::info;
use utility::logger::{Level, LogWriter, Logger, StdSink};
use utility::thread::pool::Pool;
use utility::profiler::Profiler;

fn main() -> AetherResult<()> {
  Logger::init(
    vec![
      Box::new(StdSink::new(std::io::stdout()).capacity(1)),
    ],
    Level::Trace
  );

  Profiler::init();

  let pool = Pool::default();

  let dims = [1000, 1];
  let mesh = Arc::new(StructuredBlock::uniform(
    [0.0, 0.0].into(),
    [1.0, 0.01],
    dims,
    Box::new(IdentityMap::<2>),
  ));

  let gamma = 1.4;
  let num_partitions = 12;
  let decomp = decompose_structured(Arc::clone(&mesh), dims, num_partitions, 1);

  // Create per-partition fields
  let mut states: Vec<SoaField<4>> = decomp.partitions.iter()
    .map(|p| SoaField::from_fn(p.cell_count(), |cell| {
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
    })).collect();

  let mut residuals = decomp.partitions.iter()
    .map(|p| 
      SoaField::zeros(p.cell_count()
      )).collect::<Vec<SoaField<4>>>();

  let mut bcs = BoundaryRegistry::new();
  bcs.register(BoundaryTag::Left, Transmissive);
  bcs.register(BoundaryTag::Right, Transmissive);
  bcs.register(BoundaryTag::Bottom, ReflectiveWall);
  bcs.register(BoundaryTag::Top, ReflectiveWall);

  let config = SolverConfig::new(0.5, 1e-8, TimeIntegration::ForwardEuler);
  let solver = FvmSolver::new(config, Euler2D::new(1.4), RusanovFlux);

  let mut time = 0.0;
  let mut step = 0;
  while time < 0.2 {
    let dt = solver.parallel_step(&pool, &decomp, &mut states, &mut residuals, &bcs);
    time += dt;
    step += 1;
    info!("step={}, t={:.6}, dt={:.6}", step, time, dt);
  }

  // Print density profile — gather from partitions
  for (i, partition) in decomp.partitions.iter().enumerate() {
    for j in 0..partition.num_owned() {
      let cell = CellId::from(j);
      let global = partition.local_to_global(cell);
      let s = states[i].state(cell);
      info!("{:.4} {:.4}", mesh.cell_centroid(global)[0], s.as_state()[0]);
    }
  }

  pool.flush_profiler();
  Profiler::print(&mut LogWriter::new(Level::Info));
  Ok(())
}





