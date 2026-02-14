use continuum::field::{SoaField, CellView, FieldStorage};
use continuum::geometry::{CellId, IdentityMap, CellGeometry};
use continuum::mesh::StructuredBlock;
use continuum::boundary::{BoundaryRegistry, ReflectiveWall, Transmissive};
use continuum::model::{Euler2D, RusanovFlux};
use continuum::solver::{FvmSolver, SolverConfig, TimeIntegration};
use continuum::topology::BoundaryTag;

use utility::error::AetherResult;
use utility::info;
use utility::logger::{Level, Logger, StdSink};
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
  // 100x1 grid, domain [0, 1] x [0, 0.01]
  let mesh = StructuredBlock::uniform(
    [0.0, 0.0].into(),
    [1.0, 0.01],
    [100, 1],
    Box::new(IdentityMap::<2>),
  );

  // Sod shock tube initial condition
  // Left:  rho=1.0, p=1.0, vel=0
  // Right: rho=0.125, p=0.1, vel=0
  let gamma = 1.4;
  let mut state = SoaField::<4>::from_fn(mesh.cell_count(), |cell| {
    let x = mesh.cell_centroid(cell)[0];
    if x < 0.5 {
      let rho = 1.0;
      let p = 1.0;
      [rho, 0.0, 0.0, p / (gamma - 1.0)]
    } else {
      let rho = 0.125;
      let p = 0.1;
      [rho, 0.0, 0.0, p / (gamma - 1.0)]
    }
  });

  let mut residual = SoaField::<4>::zeros(mesh.cell_count());

  // Transmissive on all boundaries
  let mut bcs = BoundaryRegistry::new();
  bcs.register(BoundaryTag::Left, Transmissive);
  bcs.register(BoundaryTag::Right, Transmissive);
  bcs.register(BoundaryTag::Bottom, ReflectiveWall);
  bcs.register(BoundaryTag::Top, ReflectiveWall);

  let config = SolverConfig::new(
    0.5,
    0.001,
    TimeIntegration::ForwardEuler,
  );

  let mut solver = FvmSolver::new(
    config,
    Euler2D::new(1.4),
    RusanovFlux,
  );

  // Run to t=0.2
  while solver.time() < 0.2 {
    let dt = solver.step(&mut state, &mut residual, &mesh, &bcs);
    info!("step={}, t={:.6}, dt={:.6}", solver.current_step(), solver.time(), dt);
  }

  // Print density profile
  for i in 0..100 {
    let cell = CellId::from(i);
    let s = state.state(cell);
    info!("{:.4} {:.4}", mesh.cell_centroid(cell)[0], s.as_state()[0]);
  }
  pool.flush_profiler();
  //Profiler::print(&mut LogWriter::new(Level::Debug));

  Ok(())
}







