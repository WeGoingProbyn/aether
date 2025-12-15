use crate::geometry::{IdentityMap, MappedGeometry};
use crate::grid::grid::Grid;
use crate::solver::fv::{CflConfig, FiniteVolumeSolver, TimeIntegrator};
use crate::solver::temperature::TemperatureAdvectionDiffusion;
use crate::topology::StructuredTopology;

#[test]
fn cfl_estimate_matches_uniform_advection_limit() {
  // 1D advection embedded in 2D grid (ny = 1).
  let nx = 10usize;
  let grid = Grid::<2>::new([nx, 1]);
  let topo = StructuredTopology::<2>::new(grid.clone());
  let geom = MappedGeometry::<2, _>::new(grid.clone(), IdentityMap);

  // Constant velocity in +x, no diffusion.
  let model = TemperatureAdvectionDiffusion::<2>::new([1.0, 0.0], 0.0).with_outflow();
  let integrator = TimeIntegrator::ExplicitEuler;
  let mut solver: FiniteVolumeSolver<_, _, _, 2, 1> =
    FiniteVolumeSolver::new(topo, geom, model, integrator);

  // Non-zero state so that wave speed is exercised.
  solver.initialize_with(|_| [1.0]);

  let cfg = CflConfig {
    cfl: 0.9,
    diffusion_cfl: 1.0,
    max_dt: 10.0,
    max_substeps: 4,
  };

  let est = solver.estimate_cfl_dt(&cfg);

  // Rusanov CFL for this setup: dt <= cfl * (h_x) / a, with h_x = 1 / nx, a = |v| = 1.
  let expected = 0.9 / (nx as f64);
  assert!(
    (est.dt - expected).abs() < 1e-12,
    "expected dt={}, got {:?}",
    expected,
    est
  );
}
