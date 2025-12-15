use crate::geometry::{IdentityMap, MappedGeometry};
use crate::grid::grid::Grid;
use crate::solver::fv::{FiniteVolumeSolver, TimeIntegrator};
use crate::solver::gmres::GmresConfig;
use crate::solver::implicit::{ImplicitConfig, NewtonConfig};
use crate::topology::StructuredTopology;
use crate::solver::fv::{Flux, Model, State};
use crate::topology::BoundaryId;
use crate::geometry::VecN;

/// A model with zero flux and time-dependent source f(t) = t.
#[derive(Debug, Clone)]
struct TimeSourceModel;

impl Model<2, 1> for TimeSourceModel {
  fn flux(&self, _u: &State<1>) -> Flux<2, 1> { [[0.0; 2]; 1] }

  fn max_wave_speed(&self, _u: &State<1>, _n_unit: VecN<2>) -> f64 { 0.0 }

  fn boundary_state(
    &self,
    _boundary: BoundaryId,
    interior: &State<1>,
    _x_face: VecN<2>,
    _t: f64,
  ) -> State<1> { *interior }

  fn source(&self, _u: &State<1>, _x_cell: VecN<2>, t: f64) -> State<1> { [t] }
}

#[test]
fn implicit_uses_future_time_in_source() {
  let grid = Grid::<2>::new([1, 1]);
  let topo = StructuredTopology::<2>::new(grid.clone());
  let geom = MappedGeometry::<2, _>::new(grid.clone(), IdentityMap);

  let implicit_cfg = ImplicitConfig {
    newton: NewtonConfig { max_iters: 6, tol: 1e-10, fd_epsilon: 1e-6 },
    gmres: GmresConfig { restart: 4, max_iters: 20, tol: 1e-10 },
  };

  let integrator = TimeIntegrator::ImplicitBackwardEuler(implicit_cfg);
  let mut solver: FiniteVolumeSolver<_, _, _, 2, 1> =
    FiniteVolumeSolver::new(topo, geom, TimeSourceModel, integrator);

  solver.initialize_with(|_| [0.0]);

  // With source = t evaluated at t_{n+1}, a single BE step of dt=1 should give u = 1.
  solver.step(1.0);
  let u = solver.u[0][0];
  assert!(
    (u - 1.0).abs() < 1e-8,
    "implicit step should use t_{{n+1}}; got u={}",
    u
  );
}
