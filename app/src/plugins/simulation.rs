use bevy::prelude::*;

use crate::resources::Simulation;

use continuum::grid::grid::Grid;
use continuum::topology::StructuredTopology;
use continuum::geometry::{MappedGeometry, IdentityMap};
use continuum::solver::explicit::FiniteVolumeSolver;
use continuum::solver::temperature::TemperatureAdvectionDiffusion;

pub struct SimulationPlugin;

impl Plugin for SimulationPlugin {
  fn build(&self, app: &mut App) {
    let nx: usize = 64;
    let ny: usize = 64;

    // 1) Computational grid on [0,1]^2
    let grid = Grid::<2>::new([nx, ny]);

    // 2) Topology (neighbors/boundaries). Optional: set periodic if you want.
    let topo = StructuredTopology::<2>::new(grid.clone());
    // let topo = StructuredTopology::<2>::new(grid.clone())
    //   .with_periodic(0, true)
    //   .with_periodic(1, true);

    // 3) Geometry (physical measures). Here: identity Cartesian mapping.
    let geom = MappedGeometry::<2, _>::new(grid.clone(), IdentityMap);

    // 4) Model: advection + diffusion of temperature (scalar, NV=1)
    let model = TemperatureAdvectionDiffusion::<2>::new([0.0, 0.0], 0.005)
      //.with_outflow(); 
      .with_dirichlet(0.0);

    // 5) Solver (D=2, NV=1)
    let mut solver: FiniteVolumeSolver<_, _, _, 2, 1> =
    FiniteVolumeSolver::new(topo, geom, model);

    // Initial condition: hot blob in center, expressed in *physical* coords x in [0,1]^2
    let cx = 0.5_f64;
    let cy = 0.5_f64;
    let r2 = (0.12_f64).powi(2);

    solver.initialize_with(|x| {
      let dx = x[0] - cx;
      let dy = x[1] - cy;
      let t = if dx * dx + dy * dy < r2 { 1.0 } else { 0.0 };
      [t]
    });

    app.insert_resource(Simulation { solver, dt: 0.01 })
      .add_systems(FixedUpdate, step_simulation);
  }
}

fn step_simulation(mut sim: ResMut<Simulation>) {
  let dt = sim.dt;
  sim.solver.step_explicit(dt);
}

