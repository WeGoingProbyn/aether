use bevy::prelude::*;
use continuum::solver::gmres::GmresConfig;
use continuum::solver::implicit::{ImplicitConfig, NewtonConfig};
use bevy::log::info;

use crate::resources::Simulation;

use continuum::grid::grid::Grid;
use continuum::topology::StructuredTopology;
use continuum::geometry::{MappedGeometry, IdentityMap};
use continuum::solver::fv::{CflConfig, FiniteVolumeSolver, HybridConfig, HybridState, TimeIntegrator};
use continuum::solver::temperature::TemperatureAdvectionDiffusion;

pub struct SimulationPlugin;

impl Plugin for SimulationPlugin {
  fn build(&self, app: &mut App) {
    let nx: usize = 128;
    let ny: usize = 128;

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
      // .with_outflow(); 
      .with_dirichlet(0.0);

    let implicit_cfg = ImplicitConfig {
      newton: NewtonConfig { max_iters: 8, tol: 1e-8, fd_epsilon: 1e-6 },
      gmres: GmresConfig { restart: 30, max_iters: 200, tol: 1e-8 },
    };

    let integrator = TimeIntegrator::Hybrid(
      implicit_cfg,
      HybridConfig::default(),
      HybridState::default(),
    );

    // 5) Solver (D=2, NV=1)
    let mut solver: FiniteVolumeSolver<_, _, _, 2, 1> =
    FiniteVolumeSolver::new(topo, geom, model, integrator);

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

    app.insert_resource(Simulation {
      solver,
      dt: 1.0 / 60.0,
      cfl: CflConfig {
        cfl: 0.45,
        diffusion_cfl: 0.3,
        max_dt: 1.0 / 60.0,
        max_substeps: 32,
      },
    })
      .add_systems(FixedUpdate, step_simulation);
  }
}

fn step_simulation(mut sim: ResMut<Simulation>) {
  let target_dt = sim.dt;
  let cfl_cfg = sim.cfl.clone();
  let integrator_before = match &sim.solver.integrator {
    TimeIntegrator::ExplicitEuler => "explicit",
    TimeIntegrator::ImplicitBackwardEuler(_) => "implicit",
    TimeIntegrator::Hybrid(_, _, st) => match st.phase {
      continuum::solver::fv::HybridPhase::Explicit => "hybrid-explicit",
      continuum::solver::fv::HybridPhase::Implicit => "hybrid-implicit",
    },
  };

  let stats = sim.solver.step_cfl(target_dt, &cfl_cfg);
  let integrator_after = match &sim.solver.integrator {
    TimeIntegrator::ExplicitEuler => "explicit",
    TimeIntegrator::ImplicitBackwardEuler(_) => "implicit",
    TimeIntegrator::Hybrid(_, _, st) => match st.phase {
      continuum::solver::fv::HybridPhase::Explicit => "hybrid-explicit",
      continuum::solver::fv::HybridPhase::Implicit => "hybrid-implicit",
    },
  };

  info!(
    "Frame dt={:.3e}s, integrator {} -> {}, time={:.3e}, advanced={:.3e}s in {} substeps",
    target_dt,
    integrator_before,
    integrator_after,
    sim.solver.time,
    stats.advanced_dt,
    stats.substeps,
  );
}
