use bevy::prelude::*;
use crate::resources::Simulation;

use continuum::field::scalar::ScalarField;
use continuum::coords::cartesian::CartesianGrid2D;
use continuum::solver::temperature::TemperatureSolver2D;

pub struct SimulationPlugin;

impl Plugin for SimulationPlugin {
  fn build(&self, app: &mut App) {
    let nx = 128;
    let ny = 128;

    let grid = CartesianGrid2D { 
      nx, 
      ny, 
      dx: 1.0 / nx as f32, 
      dy: 1.0 / ny as f32, 
      total_cells: nx * ny 
    };
    let mut temperature = ScalarField::new(nx * ny);

    // hot blob in the center
    let cx = nx as f32 * 0.5;
    let cy = ny as f32 * 0.5;
    let r2 = (nx.min(ny) as f32 * 0.12).powi(2);

    for j in 0..ny {
      for i in 0..nx {
        let dx = i as f32 - cx;
        let dy = j as f32 - cy;
        let val = if dx * dx + dy * dy < r2 { 1.0 } else { 0.0 };
        temperature.field[j * nx + i] = val;
      }
    }

    let solver = TemperatureSolver2D {
      grid,
      temperature,
      velocity: [0.0, 0.0],
      diffusivity: 0.001,
    };

    let (mut tmin, mut tmax) = (f32::INFINITY, f32::NEG_INFINITY);
    for &t in &solver.temperature.field {
      tmin = tmin.min(t);
      tmax = tmax.max(t);
    }
    println!("init temperature: min={tmin}, max={tmax}");

    app.insert_resource(Simulation { solver, dt: 0.01 })
      .add_systems(FixedUpdate, step_simulation);
  }
}

fn step_simulation(mut sim: ResMut<Simulation>) {
  let dt = sim.dt.clone();
  sim.solver.step(dt);
}
