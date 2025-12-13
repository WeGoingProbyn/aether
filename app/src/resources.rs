use bevy::prelude::*;
use continuum::solver::temperature::TemperatureSolver2D;

#[derive(Resource)]
pub struct Simulation {
  pub solver: TemperatureSolver2D,
  pub dt: f32,
}

#[derive(Resource)]
pub struct TemperatureTexture {
  pub handle: Handle<Image>,
}
