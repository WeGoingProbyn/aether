use bevy::prelude::*;
use continuum::geometry::{MappedGeometry, IdentityMap};
use continuum::grid::grid::Grid;
use continuum::solver::fv::FiniteVolumeSolver;
use continuum::solver::fv::CflConfig;
use continuum::solver::temperature::TemperatureAdvectionDiffusion;
use continuum::topology::StructuredTopology;

#[derive(Resource)]
pub struct Simulation {
  pub solver: FiniteVolumeSolver<
    StructuredTopology<2>,
    MappedGeometry<2, IdentityMap>,
    TemperatureAdvectionDiffusion<2>,
    2,
    1
  >,
  pub dt: f64,
  pub cfl: CflConfig,
}

#[derive(Resource)]
pub struct TemperatureTexture {
  pub handle: Handle<Image>,
}
