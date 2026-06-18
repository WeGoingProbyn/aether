// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Full-stack integration: cosmo seed → WorldFactory → atmosphere
//! (aer) + surface (terra) + radiation (lumen) running together.
//!
//! Verifies that:
//! - The DAG schedules cleanly with all three crates contributing.
//! - After several ticks every conserved scalar (mass, energy, surface
//!   temperature) stays finite and physically plausible.
//! - The day-side surface receives net heating and warms relative to
//!   the night-side under a fixed sun direction.

use std::collections::HashMap;

use aer::{AtmosphereModel, AtmosphereShellLayout};
use aether::{
  core::{Aether, System},
  factory::WorldFactory,
};
use cosmo::factory as cosmo_factory;
use lumen::{RadiationCoefficients, RadiationModel};
use nexus::{FieldStorage, MeshKey, SoaField, WorldId};
use terra::SurfaceThermalModel;
use tessera::geometry::CellGeometry;
use utility::{
  domain::{CellId, SystemId},
  error::AetherResult,
  thread::pool::Pool,
};

#[test]
fn full_stack_atm_surface_radiation_runs_without_blowup() -> AetherResult<()> {
  let world_id = WorldId(0);
  let angular_dims = [2, 2];
  let surface_radial_layers = 1;
  let atmosphere_radial_layers = 3;
  let atmosphere_height = 20_000.0;
  let surface_depth = 10_000.0;

  let mut factory = WorldFactory::new(world_id, cosmo_factory::earth())
    .with_primary(cosmo_factory::sun());
  let constants = factory.constants();
  let shell_layout =
    AtmosphereShellLayout::new(&constants, atmosphere_height, surface_depth)?;

  factory = factory.cube_sphere_surface(
    shell_layout.surface_shell_spec(angular_dims, surface_radial_layers),
  );
  factory = factory.cube_sphere_atmosphere(
    shell_layout.atmosphere_shell_spec(angular_dims, atmosphere_radial_layers),
  );

  let surface_mesh = factory
    .tessera()
    .mesh(MeshKey::SURFACE)
    .expect("surface mesh registered")
    .clone();
  let atmosphere_mesh = factory
    .tessera()
    .mesh(MeshKey::ATMOSPHERE)
    .expect("atmosphere mesh registered")
    .clone();

  let reference_temperature = constants
    .atmosphere
    .expect("earth has an atmosphere")
    .reference_temperature;

  let surface_model = SurfaceThermalModel::new(MeshKey::SURFACE)
    .with_initial_temperature(reference_temperature)
    // Tiny effective heat capacity so a few ticks at dt=1s already
    // produce a measurable day/night temperature contrast.
    .with_heat_capacity_per_area(1.0e3);
  let surface_fields = surface_model.fields();

  let atmosphere_model = AtmosphereModel::new(MeshKey::ATMOSPHERE)
    .with_cfl(0.25)
    .with_current_state_background_correction()
    .with_radiative_heating();
  let atmosphere_fields = atmosphere_model.fields();

  let radiation_model = RadiationModel::from_world_constants(
    MeshKey::ATMOSPHERE,
    MeshKey::SURFACE,
    &constants,
    RadiationCoefficients::default(),
  )?;

  surface_model
    .register_fields(factory.pleroma_mut(), surface_mesh.as_ref())?;
  atmosphere_model.register_fields(
    factory.pleroma_mut(),
    atmosphere_mesh.as_ref(),
    &constants,
    shell_layout.reference_radius(),
  )?;
  radiation_model.register_fields(
    factory.pleroma_mut(),
    atmosphere_mesh.as_ref(),
    surface_mesh.as_ref(),
  )?;
  // Sun on +x: cells with centroid x > 0 are illuminated.
  radiation_model
    .register_default_sun_position(factory.pleroma_mut(), [1.0, 0.0, 0.0]);

  surface_model.add_stages(factory.nexus_mut())?;
  radiation_model.add_stages(factory.nexus_mut())?;
  atmosphere_model.add_stages(factory.nexus_mut())?;

  let world = factory.build()?;
  let system_id = SystemId(0);
  let mut systems = HashMap::new();
  systems.insert(system_id, System::single(system_id, world));

  let mut aether = Aether::new(systems, Pool::default());
  for _ in 0..10 {
    aether.step(1.0)?;
  }

  let world = aether
    .world(world_id)
    .expect("world should remain registered");

  // 1. Every conserved scalar stays finite and positive.
  let surface_temperature: &SoaField<1> = world
    .pleroma()
    .read(surface_fields.temperature)
    .expect("surface temperature");
  let atmosphere_state: &SoaField<6> = world
    .pleroma()
    .read(atmosphere_fields.euler_state)
    .expect("atmosphere euler state");

  for i in 0..surface_temperature.len() {
    let t = surface_temperature.state(CellId::from(i))[0];
    assert!(t.is_finite() && t > 0.0, "surface cell {i} has T={t}");
  }
  for i in 0..atmosphere_state.len() {
    let s = atmosphere_state.state(CellId::from(i));
    assert!(
      s.iter().all(|v| v.is_finite()),
      "atm cell {i} has non-finite state {:?}",
      s
    );
    assert!(s[0] > 0.0, "atm cell {i} non-positive density");
    assert!(s[4] > 0.0, "atm cell {i} non-positive energy");
  }

  // 2. The day side warms relative to the night side. We pick the
  // hottest day-side cell and the coldest night-side cell — even with
  // only a few ticks the contrast should be unambiguous.
  let mut day_max = f64::NEG_INFINITY;
  let mut night_min = f64::INFINITY;
  for i in 0..surface_mesh.cell_count() {
    let centroid = surface_mesh.cell_centroid(CellId::from(i));
    let r =
      (centroid[0].powi(2) + centroid[1].powi(2) + centroid[2].powi(2)).sqrt();
    let mu = centroid[0] / r;
    let t = surface_temperature.state(CellId::from(i))[0];
    if mu > 0.0 {
      day_max = day_max.max(t);
    } else if mu < 0.0 {
      night_min = night_min.min(t);
    }
  }
  assert!(
    day_max > night_min,
    "day-side max ({day_max}) should exceed night-side min ({night_min})"
  );

  Ok(())
}
