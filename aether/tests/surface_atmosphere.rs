// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use aer::{AtmosphereModel, AtmosphereShellLayout};
use aether::{
  core::{Aether, System},
  factory::WorldFactory,
};
use cosmo::factory as cosmo_factory;
use nexus::{FieldStorage, MeshKey, SoaField, WorldId};
use terra::SurfaceThermalModel;
use utility::{
  domain::{CellId, SystemId},
  error::AetherResult,
  thread::pool::Pool,
};

#[test]
fn surface_atmosphere_world_ticks_with_coupled_models() -> AetherResult<()> {
  let world_id = WorldId(0);
  let angular_dims = [2, 2];
  let surface_radial_layers = 1;
  let atmosphere_radial_layers = 3;
  let atmosphere_height = 20_000.0;
  let surface_depth = 10_000.0;

  let mut factory = WorldFactory::new(world_id, cosmo_factory::earth());
  let constants = factory.constants();
  let shell_layout =
    AtmosphereShellLayout::new(&constants, atmosphere_height, surface_depth)?;

  factory = factory.cube_sphere_surface(
    shell_layout.surface_shell_spec(angular_dims, surface_radial_layers),
  );
  factory = factory.cube_sphere_atmosphere(
    shell_layout.atmosphere_shell_spec(angular_dims, atmosphere_radial_layers),
  );
  let coupler_index =
    factory.add_radial_stack_coupler(MeshKey::SURFACE, MeshKey::ATMOSPHERE)?;
  assert_eq!(
    factory
      .tessera()
      .coupler_view(coupler_index)
      .expect("registered coupler should have a view")
      .pair_count(),
    6 * angular_dims[0] * angular_dims[1]
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
    .expect("earth seed should have atmosphere constants")
    .reference_temperature;
  let surface_model = SurfaceThermalModel::new(MeshKey::SURFACE)
    .with_initial_temperature(reference_temperature)
    .with_target_temperature(reference_temperature);
  let surface_fields = surface_model.fields();
  let atmosphere_model = AtmosphereModel::new(MeshKey::ATMOSPHERE)
    .with_cfl(0.25)
    .with_current_state_background_correction();
  let atmosphere_fields = atmosphere_model.fields();

  surface_model
    .register_fields(factory.pleroma_mut(), surface_mesh.as_ref())?;
  atmosphere_model.register_fields(
    factory.pleroma_mut(),
    atmosphere_mesh.as_ref(),
    &constants,
    shell_layout.reference_radius(),
  )?;

  surface_model.add_stages(factory.nexus_mut())?;
  factory.add_scalar_interface_flux(
    coupler_index,
    surface_fields.temperature,
    atmosphere_fields.temperature,
    atmosphere_fields.temperature_tendency,
    1.0e-15,
  )?;
  atmosphere_model.add_stages(factory.nexus_mut())?;

  let world = factory.build()?;
  let system_id = SystemId(0);
  let mut systems = HashMap::new();
  systems.insert(system_id, System::single(system_id, world));

  let mut aether = Aether::new(systems, Pool::default());
  aether.step(0.05)?;
  aether.step(0.05)?;

  let world = aether
    .world(world_id)
    .expect("world should remain registered");
  let surface_temperature: &SoaField<1> = world
    .pleroma()
    .read(surface_fields.temperature)
    .expect("surface temperature should be registered");
  assert_eq!(surface_temperature.len(), surface_mesh.cell_count());
  assert!(
    surface_temperature
      .component(0)
      .as_ref()
      .iter()
      .all(|value| value.is_finite() && *value > 0.0)
  );

  let atmosphere_state: &SoaField<5> = world
    .pleroma()
    .read(atmosphere_fields.euler_state)
    .expect("atmosphere euler state should be registered");
  assert_eq!(atmosphere_state.len(), atmosphere_mesh.cell_count());
  for cell in 0..atmosphere_state.len() {
    let state = atmosphere_state.state(CellId::from(cell));
    assert!(
      state.iter().all(|value| value.is_finite()),
      "cell {} has non-finite euler state {:?}",
      cell,
      state
    );
    assert!(state[0] > 0.0, "cell {} has non-positive density", cell);
    assert!(state[4] > 0.0, "cell {} has non-positive energy", cell);
  }

  let pressure: &SoaField<1> = world
    .pleroma()
    .read(atmosphere_fields.pressure)
    .expect("atmosphere pressure diagnostic should be registered");
  assert_eq!(pressure.len(), atmosphere_mesh.cell_count());
  assert!(
    pressure
      .component(0)
      .as_ref()
      .iter()
      .all(|value| value.is_finite() && *value > 0.0)
  );

  let tendency: &SoaField<1> = world
    .pleroma()
    .read(atmosphere_fields.temperature_tendency)
    .expect("atmosphere tendency should be registered");
  assert_eq!(tendency.len(), atmosphere_mesh.cell_count());
  assert!(
    tendency
      .component(0)
      .as_ref()
      .iter()
      .all(|value| value.is_finite())
  );

  Ok(())
}
