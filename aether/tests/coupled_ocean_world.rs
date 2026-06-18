// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! First fully-coupled ocean world: cosmo seed → WorldFactory assembling a
//! moist rotating atmosphere (aer), a thermodynamic ocean (thalassa), gray
//! radiation (lumen), the air–sea water cycle (evaporation + microphysics),
//! and a diurnal sun — advanced by the multirate scheduler with the
//! atmosphere on a fast clock and the ocean on a slow one.
//!
//! This is the integration proof for the whole build. It asserts the
//! coupled DAG schedules and runs stably for many ticks, and that the
//! hydrological cycle is actually live: evaporation injects water vapour
//! that the moist Euler solver transports, so atmospheric water grows from
//! a dry start while every conserved field stays finite and physical.

use std::collections::HashMap;

use aer::{
  AtmosphereModel, AtmosphereShellLayout, EvaporationStep,
  SaturationAdjustmentStep, ShellColumns,
};
use aether::{
  core::{Aether, System},
  factory::WorldFactory,
};
use cosmo::factory as cosmo_factory;
use lumen::{DiurnalSunStep, RadiationCoefficients, RadiationModel};
use nexus::{
  FieldKey, FieldName, FieldStorage, MeshKey, SoaField, SubsystemId, WorldId,
};
use syzygy::ScalarRelaxation;
use tessera::cube_sphere::CubeSphereShellSpec;
use thalassa::{OceanColumnLayout, OceanModel};
use utility::{
  domain::CellId, domain::SystemId, error::AetherResult, thread::pool::Pool,
};

const OCEAN_SUBSYSTEM: SubsystemId = SubsystemId(1);

#[test]
fn coupled_ocean_world_runs_and_water_cycle_is_active() -> AetherResult<()> {
  let world_id = WorldId(0);
  let angular_dims = [2, 2];
  let atmosphere_layers = 3;
  let ocean_layers = 2;
  let atmosphere_height = 20_000.0;
  let surface_depth = 10_000.0;

  let mut factory = WorldFactory::new(world_id, cosmo_factory::earth())
    .with_primary(cosmo_factory::sun());
  let constants = factory.constants();
  let shell_layout =
    AtmosphereShellLayout::new(&constants, atmosphere_height, surface_depth)?;
  let reference_radius = shell_layout.reference_radius();
  let angular_velocity = constants
    .atmosphere
    .expect("earth has an atmosphere")
    .angular_velocity;
  let reference_temperature = constants
    .atmosphere
    .expect("earth has an atmosphere")
    .reference_temperature;

  // Moist rotating atmosphere shell, and an ocean column shell just below
  // the reference radius (shared angular dims so they couple radially).
  factory = factory.cube_sphere_atmosphere(
    shell_layout.atmosphere_shell_spec(angular_dims, atmosphere_layers),
  );
  factory = factory.cube_sphere_ocean(CubeSphereShellSpec::uniform(
    [angular_dims[0], angular_dims[1], ocean_layers],
    reference_radius * 0.98,
    reference_radius,
  ));

  // OCEAN (lower) ↔ ATMOSPHERE (upper): ocean top touches atmosphere base.
  let coupler =
    factory.add_radial_stack_coupler(MeshKey::OCEAN, MeshKey::ATMOSPHERE)?;

  let atmosphere_mesh =
    factory.tessera().mesh(MeshKey::ATMOSPHERE).unwrap().clone();
  let ocean_mesh = factory.tessera().mesh(MeshKey::OCEAN).unwrap().clone();

  // Models.
  let atmosphere_model = AtmosphereModel::new(MeshKey::ATMOSPHERE)
    .with_cfl(0.25)
    .with_current_state_background_correction()
    .with_radiative_heating()
    .with_rotation();
  let atmosphere_fields = atmosphere_model.fields();

  let ocean_model = OceanModel::new(
    MeshKey::OCEAN,
    OceanColumnLayout::cube_sphere(angular_dims, ocean_layers),
  )
  .with_initial_temperature(reference_temperature + 2.0)
  .with_subsystem(OCEAN_SUBSYSTEM);
  let ocean_fields = ocean_model.fields();

  // Lumen treats the ocean surface as the radiating surface: it reads ocean
  // temperature and deposits the net surface flux back into the ocean.
  let radiation_model = RadiationModel::from_world_constants(
    MeshKey::ATMOSPHERE,
    MeshKey::OCEAN,
    &constants,
    RadiationCoefficients::default(),
  )?;

  // Field keys for the water-cycle plumbing on the atmosphere mesh.
  let sst =
    FieldKey::new(MeshKey::ATMOSPHERE, FieldName::SeaSurfaceTemperature);
  let evaporation =
    FieldKey::new(MeshKey::ATMOSPHERE, FieldName::EvaporationFlux);
  let precipitation =
    FieldKey::new(MeshKey::ATMOSPHERE, FieldName::PrecipitationFlux);

  // Register state.
  atmosphere_model.register_fields(
    factory.pleroma_mut(),
    atmosphere_mesh.as_ref(),
    &constants,
    reference_radius,
  )?;
  ocean_model.register_fields(factory.pleroma_mut(), ocean_mesh.as_ref())?;
  radiation_model.register_fields(
    factory.pleroma_mut(),
    atmosphere_mesh.as_ref(),
    ocean_mesh.as_ref(),
  )?;
  radiation_model
    .register_default_sun_position(factory.pleroma_mut(), [1.0, 0.0, 0.0]);
  factory.register_field(
    sst,
    SoaField::<1>::from_fn(atmosphere_mesh.cell_count(), |_| {
      [reference_temperature]
    }),
  );
  factory.register_field(
    evaporation,
    SoaField::<1>::zeros(atmosphere_mesh.cell_count()),
  );
  factory.register_field(
    precipitation,
    SoaField::<1>::zeros(atmosphere_mesh.cell_count()),
  );

  // Stages — add order fixes coupling resolution (earlier-added runs first
  // on any data conflict). Atmosphere group stays on the default subsystem;
  // the ocean is on its own slower subsystem.
  // 1. Radiation: atmosphere heating + ocean surface flux.
  radiation_model.add_stages(factory.nexus_mut())?;
  // 2. Map ocean-surface temperature onto the atmosphere's SST field.
  // Relaxation coefficient rate·dt must stay ≤ 1 for stability; at the
  // atmosphere's inner dt this pulls SST a fraction of the way to the ocean
  // temperature each substep, converging within a few steps.
  let sst_relax = ScalarRelaxation::from_coupler(
    factory.tessera(),
    coupler,
    ocean_fields.temperature,
    sst,
    1.0,
  )?;
  factory.add_stage(sst_relax);
  // 3. Atmosphere thermodynamics + dynamics + diagnostics.
  atmosphere_model.add_stages(factory.nexus_mut())?;
  // 4. Evaporation injects vapour into the atmosphere's bottom layer.
  factory.add_stage(EvaporationStep::new(
    MeshKey::ATMOSPHERE,
    atmosphere_fields.euler_state,
    sst,
    evaporation,
    ShellColumns::cube_sphere(angular_dims, atmosphere_layers),
    1.0e-2,
  )?);
  // 5. Microphysics condenses supersaturated vapour into precipitation.
  factory.add_stage(SaturationAdjustmentStep::new(
    MeshKey::ATMOSPHERE,
    atmosphere_fields.euler_state,
    precipitation,
  )?);
  // 6. Diurnal sun rotation.
  factory.add_stage(DiurnalSunStep::new(angular_velocity));
  // 7. Ocean column thermodynamics (slow subsystem).
  ocean_model.add_stages(factory.nexus_mut())?;

  // Multirate: subcycle the atmosphere group; ocean steps once per outer dt.
  factory
    .nexus_mut()
    .set_subsystem_cadence(SubsystemId::DEFAULT, 0.5);

  let world = factory.build()?;
  let system_id = SystemId(0);
  let mut systems = HashMap::new();
  systems.insert(system_id, System::single(system_id, world));
  let mut aether = Aether::new(systems, Pool::default());

  // Run a number of coupled outer steps.
  for _ in 0..20 {
    aether.step(1.0)?;
  }

  let world = aether.world(world_id).expect("world remains registered");
  let pleroma = world.pleroma();

  // 1. Atmosphere state stays finite and physical.
  let state: &SoaField<6> =
    pleroma.read(atmosphere_fields.euler_state).unwrap();
  let mut total_vapour = 0.0;
  for i in 0..state.len() {
    let s = state.state(CellId::from(i));
    assert!(
      s.iter().all(|v| v.is_finite()),
      "atm cell {i} non-finite: {s:?}"
    );
    assert!(s[0] > 0.0, "atm cell {i} non-positive density");
    assert!(s[4] > 0.0, "atm cell {i} non-positive energy");
    assert!(s[5] >= 0.0, "atm cell {i} negative water mass");
    total_vapour += s[5];
  }

  // 2. The water cycle is live: evaporation has injected vapour into the
  //    initially dry atmosphere.
  assert!(
    total_vapour > 0.0,
    "expected atmospheric water vapour from evaporation, got {total_vapour}"
  );

  // 3. Ocean temperature stays finite and physical.
  let ocean_t: &SoaField<1> = pleroma.read(ocean_fields.temperature).unwrap();
  for i in 0..ocean_t.len() {
    let t = ocean_t.state(CellId::from(i))[0];
    assert!(t.is_finite() && t > 0.0, "ocean cell {i} has T={t}");
  }

  // 4. SST was coupled onto the atmosphere mesh (bottom cells ~ ocean temp).
  let sst_field: &SoaField<1> = pleroma.read(sst).unwrap();
  let stride = angular_dims[0] * angular_dims[1];
  for cell in 0..stride {
    let t = sst_field.state(CellId::from(cell))[0];
    assert!(t.is_finite() && t > 250.0, "SST cell {cell} = {t}");
  }

  Ok(())
}
