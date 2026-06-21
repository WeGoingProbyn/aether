// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Shared world-build logic for the sandbox demo and its VTK
//! regression test. Keeps the physics setup in one place so the
//! bevy demo and the headless regression test never drift.

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
use eidolon::extract::{
  ExtractConfig, MeshConfig, ScalarLayerConfig, scalar_component_layer,
  tessera_debug_frame,
};
use eidolon::ir::{
  LayerId, MeshRepresentation, Palette, RenderFrame, RenderLayer, RenderMeshId,
};
use lumen::{DiurnalSunStep, RadiationCoefficients, RadiationModel};
use nexus::{MeshKey, SoaField, SubsystemId, WorldId};
use syzygy::ScalarRelaxation;
use terra::SurfaceThermalModel;
use tessera::cube_sphere::CubeSphereShellSpec;
use thalassa::{OceanColumnLayout, OceanModel};
use utility::domain::{FieldKey, FieldName, SystemId};
use utility::error::AetherResult;
use utility::profile;
use utility::thread::pool::Pool;

/// Subsystem clock the ocean advances on — slower than the (default,
/// CFL-limited) atmosphere subsystem.
const OCEAN_SUBSYSTEM: SubsystemId = SubsystemId(1);

pub const SANDBOX_WORLD_ID: WorldId = WorldId(0);

/// Build the demo Aether (Earth + Sun, surface+atmosphere shells,
/// radiation + heat-flux + surface thermal stages registered).
///
/// Returns the assembled Aether plus the layout used to size the
/// shells — the bevy demo uses `reference_radius()` for the camera.
#[profile]
pub fn build_demo_aether() -> AetherResult<(Aether, AtmosphereShellLayout)> {
  let angular_dims = [128, 128];
  let surface_radial_layers = 10;
  let atmosphere_radial_layers = 30;
  let atmosphere_height = 20_000.0;
  let surface_depth = 10_000.0;
  let seed = cosmo_factory::earth();
  let primary = cosmo_factory::sun();

  let mut world_factory = WorldFactory::new(SANDBOX_WORLD_ID, seed.clone())
    .with_primary(primary.clone());
  let world_constants = world_factory.constants();
  let shell_layout = AtmosphereShellLayout::new(
    &world_constants,
    atmosphere_height,
    surface_depth,
  )?;

  world_factory = world_factory
    .cube_sphere_surface(
      shell_layout.surface_shell_spec(angular_dims, surface_radial_layers),
    )
    .with_partition_count(6);
  world_factory = world_factory
    .cube_sphere_atmosphere(
      shell_layout
        .atmosphere_shell_spec(angular_dims, atmosphere_radial_layers),
    )
    .with_partition_count(6);
  let radial_coupler_index = world_factory
    .add_radial_stack_coupler(MeshKey::SURFACE, MeshKey::ATMOSPHERE)?;

  let surface_mesh = world_factory
    .tessera()
    .mesh(MeshKey::SURFACE)
    .expect("surface mesh was just registered")
    .clone();
  let atmosphere_mesh = world_factory
    .tessera()
    .mesh(MeshKey::ATMOSPHERE)
    .expect("atmosphere mesh was just registered")
    .clone();

  let surface_model = SurfaceThermalModel::new(MeshKey::SURFACE)
    .with_initial_temperature(
      world_constants
        .atmosphere
        .map_or(288.0, |atmosphere| atmosphere.reference_temperature),
    )
    .with_heat_capacity_per_area(1.0e7);
  let surface_fields = surface_model.fields();
  let atmosphere_model = AtmosphereModel::new(MeshKey::ATMOSPHERE)
    .with_cfl(0.25)
    .with_current_state_background_correction()
    .with_radiative_heating();
  let atmosphere_fields = atmosphere_model.fields();
  let radiation_model = RadiationModel::from_world_constants(
    MeshKey::ATMOSPHERE,
    MeshKey::SURFACE,
    &world_constants,
    RadiationCoefficients::default(),
  )?;

  surface_model
    .register_fields(world_factory.pleroma_mut(), surface_mesh.as_ref())?;
  atmosphere_model.register_fields(
    world_factory.pleroma_mut(),
    atmosphere_mesh.as_ref(),
    &world_constants,
    shell_layout.reference_radius(),
  )?;
  radiation_model.register_fields(
    world_factory.pleroma_mut(),
    atmosphere_mesh.as_ref(),
    surface_mesh.as_ref(),
  )?;
  radiation_model.register_default_sun_position(
    world_factory.pleroma_mut(),
    [1.0, 0.0, 0.0],
  );

  surface_model.add_stages(world_factory.nexus_mut())?;
  world_factory.add_scalar_interface_flux(
    radial_coupler_index,
    surface_fields.temperature,
    atmosphere_fields.temperature,
    atmosphere_fields.temperature_tendency,
    1.0e-15,
  )?;
  radiation_model.add_stages(world_factory.nexus_mut())?;
  atmosphere_model.add_stages(world_factory.nexus_mut())?;
  let world = world_factory.build()?;

  let system_id = SystemId(0);
  let mut systems = HashMap::new();
  systems.insert(system_id, System::single(system_id, world));

  Ok((Aether::new(systems, Pool::default()), shell_layout))
}

/// Build the fully-coupled ocean world: a moist, rotating atmosphere
/// (aer) over a thermodynamic ocean (thalassa), driven by gray radiation
/// (lumen), a closed air–sea water cycle (evaporation + microphysics) and
/// a diurnal sun, advanced by the multirate scheduler with the atmosphere
/// on a fast clock and the ocean on a slow one.
///
/// This is the runnable counterpart to the `coupled_ocean_world`
/// integration test. Modest resolution keeps the serial moist solver
/// responsive for the live bevy view.
#[profile]
pub fn build_ocean_world_aether()
-> AetherResult<(Aether, AtmosphereShellLayout)> {
  build_ocean_world_scheme(aer::AtmosphereScheme::Hevi)
}

/// Which coupling stages to wire into the ocean world — for bisecting which
/// source drives the coupled instability. Default = the full demo.
#[derive(Clone, Copy, Debug)]
pub struct OceanWorldCoupling {
  pub radiation: bool,
  pub evaporation: bool,
  pub saturation: bool,
}

impl Default for OceanWorldCoupling {
  fn default() -> Self {
    Self {
      radiation: true,
      evaporation: true,
      saturation: true,
    }
  }
}

/// Build the ocean world with a chosen atmosphere time-stepping scheme — for
/// A/B comparison of explicit vs HEVI dynamics.
pub fn build_ocean_world_scheme(
  scheme: aer::AtmosphereScheme,
) -> AetherResult<(Aether, AtmosphereShellLayout)> {
  build_ocean_world_configured(scheme, OceanWorldCoupling::default())
}

/// Build the ocean world with a chosen scheme and a chosen subset of coupling
/// stages enabled.
pub fn build_ocean_world_configured(
  scheme: aer::AtmosphereScheme,
  coupling: OceanWorldCoupling,
) -> AetherResult<(Aether, AtmosphereShellLayout)> {
  let angular_dims = [16, 16];
  let atmosphere_layers = 6;
  let ocean_layers = 2;
  let atmosphere_height = 20_000.0;
  let surface_depth = 10_000.0;

  // 6-way partitioning runs the (CFL-substepping) atmosphere panels in
  // parallel so each 60 s frame stays responsive once weather develops.
  let mut factory = WorldFactory::new(SANDBOX_WORLD_ID, cosmo_factory::earth())
    .with_primary(cosmo_factory::sun())
    .with_partition_count(6);
  let constants = factory.constants();
  let shell_layout =
    AtmosphereShellLayout::new(&constants, atmosphere_height, surface_depth)?;
  let reference_radius = shell_layout.reference_radius();
  let atmosphere = constants.atmosphere.expect("earth has an atmosphere");
  let angular_velocity = atmosphere.angular_velocity;
  let reference_temperature = atmosphere.reference_temperature;

  factory = factory.cube_sphere_atmosphere(
    shell_layout.atmosphere_shell_spec(angular_dims, atmosphere_layers),
  );
  factory = factory.cube_sphere_ocean(CubeSphereShellSpec::uniform(
    [angular_dims[0], angular_dims[1], ocean_layers],
    reference_radius * 0.98,
    reference_radius,
  ));
  let coupler =
    factory.add_radial_stack_coupler(MeshKey::OCEAN, MeshKey::ATMOSPHERE)?;

  let atmosphere_mesh =
    factory.tessera().mesh(MeshKey::ATMOSPHERE).unwrap().clone();
  let ocean_mesh = factory.tessera().mesh(MeshKey::OCEAN).unwrap().clone();

  let atmosphere_model = AtmosphereModel::new(MeshKey::ATMOSPHERE)
    .with_cfl(0.25)
    .with_current_state_background_correction()
    .with_radiative_heating()
    .with_rotation()
    // Vertically-implicit dynamics: large stable steps on the thin shell,
    // one per-panel HEVI solver under nexus' partitioned dispatch.
    .with_hevi();
  let atmosphere_fields = atmosphere_model.fields();

  let ocean_model = OceanModel::new(
    MeshKey::OCEAN,
    OceanColumnLayout::cube_sphere(angular_dims, ocean_layers),
  )
  .with_initial_temperature(reference_temperature + 2.0)
  // A thin mixed layer so the sea surface visibly responds to the moving
  // day/night radiative forcing within a viewing session.
  .with_layer_thickness(2.0)
  .with_subsystem(OCEAN_SUBSYSTEM);
  let ocean_fields = ocean_model.fields();

  let radiation_model = RadiationModel::from_world_constants(
    MeshKey::ATMOSPHERE,
    MeshKey::OCEAN,
    &constants,
    RadiationCoefficients::default(),
  )?;

  let sst =
    FieldKey::new(MeshKey::ATMOSPHERE, FieldName::SeaSurfaceTemperature);
  let evaporation =
    FieldKey::new(MeshKey::ATMOSPHERE, FieldName::EvaporationFlux);
  let precipitation =
    FieldKey::new(MeshKey::ATMOSPHERE, FieldName::PrecipitationFlux);

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

  // Add order fixes coupling resolution (earlier-added runs first on any
  // data conflict). Atmosphere group is the default subsystem; the ocean
  // is its own slower subsystem.
  if coupling.radiation {
    radiation_model.add_stages(factory.nexus_mut())?;
  }
  let sst_relax = ScalarRelaxation::from_coupler(
    factory.tessera(),
    coupler,
    ocean_fields.temperature,
    sst,
    1.0,
  )?;
  factory.add_stage(sst_relax);
  atmosphere_model.add_stages(factory.nexus_mut())?;
  if coupling.evaporation {
    factory.add_stage(EvaporationStep::new(
      MeshKey::ATMOSPHERE,
      atmosphere_fields.euler_state,
      sst,
      evaporation,
      ShellColumns::cube_sphere(angular_dims, atmosphere_layers),
      1.0e-2,
    )?);
  }
  if coupling.saturation {
    factory.add_stage(SaturationAdjustmentStep::new(
      MeshKey::ATMOSPHERE,
      atmosphere_fields.euler_state,
      precipitation,
    )?);
  }
  factory.add_stage(DiurnalSunStep::new(angular_velocity));
  ocean_model.add_stages(factory.nexus_mut())?;

  // Atmosphere (default subsystem) and ocean (its own subsystem) each step
  // once per outer tick; the atmosphere CFL-substeps internally. No fast
  // cadence here — at a 60 s tick it would force ~120 subcycles per frame.

  let world = factory.build()?;
  let system_id = SystemId(0);
  let mut systems = HashMap::new();
  systems.insert(system_id, System::single(system_id, world));

  Ok((Aether::new(systems, Pool::default()), shell_layout))
}

/// Producer config for the ocean-world demo: ocean SST plus atmosphere
/// humidity / pressure / temperature scalar layers.
pub fn ocean_world_extract_config() -> ExtractConfig {
  let scalar = |id: &'static str,
                label: &str,
                mesh: MeshKey,
                field: FieldName,
                palette: Palette| ScalarLayerConfig {
    id: LayerId::from_static(id),
    label: label.into(),
    target_mesh: mesh,
    target_representation: MeshRepresentation::BoundaryFaces,
    field: FieldKey::new(mesh, field),
    component: 0,
    palette,
  };
  ExtractConfig {
    world_label: "ocean world".into(),
    world_scale: 1.0,
    meshes: vec![
      MeshConfig {
        mesh_key: MeshKey::OCEAN,
        representation: MeshRepresentation::BoundaryFaces,
        label: "ocean".into(),
      },
      MeshConfig {
        mesh_key: MeshKey::ATMOSPHERE,
        representation: MeshRepresentation::BoundaryFaces,
        label: "atmosphere".into(),
      },
    ],
    layers: vec![
      scalar(
        "ocean_temperature",
        "ocean_temperature",
        MeshKey::OCEAN,
        FieldName::Temperature,
        Palette::thermal(),
      ),
      // First atmosphere layer is the default binding for that mesh —
      // temperature shows the travelling day/night pattern immediately.
      scalar(
        "atmosphere_temperature",
        "atmosphere_temperature",
        MeshKey::ATMOSPHERE,
        FieldName::Temperature,
        Palette::thermal(),
      ),
      scalar(
        "atmosphere_humidity",
        "atmosphere_humidity",
        MeshKey::ATMOSPHERE,
        FieldName::Humidity,
        Palette::thermal(),
      ),
      scalar(
        "atmosphere_pressure",
        "atmosphere_pressure",
        MeshKey::ATMOSPHERE,
        FieldName::Pressure,
        Palette::thermal(),
      ),
    ],
    track_sun_direction: true,
  }
}

/// Producer config the bevy demo uses. Tracks the surface mesh and a
/// handful of scalar layers (temperature, pressure, …). The VTK test
/// instead builds a debug frame manually since it needs the snapshot
/// shape, but the *fields* it cares about line up with this list.
pub fn demo_extract_config() -> ExtractConfig {
  ExtractConfig {
    world_label: "earth".into(),
    world_scale: 1.0,
    meshes: vec![
      MeshConfig {
        mesh_key: MeshKey::SURFACE,
        representation: MeshRepresentation::BoundaryFaces,
        label: "earth surface".into(),
      },
      MeshConfig {
        mesh_key: MeshKey::ATMOSPHERE,
        representation: MeshRepresentation::BoundaryFaces,
        label: "atmosphere shell".into(),
      },
    ],
    layers: vec![
      ScalarLayerConfig {
        id: LayerId::from_static("surface_temperature"),
        label: "surface_temperature".into(),
        target_mesh: MeshKey::SURFACE,
        target_representation: MeshRepresentation::BoundaryFaces,
        field: FieldKey::new(MeshKey::SURFACE, FieldName::Temperature),
        component: 0,
        palette: Palette::thermal(),
      },
      ScalarLayerConfig {
        id: LayerId::from_static("atmosphere_temperature"),
        label: "atmosphere_temperature".into(),
        target_mesh: MeshKey::ATMOSPHERE,
        target_representation: MeshRepresentation::BoundaryFaces,
        field: FieldKey::new(MeshKey::ATMOSPHERE, FieldName::Temperature),
        component: 0,
        palette: Palette::thermal(),
      },
      ScalarLayerConfig {
        id: LayerId::from_static("atmosphere_pressure"),
        label: "atmosphere_pressure".into(),
        target_mesh: MeshKey::ATMOSPHERE,
        target_representation: MeshRepresentation::BoundaryFaces,
        field: FieldKey::new(MeshKey::ATMOSPHERE, FieldName::Pressure),
        component: 0,
        palette: Palette::thermal(),
      },
    ],
    track_sun_direction: true,
  }
}

/// Build a snapshot-shaped `RenderFrame` from the aether's current
/// state. Used by the headless VTK regression test.
pub fn debug_render_frame(
  aether: &aether::core::Aether,
  frame: u64,
  sim_time: f64,
) -> RenderFrame {
  let world = aether
    .world(SANDBOX_WORLD_ID)
    .expect("sandbox world should be registered");
  let mut frame_data =
    tessera_debug_frame(frame, sim_time, world.id(), world.tessera());

  let surface_temperature_key =
    FieldKey::new(MeshKey::SURFACE, FieldName::Temperature);
  if let Some(samples) =
    world.pleroma().read::<SoaField<1>>(surface_temperature_key)
  {
    let target = RenderMeshId {
      world: world.id(),
      mesh: MeshKey::SURFACE,
      representation: MeshRepresentation::BoundaryFaces,
    };
    let mut layer = scalar_component_layer(
      LayerId::from_static("surface_temperature"),
      "surface_temperature",
      target,
      surface_temperature_key,
      samples,
      0,
    );
    layer.palette = Palette::thermal();
    frame_data.worlds[0].layers.push(RenderLayer::Scalar(layer));
  }

  let atmosphere_temperature_key =
    FieldKey::new(MeshKey::ATMOSPHERE, FieldName::Temperature);
  if let Some(samples) = world
    .pleroma()
    .read::<SoaField<1>>(atmosphere_temperature_key)
  {
    let target = RenderMeshId {
      world: world.id(),
      mesh: MeshKey::ATMOSPHERE,
      representation: MeshRepresentation::BoundaryFaces,
    };
    let mut layer = scalar_component_layer(
      LayerId::from_static("atmosphere_temperature"),
      "atmosphere_temperature",
      target,
      atmosphere_temperature_key,
      samples,
      0,
    );
    layer.palette = Palette::thermal();
    frame_data.worlds[0].layers.push(RenderLayer::Scalar(layer));
  }

  frame_data
}
