// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Shared world-build logic for the sandbox demo and its VTK
//! regression test. Keeps the physics setup in one place so the
//! bevy demo and the headless regression test never drift.

use std::collections::HashMap;

use aer::{AtmosphereModel, AtmosphereShellLayout};
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
use lumen::{RadiationCoefficients, RadiationModel};
use nexus::{MeshKey, SoaField, WorldId};
use terra::SurfaceThermalModel;
use utility::domain::{FieldKey, FieldName, SystemId};
use utility::error::AetherResult;
use utility::profile;
use utility::thread::pool::Pool;

pub const SANDBOX_WORLD_ID: WorldId = WorldId(0);

/// Build the demo Aether (Earth + Sun, surface+atmosphere shells,
/// radiation + heat-flux + surface thermal stages registered).
///
/// Returns the assembled Aether plus the layout used to size the
/// shells — the bevy demo uses `reference_radius()` for the camera.
#[profile]
pub fn build_demo_aether() -> AetherResult<(Aether, AtmosphereShellLayout)> {
  let angular_dims = [128, 128];
  let surface_radial_layers = 5;
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

  world_factory = world_factory.cube_sphere_surface(
    shell_layout.surface_shell_spec(angular_dims, surface_radial_layers),
  );
  world_factory = world_factory.cube_sphere_atmosphere(
    shell_layout.atmosphere_shell_spec(angular_dims, atmosphere_radial_layers),
  );
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
