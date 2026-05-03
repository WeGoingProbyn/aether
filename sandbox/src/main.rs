// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use aer::{AtmosphereModel, AtmosphereShellLayout};
use aether::{
  core::{Aether, System},
  factory::WorldFactory,
};
use cosmo::factory as cosmo_factory;
use eidolon::{
  export::write_render_frame_via_registry,
  extract::{scalar_component_layer, tessera_debug_frame},
  ir::{LayerId, MeshRepresentation, Palette, RenderLayer, RenderMeshId},
};
use lumen::{RadiationCoefficients, RadiationModel};
use nexus::{MeshKey, SoaField, WorldId};
use terra::SurfaceThermalModel;
use utility::info;
use utility::logger::{Level, LogWriter, Logger, StdSink};
use utility::profiler::Profiler;
use utility::thread::pool::Pool;
use utility::{domain::SystemId, error::AetherResult};

fn main() -> AetherResult<()> {
  Logger::init(
    vec![Box::new(StdSink::new(std::io::stdout()).capacity(1))],
    Level::Trace,
  );

  Profiler::init();

  let angular_dims = [4, 4];
  let surface_radial_layers = 2;
  let atmosphere_radial_layers = 10;
  let atmosphere_height = 20_000.0;
  let surface_depth = 10_000.0;
  let world_id = WorldId(0);
  let seed = cosmo_factory::earth();
  let primary = cosmo_factory::sun();
  let mut world_factory =
    WorldFactory::new(world_id, seed).with_primary(primary);
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
  let radial_pair_count = world_factory
    .tessera()
    .coupler_view(radial_coupler_index)
    .map(|view| view.pair_count())
    .unwrap_or(0);
  info!(
    "registered surface-atmosphere radial coupler with {} face pairs",
    radial_pair_count
  );

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
  let radiation_fields = radiation_model.fields();

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

  let mut aether = Aether::new(systems, Pool::default());
  aether.step(0.05)?;
  aether.step(0.05)?;

  let world = aether
    .world(world_id)
    .expect("sandbox world should still be registered");
  let mut frame = tessera_debug_frame(0, 0.1, world.id(), world.tessera());
  if let Some(surface_temperature) = world
    .pleroma()
    .read::<SoaField<1>>(surface_fields.temperature)
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
      surface_fields.temperature,
      surface_temperature,
      0,
    );
    layer.palette = Palette::thermal();
    frame.worlds[0].layers.push(RenderLayer::Scalar(layer));
  }
  if let Some(atmosphere_temperature) = world
    .pleroma()
    .read::<SoaField<1>>(atmosphere_fields.temperature)
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
      atmosphere_fields.temperature,
      atmosphere_temperature,
      0,
    );
    layer.palette = Palette::thermal();
    frame.worlds[0].layers.push(RenderLayer::Scalar(layer));
  }
  if let Some(atmosphere_temperature_tendency) = world
    .pleroma()
    .read::<SoaField<1>>(atmosphere_fields.temperature_tendency)
  {
    let target = RenderMeshId {
      world: world.id(),
      mesh: MeshKey::ATMOSPHERE,
      representation: MeshRepresentation::BoundaryFaces,
    };
    let mut layer = scalar_component_layer(
      LayerId::from_static("atmosphere_temperature_tendency"),
      "atmosphere_temperature_tendency",
      target,
      atmosphere_fields.temperature_tendency,
      atmosphere_temperature_tendency,
      0,
    );
    layer.palette = Palette::thermal();
    frame.worlds[0].layers.push(RenderLayer::Scalar(layer));
  }
  if let Some(atmosphere_pressure) = world
    .pleroma()
    .read::<SoaField<1>>(atmosphere_fields.pressure)
  {
    let target = RenderMeshId {
      world: world.id(),
      mesh: MeshKey::ATMOSPHERE,
      representation: MeshRepresentation::BoundaryFaces,
    };
    let mut layer = scalar_component_layer(
      LayerId::from_static("atmosphere_pressure"),
      "atmosphere_pressure",
      target,
      atmosphere_fields.pressure,
      atmosphere_pressure,
      0,
    );
    layer.palette = Palette::thermal();
    frame.worlds[0].layers.push(RenderLayer::Scalar(layer));
  }
  if let Some(radiative_heating) = world
    .pleroma()
    .read::<SoaField<1>>(radiation_fields.heating_tendency)
  {
    let target = RenderMeshId {
      world: world.id(),
      mesh: MeshKey::ATMOSPHERE,
      representation: MeshRepresentation::BoundaryFaces,
    };
    let mut layer = scalar_component_layer(
      LayerId::from_static("radiative_heating"),
      "radiative_heating",
      target,
      radiation_fields.heating_tendency,
      radiative_heating,
      0,
    );
    layer.palette = Palette::thermal();
    frame.worlds[0].layers.push(RenderLayer::Scalar(layer));
  }
  if let Some(net_surface_flux) = world
    .pleroma()
    .read::<SoaField<1>>(radiation_fields.net_surface_flux)
  {
    let target = RenderMeshId {
      world: world.id(),
      mesh: MeshKey::SURFACE,
      representation: MeshRepresentation::BoundaryFaces,
    };
    let mut layer = scalar_component_layer(
      LayerId::from_static("net_surface_flux"),
      "net_surface_flux",
      target,
      radiation_fields.net_surface_flux,
      net_surface_flux,
      0,
    );
    layer.palette = Palette::thermal();
    frame.worlds[0].layers.push(RenderLayer::Scalar(layer));
  }
  if let Some(atmosphere_euler_state) = world
    .pleroma()
    .read::<SoaField<5>>(atmosphere_fields.euler_state)
  {
    let target = RenderMeshId {
      world: world.id(),
      mesh: MeshKey::ATMOSPHERE,
      representation: MeshRepresentation::BoundaryFaces,
    };
    let mut layer = scalar_component_layer(
      LayerId::from_static("atmosphere_density"),
      "atmosphere_density",
      target,
      atmosphere_fields.euler_state,
      atmosphere_euler_state,
      0,
    );
    layer.palette = Palette::thermal();
    frame.worlds[0].layers.push(RenderLayer::Scalar(layer));
  }

  let written = write_render_frame_via_registry(&frame, "output/eidolon")?;
  info!(
    "eidolon wrote {} debug VTK files to output/eidolon",
    written.len()
  );
  for path in written {
    info!("eidolon debug VTK: {}", path.display());
  }

  Profiler::print(&mut LogWriter::new(Level::Info));
  Ok(())
}
