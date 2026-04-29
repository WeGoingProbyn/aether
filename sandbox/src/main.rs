// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use aer::{AtmosphereModel, AtmosphereShellLayout, AtmosphereSpec};
use aether::{core::Aether, factory::WorldFactory};
use cosmo::factory as cosmo_factory;
use eidolon::{
  export::write_render_frame_vtu,
  extract::{scalar_component_layer, tessera_debug_frame},
  ir::{LayerId, MeshRepresentation, Palette, RenderLayer, RenderMeshId},
};
use nexus::{
  FieldKey, FieldName, FieldStorage, MeshKey, SoaField, Stage, StageContext,
  WorldId,
};
use utility::domain::CellId;
use utility::error::AetherResult;
use utility::info;
use utility::logger::{Level, LogWriter, Logger, StdSink};
use utility::profiler::Profiler;
use utility::thread::pool::Pool;

const SURFACE_TEMPERATURE: FieldKey =
  FieldKey::new(MeshKey::SURFACE, FieldName::Temperature);

struct DummySurfaceHeating {
  writes: [FieldKey; 1],
  reads: [FieldKey; 1],
}

impl DummySurfaceHeating {
  fn new() -> Self {
    Self {
      writes: [SURFACE_TEMPERATURE],
      reads: [SURFACE_TEMPERATURE],
    }
  }
}

impl Stage for DummySurfaceHeating {
  fn name(&self) -> &'static str {
    "dummy_surface_heating"
  }

  fn reads(&self) -> &[FieldKey] {
    &self.reads
  }

  fn writes(&self) -> &[FieldKey] {
    &self.writes
  }

  fn run(&mut self, mut ctx: StageContext<'_>) -> AetherResult<()> {
    let mesh_cell_count = ctx
      .world
      .tessera
      .mesh(MeshKey::SURFACE)
      .map(|mesh| mesh.cell_count())
      .unwrap_or(0);

    let field: &SoaField<1> = ctx
      .world
      .fields
      .read(SURFACE_TEMPERATURE)
      .expect("dummy stage declares surface temperature as a read");
    let before = field.state(CellId::from(0));

    let field: &mut SoaField<1> = ctx
      .world
      .fields
      .write(SURFACE_TEMPERATURE)
      .expect("dummy stage declares surface temperature as a write");

    for cell in 0..field.len() {
      field.write(CellId::from(cell), &[288.0 + ctx.world.dt]);
    }

    let field: &SoaField<1> = ctx
      .world
      .fields
      .read(SURFACE_TEMPERATURE)
      .expect("dummy stage declares surface temperature as a read");
    let after = field.state(CellId::from(0));

    info!(
      "dummy stage wrote surface temperature for {:?}: {} cells (mesh has {}), sample {:?} -> {:?}",
      ctx.world.world_id,
      field.len(),
      mesh_cell_count,
      before,
      after
    );

    Ok(())
  }
}

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
  let mut world_factory = WorldFactory::new(world_id, seed);
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
  let surface_cell_count = surface_mesh.cell_count();
  let atmosphere_model = AtmosphereModel::new(MeshKey::ATMOSPHERE)
    .with_cfl(0.25)
    .with_current_state_background_correction();
  let atmosphere_fields = atmosphere_model.fields();

  let atmosphere_spec = AtmosphereSpec::from_world_constants(&world_constants)?;
  world_factory.register_field(
    SURFACE_TEMPERATURE,
    atmosphere_spec.temperature_field(surface_cell_count),
  );
  atmosphere_model.register_fields(
    world_factory.pleroma_mut(),
    atmosphere_mesh.as_ref(),
    &world_constants,
    shell_layout.reference_radius(),
  )?;

  world_factory.add_stage(DummySurfaceHeating::new());
  world_factory.add_scalar_interface_flux(
    radial_coupler_index,
    SURFACE_TEMPERATURE,
    atmosphere_fields.temperature,
    atmosphere_fields.temperature_tendency,
    1.0e-15,
  )?;
  atmosphere_model.add_stages(world_factory.nexus_mut())?;
  let world = world_factory.build()?;

  let mut worlds = HashMap::new();
  worlds.insert(world_id, world);

  let mut aether = Aether::new(worlds, Pool::default());
  aether.step(0.05)?;
  aether.step(0.05)?;

  let world = aether
    .world(world_id)
    .expect("sandbox world should still be registered");
  let mut frame = tessera_debug_frame(0, 0.1, world.id(), world.tessera());
  if let Some(surface_temperature) =
    world.pleroma().read::<SoaField<1>>(SURFACE_TEMPERATURE)
  {
    let target = RenderMeshId {
      world: world.id(),
      mesh: MeshKey::SURFACE,
      representation: MeshRepresentation::BoundaryFaces,
    };
    let mut layer = scalar_component_layer(
      LayerId("surface_temperature"),
      "surface_temperature",
      target,
      SURFACE_TEMPERATURE,
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
      LayerId("atmosphere_temperature"),
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
      LayerId("atmosphere_temperature_tendency"),
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
      LayerId("atmosphere_pressure"),
      "atmosphere_pressure",
      target,
      atmosphere_fields.pressure,
      atmosphere_pressure,
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
      LayerId("atmosphere_density"),
      "atmosphere_density",
      target,
      atmosphere_fields.euler_state,
      atmosphere_euler_state,
      0,
    );
    layer.palette = Palette::thermal();
    frame.worlds[0].layers.push(RenderLayer::Scalar(layer));
  }

  let written = write_render_frame_vtu(&frame, "output/eidolon")?;
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
