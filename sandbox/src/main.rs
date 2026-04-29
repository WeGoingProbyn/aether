// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashMap, sync::Arc};

use aether::core::{Aether, World};
use cosmo::factory;
use eidolon::{
  export::write_render_frame_vtu,
  extract::{scalar_component_layer, tessera_debug_frame},
  ir::{LayerId, MeshRepresentation, Palette, RenderLayer, RenderMeshId},
};
use nexus::{
  FieldKey, FieldName, FieldStorage, MeshKey, Nexus, SoaField, Stage,
  StageContext, WorldId,
};
use pleroma::Pleroma;
use syzygy::ScalarRelaxation;
use tessera::{
  coupling::MeshCoupler,
  cube_sphere::{CubeSphere, CubeSphereShellSpec},
  geometry::CellGeometry,
  mesh::Mesh,
  radial_stack::RadialStackCoupler,
  world_mesh::Tessera,
};

use utility::domain::{BoundaryTag, CellId};
use utility::error::AetherResult;
use utility::info;
use utility::logger::{Level, LogWriter, Logger, StdSink};
use utility::profiler::Profiler;
use utility::thread::pool::Pool;

const SURFACE_TEMPERATURE: FieldKey =
  FieldKey::new(MeshKey::SURFACE, FieldName::Temperature);
const ATMOSPHERE_TEMPERATURE: FieldKey =
  FieldKey::new(MeshKey::ATMOSPHERE, FieldName::Temperature);

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

  let surface_mesh = Arc::new(CubeSphere::shell(
    CubeSphereShellSpec::uniform(
      [angular_dims[0], angular_dims[1], surface_radial_layers],
      0.9,
      1.0,
    )
    .with_boundaries(BoundaryTag::Ground, BoundaryTag::AtmosphereEdge),
  ));
  let atmosphere_mesh = Arc::new(CubeSphere::shell(
    CubeSphereShellSpec::uniform(
      [angular_dims[0], angular_dims[1], atmosphere_radial_layers],
      1.0,
      1.2,
    )
    .with_boundaries(BoundaryTag::Ground, BoundaryTag::AtmosphereEdge),
  ));
  let surface_cell_count = surface_mesh.cell_count();
  let atmosphere_cell_count = atmosphere_mesh.cell_count();

  let mut tessera = Tessera::new();
  let surface_for_registry: Arc<dyn Mesh<3>> = surface_mesh;
  let atmosphere_for_registry: Arc<dyn Mesh<3>> = atmosphere_mesh;
  tessera.register_mesh(MeshKey::SURFACE, surface_for_registry);
  tessera.register_mesh(MeshKey::ATMOSPHERE, atmosphere_for_registry);

  let radial_coupler = RadialStackCoupler::new(
    angular_dims,
    surface_radial_layers,
    atmosphere_radial_layers,
  );
  let radial_pair_count = radial_coupler.pairs().len();
  let radial_coupler_index =
    tessera.add_coupler(MeshKey::SURFACE, MeshKey::ATMOSPHERE, radial_coupler);
  info!(
    "registered surface-atmosphere radial coupler with {} face pairs",
    radial_pair_count
  );

  let mut pleroma = Pleroma::new();
  pleroma.register_field(
    SURFACE_TEMPERATURE,
    SoaField::<1>::zeros(surface_cell_count),
  );
  pleroma.register_field(
    ATMOSPHERE_TEMPERATURE,
    SoaField::<1>::zeros(atmosphere_cell_count),
  );

  let mut nexus = Nexus::new();
  nexus.add(DummySurfaceHeating::new());
  nexus.add(ScalarRelaxation::new(
    radial_coupler_index,
    SURFACE_TEMPERATURE,
    ATMOSPHERE_TEMPERATURE,
    0.01,
  ));
  let compiled_nexus = nexus.build(&pleroma)?;

  let world_id = WorldId(0);
  let world =
    World::new(world_id, factory::earth(), tessera, pleroma, compiled_nexus);

  let mut worlds = HashMap::new();
  worlds.insert(world_id, world);

  let mut aether = Aether::new(worlds, Pool::default());
  aether.step(60.0)?;
  aether.step(20.0)?;

  let world = aether
    .world(world_id)
    .expect("sandbox world should still be registered");
  let mut frame = tessera_debug_frame(0, 80.0, world.id(), world.tessera());
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
  if let Some(atmosphere_temperature) =
    world.pleroma().read::<SoaField<1>>(ATMOSPHERE_TEMPERATURE)
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
      ATMOSPHERE_TEMPERATURE,
      atmosphere_temperature,
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
