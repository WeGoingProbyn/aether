// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Showcase: a fully-coupled aether world (terrain + ocean + moist atmosphere)
//! rendered through eidolon. The simulation runs on a background thread and
//! streams eidolon Update batches into a bounded channel; the bevy backend
//! applies them. This is the reference for *how a consumer builds a game world
//! on top of aether* — eidolon supplies semantic, art-free data (geometry, a
//! terrain heightfield to displace by, a land/ocean/ice categorical layer, and
//! debug scalar fields); the renderer decides how it all looks.
//!
//! Keys:
//!   1 / 2 / 3  — atmosphere overlay shows temperature / humidity / pressure
//!   4 / 5      — surface shows elevation / albedo (debug fields)

use std::time::{Duration, Instant};

use bevy::prelude::*;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};

use eidolon::{
  bevy::{AetherBevyPlugin, RenderRegistry, SunLight},
  extract::FrameProducer,
  ir::{LayerHandle, LayerId, MeshRepresentation, RenderMeshId},
  runtime::{render_channel, spawn_runner},
};
use sandbox::{
  SANDBOX_WORLD_ID, atmosphere::ShowcaseRenderPlugin, build_showcase_world,
  showcase_extract_config,
};
use utility::domain::MeshKey;
use utility::error::AetherResult;
use utility::info;
use utility::logger::{Level, Logger, StdSink};
use utility::profiler::Profiler;

/// Outer simulation step per tick (s). The coupled world is stable to ~30 s;
/// 20 s sits safely inside that, and HEVI clears it in one atmosphere solve.
/// Eidolon's frame interpolation smooths the large steps for the renderer.
const TICK_DT: f64 = 20.0;

/// Wall-clock pacing for the runner thread (~60 Hz).
const TICK_PERIOD: Duration = Duration::from_micros(16_667);

fn main() -> AetherResult<()> {
  Logger::init(
    vec![Box::new(StdSink::new(std::io::stdout()).capacity(1))],
    Level::Trace,
  );
  Profiler::init();

  let (mut aether, _layout) = build_showcase_world()?;
  let mut producer = FrameProducer::new(showcase_extract_config());

  let (tx, rx) = render_channel(64);

  let mut sim_time: f64 = 0.0;
  let mut frame: u64 = 0;
  let mut next_tick = Instant::now();
  let runner = spawn_runner(tx, move |_shutdown| {
    aether.step(TICK_DT)?;
    sim_time += TICK_DT;
    frame = frame.wrapping_add(1);
    let world = aether
      .world(SANDBOX_WORLD_ID)
      .expect("sandbox world should be registered");
    let batch = producer.extract(
      SANDBOX_WORLD_ID,
      world.tessera(),
      world.pleroma(),
      None,
      sim_time,
      frame,
    );

    next_tick += TICK_PERIOD;
    let now = Instant::now();
    if let Some(remaining) = next_tick.checked_duration_since(now) {
      std::thread::sleep(remaining);
    } else {
      next_tick = now;
    }

    Profiler::flush_local();
    Ok(Some(batch))
  });

  info!(
    "sandbox showcase: terrain + ocean + moist atmosphere. \
     Tab toggles debug/rendered; 1/2/3 swap atmosphere temp/humidity/pressure; \
     4/5 swap surface elevation/albedo"
  );
  App::new()
    .add_plugins(DefaultPlugins)
    .add_plugins(PanOrbitCameraPlugin)
    .add_plugins(AetherBevyPlugin::new(rx))
    .add_plugins(ShowcaseRenderPlugin)
    .add_systems(Startup, spawn_camera_and_light)
    .add_systems(Update, layer_toggle_input)
    .run();

  runner.shutdown_and_join()?;
  Profiler::print(&mut std::io::stdout());
  Ok(())
}

fn spawn_camera_and_light(mut commands: Commands) {
  let distance = 2.5e7_f32;
  commands.spawn((
    Camera3d::default(),
    Transform::from_xyz(distance, distance * 0.4, distance)
      .looking_at(Vec3::ZERO, Vec3::Y),
    PanOrbitCamera {
      focus: Vec3::ZERO,
      radius: Some(distance * 1.7),
      ..default()
    },
  ));

  commands.spawn((
    DirectionalLight {
      illuminance: 12_000.0,
      shadows_enabled: false,
      ..default()
    },
    Transform::from_xyz(1.0, 0.4, 0.0).looking_at(Vec3::ZERO, Vec3::Y),
    SunLight,
  ));
}

/// Rebind which field colours the atmosphere (1/2/3) or surface (4/5) mesh.
fn layer_toggle_input(
  keys: Res<ButtonInput<KeyCode>>,
  mut registry: ResMut<RenderRegistry>,
) {
  let mesh_handle = |mesh: MeshKey| {
    RenderMeshId {
      world: SANDBOX_WORLD_ID,
      mesh,
      representation: MeshRepresentation::BoundaryFaces,
    }
    .handle()
  };
  let atmosphere = mesh_handle(MeshKey::ATMOSPHERE);
  let surface = mesh_handle(MeshKey::SURFACE);
  let layer = |name: &'static str, mesh| {
    LayerHandle::for_target(LayerId::from_static(name), mesh)
  };

  let mut rebind = |mesh, handle, label: &str| {
    registry.bindings.insert(mesh, handle);
    registry.dirty_meshes.insert(mesh);
    info!("{label}");
  };

  if keys.just_pressed(KeyCode::Digit1) {
    rebind(
      atmosphere,
      layer("atmosphere_temperature", atmosphere),
      "atmosphere ← temperature",
    );
  }
  if keys.just_pressed(KeyCode::Digit2) {
    rebind(
      atmosphere,
      layer("atmosphere_humidity", atmosphere),
      "atmosphere ← humidity",
    );
  }
  if keys.just_pressed(KeyCode::Digit3) {
    rebind(
      atmosphere,
      layer("atmosphere_pressure", atmosphere),
      "atmosphere ← pressure",
    );
  }
  if keys.just_pressed(KeyCode::Digit4) {
    rebind(
      surface,
      layer("surface_elevation", surface),
      "surface ← elevation",
    );
  }
  if keys.just_pressed(KeyCode::Digit5) {
    rebind(
      surface,
      layer("surface_albedo", surface),
      "surface ← albedo",
    );
  }
}
