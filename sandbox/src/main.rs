// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Bevy demo for the eidolon Update protocol. Runs the full
//! Earth+Sun+atmosphere stack on a background thread, streams
//! batches into a bounded SPSC channel, and renders the result with
//! a pan-orbit camera. Press `1`/`2`/`3` to swap which scalar
//! colours the surface mesh.

use std::time::{Duration, Instant};

use bevy::prelude::*;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};

use eidolon::{
  bevy::{AetherBevyPlugin, RenderRegistry},
  extract::FrameProducer,
  ir::{LayerHandle, LayerId, MeshRepresentation, RenderMeshId},
  runtime::{render_channel, spawn_runner},
};
use sandbox::{SANDBOX_WORLD_ID, build_demo_aether, demo_extract_config};
use utility::error::AetherResult;
use utility::logger::{Level, Logger, StdSink};
use utility::profiler::Profiler;
use utility::{domain::MeshKey, info};

/// Simulation timestep, in simulation seconds.
const TICK_DT: f64 = 0.05;

/// Wall-clock pacing for the runner thread. 60 Hz matches typical
/// monitor refresh and means the producer extracts at the same rate
/// the renderer consumes — no point producing faster than the bevy
/// main thread can drain.
const TICK_PERIOD: Duration = Duration::from_micros(16_667);

fn main() -> AetherResult<()> {
  Logger::init(
    vec![Box::new(StdSink::new(std::io::stdout()).capacity(1))],
    Level::Trace,
  );
  Profiler::init();

  let (mut aether, _layout) = build_demo_aether()?;
  let mut producer = FrameProducer::new(demo_extract_config());

  let (tx, rx) = render_channel(64);

  // Worker thread: tick the simulation, extract a batch, send it.
  // Wall-clock paced to TICK_PERIOD; without this the loop spins at
  // hundreds of kHz and starves the bevy main thread of CPU.
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
      // We fell behind (heavy tick or paused thread). Resync the
      // schedule to "now" so we don't burn CPU trying to catch up.
      next_tick = now;
    }

    Profiler::flush_local();
    Ok(Some(batch))
  });

  info!("sandbox: starting bevy app — keys 1/2/3 swap surface scalar");
  App::new()
    .add_plugins(DefaultPlugins)
    .add_plugins(PanOrbitCameraPlugin)
    .add_plugins(AetherBevyPlugin::new(rx))
    .add_systems(Startup, spawn_camera_and_light)
    .add_systems(Update, layer_toggle_input)
    .run();

  // Bevy returns when the window is closed.
  runner.shutdown_and_join()?;

  Profiler::print(&mut std::io::stdout());
  Ok(())
}

fn spawn_camera_and_light(mut commands: Commands) {
  // Earth radius is ~6.371e6 m; the world transform stays at the
  // origin since the producer is configured with `world_scale = 1.0`.
  // We frame the camera ~3 radii out so the planet fits comfortably.
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
  ));
}

/// On `1`/`2`/`3`, rebind the surface mesh to the temperature /
/// atmosphere-temperature / pressure scalar. Quick UX for the demo.
fn layer_toggle_input(
  keys: Res<ButtonInput<KeyCode>>,
  mut registry: ResMut<RenderRegistry>,
) {
  let surface_mesh = RenderMeshId {
    world: SANDBOX_WORLD_ID,
    mesh: MeshKey::SURFACE,
    representation: MeshRepresentation::BoundaryFaces,
  }
  .handle();
  let atmosphere_mesh = RenderMeshId {
    world: SANDBOX_WORLD_ID,
    mesh: MeshKey::ATMOSPHERE,
    representation: MeshRepresentation::BoundaryFaces,
  }
  .handle();

  let surface_temp = LayerHandle::for_target(
    LayerId::from_static("surface_temperature"),
    surface_mesh,
  );
  let atmosphere_temp = LayerHandle::for_target(
    LayerId::from_static("atmosphere_temperature"),
    atmosphere_mesh,
  );
  let atmosphere_pressure = LayerHandle::for_target(
    LayerId::from_static("atmosphere_pressure"),
    atmosphere_mesh,
  );

  if keys.just_pressed(KeyCode::Digit1) {
    registry.bindings.insert(surface_mesh, surface_temp);
    registry.dirty_meshes.insert(surface_mesh);
    info!("surface ← surface_temperature");
  }
  if keys.just_pressed(KeyCode::Digit2) {
    registry.bindings.insert(atmosphere_mesh, atmosphere_temp);
    registry.dirty_meshes.insert(atmosphere_mesh);
    info!("atmosphere ← atmosphere_temperature");
  }
  if keys.just_pressed(KeyCode::Digit3) {
    registry
      .bindings
      .insert(atmosphere_mesh, atmosphere_pressure);
    registry.dirty_meshes.insert(atmosphere_mesh);
    info!("atmosphere ← atmosphere_pressure");
  }
}
