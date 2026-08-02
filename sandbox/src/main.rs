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
//!   Tab        — toggle debug field views / rendered look (scattering +
//!                land/ocean/ice surfaces + terrain relief)
//!   H          — hide / show the atmosphere shell (it occludes the inner
//!                shells in the opaque debug view)
//!   1 / 2 / 3  — atmosphere overlay shows temperature / humidity / pressure
//!   4 / 5      — surface shows elevation / albedo (debug fields)
//!   G          — toggle the AMR cell-outline wireframe (the grid densifies
//!                where the surface mesh is adaptively refined)
//!   6 / 7 / 8  — atmosphere overlay shows the climatology (slowly-varying
//!                time-mean) of temperature / humidity / pressure; compare
//!                against 1 / 2 / 3 to see the aggregate smooth the weather
//!   C          — toggle live ↔ climatology time-advance regime: live
//!                integrates every game second; climatology bursts one stable
//!                step then holds, racing the game clock ahead of sim time
//!                (see the runner's periodic regime log)

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use aether::adapt::RefinementFocus;
use bevy::prelude::*;
use chronos::{Regime, RegimeConfig};

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
use utility::domain::{MeshKey, SystemId};
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

/// Radial scale applied to the cell-outline wireframe so it floats just above the
/// terrain the reference renderer displaces outward (~6% of the radius at the
/// showcase's 200× exaggeration) instead of being buried under the raised land.
const OUTLINE_LIFT: f32 = 1.08;

/// Game time advanced per frame in the climatology regime. Far larger than the
/// burst the solver actually integrates (`TICK_DT`), so each frame covers a big
/// span of game time for the cost of a single stable atmosphere step — the
/// climatology stays current while the game clock races ahead of sim time.
const CLIMATE_GAME_DT: f64 = 3600.0;

/// Shared live↔climatology regime flag. The bevy `C`-key system flips it; the
/// runner thread reads it each tick to pick the regime and game step.
#[derive(Resource, Clone)]
struct RegimeToggle(Arc<AtomicBool>);

fn main() -> AetherResult<()> {
  Logger::init(
    vec![Box::new(StdSink::new(std::io::stdout()).capacity(1))],
    Level::Trace,
  );
  Profiler::init();

  let (mut aether, layout) = build_showcase_world()?;
  let mut producer = FrameProducer::new(showcase_extract_config());

  // The **host** owns the camera. `camera_control_system` (WASD orbit, Q/E zoom)
  // moves it and writes the camera transform directly, every render frame, so the
  // view is never gated on a sim tick. It also publishes the eye into this shared
  // cell; the runner hands that to the world as a `RefinementFocus`, and the LOD
  // criterion refines the surface toward it. Two independent consumers of one
  // host-owned value — the view does not round-trip through the simulation.
  let camera_radius = layout.reference_radius() * 2.5;
  let camera_eye = Arc::new(Mutex::new(orbit_eye(0.0, 0.2, camera_radius)));
  let camera_eye_runner = Arc::clone(&camera_eye);

  // One stable atmosphere step per climatology burst, so entering the regime
  // never takes an unstable jump — the held span is what makes it cheap.
  aether
    .system_mut(SystemId(0))
    .and_then(|s| s.world_mut(SANDBOX_WORLD_ID))
    .expect("sandbox world should be registered")
    .set_regime_config(RegimeConfig::new(1, TICK_DT));

  let (tx, rx) = render_channel(64);

  // Live by default; the `C` key flips this and the runner reacts next tick.
  let climatology = Arc::new(AtomicBool::new(false));
  let climatology_runner = Arc::clone(&climatology);

  let mut frame: u64 = 0;
  let mut next_tick = Instant::now();
  let runner = spawn_runner(tx, move |_shutdown| {
    let climatology_mode = climatology_runner.load(Ordering::Relaxed);
    {
      let world = aether
        .system_mut(SystemId(0))
        .and_then(|s| s.world_mut(SANDBOX_WORLD_ID))
        .expect("sandbox world should be registered");
      world.set_regime(if climatology_mode {
        Regime::Climatology
      } else {
        Regime::Live
      });
      // Feed the host-controlled camera (updated by `camera_control_system`) into
      // the world: the LOD criterion refines toward it and eidolon presents it.
      let eye = *camera_eye_runner.lock().expect("camera eye lock");
      world.set_refinement_focus(RefinementFocus { position: eye });
    }
    // Live integrates the full game step; climatology bursts one stable step and
    // holds the rest, so the game clock races ahead of integrated sim time.
    let game_dt = if climatology_mode {
      CLIMATE_GAME_DT
    } else {
      TICK_DT
    };
    aether.advance(game_dt)?;
    frame = frame.wrapping_add(1);
    let world = aether
      .world(SANDBOX_WORLD_ID)
      .expect("sandbox world should be registered");
    // Timestamp the frame by integrated sim time, which matches the state the
    // solver actually produced (in climatology holds the fields are frozen).
    let sim_time = world.sim_time();
    if frame % 120 == 0 {
      info!(
        "regime {:?}: game {:.0}s / sim {:.0}s",
        world.regime(),
        world.game_clock(),
        sim_time
      );
      // Runtime health: per-field non-finite count, conservation drift, and the
      // volume-integrated conserved totals published by the in-DAG monitor.
      if let Some(diagnostics) = world.diagnostics() {
        for (field, report) in &diagnostics.fields {
          let conserved: Vec<String> = report
            .conserved
            .iter()
            .map(|(name, total)| format!("{name}={total:.3e}"))
            .collect();
          info!(
            "diagnostics {:?} [{:?}]: {} non-finite, drift {:.2e} | {}",
            field,
            diagnostics.policy,
            report.non_finite_cells,
            report.max_relative_drift,
            conserved.join(" ")
          );
        }
      }
    }
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
     WASD orbits the camera, Q/E zooms (surface LOD follows the view); \
     Tab toggles debug/rendered; 1/2/3 swap atmosphere temp/humidity/pressure; \
     4/5 swap surface elevation/albedo; \
     6/7/8 swap atmosphere climatology mean temp/humidity/pressure; \
     G toggles the AMR cell-outline wireframe; \
     C toggles live ↔ climatology regime (burst-then-hold)"
  );
  App::new()
    .add_plugins(DefaultPlugins)
    .add_plugins(AetherBevyPlugin::new(rx))
    .add_plugins(ShowcaseRenderPlugin)
    .insert_resource(RegimeToggle(climatology))
    .insert_resource(OrbitCamera {
      azimuth: 0.0,
      colatitude: 0.2,
      radius: camera_radius,
      min_radius: layout.reference_radius() * 1.1,
      eye: camera_eye,
    })
    .add_systems(Startup, spawn_camera_and_light)
    .add_systems(
      Update,
      (
        layer_toggle_input,
        regime_toggle_input,
        outline_view_input,
        camera_control_system,
      ),
    )
    .run();

  runner.shutdown_and_join()?;
  Profiler::print(&mut std::io::stdout());
  Ok(())
}

/// Host-side orbit-camera state. The host owns the camera outright: input updates
/// this controller, which both positions the bevy camera directly (at render
/// rate) and publishes the eye into the shared cell the runner feeds to
/// `World::set_refinement_focus` as an LOD region of interest.
#[derive(Resource)]
struct OrbitCamera {
  azimuth: f64,
  colatitude: f64,
  radius: f64,
  min_radius: f64,
  eye: Arc<Mutex<[f64; 3]>>,
}

/// World-space eye position for an orbit looking at the planet centre, at polar
/// angle `colatitude` from +z and `azimuth` about it.
fn orbit_eye(azimuth: f64, colatitude: f64, radius: f64) -> [f64; 3] {
  let (sc, cc) = colatitude.sin_cos();
  let (sa, ca) = azimuth.sin_cos();
  [radius * sc * ca, radius * sc * sa, radius * cc]
}

/// WASD orbits the view (A/D azimuth, W/S colatitude), Q/E zoom. Updates the
/// orbit state, moves the camera immediately, and publishes the eye for the
/// runner to feed to `set_refinement_focus`. Clamped off the poles and above the
/// surface.
///
/// The camera is written here, on the render thread, from this frame's input —
/// so view motion is smooth regardless of what the simulation costs per tick.
fn camera_control_system(
  time: Res<Time>,
  keys: Res<ButtonInput<KeyCode>>,
  mut orbit: ResMut<OrbitCamera>,
  mut cameras: Query<&mut Transform, With<Camera3d>>,
) {
  let dt = time.delta_secs() as f64;
  let orbit_speed = 1.2; // rad/s
  let zoom_rate = 1.5; // fraction of the radius per second
  if keys.pressed(KeyCode::KeyA) {
    orbit.azimuth -= orbit_speed * dt;
  }
  if keys.pressed(KeyCode::KeyD) {
    orbit.azimuth += orbit_speed * dt;
  }
  if keys.pressed(KeyCode::KeyW) {
    orbit.colatitude -= orbit_speed * dt;
  }
  if keys.pressed(KeyCode::KeyS) {
    orbit.colatitude += orbit_speed * dt;
  }
  if keys.pressed(KeyCode::KeyQ) {
    orbit.radius *= 1.0 - zoom_rate * dt;
  }
  if keys.pressed(KeyCode::KeyE) {
    orbit.radius *= 1.0 + zoom_rate * dt;
  }
  orbit.colatitude = orbit.colatitude.clamp(0.05, std::f64::consts::PI - 0.05);
  orbit.radius = orbit.radius.max(orbit.min_radius);

  let eye = orbit_eye(orbit.azimuth, orbit.colatitude, orbit.radius);
  // Present: drive the camera now, this frame.
  let goal = Transform::from_translation(Vec3::new(
    eye[0] as f32,
    eye[1] as f32,
    eye[2] as f32,
  ))
  .looking_at(Vec3::ZERO, Vec3::Z);
  for mut transform in &mut cameras {
    *transform = goal;
  }
  // Simulate: publish the same eye as the LOD region of interest. Latest-value,
  // so the sim reading it at its own (slower) cadence is fine.
  if let Ok(mut shared) = orbit.eye.lock() {
    *shared = eye;
  }
}

fn spawn_camera_and_light(mut commands: Commands) {
  let distance = 2.5e7_f32;
  // Host-owned: `camera_control_system` writes this transform every frame. The
  // initial pose is a placeholder until the first input frame runs.
  commands.spawn((
    Camera3d::default(),
    Transform::from_xyz(distance, distance * 0.4, distance)
      .looking_at(Vec3::ZERO, Vec3::Z),
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

/// Rebind which field colours the atmosphere (1/2/3) or surface (4/5) mesh, and
/// hide/show the atmosphere shell (H) so it stops occluding the inner shells in
/// the opaque debug view.
fn layer_toggle_input(
  keys: Res<ButtonInput<KeyCode>>,
  mut registry: ResMut<RenderRegistry>,
  mut commands: Commands,
  mut atmosphere_hidden: Local<bool>,
) {
  if keys.just_pressed(KeyCode::KeyH) {
    *atmosphere_hidden = !*atmosphere_hidden;
    let handle = RenderMeshId {
      world: SANDBOX_WORLD_ID,
      mesh: MeshKey::ATMOSPHERE,
      representation: MeshRepresentation::BoundaryFaces,
    }
    .handle();
    if let Some(entry) = registry.meshes.get(&handle) {
      let visibility = if *atmosphere_hidden {
        Visibility::Hidden
      } else {
        Visibility::Inherited
      };
      commands.entity(entry.entity).insert(visibility);
      info!(
        "atmosphere {}",
        if *atmosphere_hidden {
          "hidden"
        } else {
          "shown"
        }
      );
    }
  }

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
  // Climatology (slowly-varying time-means) of the atmosphere primitives —
  // compare against the live fields on 1/2/3 to see the aggregate smooth out
  // the instantaneous weather.
  if keys.just_pressed(KeyCode::Digit6) {
    rebind(
      atmosphere,
      layer("atmosphere_mean_temperature", atmosphere),
      "atmosphere ← mean temperature (climatology)",
    );
  }
  if keys.just_pressed(KeyCode::Digit7) {
    rebind(
      atmosphere,
      layer("atmosphere_mean_humidity", atmosphere),
      "atmosphere ← mean humidity (climatology)",
    );
  }
  if keys.just_pressed(KeyCode::Digit8) {
    rebind(
      atmosphere,
      layer("atmosphere_mean_pressure", atmosphere),
      "atmosphere ← mean pressure (climatology)",
    );
  }
}

/// The AMR debug view (G): the surface cell-outline wireframe. It is lifted just
/// above the displaced terrain each frame (idempotent, so it survives the mesh
/// being rebuilt when AMR refines), and `G` toggles its visibility. Where the
/// surface mesh is refined, this grid visibly densifies — that is where AMR is
/// being applied.
fn outline_view_input(
  keys: Res<ButtonInput<KeyCode>>,
  registry: Res<RenderRegistry>,
  mut commands: Commands,
  mut hidden: Local<bool>,
) {
  if keys.just_pressed(KeyCode::KeyG) {
    *hidden = !*hidden;
    info!("cell outlines {}", if *hidden { "hidden" } else { "shown" });
  }

  let handle = RenderMeshId {
    world: SANDBOX_WORLD_ID,
    mesh: MeshKey::SURFACE,
    representation: MeshRepresentation::Wireframe,
  }
  .handle();
  if let Some(entry) = registry.meshes.get(&handle) {
    let visibility = if *hidden {
      Visibility::Hidden
    } else {
      Visibility::Inherited
    };
    // Re-applied every frame so a re-meshed (refined) wireframe keeps its lift
    // and visibility even if the backend rebuilds the entity.
    commands
      .entity(entry.entity)
      .insert((Transform::from_scale(Vec3::splat(OUTLINE_LIFT)), visibility));
  }
}

/// Toggle the world's time-advance regime (C). In live mode the solver
/// integrates the full game step; in climatology mode it bursts one stable step
/// then holds, so game time races ahead of integrated sim time (watch the
/// runner's periodic `regime …: game … / sim …` log) while the climatology
/// means (6/7/8) stay current.
fn regime_toggle_input(
  keys: Res<ButtonInput<KeyCode>>,
  toggle: Res<RegimeToggle>,
) {
  if keys.just_pressed(KeyCode::KeyC) {
    let now = !toggle.0.fetch_xor(true, Ordering::Relaxed);
    info!(
      "regime → {}",
      if now {
        "climatology (burst-then-hold)"
      } else {
        "live"
      }
    );
  }
}
