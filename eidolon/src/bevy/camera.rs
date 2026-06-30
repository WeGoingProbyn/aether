// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Helper for spawning a default `bevy_panorbit_camera` looking at
//! the planet. The plugin itself isn't added here so callers can
//! customise it; just call `App::add_plugins(PanOrbitCameraPlugin)`
//! alongside `AetherBevyPlugin`.

use bevy::prelude::*;
use bevy_panorbit_camera::PanOrbitCamera;

use crate::ir::RenderCamera;

/// Spawn a default orbit camera at `distance` from the origin.
///
/// The camera is in world space; combine with a parented sun marker
/// or a focused mesh to keep the framing useful as the planet moves.
pub fn spawn_orbit_camera(commands: &mut Commands, distance: f32) {
  commands.spawn((
    Camera3d::default(),
    Transform::from_translation(Vec3::new(0.0, 0.0, distance)),
    PanOrbitCamera {
      focus: Vec3::ZERO,
      radius: Some(distance),
      ..default()
    },
  ));
}

/// Latest simulation-owned view from the `SetCamera` update stream. When the
/// simulation owns the camera (e.g. view-dependent LOD), the apply system records
/// each `Update::SetCamera` here and [`position_camera_from_view_system`] drives
/// any [`SimDrivenCamera`] from it. `None` ⇒ the backend's own camera is in
/// control (the forward-only contract: eidolon presents the view, never replies).
#[derive(Resource, Default)]
pub struct SimCamera {
  pub view: Option<RenderCamera>,
}

/// Marker for a `Camera3d` whose transform should follow the simulation-owned
/// view in [`SimCamera`]. Put this on the camera entity *instead of* an
/// interactive controller (e.g. `PanOrbitCamera`) when the sim owns the camera.
#[derive(Component)]
pub struct SimDrivenCamera;

/// How quickly a [`SimDrivenCamera`] chases the simulation's view (1/seconds).
/// The camera closes ~`rate·dt` of the remaining gap each render frame, so the
/// view stays smooth *at render rate* even though the simulation only updates the
/// camera at its (slower, possibly jittery) tick rate — the camera is sim state
/// presented forward and interpolated, exactly like the mesh.
const CAMERA_FOLLOW_RATE: f32 = 6.0;

/// Smoothly move every [`SimDrivenCamera`] toward the latest simulation-owned
/// view. No-op until the simulation has emitted a camera, so a backend with its
/// own camera is unaffected. On the first update it snaps (no prior pose to
/// interpolate from); afterwards it eases, decoupling the *visible* motion from
/// the sim tick cadence.
pub fn position_camera_from_view_system(
  time: Res<Time>,
  sim: Res<SimCamera>,
  mut cameras: Query<&mut Transform, With<SimDrivenCamera>>,
) {
  let Some(view) = sim.view else {
    return;
  };
  let eye = Vec3::new(
    view.position[0] as f32,
    view.position[1] as f32,
    view.position[2] as f32,
  );
  let target = Vec3::new(
    view.target[0] as f32,
    view.target[1] as f32,
    view.target[2] as f32,
  );
  let up = Vec3::new(view.up[0] as f32, view.up[1] as f32, view.up[2] as f32);
  let goal = Transform::from_translation(eye).looking_at(target, up);

  // Frame-rate-independent exponential smoothing toward the goal.
  let alpha = 1.0 - (-CAMERA_FOLLOW_RATE * time.delta_secs()).exp();
  for mut transform in &mut cameras {
    // Snap on the first frame (an identity/placeholder transform would otherwise
    // make the camera sweep in from the origin).
    if transform.translation == Vec3::ZERO {
      *transform = goal;
    } else {
      transform.translation =
        transform.translation.lerp(goal.translation, alpha);
      transform.rotation = transform.rotation.slerp(goal.rotation, alpha);
    }
  }
}
