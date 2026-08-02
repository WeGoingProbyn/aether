// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Helper for spawning a default `bevy_panorbit_camera` looking at
//! the planet. The plugin itself isn't added here so callers can
//! customise it; just call `App::add_plugins(PanOrbitCameraPlugin)`
//! alongside `AetherBevyPlugin`.
//!
//! **The camera is the host's, not the simulation's.** eidolon presents state
//! the simulation produced; it does not carry the host's view back to the
//! host's renderer. A host driving view-dependent LOD writes a
//! [`RefinementFocus`](utility::domain::RefinementFocus) into the world *and*
//! positions its own camera directly, at render rate — so the view never waits
//! on a sim tick.

use bevy::prelude::*;
use bevy_panorbit_camera::PanOrbitCamera;

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
