// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Drives a `DirectionalLight` from the simulation's sun-direction stream so
//! shading tracks the (orbiting) sun. The apply system records each
//! `UpdateSunDirection` into [`SunDirection`]; [`orient_sun_light_system`] then
//! aims every light tagged [`SunLight`] so it shines from the sun toward the
//! world origin.

use bevy::prelude::*;

/// Latest sun direction from the simulation — a unit vector from the world
/// centre toward the sun, in world coordinates. Updated by the apply system.
#[derive(Resource, Default)]
pub struct SunDirection {
  pub direction: Option<Vec3>,
}

/// Marker for a `DirectionalLight` whose orientation should track the
/// simulation sun.
#[derive(Component)]
pub struct SunLight;

/// Aim every [`SunLight`] directional light so its rays travel from the sun
/// toward the world origin (forward = −sun_direction). The sun orbits in the
/// equatorial (x-y) plane about the world +z pole, so +z is a stable "up"
/// reference that is never parallel to the in-plane sun direction.
pub fn orient_sun_light_system(
  sun: Res<SunDirection>,
  mut lights: Query<&mut Transform, With<SunLight>>,
) {
  let Some(direction) = sun.direction else {
    return;
  };
  if direction.length_squared() < 1e-12 {
    return;
  }
  let rotation = Transform::default()
    .looking_to(-direction.normalize(), Vec3::Z)
    .rotation;
  for mut transform in &mut lights {
    transform.rotation = rotation;
  }
}
