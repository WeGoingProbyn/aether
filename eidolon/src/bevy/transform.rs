// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Eidolon `Transform` (f64) → bevy `Transform` (f32).
//!
//! The cast is concentrated here so the f64→f32 jump for planet-scale
//! coordinates lives in exactly one place. Geometry produced by the
//! extract layer is local-space (vertices ÷ world scale), so the
//! transform's `centre` is what carries the absolute position; bevy's
//! own `GlobalTransform` re-centres correctly.

use bevy::math::{Quat, Vec3};
use bevy::prelude::Transform as BevyTransform;

use crate::ir::Transform as EidolonTransform;

pub fn to_bevy_transform(t: &EidolonTransform) -> BevyTransform {
  BevyTransform {
    translation: Vec3::new(
      t.centre[0] as f32,
      t.centre[1] as f32,
      t.centre[2] as f32,
    ),
    rotation: Quat::from_xyzw(
      t.orientation[1] as f32,
      t.orientation[2] as f32,
      t.orientation[3] as f32,
      t.orientation[0] as f32,
    )
    .normalize(),
    scale: Vec3::splat(t.scale as f32),
  }
}
