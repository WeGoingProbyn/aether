// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Per-resource similarity transform used to bridge f64 simulation
//! coordinates to whatever an engine wants on its GPU.
//!
//! Geometry inside a `RenderMesh` is expected to live in *local* space
//! (~unit-sized when normalised by `scale`). The mesh's `transform`
//! tells the backend how to place that local space into world space:
//!
//! `world_pos = centre + scale · rotate(orientation, local_pos)`
//!
//! Backends typically apply this on a parent transform/entity, so the
//! mesh's local-space geometry can stay in f32 without losing
//! precision at planet-radius scales.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform {
  /// Absolute centre in simulation/world coordinates (metres for our
  /// physics).
  pub centre: [f64; 3],
  /// Unit quaternion `(w, x, y, z)`. Identity is `(1, 0, 0, 0)`.
  pub orientation: [f64; 4],
  /// Uniform scale factor applied after rotation. `1.0` means
  /// local-space geometry is already in metres.
  pub scale: f64,
}

impl Transform {
  pub const IDENTITY: Self = Self {
    centre: [0.0, 0.0, 0.0],
    orientation: [1.0, 0.0, 0.0, 0.0],
    scale: 1.0,
  };

  /// Pure translation (no rotation, unit scale).
  pub const fn translation(centre: [f64; 3]) -> Self {
    Self {
      centre,
      orientation: [1.0, 0.0, 0.0, 0.0],
      scale: 1.0,
    }
  }

  /// Pure scale at the origin.
  pub const fn scaling(scale: f64) -> Self {
    Self {
      centre: [0.0, 0.0, 0.0],
      orientation: [1.0, 0.0, 0.0, 0.0],
      scale,
    }
  }

  /// Translation + scale, identity rotation. Common case for placing a
  /// planet shell at its body position with vertices stored in unit
  /// space.
  pub const fn translation_scaling(centre: [f64; 3], scale: f64) -> Self {
    Self {
      centre,
      orientation: [1.0, 0.0, 0.0, 0.0],
      scale,
    }
  }
}

impl Default for Transform {
  fn default() -> Self {
    Self::IDENTITY
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn identity_is_identity() {
    let t = Transform::IDENTITY;
    assert_eq!(t.centre, [0.0; 3]);
    assert_eq!(t.orientation, [1.0, 0.0, 0.0, 0.0]);
    assert_eq!(t.scale, 1.0);
    assert_eq!(t, Transform::default());
  }

  #[test]
  fn translation_carries_centre_only() {
    let t = Transform::translation([1.0, 2.0, 3.0]);
    assert_eq!(t.centre, [1.0, 2.0, 3.0]);
    assert_eq!(t.orientation, Transform::IDENTITY.orientation);
    assert_eq!(t.scale, 1.0);
  }

  #[test]
  fn translation_scaling_combines_both() {
    let t = Transform::translation_scaling([10.0, 0.0, 0.0], 2.5);
    assert_eq!(t.centre, [10.0, 0.0, 0.0]);
    assert_eq!(t.scale, 2.5);
  }
}
