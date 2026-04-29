// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use nexus::WorldConstants;
use tessera::cube_sphere::CubeSphereShellSpec;
use utility::{
  domain::BoundaryTag,
  error::{AetherError, AetherResult},
};

use crate::error::AerError;

/// Physical shell radii for a surface-atmosphere validation mesh.
///
/// Aer does not own meshes; Tessera still does. This type only turns neutral
/// world constants plus caller-chosen shell thicknesses into cube-sphere shell
/// specs with consistent length units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AtmosphereShellLayout {
  surface_inner_radius: f64,
  surface_outer_radius: f64,
  atmosphere_inner_radius: f64,
  atmosphere_outer_radius: f64,
}

impl AtmosphereShellLayout {
  pub fn new(
    constants: &WorldConstants,
    atmosphere_height: f64,
    surface_depth: f64,
  ) -> AetherResult<Self> {
    let radius = constants.radius;
    if !radius.is_finite() || radius <= 0.0 {
      return Err(
        AetherError::new(AerError::InvalidAtmosphereConstants)
          .context(format!("world radius {}", radius)),
      );
    }
    if !atmosphere_height.is_finite() || atmosphere_height <= 0.0 {
      return Err(
        AetherError::new(AerError::InvalidAtmosphereConstants)
          .context(format!("atmosphere height {}", atmosphere_height)),
      );
    }
    if !surface_depth.is_finite()
      || surface_depth <= 0.0
      || surface_depth >= radius
    {
      return Err(
        AetherError::new(AerError::InvalidAtmosphereConstants)
          .context(format!("surface depth {}", surface_depth)),
      );
    }

    Ok(Self {
      surface_inner_radius: radius - surface_depth,
      surface_outer_radius: radius,
      atmosphere_inner_radius: radius,
      atmosphere_outer_radius: radius + atmosphere_height,
    })
  }

  pub fn surface_inner_radius(&self) -> f64 {
    self.surface_inner_radius
  }

  pub fn surface_outer_radius(&self) -> f64 {
    self.surface_outer_radius
  }

  pub fn atmosphere_inner_radius(&self) -> f64 {
    self.atmosphere_inner_radius
  }

  pub fn atmosphere_outer_radius(&self) -> f64 {
    self.atmosphere_outer_radius
  }

  pub fn reference_radius(&self) -> f64 {
    self.atmosphere_inner_radius
  }

  pub fn atmosphere_height(&self) -> f64 {
    self.atmosphere_outer_radius - self.atmosphere_inner_radius
  }

  pub fn surface_depth(&self) -> f64 {
    self.surface_outer_radius - self.surface_inner_radius
  }

  pub fn surface_shell_spec(
    &self,
    angular_dims: [usize; 2],
    radial_layers: usize,
  ) -> CubeSphereShellSpec {
    CubeSphereShellSpec::uniform(
      [angular_dims[0], angular_dims[1], radial_layers],
      self.surface_inner_radius,
      self.surface_outer_radius,
    )
    .with_boundaries(BoundaryTag::Ground, BoundaryTag::AtmosphereEdge)
  }

  pub fn atmosphere_shell_spec(
    &self,
    angular_dims: [usize; 2],
    radial_layers: usize,
  ) -> CubeSphereShellSpec {
    CubeSphereShellSpec::uniform(
      [angular_dims[0], angular_dims[1], radial_layers],
      self.atmosphere_inner_radius,
      self.atmosphere_outer_radius,
    )
    .with_boundaries(BoundaryTag::Ground, BoundaryTag::AtmosphereEdge)
  }
}

#[cfg(test)]
mod tests {
  use nexus::WorldConstants;

  use super::*;

  #[test]
  fn shell_layout_uses_world_radius_units() {
    let constants = WorldConstants {
      radius: 6_371_000.0,
      ..WorldConstants::default()
    };

    let layout =
      AtmosphereShellLayout::new(&constants, 20_000.0, 10_000.0).unwrap();
    assert_eq!(layout.surface_inner_radius(), 6_361_000.0);
    assert_eq!(layout.surface_outer_radius(), 6_371_000.0);
    assert_eq!(layout.atmosphere_inner_radius(), 6_371_000.0);
    assert_eq!(layout.atmosphere_outer_radius(), 6_391_000.0);

    let spec = layout.atmosphere_shell_spec([4, 4], 8);
    assert_eq!(spec.radial_edges[0], 6_371_000.0);
    assert_eq!(*spec.radial_edges.last().unwrap(), 6_391_000.0);
  }

  #[test]
  fn shell_layout_rejects_normalized_or_invalid_radius() {
    let constants = WorldConstants {
      radius: 0.0,
      ..WorldConstants::default()
    };
    assert!(
      AtmosphereShellLayout::new(&constants, 20_000.0, 10_000.0).is_err()
    );
  }
}
