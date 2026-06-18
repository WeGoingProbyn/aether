// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Day/night forcing. `DiurnalSunStep` advances the world's
//! `ResourceKey::SunPosition` direction once per tick by rotating it about
//! the world +z (spin) axis at the planet's angular velocity. The moving
//! sun gives the radiative transfer a travelling day side, which — coupled
//! with rotation (Coriolis) and the water cycle — is what makes weather
//! emerge rather than relax to a steady state.

use nexus::{ResourceKey, Stage, StageContext};
use utility::error::{AetherError, AetherResult, ErrorDomain};

/// Rotates the sun direction about world +z by `−angular_velocity · dt`
/// each tick (the sun appears to move westward as the planet spins east).
pub struct DiurnalSunStep {
  angular_velocity: f64,
  resource_reads: [ResourceKey; 1],
  resource_writes: [ResourceKey; 1],
}

impl DiurnalSunStep {
  pub fn new(angular_velocity: f64) -> Self {
    Self {
      angular_velocity,
      resource_reads: [ResourceKey::SunPosition],
      resource_writes: [ResourceKey::SunPosition],
    }
  }

  pub fn angular_velocity(&self) -> f64 {
    self.angular_velocity
  }
}

impl Stage for DiurnalSunStep {
  fn name(&self) -> &'static str {
    "lumen_diurnal_sun"
  }

  fn reads(&self) -> &[nexus::FieldKey] {
    &[]
  }

  fn writes(&self) -> &[nexus::FieldKey] {
    &[]
  }

  fn resource_reads(&self) -> &[ResourceKey] {
    &self.resource_reads
  }

  fn resource_writes(&self) -> &[ResourceKey] {
    &self.resource_writes
  }

  fn run(&mut self, mut ctx: StageContext<'_>) -> AetherResult<()> {
    let dt = ctx.world.dt;
    if !dt.is_finite() || dt <= 0.0 {
      return Err(
        AetherError::new(DiurnalError::InvalidTimeStep)
          .context(format!("dt {}", dt)),
      );
    }
    let theta = -self.angular_velocity * dt;
    let (sin, cos) = theta.sin_cos();

    let sun = ctx
      .world
      .fields
      .resource_mut::<[f64; 3]>(ResourceKey::SunPosition)
      .ok_or_else(|| {
        AetherError::new(DiurnalError::MissingResource)
          .context("ResourceKey::SunPosition")
      })?;

    let [x, y, z] = *sun;
    *sun = [x * cos - y * sin, x * sin + y * cos, z];
    Ok(())
  }
}

#[derive(Debug)]
pub enum DiurnalError {
  MissingResource,
  InvalidTimeStep,
}

impl ErrorDomain for DiurnalError {
  fn domain(&self) -> &str {
    "lumen_diurnal"
  }
}

impl std::fmt::Display for DiurnalError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      DiurnalError::MissingResource => {
        write!(f, "sun position resource is not registered")
      }
      DiurnalError::InvalidTimeStep => {
        write!(f, "dt must be finite and positive")
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use std::f64::consts::PI;

  use nexus::{Nexus, Pleroma, WorldConstants, WorldId};
  use tessera::world_mesh::Tessera;
  use utility::thread::pool::Pool;

  use super::*;

  #[test]
  fn sun_rotates_about_z_each_tick() {
    let mut pleroma = Pleroma::new();
    pleroma.register_resource(ResourceKey::SunPosition, [1.0, 0.0, 0.0]);

    // ω·dt = π/2 → a quarter turn: (1,0,0) → (0,-1,0) for θ = −π/2.
    let omega = PI / 2.0;
    let mut nexus = Nexus::new();
    nexus.add(DiurnalSunStep::new(omega));
    let mut compiled = nexus.build(&pleroma).unwrap();
    compiled
      .tick(
        WorldId(0),
        &Tessera::default(),
        &WorldConstants::default(),
        &mut pleroma,
        &Pool::default(),
        1.0,
      )
      .unwrap();

    let sun = pleroma
      .read_resource::<[f64; 3]>(ResourceKey::SunPosition)
      .unwrap();
    assert!((sun[0] - 0.0).abs() < 1e-9, "x={}", sun[0]);
    assert!((sun[1] + 1.0).abs() < 1e-9, "y={}", sun[1]);
    assert_eq!(sun[2], 0.0);
  }
}
