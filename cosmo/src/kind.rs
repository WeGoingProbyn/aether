// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use utility::{
  constants::{gravity, stellar_luminosity},
  maths::vector::Vector,
};

use crate::body::{Atmosphere, Body};

/// Stellar-specific properties. Mass and radius live on `CelestialBody::body`.
#[derive(Clone, Debug)]
pub struct Star {
  pub surface_temperature: f64, // K (photosphere)
  pub core_temperature: f64,    // K
}

/// Rocky body with optional thin atmosphere — Mercury through Mars.
#[derive(Clone, Debug)]
pub struct RockyBody {
  /// Mean surface temperature in K. Set from observation rather than
  /// equilibrium temperature to capture greenhouse / heat-redistribution
  /// effects (Venus is ~735 K vs equilibrium ~230 K).
  pub surface_temperature: f64,
  /// Surface pressure in Pa. Zero for airless bodies.
  pub surface_pressure: f64,
  /// Sidereal rotation rate in rad/s. Negative for retrograde rotators.
  pub angular_velocity: f64,
  /// Obliquity in radians.
  pub axial_tilt: f64,
  pub atmosphere: Option<Atmosphere>,
}

/// Gas giant — no solid surface, so reference values are at the 1-bar level
/// (the convention used for Jupiter / Saturn / Neptune in the literature).
#[derive(Clone, Debug)]
pub struct GasGiant {
  pub reference_temperature: f64, // K at 1-bar
  pub reference_pressure: f64,    // Pa (typically 1.0e5)
  pub angular_velocity: f64,
  pub axial_tilt: f64,
  /// Internal heat factor: ratio of total thermal output to absorbed solar.
  /// Jupiter ≈ 1.6, Saturn ≈ 2.3, Neptune ≈ 2.7.
  pub heat_factor: f64,
  pub atmosphere: Atmosphere,
}

#[derive(Clone, Debug)]
pub enum BodyKind {
  Star(Star),
  GasGiant(GasGiant),
  RockyBody(RockyBody),
}

#[derive(Clone, Debug)]
pub struct CelestialBody {
  body: Body,
  kind: BodyKind,
}

impl CelestialBody {
  pub fn new(
    mass: f64,
    radius: f64,
    position: Vector<f64, 3>,
    velocity: Vector<f64, 3>,
    kind: BodyKind,
  ) -> Self {
    Self {
      body: Body::new(mass, radius, position, velocity),
      kind,
    }
  }

  pub fn mass(&self) -> f64 {
    self.body.mass
  }
  pub fn radius(&self) -> f64 {
    self.body.radius
  }
  pub fn position(&self) -> &Vector<f64, 3> {
    &self.body.position
  }
  pub fn velocity(&self) -> &Vector<f64, 3> {
    &self.body.velocity
  }
  pub fn kind(&self) -> &BodyKind {
    &self.kind
  }
  pub fn kind_mut(&mut self) -> &mut BodyKind {
    &mut self.kind
  }

  /// Surface gravity = G·M/r². For gas giants, "surface" is the 1-bar level
  /// (which is what `radius` refers to for them).
  pub fn surface_gravity(&self) -> f64 {
    gravity(self.body.mass, self.body.radius)
  }

  /// Stefan–Boltzmann luminosity, only meaningful for stars. Returns `None`
  /// for planets — they have no intrinsic luminosity at this scale (gas
  /// giants do radiate internally via `heat_factor`, but that's a multiplier
  /// on absorbed solar, not a standalone luminosity).
  pub fn luminosity(&self) -> Option<f64> {
    match &self.kind {
      BodyKind::Star(s) => {
        Some(stellar_luminosity(self.body.radius, s.surface_temperature))
      }
      _ => None,
    }
  }
}
