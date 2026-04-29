// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

/// Immutable per-world constants exposed to physics stages.
///
/// This intentionally contains plain values, not `cosmo` types. `aether`
/// derives it from a `cosmo::CelestialBody` during world setup, while physics
/// crates consume only this neutral view.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct WorldConstants {
  pub mass: f64,
  pub radius: f64,
  pub surface_gravity: f64,
  pub atmosphere: Option<AtmosphereConstants>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AtmosphereConstants {
  pub reference_temperature: f64,
  pub reference_pressure: f64,
  pub gamma: f64,
  pub gas_constant: f64,
  pub molar_mass: f64,
  pub albedo: Option<f64>,
  pub angular_velocity: f64,
  pub axial_tilt: f64,
}
