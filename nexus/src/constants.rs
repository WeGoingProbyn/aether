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
  pub radiation: Option<RadiationConstants>,
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

/// Radiation-related physical constants for a world. Derived by `aether`
/// from cosmo's `CelestialBody` + the system's primary star, then
/// consumed by `lumen` to size shortwave/longwave terms. None when the
/// world has no resolvable primary (rogue planet, isolated test).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RadiationConstants {
  /// Top-of-atmosphere solar irradiance at the planet's orbit (W/m²).
  pub solar_irradiance: f64,
  /// Bond / surface short-wave albedo (0..1).
  pub surface_albedo: f64,
  /// Surface long-wave emissivity (0..1).
  pub surface_emissivity: f64,
}
