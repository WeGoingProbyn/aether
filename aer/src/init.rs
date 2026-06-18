// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use continuum::model::MoistEuler3D;
use nexus::{SoaField, WorldConstants};
use tessera::mesh::Mesh;
use utility::domain::CellId;
use utility::error::{AetherError, AetherResult};

use crate::error::AerError;

/// Atmosphere setup values consumed by Aer.
///
/// This is built from `nexus::WorldConstants`, not from `cosmo`, so Aer stays
/// independent of seed/catalogue types.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AtmosphereSpec {
  reference_temperature: f64,
  reference_pressure: f64,
  surface_gravity: f64,
  gamma: f64,
  gas_constant: f64,
  molar_mass: f64,
  albedo: Option<f64>,
  angular_velocity: f64,
  axial_tilt: f64,
}

impl AtmosphereSpec {
  pub fn from_world_constants(
    constants: &WorldConstants,
  ) -> AetherResult<Self> {
    let atmosphere = constants
      .atmosphere
      .ok_or_else(|| AetherError::new(AerError::MissingAtmosphereConstants))?;
    let spec = Self {
      reference_temperature: atmosphere.reference_temperature,
      reference_pressure: atmosphere.reference_pressure,
      surface_gravity: constants.surface_gravity,
      gamma: atmosphere.gamma,
      gas_constant: atmosphere.gas_constant,
      molar_mass: atmosphere.molar_mass,
      albedo: atmosphere.albedo,
      angular_velocity: atmosphere.angular_velocity,
      axial_tilt: atmosphere.axial_tilt,
    };
    spec.validate()?;
    Ok(spec)
  }

  pub fn reference_temperature(&self) -> f64 {
    self.reference_temperature
  }

  pub fn reference_pressure(&self) -> f64 {
    self.reference_pressure
  }

  pub fn surface_gravity(&self) -> f64 {
    self.surface_gravity
  }

  pub fn gamma(&self) -> f64 {
    self.gamma
  }

  pub fn gas_constant(&self) -> f64 {
    self.gas_constant
  }

  pub fn molar_mass(&self) -> f64 {
    self.molar_mass
  }

  pub fn albedo(&self) -> Option<f64> {
    self.albedo
  }

  pub fn angular_velocity(&self) -> f64 {
    self.angular_velocity
  }

  pub fn axial_tilt(&self) -> f64 {
    self.axial_tilt
  }

  pub fn reference_density(&self) -> f64 {
    self.reference_pressure / (self.gas_constant * self.reference_temperature)
  }

  /// Conserved moist Euler state `[rho, rho_u, rho_v, rho_w, energy, rho_q]`
  /// at rest. The atmosphere is initialised dry (`rho_q = 0`); moisture
  /// enters via evaporation at the air–sea interface.
  pub fn euler_rest_state(&self) -> [f64; 6] {
    let rho = self.reference_density();
    let internal_energy = self.reference_pressure / (self.gamma - 1.0);
    [rho, 0.0, 0.0, 0.0, internal_energy, 0.0]
  }

  pub fn temperature_field(&self, cell_count: usize) -> SoaField<1> {
    SoaField::<1>::from_fn(cell_count, |_| [self.reference_temperature])
  }

  pub fn euler_state_field(&self, cell_count: usize) -> SoaField<6> {
    let state = self.euler_rest_state();
    SoaField::<6>::from_fn(cell_count, |_| state)
  }

  /// Isothermal hydrostatic Euler state at rest over `mesh`.
  ///
  /// Uses `p(r) = p0 * exp(-(r - r0) / H)` where
  /// `H = R * T0 / g`. The mesh radius and `reference_radius` must use the
  /// same length units as the constants used to build this spec.
  pub fn isothermal_hydrostatic_state_field<M>(
    &self,
    mesh: &M,
    reference_radius: f64,
  ) -> AetherResult<SoaField<6>>
  where
    M: Mesh<3> + ?Sized,
  {
    if !reference_radius.is_finite() || reference_radius <= 0.0 {
      return Err(
        AetherError::new(AerError::InvalidAtmosphereConstants)
          .context(format!("reference radius {}", reference_radius)),
      );
    }

    let scale_height = if self.surface_gravity > 0.0 {
      self.gas_constant * self.reference_temperature / self.surface_gravity
    } else {
      f64::INFINITY
    };
    if scale_height <= 0.0 || scale_height.is_nan() {
      return Err(
        AetherError::new(AerError::InvalidAtmosphereConstants)
          .context(format!("scale height {}", scale_height)),
      );
    }

    Ok(SoaField::<6>::from_fn(mesh.cell_count(), |cell| {
      let r = cell_radius(mesh, cell);
      let pressure = if scale_height.is_infinite() {
        self.reference_pressure
      } else {
        self.reference_pressure * (-(r - reference_radius) / scale_height).exp()
      };
      let rho = pressure / (self.gas_constant * self.reference_temperature);
      let energy = pressure / (self.gamma - 1.0);
      [rho, 0.0, 0.0, 0.0, energy, 0.0]
    }))
  }

  pub fn moist_euler3d(&self) -> MoistEuler3D {
    MoistEuler3D::new(self.gamma)
  }

  pub fn moist_euler3d_with_radial_gravity(
    &self,
    gravity: Vec<[f64; 3]>,
  ) -> MoistEuler3D {
    MoistEuler3D::with_per_cell_gravity(self.gamma, gravity)
  }

  fn validate(&self) -> AetherResult<()> {
    if self.reference_temperature > 0.0
      && self.reference_temperature.is_finite()
      && self.reference_pressure > 0.0
      && self.reference_pressure.is_finite()
      && self.surface_gravity >= 0.0
      && self.surface_gravity.is_finite()
      && self.gamma > 1.0
      && self.gamma.is_finite()
      && self.gas_constant > 0.0
      && self.gas_constant.is_finite()
      && self.molar_mass > 0.0
      && self.molar_mass.is_finite()
      && self.angular_velocity.is_finite()
      && self.axial_tilt.is_finite()
    {
      Ok(())
    } else {
      Err(
        AetherError::new(AerError::InvalidAtmosphereConstants)
          .context(format!("atmosphere spec {:?}", self)),
      )
    }
  }
}

fn cell_radius<M>(mesh: &M, cell: CellId) -> f64
where
  M: Mesh<3> + ?Sized,
{
  let p = mesh.cell_world_centroid(cell);
  (p[0].powi(2) + p[1].powi(2) + p[2].powi(2)).sqrt()
}

#[cfg(test)]
mod tests {
  use nexus::{AtmosphereConstants, FieldStorage, WorldConstants};

  use super::*;

  fn earth_like_constants() -> WorldConstants {
    WorldConstants {
      mass: 5.97e24,
      radius: 6.371e6,
      surface_gravity: 9.81,
      atmosphere: Some(AtmosphereConstants {
        reference_temperature: 288.0,
        reference_pressure: 101_325.0,
        gamma: 1.4,
        gas_constant: 287.0,
        molar_mass: 0.02897,
        albedo: Some(0.3),
        angular_velocity: 7.292e-5,
        axial_tilt: 0.409,
      }),
      radiation: None,
    }
  }

  #[test]
  fn atmosphere_spec_builds_from_neutral_world_constants() {
    let spec =
      AtmosphereSpec::from_world_constants(&earth_like_constants()).unwrap();
    assert_eq!(spec.reference_temperature(), 288.0);
    assert_eq!(spec.reference_pressure(), 101_325.0);
    assert!((spec.reference_density() - 1.226).abs() < 0.01);
    let state = spec.euler_rest_state();
    assert_eq!(state[1], 0.0);
    assert_eq!(state[2], 0.0);
    assert_eq!(state[3], 0.0);
    assert!(state[0] > 0.0);
    assert!(state[4] > 0.0);
  }

  #[test]
  fn atmosphere_spec_rejects_world_without_atmosphere() {
    let result =
      AtmosphereSpec::from_world_constants(&WorldConstants::default());
    assert!(result.is_err());
  }

  #[test]
  fn hydrostatic_state_pressure_decays_with_radius() {
    use std::sync::Arc;

    use tessera::cube_sphere::{CubeSphere, CubeSphereShellSpec};

    let constants = WorldConstants {
      mass: 1.0,
      radius: 1.0,
      surface_gravity: 10.0,
      atmosphere: Some(nexus::AtmosphereConstants {
        reference_temperature: 10.0,
        reference_pressure: 100.0,
        gamma: 1.4,
        gas_constant: 10.0,
        molar_mass: 1.0,
        albedo: None,
        angular_velocity: 0.0,
        axial_tilt: 0.0,
      }),
      radiation: None,
    };
    let spec = AtmosphereSpec::from_world_constants(&constants).unwrap();
    let mesh = Arc::new(CubeSphere::shell(CubeSphereShellSpec::uniform(
      [2, 2, 2],
      1.0,
      3.0,
    )));

    let field = spec
      .isothermal_hydrostatic_state_field(mesh.as_ref(), 1.0)
      .unwrap();
    let gamma = spec.gamma();

    let inner = CellId::from(0);
    let outer = CellId::from(2 * 2);
    assert!(
      mesh.cell_world_centroid(outer).magnitude()
        > mesh.cell_world_centroid(inner).magnitude()
    );

    let inner_state = field.state(inner);
    let outer_state = field.state(outer);
    let inner_pressure = (gamma - 1.0) * inner_state[4];
    let outer_pressure = (gamma - 1.0) * outer_state[4];
    assert!(outer_pressure < inner_pressure);
    assert_eq!(inner_state[1], 0.0);
    assert_eq!(outer_state[3], 0.0);
  }
}
