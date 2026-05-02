// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Single-stage radiative transfer.
//!
//! `RadiativeTransferStep` reads the sun direction (a `[f64; 3]` resource
//! at `ResourceKey::SunPosition`), atmospheric and surface temperature
//! fields, and writes:
//!
//! - per-atmosphere-cell radiative heating tendency (K/s)
//! - per-surface-cell net radiative flux (W/m²)
//!
//! The physics is intentionally minimal for an MVP: a single-band gray
//! atmosphere with shortwave absorption fraction `f_sw`, surface albedo
//! `α`, surface emissivity `ε`, greenhouse factor `g` (fraction of
//! upward longwave trapped), and a linear atmospheric longwave damping
//! around a reference temperature. Real radiative transfer (multi-band,
//! line-by-line, scattering) is a follow-up.

use nexus::{
  FieldKey, FieldStorage, MeshKey, ResourceKey, SoaField, Stage, StageContext,
  WorldConstants,
};
use utility::{
  constants::STEFAN_BOLTZMANN,
  domain::CellId,
  error::{AetherError, AetherResult},
};

use crate::{error::LumenError, optical::zenith_cosine};

/// Model-only tuning knobs for `RadiativeTransferStep`. These are the
/// parts of the radiation parameter block that cosmo / WorldConstants
/// can't supply — they're a function of how aggressively this gray-band
/// MVP approximates the real atmosphere.
#[derive(Clone, Copy, Debug)]
pub struct RadiationCoefficients {
  /// Fraction of upwelling LW from the surface trapped by the atmosphere
  /// (0..1). 0 = transparent atmosphere, 1 = perfect greenhouse.
  pub greenhouse_factor: f64,
  /// Fraction of incident shortwave absorbed by the atmospheric column
  /// before reaching the surface (0..1).
  pub atmospheric_absorption: f64,
  /// Volumetric heat capacity of the atmosphere (J/(K·m³)). Used to
  /// convert the absorbed shortwave flux density to a temperature
  /// tendency.
  pub atm_heat_capacity: f64,
  /// Linear damping coefficient (1/s) — atmospheric heating tendency
  /// includes `-coeff * (T_atm - T_ref)`.
  pub atm_longwave_damping: f64,
}

impl Default for RadiationCoefficients {
  fn default() -> Self {
    Self {
      greenhouse_factor: 0.40,
      atmospheric_absorption: 0.20,
      atm_heat_capacity: 1206.0,
      atm_longwave_damping: 1.0e-5,
    }
  }
}

/// Full parameter block for `RadiativeTransferStep`. Splits cleanly
/// into:
///
/// - **Physical params** (`solar_constant`, `surface_albedo`,
///   `surface_emissivity`, `atm_reference_temperature`) — derived from
///   `WorldConstants` (i.e. cosmo) via `from_world_constants`.
/// - **Model coefficients** (greenhouse / absorption / heat capacity /
///   longwave damping) — knobs of this gray-band MVP, see
///   `RadiationCoefficients`.
///
/// `Default::default()` returns Earth-ish values for both halves;
/// it's convenient for tests and bootstrapping but not authoritative
/// — prefer `from_world_constants` once a `WorldConstants` is in hand.
#[derive(Clone, Copy, Debug)]
pub struct RadiationParameters {
  /// Top-of-atmosphere solar irradiance (W/m²). From cosmo.
  pub solar_constant: f64,
  /// Surface short-wave albedo (0..1). From cosmo.
  pub surface_albedo: f64,
  /// Surface long-wave emissivity (0..1). From cosmo.
  pub surface_emissivity: f64,
  /// Reference atmospheric temperature for the linear longwave damping
  /// term (K). From cosmo.
  pub atm_reference_temperature: f64,
  /// Fraction of upwelling LW from the surface trapped by the atmosphere
  /// (0..1). 0 = transparent atmosphere, 1 = perfect greenhouse.
  pub greenhouse_factor: f64,
  /// Fraction of incident shortwave absorbed by the atmospheric column
  /// before reaching the surface (0..1).
  pub atmospheric_absorption: f64,
  /// Volumetric heat capacity of the atmosphere (J/(K·m³)). Used to
  /// convert the absorbed shortwave flux density to a temperature
  /// tendency.
  pub atm_heat_capacity: f64,
  /// Linear damping coefficient (1/s) — atmospheric heating tendency
  /// includes `-coeff * (T_atm - T_ref)`.
  pub atm_longwave_damping: f64,
}

impl Default for RadiationParameters {
  fn default() -> Self {
    let coefficients = RadiationCoefficients::default();
    Self {
      solar_constant: 1361.0,
      surface_albedo: 0.30,
      surface_emissivity: 0.95,
      atm_reference_temperature: 250.0,
      greenhouse_factor: coefficients.greenhouse_factor,
      atmospheric_absorption: coefficients.atmospheric_absorption,
      atm_heat_capacity: coefficients.atm_heat_capacity,
      atm_longwave_damping: coefficients.atm_longwave_damping,
    }
  }
}

impl RadiationParameters {
  /// Build a parameter block from a cosmo-derived `WorldConstants` plus
  /// the model coefficient knobs. Errors when the world has no
  /// `RadiationConstants` (no resolvable primary star) or no
  /// `AtmosphereConstants` (no atmosphere to give a reference
  /// temperature).
  pub fn from_world_constants(
    constants: &WorldConstants,
    coefficients: RadiationCoefficients,
  ) -> AetherResult<Self> {
    let radiation = constants.radiation.ok_or_else(|| {
      AetherError::new(LumenError::MissingRadiationConstants)
        .context("WorldConstants::radiation is None")
    })?;
    let atmosphere = constants.atmosphere.ok_or_else(|| {
      AetherError::new(LumenError::MissingAtmosphereConstants)
        .context("WorldConstants::atmosphere is None")
    })?;
    let params = Self {
      solar_constant: radiation.solar_irradiance,
      surface_albedo: radiation.surface_albedo,
      surface_emissivity: radiation.surface_emissivity,
      atm_reference_temperature: atmosphere.reference_temperature,
      greenhouse_factor: coefficients.greenhouse_factor,
      atmospheric_absorption: coefficients.atmospheric_absorption,
      atm_heat_capacity: coefficients.atm_heat_capacity,
      atm_longwave_damping: coefficients.atm_longwave_damping,
    };
    params.validate()?;
    Ok(params)
  }
}

impl RadiationParameters {
  pub fn validate(&self) -> AetherResult<()> {
    let in_unit = |x: f64| (0.0..=1.0).contains(&x);
    let positive = |x: f64| x.is_finite() && x > 0.0;
    let non_negative = |x: f64| x.is_finite() && x >= 0.0;

    if positive(self.solar_constant)
      && in_unit(self.surface_albedo)
      && in_unit(self.surface_emissivity)
      && in_unit(self.greenhouse_factor)
      && in_unit(self.atmospheric_absorption)
      && positive(self.atm_heat_capacity)
      && positive(self.atm_reference_temperature)
      && non_negative(self.atm_longwave_damping)
    {
      Ok(())
    } else {
      Err(
        AetherError::new(LumenError::InvalidParameters)
          .context(format!("{:?}", self)),
      )
    }
  }
}

/// One-stage radiative transfer between an atmosphere mesh and a surface
/// mesh. Reads `SunPosition` (resource) plus both meshes' temperature
/// fields and writes heating tendency + net surface flux.
pub struct RadiativeTransferStep {
  atm_mesh: MeshKey,
  surface_mesh: MeshKey,
  atm_temperature: FieldKey,
  surface_temperature: FieldKey,
  heating_tendency: FieldKey,
  net_surface_flux: FieldKey,
  params: RadiationParameters,
  reads: [FieldKey; 2],
  writes: [FieldKey; 2],
  resource_reads: [ResourceKey; 1],
}

impl RadiativeTransferStep {
  pub fn new(
    atm_mesh: MeshKey,
    surface_mesh: MeshKey,
    atm_temperature: FieldKey,
    surface_temperature: FieldKey,
    heating_tendency: FieldKey,
    net_surface_flux: FieldKey,
    params: RadiationParameters,
  ) -> AetherResult<Self> {
    if atm_temperature.mesh() != atm_mesh || heating_tendency.mesh() != atm_mesh
    {
      return Err(AetherError::new(LumenError::FieldMeshMismatch).context(
        format!(
          "atm_mesh {:?}, atm_temperature {:?}, heating_tendency {:?}",
          atm_mesh, atm_temperature, heating_tendency
        ),
      ));
    }
    if surface_temperature.mesh() != surface_mesh
      || net_surface_flux.mesh() != surface_mesh
    {
      return Err(AetherError::new(LumenError::FieldMeshMismatch).context(
        format!(
          "surface_mesh {:?}, surface_temperature {:?}, net_surface_flux {:?}",
          surface_mesh, surface_temperature, net_surface_flux
        ),
      ));
    }
    params.validate()?;

    Ok(Self {
      atm_mesh,
      surface_mesh,
      atm_temperature,
      surface_temperature,
      heating_tendency,
      net_surface_flux,
      params,
      reads: [atm_temperature, surface_temperature],
      writes: [heating_tendency, net_surface_flux],
      resource_reads: [ResourceKey::SunPosition],
    })
  }

  pub fn atm_mesh(&self) -> MeshKey {
    self.atm_mesh
  }

  pub fn surface_mesh(&self) -> MeshKey {
    self.surface_mesh
  }

  pub fn parameters(&self) -> &RadiationParameters {
    &self.params
  }
}

impl Stage for RadiativeTransferStep {
  fn name(&self) -> &'static str {
    "lumen_radiative_transfer"
  }

  fn reads(&self) -> &[FieldKey] {
    &self.reads
  }

  fn writes(&self) -> &[FieldKey] {
    &self.writes
  }

  fn resource_reads(&self) -> &[ResourceKey] {
    &self.resource_reads
  }

  fn run(&mut self, mut ctx: StageContext<'_>) -> AetherResult<()> {
    // Pull mesh handles upfront so we can iterate cells without re-borrowing
    // tessera mid-iteration.
    let atm_mesh = ctx.world.tessera.mesh(self.atm_mesh).ok_or_else(|| {
      AetherError::new(LumenError::MissingMesh)
        .context(format!("{:?}", self.atm_mesh))
    })?;
    let surface_mesh =
      ctx.world.tessera.mesh(self.surface_mesh).ok_or_else(|| {
        AetherError::new(LumenError::MissingMesh)
          .context(format!("{:?}", self.surface_mesh))
      })?;

    let atm_cell_count = atm_mesh.cell_count();
    let surface_cell_count = surface_mesh.cell_count();

    // Compute per-cell zenith cosines for both meshes ahead of any field
    // mutation so we can drop the tessera borrows by the time we touch
    // pleroma.
    let sun = *ctx
      .world
      .fields
      .resource::<[f64; 3]>(ResourceKey::SunPosition)
      .ok_or_else(|| {
        AetherError::new(LumenError::MissingResource)
          .context("ResourceKey::SunPosition")
      })?;

    let atm_mu: Vec<f64> = (0..atm_cell_count)
      .map(|i| {
        let centroid = atm_mesh.cell_centroid(CellId::from(i));
        zenith_cosine(centroid, &sun)
      })
      .collect();
    let surface_mu: Vec<f64> = (0..surface_cell_count)
      .map(|i| {
        let centroid = surface_mesh.cell_centroid(CellId::from(i));
        zenith_cosine(centroid, &sun)
      })
      .collect();

    // ---- Atmospheric heating tendency ----
    let atm_temperatures = {
      let field: &SoaField<1> =
        ctx.world.fields.read(self.atm_temperature).ok_or_else(|| {
          AetherError::new(LumenError::MissingReadField)
            .context(format!("{:?}", self.atm_temperature))
        })?;
      if field.len() != atm_cell_count {
        return Err(AetherError::new(LumenError::FieldLengthMismatch).context(
          format!(
            "atm_temperature len {}, atm cell count {}",
            field.len(),
            atm_cell_count
          ),
        ));
      }
      field.component(0).as_ref().to_vec()
    };

    let p = self.params;
    let heating: Vec<f64> = atm_mu
      .iter()
      .zip(atm_temperatures.iter())
      .map(|(mu, t_atm)| {
        let solar_heating = p.atmospheric_absorption * p.solar_constant * *mu
          / p.atm_heat_capacity;
        let lw_damping =
          p.atm_longwave_damping * (t_atm - p.atm_reference_temperature);
        solar_heating - lw_damping
      })
      .collect();

    {
      let field: &mut SoaField<1> = ctx
        .world
        .fields
        .write(self.heating_tendency)
        .ok_or_else(|| {
          AetherError::new(LumenError::MissingWriteField)
            .context(format!("{:?}", self.heating_tendency))
        })?;
      if field.len() != atm_cell_count {
        return Err(AetherError::new(LumenError::FieldLengthMismatch).context(
          format!(
            "heating_tendency len {}, atm cell count {}",
            field.len(),
            atm_cell_count
          ),
        ));
      }
      for (cell, value) in heating.iter().enumerate() {
        field.write(CellId::from(cell), &[*value]);
      }
    }

    // ---- Surface net radiative flux ----
    let surface_temperatures = {
      let field: &SoaField<1> = ctx
        .world
        .fields
        .read(self.surface_temperature)
        .ok_or_else(|| {
          AetherError::new(LumenError::MissingReadField)
            .context(format!("{:?}", self.surface_temperature))
        })?;
      if field.len() != surface_cell_count {
        return Err(AetherError::new(LumenError::FieldLengthMismatch).context(
          format!(
            "surface_temperature len {}, surface cell count {}",
            field.len(),
            surface_cell_count
          ),
        ));
      }
      field.component(0).as_ref().to_vec()
    };

    let net_flux: Vec<f64> = surface_mu
      .iter()
      .zip(surface_temperatures.iter())
      .map(|(mu, t_s)| {
        let incoming_sw =
          (1.0 - p.atmospheric_absorption) * p.solar_constant * *mu;
        let absorbed_sw = (1.0 - p.surface_albedo) * incoming_sw;
        let outgoing_lw = p.surface_emissivity * STEFAN_BOLTZMANN * t_s.powi(4);
        absorbed_sw - (1.0 - p.greenhouse_factor) * outgoing_lw
      })
      .collect();

    {
      let field: &mut SoaField<1> = ctx
        .world
        .fields
        .write(self.net_surface_flux)
        .ok_or_else(|| {
          AetherError::new(LumenError::MissingWriteField)
            .context(format!("{:?}", self.net_surface_flux))
        })?;
      if field.len() != surface_cell_count {
        return Err(AetherError::new(LumenError::FieldLengthMismatch).context(
          format!(
            "net_surface_flux len {}, surface cell count {}",
            field.len(),
            surface_cell_count
          ),
        ));
      }
      for (cell, value) in net_flux.iter().enumerate() {
        field.write(CellId::from(cell), &[*value]);
      }
    }

    Ok(())
  }
}
