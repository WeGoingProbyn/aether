// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Warm-rain microphysics: the condensation/precipitation half of the
//! hydrological cycle. The atmosphere carries water vapour as the 6th
//! moist-Euler component (`ρq`); this module removes any super-saturated
//! vapour, releases its latent heat into the energy equation, and books
//! the removed mass as precipitation.
//!
//! The first-proof scheme is an instantaneous *saturation adjustment*:
//! wherever specific humidity `q` exceeds the saturation value
//! `q_sat(T, p)`, the excess condenses and precipitates out in the same
//! step (no prognostic cloud water). This conserves total water per cell
//! (vapour lost == precipitation produced) and is unconditionally stable.

use nexus::{
  FieldKey, FieldName, FieldStorage, MeshKey, SoaField, Stage, StageContext,
};
use utility::{
  domain::CellId,
  error::{AetherError, AetherResult},
};

use crate::{error::AerError, init::AtmosphereSpec};

/// Latent heat of vaporisation of water (J/kg), treated as constant.
pub const LATENT_HEAT_VAPORISATION: f64 = 2.5e6;

/// Saturation vapour pressure (Pa) over liquid water via the Tetens
/// formula. `temperature` is in kelvin.
pub fn saturation_vapour_pressure(temperature: f64) -> f64 {
  let t_celsius = temperature - 273.15;
  611.2 * (17.67 * t_celsius / (t_celsius + 243.5)).exp()
}

/// Saturation specific humidity `q_sat` (kg water / kg moist air) at the
/// given temperature (K) and pressure (Pa). Uses the standard ratio of the
/// molar masses of water and dry air (≈ 0.622).
pub fn saturation_specific_humidity(temperature: f64, pressure: f64) -> f64 {
  const EPSILON: f64 = 0.622;
  let e_sat = saturation_vapour_pressure(temperature);
  // Guard against the (unphysical) e_sat ≥ p regime near boiling.
  let denom = (pressure - (1.0 - EPSILON) * e_sat).max(1.0);
  (EPSILON * e_sat / denom).clamp(0.0, 1.0)
}

/// Instantaneous saturation-adjustment / precipitation stage on the moist
/// atmosphere. Reads & writes the prognostic `SoaField<6>` Euler state and
/// writes a per-cell precipitation mass rate (kg/m³/s) diagnostic.
pub struct SaturationAdjustmentStep {
  mesh: MeshKey,
  state: FieldKey,
  precipitation: FieldKey,
  reads: [FieldKey; 1],
  writes: [FieldKey; 2],
}

impl SaturationAdjustmentStep {
  pub fn new(
    mesh: MeshKey,
    state: FieldKey,
    precipitation: FieldKey,
  ) -> AetherResult<Self> {
    if state.mesh() != mesh || precipitation.mesh() != mesh {
      return Err(AetherError::new(AerError::FieldMeshMismatch).context(
        format!(
          "mesh {:?}, state {:?}, precipitation {:?}",
          mesh, state, precipitation
        ),
      ));
    }
    Ok(Self {
      mesh,
      state,
      precipitation,
      reads: [state],
      writes: [state, precipitation],
    })
  }

  pub fn mesh(&self) -> MeshKey {
    self.mesh
  }
}

impl Stage for SaturationAdjustmentStep {
  fn name(&self) -> &'static str {
    "aer_saturation_adjustment"
  }

  fn reads(&self) -> &[FieldKey] {
    &self.reads
  }

  fn writes(&self) -> &[FieldKey] {
    &self.writes
  }

  fn run(&mut self, mut ctx: StageContext<'_>) -> AetherResult<()> {
    let mesh_cell_count = ctx
      .world
      .tessera
      .mesh(self.mesh)
      .ok_or_else(|| {
        AetherError::new(AerError::MissingMesh)
          .context(format!("{:?}", self.mesh))
      })?
      .cell_count();

    let dt = ctx.world.dt;
    if !dt.is_finite() || dt <= 0.0 {
      return Err(
        AetherError::new(AerError::InvalidTimeStep)
          .context(format!("dt {}", dt)),
      );
    }

    let spec = AtmosphereSpec::from_world_constants(ctx.world.constants)?;
    let gamma = spec.gamma();
    let gas_constant = spec.gas_constant();

    // Compute the per-cell condensed mass (kg/m³) and updated states first,
    // borrowing the state immutably, then write both fields.
    let mut precip_rate = vec![0.0; mesh_cell_count];
    let new_states: Vec<[f64; 6]> = {
      let state: &SoaField<6> =
        ctx.world.fields.read(self.state).ok_or_else(|| {
          AetherError::new(AerError::MissingReadField)
            .context(format!("{:?}", self.state))
        })?;
      if state.len() != mesh_cell_count {
        return Err(AetherError::new(AerError::FieldLengthMismatch));
      }

      let mut out = Vec::with_capacity(mesh_cell_count);
      for i in 0..mesh_cell_count {
        let mut s = state.state(CellId::from(i));
        let rho = s[0];
        if !rho.is_finite() || rho <= 0.0 {
          return Err(
            AetherError::new(AerError::InvalidAtmosphereState)
              .context(format!("cell {} density {}", i, rho)),
          );
        }
        let kinetic = 0.5 / rho * (s[1] * s[1] + s[2] * s[2] + s[3] * s[3]);
        let p = (gamma - 1.0) * (s[4] - kinetic);
        let t = p / (rho * gas_constant);
        if p > 0.0 && t > 0.0 {
          let q = s[5] / rho;
          let q_sat = saturation_specific_humidity(t, p);
          if q > q_sat {
            // Condense the excess: remove vapour mass, release latent
            // heat into energy, book the rest as precipitation.
            let condensed = rho * (q - q_sat); // kg/m³
            s[5] -= condensed;
            if s[5] < 0.0 {
              s[5] = 0.0;
            }
            s[4] += LATENT_HEAT_VAPORISATION * condensed;
            precip_rate[i] = condensed / dt;
          }
        }
        out.push(s);
      }
      out
    };

    {
      let state: &mut SoaField<6> =
        ctx.world.fields.write(self.state).ok_or_else(|| {
          AetherError::new(AerError::MissingWriteField)
            .context(format!("{:?}", self.state))
        })?;
      for (i, s) in new_states.into_iter().enumerate() {
        state.write(CellId::from(i), &s);
      }
    }

    {
      let precip: &mut SoaField<1> =
        ctx.world.fields.write(self.precipitation).ok_or_else(|| {
          AetherError::new(AerError::MissingWriteField)
            .context(format!("{:?}", self.precipitation))
        })?;
      if precip.len() != mesh_cell_count {
        return Err(AetherError::new(AerError::FieldLengthMismatch));
      }
      for (i, rate) in precip_rate.into_iter().enumerate() {
        precip.write(CellId::from(i), &[rate]);
      }
    }

    Ok(())
  }
}

/// Convenience constructor for the standard precipitation field key on a
/// mesh.
pub fn precipitation_field(mesh: MeshKey) -> FieldKey {
  FieldKey::new(mesh, FieldName::PrecipitationFlux)
}
