// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use nexus::{FieldKey, FieldStorage, MeshKey, SoaField, Stage, StageContext};
use utility::{
  domain::CellId,
  error::{AetherError, AetherResult},
};

use crate::{error::AerError, init::AtmosphereSpec};

/// Applies one or more scalar temperature tendencies (K/s) to the Euler
/// energy component.
///
/// Multiple physics modules (syzygy interface fluxes, lumen radiative
/// heating, microphysics latent-heat release, etc.) all express their
/// effect as `dT/dt` on the same atmosphere mesh. This stage sums those
/// tendency fields per cell and applies the result to `EulerState`:
///
/// `dE/dt = rho * c_v * Σ_i dT_i/dt`, with `c_v = R / (gamma - 1)`.
pub struct TemperatureTendencyToEulerEnergyStep {
  mesh: MeshKey,
  state: FieldKey,
  tendencies: Vec<FieldKey>,
  reads: Vec<FieldKey>,
  writes: [FieldKey; 1],
}

impl TemperatureTendencyToEulerEnergyStep {
  pub fn new(
    mesh: MeshKey,
    state: FieldKey,
    tendency: FieldKey,
  ) -> AetherResult<Self> {
    Self::with_tendencies(mesh, state, vec![tendency])
  }

  pub fn with_tendencies(
    mesh: MeshKey,
    state: FieldKey,
    tendencies: Vec<FieldKey>,
  ) -> AetherResult<Self> {
    if state.mesh() != mesh
      || tendencies.iter().any(|t| t.mesh() != mesh)
      || tendencies.is_empty()
    {
      return Err(AetherError::new(AerError::FieldMeshMismatch).context(
        format!(
          "mesh {:?}, state {:?}, tendencies {:?}",
          mesh, state, tendencies
        ),
      ));
    }

    Ok(Self {
      mesh,
      state,
      reads: tendencies.clone(),
      tendencies,
      writes: [state],
    })
  }

  pub fn mesh(&self) -> MeshKey {
    self.mesh
  }

  pub fn state(&self) -> FieldKey {
    self.state
  }

  pub fn tendencies(&self) -> &[FieldKey] {
    &self.tendencies
  }
}

impl Stage for TemperatureTendencyToEulerEnergyStep {
  fn name(&self) -> &'static str {
    "aer_temperature_tendency_to_euler_energy"
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
          .context(format!("requested dt {}", dt)),
      );
    }

    let spec = AtmosphereSpec::from_world_constants(ctx.world.constants)?;
    let cv = spec.gas_constant() / (spec.gamma() - 1.0);
    let mut tendency_sum = vec![0.0; mesh_cell_count];
    for key in &self.tendencies {
      let tendency: &SoaField<1> =
        ctx.world.fields.read(*key).ok_or_else(|| {
          AetherError::new(AerError::MissingReadField)
            .context(format!("{:?}", key))
        })?;
      if tendency.len() != mesh_cell_count {
        return Err(AetherError::new(AerError::FieldLengthMismatch).context(
          format!(
            "tendency {:?} len {}, mesh cell count {}",
            key,
            tendency.len(),
            mesh_cell_count
          ),
        ));
      }
      for (i, value) in tendency.component(0).as_ref().iter().enumerate() {
        tendency_sum[i] += *value;
      }
    }

    let state: &mut SoaField<5> =
      ctx.world.fields.write(self.state).ok_or_else(|| {
        AetherError::new(AerError::MissingWriteField)
          .context(format!("{:?}", self.state))
      })?;
    if state.len() != mesh_cell_count {
      return Err(AetherError::new(AerError::FieldLengthMismatch).context(
        format!(
          "state len {}, mesh cell count {}",
          state.len(),
          mesh_cell_count
        ),
      ));
    }

    for (cell, tendency) in tendency_sum.into_iter().enumerate() {
      let cell = CellId::from(cell);
      let mut s = state.state(cell);
      let rho = s[0];
      if !rho.is_finite() || rho <= 0.0 {
        return Err(
          AetherError::new(AerError::InvalidAtmosphereState).context(format!(
            "cell {} density {}",
            cell.index(),
            rho
          )),
        );
      }
      s[4] += dt * rho * cv * tendency;
      if !s[4].is_finite() {
        return Err(
          AetherError::new(AerError::InvalidAtmosphereState).context(format!(
            "cell {} energy {}",
            cell.index(),
            s[4]
          )),
        );
      }
      state.write(cell, &s);
    }

    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use std::sync::Arc;

  use nexus::{
    AtmosphereConstants, FieldName, Nexus, Pleroma, WorldConstants, WorldId,
  };
  use tessera::{
    geometry::IdentityMap,
    mesh::{Mesh, StructuredBlock},
    world_mesh::Tessera,
  };
  use utility::thread::pool::Pool;

  use super::*;

  const STATE: FieldKey =
    FieldKey::new(MeshKey::ATMOSPHERE, FieldName::EulerState);
  const TENDENCY: FieldKey =
    FieldKey::new(MeshKey::ATMOSPHERE, FieldName::TemperatureTendency);

  fn constants() -> WorldConstants {
    WorldConstants {
      mass: 1.0,
      radius: 1.0,
      surface_gravity: 0.0,
      atmosphere: Some(AtmosphereConstants {
        reference_temperature: 300.0,
        reference_pressure: 120_000.0,
        gamma: 1.4,
        gas_constant: 300.0,
        molar_mass: 1.0,
        albedo: None,
        angular_velocity: 0.0,
        axial_tilt: 0.0,
      }),
      radiation: None,
    }
  }

  #[test]
  fn temperature_tendency_updates_euler_energy() {
    let mut tessera = Tessera::new();
    let mesh = Arc::new(StructuredBlock::uniform(
      [0.0; 3].into(),
      [1.0; 3],
      [1, 1, 1],
      Box::new(IdentityMap::<3>),
    ));
    let mesh_for_registry: Arc<dyn Mesh<3>> = mesh;
    tessera.register_mesh(MeshKey::ATMOSPHERE, mesh_for_registry);

    let mut pleroma = Pleroma::new();
    pleroma.register_field(
      STATE,
      SoaField::<5>::from_fn(1, |_| [2.0, 0.0, 0.0, 0.0, 10.0]),
    );
    pleroma.register_field(TENDENCY, SoaField::<1>::from_fn(1, |_| [0.5]));

    let mut nexus = Nexus::new();
    nexus.add(
      TemperatureTendencyToEulerEnergyStep::new(
        MeshKey::ATMOSPHERE,
        STATE,
        TENDENCY,
      )
      .unwrap(),
    );
    let mut compiled = nexus.build(&pleroma).unwrap();
    compiled
      .tick(
        WorldId(0),
        &tessera,
        &constants(),
        &mut pleroma,
        &Pool::default(),
        4.0,
      )
      .unwrap();

    let state: &SoaField<5> = pleroma.read(STATE).unwrap();
    let updated = state.state(CellId::from(0));
    let cv = 300.0 / (1.4 - 1.0);
    assert_eq!(updated[4], 10.0 + 4.0 * 2.0 * cv * 0.5);
  }
}
