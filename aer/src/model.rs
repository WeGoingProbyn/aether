// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use nexus::{
  FieldKey, FieldName, MeshKey, Nexus, Pleroma, SoaField, StageId,
  WorldConstants,
};
use tessera::mesh::Mesh;
use utility::error::{AetherError, AetherResult};

use crate::{
  diagnostics::EulerDiagnosticsStep,
  dynamics::{AtmosphereScheme, EulerAtmosphereStep, RotationMode},
  error::AerError,
  init::AtmosphereSpec,
  thermal::TemperatureTendencyToEulerEnergyStep,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AtmosphereFields {
  pub temperature: FieldKey,
  pub temperature_tendency: FieldKey,
  pub euler_state: FieldKey,
  pub pressure: FieldKey,
  pub velocity_x: FieldKey,
  pub velocity_y: FieldKey,
  pub velocity_z: FieldKey,
  /// Specific humidity `q` diagnosed from the prognostic `ρq` carried in
  /// the 6th Euler component. Read by microphysics and the air–sea flux.
  pub humidity: FieldKey,
}

impl AtmosphereFields {
  pub const fn for_mesh(mesh: MeshKey) -> Self {
    Self {
      temperature: FieldKey::new(mesh, FieldName::Temperature),
      temperature_tendency: FieldKey::new(mesh, FieldName::TemperatureTendency),
      euler_state: FieldKey::new(mesh, FieldName::EulerState),
      pressure: FieldKey::new(mesh, FieldName::Pressure),
      velocity_x: FieldKey::new(mesh, FieldName::VelocityX),
      velocity_y: FieldKey::new(mesh, FieldName::VelocityY),
      velocity_z: FieldKey::new(mesh, FieldName::VelocityZ),
      humidity: FieldKey::new(mesh, FieldName::Humidity),
    }
  }

  pub fn all(self) -> [FieldKey; 8] {
    [
      self.temperature,
      self.temperature_tendency,
      self.euler_state,
      self.pressure,
      self.velocity_x,
      self.velocity_y,
      self.velocity_z,
      self.humidity,
    ]
  }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AtmosphereStageIds {
  pub tendency_to_energy: StageId,
  pub dynamics: StageId,
  pub diagnostics: StageId,
}

#[derive(Clone, Debug)]
pub struct AtmosphereModel {
  mesh: MeshKey,
  fields: AtmosphereFields,
  cfl: f64,
  rotation: RotationMode,
  scheme: AtmosphereScheme,
  max_substeps: usize,
  /// Extra dT/dt fields aer's energy stage should sum into the Euler
  /// energy update on top of `fields.temperature_tendency`. Lumen's
  /// `RadiativeHeatingTendency` is the typical first entry.
  extra_tendencies: Vec<FieldKey>,
}

impl AtmosphereModel {
  pub fn new(mesh: MeshKey) -> Self {
    Self {
      mesh,
      fields: AtmosphereFields::for_mesh(mesh),
      cfl: 0.25,
      rotation: RotationMode::None,
      scheme: AtmosphereScheme::Explicit,
      max_substeps: 10_000,
      extra_tendencies: Vec::new(),
    }
  }

  /// Cap the number of inner CFL sub-steps the dynamics may take per outer
  /// tick (it errors past this). Useful to assert a scheme actually removes the
  /// sub-step explosion.
  pub fn with_max_substeps(mut self, max_substeps: usize) -> Self {
    self.max_substeps = max_substeps.max(1);
    self
  }

  /// Enable the planetary Coriolis source (rotation about world +z at the
  /// body's angular velocity) — the dynamical driver of weather systems.
  pub fn with_rotation(mut self) -> Self {
    self.rotation = RotationMode::Planetary;
    self
  }

  /// Use the vertically-implicit (HEVI) scheme for the dynamics — large stable
  /// steps on the thin atmospheric shell (removes the vertical acoustic CFL).
  pub fn with_hevi(self) -> Self {
    self.with_scheme(AtmosphereScheme::Hevi)
  }

  /// Select the dynamics time-stepping scheme.
  pub fn with_scheme(mut self, scheme: AtmosphereScheme) -> Self {
    self.scheme = scheme;
    self
  }

  pub fn with_fields(mut self, fields: AtmosphereFields) -> Self {
    self.fields = fields;
    self
  }

  pub fn with_cfl(mut self, cfl: f64) -> Self {
    self.cfl = cfl;
    self
  }

  /// Add an extra dT/dt source to be summed into the Euler energy
  /// update. The field must live on the same mesh as the atmosphere
  /// model. Repeated calls accumulate sources.
  pub fn with_extra_tendency(mut self, key: FieldKey) -> Self {
    self.extra_tendencies.push(key);
    self
  }

  /// Convenience for the common case: lumen writes
  /// `FieldName::RadiativeHeatingTendency` to the same atmosphere mesh.
  pub fn with_radiative_heating(self) -> Self {
    let key = FieldKey::new(self.mesh, FieldName::RadiativeHeatingTendency);
    self.with_extra_tendency(key)
  }

  pub fn extra_tendencies(&self) -> &[FieldKey] {
    &self.extra_tendencies
  }

  pub fn mesh(&self) -> MeshKey {
    self.mesh
  }

  pub fn fields(&self) -> AtmosphereFields {
    self.fields
  }

  pub fn cfl(&self) -> f64 {
    self.cfl
  }

  pub fn register_fields<M>(
    &self,
    pleroma: &mut Pleroma,
    mesh: &M,
    constants: &WorldConstants,
    reference_radius: f64,
  ) -> AetherResult<()>
  where
    M: Mesh<3> + ?Sized,
  {
    self.validate()?;

    let cell_count = mesh.cell_count();
    let spec = AtmosphereSpec::from_world_constants(constants)?;
    let euler_state =
      spec.isothermal_hydrostatic_state_field(mesh, reference_radius)?;

    pleroma.register_field(
      self.fields.temperature,
      spec.temperature_field(cell_count),
    );
    pleroma.register_field(
      self.fields.temperature_tendency,
      SoaField::<1>::zeros(cell_count),
    );
    pleroma.register_field(self.fields.euler_state, euler_state);
    pleroma
      .register_field(self.fields.pressure, SoaField::<1>::zeros(cell_count));
    pleroma
      .register_field(self.fields.velocity_x, SoaField::<1>::zeros(cell_count));
    pleroma
      .register_field(self.fields.velocity_y, SoaField::<1>::zeros(cell_count));
    pleroma
      .register_field(self.fields.velocity_z, SoaField::<1>::zeros(cell_count));
    pleroma
      .register_field(self.fields.humidity, SoaField::<1>::zeros(cell_count));

    Ok(())
  }

  pub fn add_stages(
    &self,
    nexus: &mut Nexus,
  ) -> AetherResult<AtmosphereStageIds> {
    self.validate()?;

    let mut tendencies = vec![self.fields.temperature_tendency];
    tendencies.extend(self.extra_tendencies.iter().copied());
    let tendency_to_energy =
      nexus.add(TemperatureTendencyToEulerEnergyStep::with_tendencies(
        self.mesh,
        self.fields.euler_state,
        tendencies,
      )?);
    let dynamics = nexus.add(
      EulerAtmosphereStep::forward_euler(
        self.mesh,
        self.fields.euler_state,
        self.cfl,
      )?
      .with_rotation_mode(self.rotation)
      .with_scheme(self.scheme)
      .with_max_substeps(self.max_substeps),
    );
    let diagnostics = nexus.add(EulerDiagnosticsStep::new(
      self.mesh,
      self.fields.euler_state,
      self.fields.temperature,
      self.fields.pressure,
      self.fields.velocity_x,
      self.fields.velocity_y,
      self.fields.velocity_z,
      self.fields.humidity,
    )?);

    Ok(AtmosphereStageIds {
      tendency_to_energy,
      dynamics,
      diagnostics,
    })
  }

  fn validate(&self) -> AetherResult<()> {
    if !self.cfl.is_finite() || self.cfl <= 0.0 {
      return Err(
        AetherError::new(AerError::InvalidTimeStep)
          .context(format!("cfl {}", self.cfl)),
      );
    }

    let core_ok = self
      .fields
      .all()
      .iter()
      .all(|field| field.mesh() == self.mesh);
    let extras_ok = self
      .extra_tendencies
      .iter()
      .all(|field| field.mesh() == self.mesh);
    if core_ok && extras_ok {
      Ok(())
    } else {
      Err(
        AetherError::new(AerError::FieldMeshMismatch).context(format!(
          "mesh {:?}, fields {:?}, extra_tendencies {:?}",
          self.mesh, self.fields, self.extra_tendencies
        )),
      )
    }
  }
}

impl Default for AtmosphereModel {
  fn default() -> Self {
    Self::new(MeshKey::ATMOSPHERE)
  }
}

#[cfg(test)]
mod tests {
  use std::sync::Arc;

  use nexus::{
    AtmosphereConstants, FieldStorage, Pleroma, WorldConstants, WorldId,
  };
  use tessera::{
    cube_sphere::{CubeSphere, CubeSphereShellSpec},
    geometry::CellGeometry,
    world_mesh::Tessera,
  };
  use utility::thread::pool::Pool;

  use super::*;

  fn constants() -> WorldConstants {
    WorldConstants {
      mass: 1.0,
      radius: 1.0,
      surface_gravity: 1.0,
      atmosphere: Some(AtmosphereConstants {
        reference_temperature: 1.0,
        reference_pressure: 1.0,
        gamma: 1.4,
        gas_constant: 1.0,
        molar_mass: 1.0,
        albedo: None,
        angular_velocity: 0.0,
        axial_tilt: 0.0,
      }),
      radiation: None,
    }
  }

  #[test]
  fn atmosphere_model_registers_fields_and_stages() {
    let mesh = Arc::new(CubeSphere::shell(CubeSphereShellSpec::uniform(
      [2, 2, 2],
      1.0,
      1.2,
    )));
    let mut tessera = Tessera::new();
    tessera.register_mesh(MeshKey::ATMOSPHERE, mesh.clone());

    let model = AtmosphereModel::default().with_cfl(0.25);
    let fields = model.fields();
    let mut pleroma = Pleroma::new();
    model
      .register_fields(&mut pleroma, mesh.as_ref(), &constants(), 1.0)
      .unwrap();

    assert_eq!(
      pleroma.cell_count(fields.euler_state),
      Some(mesh.cell_count())
    );
    assert_eq!(pleroma.cell_count(fields.pressure), Some(mesh.cell_count()));

    let mut nexus = Nexus::new();
    let stage_ids = model.add_stages(&mut nexus).unwrap();
    assert_eq!(stage_ids.tendency_to_energy.index(), 0);
    assert_eq!(stage_ids.dynamics.index(), 1);
    assert_eq!(stage_ids.diagnostics.index(), 2);

    let mut compiled = nexus.build(&pleroma).unwrap();
    compiled
      .tick(
        WorldId(0),
        &tessera,
        &constants(),
        &mut pleroma,
        &Pool::default(),
        0.01,
      )
      .unwrap();

    let pressure: &SoaField<1> = pleroma.read(fields.pressure).unwrap();
    for i in 0..pressure.len() {
      assert!(pressure.state(utility::domain::CellId::from(i))[0] > 0.0);
    }
  }
}
