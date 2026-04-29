// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use nexus::{
  FieldKey, FieldStorage, MeshKey, SoaField, Stage, StageContext, WorldAccess,
};
use utility::{
  domain::CellId,
  error::{AetherError, AetherResult},
};

use crate::{error::AerError, init::AtmosphereSpec};

/// Derives scalar diagnostics from the prognostic Euler atmosphere state.
///
/// Reads `[rho, rho_u, rho_v, rho_w, energy]` and writes temperature,
/// pressure, and world-frame velocity components. These fields are intended
/// for coupling, rendering, and diagnostics, not as independent atmosphere
/// prognostic state.
pub struct EulerDiagnosticsStep {
  mesh: MeshKey,
  state: FieldKey,
  temperature: FieldKey,
  pressure: FieldKey,
  velocity_x: FieldKey,
  velocity_y: FieldKey,
  velocity_z: FieldKey,
  reads: [FieldKey; 1],
  writes: [FieldKey; 5],
}

impl EulerDiagnosticsStep {
  pub fn new(
    mesh: MeshKey,
    state: FieldKey,
    temperature: FieldKey,
    pressure: FieldKey,
    velocity_x: FieldKey,
    velocity_y: FieldKey,
    velocity_z: FieldKey,
  ) -> AetherResult<Self> {
    validate_mesh_fields(
      mesh,
      &[
        state,
        temperature,
        pressure,
        velocity_x,
        velocity_y,
        velocity_z,
      ],
    )?;

    Ok(Self {
      mesh,
      state,
      temperature,
      pressure,
      velocity_x,
      velocity_y,
      velocity_z,
      reads: [state],
      writes: [temperature, pressure, velocity_x, velocity_y, velocity_z],
    })
  }

  pub fn mesh(&self) -> MeshKey {
    self.mesh
  }

  pub fn state(&self) -> FieldKey {
    self.state
  }

  pub fn temperature(&self) -> FieldKey {
    self.temperature
  }

  pub fn pressure(&self) -> FieldKey {
    self.pressure
  }
}

impl Stage for EulerDiagnosticsStep {
  fn name(&self) -> &'static str {
    "aer_euler_diagnostics"
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

    let spec = AtmosphereSpec::from_world_constants(ctx.world.constants)?;
    let diagnostics = {
      let state: &SoaField<5> =
        ctx.world.fields.read(self.state).ok_or_else(|| {
          AetherError::new(AerError::MissingReadField)
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

      derive_diagnostics(state, spec.gamma(), spec.gas_constant())?
    };

    write_scalar_field(
      &mut ctx.world.fields,
      self.temperature,
      &diagnostics.temperature,
      mesh_cell_count,
    )?;
    write_scalar_field(
      &mut ctx.world.fields,
      self.pressure,
      &diagnostics.pressure,
      mesh_cell_count,
    )?;
    write_scalar_field(
      &mut ctx.world.fields,
      self.velocity_x,
      &diagnostics.velocity_x,
      mesh_cell_count,
    )?;
    write_scalar_field(
      &mut ctx.world.fields,
      self.velocity_y,
      &diagnostics.velocity_y,
      mesh_cell_count,
    )?;
    write_scalar_field(
      &mut ctx.world.fields,
      self.velocity_z,
      &diagnostics.velocity_z,
      mesh_cell_count,
    )?;

    Ok(())
  }
}

struct EulerDiagnostics {
  temperature: Vec<f64>,
  pressure: Vec<f64>,
  velocity_x: Vec<f64>,
  velocity_y: Vec<f64>,
  velocity_z: Vec<f64>,
}

fn derive_diagnostics(
  state: &SoaField<5>,
  gamma: f64,
  gas_constant: f64,
) -> AetherResult<EulerDiagnostics> {
  let mut temperature = Vec::with_capacity(state.len());
  let mut pressure = Vec::with_capacity(state.len());
  let mut velocity_x = Vec::with_capacity(state.len());
  let mut velocity_y = Vec::with_capacity(state.len());
  let mut velocity_z = Vec::with_capacity(state.len());

  for i in 0..state.len() {
    let cell = CellId::from(i);
    let s = state.state(cell);
    let rho = s[0];
    if !rho.is_finite() || rho <= 0.0 {
      return Err(
        AetherError::new(AerError::InvalidAtmosphereState)
          .context(format!("cell {} density {}", i, rho)),
      );
    }

    let inv_rho = 1.0 / rho;
    let u = s[1] * inv_rho;
    let v = s[2] * inv_rho;
    let w = s[3] * inv_rho;
    let kinetic = 0.5 * rho * (u * u + v * v + w * w);
    let p = (gamma - 1.0) * (s[4] - kinetic);
    let t = p / (rho * gas_constant);
    if !p.is_finite() || p <= 0.0 || !t.is_finite() || t <= 0.0 {
      return Err(
        AetherError::new(AerError::InvalidAtmosphereState)
          .context(format!("cell {} pressure {}, temperature {}", i, p, t)),
      );
    }

    temperature.push(t);
    pressure.push(p);
    velocity_x.push(u);
    velocity_y.push(v);
    velocity_z.push(w);
  }

  Ok(EulerDiagnostics {
    temperature,
    pressure,
    velocity_x,
    velocity_y,
    velocity_z,
  })
}

fn write_scalar_field(
  fields: &mut WorldAccess<'_>,
  key: FieldKey,
  values: &[f64],
  mesh_cell_count: usize,
) -> AetherResult<()> {
  let field: &mut SoaField<1> = fields.write(key).ok_or_else(|| {
    AetherError::new(AerError::MissingWriteField).context(format!("{:?}", key))
  })?;
  if field.len() != values.len() || field.len() != mesh_cell_count {
    return Err(AetherError::new(AerError::FieldLengthMismatch).context(
      format!(
        "field {:?} len {}, values len {}, mesh cell count {}",
        key,
        field.len(),
        values.len(),
        mesh_cell_count
      ),
    ));
  }

  for (cell, value) in values.iter().enumerate() {
    field.write(CellId::from(cell), &[*value]);
  }
  Ok(())
}

fn validate_mesh_fields(
  mesh: MeshKey,
  fields: &[FieldKey],
) -> AetherResult<()> {
  if fields.iter().all(|field| field.mesh() == mesh) {
    Ok(())
  } else {
    Err(
      AetherError::new(AerError::FieldMeshMismatch)
        .context(format!("mesh {:?}, fields {:?}", mesh, fields)),
    )
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
  const TEMPERATURE: FieldKey =
    FieldKey::new(MeshKey::ATMOSPHERE, FieldName::Temperature);
  const PRESSURE: FieldKey =
    FieldKey::new(MeshKey::ATMOSPHERE, FieldName::Pressure);
  const VELOCITY_X: FieldKey =
    FieldKey::new(MeshKey::ATMOSPHERE, FieldName::VelocityX);
  const VELOCITY_Y: FieldKey =
    FieldKey::new(MeshKey::ATMOSPHERE, FieldName::VelocityY);
  const VELOCITY_Z: FieldKey =
    FieldKey::new(MeshKey::ATMOSPHERE, FieldName::VelocityZ);

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
    }
  }

  #[test]
  fn diagnostics_derive_temperature_pressure_and_velocity() {
    let mut tessera = Tessera::new();
    let mesh = Arc::new(StructuredBlock::uniform(
      [0.0; 3].into(),
      [1.0; 3],
      [1, 1, 1],
      Box::new(IdentityMap::<3>),
    ));
    let mesh_for_registry: Arc<dyn Mesh<3>> = mesh;
    tessera.register_mesh(MeshKey::ATMOSPHERE, mesh_for_registry);

    let rho = 2.0;
    let velocity = [3.0, 4.0, 5.0];
    let pressure = 120_000.0;
    let gamma = 1.4;
    let kinetic = 0.5
      * rho
      * (velocity[0] * velocity[0]
        + velocity[1] * velocity[1]
        + velocity[2] * velocity[2]);
    let energy = pressure / (gamma - 1.0) + kinetic;

    let mut pleroma = Pleroma::new();
    pleroma.register_field(
      STATE,
      SoaField::<5>::from_fn(1, |_| {
        [
          rho,
          rho * velocity[0],
          rho * velocity[1],
          rho * velocity[2],
          energy,
        ]
      }),
    );
    pleroma.register_field(TEMPERATURE, SoaField::<1>::zeros(1));
    pleroma.register_field(PRESSURE, SoaField::<1>::zeros(1));
    pleroma.register_field(VELOCITY_X, SoaField::<1>::zeros(1));
    pleroma.register_field(VELOCITY_Y, SoaField::<1>::zeros(1));
    pleroma.register_field(VELOCITY_Z, SoaField::<1>::zeros(1));

    let mut nexus = Nexus::new();
    nexus.add(
      EulerDiagnosticsStep::new(
        MeshKey::ATMOSPHERE,
        STATE,
        TEMPERATURE,
        PRESSURE,
        VELOCITY_X,
        VELOCITY_Y,
        VELOCITY_Z,
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
        1.0,
      )
      .unwrap();

    let temperature: &SoaField<1> = pleroma.read(TEMPERATURE).unwrap();
    let pressure_field: &SoaField<1> = pleroma.read(PRESSURE).unwrap();
    let u: &SoaField<1> = pleroma.read(VELOCITY_X).unwrap();
    let v: &SoaField<1> = pleroma.read(VELOCITY_Y).unwrap();
    let w: &SoaField<1> = pleroma.read(VELOCITY_Z).unwrap();

    assert_eq!(temperature.state(CellId::from(0))[0], 200.0);
    assert_eq!(pressure_field.state(CellId::from(0))[0], pressure);
    assert_eq!(u.state(CellId::from(0))[0], velocity[0]);
    assert_eq!(v.state(CellId::from(0))[0], velocity[1]);
    assert_eq!(w.state(CellId::from(0))[0], velocity[2]);
  }
}
