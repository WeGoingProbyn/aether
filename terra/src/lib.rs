// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use nexus::{
  FieldKey, FieldName, FieldStorage, MeshKey, Nexus, Pleroma, SoaField, Stage,
  StageContext, StageId,
};
use tessera::mesh::Mesh;
use utility::{
  domain::CellId,
  error::{AetherError, AetherResult, ErrorDomain},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SurfaceFields {
  pub temperature: FieldKey,
}

impl SurfaceFields {
  pub const fn for_mesh(mesh: MeshKey) -> Self {
    Self {
      temperature: FieldKey::new(mesh, FieldName::Temperature),
    }
  }
}

#[derive(Clone, Debug)]
pub struct SurfaceThermalModel {
  mesh: MeshKey,
  fields: SurfaceFields,
  initial_temperature: f64,
  target_temperature: f64,
}

impl SurfaceThermalModel {
  pub fn new(mesh: MeshKey) -> Self {
    Self {
      mesh,
      fields: SurfaceFields::for_mesh(mesh),
      initial_temperature: 288.0,
      target_temperature: 288.0,
    }
  }

  pub fn with_fields(mut self, fields: SurfaceFields) -> Self {
    self.fields = fields;
    self
  }

  pub fn with_initial_temperature(mut self, temperature: f64) -> Self {
    self.initial_temperature = temperature;
    self
  }

  pub fn with_target_temperature(mut self, temperature: f64) -> Self {
    self.target_temperature = temperature;
    self
  }

  pub fn mesh(&self) -> MeshKey {
    self.mesh
  }

  pub fn fields(&self) -> SurfaceFields {
    self.fields
  }

  pub fn register_fields<M>(
    &self,
    pleroma: &mut Pleroma,
    mesh: &M,
  ) -> AetherResult<()>
  where
    M: Mesh<3> + ?Sized,
  {
    self.validate()?;
    let field =
      SoaField::<1>::from_fn(mesh.cell_count(), |_| [self.initial_temperature]);
    pleroma.register_field(self.fields.temperature, field);
    Ok(())
  }

  pub fn add_stages(&self, nexus: &mut Nexus) -> AetherResult<StageId> {
    self.validate()?;
    Ok(nexus.add(SurfaceTemperatureRelaxationStep::new(
      self.mesh,
      self.fields.temperature,
      self.target_temperature,
    )?))
  }

  fn validate(&self) -> AetherResult<()> {
    if self.fields.temperature.mesh() != self.mesh {
      return Err(
        AetherError::new(TerraError::FieldMeshMismatch)
          .context(format!("mesh {:?}, fields {:?}", self.mesh, self.fields)),
      );
    }

    if self.initial_temperature.is_finite()
      && self.target_temperature.is_finite()
      && self.initial_temperature > 0.0
      && self.target_temperature > 0.0
    {
      Ok(())
    } else {
      Err(
        AetherError::new(TerraError::InvalidSurfaceTemperature).context(
          format!(
            "initial {}, target {}",
            self.initial_temperature, self.target_temperature
          ),
        ),
      )
    }
  }
}

impl Default for SurfaceThermalModel {
  fn default() -> Self {
    Self::new(MeshKey::SURFACE)
  }
}

/// Placeholder surface temperature stage.
///
/// This intentionally mirrors the old sandbox dummy heating behavior while
/// keeping surface-owned state and stages inside Terra. It relaxes immediately
/// to `target_temperature + dt`, giving the atmosphere coupling a non-zero
/// validation signal without pretending this is a physical surface model.
pub struct SurfaceTemperatureRelaxationStep {
  mesh: MeshKey,
  temperature: FieldKey,
  target_temperature: f64,
  reads: [FieldKey; 1],
  writes: [FieldKey; 1],
}

impl SurfaceTemperatureRelaxationStep {
  pub fn new(
    mesh: MeshKey,
    temperature: FieldKey,
    target_temperature: f64,
  ) -> AetherResult<Self> {
    if temperature.mesh() != mesh {
      return Err(
        AetherError::new(TerraError::FieldMeshMismatch)
          .context(format!("mesh {:?}, temperature {:?}", mesh, temperature)),
      );
    }
    if !target_temperature.is_finite() || target_temperature <= 0.0 {
      return Err(
        AetherError::new(TerraError::InvalidSurfaceTemperature)
          .context(format!("target {}", target_temperature)),
      );
    }

    Ok(Self {
      mesh,
      temperature,
      target_temperature,
      reads: [temperature],
      writes: [temperature],
    })
  }

  pub fn mesh(&self) -> MeshKey {
    self.mesh
  }

  pub fn temperature(&self) -> FieldKey {
    self.temperature
  }
}

impl Stage for SurfaceTemperatureRelaxationStep {
  fn name(&self) -> &'static str {
    "terra_surface_temperature_relaxation"
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
        AetherError::new(TerraError::MissingMesh)
          .context(format!("{:?}", self.mesh))
      })?
      .cell_count();

    let field: &mut SoaField<1> =
      ctx.world.fields.write(self.temperature).ok_or_else(|| {
        AetherError::new(TerraError::MissingWriteField)
          .context(format!("{:?}", self.temperature))
      })?;
    if field.len() != mesh_cell_count {
      return Err(AetherError::new(TerraError::FieldLengthMismatch).context(
        format!(
          "temperature len {}, mesh cell count {}",
          field.len(),
          mesh_cell_count
        ),
      ));
    }

    let temperature = self.target_temperature + ctx.world.dt;
    for cell in 0..field.len() {
      field.write(CellId::from(cell), &[temperature]);
    }

    Ok(())
  }
}

#[derive(Debug)]
pub enum TerraError {
  MissingMesh,
  MissingWriteField,
  FieldMeshMismatch,
  FieldLengthMismatch,
  InvalidSurfaceTemperature,
}

impl ErrorDomain for TerraError {
  fn domain(&self) -> &str {
    "terra"
  }
}

impl std::fmt::Display for TerraError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      TerraError::MissingMesh => {
        write!(f, "surface mesh is not registered in tessera")
      }
      TerraError::MissingWriteField => {
        write!(f, "declared write field is missing or has the wrong type")
      }
      TerraError::FieldMeshMismatch => {
        write!(f, "stage fields must live on the stage mesh")
      }
      TerraError::FieldLengthMismatch => {
        write!(f, "field and mesh cell counts do not match")
      }
      TerraError::InvalidSurfaceTemperature => {
        write!(f, "surface temperature is non-physical")
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use std::sync::Arc;

  use nexus::{Pleroma, WorldConstants, WorldId};
  use tessera::{
    cube_sphere::{CubeSphere, CubeSphereShellSpec},
    geometry::CellGeometry,
    world_mesh::Tessera,
  };
  use utility::thread::pool::Pool;

  use super::*;

  #[test]
  fn surface_thermal_model_registers_field_and_updates_temperature() {
    let mesh = Arc::new(CubeSphere::shell(CubeSphereShellSpec::uniform(
      [2, 2, 1],
      0.9,
      1.0,
    )));
    let mut tessera = Tessera::new();
    tessera.register_mesh(MeshKey::SURFACE, mesh.clone());

    let model = SurfaceThermalModel::default()
      .with_initial_temperature(280.0)
      .with_target_temperature(288.0);
    let fields = model.fields();

    let mut pleroma = Pleroma::new();
    model.register_fields(&mut pleroma, mesh.as_ref()).unwrap();
    assert_eq!(
      pleroma.cell_count(fields.temperature),
      Some(mesh.cell_count())
    );

    let mut nexus = Nexus::new();
    let stage_id = model.add_stages(&mut nexus).unwrap();
    assert_eq!(stage_id.index(), 0);

    let mut compiled = nexus.build(&pleroma).unwrap();
    compiled
      .tick(
        WorldId(0),
        &tessera,
        &WorldConstants::default(),
        &mut pleroma,
        &Pool::default(),
        0.5,
      )
      .unwrap();

    let temperature: &SoaField<1> = pleroma.read(fields.temperature).unwrap();
    for i in 0..temperature.len() {
      assert_eq!(temperature.state(CellId::from(i))[0], 288.5);
    }
  }
}
