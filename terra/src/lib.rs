// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Surface thermal physics. Terra owns the *logic* of stepping a
//! surface temperature field forward under a net radiative flux supplied
//! by lumen on the same surface mesh. All field storage lives in
//! pleroma — terra only registers the temperature field at world setup
//! and names the flux field it consumes. It also owns first-class terrain:
//! elevation, a land/ocean/ice mask, and a per-cell albedo field.
//!
//! See `terra/docs/overview.md` for the terrain fields and the albedo seam.

use nexus::{
  FieldKey, FieldName, FieldStorage, MeshKey, Nexus, Pleroma, SoaField, Stage,
  StageContext, StageId,
};
use tessera::mesh::Mesh;
use utility::{
  domain::CellId,
  error::{AetherError, AetherResult, ErrorDomain},
};

pub mod terrain;
pub use terrain::{
  AlbedoTable, SurfaceAlbedoStep, TerrainFields, TerrainModel, TerrainSample,
  earthlike_terrain,
};
pub use utility::domain::SurfaceClass;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SurfaceFields {
  pub temperature: FieldKey,
  pub net_flux: FieldKey,
}

impl SurfaceFields {
  pub const fn for_mesh(mesh: MeshKey) -> Self {
    Self {
      temperature: FieldKey::new(mesh, FieldName::Temperature),
      net_flux: FieldKey::new(mesh, FieldName::NetSurfaceFlux),
    }
  }
}

/// Surface thermal slab driven by a net radiative flux (W/m²).
///
/// `dT/dt = NetSurfaceFlux / heat_capacity_per_area`
///
/// `heat_capacity_per_area` (J/(K·m²)) collapses slab density, specific
/// heat and depth into one tunable knob. Earth's ocean mixed layer is
/// O(1e8); a thin solid slab is O(1e5–1e6). Pick a value matching the
/// time-scale you want to observe.
#[derive(Clone, Debug)]
pub struct SurfaceThermalModel {
  mesh: MeshKey,
  fields: SurfaceFields,
  initial_temperature: f64,
  heat_capacity_per_area: f64,
}

impl SurfaceThermalModel {
  pub fn new(mesh: MeshKey) -> Self {
    Self {
      mesh,
      fields: SurfaceFields::for_mesh(mesh),
      initial_temperature: 288.0,
      heat_capacity_per_area: 1.0e6,
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

  pub fn with_heat_capacity_per_area(mut self, capacity: f64) -> Self {
    self.heat_capacity_per_area = capacity;
    self
  }

  pub fn mesh(&self) -> MeshKey {
    self.mesh
  }

  pub fn fields(&self) -> SurfaceFields {
    self.fields
  }

  pub fn heat_capacity_per_area(&self) -> f64 {
    self.heat_capacity_per_area
  }

  /// Register the surface temperature field. The net-flux field is
  /// expected to be registered by lumen (the producer); terra only
  /// reads it.
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
    Ok(nexus.add(SurfaceEnergyBalanceStep::new(
      self.mesh,
      self.fields.temperature,
      self.fields.net_flux,
      self.heat_capacity_per_area,
    )?))
  }

  fn validate(&self) -> AetherResult<()> {
    if self.fields.temperature.mesh() != self.mesh
      || self.fields.net_flux.mesh() != self.mesh
    {
      return Err(
        AetherError::new(TerraError::FieldMeshMismatch)
          .context(format!("mesh {:?}, fields {:?}", self.mesh, self.fields)),
      );
    }

    if !self.initial_temperature.is_finite() || self.initial_temperature <= 0.0
    {
      return Err(
        AetherError::new(TerraError::InvalidSurfaceTemperature)
          .context(format!("initial_temperature {}", self.initial_temperature)),
      );
    }
    if !self.heat_capacity_per_area.is_finite()
      || self.heat_capacity_per_area <= 0.0
    {
      return Err(AetherError::new(TerraError::InvalidHeatCapacity).context(
        format!("heat_capacity_per_area {}", self.heat_capacity_per_area),
      ));
    }
    Ok(())
  }
}

impl Default for SurfaceThermalModel {
  fn default() -> Self {
    Self::new(MeshKey::SURFACE)
  }
}

/// Forward-Euler surface energy balance.
///
/// Reads `net_flux` (W/m²) and the current `temperature`, advances:
///
/// `T <- T + dt * net_flux / heat_capacity_per_area`
pub struct SurfaceEnergyBalanceStep {
  mesh: MeshKey,
  temperature: FieldKey,
  net_flux: FieldKey,
  heat_capacity_per_area: f64,
  reads: [FieldKey; 2],
  writes: [FieldKey; 1],
}

impl SurfaceEnergyBalanceStep {
  pub fn new(
    mesh: MeshKey,
    temperature: FieldKey,
    net_flux: FieldKey,
    heat_capacity_per_area: f64,
  ) -> AetherResult<Self> {
    if temperature.mesh() != mesh || net_flux.mesh() != mesh {
      return Err(AetherError::new(TerraError::FieldMeshMismatch).context(
        format!(
          "mesh {:?}, temperature {:?}, net_flux {:?}",
          mesh, temperature, net_flux
        ),
      ));
    }
    if !heat_capacity_per_area.is_finite() || heat_capacity_per_area <= 0.0 {
      return Err(
        AetherError::new(TerraError::InvalidHeatCapacity)
          .context(format!("{}", heat_capacity_per_area)),
      );
    }

    Ok(Self {
      mesh,
      temperature,
      net_flux,
      heat_capacity_per_area,
      reads: [net_flux, temperature],
      writes: [temperature],
    })
  }

  pub fn mesh(&self) -> MeshKey {
    self.mesh
  }

  pub fn temperature(&self) -> FieldKey {
    self.temperature
  }

  pub fn net_flux(&self) -> FieldKey {
    self.net_flux
  }

  pub fn heat_capacity_per_area(&self) -> f64 {
    self.heat_capacity_per_area
  }
}

impl Stage for SurfaceEnergyBalanceStep {
  fn name(&self) -> &'static str {
    "terra_surface_energy_balance"
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

    let dt = ctx.world.dt;
    if !dt.is_finite() || dt <= 0.0 {
      return Err(
        AetherError::new(TerraError::InvalidTimeStep)
          .context(format!("dt {}", dt)),
      );
    }

    let coefficient = dt / self.heat_capacity_per_area;

    let flux_values: Vec<f64> = {
      let flux: &SoaField<1> =
        ctx.world.fields.read(self.net_flux).ok_or_else(|| {
          AetherError::new(TerraError::MissingReadField)
            .context(format!("{:?}", self.net_flux))
        })?;
      if flux.len() != mesh_cell_count {
        return Err(AetherError::new(TerraError::FieldLengthMismatch).context(
          format!(
            "net_flux len {}, mesh cell count {}",
            flux.len(),
            mesh_cell_count
          ),
        ));
      }
      flux.component(0).as_ref().to_vec()
    };

    let temperature: &mut SoaField<1> =
      ctx.world.fields.write(self.temperature).ok_or_else(|| {
        AetherError::new(TerraError::MissingWriteField)
          .context(format!("{:?}", self.temperature))
      })?;
    if temperature.len() != mesh_cell_count {
      return Err(AetherError::new(TerraError::FieldLengthMismatch).context(
        format!(
          "temperature len {}, mesh cell count {}",
          temperature.len(),
          mesh_cell_count
        ),
      ));
    }

    for (cell, flux) in flux_values.into_iter().enumerate() {
      let cell = CellId::from(cell);
      let mut t = temperature.state(cell)[0];
      t += coefficient * flux;
      if !t.is_finite() {
        return Err(
          AetherError::new(TerraError::InvalidSurfaceTemperature)
            .context(format!("cell {} temperature {}", cell.index(), t)),
        );
      }
      temperature.write(cell, &[t]);
    }

    Ok(())
  }
}

#[derive(Debug)]
pub enum TerraError {
  MissingMesh,
  MissingReadField,
  MissingWriteField,
  FieldMeshMismatch,
  FieldLengthMismatch,
  InvalidSurfaceTemperature,
  InvalidHeatCapacity,
  InvalidTimeStep,
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
      TerraError::MissingReadField => {
        write!(f, "declared read field is missing or has the wrong type")
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
      TerraError::InvalidHeatCapacity => {
        write!(f, "heat_capacity_per_area must be finite and positive")
      }
      TerraError::InvalidTimeStep => {
        write!(f, "dt must be finite and positive")
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
  fn surface_thermal_model_advances_temperature_under_constant_flux() {
    let mesh = Arc::new(CubeSphere::shell(CubeSphereShellSpec::uniform(
      [2, 2, 1],
      0.9,
      1.0,
    )));
    let mut tessera = Tessera::new();
    tessera.register_mesh(MeshKey::SURFACE, mesh.clone());

    let model = SurfaceThermalModel::default()
      .with_initial_temperature(280.0)
      .with_heat_capacity_per_area(1.0e6);
    let fields = model.fields();

    let mut pleroma = Pleroma::new();
    model.register_fields(&mut pleroma, mesh.as_ref()).unwrap();
    pleroma.register_field(
      fields.net_flux,
      SoaField::<1>::from_fn(mesh.cell_count(), |_| [1000.0]),
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
        100.0,
      )
      .unwrap();

    let expected = 280.0 + 100.0 * 1000.0 / 1.0e6;
    let temperature: &SoaField<1> = pleroma.read(fields.temperature).unwrap();
    for i in 0..temperature.len() {
      assert!(
        (temperature.state(CellId::from(i))[0] - expected).abs() < 1.0e-9,
        "cell {} expected {} got {}",
        i,
        expected,
        temperature.state(CellId::from(i))[0]
      );
    }
  }
}
