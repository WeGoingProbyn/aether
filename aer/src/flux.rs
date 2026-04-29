// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use nexus::{
  CellView, FieldKey, FieldStorage, MeshKey, SoaField, Stage, StageContext,
};
use utility::{
  domain::CellId,
  error::{AetherError, AetherResult},
};

use crate::error::AerError;

/// Minimal atmosphere integration stage for scalar temperature tendencies.
///
/// This is intentionally simple: Syzygy writes a target-side tendency, and
/// Aer applies it to the prognostic temperature field with explicit Euler:
///
/// `temperature += dt * temperature_tendency`
pub struct TemperatureTendencyStep {
  mesh: MeshKey,
  temperature: FieldKey,
  tendency: FieldKey,
  reads: [FieldKey; 1],
  writes: [FieldKey; 1],
}

impl TemperatureTendencyStep {
  pub fn new(
    mesh: MeshKey,
    temperature: FieldKey,
    tendency: FieldKey,
  ) -> AetherResult<Self> {
    if temperature.mesh() != mesh || tendency.mesh() != mesh {
      return Err(AetherError::new(AerError::FieldMeshMismatch).context(
        format!(
          "mesh {:?}, temperature {:?}, tendency {:?}",
          mesh, temperature, tendency
        ),
      ));
    }

    Ok(Self {
      mesh,
      temperature,
      tendency,
      reads: [tendency],
      writes: [temperature],
    })
  }

  pub fn mesh(&self) -> MeshKey {
    self.mesh
  }

  pub fn temperature(&self) -> FieldKey {
    self.temperature
  }

  pub fn tendency(&self) -> FieldKey {
    self.tendency
  }
}

impl Stage for TemperatureTendencyStep {
  fn name(&self) -> &'static str {
    "aer_temperature_tendency_step"
  }

  fn reads(&self) -> &[FieldKey] {
    &self.reads
  }

  fn writes(&self) -> &[FieldKey] {
    &self.writes
  }

  fn run(&mut self, mut ctx: StageContext<'_>) -> AetherResult<()> {
    let tendency_values = {
      let tendency: &SoaField<1> =
        ctx.world.fields.read(self.tendency).ok_or_else(|| {
          AetherError::new(AerError::MissingReadField)
            .context(format!("{:?}", self.tendency))
        })?;
      tendency.component(0).as_ref().to_vec()
    };

    let temperature: &mut SoaField<1> =
      ctx.world.fields.write(self.temperature).ok_or_else(|| {
        AetherError::new(AerError::MissingWriteField)
          .context(format!("{:?}", self.temperature))
      })?;

    validate_lengths(
      temperature.len(),
      tendency_values.len(),
      ctx
        .world
        .tessera
        .mesh(self.mesh)
        .map(|mesh| mesh.cell_count()),
    )?;

    for (cell, tendency) in tendency_values.into_iter().enumerate() {
      let cell = CellId::from(cell);
      let current = temperature.state(cell).as_state()[0];
      temperature.write(cell, &[current + ctx.world.dt * tendency]);
    }

    Ok(())
  }
}

fn validate_lengths(
  temperature_len: usize,
  tendency_len: usize,
  mesh_cell_count: Option<usize>,
) -> AetherResult<()> {
  if temperature_len != tendency_len {
    return Err(AetherError::new(AerError::FieldLengthMismatch).context(
      format!(
        "temperature len {}, tendency len {}",
        temperature_len, tendency_len
      ),
    ));
  }

  if let Some(mesh_cell_count) = mesh_cell_count
    && temperature_len != mesh_cell_count
  {
    return Err(AetherError::new(AerError::FieldLengthMismatch).context(
      format!(
        "temperature len {}, mesh cell count {}",
        temperature_len, mesh_cell_count
      ),
    ));
  }

  Ok(())
}

#[cfg(test)]
mod tests {
  use std::sync::Arc;

  use nexus::{FieldName, Nexus, Pleroma, WorldConstants, WorldId};
  use tessera::{
    geometry::IdentityMap,
    mesh::{Mesh, StructuredBlock},
    world_mesh::Tessera,
  };
  use utility::thread::pool::Pool;

  use super::*;

  const TEMPERATURE: FieldKey =
    FieldKey::new(MeshKey::ATMOSPHERE, FieldName::Temperature);
  const TEMPERATURE_TENDENCY: FieldKey =
    FieldKey::new(MeshKey::ATMOSPHERE, FieldName::TemperatureTendency);

  #[test]
  fn temperature_tendency_step_integrates_temperature() {
    let mut tessera = Tessera::new();
    let mesh = Arc::new(StructuredBlock::uniform(
      [0.0; 3].into(),
      [1.0; 3],
      [2, 1, 1],
      Box::new(IdentityMap::<3>),
    ));
    let mesh_for_registry: Arc<dyn Mesh<3>> = mesh;
    tessera.register_mesh(MeshKey::ATMOSPHERE, mesh_for_registry);

    let mut pleroma = Pleroma::new();
    pleroma.register_field(
      TEMPERATURE,
      SoaField::<1>::from_fn(2, |cell| [280.0 + cell.index() as f64]),
    );
    pleroma.register_field(
      TEMPERATURE_TENDENCY,
      SoaField::<1>::from_fn(2, |cell| [0.5 + cell.index() as f64]),
    );

    let mut nexus = Nexus::new();
    nexus.add(
      TemperatureTendencyStep::new(
        MeshKey::ATMOSPHERE,
        TEMPERATURE,
        TEMPERATURE_TENDENCY,
      )
      .unwrap(),
    );
    let mut compiled = nexus.build(&pleroma).unwrap();
    compiled
      .tick(
        WorldId(0),
        &tessera,
        &WorldConstants::default(),
        &mut pleroma,
        &Pool::default(),
        10.0,
      )
      .unwrap();

    let temperature: &SoaField<1> = pleroma.read(TEMPERATURE).unwrap();
    assert_eq!(temperature.state(CellId::from(0)).as_state(), &[285.0]);
    assert_eq!(temperature.state(CellId::from(1)).as_state(), &[296.0]);
  }

  #[test]
  fn constructor_rejects_wrong_mesh_fields() {
    let result = TemperatureTendencyStep::new(
      MeshKey::ATMOSPHERE,
      FieldKey::new(MeshKey::SURFACE, FieldName::Temperature),
      TEMPERATURE_TENDENCY,
    );
    assert!(result.is_err());
  }
}
