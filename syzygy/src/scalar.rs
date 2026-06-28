// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use nexus::{CellView, FieldKey, FieldStorage, SoaField, Stage, StageContext};
use tessera::world_mesh::Tessera;
use utility::{
  domain::CellId,
  error::{AetherError, AetherResult},
};

use crate::{error::SyzygyError, stencil::CouplingStencil};

/// Minimal scalar coupling law for validating cross-mesh data flow.
///
/// For every stencil entry, this relaxes the target cell toward the source
/// cell:
///
/// `target += rate_per_second * dt * (source - target)`
///
/// This is not intended to be a final physical law; it is the first Syzygy
/// stage shape that proves ownership and scheduling.
pub struct ScalarRelaxation {
  stencil: CouplingStencil,
  source: FieldKey,
  target: FieldKey,
  rate_per_second: f64,
  reads: [FieldKey; 2],
  writes: [FieldKey; 1],
}

impl ScalarRelaxation {
  pub fn new(
    stencil: CouplingStencil,
    source: FieldKey,
    target: FieldKey,
    rate_per_second: f64,
  ) -> AetherResult<Self> {
    validate_rate(rate_per_second)?;
    validate_field_meshes(
      source.mesh(),
      target.mesh(),
      stencil.source_mesh(),
      stencil.target_mesh(),
    )?;
    Ok(Self {
      stencil,
      source,
      target,
      rate_per_second,
      reads: [source, target],
      writes: [target],
    })
  }

  pub fn from_coupler(
    tessera: &Tessera,
    coupler_index: usize,
    source: FieldKey,
    target: FieldKey,
    rate_per_second: f64,
  ) -> AetherResult<Self> {
    let stencil = CouplingStencil::from_tessera_coupler(
      tessera,
      coupler_index,
      source.mesh(),
      target.mesh(),
    )?;
    Self::new(stencil, source, target, rate_per_second)
  }

  pub fn stencil(&self) -> &CouplingStencil {
    &self.stencil
  }

  pub fn source(&self) -> FieldKey {
    self.source
  }

  pub fn target(&self) -> FieldKey {
    self.target
  }

  pub fn rate_per_second(&self) -> f64 {
    self.rate_per_second
  }
}

impl Stage for ScalarRelaxation {
  fn name(&self) -> &'static str {
    "syzygy_scalar_relaxation"
  }

  fn reads(&self) -> &[FieldKey] {
    &self.reads
  }

  fn writes(&self) -> &[FieldKey] {
    &self.writes
  }

  fn run(&mut self, mut ctx: StageContext<'_>) -> AetherResult<()> {
    let coefficient = self.rate_per_second * ctx.world.dt;
    let updates = {
      let source: &SoaField<1> =
        ctx.world.fields.read(self.source).ok_or_else(|| {
          AetherError::new(SyzygyError::MissingReadField)
            .context(format!("{:?}", self.source))
        })?;
      let target: &SoaField<1> =
        ctx.world.fields.read(self.target).ok_or_else(|| {
          AetherError::new(SyzygyError::MissingReadField)
            .context(format!("{:?}", self.target))
        })?;

      proposed_updates(&self.stencil, source, target, coefficient)?
    };

    let target: &mut SoaField<1> =
      ctx.world.fields.write(self.target).ok_or_else(|| {
        AetherError::new(SyzygyError::MissingWriteField)
          .context(format!("{:?}", self.target))
      })?;
    for (cell, value) in updates {
      target.write(cell, &[value]);
    }

    Ok(())
  }
}

fn validate_rate(rate_per_second: f64) -> AetherResult<()> {
  if rate_per_second.is_finite() {
    Ok(())
  } else {
    Err(
      AetherError::new(SyzygyError::InvalidRate)
        .context(format!("rate_per_second = {}", rate_per_second)),
    )
  }
}

fn validate_field_meshes(
  source: utility::domain::MeshKey,
  target: utility::domain::MeshKey,
  stencil_source: utility::domain::MeshKey,
  stencil_target: utility::domain::MeshKey,
) -> AetherResult<()> {
  if source == stencil_source && target == stencil_target {
    Ok(())
  } else {
    Err(
      AetherError::new(SyzygyError::FieldMeshMismatch).context(format!(
        "source {:?}, target {:?}, stencil {:?} -> {:?}",
        source, target, stencil_source, stencil_target
      )),
    )
  }
}

fn proposed_updates(
  stencil: &CouplingStencil,
  source: &SoaField<1>,
  target: &SoaField<1>,
  coefficient: f64,
) -> AetherResult<Vec<(CellId, f64)>> {
  let mut sums = vec![0.0; target.len()];
  let mut weights = vec![0.0; target.len()];

  for entry in stencil.entries() {
    ensure_cell_in_bounds(entry.source_cell, source.len(), "source")?;
    ensure_cell_in_bounds(entry.target_cell, target.len(), "target")?;

    let source_value = source.state(entry.source_cell).as_state()[0];
    let target_value = target.state(entry.target_cell).as_state()[0];
    let proposed = target_value + coefficient * (source_value - target_value);
    let index = entry.target_cell.index();
    // Gather: weight each source by its share of the target's interface area.
    sums[index] += proposed * entry.target_weight;
    weights[index] += entry.target_weight;
  }

  Ok(
    sums
      .into_iter()
      .zip(weights)
      .enumerate()
      .filter_map(|(index, (sum, weight))| {
        (weight > 0.0).then(|| (CellId::from(index), sum / weight))
      })
      .collect(),
  )
}

fn ensure_cell_in_bounds(
  cell: CellId,
  len: usize,
  label: &'static str,
) -> AetherResult<()> {
  if cell.index() < len {
    Ok(())
  } else {
    Err(
      AetherError::new(SyzygyError::CellOutOfBounds).context(format!(
        "{} cell {} outside field length {}",
        label,
        cell.index(),
        len
      )),
    )
  }
}

#[cfg(test)]
mod tests {
  use std::sync::Arc;

  use nexus::{FieldName, MeshKey, Nexus, Pleroma, WorldConstants, WorldId};
  use tessera::{
    coupling::MeshCoupler, cube_sphere::CubeSphere, geometry::CellGeometry,
    mesh::Mesh, radial_stack::RadialStackCoupler, world_mesh::Tessera,
  };
  use utility::thread::pool::Pool;

  use super::*;

  const SURFACE_TEMPERATURE: FieldKey =
    FieldKey::new(MeshKey::SURFACE, FieldName::Temperature);
  const ATMOSPHERE_TEMPERATURE: FieldKey =
    FieldKey::new(MeshKey::ATMOSPHERE, FieldName::Temperature);

  #[test]
  fn scalar_relaxation_uses_tessera_coupler_geometry() {
    let angular_dims = [2, 2];
    let surface_layers = 2;
    let atmosphere_layers = 2;
    let surface = Arc::new(CubeSphere::new(
      [angular_dims[0], angular_dims[1], surface_layers],
      0.9,
      1.0,
    ));
    let atmosphere = Arc::new(CubeSphere::new(
      [angular_dims[0], angular_dims[1], atmosphere_layers],
      1.0,
      1.2,
    ));
    let surface_cell_count = surface.cell_count();
    let atmosphere_cell_count = atmosphere.cell_count();

    let mut tessera = Tessera::new();
    let surface_for_registry: Arc<dyn Mesh<3>> = surface;
    let atmosphere_for_registry: Arc<dyn Mesh<3>> = atmosphere;
    tessera.register_mesh(MeshKey::SURFACE, surface_for_registry);
    tessera.register_mesh(MeshKey::ATMOSPHERE, atmosphere_for_registry);
    let coupler =
      RadialStackCoupler::new(angular_dims, surface_layers, atmosphere_layers);
    let pair_count = coupler.pairs().len();
    let coupler_index =
      tessera.add_coupler(MeshKey::SURFACE, MeshKey::ATMOSPHERE, coupler);

    let mut pleroma = Pleroma::new();
    pleroma.register_field(
      SURFACE_TEMPERATURE,
      SoaField::<1>::from_fn(surface_cell_count, |_| [300.0]),
    );
    pleroma.register_field(
      ATMOSPHERE_TEMPERATURE,
      SoaField::<1>::zeros(atmosphere_cell_count),
    );

    let mut nexus = Nexus::new();
    nexus.add(
      ScalarRelaxation::from_coupler(
        &tessera,
        coupler_index,
        SURFACE_TEMPERATURE,
        ATMOSPHERE_TEMPERATURE,
        1.0,
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
        0.5,
      )
      .unwrap();

    let atmosphere: &SoaField<1> =
      pleroma.read(ATMOSPHERE_TEMPERATURE).unwrap();
    let stencil = CouplingStencil::from_tessera_coupler(
      &tessera,
      coupler_index,
      MeshKey::SURFACE,
      MeshKey::ATMOSPHERE,
    )
    .unwrap();
    let touched = stencil
      .entries()
      .iter()
      .filter(|entry| {
        atmosphere
          .state(entry.target_cell)
          .as_state()
          .first()
          .is_some_and(|value| (*value - 150.0).abs() < 1e-10)
      })
      .count();
    assert_eq!(touched, pair_count);
  }
}
