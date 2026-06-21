// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use nexus::{CellView, FieldKey, FieldStorage, SoaField, Stage, StageContext};
use tessera::world_mesh::Tessera;
use utility::{
  domain::CellId,
  error::{AetherError, AetherResult},
};

use crate::{error::SyzygyError, stencil::CouplingStencil};

/// Computes a target-side scalar tendency from an interface stencil.
///
/// The stage reads a source scalar field and a target scalar field, then
/// overwrites a target-mesh tendency field:
///
/// `tendency += conductance * weight * area / distance * (source - target)`
///
/// This keeps Syzygy responsible for cross-physics exchange terms without
/// directly mutating the target prognostic state.
pub struct ScalarInterfaceFlux {
  stencil: CouplingStencil,
  source: FieldKey,
  target: FieldKey,
  tendency: FieldKey,
  conductance: f64,
  reads: [FieldKey; 2],
  writes: [FieldKey; 1],
}

impl ScalarInterfaceFlux {
  pub fn new(
    stencil: CouplingStencil,
    source: FieldKey,
    target: FieldKey,
    tendency: FieldKey,
    conductance: f64,
  ) -> AetherResult<Self> {
    validate_conductance(conductance)?;
    validate_field_meshes(
      source,
      target,
      tendency,
      stencil.source_mesh(),
      stencil.target_mesh(),
    )?;
    Ok(Self {
      stencil,
      source,
      target,
      tendency,
      conductance,
      reads: [source, target],
      writes: [tendency],
    })
  }

  pub fn from_coupler(
    tessera: &Tessera,
    coupler_index: usize,
    source: FieldKey,
    target: FieldKey,
    tendency: FieldKey,
    conductance: f64,
  ) -> AetherResult<Self> {
    let stencil = CouplingStencil::from_tessera_coupler(
      tessera,
      coupler_index,
      source.mesh(),
      target.mesh(),
    )?;
    Self::new(stencil, source, target, tendency, conductance)
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

  pub fn tendency(&self) -> FieldKey {
    self.tendency
  }

  pub fn conductance(&self) -> f64 {
    self.conductance
  }
}

impl Stage for ScalarInterfaceFlux {
  fn name(&self) -> &'static str {
    "syzygy_scalar_interface_flux"
  }

  fn reads(&self) -> &[FieldKey] {
    &self.reads
  }

  fn writes(&self) -> &[FieldKey] {
    &self.writes
  }

  fn run(&mut self, mut ctx: StageContext<'_>) -> AetherResult<()> {
    let tendencies = {
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

      compute_tendencies(&self.stencil, source, target, self.conductance)?
    };

    let tendency: &mut SoaField<1> =
      ctx.world.fields.write(self.tendency).ok_or_else(|| {
        AetherError::new(SyzygyError::MissingWriteField)
          .context(format!("{:?}", self.tendency))
      })?;

    if tendency.len() != tendencies.len() {
      return Err(AetherError::new(SyzygyError::CellOutOfBounds).context(
        format!(
          "tendency field length {} does not match target field length {}",
          tendency.len(),
          tendencies.len()
        ),
      ));
    }

    for (cell, value) in tendencies.into_iter().enumerate() {
      tendency.write(CellId::from(cell), &[value]);
    }

    Ok(())
  }
}

/// Deposits a scaled source-side scalar onto a target-side scalar across an
/// interface stencil — `target[t] += scale · weight · source[s]`.
///
/// Unlike [`ScalarInterfaceFlux`] (a gradient-driven exchange written to a
/// separate tendency), this *accumulates directly* onto an existing target
/// field, so it composes with other contributors to that field (e.g. radiation
/// writing the ocean's net surface flux first). It is the conservative air–sea
/// latent-heat sink: the atmosphere's evaporative mass flux, scaled by
/// `-L_v · Δz`, debits the ocean surface energy the vapour will later release
/// on condensation.
pub struct ScalarInterfaceDeposition {
  stencil: CouplingStencil,
  source: FieldKey,
  target: FieldKey,
  scale: f64,
  reads: [FieldKey; 1],
  writes: [FieldKey; 1],
}

impl ScalarInterfaceDeposition {
  pub fn new(
    stencil: CouplingStencil,
    source: FieldKey,
    target: FieldKey,
    scale: f64,
  ) -> AetherResult<Self> {
    validate_conductance(scale)?;
    if source.mesh() != stencil.source_mesh()
      || target.mesh() != stencil.target_mesh()
    {
      return Err(AetherError::new(SyzygyError::FieldMeshMismatch).context(
        format!(
          "source {:?}, target {:?}, stencil {:?} -> {:?}",
          source,
          target,
          stencil.source_mesh(),
          stencil.target_mesh()
        ),
      ));
    }
    Ok(Self {
      stencil,
      source,
      target,
      scale,
      reads: [source],
      writes: [target],
    })
  }

  pub fn from_coupler(
    tessera: &Tessera,
    coupler_index: usize,
    source: FieldKey,
    target: FieldKey,
    scale: f64,
  ) -> AetherResult<Self> {
    let stencil = CouplingStencil::from_tessera_coupler(
      tessera,
      coupler_index,
      source.mesh(),
      target.mesh(),
    )?;
    Self::new(stencil, source, target, scale)
  }
}

impl Stage for ScalarInterfaceDeposition {
  fn name(&self) -> &'static str {
    "syzygy_scalar_interface_deposition"
  }

  fn reads(&self) -> &[FieldKey] {
    &self.reads
  }

  fn writes(&self) -> &[FieldKey] {
    &self.writes
  }

  fn run(&mut self, mut ctx: StageContext<'_>) -> AetherResult<()> {
    // Accumulate per target cell from the source field first.
    let deposits = {
      let source: &SoaField<1> =
        ctx.world.fields.read(self.source).ok_or_else(|| {
          AetherError::new(SyzygyError::MissingReadField)
            .context(format!("{:?}", self.source))
        })?;
      let mut deposits = vec![0.0; source.len().max(1)];
      let mut target_len = 0usize;
      for entry in self.stencil.entries() {
        ensure_cell_in_bounds(entry.source_cell, source.len(), "source")?;
        target_len = target_len.max(entry.target_cell.index() + 1);
        if deposits.len() < target_len {
          deposits.resize(target_len, 0.0);
        }
        let s = source.state(entry.source_cell).as_state()[0];
        deposits[entry.target_cell.index()] += self.scale * entry.weight * s;
      }
      deposits
    };

    let target: &mut SoaField<1> =
      ctx.world.fields.write(self.target).ok_or_else(|| {
        AetherError::new(SyzygyError::MissingWriteField)
          .context(format!("{:?}", self.target))
      })?;
    for (cell, add) in deposits.into_iter().enumerate() {
      if add != 0.0 && cell < target.len() {
        let current = target.state(CellId::from(cell)).as_state()[0];
        target.write(CellId::from(cell), &[current + add]);
      }
    }
    Ok(())
  }
}

fn validate_conductance(conductance: f64) -> AetherResult<()> {
  if conductance.is_finite() {
    Ok(())
  } else {
    Err(
      AetherError::new(SyzygyError::InvalidConductance)
        .context(format!("conductance = {}", conductance)),
    )
  }
}

fn validate_field_meshes(
  source: FieldKey,
  target: FieldKey,
  tendency: FieldKey,
  stencil_source: utility::domain::MeshKey,
  stencil_target: utility::domain::MeshKey,
) -> AetherResult<()> {
  if source.mesh() == stencil_source
    && target.mesh() == stencil_target
    && tendency.mesh() == stencil_target
  {
    Ok(())
  } else {
    Err(
      AetherError::new(SyzygyError::FieldMeshMismatch).context(format!(
        "source {:?}, target {:?}, tendency {:?}, stencil {:?} -> {:?}",
        source, target, tendency, stencil_source, stencil_target
      )),
    )
  }
}

fn compute_tendencies(
  stencil: &CouplingStencil,
  source: &SoaField<1>,
  target: &SoaField<1>,
  conductance: f64,
) -> AetherResult<Vec<f64>> {
  let mut tendencies = vec![0.0; target.len()];

  for entry in stencil.entries() {
    ensure_cell_in_bounds(entry.source_cell, source.len(), "source")?;
    ensure_cell_in_bounds(entry.target_cell, target.len(), "target")?;

    let source_value = source.state(entry.source_cell).as_state()[0];
    let target_value = target.state(entry.target_cell).as_state()[0];
    let distance = entry.distance.max(f64::EPSILON);
    let exchange = conductance * entry.weight * entry.area / distance
      * (source_value - target_value);
    tendencies[entry.target_cell.index()] += exchange;
  }

  Ok(tendencies)
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
    cube_sphere::CubeSphere, geometry::CellGeometry, mesh::Mesh,
    radial_stack::RadialStackCoupler, world_mesh::Tessera,
  };
  use utility::thread::pool::Pool;

  use super::*;

  const SURFACE_TEMPERATURE: FieldKey =
    FieldKey::new(MeshKey::SURFACE, FieldName::Temperature);
  const ATMOSPHERE_TEMPERATURE: FieldKey =
    FieldKey::new(MeshKey::ATMOSPHERE, FieldName::Temperature);
  const ATMOSPHERE_TEMPERATURE_TENDENCY: FieldKey =
    FieldKey::new(MeshKey::ATMOSPHERE, FieldName::TemperatureTendency);

  #[test]
  fn scalar_interface_flux_writes_target_tendency_without_mutating_target() {
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
    let coupler_index = tessera.add_coupler(
      MeshKey::SURFACE,
      MeshKey::ATMOSPHERE,
      RadialStackCoupler::new(angular_dims, surface_layers, atmosphere_layers),
    );

    let mut pleroma = Pleroma::new();
    pleroma.register_field(
      SURFACE_TEMPERATURE,
      SoaField::<1>::from_fn(surface_cell_count, |_| [300.0]),
    );
    pleroma.register_field(
      ATMOSPHERE_TEMPERATURE,
      SoaField::<1>::from_fn(atmosphere_cell_count, |_| [250.0]),
    );
    pleroma.register_field(
      ATMOSPHERE_TEMPERATURE_TENDENCY,
      SoaField::<1>::from_fn(atmosphere_cell_count, |_| [-1.0]),
    );

    let mut nexus = Nexus::new();
    nexus.add(
      ScalarInterfaceFlux::from_coupler(
        &tessera,
        coupler_index,
        SURFACE_TEMPERATURE,
        ATMOSPHERE_TEMPERATURE,
        ATMOSPHERE_TEMPERATURE_TENDENCY,
        0.5,
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
        1.0,
      )
      .unwrap();

    let atmosphere: &SoaField<1> =
      pleroma.read(ATMOSPHERE_TEMPERATURE).unwrap();
    let tendency: &SoaField<1> =
      pleroma.read(ATMOSPHERE_TEMPERATURE_TENDENCY).unwrap();

    assert_eq!(atmosphere.state(CellId::from(0)).as_state(), &[250.0]);
    assert!(
      tendency
        .component(0)
        .as_ref()
        .iter()
        .any(|value| *value > 0.0),
      "warm surface should produce positive atmosphere-side tendencies"
    );
    assert!(
      tendency
        .component(0)
        .as_ref()
        .iter()
        .any(|value| *value == 0.0),
      "uncoupled atmosphere cells should be reset to zero"
    );
  }

  #[test]
  fn deposition_accumulates_scaled_source_onto_existing_target() {
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
    let coupler_index = tessera.add_coupler(
      MeshKey::SURFACE,
      MeshKey::ATMOSPHERE,
      RadialStackCoupler::new(angular_dims, surface_layers, atmosphere_layers),
    );

    // A source flux on the atmosphere, an existing (radiation-written) value on
    // the surface target — deposition must add, not overwrite.
    let atmosphere_flux =
      FieldKey::new(MeshKey::ATMOSPHERE, FieldName::EvaporationFlux);
    let surface_net_flux =
      FieldKey::new(MeshKey::SURFACE, FieldName::NetSurfaceFlux);
    let mut pleroma = Pleroma::new();
    pleroma.register_field(
      atmosphere_flux,
      SoaField::<1>::from_fn(atmosphere_cell_count, |_| [2.0]),
    );
    pleroma.register_field(
      surface_net_flux,
      SoaField::<1>::from_fn(surface_cell_count, |_| [100.0]),
    );

    let mut nexus = Nexus::new();
    nexus.add(
      ScalarInterfaceDeposition::from_coupler(
        &tessera,
        coupler_index,
        atmosphere_flux,
        surface_net_flux,
        -3.0,
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
        1.0,
      )
      .unwrap();

    let net_flux: &SoaField<1> = pleroma.read(surface_net_flux).unwrap();
    // Surface cells paired across the interface get 100 + (-3·1·2) = 94;
    // unpaired (deep) cells keep their original 100 (add, not overwrite).
    let values: Vec<f64> =
      net_flux.component(0).as_ref().iter().copied().collect();
    assert!(
      values.iter().any(|v| (v - 94.0).abs() < 1e-9),
      "coupled surface cells should be debited to 94, got {values:?}"
    );
    assert!(
      values.iter().any(|v| (v - 100.0).abs() < 1e-9),
      "uncoupled surface cells must keep their existing value"
    );
  }
}
