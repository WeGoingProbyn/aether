// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use nexus::{CellView, FieldKey, FieldStorage, SoaField, Stage, StageContext};
use tessera::coupling::CoupledFace;
use utility::{
  domain::{CellId, MeshKey},
  error::{AetherError, AetherResult, ErrorDomain},
};

/// Minimal scalar coupling law for validating cross-mesh data flow.
///
/// For every coupled face, this relaxes the target owner cell toward the
/// source owner cell:
///
/// `target += rate_per_second * dt * (source - target)`
///
/// This is not intended to be a final physical law; it is the first Syzygy
/// stage shape that proves ownership and scheduling.
pub struct ScalarRelaxation {
  coupler_index: usize,
  source: FieldKey,
  target: FieldKey,
  rate_per_second: f64,
  reads: [FieldKey; 2],
  writes: [FieldKey; 1],
}

impl ScalarRelaxation {
  pub fn new(
    coupler_index: usize,
    source: FieldKey,
    target: FieldKey,
    rate_per_second: f64,
  ) -> Self {
    Self {
      coupler_index,
      source,
      target,
      rate_per_second,
      reads: [source, target],
      writes: [target],
    }
  }

  pub fn coupler_index(&self) -> usize {
    self.coupler_index
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
    if !self.rate_per_second.is_finite() {
      return Err(
        AetherError::new(SyzygyError::InvalidRate)
          .context(format!("rate_per_second = {}", self.rate_per_second)),
      );
    }

    let coupler = ctx
      .world
      .tessera
      .coupler_view(self.coupler_index)
      .ok_or_else(|| {
        AetherError::new(SyzygyError::MissingCoupler)
          .context(format!("coupler index {}", self.coupler_index))
      })?;

    validate_field_meshes(
      self.source.mesh(),
      self.target.mesh(),
      coupler.mesh_a(),
      coupler.mesh_b(),
    )?;

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

      proposed_updates(
        coupler.faces(),
        self.source.mesh(),
        source,
        self.target.mesh(),
        target,
        coefficient,
      )?
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

fn validate_field_meshes(
  source: MeshKey,
  target: MeshKey,
  mesh_a: MeshKey,
  mesh_b: MeshKey,
) -> AetherResult<()> {
  let forward = source == mesh_a && target == mesh_b;
  let reverse = source == mesh_b && target == mesh_a;
  if forward || reverse {
    Ok(())
  } else {
    Err(
      AetherError::new(SyzygyError::FieldMeshMismatch).context(format!(
        "source {:?}, target {:?}, coupler {:?} <-> {:?}",
        source, target, mesh_a, mesh_b
      )),
    )
  }
}

fn proposed_updates(
  faces: impl Iterator<Item = CoupledFace>,
  source_mesh: MeshKey,
  source: &SoaField<1>,
  target_mesh: MeshKey,
  target: &SoaField<1>,
  coefficient: f64,
) -> AetherResult<Vec<(CellId, f64)>> {
  let mut sums = vec![0.0; target.len()];
  let mut counts = vec![0usize; target.len()];

  for face in faces {
    let (source_cell, target_cell) =
      source_target_cells(&face, source_mesh, target_mesh)?;
    ensure_cell_in_bounds(source_cell, source.len(), "source")?;
    ensure_cell_in_bounds(target_cell, target.len(), "target")?;

    let source_value = source.state(source_cell).as_state()[0];
    let target_value = target.state(target_cell).as_state()[0];
    let proposed = target_value + coefficient * (source_value - target_value);
    let index = target_cell.index();
    sums[index] += proposed;
    counts[index] += 1;
  }

  Ok(
    sums
      .into_iter()
      .zip(counts)
      .enumerate()
      .filter_map(|(index, (sum, count))| {
        (count > 0).then(|| (CellId::from(index), sum / count as f64))
      })
      .collect(),
  )
}

fn source_target_cells(
  face: &CoupledFace,
  source_mesh: MeshKey,
  target_mesh: MeshKey,
) -> AetherResult<(CellId, CellId)> {
  match (source_mesh, target_mesh) {
    (source, target) if source == face.mesh_a && target == face.mesh_b => {
      Ok((face.owner_a, face.owner_b))
    }
    (source, target) if source == face.mesh_b && target == face.mesh_a => {
      Ok((face.owner_b, face.owner_a))
    }
    _ => Err(AetherError::new(SyzygyError::FieldMeshMismatch).context(
      format!(
        "source {:?}, target {:?}, coupled face {:?} <-> {:?}",
        source_mesh, target_mesh, face.mesh_a, face.mesh_b
      ),
    )),
  }
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

#[derive(Debug)]
pub enum SyzygyError {
  MissingCoupler,
  FieldMeshMismatch,
  MissingReadField,
  MissingWriteField,
  CellOutOfBounds,
  InvalidRate,
}

impl ErrorDomain for SyzygyError {
  fn domain(&self) -> &str {
    "syzygy"
  }
}

impl std::fmt::Display for SyzygyError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      SyzygyError::MissingCoupler => {
        write!(f, "coupler is not registered in tessera")
      }
      SyzygyError::FieldMeshMismatch => {
        write!(f, "field meshes do not match the selected coupler")
      }
      SyzygyError::MissingReadField => {
        write!(f, "declared read field is missing or has the wrong type")
      }
      SyzygyError::MissingWriteField => {
        write!(f, "declared write field is missing or has the wrong type")
      }
      SyzygyError::CellOutOfBounds => {
        write!(f, "coupled cell id is outside the field storage")
      }
      SyzygyError::InvalidRate => {
        write!(f, "scalar relaxation rate must be finite")
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use std::sync::Arc;

  use nexus::{FieldName, MeshKey, Nexus, Pleroma, WorldId};
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
    nexus.add(ScalarRelaxation::new(
      coupler_index,
      SURFACE_TEMPERATURE,
      ATMOSPHERE_TEMPERATURE,
      1.0,
    ));
    let mut compiled = nexus.build(&pleroma).unwrap();
    compiled
      .tick(WorldId(0), &tessera, &mut pleroma, &Pool::default(), 0.5)
      .unwrap();

    let atmosphere: &SoaField<1> =
      pleroma.read(ATMOSPHERE_TEMPERATURE).unwrap();
    let view = tessera.coupler_view(coupler_index).unwrap();
    let touched = view
      .faces()
      .filter(|face| {
        atmosphere
          .state(face.owner_b)
          .as_state()
          .first()
          .is_some_and(|value| (*value - 150.0).abs() < 1e-10)
      })
      .count();
    assert_eq!(touched, pair_count);
  }
}
