// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use tessera::{coupling::CoupledFace, world_mesh::Tessera};
use utility::{
  domain::{CellId, MeshKey},
  error::{AetherError, AetherResult},
};

use crate::error::SyzygyError;

/// One source↔target interface overlap. Many entries may share a `target_cell`
/// (a coarse target gathering from several fine sources) or a `source_cell` (a
/// coarse source scattering to several fine targets) — that is how level-mismatch
/// (N:M) coupling under AMR is expressed; a conforming 1:1 interface is the
/// degenerate case with one entry per cell.
///
/// `area` is the raw overlap physical area; the two normalised weights are derived
/// from it in [`CouplingStencil::new`] so a consumer picks the one matching its
/// semantics (and they each sum to 1 over their axis — see that constructor).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CouplingEntry {
  pub source_cell: CellId,
  pub target_cell: CellId,
  /// Raw overlap physical area of this pair.
  pub area: f64,
  pub distance: f64,
  pub normal: [f64; 3],
  /// `area` normalised so the weights over all entries sharing this `target_cell`
  /// sum to 1 — the **gather** weight (relaxation / interpolation read).
  pub target_weight: f64,
  /// `area` normalised so the weights over all entries sharing this `source_cell`
  /// sum to 1 — the conservative **scatter** fraction (deposition).
  pub source_weight: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CouplingStencil {
  source_mesh: MeshKey,
  target_mesh: MeshKey,
  entries: Vec<CouplingEntry>,
}

impl CouplingStencil {
  pub fn new(
    source_mesh: MeshKey,
    target_mesh: MeshKey,
    mut entries: Vec<CouplingEntry>,
  ) -> AetherResult<Self> {
    if source_mesh == target_mesh {
      return Err(AetherError::new(SyzygyError::FieldMeshMismatch).context(
        format!("source and target mesh are both {:?}", source_mesh),
      ));
    }
    for entry in &entries {
      validate_entry(entry)?;
    }
    normalise_weights(&mut entries);
    Ok(Self {
      source_mesh,
      target_mesh,
      entries,
    })
  }

  pub fn from_tessera_coupler(
    tessera: &Tessera,
    coupler_index: usize,
    source_mesh: MeshKey,
    target_mesh: MeshKey,
  ) -> AetherResult<Self> {
    let coupler = tessera.coupler_view(coupler_index).ok_or_else(|| {
      AetherError::new(SyzygyError::MissingCoupler)
        .context(format!("coupler index {}", coupler_index))
    })?;
    validate_coupler_meshes(
      source_mesh,
      target_mesh,
      coupler.mesh_a(),
      coupler.mesh_b(),
    )?;

    let entries = coupler
      .faces()
      .map(|face| entry_from_face(face, source_mesh, target_mesh))
      .collect::<AetherResult<Vec<_>>>()?;

    Self::new(source_mesh, target_mesh, entries)
  }

  pub fn source_mesh(&self) -> MeshKey {
    self.source_mesh
  }

  pub fn target_mesh(&self) -> MeshKey {
    self.target_mesh
  }

  pub fn entries(&self) -> &[CouplingEntry] {
    &self.entries
  }

  pub fn len(&self) -> usize {
    self.entries.len()
  }

  pub fn is_empty(&self) -> bool {
    self.entries.is_empty()
  }
}

fn validate_coupler_meshes(
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

fn entry_from_face(
  face: CoupledFace,
  source_mesh: MeshKey,
  target_mesh: MeshKey,
) -> AetherResult<CouplingEntry> {
  let (source_cell, target_cell, normal) = match (source_mesh, target_mesh) {
    (source, target) if source == face.mesh_a && target == face.mesh_b => (
      face.owner_a,
      face.owner_b,
      vector_to_array(&face.normal_a_to_b),
    ),
    (source, target) if source == face.mesh_b && target == face.mesh_a => (
      face.owner_b,
      face.owner_a,
      [
        -face.normal_a_to_b[0],
        -face.normal_a_to_b[1],
        -face.normal_a_to_b[2],
      ],
    ),
    _ => {
      return Err(AetherError::new(SyzygyError::FieldMeshMismatch).context(
        format!(
          "source {:?}, target {:?}, coupled face {:?} <-> {:?}",
          source_mesh, target_mesh, face.mesh_a, face.mesh_b
        ),
      ));
    }
  };

  let entry = CouplingEntry {
    source_cell,
    target_cell,
    area: face.area,
    distance: face.distance,
    normal,
    // Filled by `normalise_weights` in `CouplingStencil::new`.
    target_weight: 0.0,
    source_weight: 0.0,
  };
  validate_entry(&entry)?;
  Ok(entry)
}

fn validate_entry(entry: &CouplingEntry) -> AetherResult<()> {
  let normal_magnitude = entry.normal.iter().map(|x| x * x).sum::<f64>().sqrt();
  if entry.area.is_finite()
    && entry.area > 0.0
    && entry.distance.is_finite()
    && entry.distance >= 0.0
    && normal_magnitude.is_finite()
  {
    Ok(())
  } else {
    Err(
      AetherError::new(SyzygyError::InvalidStencil)
        .context(format!("entry {:?}", entry)),
    )
  }
}

/// Derive each entry's `target_weight` / `source_weight` from its raw overlap
/// `area`, normalised so the weights sum to 1 over all entries sharing a target
/// (gather) and over all entries sharing a source (scatter), respectively. Doing
/// it here — once, at construction — means the geometric area computation's
/// floating-point slop is absorbed into exact fractions, so conservative
/// deposition (which splits by `source_weight`) cannot silently gain/lose mass.
fn normalise_weights(entries: &mut [CouplingEntry]) {
  use std::collections::HashMap;
  let mut target_area: HashMap<usize, f64> = HashMap::new();
  let mut source_area: HashMap<usize, f64> = HashMap::new();
  for e in entries.iter() {
    *target_area.entry(e.target_cell.index()).or_default() += e.area;
    *source_area.entry(e.source_cell.index()).or_default() += e.area;
  }
  for e in entries.iter_mut() {
    let ts = target_area[&e.target_cell.index()];
    let ss = source_area[&e.source_cell.index()];
    e.target_weight = if ts > 0.0 { e.area / ts } else { 0.0 };
    e.source_weight = if ss > 0.0 { e.area / ss } else { 0.0 };
  }

  if cfg!(debug_assertions) {
    let mut tsum: HashMap<usize, f64> = HashMap::new();
    let mut ssum: HashMap<usize, f64> = HashMap::new();
    for e in entries.iter() {
      *tsum.entry(e.target_cell.index()).or_default() += e.target_weight;
      *ssum.entry(e.source_cell.index()).or_default() += e.source_weight;
    }
    for (_, s) in tsum {
      debug_assert!((s - 1.0).abs() < 1e-9, "target weights sum to {s}, not 1");
    }
    for (_, s) in ssum {
      debug_assert!((s - 1.0).abs() < 1e-9, "source weights sum to {s}, not 1");
    }
  }
}

fn vector_to_array(
  vector: &utility::maths::vector::Vector<f64, 3>,
) -> [f64; 3] {
  [vector[0], vector[1], vector[2]]
}

#[cfg(test)]
mod tests {
  use std::sync::Arc;

  use tessera::{
    coupling::MeshCoupler, cube_sphere::CubeSphere, geometry::CellGeometry,
    mesh::Mesh, radial_stack::RadialStackCoupler, world_mesh::Tessera,
  };
  use utility::domain::MeshKey;

  use super::*;

  #[test]
  fn compiles_radial_coupler_into_directional_stencil() {
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
    assert!(surface.cell_count() > 0);
    assert!(atmosphere.cell_count() > 0);

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

    let forward = CouplingStencil::from_tessera_coupler(
      &tessera,
      coupler_index,
      MeshKey::SURFACE,
      MeshKey::ATMOSPHERE,
    )
    .unwrap();
    let reverse = CouplingStencil::from_tessera_coupler(
      &tessera,
      coupler_index,
      MeshKey::ATMOSPHERE,
      MeshKey::SURFACE,
    )
    .unwrap();

    assert_eq!(forward.len(), pair_count);
    assert_eq!(reverse.len(), pair_count);
    assert_eq!(forward.source_mesh(), MeshKey::SURFACE);
    assert_eq!(forward.target_mesh(), MeshKey::ATMOSPHERE);

    let a = forward.entries()[0];
    let b = reverse.entries()[0];
    assert_eq!(a.source_cell, b.target_cell);
    assert_eq!(a.target_cell, b.source_cell);
    // A conforming 1:1 interface: each cell has a single partner, so both
    // normalised weights are 1.
    assert_eq!(a.target_weight, 1.0);
    assert_eq!(a.source_weight, 1.0);
    assert!(a.area > 0.0);
    assert!(a.distance > 0.0);
    assert!(
      (a.normal.iter().map(|x| x * x).sum::<f64>().sqrt() - 1.0).abs() < 1e-10
    );
    assert_eq!(a.normal, [-b.normal[0], -b.normal[1], -b.normal[2]]);
  }
}
