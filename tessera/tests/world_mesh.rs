// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use tessera::coupling::{FacePair, MeshCoupler, Side};
use tessera::geometry::IdentityMap;
use tessera::mesh::{Mesh, StructuredBlock};
use tessera::partition::decompose_structured;
use tessera::world_mesh::{DecompositionKey, Tessera};
use utility::domain::{CellId, FaceId, MeshKey};

struct EmptyCoupler;

impl MeshCoupler for EmptyCoupler {
  fn paired_face(&self, _side: Side, _face: FaceId) -> Option<(Side, FaceId)> {
    None
  }

  fn paired_cell(&self, _side: Side, _cell: CellId) -> Option<(Side, CellId)> {
    None
  }

  fn pairs(&self) -> &[FacePair] {
    &[]
  }
}

#[test]
fn tessera_registers_meshes_and_couplers() {
  let mut tessera = Tessera::new();
  let mesh = Arc::new(StructuredBlock::uniform(
    [0.0; 3].into(),
    [1.0; 3],
    [2, 2, 2],
    Box::new(IdentityMap::<3>),
  ));
  let mesh_for_registry: Arc<dyn Mesh<3>> = mesh.clone();

  assert!(!tessera.contains_mesh(MeshKey::SURFACE));
  assert!(
    tessera
      .register_mesh(MeshKey::SURFACE, mesh_for_registry)
      .is_none()
  );
  assert!(tessera.contains_mesh(MeshKey::SURFACE));
  assert_eq!(tessera.meshes().count(), 1);

  let id =
    tessera.add_coupler(MeshKey::SURFACE, MeshKey::ATMOSPHERE, EmptyCoupler);
  assert_eq!(id, 0);
  assert_eq!(tessera.couplers().len(), 1);
  assert_eq!(
    tessera
      .couplers_between(MeshKey::ATMOSPHERE, MeshKey::SURFACE)
      .count(),
    1
  );
}

#[test]
fn tessera_owns_decompositions_per_mesh() {
  let mut tessera = Tessera::new();
  let dims = [4, 2, 1];
  let mesh = Arc::new(StructuredBlock::uniform(
    [0.0; 3].into(),
    [1.0; 3],
    dims,
    Box::new(IdentityMap::<3>),
  ));
  let mesh_for_registry: Arc<dyn Mesh<3>> = mesh.clone();
  tessera.register_mesh(MeshKey::SURFACE, mesh_for_registry);

  let decomposition = decompose_structured(mesh, dims, 2, 1);
  assert!(
    tessera
      .register_decomposition(
        MeshKey::SURFACE,
        DecompositionKey::DEFAULT,
        decomposition,
      )
      .is_none()
  );

  assert!(
    tessera
      .contains_decomposition(MeshKey::SURFACE, DecompositionKey::DEFAULT,)
  );
  let borrowed = tessera
    .decomposition::<StructuredBlock<3>>(
      MeshKey::SURFACE,
      DecompositionKey::DEFAULT,
    )
    .unwrap();
  assert_eq!(borrowed.partitions.len(), 2);
}
