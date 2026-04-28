// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use tessera::coupling::{FacePair, MeshCoupler, Side};
use tessera::geometry::IdentityMap;
use tessera::mesh::{Mesh, StructuredBlock};
use tessera::world_mesh::Tessera;
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
  let mesh: Arc<dyn Mesh<3>> = Arc::new(StructuredBlock::uniform(
    [0.0; 3].into(),
    [1.0; 3],
    [2, 2, 2],
    Box::new(IdentityMap::<3>),
  ));

  assert!(!tessera.contains_mesh(MeshKey::SURFACE));
  assert!(
    tessera
      .register_mesh(MeshKey::SURFACE, Arc::clone(&mesh))
      .is_none()
  );
  assert!(tessera.contains_mesh(MeshKey::SURFACE));
  assert!(Arc::ptr_eq(tessera.mesh(MeshKey::SURFACE).unwrap(), &mesh));

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
