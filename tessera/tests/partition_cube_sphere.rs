// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::sync::Arc;

use tessera::cube_sphere::{CubeSphere, CubeSphereShellSpec};
use tessera::geometry::{CellGeometry, FaceGeometry};
use tessera::partition::decompose_cube_sphere_panels;
use tessera::topology::{FaceConnection, Topology};
use utility::domain::{BoundaryTag, CellId};

#[test]
fn cube_sphere_panel_decomposition_owns_one_panel_per_partition() {
  let dims = [4, 4, 2];
  let mesh = Arc::new(CubeSphere::shell(CubeSphereShellSpec::uniform(
    dims, 1.0, 2.0,
  )));
  let decomposition = decompose_cube_sphere_panels(Arc::clone(&mesh));
  let cells_per_panel = dims.iter().product::<usize>();

  assert_eq!(decomposition.partitions.len(), 6);

  let mut owned_globals = HashSet::new();
  for (partition_index, partition) in
    decomposition.partitions.iter().enumerate()
  {
    assert_eq!(partition.num_owned(), cells_per_panel);
    assert_eq!(partition.ghost_cells().len(), 4 * dims[0] * dims[2]);
    assert_eq!(
      partition.local_cell_count(),
      partition.num_owned() + partition.ghost_cells().len(),
    );

    for local in 0..partition.num_owned() {
      let global = partition.local_to_global(CellId::from(local));
      assert_eq!(global.index() / cells_per_panel, partition_index);
      assert!(owned_globals.insert(global));
    }

    for ghost in partition.ghost_cells() {
      assert!(ghost.local_cell.index() >= partition.num_owned());
      assert_ne!(ghost.source_partition, partition_index);
      assert!(ghost.source_local_cell.index() < cells_per_panel);
    }
  }

  assert_eq!(owned_globals.len(), mesh.cell_count());
}

#[test]
fn cube_sphere_panel_partition_topology_is_localized() {
  let dims = [3, 3, 2];
  let mesh = Arc::new(CubeSphere::shell(CubeSphereShellSpec::uniform(
    dims, 1.0, 2.0,
  )));
  let decomposition = decompose_cube_sphere_panels(mesh);

  for partition in &decomposition.partitions {
    for &(face, owner, neighbour) in partition.interior_faces() {
      assert!(face.index() < partition.face_count());
      assert!(owner.index() < partition.local_cell_count());
      assert!(neighbour.index() < partition.local_cell_count());
      assert!(matches!(
        partition.face_connection(face),
        FaceConnection::Interior { .. }
      ));
    }

    for tag in partition.boundary_tags() {
      assert!(matches!(
        tag,
        BoundaryTag::Ground | BoundaryTag::AtmosphereEdge
      ));
      for &(face, owner) in partition.boundary_faces(tag) {
        assert!(face.index() < partition.face_count());
        assert!(owner.index() < partition.num_owned());
        assert!(matches!(
          partition.face_connection(face),
          FaceConnection::Boundary { .. }
        ));
      }
    }
  }
}
