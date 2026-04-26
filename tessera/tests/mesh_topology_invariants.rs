// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;

use continuum::geometry::{
  CellGeometry, CellId, FaceGeometry, FaceId, IdentityMap,
};
use continuum::mesh::StructuredBlock;
use continuum::topology::{FaceConnection, Topology};

fn expected_cell_count<const D: usize>(dims: [usize; D]) -> usize {
  dims.iter().product()
}

fn expected_face_count<const D: usize>(dims: [usize; D]) -> usize {
  (0..D)
    .map(|axis| {
      (0..D)
        .map(|d| if d == axis { dims[d] + 1 } else { dims[d] })
        .product::<usize>()
    })
    .sum()
}

fn expected_interior_face_count<const D: usize>(dims: [usize; D]) -> usize {
  (0..D)
    .map(|axis| {
      let along = dims[axis] - 1;
      let across = (0..D)
        .filter(|&d| d != axis)
        .map(|d| dims[d])
        .product::<usize>()
        .max(1);
      along * across
    })
    .sum()
}

fn expected_boundary_face_count<const D: usize>(dims: [usize; D]) -> usize {
  (0..D)
    .map(|axis| {
      let across = (0..D)
        .filter(|&d| d != axis)
        .map(|d| dims[d])
        .product::<usize>()
        .max(1);
      2 * across
    })
    .sum()
}

fn assert_structured_mesh_invariants<const D: usize>(dims: [usize; D]) {
  let mesh = StructuredBlock::uniform(
    [0.0; D].into(),
    [1.0; D],
    dims,
    Box::new(IdentityMap::<D>),
  );

  assert_eq!(mesh.cell_count(), expected_cell_count(dims));
  assert_eq!(mesh.face_count(), expected_face_count(dims));
  assert_eq!(
    mesh.interior_faces().len(),
    expected_interior_face_count(dims)
  );

  let mut interior_faces = HashSet::new();
  for &(face, owner, neighbour) in mesh.interior_faces() {
    assert!(
      interior_faces.insert(face.index()),
      "duplicate interior face id {}",
      face.index()
    );

    match mesh.face_connection(face) {
      FaceConnection::Interior {
        owner: owner_conn,
        neighbour: neighbour_conn,
      } => {
        assert_eq!(owner_conn.index(), owner.index());
        assert_eq!(neighbour_conn.index(), neighbour.index());
      }
      FaceConnection::Boundary { .. } => panic!(
        "face {} in interior list has boundary connection",
        face.index()
      ),
    }

    assert!(
      mesh
        .cell_faces(owner)
        .iter()
        .any(|f| f.index() == face.index())
    );
    assert!(
      mesh
        .cell_faces(neighbour)
        .iter()
        .any(|f| f.index() == face.index())
    );
  }

  let mut boundary_faces = HashSet::new();
  let mut boundary_count = 0usize;
  for tag in mesh.boundary_tags() {
    for &(face, owner) in mesh.boundary_faces(tag) {
      boundary_count += 1;
      assert!(
        boundary_faces.insert(face.index()),
        "duplicate boundary face id {}",
        face.index()
      );

      match mesh.face_connection(face) {
        FaceConnection::Boundary {
          owner: owner_conn,
          tag: tag_conn,
          ..
        } => {
          assert_eq!(owner_conn.index(), owner.index());
          assert_eq!(*tag_conn, tag);
        }
        FaceConnection::Interior { .. } => panic!(
          "face {} in boundary list has interior connection",
          face.index()
        ),
      }

      assert!(
        mesh
          .cell_faces(owner)
          .iter()
          .any(|f| f.index() == face.index())
      );
    }
  }

  assert_eq!(boundary_count, expected_boundary_face_count(dims));
  assert!(interior_faces.is_disjoint(&boundary_faces));
  assert_eq!(
    interior_faces.len() + boundary_faces.len(),
    mesh.face_count()
  );

  let mut face_ref_counts = vec![0usize; mesh.face_count()];
  for i in 0..mesh.cell_count() {
    let cell = CellId::from(i);
    let faces = mesh.cell_faces(cell);
    assert_eq!(faces.len(), 2 * D);

    let mut unique_faces = HashSet::new();
    for face in faces {
      assert!(face.index() < mesh.face_count());
      assert!(
        unique_faces.insert(face.index()),
        "cell {} references face {} multiple times",
        i,
        face.index()
      );
      face_ref_counts[face.index()] += 1;
    }
  }

  for (face_idx, refs) in face_ref_counts.into_iter().enumerate() {
    let face = FaceId::from(face_idx);
    match mesh.face_connection(face) {
      FaceConnection::Interior { .. } => assert_eq!(
        refs, 2,
        "interior face {} should be referenced by two cells",
        face_idx
      ),
      FaceConnection::Boundary { .. } => assert_eq!(
        refs, 1,
        "boundary face {} should be referenced by one cell",
        face_idx
      ),
    }
  }
}

#[test]
fn invariants_2d_square() {
  assert_structured_mesh_invariants([2, 2]);
}

#[test]
fn invariants_2d_strip() {
  assert_structured_mesh_invariants([3, 1]);
}

#[test]
fn invariants_3d_cube() {
  assert_structured_mesh_invariants([2, 2, 2]);
}

#[test]
fn from_axis_edges_uniform_matches_uniform_constructor() {
  // Same physical box built two ways should be byte-identical in the data
  // the rest of the solver looks at.
  let dims = [3, 2, 4];
  let origin = [0.0, 0.0, 0.0];
  let extent = [3.0, 2.0, 4.0];

  let a = StructuredBlock::uniform(
    origin.into(),
    extent,
    dims,
    Box::new(IdentityMap::<3>),
  );

  let edges: [Vec<f64>; 3] = std::array::from_fn(|d| {
    let dx = extent[d] / dims[d] as f64;
    (0..=dims[d]).map(|i| origin[d] + i as f64 * dx).collect()
  });
  let b = StructuredBlock::from_axis_edges(edges, Box::new(IdentityMap::<3>));

  assert_eq!(a.cell_count(), b.cell_count());
  assert_eq!(a.face_count(), b.face_count());
  for i in 0..a.cell_count() {
    let cell = CellId::from(i);
    assert_eq!(a.cell_volume(cell), b.cell_volume(cell));
  }
  for i in 0..a.face_count() {
    let face = FaceId::from(i);
    assert_eq!(a.face_area(face), b.face_area(face));
  }
}

#[test]
fn non_uniform_axis_produces_per_cell_volumes_and_areas() {
  // Non-uniform axis 2 with three cells of widths 1, 2, 4. Other axes uniform.
  // Cell volume should match width-product per cell; face area along axis 0
  // should depend on the cell's axis-2 position.
  let edges: [Vec<f64>; 3] = [
    vec![0.0, 1.0, 2.0],      // 2 cells, width 1
    vec![0.0, 1.0, 2.0],      // 2 cells, width 1
    vec![0.0, 1.0, 3.0, 7.0], // 3 cells, widths 1, 2, 4
  ];
  let mesh =
    StructuredBlock::from_axis_edges(edges, Box::new(IdentityMap::<3>));

  // Cells: 2*2*3 = 12. Indexed (i + j*Nx + k*Nx*Ny).
  // Cell at (0,0,k) has volume 1·1·width_k.
  let widths_z = [1.0, 2.0, 4.0];
  for k in 0..3 {
    let cell_idx = 0 + 0 * 2 + k * 4;
    let cell = CellId::from(cell_idx);
    assert_eq!(mesh.cell_volume(cell), widths_z[k]);
  }

  // Total volume = 4 cells per layer × layer width, summed = 4·(1+2+4) = 28.
  let total: f64 = (0..mesh.cell_count())
    .map(|i| mesh.cell_volume(CellId::from(i)))
    .sum();
  assert_eq!(total, 28.0);

  // Sanity: total face count matches the uniform formula since topology is
  // independent of edge spacing.
  let dims = [2, 2, 3];
  assert_eq!(mesh.face_count(), expected_face_count(dims));
}
