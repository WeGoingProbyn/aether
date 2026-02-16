use std::collections::HashSet;

use continuum::geometry::{CellGeometry, CellId, FaceGeometry, FaceId, IdentityMap};
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
  assert_eq!(mesh.interior_faces().len(), expected_interior_face_count(dims));

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

    assert!(mesh.cell_faces(owner).iter().any(|f| f.index() == face.index()));
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

      assert!(mesh.cell_faces(owner).iter().any(|f| f.index() == face.index()));
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
