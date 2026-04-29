// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashSet, sync::Arc};

use tessera::coupling::{MeshCoupler, Side};
use tessera::cube_sphere::{CubeSphere, GnomonicShellPanel, PanelId};
use tessera::geometry::{FaceGeometry, GeometryMap};
use tessera::mesh::Mesh;
use tessera::radial_stack::RadialStackCoupler;
use tessera::topology::{FaceConnection, Topology};
use tessera::world_mesh::Tessera;
use utility::domain::{CellId, FaceId, MeshKey, Point};

fn panel_of(cell: CellId, dims: [usize; 3]) -> GnomonicShellPanel {
  const PANELS: [PanelId; 6] = [
    PanelId::XP,
    PanelId::XN,
    PanelId::YP,
    PanelId::YN,
    PanelId::ZP,
    PanelId::ZN,
  ];
  let panel = cell.index() / (dims[0] * dims[1] * dims[2]);
  GnomonicShellPanel::new(PANELS[panel])
}

fn boundary_owner(mesh: &CubeSphere, face: FaceId) -> CellId {
  match mesh.face_connection(face) {
    FaceConnection::Boundary { owner, .. } => *owner,
    FaceConnection::Interior { .. } => {
      panic!("expected boundary face {}, got interior", face.index())
    }
  }
}

fn face_world_centroid(
  mesh: &CubeSphere,
  face: FaceId,
  dims: [usize; 3],
) -> Point<3> {
  let owner = boundary_owner(mesh, face);
  panel_of(owner, dims).to_physical(mesh.face_centroid(face))
}

fn outward_unit_normal(mesh: &CubeSphere, face: FaceId) -> [f64; 3] {
  let out_sign = match mesh.face_connection(face) {
    FaceConnection::Boundary { out_sign, .. } => *out_sign,
    FaceConnection::Interior { .. } => {
      panic!("expected boundary face {}, got interior", face.index())
    }
  };
  let area = mesh.face_area(face);
  let area_vector = mesh.face_area_vector(face);
  [
    out_sign * area_vector[0] / area,
    out_sign * area_vector[1] / area,
    out_sign * area_vector[2] / area,
  ]
}

#[test]
fn radial_stack_pairs_cube_sphere_interface_faces() {
  let n = 4;
  let lower_radial_layers = 2;
  let upper_radial_layers = 3;
  let lower_dims = [n, n, lower_radial_layers];
  let upper_dims = [n, n, upper_radial_layers];
  let lower = CubeSphere::new(lower_dims, 1.0, 2.0);
  let upper = CubeSphere::new(upper_dims, 2.0, 3.5);
  let coupler =
    RadialStackCoupler::new([n, n], lower_radial_layers, upper_radial_layers);

  assert_eq!(coupler.pairs().len(), 6 * n * n);
  assert_eq!(coupler.panel_count(), 6);
  assert_eq!(coupler.angular_dims(), [n, n]);
  assert_eq!(coupler.lower_top_layer_idx(), lower_radial_layers - 1);
  assert_eq!(coupler.upper_bottom_layer_idx(), 0);

  let mut lower_faces = HashSet::new();
  let mut upper_faces = HashSet::new();

  for pair in coupler.pairs() {
    assert!(lower_faces.insert(pair.a()));
    assert!(upper_faces.insert(pair.b()));

    assert_eq!(
      coupler.paired_face(Side::A, pair.a()),
      Some((Side::B, pair.b()))
    );
    assert_eq!(
      coupler.paired_face(Side::B, pair.b()),
      Some((Side::A, pair.a()))
    );

    let lower_centroid = face_world_centroid(&lower, pair.a(), lower_dims);
    let upper_centroid = face_world_centroid(&upper, pair.b(), upper_dims);
    assert!(
      lower_centroid.distance(&upper_centroid) < 1e-10,
      "paired faces {} and {} do not meet: {:?} vs {:?}",
      pair.a().index(),
      pair.b().index(),
      lower_centroid,
      upper_centroid
    );

    let lower_normal = outward_unit_normal(&lower, pair.a());
    let upper_normal = outward_unit_normal(&upper, pair.b());
    let dot = lower_normal
      .iter()
      .zip(upper_normal.iter())
      .map(|(a, b)| a * b)
      .sum::<f64>();
    assert!(
      (dot + 1.0).abs() < 1e-10,
      "paired face outward normals should oppose; dot={dot}"
    );
  }

  assert_eq!(lower_faces.len(), 6 * n * n);
  assert_eq!(upper_faces.len(), 6 * n * n);
}

#[test]
fn radial_stack_pairs_interface_cells_bidirectionally() {
  let n = 3;
  let lower_radial_layers = 3;
  let upper_radial_layers = 2;
  let lower = CubeSphere::new([n, n, lower_radial_layers], 1.0, 2.0);
  let upper = CubeSphere::new([n, n, upper_radial_layers], 2.0, 3.0);
  let coupler =
    RadialStackCoupler::new([n, n], lower_radial_layers, upper_radial_layers);

  for pair in coupler.pairs() {
    let lower_owner = boundary_owner(&lower, pair.a());
    let upper_owner = boundary_owner(&upper, pair.b());
    assert_eq!(
      coupler.paired_cell(Side::A, lower_owner),
      Some((Side::B, upper_owner))
    );
    assert_eq!(
      coupler.paired_cell(Side::B, upper_owner),
      Some((Side::A, lower_owner))
    );
  }

  assert_eq!(coupler.paired_cell(Side::A, CellId::from(0)), None);
  let upper_non_interface = CellId::from(n * n);
  assert_eq!(coupler.paired_cell(Side::B, upper_non_interface), None);
}

#[test]
fn radial_stack_rejects_non_interface_faces() {
  let n = 4;
  let lower_radial_layers = 2;
  let upper_radial_layers = 2;
  let lower = CubeSphere::new([n, n, lower_radial_layers], 1.0, 2.0);
  let upper = CubeSphere::new([n, n, upper_radial_layers], 2.0, 3.0);
  let coupler =
    RadialStackCoupler::new([n, n], lower_radial_layers, upper_radial_layers);

  let lower_ground_face = lower
    .boundary_faces(utility::domain::BoundaryTag::Ground)
    .first()
    .unwrap()
    .0;
  let upper_outer_face = upper
    .boundary_faces(utility::domain::BoundaryTag::AtmosphereEdge)
    .first()
    .unwrap()
    .0;

  assert_eq!(coupler.paired_face(Side::A, lower_ground_face), None);
  assert_eq!(coupler.paired_face(Side::B, upper_outer_face), None);
}

#[test]
fn tessera_coupler_view_exposes_interface_geometry() {
  let n = 2;
  let lower_radial_layers = 2;
  let upper_radial_layers = 2;
  let lower = Arc::new(CubeSphere::new([n, n, lower_radial_layers], 1.0, 2.0));
  let upper = Arc::new(CubeSphere::new([n, n, upper_radial_layers], 2.0, 3.0));
  let lower_for_registry: Arc<dyn Mesh<3>> = lower;
  let upper_for_registry: Arc<dyn Mesh<3>> = upper;

  let mut tessera = Tessera::new();
  tessera.register_mesh(MeshKey::SURFACE, lower_for_registry);
  tessera.register_mesh(MeshKey::ATMOSPHERE, upper_for_registry);
  let coupler_index = tessera.add_coupler(
    MeshKey::SURFACE,
    MeshKey::ATMOSPHERE,
    RadialStackCoupler::new([n, n], lower_radial_layers, upper_radial_layers),
  );

  let view = tessera
    .coupler_view(coupler_index)
    .expect("registered coupler should resolve both meshes");
  assert_eq!(view.pair_count(), 6 * n * n);

  let coupled = view.faces().next().expect("coupler should have faces");
  assert_eq!(coupled.mesh_a, MeshKey::SURFACE);
  assert_eq!(coupled.mesh_b, MeshKey::ATMOSPHERE);
  assert_eq!(coupled.owner_for(MeshKey::SURFACE), Some(coupled.owner_a));
  assert_eq!(
    coupled.owner_for(MeshKey::ATMOSPHERE),
    Some(coupled.owner_b)
  );
  assert!(coupled.area > 0.0);
  assert!((coupled.normal_a_to_b.magnitude() - 1.0).abs() < 1e-10);
  assert!(
    (coupled.centroid.magnitude() - 2.0).abs() < 1e-10,
    "interface centroid should lie on shared radius"
  );
}
