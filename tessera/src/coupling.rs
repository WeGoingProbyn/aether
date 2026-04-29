// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use utility::{
  domain::{CellId, FaceId, MeshKey, Point},
  maths::vector::Vector,
};

use crate::{mesh::Mesh, topology::FaceConnection};

pub trait MeshCoupler: Send + Sync {
  fn paired_face(&self, side: Side, face: FaceId) -> Option<(Side, FaceId)>;
  fn paired_cell(&self, side: Side, cell: CellId) -> Option<(Side, CellId)>;
  fn pairs(&self) -> &[FacePair];
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Side {
  A,
  B,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FacePair {
  a: FaceId,
  b: FaceId,
}

impl FacePair {
  pub fn new(a: FaceId, b: FaceId) -> Self {
    Self { a, b }
  }

  pub fn a(&self) -> FaceId {
    self.a
  }

  pub fn b(&self) -> FaceId {
    self.b
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CoupledFace {
  pub mesh_a: MeshKey,
  pub face_a: FaceId,
  pub owner_a: CellId,
  pub mesh_b: MeshKey,
  pub face_b: FaceId,
  pub owner_b: CellId,
  pub centroid: Point<3>,
  pub normal_a_to_b: Vector<f64, 3>,
  pub area: f64,
  pub area_a: f64,
  pub area_b: f64,
}

impl CoupledFace {
  pub fn owner_for(&self, mesh: MeshKey) -> Option<CellId> {
    if mesh == self.mesh_a {
      Some(self.owner_a)
    } else if mesh == self.mesh_b {
      Some(self.owner_b)
    } else {
      None
    }
  }

  pub fn face_for(&self, mesh: MeshKey) -> Option<FaceId> {
    if mesh == self.mesh_a {
      Some(self.face_a)
    } else if mesh == self.mesh_b {
      Some(self.face_b)
    } else {
      None
    }
  }

  pub fn from_pair(
    mesh_a_key: MeshKey,
    mesh_a: &dyn Mesh<3>,
    mesh_b_key: MeshKey,
    mesh_b: &dyn Mesh<3>,
    pair: FacePair,
  ) -> Self {
    let owner_a = face_owner(mesh_a, pair.a());
    let owner_b = face_owner(mesh_b, pair.b());
    let centroid_a = mesh_a.face_world_centroid(pair.a());
    let centroid_b = mesh_b.face_world_centroid(pair.b());
    let cell_a = mesh_a.cell_world_centroid(owner_a);
    let cell_b = mesh_b.cell_world_centroid(owner_b);
    let delta = &cell_b - &cell_a;
    let normal_a_to_b =
      unit_or_fallback(delta, mesh_a.face_area_vector(pair.a()));
    let area_a = mesh_a.face_metrics(pair.a()).phys_area;
    let area_b = mesh_b.face_metrics(pair.b()).phys_area;

    Self {
      mesh_a: mesh_a_key,
      face_a: pair.a(),
      owner_a,
      mesh_b: mesh_b_key,
      face_b: pair.b(),
      owner_b,
      centroid: (&centroid_a + &centroid_b) * 0.5,
      normal_a_to_b,
      area: 0.5 * (area_a + area_b),
      area_a,
      area_b,
    }
  }
}

fn face_owner(mesh: &dyn Mesh<3>, face: FaceId) -> CellId {
  match mesh.face_connection(face) {
    FaceConnection::Boundary { owner, .. }
    | FaceConnection::Interior { owner, .. } => *owner,
  }
}

fn unit_or_fallback(
  vector: Vector<f64, 3>,
  fallback: Vector<f64, 3>,
) -> Vector<f64, 3> {
  let magnitude = vector.magnitude();
  if magnitude > f64::EPSILON {
    return &vector / &magnitude;
  }

  let fallback_magnitude = fallback.magnitude();
  if fallback_magnitude > f64::EPSILON {
    &fallback / &fallback_magnitude
  } else {
    [0.0, 0.0, 0.0].into()
  }
}
