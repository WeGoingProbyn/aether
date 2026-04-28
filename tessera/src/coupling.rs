// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use utility::domain::{CellId, FaceId};

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
