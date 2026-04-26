// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use utility::domain::{FaceId, CellId};

pub trait MeshCoupler: Send + Sync {
  fn paired_face(&self, side: Side, face: FaceId) -> Option<(Side, FaceId)>;
  fn paired_cell(&self, side: Side, cell: CellId) -> Option<(Side, CellId)>;
  fn pairs(&self) -> &[FacePair];
}

pub enum Side {
  A,
  B,
}

pub struct FacePair {
  side_a: Side,
  side_b: Side,
}
