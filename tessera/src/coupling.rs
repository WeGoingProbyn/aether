use crate::geometry::{CellId, FaceId};

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
