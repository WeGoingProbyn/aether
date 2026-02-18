use crate::geometry::{CellId, FaceId};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum BoundaryTag {
  Top,
  Left,
  Right,
  Bottom,
  Front,
  Back,

  Wall,
  Ground,
  Inflow,
  Outflow,
  AtmosphereEdge,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum CellKind {
  Owned,
  Ghost { source: u32, remote: CellId },
}

pub enum FaceConnection {
  Interior {
    owner: CellId,
    neighbour: CellId,
  },
  Boundary {
    owner: CellId,
    tag: BoundaryTag,
    out_sign: f64,
  },
}

pub trait Topology: Send + Sync {
  fn face_connection(&self, face: FaceId) -> &FaceConnection;
  fn cell_faces(&self, cell: CellId) -> &[FaceId];
  // fn face_count(&self) -> usize;
  // fn cell_count(&self) -> usize;
  fn interior_faces(&self) -> &[(FaceId, CellId, CellId)];
  fn boundary_faces(&self, tag: BoundaryTag) -> &[(FaceId, CellId)];
  fn boundary_tags(&self) -> impl Iterator<Item = BoundaryTag> + '_;
}

pub trait Partition {
  fn cell_kind(&self, cell: CellId) -> &CellKind;
  fn owned_cells(&self) -> &[CellId];
  fn ghost_cells(&self) -> &[(CellId, u32, CellId)];
}
