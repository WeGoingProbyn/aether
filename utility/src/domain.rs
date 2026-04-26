use crate::maths::vector::Vector;

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

#[derive(PartialEq, Eq, PartialOrd, Ord, Debug, Clone, Copy, Hash)]
pub enum Axis {
  X = 0,
  Y = 1,
  Z = 2,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct CellId(usize);

impl CellId {
  pub fn index(&self) -> usize {
    self.0
  }
}

impl From<usize> for CellId {
  fn from(value: usize) -> Self {
    CellId(value)
  }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct FaceId(usize);

impl FaceId {
  pub fn index(&self) -> usize {
    self.0
  }
}

impl From<usize> for FaceId {
  fn from(value: usize) -> Self {
    FaceId(value)
  }
}

pub type Point<const D: usize> = Vector<f64, D>;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum FieldKey {
  Temperature,
  VelocityX,
  VelocityY,
  VelocityZ,
  Pressure,
  Humidity,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum MeshType {
  Atmosphere,
  Surface,
  Mantle,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct MeshKey(MeshType);
