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

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct WorldId(pub usize);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SystemId(pub usize);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum FieldName {
  Temperature,
  TemperatureTendency,
  EulerState,
  VelocityX,
  VelocityY,
  VelocityZ,
  Pressure,
  Humidity,
  RadiativeHeatingTendency,
  NetSurfaceFlux,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum MeshType {
  Atmosphere,
  Surface,
  Mantle,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct MeshKey(MeshType);

impl MeshKey {
  pub const ATMOSPHERE: MeshKey = MeshKey(MeshType::Atmosphere);
  pub const SURFACE: MeshKey = MeshKey(MeshType::Surface);
  pub const MANTLE: MeshKey = MeshKey(MeshType::Mantle);

  pub const fn new(mesh_type: MeshType) -> Self {
    MeshKey(mesh_type)
  }

  pub const fn mesh_type(self) -> MeshType {
    self.0
  }
}

/// Identifier for a typed singleton in pleroma that isn't bound to a mesh.
/// Used for things like orbital body state, sun direction, planetary spin —
/// data physics stages need but that doesn't live per-cell on a mesh.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ResourceKey {
  Bodies,
  SunPosition,
  PlanetSpin,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct FieldKey {
  mesh: MeshKey,
  name: FieldName,
}

impl FieldKey {
  pub const fn new(mesh: MeshKey, name: FieldName) -> Self {
    Self { mesh, name }
  }

  pub const fn mesh(self) -> MeshKey {
    self.mesh
  }

  pub const fn name(self) -> FieldName {
    self.name
  }
}
