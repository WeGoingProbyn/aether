use utility::maths::{matrix::Matrix, vector::Vector};

pub type Point<const D: usize> = Vector<f64, D>;

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

pub trait CellGeometry<const D: usize>: Send + Sync {
  fn cell_centroid(&self, cell: CellId) -> &Point<D>;
  fn cell_volume(&self, cell: CellId) -> f64;
  fn cell_metrics(&self, cell: CellId) -> &CellMetrics<D>;
  fn cell_count(&self) -> usize;
}

pub trait FaceGeometry<const D: usize>: Send + Sync {
  fn face_centroid(&self, face: FaceId) -> &Point<D>;
  fn face_area_vector(&self, face: FaceId) -> Vector<f64, D>;
  fn face_area(&self, face: FaceId) -> f64;
  fn face_metrics(&self, face: FaceId) -> &FaceMetrics<D>;
  fn face_count(&self) -> usize;
}

pub trait GeometryMap<const D: usize, const P: usize> {
  fn to_physical(&self, comp: &Point<D>) -> Point<P>;
  fn to_computational(&self, physical: &Point<P>) -> Option<Point<D>>;
  fn jacobian(&self, comp: &Point<D>) -> Matrix<f64, D, P>;
  fn sqrt_det_metric(&self, comp: &Point<D>) -> f64;
}

pub struct CellMetrics<const D: usize> {
  pub sqrt_metric: f64,
  pub comp_volume: f64,
  pub phys_volume: f64,
}

pub struct FaceMetrics<const D: usize> {
  pub normal: Vector<f64, D>,
  pub comp_area: f64,
  pub phys_area: f64,
  pub sqrt_metric: f64,
}

/// Identity map for solution in cartesian grids requiring
/// no geometrical mappings from computational to physical
pub struct IdentityMap<const D: usize>;

impl<const D: usize> GeometryMap<D, D> for IdentityMap<D> {
  fn to_physical(&self, comp: &Point<D>) -> Point<D> {
    comp.into()
  }

  fn to_computational(&self, physical: &Point<D>) -> Option<Point<D>> {
    Some(physical.into())
  }

  fn jacobian(&self, _: &Point<D>) -> Matrix<f64, D, D> {
    Matrix::<f64, D, D>::identity(1.0f64)
  }

  fn sqrt_det_metric(&self, _: &Point<D>) -> f64 {
    1.0f64
  }
}
