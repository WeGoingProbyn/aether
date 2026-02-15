use crate::{geometry::{CellGeometry, FaceGeometry}, mesh::StructuredBlock, topology::Topology};

pub struct CubeSphere {
  inner: StructuredBlock<3>,
}

impl CellGeometry for CubeSphere {
  fn cell_count(&self) -> usize {
    
  }

  fn cell_volume(&self, cell: crate::geometry::CellId) -> f64 {
    
  }

  fn cell_metrics(&self, cell: crate::geometry::CellId) -> &crate::geometry::CellMetrics<D> {
    
  }

  fn cell_centroid(&self, cell: crate::geometry::CellId) -> &crate::geometry::Point<D> {
    
  }
}

impl FaceGeometry for CubeSphere {
  fn face_area(&self, face: crate::geometry::FaceId) -> f64 {
    
  }

  fn face_count(&self) -> usize {
    
  }
}

impl Topology for CubeSphere {

}
