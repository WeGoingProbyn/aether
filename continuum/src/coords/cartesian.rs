pub trait CoordinateSystem {
  type Coord;
  fn linear_index(&self, coords: Self::Coord) -> usize;
}

#[derive(Clone, Debug)]
pub struct CartesianGrid2D {
  pub nx: usize,
  pub ny: usize,
  
  pub dx: f32,
  pub dy: f32,

  pub total_cells: usize,
}

impl CoordinateSystem for CartesianGrid2D {
  type Coord = (usize, usize);

  fn linear_index(
    &self, 
    (i, j): Self::Coord,
  ) -> usize {
    (j * self.nx) + i
  }
}

#[derive(Clone, Debug)]
pub struct CartesianGrid3D {
  pub nx: usize,
  pub ny: usize,
  pub nz: usize,
  
  pub dx: f32,
  pub dy: f32,
  pub dz: f32,

  pub total_cells: usize,
}

impl CoordinateSystem for CartesianGrid3D {
  type Coord = (usize, usize, usize);

  fn linear_index(
    &self, 
    (i, j, k): Self::Coord,
  ) -> usize {
    (k * self.nx * self.ny) + (j * self.nx) + i 
  }
}

