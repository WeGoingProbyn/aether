//! geometry.rs
//! Mapping + metric cache: computation -> physical.

use crate::grid::grid::{CellId, FaceId, Grid};

pub type VecN<const D: usize> = [f64; D];
pub type MatN<const D: usize> = [[f64; D]; D];

fn vec_scale<const D: usize>(v: VecN<D>, s: f64) -> VecN<D> {
  let mut out = [0.0; D];
  for i in 0..D {
    out[i] = v[i] * s;
  }
  out
}

pub trait Map<const D: usize> {
  fn x(&self, xi: VecN<D>) -> VecN<D>;
  fn jacobian(&self, xi: VecN<D>) -> MatN<D>;
}

pub trait Geometry<const D: usize> {
  fn grid(&self) -> &Grid<D>;

  fn cell_center_x(&self, c: CellId<D>) -> VecN<D>;
  fn face_center_x(&self, f: FaceId<D>) -> VecN<D>;

  fn cell_volume(&self, c: CellId<D>) -> f64;
  fn face_area_vector(&self, f: FaceId<D>) -> VecN<D>; // n*A outward
}

#[derive(Debug, Clone, Copy)]
pub struct IdentityMap;

impl<const D: usize> Map<D> for IdentityMap {
  fn x(&self, xi: VecN<D>) -> VecN<D> { xi }

  fn jacobian(&self, _xi: VecN<D>) -> MatN<D> {
    let mut j = [[0.0; D]; D];
    for i in 0..D { j[i][i] = 1.0; }
    j
  }
}

fn det<const D: usize>(j: MatN<D>) -> f64 {
  match D {
    2 => j[0][0] * j[1][1] - j[0][1] * j[1][0],
    3 => {
      j[0][0] * (j[1][1] * j[2][2] - j[1][2] * j[2][1])
      - j[0][1] * (j[1][0] * j[2][2] - j[1][2] * j[2][0])
      + j[0][2] * (j[1][0] * j[2][1] - j[1][1] * j[2][0])
    }
    _ => panic!("det: only D=2 or D=3"),
  }
}

fn cofactor_col<const D: usize>(j: MatN<D>, axis: usize) -> VecN<D> {
  let mut out = [0.0; D];

  match D {
    2 => {
      // J = [[a,b],[c,d]] => cof(J) = [[d,-b],[-c,a]]
      let a = j[0][0]; let b = j[0][1];
      let c = j[1][0]; let d = j[1][1];
      let cof00 = d;
      let cof01 = -b;
      let cof10 = -c;
      let cof11 = a;

      // Return the requested column of the cofactor matrix
      if axis == 0 {
        out[0] = cof00;
        out[1] = cof10;
      } else if axis == 1 {
        out[0] = cof01;
        out[1] = cof11;
      } else {
        panic!("cofactor_col(D=2): axis out of range");
      }
      out
    }

    3 => {
      // Columns of cof(J) correspond to cross products of the other two Jacobian columns.
      let dx_dxi  = [j[0][0], j[1][0], j[2][0]];
      let dx_deta = [j[0][1], j[1][1], j[2][1]];
      let dx_dzet = [j[0][2], j[1][2], j[2][2]];

      let cross = |u: [f64; 3], v: [f64; 3]| -> [f64; 3] {
        [
          u[1] * v[2] - u[2] * v[1],
          u[2] * v[0] - u[0] * v[2],
          u[0] * v[1] - u[1] * v[0],
        ]
      };

      let s3 = match axis {
        0 => cross(dx_deta, dx_dzet),
        1 => cross(dx_dzet, dx_dxi),
        2 => cross(dx_dxi,  dx_deta),
        _ => panic!("cofactor_col(D=3): axis out of range"),
      };

      out[0] = s3[0];
      out[1] = s3[1];
      out[2] = s3[2];
      out
    }

    _ => panic!("cofactor_col: only D=2 or D=3 supported"),
  }
}


#[derive(Debug, Clone)]
pub struct MappedGeometry<const D: usize, M: Map<D>> {
  grid: Grid<D>,
  map: M,
}

impl<const D: usize, M: Map<D>> MappedGeometry<D, M> {
  pub fn new(grid: Grid<D>, map: M) -> Self {
    Self { grid, map }
  }
}

impl<const D: usize, M: Map<D>> Geometry<D> for MappedGeometry<D, M> {
  fn grid(&self) -> &Grid<D> { &self.grid }

  fn cell_center_x(&self, c: CellId<D>) -> VecN<D> {
    let xi = self.grid.cell_center_xi(c);
    self.map.x(xi)
  }

  fn face_center_x(&self, f: FaceId<D>) -> VecN<D> {
    let xi = self.grid.face_center_xi(f);
    self.map.x(xi)
  }

  fn cell_volume(&self, c: CellId<D>) -> f64 {
    let xi = self.grid.cell_center_xi(c);
    let j = self.map.jacobian(xi);
    let vol_hat: f64 = self.grid.dxi().iter().product();
    det::<D>(j).abs() * vol_hat
  }

  fn face_area_vector(&self, f: FaceId<D>) -> VecN<D> {
    let ax = f.axis.as_usize();
    let xi = self.grid.face_center_xi(f);
    let j = self.map.jacobian(xi);

    // computational face measure: product of the other dξ’s
    let h = self.grid.dxi();
    let mut area_hat = 1.0;
    for a in 0..D {
      if a != ax { area_hat *= h[a]; }
    }

    let s_dir = cofactor_col::<D>(j, ax);
    vec_scale::<D>(s_dir, f.side.sign() * area_hat)
  }
}
