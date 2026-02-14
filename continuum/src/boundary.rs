use utility::maths::vector::Vector;

use crate::topology::BoundaryTag;

pub trait BoundaryCondition<const D: usize, const N: usize>: Send + Sync {
  fn ghost_state(&self, interior: &[f64; N], normal: &Vector<f64, D>) -> [f64; N];
}

pub struct Transmissive;

impl<const D: usize, const N: usize> BoundaryCondition<D, N> for Transmissive {
  fn ghost_state(&self, interior: &[f64; N], _normal: &Vector<f64, D>) -> [f64; N] {
    *interior
  }
}

pub struct ReflectiveWall;

// Only for Euler2D — reflects velocity normal to the wall
impl BoundaryCondition<2, 4> for ReflectiveWall {
  fn ghost_state(&self, interior: &[f64; 4], normal: &Vector<f64, 2>) -> [f64; 4] {
    let rho = interior[0];
    let u = interior[1] / rho;
    let v = interior[2] / rho;

    let vn = u * normal[0] + v * normal[1];
    let u_g = u - 2.0 * vn * normal[0];
    let v_g = v - 2.0 * vn * normal[1];

    [rho, rho * u_g, rho * v_g, interior[3]]
  }
}

pub struct BoundaryRegistry<const D: usize, const N: usize> {
  entries: Vec<(BoundaryTag, Box<dyn BoundaryCondition<D, N>>)>,
}

impl<const D: usize, const N: usize> Default for BoundaryRegistry<D, N> {
  fn default() -> Self {
    BoundaryRegistry {
      entries: Vec::new(),
    }
  }
}

impl<const D: usize, const N: usize> BoundaryRegistry<D, N> {
  pub fn new() -> Self {
    Self::default()
  }

  pub fn register(&mut self, tag: BoundaryTag, bc: impl BoundaryCondition<D, N> + 'static) {
    self.entries.push((tag, Box::new(bc)));
  }

  pub fn get(&self, tag: BoundaryTag) -> Option<&dyn BoundaryCondition<D, N>> {
    self.entries.iter()
      .find(|(t, _)| *t == tag)
      .map(|(_, bc)| bc.as_ref())
  }
}
