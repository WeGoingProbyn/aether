// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use utility::{domain::BoundaryTag, maths::vector::Vector};

pub trait BoundaryCondition<const D: usize, const N: usize>:
  Send + Sync
{
  fn ghost_state(
    &self,
    interior: &[f64; N],
    normal: &Vector<f64, D>,
  ) -> [f64; N];
}

pub struct Transmissive;

impl<const D: usize, const N: usize> BoundaryCondition<D, N> for Transmissive {
  fn ghost_state(
    &self,
    interior: &[f64; N],
    _normal: &Vector<f64, D>,
  ) -> [f64; N] {
    *interior
  }
}

pub struct ReflectiveWall;

// Only for Euler2D — reflects velocity normal to the wall
impl BoundaryCondition<2, 4> for ReflectiveWall {
  fn ghost_state(
    &self,
    interior: &[f64; 4],
    normal: &Vector<f64, 2>,
  ) -> [f64; 4] {
    let rho = interior[0];
    let u = interior[1] / rho;
    let v = interior[2] / rho;

    let vn = u * normal[0] + v * normal[1];
    let u_g = u - 2.0 * vn * normal[0];
    let v_g = v - 2.0 * vn * normal[1];

    [rho, rho * u_g, rho * v_g, interior[3]]
  }
}

// 3D Euler version (5-state).
impl BoundaryCondition<3, 5> for ReflectiveWall {
  fn ghost_state(
    &self,
    interior: &[f64; 5],
    normal: &Vector<f64, 3>,
  ) -> [f64; 5] {
    let rho = interior[0];
    let u = interior[1] / rho;
    let v = interior[2] / rho;
    let w = interior[3] / rho;

    let vn = u * normal[0] + v * normal[1] + w * normal[2];
    let u_g = u - 2.0 * vn * normal[0];
    let v_g = v - 2.0 * vn * normal[1];
    let w_g = w - 2.0 * vn * normal[2];

    [rho, rho * u_g, rho * v_g, rho * w_g, interior[4]]
  }
}

// 3D moist Euler version (6-state): reflect the velocity exactly as the
// dry wall does and carry energy + the moisture tracer through unchanged
// (zero-gradient on water vapour at a solid/ground wall).
impl BoundaryCondition<3, 6> for ReflectiveWall {
  fn ghost_state(
    &self,
    interior: &[f64; 6],
    normal: &Vector<f64, 3>,
  ) -> [f64; 6] {
    let rho = interior[0];
    let u = interior[1] / rho;
    let v = interior[2] / rho;
    let w = interior[3] / rho;

    let vn = u * normal[0] + v * normal[1] + w * normal[2];
    let u_g = u - 2.0 * vn * normal[0];
    let v_g = v - 2.0 * vn * normal[1];
    let w_g = w - 2.0 * vn * normal[2];

    [
      rho,
      rho * u_g,
      rho * v_g,
      rho * w_g,
      interior[4],
      interior[5],
    ]
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

  pub fn register(
    &mut self,
    tag: BoundaryTag,
    bc: impl BoundaryCondition<D, N> + 'static,
  ) {
    self.entries.push((tag, Box::new(bc)));
  }

  pub fn get(&self, tag: BoundaryTag) -> Option<&dyn BoundaryCondition<D, N>> {
    self
      .entries
      .iter()
      .find(|(t, _)| *t == tag)
      .map(|(_, bc)| bc.as_ref())
  }
}
