// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use num_dual::Dual64;
use utility::{domain::BoundaryTag, maths::vector::Vector};

use crate::model::Scalar;

/// A boundary condition supplies a ghost state for a boundary face. It stays
/// object-safe (the [`BoundaryRegistry`] stores `dyn BoundaryCondition`) by
/// exposing one concrete method per scalar — real and dual — rather than a
/// generic method. Built-ins implement both from a single generic helper, so
/// the dual path (used for exact AD Jacobians) and the real path can never
/// disagree.
pub trait BoundaryCondition<const D: usize, const N: usize>:
  Send + Sync
{
  fn ghost_state(
    &self,
    interior: &[f64; N],
    normal: &Vector<f64, D>,
  ) -> [f64; N];

  /// Dual-number ghost state for automatic differentiation. Defaults to
  /// real evaluation of the dual's real part with a zero derivative — correct
  /// only for conditions independent of the interior state; built-ins override
  /// it. (No `ConservationLaw` boundary in this workspace relies on the
  /// default.)
  fn ghost_state_dual(
    &self,
    interior: &[Dual64; N],
    normal: &Vector<f64, D>,
  ) -> [Dual64; N] {
    let mut re = [0.0; N];
    for i in 0..N {
      re[i] = interior[i].re;
    }
    let g = self.ghost_state(&re, normal);
    let mut out = [Dual64::from(0.0); N];
    for i in 0..N {
      out[i] = Dual64::from(g[i]);
    }
    out
  }
}

/// Scalar-dispatched ghost lookup: lets the residual kernel stay generic over
/// the scalar `T` while still selecting the right concrete boundary method
/// (real for the explicit path, dual for AD) from a `dyn BoundaryCondition`.
pub trait BoundaryScalar<const D: usize, const N: usize>: Scalar {
  fn ghost(
    bc: &dyn BoundaryCondition<D, N>,
    interior: &[Self; N],
    normal: &Vector<f64, D>,
  ) -> [Self; N];
}

impl<const D: usize, const N: usize> BoundaryScalar<D, N> for f64 {
  fn ghost(
    bc: &dyn BoundaryCondition<D, N>,
    interior: &[f64; N],
    normal: &Vector<f64, D>,
  ) -> [f64; N] {
    bc.ghost_state(interior, normal)
  }
}

impl<const D: usize, const N: usize> BoundaryScalar<D, N> for Dual64 {
  fn ghost(
    bc: &dyn BoundaryCondition<D, N>,
    interior: &[Dual64; N],
    normal: &Vector<f64, D>,
  ) -> [Dual64; N] {
    bc.ghost_state_dual(interior, normal)
  }
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

  fn ghost_state_dual(
    &self,
    interior: &[Dual64; N],
    _normal: &Vector<f64, D>,
  ) -> [Dual64; N] {
    *interior
  }
}

pub struct ReflectiveWall;

impl ReflectiveWall {
  /// Reflect the wall-normal velocity, carry density / energy (and any extra
  /// scalars) through unchanged. Generic over the scalar so the real and dual
  /// ghosts share one body.
  fn reflect<const D: usize, const N: usize, T: Scalar>(
    interior: &[T; N],
    normal: &Vector<f64, D>,
  ) -> [T; N] {
    let rho = interior[0];
    // Wall-normal velocity component vn = (m·n)/ρ.
    let mut mn = T::from(0.0);
    for d in 0..D {
      mn = mn + interior[1 + d] * normal[d];
    }
    let vn = mn / rho;

    let mut out = *interior;
    for d in 0..D {
      // ρu_g = ρu − 2·(ρ·vn)·n_d.
      out[1 + d] = interior[1 + d] - rho * vn * normal[d] * 2.0;
    }
    out
  }
}

// Only for Euler2D — reflects velocity normal to the wall.
impl BoundaryCondition<2, 4> for ReflectiveWall {
  fn ghost_state(
    &self,
    interior: &[f64; 4],
    normal: &Vector<f64, 2>,
  ) -> [f64; 4] {
    Self::reflect(interior, normal)
  }

  fn ghost_state_dual(
    &self,
    interior: &[Dual64; 4],
    normal: &Vector<f64, 2>,
  ) -> [Dual64; 4] {
    Self::reflect(interior, normal)
  }
}

// 3D Euler version (5-state).
impl BoundaryCondition<3, 5> for ReflectiveWall {
  fn ghost_state(
    &self,
    interior: &[f64; 5],
    normal: &Vector<f64, 3>,
  ) -> [f64; 5] {
    Self::reflect(interior, normal)
  }

  fn ghost_state_dual(
    &self,
    interior: &[Dual64; 5],
    normal: &Vector<f64, 3>,
  ) -> [Dual64; 5] {
    Self::reflect(interior, normal)
  }
}

// 3D moist Euler version (6-state): reflect the velocity exactly as the dry
// wall does and carry energy + the moisture tracer through unchanged
// (zero-gradient on water vapour at a solid/ground wall). The generic
// `reflect` only touches components 1..=3, so 4 (energy) and 5 (ρq) pass
// through.
impl BoundaryCondition<3, 6> for ReflectiveWall {
  fn ghost_state(
    &self,
    interior: &[f64; 6],
    normal: &Vector<f64, 3>,
  ) -> [f64; 6] {
    Self::reflect(interior, normal)
  }

  fn ghost_state_dual(
    &self,
    interior: &[Dual64; 6],
    normal: &Vector<f64, 3>,
  ) -> [Dual64; 6] {
    Self::reflect(interior, normal)
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
