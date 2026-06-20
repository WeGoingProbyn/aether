// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Jacobian-vector products and the implicit system operator.
//!
//! The matrix-free machinery is built against a generic [`ResidualEval`] — a
//! map `U ↦ R(U) = dU/dt` over the mesh — rather than a conservation law
//! directly. The fully-implicit backend hands it the law's full residual; an
//! IMEX split (later) hands it only the implicit operator, with no change to
//! anything here. A [`JacobianStrategy`] forms `J·v` from that residual
//! (finite differences now, automatic differentiation later), and
//! [`ShiftedOperator`] assembles the linearly-implicit system matrix
//! `A·v = v/(γ·dt) − J·v` that GMRES solves.

use crate::implicit::linalg::LinearOperator;

/// Evaluate a residual `R(U) = dU/dt` over the whole mesh: read the per-cell
/// `state` cache, write `out`. `&mut self` so implementations may keep scratch
/// (flux accumulators) without reallocating per call.
pub trait ResidualEval<const N: usize> {
  fn eval(&mut self, state: &[[f64; N]], out: &mut [[f64; N]]);

  /// Exact directional derivative `J·v` (flattened) via automatic
  /// differentiation, if the residual supports it. Returns `false` when AD is
  /// unavailable so the caller can fall back to finite differences. The default
  /// declines.
  fn dual_jvp(
    &mut self,
    _u: &[[f64; N]],
    _v: &[f64],
    _out: &mut [f64],
  ) -> bool {
    false
  }
}

/// Strategy for forming the Jacobian-vector product `J·v` of a residual at a
/// base state `u`, given the precomputed base residual `r0 = R(u)`. Takes the
/// residual as a parameter and holds no law itself, so the same strategy serves
/// the full operator now and an IMEX implicit operator later unchanged.
pub trait JacobianStrategy<const N: usize> {
  /// Write `J·v` (flattened, length `u.len()·N`) into `out`. `v` is flattened
  /// the same way; `r0` is `R(u)` already evaluated by the caller.
  fn jvp(
    &mut self,
    residual: &mut dyn ResidualEval<N>,
    u: &[[f64; N]],
    r0: &[[f64; N]],
    v: &[f64],
    out: &mut [f64],
  );
}

impl<const N: usize, T: JacobianStrategy<N> + ?Sized> JacobianStrategy<N>
  for &mut T
{
  fn jvp(
    &mut self,
    residual: &mut dyn ResidualEval<N>,
    u: &[[f64; N]],
    r0: &[[f64; N]],
    v: &[f64],
    out: &mut [f64],
  ) {
    (**self).jvp(residual, u, r0, v, out);
  }
}

impl<const N: usize, T: ResidualEval<N> + ?Sized> ResidualEval<N> for &mut T {
  fn eval(&mut self, state: &[[f64; N]], out: &mut [[f64; N]]) {
    (**self).eval(state, out);
  }
}

/// First-order finite-difference Jacobian: `J·v ≈ (R(u + ε·v) − R(u)) / ε`.
/// Needs no derivative information from the law — the Phase-0 default before
/// the AD strategy lands.
#[derive(Default)]
pub struct FiniteDifferenceJacobian<const N: usize> {
  perturbed: Vec<[f64; N]>,
  r_perturbed: Vec<[f64; N]>,
}

impl<const N: usize> FiniteDifferenceJacobian<N> {
  pub fn new() -> Self {
    Self {
      perturbed: Vec::new(),
      r_perturbed: Vec::new(),
    }
  }
}

impl<const N: usize> JacobianStrategy<N> for FiniteDifferenceJacobian<N> {
  fn jvp(
    &mut self,
    residual: &mut dyn ResidualEval<N>,
    u: &[[f64; N]],
    r0: &[[f64; N]],
    v: &[f64],
    out: &mut [f64],
  ) {
    let cells = u.len();
    debug_assert_eq!(v.len(), cells * N);
    debug_assert_eq!(out.len(), cells * N);

    // ‖v‖ and ‖u‖ drive the step size; a zero direction gives a zero column.
    let mut v2 = 0.0;
    for &vi in v {
      v2 += vi * vi;
    }
    let vnorm = v2.sqrt();
    if vnorm <= f64::MIN_POSITIVE {
      out.iter_mut().for_each(|o| *o = 0.0);
      return;
    }
    let mut u2 = 0.0;
    for cell in u {
      for &c in cell {
        u2 += c * c;
      }
    }
    let unorm = u2.sqrt();

    // ε ≈ √(machine-eps) scaled by the state magnitude over the direction
    // magnitude — the standard Brown–Saad choice for matrix-free Newton-Krylov.
    let eps = 1e-7 * (1.0 + unorm) / vnorm;

    if self.perturbed.len() != cells {
      self.perturbed.resize(cells, [0.0; N]);
      self.r_perturbed.resize(cells, [0.0; N]);
    }
    for i in 0..cells {
      for c in 0..N {
        self.perturbed[i][c] = u[i][c] + eps * v[i * N + c];
      }
    }

    residual.eval(&self.perturbed, &mut self.r_perturbed);

    let inv_eps = 1.0 / eps;
    for i in 0..cells {
      for c in 0..N {
        out[i * N + c] = (self.r_perturbed[i][c] - r0[i][c]) * inv_eps;
      }
    }
  }
}

/// The linearly-implicit system matrix as a [`LinearOperator`]:
/// `A·v = v/(γ·dt) − J·v`, solved by GMRES for each Rosenbrock stage. It pairs
/// a residual, the base state `u`, the base residual `r0 = R(u)`, and a
/// Jacobian strategy; GMRES sees only `apply`.
pub struct ShiftedOperator<'a, const N: usize, Strat> {
  residual: &'a mut dyn ResidualEval<N>,
  strategy: Strat,
  u: &'a [[f64; N]],
  r0: &'a [[f64; N]],
  inv_gamma_dt: f64,
  jv: Vec<f64>,
}

impl<'a, const N: usize, Strat> ShiftedOperator<'a, N, Strat>
where
  Strat: JacobianStrategy<N>,
{
  /// `gamma_dt = γ·dt` is the diagonal shift `1/(γ·dt)`. `r0` must equal
  /// `R(u)` (the caller already has it as the stage RHS).
  pub fn new(
    residual: &'a mut dyn ResidualEval<N>,
    strategy: Strat,
    u: &'a [[f64; N]],
    r0: &'a [[f64; N]],
    gamma_dt: f64,
  ) -> Self {
    Self {
      residual,
      strategy,
      u,
      r0,
      inv_gamma_dt: 1.0 / gamma_dt,
      jv: vec![0.0; u.len() * N],
    }
  }
}

impl<const N: usize, Strat> LinearOperator for ShiftedOperator<'_, N, Strat>
where
  Strat: JacobianStrategy<N>,
{
  fn dim(&self) -> usize {
    self.u.len() * N
  }

  fn apply(&mut self, v: &[f64], out: &mut [f64]) {
    self
      .strategy
      .jvp(self.residual, self.u, self.r0, v, &mut self.jv);
    for i in 0..out.len() {
      out[i] = v[i] * self.inv_gamma_dt - self.jv[i];
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Linear residual `R(U) = M·U` for a fixed small matrix, so the exact
  /// Jacobian is `M` and FD-JVP should reproduce `M·v` closely.
  struct LinearResidual {
    m: [[f64; 2]; 2],
  }

  impl ResidualEval<2> for LinearResidual {
    fn eval(&mut self, state: &[[f64; 2]], out: &mut [[f64; 2]]) {
      for i in 0..state.len() {
        let s = state[i];
        out[i][0] = self.m[0][0] * s[0] + self.m[0][1] * s[1];
        out[i][1] = self.m[1][0] * s[0] + self.m[1][1] * s[1];
      }
    }
  }

  #[test]
  fn fd_jvp_matches_analytic_jacobian() {
    let mut res = LinearResidual {
      m: [[2.0, -1.0], [0.5, 3.0]],
    };
    let u = [[1.0, 2.0], [-1.0, 0.5]];
    let mut r0 = [[0.0; 2]; 2];
    res.eval(&u, &mut r0);

    // Direction v (flattened): exercise both cells.
    let v = [1.0, 0.0, 0.0, 1.0];
    let mut jv = [0.0; 4];
    let mut fd = FiniteDifferenceJacobian::<2>::new();
    fd.jvp(&mut res, &u, &r0, &v, &mut jv);

    // Analytic: J is block-diagonal with M on each cell, so
    // J·v = [M·(1,0) for cell0, M·(0,1) for cell1] = [2,0.5, -1,3].
    let expect = [2.0, 0.5, -1.0, 3.0];
    for i in 0..4 {
      assert!(
        (jv[i] - expect[i]).abs() < 1e-5,
        "jv[{i}] = {} want {}",
        jv[i],
        expect[i]
      );
    }
  }
}
