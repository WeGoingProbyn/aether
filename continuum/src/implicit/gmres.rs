// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Restarted, right-preconditioned GMRES(m) for the matrix-free implicit
//! solve. The solver consumes a [`LinearOperator`] (the implicit system matrix
//! `A = I/(γ·dt) − J`) and a [`Preconditioner`]; it knows nothing about the
//! conservation law or the integration scheme. Arnoldi uses modified
//! Gram–Schmidt and the least-squares problem is reduced with Givens rotations.

use crate::implicit::linalg::{
  LinearOperator, Preconditioner, axpy, copy, dot, nrm2, scale,
};

/// GMRES tuning. `restart` (the Krylov subspace size `m`) is configurable from
/// the start — it dominates memory and per-cycle cost and wants per-scene
/// tuning on a large mesh.
#[derive(Clone, Copy)]
pub struct GmresConfig {
  /// Krylov subspace size before restart (`m`).
  pub restart: usize,
  /// Maximum number of restart cycles.
  pub max_restarts: usize,
  /// Convergence relative to `‖b‖`.
  pub rel_tol: f64,
  /// Absolute convergence floor (for `‖b‖ ≈ 0`).
  pub abs_tol: f64,
}

impl Default for GmresConfig {
  fn default() -> Self {
    Self {
      restart: 30,
      max_restarts: 20,
      rel_tol: 1e-8,
      abs_tol: 1e-12,
    }
  }
}

/// Outcome of a GMRES solve.
#[derive(Clone, Copy, Debug)]
pub struct GmresResult {
  /// Whether the target tolerance was reached.
  pub converged: bool,
  /// Total matrix-vector products performed.
  pub iterations: usize,
  /// Final residual norm `‖b − A·x‖`.
  pub residual: f64,
}

/// Stable Givens rotation `(c, s)` such that `[c s; −s c]·[a; b] = [r; 0]`.
fn givens(a: f64, b: f64) -> (f64, f64) {
  if b == 0.0 {
    (1.0, 0.0)
  } else if b.abs() > a.abs() {
    let tau = a / b;
    let s = 1.0 / (1.0 + tau * tau).sqrt();
    (tau * s, s)
  } else {
    let tau = b / a;
    let c = 1.0 / (1.0 + tau * tau).sqrt();
    (c, tau * c)
  }
}

/// Solve `A·x = b` for `x` (which also seeds the initial guess) with restarted,
/// right-preconditioned GMRES(m). Returns convergence status, the matvec count,
/// and the final true residual norm.
pub fn solve<Op, P>(
  a: &mut Op,
  precond: &mut P,
  b: &[f64],
  x: &mut [f64],
  cfg: &GmresConfig,
) -> GmresResult
where
  Op: LinearOperator + ?Sized,
  P: Preconditioner + ?Sized,
{
  let n = a.dim();
  debug_assert_eq!(b.len(), n);
  debug_assert_eq!(x.len(), n);

  let m = cfg.restart.max(1);
  let bnorm = nrm2(b);
  let target = (cfg.rel_tol * bnorm).max(cfg.abs_tol);

  // Scratch.
  let mut r = vec![0.0; n]; // residual / work
  let mut w = vec![0.0; n]; // Arnoldi work
  let mut z = vec![0.0; n]; // M⁻¹·v
  let mut basis: Vec<Vec<f64>> = vec![vec![0.0; n]; m + 1];
  let mut h = vec![0.0; (m + 1) * m]; // Hessenberg, row-major (i*m + j)
  let mut cs = vec![0.0; m];
  let mut sn = vec![0.0; m];
  let mut g = vec![0.0; m + 1]; // rotated RHS

  let mut total_iters = 0;

  for _ in 0..cfg.max_restarts {
    // r = b − A·x
    a.apply(x, &mut r);
    for i in 0..n {
      r[i] = b[i] - r[i];
    }
    let beta = nrm2(&r);
    if beta <= target {
      return GmresResult {
        converged: true,
        iterations: total_iters,
        residual: beta,
      };
    }

    copy(&r, &mut basis[0]);
    scale(1.0 / beta, &mut basis[0]);
    g.iter_mut().for_each(|v| *v = 0.0);
    g[0] = beta;

    let mut k = 0; // completed inner iterations
    for j in 0..m {
      // w = A·M⁻¹·v_j  (right preconditioning)
      precond.apply(&basis[j], &mut z);
      a.apply(&z, &mut w);
      total_iters += 1;

      // Modified Gram–Schmidt against the existing basis.
      for i in 0..=j {
        let hij = dot(&w, &basis[i]);
        h[i * m + j] = hij;
        axpy(-hij, &basis[i], &mut w);
      }
      let hnext = nrm2(&w);
      h[(j + 1) * m + j] = hnext;
      if hnext > 1e-300 {
        copy(&w, &mut basis[j + 1]);
        scale(1.0 / hnext, &mut basis[j + 1]);
      }

      // Apply earlier rotations to the new Hessenberg column.
      for i in 0..j {
        let t = cs[i] * h[i * m + j] + sn[i] * h[(i + 1) * m + j];
        h[(i + 1) * m + j] = -sn[i] * h[i * m + j] + cs[i] * h[(i + 1) * m + j];
        h[i * m + j] = t;
      }

      // New rotation zeroes the sub-diagonal entry.
      let (c, s) = givens(h[j * m + j], h[(j + 1) * m + j]);
      cs[j] = c;
      sn[j] = s;
      h[j * m + j] = c * h[j * m + j] + s * h[(j + 1) * m + j];
      h[(j + 1) * m + j] = 0.0;
      let t = c * g[j] + s * g[j + 1];
      g[j + 1] = -s * g[j] + c * g[j + 1];
      g[j] = t;

      k = j + 1;
      if g[j + 1].abs() <= target {
        break;
      }
    }

    // Back-substitute H·y = g over the k×k upper-triangular block.
    let mut y = vec![0.0; k];
    for i in (0..k).rev() {
      let mut sum = g[i];
      for col in (i + 1)..k {
        sum -= h[i * m + col] * y[col];
      }
      y[i] = sum / h[i * m + i];
    }

    // x += M⁻¹·(Σ y_j v_j).
    r.iter_mut().for_each(|v| *v = 0.0);
    for j in 0..k {
      axpy(y[j], &basis[j], &mut r);
    }
    precond.apply(&r, &mut z);
    axpy(1.0, &z, x);

    // True residual check before deciding to restart.
    a.apply(x, &mut w);
    for i in 0..n {
      w[i] = b[i] - w[i];
    }
    if nrm2(&w) <= target {
      break;
    }
  }

  a.apply(x, &mut r);
  for i in 0..n {
    r[i] = b[i] - r[i];
  }
  let residual = nrm2(&r);
  GmresResult {
    converged: residual <= target,
    iterations: total_iters,
    residual,
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::implicit::linalg::Identity;

  /// Dense matrix operator for testing.
  struct Dense {
    n: usize,
    a: Vec<f64>, // row-major n×n
  }

  impl LinearOperator for Dense {
    fn dim(&self) -> usize {
      self.n
    }
    fn apply(&mut self, v: &[f64], out: &mut [f64]) {
      for i in 0..self.n {
        let mut acc = 0.0;
        for j in 0..self.n {
          acc += self.a[i * self.n + j] * v[j];
        }
        out[i] = acc;
      }
    }
  }

  #[test]
  fn solves_spd_system() {
    // A symmetric positive-definite 3×3 system with a known solution.
    let mut a = Dense {
      n: 3,
      a: vec![4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0],
    };
    let x_true = [1.0, -2.0, 3.0];
    let mut b = [0.0; 3];
    a.apply(&x_true, &mut b);

    let mut x = [0.0; 3];
    let cfg = GmresConfig {
      restart: 3,
      max_restarts: 5,
      rel_tol: 1e-12,
      abs_tol: 1e-14,
    };
    let res = solve(&mut a, &mut Identity, &b, &mut x, &cfg);

    assert!(res.converged, "GMRES did not converge: {res:?}");
    for i in 0..3 {
      assert!(
        (x[i] - x_true[i]).abs() < 1e-8,
        "x[{i}] = {} want {}",
        x[i],
        x_true[i]
      );
    }
  }

  #[test]
  fn solves_larger_diagonally_dominant_system() {
    // n=60 strongly diagonally dominant nonsymmetric system; restarted GMRES
    // must converge well within the budget.
    let n = 60;
    let mut a = vec![0.0; n * n];
    for i in 0..n {
      for j in 0..n {
        a[i * n + j] = if i == j {
          10.0
        } else {
          (((i * 7 + j * 3) % 5) as f64 - 2.0) * 0.1
        };
      }
    }
    let mut op = Dense { n, a };
    let x_true: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin()).collect();
    let mut b = vec![0.0; n];
    op.apply(&x_true, &mut b);

    let mut x = vec![0.0; n];
    let cfg = GmresConfig {
      restart: 30,
      max_restarts: 10,
      rel_tol: 1e-10,
      abs_tol: 1e-12,
    };
    let res = solve(&mut op, &mut Identity, &b, &mut x, &cfg);
    assert!(res.converged, "did not converge: {res:?}");
    for i in 0..n {
      assert!((x[i] - x_true[i]).abs() < 1e-6, "x[{i}] mismatch");
    }
  }

  #[test]
  fn solves_nonsymmetric_system() {
    let mut a = Dense {
      n: 3,
      a: vec![10.0, 1.0, 2.0, -1.0, 8.0, 0.5, 2.0, -3.0, 12.0],
    };
    let x_true = [0.5, 1.5, -2.0];
    let mut b = [0.0; 3];
    a.apply(&x_true, &mut b);

    let mut x = [0.0; 3];
    let res = solve(&mut a, &mut Identity, &b, &mut x, &GmresConfig::default());

    assert!(res.converged, "GMRES did not converge: {res:?}");
    for i in 0..3 {
      assert!((x[i] - x_true[i]).abs() < 1e-7);
    }
  }
}
