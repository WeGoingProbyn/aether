// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Flat dense-vector primitives for the matrix-free implicit solver.
//!
//! The implicit machinery (GMRES, Jacobian-vector products) operates on the
//! simulation state flattened to a single `&[f64]` of length `cells * N`,
//! because `FieldStorage` exposes `axpy`/`weighted_sum` but no inner product
//! or norm. These helpers supply the missing BLAS-1 vocabulary and a small
//! `LinearOperator` abstraction the Krylov solver consumes.

/// Inner product `a · b`. Panics in debug if the lengths differ.
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
  debug_assert_eq!(a.len(), b.len());
  a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Euclidean norm `‖a‖₂`.
pub fn nrm2(a: &[f64]) -> f64 {
  dot(a, a).sqrt()
}

/// `y ← y + alpha · x`.
pub fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
  debug_assert_eq!(x.len(), y.len());
  for (yi, xi) in y.iter_mut().zip(x) {
    *yi += alpha * xi;
  }
}

/// `x ← alpha · x`.
pub fn scale(alpha: f64, x: &mut [f64]) {
  for xi in x.iter_mut() {
    *xi *= alpha;
  }
}

/// `dst ← src`.
pub fn copy(src: &[f64], dst: &mut [f64]) {
  debug_assert_eq!(src.len(), dst.len());
  dst.copy_from_slice(src);
}

/// Flatten a per-cell state cache into a contiguous scalar vector.
pub fn flatten<const N: usize>(cells: &[[f64; N]], out: &mut Vec<f64>) {
  out.clear();
  out.reserve(cells.len() * N);
  for cell in cells {
    out.extend_from_slice(cell);
  }
}

/// Inverse of [`flatten`]: scatter a contiguous scalar vector back into a
/// per-cell state cache. `cells.len() * N` must equal `flat.len()`.
pub fn unflatten<const N: usize>(flat: &[f64], cells: &mut [[f64; N]]) {
  debug_assert_eq!(flat.len(), cells.len() * N);
  for (i, cell) in cells.iter_mut().enumerate() {
    cell.copy_from_slice(&flat[i * N..i * N + N]);
  }
}

/// A linear map `v ↦ A·v` over flattened state vectors. The implicit system
/// matrix `A = I/(γ·dt) − J` is supplied to GMRES through this trait, so the
/// Krylov solver never needs to know how `A·v` is formed (finite-difference
/// JVP, AD, etc.). `apply` takes `&mut self` so the operator may use internal
/// scratch buffers and re-enter the residual kernel.
pub trait LinearOperator {
  /// Dimension of the (square) operator — the flattened state length.
  fn dim(&self) -> usize;

  /// Write `A·v` into `out`. `v` and `out` both have length [`dim`].
  fn apply(&mut self, v: &[f64], out: &mut [f64]);
}

/// Right preconditioner `M⁻¹`: an approximation to `A⁻¹` applied to a vector
/// to accelerate Krylov convergence. The identity preconditioner ([`Identity`])
/// is the no-op baseline; block-Jacobi arrives with the AD Jacobian.
pub trait Preconditioner {
  /// Write `M⁻¹·v` into `out`. Both have length equal to the operator dim.
  fn apply(&mut self, v: &[f64], out: &mut [f64]);
}

/// No-op preconditioner — `M⁻¹ = I`.
pub struct Identity;

impl Preconditioner for Identity {
  fn apply(&mut self, v: &[f64], out: &mut [f64]) {
    copy(v, out);
  }
}

impl<T: LinearOperator + ?Sized> LinearOperator for &mut T {
  fn dim(&self) -> usize {
    (**self).dim()
  }
  fn apply(&mut self, v: &[f64], out: &mut [f64]) {
    (**self).apply(v, out);
  }
}

impl<T: Preconditioner + ?Sized> Preconditioner for &mut T {
  fn apply(&mut self, v: &[f64], out: &mut [f64]) {
    (**self).apply(v, out);
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn dot_and_norm() {
    let a = [3.0, 4.0];
    assert_eq!(dot(&a, &a), 25.0);
    assert_eq!(nrm2(&a), 5.0);
  }

  #[test]
  fn axpy_scale_roundtrip() {
    let x = [1.0, 2.0, 3.0];
    let mut y = [10.0, 10.0, 10.0];
    axpy(2.0, &x, &mut y);
    assert_eq!(y, [12.0, 14.0, 16.0]);
    scale(0.5, &mut y);
    assert_eq!(y, [6.0, 7.0, 8.0]);
  }

  #[test]
  fn flatten_unflatten_roundtrip() {
    let cells = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let mut flat = Vec::new();
    flatten(&cells, &mut flat);
    assert_eq!(flat, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let mut back = [[0.0; 2]; 3];
    unflatten(&flat, &mut back);
    assert_eq!(back, cells);
  }
}
