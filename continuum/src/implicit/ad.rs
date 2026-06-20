// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Automatic-differentiation Jacobian strategy.
//!
//! Forward-mode dual numbers give an *exact* Jacobian-vector product in a
//! single residual sweep: seed each state component with the direction in its
//! dual part (`U + ε·v`) and the residual's dual part is `J·v` to machine
//! precision — no finite-difference step-size tuning, no truncation error. The
//! work is delegated to the residual itself (which knows how to run its kernel
//! on dual numbers via [`ResidualEval::dual_jvp`]); this strategy just routes
//! to it, so it slots in behind the same [`JacobianStrategy`] seam as the
//! finite-difference default.

use crate::implicit::jacobian::{JacobianStrategy, ResidualEval};
use crate::implicit::linalg::Preconditioner;

/// Exact Jacobian-vector products via forward-mode automatic differentiation.
/// Requires a residual whose [`ResidualEval::dual_jvp`] is implemented;
/// otherwise [`jvp`](JacobianStrategy::jvp) panics (a programming error — pair
/// it only with AD-capable residuals, or use the finite-difference strategy).
#[derive(Default)]
pub struct AutoDiffJacobian;

impl AutoDiffJacobian {
  pub fn new() -> Self {
    Self
  }
}

impl<const N: usize> JacobianStrategy<N> for AutoDiffJacobian {
  fn jvp(
    &mut self,
    residual: &mut dyn ResidualEval<N>,
    _u: &[[f64; N]],
    _r0: &[[f64; N]],
    v: &[f64],
    out: &mut [f64],
  ) {
    // The base residual r0 is unused: AD gives the derivative directly, with
    // no need for the difference quotient's reference evaluation.
    let supported = residual.dual_jvp(_u, v, out);
    assert!(
      supported,
      "AutoDiffJacobian requires an AD-capable residual (dual_jvp); \
       pair it with a law-backed residual or use FiniteDifferenceJacobian"
    );
  }
}

/// Greedy graph colouring of the cells from the mesh's interior-face adjacency,
/// so that no two face-adjacent cells share a colour. The Rusanov stencil only
/// couples face neighbours, so cells of one colour have mutually independent
/// diagonal Jacobian blocks and can be probed together in a single dual sweep.
pub fn face_adjacency_colouring(
  cell_count: usize,
  interior_faces: &[(
    utility::domain::FaceId,
    utility::domain::CellId,
    utility::domain::CellId,
  )],
) -> (Vec<usize>, usize) {
  let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); cell_count];
  for &(_, a, b) in interior_faces {
    adjacency[a.index()].push(b.index());
    adjacency[b.index()].push(a.index());
  }

  let mut colour = vec![usize::MAX; cell_count];
  let mut num_colours = 0;
  let mut neighbour_colours: Vec<bool> = Vec::new();
  for cell in 0..cell_count {
    neighbour_colours.iter_mut().for_each(|b| *b = false);
    for &n in &adjacency[cell] {
      let c = colour[n];
      if c != usize::MAX {
        if c >= neighbour_colours.len() {
          neighbour_colours.resize(c + 1, false);
        }
        neighbour_colours[c] = true;
      }
    }
    let chosen = (0..)
      .find(|&c| c >= neighbour_colours.len() || !neighbour_colours[c])
      .unwrap();
    colour[cell] = chosen;
    num_colours = num_colours.max(chosen + 1);
  }
  (colour, num_colours)
}

/// Invert an `N×N` matrix by Gauss–Jordan elimination with partial pivoting.
/// Returns `None` if (near-)singular.
pub(crate) fn invert<const N: usize>(
  mut a: [[f64; N]; N],
) -> Option<[[f64; N]; N]> {
  let mut inv = [[0.0; N]; N];
  for (i, row) in inv.iter_mut().enumerate() {
    row[i] = 1.0;
  }
  for col in 0..N {
    // Partial pivot.
    let mut pivot = col;
    let mut best = a[col][col].abs();
    for r in (col + 1)..N {
      let v = a[r][col].abs();
      if v > best {
        best = v;
        pivot = r;
      }
    }
    if best < 1e-14 {
      return None;
    }
    a.swap(col, pivot);
    inv.swap(col, pivot);

    let diag = a[col][col];
    for k in 0..N {
      a[col][k] /= diag;
      inv[col][k] /= diag;
    }
    for r in 0..N {
      if r == col {
        continue;
      }
      let factor = a[r][col];
      if factor != 0.0 {
        for k in 0..N {
          a[r][k] -= factor * a[col][k];
          inv[r][k] -= factor * inv[col][k];
        }
      }
    }
  }
  Some(inv)
}

/// Block-Jacobi preconditioner: `M = block-diag(A)` with `A = I/(γ·dt) − J`.
/// Each cell's exact `N×N` diagonal block `∂R_i/∂U_i` is obtained from
/// graph-coloured forward-mode AD sweeps, shifted by the diagonal term, and
/// inverted. Applying `M⁻¹` is then an independent small solve per cell — a
/// cheap, strong preconditioner that sharply cuts GMRES iterations on the
/// stiff acoustic system.
pub struct BlockJacobi<const N: usize> {
  colour: Vec<usize>,
  num_colours: usize,
  /// Inverse of each cell's shifted diagonal block (identity if singular).
  inv: Vec<[[f64; N]; N]>,
  // Scratch reused across rebuilds.
  seed: Vec<f64>,
  jv: Vec<f64>,
}

impl<const N: usize> BlockJacobi<N> {
  pub fn new(colour: Vec<usize>, num_colours: usize) -> Self {
    let cells = colour.len();
    Self {
      colour,
      num_colours,
      inv: vec![identity::<N>(); cells],
      seed: vec![0.0; cells * N],
      jv: vec![0.0; cells * N],
    }
  }

  pub fn cell_count(&self) -> usize {
    self.colour.len()
  }

  /// (Re)assemble and factor the blocks at the current linearization. `residual`
  /// must be AD-capable; `gamma_dt = γ·dt` is the diagonal shift.
  pub fn rebuild(
    &mut self,
    residual: &mut dyn ResidualEval<N>,
    u: &[[f64; N]],
    gamma_dt: f64,
  ) {
    let cells = u.len();
    if self.inv.len() != cells {
      // Topology changed underneath us — fall back to identity blocks.
      self.inv.resize(cells, identity::<N>());
      self.seed.resize(cells * N, 0.0);
      self.jv.resize(cells * N, 0.0);
    }

    // Accumulate each cell's diagonal block J_ii, column by column.
    let mut blocks = vec![[[0.0; N]; N]; cells];
    for c in 0..N {
      for colour in 0..self.num_colours {
        self.seed.iter_mut().for_each(|s| *s = 0.0);
        for i in 0..cells {
          if self.colour[i] == colour {
            self.seed[i * N + c] = 1.0;
          }
        }
        residual.dual_jvp(u, &self.seed, &mut self.jv);
        for i in 0..cells {
          if self.colour[i] == colour {
            for r in 0..N {
              blocks[i][r][c] = self.jv[i * N + r];
            }
          }
        }
      }
    }

    // M_ii = I/(γ·dt) − J_ii, then invert (identity fallback if singular).
    let shift = 1.0 / gamma_dt;
    for i in 0..cells {
      let mut m = blocks[i];
      for r in 0..N {
        for col in 0..N {
          m[r][col] = -m[r][col];
        }
        m[r][r] += shift;
      }
      self.inv[i] = invert(m).unwrap_or_else(identity::<N>);
    }
  }
}

fn identity<const N: usize>() -> [[f64; N]; N] {
  let mut m = [[0.0; N]; N];
  for (i, row) in m.iter_mut().enumerate() {
    row[i] = 1.0;
  }
  m
}

impl<const N: usize> Preconditioner for BlockJacobi<N> {
  fn apply(&mut self, v: &[f64], out: &mut [f64]) {
    let cells = self.inv.len();
    for i in 0..cells {
      let block = &self.inv[i];
      for r in 0..N {
        let mut acc = 0.0;
        for col in 0..N {
          acc += block[r][col] * v[i * N + col];
        }
        out[i * N + r] = acc;
      }
    }
  }
}

/// The preconditioner choice carried by the implicit backend. Concrete (not a
/// trait object) so the backend can call [`BlockJacobi::rebuild`] each step.
pub enum PreconditionerKind<const N: usize> {
  Identity,
  BlockJacobi(BlockJacobi<N>),
}

impl<const N: usize> PreconditionerKind<N> {
  /// Re-linearize the (block-Jacobi) preconditioner at the current state.
  /// A no-op for the identity preconditioner.
  pub fn rebuild(
    &mut self,
    residual: &mut dyn ResidualEval<N>,
    u: &[[f64; N]],
    gamma_dt: f64,
  ) {
    if let PreconditionerKind::BlockJacobi(bj) = self {
      bj.rebuild(residual, u, gamma_dt);
    }
  }
}

impl<const N: usize> Preconditioner for PreconditionerKind<N> {
  fn apply(&mut self, v: &[f64], out: &mut [f64]) {
    match self {
      PreconditionerKind::Identity => out.copy_from_slice(v),
      PreconditionerKind::BlockJacobi(bj) => bj.apply(v, out),
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::implicit::gmres::{self, GmresConfig};
  use crate::implicit::jacobian::{FiniteDifferenceJacobian, ShiftedOperator};
  use crate::implicit::linalg::flatten;

  /// A source-stiff, weakly-coupled residual on a 1-D chain of `cells` cells,
  /// each carrying `N=2` components:
  ///   `R_i = −Λ_i·U_i + κ·(U_{i−1} + U_{i+1} − 2·U_i)`
  /// with a *per-cell, widely varying* stiff diagonal `Λ_i` and small coupling
  /// `κ`. The wide spread of diagonal magnitudes makes the system badly
  /// conditioned for unpreconditioned GMRES, while the stiffness is cell-local
  /// so the exact diagonal block captures it almost entirely — exactly the
  /// regime where block-Jacobi collapses the iteration count.
  struct StiffChain {
    cells: usize,
    lambda: Vec<[f64; 2]>,
    kappa: f64,
  }

  impl StiffChain {
    fn r(&self, state: &[[f64; 2]], out: &mut [[f64; 2]]) {
      for i in 0..self.cells {
        for c in 0..2 {
          let mut coupling = -2.0 * state[i][c];
          if i > 0 {
            coupling += state[i - 1][c];
          }
          if i + 1 < self.cells {
            coupling += state[i + 1][c];
          }
          out[i][c] = -self.lambda[i][c] * state[i][c] + self.kappa * coupling;
        }
      }
    }
  }

  impl ResidualEval<2> for StiffChain {
    fn eval(&mut self, state: &[[f64; 2]], out: &mut [[f64; 2]]) {
      self.r(state, out);
    }

    fn dual_jvp(&mut self, u: &[[f64; 2]], v: &[f64], out: &mut [f64]) -> bool {
      // Linear residual: J·v is just R(v) (homogeneous part), exactly.
      let mut vc = vec![[0.0; 2]; self.cells];
      for i in 0..self.cells {
        vc[i] = [v[i * 2], v[i * 2 + 1]];
      }
      let mut jv = vec![[0.0; 2]; self.cells];
      self.r(&vc, &mut jv);
      let _ = u;
      for i in 0..self.cells {
        out[i * 2] = jv[i][0];
        out[i * 2 + 1] = jv[i][1];
      }
      true
    }
  }

  fn solve_iters(
    chain: &mut StiffChain,
    precond: &mut PreconditionerKind<2>,
    gamma_dt: f64,
  ) -> (usize, bool) {
    let cells = chain.cells;
    let u = vec![[1.0, -1.0]; cells];
    let mut r0 = vec![[0.0; 2]; cells];
    chain.eval(&u, &mut r0);
    precond.rebuild(chain, &u, gamma_dt);

    let mut fd = FiniteDifferenceJacobian::<2>::new();
    let mut op = ShiftedOperator::new(chain, &mut fd, &u, &r0, gamma_dt);
    let mut b = Vec::new();
    flatten(&r0, &mut b);
    let mut x = vec![0.0; cells * 2];
    let cfg = GmresConfig {
      restart: 40,
      max_restarts: 20,
      rel_tol: 1e-9,
      abs_tol: 1e-14,
    };
    let res = gmres::solve(&mut op, precond, &b, &mut x, &cfg);
    (res.iterations, res.converged)
  }

  #[test]
  fn block_jacobi_cuts_iterations_on_source_stiff_system() {
    let cells = 40;
    let gamma_dt = 1.0; // implicit step size factor
    // Per-cell stiffness spanning ~3 orders of magnitude → badly conditioned
    // for plain GMRES, but block-diagonal in structure.
    let lambda = (0..cells)
      .map(|i| {
        let s = 1.0 + (i as f64 / cells as f64) * 1000.0;
        [s, s * 0.3]
      })
      .collect();
    let mut chain = StiffChain {
      cells,
      lambda,
      kappa: 0.5,
    };
    let (colour, n) = face_adjacency_colouring(
      cells,
      // 1-D chain adjacency as interior faces.
      &(0..cells - 1)
        .map(|i| {
          (
            utility::domain::FaceId::from(i),
            utility::domain::CellId::from(i),
            utility::domain::CellId::from(i + 1),
          )
        })
        .collect::<Vec<_>>(),
    );

    let (id_iters, id_conv) =
      solve_iters(&mut chain, &mut PreconditionerKind::Identity, gamma_dt);
    let (bj_iters, bj_conv) = solve_iters(
      &mut chain,
      &mut PreconditionerKind::BlockJacobi(BlockJacobi::new(colour, n)),
      gamma_dt,
    );

    assert!(id_conv && bj_conv, "a solve did not converge");
    assert!(
      bj_iters * 3 < id_iters,
      "block-Jacobi ({bj_iters}) did not substantially beat identity \
       ({id_iters}) on the source-stiff system"
    );
  }
}
