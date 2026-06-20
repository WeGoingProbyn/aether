// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Horizontally-explicit / vertically-implicit (HEVI) stepping.
//!
//! On a thin atmospheric shell the explicit CFL is set almost entirely by the
//! *vertical* (radial) cell spacing — measured at ~80× tighter than horizontal
//! for the demo geometry. HEVI integrates only the **vertical acoustic** terms
//! implicitly (mass flux, pressure gradient, pressure work through radial
//! faces) and everything else — horizontal acoustics, all advection —
//! explicitly. The implicit coupling is then purely between vertically-adjacent
//! cells, so per radial column the linear system `(I/dt − J_vert)·ΔU = R(U)` is
//! **block-tridiagonal** and is solved exactly by block-Thomas elimination —
//! no Krylov iteration, no global coupling, and each column is independent.
//!
//! This module currently provides the block-tridiagonal solver; the column
//! extraction, AD block assembly and backend build on top of it.

use num_dual::Dual64;
use pleroma::core::storage::FieldStorage;
use tessera::{mesh::Mesh, topology::FaceConnection};
use utility::{
  domain::{CellId, FaceId},
  maths::vector::Vector,
};

use crate::{
  boundary::{BoundaryRegistry, BoundaryScalar},
  implicit::ad::invert,
  kernel,
  model::{ConservationLaw, NumericalFlux},
  solver::{FvmBackend, SolverConfig},
};

/// One radial column of a shell mesh: cells ordered bottom→top, the interior
/// radial faces linking consecutive cells, and the boundary caps. This is the
/// geometry the vertically-implicit solver needs; it is built by the caller
/// (who knows the mesh is a shell) so continuum stays domain-neutral.
pub struct RadialColumn {
  /// Cells from innermost (bottom) to outermost (top).
  pub cells: Vec<CellId>,
  /// `up_faces[k]` is the interior radial face between `cells[k]` and
  /// `cells[k+1]` (length `cells.len() - 1`).
  pub up_faces: Vec<FaceId>,
  /// Boundary radial face below the bottom cell (e.g. Ground), if any.
  pub bottom_face: Option<FaceId>,
  /// Boundary radial face above the top cell (e.g. AtmosphereEdge), if any.
  pub top_face: Option<FaceId>,
}

/// Extract radial columns from a shell mesh. `world_pos(cell)` gives each
/// cell's world position (so "radial" = the direction from the origin); an
/// interior face is *radial* when its normal aligns with that direction. Works
/// for any origin-centred shell (cube-sphere, radial stack).
pub fn radial_columns_from_geometry<const D: usize, M, W>(
  mesh: &M,
  world_pos: W,
) -> Vec<RadialColumn>
where
  M: Mesh<D> + ?Sized,
  W: Fn(CellId) -> [f64; 3],
{
  let cells = mesh.cell_count();
  let radius = |c: CellId| {
    let p = world_pos(c);
    (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt()
  };
  let r_hat = |c: CellId| {
    let p = world_pos(c);
    let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
    [p[0] / r, p[1] / r, p[2] / r]
  };

  // Per-cell radial links: (face, neighbour) above and below.
  let mut up: Vec<Option<(FaceId, CellId)>> = vec![None; cells];
  let mut down: Vec<Option<(FaceId, CellId)>> = vec![None; cells];

  for &(face, a, b) in mesh.interior_faces() {
    let area = mesh.face_area(face);
    if area <= 0.0 {
      continue;
    }
    let av = mesh.face_area_vector(face);
    let n = [av[0] / area, av[1] / area, av[2] / area];
    let rh = r_hat(a);
    let radiality = (n[0] * rh[0] + n[1] * rh[1] + n[2] * rh[2]).abs();
    if radiality <= 0.7 {
      continue; // tangential face
    }
    // Radial face: the higher-radius cell is "up".
    let (lower, upper) = if radius(a) < radius(b) {
      (a, b)
    } else {
      (b, a)
    };
    up[lower.index()] = Some((face, upper));
    down[upper.index()] = Some((face, lower));
  }

  // Boundary radial caps per cell.
  let mut bottom_cap: Vec<Option<FaceId>> = vec![None; cells];
  let mut top_cap: Vec<Option<FaceId>> = vec![None; cells];
  for tag in mesh.boundary_tags() {
    for &(face, owner) in mesh.boundary_faces(tag) {
      let area = mesh.face_area(face);
      if area <= 0.0 {
        continue;
      }
      let out_sign = match mesh.face_connection(face) {
        FaceConnection::Boundary { out_sign, .. } => *out_sign,
        _ => continue,
      };
      let av = mesh.face_area_vector(face);
      let rh = r_hat(owner);
      // True outward normal points away from the cell.
      let outward =
        (av[0] * rh[0] + av[1] * rh[1] + av[2] * rh[2]) / area * out_sign;
      if outward.abs() <= 0.7 {
        continue;
      }
      // The boundary cap with no radial neighbour on that side.
      if outward > 0.0 && up[owner.index()].is_none() {
        top_cap[owner.index()] = Some(face);
      } else if outward < 0.0 && down[owner.index()].is_none() {
        bottom_cap[owner.index()] = Some(face);
      }
    }
  }

  // Walk each column from its bottom (no radial neighbour below).
  let mut columns = Vec::new();
  for start in 0..cells {
    if down[start].is_some() {
      continue;
    }
    let mut col_cells = vec![CellId::from(start)];
    let mut up_faces = Vec::new();
    let mut cur = start;
    while let Some((face, next)) = up[cur] {
      up_faces.push(face);
      col_cells.push(next);
      cur = next.index();
    }
    let top = col_cells.last().unwrap().index();
    columns.push(RadialColumn {
      bottom_face: bottom_cap[start],
      top_face: top_cap[top],
      cells: col_cells,
      up_faces,
    });
  }
  columns
}

/// One row of a block-tridiagonal system: `sub·x_{k-1} + diag·x_k +
/// sup·x_{k+1} = rhs`. For the first row `sub` is ignored, for the last `sup`.
#[derive(Clone)]
pub struct BlockRow<const N: usize> {
  pub sub: [[f64; N]; N],
  pub diag: [[f64; N]; N],
  pub sup: [[f64; N]; N],
  pub rhs: [f64; N],
}

fn matvec<const N: usize>(a: &[[f64; N]; N], x: &[f64; N]) -> [f64; N] {
  let mut y = [0.0; N];
  for r in 0..N {
    let mut acc = 0.0;
    for c in 0..N {
      acc += a[r][c] * x[c];
    }
    y[r] = acc;
  }
  y
}

fn matmul<const N: usize>(
  a: &[[f64; N]; N],
  b: &[[f64; N]; N],
) -> [[f64; N]; N] {
  let mut m = [[0.0; N]; N];
  for r in 0..N {
    for c in 0..N {
      let mut acc = 0.0;
      for k in 0..N {
        acc += a[r][k] * b[k][c];
      }
      m[r][c] = acc;
    }
  }
  m
}

fn sub_assign<const N: usize>(a: &mut [[f64; N]; N], b: &[[f64; N]; N]) {
  for r in 0..N {
    for c in 0..N {
      a[r][c] -= b[r][c];
    }
  }
}

fn vsub<const N: usize>(a: &[f64; N], b: &[f64; N]) -> [f64; N] {
  let mut y = [0.0; N];
  for i in 0..N {
    y[i] = a[i] - b[i];
  }
  y
}

/// Solve a block-tridiagonal system in place by block-Thomas elimination,
/// writing the solution into `out`. `rows.len()` is the column height; each
/// block is `N×N`. Returns `false` if a pivot block is singular (the caller
/// should then fall back to an explicit step for that column).
pub fn block_thomas<const N: usize>(
  rows: &[BlockRow<N>],
  out: &mut [[f64; N]],
) -> bool {
  let m = rows.len();
  debug_assert_eq!(out.len(), m);
  if m == 0 {
    return true;
  }

  // Forward elimination: Δ_k and modified rhs y_k.
  let mut delta_inv: Vec<[[f64; N]; N]> = Vec::with_capacity(m);
  let mut y: Vec<[f64; N]> = Vec::with_capacity(m);

  let Some(d0_inv) = invert(rows[0].diag) else {
    return false;
  };
  delta_inv.push(d0_inv);
  y.push(rows[0].rhs);

  for k in 1..m {
    // factor = sub_k · Δ_{k-1}^{-1}
    let factor = matmul(&rows[k].sub, &delta_inv[k - 1]);
    // Δ_k = diag_k − factor · sup_{k-1}
    let mut dk = rows[k].diag;
    sub_assign(&mut dk, &matmul(&factor, &rows[k - 1].sup));
    let Some(dk_inv) = invert(dk) else {
      return false;
    };
    delta_inv.push(dk_inv);
    // y_k = rhs_k − factor · y_{k-1}
    let yk = vsub(&rows[k].rhs, &matvec(&factor, &y[k - 1]));
    y.push(yk);
  }

  // Back substitution.
  out[m - 1] = matvec(&delta_inv[m - 1], &y[m - 1]);
  for k in (0..m - 1).rev() {
    // x_k = Δ_k^{-1} · (y_k − sup_k · x_{k+1})
    let rhs = vsub(&y[k], &matvec(&rows[k].sup, &out[k + 1]));
    out[k] = matvec(&delta_inv[k], &rhs);
  }
  true
}

/// Real geometry of one interior radial face, precomputed once. `owner_is_lower`
/// records whether the face's canonical owner is the lower (inner) cell, which
/// fixes the flux sign on each side.
struct UpFace<const D: usize> {
  normal: Vector<f64, D>,
  scale: f64,
  owner_is_lower: bool,
}

fn build_up_faces<const D: usize, M>(
  mesh: &M,
  column: &RadialColumn,
) -> Vec<UpFace<D>>
where
  M: Mesh<D> + ?Sized,
{
  column
    .up_faces
    .iter()
    .enumerate()
    .map(|(k, &face)| {
      let lower = column.cells[k];
      let area = mesh.face_area(face);
      let owner_is_lower = match mesh.face_connection(face) {
        FaceConnection::Interior { owner, .. } => *owner == lower,
        _ => true,
      };
      UpFace {
        normal: mesh.face_area_vector(face) / area,
        scale: area * mesh.face_metrics(face).sqrt_metric,
        owner_is_lower,
      }
    })
    .collect()
}

/// Real geometry of a boundary radial cap (Ground / AtmosphereEdge): the
/// *outward* normal, the flux scale, and its boundary tag. Included in the
/// implicit operator so the end cells' vertical acoustic (wall reflection) is
/// implicit too — without it their diagonal block is rank-deficient.
struct CapFace<const D: usize> {
  normal: Vector<f64, D>,
  scale: f64,
  tag: utility::domain::BoundaryTag,
}

fn build_cap<const D: usize, M>(
  mesh: &M,
  face: Option<FaceId>,
) -> Option<CapFace<D>>
where
  M: Mesh<D> + ?Sized,
{
  let face = face?;
  let area = mesh.face_area(face);
  let (tag, out_sign) = match mesh.face_connection(face) {
    FaceConnection::Boundary { tag, out_sign, .. } => (*tag, *out_sign),
    _ => return None,
  };
  Some(CapFace {
    normal: mesh.face_area_vector(face) / area * out_sign,
    scale: area * mesh.face_metrics(face).sqrt_metric,
    tag,
  })
}

/// The vertical (radial) implicit residual at column position `k`, generic over
/// the scalar so its exact Jacobian can be taken by AD. It is the implicit
/// (acoustic) flux through cell `k`'s up and down radial faces, divided by
/// volume. Interior radial faces only — the boundary caps stay in the explicit
/// RHS, so the per-column system is purely tridiagonal.
#[allow(clippy::too_many_arguments)]
fn radial_residual<const D: usize, const N: usize, L, F, T>(
  law: &L,
  flux: &F,
  up: &[UpFace<D>],
  bottom: Option<&CapFace<D>>,
  top: Option<&CapFace<D>>,
  bcs: &BoundaryRegistry<D, N>,
  k: usize,
  m: usize,
  inv_vol: f64,
  s_down: &[T; N],
  s_k: &[T; N],
  s_up: &[T; N],
) -> [T; N]
where
  L: ConservationLaw<D, N>,
  F: NumericalFlux<D, N>,
  T: BoundaryScalar<D, N>,
{
  let mut acc = [T::from(0.0); N];

  // Up face between cells[k] (lower) and cells[k+1] (upper).
  if k + 1 < m {
    let uf = &up[k];
    let (left, right) = if uf.owner_is_lower {
      (s_k, s_up)
    } else {
      (s_up, s_k)
    };
    let f = flux.compute_implicit(law, left, right, &uf.normal);
    let sign = if uf.owner_is_lower { -1.0 } else { 1.0 };
    for i in 0..N {
      acc[i] = acc[i] + f[i] * (sign * uf.scale);
    }
  }

  // Down face = up_faces[k-1], between cells[k-1] (lower) and cells[k] (upper).
  if k > 0 {
    let uf = &up[k - 1];
    let (left, right) = if uf.owner_is_lower {
      (s_down, s_k)
    } else {
      (s_k, s_down)
    };
    let f = flux.compute_implicit(law, left, right, &uf.normal);
    // cells[k] is the upper cell of this face.
    let sign = if uf.owner_is_lower { 1.0 } else { -1.0 };
    for i in 0..N {
      acc[i] = acc[i] + f[i] * (sign * uf.scale);
    }
  }

  // Boundary radial caps at the column ends (owner always loses through the
  // boundary face): ghost depends only on this cell, so it stays on the
  // diagonal block.
  if k == 0
    && let Some(cap) = bottom
    && let Some(bc) = bcs.get(cap.tag)
  {
    let ghost = T::ghost(bc, s_k, &cap.normal);
    let f = flux.compute_implicit(law, s_k, &ghost, &cap.normal);
    for i in 0..N {
      acc[i] = acc[i] - f[i] * cap.scale;
    }
  }
  if k + 1 == m
    && let Some(cap) = top
    && let Some(bc) = bcs.get(cap.tag)
  {
    let ghost = T::ghost(bc, s_k, &cap.normal);
    let f = flux.compute_implicit(law, s_k, &ghost, &cap.normal);
    for i in 0..N {
      acc[i] = acc[i] - f[i] * cap.scale;
    }
  }

  let mut out = [T::from(0.0); N];
  for i in 0..N {
    out[i] = acc[i] * inv_vol;
  }
  out
}

/// Build the `N×N` Jacobian block `∂R_vert_k/∂U_j` by forward-mode AD: seed each
/// component of the `j` state's dual part in turn and read the residual's dual
/// part. `which` selects which neighbour state is the variable.
#[allow(clippy::too_many_arguments)]
fn jacobian_block<const D: usize, const N: usize, L, F>(
  law: &L,
  flux: &F,
  up: &[UpFace<D>],
  bottom: Option<&CapFace<D>>,
  top: Option<&CapFace<D>>,
  bcs: &BoundaryRegistry<D, N>,
  k: usize,
  m: usize,
  inv_vol: f64,
  s_down: &[f64; N],
  s_k: &[f64; N],
  s_up: &[f64; N],
  which: Var,
) -> [[f64; N]; N]
where
  L: ConservationLaw<D, N>,
  F: NumericalFlux<D, N>,
{
  let dual = |s: &[f64; N], seed: Option<usize>| {
    let mut d = [Dual64::from(0.0); N];
    for i in 0..N {
      d[i] = Dual64::new(s[i], if seed == Some(i) { 1.0 } else { 0.0 });
    }
    d
  };
  let mut block = [[0.0; N]; N];
  for c in 0..N {
    let dd = dual(s_down, (which == Var::Down).then_some(c));
    let dk = dual(s_k, (which == Var::Self_).then_some(c));
    let du = dual(s_up, (which == Var::Up).then_some(c));
    let r = radial_residual(
      law, flux, up, bottom, top, bcs, k, m, inv_vol, &dd, &dk, &du,
    );
    for (row, rv) in r.iter().enumerate() {
      block[row][c] = rv.eps;
    }
  }
  block
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Var {
  Down,
  Self_,
  Up,
}

/// Vertically-implicit (HEVI) finite-volume backend. The full residual `R(U)`
/// is evaluated explicitly; only the vertical acoustic Jacobian `J_vert` is
/// taken implicitly, giving a per-column block-tridiagonal solve
/// `(I/dt − J_vert)·ΔU = R(U)`. Stable to the (much larger) horizontal CFL.
///
/// Built with the mesh's radial column structure (caller supplies it via
/// [`radial_columns_from_geometry`] so continuum stays domain-neutral). On a
/// thin shell this lifts the step size by ~the vertical/horizontal aspect ratio.
pub struct HeviBackend<const N: usize> {
  columns: Vec<RadialColumn>,
  up_faces: Vec<Vec<UpFace<3>>>, // per column, built lazily from the mesh
  caps: Vec<(Option<CapFace<3>>, Option<CapFace<3>>)>, // (bottom, top)
  built: bool,
  state_cache: Vec<[f64; N]>,
  residual_cache: Vec<[f64; N]>,
  accum: Vec<[f64; N]>,
  delta: Vec<[f64; N]>,
  fell_back: usize,
}

impl<const N: usize> HeviBackend<N> {
  pub fn new(columns: Vec<RadialColumn>) -> Self {
    Self {
      columns,
      up_faces: Vec::new(),
      caps: Vec::new(),
      built: false,
      state_cache: Vec::new(),
      residual_cache: Vec::new(),
      accum: Vec::new(),
      delta: Vec::new(),
      fell_back: 0,
    }
  }

  /// Columns where the tridiagonal solve was singular and fell back to an
  /// explicit (forward-Euler) update, in the most recent step.
  pub fn fallback_columns(&self) -> usize {
    self.fell_back
  }

  fn advance<L, F, S, M>(
    &mut self,
    law: &L,
    flux: &F,
    dt: f64,
    state: &mut S,
    mesh: &M,
    bcs: &BoundaryRegistry<3, N>,
  ) where
    L: ConservationLaw<3, N>,
    F: NumericalFlux<3, N>,
    S: FieldStorage<N>,
    M: Mesh<3> + ?Sized,
  {
    let cells = mesh.cell_count();
    if self.state_cache.len() != cells {
      self.state_cache.resize(cells, [0.0; N]);
      self.residual_cache.resize(cells, [0.0; N]);
      self.delta.resize(cells, [0.0; N]);
    }
    if !self.built {
      self.up_faces = self
        .columns
        .iter()
        .map(|c| build_up_faces(mesh, c))
        .collect();
      self.caps = self
        .columns
        .iter()
        .map(|c| (build_cap(mesh, c.bottom_face), build_cap(mesh, c.top_face)))
        .collect();
      self.built = true;
    }

    kernel::gather_state_cache(state, mesh, &mut self.state_cache);
    // Full explicit residual R(U) — the system right-hand side.
    kernel::compute_residual_into(
      law,
      flux,
      &self.state_cache,
      &mut self.accum,
      &mut self.residual_cache,
      mesh,
      bcs,
    );

    self.fell_back = 0;
    let zero = [0.0; N];
    let mut rows: Vec<BlockRow<N>> = Vec::new();
    let mut sol: Vec<[f64; N]> = Vec::new();

    for ((col, up), caps) in
      self.columns.iter().zip(&self.up_faces).zip(&self.caps)
    {
      let (bottom, top) = (caps.0.as_ref(), caps.1.as_ref());
      let m = col.cells.len();
      rows.clear();
      let inv_dt = 1.0 / dt;
      for k in 0..m {
        let ck = col.cells[k].index();
        let s_k = self.state_cache[ck];
        let s_down = if k > 0 {
          self.state_cache[col.cells[k - 1].index()]
        } else {
          zero
        };
        let s_up = if k + 1 < m {
          self.state_cache[col.cells[k + 1].index()]
        } else {
          zero
        };

        let inv_vol = 1.0 / mesh.cell_metrics(col.cells[k]).phys_volume;

        // diag = I/dt − ∂R_vert_k/∂U_k
        let mut diag = jacobian_block(
          law,
          flux,
          up,
          bottom,
          top,
          bcs,
          k,
          m,
          inv_vol,
          &s_down,
          &s_k,
          &s_up,
          Var::Self_,
        );
        for r in 0..N {
          for c in 0..N {
            diag[r][c] = -diag[r][c];
          }
          diag[r][r] += inv_dt;
        }
        let sub = if k > 0 {
          let mut b = jacobian_block(
            law,
            flux,
            up,
            bottom,
            top,
            bcs,
            k,
            m,
            inv_vol,
            &s_down,
            &s_k,
            &s_up,
            Var::Down,
          );
          negate(&mut b);
          b
        } else {
          [[0.0; N]; N]
        };
        let sup = if k + 1 < m {
          let mut b = jacobian_block(
            law,
            flux,
            up,
            bottom,
            top,
            bcs,
            k,
            m,
            inv_vol,
            &s_down,
            &s_k,
            &s_up,
            Var::Up,
          );
          negate(&mut b);
          b
        } else {
          [[0.0; N]; N]
        };

        rows.push(BlockRow {
          sub,
          diag,
          sup,
          rhs: self.residual_cache[ck],
        });
      }

      sol.resize(m, [0.0; N]);
      if block_thomas(&rows, &mut sol) {
        for k in 0..m {
          self.delta[col.cells[k].index()] = sol[k];
        }
      } else {
        // Singular column: fall back to an explicit forward-Euler update.
        self.fell_back += 1;
        for k in 0..m {
          let ck = col.cells[k].index();
          let mut d = [0.0; N];
          for i in 0..N {
            d[i] = dt * self.residual_cache[ck][i];
          }
          self.delta[ck] = d;
        }
      }
    }

    // U_new = U + ΔU, then enforce physical bounds.
    for i in 0..cells {
      for k in 0..N {
        self.state_cache[i][k] += self.delta[i][k];
      }
      law.fix_state(&mut self.state_cache[i]);
      state.write(CellId::from(i), &self.state_cache[i]);
    }
  }
}

fn negate<const N: usize>(b: &mut [[f64; N]; N]) {
  for row in b.iter_mut() {
    for v in row.iter_mut() {
      *v = -*v;
    }
  }
}

impl<const N: usize, L, F> FvmBackend<3, N, L, F> for HeviBackend<N>
where
  L: ConservationLaw<3, N>,
  F: NumericalFlux<3, N>,
{
  fn step<S, M>(
    &mut self,
    config: &SolverConfig,
    law: &L,
    flux: &F,
    state: &mut S,
    residual: &mut S,
    mesh: &M,
    bcs: &BoundaryRegistry<3, N>,
  ) -> f64
  where
    S: FieldStorage<N>,
    M: Mesh<3> + ?Sized,
  {
    let _ = residual;
    if self.state_cache.len() != mesh.cell_count() {
      self.state_cache.resize(mesh.cell_count(), [0.0; N]);
    }
    kernel::gather_state_cache(state, mesh, &mut self.state_cache);
    // HEVI is stable to the horizontal (advective+acoustic) CFL.
    let dt = kernel::compute_explicit_dt_from_cache(
      config,
      law,
      &self.state_cache,
      mesh,
    );
    self.advance(law, flux, dt, state, mesh, bcs);
    dt
  }

  fn step_with_dt<S, M>(
    &mut self,
    config: &SolverConfig,
    law: &L,
    flux: &F,
    dt: f64,
    state: &mut S,
    residual: &mut S,
    mesh: &M,
    bcs: &BoundaryRegistry<3, N>,
  ) where
    S: FieldStorage<N>,
    M: Mesh<3> + ?Sized,
  {
    let _ = (config, residual);
    self.advance(law, flux, dt, state, mesh, bcs);
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Build a random-ish scalar (N=1) tridiagonal system with a known solution
  /// and check block-Thomas recovers it.
  #[test]
  fn block_thomas_scalar_tridiagonal() {
    let m = 8;
    let x_true: Vec<f64> = (0..m).map(|i| (i as f64 * 0.5).sin()).collect();
    // Diagonally-dominant tridiagonal: diag 4, off -1.
    let mut rows: Vec<BlockRow<1>> = Vec::new();
    for k in 0..m {
      let mut rhs = 4.0 * x_true[k];
      if k > 0 {
        rhs += -1.0 * x_true[k - 1];
      }
      if k + 1 < m {
        rhs += -1.0 * x_true[k + 1];
      }
      rows.push(BlockRow {
        sub: [[-1.0]],
        diag: [[4.0]],
        sup: [[-1.0]],
        rhs: [rhs],
      });
    }
    let mut out = vec![[0.0; 1]; m];
    assert!(block_thomas(&rows, &mut out));
    for k in 0..m {
      assert!(
        (out[k][0] - x_true[k]).abs() < 1e-12,
        "row {k}: {} vs {}",
        out[k][0],
        x_true[k]
      );
    }
  }

  /// A 2×2-block tridiagonal system with a known solution.
  #[test]
  fn block_thomas_2x2_blocks() {
    let m = 5;
    let diag = [[6.0, 1.0], [-1.0, 5.0]];
    let sub = [[-1.0, 0.0], [0.0, -1.0]];
    let sup = [[-1.0, 0.5], [0.0, -1.0]];
    let x_true: Vec<[f64; 2]> = (0..m)
      .map(|i| [(i as f64).cos(), (i as f64 * 0.3).sin()])
      .collect();

    let mut rows: Vec<BlockRow<2>> = Vec::new();
    for k in 0..m {
      let mut rhs = mv(&diag, &x_true[k]);
      if k > 0 {
        let t = mv(&sub, &x_true[k - 1]);
        rhs = [rhs[0] + t[0], rhs[1] + t[1]];
      }
      if k + 1 < m {
        let t = mv(&sup, &x_true[k + 1]);
        rhs = [rhs[0] + t[0], rhs[1] + t[1]];
      }
      rows.push(BlockRow {
        sub,
        diag,
        sup,
        rhs,
      });
    }
    let mut out = vec![[0.0; 2]; m];
    assert!(block_thomas(&rows, &mut out));
    for k in 0..m {
      for c in 0..2 {
        assert!((out[k][c] - x_true[k][c]).abs() < 1e-10, "row {k} comp {c}");
      }
    }
  }

  fn mv(a: &[[f64; 2]; 2], x: &[f64; 2]) -> [f64; 2] {
    [
      a[0][0] * x[0] + a[0][1] * x[1],
      a[1][0] * x[0] + a[1][1] * x[1],
    ]
  }

  #[test]
  fn radial_columns_on_cube_sphere() {
    use tessera::cube_sphere::CubeSphere;
    use tessera::geometry::CellGeometry;
    let (na, nr) = (6, 5);
    let mesh = CubeSphere::new([na, na, nr], 1.0, 1.2);
    let columns = radial_columns_from_geometry(&mesh, |c| {
      let p = mesh.cell_world_centroid(c);
      [p[0], p[1], p[2]]
    });

    // One column per angular cell across all 6 panels; each is nr cells tall,
    // capped top and bottom, with nr-1 interior radial links.
    assert_eq!(columns.len(), 6 * na * na, "column count");
    let total: usize = columns.iter().map(|c| c.cells.len()).sum();
    assert_eq!(total, mesh.cell_count(), "all cells covered exactly once");
    for col in &columns {
      assert_eq!(col.cells.len(), nr, "column height");
      assert_eq!(col.up_faces.len(), nr - 1);
      assert!(col.bottom_face.is_some(), "ground cap");
      assert!(col.top_face.is_some(), "atmosphere-edge cap");
      // Strictly increasing radius bottom→top.
      let radii: Vec<f64> = col
        .cells
        .iter()
        .map(|&c| {
          let p = mesh.cell_world_centroid(c);
          (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt()
        })
        .collect();
      for w in radii.windows(2) {
        assert!(w[0] < w[1], "column ordered bottom→top");
      }
    }
  }
}
