//! solver.rs
//! Generic explicit finite-volume solver using Grid + Topology + Geometry.
//!
//! - Logical domain: structured Cartesian indices over [0,1]^D (Grid)
//! - Connectivity: neighbors/boundaries (Topology)
//! - Physical measures: V_i and S_f (Geometry)
//! - Model provides flux and characteristic speeds (user-defined)

use crate::geometry::{Geometry, VecN};
use crate::grid::grid::{CellId, FaceId};
use crate::topology::{BoundaryId, Neighbor, Topology};

/// Conserved state per cell.
pub type State<const NV: usize> = [f64; NV];

/// Flux tensor: for each conserved component, a physical-space flux vector in R^D.
pub type Flux<const D: usize, const NV: usize> = [[f64; D]; NV];

fn dot<const D: usize>(a: VecN<D>, b: VecN<D>) -> f64 {
  let mut s = 0.0;
  for i in 0..D {
    s += a[i] * b[i];
  }
  s
}

fn norm<const D: usize>(v: VecN<D>) -> f64 {
  dot::<D>(v, v).sqrt()
}

fn add_state<const NV: usize>(a: &mut State<NV>, b: State<NV>) {
  for m in 0..NV {
    a[m] += b[m];
  }
}

fn sub_state<const NV: usize>(a: &mut State<NV>, b: State<NV>) {
  for m in 0..NV {
    a[m] -= b[m];
  }
}

fn scale_state<const NV: usize>(a: &mut State<NV>, s: f64) {
  for m in 0..NV {
    a[m] *= s;
  }
}

fn zero_state<const NV: usize>() -> State<NV> {
  [0.0; NV]
}

/// User model: provides physical fluxes and wave speeds.
/// Boundary conditions are handled by constructing an "exterior" state for boundary faces.
///
/// This keeps the solver generic: it only needs a numerical flux per face.
pub trait Model<const D: usize, const NV: usize> {
  /// Physical flux F(U) as a vector in physical space for each component.
  fn flux(&self, u: &State<NV>) -> Flux<D, NV>;

  /// Maximum signal speed in direction of unit normal `n_unit`.
  /// Used by Rusanov flux.
  fn max_wave_speed(&self, u: &State<NV>, n_unit: VecN<D>) -> f64;

  /// Provide exterior/ghost state for a boundary face.
  /// - `boundary`: which boundary (axis + side) this is
  /// - `interior`: state inside the domain
  /// - `x_face`: physical face center (often useful for BCs)
  /// - `t`: current time
  fn boundary_state(
    &self,
    boundary: BoundaryId,
    interior: &State<NV>,
    x_face: VecN<D>,
    t: f64,
  ) -> State<NV>;

  /// Optional source term (cell-averaged) dU/dt += S(U, x, t).
  fn source(&self, _u: &State<NV>, _x_cell: VecN<D>, _t: f64) -> State<NV> {
    zero_state::<NV>()
  }

  /// Optional: diffusive normal flux per unit area (i.e. component-wise F_diff · n).
  /// `dist_n` is the (positive) distance between left/right states along the normal direction.
  fn diffusive_flux_unit_normal(
    &self,
    _ul: &State<NV>,
    _ur: &State<NV>,
    _n_unit: VecN<D>,
    _dist_n: f64,
    _t: f64,
  ) -> State<NV> {
    [0.0; NV]
  }
}

/// Generic explicit finite-volume solver (Euler forward).
#[derive(Debug, Clone)]
pub struct FiniteVolumeSolver<Topo, Geom, M, const D: usize, const NV: usize>
where
    Topo: Topology<D>,
    Geom: Geometry<D>,
    M: Model<D, NV>,
  {
    pub topo: Topo,
    pub geom: Geom,
    pub model: M,

    /// Conservative cell-averages U_i (length = grid.cell_count()).
    pub u: Vec<State<NV>>,

    /// Current simulation time (optional convenience).
    pub time: f64,
  }

impl<Topo, Geom, M, const D: usize, const NV: usize> FiniteVolumeSolver<Topo, Geom, M, D, NV>
where
  Topo: Topology<D>,
  Geom: Geometry<D>,
  M: Model<D, NV>,
{
  pub fn new(topo: Topo, geom: Geom, model: M) -> Self {
    let grid = topo.grid();
    let n = grid.cell_count();
    Self {
      topo,
      geom,
      model,
      u: vec![[0.0; NV]; n],
      time: 0.0,
    }
  }

  /// Set initial condition from a closure in physical space.
  pub fn initialize_with<F>(&mut self, mut f: F)
where
    F: FnMut(VecN<D>) -> State<NV>,
  {
    let grid = self.topo.grid();
    for cell in grid.iter_cells() {
      let idx = grid.cell_linear(cell);
      let x = self.geom.cell_center_x(cell);
      self.u[idx] = f(x);
    }
  }

  /// A standard Rusanov (local Lax-Friedrichs) numerical flux through a *unit* normal.
  /// Returns flux per unit area: \hat{F} · n (one scalar per conserved component).
  fn rusanov_flux_unit_normal(&self, ul: &State<NV>, ur: &State<NV>, n_unit: VecN<D>) -> State<NV> {
    let fl = self.model.flux(ul);
    let fr = self.model.flux(ur);

    // Compute normal flux components: (F · n)
    let mut fn_l = [0.0; NV];
    let mut fn_r = [0.0; NV];
    for m in 0..NV {
      fn_l[m] = dot::<D>(fl[m], n_unit);
      fn_r[m] = dot::<D>(fr[m], n_unit);
    }

    // Max wave speed across interface
    let a_l = self.model.max_wave_speed(ul, n_unit);
    let a_r = self.model.max_wave_speed(ur, n_unit);
    let a = a_l.max(a_r);

    // Rusanov: 0.5(Fn(UL)+Fn(UR)) - 0.5*a*(UR-UL)
    let mut out = [0.0; NV];
    for m in 0..NV {
      out[m] = 0.5 * (fn_l[m] + fn_r[m]) - 0.5 * a * (ur[m] - ul[m]);
    }
    out
  }

  /// Explicit Euler step:
  /// U^{n+1}_i = U^n_i - dt/Vol_i * sum_faces( Flux * AreaVector ) + dt * Source
  pub fn step_explicit(&mut self, dt: f64) {
    let grid = self.topo.grid();
    let mut rhs = vec![[0.0; NV]; grid.cell_count()];

    // Accumulate flux divergence and sources cell-by-cell.
    for cell in grid.iter_cells() {
      let i = grid.cell_linear(cell);
      let ui = self.u[i];

      // Source term contribution (cell-averaged)
      let x_cell = self.geom.cell_center_x(cell);
      add_state::<NV>(&mut rhs[i], self.model.source(&ui, x_cell, self.time));

      // Face flux contributions
      for face in self.topo.faces_of_cell(cell) {
        let s = self.geom.face_area_vector(face); // physical area-vector (outward for this cell)
        let area = norm::<D>(s);

        // Degenerate face (shouldn't happen in sane grids/mappings, but guard anyway)
        if area == 0.0 {
          continue;
        }

        let n_unit: VecN<D> = {
          let mut out = [0.0; D];
          for d in 0..D {
            out[d] = s[d] / area;
          }
          out
        };

        // Left state = this cell
        let ul = ui;

        // Right state from neighbor or boundary
        let ur = match self.topo.neighbor_across(face) {
          Neighbor::Cell(nc) => {
            let j = grid.cell_linear(nc);
            self.u[j]
          }
          Neighbor::Boundary(bid) => {
            let x_face = self.geom.face_center_x(face);
            self.model.boundary_state(bid, &ul, x_face, self.time)
          }
        };

        // Numerical advective flux per unit area through the face (Rusanov)
        let mut fhat_n = self.rusanov_flux_unit_normal(&ul, &ur, n_unit);

        // Normal distance between L/R sample points (use cell centers)
        let x_l = self.geom.cell_center_x(face.cell);
        let x_r = match self.topo.neighbor_across(face) {
          Neighbor::Cell(nc) => self.geom.cell_center_x(nc),
          Neighbor::Boundary(bid) => {
            // For boundaries, we can approximate the "exterior" point by mirroring
            // across the face center. This gives a reasonable dist for Dirichlet/outflow.
            let xf = self.geom.face_center_x(face);
            let mut xr = [0.0; D];
            for d in 0..D {
              xr[d] = 2.0 * xf[d] - x_l[d];
            }
            xr
          }
        };

        let mut dx = [0.0; D];
        for d in 0..D { dx[d] = x_r[d] - x_l[d]; }
        let dist_n = dot::<D>(dx, n_unit).abs().max(1e-12);

        // Add diffusion: fhat_n += F_diff · n
        let f_diff_n = self.model.diffusive_flux_unit_normal(&ul, &ur, n_unit, dist_n, self.time);
        for m in 0..NV {
          fhat_n[m] += f_diff_n[m];
        }

        // Multiply by physical area to get integrated flux through the face
        let mut flux_integral = fhat_n;
        scale_state::<NV>(&mut flux_integral, area);

        // FV sign convention: outward flux leaves the cell
        // dU/dt += -(1/Vol) * sum(outward flux integrals)
        sub_state::<NV>(&mut rhs[i], flux_integral);
      }

      // Divide by cell volume
      let vol = self.geom.cell_volume(cell);
      let inv_vol = 1.0 / vol;
      scale_state::<NV>(&mut rhs[i], inv_vol);
    }

    // Apply explicit Euler update
    for i in 0..self.u.len() {
      for m in 0..NV {
        self.u[i][m] += dt * rhs[i][m];
      }
    }

    self.time += dt;
  }
}
