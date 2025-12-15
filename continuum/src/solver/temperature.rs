use crate::geometry::VecN;
use crate::solver::fv::{Flux, Model, State};
use crate::topology::BoundaryId;

/// Scalar temperature T satisfies:
///   ∂T/∂t + ∇·(v T) = ∇·(κ ∇T) + Q
///
/// FV implementation here:
/// - advection handled via Rusanov numerical flux on vT
/// - diffusion handled via centered normal gradient using (T_R - T_L)/dist_n
#[derive(Debug, Clone)]
pub struct TemperatureAdvectionDiffusion<const D: usize> {
  /// Constant advection velocity in physical space.
  pub vel: VecN<D>,
  /// Constant diffusivity κ (>= 0).
  pub kappa: f64,

  /// Optional volumetric heat source Q (adds to dT/dt).
  pub source_q: f64,

  /// Boundary temperature (Dirichlet).   
  pub dirichlet_T: f64,

  /// If true: boundary is Dirichlet; if false: simple outflow/zero-gradient.
  pub use_dirichlet: bool,
}

impl<const D: usize> TemperatureAdvectionDiffusion<D> {
  pub fn new(vel: VecN<D>, kappa: f64) -> Self {
    Self {
      vel,
      kappa: kappa.max(0.0),
      source_q: 0.0,
      dirichlet_T: 0.0,
      use_dirichlet: false,
    }
  }

  pub fn with_source(mut self, q: f64) -> Self {
    self.source_q = q;
    self
  }

  pub fn with_dirichlet(mut self, t_bc: f64) -> Self {
    self.use_dirichlet = true;
    self.dirichlet_T = t_bc;
    self
  }

  pub fn with_outflow(mut self) -> Self {
    self.use_dirichlet = false;
    self
  }
}

impl<const D: usize> Model<D, 1> for TemperatureAdvectionDiffusion<D> {
  fn flux(&self, u: &State<1>) -> Flux<D, 1> {
    // Physical flux vector for advection: F = v*T
    let t = u[0];
    let mut f = [[0.0; D]; 1];
    for d in 0..D {
      f[0][d] = self.vel[d] * t;
    }
    f
  }

  fn max_wave_speed(&self, _u: &State<1>, n_unit: VecN<D>) -> f64 {
    // |v · n|
    let mut s = 0.0;
    for d in 0..D {
      s += self.vel[d] * n_unit[d];
    }
    s.abs()
  }

  fn boundary_state(
    &self,
    _boundary: BoundaryId,
    interior: &State<1>,
    _x_face: VecN<D>,
    _t: f64,
  ) -> State<1> {
    if self.use_dirichlet {
      [self.dirichlet_T]
    } else {
      // simple outflow/zero-gradient
      *interior
    }
  }

  fn source(&self, _u: &State<1>, _x_cell: VecN<D>, _t: f64) -> State<1> {
    // Add volumetric source Q directly to dT/dt
    [self.source_q]
  }

  fn diffusive_flux_unit_normal(
    &self,
    ul: &State<1>,
    ur: &State<1>,
    _n_unit: VecN<D>,
    dist_n: f64,
    _t: f64,
  ) -> State<1> {
    if self.kappa == 0.0 {
      return [0.0];
    }

    // F_diff · n = -κ * ∂T/∂n  ≈ -κ * (T_R - T_L)/dist_n
    let dtdn = (ur[0] - ul[0]) / dist_n;
    [-self.kappa * dtdn]
  }

  fn diffusion_lipschitz_coeff(
    &self,
    _ul: &State<1>,
    _ur: &State<1>,
    _n_unit: VecN<D>,
    dist_n: f64,
    _t: f64,
  ) -> State<1> {
    if self.kappa == 0.0 {
      [0.0]
    } else {
      // |∂(F_diff·n)/∂U| ≈ κ / dist_n
      [self.kappa / dist_n]
    }
  }
}
