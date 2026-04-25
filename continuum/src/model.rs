use crate::geometry::{CellId, CellMetrics, Point};
use crate::output::LawFieldSchema;
use utility::{maths::vector::Vector, profile};

pub trait ConservationLaw<const D: usize, const N: usize>: Send + Sync {
  fn fix_state(&self, state: &mut [f64; N]);
  fn flux(&self, state: &[f64; N]) -> [[f64; N]; D];
  fn max_wave_speed(&self, state: &[f64; N]) -> f64;
  /// Per-cell source term. `cell` is the global cell ID — laws that carry
  /// pre-computed per-cell data (e.g. radial gravity vectors) index into
  /// their own arrays with it. Laws with no spatial dependence ignore it.
  fn source(
    &self,
    state: &[f64; N],
    cell: CellId,
    centroid: &Point<D>,
    metrics: &CellMetrics<D>,
  ) -> [f64; N];
}

pub trait NumericalFlux<const D: usize, const N: usize>: Send + Sync {
  fn compute(
    &self,
    law: &dyn ConservationLaw<D, N>,
    left: &[f64; N],
    right: &[f64; N],
    normal: &Vector<f64, D>,
  ) -> [f64; N];
}

pub struct RusanovFlux;

impl<const D: usize, const N: usize> NumericalFlux<D, N> for RusanovFlux {
  #[profile]
  fn compute(
    &self,
    law: &dyn ConservationLaw<D, N>,
    left: &[f64; N],
    right: &[f64; N],
    normal: &Vector<f64, D>,
  ) -> [f64; N] {
    let fl = law.flux(left);
    let fr = law.flux(right);
    let s_max = law.max_wave_speed(left).max(law.max_wave_speed(right));

    let mut result = [0.0; N];
    for i in 0..N {
      let mut fn_avg = 0.0;
      for d in 0..D {
        fn_avg += 0.5 * (fl[d][i] + fr[d][i]) * normal[d];
      }
      result[i] = fn_avg - 0.5 * s_max * (right[i] - left[i]);
    }
    result
  }
}

pub struct Euler2D {
  gamma: f64, // Ratio of specific heats
}

impl Euler2D {
  pub fn new(gamma: f64) -> Euler2D {
    Euler2D { gamma }
  }

  pub fn pressure(&self, state: &[f64; 4]) -> f64 {
    let rho = state[0];
    let u = state[1] / rho;
    let v = state[2] / rho;
    (self.gamma - 1.0) * (state[3] - 0.5 * rho * (u * u + v * v))
  }
}

impl ConservationLaw<2, 4> for Euler2D {
  fn flux(&self, state: &[f64; 4]) -> [[f64; 4]; 2] {
    let rho = state[0];

    let u = state[1] / rho;
    let v = state[2] / rho;
    let p = self.pressure(state);

    let fx = [
      state[1],           // rho * u
      state[1] * u + p,   // rho * u^2 + p
      state[1] * v,       // rho * u * v
      (state[3] + p) * u, // (E + p) * u
    ];
    let fy = [
      state[2],           // rho * v
      state[2] * u,       // rho * v * u
      state[2] * v + p,   // rho * v^2 + p
      (state[3] + p) * v, // (E + p) * v
    ];
    [fx, fy]
  }

  fn max_wave_speed(&self, state: &[f64; 4]) -> f64 {
    let rho = state[0];
    let u = state[1] / rho;
    let v = state[2] / rho;
    let p = self.pressure(state);
    let c = (self.gamma * p / rho).sqrt();
    (u * u + v * v).sqrt() + c
  }

  fn source(
    &self,
    _state: &[f64; 4],
    _cell: CellId,
    _centroid: &Point<2>,
    _metrics: &CellMetrics<2>,
  ) -> [f64; 4] {
    [0.0; 4] // no source terms for basic Euler
  }

  #[profile]
  fn fix_state(&self, state: &mut [f64; 4]) {
    let floor = 1e-8;
    if state[0] < floor {
      state[0] = floor;
    }
    let rho = state[0];
    let u = state[1] / rho;
    let v = state[2] / rho;
    let ke = 0.5 * rho * (u * u + v * v);
    if state[3] - ke < floor {
      state[3] = ke + floor;
    }
  }
}

/// Body-force-per-unit-mass field for `Euler3D`. Choose between no gravity
/// (free flow), a single constant vector (flat box), or a per-cell vector
/// (radial gravity on a sphere shell, or any other spatially-varying field).
pub enum GravityField {
  None,
  Constant([f64; 3]),
  /// One gravity vector per global cell ID.
  PerCell(Vec<[f64; 3]>),
}

/// Compressible Euler in 3D with an optional body-force per unit mass.
/// State = `[ρ, ρu, ρv, ρw, E]`.
///
/// Designed for the cube-sphere shell with Cartesian world-frame momentum
/// (u, v, w in world xyz).
pub struct Euler3D {
  gamma: f64,
  gravity: GravityField,
}

impl Euler3D {
  /// Construct without gravity (free flow).
  pub fn new(gamma: f64) -> Self {
    Self { gamma, gravity: GravityField::None }
  }

  /// Construct with a constant gravity vector (force per unit mass) — right
  /// for flat boxes, e.g. `[0, 0, -9.81]`.
  pub fn with_gravity(gamma: f64, gravity: [f64; 3]) -> Self {
    Self { gamma, gravity: GravityField::Constant(gravity) }
  }

  /// Construct with a precomputed per-cell gravity field — right for radial
  /// gravity on a sphere shell. The vector is indexed by global cell ID and
  /// must have length equal to the mesh's cell count. Build it via
  /// `CubeSphere::radial_gravity_field(g)`.
  pub fn with_per_cell_gravity(gamma: f64, gravity: Vec<[f64; 3]>) -> Self {
    Self { gamma, gravity: GravityField::PerCell(gravity) }
  }

  pub fn pressure(&self, state: &[f64; 5]) -> f64 {
    let rho = state[0];
    let inv_rho = 1.0 / rho;
    let ke = 0.5 * inv_rho
      * (state[1].powi(2) + state[2].powi(2) + state[3].powi(2));
    (self.gamma - 1.0) * (state[4] - ke)
  }

  pub fn gamma(&self) -> f64 {
    self.gamma
  }
}

impl ConservationLaw<3, 5> for Euler3D {
  fn flux(&self, state: &[f64; 5]) -> [[f64; 5]; 3] {
    let rho = state[0];
    let u = state[1] / rho;
    let v = state[2] / rho;
    let w = state[3] / rho;
    let p = self.pressure(state);
    let h = state[4] + p;

    let fx = [state[1], state[1] * u + p, state[1] * v, state[1] * w, h * u];
    let fy = [state[2], state[2] * u, state[2] * v + p, state[2] * w, h * v];
    let fz = [state[3], state[3] * u, state[3] * v, state[3] * w + p, h * w];
    [fx, fy, fz]
  }

  fn max_wave_speed(&self, state: &[f64; 5]) -> f64 {
    let rho = state[0];
    let u = state[1] / rho;
    let v = state[2] / rho;
    let w = state[3] / rho;
    let p = self.pressure(state);
    let c = (self.gamma * p / rho).sqrt();
    (u * u + v * v + w * w).sqrt() + c
  }

  fn source(
    &self,
    state: &[f64; 5],
    cell: CellId,
    _: &Point<3>,
    _: &CellMetrics<3>,
  ) -> [f64; 5] {
    let g = match &self.gravity {
      GravityField::None => return [0.0; 5],
      GravityField::Constant(g) => *g,
      GravityField::PerCell(field) => field[cell.index()],
    };
    let rho = state[0];
    // Gravity: force/volume = ρ·g on momentum, work/volume = (ρu)·g on energy.
    [
      0.0,
      rho * g[0],
      rho * g[1],
      rho * g[2],
      state[1] * g[0] + state[2] * g[1] + state[3] * g[2],
    ]
  }

  #[profile]
  fn fix_state(&self, state: &mut [f64; 5]) {
    let floor = 1e-8;
    if state[0] < floor {
      state[0] = floor;
    }
    let rho = state[0];
    let ke = 0.5 / rho
      * (state[1].powi(2) + state[2].powi(2) + state[3].powi(2));
    if state[4] - ke < floor {
      state[4] = ke + floor;
    }
  }
}

impl LawFieldSchema<3, 5> for Euler3D {
  fn conserved_field_names(&self) -> [&'static str; 5] {
    ["rho", "rho_u", "rho_v", "rho_w", "energy"]
  }

  fn derived_field_names(&self) -> &'static [&'static str] {
    &["u", "v", "w", "pressure"]
  }

  fn write_derived_fields(
    &self,
    state: &[f64; 5],
    _centroid: &Point<3>,
    _metrics: &CellMetrics<3>,
    out: &mut [f64],
  ) {
    debug_assert_eq!(out.len(), 4);
    let rho = state[0];
    out[0] = state[1] / rho;
    out[1] = state[2] / rho;
    out[2] = state[3] / rho;
    out[3] = self.pressure(state);
  }
}

impl LawFieldSchema<2, 4> for Euler2D {
  fn conserved_field_names(&self) -> [&'static str; 4] {
    ["rho", "rho_u", "rho_v", "energy"]
  }

  fn derived_field_names(&self) -> &'static [&'static str] {
    &["u", "v", "pressure"]
  }

  fn write_derived_fields(
    &self,
    state: &[f64; 4],
    _centroid: &Point<2>,
    _metrics: &CellMetrics<2>,
    out: &mut [f64],
  ) {
    debug_assert_eq!(out.len(), 3);
    let rho = state[0];
    out[0] = state[1] / rho;
    out[1] = state[2] / rho;
    out[2] = self.pressure(state);
  }
}
