// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use num_dual::DualNum;
use utility::domain::{CellId, Point};
use utility::{StateDiagnostics, maths::vector::Vector};

use tessera::geometry::CellMetrics;

use crate::output::LawFieldSchema;

/// The scalar field the conservation-law math is generic over. It is `f64` for
/// the ordinary (explicit / residual) path and a forward-mode dual number for
/// exact automatic-differentiation Jacobians. `f64: Scalar` and
/// `Dual64: Scalar`, so the same law body serves both with no duplication.
///
/// Note: dual arithmetic only supports `dual * f64` (not `f64 * dual`), so law
/// bodies keep the real operand on the right of any mixed product.
pub trait Scalar: DualNum<f64> + Copy {}
impl<T: DualNum<f64> + Copy> Scalar for T {}

/// Branch-select max for scalars (Rosenbrock dual numbers compare by real
/// part), used for the Rusanov dissipation speed.
fn max_scalar<T: Scalar>(a: T, b: T) -> T {
  if a > b { a } else { b }
}

/// Regularisation added inside the velocity-magnitude `sqrt` in wave-speed
/// estimates. `sqrt` has an infinite slope at 0, so a state at rest makes the
/// automatic-differentiation Jacobian NaN; this tiny floor keeps the dual
/// derivative finite (its value at the kink, 0) while shifting the real result
/// by at most ~1e-9 m/s — utterly negligible against the ~340 m/s sound speed.
const SPEED_EPS: f64 = 1e-18;

pub trait ConservationLaw<const D: usize, const N: usize>: Send + Sync {
  /// Clamp a *real* state back to the physical set (positive density etc.).
  /// Only applied to materialised states, never to dual numbers.
  fn fix_state(&self, state: &mut [f64; N]);
  fn flux<T: Scalar>(&self, state: &[T; N]) -> [[T; N]; D];
  fn max_wave_speed<T: Scalar>(&self, state: &[T; N]) -> T;
  /// Per-cell source term. `cell` is the global cell ID — laws that carry
  /// pre-computed per-cell data (e.g. radial gravity vectors) index into
  /// their own arrays with it. Laws with no spatial dependence ignore it.
  /// Geometry (`centroid`, `metrics`) stays real; only the state is generic.
  fn source<T: Scalar>(
    &self,
    state: &[T; N],
    cell: CellId,
    centroid: &Point<D>,
    metrics: &CellMetrics<D>,
  ) -> [T; N];

  // --- IMEX operator split (optional) ---
  //
  // A law may split its dynamics into an implicit part (stiff — e.g. the
  // acoustic terms) and an explicit remainder, so an IMEX integrator can step
  // only the stiff part implicitly. The defaults make the implicit operator
  // empty, so a law that does not opt in runs fully explicitly under IMEX and
  // every existing law is unaffected. The invariant the IMEX backend relies on
  // is `R = R_explicit + R_implicit`, with `R_implicit` built from
  // `implicit_flux` / `implicit_source` and the [`acoustic_speed`] dissipation.

  /// The implicit (stiff) part of the flux. Default: none — fully explicit.
  fn implicit_flux<T: Scalar>(&self, _state: &[T; N]) -> [[T; N]; D] {
    [[T::from(0.0); N]; D]
  }

  /// The implicit (stiff) part of the source. Default: none.
  fn implicit_source<T: Scalar>(
    &self,
    _state: &[T; N],
    _cell: CellId,
    _centroid: &Point<D>,
    _metrics: &CellMetrics<D>,
  ) -> [T; N] {
    [T::from(0.0); N]
  }

  /// Dissipation speed for the implicit numerical flux (the fast/acoustic wave
  /// speed). Default 0 → the implicit operator carries no dissipation, leaving
  /// the full Rusanov dissipation in the explicit remainder.
  fn acoustic_speed<T: Scalar>(&self, _state: &[T; N]) -> T {
    T::from(0.0)
  }

  /// Wave speed bounding the *explicit* part of an IMEX split, used to size the
  /// (advective) CFL step. Default: the full wave speed (no speed-up).
  fn explicit_wave_speed<T: Scalar>(&self, state: &[T; N]) -> T {
    self.max_wave_speed(state)
  }
}

pub trait NumericalFlux<const D: usize, const N: usize>: Send + Sync {
  /// Numerical flux through a face with unit `normal` (real geometry). Generic
  /// over the scalar and the law so dual evaluations need no `&dyn` and the
  /// Jacobian can be taken by automatic differentiation.
  fn compute<T: Scalar, L: ConservationLaw<D, N>>(
    &self,
    law: &L,
    left: &[T; N],
    right: &[T; N],
    normal: &Vector<f64, D>,
  ) -> [T; N];

  /// Numerical flux of the *implicit* part of the operator (for IMEX): a
  /// central average of `implicit_flux` plus dissipation scaled by the
  /// `acoustic_speed`. With the default (empty) implicit operator this is zero.
  /// Splitting the dissipation this way keeps `R = R_explicit + R_implicit`:
  /// the explicit remainder keeps `(full_speed − acoustic_speed)` dissipation.
  fn compute_implicit<T: Scalar, L: ConservationLaw<D, N>>(
    &self,
    law: &L,
    left: &[T; N],
    right: &[T; N],
    normal: &Vector<f64, D>,
  ) -> [T; N] {
    let fl = law.implicit_flux(left);
    let fr = law.implicit_flux(right);
    let s = max_scalar(law.acoustic_speed(left), law.acoustic_speed(right));

    let mut result = [T::from(0.0); N];
    for i in 0..N {
      let mut fn_avg = T::from(0.0);
      for d in 0..D {
        fn_avg = fn_avg + (fl[d][i] + fr[d][i]) * normal[d] * 0.5;
      }
      result[i] = fn_avg - (right[i] - left[i]) * s * 0.5;
    }
    result
  }
}

pub struct RusanovFlux;

impl<const D: usize, const N: usize> NumericalFlux<D, N> for RusanovFlux {
  fn compute<T: Scalar, L: ConservationLaw<D, N>>(
    &self,
    law: &L,
    left: &[T; N],
    right: &[T; N],
    normal: &Vector<f64, D>,
  ) -> [T; N] {
    let fl = law.flux(left);
    let fr = law.flux(right);
    let s_max = max_scalar(law.max_wave_speed(left), law.max_wave_speed(right));

    let mut result = [T::from(0.0); N];
    for i in 0..N {
      let mut fn_avg = T::from(0.0);
      for d in 0..D {
        fn_avg = fn_avg + (fl[d][i] + fr[d][i]) * normal[d] * 0.5;
      }
      result[i] = fn_avg - (right[i] - left[i]) * s_max * 0.5;
    }
    result
  }
}

#[derive(StateDiagnostics)]
#[diagnostics(
  components("rho", "rho_u", "rho_v", "energy"),
  conserved(
    ("mass", 0),
    ("momentum_x", 1),
    ("momentum_y", 2),
    ("total_energy", 3),
  ),
  extras(
    ("u", self.velocity(state)[0]),
    ("v", self.velocity(state)[1]),
    ("speed", self.speed(state)),
    ("kinetic_energy_density", self.kinetic_energy_density(state)),
    ("pressure", self.pressure(state)),
  ),
)]
pub struct Euler2D {
  gamma: f64, // Ratio of specific heats
}

impl Euler2D {
  pub fn new(gamma: f64) -> Euler2D {
    Euler2D { gamma }
  }

  pub fn velocity<T: Scalar>(&self, state: &[T; 4]) -> [T; 2] {
    let rho = state[0];
    [state[1] / rho, state[2] / rho]
  }

  pub fn speed<T: Scalar>(&self, state: &[T; 4]) -> T {
    let [u, v] = self.velocity(state);
    (u * u + v * v).sqrt()
  }

  pub fn kinetic_energy_density<T: Scalar>(&self, state: &[T; 4]) -> T {
    let rho = state[0];
    (state[1] * state[1] + state[2] * state[2]) / rho * 0.5
  }

  pub fn pressure<T: Scalar>(&self, state: &[T; 4]) -> T {
    (state[3] - self.kinetic_energy_density(state)) * (self.gamma - 1.0)
  }
}

impl ConservationLaw<2, 4> for Euler2D {
  fn flux<T: Scalar>(&self, state: &[T; 4]) -> [[T; 4]; 2] {
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

  fn max_wave_speed<T: Scalar>(&self, state: &[T; 4]) -> T {
    let rho = state[0];
    let u = state[1] / rho;
    let v = state[2] / rho;
    let p = self.pressure(state);
    let c = (p / rho * self.gamma).sqrt();
    // Regularise the speed `|u|` so the dual derivative is finite at u = 0
    // (sqrt' is singular there); the offset is negligible beside the sound
    // speed and its subgradient at the kink is the correct 0.
    (u * u + v * v + SPEED_EPS).sqrt() + c
  }

  fn source<T: Scalar>(
    &self,
    _state: &[T; 4],
    _cell: CellId,
    _centroid: &Point<2>,
    _metrics: &CellMetrics<2>,
  ) -> [T; 4] {
    [T::from(0.0); 4] // no source terms for basic Euler
  }

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
#[derive(StateDiagnostics)]
#[diagnostics(
  components("rho", "rho_u", "rho_v", "rho_w", "energy"),
  conserved(
    ("mass", 0),
    ("momentum_x", 1),
    ("momentum_y", 2),
    ("momentum_z", 3),
    ("total_energy", 4),
  ),
  extras(
    ("u", self.velocity(state)[0]),
    ("v", self.velocity(state)[1]),
    ("w", self.velocity(state)[2]),
    ("speed", self.speed(state)),
    ("kinetic_energy_density", self.kinetic_energy_density(state)),
    ("pressure", self.pressure(state)),
  ),
)]
pub struct Euler3D {
  gamma: f64,
  gravity: GravityField,
}

impl Euler3D {
  /// Construct without gravity (free flow).
  pub fn new(gamma: f64) -> Self {
    Self {
      gamma,
      gravity: GravityField::None,
    }
  }

  /// Construct with a constant gravity vector (force per unit mass) — right
  /// for flat boxes, e.g. `[0, 0, -9.81]`.
  pub fn with_gravity(gamma: f64, gravity: [f64; 3]) -> Self {
    Self {
      gamma,
      gravity: GravityField::Constant(gravity),
    }
  }

  /// Construct with a precomputed per-cell gravity field — right for radial
  /// gravity on a sphere shell. The vector is indexed by global cell ID and
  /// must have length equal to the mesh's cell count. Build it via
  /// `CubeSphere::radial_gravity_field(g)`.
  pub fn with_per_cell_gravity(gamma: f64, gravity: Vec<[f64; 3]>) -> Self {
    Self {
      gamma,
      gravity: GravityField::PerCell(gravity),
    }
  }

  pub fn velocity<T: Scalar>(&self, state: &[T; 5]) -> [T; 3] {
    let rho = state[0];
    [state[1] / rho, state[2] / rho, state[3] / rho]
  }

  pub fn speed<T: Scalar>(&self, state: &[T; 5]) -> T {
    let [u, v, w] = self.velocity(state);
    (u * u + v * v + w * w).sqrt()
  }

  pub fn kinetic_energy_density<T: Scalar>(&self, state: &[T; 5]) -> T {
    let rho = state[0];
    (state[1] * state[1] + state[2] * state[2] + state[3] * state[3]) / rho
      * 0.5
  }

  pub fn pressure<T: Scalar>(&self, state: &[T; 5]) -> T {
    (state[4] - self.kinetic_energy_density(state)) * (self.gamma - 1.0)
  }

  pub fn gamma(&self) -> f64 {
    self.gamma
  }
}

impl ConservationLaw<3, 5> for Euler3D {
  fn flux<T: Scalar>(&self, state: &[T; 5]) -> [[T; 5]; 3] {
    let rho = state[0];
    let u = state[1] / rho;
    let v = state[2] / rho;
    let w = state[3] / rho;
    let p = self.pressure(state);
    let h = state[4] + p;

    let fx = [
      state[1],
      state[1] * u + p,
      state[1] * v,
      state[1] * w,
      h * u,
    ];
    let fy = [
      state[2],
      state[2] * u,
      state[2] * v + p,
      state[2] * w,
      h * v,
    ];
    let fz = [
      state[3],
      state[3] * u,
      state[3] * v,
      state[3] * w + p,
      h * w,
    ];
    [fx, fy, fz]
  }

  fn max_wave_speed<T: Scalar>(&self, state: &[T; 5]) -> T {
    let rho = state[0];
    let u = state[1] / rho;
    let v = state[2] / rho;
    let w = state[3] / rho;
    let p = self.pressure(state);
    let c = (p / rho * self.gamma).sqrt();
    // Regularise `|u|` so the dual derivative is finite at u = 0 (sqrt' is
    // singular there); negligible beside the sound speed.
    (u * u + v * v + w * w + SPEED_EPS).sqrt() + c
  }

  fn source<T: Scalar>(
    &self,
    state: &[T; 5],
    cell: CellId,
    _: &Point<3>,
    _: &CellMetrics<3>,
  ) -> [T; 5] {
    let g = match &self.gravity {
      GravityField::None => return [T::from(0.0); 5],
      GravityField::Constant(g) => *g,
      GravityField::PerCell(field) => field[cell.index()],
    };
    let rho = state[0];
    // Gravity: force/volume = ρ·g on momentum, work/volume = (ρu)·g on energy.
    [
      T::from(0.0),
      rho * g[0],
      rho * g[1],
      rho * g[2],
      state[1] * g[0] + state[2] * g[1] + state[3] * g[2],
    ]
  }

  fn fix_state(&self, state: &mut [f64; 5]) {
    let floor = 1e-8;
    if state[0] < floor {
      state[0] = floor;
    }
    let rho = state[0];
    let ke =
      0.5 / rho * (state[1].powi(2) + state[2].powi(2) + state[3].powi(2));
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

/// Compressible Euler in 3D carrying one advected moisture tracer.
/// State = `[ρ, ρu, ρv, ρw, E, ρq]`, where `q` is the specific humidity
/// (water-vapour mass fraction).
///
/// In this first-proof formulation moisture is a *passive but
/// mass-conserved* scalar: the dynamics (pressure, density, wave speeds)
/// are identical to dry [`Euler3D`], and `ρq` is transported by the same
/// velocity field. Phase changes — evaporation adding `ρq`, condensation
/// releasing latent heat into `E` and precipitating it out — are applied
/// by separate physics stages (air–sea flux, microphysics), not by this
/// law's flux. The law simply guarantees water is advected conservatively.
///
/// Implemented by composition over [`Euler3D`] so the gravity / energy /
/// positivity logic is shared rather than duplicated.
pub struct MoistEuler3D {
  dry: Euler3D,
  /// Planetary rotation vector Ω (rad/s) in the world frame. Drives the
  /// Coriolis momentum source `−2·Ω×(ρu)` that organises rotating flow
  /// into geostrophic balance (cyclones, jets). Zero = non-rotating.
  omega: [f64; 3],
}

impl MoistEuler3D {
  /// Construct without gravity (free flow).
  pub fn new(gamma: f64) -> Self {
    Self {
      dry: Euler3D::new(gamma),
      omega: [0.0; 3],
    }
  }

  /// Construct with a constant gravity vector (force per unit mass).
  pub fn with_gravity(gamma: f64, gravity: [f64; 3]) -> Self {
    Self {
      dry: Euler3D::with_gravity(gamma, gravity),
      omega: [0.0; 3],
    }
  }

  /// Construct with a precomputed per-cell gravity field — right for radial
  /// gravity on a sphere shell. See [`Euler3D::with_per_cell_gravity`].
  pub fn with_per_cell_gravity(gamma: f64, gravity: Vec<[f64; 3]>) -> Self {
    Self {
      dry: Euler3D::with_per_cell_gravity(gamma, gravity),
      omega: [0.0; 3],
    }
  }

  /// Set the planetary rotation vector Ω (rad/s, world frame) used for the
  /// Coriolis source. Defaults to no rotation.
  pub fn with_rotation(mut self, omega: [f64; 3]) -> Self {
    self.omega = omega;
    self
  }

  /// Planetary rotation vector Ω (rad/s) in the world frame.
  pub fn rotation(&self) -> [f64; 3] {
    self.omega
  }

  /// Borrow the underlying dry Euler law (pressure / velocity helpers).
  pub fn dry(&self) -> &Euler3D {
    &self.dry
  }

  /// The dry 5-component sub-state `[ρ, ρu, ρv, ρw, E]`.
  pub fn dry_state<T: Scalar>(state: &[T; 6]) -> [T; 5] {
    [state[0], state[1], state[2], state[3], state[4]]
  }

  pub fn velocity<T: Scalar>(&self, state: &[T; 6]) -> [T; 3] {
    self.dry.velocity(&Self::dry_state(state))
  }

  pub fn pressure<T: Scalar>(&self, state: &[T; 6]) -> T {
    self.dry.pressure(&Self::dry_state(state))
  }

  pub fn gamma(&self) -> f64 {
    self.dry.gamma()
  }

  /// Specific humidity `q = ρq / ρ` (water-vapour mass fraction).
  pub fn specific_humidity<T: Scalar>(&self, state: &[T; 6]) -> T {
    state[5] / state[0]
  }
}

impl ConservationLaw<3, 6> for MoistEuler3D {
  fn flux<T: Scalar>(&self, state: &[T; 6]) -> [[T; 6]; 3] {
    let dry = Self::dry_state(state);
    let dry_flux = self.dry.flux(&dry);
    let rho = state[0];
    let rho_q = state[5];
    // Advective flux of ρq in each direction is ρq · u_d = ρq · (ρu_d / ρ).
    let mut out = [[T::from(0.0); 6]; 3];
    for d in 0..3 {
      for i in 0..5 {
        out[d][i] = dry_flux[d][i];
      }
      let u_d = state[1 + d] / rho;
      out[d][5] = rho_q * u_d;
    }
    out
  }

  fn max_wave_speed<T: Scalar>(&self, state: &[T; 6]) -> T {
    // Tracer advection speed |u| never exceeds the acoustic bound used by
    // the dry law, so the dry estimate also bounds the moist system.
    self.dry.max_wave_speed(&Self::dry_state(state))
  }

  fn source<T: Scalar>(
    &self,
    state: &[T; 6],
    cell: CellId,
    centroid: &Point<3>,
    metrics: &CellMetrics<3>,
  ) -> [T; 6] {
    let dry = self
      .dry
      .source(&Self::dry_state(state), cell, centroid, metrics);
    // Coriolis: −2·Ω×(ρu) on momentum. It does no work (⊥ velocity), so
    // there is no energy term, and it does not affect the moisture tracer.
    let m = [state[1], state[2], state[3]];
    let o = self.omega;
    let coriolis = [
      (m[2] * o[1] - m[1] * o[2]) * -2.0,
      (m[0] * o[2] - m[2] * o[0]) * -2.0,
      (m[1] * o[0] - m[0] * o[1]) * -2.0,
    ];
    [
      dry[0],
      dry[1] + coriolis[0],
      dry[2] + coriolis[1],
      dry[3] + coriolis[2],
      dry[4],
      T::from(0.0),
    ]
  }

  fn fix_state(&self, state: &mut [f64; 6]) {
    let mut dry = Self::dry_state(state);
    self.dry.fix_state(&mut dry);
    state[..5].copy_from_slice(&dry);
    // Moisture mass cannot go negative.
    if state[5] < 0.0 {
      state[5] = 0.0;
    }
  }

  /// Acoustic part of the flux for an IMEX split: the terms that carry the
  /// sound speed — the mass flux `ρu` (the velocity divergence in continuity),
  /// the pressure gradient on the momentum diagonal, and the pressure work
  /// `p·u` in energy. Momentum/energy *advection* and the moisture tracer stay
  /// in the explicit remainder.
  fn implicit_flux<T: Scalar>(&self, state: &[T; 6]) -> [[T; 6]; 3] {
    let rho = state[0];
    let p = self.pressure(state);
    let mut out = [[T::from(0.0); 6]; 3];
    for d in 0..3 {
      let u_d = state[1 + d] / rho;
      out[d][0] = state[1 + d]; // mass flux ρu_d
      out[d][1 + d] = p; // pressure on the momentum diagonal
      out[d][4] = p * u_d; // pressure work
    }
    out
  }

  /// Sound speed `c = √(γ·p/ρ)` — the dissipation speed for the implicit
  /// (acoustic) flux.
  fn acoustic_speed<T: Scalar>(&self, state: &[T; 6]) -> T {
    let rho = state[0];
    let p = self.pressure(state);
    (p / rho * self.gamma()).sqrt()
  }

  /// Explicit (advective) wave speed `|u|`: with the acoustics handled
  /// implicitly, the explicit CFL is set by advection, not sound.
  fn explicit_wave_speed<T: Scalar>(&self, state: &[T; 6]) -> T {
    let rho = state[0];
    let u = state[1] / rho;
    let v = state[2] / rho;
    let w = state[3] / rho;
    (u * u + v * v + w * w + SPEED_EPS).sqrt()
  }
}

impl LawFieldSchema<3, 6> for MoistEuler3D {
  fn conserved_field_names(&self) -> [&'static str; 6] {
    ["rho", "rho_u", "rho_v", "rho_w", "energy", "rho_q"]
  }

  fn derived_field_names(&self) -> &'static [&'static str] {
    &["u", "v", "w", "pressure", "humidity"]
  }

  fn write_derived_fields(
    &self,
    state: &[f64; 6],
    _centroid: &Point<3>,
    _metrics: &CellMetrics<3>,
    out: &mut [f64],
  ) {
    debug_assert_eq!(out.len(), 5);
    let rho = state[0];
    out[0] = state[1] / rho;
    out[1] = state[2] / rho;
    out[2] = state[3] / rho;
    out[3] = self.pressure(state);
    out[4] = self.specific_humidity(state);
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
