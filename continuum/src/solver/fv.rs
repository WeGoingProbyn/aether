use log::{info, warn};

use crate::geometry::{Geometry, VecN};
use crate::grid::grid::Side;
use crate::topology::{BoundaryId, Neighbor, Topology};

pub type State<const NV: usize> = [f64; NV];
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

fn zero_state<const NV: usize>() -> State<NV> {
  [0.0; NV]
}

fn add_scaled_state<const NV: usize>(a: &mut State<NV>, b: &State<NV>, s: f64) {
  for m in 0..NV {
    a[m] += s * b[m];
  }
}

pub trait Model<const D: usize, const NV: usize> {
  fn flux(&self, u: &State<NV>) -> Flux<D, NV>;
  fn max_wave_speed(&self, u: &State<NV>, n_unit: VecN<D>) -> f64;

  fn boundary_state(
    &self,
    boundary: BoundaryId,
    interior: &State<NV>,
    x_face: VecN<D>,
    t: f64,
  ) -> State<NV>;

  fn source(&self, _u: &State<NV>, _x_cell: VecN<D>, _t: f64) -> State<NV> {
    zero_state::<NV>()
  }

  /// Optional: diffusive normal flux per unit area (component-wise F_diff · n).
  fn diffusive_flux_unit_normal(
    &self,
    _ul: &State<NV>,
    _ur: &State<NV>,
    _n_unit: VecN<D>,
    _dist_n: f64,
    _t: f64,
  ) -> State<NV> {
    zero_state::<NV>()
  }

  /// Optional: diffusion Lipschitz coefficient for preconditioning.
  ///
  /// Return `c` such that (approximately) F_diff · n ≈ -c * (U_R - U_L)
  /// componentwise. Default: 0.
  fn diffusion_lipschitz_coeff(
    &self,
    _ul: &State<NV>,
    _ur: &State<NV>,
    _n_unit: VecN<D>,
    _dist_n: f64,
    _t: f64,
  ) -> State<NV> {
    zero_state::<NV>()
  }
}

/// CFL controller settings for explicit (and hybrid) stepping.
#[derive(Debug, Clone)]
pub struct CflConfig {
  /// Advective CFL number (typical stable range: 0 < cfl <= 1).
  pub cfl: f64,
  /// Diffusive CFL number; use smaller values when diffusion dominates.
  pub diffusion_cfl: f64,
  /// Hard upper cap on any single step.
  pub max_dt: f64,
  /// Guard against pathological tiny steps per outer call.
  pub max_substeps: usize,
}

impl Default for CflConfig {
  fn default() -> Self {
    Self {
      cfl: 0.45,
      diffusion_cfl: 0.25,
      max_dt: 1e9,
      max_substeps: 32,
    }
  }
}

#[derive(Debug, Clone)]
pub struct CflEstimate {
  pub dt: f64,
  pub dt_adv: f64,
  pub dt_diff: f64,
  pub limiting_cell: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct CflStepStats {
  pub substeps: usize,
  pub advanced_dt: f64,
  pub last_estimate: CflEstimate,
}

/// Explicit / implicit / hybrid choice.
#[derive(Debug, Clone)]
pub enum TimeIntegrator {
  ExplicitEuler,
  ImplicitBackwardEuler(crate::solver::implicit::ImplicitConfig),
  Hybrid(crate::solver::implicit::ImplicitConfig, HybridConfig, HybridState),
}

/// Hybrid configuration (explicit warmup -> implicit when smooth).
#[derive(Debug, Clone)]
pub struct HybridConfig {
  pub warmup_steps: usize,
  pub enter_implicit: f64,
  pub exit_implicit: f64,
  pub consecutive_smooth_steps: usize,
  pub cooldown_steps_after_fail: usize,
  pub max_consecutive_implicit_failures: usize,
}

impl Default for HybridConfig {
  fn default() -> Self {
    Self {
      warmup_steps: 10,
      enter_implicit: 0.02,
      exit_implicit: 0.05,
      consecutive_smooth_steps: 3,
      cooldown_steps_after_fail: 5,
      max_consecutive_implicit_failures: 3,
    }
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridPhase {
  Explicit,
  Implicit,
}

/// Hybrid internal state.
#[derive(Debug, Clone)]
pub struct HybridState {
  pub phase: HybridPhase,
  pub step_count: usize,
  pub smooth_streak: usize,
  pub cooldown_left: usize,
  pub implicit_fail_streak: usize,
  pub last_smoothness: f64,
}

impl Default for HybridState {
  fn default() -> Self {
    Self {
      phase: HybridPhase::Explicit,
      step_count: 0,
      smooth_streak: 0,
      cooldown_left: 0,
      implicit_fail_streak: 0,
      last_smoothness: f64::INFINITY,
    }
  }
}

/// Main solver object: owns topology, geometry, model, state, integrator choice.
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

    pub u: Vec<State<NV>>,
    pub time: f64,

    pub integrator: TimeIntegrator,
  }

impl<Topo, Geom, M, const D: usize, const NV: usize> FiniteVolumeSolver<Topo, Geom, M, D, NV>
where
  Topo: Topology<D>,
  Geom: Geometry<D>,
  M: Model<D, NV>,
{
  pub fn new(topo: Topo, geom: Geom, model: M, integrator: TimeIntegrator) -> Self {
    let n = topo.grid().cell_count();
    Self {
      topo,
      geom,
      model,
      u: vec![[0.0; NV]; n],
      time: 0.0,
      integrator,
    }
  }

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

  /// Public step entry point.
  ///
  /// Important: we `mem::replace` integrator to avoid borrow conflicts between
  /// `self` and `self.integrator` for the Hybrid state machine.
  pub fn step(&mut self, dt: f64) {
    let integrator = std::mem::replace(&mut self.integrator, TimeIntegrator::ExplicitEuler);

    self.integrator = match integrator {
      TimeIntegrator::ExplicitEuler => {
        crate::solver::explicit::step_explicit(self, dt);
        TimeIntegrator::ExplicitEuler
      }

      TimeIntegrator::ImplicitBackwardEuler(cfg) => {
        // If implicit fails, fall back to explicit (safe default).
        if crate::solver::implicit::step_implicit_backward_euler(self, dt, cfg.clone()).is_err() {
          crate::solver::explicit::step_explicit(self, dt);
        }
        TimeIntegrator::ImplicitBackwardEuler(cfg)
      }

      TimeIntegrator::Hybrid(implicit_cfg, cfg, mut st) => {
        self.step_hybrid(dt, &implicit_cfg, &cfg, &mut st);
        TimeIntegrator::Hybrid(implicit_cfg, cfg, st)
      }
    };
  }

  fn compute_smoothness(&self) -> f64 {
    // Mean normalized jump across interior "+" faces:
    // mean(|U_i - U_j|) / (mean(|U|) + eps)
    let grid = self.topo.grid();
    let mut jump_sum = 0.0;
    let mut mag_sum = 0.0;
    let mut count = 0usize;

    for cell in grid.iter_cells() {
      let i = grid.cell_linear(cell);
      let ui = self.u[i];

      for m in 0..NV {
        mag_sum += ui[m].abs();
      }

      for face in self.topo.faces_of_cell(cell) {
        if face.side != Side::Plus {
          continue; // avoid double counting
        }
        match self.topo.neighbor_across(face) {
          Neighbor::Cell(nc) => {
            let j = grid.cell_linear(nc);
            let uj = self.u[j];
            for m in 0..NV {
              jump_sum += (ui[m] - uj[m]).abs();
            }
            count += NV;
          }
          Neighbor::Boundary(_) => {}
        }
      }
    }

    let mean_mag = (mag_sum / (grid.cell_count() as f64 * NV as f64)).max(1e-12);
    let mean_jump = if count == 0 { 0.0 } else { jump_sum / (count as f64) };
    mean_jump / mean_mag
  }

  /// Estimate a stable timestep based on Rusanov advection and model-provided diffusion Lipschitz.
  pub fn estimate_cfl_dt(&self, cfg: &CflConfig) -> CflEstimate {
    let grid = self.topo.grid();
    let mut dt_best = f64::INFINITY;
    let mut dt_adv_best = f64::INFINITY;
    let mut dt_diff_best = f64::INFINITY;
    let mut limiting_cell = None;

    for cell in grid.iter_cells() {
      let i = grid.cell_linear(cell);
      let ui = self.u[i];

      let mut adv_sum = [0.0; NV];
      let mut diff_sum = [0.0; NV];

      for face in self.topo.faces_of_cell(cell) {
        let s = self.geom.face_area_vector(face);
        let area = norm::<D>(s);
        if area == 0.0 {
          continue;
        }

        let mut n_unit = [0.0; D];
        for d in 0..D {
          n_unit[d] = s[d] / area;
        }

        let neighbor = self.topo.neighbor_across(face);
        let ur = match neighbor {
          Neighbor::Cell(nc) => {
            let j = grid.cell_linear(nc);
            self.u[j]
          }
          Neighbor::Boundary(bid) => {
            let xf = self.geom.face_center_x(face);
            self.model.boundary_state(bid, &ui, xf, self.time)
          }
        };

        // Advective wave speed (Rusanov dissipation uses 0.5 * a)
        let a = self
          .model
          .max_wave_speed(&ui, n_unit)
          .max(self.model.max_wave_speed(&ur, n_unit));

        // Diffusion Lipschitz ~ c where F_diff · n ≈ -c (u_r - u_l)
        let x_l = self.geom.cell_center_x(face.cell);
        let x_r = match neighbor {
          Neighbor::Cell(nc) => self.geom.cell_center_x(nc),
          Neighbor::Boundary(_) => {
            let xf = self.geom.face_center_x(face);
            let mut xr = [0.0; D];
            for d in 0..D {
              xr[d] = 2.0 * xf[d] - x_l[d];
            }
            xr
          }
        };
        let mut dx = [0.0; D];
        for d in 0..D {
          dx[d] = x_r[d] - x_l[d];
        }
        let dist_n = dot::<D>(dx, n_unit).abs().max(1e-12);
        let c = self
          .model
          .diffusion_lipschitz_coeff(&ui, &ur, n_unit, dist_n, self.time);

        for m in 0..NV {
          adv_sum[m] += area * 0.5 * a;
          diff_sum[m] += area * c[m].max(0.0);
        }
      }

      let vol = self.geom.cell_volume(cell);
      for m in 0..NV {
        let dt_adv = if adv_sum[m] > 0.0 {
          cfg.cfl * vol / adv_sum[m]
        } else {
          f64::INFINITY
        };
        let dt_diff = if diff_sum[m] > 0.0 {
          cfg.diffusion_cfl * vol / diff_sum[m]
        } else {
          f64::INFINITY
        };
        let dt_cell = dt_adv.min(dt_diff);

        dt_adv_best = dt_adv_best.min(dt_adv);
        dt_diff_best = dt_diff_best.min(dt_diff);

        if dt_cell < dt_best {
          dt_best = dt_cell;
          limiting_cell = Some(i);
        }
      }
    }

    let dt_raw = if dt_best.is_finite() {
      dt_best
    } else {
      cfg.max_dt
    };

    let dt = dt_raw.min(cfg.max_dt);
    let dt_adv = dt_adv_best.min(cfg.max_dt);
    let dt_diff = dt_diff_best.min(cfg.max_dt);

    CflEstimate {
      dt,
      dt_adv,
      dt_diff,
      limiting_cell,
    }
  }

  /// Advance up to `target_dt` using CFL-limited substeps. Returns stats about the last estimate.
  pub fn step_cfl(&mut self, target_dt: f64, cfg: &CflConfig) -> CflStepStats {
    let mut remaining = target_dt.max(0.0);
    let mut substeps = 0usize;
    let mut last_est = self.estimate_cfl_dt(cfg);

    while remaining > 0.0 && substeps < cfg.max_substeps {
      last_est = self.estimate_cfl_dt(cfg);
      let dt_step = last_est.dt.min(remaining).min(cfg.max_dt);

      if !dt_step.is_finite() || dt_step <= 0.0 {
        warn!("CFL estimate produced non-positive dt; aborting step");
        break;
      }

      self.step(dt_step);
      remaining -= dt_step;
      substeps += 1;
    }

    if remaining > 0.0 {
      warn!(
        "CFL step capped at {} substeps; {} time left unadvanced",
        cfg.max_substeps,
        remaining
      );
    }

    let stats = CflStepStats {
      substeps,
      advanced_dt: target_dt - remaining,
      last_estimate: last_est,
    };

    info!(
      "CFL step: advanced_dt={:.3e}, substeps={}, dt_est={:.3e} (adv={:.3e}, diff={:.3e}), limiter_cell={:?}",
      stats.advanced_dt,
      stats.substeps,
      stats.last_estimate.dt,
      stats.last_estimate.dt_adv,
      stats.last_estimate.dt_diff,
      stats.last_estimate.limiting_cell,
    );

    stats
  }

  fn step_hybrid(
    &mut self,
    dt: f64,
    implicit_cfg: &crate::solver::implicit::ImplicitConfig,
    cfg: &HybridConfig,
    st: &mut HybridState,
  ) {
    st.step_count += 1;

    // Warmup: force explicit
    if st.step_count <= cfg.warmup_steps {
      if st.step_count == 1 { 
        info!("Running {} explicit steps on startup", cfg.warmup_steps);
      }
      st.phase = HybridPhase::Explicit;
      crate::solver::explicit::step_explicit(self, dt);
      st.last_smoothness = self.compute_smoothness();
      return;
    }

    // Cooldown after implicit failure
    if st.cooldown_left > 0 {
      if st.cooldown_left == cfg.cooldown_steps_after_fail {
        info!("Implicit step failure, falling back to explicit!");
      }
      st.cooldown_left -= 1;
      st.phase = HybridPhase::Explicit;
      crate::solver::explicit::step_explicit(self, dt);
      st.last_smoothness = self.compute_smoothness();
      return;
    }

    // Measure smoothness
    st.last_smoothness = self.compute_smoothness();

    match st.phase {
      HybridPhase::Explicit => {
        if st.last_smoothness < cfg.enter_implicit {
          st.smooth_streak += 1;
        } else {
          st.smooth_streak = 0;
        }

        let can_try_implicit = st.smooth_streak >= cfg.consecutive_smooth_steps
        && st.implicit_fail_streak < cfg.max_consecutive_implicit_failures;

        if can_try_implicit {
          // Try implicit; fallback on failure.
          // info!("Trying implict solver for step {}", st.step_count);
          if self.try_implicit_or_fallback(dt, implicit_cfg, cfg, st) {
            st.phase = HybridPhase::Implicit;
            return;
          }
        }

        // info!("Running explicit solver for step {}", st.step_count);
        crate::solver::explicit::step_explicit(self, dt);
      }

      HybridPhase::Implicit => {
        // Hysteresis: if it becomes rough again, go back to explicit
        if st.last_smoothness > cfg.exit_implicit {
          st.phase = HybridPhase::Explicit;
          st.smooth_streak = 0;
          crate::solver::explicit::step_explicit(self, dt);
          return;
        }

        // Stay implicit; fallback if needed.
        let _ = self.try_implicit_or_fallback(dt, implicit_cfg, cfg, st);
      }
    }
  }

  /// Returns true if implicit succeeded, false if it fell back to explicit.
  fn try_implicit_or_fallback(
    &mut self,
    dt: f64,
    implicit_cfg: &crate::solver::implicit::ImplicitConfig,
    cfg: &HybridConfig,
    st: &mut HybridState,
  ) -> bool {
    match crate::solver::implicit::step_implicit_backward_euler(self, dt, implicit_cfg.clone()) {
      Ok(()) => {
        info!("Implicit step success for step {}", st.step_count);
        st.implicit_fail_streak = 0;
        true
      }
      Err(_e) => {
        info!(
          "Implicit step failed for step {}, falling back to explicit for {} steps", 
          st.step_count,
          cfg.cooldown_steps_after_fail,
        );
        st.implicit_fail_streak += 1;
        st.cooldown_left = cfg.cooldown_steps_after_fail;
        st.phase = HybridPhase::Explicit;
        st.smooth_streak = 0;

        crate::solver::explicit::step_explicit(self, dt);
        false
      }
    }
  }

  /// Compute L(U) into `out` such that dU/dt = L(U).
  /// Shared FV operator (used by both explicit and implicit).
  pub fn compute_l(&self, u_in: &[State<NV>], t: f64, out: &mut [State<NV>]) {
    let grid = self.topo.grid();
    debug_assert_eq!(u_in.len(), grid.cell_count());
    debug_assert_eq!(out.len(), grid.cell_count());

    for i in 0..out.len() {
      out[i] = [0.0; NV];
    }

    for cell in grid.iter_cells() {
      let i = grid.cell_linear(cell);
      let ui = u_in[i];

      // source term
      let x_cell = self.geom.cell_center_x(cell);
      let src = self.model.source(&ui, x_cell, t);
      add_scaled_state::<NV>(&mut out[i], &src, 1.0);

      for face in self.topo.faces_of_cell(cell) {
        let s = self.geom.face_area_vector(face);
        let area = norm::<D>(s);
        if area == 0.0 {
          continue;
        }

        let mut n_unit = [0.0; D];
        for d in 0..D {
          n_unit[d] = s[d] / area;
        }

        let ul = ui;

        let neighbor = self.topo.neighbor_across(face);
        let ur = match neighbor {
          Neighbor::Cell(nc) => {
            let j = grid.cell_linear(nc);
            u_in[j]
          }
          Neighbor::Boundary(bid) => {
            let xf = self.geom.face_center_x(face);
            self.model.boundary_state(bid, &ul, xf, t)
          }
        };

        // Rusanov advective flux through unit normal
        let fl = self.model.flux(&ul);
        let fr = self.model.flux(&ur);

        let mut fn_l = [0.0; NV];
        let mut fn_r = [0.0; NV];
        for m in 0..NV {
          fn_l[m] = dot::<D>(fl[m], n_unit);
          fn_r[m] = dot::<D>(fr[m], n_unit);
        }

        let a = self
          .model
          .max_wave_speed(&ul, n_unit)
          .max(self.model.max_wave_speed(&ur, n_unit));

        let mut fhat_n = [0.0; NV];
        for m in 0..NV {
          fhat_n[m] = 0.5 * (fn_l[m] + fn_r[m]) - 0.5 * a * (ur[m] - ul[m]);
        }

        // Diffusion normal flux (optional)
        let x_l = self.geom.cell_center_x(face.cell);
        let x_r = match neighbor {
          Neighbor::Cell(nc) => self.geom.cell_center_x(nc),
          Neighbor::Boundary(_) => {
            let xf = self.geom.face_center_x(face);
            let mut xr = [0.0; D];
            for d in 0..D {
              xr[d] = 2.0 * xf[d] - x_l[d];
            }
            xr
          }
        };
        let mut dx = [0.0; D];
        for d in 0..D {
          dx[d] = x_r[d] - x_l[d];
        }
        let dist_n = dot::<D>(dx, n_unit).abs().max(1e-12);

        let f_diff_n = self.model.diffusive_flux_unit_normal(&ul, &ur, n_unit, dist_n, t);
        for m in 0..NV {
          fhat_n[m] += f_diff_n[m];
        }

        // Integrated flux = fhat_n * area
        for m in 0..NV {
          out[i][m] -= fhat_n[m] * area;
        }
      }

      // divide by volume
      let vol = self.geom.cell_volume(cell);
      let inv_vol = 1.0 / vol;
      for m in 0..NV {
        out[i][m] *= inv_vol;
      }
    }
  }
}
