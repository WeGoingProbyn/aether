// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Hybrid implicit/explicit backend.
//!
//! Implicit stepping is unconditionally *stable* but its inner solves can still
//! fail — GMRES may not converge, or a converged-but-poorly-linearized step can
//! produce a non-physical or wildly-moving state. Rather than let that stall or
//! corrupt the simulation, the hybrid backend *attempts* an implicit step, then
//! **validates** it; if the step is rejected it restores the pre-step state and
//! advances the same interval with a burst of small, CFL-limited explicit
//! sub-steps. The simulation always makes progress, taking the big implicit
//! step when it can and degrading gracefully when it can't.
//!
//! Validation (all cheap):
//! 1. every stage's GMRES converged;
//! 2. the new state is finite and density stays positive;
//! 3. the relative motion `‖ΔU‖/‖U‖` is within a bound — a converged, positive
//!    step can still be physically wrong if the linearization was too far off,
//!    and this catches that for the cost of one norm.

use pleroma::core::storage::FieldStorage;
use tessera::mesh::Mesh;
use utility::domain::CellId;

use crate::{
  boundary::BoundaryRegistry,
  cpu::CpuBackend,
  implicit::backend::ImplicitBackend,
  kernel,
  model::{ConservationLaw, NumericalFlux},
  solver::{FvmBackend, SolverConfig},
};

/// Running tally of how the hybrid backend has been stepping — surfaced for
/// diagnostics (how often the implicit path is actually winning).
#[derive(Clone, Copy, Debug, Default)]
pub struct HybridReport {
  /// Implicit steps that passed validation and were accepted.
  pub implicit_accepted: usize,
  /// Steps where the implicit attempt was rejected and fell back to explicit.
  pub fallbacks: usize,
  /// Total explicit sub-steps taken across all fallbacks.
  pub explicit_substeps: usize,
}

pub struct HybridBackend<const N: usize> {
  implicit: ImplicitBackend<N>,
  explicit: CpuBackend<N>,
  /// Reject an implicit step whose `‖ΔU‖/‖U‖` exceeds this.
  max_relative_step: f64,
  /// Safety cap on explicit sub-steps per fallback (avoids pathological loops).
  max_substeps: usize,
  snapshot: Vec<[f64; N]>,
  cache: Vec<[f64; N]>,
  report: HybridReport,
}

impl<const N: usize> HybridBackend<N> {
  pub fn new(implicit: ImplicitBackend<N>) -> Self {
    Self {
      implicit,
      explicit: CpuBackend::default(),
      max_relative_step: 0.5,
      max_substeps: 100_000,
      snapshot: Vec::new(),
      cache: Vec::new(),
      report: HybridReport::default(),
    }
  }

  /// Set the relative-motion rejection threshold (`‖ΔU‖/‖U‖`).
  pub fn with_max_relative_step(mut self, threshold: f64) -> Self {
    self.max_relative_step = threshold;
    self
  }

  pub fn report(&self) -> HybridReport {
    self.report
  }

  /// Validate the freshly-stepped `cache` against the pre-step `snapshot`.
  fn validate(&self, norm_u: f64) -> bool {
    let mut diff_sq = 0.0;
    for (new, old) in self.cache.iter().zip(&self.snapshot) {
      if new[0] <= 0.0 {
        return false; // density must stay positive
      }
      for c in 0..N {
        if !new[c].is_finite() {
          return false;
        }
        let d = new[c] - old[c];
        diff_sq += d * d;
      }
    }
    let rel = diff_sq.sqrt() / norm_u.max(1e-30);
    rel < self.max_relative_step
  }

  fn explicit_fallback<const D: usize, L, F, S, M>(
    &mut self,
    config: &SolverConfig,
    law: &L,
    flux: &F,
    dt: f64,
    state: &mut S,
    residual: &mut S,
    mesh: &M,
    bcs: &BoundaryRegistry<D, N>,
  ) where
    L: ConservationLaw<D, N>,
    F: NumericalFlux<D, N>,
    S: FieldStorage<N>,
    M: Mesh<D> + ?Sized,
  {
    // Restore the pre-step state, then cover [0, dt] with CFL-limited steps.
    for (i, s) in self.snapshot.iter().enumerate() {
      state.write(CellId::from(i), s);
    }
    self.report.fallbacks += 1;

    let mut elapsed = 0.0;
    let eps = 1e-12 * dt.abs().max(1.0);
    let mut taken = 0;
    while elapsed < dt - eps && taken < self.max_substeps {
      kernel::gather_state_cache(state, mesh, &mut self.cache);
      let cfl = kernel::compute_dt_from_cache(config, law, &self.cache, mesh);
      let step = if cfl.is_finite() && cfl > 0.0 {
        (dt - elapsed).min(cfl)
      } else {
        dt - elapsed
      };
      self
        .explicit
        .step_with_dt(config, law, flux, step, state, residual, mesh, bcs);
      elapsed += step;
      taken += 1;
    }
    self.report.explicit_substeps += taken;
  }

  fn attempt<const D: usize, L, F, S, M>(
    &mut self,
    config: &SolverConfig,
    law: &L,
    flux: &F,
    dt: f64,
    state: &mut S,
    residual: &mut S,
    mesh: &M,
    bcs: &BoundaryRegistry<D, N>,
  ) where
    L: ConservationLaw<D, N>,
    F: NumericalFlux<D, N>,
    S: FieldStorage<N>,
    M: Mesh<D> + ?Sized,
  {
    kernel::gather_state_cache(state, mesh, &mut self.snapshot);
    let mut norm_sq = 0.0;
    for cell in &self.snapshot {
      for &c in cell {
        norm_sq += c * c;
      }
    }
    let norm_u = norm_sq.sqrt();

    self
      .implicit
      .step_with_dt(config, law, flux, dt, state, residual, mesh, bcs);

    kernel::gather_state_cache(state, mesh, &mut self.cache);
    let converged = self
      .implicit
      .last_report()
      .map(|r| r.converged)
      .unwrap_or(false);

    if converged && self.validate(norm_u) {
      self.report.implicit_accepted += 1;
    } else {
      self.explicit_fallback(config, law, flux, dt, state, residual, mesh, bcs);
    }
  }
}

impl<const D: usize, const N: usize, L, F> FvmBackend<D, N, L, F>
  for HybridBackend<N>
where
  L: ConservationLaw<D, N>,
  F: NumericalFlux<D, N>,
{
  fn step<S, M>(
    &mut self,
    config: &SolverConfig,
    law: &L,
    flux: &F,
    state: &mut S,
    residual: &mut S,
    mesh: &M,
    bcs: &BoundaryRegistry<D, N>,
  ) -> f64
  where
    S: FieldStorage<N>,
    M: Mesh<D> + ?Sized,
  {
    // No target dt given: take the explicit CFL step, where implicit and
    // explicit agree and validation always passes.
    kernel::gather_state_cache(state, mesh, &mut self.cache);
    let dt = kernel::compute_dt_from_cache(config, law, &self.cache, mesh);
    self.attempt(config, law, flux, dt, state, residual, mesh, bcs);
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
    bcs: &BoundaryRegistry<D, N>,
  ) where
    S: FieldStorage<N>,
    M: Mesh<D> + ?Sized,
  {
    self.attempt(config, law, flux, dt, state, residual, mesh, bcs);
  }
}
