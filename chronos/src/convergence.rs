// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Convergence instrumentation for climatology aggregates.
//!
//! The burst length used by the climatology regime (how long to run the live
//! solver before holding it and exposing aggregates) is otherwise an
//! unprincipled free parameter: too short and the climatology averages noise
//! without developing realistic spatial structure; too long and entering
//! climatology mode lags before the aggregates settle — and the right value
//! differs per subsystem. So we *measure* convergence rather than guess.
//!
//! These are pure analysis helpers, deliberately decoupled from the hot
//! [`crate::accumulator::ClimatologyAccumulatorStep`]: a caller (a tuning
//! harness, a test, or the regime driver) snapshots a mean field before and
//! after a tick, calls [`residual`] to get one [`ConvergenceSample`], collects
//! a trace over successive ticks, then derives a settling time / suggested
//! burst length from it. Nothing here feeds physics.

/// One measurement of how much a climatology aggregate moved over a tick.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConvergenceSample {
  /// `‖meanₙ − meanₙ₋₁‖₂` over cells — absolute residual change this tick.
  pub residual_l2: f64,
  /// `residual_l2 / ‖meanₙ‖₂` — scale-free residual; `0` when the mean is all
  /// zero. This is the quantity to threshold for "has it settled".
  pub residual_rel: f64,
  /// Accumulated simulation time at the end of this tick (s).
  pub elapsed: f64,
  /// Number of accumulator steps taken up to and including this tick.
  pub steps: usize,
}

/// Compute the residual change between the previous and current mean fields.
///
/// `prev` and `curr` must be the same length (the per-cell aggregate before and
/// after one accumulator step). `elapsed` / `steps` are carried through so a
/// collected trace can be read as a function of sim time.
pub fn residual(
  prev: &[f64],
  curr: &[f64],
  elapsed: f64,
  steps: usize,
) -> ConvergenceSample {
  let n = prev.len().min(curr.len());
  let mut diff_sq = 0.0;
  let mut curr_sq = 0.0;
  for i in 0..n {
    let d = curr[i] - prev[i];
    diff_sq += d * d;
    curr_sq += curr[i] * curr[i];
  }
  let residual_l2 = diff_sq.sqrt();
  let residual_rel = if curr_sq > 0.0 {
    residual_l2 / curr_sq.sqrt()
  } else {
    0.0
  };
  ConvergenceSample {
    residual_l2,
    residual_rel,
    elapsed,
    steps,
  }
}

/// The first elapsed sim-time at which the relative residual has dropped below
/// `eps_rel` *and stays below it* for the remainder of the trace — i.e. the
/// aggregate has settled. `None` if it never settles within the trace.
///
/// Requiring it to stay below the threshold (not just touch it once) guards
/// against a transient dip while the aggregate is still developing structure.
pub fn settling_time(trace: &[ConvergenceSample], eps_rel: f64) -> Option<f64> {
  let mut candidate: Option<f64> = None;
  for sample in trace {
    if sample.residual_rel < eps_rel {
      if candidate.is_none() {
        candidate = Some(sample.elapsed);
      }
    } else {
      candidate = None;
    }
  }
  candidate
}

/// Suggest a burst length (number of steps at `dt`) that covers the measured
/// settling time, so the climatology regime refreshes its aggregates long
/// enough for them to develop before holding. Falls back to the full trace
/// length when the aggregate never settled within the trace (be conservative
/// rather than under-develop the climatology).
pub fn suggest_burst_steps(
  trace: &[ConvergenceSample],
  eps_rel: f64,
  dt: f64,
) -> usize {
  match settling_time(trace, eps_rel) {
    Some(t) if dt.is_finite() && dt > 0.0 => (t / dt).ceil() as usize,
    _ => trace.len(),
  }
  .max(1)
}
