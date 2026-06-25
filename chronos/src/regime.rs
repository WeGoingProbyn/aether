// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Timescale regimes: how long game-time is advanced relative to the live
//! solver.
//!
//! A CFL-bound Euler solver cannot be integrated for a simulated millennium, so
//! two regimes coexist behind one interface:
//!
//! - [`Regime::Live`] — the full solver runs at game time; a fast consumer
//!   reads instantaneous fields. Advancing by `game_dt` integrates the world by
//!   `game_dt`.
//! - [`Regime::Climatology`] — the solver runs only in short *bursts* to keep
//!   the climatology aggregates current, then long game-time advances by
//!   *holding* the Euler state. Advancing by a large `game_dt` costs only the
//!   burst, not a full integration of `game_dt`.
//!
//! The climatology regime deliberately does **not** claim instantaneous Euler
//! fidelity over the held span — it claims the *climatology* is current. The
//! game clock and the integrated sim time diverge by design (see the driver in
//! `aether`); that divergence is the cost saving.

/// Which time-advance regime a world is in.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Regime {
  /// Full solver at game time; reads see instantaneous fields.
  Live,
  /// Burst-then-hold: refresh the climatology in short bursts, then advance
  /// game-time by holding the live state.
  Climatology,
}

impl Default for Regime {
  fn default() -> Self {
    Regime::Live
  }
}

/// Parameters for the climatology regime's burst-then-hold advance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegimeConfig {
  /// Number of live solver steps run at the start of a climatology advance to
  /// refresh the aggregates before holding. Choose this from a measured
  /// convergence trace (see [`crate::convergence::suggest_burst_steps`]) rather
  /// than guessing: too few and the climatology averages noise without
  /// developing structure; too many and entering the regime lags.
  pub burst_steps: usize,
  /// The dt of each burst step (s).
  pub burst_dt: f64,
}

impl RegimeConfig {
  pub fn new(burst_steps: usize, burst_dt: f64) -> Self {
    Self {
      burst_steps: burst_steps.max(1),
      burst_dt,
    }
  }

  /// Total simulation time integrated by one climatology advance's burst.
  pub fn burst_span(&self) -> f64 {
    self.burst_steps as f64 * self.burst_dt
  }
}

impl Default for RegimeConfig {
  fn default() -> Self {
    Self {
      burst_steps: 1,
      burst_dt: 60.0,
    }
  }
}

/// Direction of a live↔climatology handoff.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum TransitionKind {
  /// Zoom in / wake: switch a region from reading the climatology aggregate to
  /// the live solver. The live state is seeded from the climatology and then
  /// relaxed toward free evolution over the window (a runtime spin-up).
  ClimatologyToLive,
  /// Zoom out / sleep: switch from live back to the climatology aggregate. The
  /// climatology is seeded from the current live state at the instant of switch
  /// so the aggregate is continuous with the last live value shown.
  LiveToClimatology,
}

/// A live↔climatology handoff in progress. `progress` runs 0→1 over the
/// transition; the nudge relaxation fraction ramps from `1` (hold the live
/// state on the climatology at the switch) to `0` (free evolution) as it
/// completes, so the consumer sees no discontinuity across the handoff.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransitionState {
  pub kind: TransitionKind,
  /// Fraction of the window elapsed, in `[0, 1]`.
  pub progress: f64,
  /// Length of the transition window in game seconds.
  pub window: f64,
}

impl TransitionState {
  pub fn new(kind: TransitionKind, window: f64) -> Self {
    Self {
      kind,
      progress: 0.0,
      window,
    }
  }

  /// The relaxation fraction applied this tick: `1 − progress`, so the live
  /// state is held on the climatology at the start of the window and released
  /// to free evolution by the end. Clamped to `[0, 1]`.
  pub fn nudge_fraction(&self) -> f64 {
    (1.0 - self.progress).clamp(0.0, 1.0)
  }

  /// Advance the window by `game_dt`, clamping `progress` to 1. Returns `true`
  /// once the transition has completed (progress reached 1).
  pub fn advance(&mut self, game_dt: f64) -> bool {
    if self.window > 0.0 {
      self.progress = (self.progress + game_dt / self.window).min(1.0);
    } else {
      self.progress = 1.0;
    }
    self.is_complete()
  }

  pub fn is_complete(&self) -> bool {
    self.progress >= 1.0
  }
}
