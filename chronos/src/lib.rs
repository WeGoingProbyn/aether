// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Chronos owns the *timescale spectrum*: turning live, CFL-bound physics into
//! slowly-varying climatology a days–centuries consumer can read, and the
//! regime machinery that lets long game-time advance without integrating the
//! Euler solver for a millennium.
//!
//! Like the physics crates it stores nothing globally. It builds on the
//! multirate scheduler already in `nexus` (`SubsystemId` + subsystem cadences +
//! the operator-split `multirate_tick`): a climatology aggregate is just an
//! ordinary pleroma field updated by an ordinary nexus stage placed on a slow
//! subsystem. Aggregates then flow to consumers through the existing
//! `eidolon::query` snapshot path unchanged.
//!
//! Layers:
//! - [`ClimatologyModel`] / [`ClimatologyAccumulatorStep`] — inert time-mean
//!   aggregation (an exponential moving average toward the live value).
//! - [`convergence`] — instrumentation to *measure* how fast aggregates settle,
//!   so the regime burst length is chosen from data, not guessed.
//!
//! Out of scope (deferred with AMR per the roadmap): fidelity-LOD /
//! multi-resolution region wake.
//!
//! See `chronos/docs/overview.md` and `docs/physics.md` for the full narrative.

pub mod accumulator;
pub mod convergence;
pub mod error;
pub mod model;
pub mod nudge;
pub mod regime;

pub use accumulator::ClimatologyAccumulatorStep;
pub use convergence::{
  ConvergenceSample, residual, settling_time, suggest_burst_steps,
};
pub use error::ChronosError;
pub use model::{ClimateQuantity, ClimatologyModel, DEFAULT_TIMESCALE};
pub use nudge::{ClimatologyNudgeStep, copy_field};
pub use regime::{Regime, RegimeConfig, TransitionKind, TransitionState};
