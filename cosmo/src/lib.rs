// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Immutable initial conditions: what bodies exist and what they are made of.
//! Pure configuration — no mutable state, geometry, or stages. A `cosmo` seed
//! is the fixed input a world is grown from at setup time.
//!
//! See `cosmo/docs/overview.md` for how a seed becomes a runnable world.

pub mod body;
pub mod factory;
pub mod kind;
pub mod system;
