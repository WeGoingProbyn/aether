// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! The runtime facade that ties everything into a runnable thing. `WorldFactory`
//! assembles a `cosmo` seed into meshes, state, and stages; `World` / `System` /
//! `Aether` advance it in time (`tick`, `step`, and regime-aware `advance`). Holds
//! no physics of its own — it is the conductor, not an instrument.
//!
//! See `aether/docs/overview.md` for the runtime hierarchy and the advance modes.

pub mod adapt;
pub mod core;
pub mod factory;
