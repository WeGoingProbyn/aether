// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Nexus is the DAG scheduler. Physics crates depend on `nexus` alone for
//! their state-access vocabulary; the relevant pleroma items are re-exported
//! here so a typical `aer::Cargo.toml` lists `nexus + tessera + continuum +
//! utility` and never `pleroma` directly.

pub mod constants;
pub mod schedule;
pub mod stage;

pub use pleroma::Pleroma;
pub use pleroma::prelude::*;
pub use utility::domain::WorldId;

pub use constants::*;
pub use schedule::*;
pub use stage::*;
