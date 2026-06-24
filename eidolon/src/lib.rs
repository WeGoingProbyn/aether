// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Engine-neutral presentation IR for simulation visualization.
//!
//! Eidolon observes simulation state and emits owned render/diagnostic data.
//! Backends such as Bevy should consume this crate; simulation crates should
//! not depend on concrete engine types.

pub mod backend;
pub mod bevy;
pub mod export;
pub mod extract;
pub mod ir;
pub mod playback;
pub mod query;
pub mod registry;
pub mod render_ops;
pub mod runtime;
