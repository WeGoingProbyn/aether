// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Read-only extraction from simulation state into Eidolon IR.

pub mod coupler_debug;
pub mod diagnostics;
pub mod frame;
pub mod layer;
pub mod mesh;

pub use coupler_debug::*;
pub use frame::*;
pub use layer::*;
pub use mesh::*;
