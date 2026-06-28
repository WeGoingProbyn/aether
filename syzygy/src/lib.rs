// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Syzygy owns coupling semantics between physics modules. It consumes
//! read-only mesh/coupler geometry from Tessera and field access from Nexus.
//!
//! See `syzygy/docs/overview.md` for the stencil/flux/relaxation vocabulary.

pub mod error;
pub mod flux;
pub mod scalar;
pub mod stencil;

pub use error::SyzygyError;
pub use flux::{ScalarInterfaceDeposition, ScalarInterfaceFlux};
pub use scalar::ScalarRelaxation;
pub use stencil::{CouplingEntry, CouplingStencil, StencilSource};
