// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Syzygy owns coupling semantics between physics modules. It consumes
//! read-only mesh/coupler geometry from Tessera and field access from Nexus.

pub mod scalar;

pub use scalar::{ScalarRelaxation, SyzygyError};
