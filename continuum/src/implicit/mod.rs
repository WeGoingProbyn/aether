// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Matrix-free implicit time integration for the finite-volume solver.
//!
//! The acoustic CFL limit (`dt ≤ cfl·dx/(|u|+c)`) forces the explicit
//! [`CpuBackend`](crate::cpu::CpuBackend) into tiny steps even when the
//! physics of interest evolves slowly. This module adds an implicit path: a
//! Rosenbrock backend that linearizes the residual and solves the resulting
//! system with matrix-free GMRES, removing the sound speed from the step-size
//! constraint. It plugs in behind the existing
//! [`FvmBackend`](crate::solver::FvmBackend) trait; the explicit path and the
//! conservation laws are untouched.

pub mod ad;
pub mod backend;
pub mod dispatch;
pub mod gmres;
pub mod hybrid;
pub mod jacobian;
pub mod linalg;
pub mod schemes;
