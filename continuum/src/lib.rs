// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Domain-neutral finite-volume solver, generic over dimension, state size, the
//! `ConservationLaw`, and the `NumericalFlux`. A deliberately serial CPU solver
//! — parallelism is the scheduler's job (N solvers over N partitions). Explicit,
//! implicit (matrix-free GMRES), and IMEX backends sit behind one `FvmBackend`.
//!
//! See `continuum/docs/overview.md` for the generic core and the AD gotcha.

pub mod boundary;
pub mod cpu;
pub mod diagnostics;
pub mod implicit;
mod kernel;
pub mod model;
pub mod output;
pub mod solver;
