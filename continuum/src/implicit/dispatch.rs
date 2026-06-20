// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! A runtime backend selector. Solvers are parameterised by a concrete backend
//! type, so a consumer that wants to *choose* its time-stepping strategy at
//! runtime (explicit vs implicit vs hybrid) would otherwise need a generic
//! solver field per choice. [`BackendKind`] is a single enum that implements
//! [`FvmBackend`] by dispatch, so the consumer keeps one concrete
//! `FvmSolver<…, BackendKind<N>>` and picks the variant by configuration.

use pleroma::core::storage::FieldStorage;
use tessera::mesh::Mesh;

use crate::{
  boundary::BoundaryRegistry,
  cpu::CpuBackend,
  implicit::{backend::ImplicitBackend, hybrid::HybridBackend},
  model::{ConservationLaw, NumericalFlux},
  solver::{FvmBackend, SolverConfig},
};

/// One of the available finite-volume backends, selected at runtime.
pub enum BackendKind<const N: usize> {
  /// Explicit CFL-limited stepping (the default, ground-truth path).
  Explicit(CpuBackend<N>),
  /// Implicit Rosenbrock stepping (large stable steps, may fail to converge).
  Implicit(Box<ImplicitBackend<N>>),
  /// Implicit with explicit fallback on rejection (robust large steps).
  Hybrid(Box<HybridBackend<N>>),
}

impl<const N: usize> BackendKind<N> {
  pub fn explicit() -> Self {
    BackendKind::Explicit(CpuBackend::default())
  }

  pub fn implicit(backend: ImplicitBackend<N>) -> Self {
    BackendKind::Implicit(Box::new(backend))
  }

  pub fn hybrid(backend: HybridBackend<N>) -> Self {
    BackendKind::Hybrid(Box::new(backend))
  }
}

impl<const D: usize, const N: usize, L, F> FvmBackend<D, N, L, F>
  for BackendKind<N>
where
  L: ConservationLaw<D, N>,
  F: NumericalFlux<D, N>,
{
  fn step<S, M>(
    &mut self,
    config: &SolverConfig,
    law: &L,
    flux: &F,
    state: &mut S,
    residual: &mut S,
    mesh: &M,
    bcs: &BoundaryRegistry<D, N>,
  ) -> f64
  where
    S: FieldStorage<N>,
    M: Mesh<D> + ?Sized,
  {
    match self {
      BackendKind::Explicit(b) => {
        b.step(config, law, flux, state, residual, mesh, bcs)
      }
      BackendKind::Implicit(b) => {
        b.step(config, law, flux, state, residual, mesh, bcs)
      }
      BackendKind::Hybrid(b) => {
        b.step(config, law, flux, state, residual, mesh, bcs)
      }
    }
  }

  fn step_with_dt<S, M>(
    &mut self,
    config: &SolverConfig,
    law: &L,
    flux: &F,
    dt: f64,
    state: &mut S,
    residual: &mut S,
    mesh: &M,
    bcs: &BoundaryRegistry<D, N>,
  ) where
    S: FieldStorage<N>,
    M: Mesh<D> + ?Sized,
  {
    match self {
      BackendKind::Explicit(b) => {
        b.step_with_dt(config, law, flux, dt, state, residual, mesh, bcs)
      }
      BackendKind::Implicit(b) => {
        b.step_with_dt(config, law, flux, dt, state, residual, mesh, bcs)
      }
      BackendKind::Hybrid(b) => {
        b.step_with_dt(config, law, flux, dt, state, residual, mesh, bcs)
      }
    }
  }
}
