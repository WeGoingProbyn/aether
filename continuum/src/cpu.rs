// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Serial CPU backend for finite-volume solver steps.
//!
//! This backend owns reusable CPU scratch buffers and runs the local solver
//! kernels synchronously. Parallel scheduling belongs above continuum.

use pleroma::core::storage::FieldStorage;
use tessera::mesh::Mesh;

use crate::{
  boundary::BoundaryRegistry,
  kernel,
  model::{ConservationLaw, NumericalFlux},
  solver::{FvmBackend, SolverConfig, TimeIntegration},
};

#[derive(Clone)]
pub struct CpuBackend<const N: usize> {
  scratch: CpuScratch<N>,
}

#[derive(Clone)]
struct CpuScratch<const N: usize> {
  state_cache: Vec<[f64; N]>,
  residual_accum: Vec<[f64; N]>,
  cell_state: [f64; N],
}

impl<const N: usize> Default for CpuBackend<N> {
  fn default() -> Self {
    Self {
      scratch: CpuScratch::default(),
    }
  }
}

impl<const N: usize> Default for CpuScratch<N> {
  fn default() -> Self {
    Self {
      state_cache: Vec::new(),
      residual_accum: Vec::new(),
      cell_state: [0.0; N],
    }
  }
}

impl<const N: usize> CpuScratch<N> {
  fn ensure_len(&mut self, count: usize) {
    if self.state_cache.len() != count {
      self.state_cache.resize(count, [0.0; N]);
    }
    if self.residual_accum.len() != count {
      self.residual_accum.resize(count, [0.0; N]);
    }
  }
}

impl<const N: usize> CpuBackend<N> {
  fn advance_cached<const D: usize, L, F, S, M>(
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
    L: ConservationLaw<D, N>,
    F: NumericalFlux<D, N>,
    S: FieldStorage<N>,
    M: Mesh<D> + ?Sized,
  {
    match config.integrator() {
      TimeIntegration::ForwardEuler => {
        kernel::compute_residual_from_cache_with_accum(
          law,
          flux,
          &self.scratch.state_cache,
          &mut self.scratch.residual_accum,
          residual,
          mesh,
          bcs,
        );
        state.axpy(dt, residual);
      }

      TimeIntegration::Rk2 => {
        let u_old = state.clone_state();

        kernel::compute_residual_from_cache_with_accum(
          law,
          flux,
          &self.scratch.state_cache,
          &mut self.scratch.residual_accum,
          residual,
          mesh,
          bcs,
        );
        state.axpy(dt, residual);

        kernel::gather_state_cache(state, mesh, &mut self.scratch.state_cache);
        kernel::compute_residual_from_cache_with_accum(
          law,
          flux,
          &self.scratch.state_cache,
          &mut self.scratch.residual_accum,
          residual,
          mesh,
          bcs,
        );
        state.axpy(dt, residual);

        let stage2 = state.clone_state();
        state.weighted_sum(0.5, &u_old, 0.5, &stage2);
      }
    }

    kernel::fix_state(law, state, mesh, &mut self.scratch.cell_state);
  }
}

impl<const D: usize, const N: usize, L, F> FvmBackend<D, N, L, F>
  for CpuBackend<N>
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
    self.scratch.ensure_len(mesh.cell_count());
    kernel::gather_state_cache(state, mesh, &mut self.scratch.state_cache);
    let dt = kernel::compute_dt_from_cache(
      config,
      law,
      &self.scratch.state_cache,
      mesh,
    );

    self.advance_cached(config, law, flux, dt, state, residual, mesh, bcs);
    dt
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
    self.scratch.ensure_len(mesh.cell_count());
    kernel::gather_state_cache(state, mesh, &mut self.scratch.state_cache);

    self.advance_cached(config, law, flux, dt, state, residual, mesh, bcs);
  }
}
