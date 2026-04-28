// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! CPU execution policy for partitioned finite-volume steps.
//!
//! This module is intentionally the place where continuum knows about the
//! host thread pool. The solver and model modules keep the numerical state
//! and kernels; this runner decides how to fan those kernels out on CPU.

use pleroma::core::{exchange::exchange_ghosts, storage::FieldStorage};
use tessera::{geometry::CellGeometry, mesh::Mesh, partition::Decomposition};
use utility::{domain::CellId, profile, thread::pool::Pool};

use crate::{
  boundary::BoundaryRegistry,
  model::{ConservationLaw, NumericalFlux},
  solver::{FvmSolver, SolverScratch, TimeIntegration},
};

pub struct CpuFvmRunner<'a> {
  pool: &'a Pool,
}

impl<'a> CpuFvmRunner<'a> {
  pub fn new(pool: &'a Pool) -> Self {
    Self { pool }
  }

  #[profile]
  pub fn step<const D: usize, const N: usize, L, F, M, S>(
    &self,
    solver: &mut FvmSolver<D, N, L, F>,
    decomp: &Decomposition<D, M>,
    states: &mut [S],
    residuals: &mut [S],
    bcs: &BoundaryRegistry<D, N>,
  ) -> f64
  where
    M: Mesh<D>,
    L: ConservationLaw<D, N> + Sync,
    F: NumericalFlux<D, N> + Sync,
    S: FieldStorage<N>,
  {
    exchange_ghosts(decomp, states);

    let dt = {
      let (law, flux, config, scratches) =
        solver.partitioned_parts(decomp.partitions.len());

      refresh_parallel_state_caches::<D, N, L, F, M, S>(
        decomp, states, scratches,
      );

      let dt = decomp
        .partitions
        .iter()
        .enumerate()
        .map(|(i, partition)| {
          FvmSolver::<D, N, L, F>::compute_dt_from_cache(
            config,
            law,
            &scratches[i].state_cache,
            partition,
          )
        })
        .fold(config.dt_max(), f64::min);

      match config.integrator() {
        TimeIntegration::ForwardEuler => {
          parallel_compute_residuals_from_cache::<D, N, L, F, M, S>(
            self.pool, law, flux, decomp, scratches, residuals, bcs,
          );
          parallel_axpy(self.pool, states, residuals, dt);
        }

        TimeIntegration::Rk2 => {
          let u_old: Vec<S> =
            states.iter().map(|state| state.clone_state()).collect();

          parallel_compute_residuals_from_cache::<D, N, L, F, M, S>(
            self.pool, law, flux, decomp, scratches, residuals, bcs,
          );
          parallel_axpy(self.pool, states, residuals, dt);

          exchange_ghosts(decomp, states);
          refresh_parallel_state_caches::<D, N, L, F, M, S>(
            decomp, states, scratches,
          );
          parallel_compute_residuals_from_cache::<D, N, L, F, M, S>(
            self.pool, law, flux, decomp, scratches, residuals, bcs,
          );
          parallel_axpy(self.pool, states, residuals, dt);

          let tasks: Vec<_> = states
            .iter_mut()
            .zip(u_old.iter())
            .map(|(state, old_state)| {
              move || {
                let stage2 = state.clone_state();
                state.weighted_sum(0.5, old_state, 0.5, &stage2);
              }
            })
            .collect();

          self.pool.dispatch(tasks);
        }
      }

      parallel_fix_owned::<D, N, L, M, S>(self.pool, law, decomp, states);
      dt
    };

    solver.advance_clock(dt);
    dt
  }
}

fn refresh_parallel_state_caches<const D: usize, const N: usize, L, F, M, S>(
  decomp: &Decomposition<D, M>,
  states: &[S],
  scratches: &mut [SolverScratch<N>],
) where
  M: Mesh<D>,
  L: ConservationLaw<D, N>,
  F: NumericalFlux<D, N>,
  S: FieldStorage<N>,
{
  for ((partition, state), scratch) in decomp
    .partitions
    .iter()
    .zip(states.iter())
    .zip(scratches.iter_mut())
  {
    scratch.ensure_len(partition.cell_count());
    FvmSolver::<D, N, L, F>::gather_state_cache(
      state,
      partition,
      &mut scratch.state_cache,
    );
  }
}

fn parallel_compute_residuals_from_cache<
  const D: usize,
  const N: usize,
  L,
  F,
  M,
  S,
>(
  pool: &Pool,
  law: &L,
  flux_solver: &F,
  decomp: &Decomposition<D, M>,
  scratches: &mut [SolverScratch<N>],
  residuals: &mut [S],
  bcs: &BoundaryRegistry<D, N>,
) where
  M: Mesh<D>,
  L: ConservationLaw<D, N> + Sync,
  F: NumericalFlux<D, N> + Sync,
  S: FieldStorage<N>,
{
  let tasks: Vec<_> = scratches
    .iter_mut()
    .zip(residuals.iter_mut())
    .zip(decomp.partitions.iter())
    .map(|((scratch, residual), partition)| {
      move || {
        FvmSolver::<D, N, L, F>::compute_residual_from_cache_with_accum(
          law,
          flux_solver,
          &scratch.state_cache,
          &mut scratch.residual_accum,
          residual,
          partition,
          bcs,
        );
      }
    })
    .collect();

  pool.dispatch(tasks);
}

fn parallel_axpy<const N: usize, S>(
  pool: &Pool,
  states: &mut [S],
  residuals: &[S],
  alpha: f64,
) where
  S: FieldStorage<N>,
{
  let tasks: Vec<_> = states
    .iter_mut()
    .zip(residuals.iter())
    .map(|(state, residual)| {
      move || {
        state.axpy(alpha, residual);
      }
    })
    .collect();

  pool.dispatch(tasks);
}

fn parallel_fix_owned<const D: usize, const N: usize, L, M, S>(
  pool: &Pool,
  law: &L,
  decomp: &Decomposition<D, M>,
  states: &mut [S],
) where
  M: Mesh<D>,
  L: ConservationLaw<D, N> + Sync,
  S: FieldStorage<N>,
{
  let tasks: Vec<_> = states
    .iter_mut()
    .zip(decomp.partitions.iter())
    .map(|(state, partition)| {
      move || {
        let mut cell_state = [0.0; N];
        for i in 0..partition.num_owned() {
          let cell = CellId::from(i);
          state.state_into(cell, &mut cell_state);
          law.fix_state(&mut cell_state);
          state.write(cell, &cell_state);
        }
      }
    })
    .collect();

  pool.dispatch(tasks);
}
