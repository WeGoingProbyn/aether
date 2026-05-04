// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use pleroma::core::storage::FieldStorage;
use tessera::mesh::Mesh;
use utility::{domain::CellId, profile};

use crate::{
  boundary::BoundaryRegistry,
  cpu::CpuBackend,
  kernel,
  model::{ConservationLaw, NumericalFlux},
};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum TimeIntegration {
  ForwardEuler,
  Rk2,
}

#[derive(Clone)]
pub struct SolverConfig {
  cfl: f64,
  dt_max: f64,
  integrator: TimeIntegration,
}

impl SolverConfig {
  pub fn new(
    cfl: f64,
    dt_max: f64,
    integrator: TimeIntegration,
  ) -> SolverConfig {
    SolverConfig {
      cfl,
      dt_max,
      integrator,
    }
  }

  pub fn dt_max(&self) -> f64 {
    self.dt_max
  }

  pub fn cfl(&self) -> f64 {
    self.cfl
  }

  pub fn set_dt_max(&mut self, dt_max: f64) {
    self.dt_max = dt_max;
  }

  pub fn integrator(&self) -> TimeIntegration {
    self.integrator
  }
}

pub trait FvmBackend<const D: usize, const N: usize, L, F>
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
    M: Mesh<D> + ?Sized;
}

#[derive(Clone)]
pub struct FvmSolver<const D: usize, const N: usize, L, F, B = CpuBackend<N>>
where
  L: ConservationLaw<D, N>,
  F: NumericalFlux<D, N>,
  B: FvmBackend<D, N, L, F>,
{
  config: SolverConfig,
  time: f64,
  step: usize,
  law: L,
  flux: F,
  backend: B,
}

impl<const D: usize, const N: usize, L, F> FvmSolver<D, N, L, F, CpuBackend<N>>
where
  L: ConservationLaw<D, N>,
  F: NumericalFlux<D, N>,
{
  pub fn new(config: SolverConfig, law: L, flux: F) -> Self {
    Self::with_backend(config, law, flux, CpuBackend::default())
  }
}

impl<const D: usize, const N: usize, L, F, B> FvmSolver<D, N, L, F, B>
where
  L: ConservationLaw<D, N>,
  F: NumericalFlux<D, N>,
  B: FvmBackend<D, N, L, F>,
{
  pub fn with_backend(
    config: SolverConfig,
    law: L,
    flux: F,
    backend: B,
  ) -> Self {
    FvmSolver {
      config,
      time: 0.0,
      step: 0,
      law,
      flux,
      backend,
    }
  }

  pub fn time(&self) -> f64 {
    self.time
  }

  pub fn current_step(&self) -> usize {
    self.step
  }

  pub fn law(&self) -> &L {
    &self.law
  }

  pub fn config(&self) -> &SolverConfig {
    &self.config
  }

  pub fn config_mut(&mut self) -> &mut SolverConfig {
    &mut self.config
  }

  #[profile]
  pub fn compute_dt<S, M>(&self, state: &S, mesh: &M) -> f64
  where
    S: FieldStorage<N>,
    M: Mesh<D> + ?Sized,
  {
    let mut dt_min = self.config.dt_max;
    let mut cell_state = [0.0; N];

    for i in 0..mesh.cell_count() {
      let cell = CellId::from(i);
      state.state_into(cell, &mut cell_state);

      let speed = self.law.max_wave_speed(&cell_state);
      if speed > 1e-14 {
        let dx = kernel::characteristic_length(mesh, cell);
        let dt_local = self.config.cfl() * dx / speed;
        dt_min = dt_min.min(dt_local);
      }
    }

    dt_min
  }

  #[profile]
  pub fn compute_residual<S, M>(
    &self,
    state: &S,
    residual: &mut S,
    mesh: &M,
    bcs: &BoundaryRegistry<D, N>,
  ) where
    S: FieldStorage<N>,
    M: Mesh<D> + ?Sized,
  {
    let mut state_cache = Vec::new();
    let mut residual_accum = Vec::new();
    kernel::gather_state_cache(state, mesh, &mut state_cache);
    kernel::compute_residual_from_cache_with_accum(
      &self.law,
      &self.flux,
      &state_cache,
      &mut residual_accum,
      residual,
      mesh,
      bcs,
    );
  }

  #[profile]
  pub fn step<S, M>(
    &mut self,
    state: &mut S,
    residual: &mut S,
    mesh: &M,
    bcs: &BoundaryRegistry<D, N>,
  ) -> f64
  where
    S: FieldStorage<N>,
    M: Mesh<D> + ?Sized,
  {
    let dt = self.backend.step(
      &self.config,
      &self.law,
      &self.flux,
      state,
      residual,
      mesh,
      bcs,
    );
    self.time += dt;
    self.step += 1;
    dt
  }
}
