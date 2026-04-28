// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use utility::{domain::CellId, profile};

use pleroma::core::storage::FieldStorage;

use tessera::geometry::{CellGeometry, FaceGeometry};
use tessera::mesh::Mesh;
use tessera::topology::{FaceConnection, Topology};

use crate::{
  boundary::BoundaryRegistry,
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

  pub fn integrator(&self) -> TimeIntegration {
    self.integrator
  }
}

#[derive(Clone)]
pub(crate) struct SolverScratch<const N: usize> {
  pub(crate) state_cache: Vec<[f64; N]>,
  pub(crate) residual_accum: Vec<[f64; N]>,
  cell_state: [f64; N],
}

impl<const N: usize> Default for SolverScratch<N> {
  fn default() -> Self {
    SolverScratch {
      state_cache: Vec::new(),
      residual_accum: Vec::new(),
      cell_state: [0.0; N],
    }
  }
}

impl<const N: usize> SolverScratch<N> {
  pub(crate) fn ensure_len(&mut self, count: usize) {
    if self.state_cache.len() != count {
      self.state_cache.resize(count, [0.0; N]);
    }
    if self.residual_accum.len() != count {
      self.residual_accum.resize(count, [0.0; N]);
    }
  }
}

#[derive(Clone)]
pub struct FvmSolver<const D: usize, const N: usize, L, F>
where
  L: ConservationLaw<D, N>,
  F: NumericalFlux<D, N>,
{
  config: SolverConfig,
  time: f64,
  step: usize,
  law: L,
  flux: F,
  scratches: Vec<SolverScratch<N>>,
}

impl<const D: usize, const N: usize, L, F> FvmSolver<D, N, L, F>
where
  L: ConservationLaw<D, N>,
  F: NumericalFlux<D, N>,
{
  pub fn new(config: SolverConfig, law: L, flux: F) -> Self {
    FvmSolver {
      config,
      time: 0.0,
      step: 0,
      law,
      flux,
      scratches: vec![SolverScratch::default()],
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

  pub(crate) fn gather_state_cache<S, G>(
    state: &S,
    geometry: &G,
    cache: &mut Vec<[f64; N]>,
  ) where
    S: FieldStorage<N>,
    G: CellGeometry<D>,
  {
    let cell_count = geometry.cell_count();
    if cache.len() != cell_count {
      cache.resize(cell_count, [0.0; N]);
    }

    for (i, cell_state) in cache.iter_mut().enumerate().take(cell_count) {
      state.state_into(CellId::from(i), cell_state);
    }
  }

  pub(crate) fn compute_dt_from_cache<G>(
    config: &SolverConfig,
    law: &L,
    state_cache: &[[f64; N]],
    mesh: &G,
  ) -> f64
  where
    G: CellGeometry<D>,
  {
    let mut dt_min = config.dt_max;

    for (i, cell_state) in
      state_cache.iter().enumerate().take(mesh.cell_count())
    {
      let speed = law.max_wave_speed(cell_state);
      if speed > 1e-14 {
        let vol = mesh.cell_volume(CellId::from(i));
        let dx = vol.powf(1.0 / D as f64);
        let dt_local = config.cfl * dx / speed;
        dt_min = dt_min.min(dt_local);
      }
    }

    dt_min
  }

  pub(crate) fn compute_residual_from_cache_with_accum<S, M>(
    law: &L,
    flux_solver: &F,
    state_cache: &[[f64; N]],
    accumulated: &mut Vec<[f64; N]>,
    residual: &mut S,
    mesh: &M,
    bcs: &BoundaryRegistry<D, N>,
  ) where
    S: FieldStorage<N>,
    M: Mesh<D>,
  {
    let cell_count = mesh.cell_count();
    debug_assert_eq!(state_cache.len(), cell_count);

    if accumulated.len() != cell_count {
      accumulated.resize(cell_count, [0.0; N]);
    }
    accumulated.fill([0.0; N]);

    // Interior faces
    for &(face, owner, neighbour) in mesh.interior_faces() {
      let area_vec = mesh.face_area_vector(face);
      let area = mesh.face_area(face);
      let normal = &area_vec / &area;

      let flux = flux_solver.compute(
        law,
        &state_cache[owner.index()],
        &state_cache[neighbour.index()],
        &normal,
      );

      let owner_index = owner.index();
      let neighbour_index = neighbour.index();
      let face_scale = area * mesh.face_metrics(face).sqrt_metric;

      for i in 0..N {
        let scaled = flux[i] * face_scale;
        accumulated[owner_index][i] -= scaled;
        accumulated[neighbour_index][i] += scaled;
      }
    }

    // Boundary faces
    for tag in mesh.boundary_tags() {
      if let Some(bc) = bcs.get(tag) {
        for &(face, owner) in mesh.boundary_faces(tag) {
          let area_vec = mesh.face_area_vector(face);
          let area = mesh.face_area(face);
          let out_sign = match mesh.face_connection(face) {
            FaceConnection::Boundary { out_sign, .. } => *out_sign,
            _ => unreachable!(),
          };
          let normal = &area_vec / &area * out_sign;

          let owner_index = owner.index();
          let interior = &state_cache[owner_index];
          let ghost = bc.ghost_state(interior, &normal);
          let flux = flux_solver.compute(law, interior, &ghost, &normal);
          let face_scale = area * mesh.face_metrics(face).sqrt_metric;

          for i in 0..N {
            accumulated[owner_index][i] -= flux[i] * face_scale;
          }
        }
      }
    }

    // Divide by volume + add source terms
    for (i, accum_state) in accumulated.iter().enumerate().take(cell_count) {
      let cell = CellId::from(i);
      let vol = mesh.cell_volume(cell);
      let metrics = mesh.cell_metrics(cell);

      let source =
        law.source(&state_cache[i], cell, mesh.cell_centroid(cell), metrics);

      let mut out = [0.0; N];
      for j in 0..N {
        out[j] = accum_state[j] / vol + source[j] * metrics.sqrt_metric;
      }

      residual.write(cell, &out);
    }
  }

  #[profile]
  pub fn compute_dt(
    &self,
    state: &impl FieldStorage<N>,
    mesh: &impl Mesh<D>,
  ) -> f64 {
    let mut dt_min = self.config.dt_max;
    let mut cell_state = [0.0; N];

    for i in 0..mesh.cell_count() {
      let cell = CellId::from(i);
      state.state_into(cell, &mut cell_state);

      let speed = self.law.max_wave_speed(&cell_state);
      if speed > 1e-14 {
        let vol = mesh.cell_volume(cell);
        let dx = vol.powf(1.0 / D as f64);
        let dt_local = self.config.cfl * dx / speed;
        dt_min = dt_min.min(dt_local);
      }
    }

    dt_min
  }

  #[profile]
  pub fn compute_residual<S>(
    &self,
    state: &S,
    residual: &mut S,
    mesh: &impl Mesh<D>,
    bcs: &BoundaryRegistry<D, N>,
  ) where
    S: FieldStorage<N>,
  {
    let mut state_cache = Vec::new();
    let mut residual_accum = Vec::new();
    Self::gather_state_cache(state, mesh, &mut state_cache);
    Self::compute_residual_from_cache_with_accum(
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
  pub fn step<S: FieldStorage<N>>(
    &mut self,
    state: &mut S,
    residual: &mut S,
    mesh: &(impl CellGeometry<D> + FaceGeometry<D> + Topology),
    bcs: &BoundaryRegistry<D, N>,
  ) -> f64 {
    let dt = {
      self.ensure_scratch_slots(1);
      let (law, flux, config, scratch) =
        (&self.law, &self.flux, &self.config, &mut self.scratches[0]);

      scratch.ensure_len(mesh.cell_count());
      Self::gather_state_cache(state, mesh, &mut scratch.state_cache);
      let dt =
        Self::compute_dt_from_cache(config, law, &scratch.state_cache, mesh);

      match config.integrator {
        TimeIntegration::ForwardEuler => {
          Self::compute_residual_from_cache_with_accum(
            law,
            flux,
            &scratch.state_cache,
            &mut scratch.residual_accum,
            residual,
            mesh,
            bcs,
          );
          state.axpy(dt, residual);
        }

        TimeIntegration::Rk2 => {
          let u_old = state.clone_state();

          // Stage 1: state = u + dt * R(u)
          Self::compute_residual_from_cache_with_accum(
            law,
            flux,
            &scratch.state_cache,
            &mut scratch.residual_accum,
            residual,
            mesh,
            bcs,
          );
          state.axpy(dt, residual);

          // Stage 2: state = u* + dt * R(u*)
          Self::gather_state_cache(state, mesh, &mut scratch.state_cache);
          Self::compute_residual_from_cache_with_accum(
            law,
            flux,
            &scratch.state_cache,
            &mut scratch.residual_accum,
            residual,
            mesh,
            bcs,
          );
          state.axpy(dt, residual);

          let stage2 = state.clone_state();
          state.weighted_sum(0.5, &u_old, 0.5, &stage2);
        }
      }

      for i in 0..mesh.cell_count() {
        let cell = CellId::from(i);
        state.state_into(cell, &mut scratch.cell_state);
        law.fix_state(&mut scratch.cell_state);
        state.write(cell, &scratch.cell_state);
      }

      dt
    };

    self.time += dt;
    self.step += 1;
    dt
  }

  pub(crate) fn ensure_scratch_slots(&mut self, count: usize) {
    let required = count.max(1);
    if self.scratches.len() >= required {
      return;
    }
    self.scratches.resize_with(required, SolverScratch::default);
  }

  pub(crate) fn partitioned_parts(
    &mut self,
    partition_count: usize,
  ) -> (&L, &F, &SolverConfig, &mut [SolverScratch<N>]) {
    self.ensure_scratch_slots(partition_count);
    (
      &self.law,
      &self.flux,
      &self.config,
      &mut self.scratches[..partition_count],
    )
  }

  pub(crate) fn advance_clock(&mut self, dt: f64) {
    self.time += dt;
    self.step += 1;
  }
}
