// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use pleroma::core::storage::FieldStorage;
use tessera::{geometry::CellGeometry, mesh::Mesh, topology::FaceConnection};
use utility::{domain::CellId, profile};

use crate::{
  boundary::{BoundaryRegistry, BoundaryScalar},
  model::{ConservationLaw, NumericalFlux},
  solver::SolverConfig,
};

pub(crate) fn gather_state_cache<const D: usize, const N: usize, S, G>(
  state: &S,
  geometry: &G,
  cache: &mut Vec<[f64; N]>,
) where
  S: FieldStorage<N>,
  G: CellGeometry<D> + ?Sized,
{
  let cell_count = geometry.cell_count();
  if cache.len() != cell_count {
    cache.resize(cell_count, [0.0; N]);
  }

  for (i, cell_state) in cache.iter_mut().enumerate().take(cell_count) {
    state.state_into(CellId::from(i), cell_state);
  }
}

pub(crate) fn compute_dt_from_cache<const D: usize, const N: usize, L, G>(
  config: &SolverConfig,
  law: &L,
  state_cache: &[[f64; N]],
  mesh: &G,
) -> f64
where
  L: ConservationLaw<D, N>,
  G: Mesh<D> + ?Sized,
{
  let mut dt_min = config.dt_max();

  for (i, cell_state) in state_cache.iter().enumerate().take(mesh.cell_count())
  {
    let speed = law.max_wave_speed(cell_state);
    if speed > 1e-14 {
      let cell = CellId::from(i);
      let dx = characteristic_length(mesh, cell);
      let dt_local = config.cfl() * dx / speed;
      dt_min = dt_min.min(dt_local);
    }
  }

  dt_min
}

/// CFL step sized by the *explicit* (advective) wave speed — for IMEX, where
/// the acoustic terms are handled implicitly and the stable step is set by
/// advection rather than sound.
pub(crate) fn compute_explicit_dt_from_cache<
  const D: usize,
  const N: usize,
  L,
  G,
>(
  config: &SolverConfig,
  law: &L,
  state_cache: &[[f64; N]],
  mesh: &G,
) -> f64
where
  L: ConservationLaw<D, N>,
  G: Mesh<D> + ?Sized,
{
  let mut dt_min = config.dt_max();
  for (i, cell_state) in state_cache.iter().enumerate().take(mesh.cell_count())
  {
    let speed = law.explicit_wave_speed(cell_state);
    if speed > 1e-14 {
      let cell = CellId::from(i);
      let dx = characteristic_length(mesh, cell);
      dt_min = dt_min.min(config.cfl() * dx / speed);
    }
  }
  dt_min
}

/// Accumulate the net face-flux integral for every cell into `accumulated`
/// (interior + boundary faces). This is the part of `R(U)` that couples cells;
/// the per-cell `source` and the `1/volume` scaling are applied by the callers
/// below. Shared by every residual entry point so they cannot diverge, and
/// generic over the scalar `T` (real for the explicit path, dual for AD) — the
/// boundary ghost is selected per scalar through [`BoundaryScalar`].
fn accumulate_face_fluxes<
  const D: usize,
  const N: usize,
  L,
  F,
  M,
  T: BoundaryScalar<D, N>,
>(
  law: &L,
  flux_solver: &F,
  state_cache: &[[T; N]],
  accumulated: &mut Vec<[T; N]>,
  mesh: &M,
  bcs: &BoundaryRegistry<D, N>,
) where
  L: ConservationLaw<D, N>,
  F: NumericalFlux<D, N>,
  M: Mesh<D> + ?Sized,
{
  let cell_count = mesh.cell_count();
  debug_assert_eq!(state_cache.len(), cell_count);

  let zero = [T::from(0.0); N];
  if accumulated.len() != cell_count {
    accumulated.resize(cell_count, zero);
  }
  accumulated.fill(zero);

  for &(face, owner, neighbour) in mesh.interior_faces() {
    let area_vec = mesh.face_area_vector(face);
    let area = mesh.face_area(face);
    let normal = area_vec / area;

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
      accumulated[owner_index][i] = accumulated[owner_index][i] - scaled;
      accumulated[neighbour_index][i] =
        accumulated[neighbour_index][i] + scaled;
    }
  }

  for tag in mesh.boundary_tags() {
    if let Some(bc) = bcs.get(tag) {
      for &(face, owner) in mesh.boundary_faces(tag) {
        let area_vec = mesh.face_area_vector(face);
        let area = mesh.face_area(face);
        let out_sign = match mesh.face_connection(face) {
          FaceConnection::Boundary { out_sign, .. } => *out_sign,
          _ => unreachable!(),
        };
        let normal = area_vec / area * out_sign;

        let owner_index = owner.index();
        let interior = &state_cache[owner_index];
        let ghost = T::ghost(bc, interior, &normal);
        let flux = flux_solver.compute(law, interior, &ghost, &normal);
        let face_scale = area * mesh.face_metrics(face).sqrt_metric;

        for i in 0..N {
          accumulated[owner_index][i] =
            accumulated[owner_index][i] - flux[i] * face_scale;
        }
      }
    }
  }
}

#[profile]
pub(crate) fn compute_residual_from_cache_with_accum<
  const D: usize,
  const N: usize,
  L,
  F,
  S,
  M,
>(
  law: &L,
  flux_solver: &F,
  state_cache: &[[f64; N]],
  accumulated: &mut Vec<[f64; N]>,
  residual: &mut S,
  mesh: &M,
  bcs: &BoundaryRegistry<D, N>,
) where
  L: ConservationLaw<D, N>,
  F: NumericalFlux<D, N>,
  S: FieldStorage<N>,
  M: Mesh<D> + ?Sized,
{
  accumulate_face_fluxes(law, flux_solver, state_cache, accumulated, mesh, bcs);

  for (i, accum_state) in accumulated.iter().enumerate().take(mesh.cell_count())
  {
    let cell = CellId::from(i);
    let metrics = mesh.cell_metrics(cell);
    let vol = metrics.phys_volume;

    let source =
      law.source(&state_cache[i], cell, mesh.cell_centroid(cell), metrics);

    let mut out = [0.0; N];
    for j in 0..N {
      out[j] = accum_state[j] / vol + source[j];
    }

    residual.write(cell, &out);
  }
}

/// Compute `R(U) = dU/dt` from a state cache directly into a plain slice — the
/// form the implicit solver's matrix-free machinery needs (no `FieldStorage`).
/// Generic over the scalar: `T = f64` is the ordinary residual, `T = Dual64`
/// yields an exact directional derivative `J·v` when the cache carries the
/// seed in its dual part.
///
/// `out` is written for every cell as `net_face_flux / volume + source`. The
/// function reads only the `state_cache` it is handed, so a finite-difference
/// Jacobian can perturb a copy of the cache and re-evaluate with no hidden,
/// potentially-stale flux cache in play.
pub(crate) fn compute_residual_generic<
  const D: usize,
  const N: usize,
  L,
  F,
  M,
  T: BoundaryScalar<D, N>,
>(
  law: &L,
  flux_solver: &F,
  state_cache: &[[T; N]],
  accumulated: &mut Vec<[T; N]>,
  out: &mut [[T; N]],
  mesh: &M,
  bcs: &BoundaryRegistry<D, N>,
) where
  L: ConservationLaw<D, N>,
  F: NumericalFlux<D, N>,
  M: Mesh<D> + ?Sized,
{
  let cell_count = mesh.cell_count();
  debug_assert_eq!(out.len(), cell_count);

  accumulate_face_fluxes(law, flux_solver, state_cache, accumulated, mesh, bcs);

  for (i, accum_state) in accumulated.iter().enumerate().take(cell_count) {
    let cell = CellId::from(i);
    let metrics = mesh.cell_metrics(cell);
    let vol = metrics.phys_volume;

    let source =
      law.source(&state_cache[i], cell, mesh.cell_centroid(cell), metrics);

    for j in 0..N {
      out[i][j] = accum_state[j] / vol + source[j];
    }
  }
}

/// Accumulate the net *implicit-part* face-flux integral for every cell — the
/// IMEX counterpart of [`accumulate_face_fluxes`], using the numerical flux's
/// `compute_implicit` (central `implicit_flux` + acoustic dissipation).
fn accumulate_implicit_face_fluxes<
  const D: usize,
  const N: usize,
  L,
  F,
  M,
  T: BoundaryScalar<D, N>,
>(
  law: &L,
  flux_solver: &F,
  state_cache: &[[T; N]],
  accumulated: &mut Vec<[T; N]>,
  mesh: &M,
  bcs: &BoundaryRegistry<D, N>,
) where
  L: ConservationLaw<D, N>,
  F: NumericalFlux<D, N>,
  M: Mesh<D> + ?Sized,
{
  let cell_count = mesh.cell_count();
  debug_assert_eq!(state_cache.len(), cell_count);

  let zero = [T::from(0.0); N];
  if accumulated.len() != cell_count {
    accumulated.resize(cell_count, zero);
  }
  accumulated.fill(zero);

  for &(face, owner, neighbour) in mesh.interior_faces() {
    let area_vec = mesh.face_area_vector(face);
    let area = mesh.face_area(face);
    let normal = area_vec / area;

    let flux = flux_solver.compute_implicit(
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
      accumulated[owner_index][i] = accumulated[owner_index][i] - scaled;
      accumulated[neighbour_index][i] =
        accumulated[neighbour_index][i] + scaled;
    }
  }

  for tag in mesh.boundary_tags() {
    if let Some(bc) = bcs.get(tag) {
      for &(face, owner) in mesh.boundary_faces(tag) {
        let area_vec = mesh.face_area_vector(face);
        let area = mesh.face_area(face);
        let out_sign = match mesh.face_connection(face) {
          FaceConnection::Boundary { out_sign, .. } => *out_sign,
          _ => unreachable!(),
        };
        let normal = area_vec / area * out_sign;

        let owner_index = owner.index();
        let interior = &state_cache[owner_index];
        let ghost = T::ghost(bc, interior, &normal);
        let flux = flux_solver.compute_implicit(law, interior, &ghost, &normal);
        let face_scale = area * mesh.face_metrics(face).sqrt_metric;

        for i in 0..N {
          accumulated[owner_index][i] =
            accumulated[owner_index][i] - flux[i] * face_scale;
        }
      }
    }
  }
}

/// Assemble the IMEX *implicit operator* residual `R_im(U)` into `out`, using
/// `implicit_flux` / `implicit_source`. Generic over the scalar so its exact
/// Jacobian can be taken by automatic differentiation — that Jacobian is what
/// the IMEX backend linearizes (only the stiff acoustic part).
pub(crate) fn compute_residual_implicit<
  const D: usize,
  const N: usize,
  L,
  F,
  M,
  T: BoundaryScalar<D, N>,
>(
  law: &L,
  flux_solver: &F,
  state_cache: &[[T; N]],
  accumulated: &mut Vec<[T; N]>,
  out: &mut [[T; N]],
  mesh: &M,
  bcs: &BoundaryRegistry<D, N>,
) where
  L: ConservationLaw<D, N>,
  F: NumericalFlux<D, N>,
  M: Mesh<D> + ?Sized,
{
  let cell_count = mesh.cell_count();
  debug_assert_eq!(out.len(), cell_count);

  accumulate_implicit_face_fluxes(
    law,
    flux_solver,
    state_cache,
    accumulated,
    mesh,
    bcs,
  );

  for (i, accum_state) in accumulated.iter().enumerate().take(cell_count) {
    let cell = CellId::from(i);
    let metrics = mesh.cell_metrics(cell);
    let vol = metrics.phys_volume;

    let source = law.implicit_source(
      &state_cache[i],
      cell,
      mesh.cell_centroid(cell),
      metrics,
    );

    for j in 0..N {
      out[i][j] = accum_state[j] / vol + source[j];
    }
  }
}

/// Real-valued convenience wrapper around [`compute_residual_generic`].
pub(crate) fn compute_residual_into<const D: usize, const N: usize, L, F, M>(
  law: &L,
  flux_solver: &F,
  state_cache: &[[f64; N]],
  accumulated: &mut Vec<[f64; N]>,
  out: &mut [[f64; N]],
  mesh: &M,
  bcs: &BoundaryRegistry<D, N>,
) where
  L: ConservationLaw<D, N>,
  F: NumericalFlux<D, N>,
  M: Mesh<D> + ?Sized,
{
  compute_residual_generic(
    law,
    flux_solver,
    state_cache,
    accumulated,
    out,
    mesh,
    bcs,
  );
}

pub(crate) fn fix_state<const D: usize, const N: usize, L, S, M>(
  law: &L,
  state: &mut S,
  mesh: &M,
  cell_state: &mut [f64; N],
) where
  L: ConservationLaw<D, N>,
  S: FieldStorage<N>,
  M: CellGeometry<D> + ?Sized,
{
  for i in 0..mesh.cell_count() {
    let cell = CellId::from(i);
    state.state_into(cell, cell_state);
    law.fix_state(cell_state);
    state.write(cell, cell_state);
  }
}

pub(crate) fn characteristic_length<const D: usize, M>(
  mesh: &M,
  cell: CellId,
) -> f64
where
  M: Mesh<D> + ?Sized,
{
  let volume = mesh.cell_metrics(cell).phys_volume;
  let max_face_area = mesh
    .cell_faces(cell)
    .iter()
    .map(|&face| mesh.face_metrics(face).phys_area)
    .fold(0.0, f64::max);

  if max_face_area > 0.0 && max_face_area.is_finite() {
    volume / max_face_area
  } else {
    volume.powf(1.0 / D as f64)
  }
}

/// The cell's *largest* directional spacing — `volume / min_face_area`. On a
/// thin shell cell the smallest face is a side wall (`dx_h · dz`), so this
/// returns the horizontal spacing `dx_h`. HEVI handles the vertical acoustics
/// implicitly, so its explicit stability is set by this horizontal spacing, not
/// the thin vertical one that [`characteristic_length`] returns.
pub(crate) fn horizontal_characteristic_length<const D: usize, M>(
  mesh: &M,
  cell: CellId,
) -> f64
where
  M: Mesh<D> + ?Sized,
{
  let volume = mesh.cell_metrics(cell).phys_volume;
  let min_face_area = mesh
    .cell_faces(cell)
    .iter()
    .map(|&face| mesh.face_metrics(face).phys_area)
    .filter(|a| *a > 0.0 && a.is_finite())
    .fold(f64::INFINITY, f64::min);

  if min_face_area.is_finite() && min_face_area > 0.0 {
    volume / min_face_area
  } else {
    volume.powf(1.0 / D as f64)
  }
}
