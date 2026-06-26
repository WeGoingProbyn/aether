// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Volume-weighted reductions over a state field. Lives in continuum
//! because it's a numerical operation on world state — pleroma stays
//! out of the physics-aware part of the diagnostics layer.
//!
//! The reducer is generic over any law that declares
//! `ConservationQuantities<N>`; callers typically derive that via
//! `#[derive(StateDiagnostics)]` and never name this trait directly.

use pleroma::core::storage::FieldStorage;
use tessera::geometry::CellGeometry;
use utility::diagnostics::ConservationQuantities;
use utility::domain::CellId;

/// Volume-integrate each declared conserved component, returning
/// `(name, total)` pairs in the order declared on the law `L`.
///
/// The law contributes only its associated `CONSERVED_QUANTITIES` const, so
/// `L` is a turbofish type parameter rather than an instance argument — the
/// caller never has to construct a law just to read off its conserved
/// component indices: `integrate_conserved::<N, _, _, MyLaw>(mesh, field)`.
///
/// Mesh dim is hard-coded to 3 to match the rest of the workspace; the
/// signature can be generalised later if a 1D / 2D mesh starts seeing real
/// use.
pub fn integrate_conserved<const N: usize, M, S, L>(
  mesh: &M,
  field: &S,
) -> Vec<(&'static str, f64)>
where
  M: CellGeometry<3> + ?Sized,
  S: FieldStorage<N>,
  L: ConservationQuantities<N>,
{
  let quantities = L::CONSERVED_QUANTITIES;
  let mut totals = vec![0.0f64; quantities.len()];
  let mut state = [0.0f64; N];

  for i in 0..mesh.cell_count() {
    let cell = CellId::from(i);
    let volume = mesh.cell_volume(cell);
    field.state_into(cell, &mut state);

    for (q, total) in quantities.iter().zip(totals.iter_mut()) {
      *total += state[q.component] * volume;
    }
  }

  quantities
    .iter()
    .zip(totals)
    .map(|(q, total)| (q.name, total))
    .collect()
}

/// Count cells whose state has at least one non-finite (NaN/Inf) component.
///
/// Law-agnostic: needs only the state size `N`, so it guards any field — a
/// cheap per-tick blow-up detector. Returns `0` for a clean field.
pub fn count_non_finite<const N: usize, S>(field: &S) -> usize
where
  S: FieldStorage<N>,
{
  let mut state = [0.0f64; N];
  let mut count = 0;
  for i in 0..field.len() {
    field.state_into(CellId::from(i), &mut state);
    if state.iter().any(|v| !v.is_finite()) {
      count += 1;
    }
  }
  count
}
