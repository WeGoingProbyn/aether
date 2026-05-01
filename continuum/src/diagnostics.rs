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
/// `(name, total)` pairs in the order declared on the law.
///
/// Mesh dim is hard-coded to 3 to match the rest of the workspace; the
/// signature can be generalised later if a 1D / 2D mesh starts seeing real
/// use.
pub fn integrate_conserved<const N: usize, M, S, L>(
  _law: &L,
  mesh: &M,
  field: &S,
) -> Vec<(&'static str, f64)>
where
  M: CellGeometry<3>,
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
