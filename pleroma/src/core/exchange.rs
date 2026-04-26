// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Ghost-cell exchange between partition fields.
//!
//! Tessera owns the *topology* (which ghost cell on which partition mirrors
//! which owned cell on which other partition) via `GhostDescriptor`. Pleroma
//! owns the *field data*, so the actual scatter/gather lives here as a free
//! function over any `FieldStorage<N>`.

use tessera::geometry::{CellGeometry, FaceGeometry};
use tessera::partition::Decomposition;
use tessera::topology::Topology;
use utility::domain::CellId;
use utility::profile;

use crate::core::storage::{CellView, FieldStorage};

/// Exchange ghost-cell data between partition fields.
///
/// Two-pass to avoid aliasing: first collect all ghost values from their
/// source partitions, then scatter them into destination partitions' ghost
/// slots.
#[profile]
pub fn exchange_ghosts<const D: usize, const N: usize, M, S>(
  decomp: &Decomposition<D, M>,
  fields: &mut [S],
) where
  M: CellGeometry<D> + FaceGeometry<D> + Topology,
  S: FieldStorage<N>,
{
  let ghost_data: Vec<Vec<(CellId, [f64; N])>> = decomp
    .partitions
    .iter()
    .map(|p| {
      p.ghost_cells()
        .iter()
        .map(|g| {
          let val = *fields[g.source_partition]
            .state(g.source_local_cell)
            .as_state();
          (g.local_cell, val)
        })
        .collect()
    })
    .collect();

  for (dest, data) in ghost_data.into_iter().enumerate() {
    for (cell, val) in data {
      fields[dest].write(cell, &val);
    }
  }
}
