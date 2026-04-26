// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashMap, sync::Arc};

use utility::{maths::vector::Vector, profile};

use crate::{
  geometry::{
    CellGeometry, CellId, CellMetrics, FaceGeometry, FaceId, FaceMetrics, Point,
  },
  mesh::{Mesh, StructuredBlock},
  topology::{BoundaryTag, FaceConnection, Topology},
};

/// Describes a ghost cell in a partition
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GhostDescriptor {
  pub local_cell: CellId, // local index in this partition (after owned cells)
  pub source_partition: usize, // which partition owns the original
  pub source_local_cell: CellId, // local index in the source partition
}

/// A partition-local view of any mesh with ghost cells.
pub struct PartitionMesh<const D: usize, M>
where
  M: CellGeometry<D> + FaceGeometry<D>,
{
  mesh: Arc<M>,

  // Cell mapping: local → global
  local_to_global_cell: Vec<CellId>, // [0..num_owned] = owned, [num_owned..] = ghost
  num_owned: usize,

  // Face mapping: local → global
  local_to_global_face: Vec<FaceId>,

  // Ghost cell info
  ghost_cells: Vec<GhostDescriptor>,

  // Topology (all using LOCAL cell IDs and LOCAL face IDs)
  face_connections: Vec<FaceConnection>,
  cell_face_adj: Vec<Vec<FaceId>>,
  interior_and_halo_faces: Vec<(FaceId, CellId, CellId)>, // includes halo faces
  boundary_face_lists: Vec<(BoundaryTag, Vec<(FaceId, CellId)>)>,
}

impl<const D: usize, M> PartitionMesh<D, M>
where
  M: Mesh<D>,
{
  pub fn local_cell_count(&self) -> usize {
    self.local_to_global_cell.len()
  }

  pub fn num_owned(&self) -> usize {
    self.num_owned
  }

  pub fn local_to_global(&self, cell: CellId) -> CellId {
    self.local_to_global_cell[cell.index()]
  }

  pub fn mesh(&self) -> &M {
    self.mesh.as_ref()
  }

  pub fn local_to_global_cells(&self) -> &[CellId] {
    &self.local_to_global_cell
  }

  pub fn ghost_cells(&self) -> &[GhostDescriptor] {
    &self.ghost_cells
  }
}

impl<const D: usize, M> CellGeometry<D> for PartitionMesh<D, M>
where
  M: CellGeometry<D> + FaceGeometry<D>,
{
  fn cell_centroid(&self, cell: CellId) -> &Point<D> {
    let global = self.local_to_global_cell[cell.index()];
    self.mesh.cell_centroid(global)
  }

  fn cell_volume(&self, cell: CellId) -> f64 {
    let global = self.local_to_global_cell[cell.index()];
    self.mesh.cell_volume(global)
  }

  fn cell_metrics(&self, cell: CellId) -> &CellMetrics<D> {
    let global = self.local_to_global_cell[cell.index()];
    self.mesh.cell_metrics(global)
  }

  fn cell_count(&self) -> usize {
    self.local_to_global_cell.len() // owned + ghost
  }
}

impl<const D: usize, M> FaceGeometry<D> for PartitionMesh<D, M>
where
  M: Mesh<D>,
{
  fn face_centroid(&self, face: FaceId) -> &Point<D> {
    let global = self.local_to_global_face[face.index()];
    self.mesh.face_centroid(global)
  }

  fn face_area_vector(&self, face: FaceId) -> Vector<f64, D> {
    let global = self.local_to_global_face[face.index()];
    self.mesh.face_area_vector(global)
  }

  fn face_area(&self, face: FaceId) -> f64 {
    let global = self.local_to_global_face[face.index()];
    self.mesh.face_area(global)
  }

  fn face_metrics(&self, face: FaceId) -> &FaceMetrics<D> {
    let global = self.local_to_global_face[face.index()];
    self.mesh.face_metrics(global)
  }

  fn face_count(&self) -> usize {
    self.local_to_global_face.len()
  }
}

impl<const D: usize, M> Topology for PartitionMesh<D, M>
where
  M: Mesh<D>,
{
  fn face_connection(&self, face: FaceId) -> &FaceConnection {
    &self.face_connections[face.index()]
  }

  fn cell_faces(&self, cell: CellId) -> &[FaceId] {
    &self.cell_face_adj[cell.index()]
  }

  fn interior_faces(&self) -> &[(FaceId, CellId, CellId)] {
    // Includes halo faces — ghost cells have valid state after exchange,
    // so the solver processes them identically to true interior faces
    &self.interior_and_halo_faces
  }

  fn boundary_faces(&self, tag: BoundaryTag) -> &[(FaceId, CellId)] {
    self
      .boundary_face_lists
      .iter()
      .find(|(t, _)| *t == tag)
      .map(|(_, list)| list.as_slice())
      .unwrap_or(&[])
  }

  fn boundary_tags(&self) -> Box<dyn Iterator<Item = BoundaryTag> + '_> {
    Box::new(self.boundary_face_lists.iter().map(|(tag, _)| *tag))
  }
}

/// Result of decomposing a mesh. Generic over the source mesh type M.
pub struct Decomposition<const D: usize, M>
where
  M: CellGeometry<D> + FaceGeometry<D> + Topology,
{
  pub partitions: Vec<PartitionMesh<D, M>>,
}

impl<const D: usize, M: CellGeometry<D> + FaceGeometry<D> + Topology>
  Decomposition<D, M>
{
  /// Exchange ghost cell data between partition fields.
  /// Two-pass to avoid aliasing: collect all ghost values, then scatter.
  #[profile]
  pub fn exchange_ghosts<const N: usize, S: FieldStorage<N>>(
    &self,
    fields: &mut [S],
  ) {
    // Pass 1: collect ghost values from source partitions
    let ghost_data: Vec<Vec<(CellId, [f64; N])>> = self
      .partitions
      .iter()
      .map(|p| {
        p.ghost_cells
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

    // Pass 2: write ghost values to destination fields
    for (dest, data) in ghost_data.into_iter().enumerate() {
      for (cell, val) in data {
        fields[dest].write(cell, &val);
      }
    }
  }
}

/// Structured grid decomposition — slab partitioning along the longest axis.
/// Lives as a standalone function (not on Decomposition) since it's specific
/// to StructuredBlock, while the Decomposition/PartitionMesh types are generic.
pub fn decompose_structured<const D: usize>(
  mesh: Arc<StructuredBlock<D>>,
  dims: [usize; D],
  num_parts: usize,
  ghost_depth: usize,
) -> Decomposition<D, StructuredBlock<D>> {
  let split_axis = (0..D).max_by_key(|&d| dims[d]).unwrap();
  let n = dims[split_axis];

  let slab_size = n / num_parts;
  let remainder = n % num_parts;
  let boundaries: Vec<usize> = (0..=num_parts)
    .scan(0usize, |acc, p| {
      let prev = *acc;
      if p < num_parts {
        *acc += slab_size + if p < remainder { 1 } else { 0 };
      }
      Some(prev)
    })
    .collect();

  // Number of cells per split-axis position (product of all other dims)
  let cross_section: usize = (0..D)
    .filter(|&d| d != split_axis)
    .map(|d| dims[d])
    .product::<usize>()
    .max(1); // D=1 case

  let partitions = (0..num_parts)
    .map(|p| {
      let start = boundaries[p];
      let end = boundaries[p + 1];

      // Ghost range clamped to domain
      let ghost_left = start.saturating_sub(ghost_depth);
      let ghost_right = (end + ghost_depth).min(n);

      // --- Cell enumeration ---
      // Owned cells first, then left ghosts, then right ghosts.
      // Within each group: iterate split_axis position, then cross-section.

      let mut local_to_global_cell = Vec::new();
      let mut global_to_local_cell = HashMap::new();
      let mut ghost_descriptors = Vec::new();

      // Helper: convert (split_axis_pos, cross_section_idx) → global ijk → global CellId
      let to_global_cell = |sa: usize, cs: usize| -> CellId {
        let ijk = build_ijk::<D>(&dims, split_axis, sa, cs);
        CellId::from(StructuredBlock::<D>::cell_index(&dims, &ijk))
      };

      // Owned cells: split_axis in [start, end)
      for sa in start..end {
        for cs in 0..cross_section {
          let global = to_global_cell(sa, cs);
          let local = local_to_global_cell.len();
          global_to_local_cell.insert(global, local);
          local_to_global_cell.push(global);
        }
      }
      let num_owned = local_to_global_cell.len();

      // Left ghost cells: split_axis in [ghost_left, start)
      for sa in ghost_left..start {
        let source_partition = find_owning_partition(&boundaries, sa);
        let source_offset = sa - boundaries[source_partition];
        for cs in 0..cross_section {
          let global = to_global_cell(sa, cs);
          let local = local_to_global_cell.len();
          global_to_local_cell.insert(global, local);
          local_to_global_cell.push(global);
          ghost_descriptors.push(GhostDescriptor {
            local_cell: CellId::from(local),
            source_partition,
            source_local_cell: CellId::from(source_offset * cross_section + cs),
          });
        }
      }

      // Right ghost cells: split_axis in [end, ghost_right)
      for sa in end..ghost_right {
        let source_partition = find_owning_partition(&boundaries, sa);
        let source_offset = sa - boundaries[source_partition];
        for cs in 0..cross_section {
          let global = to_global_cell(sa, cs);
          let local = local_to_global_cell.len();
          global_to_local_cell.insert(global, local);
          local_to_global_cell.push(global);
          ghost_descriptors.push(GhostDescriptor {
            local_cell: CellId::from(local),
            source_partition,
            source_local_cell: CellId::from(source_offset * cross_section + cs),
          });
        }
      }

      // --- Face enumeration ---
      // Iterate all global faces. A face is relevant if at least one
      // of its adjacent cells is in our local set (owned or ghost).
      // Use the global mesh's face data to classify.

      let mut local_to_global_face = Vec::new();
      let mut face_connections = Vec::new();
      let mut cell_face_adj: Vec<Vec<FaceId>> =
        vec![Vec::new(); local_to_global_cell.len()];
      let mut interior_and_halo = Vec::new();
      let mut boundary_map: HashMap<BoundaryTag, Vec<(FaceId, CellId)>> =
        HashMap::new();

      // Iterate faces by axis, same order as StructuredBlock::build_topology
      let mut global_face = 0usize;
      for axis in 0..D {
        let face_count = StructuredBlock::<D>::face_count_for_axis(&dims, axis);
        for local_in_axis in 0..face_count {
          let face_ijk =
            StructuredBlock::<D>::face_indices(&dims, axis, local_in_axis);
          let gf = FaceId::from(global_face);

          // Determine which cells this face connects
          let conn = if face_ijk[axis] == 0 {
            // Min boundary
            let cell_global =
              CellId::from(StructuredBlock::<D>::cell_index(&dims, &face_ijk));
            FaceKind::Boundary {
              owner_global: cell_global,
              tag: StructuredBlock::<D>::boundary_tag(axis, 0),
              out_sign: -1.0,
            }
          } else if face_ijk[axis] == dims[axis] {
            // Max boundary
            let mut cell_ijk = face_ijk;
            cell_ijk[axis] = dims[axis] - 1;
            let cell_global =
              CellId::from(StructuredBlock::<D>::cell_index(&dims, &cell_ijk));
            FaceKind::Boundary {
              owner_global: cell_global,
              tag: StructuredBlock::<D>::boundary_tag(axis, 1),
              out_sign: 1.0,
            }
          } else {
            // Interior
            let mut owner_ijk = face_ijk;
            owner_ijk[axis] -= 1;
            let owner_global =
              CellId::from(StructuredBlock::<D>::cell_index(&dims, &owner_ijk));
            let neighbour_global =
              CellId::from(StructuredBlock::<D>::cell_index(&dims, &face_ijk));
            FaceKind::Interior {
              owner_global,
              neighbour_global,
            }
          };

          // Check if this face involves any local cells
          match conn {
            FaceKind::Interior {
              owner_global,
              neighbour_global,
            } => {
              let owner_local =
                global_to_local_cell.get(&owner_global).copied();
              let nbr_local =
                global_to_local_cell.get(&neighbour_global).copied();

              if let (Some(ol), Some(nl)) = (owner_local, nbr_local) {
                let lf = FaceId::from(local_to_global_face.len());
                local_to_global_face.push(gf);

                let ol_id = CellId::from(ol);
                let nl_id = CellId::from(nl);
                face_connections.push(FaceConnection::Interior {
                  owner: ol_id,
                  neighbour: nl_id,
                });
                interior_and_halo.push((lf, ol_id, nl_id));
                cell_face_adj[ol].push(lf);
                cell_face_adj[nl].push(lf);
              }
              // If neither cell is local, skip this face entirely
            }

            FaceKind::Boundary {
              owner_global,
              tag,
              out_sign,
            } => {
              if let Some(&ol) = global_to_local_cell.get(&owner_global) {
                // Only include if owner is OWNED (not ghost)
                if ol < num_owned {
                  let lf = FaceId::from(local_to_global_face.len());
                  local_to_global_face.push(gf);

                  let ol_id = CellId::from(ol);
                  face_connections.push(FaceConnection::Boundary {
                    owner: ol_id,
                    tag,
                    out_sign,
                  });
                  boundary_map.entry(tag).or_default().push((lf, ol_id));
                  cell_face_adj[ol].push(lf);
                }
              }
            }
          }

          global_face += 1;
        }
      }

      let boundary_face_lists: Vec<_> = boundary_map.into_iter().collect();

      PartitionMesh {
        mesh: Arc::clone(&mesh),
        local_to_global_cell,
        num_owned,
        local_to_global_face,
        ghost_cells: ghost_descriptors,
        face_connections,
        cell_face_adj,
        interior_and_halo_faces: interior_and_halo,
        boundary_face_lists,
      }
    })
    .collect();

  Decomposition { partitions }
}

// --- Helpers ---

/// Build a D-dimensional ijk index from a split-axis position and
/// a flattened index across all other dimensions.
fn build_ijk<const D: usize>(
  dims: &[usize; D],
  split_axis: usize,
  sa_pos: usize,
  cross_flat: usize,
) -> [usize; D] {
  let mut ijk = [0; D];
  ijk[split_axis] = sa_pos;
  let mut remaining = cross_flat;
  for d in 0..D {
    if d == split_axis {
      continue;
    }
    ijk[d] = remaining % dims[d];
    remaining /= dims[d];
  }
  ijk
}

/// Find which partition owns a given split-axis position.
fn find_owning_partition(boundaries: &[usize], sa_pos: usize) -> usize {
  // boundaries = [0, b1, b2, ..., n]
  // Partition p owns [boundaries[p], boundaries[p+1])
  boundaries
    .windows(2)
    .position(|w| sa_pos >= w[0] && sa_pos < w[1])
    .unwrap()
}

/// Temporary enum for face classification during decomposition
enum FaceKind {
  Interior {
    owner_global: CellId,
    neighbour_global: CellId,
  },
  Boundary {
    owner_global: CellId,
    tag: BoundaryTag,
    out_sign: f64,
  },
}
