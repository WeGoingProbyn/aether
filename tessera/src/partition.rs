// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use std::{
  collections::{HashMap, HashSet},
  sync::Arc,
};

use utility::{
  domain::{BoundaryTag, CellId, FaceId, Point},
  maths::vector::Vector,
};

use crate::{
  cube_sphere::CubeSphere,
  geometry::{CellGeometry, CellMetrics, FaceGeometry, FaceMetrics},
  mesh::{Mesh, StructuredBlock},
  topology::{FaceConnection, Topology},
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
  M: CellGeometry<D> + FaceGeometry<D> + Topology + ?Sized,
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
  M: CellGeometry<D> + FaceGeometry<D> + Topology + ?Sized,
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
  M: CellGeometry<D> + FaceGeometry<D> + Topology + ?Sized,
{
  fn cell_centroid(&self, cell: CellId) -> &Point<D> {
    let global = self.local_to_global_cell[cell.index()];
    self.mesh.cell_centroid(global)
  }

  fn cell_world_centroid(&self, cell: CellId) -> Point<D> {
    // Forward to the underlying mesh's (possibly overridden) world mapping —
    // the default would return the *computational* centroid, breaking any
    // caller that needs real world positions (e.g. HEVI's radial direction).
    let global = self.local_to_global_cell[cell.index()];
    self.mesh.cell_world_centroid(global)
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
  M: CellGeometry<D> + FaceGeometry<D> + Topology + ?Sized,
{
  fn face_centroid(&self, face: FaceId) -> &Point<D> {
    let global = self.local_to_global_face[face.index()];
    self.mesh.face_centroid(global)
  }

  fn face_world_centroid(&self, face: FaceId) -> Point<D> {
    let global = self.local_to_global_face[face.index()];
    self.mesh.face_world_centroid(global)
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
  M: CellGeometry<D> + FaceGeometry<D> + Topology + ?Sized,
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
  M: CellGeometry<D> + FaceGeometry<D> + Topology + ?Sized,
{
  pub partitions: Vec<PartitionMesh<D, M>>,
}

impl<const D: usize, M> Decomposition<D, M>
where
  M: CellGeometry<D> + FaceGeometry<D> + Topology + ?Sized,
{
  /// All ghost descriptors, grouped per partition. Pleroma uses these to
  /// drive the actual field-data exchange — tessera owns the topology, not
  /// the field storage.
  pub fn ghost_descriptors_per_partition(&self) -> Vec<&[GhostDescriptor]> {
    self.partitions.iter().map(|p| p.ghost_cells()).collect()
  }
}

/// Decompose a mesh from explicit owned-cell sets, adding a one-face ghost
/// layer around partition boundaries.
///
/// Owned cells are kept in caller-provided order and ghost cells are appended
/// after them. Ghost descriptors point back to the partition/local owned cell
/// that supplies each ghost value.
pub fn decompose_by_owned_cells<const D: usize, M>(
  mesh: Arc<M>,
  owned_cells: Vec<Vec<CellId>>,
) -> Decomposition<D, M>
where
  M: CellGeometry<D> + FaceGeometry<D> + Topology + ?Sized,
{
  assert!(
    !owned_cells.is_empty(),
    "decomposition must contain at least one partition"
  );

  let mut owner: HashMap<CellId, (usize, CellId)> = HashMap::new();
  for (partition, cells) in owned_cells.iter().enumerate() {
    for (local, &cell) in cells.iter().enumerate() {
      assert!(
        owner
          .insert(cell, (partition, CellId::from(local)))
          .is_none(),
        "cell {} appears in more than one partition",
        cell.index(),
      );
    }
  }

  assert_eq!(
    owner.len(),
    mesh.cell_count(),
    "owned cell sets must cover the whole mesh exactly once",
  );

  let mut ghost_cells: Vec<Vec<CellId>> = vec![Vec::new(); owned_cells.len()];
  let mut ghost_seen: Vec<HashSet<CellId>> =
    vec![HashSet::new(); owned_cells.len()];

  for &(_, owner_cell, neighbour_cell) in mesh.interior_faces() {
    let Some(&(owner_partition, _)) = owner.get(&owner_cell) else {
      panic!("interior face owner cell is not owned by any partition");
    };
    let Some(&(neighbour_partition, _)) = owner.get(&neighbour_cell) else {
      panic!("interior face neighbour cell is not owned by any partition");
    };

    if owner_partition == neighbour_partition {
      continue;
    }

    if ghost_seen[owner_partition].insert(neighbour_cell) {
      ghost_cells[owner_partition].push(neighbour_cell);
    }
    if ghost_seen[neighbour_partition].insert(owner_cell) {
      ghost_cells[neighbour_partition].push(owner_cell);
    }
  }

  let partitions = owned_cells
    .into_iter()
    .enumerate()
    .map(|(partition, owned)| {
      let ghosts = ghost_cells[partition]
        .iter()
        .map(|&global| {
          let &(source_partition, source_local_cell) =
            owner.get(&global).unwrap();
          (global, source_partition, source_local_cell)
        })
        .collect();
      build_partition_mesh_from_cells(Arc::clone(&mesh), owned, ghosts)
    })
    .collect();

  Decomposition { partitions }
}

/// Six-way cubed-sphere decomposition with one owned partition per panel.
///
/// Returns a mesh-type-erased `Decomposition<3, dyn Mesh<3>>` so the partitioned
/// solver reads a single decomposition type regardless of whether the source mesh
/// is a plain [`CubeSphere`] or an adapted wrapper around one (see
/// [`decompose_panels`]).
pub fn decompose_cube_sphere_panels(
  mesh: Arc<CubeSphere>,
) -> Decomposition<3, dyn Mesh<3>> {
  let cells_per_panel = mesh.dims().iter().product::<usize>();
  let owned_cells = (0..6)
    .map(|panel| {
      (0..cells_per_panel)
        .map(|local| CellId::from(panel * cells_per_panel + local))
        .collect::<Vec<_>>()
    })
    .collect();

  let mesh: Arc<dyn Mesh<3>> = mesh;
  decompose_by_owned_cells(mesh, owned_cells)
}

/// Decompose any 3-D mesh into `panel_count` partitions by grouping each cell
/// under the panel returned by `panel_of`. The mesh-agnostic counterpart of
/// [`decompose_cube_sphere_panels`]: an
/// [`AdaptiveMesh`](crate::adaptive::AdaptiveMesh) supplies a `panel_of` that maps
/// each leaf to its base cell's panel, so a refined cube-sphere stays partitioned
/// by panel (uneven counts after refinement — load balancing is a later concern).
/// Empty panels are dropped so every partition owns at least one cell.
pub fn decompose_panels(
  mesh: Arc<dyn Mesh<3>>,
  panel_count: usize,
  panel_of: impl Fn(CellId) -> usize,
) -> Decomposition<3, dyn Mesh<3>> {
  let mut owned_cells: Vec<Vec<CellId>> = vec![Vec::new(); panel_count];
  for cell in 0..mesh.cell_count() {
    let cell = CellId::from(cell);
    owned_cells[panel_of(cell)].push(cell);
  }
  owned_cells.retain(|cells| !cells.is_empty());
  decompose_by_owned_cells(mesh, owned_cells)
}

fn build_partition_mesh_from_cells<const D: usize, M>(
  mesh: Arc<M>,
  owned_cells: Vec<CellId>,
  ghost_cells: Vec<(CellId, usize, CellId)>,
) -> PartitionMesh<D, M>
where
  M: CellGeometry<D> + FaceGeometry<D> + Topology + ?Sized,
{
  let num_owned = owned_cells.len();
  let mut local_to_global_cell = owned_cells;
  let mut ghost_descriptors = Vec::with_capacity(ghost_cells.len());

  for (global, source_partition, source_local_cell) in ghost_cells {
    let local_cell = CellId::from(local_to_global_cell.len());
    local_to_global_cell.push(global);
    ghost_descriptors.push(GhostDescriptor {
      local_cell,
      source_partition,
      source_local_cell,
    });
  }

  let global_to_local_cell: HashMap<CellId, usize> = local_to_global_cell
    .iter()
    .enumerate()
    .map(|(local, &global)| (global, local))
    .collect();

  let mut local_to_global_face = Vec::new();
  let mut face_connections = Vec::new();
  let mut cell_face_adj: Vec<Vec<FaceId>> =
    vec![Vec::new(); local_to_global_cell.len()];
  let mut interior_and_halo = Vec::new();
  let mut boundary_map: HashMap<BoundaryTag, Vec<(FaceId, CellId)>> =
    HashMap::new();

  for &(global_face, owner_global, neighbour_global) in mesh.interior_faces() {
    let owner_local = global_to_local_cell.get(&owner_global).copied();
    let neighbour_local = global_to_local_cell.get(&neighbour_global).copied();

    if let (Some(owner_local), Some(neighbour_local)) =
      (owner_local, neighbour_local)
    {
      let local_face = FaceId::from(local_to_global_face.len());
      let owner = CellId::from(owner_local);
      let neighbour = CellId::from(neighbour_local);

      local_to_global_face.push(global_face);
      face_connections.push(FaceConnection::Interior { owner, neighbour });
      interior_and_halo.push((local_face, owner, neighbour));
      cell_face_adj[owner_local].push(local_face);
      cell_face_adj[neighbour_local].push(local_face);
    }
  }

  for tag in mesh.boundary_tags() {
    for &(global_face, owner_global) in mesh.boundary_faces(tag) {
      let Some(&owner_local) = global_to_local_cell.get(&owner_global) else {
        continue;
      };

      if owner_local >= num_owned {
        continue;
      }

      let out_sign = match mesh.face_connection(global_face) {
        FaceConnection::Boundary { out_sign, .. } => *out_sign,
        _ => unreachable!(),
      };
      let local_face = FaceId::from(local_to_global_face.len());
      let owner = CellId::from(owner_local);

      local_to_global_face.push(global_face);
      face_connections.push(FaceConnection::Boundary {
        owner,
        tag,
        out_sign,
      });
      boundary_map
        .entry(tag)
        .or_default()
        .push((local_face, owner));
      cell_face_adj[owner_local].push(local_face);
    }
  }

  PartitionMesh {
    mesh,
    local_to_global_cell,
    num_owned,
    local_to_global_face,
    ghost_cells: ghost_descriptors,
    face_connections,
    cell_face_adj,
    interior_and_halo_faces: interior_and_halo,
    boundary_face_lists: boundary_map.into_iter().collect(),
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
