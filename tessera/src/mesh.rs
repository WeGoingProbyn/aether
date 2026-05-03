// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use utility::{
  domain::{BoundaryTag, CellId, FaceId, Point},
  maths::vector::Vector,
  profile,
};

use crate::{
  geometry::{
    CellGeometry, CellMetrics, FaceGeometry, FaceMetrics, GeometryMap,
  },
  topology::{FaceConnection, Topology},
};

pub trait Mesh<const D: usize>:
  CellGeometry<D> + FaceGeometry<D> + Topology
{
}

impl<const D: usize, T> Mesh<D> for T where
  T: CellGeometry<D> + FaceGeometry<D> + Topology
{
}

pub struct StructuredBlock<const D: usize> {
  dims: [usize; D], // x, y, z

  // Geometry - per cell
  cell_centroids: Vec<Point<D>>,
  cell_volumes: Vec<f64>,
  cell_metrics: Vec<CellMetrics<D>>,

  // Geometry - per face
  face_centroids: Vec<Point<D>>,
  face_area_vectors: Vec<Vector<f64, D>>,
  face_areas: Vec<f64>,
  face_metrics: Vec<FaceMetrics<D>>,

  // Topology - pre-built lists
  face_connections: Vec<FaceConnection>,
  cell_face_adj: Vec<Vec<FaceId>>,
  interior_face_list: Vec<(FaceId, CellId, CellId)>,
  boundary_face_lists: Vec<(BoundaryTag, Vec<(FaceId, CellId)>)>,
  // The coordinate mapping
  // coord_map: Box<dyn GeometryMap<D, P>>,
}

type Topo = (
  Vec<FaceConnection>,
  Vec<(FaceId, CellId, CellId)>,
  Vec<(BoundaryTag, Vec<(FaceId, CellId)>)>,
  Vec<Vec<FaceId>>,
);

impl<const D: usize> StructuredBlock<D> {
  // ---- Counts ----
  fn total_cells(dims: &[usize; D]) -> usize {
    dims.iter().product()
  }

  pub fn face_count_for_axis(dims: &[usize; D], axis: usize) -> usize {
    (0..D)
      .map(|d| if d == axis { dims[d] + 1 } else { dims[d] })
      .product()
  }

  fn total_faces(dims: &[usize; D]) -> usize {
    (0..D).map(|a| Self::face_count_for_axis(dims, a)).sum()
  }

  // ---- Index conversions ----
  fn cell_indices(dims: &[usize; D], index: usize) -> [usize; D] {
    let mut ijk = [0; D];
    let mut remaining = index;
    for d in 0..D {
      ijk[d] = remaining % dims[d];
      remaining /= dims[d];
    }
    ijk
  }

  pub fn cell_index(dims: &[usize; D], ijk: &[usize; D]) -> usize {
    let mut idx = 0;
    let mut stride = 1;
    for d in 0..D {
      idx += ijk[d] * stride;
      stride *= dims[d];
    }
    idx
  }

  // Face ijk within an axis's face set. Along `axis`, size is dims[axis]+1; others are dims[d]
  pub fn face_indices(
    dims: &[usize; D],
    axis: usize,
    local: usize,
  ) -> [usize; D] {
    let mut ijk = [0; D];
    let mut remaining = local;
    for d in 0..D {
      let size = if d == axis { dims[d] + 1 } else { dims[d] };
      ijk[d] = remaining % size;
      remaining /= size;
    }
    ijk
  }

  pub fn boundary_tag(axis: usize, side: usize) -> BoundaryTag {
    match (axis, side) {
      (0, 0) => BoundaryTag::Left,
      (0, 1) => BoundaryTag::Right,
      (1, 0) => BoundaryTag::Bottom,
      (1, 1) => BoundaryTag::Top,
      (2, 0) => BoundaryTag::Front,
      (2, 1) => BoundaryTag::Back,
      _ => unreachable!(),
    }
  }

  // ---- Builders ----
  #[profile]
  fn build_cell_geometry(
    axis_edges: &[Vec<f64>; D],
    dims: &[usize; D],
  ) -> (Vec<Point<D>>, Vec<f64>) {
    let count = Self::total_cells(dims);

    let mut centroids = Vec::with_capacity(count);
    let mut volumes = Vec::with_capacity(count);

    for flat in 0..count {
      let ijk = Self::cell_indices(dims, flat);
      let pos: [f64; D] = std::array::from_fn(|d| {
        let lo = axis_edges[d][ijk[d]];
        let hi = axis_edges[d][ijk[d] + 1];
        0.5 * (lo + hi)
      });
      let spacing: [f64; D] = std::array::from_fn(|d| {
        axis_edges[d][ijk[d] + 1] - axis_edges[d][ijk[d]]
      });
      let vol: f64 = spacing.iter().product();
      centroids.push(Vector::from(pos));
      volumes.push(vol);
    }

    (centroids, volumes)
  }

  #[profile]
  fn build_face_geometry(
    axis_edges: &[Vec<f64>; D],
    dims: &[usize; D],
  ) -> (Vec<Point<D>>, Vec<Vector<f64, D>>, Vec<f64>) {
    let total = Self::total_faces(dims);

    let mut centroids = Vec::with_capacity(total);
    let mut area_vectors = Vec::with_capacity(total);
    let mut areas = Vec::with_capacity(total);

    for axis in 0..D {
      let count = Self::face_count_for_axis(dims, axis);
      for local in 0..count {
        let ijk = Self::face_indices(dims, axis, local);

        // Face sits on a vertex along its own axis, cell-center along others.
        let pos: [f64; D] = std::array::from_fn(|d| {
          if d == axis {
            axis_edges[d][ijk[d]]
          } else {
            0.5 * (axis_edges[d][ijk[d]] + axis_edges[d][ijk[d] + 1])
          }
        });
        centroids.push(Vector::from(pos));

        // Face area = product of cell extents along the OTHER axes at this
        // face's position. Varies per face when those axes are non-uniform.
        let area: f64 = (0..D)
          .filter(|&d| d != axis)
          .map(|d| axis_edges[d][ijk[d] + 1] - axis_edges[d][ijk[d]])
          .product();

        // Normal points in +axis direction.
        let mut av = [0.0; D];
        av[axis] = area;
        area_vectors.push(Vector::from(av));

        areas.push(area);
      }
    }

    (centroids, area_vectors, areas)
  }

  #[profile]
  fn build_topology(dims: &[usize; D]) -> Topo {
    let total_faces = Self::total_faces(dims);
    let total_cells = Self::total_cells(dims);

    let mut connections = Vec::with_capacity(total_faces);
    let mut interior = Vec::new();
    let mut cell_faces = vec![Vec::new(); total_cells];

    // One boundary list per (axis, side) pair
    let mut boundary_map: Vec<(BoundaryTag, Vec<(FaceId, CellId)>)> = (0..D)
      .flat_map(|axis| {
        (0..2).map(move |side| (Self::boundary_tag(axis, side), Vec::new()))
      })
      .collect();

    let mut global_face = 0usize;

    for axis in 0..D {
      let count = Self::face_count_for_axis(dims, axis);
      for local in 0..count {
        let ijk = Self::face_indices(dims, axis, local);
        let face = FaceId::from(global_face);

        if ijk[axis] == 0 {
          // Min boundary along this axis
          let owner = CellId::from(Self::cell_index(dims, &ijk));
          let tag = Self::boundary_tag(axis, 0);
          connections.push(FaceConnection::Boundary {
            owner,
            tag,
            out_sign: -1.0,
          });
          boundary_map[axis * 2].1.push((face, owner));
          cell_faces[owner.index()].push(face);
        } else if ijk[axis] == dims[axis] {
          // Max boundary along this axis
          let mut cell_ijk = ijk;
          cell_ijk[axis] = dims[axis] - 1;
          let owner = CellId::from(Self::cell_index(dims, &cell_ijk));
          let tag = Self::boundary_tag(axis, 1);
          connections.push(FaceConnection::Boundary {
            owner,
            tag,
            out_sign: 1.0,
          });
          boundary_map[axis * 2 + 1].1.push((face, owner));
          cell_faces[owner.index()].push(face);
        } else {
          // Interior face between cell at (ijk[axis]-1) and cell at (ijk[axis])
          let mut owner_ijk = ijk;
          owner_ijk[axis] -= 1;
          let owner = CellId::from(Self::cell_index(dims, &owner_ijk));
          let neighbour = CellId::from(Self::cell_index(dims, &ijk));
          connections.push(FaceConnection::Interior { owner, neighbour });
          interior.push((face, owner, neighbour));
          cell_faces[owner.index()].push(face);
          cell_faces[neighbour.index()].push(face);
        }

        global_face += 1;
      }
    }

    (connections, interior, boundary_map, cell_faces)
  }

  #[profile]
  fn compute_metrics<const P: usize>(
    cell_centroids: &[Point<D>],
    cell_volumes: &[f64],
    face_centroids: &[Point<D>],
    face_area_vectors: &[Vector<f64, D>],
    face_areas: &[f64],
    coord_map: &dyn GeometryMap<D, P>,
  ) -> (Vec<CellMetrics<D>>, Vec<FaceMetrics<D>>) {
    let cell_metrics = cell_centroids
      .iter()
      .zip(cell_volumes.iter())
      .map(|(centroid, &vol)| {
        let sqrt_g = coord_map.sqrt_det_metric(centroid);
        CellMetrics {
          sqrt_metric: sqrt_g,
          comp_volume: vol,
          phys_volume: vol * sqrt_g,
        }
      })
      .collect();

    let face_metrics = face_centroids
      .iter()
      .zip(face_area_vectors.iter())
      .zip(face_areas.iter())
      .map(|((centroid, area_vec), &area)| {
        let sqrt_g = coord_map.sqrt_det_metric(centroid);
        let normal = area_vec / &area;
        FaceMetrics {
          normal,
          comp_area: area,
          phys_area: area * sqrt_g,
          sqrt_metric: sqrt_g,
        }
      })
      .collect();

    (cell_metrics, face_metrics)
  }

  // ---- Constructors ----

  /// Build a block from per-axis edge positions. `axis_edges[d]` lists the
  /// `dims[d] + 1` cell-edge positions along axis `d`, in strictly increasing
  /// order. Allows non-uniform cell widths along any axis (e.g. atmospheric
  /// stretching toward the surface).
  #[profile]
  pub fn from_axis_edges<const P: usize>(
    axis_edges: [Vec<f64>; D],
    coord_map: Box<dyn GeometryMap<D, P>>,
  ) -> Self {
    let dims: [usize; D] = std::array::from_fn(|d| {
      assert!(
        axis_edges[d].len() >= 2,
        "axis {} needs at least two edges (got {})",
        d,
        axis_edges[d].len()
      );
      for i in 1..axis_edges[d].len() {
        assert!(
          axis_edges[d][i] > axis_edges[d][i - 1],
          "axis {} edges must be strictly increasing",
          d
        );
      }
      axis_edges[d].len() - 1
    });

    let (cell_centroids, cell_volumes) =
      Self::build_cell_geometry(&axis_edges, &dims);
    let (face_centroids, face_area_vectors, face_areas) =
      Self::build_face_geometry(&axis_edges, &dims);
    let (
      face_connections,
      interior_face_list,
      boundary_face_lists,
      cell_face_adj,
    ) = Self::build_topology(&dims);
    let (cell_metrics, face_metrics) = Self::compute_metrics(
      &cell_centroids,
      &cell_volumes,
      &face_centroids,
      &face_area_vectors,
      &face_areas,
      coord_map.as_ref(),
    );

    StructuredBlock {
      dims,
      cell_centroids,
      cell_volumes,
      cell_metrics,
      face_centroids,
      face_area_vectors,
      face_areas,
      face_metrics,
      face_connections,
      cell_face_adj,
      interior_face_list,
      boundary_face_lists,
    }
  }

  /// Convenience constructor with uniform spacing along every axis. Equivalent
  /// to calling `from_axis_edges` with arithmetically-spaced edges.
  pub fn uniform<const P: usize>(
    origin: Point<D>,
    extent: [f64; D],
    dims: [usize; D],
    coord_map: Box<dyn GeometryMap<D, P>>,
  ) -> Self {
    let axis_edges: [Vec<f64>; D] = std::array::from_fn(|d| {
      let dx = extent[d] / dims[d] as f64;
      (0..=dims[d]).map(|i| origin[d] + i as f64 * dx).collect()
    });
    Self::from_axis_edges(axis_edges, coord_map)
  }

  pub fn dims(&self) -> &[usize; D] {
    &self.dims
  }
}

impl<const D: usize> CellGeometry<D> for StructuredBlock<D> {
  fn cell_centroid(&self, cell: CellId) -> &Point<D> {
    &self.cell_centroids[cell.index()]
  }

  fn cell_volume(&self, cell: CellId) -> f64 {
    self.cell_volumes[cell.index()]
  }

  fn cell_count(&self) -> usize {
    self.cell_centroids.len()
  }

  fn cell_metrics(&self, cell: CellId) -> &CellMetrics<D> {
    &self.cell_metrics[cell.index()]
  }
}

impl<const D: usize> FaceGeometry<D> for StructuredBlock<D> {
  fn face_centroid(&self, face: FaceId) -> &Point<D> {
    &self.face_centroids[face.index()]
  }

  fn face_area_vector(&self, face: FaceId) -> Vector<f64, D> {
    self.face_area_vectors[face.index()].clone()
  }

  fn face_area(&self, face: FaceId) -> f64 {
    self.face_areas[face.index()]
  }

  fn face_count(&self) -> usize {
    self.face_centroids.len()
  }

  fn face_metrics(&self, face: FaceId) -> &FaceMetrics<D> {
    &self.face_metrics[face.index()]
  }
}

impl<const D: usize> Topology for StructuredBlock<D> {
  fn face_connection(&self, face: FaceId) -> &FaceConnection {
    &self.face_connections[face.index()]
  }

  fn cell_faces(&self, cell: CellId) -> &[FaceId] {
    &self.cell_face_adj[cell.index()]
  }

  // fn face_count(&self) -> usize {
  //   self.face_connections.len()
  // }
  //
  // fn cell_count(&self) -> usize {
  //   self.cell_centroids.len()
  // }

  fn interior_faces(&self) -> &[(FaceId, CellId, CellId)] {
    &self.interior_face_list
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
