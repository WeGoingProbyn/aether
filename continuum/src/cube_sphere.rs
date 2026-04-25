use std::collections::HashMap;

use utility::maths::{matrix::Matrix, vector::Vector};

use crate::{
  geometry::{
    CellGeometry, CellId, CellMetrics, FaceGeometry, FaceId, FaceMetrics,
    GeometryMap, Point,
  },
  mesh::StructuredBlock,
  topology::{BoundaryTag, FaceConnection, Topology},
};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum PanelId {
  XP,
  XN,
  YP,
  YN,
  ZP,
  ZN
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Edge {
  North,
  South,
  East,
  West
}

impl Edge {
  /// Computational midpoint of this edge in (ξ, η).
  /// North/South pin η; East/West pin ξ.
  pub fn midpoint_comp(self) -> Point<2> {
    let b = std::f64::consts::FRAC_PI_4;
    match self {
      Edge::North => [0.0,  b].into(),
      Edge::South => [0.0, -b].into(),
      Edge::East  => [ b, 0.0].into(),
      Edge::West  => [-b, 0.0].into(),
    }
  }
}

/// The 12 edges of the cube, each shared by exactly two panels.
/// Pairing depends on the axis convention in `panel_axes`; if that table
/// changes, the verification test will catch any mismatch here.
pub const CUBE_EDGES: [(PanelId, Edge, PanelId, Edge); 12] = [
  // Top ring (shared with ZP).
  (PanelId::ZP, Edge::East,  PanelId::XP, Edge::North),
  (PanelId::ZP, Edge::West,  PanelId::XN, Edge::North),
  (PanelId::ZP, Edge::North, PanelId::YP, Edge::North),
  (PanelId::ZP, Edge::South, PanelId::YN, Edge::North),
  // Bottom ring (shared with ZN).
  (PanelId::ZN, Edge::East,  PanelId::XP, Edge::South),
  (PanelId::ZN, Edge::West,  PanelId::XN, Edge::South),
  (PanelId::ZN, Edge::North, PanelId::YN, Edge::South),
  (PanelId::ZN, Edge::South, PanelId::YP, Edge::South),
  // Vertical ring (between X and Y panels).
  (PanelId::XP, Edge::East,  PanelId::YP, Edge::West),
  (PanelId::XP, Edge::West,  PanelId::YN, Edge::East),
  (PanelId::XN, Edge::East,  PanelId::YN, Edge::West),
  (PanelId::XN, Edge::West,  PanelId::YP, Edge::East),
];

/// Number of cells along the given edge, given a 2D panel resolution.
pub fn cells_along_edge(edge: Edge, dims: [usize; 2]) -> usize {
  match edge {
    Edge::East | Edge::West => dims[1],
    Edge::North | Edge::South => dims[0],
  }
}

/// Local cell index (i + j·Nx) of the k-th cell along `edge` for a panel
/// with `dims = [Nx, Ny]`.
pub fn edge_cell_index(edge: Edge, dims: [usize; 2], k: usize) -> CellId {
  let nx = dims[0];
  let ny = dims[1];
  let (i, j) = match edge {
    Edge::East  => (nx - 1, k),
    Edge::West  => (0,      k),
    Edge::North => (k,      ny - 1),
    Edge::South => (k,      0),
  };
  CellId::from(i + j * nx)
}

/// Computational (ξ, η) of the face centroid of the k-th boundary face along
/// `edge`. Cells are uniformly spaced over [-π/4, π/4]² in both directions.
pub fn edge_face_centroid_comp(edge: Edge, dims: [usize; 2], k: usize) -> Point<2> {
  let bound = std::f64::consts::FRAC_PI_4;
  let dxi   = 2.0 * bound / dims[0] as f64;
  let deta  = 2.0 * bound / dims[1] as f64;
  match edge {
    Edge::East  => [ bound, -bound + (k as f64 + 0.5) * deta].into(),
    Edge::West  => [-bound, -bound + (k as f64 + 0.5) * deta].into(),
    Edge::North => [-bound + (k as f64 + 0.5) * dxi,  bound].into(),
    Edge::South => [-bound + (k as f64 + 0.5) * dxi, -bound].into(),
  }
}

/// Pair every boundary cell on `(panel_a, edge_a)` with its world-space mate
/// on `(panel_b, edge_b)` by nearest-face-centroid matching. Both edges must
/// have the same number of cells along them — gnomonic edges coincide
/// point-by-point in world space at equal resolution, so the match is exact
/// up to floating-point noise.
pub fn match_edge_cells(
  panel_a: &GnomonicPanel,
  edge_a: Edge,
  dims_a: [usize; 2],
  panel_b: &GnomonicPanel,
  edge_b: Edge,
  dims_b: [usize; 2],
) -> Vec<(CellId, CellId)> {
  let n_a = cells_along_edge(edge_a, dims_a);
  let n_b = cells_along_edge(edge_b, dims_b);
  assert_eq!(n_a, n_b, "edge resolutions must match");

  let world_a: Vec<(CellId, Point<3>)> = (0..n_a)
    .map(|k| (
      edge_cell_index(edge_a, dims_a, k),
      panel_a.to_physical(&edge_face_centroid_comp(edge_a, dims_a, k)),
    ))
    .collect();

  let world_b: Vec<(CellId, Point<3>)> = (0..n_b)
    .map(|k| (
      edge_cell_index(edge_b, dims_b, k),
      panel_b.to_physical(&edge_face_centroid_comp(edge_b, dims_b, k)),
    ))
    .collect();

  world_a.iter().map(|(cell_a, pa)| {
    let (cell_b, _) = world_b.iter()
      .min_by(|(_, q1), (_, q2)| {
        pa.distance(q1).partial_cmp(&pa.distance(q2)).unwrap()
      })
      .unwrap();
    (*cell_a, *cell_b)
  }).collect()
}

/// (ξ, η) ∈ [-π/4, π/4]²
pub struct GnomonicPanel {
  panel: PanelId,
  rotation: Matrix<f64, 3, 3>,
  radius: f64,
}

impl GnomonicPanel {
  pub fn new(panel: PanelId, radius: f64) -> GnomonicPanel {
    GnomonicPanel {
      panel,
      rotation: panel_axes(panel),
      radius
    }
  }
}

/// Rotation that takes a panel-local frame vector (ξ̂, η̂, outward) into world
/// xyz. Stored as rows so that `v_local_row * rotation` gives the world-frame
/// vector (this codebase's `Vector` is a row vector).
pub fn panel_axes(panel: PanelId) -> Matrix<f64, 3, 3> {
  let (xi, eta, n): ([f64; 3], [f64; 3], [f64; 3]) = match panel {
    PanelId::ZP => ([1.0, 0.0, 0.0], [0.0,  1.0, 0.0], [0.0, 0.0,  1.0]),
    PanelId::ZN => ([1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]),
    PanelId::XP => ([0.0, 1.0, 0.0], [0.0, 0.0,  1.0], [ 1.0, 0.0, 0.0]),
    PanelId::XN => ([0.0,-1.0, 0.0], [0.0, 0.0,  1.0], [-1.0, 0.0, 0.0]),
    PanelId::YP => ([-1.0,0.0, 0.0], [0.0, 0.0,  1.0], [0.0,  1.0, 0.0]),
    PanelId::YN => ([ 1.0,0.0, 0.0], [0.0, 0.0,  1.0], [0.0, -1.0, 0.0]),
  };
  [xi, eta, n].into()
}

impl GeometryMap<2, 3> for GnomonicPanel {
  fn to_physical(&self, comp: &Point<2>) -> Point<3> {
    let tan_zeta: f64 = comp[0].tan();
    let tan_eta: f64 = comp[1].tan();
    let g: f64 = 1f64 + tan_zeta.powi(2) + tan_eta.powi(2);
    let numerator: Vector<f64, 3> = [tan_zeta, tan_eta, 1f64].into();
    let v_local: Vector<f64, 3> = numerator / g.sqrt();
    (&v_local * &self.rotation) * self.radius
  }

  fn to_computational(&self, physical: &Point<3>) -> Option<Point<2>> {
    let local: Vector<f64, 3> = physical * &self.rotation.transpose();
    if local[2] <= 0f64 { return None; }

    let xi: f64 = (local[0] / local[2]).atan();
    let eta: f64 = (local[1] / local[2]).atan();

    let bound: f64 = std::f64::consts::FRAC_PI_4;
    if xi.abs() > bound || eta.abs() > bound { return None; }

    Some([xi, eta].into())
    
  }

  /// ∂v_local/∂ξ = (sec²ξ / g^(3/2)) · ( 1 + tan²η,  -tan ξ · tan η,  -tan ξ)
  /// ∂v_local/∂η = (sec²η / g^(3/2)) · (-tan ξ · tan η,  1 + tan²ξ,  -tan η)  
  fn jacobian(&self, comp: &Point<2>) -> Matrix<f64, 2, 3> {
    unimplemented!()
  }

  ///sqrt(det g) = R² · sec²ξ · sec²η / (1 + tan²ξ + tan²η)^(3/2)
  fn sqrt_det_metric(&self, comp: &Point<2>) -> f64 {
    let sec2_xi = comp[0].cos().powi(-2);
    let sec2_eta = comp[1].cos().powi(-2);
    let g = 1.0 + comp[0].tan().powi(2) + comp[1].tan().powi(2);
    (self.radius.powi(2) * sec2_xi * sec2_eta) / (g * g.sqrt())
  }
}

/// One panel of a 3D spherical shell. Computational coords are
/// (ξ, η, r) ∈ [-π/4, π/4]² × [r_inner, r_outer]; physical is (x, y, z).
/// The radial coord is just the radial distance — no projection involved.
pub struct GnomonicShellPanel {
  panel: PanelId,
  rotation: Matrix<f64, 3, 3>,
}

impl GnomonicShellPanel {
  pub fn new(panel: PanelId) -> GnomonicShellPanel {
    GnomonicShellPanel { panel, rotation: panel_axes(panel) }
  }
}

impl GeometryMap<3, 3> for GnomonicShellPanel {
  fn to_physical(&self, comp: &Point<3>) -> Point<3> {
    let tan_xi = comp[0].tan();
    let tan_eta = comp[1].tan();
    let r = comp[2];
    let g = 1.0 + tan_xi.powi(2) + tan_eta.powi(2);
    let v_local: Vector<f64, 3> = [tan_xi, tan_eta, 1.0].into();
    let v_unit = v_local / g.sqrt();
    (&v_unit * &self.rotation) * r
  }

  fn to_computational(&self, physical: &Point<3>) -> Option<Point<3>> {
    let local: Vector<f64, 3> = physical * &self.rotation.transpose();
    if local[2] <= 0.0 { return None; }

    let xi = (local[0] / local[2]).atan();
    let eta = (local[1] / local[2]).atan();
    let bound = std::f64::consts::FRAC_PI_4;
    if xi.abs() > bound || eta.abs() > bound { return None; }

    let r = physical.magnitude();
    Some([xi, eta, r].into())
  }

  /// Columns are ∂world/∂ξ, ∂world/∂η, ∂world/∂r.
  ///
  /// Local-frame partials (before applying panel rotation):
  ///   ∂v_local/∂ξ = (sec²ξ / g^(3/2)) · ( 1 + tan²η,  -tan ξ · tan η,  -tan ξ)
  ///   ∂v_local/∂η = (sec²η / g^(3/2)) · (-tan ξ · tan η,  1 + tan²ξ,  -tan η)
  ///   ∂v_local/∂r = v_local
  /// Then ∂world/∂ξ = r · (∂v_local/∂ξ · rotation), likewise for η,
  /// and ∂world/∂r = v_local · rotation (no r factor).
  fn jacobian(&self, comp: &Point<3>) -> Matrix<f64, 3, 3> {
    let s = comp[0].tan();
    let t = comp[1].tan();
    let r = comp[2];
    let g = 1.0 + s * s + t * t;
    let g32 = g * g.sqrt();
    let sec2_xi = comp[0].cos().powi(-2);
    let sec2_eta = comp[1].cos().powi(-2);

    let dv_dxi: Vector<f64, 3> = [
      sec2_xi * (1.0 + t * t) / g32,
      -sec2_xi * s * t / g32,
      -sec2_xi * s / g32,
    ].into();

    let dv_deta: Vector<f64, 3> = [
      -sec2_eta * s * t / g32,
      sec2_eta * (1.0 + s * s) / g32,
      -sec2_eta * t / g32,
    ].into();

    let v_unit: Vector<f64, 3> =
      [s / g.sqrt(), t / g.sqrt(), 1.0 / g.sqrt()].into();

    let dw_dxi  = (&dv_dxi  * &self.rotation) * r;
    let dw_deta = (&dv_deta * &self.rotation) * r;
    let dw_dr   = &v_unit   * &self.rotation;

    // Pack columns into a Matrix<f64, 3, 3> (row-major: inner[r][c]).
    [
      [dw_dxi[0], dw_deta[0], dw_dr[0]],
      [dw_dxi[1], dw_deta[1], dw_dr[1]],
      [dw_dxi[2], dw_deta[2], dw_dr[2]],
    ].into()
  }

  /// sqrt(det g) = r² · sec²ξ · sec²η / (1 + tan²ξ + tan²η)^(3/2)
  ///
  /// Same form as the 2D panel with the constant radius replaced by the local
  /// radial coord. Derivation: ∂world/∂r = v_world (unit, normal to the shell);
  /// ∂world/∂ξ and ∂world/∂η are tangent to the shell and scale linearly with r.
  /// |det J| = r² · |dA_unit_sphere| where |dA_unit_sphere| is the 2D form.
  fn sqrt_det_metric(&self, comp: &Point<3>) -> f64 {
    let sec2_xi = comp[0].cos().powi(-2);
    let sec2_eta = comp[1].cos().powi(-2);
    let r = comp[2];
    let g = 1.0 + comp[0].tan().powi(2) + comp[1].tan().powi(2);
    (r.powi(2) * sec2_xi * sec2_eta) / (g * g.sqrt())
  }
}

/// Maps an angular `Edge` to the `BoundaryTag` that `StructuredBlock` assigns
/// to the corresponding axis 0/1 boundary (see `mesh.rs::boundary_tag`).
fn boundary_tag_for(edge: Edge) -> BoundaryTag {
  match edge {
    Edge::East  => BoundaryTag::Right,
    Edge::West  => BoundaryTag::Left,
    Edge::North => BoundaryTag::Top,
    Edge::South => BoundaryTag::Bottom,
  }
}

/// Pull the c-th column out of a 3×3 matrix as a `Vector<f64, 3>`.
fn col3(m: &Matrix<f64, 3, 3>, c: usize) -> Vector<f64, 3> {
  [m[0][c], m[1][c], m[2][c]].into()
}

/// World-space unit normal at a face on `panel`, given the face's computational
/// centroid and which axis it's perpendicular to. The normal points in panel A's
/// +axis direction in world coords; the per-cell Jacobian captures sphere
/// curvature, so this is exact (not the panel-center approximation).
fn world_unit_normal(
  panel_id: PanelId,
  comp_centroid: &Point<3>,
  axis: usize,
) -> Vector<f64, 3> {
  let map = GnomonicShellPanel::new(panel_id);
  let j = map.jacobian(comp_centroid);
  let cross = match axis {
    0 => col3(&j, 1).cross(&col3(&j, 2)),  // +ξ direction
    1 => col3(&j, 2).cross(&col3(&j, 0)),  // +η direction
    2 => col3(&j, 0).cross(&col3(&j, 1)),  // +r direction
    _ => unreachable!(),
  };
  cross.normalise()
}

/// Find the dominant component of an axis-aligned area vector. Robust to the
/// near-zero noise StructuredBlock won't produce but that we'd rather not
/// trip over.
fn axis_of(av: &Vector<f64, 3>) -> usize {
  let (a, b, c) = (av[0].abs(), av[1].abs(), av[2].abs());
  if a >= b && a >= c { 0 }
  else if b >= c { 1 }
  else { 2 }
}

/// Fixed slot for each panel in the `CubeSphere::panels` array.
fn panel_index(id: PanelId) -> usize {
  match id {
    PanelId::XP => 0,
    PanelId::XN => 1,
    PanelId::YP => 2,
    PanelId::YN => 3,
    PanelId::ZP => 4,
    PanelId::ZN => 5,
  }
}

const PANEL_ORDER: [PanelId; 6] = [
  PanelId::XP, PanelId::XN,
  PanelId::YP, PanelId::YN,
  PanelId::ZP, PanelId::ZN,
];

/// 3D cubed-sphere shell. Six `StructuredBlock<3>` panels are stitched along
/// the 12 cube edges into a single global cell/face index space. Per-panel
/// angular boundary faces (Left/Right/Top/Bottom) are reclassified as
/// `FaceConnection::Interior` connections to the matched cell on the adjacent
/// panel; per-panel radial boundaries (Front/Back) are relabeled as
/// `Ground` (inner) and `AtmosphereEdge` (outer) and remain true boundaries.
///
/// Cell IDs: `global = panel_index * cells_per_panel + panel_local`.
/// Face IDs: assigned during construction; geometry of each face is taken
/// from the panel that contributed it (for inter-panel faces, this is the
/// `panel_a` half from `CUBE_EDGES`).
pub struct CubeSphere {
  panels: [StructuredBlock<3>; 6],
  dims: [usize; 3],
  cells_per_panel: usize,

  // Each global face references one panel's local face for its geometry.
  face_panel: Vec<u8>,
  face_local: Vec<FaceId>,

  // World-frame area vector per face: direction is the world-space outward
  // normal (in panel A's +axis convention for inter-panel faces), magnitude
  // is the computational area. `face_area_vector / face_area` therefore
  // yields a unit world normal, and `face_area · sqrt_metric` still gives
  // physical area.
  face_area_vectors_world: Vec<Vector<f64, 3>>,

  // Topology in global IDs.
  face_connections: Vec<FaceConnection>,
  cell_face_adj: Vec<Vec<FaceId>>,
  interior_face_list: Vec<(FaceId, CellId, CellId)>,
  boundary_face_lists: Vec<(BoundaryTag, Vec<(FaceId, CellId)>)>,
}

impl CubeSphere {
  /// Construct with uniform radial layers. Convenience wrapper over
  /// `with_radial_edges`.
  pub fn new(dims: [usize; 3], r_inner: f64, r_outer: f64) -> Self {
    let radial_edges: Vec<f64> = (0..=dims[2])
      .map(|k| r_inner + (r_outer - r_inner) * k as f64 / dims[2] as f64)
      .collect();
    Self::with_radial_edges([dims[0], dims[1]], radial_edges)
  }

  /// Construct with caller-supplied radial layer edges. Lets atmospheric
  /// callers concentrate cells near the surface (or wherever else they want
  /// extra resolution). `radial_edges` must be strictly increasing and have
  /// length `n_radial_layers + 1`. Angular axes stay uniform — required for
  /// the inter-panel matching to be exact.
  pub fn with_radial_edges(
    angular_dims: [usize; 2],
    radial_edges: Vec<f64>,
  ) -> Self {
    assert_eq!(angular_dims[0], angular_dims[1],
      "angular dims must be equal so adjacent panel edges have matching cell counts");
    assert!(radial_edges.len() >= 2, "need at least two radial edges");
    assert!(radial_edges[0] > 0.0, "inner radius must be positive");
    for k in 1..radial_edges.len() {
      assert!(
        radial_edges[k] > radial_edges[k - 1],
        "radial edges must be strictly increasing"
      );
    }

    let dims: [usize; 3] =
      [angular_dims[0], angular_dims[1], radial_edges.len() - 1];

    let bound = std::f64::consts::FRAC_PI_4;
    let xi_edges: Vec<f64> = (0..=angular_dims[0])
      .map(|i| -bound + 2.0 * bound * i as f64 / angular_dims[0] as f64)
      .collect();
    let eta_edges: Vec<f64> = (0..=angular_dims[1])
      .map(|i| -bound + 2.0 * bound * i as f64 / angular_dims[1] as f64)
      .collect();

    let panels: [StructuredBlock<3>; 6] = PANEL_ORDER.map(|p| {
      let map: Box<dyn GeometryMap<3, 3>> = Box::new(GnomonicShellPanel::new(p));
      StructuredBlock::from_axis_edges(
        [xi_edges.clone(), eta_edges.clone(), radial_edges.clone()],
        map,
      )
    });

    let cells_per_panel: usize = dims.iter().product();
    let total_cells = 6 * cells_per_panel;

    let mut face_panel: Vec<u8> = Vec::new();
    let mut face_local: Vec<FaceId> = Vec::new();
    let mut face_area_vectors_world: Vec<Vector<f64, 3>> = Vec::new();
    let mut face_connections: Vec<FaceConnection> = Vec::new();
    let mut cell_face_adj: Vec<Vec<FaceId>> = vec![Vec::new(); total_cells];
    let mut interior_face_list: Vec<(FaceId, CellId, CellId)> = Vec::new();
    let mut boundary_map: HashMap<BoundaryTag, Vec<(FaceId, CellId)>> =
      HashMap::new();

    let to_global = |panel: usize, local: CellId| -> CellId {
      CellId::from(panel * cells_per_panel + local.index())
    };

    // Build the world-frame area vector for a face on `panel_id`'s panel-local
    // face `local_f`. Magnitude = computational area; direction = world unit
    // normal in panel A's +axis convention.
    let world_av_for = |panel: &StructuredBlock<3>,
                        panel_id: PanelId,
                        local_f: FaceId|
     -> Vector<f64, 3> {
      let comp_av = panel.face_area_vector(local_f);
      let comp_area = panel.face_area(local_f);
      let centroid = panel.face_centroid(local_f);
      let axis = axis_of(&comp_av);
      world_unit_normal(panel_id, centroid, axis) * comp_area
    };

    // 1. Per-panel: keep interior faces and radial boundaries; skip angular
    //    boundaries (those become inter-panel faces in step 2).
    for (p, panel) in panels.iter().enumerate() {
      let panel_id = PANEL_ORDER[p];

      for &(local_f, owner, nbr) in panel.interior_faces() {
        let gf = FaceId::from(face_panel.len());
        face_panel.push(p as u8);
        face_local.push(local_f);
        face_area_vectors_world.push(world_av_for(panel, panel_id, local_f));
        let go = to_global(p, owner);
        let gn = to_global(p, nbr);
        face_connections.push(FaceConnection::Interior {
          owner: go, neighbour: gn,
        });
        interior_face_list.push((gf, go, gn));
        cell_face_adj[go.index()].push(gf);
        cell_face_adj[gn.index()].push(gf);
      }

      for (panel_tag, sphere_tag) in [
        (BoundaryTag::Front, BoundaryTag::Ground),
        (BoundaryTag::Back,  BoundaryTag::AtmosphereEdge),
      ] {
        for &(local_f, owner) in panel.boundary_faces(panel_tag) {
          let out_sign = match panel.face_connection(local_f) {
            FaceConnection::Boundary { out_sign, .. } => *out_sign,
            _ => unreachable!(),
          };
          let gf = FaceId::from(face_panel.len());
          face_panel.push(p as u8);
          face_local.push(local_f);
          face_area_vectors_world.push(world_av_for(panel, panel_id, local_f));
          let go = to_global(p, owner);
          face_connections.push(FaceConnection::Boundary {
            owner: go, tag: sphere_tag, out_sign,
          });
          boundary_map.entry(sphere_tag).or_default().push((gf, go));
          cell_face_adj[go.index()].push(gf);
        }
      }
    }

    // 2. Stitch inter-panel faces. One face per (cube edge × radial layer ×
    //    angular cell). Geometry comes from panel A; owner/neighbour follow
    //    the direction of panel A's outward normal at that edge.
    let nx = dims[0];
    let ny = dims[1];
    let nz = dims[2];

    for &(panel_a_id, edge_a, panel_b_id, edge_b) in CUBE_EDGES.iter() {
      let pa = panel_index(panel_a_id);
      let pb = panel_index(panel_b_id);

      // The 2D matcher only depends on direction; radius cancels in the
      // nearest-distance comparison, so any positive value works.
      let map_a = GnomonicPanel::new(panel_a_id, 1.0);
      let map_b = GnomonicPanel::new(panel_b_id, 1.0);
      let pairs_2d = match_edge_cells(
        &map_a, edge_a, [nx, ny],
        &map_b, edge_b, [nx, ny],
      );

      // Cell→face map for panel A's edge. Includes all 3D cells along the
      // edge (Ny·Nz or Nx·Nz entries).
      let face_for_cell_a: HashMap<CellId, FaceId> = panels[pa]
        .boundary_faces(boundary_tag_for(edge_a))
        .iter()
        .map(|&(f, c)| (c, f))
        .collect();

      for k in 0..nz {
        for &(cell_a_2d, cell_b_2d) in &pairs_2d {
          let (i_a, j_a) = (cell_a_2d.index() % nx, cell_a_2d.index() / nx);
          let (i_b, j_b) = (cell_b_2d.index() % nx, cell_b_2d.index() / nx);
          let cell_a_local = CellId::from(i_a + j_a * nx + k * nx * ny);
          let cell_b_local = CellId::from(i_b + j_b * nx + k * nx * ny);
          let cell_a_global = to_global(pa, cell_a_local);
          let cell_b_global = to_global(pb, cell_b_local);

          let local_face_a = face_for_cell_a[&cell_a_local];

          // Panel A's face_area_vector is in panel A's +axis direction.
          // East/North: that's outward from panel A → points cell_a → cell_b
          //             → owner = cell_a.
          // West/South: that's inward into panel A → points cell_b → cell_a
          //             → owner = cell_b.
          let (owner, neighbour) = match edge_a {
            Edge::East | Edge::North => (cell_a_global, cell_b_global),
            Edge::West | Edge::South => (cell_b_global, cell_a_global),
          };

          let gf = FaceId::from(face_panel.len());
          face_panel.push(pa as u8);
          face_local.push(local_face_a);
          face_area_vectors_world.push(
            world_av_for(&panels[pa], panel_a_id, local_face_a)
          );
          face_connections.push(FaceConnection::Interior { owner, neighbour });
          interior_face_list.push((gf, owner, neighbour));
          cell_face_adj[cell_a_global.index()].push(gf);
          cell_face_adj[cell_b_global.index()].push(gf);
        }
      }
    }

    let boundary_face_lists: Vec<(BoundaryTag, Vec<(FaceId, CellId)>)> =
      boundary_map.into_iter().collect();

    CubeSphere {
      panels,
      dims,
      cells_per_panel,
      face_panel,
      face_local,
      face_area_vectors_world,
      face_connections,
      cell_face_adj,
      interior_face_list,
      boundary_face_lists,
    }
  }

  pub fn dims(&self) -> [usize; 3] { self.dims }

  pub fn panels(&self) -> &[StructuredBlock<3>; 6] { &self.panels }

  /// World-space centroid of `cell`. Cube-sphere `cell_centroid` returns
  /// computational `(ξ, η, r)`; this projects through the owning panel's
  /// gnomonic map to give an actual `(x, y, z)`.
  pub fn cell_world_centroid(&self, cell: CellId) -> Point<3> {
    let (panel_idx, local) = self.cell_to_panel(cell);
    let panel_id = PANEL_ORDER[panel_idx];
    GnomonicShellPanel::new(panel_id)
      .to_physical(self.panels[panel_idx].cell_centroid(local))
  }

  /// Build a per-cell radial gravity field — `g · (-r̂)` evaluated at each
  /// cell's world centroid. Magnitude is constant (uniform-g approximation
  /// valid for atmospheres thin compared to the planet radius). Pass the
  /// returned vector to `Euler3D::with_per_cell_gravity`.
  pub fn radial_gravity_field(&self, surface_g: f64) -> Vec<[f64; 3]> {
    (0..self.cell_count())
      .map(|i| {
        let p = self.cell_world_centroid(CellId::from(i));
        let r = (p[0].powi(2) + p[1].powi(2) + p[2].powi(2)).sqrt();
        [
          -surface_g * p[0] / r,
          -surface_g * p[1] / r,
          -surface_g * p[2] / r,
        ]
      })
      .collect()
  }

  /// Build a radial-edge array in `[r_inner, r_outer]` whose spacing
  /// concentrates layers near the inner radius (= the ground for an
  /// atmospheric shell). `beta` controls how aggressive the stretching is:
  ///
  ///   * `beta` → 0    ⇒ uniform spacing
  ///   * `beta` ≈ 1–2  ⇒ moderate near-surface refinement
  ///   * `beta` ≳ 3    ⇒ aggressive near-surface refinement
  ///
  /// Uses `sinh(β·s) / sinh(β)` where `s ∈ [0, 1]` runs from inner to outer.
  /// `sinh`'s derivative is small near `s = 0` and grows toward `s = 1`, so
  /// the resulting edge spacing is small near the inner surface and large
  /// near the outer.
  pub fn radial_edges_stretched(
    r_inner: f64,
    r_outer: f64,
    n_layers: usize,
    beta: f64,
  ) -> Vec<f64> {
    assert!(beta >= 0.0 && beta.is_finite(), "beta must be a non-negative finite number");
    assert!(n_layers >= 1);
    if beta == 0.0 {
      return (0..=n_layers)
        .map(|k| r_inner + (r_outer - r_inner) * k as f64 / n_layers as f64)
        .collect();
    }
    let denom = beta.sinh();
    (0..=n_layers)
      .map(|k| {
        let s = k as f64 / n_layers as f64;
        r_inner + (r_outer - r_inner) * (beta * s).sinh() / denom
      })
      .collect()
  }

  fn cell_to_panel(&self, cell: CellId) -> (usize, CellId) {
    let g = cell.index();
    (g / self.cells_per_panel, CellId::from(g % self.cells_per_panel))
  }

  fn face_to_panel(&self, face: FaceId) -> (&StructuredBlock<3>, FaceId) {
    let p = self.face_panel[face.index()] as usize;
    (&self.panels[p], self.face_local[face.index()])
  }
}

impl CellGeometry<3> for CubeSphere {
  fn cell_centroid(&self, cell: CellId) -> &Point<3> {
    let (p, l) = self.cell_to_panel(cell);
    self.panels[p].cell_centroid(l)
  }
  fn cell_volume(&self, cell: CellId) -> f64 {
    let (p, l) = self.cell_to_panel(cell);
    self.panels[p].cell_volume(l)
  }
  fn cell_metrics(&self, cell: CellId) -> &CellMetrics<3> {
    let (p, l) = self.cell_to_panel(cell);
    self.panels[p].cell_metrics(l)
  }
  fn cell_count(&self) -> usize {
    6 * self.cells_per_panel
  }
}

impl FaceGeometry<3> for CubeSphere {
  fn face_centroid(&self, face: FaceId) -> &Point<3> {
    let (panel, local) = self.face_to_panel(face);
    panel.face_centroid(local)
  }
  fn face_area_vector(&self, face: FaceId) -> Vector<f64, 3> {
    self.face_area_vectors_world[face.index()].clone()
  }
  fn face_area(&self, face: FaceId) -> f64 {
    let (panel, local) = self.face_to_panel(face);
    panel.face_area(local)
  }
  fn face_metrics(&self, face: FaceId) -> &FaceMetrics<3> {
    let (panel, local) = self.face_to_panel(face);
    panel.face_metrics(local)
  }
  fn face_count(&self) -> usize {
    self.face_panel.len()
  }
}

impl Topology for CubeSphere {
  fn face_connection(&self, face: FaceId) -> &FaceConnection {
    &self.face_connections[face.index()]
  }
  fn cell_faces(&self, cell: CellId) -> &[FaceId] {
    &self.cell_face_adj[cell.index()]
  }
  fn interior_faces(&self) -> &[(FaceId, CellId, CellId)] {
    &self.interior_face_list
  }
  fn boundary_faces(&self, tag: BoundaryTag) -> &[(FaceId, CellId)] {
    self.boundary_face_lists.iter()
      .find(|(t, _)| *t == tag)
      .map(|(_, l)| l.as_slice())
      .unwrap_or(&[])
  }
  fn boundary_tags(&self) -> impl Iterator<Item = BoundaryTag> + '_ {
    self.boundary_face_lists.iter().map(|(t, _)| *t)
  }
}
