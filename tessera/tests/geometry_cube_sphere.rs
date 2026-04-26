// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use continuum::{
  cube_sphere::{
    CUBE_EDGES, CubeSphere, GnomonicPanel, GnomonicShellPanel, PanelId,
    cells_along_edge, edge_cell_index, edge_face_centroid_comp,
    match_edge_cells,
  },
  geometry::{CellGeometry, CellId, FaceGeometry, FaceId, GeometryMap, Point},
  topology::{BoundaryTag, FaceConnection, Topology},
};

fn all_panels() -> [PanelId; 6] {
  [
    PanelId::XP,
    PanelId::XN,
    PanelId::YP,
    PanelId::YN,
    PanelId::ZP,
    PanelId::ZN,
  ]
}

#[test]
fn round_trip_recovers_input() {
  for panel_id in [
    PanelId::XP,
    PanelId::XN,
    PanelId::YP,
    PanelId::YN,
    PanelId::ZP,
    PanelId::ZN,
  ] {
    let panel = GnomonicPanel::new(panel_id, 7.0);
    let bound = std::f64::consts::FRAC_PI_4;
    for i in 0..10 {
      for j in 0..10 {
        let xi = -bound + (i as f64 + 0.5) * (2.0 * bound / 10.0);
        let eta = -bound + (j as f64 + 0.5) * (2.0 * bound / 10.0);
        let comp_in: Point<2> = [xi, eta].into();
        let world = panel.to_physical(&comp_in);
        let comp_out = panel.to_computational(&world).expect("on-panel");
        assert!((comp_out[0] - xi).abs() < 1e-12);
        assert!((comp_out[1] - eta).abs() < 1e-12);
      }
    }
  }
}

#[test]
fn panel_area_is_one_sixth_of_sphere() {
  let panel = GnomonicPanel::new(PanelId::ZP, 1.0);
  let n = 64;
  let h = (std::f64::consts::FRAC_PI_2) / n as f64; // π/2 total range, n cells
  let lo = -std::f64::consts::FRAC_PI_4;
  let mut area = 0.0;
  for i in 0..n {
    for j in 0..n {
      let xi = lo + (i as f64 + 0.5) * h;
      let eta = lo + (j as f64 + 0.5) * h;
      area += panel.sqrt_det_metric(&[xi, eta].into()) * h * h;
    }
  }
  let expected = 4.0 * std::f64::consts::PI / 6.0;
  assert!((area - expected).abs() < 1e-3);
}

#[test]
fn cube_edges_meet_in_world_space() {
  let radius = 3.0;
  let panels: std::collections::HashMap<PanelId, GnomonicPanel> = all_panels()
    .into_iter()
    .map(|p| (p, GnomonicPanel::new(p, radius)))
    .collect();

  for &(panel_a, edge_a, panel_b, edge_b) in CUBE_EDGES.iter() {
    let mid_a = panels[&panel_a].to_physical(&edge_a.midpoint_comp());
    let mid_b = panels[&panel_b].to_physical(&edge_b.midpoint_comp());
    let d = mid_a.distance(&mid_b);
    assert!(
      d < 1e-12,
      "{:?}-{:?} and {:?}-{:?} disagree at midpoint: {:?} vs {:?} (d = {:e})",
      panel_a,
      edge_a,
      panel_b,
      edge_b,
      mid_a,
      mid_b,
      d
    );
  }
}

#[test]
fn match_edge_cells_pairs_face_centroids_exactly() {
  let radius = 5.0;
  let dims = [16, 16];
  let panels: std::collections::HashMap<PanelId, GnomonicPanel> = all_panels()
    .into_iter()
    .map(|p| (p, GnomonicPanel::new(p, radius)))
    .collect();

  for &(pa, ea, pb, eb) in CUBE_EDGES.iter() {
    let pairs =
      match_edge_cells(&panels[&pa], ea, dims, &panels[&pb], eb, dims);

    let n = cells_along_edge(ea, dims);
    assert_eq!(
      pairs.len(),
      n,
      "{:?}-{:?} ↔ {:?}-{:?}: expected {} pairs",
      pa,
      ea,
      pb,
      eb,
      n
    );

    // Bijective: every cell on side B is used exactly once.
    let used_b: std::collections::HashSet<_> =
      pairs.iter().map(|&(_, b)| b).collect();
    assert_eq!(
      used_b.len(),
      pairs.len(),
      "{:?}-{:?} ↔ {:?}-{:?}: B cells reused",
      pa,
      ea,
      pb,
      eb
    );

    // Recover face centroids for each match and verify world-space coincidence.
    // At equal resolution, gnomonic edges line up exactly — distance should be
    // floating-point noise, not a fraction of a cell.
    for (k_a, &(cell_a, cell_b)) in pairs.iter().enumerate() {
      assert_eq!(cell_a, edge_cell_index(ea, dims, k_a));

      // Find which k on side B this cell corresponds to.
      let k_b = (0..n)
        .find(|&k| edge_cell_index(eb, dims, k) == cell_b)
        .expect("cell_b lies on edge_b");

      let world_a =
        panels[&pa].to_physical(&edge_face_centroid_comp(ea, dims, k_a));
      let world_b =
        panels[&pb].to_physical(&edge_face_centroid_comp(eb, dims, k_b));
      let d = world_a.distance(&world_b);
      assert!(
        d < 1e-10,
        "{:?}-{:?}[{}] ↔ {:?}-{:?}[{}] mismatch: d = {:e}",
        pa,
        ea,
        k_a,
        pb,
        eb,
        k_b,
        d
      );
    }
  }
}

#[test]
fn cube_edges_table_covers_every_panel_edge_once() {
  // Every (panel, edge) pair must appear exactly once across the 12 entries.
  let mut seen = std::collections::HashSet::new();
  for &(pa, ea, pb, eb) in CUBE_EDGES.iter() {
    assert!(
      seen.insert((pa, ea)),
      "duplicate entry for {:?}-{:?}",
      pa,
      ea
    );
    assert!(
      seen.insert((pb, eb)),
      "duplicate entry for {:?}-{:?}",
      pb,
      eb
    );
  }
  assert_eq!(
    seen.len(),
    24,
    "expected 6 panels × 4 edges = 24 (panel,edge) pairs"
  );
}

#[test]
fn cube_sphere_topology_is_consistent() {
  let n = 4;
  let nz = 2;
  let mesh = CubeSphere::new([n, n, nz], 1.0, 2.0);

  // Cell count: 6 panels × N × N × Nz.
  let expected_cells = 6 * n * n * nz;
  assert_eq!(mesh.cell_count(), expected_cells);

  // Boundary face counts: one face per cell on each radial cap.
  let ground = mesh.boundary_faces(BoundaryTag::Ground).len();
  let edge = mesh.boundary_faces(BoundaryTag::AtmosphereEdge).len();
  assert_eq!(ground, 6 * n * n);
  assert_eq!(edge, 6 * n * n);

  // Total face count: 6·N²·(3·Nz + 1). Derivation:
  //   per panel internal + radial:
  //     axis-0 internal (N-1)·N·Nz + axis-1 internal N·(N-1)·Nz
  //     + axis-2 all N·N·(Nz+1) = 2N(N-1)Nz + N²(Nz+1)
  //   inter-panel: 12 cube edges × N × Nz = 12·N·Nz
  //   sum: 6·[2N(N-1)Nz + N²(Nz+1)] + 12·N·Nz
  //      = 12N²Nz + 6N²(Nz+1) = 6N²(3Nz + 1)
  let expected_faces = 6 * n * n * (3 * nz + 1);
  assert_eq!(mesh.face_count(), expected_faces);

  // Interior + boundary partition the face set.
  assert_eq!(mesh.interior_faces().len() + ground + edge, expected_faces);

  // Every cell touches exactly 6 faces (one per ±axis direction; inter-panel
  // faces fill in for the would-be missing angular neighbours).
  for c in 0..mesh.cell_count() {
    let faces = mesh.cell_faces(CellId::from(c));
    assert_eq!(faces.len(), 6, "cell {} has {} faces", c, faces.len());
  }

  // Every interior face has two distinct cell IDs.
  for &(_, owner, neighbour) in mesh.interior_faces() {
    assert_ne!(owner, neighbour);
  }

  // Every face's connection agrees with the interior/boundary split.
  let mut interior_count = 0;
  let mut boundary_count = 0;
  for f in 0..mesh.face_count() {
    match mesh.face_connection(continuum::geometry::FaceId::from(f)) {
      FaceConnection::Interior { .. } => interior_count += 1,
      FaceConnection::Boundary { .. } => boundary_count += 1,
    }
  }
  assert_eq!(interior_count, mesh.interior_faces().len());
  assert_eq!(boundary_count, ground + edge);
}

// Mirrors CubeSphere's internal cell→panel decoding so tests can reproject
// computational centroids into world coords through the right panel.
fn panel_of(cell: CellId, dims: [usize; 3]) -> GnomonicShellPanel {
  const PANELS: [PanelId; 6] = [
    PanelId::XP,
    PanelId::XN,
    PanelId::YP,
    PanelId::YN,
    PanelId::ZP,
    PanelId::ZN,
  ];
  let p = cell.index() / (dims[0] * dims[1] * dims[2]);
  GnomonicShellPanel::new(PANELS[p])
}

#[test]
fn cube_sphere_face_normals_point_owner_to_neighbour() {
  // For every interior face, the world-frame area vector should point from
  // owner's world centroid toward neighbour's. The flux solver assumes this
  // — if it fails the residual gets the wrong sign and Δt accumulates the
  // wrong way.
  let dims = [4, 4, 2];
  let mesh = CubeSphere::new(dims, 1.0, 2.0);

  for &(face, owner, neighbour) in mesh.interior_faces() {
    let av = mesh.face_area_vector(face);
    let world_o = panel_of(owner, dims).to_physical(mesh.cell_centroid(owner));
    let world_n =
      panel_of(neighbour, dims).to_physical(mesh.cell_centroid(neighbour));
    let dot = (0..3)
      .map(|i| av[i] * (world_n[i] - world_o[i]))
      .sum::<f64>();
    assert!(
      dot > 0.0,
      "face {} owner={:?} neighbour={:?}: av·Δ = {}",
      face.index(),
      owner,
      neighbour,
      dot
    );
  }
}

#[test]
fn cube_sphere_radial_face_normals_are_radial() {
  // Ground/AtmosphereEdge face normals must be parallel to the radial
  // unit vector through the face centroid (axis-2 normal = +r direction
  // by panel convention, so dot ≈ +1).
  let dims = [6, 6, 3];
  let mesh = CubeSphere::new(dims, 1.0, 2.5);

  for tag in [BoundaryTag::Ground, BoundaryTag::AtmosphereEdge] {
    for &(face, owner) in mesh.boundary_faces(tag) {
      let av = mesh.face_area_vector(face);
      let area = mesh.face_area(face);
      let panel = panel_of(owner, dims);
      let world_c = panel.to_physical(mesh.face_centroid(face));
      let r = (0..3).map(|i| world_c[i].powi(2)).sum::<f64>().sqrt();
      let dot = (0..3)
        .map(|i| (av[i] / area) * (world_c[i] / r))
        .sum::<f64>();
      assert!(
        (dot - 1.0).abs() < 1e-10,
        "{:?} face {}: dot={}",
        tag,
        face.index(),
        dot
      );
    }
  }
}

#[test]
fn cube_sphere_stretched_radial_layers_concentrate_near_inner_surface() {
  // tanh-stretched edges should put more cells near r_inner. Verify:
  //   * the stretched edges are monotonic and span [r_inner, r_outer],
  //   * the innermost layer is thinner than the outermost.
  let r_inner = 1.0;
  let r_outer = 2.0;
  let n = 8;
  let beta = 2.0;
  let edges = CubeSphere::radial_edges_stretched(r_inner, r_outer, n, beta);

  assert_eq!(edges.len(), n + 1);
  assert!((edges[0] - r_inner).abs() < 1e-12);
  assert!((edges[n] - r_outer).abs() < 1e-12);
  for k in 1..edges.len() {
    assert!(edges[k] > edges[k - 1]);
  }

  let dr_inner = edges[1] - edges[0];
  let dr_outer = edges[n] - edges[n - 1];
  assert!(
    dr_outer > 3.0 * dr_inner,
    "expected outer layer > 3 × inner; got dr_inner={} dr_outer={}",
    dr_inner,
    dr_outer
  );
}

#[test]
fn cube_sphere_with_stretched_radial_still_matches_shell_volume() {
  // The discrete physical-volume sum is independent of how the radial axis
  // is partitioned — uniform or stretched, the integral of sqrt(det g) over
  // [r_inner, r_outer] is exact for axis-2 (the shell metric factorizes as
  // r²·sec²ξ·sec²η/g^(3/2), and ∫r² dr is exact under any partition since
  // each cell uses the actual cell-centre r).
  //
  // Numerically this means the stretched mesh should match shell volume to
  // the same tolerance as the uniform mesh.
  let r_inner = 1.0;
  let r_outer = 2.0;
  let n_ang = 16;
  let n_rad = 8;

  let edges = CubeSphere::radial_edges_stretched(r_inner, r_outer, n_rad, 1.5);
  let mesh = CubeSphere::with_radial_edges([n_ang, n_ang], edges);

  let total_phys: f64 = (0..mesh.cell_count())
    .map(|i| {
      let cell = CellId::from(i);
      mesh.cell_volume(cell) * mesh.cell_metrics(cell).sqrt_metric
    })
    .sum();

  let expected =
    4.0 * std::f64::consts::PI / 3.0 * (r_outer.powi(3) - r_inner.powi(3));
  let rel_err = (total_phys - expected).abs() / expected;
  assert!(
    rel_err < 5e-2,
    "stretched cube sphere volume {} vs expected {} (rel err {})",
    total_phys,
    expected,
    rel_err
  );
}

#[test]
fn cube_sphere_stretched_face_normals_still_point_owner_to_neighbour() {
  // Same invariant as the uniform-radial case — the world-frame area vector
  // must point from owner toward neighbour. With non-uniform radial spacing
  // we want to confirm the Jacobian-based normal computation hasn't picked
  // up a sign error from the per-cell width variation.
  let n_ang = 4;
  let n_rad = 4;
  let edges = CubeSphere::radial_edges_stretched(1.0, 3.0, n_rad, 2.0);
  let dims = [n_ang, n_ang, n_rad];
  let mesh = CubeSphere::with_radial_edges([n_ang, n_ang], edges);

  for &(face, owner, neighbour) in mesh.interior_faces() {
    let av = mesh.face_area_vector(face);
    let world_o = panel_of(owner, dims).to_physical(mesh.cell_centroid(owner));
    let world_n =
      panel_of(neighbour, dims).to_physical(mesh.cell_centroid(neighbour));
    let dot = (0..3)
      .map(|i| av[i] * (world_n[i] - world_o[i]))
      .sum::<f64>();
    assert!(
      dot > 0.0,
      "face {} owner={:?} neighbour={:?}: av·Δ = {}",
      face.index(),
      owner,
      neighbour,
      dot
    );
  }
}

#[test]
fn cube_sphere_volume_sums_to_shell_volume() {
  // Sum of cell volumes should match (4π/3)(R_outer³ - R_inner³).
  // Cell volumes are computational; physical volume = vol · sqrt_metric.
  let r_inner = 1.0;
  let r_outer = 2.0;
  let n = 16;
  let nz = 4;
  let mesh = CubeSphere::new([n, n, nz], r_inner, r_outer);

  let mut total_phys = 0.0;
  for c in 0..mesh.cell_count() {
    let cell = CellId::from(c);
    let vol = mesh.cell_volume(cell);
    let m = mesh.cell_metrics(cell);
    total_phys += vol * m.sqrt_metric;
  }

  let expected =
    4.0 * std::f64::consts::PI / 3.0 * (r_outer.powi(3) - r_inner.powi(3));
  let rel_err = (total_phys - expected).abs() / expected;
  assert!(
    rel_err < 1e-2,
    "volume {} vs expected {} (rel err {})",
    total_phys,
    expected,
    rel_err
  );
}

#[test]
fn shell_panel_round_trip_recovers_input() {
  let bound = std::f64::consts::FRAC_PI_4;
  let r_inner = 1.0;
  let r_outer = 2.5;
  for panel_id in all_panels() {
    let panel = GnomonicShellPanel::new(panel_id);
    for i in 0..6 {
      for j in 0..6 {
        for k in 0..4 {
          let xi = -bound + (i as f64 + 0.5) * (2.0 * bound / 6.0);
          let eta = -bound + (j as f64 + 0.5) * (2.0 * bound / 6.0);
          let r = r_inner + (k as f64 + 0.5) * (r_outer - r_inner) / 4.0;
          let comp_in: Point<3> = [xi, eta, r].into();
          let world = panel.to_physical(&comp_in);
          let comp_out = panel.to_computational(&world).expect("on-panel");
          assert!((comp_out[0] - xi).abs() < 1e-12);
          assert!((comp_out[1] - eta).abs() < 1e-12);
          assert!((comp_out[2] - r).abs() < 1e-12);
        }
      }
    }
  }
}

#[test]
fn shell_panel_volume_is_one_sixth_of_shell() {
  // Integrate sqrt(det g) over the panel's computational box. One panel covers
  // 1/6 of the spherical shell, so the result should be (4π/3)(R³ - r³) / 6.
  let panel = GnomonicShellPanel::new(PanelId::ZP);
  let r_inner = 1.0;
  let r_outer = 2.0;
  let n_ang = 32;
  let n_rad = 16;
  let bound = std::f64::consts::FRAC_PI_4;
  let h_ang = (std::f64::consts::FRAC_PI_2) / n_ang as f64;
  let h_rad = (r_outer - r_inner) / n_rad as f64;

  let mut volume = 0.0;
  for i in 0..n_ang {
    for j in 0..n_ang {
      for k in 0..n_rad {
        let xi = -bound + (i as f64 + 0.5) * h_ang;
        let eta = -bound + (j as f64 + 0.5) * h_ang;
        let r = r_inner + (k as f64 + 0.5) * h_rad;
        volume +=
          panel.sqrt_det_metric(&[xi, eta, r].into()) * h_ang * h_ang * h_rad;
      }
    }
  }

  let shell_volume =
    4.0 * std::f64::consts::PI / 3.0 * (r_outer.powi(3) - r_inner.powi(3));
  let expected = shell_volume / 6.0;
  let rel_err = (volume - expected).abs() / expected;
  assert!(
    rel_err < 1e-3,
    "volume {} vs expected {} (rel err {})",
    volume,
    expected,
    rel_err
  );
}
