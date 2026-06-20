// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Diagnostic: where does the cube-sphere atmosphere's explicit CFL actually
//! bind — the thin radial (vertical) spacing, or the compressed tangential
//! (horizontal) spacing at panel corners? This decides whether a
//! horizontally-explicit / vertically-implicit (HEVI) scheme — a cheap
//! per-column tridiagonal solve — would speed the demo up, and by how much.
//!
//! Run: `cargo test -p continuum --test cfl_anatomy --release -- --ignored --nocapture`

use tessera::cube_sphere::CubeSphere;
use tessera::geometry::{CellGeometry, FaceGeometry};
use tessera::mesh::Mesh;
use tessera::topology::Topology;
use utility::domain::CellId;

/// Standard sea-level-ish acoustic speed (γ=1.4, p=1e5, ρ=1.2) ≈ 341 m/s; |u|≈0.
const SOUND_SPEED: f64 = 341.0;
const CFL: f64 = 0.25;

fn report(angular: usize, radial: usize) {
  // Earth radius, 20 km atmosphere shell — the demo's geometry.
  let r_inner = 6.371e6;
  let r_outer = r_inner + 20_000.0;
  let mesh = CubeSphere::new([angular, angular, radial], r_inner, r_outer);
  let cells = mesh.cell_count();

  // Per cell, split its faces into radial (normal ∥ r̂) and tangential, and form
  // the directional cell spacing dx = volume / face_area for each.
  let mut min_dt_radial = f64::INFINITY; // vertical-bound CFL (current)
  let mut min_dt_tangential = f64::INFINITY; // horizontal-bound CFL (HEVI)
  let mut radial_binds = 0usize;

  for i in 0..cells {
    let cell = CellId::from(i);
    let c = mesh.cell_world_centroid(cell);
    let r = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
    let r_hat = [c[0] / r, c[1] / r, c[2] / r];

    let vol = mesh.cell_metrics(cell).phys_volume;
    let mut radial_area = 0.0_f64;
    let mut tangential_area = 0.0_f64;
    for &face in mesh.cell_faces(cell) {
      let area = mesh.face_area(face);
      if area <= 0.0 {
        continue;
      }
      let av = mesh.face_area_vector(face);
      let n = [av[0] / area, av[1] / area, av[2] / area];
      let radiality =
        (n[0] * r_hat[0] + n[1] * r_hat[1] + n[2] * r_hat[2]).abs();
      let phys_area = mesh.face_metrics(face).phys_area;
      if radiality > 0.7 {
        radial_area = radial_area.max(phys_area);
      } else {
        tangential_area = tangential_area.max(phys_area);
      }
    }

    // dt limited by the radial faces (vertical transport) vs the tangential
    // faces (horizontal transport).
    if radial_area > 0.0 {
      let dz = vol / radial_area;
      min_dt_radial = min_dt_radial.min(CFL * dz / SOUND_SPEED);
    }
    if tangential_area > 0.0 {
      let dxh = vol / tangential_area;
      min_dt_tangential = min_dt_tangential.min(CFL * dxh / SOUND_SPEED);
    }
    if radial_area > 0.0 && tangential_area > 0.0 {
      let dz = vol / radial_area;
      let dxh = vol / tangential_area;
      if dz < dxh {
        radial_binds += 1;
      }
    }
  }

  let current = min_dt_radial.min(min_dt_tangential);
  let hevi = min_dt_tangential;
  eprintln!(
    "[{angular}x{angular}x{radial}, {cells} cells]\n  \
     current explicit dt (min over all dirs):  {current:.4} s\n  \
     vertical-bound dt:    {min_dt_radial:.4} s\n  \
     horizontal-bound dt:  {min_dt_tangential:.4} s\n  \
     HEVI dt (vertical implicit) ≈ horizontal: {hevi:.4} s\n  \
     => HEVI potential speedup: {:.1}x   (radial binds in {:.1}% of cells)",
    hevi / current,
    100.0 * radial_binds as f64 / cells as f64
  );
}

#[test]
#[ignore = "diagnostic; run with --release --ignored --nocapture"]
fn cube_sphere_cfl_anatomy() {
  // The demo uses 128x128 angular x 30 radial; smaller resolutions show how the
  // ratio scales (horizontal dx ∝ 1/angular, vertical dx fixed by shell height).
  report(32, 30);
  report(64, 30);
  report(128, 30);
}
