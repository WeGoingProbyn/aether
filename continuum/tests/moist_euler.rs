// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Phase 1: `MoistEuler3D` is dry `Euler3D` plus one advected moisture
//! tracer. These pin the two invariants the rest of the water cycle leans
//! on: the dry dynamics are unchanged, and `ρq` is transported by the
//! velocity field (so atmospheric water moves with the flow and is
//! conserved by the finite-volume update).

use continuum::model::{ConservationLaw, Euler3D, MoistEuler3D};
use tessera::geometry::CellMetrics;
use utility::domain::{CellId, Point};

const GAMMA: f64 = 1.4;

fn metrics() -> CellMetrics<3> {
  CellMetrics {
    sqrt_metric: 1.0,
    comp_volume: 1.0,
    phys_volume: 1.0,
  }
}

fn origin() -> Point<3> {
  Point::<3>::from([0.0; 3])
}

/// A representative moving, moist state: ρ=1.2, velocity (3,−2,1), some
/// internal energy, and ρq for q≈0.01.
fn sample() -> [f64; 6] {
  let rho = 1.2;
  let (u, v, w) = (3.0, -2.0, 1.0);
  let internal = 2.5e5;
  let ke = 0.5 * rho * (u * u + v * v + w * w);
  [rho, rho * u, rho * v, rho * w, internal + ke, rho * 0.01]
}

#[test]
fn dry_components_match_euler3d() {
  let moist = MoistEuler3D::new(GAMMA);
  let dry = Euler3D::new(GAMMA);
  let s = sample();
  let ds = MoistEuler3D::dry_state(&s);

  assert_eq!(moist.pressure(&s), dry.pressure(&ds));
  assert_eq!(moist.velocity(&s), dry.velocity(&ds));
  assert_eq!(moist.max_wave_speed(&s), dry.max_wave_speed(&ds));

  let mf = moist.flux(&s);
  let df = dry.flux(&ds);
  for d in 0..3 {
    for i in 0..5 {
      assert_eq!(mf[d][i], df[d][i], "flux mismatch at dir {d} comp {i}");
    }
  }
}

#[test]
fn moisture_is_advected_with_velocity() {
  let moist = MoistEuler3D::new(GAMMA);
  let s = sample();
  let flux = moist.flux(&s);
  let rho_q = s[5];
  let u = [s[1] / s[0], s[2] / s[0], s[3] / s[0]];
  for d in 0..3 {
    let expected = rho_q * u[d];
    assert!(
      (flux[d][5] - expected).abs() < 1e-12,
      "tracer flux dir {d}: got {} want {}",
      flux[d][5],
      expected
    );
  }
}

#[test]
fn fix_state_clamps_negative_moisture_and_floors_density() {
  let moist = MoistEuler3D::new(GAMMA);
  let mut s = sample();
  s[5] = -3.0; // unphysical negative water mass
  moist.fix_state(&mut s);
  assert!(s[5] >= 0.0, "moisture clamped to non-negative");

  let mut s2 = [-1.0, 0.0, 0.0, 0.0, 1.0, 0.5];
  moist.fix_state(&mut s2);
  assert!(s2[0] > 0.0, "density floored positive");
  assert!(s2[5] >= 0.0);
}

#[test]
fn coriolis_deflects_momentum_without_doing_work() {
  // Spin about +z. Coriolis on horizontal momentum is perpendicular to it.
  let omega = 7.29e-5;
  let moist = MoistEuler3D::new(GAMMA).with_rotation([0.0, 0.0, omega]);
  let s = sample(); // moving (u=3, v=-2, w=1)
  let src = moist.source(&s, CellId::from(0), &origin(), &metrics());

  // No gravity here, so the source is pure Coriolis: −2 Ω×m.
  // Ω = (0,0,ω): −2 Ω×m = (2ω·m_y, −2ω·m_x, 0).
  let m = [s[1], s[2], s[3]];
  assert!((src[1] - 2.0 * omega * m[1]).abs() < 1e-12);
  assert!((src[2] + 2.0 * omega * m[0]).abs() < 1e-12);
  assert_eq!(src[3], 0.0);
  // Coriolis does no work and adds no mass / moisture.
  assert_eq!(src[0], 0.0);
  assert_eq!(src[4], 0.0);
  assert_eq!(src[5], 0.0);
  // Force is perpendicular to horizontal momentum (zero power).
  let power = src[1] * m[0] + src[2] * m[1] + src[3] * m[2];
  assert!(
    power.abs() < 1e-9,
    "Coriolis power should be ~0, got {power}"
  );
}

#[test]
fn coriolis_vanishes_at_rest() {
  let moist = MoistEuler3D::new(GAMMA).with_rotation([0.0, 0.0, 1.0]);
  let rest = [1.2, 0.0, 0.0, 0.0, 2.5e5, 0.01];
  let src = moist.source(&rest, CellId::from(0), &origin(), &metrics());
  assert_eq!(src, [0.0; 6]);
}

#[test]
fn source_leaves_moisture_untouched() {
  // Gravity acts on momentum/energy; the law itself adds no moisture
  // source (evaporation/condensation are separate stages).
  let moist = MoistEuler3D::with_gravity(GAMMA, [0.0, 0.0, -9.81]);
  let s = sample();
  let src = moist.source(&s, CellId::from(0), &origin(), &metrics());
  assert_eq!(src[5], 0.0);
  // Momentum-z source is ρ·g_z.
  assert!((src[3] - s[0] * -9.81).abs() < 1e-9);
}
