// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Pillar 3 milestone (b), end-to-end: terrain → atmosphere coupling via
//! orographic lift, exercised across a real multi-step solve.
//!
//! We run two otherwise-identical coupled worlds — one with the lift coupling
//! on, one with it off — from the same uniform zonal-wind initial state, and
//! attribute the *difference* in the atmosphere bottom-layer vertical velocity
//! to the terrain. That difference must (1) stay finite (the cross-model
//! stability guard) and (2) track `w = u_h·∇h`: air rises over windward slopes
//! and sinks in the lee.

use aer::LiftSite;
use nexus::{FieldStorage, SoaField};
use sandbox::{SANDBOX_WORLD_ID, build_terrain_world_configured};
use terra::TerrainSample;
use tessera::geo::GeoCoord;
use utility::domain::{
  CellId, FieldKey, FieldName, MeshKey, SurfaceClass, SystemId,
};

fn state_key() -> FieldKey {
  FieldKey::new(MeshKey::ATMOSPHERE, FieldName::EulerState)
}

/// A steep meridional mountain ridge on the prime meridian: a Gaussian in
/// longitude, wide enough to resolve on the grid. An eastward wind climbs the
/// windward (west) face and descends the lee (east) face, so the sign of the
/// induced vertical velocity flips across the crest.
fn ridge(geo: GeoCoord) -> TerrainSample {
  let elevation = 6000.0 * (-(geo.lon / 0.35).powi(2)).exp();
  TerrainSample {
    elevation,
    class: SurfaceClass::Land,
  }
}

/// Seed a uniform world-frame momentum (+y, purely zonal along the equator)
/// across every atmosphere cell.
fn seed_zonal_wind(aether: &mut aether::core::Aether, speed: f64) {
  let world = aether
    .system_mut(SystemId(0))
    .unwrap()
    .world_mut(SANDBOX_WORLD_ID)
    .unwrap();
  let state: &mut SoaField<6> = world.pleroma_mut().write(state_key()).unwrap();
  for i in 0..state.len() {
    let cur = state.state(CellId::from(i));
    let rho = cur[0];
    let m = [0.0, rho * speed, 0.0];
    let ke = 0.5 * (m[0] * m[0] + m[1] * m[1] + m[2] * m[2]) / rho;
    state.write(
      CellId::from(i),
      &[rho, m[0], m[1], m[2], cur[4] + ke, cur[5]],
    );
  }
}

fn run(aether: &mut aether::core::Aether, dt: f64, steps: usize) {
  for step in 1..=steps {
    aether
      .step(dt)
      .unwrap_or_else(|e| panic!("step {step} failed: {e:?}"));
  }
}

/// Radial velocity of cell `c` in the given state.
fn radial_velocity(state: &SoaField<6>, site: &LiftSite) -> f64 {
  let s = state.state(site.target);
  let rho = s[0];
  (s[1] * site.r_hat[0] + s[2] * site.r_hat[1] + s[3] * site.r_hat[2]) / rho
}

#[test]
fn orographic_lift_is_stable_and_tracks_terrain_slope() {
  let dt = 2.0;
  let steps = 3;
  let speed = 20.0;

  // Identical worlds (same ridge), lift on vs off.
  let (mut on, _l1, sites) =
    build_terrain_world_configured(0.5, ridge).unwrap();
  let (mut off, _l2, _s2) = build_terrain_world_configured(0.0, ridge).unwrap();
  seed_zonal_wind(&mut on, speed);
  seed_zonal_wind(&mut off, speed);
  run(&mut on, dt, steps);
  run(&mut off, dt, steps);

  let on_world = on.world(SANDBOX_WORLD_ID).unwrap();
  let off_world = off.world(SANDBOX_WORLD_ID).unwrap();
  let on_state: &SoaField<6> = on_world.pleroma().read(state_key()).unwrap();
  let off_state: &SoaField<6> = off_world.pleroma().read(state_key()).unwrap();

  // (1) Stability: the lift-coupled world stays finite everywhere.
  for i in 0..on_state.len() {
    let s = on_state.state(CellId::from(i));
    assert!(
      (0..6).all(|k| s[k].is_finite()),
      "non-finite atmosphere state at cell {i}"
    );
  }

  // (2) Attribute the bottom-layer vertical-velocity difference to terrain. The
  // reference target w = u_h·∇h uses the lift-off ("ambient") wind so it is
  // independent of the coupling being measured. On sites with a meaningful
  // slope in the wind direction, the terrain-induced velocity change Δvr must
  // have the same sign as w: uplift over windward slopes, subsidence in the lee.
  // Per site, the ambient target w and the terrain-induced Δvr.
  let per_site: Vec<(f64, f64)> = sites
    .iter()
    .filter_map(|site| {
      let rho = off_state.state(site.target)[0];
      if !rho.is_finite() || rho <= 0.0 {
        return None;
      }
      let m = [
        off_state.state(site.target)[1],
        off_state.state(site.target)[2],
        off_state.state(site.target)[3],
      ];
      let proj = |b: &[f64; 3]| (m[0] * b[0] + m[1] * b[1] + m[2] * b[2]) / rho;
      let w = (proj(&site.east) * site.grad[0]
        + proj(&site.north) * site.grad[1])
        .clamp(-20.0, 20.0);
      let dvr =
        radial_velocity(on_state, site) - radial_velocity(off_state, site);
      Some((w, dvr))
    })
    .collect();

  let max_w = per_site.iter().fold(0.0_f64, |a, &(w, _)| a.max(w.abs()));
  let max_abs_dvr = per_site.iter().fold(0.0_f64, |a, &(_, d)| a.max(d.abs()));
  assert!(
    max_w > 1e-3,
    "ridge produced no resolvable slope (max w {max_w})"
  );
  assert!(
    max_abs_dvr > 1e-4,
    "orographic lift never moved the vertical velocity"
  );

  // On the genuinely sloped sites (windward / lee faces of the ridge), the
  // terrain-induced vertical velocity must follow the sign of u·∇h.
  let threshold = 0.3 * max_w;
  let (mut agree, mut total) = (0u32, 0u32);
  for &(w, dvr) in &per_site {
    if w.abs() > threshold {
      total += 1;
      if dvr.signum() == w.signum() {
        agree += 1;
      }
    }
  }
  assert!(total >= 4, "too few sloped sites to judge ({total})");
  let agreement = agree as f64 / total as f64;
  assert!(
    agreement > 0.75,
    "Δvr should follow u·∇h on sloped sites: {agree}/{total} = {agreement}"
  );
}
