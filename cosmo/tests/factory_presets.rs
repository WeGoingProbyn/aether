// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Sanity-check that the Sol-system factory functions return bodies whose
//! gross properties match well-known reference values, and that the system
//! container exposes them correctly.

use cosmo::{
  factory::{earth, jupiter, mars, sol, sun, venus},
  kind::BodyKind,
};
use utility::constants::{
  AU, EARTH_MASS, EARTH_RADIUS, JUPITER_MASS, MARS_MASS, SOLAR_LUMIN,
  SOLAR_MASS, VENUS_MASS,
};

#[test]
fn sun_returns_solar_kinematic_constants() {
  let s = sun();
  assert_eq!(s.mass(), SOLAR_MASS);
  assert!(matches!(s.kind(), BodyKind::Star(_)));
  assert_eq!(s.position()[0], 0.0);
  assert_eq!(s.velocity().magnitude(), 0.0);
  // Stefan–Boltzmann luminosity should match SOLAR_LUMIN to within ~1%.
  let l = s.luminosity().expect("star has luminosity");
  let rel_err = (l - SOLAR_LUMIN).abs() / SOLAR_LUMIN;
  assert!(
    rel_err < 0.01,
    "L = {}, expected ~{} (rel err {})",
    l,
    SOLAR_LUMIN,
    rel_err
  );
}

#[test]
fn earth_has_atmosphere_with_correct_eos_at_surface_temperature() {
  let e = earth();
  assert_eq!(e.mass(), EARTH_MASS);
  assert_eq!(e.radius(), EARTH_RADIUS);

  let r = e.position()[0];
  assert!(
    (r - AU).abs() < 1e-3,
    "Earth orbit radius = {} (expected AU = {})",
    r,
    AU
  );

  // Surface gravity should be close to 9.81 m/s².
  let g = e.surface_gravity();
  assert!((g - 9.81).abs() < 0.1, "g = {}", g);

  match e.kind() {
    BodyKind::RockyBody(rb) => {
      let atm = rb.atmosphere.as_ref().expect("Earth has atmosphere");
      let p = atm.properties(rb.surface_temperature);
      assert!((p.gamma - 1.4).abs() < 1e-12, "γ = {}", p.gamma);
      assert!(
        (p.gas_constant - 288.0).abs() < 1.5,
        "R_specific = {}",
        p.gas_constant
      );
    }
    _ => panic!("Earth should be a RockyBody"),
  }
}

#[test]
fn mars_has_co2_atmosphere_with_lower_gamma() {
  let m = mars();
  assert_eq!(m.mass(), MARS_MASS);
  match m.kind() {
    BodyKind::RockyBody(rb) => {
      assert!(
        rb.surface_pressure < 1000.0,
        "Mars p = {}",
        rb.surface_pressure
      );
      let atm = rb.atmosphere.as_ref().unwrap();
      let p = atm.properties(rb.surface_temperature);
      // CO₂ at 210 K → 6 DOF → γ = 4/3 ≈ 1.333.
      assert!(
        p.gamma < 1.4,
        "Mars γ = {} should be lower than diatomic 1.4",
        p.gamma
      );
      assert!((p.gamma - 4.0 / 3.0).abs() < 0.02);
    }
    _ => panic!("Mars should be a RockyBody"),
  }
}

#[test]
fn venus_rotation_is_retrograde() {
  let v = venus();
  assert_eq!(v.mass(), VENUS_MASS);
  match v.kind() {
    BodyKind::RockyBody(rb) => {
      assert!(
        rb.angular_velocity < 0.0,
        "Venus should be retrograde, ω = {}",
        rb.angular_velocity
      );
    }
    _ => panic!("Venus should be a RockyBody"),
  }
}

#[test]
fn jupiter_is_a_gas_giant_with_hydrogen_dominated_atmosphere() {
  let j = jupiter();
  assert_eq!(j.mass(), JUPITER_MASS);
  match j.kind() {
    BodyKind::GasGiant(gg) => {
      let p = gg.atmosphere.properties(gg.reference_temperature);
      // H₂/He mix should give γ between monatomic (5/3) and diatomic (7/5),
      // closer to diatomic since H₂ dominates.
      assert!(p.gamma > 1.4 && p.gamma < 1.5, "Jupiter γ = {}", p.gamma);
      // Mixture molar mass is much smaller than air (Jupiter is mostly H₂).
      assert!(p.molar_mass < 0.005, "Jupiter M = {} kg/mol", p.molar_mass);
    }
    _ => panic!("Jupiter should be a GasGiant"),
  }
}

#[test]
fn sol_contains_one_star_and_seven_planets() {
  let s = sol();
  assert_eq!(s.bodies.len(), 8);
  assert_eq!(s.planets().count(), 7);
  let star = s.star().expect("sol has a star");
  assert_eq!(star.mass(), SOLAR_MASS);
}

#[test]
fn planet_orbital_velocities_decrease_with_distance() {
  // Kepler's third law: outer planets move slower. Check a few in order.
  let speeds = [
    earth().velocity().magnitude(),
    mars().velocity().magnitude(),
    jupiter().velocity().magnitude(),
  ];
  assert!(
    speeds[0] > speeds[1],
    "Earth {} > Mars {}",
    speeds[0],
    speeds[1]
  );
  assert!(
    speeds[1] > speeds[2],
    "Mars {} > Jupiter {}",
    speeds[1],
    speeds[2]
  );
}
