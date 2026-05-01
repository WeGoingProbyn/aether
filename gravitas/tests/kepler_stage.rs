// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! End-to-end test for `KeplerStage` running through nexus + pleroma.
//! Confirms that `BodyState` registered as `ResourceKey::Bodies` is mutated
//! in place across `CompiledNexus::tick` calls and that the resulting
//! orbit lies near the analytic circular-orbit radius.

use gravitas::{BodyState, KeplerStage, NBodyGravity, PointMassBody};
use nexus::{Nexus, Pleroma, ResourceKey, WorldConstants, WorldId};
use tessera::world_mesh::Tessera;
use utility::constants::NEWTON_G;
use utility::thread::pool::Pool;

#[test]
fn kepler_stage_evolves_two_body_orbit_through_nexus() {
  let solar_mass = 1.988_47e30;
  let earth_mass = 5.972_2e24;
  let radius = 1.495_978_707e11;
  let speed = (NEWTON_G * solar_mass / radius).sqrt();
  let bodies = BodyState::<3>::from_bodies([
    PointMassBody::new(
      solar_mass,
      [0.0, 0.0, 0.0].into(),
      [0.0, 0.0, 0.0].into(),
    ),
    PointMassBody::new(
      earth_mass,
      [radius, 0.0, 0.0].into(),
      [0.0, speed, 0.0].into(),
    ),
  ]);
  let gravity = NBodyGravity::new(bodies.masses().to_vec());

  let mut pleroma = Pleroma::new();
  pleroma.register_resource(ResourceKey::Bodies, bodies);

  let mut nexus = Nexus::new();
  nexus.add(KeplerStage::<3>::new(gravity));

  let mut compiled = nexus.build(&pleroma).unwrap();
  let tessera = Tessera::new();
  let constants = WorldConstants::default();
  let pool = Pool::default();

  let day = 86_400.0;
  for _ in 0..365 {
    compiled
      .tick(WorldId(0), &tessera, &constants, &mut pleroma, &pool, day)
      .unwrap();
  }

  let bodies: &BodyState<3> =
    pleroma.read_resource(ResourceKey::Bodies).unwrap();
  let earth = bodies.body(1);
  let pos = earth.position();
  let final_radius = (pos[0] * pos[0] + pos[1] * pos[1]).sqrt();
  assert!(((final_radius - radius) / radius).abs() < 5.0e-4);
  // Time accumulated in BodyState by the stage should match.
  assert!((bodies.time() - day * 365.0).abs() < 1.0);
}

#[test]
fn kepler_stage_returns_error_when_bodies_missing() {
  let gravity = NBodyGravity::new(vec![1.0, 1.0]);

  // Pleroma without any registered resource.
  let mut pleroma = Pleroma::new();
  let mut nexus = Nexus::new();
  nexus.add(KeplerStage::<3>::new(gravity));

  let mut compiled = nexus.build(&pleroma).unwrap();
  let tessera = Tessera::new();
  let constants = WorldConstants::default();
  let pool = Pool::default();

  let result =
    compiled.tick(WorldId(0), &tessera, &constants, &mut pleroma, &pool, 1.0);
  assert!(result.is_err());
}
