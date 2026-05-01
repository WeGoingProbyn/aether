// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Direct (non-scheduled) access to pleroma's non-mesh-bound resource API.
//! Verifies `register_resource` + `read_resource::<R>` + `write_resource::<R>`
//! round-trip correctly and that type / key mismatches surface as `None`
//! rather than UB.

use pleroma::Pleroma;
use utility::domain::ResourceKey;

#[derive(Clone, Debug, PartialEq)]
struct Bodies {
  positions: Vec<[f64; 3]>,
}

#[derive(Clone, Debug, PartialEq)]
struct SunDirection([f64; 3]);

#[test]
fn register_then_read_round_trip() {
  let mut world = Pleroma::new();
  world.register_resource(
    ResourceKey::Bodies,
    Bodies {
      positions: vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    },
  );

  let bodies: &Bodies = world
    .read_resource(ResourceKey::Bodies)
    .expect("registered");
  assert_eq!(bodies.positions, vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
}

#[test]
fn write_mutates_resource_in_place() {
  let mut world = Pleroma::new();
  world.register_resource(
    ResourceKey::Bodies,
    Bodies {
      positions: vec![[0.0, 0.0, 0.0]],
    },
  );

  {
    let bodies: &mut Bodies =
      world.write_resource(ResourceKey::Bodies).unwrap();
    bodies.positions[0] = [10.0, 20.0, 30.0];
  }

  let bodies: &Bodies = world.read_resource(ResourceKey::Bodies).unwrap();
  assert_eq!(bodies.positions[0], [10.0, 20.0, 30.0]);
}

#[test]
fn type_mismatch_returns_none() {
  let mut world = Pleroma::new();
  world
    .register_resource(ResourceKey::SunPosition, SunDirection([0.0, 0.0, 1.0]));

  // Wrong type — TypeId catches it.
  assert!(
    world
      .read_resource::<Bodies>(ResourceKey::SunPosition)
      .is_none()
  );
  // Right type — fine.
  assert!(
    world
      .read_resource::<SunDirection>(ResourceKey::SunPosition)
      .is_some()
  );
}

#[test]
fn missing_key_returns_none() {
  let world = Pleroma::new();
  assert!(
    world
      .read_resource::<Bodies>(ResourceKey::PlanetSpin)
      .is_none()
  );
}

#[test]
fn fields_and_resources_are_independent_namespaces() {
  // Same Pleroma can hold both, registered under their own keyspaces.
  let mut world = Pleroma::new();
  world.register_resource(
    ResourceKey::Bodies,
    Bodies {
      positions: vec![[1.0; 3]],
    },
  );
  world
    .register_resource(ResourceKey::SunPosition, SunDirection([0.0, 1.0, 0.0]));

  assert!(world.read_resource::<Bodies>(ResourceKey::Bodies).is_some());
  assert!(
    world
      .read_resource::<SunDirection>(ResourceKey::SunPosition)
      .is_some()
  );
}
