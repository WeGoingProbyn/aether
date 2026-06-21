// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Diagnostic: at the demo's dt, how much do the surface-visible fields
//! and the sun direction actually move over a few hundred ticks? Run with
//! `cargo test -p sandbox --test ocean_world_dynamics -- --nocapture`.

use nexus::{FieldStorage, MeshKey, ResourceKey, SoaField};
use sandbox::{SANDBOX_WORLD_ID, build_ocean_world_aether};
use utility::domain::{CellId, FieldKey, FieldName};

fn spread(world: &aether::core::World, key: FieldKey) -> (f64, f64, f64) {
  let f: &SoaField<1> = world.pleroma().read(key).unwrap();
  let (mut min, mut max, mut sum) = (f64::INFINITY, f64::NEG_INFINITY, 0.0);
  for i in 0..f.len() {
    let v = f.state(CellId::from(i))[0];
    min = min.min(v);
    max = max.max(v);
    sum += v;
  }
  (min, max, sum / f.len() as f64)
}

// Slow (builds + steps the real demo world); opt in with
// `cargo test -p sandbox --test ocean_world_dynamics -- --ignored --nocapture`.
#[test]
#[ignore = "diagnostic: builds + steps the full demo world; ~15s"]
fn report_field_evolution_at_demo_dt() {
  let dt = 20.0_f64;
  let steps = 8;
  let ocean_t = FieldKey::new(MeshKey::OCEAN, FieldName::Temperature);
  let humidity = FieldKey::new(MeshKey::ATMOSPHERE, FieldName::Humidity);

  let (mut aether, _) = build_ocean_world_aether().unwrap();
  let sun0 = *aether
    .world(SANDBOX_WORLD_ID)
    .unwrap()
    .pleroma()
    .read_resource::<[f64; 3]>(ResourceKey::SunPosition)
    .unwrap();

  for _ in 0..steps {
    aether.step(dt).unwrap();
  }

  let world = aether.world(SANDBOX_WORLD_ID).unwrap();
  let sun1 = *world
    .pleroma()
    .read_resource::<[f64; 3]>(ResourceKey::SunPosition)
    .unwrap();
  let (o_min, o_max, o_mean) = spread(world, ocean_t);
  let (h_min, h_max, h_mean) = spread(world, humidity);

  eprintln!(
    "after {steps} steps @ dt={dt} (sim {:.1}s):",
    dt * steps as f64
  );
  eprintln!("  sun: {sun0:?} -> {sun1:?}");
  eprintln!(
    "  ocean T:  min {o_min:.5} max {o_max:.5} spread {:.3e} mean {o_mean:.4}",
    o_max - o_min
  );
  eprintln!(
    "  humidity: min {h_min:.3e} max {h_max:.3e} spread {:.3e} mean {h_mean:.3e}",
    h_max - h_min
  );
}
