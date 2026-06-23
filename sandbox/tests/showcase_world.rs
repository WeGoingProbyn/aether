// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Stability smoke test for the fully-coupled showcase world: terrain + ocean +
//! moist atmosphere, all couplings live (radiation with terrain albedo, air-sea
//! water cycle, orographic lift). It must assemble and advance many steps with
//! every field staying finite — the foundation the render showcase and future
//! end-to-end tests build on.

use eidolon::extract::FrameProducer;
use eidolon::ir::{LayerKind, Update};
use nexus::{FieldStorage, SoaField};
use sandbox::{
  SANDBOX_WORLD_ID, build_showcase_world, showcase_extract_config,
};
use utility::domain::{CellId, FieldKey, FieldName, MeshKey};

fn all_finite_scalar(
  aether: &aether::core::Aether,
  mesh: MeshKey,
  field: FieldName,
) -> bool {
  let world = aether.world(SANDBOX_WORLD_ID).unwrap();
  let f: &SoaField<1> = world
    .pleroma()
    .read(FieldKey::new(mesh, field))
    .unwrap_or_else(|| panic!("{mesh:?}/{field:?} should be registered"));
  (0..f.len()).all(|i| f.state(CellId::from(i))[0].is_finite())
}

#[test]
fn showcase_world_is_stable_over_many_steps() {
  let (mut aether, _layout) = build_showcase_world().unwrap();

  // The terrain fields are present and finite from the start.
  assert!(all_finite_scalar(
    &aether,
    MeshKey::SURFACE,
    FieldName::SurfaceElevation
  ));
  assert!(all_finite_scalar(
    &aether,
    MeshKey::SURFACE,
    FieldName::SurfaceType
  ));
  assert!(all_finite_scalar(
    &aether,
    MeshKey::OCEAN,
    FieldName::SurfaceAlbedo
  ));

  let dt = 20.0;
  for step in 1..=30 {
    aether.step(dt).unwrap_or_else(|e| {
      panic!("step {step} failed: display=[{e}] debug=[{e:#?}]")
    });

    // Atmosphere conserved state finite.
    let world = aether.world(SANDBOX_WORLD_ID).unwrap();
    let euler: &SoaField<6> = world
      .pleroma()
      .read(FieldKey::new(MeshKey::ATMOSPHERE, FieldName::EulerState))
      .unwrap();
    for i in 0..euler.len() {
      let s = euler.state(CellId::from(i));
      assert!(
        (0..6).all(|k| s[k].is_finite()),
        "non-finite atmosphere state at step {step}, cell {i}"
      );
      assert!(s[0] > 0.0, "non-positive density at step {step}, cell {i}");
    }
  }

  // Ocean temperature and atmosphere humidity stay finite and physical after
  // the full air-sea cycle has run.
  assert!(all_finite_scalar(
    &aether,
    MeshKey::OCEAN,
    FieldName::Temperature
  ));
  assert!(all_finite_scalar(
    &aether,
    MeshKey::ATMOSPHERE,
    FieldName::Humidity
  ));
}

/// The render config gives a consumer the art-free terrain data it needs: a
/// categorical land/ocean/ice layer and an elevation data layer on the surface,
/// plus the atmosphere overlay fields.
#[test]
fn showcase_producer_emits_terrain_and_atmosphere_layers() {
  let (mut aether, _layout) = build_showcase_world().unwrap();
  aether.step(20.0).unwrap();

  let mut producer = FrameProducer::new(showcase_extract_config());
  let world = aether.world(SANDBOX_WORLD_ID).unwrap();
  let batch = producer.extract(
    SANDBOX_WORLD_ID,
    world.tessera(),
    world.pleroma(),
    None,
    20.0,
    1,
  );

  let has_categorical = batch.updates.iter().any(|u| {
    matches!(
      u,
      Update::RegisterLayer {
        kind: LayerKind::Categorical { .. },
        ..
      }
    )
  });
  assert!(
    has_categorical,
    "surface land/ocean/ice categorical layer missing"
  );

  let scalar_layers = batch
    .updates
    .iter()
    .filter(|u| {
      matches!(
        u,
        Update::RegisterLayer {
          kind: LayerKind::Scalar { .. },
          ..
        }
      )
    })
    .count();
  // elevation + albedo + ocean SST + atmosphere temp/humidity/pressure.
  assert!(
    scalar_layers >= 6,
    "expected the terrain + atmosphere scalar layers, got {scalar_layers}"
  );
}
