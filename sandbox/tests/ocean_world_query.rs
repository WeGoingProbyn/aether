// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! End-to-end check of the semantic query API (Pillar 1) against the live
//! ocean-world demo: build the world, step it so the diagnostic fields evolve,
//! snapshot the quantity channels, and confirm `WorldQuery` returns finite,
//! physically plausible values that match a direct field read — exercising the
//! whole chain field → snapshot channel → geographic locate → sample.

use aer::AtmosphereScheme;
use eidolon::extract::{default_atmosphere_quantities, extract_quantity_frame};
use eidolon::playback::FrameInterpolator;
use eidolon::query::{Sample, ScalarQuantity, WorldQuery};
use nexus::{FieldStorage, SoaField};
use sandbox::{SANDBOX_WORLD_ID, build_ocean_world_scheme};
use tessera::geo::GeoCoord;
use utility::domain::{FieldKey, FieldName, MeshKey, MeshType};

#[test]
fn ocean_world_semantic_queries_return_physical_values() {
  let (mut aether, shell) =
    build_ocean_world_scheme(AtmosphereScheme::Hevi).unwrap();
  let surface_radius = shell.reference_radius();
  let channels = default_atmosphere_quantities();
  let dt = 20.0;

  // Spin a few ticks so the diagnostic fields reflect evolved state.
  for _ in 0..3 {
    aether.step(dt).unwrap();
  }

  // Build the query view once from the static mesh geometry (owns its indices,
  // so it does not borrow the world).
  let query = {
    let world = aether.world(SANDBOX_WORLD_ID).unwrap();
    WorldQuery::new(world.tessera(), surface_radius)
  };

  // One snapshot frame from the current live diagnostic fields.
  let mut snapshot = FrameInterpolator::new();
  let frame = {
    let world = aether.world(SANDBOX_WORLD_ID).unwrap();
    extract_quantity_frame(world.pleroma(), 3.0 * dt, &channels)
  };
  snapshot.push(frame);

  let at = GeoCoord::from_degrees(12.0, 34.0, 1000.0);

  // Temperature is available, finite, and physically plausible.
  let temperature =
    query.sample_scalar(&snapshot, ScalarQuantity::Temperature, at);
  let t = temperature
    .value()
    .expect("temperature should be available");
  assert!(
    t.is_finite() && t > 150.0 && t < 400.0,
    "temperature {t} K out of plausible range"
  );

  // A single buffered frame cannot be interpolated, so the value is served by
  // snapping — flagged Stale, not Ok. (Documents the quality contract.)
  assert!(
    matches!(temperature, Sample::Stale(_)),
    "single-frame snapshot should be Stale, got {temperature:?}"
  );

  // Cross-check the full pipeline: the sampled value equals a direct read of
  // the field at the cell the query locates (exact, since no interpolation).
  {
    let world = aether.world(SANDBOX_WORLD_ID).unwrap();
    let cell = query.locate(MeshType::Atmosphere, at).expect("on the mesh");
    let field = world
      .pleroma()
      .read::<SoaField<1>>(FieldKey::new(
        MeshKey::ATMOSPHERE,
        FieldName::Temperature,
      ))
      .expect("temperature field exists");
    let direct = field.state(cell)[0];
    assert!(
      (t - direct).abs() < 1e-9,
      "sampled {t} != direct field read {direct}"
    );
  }

  // Wind resolves to a finite east-north-up vector.
  let wind = query
    .sample_wind(&snapshot, at)
    .value()
    .expect("wind should be available");
  assert!(
    wind.iter().all(|c| c.is_finite()),
    "wind ENU not finite: {wind:?}"
  );

  // Pressure and humidity are wired through the same path and finite.
  for q in [ScalarQuantity::Pressure, ScalarQuantity::Humidity] {
    let v = query
      .sample_scalar(&snapshot, q, at)
      .value()
      .unwrap_or_else(|| panic!("{q:?} should be available"));
    assert!(v.is_finite(), "{q:?} = {v} not finite");
  }
}
