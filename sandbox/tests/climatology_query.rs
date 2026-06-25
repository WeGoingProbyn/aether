// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! End-to-end check of the climatology (Pillar 2) read path against the live
//! showcase world: the chronos accumulators EMA-aggregate the atmosphere
//! diagnostics on their own slow subsystem, and a consumer reads the resulting
//! slowly-varying means through the same semantic query API as live fields —
//! exercising mean field → snapshot channel → geographic locate → sample.

use eidolon::extract::{
  default_climatology_quantities, extract_quantity_frame,
};
use eidolon::playback::FrameInterpolator;
use eidolon::query::{Sample, ScalarQuantity, WorldQuery};
use nexus::{FieldStorage, SoaField};
use sandbox::{SANDBOX_WORLD_ID, build_showcase_world};
use tessera::geo::GeoCoord;
use utility::domain::{FieldKey, FieldName, MeshKey, MeshType};

#[test]
fn showcase_world_exposes_climatology_means() {
  let (mut aether, shell) = build_showcase_world().unwrap();
  let surface_radius = shell.reference_radius();
  let channels = default_climatology_quantities();
  let dt = 20.0;

  // A few outer ticks: the atmosphere refreshes its diagnostics and the
  // climatology subsystem EMAs toward them each tick.
  for _ in 0..4 {
    aether.step(dt).unwrap();
  }

  let query = {
    let world = aether.world(SANDBOX_WORLD_ID).unwrap();
    WorldQuery::new(world.tessera(), surface_radius)
  };

  let mut snapshot = FrameInterpolator::new();
  let frame = {
    let world = aether.world(SANDBOX_WORLD_ID).unwrap();
    extract_quantity_frame(world.pleroma(), 4.0 * dt, &channels)
  };
  snapshot.push(frame);

  let at = GeoCoord::from_degrees(12.0, 34.0, 1000.0);

  // Mean temperature is available, finite, and physically plausible (it seeds
  // from the hydrostatic temperature and EMAs toward the evolving live field).
  let mean_t =
    query.sample_scalar(&snapshot, ScalarQuantity::MeanTemperature, at);
  let t = mean_t
    .value()
    .expect("mean temperature should be available");
  assert!(
    t.is_finite() && t > 150.0 && t < 400.0,
    "mean temperature {t} K out of plausible range"
  );
  // Single buffered frame → served by snapping, flagged Stale (quality contract).
  assert!(
    matches!(mean_t, Sample::Stale(_)),
    "single-frame snapshot should be Stale, got {mean_t:?}"
  );

  // Full-pipeline cross-check: the sampled mean equals a direct read of the
  // MeanTemperature field at the cell the query locates (exact, no interpolation).
  {
    let world = aether.world(SANDBOX_WORLD_ID).unwrap();
    let cell = query.locate(MeshType::Atmosphere, at).expect("on the mesh");
    let field = world
      .pleroma()
      .read::<SoaField<1>>(FieldKey::new(
        MeshKey::ATMOSPHERE,
        FieldName::MeanTemperature,
      ))
      .expect("mean temperature field exists");
    let direct = field.state(cell)[0];
    assert!(
      (t - direct).abs() < 1e-9,
      "sampled {t} != direct field read {direct}"
    );
  }

  // Liveness: MeanPressure seeds from the zero-initialised pressure field, so a
  // strictly positive mean after stepping proves the accumulator actually ran
  // and aggregated the (now non-zero) live pressure diagnostic.
  let mean_p = query
    .sample_scalar(&snapshot, ScalarQuantity::MeanPressure, at)
    .value()
    .expect("mean pressure should be available");
  assert!(
    mean_p.is_finite() && mean_p > 0.0,
    "mean pressure {mean_p} did not aggregate from zero"
  );

  // Mean humidity is wired through the same path and finite.
  let mean_q = query
    .sample_scalar(&snapshot, ScalarQuantity::MeanHumidity, at)
    .value()
    .expect("mean humidity should be available");
  assert!(mean_q.is_finite(), "mean humidity {mean_q} not finite");

  // A quantity with no climatology source (SST climatology is not wired in this
  // world) reports Unavailable rather than fabricating a value.
  assert!(
    query
      .sample_scalar(&snapshot, ScalarQuantity::MeanSeaSurfaceTemperature, at)
      .is_unavailable(),
    "unwired climatology quantity should be Unavailable"
  );
}
