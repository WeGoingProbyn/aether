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
use utility::diagnostics::DiagnosticsPolicy;
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

/// Land–sea masking is live end-to-end: the tessera ocean cell-mask exists and
/// stops the ocean solver evolving land columns (they stay at their inert initial
/// temperature), while active ocean columns evolve.
#[test]
fn showcase_masks_the_ocean_solver_to_open_water() {
  let (mut aether, _layout) = build_showcase_world().unwrap();
  let ocean_temp = FieldKey::new(MeshKey::OCEAN, FieldName::Temperature);

  // The mask was built alongside the couplers, with both land and ocean columns.
  let initial: Vec<f64> = {
    let world = aether.world(SANDBOX_WORLD_ID).unwrap();
    let mask = world
      .tessera()
      .cell_mask(MeshKey::OCEAN)
      .expect("showcase builds the ocean cell mask");
    assert!(
      mask.active_count() > 0 && mask.inactive_count() > 0,
      "the ocean mask must mark both ocean ({}) and land ({}) cells",
      mask.active_count(),
      mask.inactive_count()
    );
    let f: &SoaField<1> = world.pleroma().read(ocean_temp).unwrap();
    (0..f.len()).map(|i| f.state(CellId::from(i))[0]).collect()
  };

  for _ in 0..10 {
    aether.step(20.0).unwrap();
  }

  let world = aether.world(SANDBOX_WORLD_ID).unwrap();
  let mask = world.tessera().cell_mask(MeshKey::OCEAN).unwrap();
  let temp: &SoaField<1> = world.pleroma().read(ocean_temp).unwrap();
  let mut any_active_changed = false;
  for i in 0..temp.len() {
    let cell = CellId::from(i);
    let t = temp.state(cell)[0];
    if mask.is_active(cell) {
      if (t - initial[i]).abs() > 1e-9 {
        any_active_changed = true;
      }
    } else {
      assert_eq!(
        t, initial[i],
        "masked land ocean-cell {i} must stay at its inert initial temperature"
      );
    }
  }
  assert!(
    any_active_changed,
    "active ocean columns should evolve under radiation / SST coupling"
  );
}

/// Evaporation is gated to open water: land cells (moisture availability 0) inject
/// no vapour, so mean evaporation over ocean strictly exceeds land.
#[test]
fn showcase_gates_evaporation_to_ocean() {
  let (mut aether, _layout) = build_showcase_world().unwrap();
  for _ in 0..1 {
    aether.step(20.0).unwrap();
  }

  let world = aether.world(SANDBOX_WORLD_ID).unwrap();
  let avail: &SoaField<1> = world
    .pleroma()
    .read(FieldKey::new(
      MeshKey::ATMOSPHERE,
      FieldName::MoistureAvailability,
    ))
    .unwrap();
  let evap: &SoaField<1> = world
    .pleroma()
    .read(FieldKey::new(
      MeshKey::ATMOSPHERE,
      FieldName::EvaporationFlux,
    ))
    .unwrap();

  let (mut ocean_sum, mut ocean_n, mut land_sum, mut land_n) =
    (0.0, 0usize, 0.0, 0usize);
  for i in 0..evap.len() {
    let cell = CellId::from(i);
    let e = evap.state(cell)[0];
    if avail.state(cell)[0] > 0.5 {
      ocean_sum += e;
      ocean_n += 1;
    } else {
      land_sum += e;
      land_n += 1;
    }
  }
  assert!(
    ocean_n > 0 && land_n > 0,
    "both ocean and land cells present"
  );
  assert_eq!(land_sum, 0.0, "land injects no evaporation (gated to zero)");
  assert!(
    ocean_sum > 0.0,
    "open ocean evaporates on the first step (before the air saturates)"
  );
  assert!(
    ocean_sum / ocean_n as f64 > land_sum / land_n as f64,
    "ocean must evaporate more than land (ocean {ocean_sum}, land {land_sum})"
  );
}

/// The showcase enables the in-DAG conservation monitor, so after stepping the
/// world publishes a health report (what the demo runner prints periodically):
/// a finite-state, finite-conserved-totals report for the atmosphere Euler
/// state under the default Warn policy.
#[test]
fn showcase_world_publishes_runtime_diagnostics() {
  let (mut aether, _layout) = build_showcase_world().unwrap();
  for _ in 0..5 {
    aether.step(20.0).unwrap();
  }

  let world = aether.world(SANDBOX_WORLD_ID).unwrap();
  let diagnostics = world
    .diagnostics()
    .expect("showcase world registers the Diagnostics resource");
  assert_eq!(diagnostics.policy, DiagnosticsPolicy::Warn);
  assert!(
    !diagnostics.has_non_finite(),
    "stable showcase has no NaN/Inf"
  );

  let euler_state = FieldKey::new(MeshKey::ATMOSPHERE, FieldName::EulerState);
  let report = diagnostics
    .fields
    .get(&euler_state)
    .expect("monitor published the atmosphere Euler-state report");
  assert_eq!(report.non_finite_cells, 0);
  assert_eq!(report.conserved.len(), 6);
  assert!(report.conserved.iter().all(|(_, total)| total.is_finite()));
}

/// AMR is live in the showcase: the surface mesh refines a panel-interior cap
/// after the adapter's cadence elapses, broadcasting a `TopologyChanged`, and the
/// producer emits the cell-outline wireframe (line geometry) that visualises it.
#[test]
fn showcase_refines_the_surface_and_emits_a_wireframe() {
  use eidolon::ir::RenderGeometry;
  use tessera::geometry::CellGeometry;
  use utility::events::Event;

  use aether::adapt::CameraView;
  use utility::domain::SystemId;

  let (mut aether, layout) = build_showcase_world().unwrap();
  let initial = aether
    .world(SANDBOX_WORLD_ID)
    .unwrap()
    .tessera()
    .mesh(MeshKey::SURFACE)
    .unwrap()
    .cell_count();

  // The surface uses *view-dependent* LOD, so it only refines once the host has
  // placed the camera. Put it above the +z pole (panel-interior, clear of seams),
  // so the near cells refine.
  let radius = layout.reference_radius();
  aether
    .system_mut(SystemId(0))
    .and_then(|s| s.world_mut(SANDBOX_WORLD_ID))
    .unwrap()
    .set_camera(CameraView {
      position: [0.0, 0.0, radius * 2.5],
    });

  // The surface adapter fires on a 15-tick cadence; step exactly to the first
  // firing so the TopologyChanged event is in this tick's buffer.
  for _ in 0..15 {
    aether.step(20.0).unwrap();
  }

  let world = aether.world(SANDBOX_WORLD_ID).unwrap();
  let refined = world.tessera().mesh(MeshKey::SURFACE).unwrap().cell_count();
  assert!(
    refined > initial,
    "AMR should refine the surface cap: {initial} -> {refined}"
  );

  // The topology change was broadcast on the tick it happened.
  assert!(
    world.events().iter().any(|e| matches!(
      e,
      Event::TopologyChanged { mesh, .. } if *mesh == MeshKey::SURFACE
    )),
    "expected a TopologyChanged event for the refined surface, got {:?}",
    world.events()
  );

  // The producer emits the surface cell-outline wireframe (a line mesh).
  let mut producer = FrameProducer::new(showcase_extract_config());
  let batch = producer.extract(
    SANDBOX_WORLD_ID,
    world.tessera(),
    world.pleroma(),
    None,
    0.0,
    0,
  );
  assert!(
    batch.updates.iter().any(|u| matches!(
      u,
      Update::RegisterMesh {
        geometry: RenderGeometry::Lines(_),
        ..
      }
    )),
    "expected a wireframe (line) mesh for the surface cell outlines"
  );

  // The simulation-owned camera is emitted *forward* into the IR (the backend
  // positions its view from it). This is the same view that drove the LOD above.
  assert!(
    batch.updates.iter().any(|u| matches!(
      u,
      Update::SetCamera { camera }
        if (camera.position[2] - radius * 2.5).abs() < 1.0
    )),
    "expected the simulation camera to be emitted forward as SetCamera"
  );
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
  // elevation + albedo + atmosphere temp/humidity/pressure.
  assert!(
    scalar_layers >= 5,
    "expected the terrain + atmosphere scalar layers, got {scalar_layers}"
  );
}
