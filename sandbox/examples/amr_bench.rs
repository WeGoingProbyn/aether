// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Headless timing rig for the showcase world's adapt barrier.
//!
//! Mirrors what the bevy showcase runner does each tick — advance the world,
//! move the LOD focus, extract a render batch — without a window, so the
//! profiler numbers are comparable to a `cargo run -p sandbox` capture but
//! reproducible on a machine with no GPU.
//!
//! Run it in release; a debug build measures the compiler, not the code:
//!
//! ```text
//! cargo run --release -p sandbox --example amr_bench
//! ```
//!
//! Read `run_adapters.refine` (the re-mesh) and `extract` (the render-batch
//! build) — those are the two costs that land on the simulation tick and so
//! bound how often the renderer gets a fresh frame.

use std::f64::consts::TAU;

use aether::adapt::RefinementFocus;
use eidolon::extract::FrameProducer;
use sandbox::{
  SANDBOX_WORLD_ID, build_showcase_world, showcase_extract_config,
};
use utility::domain::SystemId;
use utility::error::AetherResult;
use utility::logger::{Level, Logger, StdSink};
use utility::profiler::Profiler;

/// Matches the showcase runner's outer step.
const TICK_DT: f64 = 20.0;

/// Enough ticks for the surface adapter (15-tick cadence) to fire ~20 times,
/// which is the same order as the capture this rig was built to compare against.
const TICKS: usize = 320;

fn main() -> AetherResult<()> {
  Logger::init(
    vec![Box::new(StdSink::new(std::io::stdout()).capacity(1))],
    Level::Debug,
  );
  Profiler::init();

  let (mut aether, layout) = build_showcase_world()?;
  let mut producer = FrameProducer::new(showcase_extract_config());
  let radius = layout.reference_radius() * 2.5;

  for tick in 0..TICKS {
    // Orbit the focus so the LOD set genuinely churns — a stationary focus
    // refines once and then costs nothing, which would flatter the measurement.
    let azimuth = (tick as f64 / TICKS as f64) * TAU;
    let colatitude: f64 = 0.2;
    let (sc, cc) = colatitude.sin_cos();
    let (sa, ca) = azimuth.sin_cos();
    let eye = [radius * sc * ca, radius * sc * sa, radius * cc];

    aether
      .system_mut(SystemId(0))
      .and_then(|s| s.world_mut(SANDBOX_WORLD_ID))
      .expect("sandbox world should be registered")
      .set_refinement_focus(RefinementFocus { position: eye });

    aether.advance(TICK_DT)?;

    let world = aether
      .world(SANDBOX_WORLD_ID)
      .expect("sandbox world should be registered");
    let _batch = producer.extract(
      SANDBOX_WORLD_ID,
      world.tessera(),
      world.pleroma(),
      None,
      world.sim_time(),
      tick as u64,
    );
  }

  Profiler::flush_local();
  Profiler::print(&mut std::io::stdout());
  Ok(())
}
