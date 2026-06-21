// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Longevity A/B: run the coupled ocean world for many steps under explicit vs
//! HEVI dynamics and watch for a growing high-frequency (checkerboard) mode —
//! tracked as the atmosphere density spread (max-min). Localizes whether the
//! odd-even decoupling seen in the live demo is HEVI-specific or general.
//!
//! `cargo test -p sandbox --release --test ocean_world_checkerboard -- --ignored --nocapture`

use aer::AtmosphereScheme;
use nexus::{FieldStorage, MeshKey, SoaField};
use sandbox::{SANDBOX_WORLD_ID, build_ocean_world_scheme};
use utility::domain::{CellId, FieldKey, FieldName};

fn density_spread(aether: &aether::core::Aether) -> Option<(f64, f64)> {
  let world = aether.world(SANDBOX_WORLD_ID).unwrap();
  let key = FieldKey::new(MeshKey::ATMOSPHERE, FieldName::EulerState);
  let state = world.pleroma().read::<SoaField<6>>(key)?;
  let mut min = f64::INFINITY;
  let mut max = f64::NEG_INFINITY;
  for i in 0..state.len() {
    let rho = state.state(CellId::from(i))[0];
    if !rho.is_finite() {
      return None;
    }
    min = min.min(rho);
    max = max.max(rho);
  }
  Some((min, max))
}

fn run(scheme: AtmosphereScheme, label: &str) {
  let dt = 20.0;
  let steps = 400;
  let (mut aether, _) = build_ocean_world_scheme(scheme).unwrap();
  for step in 1..=steps {
    if aether.step(dt).is_err() {
      eprintln!("[{label}] step {step}: STEP ERROR (non-physical)");
      return;
    }
    let Some((min, max)) = density_spread(&aether) else {
      eprintln!("[{label}] step {step}: NaN density");
      return;
    };
    if step % 50 == 0 || max - min > 1.0 {
      eprintln!(
        "[{label}] step {step:4} (sim {:6.0}s): rho [{min:.4}, {max:.4}] spread {:.4}",
        step as f64 * dt,
        max - min
      );
    }
    if max - min > 5.0 {
      eprintln!("[{label}] step {step}: spread runaway — checkerboard");
      return;
    }
  }
  eprintln!("[{label}] survived all {steps} steps");
}

#[test]
#[ignore = "diagnostic: long coupled run per scheme; slow"]
fn explicit_vs_hevi_checkerboard_growth() {
  run(AtmosphereScheme::Explicit, "explicit");
  run(AtmosphereScheme::Hevi, "hevi");
}
