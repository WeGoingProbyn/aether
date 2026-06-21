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

struct Stats {
  min_rho: f64,
  max_rho: f64,
  total_mass: f64,
  total_energy: f64,
  total_moisture: f64,
}

fn stats(aether: &aether::core::Aether) -> Option<Stats> {
  let world = aether.world(SANDBOX_WORLD_ID).unwrap();
  let key = FieldKey::new(MeshKey::ATMOSPHERE, FieldName::EulerState);
  let state = world.pleroma().read::<SoaField<6>>(key)?;
  let mut s = Stats {
    min_rho: f64::INFINITY,
    max_rho: f64::NEG_INFINITY,
    total_mass: 0.0,
    total_energy: 0.0,
    total_moisture: 0.0,
  };
  for i in 0..state.len() {
    let cell = state.state(CellId::from(i));
    let rho = cell[0];
    if !rho.is_finite() {
      return None;
    }
    s.min_rho = s.min_rho.min(rho);
    s.max_rho = s.max_rho.max(rho);
    s.total_mass += rho;
    s.total_energy += cell[4];
    s.total_moisture += cell[5];
  }
  Some(s)
}

fn run(scheme: AtmosphereScheme, label: &str) {
  let dt = 20.0;
  let steps = 400;
  let (mut aether, _) = build_ocean_world_scheme(scheme).unwrap();
  let m0 = stats(&aether).unwrap().total_mass;
  for step in 1..=steps {
    if aether.step(dt).is_err() {
      eprintln!("[{label}] step {step}: STEP ERROR (non-physical)");
      return;
    }
    let Some(s) = stats(&aether) else {
      eprintln!("[{label}] step {step}: NaN density");
      return;
    };
    let spread = s.max_rho - s.min_rho;
    if step % 5 == 0 || spread > 1.0 {
      eprintln!(
        "[{label}] step {step:4}: rho [{:.3},{:.3}] | mass {:.2} ({:+.1}%) energy {:.3e} moist {:.3e}",
        s.min_rho,
        s.max_rho,
        s.total_mass,
        100.0 * (s.total_mass - m0) / m0,
        s.total_energy,
        s.total_moisture,
      );
    }
    if spread > 5.0 {
      eprintln!("[{label}] step {step}: runaway");
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
