// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Find the largest outer dt the *coupled* HEVI ocean world stays physical at —
//! HEVI removes the atmosphere's acoustic CFL, but the radiation/ocean coupling
//! imposes its own step limit. Diagnostic.
//!
//! `cargo test -p sandbox --release --test ocean_world_hevi_dt -- --ignored --nocapture`

use nexus::{FieldStorage, MeshKey, SoaField};
use sandbox::{SANDBOX_WORLD_ID, build_ocean_world_aether};
use utility::domain::{CellId, FieldKey, FieldName};

fn atmosphere_finite(aether: &aether::core::Aether) -> bool {
  let world = aether.world(SANDBOX_WORLD_ID).unwrap();
  let key = FieldKey::new(MeshKey::ATMOSPHERE, FieldName::EulerState);
  let Some(state) = world.pleroma().read::<SoaField<6>>(key) else {
    return false;
  };
  (0..state.len()).all(|i| {
    let s = state.state(CellId::from(i));
    s.iter().all(|v| v.is_finite()) && s[0] > 0.0 && s[4] > 0.0
  })
}

#[test]
#[ignore = "diagnostic: builds the coupled HEVI ocean world per dt; slow"]
fn coupled_hevi_stable_dt() {
  for &dt in &[60.0_f64, 30.0, 15.0, 5.0, 1.0] {
    let (mut aether, _) = build_ocean_world_aether().unwrap();
    let mut ok = true;
    let mut survived = 0;
    for _ in 0..8 {
      if aether.step(dt).is_err() || !atmosphere_finite(&aether) {
        ok = false;
        break;
      }
      survived += 1;
    }
    eprintln!(
      "dt = {dt:6.1}s: {} ({survived}/8 steps finite)",
      if ok { "STABLE" } else { "BLEW UP" }
    );
  }
}
