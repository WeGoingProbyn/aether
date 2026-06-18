// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Phase 0A: subsystem cadence metadata. These assert the build-time
//! plumbing only — execution stays single-rate until the multirate driver
//! (Phase 0B) consults it. A nexus with no overrides must look exactly
//! like today: one subsystem, no cadences, `is_multirate() == false`.

use nexus::{Nexus, Stage, StageContext, SubsystemId};
use pleroma::Pleroma;
use pleroma::prelude::{FieldKey, FieldName, MeshKey};
use utility::error::AetherResult;

const ATM_STATE: FieldKey =
  FieldKey::new(MeshKey::ATMOSPHERE, FieldName::EulerState);
const SURFACE_TEMP: FieldKey =
  FieldKey::new(MeshKey::SURFACE, FieldName::Temperature);

struct Tagged {
  name: &'static str,
  writes: Vec<FieldKey>,
  subsystem: SubsystemId,
}

impl Stage for Tagged {
  fn name(&self) -> &'static str {
    self.name
  }
  fn reads(&self) -> &[FieldKey] {
    &[]
  }
  fn writes(&self) -> &[FieldKey] {
    &self.writes
  }
  fn subsystem(&self) -> SubsystemId {
    self.subsystem
  }
  fn run(&mut self, _ctx: StageContext<'_>) -> AetherResult<()> {
    Ok(())
  }
}

fn tagged(
  name: &'static str,
  write: FieldKey,
  subsystem: SubsystemId,
) -> Tagged {
  Tagged {
    name,
    writes: vec![write],
    subsystem,
  }
}

#[test]
fn default_nexus_is_single_rate() {
  let mut nexus = Nexus::new();
  nexus.add(tagged("a", ATM_STATE, SubsystemId::DEFAULT));
  nexus.add(tagged("b", SURFACE_TEMP, SubsystemId::DEFAULT));

  let compiled = nexus.build(&Pleroma::new()).unwrap();
  assert_eq!(compiled.subsystems(), vec![SubsystemId::DEFAULT]);
  assert!(!compiled.is_multirate());
  assert_eq!(compiled.cadence(SubsystemId::DEFAULT), None);
}

#[test]
fn build_records_per_stage_subsystem_and_cadences() {
  const ATMOSPHERE: SubsystemId = SubsystemId(1);
  const OCEAN: SubsystemId = SubsystemId(2);

  let mut nexus = Nexus::new();
  let atm = nexus.add(tagged("atmosphere", ATM_STATE, ATMOSPHERE));
  let ocean = nexus.add(tagged("ocean", SURFACE_TEMP, OCEAN));
  nexus.set_subsystem_cadence(ATMOSPHERE, 1.0);
  nexus.set_subsystem_cadence(OCEAN, 3600.0);

  let compiled = nexus.build(&Pleroma::new()).unwrap();

  assert_eq!(compiled.subsystem_of(atm), ATMOSPHERE);
  assert_eq!(compiled.subsystem_of(ocean), OCEAN);
  assert_eq!(compiled.subsystems(), vec![ATMOSPHERE, OCEAN]);
  assert_eq!(compiled.cadence(ATMOSPHERE), Some(1.0));
  assert_eq!(compiled.cadence(OCEAN), Some(3600.0));
  assert!(compiled.is_multirate());
}
