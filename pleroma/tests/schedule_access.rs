// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Scheduled access — the unsafe split-borrow path nexus uses to fan out
//! parallel stages. The test simulates one DAG layer with two stages whose
//! writes are disjoint, runs them concurrently in scoped threads, then
//! verifies both fields received their writes.

use std::thread;

use pleroma::Pleroma;
use pleroma::core::storage::{CellView, FieldStorage, SoaField};
use utility::domain::{CellId, FieldKey, FieldName, MeshKey};

const N: usize = 1;
const TEMPERATURE: FieldKey =
  FieldKey::new(MeshKey::SURFACE, FieldName::Temperature);
const PRESSURE: FieldKey = FieldKey::new(MeshKey::SURFACE, FieldName::Pressure);
const HUMIDITY: FieldKey = FieldKey::new(MeshKey::SURFACE, FieldName::Humidity);

#[test]
fn view_for_honors_declared_keys() {
  let mut world = Pleroma::new();
  world.register_field(TEMPERATURE, SoaField::<N>::zeros(2));
  world.register_field(PRESSURE, SoaField::<N>::zeros(2));

  let access = world.schedule_access();
  // SAFETY: the only view alive at any time has disjoint reads/writes.
  let mut view =
    unsafe { access.view_for(&[TEMPERATURE], &[PRESSURE], &[], &[]) };

  // Pressure was declared as a write — should resolve.
  assert!(view.write::<SoaField<N>>(PRESSURE).is_some());

  // Temperature was declared as a read only — write must reject.
  assert!(view.write::<SoaField<N>>(TEMPERATURE).is_none());
  // …but read works.
  assert!(view.read::<SoaField<N>>(TEMPERATURE).is_some());

  // Humidity wasn't declared at all — both directions reject.
  assert!(view.read::<SoaField<N>>(HUMIDITY).is_none());
  assert!(view.write::<SoaField<N>>(HUMIDITY).is_none());
}

#[test]
fn disjoint_writes_run_in_parallel() {
  let mut world = Pleroma::new();
  world.register_field(TEMPERATURE, SoaField::<N>::zeros(4));
  world.register_field(PRESSURE, SoaField::<N>::zeros(4));

  let access = world.schedule_access();

  // SAFETY: the two views write to *different* keys; reads/writes are
  // pairwise disjoint, so simultaneous typed mutation is sound.
  let mut view_t = unsafe { access.view_for(&[], &[TEMPERATURE], &[], &[]) };
  let mut view_p = unsafe { access.view_for(&[], &[PRESSURE], &[], &[]) };

  thread::scope(|s| {
    s.spawn(move || {
      let f = view_t.write::<SoaField<N>>(TEMPERATURE).unwrap();
      for i in 0..4 {
        f.write(CellId::from(i), &[i as f64 + 100.0]);
      }
    });
    s.spawn(move || {
      let f = view_p.write::<SoaField<N>>(PRESSURE).unwrap();
      for i in 0..4 {
        f.write(CellId::from(i), &[i as f64 + 200.0]);
      }
    });
  });

  drop(access);

  let t = world.read::<SoaField<N>>(TEMPERATURE).unwrap();
  let p = world.read::<SoaField<N>>(PRESSURE).unwrap();
  for i in 0..4 {
    assert_eq!(t.state(CellId::from(i)).as_state(), &[i as f64 + 100.0]);
    assert_eq!(p.state(CellId::from(i)).as_state(), &[i as f64 + 200.0]);
  }
}
