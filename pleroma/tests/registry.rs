// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Direct (non-scheduled) access to the Pleroma registry. Verifies that
//! `register_field` + `read::<S>` + `write::<S>` round-trip correctly and
//! that type and key mismatches surface as `None` rather than UB.

use pleroma::Pleroma;
use pleroma::core::storage::{AosField, CellView, FieldStorage, SoaField};
use utility::domain::{CellId, FieldKey};

const N: usize = 4;

#[test]
fn register_then_read_round_trip() {
  let mut world = Pleroma::new();
  let initial = SoaField::<N>::from_fn(8, |c| {
    let i = c.index() as f64;
    [i, i + 1.0, i + 2.0, i + 3.0]
  });

  world.register_field(FieldKey::Pressure, initial);

  let stored: &SoaField<N> =
    world.read(FieldKey::Pressure).expect("field is registered");

  assert_eq!(stored.len(), 8);
  assert_eq!(world.cell_count(FieldKey::Pressure), Some(8));

  for i in 0..8 {
    let v = stored.state(CellId::from(i));
    assert_eq!(
      v.as_state(),
      &[i as f64, i as f64 + 1.0, i as f64 + 2.0, i as f64 + 3.0]
    );
  }
}

#[test]
fn write_mutates_field_in_place() {
  let mut world = Pleroma::new();
  world.register_field(FieldKey::Temperature, SoaField::<N>::zeros(4));

  {
    let f: &mut SoaField<N> = world.write(FieldKey::Temperature).unwrap();
    f.write(CellId::from(0), &[1.0, 2.0, 3.0, 4.0]);
    f.write(CellId::from(2), &[10.0, 20.0, 30.0, 40.0]);
  }

  let f: &SoaField<N> = world.read(FieldKey::Temperature).unwrap();
  assert_eq!(f.state(CellId::from(0)).as_state(), &[1.0, 2.0, 3.0, 4.0]);
  assert_eq!(f.state(CellId::from(1)).as_state(), &[0.0; 4]);
  assert_eq!(
    f.state(CellId::from(2)).as_state(),
    &[10.0, 20.0, 30.0, 40.0]
  );
}

#[test]
fn type_mismatch_returns_none() {
  let mut world = Pleroma::new();
  world.register_field(FieldKey::Pressure, SoaField::<4>::zeros(4));

  // SoaField<5> is a distinct type from SoaField<4> — TypeId catches it.
  assert!(world.read::<SoaField<5>>(FieldKey::Pressure).is_none());
  // AosField<4> is a different storage layout; also rejected.
  assert!(world.read::<AosField<4>>(FieldKey::Pressure).is_none());
  // Right type — fine.
  assert!(world.read::<SoaField<4>>(FieldKey::Pressure).is_some());
}

#[test]
fn missing_key_returns_none() {
  let world = Pleroma::new();
  assert!(world.read::<SoaField<4>>(FieldKey::Humidity).is_none());
  assert_eq!(world.cell_count(FieldKey::Humidity), None);
}
