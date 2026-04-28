// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Nexus build-time behaviour: edge derivation from declared
//! reads/writes, layer construction, cycle detection, and `before` hints.

use std::sync::{Arc, Mutex};

use nexus::{Nexus, Stage, StageContext, StageId};
use pleroma::Pleroma;
use pleroma::prelude::{FieldKey, FieldName, MeshKey};
use utility::error::AetherResult;

const PRESSURE: FieldKey = FieldKey::new(MeshKey::SURFACE, FieldName::Pressure);
const ATMOSPHERE_PRESSURE: FieldKey =
  FieldKey::new(MeshKey::ATMOSPHERE, FieldName::Pressure);
const TEMPERATURE: FieldKey =
  FieldKey::new(MeshKey::SURFACE, FieldName::Temperature);
const HUMIDITY: FieldKey = FieldKey::new(MeshKey::SURFACE, FieldName::Humidity);

/// Test fixture: a stage with arbitrary reads/writes that records its name
/// when it runs. Useful for asserting layer membership / ordering.
struct Probe {
  name: &'static str,
  reads: Vec<FieldKey>,
  writes: Vec<FieldKey>,
  log: Arc<Mutex<Vec<&'static str>>>,
}

impl Stage for Probe {
  fn name(&self) -> &'static str {
    self.name
  }
  fn reads(&self) -> &[FieldKey] {
    &self.reads
  }
  fn writes(&self) -> &[FieldKey] {
    &self.writes
  }
  fn run(&self, _ctx: StageContext<'_>) -> AetherResult<()> {
    self.log.lock().unwrap().push(self.name);
    Ok(())
  }
}

fn probe(
  name: &'static str,
  reads: &[FieldKey],
  writes: &[FieldKey],
  log: &Arc<Mutex<Vec<&'static str>>>,
) -> Probe {
  Probe {
    name,
    reads: reads.to_vec(),
    writes: writes.to_vec(),
    log: Arc::clone(log),
  }
}

#[test]
fn independent_stages_collapse_into_one_layer() {
  let log = Arc::new(Mutex::new(Vec::new()));
  let mut s = Nexus::new();
  s.add(probe("a", &[], &[PRESSURE], &log));
  s.add(probe("b", &[], &[TEMPERATURE], &log));
  s.add(probe("c", &[], &[HUMIDITY], &log));

  let world = Pleroma::new();
  let compiled = s.build(&world).unwrap();
  assert_eq!(compiled.layer_count(), 1);
  assert_eq!(compiled.layers()[0].len(), 3);
}

#[test]
fn same_field_name_on_different_meshes_is_independent() {
  let log = Arc::new(Mutex::new(Vec::new()));
  let mut s = Nexus::new();
  s.add(probe("surface", &[], &[PRESSURE], &log));
  s.add(probe("atmosphere", &[], &[ATMOSPHERE_PRESSURE], &log));

  let world = Pleroma::new();
  let compiled = s.build(&world).unwrap();
  assert_eq!(compiled.layer_count(), 1);
  assert_eq!(compiled.layers()[0].len(), 2);
}

#[test]
fn raw_dependency_creates_two_layers() {
  let log = Arc::new(Mutex::new(Vec::new()));
  let mut s = Nexus::new();
  s.add(probe("writer", &[], &[PRESSURE], &log));
  s.add(probe("reader", &[PRESSURE], &[], &log));

  let compiled = s.build(&Pleroma::new()).unwrap();
  assert_eq!(compiled.layer_count(), 2);
  assert_eq!(layer_indices(compiled.layers()[0].as_slice()), vec![0]);
  assert_eq!(layer_indices(compiled.layers()[1].as_slice()), vec![1]);
}

#[test]
fn waw_serialises_in_add_order() {
  let log = Arc::new(Mutex::new(Vec::new()));
  let mut s = Nexus::new();
  s.add(probe("first", &[], &[PRESSURE], &log));
  s.add(probe("second", &[], &[PRESSURE], &log));

  let compiled = s.build(&Pleroma::new()).unwrap();
  assert_eq!(compiled.layer_count(), 2);
  assert_eq!(layer_indices(compiled.layers()[0].as_slice()), vec![0]);
  assert_eq!(layer_indices(compiled.layers()[1].as_slice()), vec![1]);
}

#[test]
fn war_orders_reader_before_writer() {
  // a reads X, b writes X → a must run first so it sees the pre-tick value
  // before b overwrites it.
  let log = Arc::new(Mutex::new(Vec::new()));
  let mut s = Nexus::new();
  s.add(probe("reader", &[PRESSURE], &[], &log));
  s.add(probe("writer", &[], &[PRESSURE], &log));

  let compiled = s.build(&Pleroma::new()).unwrap();
  assert_eq!(compiled.layer_count(), 2);
  assert_eq!(layer_indices(compiled.layers()[0].as_slice()), vec![0]);
  assert_eq!(layer_indices(compiled.layers()[1].as_slice()), vec![1]);
}

#[test]
fn diamond_dependency_yields_three_layers() {
  // a writes P
  // b reads P, writes T
  // c reads P, writes H
  // d reads T, reads H
  let log = Arc::new(Mutex::new(Vec::new()));
  let mut s = Nexus::new();
  s.add(probe("a", &[], &[PRESSURE], &log));
  s.add(probe("b", &[PRESSURE], &[TEMPERATURE], &log));
  s.add(probe("c", &[PRESSURE], &[HUMIDITY], &log));
  s.add(probe("d", &[TEMPERATURE, HUMIDITY], &[], &log));

  let compiled = s.build(&Pleroma::new()).unwrap();
  assert_eq!(compiled.layer_count(), 3);
  assert_eq!(layer_indices(compiled.layers()[0].as_slice()), vec![0]);
  let mid = layer_indices(compiled.layers()[1].as_slice());
  assert_eq!(mid.len(), 2);
  assert!(mid.contains(&1));
  assert!(mid.contains(&2));
  assert_eq!(layer_indices(compiled.layers()[2].as_slice()), vec![3]);
}

fn layer_indices(layer: &[StageId]) -> Vec<usize> {
  layer.iter().map(|id| id.index()).collect()
}

#[test]
fn explicit_before_adds_edge_between_independent_stages() {
  let log = Arc::new(Mutex::new(Vec::new()));
  let mut s = Nexus::new();
  let a = s.add(probe("a", &[], &[PRESSURE], &log));
  let b = s.add(probe("b", &[], &[TEMPERATURE], &log));
  s.before(a, b);

  let compiled = s.build(&Pleroma::new()).unwrap();
  assert_eq!(compiled.layer_count(), 2);
  assert_eq!(layer_indices(compiled.layers()[0].as_slice()), vec![0]);
  assert_eq!(layer_indices(compiled.layers()[1].as_slice()), vec![1]);
}

#[test]
fn contradictory_before_hints_error_as_cycle() {
  let log = Arc::new(Mutex::new(Vec::new()));
  let mut s = Nexus::new();
  let a = s.add(probe("a", &[], &[PRESSURE], &log));
  let b = s.add(probe("b", &[], &[TEMPERATURE], &log));
  s.before(a, b);
  s.before(b, a);

  assert!(s.build(&Pleroma::new()).is_err());
}
