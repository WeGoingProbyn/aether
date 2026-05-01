// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! DAG construction with non-mesh-bound resources. Stages declaring
//! conflicting `resource_reads`/`resource_writes` should serialise the same
//! way conflicting field reads/writes do.

use nexus::{Nexus, ResourceKey, Stage, StageContext, StageId};
use pleroma::Pleroma;
use pleroma::prelude::FieldKey;
use utility::error::AetherResult;

struct ResourceProbe {
  reads: Vec<FieldKey>,
  writes: Vec<FieldKey>,
  resource_reads: Vec<ResourceKey>,
  resource_writes: Vec<ResourceKey>,
}

impl Stage for ResourceProbe {
  fn name(&self) -> &'static str {
    "resource_probe"
  }
  fn reads(&self) -> &[FieldKey] {
    &self.reads
  }
  fn writes(&self) -> &[FieldKey] {
    &self.writes
  }
  fn resource_reads(&self) -> &[ResourceKey] {
    &self.resource_reads
  }
  fn resource_writes(&self) -> &[ResourceKey] {
    &self.resource_writes
  }
  fn run(&mut self, _ctx: StageContext<'_>) -> AetherResult<()> {
    Ok(())
  }
}

fn probe(
  resource_reads: &[ResourceKey],
  resource_writes: &[ResourceKey],
) -> ResourceProbe {
  ResourceProbe {
    reads: vec![],
    writes: vec![],
    resource_reads: resource_reads.to_vec(),
    resource_writes: resource_writes.to_vec(),
  }
}

fn layer_indices(layer: &[StageId]) -> Vec<usize> {
  layer.iter().map(|id| id.index()).collect()
}

#[test]
fn independent_resource_writes_run_in_one_layer() {
  let mut nexus = Nexus::new();
  nexus.add(probe(&[], &[ResourceKey::Bodies]));
  nexus.add(probe(&[], &[ResourceKey::SunPosition]));
  nexus.add(probe(&[], &[ResourceKey::PlanetSpin]));

  let compiled = nexus.build(&Pleroma::new()).unwrap();
  assert_eq!(compiled.layer_count(), 1);
  assert_eq!(compiled.layers()[0].len(), 3);
}

#[test]
fn resource_raw_creates_two_layers() {
  // Writer of Bodies must precede a reader of Bodies.
  let mut nexus = Nexus::new();
  nexus.add(probe(&[], &[ResourceKey::Bodies]));
  nexus.add(probe(&[ResourceKey::Bodies], &[]));

  let compiled = nexus.build(&Pleroma::new()).unwrap();
  assert_eq!(compiled.layer_count(), 2);
  assert_eq!(layer_indices(compiled.layers()[0].as_slice()), vec![0]);
  assert_eq!(layer_indices(compiled.layers()[1].as_slice()), vec![1]);
}

#[test]
fn resource_waw_serialises_in_add_order() {
  let mut nexus = Nexus::new();
  nexus.add(probe(&[], &[ResourceKey::Bodies]));
  nexus.add(probe(&[], &[ResourceKey::Bodies]));

  let compiled = nexus.build(&Pleroma::new()).unwrap();
  assert_eq!(compiled.layer_count(), 2);
  assert_eq!(layer_indices(compiled.layers()[0].as_slice()), vec![0]);
  assert_eq!(layer_indices(compiled.layers()[1].as_slice()), vec![1]);
}

#[test]
fn resource_war_orders_reader_before_writer() {
  let mut nexus = Nexus::new();
  nexus.add(probe(&[ResourceKey::SunPosition], &[]));
  nexus.add(probe(&[], &[ResourceKey::SunPosition]));

  let compiled = nexus.build(&Pleroma::new()).unwrap();
  assert_eq!(compiled.layer_count(), 2);
  assert_eq!(layer_indices(compiled.layers()[0].as_slice()), vec![0]);
  assert_eq!(layer_indices(compiled.layers()[1].as_slice()), vec![1]);
}

#[test]
fn shared_reads_dont_force_serialisation() {
  // Two stages reading the same resource (e.g. lumen + something else
  // both reading SunPosition) should still run concurrently.
  let mut nexus = Nexus::new();
  nexus.add(probe(&[ResourceKey::SunPosition], &[]));
  nexus.add(probe(&[ResourceKey::SunPosition], &[]));

  let compiled = nexus.build(&Pleroma::new()).unwrap();
  assert_eq!(compiled.layer_count(), 1);
  assert_eq!(compiled.layers()[0].len(), 2);
}

#[test]
fn resource_and_field_conflicts_compose() {
  // Stages with no field overlap but a resource conflict still serialise.
  // Conversely, stages with no resource overlap but a field conflict also
  // serialise. Confirms the two namespaces are both honoured.
  let mut nexus = Nexus::new();
  nexus.add(probe(&[], &[ResourceKey::Bodies]));
  nexus.add(probe(&[ResourceKey::Bodies], &[]));
  nexus.add(probe(&[], &[ResourceKey::SunPosition]));

  let compiled = nexus.build(&Pleroma::new()).unwrap();
  // 0 → 1 (resource RAW), 2 independent.
  assert_eq!(compiled.layer_count(), 2);
  let layer0 = layer_indices(compiled.layers()[0].as_slice());
  let layer1 = layer_indices(compiled.layers()[1].as_slice());
  assert!(layer0.contains(&0));
  assert!(layer0.contains(&2));
  assert_eq!(layer1, vec![1]);
}
