// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Public view types stage authors and the scheduler hand around. The
//! storage backing them lives in `crate::runtime`; only that module can
//! construct these views — the `pub(in crate::runtime)` field guards keep
//! external callers from forging one.

use std::sync::Arc;

use tessera::mesh::Mesh;
use utility::domain::{FieldKey, MeshKey};

use crate::runtime::slot::SlotView;
use crate::runtime::split::SplitBorrow;

/// The typed view a `Stage::run` body sees. Reads/writes are checked against
/// the keys declared on the stage; calling `read`/`write` for a key the
/// stage didn't declare returns `None`.
///
/// Field visibility is `pub(crate)` so only code inside `pleroma` can
/// construct one — by convention this only happens inside `crate::runtime`.
pub struct WorldAccess<'a> {
  pub(crate) slot_view: SlotView<'a>,
}

impl<'a> WorldAccess<'a> {
  pub fn mesh_for(&self, _key: MeshKey) -> Option<&Arc<dyn Mesh<3>>> {
    unimplemented!("WorldAccess::mesh_for is awaiting runtime impl")
  }

  pub fn read<S: 'static>(&self, _key: FieldKey) -> Option<&S> {
    unimplemented!("WorldAccess::read is awaiting runtime impl")
  }

  pub fn write<S: 'static>(&mut self, _key: FieldKey) -> Option<&mut S> {
    unimplemented!("WorldAccess::write is awaiting runtime impl")
  }
}

/// One DAG-layer's split-borrow handle. Nexus pulls this from
/// `Pleroma::schedule_access` and calls `view_for` once per parallel stage.
pub struct ScheduleAccess<'a> {
  pub(crate) inner: SplitBorrow<'a>,
}

impl<'a> ScheduleAccess<'a> {
  /// Hand out a `WorldAccess` scoped to one stage's declared keys.
  ///
  /// # Safety
  /// The caller (nexus) must guarantee that across every `view_for` call
  /// that produces a `WorldAccess` alive at the same time, the union of all
  /// `reads` is disjoint from the union of all `writes`, and no two
  /// `writes` overlap.
  pub unsafe fn view_for(
    &self,
    _reads: &[FieldKey],
    _writes: &[FieldKey],
  ) -> WorldAccess<'a> {
    unimplemented!("ScheduleAccess::view_for is awaiting runtime impl")
  }
}
