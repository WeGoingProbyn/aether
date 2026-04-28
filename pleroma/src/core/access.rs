// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Public view types stage authors and the scheduler hand around. The
//! storage backing them lives in `crate::runtime`; only that module can
//! construct these views — the `pub(crate)` field guards keep external
//! callers from forging one.

use std::any::TypeId;

use utility::domain::FieldKey;

use crate::runtime::slot::SlotView;
use crate::runtime::split::SplitBorrow;

/// The typed view a `Stage::run` body sees. Reads/writes are checked against
/// the keys declared on the stage; calling `read`/`write` for a key the
/// stage didn't declare returns `None`. Calling `read`/`write` with the
/// wrong concrete `S` likewise returns `None` (TypeId mismatch).
///
/// Field visibility is `pub(crate)` so only code inside `pleroma` can
/// construct one — by convention this only happens inside `crate::runtime`.
pub struct WorldAccess<'a> {
  pub(crate) slot_view: SlotView<'a>,
}

impl<'a> WorldAccess<'a> {
  /// Read a field. Returns `None` if the key wasn't declared as a read or
  /// write, the field isn't registered, or the requested type `S` doesn't
  /// match the stored type.
  pub fn read<S: 'static>(&self, key: FieldKey) -> Option<&S> {
    if !self.slot_view.reads.contains(&key)
      && !self.slot_view.writes.contains(&key)
    {
      return None;
    }
    // SAFETY: see crate-level safety note. The split-borrow precondition
    // means no other reference (mut or shared) to this slot is active for
    // any `key` in this view's declared set.
    let fields = unsafe { &*self.slot_view.fields };
    let slot = fields.get(&key)?;
    if slot.type_id != TypeId::of::<S>() {
      return None;
    }
    unsafe {
      let boxed = &*slot.data.get();
      boxed.downcast_ref::<S>()
    }
  }

  /// Mutably borrow a field. Returns `None` unless the key was declared as a
  /// write and the requested type matches.
  pub fn write<S: 'static>(&mut self, key: FieldKey) -> Option<&mut S> {
    if !self.slot_view.writes.contains(&key) {
      return None;
    }
    let fields = unsafe { &*self.slot_view.fields };
    let slot = fields.get(&key)?;
    if slot.type_id != TypeId::of::<S>() {
      return None;
    }
    // SAFETY: `key` is in this view's `writes` set, so by the schedule's
    // non-overlap precondition no other view alive at the same time can
    // observe this slot. We are the unique writer.
    unsafe {
      let boxed = &mut *slot.data.get();
      boxed.downcast_mut::<S>()
    }
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
    reads: &[FieldKey],
    writes: &[FieldKey],
  ) -> WorldAccess<'a> {
    WorldAccess {
      slot_view: SlotView {
        fields: self.inner.fields,
        reads: reads.to_vec(),
        writes: writes.to_vec(),
        _phantom: std::marker::PhantomData,
      },
    }
  }
}
