// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Field storage slot — the type-erased home of one registered field.
//!
//! Stays inside `pub(crate) mod runtime` so the `UnsafeCell` is invisible to
//! every other crate. Outside callers only see `WorldAccess`/`ScheduleAccess`,
//! which the registry constructs from these slots under verified borrow
//! discipline.

use std::any::{Any, TypeId};
use std::cell::UnsafeCell;
use std::marker::PhantomData;

/// One registered field. Holds an `UnsafeCell<Box<dyn Any + Send + Sync>>` so
/// the registry can hand out aliased typed references when the schedule has
/// proved non-overlap.
pub(crate) struct FieldSlot {
  pub(crate) data: UnsafeCell<Box<dyn Any + Send + Sync>>,
  pub(crate) type_id: TypeId,
  pub(crate) cell_count: usize,
}

// SAFETY: the schedule layer guarantees that any concurrent access to a
// `FieldSlot` is non-overlapping; downcast typing checks happen on every
// read/write.
unsafe impl Send for FieldSlot {}
unsafe impl Sync for FieldSlot {}

/// Typed handle into a `FieldSlot` carried inside a `WorldAccess`. Borrow
/// lifetime ties it to the parent registry, and only `crate::runtime` can
/// construct one.
pub(crate) struct SlotView<'a> {
  pub(crate) _phantom: PhantomData<&'a FieldSlot>,
}
