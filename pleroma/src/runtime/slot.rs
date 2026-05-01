// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Storage slots — the type-erased homes of registered fields and resources.
//!
//! Stays inside `pub(crate) mod runtime` so the `UnsafeCell` is invisible to
//! every other crate. Outside callers only see `WorldAccess`/`ScheduleAccess`,
//! which the registry constructs from these slots under verified borrow
//! discipline.

use std::any::{Any, TypeId};
use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::marker::PhantomData;

use utility::domain::{FieldKey, ResourceKey};

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

/// One registered resource — a typed singleton not bound to a mesh. Same
/// shape as `FieldSlot` minus the per-cell metadata.
pub(crate) struct ResourceSlot {
  pub(crate) data: UnsafeCell<Box<dyn Any + Send + Sync>>,
  pub(crate) type_id: TypeId,
}

// SAFETY: same as `FieldSlot`.
unsafe impl Send for ResourceSlot {}
unsafe impl Sync for ResourceSlot {}

/// Typed handle into the registry carried inside a `WorldAccess`. Holds raw
/// pointers to the field and resource maps and the keys this view is allowed
/// to touch. Construction is `pub(crate)` — only `crate::runtime` can build
/// one.
pub(crate) struct SlotView<'a> {
  pub(crate) fields: *const HashMap<FieldKey, FieldSlot>,
  pub(crate) resources: *const HashMap<ResourceKey, ResourceSlot>,
  pub(crate) reads: Vec<FieldKey>,
  pub(crate) writes: Vec<FieldKey>,
  pub(crate) resource_reads: Vec<ResourceKey>,
  pub(crate) resource_writes: Vec<ResourceKey>,
  pub(crate) _phantom: PhantomData<&'a HashMap<FieldKey, FieldSlot>>,
}

// SAFETY: the registry maps live behind a phantom borrow tied to `'a`; raw
// pointers are valid for that lifetime. Send/Sync of contents is guaranteed
// by `FieldSlot`/`ResourceSlot`'s own bounds and the registered storage's
// `Send + Sync`.
unsafe impl Send for SlotView<'_> {}
unsafe impl Sync for SlotView<'_> {}
