// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use std::any::{Any, TypeId};
use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::marker::PhantomData;

use utility::domain::{FieldKey, ResourceKey};

use crate::core::access::ScheduleAccess;
use crate::core::storage::FieldStorage;
use crate::runtime::slot::{FieldSlot, ResourceSlot};
use crate::runtime::split::SplitBorrow;

/// Aggregator for every simulation field and non-mesh-bound resource.
/// Constructed by sandbox/init code from a cosmo seed; physics crates never
/// hold a `Pleroma` directly — they reach state via `WorldAccess` from
/// nexus.
pub struct Pleroma {
  fields: HashMap<FieldKey, FieldSlot>,
  resources: HashMap<ResourceKey, ResourceSlot>,
}

impl Default for Pleroma {
  fn default() -> Self {
    Self::new()
  }
}

impl Pleroma {
  pub fn new() -> Self {
    Pleroma {
      fields: HashMap::new(),
      resources: HashMap::new(),
    }
  }

  /// Register a field. The concrete `S` is captured in the slot's TypeId
  /// so subsequent `read::<S>` / `write::<S>` calls can downcast safely.
  pub fn register_field<S, const N: usize>(&mut self, key: FieldKey, init: S)
  where
    S: FieldStorage<N> + 'static,
  {
    let cell_count = init.len();
    let boxed: Box<dyn Any + Send + Sync> = Box::new(init);
    self.fields.insert(
      key,
      FieldSlot {
        data: UnsafeCell::new(boxed),
        type_id: TypeId::of::<S>(),
        cell_count,
      },
    );
  }

  /// Register a non-mesh-bound resource (e.g. orbital body state, sun
  /// direction). The concrete `R` is captured in the slot's TypeId so
  /// subsequent `read_resource::<R>` / `write_resource::<R>` calls can
  /// downcast safely.
  pub fn register_resource<R>(&mut self, key: ResourceKey, init: R)
  where
    R: 'static + Send + Sync,
  {
    let boxed: Box<dyn Any + Send + Sync> = Box::new(init);
    self.resources.insert(
      key,
      ResourceSlot {
        data: UnsafeCell::new(boxed),
        type_id: TypeId::of::<R>(),
      },
    );
  }

  /// Direct read against the registry. Safe; takes `&self`. Returns `None`
  /// if the key isn't registered or the requested `S` doesn't match the
  /// stored type.
  pub fn read<S: 'static>(&self, key: FieldKey) -> Option<&S> {
    let slot = self.fields.get(&key)?;
    if slot.type_id != TypeId::of::<S>() {
      return None;
    }
    // SAFETY: `&self` borrow on Pleroma blocks any concurrent `write`/`view_for`
    // call. We hand out only a shared reference.
    unsafe {
      let boxed = &*slot.data.get();
      boxed.downcast_ref::<S>()
    }
  }

  /// Direct mutable borrow against the registry. Safe; takes `&mut self`.
  pub fn write<S: 'static>(&mut self, key: FieldKey) -> Option<&mut S> {
    let slot = self.fields.get_mut(&key)?;
    if slot.type_id != TypeId::of::<S>() {
      return None;
    }
    let boxed = slot.data.get_mut();
    boxed.downcast_mut::<S>()
  }

  /// Direct read of a resource. Safe; takes `&self`.
  pub fn read_resource<R: 'static>(&self, key: ResourceKey) -> Option<&R> {
    let slot = self.resources.get(&key)?;
    if slot.type_id != TypeId::of::<R>() {
      return None;
    }
    // SAFETY: `&self` borrow on Pleroma blocks any concurrent
    // `write_resource`/`view_for` call.
    unsafe {
      let boxed = &*slot.data.get();
      boxed.downcast_ref::<R>()
    }
  }

  /// Direct mutable borrow of a resource. Safe; takes `&mut self`.
  pub fn write_resource<R: 'static>(
    &mut self,
    key: ResourceKey,
  ) -> Option<&mut R> {
    let slot = self.resources.get_mut(&key)?;
    if slot.type_id != TypeId::of::<R>() {
      return None;
    }
    let boxed = slot.data.get_mut();
    boxed.downcast_mut::<R>()
  }

  /// Hand a `ScheduleAccess` to the nexus scheduler for one DAG layer. The
  /// returned handle holds a phantom mutable borrow on `self`, so the
  /// registry is exclusively held until it is dropped.
  pub fn schedule_access(&mut self) -> ScheduleAccess<'_> {
    ScheduleAccess {
      inner: SplitBorrow {
        fields: &self.fields as *const _,
        resources: &self.resources as *const _,
        _phantom: PhantomData,
      },
    }
  }

  /// Cell count of a registered field, without needing to know its concrete
  /// storage type. Useful for sanity-checking buffer sizes during init.
  pub fn cell_count(&self, key: FieldKey) -> Option<usize> {
    self.fields.get(&key).map(|s| s.cell_count)
  }
}
