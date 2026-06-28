// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use std::any::{Any, TypeId};
use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::marker::PhantomData;

use utility::domain::{FieldKey, ResourceKey};
use utility::error::AetherResult;
use utility::serial::deserialize::Deserialize;
use utility::serial::serialize::Serialize;

use crate::core::access::ScheduleAccess;
use crate::core::checkpoint::{
  self, CHECKPOINT_VERSION, FieldRecord, PleromaCheckpoint, ResourceRecord,
};
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
    S: FieldStorage<N> + Serialize + Deserialize + 'static,
  {
    let cell_count = init.len();
    let boxed: Box<dyn Any + Send + Sync> = Box::new(init);
    self.fields.insert(
      key,
      FieldSlot {
        data: UnsafeCell::new(boxed),
        type_id: TypeId::of::<S>(),
        cell_count,
        codec: checkpoint::field_codec::<N, S>(),
        remapper: crate::core::storage::field_remapper::<N, S>(),
        resizer: crate::core::storage::field_resizer::<N, S>(),
      },
    );
  }

  /// Resize every field on `mesh` to `cell_count` zeroed cells, updating each
  /// slot's `cell_count`. Used to grow/shrink a mesh's fields to a reconstructed
  /// (e.g. checkpoint-restored adaptive) topology before the stored values are
  /// loaded over them. Fields on other meshes are untouched.
  pub fn resize_mesh_fields(
    &mut self,
    mesh: utility::domain::MeshKey,
    cell_count: usize,
  ) {
    for (key, slot) in self.fields.iter_mut() {
      if key.mesh() != mesh {
        continue;
      }
      let boxed = slot.data.get_mut();
      *boxed = (slot.resizer)(cell_count);
      slot.cell_count = cell_count;
    }
  }

  /// Remap every field on `mesh` across a topology adapt, in place. For each such
  /// field the captured remapper builds new-length storage from `remap` (children
  /// prolong from their parent, merges restrict by volume), then it is swapped in
  /// and the slot's `cell_count` updated to `remap.new_count()`. `old_volumes` is
  /// the pre-adapt per-cell volume of `mesh`, indexed by old [`CellId`], needed
  /// for the conservative coarsening average. Fields on other meshes are
  /// untouched.
  ///
  /// Swap-to-new: each field's old buffer stays live until its replacement is
  /// built, so a remap never leaves a field half-resized. Call at the end-of-tick
  /// adapt barrier, never mid-DAG.
  pub fn remap_mesh_fields(
    &mut self,
    mesh: utility::domain::MeshKey,
    remap: &utility::domain::CellRemap,
    old_volumes: &[f64],
  ) {
    for (key, slot) in self.fields.iter_mut() {
      if key.mesh() != mesh {
        continue;
      }
      let boxed = slot.data.get_mut();
      let new_box = (slot.remapper)(boxed.as_ref(), remap, old_volumes);
      *boxed = new_box;
      slot.cell_count = remap.new_count();
    }
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
        codec: None,
      },
    );
  }

  /// Register a resource that is part of the persistent simulation state, so a
  /// checkpoint round-trips it. Identical to [`register_resource`] but installs a
  /// serialize/deserialize codec. Derived / transient resources (e.g. the
  /// diagnostics report) should use the plain [`register_resource`] — a
  /// checkpoint skips them and world assembly rebuilds them on load.
  ///
  /// [`register_resource`]: Self::register_resource
  pub fn register_checkpointed_resource<R>(&mut self, key: ResourceKey, init: R)
  where
    R: Serialize + Deserialize + 'static + Send + Sync,
  {
    let boxed: Box<dyn Any + Send + Sync> = Box::new(init);
    self.resources.insert(
      key,
      ResourceSlot {
        data: UnsafeCell::new(boxed),
        type_id: TypeId::of::<R>(),
        codec: Some(checkpoint::resource_codec::<R>()),
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

  /// Snapshot every field and every checkpointed resource into a serialisable
  /// [`PleromaCheckpoint`]. Records are emitted in sorted key order so the output
  /// is deterministic and diffable. Refuses to save if any field holds non-finite
  /// values (the JSON backend cannot round-trip NaN/Inf, and a blown-up state is
  /// not worth checkpointing). Derived / transient resources are skipped.
  pub fn save(&self) -> AetherResult<PleromaCheckpoint> {
    let mut field_keys: Vec<FieldKey> = self.fields.keys().copied().collect();
    field_keys.sort();
    let mut fields = Vec::with_capacity(field_keys.len());
    for key in field_keys {
      let slot = &self.fields[&key];
      // SAFETY: `&self` blocks any concurrent `write`/`view_for`; we only read.
      let any: &dyn Any = unsafe { &**slot.data.get() };
      let json = (slot.codec.save)(any)?;
      fields.push(FieldRecord {
        key: format!("{key:?}"),
        cell_count: slot.cell_count as u64,
        type_name: slot.codec.type_name.to_string(),
        json,
      });
    }

    let mut resource_keys: Vec<ResourceKey> =
      self.resources.keys().copied().collect();
    resource_keys.sort();
    let mut resources = Vec::new();
    for key in resource_keys {
      let slot = &self.resources[&key];
      let Some(codec) = &slot.codec else { continue };
      // SAFETY: as above.
      let any: &dyn Any = unsafe { &**slot.data.get() };
      let json = (codec.save)(any)?;
      resources.push(ResourceRecord {
        key: format!("{key:?}"),
        type_name: codec.type_name.to_string(),
        json,
      });
    }

    Ok(PleromaCheckpoint {
      version: CHECKPOINT_VERSION,
      fields,
      resources,
    })
  }

  /// Restore state from a [`PleromaCheckpoint`] **into the already-registered
  /// schema**: every live field (and every checkpointed resource) must have a
  /// matching record of the same type and size, or a clear schema/type-mismatch
  /// error is returned and no state is left partially applied beyond the slots
  /// already restored. Records present in the file but absent from the live world
  /// are ignored (a superset checkpoint is allowed).
  pub fn load(&mut self, ckpt: &PleromaCheckpoint) -> AetherResult<()> {
    if ckpt.version != CHECKPOINT_VERSION {
      return Err(checkpoint::version_mismatch(format!(
        "checkpoint version {} != supported {CHECKPOINT_VERSION}",
        ckpt.version,
      )));
    }

    let field_recs: HashMap<&str, &FieldRecord> =
      ckpt.fields.iter().map(|r| (r.key.as_str(), r)).collect();
    for (key, slot) in self.fields.iter_mut() {
      let key_str = format!("{key:?}");
      let rec = field_recs.get(key_str.as_str()).ok_or_else(|| {
        checkpoint::schema_mismatch(format!(
          "checkpoint is missing field {key_str}"
        ))
      })?;
      if rec.type_name != slot.codec.type_name {
        return Err(checkpoint::type_mismatch(format!(
          "field {key_str}: checkpoint type {} != live {}",
          rec.type_name, slot.codec.type_name,
        )));
      }
      if rec.cell_count != slot.cell_count as u64 {
        return Err(checkpoint::type_mismatch(format!(
          "field {key_str}: checkpoint has {} cells != live {}",
          rec.cell_count, slot.cell_count,
        )));
      }
      let load_fn = slot.codec.load;
      let any: &mut dyn Any = slot.data.get_mut().as_mut();
      load_fn(any, &rec.json)?;
    }

    let resource_recs: HashMap<&str, &ResourceRecord> =
      ckpt.resources.iter().map(|r| (r.key.as_str(), r)).collect();
    for (key, slot) in self.resources.iter_mut() {
      let Some(codec) = &slot.codec else { continue };
      let key_str = format!("{key:?}");
      let rec = resource_recs.get(key_str.as_str()).ok_or_else(|| {
        checkpoint::schema_mismatch(format!(
          "checkpoint is missing resource {key_str}"
        ))
      })?;
      if rec.type_name != codec.type_name {
        return Err(checkpoint::type_mismatch(format!(
          "resource {key_str}: checkpoint type {} != live {}",
          rec.type_name, codec.type_name,
        )));
      }
      let load_fn = codec.load;
      let any: &mut dyn Any = slot.data.get_mut().as_mut();
      load_fn(any, &rec.json)?;
    }

    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use utility::domain::{
    CellId, CellRemap, FieldName, MeshKey, NewCellSource, TopologyEpoch,
  };

  use super::*;
  use crate::core::storage::{FieldStorage, SoaField};

  #[test]
  fn remap_mesh_fields_resizes_and_remaps_only_the_target_mesh() {
    let atmos = FieldKey::new(MeshKey::ATMOSPHERE, FieldName::Temperature);
    let ocean = FieldKey::new(MeshKey::OCEAN, FieldName::Temperature);

    let mut pleroma = Pleroma::new();
    // Atmosphere: 3 cells; ocean: 3 cells (must be left untouched).
    pleroma.register_field(
      atmos,
      SoaField::<1>::from_fn(3, |c| [10.0 + c.index() as f64]),
    );
    pleroma.register_field(ocean, SoaField::<1>::from_fn(3, |_| [99.0]));

    // Refine atmosphere cell 0 into two children; merge cells 1,2.
    let remap = CellRemap::new(
      TopologyEpoch::ZERO,
      TopologyEpoch::ZERO.next(),
      vec![None, Some(CellId::from(2)), Some(CellId::from(2))],
      vec![
        NewCellSource::Child {
          parent: CellId::from(0),
        },
        NewCellSource::Child {
          parent: CellId::from(0),
        },
        NewCellSource::Merge {
          children: vec![CellId::from(1), CellId::from(2)],
        },
      ],
    );
    pleroma.remap_mesh_fields(MeshKey::ATMOSPHERE, &remap, &[1.0, 1.0, 1.0]);

    // Atmosphere field swapped to the new length with prolonged children.
    assert_eq!(pleroma.cell_count(atmos), Some(3));
    let atmos_field: &SoaField<1> = pleroma.read(atmos).unwrap();
    assert_eq!(atmos_field.state(CellId::from(0)), [10.0]);
    assert_eq!(atmos_field.state(CellId::from(1)), [10.0]);
    // Equal volumes ⇒ merge is the plain mean of cells 1,2: (11 + 12) / 2.
    assert!((atmos_field.state(CellId::from(2))[0] - 11.5).abs() < 1e-12);

    // Ocean field is on another mesh: untouched.
    assert_eq!(pleroma.cell_count(ocean), Some(3));
    let ocean_field: &SoaField<1> = pleroma.read(ocean).unwrap();
    for i in 0..3 {
      assert_eq!(ocean_field.state(CellId::from(i)), [99.0]);
    }
  }
}
