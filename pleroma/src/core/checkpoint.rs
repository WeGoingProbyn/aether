// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Checkpoint codec + container for the pleroma registry.
//!
//! A checkpoint is **loaded into a pre-registered schema**: assemble the world
//! normally (which registers every field/resource), then [`Pleroma::load`]
//! overwrites the *values* in place. This sidesteps reconstructing types from
//! nothing and makes the type-erased registry restorable.
//!
//! Each slot serialises independently through a [`SlotCodec`] captured at
//! registration (the `Serializer` trait is not object-safe, so the codec commits
//! to the concrete JSON backend). The container frames the per-slot JSON
//! documents, keyed by the stable `Debug` rendering of the slot's key so the file
//! is robust to enum-variant reordering.

use std::any::{Any, type_name};

use utility::domain::CellId;
use utility::error::{AetherError, AetherResult, ErrorDomain};
use utility::serial::deserialize::Deserialize;
use utility::serial::json::{JsonDeserializer, JsonSerializer};
use utility::serial::serialize::Serialize;

use crate::core::storage::FieldStorage;
use crate::runtime::slot::SlotCodec;

/// Bumped when the on-disk container layout changes incompatibly.
pub const CHECKPOINT_VERSION: u32 = 1;

#[derive(Debug)]
pub enum CheckpointError {
  /// A slot's stored value did not downcast to its registered type — an
  /// internal invariant violation, not a user error.
  Downcast,
  /// `save` refused to checkpoint a field holding NaN/Inf.
  NonFiniteState,
  /// The world being loaded into is missing a key present at save time (or vice
  /// versa) — the schemas do not match.
  SchemaMismatch,
  /// A record's stored type or cell count disagrees with the live slot.
  TypeMismatch,
  /// The container version is not understood by this build.
  VersionMismatch,
}

impl ErrorDomain for CheckpointError {
  fn domain(&self) -> &str {
    "pleroma checkpoint"
  }
}

impl std::fmt::Display for CheckpointError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let msg = match self {
      CheckpointError::Downcast => "checkpoint slot failed to downcast",
      CheckpointError::NonFiniteState => {
        "refusing to checkpoint a field with non-finite values"
      }
      CheckpointError::SchemaMismatch => {
        "checkpoint schema does not match the assembled world"
      }
      CheckpointError::TypeMismatch => {
        "checkpoint record type or size disagrees with the live slot"
      }
      CheckpointError::VersionMismatch => {
        "checkpoint version is not understood by this build"
      }
    };
    write!(f, "{msg}")
  }
}

fn err(kind: CheckpointError, ctx: impl Into<String>) -> AetherError {
  AetherError::new(kind).context(ctx)
}

// One serialised slot. Primitive fields only, so the existing struct derives
// apply; the key is the slot key's stable `Debug` string.
#[derive(utility::Serialize, utility::Deserialize)]
pub struct FieldRecord {
  pub key: String,
  pub cell_count: u64,
  pub type_name: String,
  pub json: String,
}

#[derive(utility::Serialize, utility::Deserialize)]
pub struct ResourceRecord {
  pub key: String,
  pub type_name: String,
  pub json: String,
}

/// The serialisable container: a versioned, key-sorted set of field and resource
/// records. `Serialize`/`Deserialize` so file I/O reuses the JSON backend.
#[derive(utility::Serialize, utility::Deserialize)]
pub struct PleromaCheckpoint {
  pub version: u32,
  pub fields: Vec<FieldRecord>,
  pub resources: Vec<ResourceRecord>,
}

fn to_json<T: Serialize>(value: &T) -> AetherResult<String> {
  let mut buf = Vec::new();
  {
    let mut ser = JsonSerializer::new(&mut buf);
    value.serialize(&mut ser)?;
  }
  String::from_utf8(buf)
    .map_err(|e| err(CheckpointError::Downcast, format!("invalid utf8: {e}")))
}

fn from_json<T: Deserialize>(json: &str) -> AetherResult<T> {
  let mut de = JsonDeserializer::new(json.as_bytes());
  T::deserialize(&mut de)
}

// --- Field codec ----------------------------------------------------------

/// Build the codec for a field storage type. Captured at `register_field` where
/// both `N` and `S` are known, so the returned fn pointers bake them in.
pub(crate) fn field_codec<const N: usize, S>() -> SlotCodec
where
  S: FieldStorage<N> + Serialize + Deserialize + 'static,
{
  SlotCodec {
    type_name: type_name::<S>(),
    save: save_field::<N, S>,
    load: load_field::<S>,
  }
}

fn save_field<const N: usize, S>(any: &dyn Any) -> AetherResult<String>
where
  S: FieldStorage<N> + Serialize + 'static,
{
  let field = any.downcast_ref::<S>().ok_or_else(|| {
    err(
      CheckpointError::Downcast,
      format!("field as {}", type_name::<S>()),
    )
  })?;
  // Refuse to checkpoint a blown-up field: the JSON backend cannot round-trip
  // NaN/Inf, and a non-finite state is not a checkpoint worth keeping.
  let mut state = [0.0f64; N];
  for i in 0..field.len() {
    field.state_into(CellId::from(i), &mut state);
    if state.iter().any(|v| !v.is_finite()) {
      return Err(err(
        CheckpointError::NonFiniteState,
        format!("cell {i} of a {} field", type_name::<S>()),
      ));
    }
  }
  to_json(field)
}

fn load_field<S>(any: &mut dyn Any, json: &str) -> AetherResult<()>
where
  S: Deserialize + 'static,
{
  let field = any.downcast_mut::<S>().ok_or_else(|| {
    err(
      CheckpointError::Downcast,
      format!("field as {}", type_name::<S>()),
    )
  })?;
  *field = from_json::<S>(json)?;
  Ok(())
}

// --- Resource codec -------------------------------------------------------

pub(crate) fn resource_codec<R>() -> SlotCodec
where
  R: Serialize + Deserialize + 'static,
{
  SlotCodec {
    type_name: type_name::<R>(),
    save: save_resource::<R>,
    load: load_resource::<R>,
  }
}

fn save_resource<R: Serialize + 'static>(
  any: &dyn Any,
) -> AetherResult<String> {
  let r = any.downcast_ref::<R>().ok_or_else(|| {
    err(
      CheckpointError::Downcast,
      format!("resource as {}", type_name::<R>()),
    )
  })?;
  to_json(r)
}

fn load_resource<R: Deserialize + 'static>(
  any: &mut dyn Any,
  json: &str,
) -> AetherResult<()> {
  let r = any.downcast_mut::<R>().ok_or_else(|| {
    err(
      CheckpointError::Downcast,
      format!("resource as {}", type_name::<R>()),
    )
  })?;
  *r = from_json::<R>(json)?;
  Ok(())
}

/// Error helpers re-exported for the `save`/`load` impl in `registry.rs`.
pub(crate) fn schema_mismatch(ctx: impl Into<String>) -> AetherError {
  err(CheckpointError::SchemaMismatch, ctx)
}

pub(crate) fn type_mismatch(ctx: impl Into<String>) -> AetherError {
  err(CheckpointError::TypeMismatch, ctx)
}

pub(crate) fn version_mismatch(ctx: impl Into<String>) -> AetherError {
  err(CheckpointError::VersionMismatch, ctx)
}

#[cfg(test)]
mod tests {
  use crate::Pleroma;
  use crate::core::storage::{FieldStorage, SoaField};
  use utility::domain::{CellId, FieldKey, FieldName, MeshKey, ResourceKey};

  fn temp_key() -> FieldKey {
    FieldKey::new(MeshKey::ATMOSPHERE, FieldName::Temperature)
  }

  #[test]
  fn checkpoint_round_trips_fields_and_resources() {
    let mut source = Pleroma::new();
    source.register_field(
      temp_key(),
      SoaField::<2>::from_fn(4, |c| {
        let i = c.index() as f64;
        [i, i + 50.0]
      }),
    );
    source.register_checkpointed_resource(
      ResourceKey::SunPosition,
      [0.1, 0.2, 0.3],
    );

    let ckpt = source.save().expect("save");

    // A freshly assembled world with the same schema but different values.
    let mut target = Pleroma::new();
    target.register_field(temp_key(), SoaField::<2>::zeros(4));
    target.register_checkpointed_resource(
      ResourceKey::SunPosition,
      [0.0, 0.0, 0.0],
    );

    target.load(&ckpt).expect("load");

    let field = target.read::<SoaField<2>>(temp_key()).unwrap();
    for i in 0..4 {
      assert_eq!(field.state(CellId::from(i)), [i as f64, i as f64 + 50.0]);
    }
    let sun = target
      .read_resource::<[f64; 3]>(ResourceKey::SunPosition)
      .unwrap();
    assert_eq!(sun, &[0.1, 0.2, 0.3]);
  }

  #[test]
  fn load_rejects_a_schema_that_lacks_a_live_field() {
    let mut source = Pleroma::new();
    source.register_field(temp_key(), SoaField::<1>::zeros(2));
    let ckpt = source.save().expect("save");

    // The target has an extra field the checkpoint never recorded.
    let mut target = Pleroma::new();
    target.register_field(temp_key(), SoaField::<1>::zeros(2));
    target.register_field(
      FieldKey::new(MeshKey::ATMOSPHERE, FieldName::Pressure),
      SoaField::<1>::zeros(2),
    );

    assert!(
      target.load(&ckpt).is_err(),
      "missing field must be rejected"
    );
  }

  #[test]
  fn load_rejects_a_cell_count_mismatch() {
    let mut source = Pleroma::new();
    source.register_field(temp_key(), SoaField::<1>::zeros(4));
    let ckpt = source.save().expect("save");

    let mut target = Pleroma::new();
    target.register_field(temp_key(), SoaField::<1>::zeros(8));

    assert!(
      target.load(&ckpt).is_err(),
      "cell-count mismatch must be rejected"
    );
  }

  #[test]
  fn save_refuses_a_non_finite_field() {
    let mut source = Pleroma::new();
    source.register_field(
      temp_key(),
      SoaField::<1>::from_fn(2, |c| {
        if c.index() == 0 { [f64::NAN] } else { [1.0] }
      }),
    );
    assert!(source.save().is_err(), "a NaN field must not checkpoint");
  }
}
