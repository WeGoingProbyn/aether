// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Split-borrow primitive backing `ScheduleAccess`.
//!
//! `Pleroma::schedule_access` produces a `SplitBorrow` for one DAG layer.
//! Nexus then calls `ScheduleAccess::view_for(reads, writes)` once per
//! parallel stage; the call is `unsafe` because soundness depends on nexus
//! having already verified that all simultaneously-running stages have
//! pairwise-disjoint declared reads/writes.

use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;

use tessera::mesh::Mesh;
use utility::domain::{FieldKey, MeshKey};

use crate::runtime::slot::FieldSlot;

pub(crate) struct SplitBorrow<'a> {
  pub(crate) fields: *const HashMap<FieldKey, FieldSlot>,
  pub(crate) meshes: *const HashMap<MeshKey, Arc<dyn Mesh<3>>>,
  // PhantomData<&'a mut ()> — `schedule_access(&mut self)` is the only way
  // to produce a `SplitBorrow`, so this morally holds an exclusive borrow on
  // Pleroma for the duration. WorldAccess views are aliased slices of that
  // borrow; soundness is the caller's burden via `view_for`.
  pub(crate) _phantom: PhantomData<&'a mut ()>,
}

// SAFETY: the underlying registries are `Send + Sync` (FieldSlot manually so;
// mesh map is Arc'd). Raw pointers are valid for `'a`.
unsafe impl Send for SplitBorrow<'_> {}
unsafe impl Sync for SplitBorrow<'_> {}
