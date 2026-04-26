// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Split-borrow primitive backing `ScheduleAccess`.
//!
//! `Pleroma::schedule_access` produces a `SplitBorrow` for one DAG layer.
//! Nexus then calls `ScheduleAccess::view_for(reads, writes)` once per
//! parallel stage; the call is `unsafe` because soundness depends on nexus
//! having already verified that all simultaneously-running stages have
//! pairwise-disjoint declared reads/writes.

use std::marker::PhantomData;

pub(crate) struct SplitBorrow<'a> {
  pub(crate) _phantom: PhantomData<&'a ()>,
}
