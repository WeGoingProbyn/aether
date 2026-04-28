// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! The vocabulary of pleroma. Everything stage authors and downstream
//! consumers should reach for lives here. Nexus re-exports this module
//! verbatim so physics crates depend on `nexus` alone.

pub use crate::core::access::*;
pub use crate::core::exchange::exchange_ghosts;
pub use crate::core::storage::*;
pub use utility::domain::{FieldKey, FieldName, MeshKey, MeshType};
