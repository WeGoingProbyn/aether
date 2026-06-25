// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! The single owner of all mutable simulation state. Every field a stage evolves
//! and every non-mesh resource lives here behind a typed registry; `nexus` hands
//! stages scope-limited borrowed access (`WorldAccess`), so physics crates never
//! hold buffers of their own.
//!
//! See `pleroma/docs/overview.md` for the access/storage vocabulary.

pub mod core;
pub mod prelude;
pub(crate) mod runtime;

// Top-level handle for sandbox/init code. The registry struct is defined in
// `runtime`, but its identity is part of pleroma's public API.
pub use runtime::registry::Pleroma;
