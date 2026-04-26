// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

pub mod core;
pub mod prelude;
pub(crate) mod runtime;

// Top-level handle for sandbox/init code. The registry struct is defined in
// `runtime`, but its identity is part of pleroma's public API.
pub use runtime::registry::Pleroma;
