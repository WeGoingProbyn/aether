// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Shared foundation for the workspace: math types, a work-stealing thread
//! pool, profiler, logger, serialization, and the shared ID vocabulary
//! (`domain`) every other crate agrees on. No aether-specific concepts of its
//! own — it is the bedrock everything else builds on.
//!
//! See `utility/docs/overview.md` for the module map and how it fits.

pub mod collections;
pub mod constants;
pub mod diagnostics;
pub mod domain;
pub mod error;
pub mod events;
pub mod logger;
pub mod maths;
pub mod profiler;
pub mod serial;
pub mod thread;

pub use utility_macros::Deserialize;
pub use utility_macros::Serialize;
pub use utility_macros::StateDiagnostics;
pub use utility_macros::profile;

extern crate self as utility;

/// Start an inline profiler span. Must be paired with `end_profile!`
/// in strict LIFO order.
#[macro_export]
macro_rules! inline_profile {
  ($name:expr) => {
    $crate::profiler::Profiler::start_span($name);
  };
}

/// End an inline profiler span started with `inline_profile!`.
#[macro_export]
macro_rules! end_profile {
  ($name:expr) => {
    $crate::profiler::Profiler::end_span($name);
  };
}

/// Profile a lexical block and return the block's value.
#[macro_export]
macro_rules! profile_block {
  ($name:expr, $body:block) => {{
    let _guard = $crate::profiler::SpanGuard::new($name, module_path!());
    $body
  }};
}

pub mod consts {
  pub use crate::constants::{
    EARTH_ALBEDO, EARTH_AXIAL_TILT, EARTH_DAY, EARTH_MASS, EARTH_ORBIT,
    EARTH_RADIUS, EARTH_SURFACE_P, EARTH_SURFACE_T, JUPITER_AXIAL_TILT,
    JUPITER_DAY, JUPITER_HEAT_FACTOR, JUPITER_MASS, JUPITER_ORBIT,
    JUPITER_RADIUS, JUPITER_SURFACE_T, MARS_ALBEDO, MARS_AXIAL_TILT, MARS_DAY,
    MARS_MASS, MARS_ORBIT, MARS_RADIUS, MARS_SURFACE_P, MARS_SURFACE_T,
    MERCURY_AXIAL_TILT, MERCURY_DAY, MERCURY_MASS, MERCURY_ORBIT,
    MERCURY_RADIUS, MERCURY_SURFACE_P, MERCURY_SURFACE_T, NEPTUNE_AXIAL_TILT,
    NEPTUNE_DAY, NEPTUNE_HEAT_FACTOR, NEPTUNE_MASS, NEPTUNE_ORBIT,
    NEPTUNE_RADIUS, NEPTUNE_SURFACE_T, NOMINAL_GAS_PRESS, SATURN_AXIAL_TILT,
    SATURN_DAY, SATURN_HEAT_FACTOR, SATURN_MASS, SATURN_ORBIT, SATURN_RADIUS,
    SATURN_SURFACE_T, SOLAR_CORE_TEMP, SOLAR_MASS, SOLAR_RADIUS,
    SOLAR_SURFACE_TEMP, VENUS_ALBEDO, VENUS_AXIAL_TILT, VENUS_DAY, VENUS_MASS,
    VENUS_ORBIT, VENUS_RADIUS, VENUS_SURFACE_P, VENUS_SURFACE_T,
  };
}
