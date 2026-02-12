pub mod error;
pub mod maths;
pub mod serial;
pub mod logger;
pub mod profiler;

pub use utility_macros::profile;
pub use utility_macros::Serialize;
pub use utility_macros::Deserialize;

extern crate self as utility;
