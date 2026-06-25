// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Thalassa owns the *logic* of a thermodynamic ocean. Like the other
//! physics crates it stores nothing globally: it registers an ocean
//! temperature field at setup, names the surface-flux field it consumes,
//! and advances the water column through nexus stages. All state lives in
//! pleroma.
//!
//! The ocean is a radial stack of liquid-water layers on a cube-sphere
//! shell (`MeshKey::OCEAN`). For this first proof the physics are purely
//! thermodynamic:
//!
//! - the **top** layer (outermost radius — the sea surface) absorbs the
//!   net surface heat flux supplied by lumen/syzygy, and
//! - heat **diffuses vertically** between radial layers toward the deep
//!   ocean.
//!
//! Horizontal currents (a `continuum` ocean conservation law) are a later
//! deepening; the column model already closes the air–sea heat budget and
//! supplies the sea-surface temperature that drives evaporation.
//!
//! See `thalassa/docs/overview.md` for how it fits the coupled budget.

pub mod error;
pub mod model;
pub mod thermodynamics;

pub use error::ThalassaError;
pub use model::{OceanColumnLayout, OceanFields, OceanModel};
pub use thermodynamics::OceanThermodynamicsStep;
