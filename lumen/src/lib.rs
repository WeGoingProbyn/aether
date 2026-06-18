// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Radiative transfer for an atmosphere/surface pair.
//!
//! Lumen owns the *logic* of radiation: a single-band gray-atmosphere
//! model with shortwave / longwave / greenhouse / albedo terms. It does
//! NOT own any field data — every input and output flows through pleroma
//! via `WorldAccess`, exactly like the other physics crates.
//!
//! Aer and terra do not depend on lumen; they consume its outputs by
//! naming the same `FieldKey`s
//! (`FieldName::RadiativeHeatingTendency`, `FieldName::NetSurfaceFlux`).
//! Same-mesh dependencies (lumen heating tendency → aer energy update
//! on the atm mesh) are plain nexus DAG edges; the cross-mesh hop
//! (atm-bottom radiance ↔ surface) goes through syzygy.

pub mod diurnal;
pub mod error;
pub mod model;
pub mod optical;
pub mod transfer;

pub use diurnal::DiurnalSunStep;
pub use error::LumenError;
pub use model::{RadiationFields, RadiationModel, RadiationStageIds};
pub use optical::{normalise, zenith_cosine};
pub use transfer::{
  RadiationCoefficients, RadiationParameters, RadiativeTransferStep,
};
