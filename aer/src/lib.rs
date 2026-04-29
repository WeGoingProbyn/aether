// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

pub mod background;
pub mod diagnostics;
pub mod dynamics;
pub mod error;
pub mod flux;
pub mod init;
pub mod microphysics;
pub mod model;
pub mod radiation;
pub mod shell;
pub mod thermal;
pub mod tracers;

pub use background::BackgroundCorrectedEuler3D;
pub use diagnostics::EulerDiagnosticsStep;
pub use dynamics::{
  BackgroundCorrectionMode, EulerAtmosphereStep, GravityMode,
};
pub use error::AerError;
pub use flux::TemperatureTendencyStep;
pub use init::AtmosphereSpec;
pub use model::{AtmosphereFields, AtmosphereModel, AtmosphereStageIds};
pub use shell::AtmosphereShellLayout;
pub use thermal::TemperatureTendencyToEulerEnergyStep;
