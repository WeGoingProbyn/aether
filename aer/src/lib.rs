// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

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

pub use diagnostics::EulerDiagnosticsStep;
pub use dynamics::{
  AtmosphereScheme, EulerAtmosphereStep, GravityMode, RotationMode,
};
pub use error::AerError;
pub use flux::TemperatureTendencyStep;
pub use init::AtmosphereSpec;
pub use microphysics::{
  LATENT_HEAT_VAPORISATION, SaturationAdjustmentStep, precipitation_field,
  saturation_specific_humidity, saturation_vapour_pressure,
};
pub use model::{AtmosphereFields, AtmosphereModel, AtmosphereStageIds};
pub use shell::AtmosphereShellLayout;
pub use thermal::TemperatureTendencyToEulerEnergyStep;
pub use tracers::{EvaporationStep, ShellColumns};
