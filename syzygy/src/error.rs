// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use utility::error::ErrorDomain;

#[derive(Debug)]
pub enum SyzygyError {
  MissingCoupler,
  FieldMeshMismatch,
  MissingReadField,
  MissingWriteField,
  CellOutOfBounds,
  InvalidRate,
  InvalidConductance,
  InvalidStencil,
}

impl ErrorDomain for SyzygyError {
  fn domain(&self) -> &str {
    "syzygy"
  }
}

impl std::fmt::Display for SyzygyError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      SyzygyError::MissingCoupler => {
        write!(f, "coupler is not registered in tessera")
      }
      SyzygyError::FieldMeshMismatch => {
        write!(f, "field meshes do not match the selected coupling")
      }
      SyzygyError::MissingReadField => {
        write!(f, "declared read field is missing or has the wrong type")
      }
      SyzygyError::MissingWriteField => {
        write!(f, "declared write field is missing or has the wrong type")
      }
      SyzygyError::CellOutOfBounds => {
        write!(f, "coupled cell id is outside the field storage")
      }
      SyzygyError::InvalidRate => {
        write!(f, "scalar relaxation rate must be finite")
      }
      SyzygyError::InvalidConductance => {
        write!(f, "interface conductance must be finite")
      }
      SyzygyError::InvalidStencil => {
        write!(f, "coupling stencil contains invalid entries")
      }
    }
  }
}
