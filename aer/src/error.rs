// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use utility::error::ErrorDomain;

#[derive(Debug)]
pub enum AerError {
  MissingAtmosphereConstants,
  InvalidAtmosphereConstants,
  InvalidAtmosphereState,
  MissingMesh,
  MissingReadField,
  MissingWriteField,
  FieldMeshMismatch,
  FieldLengthMismatch,
  InvalidTimeStep,
}

impl ErrorDomain for AerError {
  fn domain(&self) -> &str {
    "aer"
  }
}

impl std::fmt::Display for AerError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      AerError::MissingAtmosphereConstants => {
        write!(f, "world constants do not contain atmosphere constants")
      }
      AerError::InvalidAtmosphereConstants => {
        write!(f, "atmosphere constants are non-physical")
      }
      AerError::InvalidAtmosphereState => {
        write!(f, "atmosphere state is non-physical")
      }
      AerError::MissingMesh => {
        write!(f, "atmosphere mesh is not registered in tessera")
      }
      AerError::MissingReadField => {
        write!(f, "declared read field is missing or has the wrong type")
      }
      AerError::MissingWriteField => {
        write!(f, "declared write field is missing or has the wrong type")
      }
      AerError::FieldMeshMismatch => {
        write!(f, "stage fields must live on the stage mesh")
      }
      AerError::FieldLengthMismatch => {
        write!(f, "field and mesh cell counts do not match")
      }
      AerError::InvalidTimeStep => {
        write!(f, "solver produced an invalid time step")
      }
    }
  }
}
