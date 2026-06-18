// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use utility::error::ErrorDomain;

#[derive(Debug)]
pub enum ThalassaError {
  MissingMesh,
  MissingReadField,
  MissingWriteField,
  FieldMeshMismatch,
  FieldLengthMismatch,
  InvalidOceanTemperature,
  InvalidColumnLayout,
  InvalidColumnProperties,
  InvalidTimeStep,
}

impl ErrorDomain for ThalassaError {
  fn domain(&self) -> &str {
    "thalassa"
  }
}

impl std::fmt::Display for ThalassaError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      ThalassaError::MissingMesh => {
        write!(f, "ocean mesh is not registered in tessera")
      }
      ThalassaError::MissingReadField => {
        write!(f, "declared read field is missing or has the wrong type")
      }
      ThalassaError::MissingWriteField => {
        write!(f, "declared write field is missing or has the wrong type")
      }
      ThalassaError::FieldMeshMismatch => {
        write!(f, "stage fields must live on the ocean mesh")
      }
      ThalassaError::FieldLengthMismatch => {
        write!(f, "field and mesh cell counts do not match")
      }
      ThalassaError::InvalidOceanTemperature => {
        write!(f, "ocean temperature is non-physical")
      }
      ThalassaError::InvalidColumnLayout => {
        write!(f, "ocean column layout is degenerate")
      }
      ThalassaError::InvalidColumnProperties => {
        write!(
          f,
          "ocean column physical properties must be finite/positive"
        )
      }
      ThalassaError::InvalidTimeStep => {
        write!(f, "dt must be finite and positive")
      }
    }
  }
}
