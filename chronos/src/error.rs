// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use utility::error::ErrorDomain;

#[derive(Debug)]
pub enum ChronosError {
  MissingReadField,
  MissingWriteField,
  FieldMeshMismatch,
  FieldLengthMismatch,
  InvalidTimeScale,
  InvalidTimeStep,
  NonFiniteAggregate,
}

impl ErrorDomain for ChronosError {
  fn domain(&self) -> &str {
    "chronos"
  }
}

impl std::fmt::Display for ChronosError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      ChronosError::MissingReadField => {
        write!(f, "declared read field is missing or has the wrong type")
      }
      ChronosError::MissingWriteField => {
        write!(f, "declared write field is missing or has the wrong type")
      }
      ChronosError::FieldMeshMismatch => {
        write!(f, "source and mean fields must live on the same mesh")
      }
      ChronosError::FieldLengthMismatch => {
        write!(f, "source and mean fields have different cell counts")
      }
      ChronosError::InvalidTimeScale => {
        write!(f, "climatology timescale must be finite and positive")
      }
      ChronosError::InvalidTimeStep => {
        write!(f, "dt must be finite and positive")
      }
      ChronosError::NonFiniteAggregate => {
        write!(f, "climatology aggregate became non-finite")
      }
    }
  }
}
