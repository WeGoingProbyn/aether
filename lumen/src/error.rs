// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use utility::error::ErrorDomain;

#[derive(Debug)]
pub enum LumenError {
  MissingMesh,
  MissingReadField,
  MissingWriteField,
  MissingResource,
  FieldMeshMismatch,
  FieldLengthMismatch,
  InvalidParameters,
  MissingRadiationConstants,
  MissingAtmosphereConstants,
}

impl ErrorDomain for LumenError {
  fn domain(&self) -> &str {
    "lumen"
  }
}

impl std::fmt::Display for LumenError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      LumenError::MissingMesh => {
        write!(f, "lumen mesh is not registered in tessera")
      }
      LumenError::MissingReadField => {
        write!(f, "declared read field is missing or has the wrong type")
      }
      LumenError::MissingWriteField => {
        write!(f, "declared write field is missing or has the wrong type")
      }
      LumenError::MissingResource => write!(
        f,
        "declared resource is not registered or has the wrong type \
         (lumen expects [f64; 3] for SunPosition)"
      ),
      LumenError::FieldMeshMismatch => {
        write!(f, "stage fields must live on their declared mesh")
      }
      LumenError::FieldLengthMismatch => {
        write!(f, "field and mesh cell counts do not match")
      }
      LumenError::InvalidParameters => {
        write!(f, "radiation model parameters are non-physical")
      }
      LumenError::MissingRadiationConstants => write!(
        f,
        "WorldConstants::radiation is None — cosmo could not derive a \
         solar irradiance for this world (no primary star?)"
      ),
      LumenError::MissingAtmosphereConstants => write!(
        f,
        "WorldConstants::atmosphere is None — radiation needs a \
         reference temperature, which only atmospheric bodies provide"
      ),
    }
  }
}
