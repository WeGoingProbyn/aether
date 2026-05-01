// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Top-level builder for lumen's radiative transfer.
//!
//! `RadiationModel` mirrors the pattern used by other physics crates
//! (`AtmosphereModel`, `SurfaceThermalModel`): the user constructs it,
//! calls `register_fields` against a pleroma + the two meshes, then
//! `add_stages` against a nexus. Lumen never holds field storage of its
//! own.

use nexus::{
  FieldKey, FieldName, MeshKey, Nexus, Pleroma, ResourceKey, SoaField, StageId,
};
use tessera::mesh::Mesh;
use utility::error::{AetherError, AetherResult};

use crate::{
  error::LumenError,
  transfer::{RadiationParameters, RadiativeTransferStep},
};

/// Field keys lumen writes to. Held on the model so callers can name them
/// uniformly (e.g. when wiring the heating tendency into aer's energy
/// stage as a same-mesh RAW dependency).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RadiationFields {
  pub heating_tendency: FieldKey,
  pub net_surface_flux: FieldKey,
  pub atm_temperature: FieldKey,
  pub surface_temperature: FieldKey,
}

impl RadiationFields {
  /// Default field keys for an atmosphere/surface mesh pair: lumen's
  /// outputs land on the conventional `RadiativeHeatingTendency` and
  /// `NetSurfaceFlux` field names, while it consumes the existing
  /// temperature fields registered by aer / terra.
  pub const fn for_meshes(atm_mesh: MeshKey, surface_mesh: MeshKey) -> Self {
    Self {
      heating_tendency: FieldKey::new(
        atm_mesh,
        FieldName::RadiativeHeatingTendency,
      ),
      net_surface_flux: FieldKey::new(surface_mesh, FieldName::NetSurfaceFlux),
      atm_temperature: FieldKey::new(atm_mesh, FieldName::Temperature),
      surface_temperature: FieldKey::new(surface_mesh, FieldName::Temperature),
    }
  }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RadiationStageIds {
  pub transfer: StageId,
}

/// Builder for the radiative-transfer stage. Carries the mesh keys, the
/// derived field keys, and a `RadiationParameters` block; nothing else.
#[derive(Clone, Debug)]
pub struct RadiationModel {
  atm_mesh: MeshKey,
  surface_mesh: MeshKey,
  fields: RadiationFields,
  params: RadiationParameters,
}

impl RadiationModel {
  pub fn new(atm_mesh: MeshKey, surface_mesh: MeshKey) -> Self {
    Self {
      atm_mesh,
      surface_mesh,
      fields: RadiationFields::for_meshes(atm_mesh, surface_mesh),
      params: RadiationParameters::default(),
    }
  }

  pub fn with_fields(mut self, fields: RadiationFields) -> Self {
    self.fields = fields;
    self
  }

  pub fn with_parameters(mut self, params: RadiationParameters) -> Self {
    self.params = params;
    self
  }

  pub fn atm_mesh(&self) -> MeshKey {
    self.atm_mesh
  }

  pub fn surface_mesh(&self) -> MeshKey {
    self.surface_mesh
  }

  pub fn fields(&self) -> RadiationFields {
    self.fields
  }

  pub fn parameters(&self) -> &RadiationParameters {
    &self.params
  }

  /// Register the fields lumen writes (heating tendency on the atm mesh,
  /// net surface flux on the surface mesh). Temperatures are assumed to
  /// be registered by the atmosphere / surface models that own them —
  /// lumen does not register state it doesn't write.
  pub fn register_fields<MA, MS>(
    &self,
    pleroma: &mut Pleroma,
    atm_mesh: &MA,
    surface_mesh: &MS,
  ) -> AetherResult<()>
  where
    MA: Mesh<3> + ?Sized,
    MS: Mesh<3> + ?Sized,
  {
    self.validate()?;

    pleroma.register_field(
      self.fields.heating_tendency,
      SoaField::<1>::zeros(atm_mesh.cell_count()),
    );
    pleroma.register_field(
      self.fields.net_surface_flux,
      SoaField::<1>::zeros(surface_mesh.cell_count()),
    );

    Ok(())
  }

  /// If the world has no other source for `ResourceKey::SunPosition`, the
  /// model can register a constant default so the stage is runnable
  /// out-of-the-box. Returns `true` if it inserted a value, `false` if
  /// one was already registered.
  pub fn register_default_sun_position(
    &self,
    pleroma: &mut Pleroma,
    direction: [f64; 3],
  ) -> bool {
    if pleroma
      .read_resource::<[f64; 3]>(ResourceKey::SunPosition)
      .is_some()
    {
      return false;
    }
    pleroma.register_resource(ResourceKey::SunPosition, direction);
    true
  }

  pub fn add_stages(
    &self,
    nexus: &mut Nexus,
  ) -> AetherResult<RadiationStageIds> {
    self.validate()?;
    let transfer = nexus.add(RadiativeTransferStep::new(
      self.atm_mesh,
      self.surface_mesh,
      self.fields.atm_temperature,
      self.fields.surface_temperature,
      self.fields.heating_tendency,
      self.fields.net_surface_flux,
      self.params,
    )?);
    Ok(RadiationStageIds { transfer })
  }

  fn validate(&self) -> AetherResult<()> {
    let on_atm = [self.fields.atm_temperature, self.fields.heating_tendency];
    let on_surface = [
      self.fields.surface_temperature,
      self.fields.net_surface_flux,
    ];
    if on_atm.iter().any(|k| k.mesh() != self.atm_mesh)
      || on_surface.iter().any(|k| k.mesh() != self.surface_mesh)
    {
      return Err(AetherError::new(LumenError::FieldMeshMismatch).context(
        format!(
          "atm_mesh {:?}, surface_mesh {:?}, fields {:?}",
          self.atm_mesh, self.surface_mesh, self.fields
        ),
      ));
    }
    self.params.validate()
  }
}
