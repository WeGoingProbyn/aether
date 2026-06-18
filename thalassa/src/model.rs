// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Top-level builder for the thermodynamic ocean, mirroring the
//! `aer::AtmosphereModel` / `terra::SurfaceThermalModel` pattern: construct
//! it, `register_fields` against a pleroma + the ocean mesh, then
//! `add_stages` against a nexus.

use nexus::{
  FieldKey, FieldName, MeshKey, Nexus, Pleroma, SoaField, StageId, SubsystemId,
};
use tessera::mesh::Mesh;
use utility::error::{AetherError, AetherResult};

use crate::{error::ThalassaError, thermodynamics::OceanThermodynamicsStep};

/// Radial-column layout of the ocean cube-sphere shell. Cells are ordered
/// `panel · (ax·ay·layers) + layer · (ax·ay) + j·ax + i`, so within a panel
/// the radial (vertical) neighbour of a cell is `±(ax·ay)` away. Layer
/// index increases outward: layer `layers - 1` is the sea surface.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OceanColumnLayout {
  pub panel_count: usize,
  pub angular_dims: [usize; 2],
  pub radial_layers: usize,
}

impl OceanColumnLayout {
  /// A standard 6-panel cube-sphere ocean shell.
  pub fn cube_sphere(angular_dims: [usize; 2], radial_layers: usize) -> Self {
    Self {
      panel_count: 6,
      angular_dims,
      radial_layers,
    }
  }

  /// Cells stacked between adjacent radial layers of one angular column.
  pub fn radial_stride(&self) -> usize {
    self.angular_dims[0] * self.angular_dims[1]
  }

  pub fn cells_per_panel(&self) -> usize {
    self.radial_stride() * self.radial_layers
  }

  pub fn cell_count(&self) -> usize {
    self.cells_per_panel() * self.panel_count
  }

  /// Outermost layer index — the sea surface that exchanges with the air.
  pub fn surface_layer(&self) -> usize {
    self.radial_layers - 1
  }

  fn is_valid(&self) -> bool {
    self.panel_count > 0
      && self.angular_dims[0] > 0
      && self.angular_dims[1] > 0
      && self.radial_layers > 0
  }
}

/// Field keys thalassa touches. Both live on the ocean mesh: the prognostic
/// sea-water temperature, and the net surface heat flux (W/m²) deposited
/// into the surface layer by radiation / air–sea coupling.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OceanFields {
  pub temperature: FieldKey,
  pub net_flux: FieldKey,
}

impl OceanFields {
  pub const fn for_mesh(mesh: MeshKey) -> Self {
    Self {
      temperature: FieldKey::new(mesh, FieldName::Temperature),
      net_flux: FieldKey::new(mesh, FieldName::NetSurfaceFlux),
    }
  }
}

/// Thermodynamic ocean: a radial stack of water layers whose surface layer
/// absorbs the net surface flux while heat diffuses vertically toward the
/// deep ocean.
#[derive(Clone, Debug)]
pub struct OceanModel {
  mesh: MeshKey,
  fields: OceanFields,
  layout: OceanColumnLayout,
  initial_temperature: f64,
  /// Thickness of each radial layer (m). Uniform for the first proof.
  layer_thickness: f64,
  /// Sea-water density (kg/m³).
  density: f64,
  /// Sea-water specific heat capacity (J/(kg·K)).
  specific_heat: f64,
  /// Vertical thermal diffusivity (m²/s) — an eddy diffusivity lumping
  /// turbulent mixing, not molecular conduction.
  vertical_diffusivity: f64,
  subsystem: SubsystemId,
}

impl OceanModel {
  pub fn new(mesh: MeshKey, layout: OceanColumnLayout) -> Self {
    Self {
      mesh,
      fields: OceanFields::for_mesh(mesh),
      layout,
      initial_temperature: 288.0,
      layer_thickness: 50.0,
      density: 1025.0,
      specific_heat: 3990.0,
      vertical_diffusivity: 1.0e-3,
      subsystem: SubsystemId::DEFAULT,
    }
  }

  pub fn with_initial_temperature(mut self, temperature: f64) -> Self {
    self.initial_temperature = temperature;
    self
  }

  pub fn with_layer_thickness(mut self, thickness: f64) -> Self {
    self.layer_thickness = thickness;
    self
  }

  pub fn with_density(mut self, density: f64) -> Self {
    self.density = density;
    self
  }

  pub fn with_specific_heat(mut self, specific_heat: f64) -> Self {
    self.specific_heat = specific_heat;
    self
  }

  pub fn with_vertical_diffusivity(mut self, diffusivity: f64) -> Self {
    self.vertical_diffusivity = diffusivity;
    self
  }

  /// Place the ocean on its own subsystem clock so the scheduler can step
  /// it slower than the (CFL-limited) atmosphere. Defaults to
  /// [`SubsystemId::DEFAULT`].
  pub fn with_subsystem(mut self, subsystem: SubsystemId) -> Self {
    self.subsystem = subsystem;
    self
  }

  pub fn mesh(&self) -> MeshKey {
    self.mesh
  }

  pub fn fields(&self) -> OceanFields {
    self.fields
  }

  pub fn layout(&self) -> OceanColumnLayout {
    self.layout
  }

  pub fn subsystem(&self) -> SubsystemId {
    self.subsystem
  }

  /// Heat capacity per unit area of a single layer, `ρ·c_p·Δz` (J/(K·m²)).
  pub fn layer_heat_capacity_per_area(&self) -> f64 {
    self.density * self.specific_heat * self.layer_thickness
  }

  /// Register the prognostic temperature field (initialised uniform) and a
  /// zeroed net-flux field. Thalassa owns the net-flux registration because
  /// it is the consumer; the air–sea coupling stage writes it each tick.
  pub fn register_fields<M>(
    &self,
    pleroma: &mut Pleroma,
    mesh: &M,
  ) -> AetherResult<()>
  where
    M: Mesh<3> + ?Sized,
  {
    self.validate(mesh.cell_count())?;
    pleroma.register_field(
      self.fields.temperature,
      SoaField::<1>::from_fn(mesh.cell_count(), |_| [self.initial_temperature]),
    );
    pleroma.register_field(
      self.fields.net_flux,
      SoaField::<1>::zeros(mesh.cell_count()),
    );
    Ok(())
  }

  pub fn add_stages(&self, nexus: &mut Nexus) -> AetherResult<StageId> {
    Ok(nexus.add(OceanThermodynamicsStep::new(
      self.mesh,
      self.fields.temperature,
      self.fields.net_flux,
      self.layout,
      self.layer_thickness,
      self.layer_heat_capacity_per_area(),
      self.vertical_diffusivity,
      self.subsystem,
    )?))
  }

  fn validate(&self, mesh_cell_count: usize) -> AetherResult<()> {
    if self.fields.temperature.mesh() != self.mesh
      || self.fields.net_flux.mesh() != self.mesh
    {
      return Err(
        AetherError::new(ThalassaError::FieldMeshMismatch)
          .context(format!("mesh {:?}, fields {:?}", self.mesh, self.fields)),
      );
    }
    if !self.layout.is_valid() || self.layout.cell_count() != mesh_cell_count {
      return Err(
        AetherError::new(ThalassaError::InvalidColumnLayout).context(format!(
          "layout {:?} implies {} cells, mesh has {}",
          self.layout,
          self.layout.cell_count(),
          mesh_cell_count
        )),
      );
    }
    if !self.initial_temperature.is_finite() || self.initial_temperature <= 0.0
    {
      return Err(
        AetherError::new(ThalassaError::InvalidOceanTemperature)
          .context(format!("initial_temperature {}", self.initial_temperature)),
      );
    }
    let props_ok = [
      self.layer_thickness,
      self.density,
      self.specific_heat,
      self.vertical_diffusivity,
    ]
    .iter()
    .all(|v| v.is_finite() && *v > 0.0);
    if !props_ok {
      return Err(AetherError::new(ThalassaError::InvalidColumnProperties));
    }
    Ok(())
  }
}
