// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use nexus::{
  FieldKey, FieldStorage, MeshKey, SoaField, Stage, StageContext, SubsystemId,
};
use utility::{
  domain::CellId,
  error::{AetherError, AetherResult},
};

use crate::{error::ThalassaError, model::OceanColumnLayout};

/// One explicit step of ocean column thermodynamics:
///
/// 1. the surface layer absorbs the net surface flux `Q` (W/m²):
///    `dT_surface = dt · Q / (ρ·c_p·Δz)`;
/// 2. heat diffuses vertically between radial layers:
///    `dT_k = dt · κ · (T_{k+1} − 2·T_k + T_{k−1}) / Δz²`,
///    with insulated top and bottom (zero-flux boundaries).
///
/// New temperatures are computed from the old field simultaneously (a
/// forward-Euler update), so the stage reads and writes the same
/// temperature field.
pub struct OceanThermodynamicsStep {
  mesh: MeshKey,
  temperature: FieldKey,
  net_flux: FieldKey,
  layout: OceanColumnLayout,
  layer_thickness: f64,
  surface_heat_capacity_per_area: f64,
  vertical_diffusivity: f64,
  subsystem: SubsystemId,
  reads: [FieldKey; 2],
  writes: [FieldKey; 1],
}

impl OceanThermodynamicsStep {
  #[allow(clippy::too_many_arguments)]
  pub fn new(
    mesh: MeshKey,
    temperature: FieldKey,
    net_flux: FieldKey,
    layout: OceanColumnLayout,
    layer_thickness: f64,
    surface_heat_capacity_per_area: f64,
    vertical_diffusivity: f64,
    subsystem: SubsystemId,
  ) -> AetherResult<Self> {
    if temperature.mesh() != mesh || net_flux.mesh() != mesh {
      return Err(AetherError::new(ThalassaError::FieldMeshMismatch).context(
        format!(
          "mesh {:?}, temperature {:?}, net_flux {:?}",
          mesh, temperature, net_flux
        ),
      ));
    }
    Ok(Self {
      mesh,
      temperature,
      net_flux,
      layout,
      layer_thickness,
      surface_heat_capacity_per_area,
      vertical_diffusivity,
      subsystem,
      reads: [net_flux, temperature],
      writes: [temperature],
    })
  }

  pub fn mesh(&self) -> MeshKey {
    self.mesh
  }
}

impl Stage for OceanThermodynamicsStep {
  fn name(&self) -> &'static str {
    "thalassa_ocean_thermodynamics"
  }

  fn reads(&self) -> &[FieldKey] {
    &self.reads
  }

  fn writes(&self) -> &[FieldKey] {
    &self.writes
  }

  fn subsystem(&self) -> SubsystemId {
    self.subsystem
  }

  fn run(&mut self, mut ctx: StageContext<'_>) -> AetherResult<()> {
    let mesh_cell_count = ctx
      .world
      .tessera
      .mesh(self.mesh)
      .ok_or_else(|| {
        AetherError::new(ThalassaError::MissingMesh)
          .context(format!("{:?}", self.mesh))
      })?
      .cell_count();

    // Cell-activity mask (e.g. land columns of a global ocean shell). Absent ⇒
    // every cell active (the backward-compatible default). The mask is keyed by
    // global CellId, the same space as the temperature field. Fetched up front
    // (it borrows `tessera`, not `fields`) so it can be consulted while the
    // temperature buffer is borrowed below.
    let cell_mask = ctx.world.tessera.cell_mask(self.mesh);
    if let Some(mask) = cell_mask {
      if mask.len() != mesh_cell_count {
        return Err(
          AetherError::new(ThalassaError::FieldLengthMismatch).context(
            format!(
              "cell mask {} cells, mesh has {}",
              mask.len(),
              mesh_cell_count
            ),
          ),
        );
      }
    }

    let dt = ctx.world.dt;
    if !dt.is_finite() || dt <= 0.0 {
      return Err(
        AetherError::new(ThalassaError::InvalidTimeStep)
          .context(format!("dt {}", dt)),
      );
    }
    if self.layout.cell_count() != mesh_cell_count {
      return Err(
        AetherError::new(ThalassaError::FieldLengthMismatch).context(format!(
          "layout cells {}, mesh cells {}",
          self.layout.cell_count(),
          mesh_cell_count
        )),
      );
    }

    let flux: Vec<f64> = {
      let field: &SoaField<1> =
        ctx.world.fields.read(self.net_flux).ok_or_else(|| {
          AetherError::new(ThalassaError::MissingReadField)
            .context(format!("{:?}", self.net_flux))
        })?;
      if field.len() != mesh_cell_count {
        return Err(AetherError::new(ThalassaError::FieldLengthMismatch));
      }
      field.component(0).as_ref().to_vec()
    };

    let temperature: &mut SoaField<1> =
      ctx.world.fields.write(self.temperature).ok_or_else(|| {
        AetherError::new(ThalassaError::MissingWriteField)
          .context(format!("{:?}", self.temperature))
      })?;
    if temperature.len() != mesh_cell_count {
      return Err(AetherError::new(ThalassaError::FieldLengthMismatch));
    }

    let old: Vec<f64> = temperature.component(0).as_ref().to_vec();
    let mut new = old.clone();

    let layout = self.layout;
    let stride = layout.radial_stride();
    let layers = layout.radial_layers;
    let surface_layer = layout.surface_layer();
    let dz2 = self.layer_thickness * self.layer_thickness;
    let diffusion = self.vertical_diffusivity * dt / dz2;
    let flux_gain = dt / self.surface_heat_capacity_per_area;

    for panel in 0..layout.panel_count {
      let panel_base = panel * layout.cells_per_panel();
      for column in 0..stride {
        let base = panel_base + column;
        // Skip masked-out (e.g. land) columns: land/ocean is a lat/lon property,
        // so the whole column shares its surface cell's mask bit. Skipped columns
        // keep their inert initial temperature — the ocean never evolves there.
        if let Some(mask) = cell_mask {
          let surface_cell = base + surface_layer * stride;
          if !mask.is_active(CellId::from(surface_cell)) {
            continue;
          }
        }
        for k in 0..layers {
          let here = base + k * stride;
          let t_here = old[here];
          let mut delta = 0.0;
          if k + 1 < layers {
            delta += diffusion * (old[here + stride] - t_here);
          }
          if k > 0 {
            delta += diffusion * (old[here - stride] - t_here);
          }
          if k == surface_layer {
            delta += flux_gain * flux[here];
          }
          let updated = t_here + delta;
          if !updated.is_finite() || updated <= 0.0 {
            return Err(
              AetherError::new(ThalassaError::InvalidOceanTemperature)
                .context(format!("cell {} temperature {}", here, updated)),
            );
          }
          new[here] = updated;
        }
      }
    }

    for (cell, value) in new.into_iter().enumerate() {
      temperature.write(CellId::from(cell), &[value]);
    }
    Ok(())
  }
}

impl OceanThermodynamicsStep {
  /// Stable explicit-diffusion timestep limit `Δz² / (2κ)` for the column.
  /// The ocean subsystem cadence should stay below this.
  pub fn max_stable_dt(&self) -> f64 {
    let dz2 = self.layer_thickness * self.layer_thickness;
    dz2 / (2.0 * self.vertical_diffusivity)
  }
}
