// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Moisture tracer sources — the evaporation half of the hydrological
//! cycle. Evaporation is the only way water vapour enters the atmosphere;
//! microphysics (condensation/precipitation) is the only way it leaves.
//!
//! `EvaporationStep` is a bulk air–sea moisture flux applied to the
//! atmosphere's bottom radial layer (the cells touching the sea surface).
//! It relaxes near-surface specific humidity toward saturation at the
//! sea-surface temperature:
//!
//! `Δq = k · dt · max(0, q_sat(T_sea, p) − q)`
//!
//! and adds the corresponding water mass to the prognostic `ρq`. The
//! sea-surface temperature arrives as a scalar field coupled onto the
//! atmosphere mesh (Phase 5 wires the ocean's surface layer to it); aer
//! never depends on the ocean crate.

use nexus::{FieldKey, FieldStorage, MeshKey, SoaField, Stage, StageContext};
use utility::{
  domain::CellId,
  error::{AetherError, AetherResult},
};

use crate::{
  error::AerError, init::AtmosphereSpec,
  microphysics::saturation_specific_humidity,
};

/// Radial-column layout of a cube-sphere shell, used to pick out the
/// bottom (surface-adjacent) layer. Cells are ordered
/// `panel · (ax·ay·layers) + layer · (ax·ay) + j·ax + i`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ShellColumns {
  pub panel_count: usize,
  pub angular_dims: [usize; 2],
  pub radial_layers: usize,
}

impl ShellColumns {
  pub fn cube_sphere(angular_dims: [usize; 2], radial_layers: usize) -> Self {
    Self {
      panel_count: 6,
      angular_dims,
      radial_layers,
    }
  }

  pub fn radial_stride(&self) -> usize {
    self.angular_dims[0] * self.angular_dims[1]
  }

  pub fn cells_per_panel(&self) -> usize {
    self.radial_stride() * self.radial_layers
  }

  pub fn cell_count(&self) -> usize {
    self.cells_per_panel() * self.panel_count
  }

  /// Iterate the global cell ids of the bottom (innermost) radial layer —
  /// the atmosphere cells adjacent to the surface.
  pub fn bottom_layer_cells(&self) -> impl Iterator<Item = usize> + '_ {
    let stride = self.radial_stride();
    (0..self.panel_count).flat_map(move |panel| {
      let base = panel * self.cells_per_panel();
      (0..stride).map(move |column| base + column)
    })
  }
}

/// Bulk air–sea evaporation into the atmosphere's bottom layer.
pub struct EvaporationStep {
  mesh: MeshKey,
  state: FieldKey,
  sea_surface_temperature: FieldKey,
  evaporation_flux: FieldKey,
  columns: ShellColumns,
  /// Relaxation rate toward sea-surface saturation (1/s).
  exchange_rate: f64,
  /// Optional per-cell open-water fraction (`FieldName::MoistureAvailability`,
  /// `[0,1]`) scaling the flux: `0` over land disables evaporation, `1` over ocean
  /// is full. `None` ⇒ `1` everywhere (the pre-masking behaviour).
  moisture_availability: Option<FieldKey>,
  reads: Vec<FieldKey>,
  writes: [FieldKey; 2],
}

impl EvaporationStep {
  pub fn new(
    mesh: MeshKey,
    state: FieldKey,
    sea_surface_temperature: FieldKey,
    evaporation_flux: FieldKey,
    columns: ShellColumns,
    exchange_rate: f64,
  ) -> AetherResult<Self> {
    let on_mesh = [state, sea_surface_temperature, evaporation_flux]
      .iter()
      .all(|f| f.mesh() == mesh);
    if !on_mesh {
      return Err(AetherError::new(AerError::FieldMeshMismatch).context(
        format!(
          "mesh {:?}, state {:?}, sst {:?}, evap {:?}",
          mesh, state, sea_surface_temperature, evaporation_flux
        ),
      ));
    }
    Ok(Self {
      mesh,
      state,
      sea_surface_temperature,
      evaporation_flux,
      columns,
      exchange_rate,
      moisture_availability: None,
      reads: vec![state, sea_surface_temperature],
      writes: [state, evaporation_flux],
    })
  }

  /// Gate evaporation by a per-cell open-water fraction
  /// (`FieldName::MoistureAvailability`, `[0,1]`) on the same mesh: `0` over land
  /// disables it, `1` over ocean is full. The key is added to `reads` so the
  /// scheduler orders its producer first. The mask is the moisture half of
  /// land–sea masking (the structural half is the tessera ocean cell-mask).
  pub fn with_moisture_availability(
    mut self,
    availability: FieldKey,
  ) -> AetherResult<Self> {
    if availability.mesh() != self.mesh {
      return Err(AetherError::new(AerError::FieldMeshMismatch).context(
        format!(
          "moisture availability {:?} not on evaporation mesh {:?}",
          availability, self.mesh
        ),
      ));
    }
    self.moisture_availability = Some(availability);
    self.reads.push(availability);
    Ok(self)
  }

  pub fn mesh(&self) -> MeshKey {
    self.mesh
  }
}

impl Stage for EvaporationStep {
  fn name(&self) -> &'static str {
    "aer_evaporation"
  }

  fn reads(&self) -> &[FieldKey] {
    &self.reads
  }

  fn writes(&self) -> &[FieldKey] {
    &self.writes
  }

  fn run(&mut self, mut ctx: StageContext<'_>) -> AetherResult<()> {
    let mesh_cell_count = ctx
      .world
      .tessera
      .mesh(self.mesh)
      .ok_or_else(|| {
        AetherError::new(AerError::MissingMesh)
          .context(format!("{:?}", self.mesh))
      })?
      .cell_count();
    if self.columns.cell_count() != mesh_cell_count {
      return Err(AetherError::new(AerError::FieldLengthMismatch).context(
        format!(
          "columns imply {} cells, mesh has {}",
          self.columns.cell_count(),
          mesh_cell_count
        ),
      ));
    }

    let dt = ctx.world.dt;
    if !dt.is_finite() || dt <= 0.0 {
      return Err(
        AetherError::new(AerError::InvalidTimeStep)
          .context(format!("dt {}", dt)),
      );
    }

    let spec = AtmosphereSpec::from_world_constants(ctx.world.constants)?;
    let gamma = spec.gamma();

    let sst: Vec<f64> = {
      let field: &SoaField<1> = ctx
        .world
        .fields
        .read(self.sea_surface_temperature)
        .ok_or_else(|| {
          AetherError::new(AerError::MissingReadField)
            .context(format!("{:?}", self.sea_surface_temperature))
        })?;
      if field.len() != mesh_cell_count {
        return Err(AetherError::new(AerError::FieldLengthMismatch));
      }
      field.component(0).as_ref().to_vec()
    };

    // Per-cell open-water fraction gating the flux. `None` ⇒ all-`1` (no masking).
    let availability: Vec<f64> = match self.moisture_availability {
      Some(key) => {
        let field: &SoaField<1> =
          ctx.world.fields.read(key).ok_or_else(|| {
            AetherError::new(AerError::MissingReadField)
              .context(format!("{:?}", key))
          })?;
        if field.len() != mesh_cell_count {
          return Err(AetherError::new(AerError::FieldLengthMismatch));
        }
        field.component(0).as_ref().to_vec()
      }
      None => vec![1.0; mesh_cell_count],
    };

    // Compute added vapour per surface cell, then write state + flux.
    let mut added = vec![0.0; mesh_cell_count];
    let new_states: Vec<(usize, [f64; 6])> = {
      let state: &SoaField<6> =
        ctx.world.fields.read(self.state).ok_or_else(|| {
          AetherError::new(AerError::MissingReadField)
            .context(format!("{:?}", self.state))
        })?;
      if state.len() != mesh_cell_count {
        return Err(AetherError::new(AerError::FieldLengthMismatch));
      }

      let mut updates = Vec::new();
      for cell in self.columns.bottom_layer_cells() {
        let mut s = state.state(CellId::from(cell));
        let rho = s[0];
        if !rho.is_finite() || rho <= 0.0 {
          continue;
        }
        let kinetic = 0.5 / rho * (s[1] * s[1] + s[2] * s[2] + s[3] * s[3]);
        let p = (gamma - 1.0) * (s[4] - kinetic);
        let t_sea = sst[cell];
        if p <= 0.0 || !t_sea.is_finite() || t_sea <= 0.0 {
          continue;
        }
        let q = s[5] / rho;
        let q_sat_sea = saturation_specific_humidity(t_sea, p);
        // Gate by the open-water fraction: 0 over land, 1 over open ocean.
        let avail = availability[cell].clamp(0.0, 1.0);
        let dq = avail * self.exchange_rate * dt * (q_sat_sea - q);
        if dq > 0.0 {
          // The vapour carries its latent heat implicitly: the air's sensible
          // energy is unchanged here (sea-surface evaporation draws its latent
          // heat from the OCEAN, not the air). The condensation half releases
          // that latent heat into the air; conservation closes by debiting the
          // ocean surface for `LATENT · evaporation_flux` (see the air–sea
          // latent-heat sink stage), so the air–sea heat flux is real and
          // bounded by the ocean's finite heat capacity.
          let delta_rho_q = rho * dq;
          s[5] += delta_rho_q;
          added[cell] = delta_rho_q / dt; // kg/m³/s diagnostic
          updates.push((cell, s));
        }
      }
      updates
    };

    {
      let state: &mut SoaField<6> =
        ctx.world.fields.write(self.state).ok_or_else(|| {
          AetherError::new(AerError::MissingWriteField)
            .context(format!("{:?}", self.state))
        })?;
      for (cell, s) in new_states {
        state.write(CellId::from(cell), &s);
      }
    }

    {
      let flux: &mut SoaField<1> = ctx
        .world
        .fields
        .write(self.evaporation_flux)
        .ok_or_else(|| {
          AetherError::new(AerError::MissingWriteField)
            .context(format!("{:?}", self.evaporation_flux))
        })?;
      if flux.len() != mesh_cell_count {
        return Err(AetherError::new(AerError::FieldLengthMismatch));
      }
      for (cell, rate) in added.into_iter().enumerate() {
        flux.write(CellId::from(cell), &[rate]);
      }
    }

    Ok(())
  }
}
