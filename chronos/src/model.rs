// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Top-level builder for climatology aggregation, mirroring the
//! `thalassa::OceanModel` / `aer::AtmosphereModel` pattern: construct it,
//! `register_fields` against a pleroma (the live source fields must already be
//! registered), then `add_stages` against a nexus.

use nexus::{
  FieldKey, FieldName, FieldStorage, MeshKey, Nexus, Pleroma, SoaField,
  StageId, SubsystemId,
};
use utility::error::{AetherError, AetherResult};

use crate::{accumulator::ClimatologyAccumulatorStep, error::ChronosError};

/// Default climatology timescale (s): the e-folding time of the exponential
/// moving average. One day is a reasonable neutral default for atmospheric
/// aggregates; override with [`ClimatologyModel::with_timescale`].
pub const DEFAULT_TIMESCALE: f64 = 86_400.0;

/// A scalar quantity aggregated into a slowly-varying climatology mean. This is
/// a small fixed vocabulary (like `eidolon::query::ScalarQuantity`); each value
/// maps a live *source* `FieldName` to its climatology *mean* `FieldName`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ClimateQuantity {
  Temperature,
  Pressure,
  Humidity,
  /// Sea-surface temperature climatology. Built on the ocean mesh, where the
  /// prognostic sea-water temperature is [`FieldName::Temperature`] — so this
  /// sources that field (the ocean's own temperature), not the atmosphere-side
  /// `SeaSurfaceTemperature` mapping. Its mean is read back through
  /// `eidolon::query::ScalarQuantity::MeanSeaSurfaceTemperature`, which is bound
  /// to the ocean mesh, keeping producer and consumer on the same mesh.
  SeaSurfaceTemperature,
}

impl ClimateQuantity {
  /// The live field this aggregate averages.
  pub fn source_name(self) -> FieldName {
    match self {
      ClimateQuantity::Temperature => FieldName::Temperature,
      ClimateQuantity::Pressure => FieldName::Pressure,
      ClimateQuantity::Humidity => FieldName::Humidity,
      // The ocean's prognostic temperature is the SST source (see the variant
      // docs); the mean lands in `MeanSeaSurfaceTemperature` on the same mesh.
      ClimateQuantity::SeaSurfaceTemperature => FieldName::Temperature,
    }
  }

  /// The climatology field this aggregate is written into.
  pub fn mean_name(self) -> FieldName {
    match self {
      ClimateQuantity::Temperature => FieldName::MeanTemperature,
      ClimateQuantity::Pressure => FieldName::MeanPressure,
      ClimateQuantity::Humidity => FieldName::MeanHumidity,
      ClimateQuantity::SeaSurfaceTemperature => {
        FieldName::MeanSeaSurfaceTemperature
      }
    }
  }

  /// Source and mean `FieldKey`s on `mesh`.
  pub fn keys(self, mesh: MeshKey) -> (FieldKey, FieldKey) {
    (
      FieldKey::new(mesh, self.source_name()),
      FieldKey::new(mesh, self.mean_name()),
    )
  }
}

/// Aggregates one or more live fields on a single mesh into slowly-varying
/// climatology means. Inert with respect to physics — it registers mean fields
/// and adds accumulator stages, and never feeds back into the live state.
#[derive(Clone, Debug)]
pub struct ClimatologyModel {
  mesh: MeshKey,
  quantities: Vec<ClimateQuantity>,
  timescale: f64,
  subsystem: SubsystemId,
}

impl ClimatologyModel {
  pub fn new(mesh: MeshKey) -> Self {
    Self {
      mesh,
      quantities: Vec::new(),
      timescale: DEFAULT_TIMESCALE,
      subsystem: SubsystemId::DEFAULT,
    }
  }

  /// Aggregate `quantity` (deduplicated).
  pub fn with_quantity(mut self, quantity: ClimateQuantity) -> Self {
    if !self.quantities.contains(&quantity) {
      self.quantities.push(quantity);
    }
    self
  }

  /// Aggregate every quantity in `quantities`.
  pub fn with_quantities(
    mut self,
    quantities: impl IntoIterator<Item = ClimateQuantity>,
  ) -> Self {
    for quantity in quantities {
      self = self.with_quantity(quantity);
    }
    self
  }

  /// Set the climatology timescale `τ` (s).
  pub fn with_timescale(mut self, timescale: f64) -> Self {
    self.timescale = timescale;
    self
  }

  /// Place the accumulators on their own subsystem clock so the scheduler can
  /// step them slower than the live physics. Defaults to
  /// [`SubsystemId::DEFAULT`].
  pub fn with_subsystem(mut self, subsystem: SubsystemId) -> Self {
    self.subsystem = subsystem;
    self
  }

  pub fn mesh(&self) -> MeshKey {
    self.mesh
  }

  pub fn quantities(&self) -> &[ClimateQuantity] {
    &self.quantities
  }

  pub fn timescale(&self) -> f64 {
    self.timescale
  }

  pub fn subsystem(&self) -> SubsystemId {
    self.subsystem
  }

  /// Register a mean field for each quantity, initialised to a copy of the
  /// corresponding live field so the climatology is immediately valid (no
  /// spurious zero state on the first query). The live source fields must
  /// already be registered.
  pub fn register_fields(&self, pleroma: &mut Pleroma) -> AetherResult<()> {
    for &quantity in &self.quantities {
      let (source, mean) = quantity.keys(self.mesh);
      let initial: SoaField<1> = {
        let live: &SoaField<1> = pleroma.read(source).ok_or_else(|| {
          AetherError::new(ChronosError::MissingReadField).context(format!(
            "live source {:?} must be registered before climatology",
            source
          ))
        })?;
        SoaField::<1>::from_fn(live.len(), |i| [live.state(i)[0]])
      };
      pleroma.register_field(mean, initial);
    }
    Ok(())
  }

  /// Add one accumulator stage per quantity. Returns their stage ids.
  pub fn add_stages(&self, nexus: &mut Nexus) -> AetherResult<Vec<StageId>> {
    let mut ids = Vec::with_capacity(self.quantities.len());
    for &quantity in &self.quantities {
      let (source, mean) = quantity.keys(self.mesh);
      ids.push(nexus.add(ClimatologyAccumulatorStep::new(
        self.mesh,
        source,
        mean,
        self.timescale,
        self.subsystem,
      )?));
    }
    Ok(ids)
  }
}
