use std::collections::HashMap;

use utility::{
  constants::{
    CO2_MOLAR_MASS, HELIUM_MOLAR_MASS, HYDROGEN_MOLAR_MASS,
    METHANE_MOLAR_MASS, NITROGEN_MOLAR_MASS, OXYGEN_MOLAR_MASS, gamma,
    specific_gas_constant,
  },
  maths::vector::Vector,
};

#[derive(Clone, Debug)]
pub(crate) struct Body {
  pub mass: f64,
  pub radius: f64,
  pub position: Vector<f64, 3>,
  pub velocity: Vector<f64, 3>,
}

impl Body {
  pub fn new(
    mass: f64,
    radius: f64,
    pos: Vector<f64, 3>,
    vel: Vector<f64, 3>,
  ) -> Body {
    Body { mass, radius, position: pos, velocity: vel }
  }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct GasProperties {
  pub molar_mass: f64,
  pub gamma: f64,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum Species {
  Hydrogen,
  Helium,
  Oxygen,
  CarbonDioxide,
  Nitrogen,
  Methane,
}

impl Species {
  /// Molar mass — temperature-independent. kg/mol.
  pub fn molar_mass(&self) -> f64 {
    match self {
      Species::Hydrogen => HYDROGEN_MOLAR_MASS,
      Species::Helium => HELIUM_MOLAR_MASS,
      Species::Oxygen => OXYGEN_MOLAR_MASS,
      Species::Nitrogen => NITROGEN_MOLAR_MASS,
      Species::CarbonDioxide => CO2_MOLAR_MASS,
      Species::Methane => METHANE_MOLAR_MASS,
    }
  }

  /// Effective degrees of freedom at the given temperature. Polyatomics gain
  /// vibrational modes as temperature rises, smoothly enough for our purposes.
  pub fn degrees_of_freedom(&self, temperature: f64) -> f64 {
    match self {
      Species::Helium => 3.0,
      Species::Hydrogen => 5.0,
      Species::Nitrogen => 5.0,
      Species::Oxygen => 5.0,
      Species::CarbonDioxide => {
        let extra = if temperature < 300.0 {
          0.0
        } else if temperature < 800.0 {
          1.5
        } else {
          3.0
        };
        6.0 + extra
      }
      Species::Methane => {
        let base = 6.0;
        if temperature < 500.0 {
          base
        } else if temperature < 1000.0 {
          base + 2.0
        } else {
          15.0
        }
      }
    }
  }

  /// Combined molar mass and γ for this species at the given temperature.
  pub fn properties(&self, temperature: f64) -> GasProperties {
    GasProperties {
      molar_mass: self.molar_mass(),
      gamma: gamma(self.degrees_of_freedom(temperature)),
    }
  }
}

#[derive(Clone, Debug)]
pub struct Atmosphere {
  pub composition: HashMap<Species, f64>,
  pub albedo: Option<f64>,
}

impl Atmosphere {
  pub fn new(
    composition: HashMap<Species, f64>,
    albedo: Option<f64>,
  ) -> Atmosphere {
    Atmosphere { composition, albedo }
  }

  /// Mole fractions sum to ≤ 1 (the residual is treated as inert / unmodelled).
  pub fn validate_elements(&self) -> bool {
    let sum: f64 = self.composition.values().sum();
    sum <= 1.0 + 1e-12
  }

  /// Rescale mole fractions to sum to 1.
  pub fn normalise_components(&mut self) {
    let sum: f64 = self.composition.values().sum();
    if sum > 0.0 {
      for v in self.composition.values_mut() {
        *v /= sum;
      }
    }
  }

  /// Bulk thermodynamic properties at the given temperature.
  ///
  /// Mixture quantities, with mole fractions `x_i` (auto-normalised so the
  /// caller doesn't have to):
  ///
  /// * **Molar mass:** `M_mix = Σ x_i · M_i` (kg/mol)
  /// * **Effective DOF:** `f_mix = Σ x_i · f_i(T)` — temperature-dependent
  ///   because polyatomics activate vibrational modes
  /// * **γ:** `1 + 2/f_mix` — the standard ideal-gas relation; this is
  ///   exactly equivalent to `(c_p/c_v)_mix` evaluated mole-fraction-weighted
  /// * **Specific gas constant:** `R/M_mix` (J/(kg·K))
  ///
  /// Panics if the composition is empty.
  pub fn properties(&self, temperature: f64) -> AtmosphereProperties {
    let total: f64 = self.composition.values().sum();
    assert!(total > 0.0, "atmosphere composition is empty");

    let mut molar_mass = 0.0;
    let mut total_dof = 0.0;
    for (species, fraction) in &self.composition {
      let normalised = fraction / total;
      molar_mass += normalised * species.molar_mass();
      total_dof += normalised * species.degrees_of_freedom(temperature);
    }

    AtmosphereProperties {
      molar_mass,
      gamma: gamma(total_dof),
      gas_constant: specific_gas_constant(molar_mass),
    }
  }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct AtmosphereProperties {
  pub molar_mass: f64,
  pub gamma: f64,
  pub gas_constant: f64,
}
