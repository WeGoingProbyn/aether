// ====== Conversions ================ //
pub const POWER_THREE:         f64 = 1_000f64;
pub const POWER_SIX:           f64 = 1_000_000f64;
pub const POWER_NINE:          f64 = 1_000_000_000f64;

// ====== Physical constants ========= //
pub const G:                   f64 = 6.6743e-11;
pub const AVOGADRO:            f64 = 6.02214076e23;
pub const BOLTZMANN:           f64 = 1.380649e-23;
pub const UNIVERSAL_GAS:       f64 = 8.314462618;
pub const STEFAN_BOLTZMANN:    f64 = 5.670374419e-8;

// ====== Orbital constants ========= //
pub const AU:                  f64 = 1.496e11; 

// ====== Fluid dynamics ============ //

// ====== Solar constants =========== //
pub const SOLAR_LUMIN:         f64 = 3.828e26;
pub const SOLAR_MASS:          f64 = 1.9885e30;        // kg
pub const SOLAR_RADIUS:        f64 = 6.9634e8;         // m
pub const SOLAR_SURFACE_TEMP:  f64 = 5772.0;           // K
pub const SOLAR_CORE_TEMP:     f64 = 1.57e7;           // K

// ====== Mercury =================== //
pub const MERCURY_MASS:        f64 = 3.30e23;                // kg
pub const MERCURY_RADIUS:      f64 = 2440f64 * POWER_THREE;  // m
pub const MERCURY_ORBIT:       f64 = 0.387 * AU;             // m
pub const MERCURY_DAY:         f64 = 5_067_000.0;
pub const MERCURY_SURFACE_T:   f64 = 440.0;
pub const MERCURY_SURFACE_P:   f64 = 0.0;
pub const MERCURY_AXIAL_TILT:  f64 = 0.034;

// ====== Venus ===================== //
pub const VENUS_MASS:          f64 = 4.87e24;                // kg
pub const VENUS_RADIUS:        f64 = 6052f64 * POWER_THREE;  // m
pub const VENUS_MEAN_MOL_MASS: f64 = 0.044;
pub const VENUS_ORBIT:         f64 = 0.723 * AU;             // m
pub const VENUS_DAY:           f64 = -20_995_200.0;
pub const VENUS_ALBEDO:        f64 = 0.75;
pub const VENUS_SURFACE_T:     f64 = 737.0;
pub const VENUS_SURFACE_P:     f64 = 9.2e6;                  // bar
pub const VENUS_AXIAL_TILT:    f64 = 177.4;

// ====== Earth ===================== //
pub const EARTH_MASS:          f64 = 5.97e24;                // kg
pub const EARTH_RADIUS:        f64 = 6371f64 * POWER_THREE;  // m
pub const EARTH_ORBIT:         f64 = AU;                     // m
pub const EARTH_DAY:           f64 = 86400.0;                // s
pub const EARTH_MEAN_MOL_MASS: f64 = 0.02897;
pub const EARTH_ALBEDO:        f64 = 0.3;
pub const EARTH_SURFACE_T:     f64 = 288.0;
pub const EARTH_SURFACE_P:     f64 = 101325.0;               // bar
pub const EARTH_AXIAL_TILT:    f64 = 23.44;


// ====== Mars ====================== //
pub const MARS_MASS:           f64 = 6.42e23;                // kg
pub const MARS_RADIUS:         f64 = 3390f64 * POWER_THREE;  // m
pub const MARS_ORBIT:          f64 = 1.523 * AU;             // m
pub const MARS_DAY:            f64 = 88775.0;                // s
pub const MARS_MEAN_MOL_MASS:  f64 = 0.043;
pub const MARS_ALBEDO:         f64 = 0.25;
pub const MARS_SURFACE_T:      f64 = 210.0;
pub const MARS_SURFACE_P:      f64 = 600.0;                  // bar
pub const MARS_AXIAL_TILT:     f64 = 25.19;

// ====== Jupiter =================== //
pub const JUPITER_MASS:        f64 = 1.90e27;                // kg
pub const JUPITER_RADIUS:      f64 = 69911f64 * POWER_THREE; // m
pub const JUPITER_ORBIT:       f64 = 5.203 * AU;             // m
pub const JUPITER_DAY:         f64 = 35730.0;                // s
pub const JUPITER_HEAT_FACTOR: f64 = 1.6;
pub const JUPITER_SURFACE_T:   f64 = 165.0;
pub const JUPITER_AXIAL_TILT:  f64 = 3.13;

// ====== Saturn ==================== //
pub const SATURN_MASS:         f64 = 5.68e26;
pub const SATURN_RADIUS:       f64 = 58232.0 * POWER_THREE;
pub const SATURN_ORBIT:        f64 = 9.537 * AU;
pub const SATURN_DAY:          f64 = 38340.0;
pub const SATURN_HEAT_FACTOR:  f64 = 2.3;
pub const SATURN_SURFACE_T:    f64 = 134.0;
pub const SATURN_AXIAL_TILT:   f64 = 26.73;

// ====== Neptune =================== //
pub const NEPTUNE_MASS:        f64 = 1.02e26;
pub const NEPTUNE_RADIUS:      f64 = 24622.0 * POWER_THREE;
pub const NEPTUNE_ORBIT:       f64 = 30.07 * AU;
pub const NEPTUNE_DAY:         f64 = 57960.0;
pub const NEPTUNE_HEAT_FACTOR: f64 = 2.3;
pub const NEPTUNE_SURFACE_T:   f64 = 72.0;
pub const NEPTUNE_AXIAL_TILT:  f64 = 2.7;

// ====== Rocky Generics ======== //
pub const DENSITY_ROCKY_MIN:   f64 = 3000.0;
pub const DENSITY_ROCKY_MAX:   f64 = 5500.0;

// ====== Gas giant generics ==== //
pub const DENSITY_GAS_MIN:     f64 = 600.0;
pub const DENSITY_GAS_MAX:     f64 = 1600.0;
pub const GAS_MEAN_MOL_MASS:   f64 = 0.0023;
pub const GAS_R:               f64 = UNIVERSAL_GAS / GAS_MEAN_MOL_MASS;
pub const NOMINAL_GAS_PRESS:   f64 = 1.0e5; // 1 bar reference

// ====== Species =============== //
pub const HYDROGEN_MOLAR_MASS: f64 = 0.002016;  // kg/mol
pub const HELIUM_MOLAR_MASS:   f64 = 0.0040026; // kg/mol
pub const OXYGEN_MOLAR_MASS:   f64 = 0.032;
pub const CO2_MOLAR_MASS:      f64 = 0.044;
pub const METHANE_MOLAR_MASS:  f64 = 0.016043;
pub const NITROGEN_MOLAR_MASS: f64 = 0.028;

// ====== Computed ================== //
pub const fn gravity(mass: f64, radius: f64) -> f64 {
  G * mass / (radius * radius)
}

pub const fn angular_velocity(day: f64) -> f64 {
  2.0 * std::f64::consts::PI / day
}

pub fn solar_flux(luminosity: f64, distance: f64) -> f64 {
  luminosity / (4.0 * std::f64::consts::PI * distance.powi(2))
}

pub fn escape_velocity(mass: f64, radius: f64) -> f64 {
  (2.0 * G * mass / radius).sqrt()
}

pub fn equilibrium_temperature(albedo: f64, luminosity: f64, distance: f64) -> f64 {
  ((1.0 - albedo) * luminosity
    / (16.0 * std::f64::consts::PI * STEFAN_BOLTZMANN * distance.powi(2)))
    .powf(0.25)
}

pub const fn specific_gas_constant(molar_mass: f64) -> f64 {
  UNIVERSAL_GAS / molar_mass
}

pub const fn scale_height(temp: f64, molar_mass: f64, gravity: f64) -> f64 {
  let particle_mass = molar_mass / AVOGADRO;
  BOLTZMANN * temp / (particle_mass * gravity)
}

pub fn pressure_at_height(p0: f64, height: f64, scale_height: f64) -> f64 {
  p0 * (-height / scale_height).exp()
}

pub fn coriolis_parameter(omega: f64, latitude: f64) -> f64 {
  2.0 * omega * latitude.sin()
}

pub const fn lapse_rate(gravity: f64, cp: f64) -> f64 {
  gravity / cp
}

pub fn speed_of_sound(gamma: f64, r_specific: f64, temp: f64) -> f64 {
  (gamma * r_specific * temp).sqrt()
}

pub fn blackbody_radiation(temp: f64) -> f64 {
  STEFAN_BOLTZMANN * temp.powi(4)
}

pub fn stellar_luminosity(radius: f64, temp: f64) -> f64 {
  4.0 * std::f64::consts::PI * radius.powi(2) * STEFAN_BOLTZMANN * temp.powi(4)
}

pub fn effective_temperature(eq_temp: f64, internal_factor: f64) -> f64 {
  (eq_temp.powi(4) * internal_factor).powf(0.25)
}

pub fn orbital_velocity(star_mass: f64, distance: f64) -> f64 {
  (G * star_mass / distance).sqrt()
}

pub fn gamma(dof: f64) -> f64 {
  1.0 + 2.0 / dof
}

