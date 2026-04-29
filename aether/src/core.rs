use std::collections::HashMap;

use cosmo::kind::{BodyKind, CelestialBody};
use nexus::{AtmosphereConstants, CompiledNexus, WorldConstants, WorldId};
use pleroma::Pleroma;
use tessera::world_mesh::Tessera;
use utility::{domain::SystemId, error::AetherResult, thread::pool::Pool};

/// Runtime state for one simulated body.
pub struct World {
  id: WorldId,
  seed: CelestialBody,
  constants: WorldConstants,
  tessera: Tessera,
  pleroma: Pleroma,
  nexus: CompiledNexus,
}

impl World {
  pub fn new(
    id: WorldId,
    seed: CelestialBody,
    tessera: Tessera,
    pleroma: Pleroma,
    nexus: CompiledNexus,
  ) -> Self {
    let constants = world_constants_from_seed(&seed);
    Self {
      id,
      seed,
      constants,
      tessera,
      pleroma,
      nexus,
    }
  }

  pub fn id(&self) -> WorldId {
    self.id
  }

  pub fn seed(&self) -> &CelestialBody {
    &self.seed
  }

  pub fn constants(&self) -> &WorldConstants {
    &self.constants
  }

  pub fn tessera(&self) -> &Tessera {
    &self.tessera
  }

  pub fn tessera_mut(&mut self) -> &mut Tessera {
    &mut self.tessera
  }

  pub fn pleroma(&self) -> &Pleroma {
    &self.pleroma
  }

  pub fn pleroma_mut(&mut self) -> &mut Pleroma {
    &mut self.pleroma
  }

  pub fn runtime_parts_mut(&mut self) -> (&Tessera, &mut Pleroma) {
    (&self.tessera, &mut self.pleroma)
  }

  pub fn tick(&mut self, pool: &Pool, dt: f64) -> AetherResult<()> {
    self.nexus.tick(
      self.id,
      &self.tessera,
      &self.constants,
      &mut self.pleroma,
      pool,
      dt,
    )
  }
}

pub fn world_constants_from_seed(seed: &CelestialBody) -> WorldConstants {
  WorldConstants {
    mass: seed.mass(),
    radius: seed.radius(),
    surface_gravity: seed.surface_gravity(),
    atmosphere: atmosphere_constants_from_seed(seed),
  }
}

fn atmosphere_constants_from_seed(
  seed: &CelestialBody,
) -> Option<AtmosphereConstants> {
  match seed.kind() {
    BodyKind::RockyBody(body) => {
      let atmosphere = body.atmosphere.as_ref()?;
      let properties = atmosphere.properties(body.surface_temperature);
      Some(AtmosphereConstants {
        reference_temperature: body.surface_temperature,
        reference_pressure: body.surface_pressure,
        gamma: properties.gamma,
        gas_constant: properties.gas_constant,
        molar_mass: properties.molar_mass,
        albedo: atmosphere.albedo,
        angular_velocity: body.angular_velocity,
        axial_tilt: body.axial_tilt,
      })
    }
    BodyKind::GasGiant(body) => {
      let properties = body.atmosphere.properties(body.reference_temperature);
      Some(AtmosphereConstants {
        reference_temperature: body.reference_temperature,
        reference_pressure: body.reference_pressure,
        gamma: properties.gamma,
        gas_constant: properties.gas_constant,
        molar_mass: properties.molar_mass,
        albedo: body.atmosphere.albedo,
        angular_velocity: body.angular_velocity,
        axial_tilt: body.axial_tilt,
      })
    }
    BodyKind::Star(_) => None,
  }
}

/// Runtime state for one generated star/planet system.
///
/// System-level physics such as N-body gravity belongs here. World-local
/// physics still lives inside each `World` and is ticked after the system
/// layer.
pub struct System {
  id: SystemId,
  worlds: HashMap<WorldId, World>,
}

impl System {
  pub fn new(id: SystemId, worlds: HashMap<WorldId, World>) -> Self {
    Self { id, worlds }
  }

  pub fn single(id: SystemId, world: World) -> Self {
    let mut worlds = HashMap::new();
    worlds.insert(world.id(), world);
    Self::new(id, worlds)
  }

  pub fn id(&self) -> SystemId {
    self.id
  }

  pub fn insert_world(&mut self, world: World) -> Option<World> {
    self.worlds.insert(world.id(), world)
  }

  pub fn world(&self, id: WorldId) -> Option<&World> {
    self.worlds.get(&id)
  }

  pub fn world_mut(&mut self, id: WorldId) -> Option<&mut World> {
    self.worlds.get_mut(&id)
  }

  pub fn worlds(&self) -> impl Iterator<Item = &World> {
    self.worlds.values()
  }

  pub fn worlds_mut(&mut self) -> impl Iterator<Item = &mut World> {
    self.worlds.values_mut()
  }

  pub fn tick(&mut self, pool: &Pool, dt: f64) -> AetherResult<()> {
    // Future system-level nexus/resources run here before world-local physics.
    for world in self.worlds.values_mut() {
      world.tick(pool, dt)?;
    }
    Ok(())
  }
}

pub struct Aether {
  systems: HashMap<SystemId, System>,
  pool: Pool,
}

impl Aether {
  pub fn new(systems: HashMap<SystemId, System>, pool: Pool) -> Self {
    Self { systems, pool }
  }

  pub fn from_worlds(worlds: HashMap<WorldId, World>, pool: Pool) -> Self {
    let mut systems = HashMap::new();
    systems.insert(SystemId(0), System::new(SystemId(0), worlds));
    Self::new(systems, pool)
  }

  pub fn step(&mut self, dt: f64) -> AetherResult<()> {
    for system in self.systems.values_mut() {
      system.tick(&self.pool, dt)?;
    }
    Ok(())
  }

  pub fn system(&self, id: SystemId) -> Option<&System> {
    self.systems.get(&id)
  }

  pub fn system_mut(&mut self, id: SystemId) -> Option<&mut System> {
    self.systems.get_mut(&id)
  }

  pub fn systems(&self) -> impl Iterator<Item = &System> {
    self.systems.values()
  }

  pub fn systems_mut(&mut self) -> impl Iterator<Item = &mut System> {
    self.systems.values_mut()
  }

  pub fn insert_system(&mut self, system: System) -> Option<System> {
    self.systems.insert(system.id(), system)
  }

  pub fn world(&self, id: WorldId) -> Option<&World> {
    self.systems.values().find_map(|system| system.world(id))
  }

  pub fn world_in_system(
    &self,
    system_id: SystemId,
    world_id: WorldId,
  ) -> Option<&World> {
    self.system(system_id)?.world(world_id)
  }

  pub fn worlds(&self) -> impl Iterator<Item = &World> {
    self.systems.values().flat_map(System::worlds)
  }
}
