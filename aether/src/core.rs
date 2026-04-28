use std::collections::HashMap;

use cosmo::kind::CelestialBody;
use nexus::{CompiledNexus, WorldId};
use pleroma::Pleroma;
use tessera::world_mesh::Tessera;
use utility::{error::AetherResult, thread::pool::Pool};

/// Runtime state for one simulated body.
pub struct World {
  id: WorldId,
  seed: CelestialBody,
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
    Self {
      id,
      seed,
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
    self
      .nexus
      .tick(self.id, &self.tessera, &mut self.pleroma, pool, dt)
  }
}

pub struct Aether {
  worlds: HashMap<WorldId, World>,
  pool: Pool,
}

impl Aether {
  pub fn new(worlds: HashMap<WorldId, World>, pool: Pool) -> Self {
    Self { worlds, pool }
  }

  pub fn step(&mut self, dt: f64) -> AetherResult<()> {
    for world in self.worlds.values_mut() {
      world.tick(&self.pool, dt)?;
    }
    Ok(())
  }
}
