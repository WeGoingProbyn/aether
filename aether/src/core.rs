use std::collections::HashMap;

use cosmo::kind::CelestialBody;
use nexus::CompiledNexus;
use pleroma::Pleroma;
use tessera::world_mesh::Tessera;
use utility::{error::AetherResult, thread::pool::Pool};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct WorldId(pub usize);

/// Runtime state for one simulated body.
pub struct World {
  seed: CelestialBody,
  tessera: Tessera,
  pleroma: Pleroma,
}

impl World {
  pub fn new(seed: CelestialBody, tessera: Tessera, pleroma: Pleroma) -> Self {
    Self {
      seed,
      tessera,
      pleroma,
    }
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
}

pub struct Aether {
  worlds: HashMap<WorldId, World>,
  nexus: CompiledNexus,
  pool: Pool,
}

impl Aether {
  pub fn new(
    worlds: HashMap<WorldId, World>,
    nexus: CompiledNexus,
    pool: Pool,
  ) -> Self {
    Self {
      worlds,
      nexus,
      pool,
    }
  }

  pub fn step(&mut self, dt: f64) -> AetherResult<()> {
    for world in self.worlds.values_mut() {
      let (tessera, pleroma) = world.runtime_parts_mut();
      self.nexus.tick(tessera, pleroma, &self.pool, dt)?;
    }
    Ok(())
  }
}
