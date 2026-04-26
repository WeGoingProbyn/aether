use utility::{error::AetherResult, thread::pool::Pool};

use crate::stage::Stage;

pub struct Schedule {

}

impl Schedule {
  pub fn add(&mut self, stage: impl Stage + 'static) -> StageId {
    unimplemented!()
  }

  pub fn before(&mut self, a: StageId, b: StageId) {
    unimplemented!()
  }

  pub fn build(self, world: &Pleroma) -> AetherResult<CompiledSchedule> {
    unimplemented!()
  }
}

pub struct CompiledSchedule {

}

impl CompiledSchedule {
  pub fn tick(&self, world: &mut Pleroma, pool: &Pool, dt: f64)
  -> AetherResult<()> {
    unimplemented!()
  }
}
