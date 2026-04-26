use pleroma::prelude::WorldAccess;
use utility::{
  domain::FieldKey,
  error::AetherResult, thread::pool::Pool,
}; 

pub trait Stage: Send + Sync {
   fn name(&self) -> &'static str;
   fn reads(&self) -> &[FieldKey];
   fn writes(&self) -> &[FieldKey];
   fn run(&self, ctx: StageContext<'_>) -> AetherResult<()>;
 }

 pub struct StageContext<'a> {
   pub world: WorldAccess<'a>,   // typed read/write into pleroma, scoped to declared keys
   pub pool: &'a Pool,            // for inner parallelism (e.g. continuum::parallel_step)
   pub dt: f64,
 }
