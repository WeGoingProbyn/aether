// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! End-to-end `CompiledNexus::tick` — nexus actually runs stages,
//! propagates writer output to readers, and surfaces stage errors.

use nexus::{
  CellView, FieldKey, FieldName, FieldStorage, MeshKey, Nexus, SoaField, Stage,
  StageContext, StagePlan, WorldConstants, WorldId,
};
use pleroma::Pleroma;
use tessera::world_mesh::Tessera;
use utility::domain::CellId;
use utility::error::{AetherError, AetherResult, ErrorDomain};
use utility::thread::pool::{Pool, ScopedScheduler, ScopedTaskGraph};

const N: usize = 1;
const PRESSURE: FieldKey = FieldKey::new(MeshKey::SURFACE, FieldName::Pressure);
const TEMPERATURE: FieldKey =
  FieldKey::new(MeshKey::SURFACE, FieldName::Temperature);
const HUMIDITY: FieldKey = FieldKey::new(MeshKey::SURFACE, FieldName::Humidity);

struct Setter {
  name: &'static str,
  reads: Vec<FieldKey>,
  writes: Vec<FieldKey>,
  target: FieldKey,
  value: f64,
}

impl Stage for Setter {
  fn name(&self) -> &'static str {
    self.name
  }
  fn reads(&self) -> &[FieldKey] {
    &self.reads
  }
  fn writes(&self) -> &[FieldKey] {
    &self.writes
  }
  fn run(&mut self, mut ctx: StageContext<'_>) -> AetherResult<()> {
    let field: &mut SoaField<N> = ctx
      .world
      .fields
      .write(self.target)
      .expect("declared write should resolve");
    field.write(CellId::from(0), &[self.value]);
    Ok(())
  }
}

struct Doubler {
  source: FieldKey,
  destination: FieldKey,
}

impl Stage for Doubler {
  fn name(&self) -> &'static str {
    "doubler"
  }
  fn reads(&self) -> &[FieldKey] {
    std::slice::from_ref(&self.source)
  }
  fn writes(&self) -> &[FieldKey] {
    std::slice::from_ref(&self.destination)
  }
  fn run(&mut self, mut ctx: StageContext<'_>) -> AetherResult<()> {
    let src_value = {
      let src: &SoaField<N> = ctx.world.fields.read(self.source).unwrap();
      *src.state(CellId::from(0)).as_state()
    };
    let dst: &mut SoaField<N> =
      ctx.world.fields.write(self.destination).unwrap();
    dst.write(CellId::from(0), &[src_value[0] * 2.0]);
    Ok(())
  }
}

struct ProgramSetter {
  writes: [FieldKey; 1],
}

impl Stage for ProgramSetter {
  fn name(&self) -> &'static str {
    "program_setter"
  }

  fn reads(&self) -> &[FieldKey] {
    &[]
  }

  fn writes(&self) -> &[FieldKey] {
    &self.writes
  }

  fn run(&mut self, _ctx: StageContext<'_>) -> AetherResult<()> {
    unreachable!("program stage should execute through plan")
  }

  fn plan<'a>(
    &'a mut self,
    mut ctx: StageContext<'a>,
  ) -> AetherResult<StagePlan<'a>> {
    Ok(StagePlan::program(
      move |scheduler: &mut ScopedScheduler| {
        let mut wave_sum = 0usize;
        let mut wave = ScopedTaskGraph::new();
        wave.add(|| {
          wave_sum += 2;
          Ok(())
        });
        scheduler.run(wave)?;

        let field: &mut SoaField<N> =
          ctx.world.fields.write(self.writes[0]).unwrap();
        field.write(CellId::from(0), &[wave_sum as f64]);
        Ok(())
      },
    ))
  }
}

fn make_world() -> Pleroma {
  let mut world = Pleroma::new();
  world.register_field(PRESSURE, SoaField::<N>::zeros(1));
  world.register_field(TEMPERATURE, SoaField::<N>::zeros(1));
  world.register_field(HUMIDITY, SoaField::<N>::zeros(1));
  world
}

#[test]
fn parallel_independent_writers_both_succeed() {
  let mut world = make_world();
  let tessera = Tessera::default();
  let pool = Pool::default();

  let mut s = Nexus::new();
  s.add(Setter {
    name: "set_p",
    reads: vec![],
    writes: vec![PRESSURE],
    target: PRESSURE,
    value: 7.0,
  });
  s.add(Setter {
    name: "set_t",
    reads: vec![],
    writes: vec![TEMPERATURE],
    target: TEMPERATURE,
    value: 11.0,
  });

  let mut compiled = s.build(&world).unwrap();
  assert_eq!(compiled.layer_count(), 1);
  compiled
    .tick(
      WorldId(0),
      &tessera,
      &WorldConstants::default(),
      &mut world,
      &pool,
      0.0,
    )
    .unwrap();

  let p: &SoaField<N> = world.read(PRESSURE).unwrap();
  let t: &SoaField<N> = world.read(TEMPERATURE).unwrap();
  assert_eq!(p.state(CellId::from(0)).as_state(), &[7.0]);
  assert_eq!(t.state(CellId::from(0)).as_state(), &[11.0]);
}

#[test]
fn raw_chain_propagates_value_through_layers() {
  let mut world = make_world();
  let tessera = Tessera::default();
  let pool = Pool::default();

  let mut s = Nexus::new();
  // layer 0: set Pressure = 5
  s.add(Setter {
    name: "set_p",
    reads: vec![],
    writes: vec![PRESSURE],
    target: PRESSURE,
    value: 5.0,
  });
  // layer 1: Temperature = Pressure * 2 = 10
  s.add(Doubler {
    source: PRESSURE,
    destination: TEMPERATURE,
  });
  // layer 2: Humidity = Temperature * 2 = 20
  s.add(Doubler {
    source: TEMPERATURE,
    destination: HUMIDITY,
  });

  let mut compiled = s.build(&world).unwrap();
  assert_eq!(compiled.layer_count(), 3);
  compiled
    .tick(
      WorldId(0),
      &tessera,
      &WorldConstants::default(),
      &mut world,
      &pool,
      0.0,
    )
    .unwrap();

  let p: &SoaField<N> = world.read(PRESSURE).unwrap();
  let t: &SoaField<N> = world.read(TEMPERATURE).unwrap();
  let h: &SoaField<N> = world.read(HUMIDITY).unwrap();
  assert_eq!(p.state(CellId::from(0)).as_state(), &[5.0]);
  assert_eq!(t.state(CellId::from(0)).as_state(), &[10.0]);
  assert_eq!(h.state(CellId::from(0)).as_state(), &[20.0]);
}

#[test]
fn program_stage_runs_on_scheduler_and_can_execute_child_wave() {
  let mut world = make_world();
  let tessera = Tessera::default();
  let pool = Pool::new(2).unwrap();

  let mut s = Nexus::new();
  s.add(ProgramSetter {
    writes: [TEMPERATURE],
  });

  let mut compiled = s.build(&world).unwrap();
  compiled
    .tick(
      WorldId(0),
      &tessera,
      &WorldConstants::default(),
      &mut world,
      &pool,
      0.0,
    )
    .unwrap();

  let t: &SoaField<N> = world.read(TEMPERATURE).unwrap();
  assert_eq!(t.state(CellId::from(0)).as_state(), &[2.0]);
}

#[derive(Debug)]
enum BoomError {
  Boom,
}

impl ErrorDomain for BoomError {
  fn domain(&self) -> &str {
    "test boom"
  }
}

impl std::fmt::Display for BoomError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "boom")
  }
}

struct Boom;

impl Stage for Boom {
  fn name(&self) -> &'static str {
    "boom"
  }
  fn reads(&self) -> &[FieldKey] {
    &[]
  }
  fn writes(&self) -> &[FieldKey] {
    &[]
  }
  fn run(&mut self, _ctx: StageContext<'_>) -> AetherResult<()> {
    Err(AetherError::new(BoomError::Boom))
  }
}

#[test]
fn stage_errors_surface_through_tick() {
  let mut world = make_world();
  let tessera = Tessera::default();
  let pool = Pool::default();

  let mut s = Nexus::new();
  s.add(Boom);

  let mut compiled = s.build(&world).unwrap();
  let result = compiled.tick(
    WorldId(0),
    &tessera,
    &WorldConstants::default(),
    &mut world,
    &pool,
    0.0,
  );
  assert!(result.is_err());
}
