// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use continuum::{
  boundary::{
    BoundaryCondition, BoundaryRegistry, ReflectiveWall, Transmissive,
  },
  model::{Euler3D, RusanovFlux},
  solver::{FvmSolver, SolverConfig, TimeIntegration},
};
use nexus::{
  FieldKey, FieldStorage, LocalPartitionField, MeshKey, SoaField, Stage,
  StageContext, StagePlan, gather_partition_field, scatter_partition_owned,
};
use std::sync::Mutex;
use tessera::{
  cube_sphere::CubeSphere, mesh::Mesh, partition::Decomposition,
  world_mesh::DecompositionKey,
};
use utility::{
  debug,
  domain::{BoundaryTag, CellId},
  end_profile,
  error::{AetherError, AetherResult, Unpoison},
  inline_profile, profile,
  thread::pool::{ScopedReduction, ScopedScheduler},
};

use crate::{
  background::BackgroundCorrectedEuler3D, error::AerError, init::AtmosphereSpec,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GravityMode {
  None,
  Radial,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackgroundCorrectionMode {
  None,
  /// Capture the current Euler field when the solver is first initialized and
  /// subtract its discrete residual as a fixed source correction.
  CurrentState,
}

/// Compressible atmosphere dynamics stage.
///
/// The prognostic state is a Pleroma `SoaField<5>` using continuum's Euler3D
/// ordering: `[rho, rho_u, rho_v, rho_w, energy]`. Aer owns the solver and
/// residual scratch; Tessera owns the mesh; constants arrive through
/// `WorldView`.
pub struct EulerAtmosphereStep {
  mesh: MeshKey,
  state: FieldKey,
  reads: [FieldKey; 1],
  writes: [FieldKey; 1],
  config: SolverConfig,
  gravity: GravityMode,
  background_correction: BackgroundCorrectionMode,
  boundaries: BoundaryRegistry<3, 5>,
  solver: Option<FvmSolver<3, 5, BackgroundCorrectedEuler3D, RusanovFlux>>,
  partition_source_correction: Option<Vec<Vec<[f64; 5]>>>,
  residual: SoaField<5>,
  last_dt: Option<f64>,
  last_substeps: usize,
  max_substeps: usize,
}

impl EulerAtmosphereStep {
  pub fn new(
    mesh: MeshKey,
    state: FieldKey,
    config: SolverConfig,
  ) -> AetherResult<Self> {
    if state.mesh() != mesh {
      return Err(
        AetherError::new(AerError::FieldMeshMismatch)
          .context(format!("mesh {:?}, state {:?}", mesh, state)),
      );
    }

    Ok(Self {
      mesh,
      state,
      reads: [state],
      writes: [state],
      config,
      gravity: GravityMode::Radial,
      background_correction: BackgroundCorrectionMode::None,
      boundaries: default_atmosphere_boundaries(),
      solver: None,
      partition_source_correction: None,
      residual: SoaField::<5>::zeros(0),
      last_dt: None,
      last_substeps: 0,
      max_substeps: 10_000,
    })
  }

  pub fn forward_euler(
    mesh: MeshKey,
    state: FieldKey,
    cfl: f64,
  ) -> AetherResult<Self> {
    Self::new(
      mesh,
      state,
      SolverConfig::new(cfl, 1.0, TimeIntegration::ForwardEuler),
    )
  }

  pub fn with_gravity_mode(mut self, gravity: GravityMode) -> Self {
    self.gravity = gravity;
    self.solver = None;
    self.partition_source_correction = None;
    self
  }

  pub fn with_background_correction(
    mut self,
    mode: BackgroundCorrectionMode,
  ) -> Self {
    self.background_correction = mode;
    self.solver = None;
    self.partition_source_correction = None;
    self
  }

  pub fn with_current_state_background_correction(self) -> Self {
    self.with_background_correction(BackgroundCorrectionMode::CurrentState)
  }

  pub fn with_boundary(
    mut self,
    tag: BoundaryTag,
    condition: impl BoundaryCondition<3, 5> + 'static,
  ) -> Self {
    self.boundaries.register(tag, condition);
    self.solver = None;
    self.partition_source_correction = None;
    self
  }

  pub fn with_max_substeps(mut self, max_substeps: usize) -> Self {
    self.max_substeps = max_substeps.max(1);
    self
  }

  pub fn mesh(&self) -> MeshKey {
    self.mesh
  }

  pub fn state(&self) -> FieldKey {
    self.state
  }

  pub fn gravity_mode(&self) -> GravityMode {
    self.gravity
  }

  pub fn background_correction(&self) -> BackgroundCorrectionMode {
    self.background_correction
  }

  pub fn last_dt(&self) -> Option<f64> {
    self.last_dt
  }

  pub fn last_substeps(&self) -> usize {
    self.last_substeps
  }

  fn ensure_solver(
    &mut self,
    constants: &nexus::WorldConstants,
    mesh: &dyn Mesh<3>,
    background_state: Option<&SoaField<5>>,
  ) -> AetherResult<()> {
    if self.solver.is_some() {
      return Ok(());
    }

    let spec = AtmosphereSpec::from_world_constants(constants)?;
    let source_correction = match self.background_correction {
      BackgroundCorrectionMode::None => Vec::new(),
      BackgroundCorrectionMode::CurrentState => {
        let background_state = background_state.ok_or_else(|| {
          AetherError::new(AerError::MissingReadField)
            .context("background correction needs the current Euler state")
        })?;
        background_residual_correction(
          &spec,
          self.gravity,
          mesh,
          &self.boundaries,
          background_state,
        )?
      }
    };

    let law = BackgroundCorrectedEuler3D::new(
      euler_law(&spec, self.gravity, mesh),
      source_correction,
    );
    self.solver = Some(FvmSolver::new(self.config.clone(), law, RusanovFlux));
    Ok(())
  }

  fn ensure_partition_source_correction(
    &mut self,
    constants: &nexus::WorldConstants,
    mesh: &dyn Mesh<3>,
    decomposition: &Decomposition<3, CubeSphere>,
    background_state: Option<&SoaField<5>>,
  ) -> AetherResult<()> {
    match self.background_correction {
      BackgroundCorrectionMode::None => Ok(()),
      BackgroundCorrectionMode::CurrentState => {
        if self.partition_source_correction.is_some() {
          return Ok(());
        }

        let background_state = background_state.ok_or_else(|| {
          AetherError::new(AerError::MissingReadField)
            .context("background correction needs the current Euler state")
        })?;
        if background_state.len() != mesh.cell_count() {
          return Err(AetherError::new(AerError::FieldLengthMismatch).context(
            format!(
              "background len {}, mesh cell count {}",
              background_state.len(),
              mesh.cell_count()
            ),
          ));
        }

        let spec = AtmosphereSpec::from_world_constants(constants)?;
        let mut partition_corrections =
          Vec::with_capacity(decomposition.partitions.len());
        for partition in &decomposition.partitions {
          let local_background =
            gather_partition_field(background_state, partition);
          let mut local_residual = LocalPartitionField::<5>::zeros(
            partition.num_owned(),
            partition.local_cell_count() - partition.num_owned(),
          );
          partition_corrections.push(
            background_residual_correction_with_residual(
              &spec,
              self.gravity,
              partition,
              &self.boundaries,
              &local_background,
              &mut local_residual,
            )?,
          );
        }

        self.partition_source_correction = Some(partition_corrections);
        Ok(())
      }
    }
  }
}

impl Stage for EulerAtmosphereStep {
  fn name(&self) -> &'static str {
    "aer_euler_atmosphere_step"
  }

  fn reads(&self) -> &[FieldKey] {
    &self.reads
  }

  fn writes(&self) -> &[FieldKey] {
    &self.writes
  }

  fn plan<'a>(
    &'a mut self,
    ctx: StageContext<'a>,
  ) -> AetherResult<StagePlan<'a>> {
    if ctx.world.partition_count == 6
      && ctx
        .world
        .tessera
        .decomposition::<CubeSphere>(self.mesh, DecompositionKey::DEFAULT)
        .is_some()
    {
      Ok(StagePlan::program(EulerPartitionProgram {
        stage: self,
        ctx,
      }))
    } else {
      let name = self.name();
      Ok(StagePlan::single(name, move || self.run(ctx)))
    }
  }

  #[profile("aer.EulerAtmosphereStep.run")]
  fn run(&mut self, mut ctx: StageContext<'_>) -> AetherResult<()> {
    let tessera = ctx.world.tessera;
    let constants = ctx.world.constants;
    let dt = ctx.world.dt;
    if !dt.is_finite() || dt <= 0.0 {
      return Err(
        AetherError::new(AerError::InvalidTimeStep)
          .context(format!("requested dt {}", dt)),
      );
    }

    let mesh = tessera.mesh(self.mesh).ok_or_else(|| {
      AetherError::new(AerError::MissingMesh)
        .context(format!("{:?}", self.mesh))
    })?;
    let cell_count = mesh.cell_count();

    let background_state = if self.solver.is_none()
      && self.background_correction == BackgroundCorrectionMode::CurrentState
    {
      let state: &SoaField<5> =
        ctx.world.fields.read(self.state).ok_or_else(|| {
          AetherError::new(AerError::MissingReadField)
            .context(format!("{:?}", self.state))
        })?;
      if state.len() != cell_count {
        return Err(AetherError::new(AerError::FieldLengthMismatch).context(
          format!("state len {}, mesh cell count {}", state.len(), cell_count),
        ));
      }
      Some(state.clone_state())
    } else {
      None
    };

    self.ensure_solver(constants, mesh.as_ref(), background_state.as_ref())?;
    if self.residual.len() != cell_count {
      self.residual = SoaField::<5>::zeros(cell_count);
    }

    let state: &mut SoaField<5> =
      ctx.world.fields.write(self.state).ok_or_else(|| {
        AetherError::new(AerError::MissingWriteField)
          .context(format!("{:?}", self.state))
      })?;

    if state.len() != cell_count {
      return Err(AetherError::new(AerError::FieldLengthMismatch).context(
        format!("state len {}, mesh cell count {}", state.len(), cell_count),
      ));
    }

    let mut remaining = dt;
    let mut total_advanced = 0.0;
    let mut substeps = 0usize;
    let tolerance = (dt.abs() * 1.0e-12).max(1.0e-15);

    inline_profile!("aer.EulerAtmosphereStep.inner_loop");
    while remaining > tolerance {
      if substeps >= self.max_substeps {
        return Err(AetherError::new(AerError::InvalidTimeStep).context(
          format!(
            "hit max_substeps {} with {} remaining from requested dt {}",
            self.max_substeps, remaining, dt
          ),
        ));
      }

      let solver = self
        .solver
        .as_mut()
        .expect("solver is initialized before stepping");
      solver.config_mut().set_dt_max(remaining);
      let advanced =
        solver.step(state, &mut self.residual, mesh.as_ref(), &self.boundaries);
      if !advanced.is_finite() || advanced <= 0.0 {
        return Err(
          AetherError::new(AerError::InvalidTimeStep)
            .context(format!("advanced dt {}", advanced)),
        );
      }
      if advanced > remaining + tolerance {
        return Err(AetherError::new(AerError::InvalidTimeStep).context(
          format!("advanced dt {} exceeds remaining {}", advanced, remaining),
        ));
      }

      remaining -= advanced.min(remaining);
      total_advanced += advanced;
      substeps += 1;
    }
    end_profile!("aer.EulerAtmosphereStep.inner_loop");
    debug!("atmosphere substeps: {}", substeps);

    self.last_dt = Some(total_advanced);
    self.last_substeps = substeps;

    Ok(())
  }
}

struct EulerPartitionProgram<'a> {
  stage: &'a mut EulerAtmosphereStep,
  ctx: StageContext<'a>,
}

impl<'a> nexus::StageProgram<'a> for EulerPartitionProgram<'a> {
  fn execute(
    self: Box<Self>,
    scheduler: &mut ScopedScheduler,
  ) -> AetherResult<()> {
    let EulerPartitionProgram { stage, mut ctx } = *self;
    let tessera = ctx.world.tessera;
    let constants = ctx.world.constants;
    let dt = ctx.world.dt;
    if !dt.is_finite() || dt <= 0.0 {
      return Err(
        AetherError::new(AerError::InvalidTimeStep)
          .context(format!("requested dt {}", dt)),
      );
    }

    let mesh = tessera.mesh(stage.mesh).ok_or_else(|| {
      AetherError::new(AerError::MissingMesh)
        .context(format!("{:?}", stage.mesh))
    })?;
    let cell_count = mesh.cell_count();
    let decomposition = tessera
      .decomposition::<CubeSphere>(stage.mesh, DecompositionKey::DEFAULT)
      .ok_or_else(|| {
        AetherError::new(AerError::MissingMesh).context(format!(
          "missing default cube-sphere decomposition for {:?}",
          stage.mesh
        ))
      })?;

    let spec = AtmosphereSpec::from_world_constants(constants)?;
    let gravity = stage.gravity;
    let config = stage.config.clone();
    let max_substeps = stage.max_substeps;
    let partition_count = decomposition.partitions.len();
    let background_state = if stage.partition_source_correction.is_none()
      && stage.background_correction == BackgroundCorrectionMode::CurrentState
    {
      let state: &SoaField<5> =
        ctx.world.fields.read(stage.state).ok_or_else(|| {
          AetherError::new(AerError::MissingReadField)
            .context(format!("{:?}", stage.state))
        })?;
      if state.len() != cell_count {
        return Err(AetherError::new(AerError::FieldLengthMismatch).context(
          format!("state len {}, mesh cell count {}", state.len(), cell_count),
        ));
      }
      Some(state.clone_state())
    } else {
      None
    };
    stage.ensure_partition_source_correction(
      constants,
      mesh.as_ref(),
      decomposition,
      background_state.as_ref(),
    )?;
    let boundaries = &stage.boundaries;
    let partition_source_correction =
      stage.partition_source_correction.as_ref();
    let mut solvers = Vec::with_capacity(partition_count);
    let mut states = Vec::with_capacity(partition_count);
    let mut residuals = Vec::with_capacity(partition_count);

    for (index, partition) in decomposition.partitions.iter().enumerate() {
      let source_correction = partition_source_correction
        .map(|corrections| corrections[index].clone())
        .unwrap_or_default();
      let law = BackgroundCorrectedEuler3D::new(
        euler_law(&spec, gravity, partition),
        source_correction,
      );
      solvers.push(Mutex::new(FvmSolver::new(
        config.clone(),
        law,
        RusanovFlux,
      )));
      states.push(Mutex::new(LocalPartitionField::<5>::zeros(
        partition.num_owned(),
        partition.local_cell_count() - partition.num_owned(),
      )));
      residuals.push(Mutex::new(LocalPartitionField::<5>::zeros(
        partition.num_owned(),
        partition.local_cell_count() - partition.num_owned(),
      )));
    }

    let state: &mut SoaField<5> =
      ctx.world.fields.write(stage.state).ok_or_else(|| {
        AetherError::new(AerError::MissingWriteField)
          .context(format!("{:?}", stage.state))
      })?;

    if state.len() != cell_count {
      return Err(AetherError::new(AerError::FieldLengthMismatch).context(
        format!("state len {}, mesh cell count {}", state.len(), cell_count),
      ));
    }

    let mut remaining = dt;
    let mut total_advanced = 0.0;
    let mut substeps = 0usize;
    let tolerance = (dt.abs() * 1.0e-12).max(1.0e-15);
    let reduction = ScopedReduction::new(partition_count);

    inline_profile!("aer.EulerAtmosphereStep.partition_loop");
    while remaining > tolerance {
      if substeps >= max_substeps {
        return Err(AetherError::new(AerError::InvalidTimeStep).context(
          format!(
            "hit max_substeps {} with {} remaining from requested dt {}",
            max_substeps, remaining, dt
          ),
        ));
      }

      for (index, partition) in decomposition.partitions.iter().enumerate() {
        *states[index].lock().unpoison() =
          gather_partition_field(state, partition);
      }

      reduction.clear();
      scheduler.map(partition_count, |index| {
        let mut solver = solvers[index].lock().unpoison();
        solver.config_mut().set_dt_max(remaining);
        let local = states[index].lock().unpoison();
        let dt_local =
          solver.compute_dt(&*local, &decomposition.partitions[index]);
        reduction.write_input(index, dt_local.min(remaining))
      })?;

      let advanced = scheduler.reduce_min(&reduction)?;
      if !advanced.is_finite() || advanced <= 0.0 {
        return Err(
          AetherError::new(AerError::InvalidTimeStep)
            .context(format!("advanced dt {}", advanced)),
        );
      }
      if advanced > remaining + tolerance {
        return Err(AetherError::new(AerError::InvalidTimeStep).context(
          format!("advanced dt {} exceeds remaining {}", advanced, remaining),
        ));
      }

      scheduler.map(partition_count, |index| {
        let mut solver = solvers[index].lock().unpoison();
        let mut local = states[index].lock().unpoison();
        let mut residual = residuals[index].lock().unpoison();
        solver.config_mut().set_dt_max(advanced);
        solver.step_with_dt(
          advanced,
          &mut *local,
          &mut *residual,
          &decomposition.partitions[index],
          boundaries,
        );
        Ok(())
      })?;

      for (index, partition) in decomposition.partitions.iter().enumerate() {
        let local = states[index].lock().unpoison();
        scatter_partition_owned(&*local, state, partition);
      }

      remaining -= advanced.min(remaining);
      total_advanced += advanced;
      substeps += 1;
    }
    end_profile!("aer.EulerAtmosphereStep.partition_loop");
    debug!("atmosphere substeps: {}", substeps);

    stage.last_dt = Some(total_advanced);
    stage.last_substeps = substeps;

    Ok(())
  }
}

fn default_atmosphere_boundaries() -> BoundaryRegistry<3, 5> {
  let mut boundaries = BoundaryRegistry::<3, 5>::new();
  boundaries.register(BoundaryTag::Ground, ReflectiveWall);
  boundaries.register(BoundaryTag::AtmosphereEdge, Transmissive);
  boundaries
}

fn euler_law<M>(
  spec: &AtmosphereSpec,
  gravity: GravityMode,
  mesh: &M,
) -> Euler3D
where
  M: Mesh<3> + ?Sized,
{
  match gravity {
    GravityMode::None => spec.euler3d(),
    GravityMode::Radial => spec.euler3d_with_radial_gravity(
      radial_gravity_field(mesh, spec.surface_gravity()),
    ),
  }
}

fn background_residual_correction(
  spec: &AtmosphereSpec,
  gravity: GravityMode,
  mesh: &dyn Mesh<3>,
  boundaries: &BoundaryRegistry<3, 5>,
  background_state: &SoaField<5>,
) -> AetherResult<Vec<[f64; 5]>> {
  let cell_count = mesh.cell_count();
  if background_state.len() != cell_count {
    return Err(AetherError::new(AerError::FieldLengthMismatch).context(
      format!(
        "background len {}, mesh cell count {}",
        background_state.len(),
        cell_count
      ),
    ));
  }

  let mut residual = SoaField::<5>::zeros(cell_count);
  background_residual_correction_with_residual(
    spec,
    gravity,
    mesh,
    boundaries,
    background_state,
    &mut residual,
  )
}

fn background_residual_correction_with_residual<S, M>(
  spec: &AtmosphereSpec,
  gravity: GravityMode,
  mesh: &M,
  boundaries: &BoundaryRegistry<3, 5>,
  background_state: &S,
  residual: &mut S,
) -> AetherResult<Vec<[f64; 5]>>
where
  S: FieldStorage<5>,
  M: Mesh<3> + ?Sized,
{
  let cell_count = mesh.cell_count();
  if background_state.len() != cell_count {
    return Err(AetherError::new(AerError::FieldLengthMismatch).context(
      format!(
        "background len {}, mesh cell count {}",
        background_state.len(),
        cell_count
      ),
    ));
  }
  if residual.len() != cell_count {
    return Err(AetherError::new(AerError::FieldLengthMismatch).context(
      format!(
        "residual len {}, mesh cell count {}",
        residual.len(),
        cell_count
      ),
    ));
  }

  let solver = FvmSolver::new(
    SolverConfig::new(1.0, 1.0, TimeIntegration::ForwardEuler),
    euler_law(spec, gravity, mesh),
    RusanovFlux,
  );
  solver.compute_residual(background_state, residual, mesh, boundaries);

  Ok(
    (0..cell_count)
      .map(|i| {
        let mut residual_state = [0.0; 5];
        residual.state_into(CellId::from(i), &mut residual_state);
        [
          -residual_state[0],
          -residual_state[1],
          -residual_state[2],
          -residual_state[3],
          -residual_state[4],
        ]
      })
      .collect(),
  )
}

fn radial_gravity_field<M>(mesh: &M, surface_gravity: f64) -> Vec<[f64; 3]>
where
  M: Mesh<3> + ?Sized,
{
  (0..mesh.cell_count())
    .map(|i| {
      let p = mesh.cell_world_centroid(CellId::from(i));
      let r = (p[0].powi(2) + p[1].powi(2) + p[2].powi(2)).sqrt();
      if r <= f64::EPSILON {
        return [0.0; 3];
      }
      [
        -surface_gravity * p[0] / r,
        -surface_gravity * p[1] / r,
        -surface_gravity * p[2] / r,
      ]
    })
    .collect()
}

#[cfg(test)]
mod tests {
  use std::sync::Arc;

  use continuum::solver::TimeIntegration;
  use nexus::{
    AtmosphereConstants, CellView, FieldName, Nexus, Pleroma, WorldConstants,
    WorldId,
  };
  use tessera::{
    cube_sphere::{CubeSphere, CubeSphereShellSpec},
    geometry::CellGeometry,
    partition::decompose_cube_sphere_panels,
    world_mesh::DecompositionKey,
    world_mesh::Tessera,
  };
  use utility::thread::pool::Pool;

  use super::*;

  const EULER_STATE: FieldKey =
    FieldKey::new(MeshKey::ATMOSPHERE, FieldName::EulerState);

  fn earth_like_constants() -> WorldConstants {
    WorldConstants {
      mass: 5.97e24,
      radius: 1.0,
      surface_gravity: 1.0,
      atmosphere: Some(AtmosphereConstants {
        reference_temperature: 1.0,
        reference_pressure: 1.0,
        gamma: 1.4,
        gas_constant: 1.0,
        molar_mass: 1.0,
        albedo: Some(0.3),
        angular_velocity: 0.0,
        axial_tilt: 0.0,
      }),
      radiation: None,
    }
  }

  #[test]
  fn euler_atmosphere_step_advances_registered_state() {
    let constants = earth_like_constants();
    let spec = AtmosphereSpec::from_world_constants(&constants).unwrap();
    let mesh = Arc::new(CubeSphere::shell(CubeSphereShellSpec::uniform(
      [2, 2, 2],
      1.0,
      1.2,
    )));
    let cell_count = mesh.cell_count();

    let mut tessera = Tessera::new();
    tessera.register_mesh(MeshKey::ATMOSPHERE, mesh);

    let mut pleroma = Pleroma::new();
    pleroma.register_field(EULER_STATE, spec.euler_state_field(cell_count));

    let mut nexus = Nexus::new();
    nexus.add(
      EulerAtmosphereStep::new(
        MeshKey::ATMOSPHERE,
        EULER_STATE,
        SolverConfig::new(0.1, 1.0, TimeIntegration::ForwardEuler),
      )
      .unwrap()
      .with_gravity_mode(GravityMode::None),
    );
    let mut compiled = nexus.build(&pleroma).unwrap();
    compiled
      .tick(
        WorldId(0),
        &tessera,
        &constants,
        &mut pleroma,
        &Pool::default(),
        0.01,
      )
      .unwrap();

    let state: &SoaField<5> = pleroma.read(EULER_STATE).unwrap();
    for i in 0..cell_count {
      let cell_state = state.state(CellId::from(i));
      assert!(cell_state[0] > 0.0);
      assert!(cell_state[4] > 0.0);
    }
  }

  #[test]
  fn euler_atmosphere_step_subcycles_to_consume_tick() {
    let constants = earth_like_constants();
    let spec = AtmosphereSpec::from_world_constants(&constants).unwrap();
    let mesh = Arc::new(CubeSphere::shell(CubeSphereShellSpec::uniform(
      [2, 2, 2],
      1.0,
      1.2,
    )));
    let cell_count = mesh.cell_count();

    let mut tessera = Tessera::new();
    tessera.register_mesh(MeshKey::ATMOSPHERE, mesh);

    let mut pleroma = Pleroma::new();
    pleroma.register_field(EULER_STATE, spec.euler_state_field(cell_count));

    let mut nexus = Nexus::new();
    nexus.add(
      EulerAtmosphereStep::new(
        MeshKey::ATMOSPHERE,
        EULER_STATE,
        SolverConfig::new(0.0001, 1.0, TimeIntegration::ForwardEuler),
      )
      .unwrap()
      .with_gravity_mode(GravityMode::None),
    );
    let mut compiled = nexus.build(&pleroma).unwrap();
    compiled
      .tick(
        WorldId(0),
        &tessera,
        &constants,
        &mut pleroma,
        &Pool::default(),
        0.01,
      )
      .unwrap();

    let state: &SoaField<5> = pleroma.read(EULER_STATE).unwrap();
    for i in 0..cell_count {
      let cell_state = state.state(CellId::from(i));
      assert!(cell_state[0] > 0.0);
      assert!(cell_state[4] > 0.0);
    }
  }

  #[test]
  fn euler_atmosphere_step_runs_partitioned_panel_program() {
    let constants = earth_like_constants();
    let spec = AtmosphereSpec::from_world_constants(&constants).unwrap();
    let mesh = Arc::new(CubeSphere::shell(CubeSphereShellSpec::uniform(
      [2, 2, 2],
      1.0,
      1.2,
    )));
    let cell_count = mesh.cell_count();

    let mut tessera = Tessera::new();
    tessera.register_mesh(MeshKey::ATMOSPHERE, mesh.clone());
    tessera.register_decomposition(
      MeshKey::ATMOSPHERE,
      DecompositionKey::DEFAULT,
      decompose_cube_sphere_panels(mesh),
    );

    let mut pleroma = Pleroma::new();
    pleroma.register_field(EULER_STATE, spec.euler_state_field(cell_count));

    let mut nexus = Nexus::new();
    nexus.add(
      EulerAtmosphereStep::new(
        MeshKey::ATMOSPHERE,
        EULER_STATE,
        SolverConfig::new(0.1, 1.0, TimeIntegration::ForwardEuler),
      )
      .unwrap()
      .with_gravity_mode(GravityMode::None),
    );
    let mut compiled = nexus.build(&pleroma).unwrap();
    compiled
      .tick_with_partition_count(
        WorldId(0),
        &tessera,
        &constants,
        &mut pleroma,
        &Pool::new(2).unwrap(),
        0.01,
        6,
      )
      .unwrap();

    let state: &SoaField<5> = pleroma.read(EULER_STATE).unwrap();
    for i in 0..cell_count {
      let cell_state = state.state(CellId::from(i));
      assert!(cell_state[0] > 0.0);
      assert!(cell_state[4] > 0.0);
    }
  }

  #[test]
  fn partitioned_euler_matches_serial_for_one_small_step() {
    let constants = earth_like_constants();
    let spec = AtmosphereSpec::from_world_constants(&constants).unwrap();
    let mesh = Arc::new(CubeSphere::shell(CubeSphereShellSpec::uniform(
      [2, 2, 2],
      1.0,
      1.2,
    )));
    let cell_count = mesh.cell_count();
    let mut initial = spec.euler_state_field(cell_count);
    let mut perturbed = *initial.state(CellId::from(0)).as_state();
    perturbed[0] *= 1.01;
    perturbed[4] *= 1.01;
    initial.write(CellId::from(0), &perturbed);

    let serial = run_euler_test_step(
      Arc::clone(&mesh),
      None,
      initial.clone_state(),
      &constants,
      1,
    );
    let partitioned = run_euler_test_step(
      Arc::clone(&mesh),
      Some(decompose_cube_sphere_panels(mesh)),
      initial,
      &constants,
      6,
    );

    for i in 0..cell_count {
      let cell = CellId::from(i);
      let a = serial.state(cell);
      let b = partitioned.state(cell);
      for component in 0..5 {
        let scale = a[component].abs().max(1.0);
        let rel = (a[component] - b[component]).abs() / scale;
        assert!(
          rel < 1.0e-10,
          "cell {} component {} serial {} partitioned {} rel {}",
          i,
          component,
          a[component],
          b[component],
          rel
        );
      }
    }
  }

  fn run_euler_test_step(
    mesh: Arc<CubeSphere>,
    decomposition: Option<tessera::partition::Decomposition<3, CubeSphere>>,
    initial: SoaField<5>,
    constants: &WorldConstants,
    partition_count: usize,
  ) -> SoaField<5> {
    let mut tessera = Tessera::new();
    tessera.register_mesh(MeshKey::ATMOSPHERE, mesh);
    if let Some(decomposition) = decomposition {
      tessera.register_decomposition(
        MeshKey::ATMOSPHERE,
        DecompositionKey::DEFAULT,
        decomposition,
      );
    }

    let mut pleroma = Pleroma::new();
    pleroma.register_field(EULER_STATE, initial);

    let mut nexus = Nexus::new();
    nexus.add(
      EulerAtmosphereStep::new(
        MeshKey::ATMOSPHERE,
        EULER_STATE,
        SolverConfig::new(0.1, 1.0, TimeIntegration::ForwardEuler),
      )
      .unwrap()
      .with_gravity_mode(GravityMode::None),
    );
    let mut compiled = nexus.build(&pleroma).unwrap();
    compiled
      .tick_with_partition_count(
        WorldId(0),
        &tessera,
        constants,
        &mut pleroma,
        &Pool::new(2).unwrap(),
        0.01,
        partition_count,
      )
      .unwrap();

    pleroma
      .read::<SoaField<5>>(EULER_STATE)
      .unwrap()
      .clone_state()
  }

  #[test]
  fn current_state_background_correction_cancels_initial_residual() {
    let constants = earth_like_constants();
    let spec = AtmosphereSpec::from_world_constants(&constants).unwrap();
    let mesh = Arc::new(CubeSphere::shell(CubeSphereShellSpec::uniform(
      [2, 2, 2],
      1.0,
      1.2,
    )));
    let cell_count = mesh.cell_count();

    let mut tessera = Tessera::new();
    tessera.register_mesh(MeshKey::ATMOSPHERE, mesh.clone());

    let initial_state = spec
      .isothermal_hydrostatic_state_field(mesh.as_ref(), constants.radius)
      .unwrap();
    let mut pleroma = Pleroma::new();
    pleroma.register_field(EULER_STATE, initial_state.clone_state());

    let mut nexus = Nexus::new();
    nexus.add(
      EulerAtmosphereStep::new(
        MeshKey::ATMOSPHERE,
        EULER_STATE,
        SolverConfig::new(0.25, 1.0, TimeIntegration::ForwardEuler),
      )
      .unwrap()
      .with_current_state_background_correction(),
    );
    let mut compiled = nexus.build(&pleroma).unwrap();
    compiled
      .tick(
        WorldId(0),
        &tessera,
        &constants,
        &mut pleroma,
        &Pool::default(),
        0.01,
      )
      .unwrap();

    let state: &SoaField<5> = pleroma.read(EULER_STATE).unwrap();
    for i in 0..cell_count {
      let cell = CellId::from(i);
      let before = initial_state.state(cell);
      let after = state.state(cell);
      for component in 0..5 {
        assert!(
          (after[component] - before[component]).abs() < 1.0e-10,
          "cell {} component {} drifted from {} to {}",
          i,
          component,
          before[component],
          after[component]
        );
      }
    }
  }

  #[test]
  fn partitioned_current_state_background_correction_cancels_initial_residual()
  {
    let constants = earth_like_constants();
    let spec = AtmosphereSpec::from_world_constants(&constants).unwrap();
    let mesh = Arc::new(CubeSphere::shell(CubeSphereShellSpec::uniform(
      [2, 2, 2],
      1.0,
      1.2,
    )));
    let cell_count = mesh.cell_count();

    let mut tessera = Tessera::new();
    tessera.register_mesh(MeshKey::ATMOSPHERE, mesh.clone());
    tessera.register_decomposition(
      MeshKey::ATMOSPHERE,
      DecompositionKey::DEFAULT,
      decompose_cube_sphere_panels(mesh.clone()),
    );

    let initial_state = spec
      .isothermal_hydrostatic_state_field(mesh.as_ref(), constants.radius)
      .unwrap();
    let mut pleroma = Pleroma::new();
    pleroma.register_field(EULER_STATE, initial_state.clone_state());

    let mut nexus = Nexus::new();
    nexus.add(
      EulerAtmosphereStep::new(
        MeshKey::ATMOSPHERE,
        EULER_STATE,
        SolverConfig::new(0.25, 1.0, TimeIntegration::ForwardEuler),
      )
      .unwrap()
      .with_current_state_background_correction(),
    );
    let mut compiled = nexus.build(&pleroma).unwrap();
    compiled
      .tick_with_partition_count(
        WorldId(0),
        &tessera,
        &constants,
        &mut pleroma,
        &Pool::new(2).unwrap(),
        0.01,
        6,
      )
      .unwrap();

    let state: &SoaField<5> = pleroma.read(EULER_STATE).unwrap();
    for i in 0..cell_count {
      let cell = CellId::from(i);
      let before = initial_state.state(cell);
      let after = state.state(cell);
      for component in 0..5 {
        assert!(
          (after[component] - before[component]).abs() < 1.0e-10,
          "cell {} component {} drifted from {} to {}",
          i,
          component,
          before[component],
          after[component]
        );
      }
    }
  }
}
