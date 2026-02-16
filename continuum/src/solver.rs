use utility::{profile, thread::pool::Pool};

use crate::{boundary::BoundaryRegistry, field::{CellView, FieldStorage}, geometry::{CellGeometry, CellId, FaceGeometry}, mesh::Mesh, model::{ConservationLaw, NumericalFlux}, partition::Decomposition, topology::{FaceConnection, Topology}};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum TimeIntegration {
  ForwardEuler,
  Rk2,
}

#[derive(Clone)]
pub struct SolverConfig {
  cfl: f64,
  dt_max: f64,
  integrator: TimeIntegration,
}

impl SolverConfig {
  pub fn new(cfl: f64, dt_max: f64, integrator: TimeIntegration) -> SolverConfig {
    SolverConfig {
      cfl,
      dt_max,
      integrator,
    }
  }

   pub fn dt_max(&self) -> f64 { 
    self.dt_max 
  }
}

#[derive(Clone)]
pub struct FvmSolver<const D: usize, const N: usize, L, F>
where 
  L: ConservationLaw<D, N>,
  F: NumericalFlux<D, N>,
{
  config: SolverConfig,
  time: f64,
  step: usize,
  law: L,
  flux: F,
}

impl<const D: usize, const N: usize, L, F> FvmSolver<D, N, L, F> 
where 
  L: ConservationLaw<D, N>,
  F: NumericalFlux<D, N>,
{
  pub fn new(config: SolverConfig, law: L, flux: F) -> Self {      
    FvmSolver { 
      config, 
      time: 0.0, 
      step: 0, 
      law, 
      flux 
    }
  }

  pub fn time(&self) -> f64 {
    self.time
  }

  pub fn current_step(&self) -> usize {
    self.step
  }

  pub fn law(&self) -> &L { 
    &self.law 
  }
  
  pub fn config(&self) -> &SolverConfig { 
    &self.config 
  }

  #[profile]
  pub fn compute_dt(
    &self,
    state: &impl FieldStorage<N>,
    mesh: &impl Mesh<D>,
  ) -> f64 {
    let mut dt_min = self.config.dt_max;

    for i in 0..mesh.cell_count() {
      let cell = CellId::from(i);
      let s = state.state(cell);
      let speed = self.law.max_wave_speed(s.as_state());
      if speed > 1e-14 {
        let vol = mesh.cell_volume(cell);
        let dx = vol.powf(1.0 / D as f64);
        let dt_local = self.config.cfl * dx / speed;
        dt_min = dt_min.min(dt_local);
      }
    }
    dt_min
  } 

  #[profile]
  pub fn compute_residual<S>(
    &self,
    state: &S,
    residual: &mut S,
    mesh: &impl Mesh<D>,
    bcs: &BoundaryRegistry<D, N>,
  ) 
  where
    S: FieldStorage<N>
  {
    // Zero residual
    for i in 0..mesh.cell_count() {
      residual.write(CellId::from(i), &[0.0; N]);
    }

    // Interior faces
    for &(face, owner, neighbour) in mesh.interior_faces() {
      let left = state.state(owner);
      let right = state.state(neighbour);

      let area_vec = mesh.face_area_vector(face);
      let area = mesh.face_area(face);
      let normal = &area_vec / &area;

      let flux = self.flux.compute(
        &self.law,
        left.as_state(),
        right.as_state(),
        &normal,
      );

      // Owner loses flux, neighbour gains flux
      let mut res_l = *residual.state(owner).as_state();
      let mut res_r = *residual.state(neighbour).as_state();
      for i in 0..N {
        let f_scaled = flux[i] * area * mesh.face_metrics(face).sqrt_metric;
        res_l[i] -= f_scaled;
        res_r[i] += f_scaled;
      }
      residual.write(owner, res_l.as_state());
      residual.write(neighbour, res_r.as_state());
    }

    // Boundary faces
    for tag in mesh.boundary_tags() {
      if let Some(bc) = bcs.get(tag) {
        for &(face, owner) in mesh.boundary_faces(tag) {
          let interior = state.state(owner);
          let area_vec = mesh.face_area_vector(face);
          let area = mesh.face_area(face);
          let out_sign = match mesh.face_connection(face) {
            FaceConnection::Boundary { out_sign, .. } => *out_sign,
            _ => unreachable!(),
          };
          let normal = &area_vec / &area * out_sign;

          let ghost = bc.ghost_state(interior.as_state(), &normal);
          let flux = self.flux.compute(
            &self.law,
            interior.as_state(),
            &ghost,
            &normal,
          );

          let mut res = *residual.state(owner).as_state();
          for i in 0..N {
            res[i] -= flux[i] * area * mesh.face_metrics(face).sqrt_metric;
          }
          residual.write(owner, res.as_state());
        }
      }
    }

    // Divide by volume + add source terms
    for i in 0..mesh.cell_count() {
      let cell = CellId::from(i);
      let vol = mesh.cell_volume(cell);
      let metrics = mesh.cell_metrics(cell);

      let mut res = *residual.state(cell).as_state();
      let s = self.law.source(
        state.state(cell).as_state(),
        mesh.cell_centroid(cell),
        metrics,
      );

      for j in 0..N {
        res[j] = res[j] / vol + s[j] * metrics.sqrt_metric;
      }
      residual.write(cell, res.as_state());
    }
  }

  #[profile]
  pub fn step<S: FieldStorage<N>>(
    &mut self,
    state: &mut S,
    residual: &mut S,
    mesh: &(impl CellGeometry<D> + FaceGeometry<D> + Topology),
    bcs: &BoundaryRegistry<D, N>,
  ) -> f64 {
  let dt = self.compute_dt(state, mesh);

    match self.config.integrator {
      TimeIntegration::ForwardEuler => {
        self.compute_residual(state, residual, mesh, bcs);
        // state = state + dt * residual
        state.axpy(dt, residual);
      }

      TimeIntegration::Rk2 => {
        let u_old = state.clone_state();

        // Stage 1: state = u + dt * R(u)
        self.compute_residual(state, residual, mesh, bcs);
        state.axpy(dt, residual);

        // Stage 2: state = 0.5 * u_old + 0.5 * (state + dt * R(state))
        self.compute_residual(state, residual, mesh, bcs);
        state.axpy(dt, residual);          // state = u* + dt*R(u*)

        let temp = state.clone_state();
        state.weighted_sum(0.5, &u_old, 0.5, &temp);
      }
    }

    for i in 0..mesh.cell_count() {
      let cell = CellId::from(i);
      let mut s = *state.state(cell).as_state();
      self.law.fix_state(&mut s);
      state.write(cell, &s);
    }

    self.time += dt;
    self.step += 1;
    dt
  }

  #[profile]
  pub fn parallel_step<S>(
    &self,
    pool: &Pool,
    decomp: &Decomposition<D, impl Mesh<D>>,
    states: &mut [S],
    residuals: &mut [S],
    bcs: &BoundaryRegistry<D, N>,
  ) -> f64
where
    L: ConservationLaw<D, N> + Sync,
    F: NumericalFlux<D, N> + Sync,
    S: FieldStorage<N>,
  {
    // 1. Exchange ghost cell data
    decomp.exchange_ghosts(states);

    // 2. Compute global dt (sequential — it's a min-reduction, fast)
    let dt = decomp.partitions.iter().enumerate()
      .map(|(i, p)| self.compute_dt(&states[i], p))
      .fold(self.config.dt_max, f64::min);

    // 3. Compute residuals per partition (parallel via dispatch)
    let tasks: Vec<_> = states.iter()
      .zip(residuals.iter_mut())
      .zip(decomp.partitions.iter())
      .map(|((state, residual), partition)| {
        move || {
          self.compute_residual(state, residual, partition, bcs);
        }
      }).collect();
    pool.dispatch(tasks);

    // 4. Update state: state += dt * residual (parallel via dispatch)
    let tasks: Vec<_> = states.iter_mut()
      .zip(residuals.iter())
      .map(|(state, residual)| {
        move || { state.axpy(dt, residual); }
      }).collect();
    pool.dispatch(tasks);

    // 5. Fix state per partition (parallel via dispatch)
    let tasks: Vec<_> = states.iter_mut()
      .zip(decomp.partitions.iter())
      .map(|(state, partition)| {
        move || {
          for i in 0..partition.num_owned() {
            let cell = CellId::from(i);
            let mut s = *state.state(cell).as_state();
            self.law.fix_state(&mut s);
            state.write(cell, &s);
          }
        }
      }).collect();
    pool.dispatch(tasks);

    dt
  }
}


