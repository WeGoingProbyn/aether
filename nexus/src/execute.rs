use pleroma::prelude::{FieldStorage, exchange_ghosts};
use tessera::{geometry::CellGeometry, mesh::Mesh, partition::Decomposition};
use utility::{domain::CellId, thread::pool::Pool};

#[profile]
pub fn parallel_step<M, S>(
  &mut self,
  pool: &Pool,
  decomp: &Decomposition<D, M>,
  states: &mut [S],
  residuals: &mut [S],
  bcs: &BoundaryRegistry<D, N>,
) -> f64
  where
  M: Mesh<D>,
  L: ConservationLaw<D, N> + Sync,
  F: NumericalFlux<D, N> + Sync,
  S: FieldStorage<N>,
{
  exchange_ghosts(decomp, states);

  let dt = {
    self.ensure_scratch_slots(decomp.partitions.len());
    let (law, flux, config, scratches) = (
      &self.law,
      &self.flux,
      &self.config,
      &mut self.scratches[..decomp.partitions.len()],
    );

    Self::refresh_parallel_state_caches(decomp, states, scratches);

    let dt = decomp
      .partitions
      .iter()
      .enumerate()
      .map(|(i, partition)| {
        Self::compute_dt_from_cache(
          config,
          law,
          &scratches[i].state_cache,
          partition,
        )
      })
      .fold(config.dt_max, f64::min);

    match config.integrator {
      TimeIntegration::ForwardEuler => {
        Self::parallel_compute_residuals_from_cache(
          pool, law, flux, decomp, scratches, residuals, bcs,
        );
        Self::parallel_axpy(pool, states, residuals, dt);
      }

      TimeIntegration::Rk2 => {
        let u_old: Vec<S> =
        states.iter().map(|state| state.clone_state()).collect();

        Self::parallel_compute_residuals_from_cache(
          pool, law, flux, decomp, scratches, residuals, bcs,
        );
        Self::parallel_axpy(pool, states, residuals, dt);

        exchange_ghosts(decomp, states);
        Self::refresh_parallel_state_caches(decomp, states, scratches);
        Self::parallel_compute_residuals_from_cache(
          pool, law, flux, decomp, scratches, residuals, bcs,
        );
        Self::parallel_axpy(pool, states, residuals, dt);

        // Combine stages: state = 0.5 * u_old + 0.5 * state
        let tasks: Vec<_> = states
          .iter_mut()
          .zip(u_old.iter())
          .map(|(state, old_state)| {
            move || {
              let stage2 = state.clone_state();
              state.weighted_sum(0.5, old_state, 0.5, &stage2);
            }
          })
          .collect();

        pool.dispatch(tasks);
      }
    }

    Self::parallel_fix_owned(pool, law, decomp, states);
    dt
  };

  self.time += dt;
  self.step += 1;

  dt
}
