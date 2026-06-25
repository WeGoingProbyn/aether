# continuum — numerical methods

`continuum` is a **domain-neutral finite-volume solver**. It knows nothing about
air, water, or planets — it provides the numerics that physics crates plug their
equations into. Keeping the method generic is what lets a new fluid be a new
*equation*, not a new solver.

## How it fits

Physics crates (chiefly `aer`) depend on `continuum` and supply a conservation
law + numerical flux; `continuum` integrates one partition's cells. It is a
deliberately **serial CPU solver** — parallelism is `nexus`/`aether`'s job (N
solvers over N partitions), which keeps the numerics simple and testable.

## The generic core

`continuum` is generic over dimension `D`, state size `N`, and two traits
(`model.rs`):

- **`ConservationLaw<D, N>`** — the physics: `flux`, `max_wave_speed`, per-cell
  `source`, and `fix_state` (clamp unphysical states).
- **`NumericalFlux<D, N>`** — how neighbours exchange flux across a face; the
  built-in is `RusanovFlux`.

## What's inside

- **`solver.rs`** — `FvmSolver` drives integration; `SolverConfig` carries CFL,
  `dt_max`, and `TimeIntegration` (`ForwardEuler` | `Rk2`). Internal `Scratch`
  buffers keep the hot loop allocation-free (don't remove them).
- **`cpu.rs`** — `CpuBackend`, partitioned execution.
- **`implicit/`** — implicit / IMEX / hybrid backends behind the `FvmBackend`
  trait; matrix-free GMRES with Jacobians from automatic differentiation
  (`num-dual`).
- **`boundary.rs`** — `BoundaryCondition<D, N>` (`ghost_state`),
  `BoundaryRegistry`, built-ins `Transmissive` / `ReflectiveWall`, keyed by
  `BoundaryTag` (`Top/Bottom/Left/Right/Wall/Ground/Inflow/Outflow/…`).
- **`output.rs`** — `LawFieldSchema<D, N>` mapping states to named arrays, plus
  VTK snapshot writing for ParaView.

## Gotcha — AD over non-smooth functions

The implicit/IMEX paths differentiate the law to build Jacobians. AD through
`sqrt(0)`, `abs`, or `max` kinks yields NaN derivatives, so wave-speed and flux
code is regularised (e.g. a small `SPEED_EPS`) and tested from rest/vacuum
states.

## See also

- Adding a fluid: [extending](../../docs/extending.md#add-a-new-conservation-law--fluid).
- Its main consumer: [`aer`](../../aer/docs/overview.md). Time integrators:
  [`tempus`](../../tempus/docs/overview.md).
