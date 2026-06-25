# The physics

This is the deep dive: what aether actually simulates and the numerical choices
behind it. For the layering that hosts all of this, read
[architecture.md](architecture.md) first.

The reference world is a **coupled planetary system**: a compressible moist
atmosphere over a thermodynamic ocean and a terrain surface, lit by a star,
on a rotating cube-sphere planet. Each process lives in its own crate; they meet
only through fields in `pleroma` and couplers in `tessera`.

## Numerical foundation — finite volume (`continuum`)

`continuum` is a **domain-neutral** finite-volume solver. It knows nothing about
air, water, or planets — it is generic over:

- the spatial dimension `D`,
- the conserved-state size `N`,
- a `ConservationLaw<D, N>` (the physics: `flux`, `max_wave_speed`, per-cell
  `source`, and `fix_state` to clamp unphysical states), and
- a `NumericalFlux<D, N>` (how neighbouring cells exchange flux across a face;
  the built-in is `RusanovFlux`).

`FvmSolver` drives integration cell-by-cell on a single partition; it is a
deliberately *dumb serial CPU solver*. Parallelism is not its job —
`nexus`/`aether` run N solvers over N partitions. Internal `Scratch` buffers
keep the hot loop allocation-free.

Time stepping (`SolverConfig`) offers `ForwardEuler` and `Rk2`, CFL control, and
a `dt_max` clamp. Boundary conditions (`Transmissive`, `ReflectiveWall`, …) are
supplied through a `BoundaryRegistry` keyed by `BoundaryTag`.

**Solver backends.** Explicit, implicit (matrix-free GMRES), IMEX, and hybrid
schemes sit behind one `FvmBackend` trait. Implicit/IMEX paths build their
Jacobians by automatic differentiation (`num-dual`). A gotcha worth knowing:
AD through non-smooth functions (`sqrt(0)`, `abs`, `max` kinks) produces NaN
Jacobians, so wave-speed and flux code is regularised (e.g. a small `SPEED_EPS`)
and tested from rest/vacuum states.

## The atmosphere (`aer`)

`aer` is compressible Euler on the cube-sphere shell, with the features a stable
planetary atmosphere needs:

- **Well-balanced reconstruction.** A planetary atmosphere is almost entirely a
  hydrostatic balance `∇p = ρg`, with weather as a tiny perturbation on top. A
  naive scheme lets discretisation error of the huge balanced part swamp the
  weather. `aer` reconstructs each cell's state to shared faces *along the local
  hydrostatic profile*, so a fluid at rest holds `∇p = ρg` to machine precision
  and never spuriously drifts. Gravity is an analytic radial geopotential
  supplied by the kernel.
- **HEVI time stepping.** On a thin shell the vertical acoustic CFL limit is far
  stricter than the horizontal one, which would force tiny global steps.
  Horizontally-Explicit / Vertically-Implicit integration treats the vertical
  acoustic coupling implicitly (a per-column solve) and everything else
  explicitly, removing that limit and buying large stable steps. It runs through
  the same partitioned `nexus` path as the explicit solver (per-panel dispatch,
  radial columns).
- **Moisture & microphysics.** Water vapour is a transported tracer. Evaporation
  injects vapour from the sea surface; saturation adjustment condenses
  supersaturated vapour into precipitation, releasing latent heat. The moist
  energy budget is conserved across the air–sea interface.
- **Rotation & diagnostics.** Optional Coriolis rotation; a diagnostics stage
  derives temperature, pressure, and world-frame velocity from the conserved
  state each tick (these are what the query API and renderer read).

## The surface and terrain (`terra`)

`terra` owns a surface thermal slab — `dT/dt = NetSurfaceFlux /
heat_capacity_per_area` — and **first-class terrain**:

- A static **heightfield** (`SurfaceElevation`) and a categorical
  **land / ocean / ice** mask (`SurfaceType`, encoded via `SurfaceClass`), set
  once at setup and not evolved (inert data).
- A per-cell **surface albedo** field — the reusable "surface property as a
  field" contract: terrain writes a base albedo, and any later producer (snow,
  ice) can blend into the same field, which radiation reads. This is what makes
  ice-albedo feedback a drop-in later.

Terrain couples into the atmosphere through ordinary fields — the first such
coupling is **orographic lift** (wind forced up the windward slope / down the
lee), implemented as a momentum forcing that holds internal energy fixed so it
cannot disturb the well-balanced hydrostatic state.

## The ocean (`thalassa`)

`thalassa` is a thermodynamic ocean column: a radial stack of water layers on a
cube-sphere shell. The surface layer absorbs the net surface heat flux; heat
diffuses vertically toward the deep ocean. It supplies the sea-surface
temperature that drives evaporation, closing the air–sea loop. Because the ocean
evolves far slower than the atmosphere, it runs on its **own subsystem clock**
(see the timescale spectrum below).

## Radiation (`lumen`)

`lumen` is single-band gray-atmosphere radiative transfer. It reads the sun
direction (a resource a diurnal-rotation stage advances), computes shortwave
absorption (using the per-cell surface albedo field), and emits a
`RadiativeHeatingTendency` for the atmosphere and a `NetSurfaceFlux` into the
surface/ocean.

## Gravity (`gravitas`)

`gravitas` is Newtonian n-body gravity as a single stage. Body state is a
`ResourceKey::Bodies` resource (a `BodyState<D>`); `KeplerStage` integrates it
with `tempus`'s Velocity-Verlet. This is the system-level physics that moves
planets between ticks.

## Coupling (`syzygy`)

Processes meet in two ways. **Same-mesh** dependencies are plain `nexus` DAG
edges — if stage B reads what stage A writes on the same mesh, the scheduler
orders them. **Cross-mesh** exchange (ocean ↔ atmosphere) goes through `syzygy`:
a `CouplingStencil` precomputes which cells of one mesh map to which cells of
another, and `ScalarInterfaceFlux` / `ScalarInterfaceDeposition` /
`ScalarRelaxation` move conserved quantities across that interface. The
air–sea coupling (radiation, SST relaxation, evaporation, latent-heat sink) is
built from these, and it conserves the moist energy budget end-to-end.

## Time integration (`tempus`)

`tempus` holds generic time-integration kernels (Runge-Kutta, Velocity-Verlet)
with **no dependency** on the scheduler, state registry, or any physics crate.
Physics crates own their state layout and call these kernels from their own
stages — keeping the integrators reusable and unit-testable in isolation.

## The timescale spectrum (`chronos`)

A CFL-bound solver cannot be integrated for a simulated millennium, yet a
strategy game wants to advance days to centuries. Aether reconciles this with a
**multi-rate scheduler** plus a **climatology regime**, both built on `nexus`'s
subsystem mechanism.

- **Multi-rate scheduling.** A `Stage` declares a `SubsystemId`; `nexus`
  operator-splits by subsystem and subcycles each at its own cadence. The
  CFL-limited atmosphere runs on a fast clock, the ocean on a slow one, within
  one outer world step — without forcing the whole world onto the fastest dt.
  The split is deterministic (ascending subsystem order, Gauss–Seidel) and
  conserves the integral over an outer step.
- **Climatology aggregation.** `chronos` accumulates slowly-varying time-means
  (an exponential moving average) of chosen quantities into climatology fields,
  on their own slow subsystem. These are the smoothed aggregates a long-horizon
  consumer reads.
- **Burst-then-hold regime.** In the climatology regime, `World::advance` runs a
  short *burst* of live steps to keep the aggregates current, then advances the
  game clock over a long span while *holding* the Euler state — so a large game
  step costs only the burst, not a full integration. The game clock and the
  integrated sim time diverge by design; that divergence is the saving.
- **Regime transition continuity.** Switching a region between the climatology
  aggregate and the live solver must not show a discontinuity. `chronos` seeds
  the target from the source at the instant of handoff (`copy_field`) and runs a
  ramped relaxation (`ClimatologyNudgeStep`) that holds the live state on the
  climatology and releases it over a window. The burst length itself is *measured*
  from a convergence trace, not guessed.

## Conservation, stability, and testing posture

The physics is held to behavioural invariants, not just "it runs":

- well-balanced states stay balanced to machine precision;
- coupled budgets (moist energy across the air–sea interface) conserve;
- multi-rate subcycling preserves the integral over an outer step;
- couplings are landed one at a time, each behind a stability check (the demo
  stays finite and physical for many steps), and A/B tests isolate a coupling's
  signal from the shared dynamics.

Integration tests in `sandbox/tests` and `aether/tests` exercise the coupled
world end-to-end; each physics crate carries unit tests for its own kernels.
