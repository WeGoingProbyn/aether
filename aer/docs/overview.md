# aer — atmosphere

`aer` is the compressible-Euler atmosphere: the planetary air on a cube-sphere
shell, with the machinery a stable, moist, rotating atmosphere needs. It is the
largest physics crate and the heaviest consumer of `continuum`.

## How it fits

`aer` owns the *logic* of stepping the atmosphere; all state lives in `pleroma`.
An `AtmosphereModel` registers the conserved Euler state (and diagnostic fields)
at setup and adds the stages that evolve them. It reads radiative heating from
`lumen`, sea-surface temperature from the air–sea coupling, and terrain slope
from `terra` (for orographic lift) — all as ordinary fields.

## Key pieces

- **`model.rs`** — `AtmosphereModel` (the builder: `register_fields`,
  `add_stages`), `AtmosphereFields`, `AtmosphereStageIds`.
- **`dynamics.rs`** — `EulerAtmosphereStep`, the time-stepping scheme selector
  (`AtmosphereScheme`: explicit or **HEVI**), `GravityMode`, `RotationMode`
  (Coriolis).
- **`flux.rs` / `thermal.rs`** — the conservation-law flux and the
  temperature-tendency ↔ Euler-energy bridges.
- **`diagnostics.rs`** — `EulerDiagnosticsStep` derives temperature, pressure,
  and world-frame velocity from the conserved state each tick (these feed the
  query API and renderer).
- **`microphysics.rs` / `tracers.rs`** — moisture: `EvaporationStep`,
  `SaturationAdjustmentStep`, latent heat, saturation vapour pressure.
- **`orographic.rs`** — `OrographicLiftStage`, terrain → atmosphere momentum
  forcing built from precomputed `LiftSite`s.
- **`shell.rs`** — `AtmosphereShellLayout`, sizing the radial atmosphere shell.

## The physics worth knowing

- **Well-balanced reconstruction** keeps a fluid at rest at `∇p = ρg` to machine
  precision, so weather isn't swamped by discretisation error of the hydrostatic
  background.
- **HEVI** (horizontally-explicit / vertically-implicit) removes the vertical
  acoustic CFL limit on thin shells, enabling large stable steps.

Both are explained in [physics.md](../../docs/physics.md#the-atmosphere-aer).

## See also

- The solver it drives: [`continuum`](../../continuum/docs/overview.md).
- Couplings into it: [`lumen`](../../lumen/docs/overview.md),
  [`terra`](../../terra/docs/overview.md), [`syzygy`](../../syzygy/docs/overview.md).
