# chronos — the timescale spectrum

`chronos` reconciles two incompatible time horizons: a CFL-bound solver that
cannot be integrated for a simulated millennium, and a strategy/4X consumer that
wants to advance days to centuries. It does this with slowly-varying climatology
aggregates and a burst-then-hold regime — all built on `nexus`'s multi-rate
subsystem mechanism, storing nothing globally.

## How it fits

`chronos` adds climatology *fields* (ordinary `pleroma` state) updated by
ordinary `nexus` stages on a slow subsystem, so the multi-rate scheduler already
shipped in `nexus` is the substrate. The aggregates flow to consumers through the
existing `eidolon` query path unchanged. The regime/transition driver lives one
level up in `aether::World::advance`, which consults `chronos`'s policy types.

## What's inside

- **`accumulator.rs`** — `ClimatologyAccumulatorStep`: an exponential moving
  average of a live field toward a climatology mean
  (`mean += (live − mean)·clamp(dt/τ, 0, 1)`), inert with respect to physics.
- **`model.rs`** — `ClimatologyModel` (builder, mirroring `OceanModel`/
  `AtmosphereModel`) and `ClimateQuantity`, the small fixed vocabulary of
  aggregated quantities.
- **`convergence.rs`** — instrumentation (`residual`, `settling_time`,
  `suggest_burst_steps`) to *measure* how fast aggregates settle, so the regime
  burst length is derived from data rather than guessed.
- **`regime.rs`** — `Regime` (Live / Climatology), `RegimeConfig` (burst length),
  and `TransitionState` / `TransitionKind` for handoffs.
- **`nudge.rs`** — `copy_field` (seed the target from the source at a handoff)
  and `ClimatologyNudgeStep` (ramped relaxation toward the climatology, reading
  the transition fraction from a resource) — the regime-transition continuity
  mechanism.

## The two regimes

- **Live** — the full solver runs at game time; consumers read instantaneous
  fields.
- **Climatology** — `World::advance` runs a short *burst* to keep aggregates
  current, then advances the game clock over a long span while *holding* the
  Euler state. The game clock and integrated sim time diverge by design; that
  divergence is the compute saving.

Switching between them seeds the target from the source for instantaneous
continuity and ramps a nudge to spin the live state up/down smoothly. The full
narrative is in [physics.md](../../docs/physics.md#the-timescale-spectrum).

## Scope

Continuity is proven on a directly-prognostic scalar (ocean temperature);
spinning up the conserved Euler state via a derived quantity, and fidelity-LOD /
region wake, are deliberate follow-ons.

## See also

- The scheduler it builds on: [`nexus`](../../nexus/docs/overview.md).
- How a consumer reads the aggregates: [rendering](../../docs/rendering.md).
