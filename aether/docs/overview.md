# aether — the runtime facade

`aether` is the thin top layer that ties everything into a runnable thing. It
assembles a world from the other crates and advances it in time. It holds no
physics of its own — it is the conductor, not an instrument.

## How it fits

`aether` sits above the simulation crates and below the assembler/consumer
(`sandbox`). It owns the runtime aggregates — `World`, `System`, `Aether` — and
the `WorldFactory` that wires a `cosmo` seed into meshes (`tessera`), state
(`pleroma`), and stages (`nexus` + physics). It depends on `chronos` for the
timescale-regime types it drives.

## What's inside

- **`factory.rs`** — `WorldFactory`: the builder that turns a `cosmo` seed plus a
  set of models into a compiled, runnable `World` (registers meshes, fields,
  couplers, and stages; derives `WorldConstants`).
- **`core.rs`** — the runtime hierarchy and the tick/advance logic:
  - **`World`** — one simulated body: its `tessera`, `pleroma`, `CompiledNexus`,
    constants, partition count, and timescale regime/clocks.
  - **`System`** — a star/planet system; system-level physics (n-body gravity)
    runs here before world-local physics.
  - **`Aether`** — the top container plus the shared thread `Pool`.

## Advancing time

Three entry points, in increasing capability:

- **`World::tick(pool, dt)`** — advance one world's DAG by `dt` (multi-rate
  inside).
- **`Aether::step(dt)`** — advance every world; non-multi-rate worlds are fused
  into one cross-world scheduler graph for parallelism.
- **`Aether::advance(game_dt)`** — regime-aware advance: live worlds integrate by
  `game_dt`; climatology worlds burst-then-hold (see
  [`chronos`](../../chronos/docs/overview.md)). Tracks `game_clock` vs `sim_time`
  per world.

## Adaptive refinement

`adapt` is the AMR driver: a `World` runs registered `MeshAdapter`s at the
end-of-tick barrier (criterion → balance → refine → remap fields → swap mesh +
bump epoch → emit `TopologyChanged`), bounded by an `AdaptGovernor`.
`RefinementCriterion` is the mesh- and consumer-agnostic seam (`GradientCriterion`,
`RegionRefinementCriterion` ship today). Checkpoints persist the adapted topology.
See [amr.md](../../docs/amr.md).

## See also

- The seed it consumes: [`cosmo`](../../cosmo/docs/overview.md).
- A concrete assembled world: [`sandbox`](../../sandbox/docs/overview.md).
- The execution model: [architecture](../../docs/architecture.md#the-execution-model).
- Runtime mesh adaptation: [amr.md](../../docs/amr.md).
