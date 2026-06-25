# nexus — the scheduler

`nexus` turns a pile of physics stages into a correctly-ordered, parallel
execution. It is the orchestration layer: it discovers dependencies from what
stages *declare* they read and write, builds a DAG, and runs everything
non-conflicting at once. It is also the vocabulary crate physics modules program
against.

## How it fits

Physics crates depend on `nexus` *alone* for their state-access vocabulary —
`nexus` re-exports `pleroma::prelude::*`, so a physics `Cargo.toml` lists
`nexus + tessera + continuum + utility` and never `pleroma` directly. `aether`
drives a `CompiledNexus` each tick; `pleroma` supplies the borrowed state.

## The `Stage` contract

A stage declares its data flow and provides its work:

- `name()`, `reads()` / `writes()` (`FieldKey`s),
- `resource_reads()` / `resource_writes()` (`ResourceKey`s),
- `subsystem()` — which cadence clock it runs on (default: once per outer tick),
- `run(StageContext)` — the work, over borrowed `WorldAccess`,
- optional `plan()` — return a `StagePlan` (a static task list, or a dynamic
  multi-task `Program`, e.g. a partitioned solve).

## How scheduling works

`Nexus::build` compiles stages into a `CompiledNexus`. Edges fall out of three
relationships between stages — **RAW** (read-after-write), **WAR**
(write-after-read), **WAW** (write-after-write) — on both fields and resources.
Whenever two stages conflict, the earlier-added one runs first; `before(a, b)`
pins extra ordering for physically-ordered stages that don't otherwise conflict.
Cycles are caught at build time. Layers of mutually-non-conflicting stages run in
parallel on the shared `Pool`.

## Multi-rate execution

Stages on different `SubsystemId`s can advance at different cadences. Register a
cadence with `set_subsystem_cadence(id, dt)`; `multirate_tick` operator-splits by
subsystem (deterministic ascending order, Gauss–Seidel) and subcycles each at its
own dt within one outer step — a fast atmosphere over a slow ocean, without
forcing the whole world onto the fastest dt. The integral over an outer step is
preserved. See [`chronos`](../../chronos/docs/overview.md) for the regime logic
layered on top.

## Also here

`WorldConstants` and friends (`AtmosphereConstants`, `RadiationConstants`) carry
per-world parameters the scheduler threads to stages via `StageContext`.

## See also

- Adding a stage: [extending](../../docs/extending.md#add-a-new-stage).
- The state it borrows: [`pleroma`](../../pleroma/docs/overview.md).
