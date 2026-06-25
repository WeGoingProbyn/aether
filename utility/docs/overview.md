# utility — shared foundation

`utility` is the bedrock every other crate builds on. It has no aether-specific
concepts of its own; it provides the math, concurrency, observability, and the
shared ID vocabulary that the rest of the workspace agrees on.

## How it fits

Everything depends on `utility` (directly or transitively). Crucially, it owns
`domain` — the *shared identifier vocabulary* (`WorldId`, `MeshKey`, `FieldKey`,
`FieldName`, `ResourceKey`, `CellId`, `SurfaceClass`, …). Keeping these in one
low-level crate is what lets a producer (`terra`) and a consumer (`eidolon`)
agree on, say, the encoding of land/ocean/ice without depending on each other.

## What's inside

- **`maths/`** — `Vector<T, D>`, `Matrix<T, R, C>`, `Quaternion<T>`. `Vector`
  wraps `Matrix<T, D, 1>`. Constructors take arrays: `Vec3::new([x, y, z])`,
  *not* separate args.
- **`thread/`** — a work-stealing `Pool` with `parallel_for()`, an `execute()`
  task DAG, `spawn()`, and the scoped schedulers (`ScopedScheduler`,
  `ScopedTaskGraph`) that `nexus`/`aether` use to run stage graphs.
- **`domain.rs`** — the ID vocabulary above. The closed enums (`FieldName`,
  `MeshType`, `ResourceKey`) are internal-by-design; consumers read through
  `eidolon`'s semantic vocabulary instead.
- **`profiler.rs`** — `#[profile]` attribute (inserts a thread-local `SpanGuard`
  at fn entry, no locking), plus `inline_profile!` / `end_profile!` for blocks.
- **`logger.rs`** — levels Trace→Fatal with `trace!`…`fatal!` macros.
- **`serial/`** — custom `Serialize`/`Deserialize` traits and derives; JSON and
  VTK XML backends.
- **`collections/`** — graph types used by the scheduler; **`constants.rs`** —
  physical/planetary constants; **`error.rs`** — `AetherResult` / `AetherError`
  with per-domain `ErrorDomain`s.

## See also

- Derives and `#[profile]` come from [`utility_macros`](../../utility_macros/docs/overview.md).
- The ID vocabulary in action: [architecture](../../docs/architecture.md),
  [pleroma](../../pleroma/docs/overview.md), [nexus](../../nexus/docs/overview.md).
