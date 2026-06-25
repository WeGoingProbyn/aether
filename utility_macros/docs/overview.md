# utility_macros — proc-macros

`utility_macros` is the procedural-macro crate behind `utility`. It exists only
to generate boilerplate; it has no runtime logic of its own and is re-exported
through `utility` (you normally write `utility::Serialize`, not
`utility_macros::Serialize`).

## What it provides

- **`#[derive(Serialize)]` / `#[derive(Deserialize)]`** — implement `utility`'s
  custom serialization traits for a type, used by the JSON and VTK backends.
- **`#[derive(StateDiagnostics)]`** — generate the diagnostic accessors used to
  introspect simulation state.
- **`#[profile]`** — wrap a function body in a profiler `SpanGuard` so it shows
  up in `utility::profiler` output with zero hand-written instrumentation.

## How it fits

It sits at the very bottom of the dependency graph alongside `utility`. Because
the derives target `utility`'s own traits (not `serde`), the workspace controls
its serialization format end-to-end — which is what makes the VTK XML export in
`eidolon` and `continuum::output` possible without an external schema.

## See also

- [`utility/docs/overview.md`](../../utility/docs/overview.md) — the traits and
  profiler these macros target.
