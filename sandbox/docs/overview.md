# sandbox — the integration rig

`sandbox` is where all the pieces are wired into a concrete, runnable world and
rendered. It is both the reference for *how a consumer assembles aether* and the
home of the headless integration tests that keep the coupled system honest. It is
not a library other crates depend on — it sits at the very top.

## How it fits

`sandbox` assembles the full pipeline: a `cosmo` seed → `tessera` meshes →
`pleroma` fields → `nexus` + physics stages → an `eidolon` render config. The
same build logic feeds both the Bevy demo (`src/main.rs`) and the headless tests
(`tests/`), so the demo and the regression tests never drift.

## What's inside

- **`lib.rs`** — world builders. `build_showcase_world()` assembles the reference
  world (terrain + ocean + moist rotating atmosphere + radiation + climatology),
  and `showcase_extract_config()` describes what `eidolon` should surface.
  Smaller builders (`build_terrain_world`, `build_ocean_world`, …) isolate a
  single coupling for focused tests.
- **`main.rs`** — the Bevy showcase: spins the sim on a background thread, streams
  `eidolon` batches over a channel to the render backend, and offers keyboard
  toggles (debug fields vs rendered look, which field/climatology to overlay, and
  the live↔climatology regime).
- **`atmosphere.rs` / `atmosphere.wgsl`** — consumer-side art: an analytic
  atmospheric-scattering material. This lives here on purpose — *the look is the
  consumer's call*, kept out of `aether`/`eidolon`.
- **`tests/`** — end-to-end checks: coupled-world stability over many steps,
  air–sea water-cycle activity, orographic lift, surface-albedo radiation,
  geographic queries, climatology, and more.

## What it demonstrates

`sandbox` is the practical answer to "how do I use this?" — it shows the assembly
order, how to register models, how to add couplings, how to configure the render
extract, and how a consumer styles art-free data into a finished planet.

## See also

- Run it: `cargo run -p sandbox`.
- The assembly model: [architecture](../../docs/architecture.md),
  [extending](../../docs/extending.md).
- The render path it drives: [rendering](../../docs/rendering.md).
