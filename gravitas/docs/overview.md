# gravitas — gravity

`gravitas` is Newtonian n-body gravity, expressed as a single `nexus` stage. It
is the system-level physics that moves bodies (planets, moons) between ticks,
as opposed to the world-local fluid physics of `aer`/`thalassa`.

## How it fits

Body state is a non-mesh resource: `ResourceKey::Bodies`, a `BodyState<D>` held
in `pleroma`. `KeplerStage` pulls it via `WorldAccess`, advances it with
`tempus`'s Velocity-Verlet, and writes it back — declaring the resource as a
read/write so `nexus` orders it against anything else that touches body state.
`eidolon` reads the resulting positions to place each world's centre per tick.

## What's inside

- **`KeplerStage`** — the integration stage: gather bodies, accumulate pairwise
  Newtonian acceleration, step with Velocity-Verlet.

## How it stays composable

`gravitas` owns no state and no integrator of its own — state is a `pleroma`
resource and the time-stepping is a `tempus` kernel. It is a compact example of
the central rule applied to *resource* state rather than mesh fields.

## See also

- The integrator it uses: [`tempus`](../../tempus/docs/overview.md).
- Resource state: [`pleroma`](../../pleroma/docs/overview.md).
