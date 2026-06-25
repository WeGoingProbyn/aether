# cosmo — initial conditions

`cosmo` describes the **immutable starting point** of a world: what bodies exist
and what they're made of. It is pure configuration — no mutable state, no
geometry, no stages. Think of it as the seed a world grows from.

## How it fits

`cosmo` is consumed at world-setup time. `aether`'s `WorldFactory` reads a
`CelestialBody` seed to derive `WorldConstants` (mass, radius, surface gravity,
atmosphere makeup, the stellar irradiance a primary star implies), which then
parameterise the meshes, fields, and physics stages. Nothing downstream mutates
`cosmo` data; it is the fixed input the rest of the pipeline is built around.

## What's inside

- **`body.rs` / `kind.rs`** — `CelestialBody` and `BodyKind`
  (`RockyBody`, `GasGiant`, `Star`), carrying physical properties: mass, radius,
  surface temperature/pressure, atmosphere composition, axial tilt, angular
  velocity, luminosity.
- **`system.rs`** — groupings of bodies into a star/planet system.
- **`factory.rs`** — ready-made bodies (`factory::earth()`, `factory::sun()`,
  `factory::mars()`, …) so demos and tests don't hand-build seeds.

## See also

- How a seed becomes a runnable world:
  [`aether/docs/overview.md`](../../aether/docs/overview.md).
- Where the derived constants are used: [physics](../../docs/physics.md).
