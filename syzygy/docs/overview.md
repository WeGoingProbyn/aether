# syzygy — coupling

`syzygy` owns **cross-mesh coupling semantics** — how a quantity on one mesh
influences another (ocean ↔ atmosphere). Same-mesh dependencies need nothing
special (they're ordinary `nexus` DAG edges); `syzygy` is for the harder case
where two distinct meshes must exchange conserved quantities.

## How it fits

`syzygy` consumes read-only mesh/coupler geometry from `tessera` and field access
from `nexus`, and provides coupling *stages* an assembler adds to the world. It
deliberately keeps the producing physics crates coupling-agnostic: a crate like
`aer` takes plain precomputed data, and the assembler (`sandbox`/`aether`) builds
the `syzygy` stencil and stages — so no physics crate depends on `syzygy`.

## What's inside

- **`stencil.rs`** — `CouplingStencil` / `CouplingEntry`: precompute, from a
  `tessera` coupler, which source-mesh cells map to which target-mesh cells.
- **`flux.rs`** — `ScalarInterfaceFlux` and `ScalarInterfaceDeposition`: move a
  scalar across the interface as a flux / deposit a quantity into the target
  (e.g. debit the ocean the latent heat that evaporation carried into the air).
- **`scalar.rs`** — `ScalarRelaxation`: relax a target field toward a source
  field across the coupler (e.g. pull the atmosphere's SST toward the ocean's
  surface temperature).

## The principle

Coupling is *explicit, precomputed data* — a stencil you can inspect and a stage
you can test — not hidden side effects. The discipline that keeps the coupled
system physical is conservation: debit one side exactly what you credit the
other.

## See also

- Air–sea coupling in context: [physics](../../docs/physics.md#coupling-syzygy).
- Adding a coupling: [extending](../../docs/extending.md#couple-two-processes).
