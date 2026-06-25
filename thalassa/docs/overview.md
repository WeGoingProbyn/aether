# thalassa — ocean

`thalassa` is a thermodynamic ocean: a radial stack of water layers on a
cube-sphere shell. For this first proof the physics are purely thermal — the
surface layer exchanges heat with the air, and heat diffuses downward toward the
deep ocean. Like every physics crate it stores nothing globally.

## How it fits

An `OceanModel` registers the prognostic sea-water temperature and the net
surface flux on the ocean mesh, then adds the column-thermodynamics stage.
Radiation/`lumen` deposits the net flux; the air–sea coupling (`syzygy`) maps the
ocean's surface temperature onto the atmosphere as the SST that drives
evaporation. The ocean evolves far slower than the atmosphere, so it runs on its
**own subsystem clock** — the canonical use of the multi-rate scheduler.

## What's inside

- **`model.rs`** — `OceanModel` (builder: `register_fields`, `add_stages`,
  `with_subsystem`), `OceanColumnLayout` (panel/angular/radial layout of the
  shell), `OceanFields` (temperature + net flux keys).
- **`thermodynamics.rs`** — `OceanThermodynamicsStep`: the surface layer absorbs
  the net flux (`dT = dt·Q / (ρ·c_p·Δz)`); interior layers diffuse vertically
  with insulated top/bottom. A `max_stable_dt` guides the subsystem cadence.

## Role in the coupled budget

The ocean's finite heat capacity is what makes the air–sea cycle physical: it
self-limits evaporation instead of acting as an infinite reservoir, and it
absorbs the latent-heat debit when vapour leaves the surface — closing the moist
energy budget across meshes.

## See also

- Air–sea coupling: [`syzygy`](../../syzygy/docs/overview.md),
  [physics](../../docs/physics.md#coupling-syzygy).
- Multi-rate scheduling: [`nexus`](../../nexus/docs/overview.md),
  [`chronos`](../../chronos/docs/overview.md).
