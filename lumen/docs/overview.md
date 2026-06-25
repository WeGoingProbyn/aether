# lumen — radiation

`lumen` is single-band gray-atmosphere radiative transfer: it turns starlight
into heating of the air and a net heat flux into the surface/ocean. It is the
energy source that drives the whole coupled climate.

## How it fits

`lumen` reads the sun direction (a `ResourceKey` that a diurnal-rotation stage
advances each tick) and the per-cell surface albedo field that `terra` maintains,
then writes two fields other crates consume: a `RadiativeHeatingTendency` for
`aer` and a `NetSurfaceFlux` for `terra`/`thalassa`. Producer and consumers meet
only through those named fields.

## What's inside

- **`model.rs`** — `RadiationModel` (builder: `register_fields`, `add_stages`,
  `from_world_constants`, `with_surface_albedo_field`), `RadiationCoefficients`.
- **`transfer.rs`** — `RadiativeTransferStep`: shortwave absorption using the
  per-cell albedo (`absorbed = (1 − albedo)·incoming`), longwave exchange, and
  the net surface flux deposit.
- **`diurnal.rs`** — `DiurnalSunStep`: rotates the `SunPosition` resource to make
  day and night.

## The albedo seam

Radiation reads albedo as a *field*, not a constant. Any producer can write or
blend into that field — terrain today, snow/ice later — so adding ice-albedo
feedback requires no change in `lumen`. This is the reusable "surface property as
a field" pattern (see [`terra`](../../terra/docs/overview.md)).

## See also

- Consumers of its outputs: [`aer`](../../aer/docs/overview.md),
  [`terra`](../../terra/docs/overview.md),
  [`thalassa`](../../thalassa/docs/overview.md).
