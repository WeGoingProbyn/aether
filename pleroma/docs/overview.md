# pleroma — mutable state

`pleroma` is the **single owner of all mutable simulation state**. Every field a
physics stage evolves and every non-mesh resource lives here, behind a typed
registry. Centralising state is what makes the central rule possible: physics
crates hold no buffers, so there is exactly one place state can be and exactly
one thing to reason about for aliasing.

## How it fits

At setup, models register their fields against `pleroma`. At run-time `nexus`
asks `pleroma` for a `ScheduleAccess`, then hands each stage a `WorldAccess`
scoped to just the keys that stage declared. Stages read and write through that
borrowed handle; they never see the registry directly. `eidolon` reads `pleroma`
(read-only) to build snapshots.

## Core vocabulary

Keys come from `utility::domain` and are re-exported in `pleroma::prelude` (and
again by `nexus`):

- **`FieldKey { mesh, name }`** — a field on a mesh; **`ResourceKey`** —
  non-mesh-bound state (body state, sun direction, …).
- **`WorldAccess`** (`core/access.rs`) — the borrowed, scope-limited handle a
  stage receives: `read::<S>(key)`, `write::<S>(key)`, `resource::<R>(key)`,
  `resource_mut`. **`ScheduleAccess`** is the planner-side counterpart.
- **Field storage** (`core/storage.rs`), the `FieldStorage<N>` trait:
  - `SoaField<N>` — structure-of-arrays, fast component-wise sweeps;
  - `AosField<N>` — array-of-structures, fast per-cell access;
  - `LocalPartitionField<N>` — owned + ghost values for one partition, with
    `gather_partition_field` / `scatter_partition_owned` moving data between
    global fields and partition-local buffers.
- **`exchange_ghosts`** synchronises ghost layers across partitions.

`Pleroma` itself is the top-level handle setup/init code uses to register and
read state.

## See also

- Adding a field: [extending](../../docs/extending.md#add-a-new-field).
- The scheduler that borrows this state: [`nexus`](../../nexus/docs/overview.md).
