# tempus — time integrators

`tempus` holds generic time-integration kernels — Runge-Kutta families,
Velocity-Verlet — and nothing else. It is the "how to step forward in time"
toolbox, kept separate from *what* is being stepped.

## How it fits

`tempus` deliberately depends on **nothing** but `utility` — not `nexus`, not
`pleroma`, not `tessera`, not any physics crate. Physics crates own their state
layout and call these kernels from their own stages: `gravitas` integrates body
state with Velocity-Verlet, `continuum`-based fluids use RK stepping. Because it
borrows no aether concepts, the integrators are reusable and unit-testable in
complete isolation.

## What's inside

- **`integrator.rs`** — the stepping kernels (explicit RK, Velocity-Verlet).
- **`ode.rs`** — the small ODE-shaped interface they operate against.

## See also

- A consumer: [`gravitas`](../../gravitas/docs/overview.md).
- Why it stays dependency-free: the central rule in
  [architecture](../../docs/architecture.md).
