# Aether

Aether is a Rust-based experimental project for developing and validating
numerical solvers for partial differential equations (PDEs), with real-time
visualization using the Bevy game engine.

It emphasizes a **modular, engine-agnostic architecture**, separating
core mathematical and numerical logic from application and rendering.

---

## Overview

Aether is designed as a foundation for experimenting with continuum mechanics,
transport equations, and eventually fluid dynamics across different coordinate
systems.

The codebase is structured to:

- Keep numerical solvers independent of any game engine
- Enable quick validation through real-time visualization
- Scale toward more complex geometries and coordinate systems
- Serve as a potential basis for interactive simulations or games
- It does not aim to solve transport equations to the detail necessary for real world applications

At present, the project focuses on solving and visualizing simple PDEs (such as
temperature advection–diffusion) on structured Cartesian grids using finite differences.
See the issues tab for a todo list of future plans.

---

## Project Structure

```
aether/
├── app/                       # Application layer (Bevy-based)
│   └── src/
│       ├── plugins/           # Bevy plugins
│       └── main.rs            # Application entry point
└── continuum/                 # Core numerical / continuum library
    └── src/
        ├── grid/              # Grid system abstractions
        ├── field/             # Scalar and field data structures
        ├── maths/             # Math utilities and primitives
        ├── solver/            # PDE solvers and time-stepping logic
        └── tests/             # Sanity and correctness tests
    └── docs/
        └── domain.pdf         # How does continuum abstract across domains
```

The `continuum` module contains **no Bevy dependencies** and is intended to be
reuseable in headless simulations or other frontends.

The `app` module integrates the solvers into a Bevy application for visualization
and interaction.

---

## Prerequisites

- [Rust](https://www.rust-lang.org/tools/install) (latest stable recommended)
- Cargo (installed with Rust)
- A GPU capable of running Bevy’s rendering backend

---

# Installation

Clone the repository:

```bash
git clone https://github.com/WeGoingProbyn/aether.git
cd aether
```

# Building
Build using cargo's compiler:

```bash
cargo b --release
```

# Running

Run the application layer to visualize the simulation:

```bash
cargo run --release
```

When running, a window will open displaying the current scalar field.

# Testing

Run the test suite (primarily targeting the continuum module):

```bash
cargo test
```
---

These tests focus on basic correctness and numerical sanity checks,
following the assumptions underpinning the mathematical basis for
continuum's solvers. Such as conservation of computed quantities,
correctness around boundary implementations etc.

# Project Modules
## app

The application layer built on top of Bevy:

- Owning simulation state as Bevy resources

- Scheduling solver time steps

- Uploading field data to GPU textures

- Rendering scalar fields for visual inspection

- Handling input and application lifecycle

This layer is intentionally thin and delegates all numerical work to continuum.

## continuum

The core numerical and mathematical library:

- Grid definitions and indexing

- Topolgy and geometry definitions

- Field storage and access

- Coordinate system abstractions

- PDE discretization and time-stepping logic

- Numerical correctness tests

This module is engine-agnostic and does not depend on Bevy or any rendering code.

---

# Development

Generate and view documentation:

cargo doc --open

---

Contributing

Contributions are welcome.

If you’re interested in numerical methods, simulation, rendering, or Rust
architecture, feel free to open an issue or submit a pull request.

## License

MIT License. See the `LICENSE` file for details.

## Authors

- [@WeGoingProbyn](https://github.com/WeGoingProbyn)
- [@TheCharlieKennedy](https://github.com/TheCharlieKennedy)

## Acknowledgments

- [Bevy](https://github.com/bevyengine/bevy)
