# Aether

## Core responsibilities

Aether is designed to be modular and provide a clear and distinct set of seperation of concerns. To this effect, Aether is split into distinct subset of crates which all own and act upon a distinct set of core responsibilities. The idea is to provide an easy to extend and interpret set of multiphysics subsystems which can be manipulated and re arranged without needing to pull down thousands of lines of fragile operations.

### utility

Provides helper and general types which don't necessarily fit into a single multiphysics stage.

### Cosmo

Provides immutable state for building simulations, provides seeding, initial conditions, chooses integration schemes and allows serializing and deserializing of these states.

### Tessera

Owns mesh, geometry and topology definitions to be used by the underlying physics crates. Including the splitting and coupling of different parts of one larger mesh.

### Plemora

Provides the global mutable state for all physics to act upon. Allows reading and writing to states and passes these down stream to determine execution orders.

### Continuum

Underlying finite volume solver abstracted over conservation laws and flux schemes. Provides the underlying solver for all PDE solutions required by physics crates.

### Aer

Definitions for atmospheric physics

### Orbit

Definitions for orbital mechanics and their evolutions

### Terra

Definitions for geophysical processes like plate tectonics, terrain height maps, erosion etc.

### Nexus

DAG based orchestrator and plugin builder. Looks at defined reads and writes and determines execution order across multiphysics domain.

### Eidolon

A viewer into Plemora and Tessera to build an engine agnostic IR to be passed into a game engine of your choice (currently only supports bevy)
