# Aether

## Overview

Aether is a modular multiphysics simulation framework designed around **strict separation of concerns** and **composable physics systems**.

The goal is to provide:

- clear boundaries between simulation layers
- extensible physics modules
- deterministic execution via dependency resolution
- decoupling of simulation, numerics, and rendering

Rather than a monolithic engine, Aether is composed of independent crates which interact through well-defined interfaces.

---

## Core Design Principles

- **Separation of concerns**  
  Geometry, state, physics, and orchestration are fully decoupled.

- **Data-oriented architecture**  
  Simulation state is stored centrally and transformed by systems.

- **Deterministic execution**  
  All physics systems declare read/write dependencies, resolved via a DAG.

- **Pluggable physics**  
  New physics modules can be added without modifying existing ones.

- **Engine agnostic**  
  Rendering is handled separately via an intermediate representation.

---

## Crate Responsibilities

---

### `utility`

Provides shared foundational components:

- math types (vectors, matrices, quaternions)
- collections and graphs
- serialization helpers
- threading utilities

This crate contains no domain-specific logic.

---

### `cosmo`

Defines the **immutable simulation configuration**.

Responsibilities:

- seeds and procedural generation inputs
- planetary and system parameters
- initial conditions (ICs)
- integration scheme selection
- serialization / deserialization

> Cosmo is the **single source of truth for starting conditions** and is never mutated at runtime.

---

### `tessera`

Defines the **spatial domain** of the simulation.

Responsibilities:

- mesh generation (cube-sphere, grids, etc.)
- topology and adjacency relationships
- geometric transformations
- coupling between domain regions (e.g. atmosphere ↔ surface)

> Tessera defines *where* physics happens, but contains no physical values.

---

### `pleroma`

Owns the **global mutable simulation state**.

Responsibilities:

- storage of all simulation fields (temperature, pressure, velocity, etc.)
- read/write access for physics systems
- field lifecycle management

> Pleroma contains **values only**, never geometry or physics logic.

---

### `continuum`

Provides the **numerical solver layer**.

Responsibilities:

- finite volume methods (FVM)
- flux computation
- reconstruction schemes
- time integration

> Continuum is domain-agnostic and operates over tessera-defined meshes.

---

### `aer`

Defines **atmospheric physics**.

Examples:

- thermodynamics
- gas dynamics
- radiative processes (future)
- composition evolution

---

### `terra`

Defines **geophysical processes**.

Examples:

- terrain generation
- erosion
- plate tectonics (future)
- crust and mantle interactions

---

### `gravitas`

Defines **orbital mechanics**.

Examples:

- n-body dynamics
- gravitational interactions
- orbital evolution

---

### `nexus`

The **execution engine** of Aether.

Responsibilities:

- builds a directed acyclic graph (DAG) of system dependencies
- resolves execution order based on read/write access
- schedules physics systems
- enables parallel execution where possible

> Nexus orchestrates *when* and *in what order* physics systems run.

---

### `eidolon`

The **rendering bridge**.

Responsibilities:

- reads from `pleroma` (fields) and `tessera` (geometry)
- constructs an engine-agnostic intermediate representation (IR)
- translates IR to specific backends (currently Bevy)

> Eidolon does not perform simulation — it only **interprets simulation state for presentation**.

---

### `sandbox`

Example binary / playground.

Used for:
- running simulations
- testing configurations
- integrating with rendering backends

---

### Naming Scheme

Generally following latin or greek naming scheme:

- Aether → the medium / substrate in which everything exists

- Cosmo → origin / ordering principle  
- Tessera → structure / tiling of space  
- Pleroma → fullness / realised state  
- Nexus → connection / causality  
- Eidolon → image / projection  

- Continuum → continuous fields / conservation laws / numerical flow  
- Aer → air / atmosphere / gaseous processes  
- Terra → earth / solid body / geophysical processes  
- Gravitas → motion / celestial mechanics / gravitational evolution 

---

## Dependency topology

```text
                 ┌──────────┐
                 │ utility  │
                 └────┬─────┘
                      │
         ┌────────────┼──────────────┐
         ▼            ▼              │ 
    ┌─────────┐  ┌─────────┐         │ 
 ┌──┤ tessera │  │  cosmo  │         │ 
 │  └────┬────┘  └────┬────┘         │ 
 │       │            │              │
 │       ▼            │              │
 │  ┌─────────┐       │              │
 ├──┤ pleroma │◀────┬─┘              │
 │  └─────────┘     │                │
 │                  │                │
 │              ┌───┴───┐            │
 │              │ nexus │            │
 │              └───┬───┘            │
 │                  │                │
 │                  ▼                ▼
 │┌─────┬───────┬──────────┬────────────────┐
 ││ aer │ terra │ gravitas │ future physics │
 │└─────┴───────┼──────────┼────────────────┘
 │              │continuum │
 │              └──────────┘            
 │              ┌──────────┐    
 └─────────────▶│ eidolon  │       (read-only viewer over pleroma + tessera for rendering)
                └────┬─────┘
                     ▼
                ┌──────────┐
                │ sandbox  │       (your binary/project)
                └──────────┘
```

---

## Execution Model

Aether does not use a linear pipeline. Instead, it operates as a dependency-driven simulation graph.

### Step 1 — Initialization

- `cosmo` defines initial conditions  
- `tessera` constructs the spatial domain  
- `pleroma` initializes fields from cosmo  

---

### Step 2 — Graph Construction

Physics systems declare:

- fields they **read**
- fields they **write**

`nexus` builds a DAG from these dependencies.

---

### Step 3 — Simulation Step

Each tick:

- `nexus` resolves execution order via topological sort  
- physics systems execute in dependency-safe order  
- fields in `plerom` are updated  

---

### Step 4 — Rendering

- `eidolon` samples:
  - geometry from `tessera`
  - fields from `pleroma`
- builds render IR  
- passes to backend (e.g. Bevy)  

---

## Data Flow

```text
cosmo → initial conditions
        ↓
tessera → geometry
        ↓
pleroma → fields (state)
        ↓
nexus → orchestrates updates
        ↓
physics systems → transform state
        ↓
eidolon → visualises state
```

---

## Architectural Rules

- No physics crate owns state
- No physics crate owns geometry
- All mutation happens in pleroma
- All geometry lives in tessera
- All execution ordering is handled by nexus
- Rendering is strictly read-only

---

## Planned Extensions

- radiative transfer systems
- chemistry / phase transitions
- magnetosphere and plasma interactions
- adaptive mesh refinement (AMR)
- multiple rendering backends

---

## Summary

Aether is structured around a simple idea:

Define the world (cosmo), shape it (tessera), store it (pleroma), evolve it (physics via nexus), and observe it (eidolon).

## Note

This is very much a passion project driven by just myself (WeGoingProbyn), I work on it in my free time when I feel motivated to do so. If you're using this project yourself, I can't guarantee API consistency and I can't guarantee fixes and updates at a pace which you might expect from other maintainers. Feel free to open issues and PRs if you wish and I will get around to them when I can.



