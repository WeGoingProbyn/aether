// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Geometry and space: meshes, topology, partitioning, cross-mesh coupling, and
//! geographic (lat/lon) addressing. Holds no mutable simulation state — only the
//! static spatial structure that `pleroma` fields live on and `eidolon` reads.
//!
//! See `tessera/docs/overview.md` for the module map and the curvilinear-metric
//! gotcha.

pub mod coupling;
pub mod cube_sphere;
pub mod geo;
pub mod geometry;
pub mod mesh;
pub mod partition;
pub mod radial_stack;
pub mod spatial;
pub mod topology;
pub mod world_mesh;
