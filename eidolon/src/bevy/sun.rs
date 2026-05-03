// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Stub for the eventual sun visualisation. v0.1 emits
//! `UpdateSunDirection` from the producer but the bevy backend
//! currently no-ops on it; the sandbox demo can spawn its own
//! directional light against the world. A future revision will use
//! this module to drive a child `DirectionalLight` whose orientation
//! tracks the sun-direction stream automatically.
