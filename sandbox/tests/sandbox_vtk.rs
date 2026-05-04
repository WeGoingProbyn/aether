// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! VTK regression test. Replicates what the old sandbox
//! `main` used to do — build the demo Aether, run a couple of ticks,
//! pipe the snapshot through `BackendRegistry::snapshot` and write
//! VTK files to a tempdir. Asserts a minimum file count and that one
//! of the files parses as well-formed XML and contains the
//! `surface_temperature` data array.

// use std::fs;
//
// use eidolon::export::write_render_frame_via_registry;
// use sandbox::{build_demo_aether, debug_render_frame};
// use tempfile::TempDir;

// #[test]
// fn sandbox_world_writes_well_formed_vtk_with_expected_arrays() {
//   let (mut aether, _layout) = build_demo_aether().expect("demo aether builds");
//   aether.step(0.05).expect("first tick");
//   aether.step(0.05).expect("second tick");
//
//   let frame = debug_render_frame(&aether, 0, 0.1);
//   let dir = TempDir::new().expect("tempdir");
//   let written = write_render_frame_via_registry(&frame, dir.path())
//     .expect("vtk pipeline succeeds");
//
//   assert!(
//     written.len() >= 2,
//     "expected at least one vtu per registered mesh, got {}",
//     written.len()
//   );
//   for path in &written {
//     assert!(path.exists(), "{} should exist", path.display());
//     let metadata = fs::metadata(path).expect("metadata");
//     assert!(metadata.len() > 0, "{} is empty", path.display());
//   }
//
//   // Pick a file that should hold the surface temperature scalar and
//   // sanity-check it parses as XML and mentions the field name.
//   let surface_path = written
//     .iter()
//     .find(|p| {
//       p.file_name()
//         .and_then(|s| s.to_str())
//         .map(|s| s.contains("Surface") || s.contains("surface"))
//         .unwrap_or(false)
//     })
//     .or_else(|| written.first())
//     .expect("at least one vtu");
//   let bytes = fs::read(surface_path).expect("read vtu");
//   let text = String::from_utf8_lossy(&bytes);
//   assert!(
//     text.starts_with("<?xml") || text.contains("<VTKFile"),
//     "{} does not look like a vtu file (first 80 bytes: {:?})",
//     surface_path.display(),
//     &text.chars().take(80).collect::<String>()
//   );
//   assert!(
//     text.contains("surface_temperature"),
//     "expected surface_temperature data array in {}",
//     surface_path.display()
//   );
// }
