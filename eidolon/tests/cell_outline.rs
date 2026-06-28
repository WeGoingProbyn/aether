// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Phase 6 (AMR debug view): the cell-outline wireframe densifies where the mesh
//! is refined, so it is visually obvious where AMR is being applied.

use std::sync::Arc;

use eidolon::extract::mesh::cell_outline_lines;
use eidolon::ir::RenderGeometry;
use tessera::adaptive::AdaptiveMesh;
use tessera::cube_sphere::CubeSphere;
use tessera::mesh::Mesh;
use tessera::refine::AdaptRequest;
use utility::domain::{CellId, MeshKey, WorldId};

fn segment_count(mesh: &dyn Mesh<3>) -> usize {
  let rendered = cell_outline_lines(WorldId(0), MeshKey::SURFACE, mesh);
  match rendered.geometry {
    RenderGeometry::Lines(lines) => lines.segments.len(),
    other => panic!("expected line geometry, got {other:?}"),
  }
}

#[test]
fn refinement_adds_cell_outline_segments() {
  let base = AdaptiveMesh::new(Arc::new(CubeSphere::new([8, 8, 1], 0.9, 1.0)));
  let before = segment_count(&base);
  assert!(before > 0, "base mesh should have outline segments");

  let (refined, _remap) = base
    .refine(&AdaptRequest {
      refine: vec![CellId::from(36)],
      coarsen: vec![],
    })
    .unwrap();
  let after = segment_count(&refined);

  // Refining one cell into four adds outline edges (the refined region's grid is
  // denser), so the wireframe shows where AMR was applied.
  assert!(
    after > before,
    "refinement should add outline segments: {before} -> {after}"
  );
}
