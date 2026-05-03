// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Phase 3: receiver-side coalescing keeps every lifecycle Update,
//! collapses stale per-attribute Updates to last-write-wins, and
//! propagates the latest sim_time / frame.

use eidolon::ir::{
  LayerHandle, LayerSamples, MeshHandle, ScalarSamples, Transform, Update,
  UpdateBatch, WorldHandle,
};
use eidolon::runtime::render_channel;

fn samples(values: &[f64]) -> LayerSamples {
  LayerSamples::Scalar(ScalarSamples::PerCell(values.to_vec()))
}

#[test]
fn stale_layer_samples_are_collapsed_to_the_last_value() {
  let (tx, rx) = render_channel(8);

  let layer = LayerHandle(42);
  for (i, sim_time) in [0.0, 0.1, 0.2, 0.3].iter().enumerate() {
    let batch = UpdateBatch {
      frame: i as u64,
      sim_time: *sim_time,
      updates: vec![
        Update::UpdateLayerSamples {
          handle: layer,
          samples: samples(&[i as f64]),
          epoch: i as u64 + 1,
        },
        Update::SetSimTime {
          sim_time: *sim_time,
          frame: i as u64,
        },
      ],
    };
    tx.send(batch).expect("receiver alive");
  }

  let merged = rx.drain_coalesced().expect("at least one batch");
  // Latest sim_time wins.
  assert_eq!(merged.frame, 3);
  assert!((merged.sim_time - 0.3).abs() < 1e-12);

  let sample_updates = merged
    .updates
    .iter()
    .filter(|u| matches!(u, Update::UpdateLayerSamples { .. }))
    .count();
  assert_eq!(sample_updates, 1, "stale sample updates should collapse");

  let set_sim_time_updates = merged
    .updates
    .iter()
    .filter(|u| matches!(u, Update::SetSimTime { .. }))
    .count();
  assert_eq!(set_sim_time_updates, 1, "only the last SetSimTime survives");

  // The kept UpdateLayerSamples must be the latest one (epoch 4).
  let kept = merged
    .updates
    .iter()
    .find_map(|u| match u {
      Update::UpdateLayerSamples { epoch, .. } => Some(*epoch),
      _ => None,
    })
    .unwrap();
  assert_eq!(kept, 4);
}

#[test]
fn lifecycle_updates_are_all_preserved_in_order() {
  let (tx, rx) = render_channel(8);

  let mesh_a = MeshHandle(1);
  let mesh_b = MeshHandle(2);
  let world = WorldHandle(0);

  // Adversarial: Register A, Update geometry A (twice), Free A,
  // Register B, Update geometry B. After Free A the late
  // UpdateMeshGeometry for A must survive because it was emitted
  // before the Free.
  tx.send(UpdateBatch {
    frame: 0,
    sim_time: 0.0,
    updates: vec![
      Update::RegisterMesh {
        handle: mesh_a,
        world,
        id: eidolon::ir::RenderMeshId {
          world: utility::domain::WorldId(0),
          mesh: utility::domain::MeshKey::SURFACE,
          representation: eidolon::ir::MeshRepresentation::BoundaryFaces,
        },
        label: "a".into(),
        source: eidolon::ir::MeshSource::TesseraMesh(
          utility::domain::MeshKey::SURFACE,
        ),
        geometry: eidolon::ir::RenderGeometry::Triangles(
          eidolon::ir::TriangleMesh::default(),
        ),
        transform: Transform::IDENTITY,
        geometry_epoch: 1,
        transform_epoch: 1,
      },
      Update::UpdateMeshGeometry {
        handle: mesh_a,
        geometry: eidolon::ir::RenderGeometry::Triangles(
          eidolon::ir::TriangleMesh::default(),
        ),
        epoch: 2,
      },
      Update::UpdateMeshGeometry {
        handle: mesh_a,
        geometry: eidolon::ir::RenderGeometry::Triangles(
          eidolon::ir::TriangleMesh::default(),
        ),
        epoch: 3,
      },
      Update::FreeMesh { handle: mesh_a },
      Update::RegisterMesh {
        handle: mesh_b,
        world,
        id: eidolon::ir::RenderMeshId {
          world: utility::domain::WorldId(0),
          mesh: utility::domain::MeshKey::ATMOSPHERE,
          representation: eidolon::ir::MeshRepresentation::BoundaryFaces,
        },
        label: "b".into(),
        source: eidolon::ir::MeshSource::TesseraMesh(
          utility::domain::MeshKey::ATMOSPHERE,
        ),
        geometry: eidolon::ir::RenderGeometry::Triangles(
          eidolon::ir::TriangleMesh::default(),
        ),
        transform: Transform::IDENTITY,
        geometry_epoch: 1,
        transform_epoch: 1,
      },
      Update::UpdateMeshGeometry {
        handle: mesh_b,
        geometry: eidolon::ir::RenderGeometry::Triangles(
          eidolon::ir::TriangleMesh::default(),
        ),
        epoch: 2,
      },
    ],
  })
  .unwrap();

  let merged = rx.drain_coalesced().expect("batch received");

  // Every Register and Free must be preserved.
  let registers = merged
    .updates
    .iter()
    .filter(|u| matches!(u, Update::RegisterMesh { .. }))
    .count();
  assert_eq!(registers, 2);
  let frees = merged
    .updates
    .iter()
    .filter(|u| matches!(u, Update::FreeMesh { .. }))
    .count();
  assert_eq!(frees, 1);

  // Per-mesh, only the last UpdateMeshGeometry survives.
  let mut a_geom_epochs = Vec::new();
  let mut b_geom_epochs = Vec::new();
  for u in &merged.updates {
    if let Update::UpdateMeshGeometry { handle, epoch, .. } = u {
      if *handle == mesh_a {
        a_geom_epochs.push(*epoch);
      } else if *handle == mesh_b {
        b_geom_epochs.push(*epoch);
      }
    }
  }
  assert_eq!(
    a_geom_epochs,
    vec![3],
    "A's stale geometry update collapsed"
  );
  assert_eq!(b_geom_epochs, vec![2]);

  // Order must still be: RegisterMesh A → UpdateMeshGeometry A →
  // FreeMesh A → RegisterMesh B → UpdateMeshGeometry B.
  let order: Vec<&str> = merged
    .updates
    .iter()
    .map(|u| match u {
      Update::RegisterMesh { .. } => "Register",
      Update::UpdateMeshGeometry { .. } => "UpdateGeometry",
      Update::FreeMesh { .. } => "Free",
      _ => "Other",
    })
    .collect();
  assert_eq!(
    order,
    vec![
      "Register",
      "UpdateGeometry",
      "Free",
      "Register",
      "UpdateGeometry"
    ]
  );
}

#[test]
fn empty_channel_returns_none() {
  let (_tx, rx) = render_channel(4);
  assert!(rx.drain_coalesced().is_none());
}
