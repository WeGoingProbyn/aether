use continuum::geometry::{
  CellGeometry, CellId, FaceGeometry, FaceId, IdentityMap,
};
use continuum::mesh::StructuredBlock;

fn assert_close(lhs: f64, rhs: f64, eps: f64, context: &str) {
  assert!(
    (lhs - rhs).abs() <= eps,
    "{}: lhs={} rhs={} |diff|={}",
    context,
    lhs,
    rhs,
    (lhs - rhs).abs()
  );
}

fn assert_identity_metrics<const D: usize>(dims: [usize; D], extent: [f64; D]) {
  let mesh = StructuredBlock::uniform(
    [0.0; D].into(),
    extent,
    dims,
    Box::new(IdentityMap::<D>),
  );
  let eps = 1e-12;

  for i in 0..mesh.cell_count() {
    let cell = CellId::from(i);
    let metrics = mesh.cell_metrics(cell);
    assert_close(metrics.sqrt_metric, 1.0, eps, "cell sqrt_metric");
    assert_close(
      metrics.phys_volume,
      metrics.comp_volume,
      eps,
      "cell physical/computational volume mismatch",
    );
    assert_close(
      metrics.comp_volume,
      mesh.cell_volume(cell),
      eps,
      "cell volume and comp_volume mismatch",
    );
  }

  for i in 0..mesh.face_count() {
    let face = FaceId::from(i);
    let metrics = mesh.face_metrics(face);
    let area_vec = mesh.face_area_vector(face);
    let area = mesh.face_area(face);
    let expected_normal = &area_vec / &area;

    assert_close(metrics.sqrt_metric, 1.0, eps, "face sqrt_metric");
    assert_close(
      metrics.phys_area,
      metrics.comp_area,
      eps,
      "face physical/computational area mismatch",
    );
    assert_close(
      metrics.comp_area,
      area,
      eps,
      "face area and comp_area mismatch",
    );
    assert_close(
      metrics.normal.magnitude(),
      1.0,
      eps,
      "face normal magnitude",
    );

    for d in 0..D {
      assert_close(
        metrics.normal[d],
        expected_normal[d],
        eps,
        "face metric normal mismatch",
      );
    }
  }
}

#[test]
fn identity_metrics_2d() {
  assert_identity_metrics([7, 3], [2.0, 0.6]);
}

#[test]
fn identity_metrics_3d() {
  assert_identity_metrics([3, 2, 4], [1.2, 0.8, 2.5]);
}
