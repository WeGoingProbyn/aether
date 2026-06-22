// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Orographic lift — the first terrain↔atmosphere coupling.
//!
//! Air flowing over rising ground is forced upward; over falling ground it
//! subsides. The induced vertical velocity at the lower boundary is
//! `w = u_h · ∇h`, where `u_h` is the horizontal wind and `∇h` the terrain
//! slope. This drives windward uplift (and cooling → precipitation) and a lee
//! rain shadow — emergent weather straight out of the heightfield.
//!
//! Design (per the inert-data-first plan): terrain lives on the surface mesh;
//! this stage relaxes the *atmosphere bottom layer's* radial velocity toward
//! `w = u_h·∇h`. It is deliberately a relaxation forcing behind a clamp and a
//! rate, so it cannot inject an unbounded impulse — the stability guard the
//! plan asks for. Internal energy (hence pressure / the well-balanced
//! hydrostatic state) is held fixed; only kinetic energy changes, as physical
//! for the work the terrain does on the flow.
//!
//! To keep `aer` free of a `syzygy` dependency, the stage consumes *plain
//! precomputed data* ([`LiftSite`]s): the surface→atmosphere cell pairing,
//! per-site tangent frame, and per-site terrain gradient are all assembled once
//! at setup by [`build_lift_sites`] from the meshes and the coupler pairing.

use nexus::{FieldKey, FieldStorage, SoaField, Stage, StageContext};
use tessera::geo::GeoCoord;
use tessera::mesh::Mesh;
use tessera::topology::FaceConnection;
use utility::{
  domain::CellId,
  error::{AetherError, AetherResult},
};

use crate::error::AerError;

fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
  a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Local radial / east / north unit vectors at a geographic coordinate.
fn enu_basis(geo: &GeoCoord) -> ([f64; 3], [f64; 3], [f64; 3]) {
  let (sin_lat, cos_lat) = geo.lat.sin_cos();
  let (sin_lon, cos_lon) = geo.lon.sin_cos();
  let r_hat = [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat];
  let east = [-sin_lon, cos_lon, 0.0];
  let north = [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat];
  (r_hat, east, north)
}

/// Per-cell horizontal terrain gradient `[∂h/∂east, ∂h/∂north]` (slope, m/m),
/// reconstructed by least squares from interior-face neighbours in the local
/// tangent frame. Working in world space keeps it correct across cube-sphere
/// panel seams; the fit of a (locally) linear field is exact up to curvature.
pub fn compute_enu_gradient<M>(
  mesh: &M,
  elevation: &[f64],
  surface_radius: f64,
) -> Vec<[f64; 2]>
where
  M: Mesh<3> + ?Sized,
{
  let n = mesh.cell_count();
  let mut grad = vec![[0.0; 2]; n];
  for i in 0..n {
    let cell = CellId::from(i);
    let centroid = mesh.cell_world_centroid(cell);
    let c = [centroid[0], centroid[1], centroid[2]];
    let geo = GeoCoord::from_world(&centroid, surface_radius);
    let (_, east, north) = enu_basis(&geo);

    // Normal equations for [g_e, g_n] minimising Σ(g_e·de + g_n·dn − dh)².
    let (mut sxx, mut sxy, mut syy, mut sxh, mut syh) =
      (0.0, 0.0, 0.0, 0.0, 0.0);
    for &face in mesh.cell_faces(cell) {
      if let FaceConnection::Interior { owner, neighbour } =
        mesh.face_connection(face)
      {
        let nb = if *owner == cell { *neighbour } else { *owner };
        let nc = mesh.cell_world_centroid(nb);
        let d = [nc[0] - c[0], nc[1] - c[1], nc[2] - c[2]];
        let de = dot3(&d, &east);
        let dn = dot3(&d, &north);
        let dh = elevation[nb.index()] - elevation[i];
        sxx += de * de;
        sxy += de * dn;
        syy += dn * dn;
        sxh += de * dh;
        syh += dn * dh;
      }
    }
    let det = sxx * syy - sxy * sxy;
    if det.abs() > 1e-9 * (sxx * syy).max(1.0) {
      grad[i] = [(syy * sxh - sxy * syh) / det, (sxx * syh - sxy * sxh) / det];
    }
  }
  grad
}

/// A single place where orographic lift is applied: an atmosphere bottom-layer
/// cell, its tangent frame, and the terrain slope of the surface cell beneath.
#[derive(Clone, Copy, Debug)]
pub struct LiftSite {
  /// The atmosphere bottom-layer cell to force.
  pub target: CellId,
  pub r_hat: [f64; 3],
  pub east: [f64; 3],
  pub north: [f64; 3],
  /// Terrain gradient `[∂h/∂east, ∂h/∂north]` at the paired surface cell.
  pub grad: [f64; 2],
}

/// Assemble the lift sites from the meshes, the surface elevation field, and
/// the surface→atmosphere cell pairing (`(surface_cell, atmosphere_cell)`,
/// typically the radial-stack coupler's entries). Surface terrain gradients are
/// reconstructed once here; the tangent frame is taken at the atmosphere cell
/// (radially aligned with its surface partner).
pub fn build_lift_sites<A, S>(
  atmosphere_mesh: &A,
  surface_mesh: &S,
  elevation: &[f64],
  surface_radius: f64,
  pairings: &[(CellId, CellId)],
) -> Vec<LiftSite>
where
  A: Mesh<3> + ?Sized,
  S: Mesh<3> + ?Sized,
{
  let grad = compute_enu_gradient(surface_mesh, elevation, surface_radius);
  pairings
    .iter()
    .map(|&(surface_cell, atmosphere_cell)| {
      let centroid = atmosphere_mesh.cell_world_centroid(atmosphere_cell);
      let geo = GeoCoord::from_world(&centroid, surface_radius);
      let (r_hat, east, north) = enu_basis(&geo);
      LiftSite {
        target: atmosphere_cell,
        r_hat,
        east,
        north,
        grad: grad[surface_cell.index()],
      }
    })
    .collect()
}

/// Relax each site's radial velocity toward `w = u_h·∇h`, holding internal
/// energy fixed. Pure (no `StageContext`) so it can be unit-tested directly.
pub fn apply_orographic_lift(
  state: &mut SoaField<6>,
  sites: &[LiftSite],
  relaxation: f64,
  max_velocity: f64,
) {
  for site in sites {
    if site.target.index() >= state.len() {
      continue;
    }
    let cur = state.state(site.target);
    let rho = cur[0];
    let m = [cur[1], cur[2], cur[3]];
    let energy = cur[4];
    let vapour = cur[5];
    if !rho.is_finite() || rho <= 0.0 {
      continue;
    }

    let ue = dot3(&m, &site.east) / rho;
    let un = dot3(&m, &site.north) / rho;
    let vr = dot3(&m, &site.r_hat) / rho;
    let w = (ue * site.grad[0] + un * site.grad[1])
      .clamp(-max_velocity, max_velocity);
    let dvr = relaxation * (w - vr);

    let mut m_new = m;
    for d in 0..3 {
      m_new[d] += rho * dvr * site.r_hat[d];
    }
    // Hold internal energy fixed: total energy moves only by the ΔKE.
    let ke_old = 0.5 * dot3(&m, &m) / rho;
    let ke_new = 0.5 * dot3(&m_new, &m_new) / rho;
    let updated = [
      rho,
      m_new[0],
      m_new[1],
      m_new[2],
      energy + (ke_new - ke_old),
      vapour,
    ];
    state.write(site.target, &updated);
  }
}

/// Nexus stage wrapping [`apply_orographic_lift`] over the atmosphere state.
pub struct OrographicLiftStage {
  state: FieldKey,
  sites: Vec<LiftSite>,
  relaxation: f64,
  max_velocity: f64,
  writes: [FieldKey; 1],
}

impl OrographicLiftStage {
  /// `relaxation` ∈ (0, 1] is the per-step fraction of the velocity gap closed;
  /// `max_velocity` (m/s) clamps the induced vertical velocity. Both keep the
  /// forcing bounded.
  pub fn new(
    state: FieldKey,
    sites: Vec<LiftSite>,
    relaxation: f64,
    max_velocity: f64,
  ) -> Self {
    Self {
      state,
      sites,
      relaxation,
      max_velocity,
      writes: [state],
    }
  }
}

impl Stage for OrographicLiftStage {
  fn name(&self) -> &'static str {
    "aer_orographic_lift"
  }

  fn reads(&self) -> &[FieldKey] {
    &[]
  }

  fn writes(&self) -> &[FieldKey] {
    &self.writes
  }

  fn run(&mut self, mut ctx: StageContext<'_>) -> AetherResult<()> {
    let state: &mut SoaField<6> =
      ctx.world.fields.write(self.state).ok_or_else(|| {
        AetherError::new(AerError::MissingWriteField)
          .context(format!("{:?}", self.state))
      })?;
    apply_orographic_lift(
      state,
      &self.sites,
      self.relaxation,
      self.max_velocity,
    );
    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::sync::Arc;
  use tessera::cube_sphere::CubeSphere;
  use tessera::geometry::CellGeometry;

  const R_INNER: f64 = 6.371e6;
  const R_OUTER: f64 = 6.381e6;

  #[test]
  fn gradient_recovers_a_linear_world_field() {
    // h = z (world). Its ENU gradient is [ẑ·east, ẑ·north] = [0, cos(lat)].
    let mesh = Arc::new(CubeSphere::new([16, 16, 1], R_INNER, R_OUTER));
    let elevation: Vec<f64> = (0..mesh.cell_count())
      .map(|i| mesh.cell_world_centroid(CellId::from(i))[2])
      .collect();
    let grad = compute_enu_gradient(mesh.as_ref(), &elevation, R_INNER);
    for i in 0..mesh.cell_count() {
      let geo = GeoCoord::from_world(
        &mesh.cell_world_centroid(CellId::from(i)),
        R_INNER,
      );
      // Skip near-polar cells where the tangent fit is most distorted.
      if geo.lat.abs() > 60_f64.to_radians() {
        continue;
      }
      assert!(
        grad[i][0].abs() < 0.03,
        "east slope {} at cell {i}",
        grad[i][0]
      );
      assert!(
        (grad[i][1] - geo.lat.cos()).abs() < 0.03,
        "north slope {} vs cos(lat) {} at cell {i}",
        grad[i][1],
        geo.lat.cos()
      );
    }
  }

  // A single site at the equator/prime meridian: r̂=+x, east=+y, north=+z.
  fn equator_site(grad: [f64; 2]) -> LiftSite {
    LiftSite {
      target: CellId::from(0),
      r_hat: [1.0, 0.0, 0.0],
      east: [0.0, 1.0, 0.0],
      north: [0.0, 0.0, 1.0],
      grad,
    }
  }

  fn eastward_wind_state(speed: f64) -> SoaField<6> {
    // ρ=1, momentum purely eastward (+y), some internal energy, no vapour.
    SoaField::<6>::from_fn(1, |_| [1.0, 0.0, speed, 0.0, 1.0e5, 0.0])
  }

  #[test]
  fn windward_slope_lifts_lee_slope_subsides() {
    // Eastward wind, ground rising to the east (∂h/∂e > 0) ⇒ uplift (+x vr).
    let mut up = eastward_wind_state(10.0);
    apply_orographic_lift(&mut up, &[equator_site([0.5, 0.0])], 0.5, 50.0);
    let vr_up = up.state(CellId::from(0))[1]; // ρ=1 ⇒ momentum_x == vr
    assert!(vr_up > 0.0, "windward uplift expected, got vr {vr_up}");

    // Same wind, ground falling to the east ⇒ subsidence (−x vr).
    let mut down = eastward_wind_state(10.0);
    apply_orographic_lift(&mut down, &[equator_site([-0.5, 0.0])], 0.5, 50.0);
    let vr_down = down.state(CellId::from(0))[1];
    assert!(vr_down < 0.0, "lee subsidence expected, got vr {vr_down}");

    // Magnitude is symmetric for symmetric slopes.
    assert!((vr_up + vr_down).abs() < 1e-9);
  }

  #[test]
  fn relaxation_is_bounded_and_converges_to_target() {
    // Iterating the forcing must stay finite and converge vr → w (= u·∇h,
    // clamped), never overshoot unboundedly — the stability guard.
    let site = equator_site([0.5, 0.0]);
    let mut state = eastward_wind_state(10.0);
    let target_w = (10.0_f64 * 0.5).min(50.0); // u_e·∂h/∂e = 5 m/s
    for _ in 0..200 {
      apply_orographic_lift(&mut state, &[site], 0.3, 50.0);
      let s = state.state(CellId::from(0));
      assert!(s.iter().all(|v| v.is_finite()), "state went non-finite");
    }
    let vr = state.state(CellId::from(0))[1];
    assert!(
      (vr - target_w).abs() < 1e-6,
      "vr {vr} did not converge to {target_w}"
    );
  }

  #[test]
  fn clamp_limits_induced_velocity() {
    // A huge slope must not produce an unbounded w: clamp caps it.
    let mut state = eastward_wind_state(100.0);
    apply_orographic_lift(&mut state, &[equator_site([10.0, 0.0])], 1.0, 2.0);
    let vr = state.state(CellId::from(0))[1];
    assert!(vr <= 2.0 + 1e-9, "vr {vr} exceeded clamp");
  }
}
