// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Backend-agnostic frame interpolation for smooth playback.
//!
//! A simulation may advance in large, irregular time steps (e.g. an implicit
//! solver taking minute-scale steps), while a renderer wants to draw smooth
//! motion at its own, much higher frame rate. This module decouples the two: it
//! buffers the two most recent frames of per-layer field samples and serves
//! them linearly interpolated at a **render clock** that advances through
//! simulation time at the rate the simulation is actually delivering frames.
//!
//! The clock never extrapolates past the newest frame (so it can only show
//! state the simulation has produced) — when the simulation falls behind, the
//! clock simply holds, dilating time rather than stuttering; when frames arrive
//! faster, the buffer window advances and playback catches up. This is the
//! sim↔render load-balancing knob.
//!
//! It is intentionally engine-neutral: it works on IR [`LayerHandle`]s, plain
//! `f64` samples, and real-time deltas, so any backend can drive it. Engine
//! specifics (clocks, mesh updates) belong in that backend.

use std::collections::HashMap;

use crate::ir::LayerHandle;

/// EMA weight for the delivered-sim-rate estimate (per frame push).
const RATE_SMOOTHING: f64 = 0.3;

/// A per-cell quantity channel carried in the snapshot for the semantic query
/// API (distinct from render [`LayerHandle`]s). Vector quantities such as wind
/// are stored as separate world-frame component channels and recombined at
/// query time; the mesh is part of the key so multi-mesh worlds stay
/// unambiguous.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MeshChannel {
  AtmosphereTemperature,
  AtmospherePressure,
  AtmosphereHumidity,
  /// World-frame air-velocity components (m/s).
  AtmosphereWindX,
  AtmosphereWindY,
  AtmosphereWindZ,
  /// Sea-surface temperature on the ocean mesh (K).
  SeaSurfaceTemperature,
  /// Static surface elevation (m) relative to the mean surface radius.
  SurfaceElevation,
  /// Static categorical surface-type code (see `utility::domain::SurfaceClass`).
  SurfaceType,
}

/// One completed frame's interpolatable field data: a simulation timestamp,
/// the per-cell scalar samples of each active render layer, and the per-cell
/// quantity channels the semantic query API reads.
#[derive(Clone, Debug, Default)]
pub struct SampleFrame {
  pub sim_time: f64,
  pub layers: HashMap<LayerHandle, Vec<f64>>,
  pub quantities: HashMap<MeshChannel, Vec<f64>>,
}

impl SampleFrame {
  pub fn new(sim_time: f64) -> Self {
    Self {
      sim_time,
      layers: HashMap::new(),
      quantities: HashMap::new(),
    }
  }

  pub fn insert(&mut self, layer: LayerHandle, samples: Vec<f64>) {
    self.layers.insert(layer, samples);
  }

  /// Attach a per-cell quantity channel used by the semantic query API.
  pub fn insert_quantity(&mut self, channel: MeshChannel, samples: Vec<f64>) {
    self.quantities.insert(channel, samples);
  }
}

/// Buffers the two most recent [`SampleFrame`]s and serves per-layer samples
/// interpolated at an adaptive render clock.
#[derive(Debug, Default)]
pub struct FrameInterpolator {
  prev: Option<SampleFrame>,
  next: Option<SampleFrame>,
  /// Playback position in simulation seconds.
  clock: f64,
  /// Estimated simulation seconds delivered per real second.
  sim_rate: f64,
  /// Real seconds accumulated since the last frame push (rate estimation).
  real_since_push: f64,
  has_rate: bool,
}

impl FrameInterpolator {
  pub fn new() -> Self {
    Self::default()
  }

  /// Ingest a freshly-completed frame, shifting the window forward and
  /// re-estimating the delivered sim-rate from how much sim-time advanced over
  /// the real time since the previous push.
  pub fn push(&mut self, frame: SampleFrame) {
    match self.next.take() {
      None => {
        self.clock = frame.sim_time;
        self.next = Some(frame);
      }
      Some(old_next) => {
        let d_sim = frame.sim_time - old_next.sim_time;
        if self.real_since_push > 1e-9 && d_sim.is_finite() {
          let observed = d_sim / self.real_since_push;
          self.sim_rate = if self.has_rate {
            self.sim_rate + RATE_SMOOTHING * (observed - self.sim_rate)
          } else {
            observed
          };
          self.has_rate = true;
        }
        self.prev = Some(old_next);
        self.next = Some(frame);
      }
    }
    self.real_since_push = 0.0;
    self.clamp_clock();
  }

  /// Advance the render clock by `real_dt` real seconds. Until a delivery rate
  /// is known the clock simply tracks the newest frame (snap).
  pub fn advance(&mut self, real_dt: f64) {
    if real_dt > 0.0 {
      self.real_since_push += real_dt;
      if self.has_rate {
        self.clock += real_dt * self.sim_rate;
      }
    }
    self.clamp_clock();
  }

  fn window(&self) -> Option<(f64, f64)> {
    match (&self.prev, &self.next) {
      (Some(p), Some(n)) => Some((p.sim_time, n.sim_time)),
      (None, Some(n)) => Some((n.sim_time, n.sim_time)),
      _ => None,
    }
  }

  fn clamp_clock(&mut self) {
    if let Some((lo, hi)) = self.window() {
      self.clock = self.clock.clamp(lo.min(hi), hi.max(lo));
      if !self.has_rate {
        self.clock = hi;
      }
    }
  }

  /// Interpolation fraction in `[0, 1]` between the buffered frames.
  pub fn alpha(&self) -> f64 {
    match self.window() {
      Some((lo, hi)) if (hi - lo).abs() > 1e-12 => {
        ((self.clock - lo) / (hi - lo)).clamp(0.0, 1.0)
      }
      _ => 1.0,
    }
  }

  /// The render-clock position, in simulation seconds.
  pub fn clock(&self) -> f64 {
    self.clock
  }

  /// Estimated delivered simulation seconds per real second (0 until known).
  pub fn sim_rate(&self) -> f64 {
    if self.has_rate { self.sim_rate } else { 0.0 }
  }

  /// Whether at least one frame has been buffered.
  pub fn is_primed(&self) -> bool {
    self.next.is_some()
  }

  /// Per-cell samples for a layer, linearly interpolated at the current clock.
  /// Falls back to the newest frame's values when there is no previous frame,
  /// the layer is new, or the sample shapes differ.
  pub fn samples(&self, layer: LayerHandle) -> Option<Vec<f64>> {
    let next = self.next.as_ref()?.layers.get(&layer)?;
    let alpha = self.alpha();
    match self.prev.as_ref().and_then(|p| p.layers.get(&layer)) {
      Some(prev) if prev.len() == next.len() && alpha < 1.0 => Some(
        prev
          .iter()
          .zip(next)
          .map(|(a, b)| a + (b - a) * alpha)
          .collect(),
      ),
      _ => Some(next.clone()),
    }
  }

  /// Per-cell values for a quantity channel, linearly interpolated at the
  /// current clock. Mirrors [`Self::samples`] but for the query-API channels.
  pub fn quantity(&self, channel: MeshChannel) -> Option<Vec<f64>> {
    let next = self.next.as_ref()?.quantities.get(&channel)?;
    let alpha = self.alpha();
    match self.prev.as_ref().and_then(|p| p.quantities.get(&channel)) {
      Some(prev) if prev.len() == next.len() && alpha < 1.0 => Some(
        prev
          .iter()
          .zip(next)
          .map(|(a, b)| a + (b - a) * alpha)
          .collect(),
      ),
      _ => Some(next.clone()),
    }
  }

  /// Whether the interpolator is genuinely blending two distinct frames (as
  /// opposed to snapping to a single newest frame). Query results served while
  /// this is false are flagged [`crate::query::Sample::Stale`].
  pub fn is_interpolating(&self) -> bool {
    match (&self.prev, &self.next) {
      (Some(p), Some(n)) => (n.sim_time - p.sim_time).abs() > 1e-12,
      _ => false,
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn frame(sim_time: f64, layer: LayerHandle, vals: &[f64]) -> SampleFrame {
    let mut f = SampleFrame::new(sim_time);
    f.insert(layer, vals.to_vec());
    f
  }

  #[test]
  fn snaps_to_first_frame_before_rate_known() {
    let l = LayerHandle(7);
    let mut fi = FrameInterpolator::new();
    fi.push(frame(0.0, l, &[1.0, 2.0]));
    assert_eq!(fi.samples(l), Some(vec![1.0, 2.0]));
    assert_eq!(fi.alpha(), 1.0);
  }

  #[test]
  fn interpolates_between_two_frames() {
    let l = LayerHandle(7);
    let mut fi = FrameInterpolator::new();
    fi.push(frame(0.0, l, &[0.0, 10.0]));
    // 1 real-second elapses, then a frame at sim_time=10 arrives ⇒ rate = 10.
    fi.advance(1.0);
    fi.push(frame(10.0, l, &[10.0, 20.0]));

    assert!((fi.alpha() - 0.0).abs() < 1e-9);
    assert_eq!(fi.samples(l), Some(vec![0.0, 10.0]));

    // Half a real second ⇒ 5 sim-seconds ⇒ alpha 0.5.
    fi.advance(0.5);
    assert!((fi.alpha() - 0.5).abs() < 1e-9, "alpha {}", fi.alpha());
    assert_eq!(fi.samples(l), Some(vec![5.0, 15.0]));

    // Past the window: clamps (no extrapolation).
    fi.advance(10.0);
    assert!((fi.alpha() - 1.0).abs() < 1e-9);
    assert_eq!(fi.samples(l), Some(vec![10.0, 20.0]));
  }

  #[test]
  fn handles_stall_then_burst() {
    // A backend that falls back to an explicit burst delivers frames slowly,
    // then a rapid catch-up flurry. The interpolator must never extrapolate
    // past the newest frame (so it only shows produced state) and must always
    // serve a valid, finite sample — through the stall (clock dilates/holds)
    // and the burst (clock catches up by advancing the window).
    let l = LayerHandle(3);
    let mut fi = FrameInterpolator::new();
    fi.push(frame(0.0, l, &[0.0]));
    fi.advance(1.0);
    fi.push(frame(10.0, l, &[10.0])); // establishes rate

    let mut newest = 10.0_f64;
    let mut max_clock_overshoot = 0.0_f64;
    let mut min_alpha = f64::INFINITY;
    let mut max_alpha = f64::NEG_INFINITY;
    let mut check = |fi: &FrameInterpolator, newest: f64| {
      // Never shows the future.
      max_clock_overshoot = max_clock_overshoot.max(fi.clock() - newest);
      let a = fi.alpha();
      min_alpha = min_alpha.min(a);
      max_alpha = max_alpha.max(a);
      assert!(
        fi.samples(l).unwrap().iter().all(|v| v.is_finite()),
        "non-finite sample"
      );
    };

    // --- Stall: render keeps ticking, no new frames for a long while. ---
    for _ in 0..50 {
      fi.advance(1.0);
      check(&fi, newest);
    }

    // --- Burst: a flurry of frames arrive with almost no render time between
    // them (the explicit catch-up). ---
    for _ in 0..8 {
      newest += 10.0;
      fi.push(frame(newest, l, &[newest]));
      fi.advance(0.05);
      check(&fi, newest);
    }

    // --- Settle back to a steady cadence. ---
    for _ in 0..20 {
      fi.advance(0.2);
      check(&fi, newest);
    }

    assert!(
      max_clock_overshoot < 1e-9,
      "clock extrapolated past newest frame by {max_clock_overshoot}"
    );
    assert!(
      min_alpha >= -1e-9 && max_alpha <= 1.0 + 1e-9,
      "alpha left [0,1]: [{min_alpha}, {max_alpha}]"
    );
    // After the burst + settle, playback has caught up near the newest frame.
    assert!(
      fi.clock() >= newest - 10.0 - 1e-9,
      "clock {} did not catch up toward newest {newest}",
      fi.clock()
    );
  }

  #[test]
  fn clock_holds_when_sim_stalls() {
    let l = LayerHandle(7);
    let mut fi = FrameInterpolator::new();
    fi.push(frame(0.0, l, &[0.0]));
    fi.advance(1.0);
    fi.push(frame(10.0, l, &[10.0]));
    for _ in 0..100 {
      fi.advance(1.0);
    }
    assert!((fi.clock() - 10.0).abs() < 1e-9);
    assert_eq!(fi.samples(l), Some(vec![10.0]));
  }

  /// Quantify smoothness: drive the interpolator on a realistic schedule (a
  /// coarse sim delivering frames slowly, a fast 60 Hz renderer) and measure
  /// the worst per-render-frame change in the painted field, plus tracking
  /// error against a known smooth ground-truth field. Interpolation should cut
  /// the worst visual jump by ~the render/sim frame ratio and track the true
  /// field far more closely than snapping to the latest frame.
  #[test]
  fn quantifies_smoothness_vs_snap() {
    const CELLS: usize = 4;
    let layer = LayerHandle(1);
    let render_dt = 1.0 / 60.0; // 60 fps
    let renders_per_sim = 12; // sim delivers a frame every 12 render frames
    let sim_dt = 10.0; // sim-seconds per sim frame
    let sim_frames = 10;

    // A smooth ground-truth field that evolves *slowly* relative to the sim
    // step — the realistic regime, since large steps are taken precisely
    // because the physics is slow. It still curves (so linear interp has a
    // real, measurable residual), just gently over a step.
    let truth = |t: f64| -> Vec<f64> {
      (0..CELLS)
        .map(|i| (0.03 * t + i as f64 * 0.7).sin())
        .collect()
    };
    let max_abs_diff = |a: &[f64], b: &[f64]| -> f64 {
      a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
    };

    let mut fi = FrameInterpolator::new();
    let mut sim_t = 0.0;
    fi.push(frame(sim_t, layer, &truth(sim_t)));
    let mut latest = truth(sim_t); // what snap-to-latest would show

    let mut prev_interp: Option<Vec<f64>> = None;
    let mut prev_snap: Option<Vec<f64>> = None;
    let (mut peak_interp, mut peak_snap) = (0.0_f64, 0.0_f64);
    let (mut err_interp, mut err_snap) = (0.0_f64, 0.0_f64);
    let mut warmup = 2_u32; // skip the first window (rate not yet known)

    for _ in 1..sim_frames {
      for _ in 0..renders_per_sim {
        fi.advance(render_dt);
        let clk = fi.clock();
        let interp = fi.samples(layer).unwrap();
        let ideal = truth(clk); // what *should* be on screen at the clock

        if warmup == 0 {
          if let Some(p) = &prev_interp {
            peak_interp = peak_interp.max(max_abs_diff(&interp, p));
          }
          if let Some(p) = &prev_snap {
            peak_snap = peak_snap.max(max_abs_diff(&latest, p));
          }
          err_interp = err_interp.max(max_abs_diff(&interp, &ideal));
          err_snap = err_snap.max(max_abs_diff(&latest, &ideal));
        }
        prev_interp = Some(interp);
        prev_snap = Some(latest.clone());
      }
      sim_t += sim_dt;
      fi.push(frame(sim_t, layer, &truth(sim_t)));
      latest = truth(sim_t);
      warmup = warmup.saturating_sub(1);
    }

    eprintln!(
      "peak per-frame jump: interp {peak_interp:.4} vs snap {peak_snap:.4} \
       ({:.1}x smoother); max tracking error: interp {err_interp:.4} vs \
       snap {err_snap:.4} ({:.1}x closer)",
      peak_snap / peak_interp,
      err_snap / err_interp,
    );

    // Worst per-frame jump cut by ~the render/sim ratio (≈12×); accept ≥6×.
    assert!(
      peak_interp * 6.0 < peak_snap,
      "interp peak {peak_interp} not << snap peak {peak_snap}"
    );
    // And it tracks the true field far better than a one-frame-stale snap.
    assert!(
      err_interp * 3.0 < err_snap,
      "interp error {err_interp} not << snap error {err_snap}"
    );
  }
}
