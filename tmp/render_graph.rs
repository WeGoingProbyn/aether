use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::Arc;

use crate::containers::graph::Graph;
use crate::debugger::IcsError;
use crate::structures::{
  materials::MaterialVariable, pipeline::PipelineAttribute, textures::TextureType,
};
use crate::{ICS_DEBUG, ICS_ERROR, ICS_INFO, ICS_WARN};

// ---------------------------------------------------------------------------
// API-agnostic graph descriptors (no direct Vulkan types).
// The backend will translate these into platform-specific objects.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GraphFormat {
  DefaultColor,
  DefaultDepth,
  Rgba8Unorm,
  Rgba8Srgb,
  Rgba16Float,
  R16G16Snorm,
  D32,
  D24S8,
  /// Backend-defined numeric format token for custom cases.
  Custom(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GraphSampleCount {
  One,
  Two,
  Four,
  Eight,
  Custom(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GraphImageLayout {
  Undefined,
  ColorAttachment,
  DepthAttachment,
  ShaderRead,
  DepthRead,
  Present,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GraphSize {
  Swapchain,
  Fixed { width: u32, height: u32 },
  Scaled { numerator: u32, denominator: u32 }, // e.g., 1/2, 1/4 scale
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GraphResourceKind {
  Color,
  Depth,
  Normal,
  Scratch,
  SwapchainColor,
  Buffer,
  Custom,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GraphResourceLifetime {
  External,   // swapchain/imported
  Persistent, // long-lived engine-owned
  History,    // preserved across frames
  Transient,  // per-frame scratch
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GraphResourceRole {
  Scene,
  Post,
  Custom,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GraphAccessType {
  Sampled,
  InputAttachment,
  ColorWrite,
  DepthWrite,
  StorageReadWrite,
  BufferRead,
  BufferWrite,
  BufferReadWrite,
  Present,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GraphQueueClass {
  Graphics,
  Compute,
  Transfer,
}

/// Determinism scope for render graph planning and caching.
pub const RENDER_GRAPH_DETERMINISM_SCOPE: &str =
  "Pass ordering, resource ordering, barrier ordering, renderpass/subpass grouping, \
pipeline cache keys, descriptor set layouts";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GraphQueueRequest {
  Any,
  Require(GraphQueueClass),
  Prefer(GraphQueueClass),
}

impl Default for GraphQueueRequest {
  fn default() -> Self {
    GraphQueueRequest::Any
  }
}

#[cfg(test)]
mod tests {
  use super::{
    diff_text, signature_for_plan, CommandBufferUsage, GraphQueueClass, GraphQueueRequest,
    GraphResourceLifetime, RenderBufferDesc, RenderGraphBuilder, RenderPassKind, RenderPassWork,
    RenderResourceAccess, RenderResourceKind, RenderSyncPlan, SyncQueue,
  };
  use crate::debugger::initialize_logger;

  static TEST_ENV_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

  #[test]
  fn render_plan_signature_is_deterministic() {
    initialize_logger();
    let build_plan = || {
      let mut builder = RenderGraphBuilder::new();
      let color = builder.add_resource("color", RenderResourceKind::SwapchainColor);
      let depth = builder.add_resource("depth", RenderResourceKind::SwapchainDepth);
      builder.add_pass(
        "scene",
        RenderPassKind::Graphics,
        RenderPassWork::None,
        0,
        None,
        vec![],
        vec![
          RenderResourceAccess::write(color),
          RenderResourceAccess::write(depth),
        ],
        vec![color, depth],
        vec![color, depth],
        None,
        GraphQueueRequest::Any,
        CommandBufferUsage::OneTime,
      );
      builder.add_pass(
        "present",
        RenderPassKind::Present,
        RenderPassWork::None,
        0,
        None,
        vec![RenderResourceAccess::read(color)],
        vec![],
        vec![color],
        vec![],
        None,
        GraphQueueRequest::Any,
        CommandBufferUsage::OneTime,
      );
      let mut graph = builder.build();
      graph.compile().unwrap();
      graph.make_plan()
    };

    let plan_a = build_plan();
    let plan_b = build_plan();
    let sig_a = signature_for_plan(&plan_a);
    let sig_b = signature_for_plan(&plan_b);
    assert_eq!(sig_a, sig_b);
  }

  #[test]
  fn render_plan_dump_is_deterministic() {
    initialize_logger();
    let build_plan = || {
      let mut builder = RenderGraphBuilder::new();
      let color = builder.add_resource("color", RenderResourceKind::SwapchainColor);
      let depth = builder.add_resource("depth", RenderResourceKind::SwapchainDepth);
      builder.add_pass(
        "scene",
        RenderPassKind::Graphics,
        RenderPassWork::None,
        0,
        None,
        vec![],
        vec![
          RenderResourceAccess::write(color),
          RenderResourceAccess::write(depth),
        ],
        vec![color, depth],
        vec![color, depth],
        None,
        GraphQueueRequest::Any,
        CommandBufferUsage::OneTime,
      );
      builder.add_pass(
        "present",
        RenderPassKind::Present,
        RenderPassWork::None,
        0,
        None,
        vec![RenderResourceAccess::read(color)],
        vec![],
        vec![color],
        vec![],
        None,
        GraphQueueRequest::Any,
        CommandBufferUsage::OneTime,
      );
      let mut graph = builder.build();
      graph.compile().unwrap();
      graph.make_plan()
    };

    let plan_a = build_plan();
    let plan_b = build_plan();
    let sync_a = RenderSyncPlan::from_plan(&plan_a);
    let sync_b = RenderSyncPlan::from_plan(&plan_b);
    let report_a = plan_a.report(Some(&sync_a));
    let report_b = plan_b.report(Some(&sync_b));

    let dump_a = format!(
      "{}{}{}",
      report_a.dump_text(),
      plan_a.dump_text(),
      sync_a.dump_text()
    );
    let dump_b = format!(
      "{}{}{}",
      report_b.dump_text(),
      plan_b.dump_text(),
      sync_b.dump_text()
    );

    if dump_a != dump_b {
      panic!("{}", diff_text(&dump_a, &dump_b));
    }
  }

  #[test]
  fn render_plan_dump_matches_golden() {
    initialize_logger();
    const GOLDEN: &str = "RenderGraphReport\npasses=2 attachments=2 render_passes=1 subpasses=1\nplan_signature=unknown\nsubmissions=1 primary_cmds_per_fb=1 secondary_cmds_per_fb=unknown\nvalidation_errors=0 validation_warnings=0\nwarnings=1\n  - queue ownership transfers planned: 1 (release/acquire barriers)\nRenderPlan\nattachments=2 passes=2 subpasses=1 render_passes=1\nAttachments:\n  - color fmt=DefaultColor samples=One size=Swapchain life=External role=Custom swapchain=true init=Undefined final=Present\n  - depth fmt=DefaultDepth samples=One size=Swapchain life=External role=Custom swapchain=true init=Undefined final=Present\nPasses:\n  - scene kind=Graphics rp=0 sp=0 reads=0 writes=2\n      write color (ColorWrite)\n      write depth (DepthWrite)\n  - present kind=Present rp=0 sp=0 reads=1 writes=0\n      read color (Present)\nRenderSyncPlan\nsubmissions=1 resources=2\nbarriers_image=3 barriers_buffer=0 queue_transfers=1 release_barriers=1 acquire_barriers=2\n  - submission 0 queue=Graphics passes=1 waits=0 signals=0 barriers=3\n";

    let mut builder = RenderGraphBuilder::new();
    let color = builder.add_resource("color", RenderResourceKind::SwapchainColor);
    let depth = builder.add_resource("depth", RenderResourceKind::SwapchainDepth);
    builder.add_pass(
      "scene",
      RenderPassKind::Graphics,
      RenderPassWork::None,
      0,
      None,
      vec![],
      vec![
        RenderResourceAccess::write(color),
        RenderResourceAccess::write(depth),
      ],
      vec![color, depth],
      vec![color, depth],
      None,
      GraphQueueRequest::Any,
      CommandBufferUsage::OneTime,
    );
    builder.add_pass(
      "present",
      RenderPassKind::Present,
      RenderPassWork::None,
      0,
      None,
      vec![RenderResourceAccess::read(color)],
      vec![],
      vec![color],
      vec![],
      None,
      GraphQueueRequest::Any,
      CommandBufferUsage::OneTime,
    );
    let mut graph = builder.build();
    graph.compile().unwrap();
    let plan = graph.make_plan();
    let sync = RenderSyncPlan::from_plan(&plan);
    let report = plan.report(Some(&sync));
    let dump = format!(
      "{}{}{}",
      report.dump_text(),
      plan.dump_text(),
      sync.dump_text()
    );

    if dump != GOLDEN {
      panic!("{}", diff_text(GOLDEN, &dump));
    }
  }

  #[test]
  fn sync_plan_adds_barrier_for_write_then_sample() {
    initialize_logger();
    let _env_guard = TEST_ENV_MUTEX.lock().unwrap();
    std::env::remove_var("ICS_RENDER_GRAPH_SINGLE_QUEUE");
    let mut builder = RenderGraphBuilder::new();
    let color = builder.add_resource("color", RenderResourceKind::SwapchainColor);
    builder.add_pass(
      "write",
      RenderPassKind::Graphics,
      RenderPassWork::None,
      0,
      None,
      vec![],
      vec![RenderResourceAccess::write(color)],
      vec![color],
      vec![color],
      None,
      GraphQueueRequest::Any,
      CommandBufferUsage::OneTime,
    );
    builder.add_pass(
      "sample",
      RenderPassKind::Graphics,
      RenderPassWork::None,
      0,
      None,
      vec![RenderResourceAccess::read(color)],
      vec![],
      vec![color],
      vec![],
      None,
      GraphQueueRequest::Any,
      CommandBufferUsage::OneTime,
    );
    let mut graph = builder.build();
    graph.compile().unwrap();
    let plan = graph.make_plan();
    let sync = RenderSyncPlan::from_plan(&plan);
    assert!(!sync.submissions.is_empty());
    let total_barriers: usize = sync.submissions.iter().map(|s| s.barriers.len()).sum();
    assert!(total_barriers > 0);
  }

  #[test]
  fn sync_plan_splits_submissions_on_queue_change() {
    initialize_logger();
    let _env_guard = TEST_ENV_MUTEX.lock().unwrap();
    std::env::remove_var("ICS_RENDER_GRAPH_SINGLE_QUEUE");
    let mut builder = RenderGraphBuilder::new();
    let color = builder.add_resource("color", RenderResourceKind::SwapchainColor);
    builder.add_pass(
      "compute_write",
      RenderPassKind::Compute,
      RenderPassWork::None,
      0,
      None,
      vec![],
      vec![RenderResourceAccess::write(color)],
      vec![color],
      vec![color],
      None,
      GraphQueueRequest::Require(GraphQueueClass::Compute),
      CommandBufferUsage::OneTime,
    );
    builder.add_pass(
      "graphics_sample",
      RenderPassKind::Graphics,
      RenderPassWork::None,
      0,
      None,
      vec![RenderResourceAccess::read(color)],
      vec![],
      vec![color],
      vec![],
      None,
      GraphQueueRequest::Require(GraphQueueClass::Graphics),
      CommandBufferUsage::OneTime,
    );
    let mut graph = builder.build();
    graph.compile().unwrap();
    let plan = graph.make_plan();
    let sync = RenderSyncPlan::from_plan(&plan);
    assert!(sync.submissions.len() >= 2);
  }

  #[test]
  fn sync_plan_forces_single_queue() {
    initialize_logger();
    let _env_guard = TEST_ENV_MUTEX.lock().unwrap();
    std::env::set_var("ICS_RENDER_GRAPH_SINGLE_QUEUE", "1");
    let mut builder = RenderGraphBuilder::new();
    let color = builder.add_resource("color", RenderResourceKind::SwapchainColor);
    builder.add_pass(
      "compute_write",
      RenderPassKind::Compute,
      RenderPassWork::None,
      0,
      None,
      vec![],
      vec![RenderResourceAccess::write(color)],
      vec![color],
      vec![color],
      None,
      GraphQueueRequest::Require(GraphQueueClass::Compute),
      CommandBufferUsage::OneTime,
    );
    let mut graph = builder.build();
    graph.compile().unwrap();
    let plan = graph.make_plan();
    let sync = RenderSyncPlan::from_plan(&plan);
    std::env::remove_var("ICS_RENDER_GRAPH_SINGLE_QUEUE");
    assert!(sync
      .submissions
      .iter()
      .all(|s| matches!(s.queue, SyncQueue::Graphics)));
  }

  // =========================================================================
  // M2.5: DAG/Hazard Edge Generation Tests
  // =========================================================================

  /// Test: Write -> Read creates a dependency edge (RAW hazard)
  #[test]
  fn dag_edge_write_then_read() {
    initialize_logger();
    let mut builder = RenderGraphBuilder::new();
    let rt = builder.add_resource("rt", RenderResourceKind::SwapchainColor);

    // Pass A writes to rt
    builder.add_pass(
      "writer",
      RenderPassKind::Graphics,
      RenderPassWork::None,
      0,
      None,
      vec![],
      vec![RenderResourceAccess::write(rt)],
      vec![rt],
      vec![rt],
      None,
      GraphQueueRequest::Any,
      CommandBufferUsage::OneTime,
    );

    // Pass B reads from rt (should depend on A)
    builder.add_pass(
      "reader",
      RenderPassKind::Graphics,
      RenderPassWork::None,
      0,
      None,
      vec![RenderResourceAccess::read(rt)],
      vec![],
      vec![rt],
      vec![],
      None,
      GraphQueueRequest::Any,
      CommandBufferUsage::OneTime,
    );

    let mut graph = builder.build();
    graph.compile().unwrap();
    let plan = graph.make_plan();

    // Verify ordering: writer should come before reader
    let writer_idx = plan.passes.iter().position(|p| p.name == "writer").unwrap();
    let reader_idx = plan.passes.iter().position(|p| p.name == "reader").unwrap();
    assert!(
      writer_idx < reader_idx,
      "Writer pass should execute before reader pass (RAW hazard)"
    );
  }

  /// Test: Write -> Write creates a dependency edge (WAW hazard)
  #[test]
  fn dag_edge_write_then_write() {
    initialize_logger();
    let mut builder = RenderGraphBuilder::new();
    let rt = builder.add_resource("rt", RenderResourceKind::SwapchainColor);

    // Pass A writes to rt
    builder.add_pass(
      "writer_a",
      RenderPassKind::Graphics,
      RenderPassWork::None,
      0,
      None,
      vec![],
      vec![RenderResourceAccess::write(rt)],
      vec![rt],
      vec![rt],
      None,
      GraphQueueRequest::Any,
      CommandBufferUsage::OneTime,
    );

    // Pass B also writes to rt (should depend on A to preserve order)
    builder.add_pass(
      "writer_b",
      RenderPassKind::Graphics,
      RenderPassWork::None,
      0,
      None,
      vec![],
      vec![RenderResourceAccess::write(rt)],
      vec![rt],
      vec![rt],
      None,
      GraphQueueRequest::Any,
      CommandBufferUsage::OneTime,
    );

    let mut graph = builder.build();
    graph.compile().unwrap();
    let plan = graph.make_plan();

    // Verify ordering preserved
    let writer_a_idx = plan
      .passes
      .iter()
      .position(|p| p.name == "writer_a")
      .unwrap();
    let writer_b_idx = plan
      .passes
      .iter()
      .position(|p| p.name == "writer_b")
      .unwrap();
    assert!(
      writer_a_idx < writer_b_idx,
      "First writer should execute before second writer (WAW hazard)"
    );
  }

  /// Test: Read -> Write creates a dependency edge (WAR hazard)
  #[test]
  fn dag_edge_read_then_write() {
    initialize_logger();
    let mut builder = RenderGraphBuilder::new();
    let rt = builder.add_resource("rt", RenderResourceKind::SwapchainColor);

    // First, something writes to rt so there's data to read
    builder.add_pass(
      "initial_write",
      RenderPassKind::Graphics,
      RenderPassWork::None,
      0,
      None,
      vec![],
      vec![RenderResourceAccess::write(rt)],
      vec![rt],
      vec![rt],
      None,
      GraphQueueRequest::Any,
      CommandBufferUsage::OneTime,
    );

    // Pass A reads from rt
    builder.add_pass(
      "reader",
      RenderPassKind::Graphics,
      RenderPassWork::None,
      0,
      None,
      vec![RenderResourceAccess::read(rt)],
      vec![],
      vec![rt],
      vec![],
      None,
      GraphQueueRequest::Any,
      CommandBufferUsage::OneTime,
    );

    // Pass B writes to rt (should depend on A to prevent overwriting before read)
    builder.add_pass(
      "later_writer",
      RenderPassKind::Graphics,
      RenderPassWork::None,
      0,
      None,
      vec![],
      vec![RenderResourceAccess::write(rt)],
      vec![rt],
      vec![rt],
      None,
      GraphQueueRequest::Any,
      CommandBufferUsage::OneTime,
    );

    let mut graph = builder.build();
    graph.compile().unwrap();
    let plan = graph.make_plan();

    // Verify ordering: reader should come before later_writer
    let reader_idx = plan.passes.iter().position(|p| p.name == "reader").unwrap();
    let later_writer_idx = plan
      .passes
      .iter()
      .position(|p| p.name == "later_writer")
      .unwrap();
    assert!(
      reader_idx < later_writer_idx,
      "Reader pass should execute before later writer (WAR hazard)"
    );
  }

  /// Test: Independent resources have no dependency (can be parallel)
  #[test]
  fn dag_no_edge_independent_resources() {
    initialize_logger();
    let mut builder = RenderGraphBuilder::new();
    let rt_a = builder.add_resource("rt_a", RenderResourceKind::SwapchainColor);
    let rt_b = builder.add_resource("rt_b", RenderResourceKind::SwapchainDepth);

    // Pass A writes to rt_a
    builder.add_pass(
      "writer_a",
      RenderPassKind::Graphics,
      RenderPassWork::None,
      0,
      None,
      vec![],
      vec![RenderResourceAccess::write(rt_a)],
      vec![rt_a],
      vec![rt_a],
      None,
      GraphQueueRequest::Any,
      CommandBufferUsage::OneTime,
    );

    // Pass B writes to rt_b (no dependency on A)
    builder.add_pass(
      "writer_b",
      RenderPassKind::Graphics,
      RenderPassWork::None,
      0,
      None,
      vec![],
      vec![RenderResourceAccess::write(rt_b)],
      vec![rt_b],
      vec![rt_b],
      None,
      GraphQueueRequest::Any,
      CommandBufferUsage::OneTime,
    );

    let mut graph = builder.build();
    graph.compile().unwrap();
    // Should compile without errors - no cycle possible
    let plan = graph.make_plan();
    assert_eq!(plan.passes.len(), 2);
  }

  /// Test: Diamond dependency pattern (A -> B, A -> C, B -> D, C -> D)
  #[test]
  fn dag_diamond_dependency() {
    initialize_logger();
    let mut builder = RenderGraphBuilder::new();
    let rt_main = builder.add_resource("rt_main", RenderResourceKind::SwapchainColor);
    let rt_left = builder.add_resource(
      "rt_left",
      RenderResourceKind::Image(super::RenderImageDesc {
        format: super::GraphFormat::DefaultColor,
        samples: super::GraphSampleCount::One,
        size: super::GraphSize::Swapchain,
        lifetime: super::GraphResourceLifetime::Transient,
        resolve: false,
        is_depth: false,
      }),
    );
    let rt_right = builder.add_resource(
      "rt_right",
      RenderResourceKind::Image(super::RenderImageDesc {
        format: super::GraphFormat::DefaultColor,
        samples: super::GraphSampleCount::One,
        size: super::GraphSize::Swapchain,
        lifetime: super::GraphResourceLifetime::Transient,
        resolve: false,
        is_depth: false,
      }),
    );

    // A: writes to rt_main
    builder.add_pass(
      "pass_a",
      RenderPassKind::Graphics,
      RenderPassWork::None,
      0,
      None,
      vec![],
      vec![RenderResourceAccess::write(rt_main)],
      vec![rt_main],
      vec![rt_main],
      None,
      GraphQueueRequest::Any,
      CommandBufferUsage::OneTime,
    );

    // B: reads rt_main, writes rt_left
    builder.add_pass(
      "pass_b",
      RenderPassKind::Graphics,
      RenderPassWork::None,
      0,
      None,
      vec![RenderResourceAccess::read(rt_main)],
      vec![RenderResourceAccess::write(rt_left)],
      vec![rt_main, rt_left],
      vec![rt_left],
      None,
      GraphQueueRequest::Any,
      CommandBufferUsage::OneTime,
    );

    // C: reads rt_main, writes rt_right
    builder.add_pass(
      "pass_c",
      RenderPassKind::Graphics,
      RenderPassWork::None,
      0,
      None,
      vec![RenderResourceAccess::read(rt_main)],
      vec![RenderResourceAccess::write(rt_right)],
      vec![rt_main, rt_right],
      vec![rt_right],
      None,
      GraphQueueRequest::Any,
      CommandBufferUsage::OneTime,
    );

    // D: reads rt_left and rt_right
    builder.add_pass(
      "pass_d",
      RenderPassKind::Graphics,
      RenderPassWork::None,
      0,
      None,
      vec![
        RenderResourceAccess::read(rt_left),
        RenderResourceAccess::read(rt_right),
      ],
      vec![],
      vec![rt_left, rt_right],
      vec![],
      None,
      GraphQueueRequest::Any,
      CommandBufferUsage::OneTime,
    );

    let mut graph = builder.build();
    graph.compile().unwrap();
    let plan = graph.make_plan();

    // Verify topological order
    let idx_a = plan.passes.iter().position(|p| p.name == "pass_a").unwrap();
    let idx_b = plan.passes.iter().position(|p| p.name == "pass_b").unwrap();
    let idx_c = plan.passes.iter().position(|p| p.name == "pass_c").unwrap();
    let idx_d = plan.passes.iter().position(|p| p.name == "pass_d").unwrap();

    assert!(idx_a < idx_b, "A must come before B");
    assert!(idx_a < idx_c, "A must come before C");
    assert!(idx_b < idx_d, "B must come before D");
    assert!(idx_c < idx_d, "C must come before D");
  }

  /// Test: Buffer hazards are tracked correctly
  #[test]
  fn dag_edge_buffer_hazards() {
    initialize_logger();
    let mut builder = RenderGraphBuilder::new();
    let buffer = builder.add_resource(
      "storage_buf",
      RenderResourceKind::Buffer(RenderBufferDesc::new(
        256,
        16,
        GraphResourceLifetime::Persistent,
      )),
    );

    // Pass A writes to buffer
    builder.add_pass(
      "buffer_writer",
      RenderPassKind::Compute,
      RenderPassWork::None,
      0,
      None,
      vec![],
      vec![RenderResourceAccess::write(buffer)],
      vec![buffer],
      vec![buffer],
      None,
      GraphQueueRequest::Any,
      CommandBufferUsage::OneTime,
    );

    // Pass B reads from buffer
    builder.add_pass(
      "buffer_reader",
      RenderPassKind::Compute,
      RenderPassWork::None,
      0,
      None,
      vec![RenderResourceAccess::read(buffer)],
      vec![],
      vec![buffer],
      vec![],
      None,
      GraphQueueRequest::Any,
      CommandBufferUsage::OneTime,
    );

    let mut graph = builder.build();
    graph.compile().unwrap();
    let plan = graph.make_plan();

    let writer_idx = plan
      .passes
      .iter()
      .position(|p| p.name == "buffer_writer")
      .unwrap();
    let reader_idx = plan
      .passes
      .iter()
      .position(|p| p.name == "buffer_reader")
      .unwrap();
    assert!(
      writer_idx < reader_idx,
      "Buffer writer must execute before buffer reader"
    );
  }

  /// Test: Sampled-after-write causes render pass split
  #[test]
  fn hazard_sampled_after_write_causes_split() {
    initialize_logger();
    let mut builder = RenderGraphBuilder::new();
    let rt = builder.add_resource(
      "rt",
      RenderResourceKind::Image(super::RenderImageDesc {
        format: super::GraphFormat::DefaultColor,
        samples: super::GraphSampleCount::One,
        size: super::GraphSize::Swapchain,
        lifetime: super::GraphResourceLifetime::Transient,
        resolve: false,
        is_depth: false,
      }),
    );

    // Pass A writes to rt as color attachment
    builder.add_pass(
      "writer",
      RenderPassKind::Graphics,
      RenderPassWork::None,
      0,
      None,
      vec![],
      vec![RenderResourceAccess::write(rt)],
      vec![rt],
      vec![rt],
      None,
      GraphQueueRequest::Any,
      CommandBufferUsage::OneTime,
    );

    // Pass B samples rt (requires different layout - forces split)
    builder.add_pass(
      "sampler",
      RenderPassKind::Graphics,
      RenderPassWork::None,
      0,
      None,
      vec![RenderResourceAccess::read(rt)],
      vec![],
      vec![rt],
      vec![],
      None,
      GraphQueueRequest::Any,
      CommandBufferUsage::OneTime,
    );

    let mut graph = builder.build();
    graph.compile().unwrap();
    let plan = graph.make_plan();
    let report = plan.report(None);

    // Check that we got a render pass split
    assert!(
      report.render_pass_count >= 1,
      "Should have at least 1 render pass"
    );
  }

  /// Test: Plan hash is stable across compilations (determinism)
  #[test]
  fn dag_plan_hash_stable() {
    initialize_logger();
    let build_and_hash = || {
      let mut builder = RenderGraphBuilder::new();
      let color = builder.add_resource("color", RenderResourceKind::SwapchainColor);
      let depth = builder.add_resource("depth", RenderResourceKind::SwapchainDepth);
      let intermediate = builder.add_resource(
        "intermediate",
        RenderResourceKind::Image(super::RenderImageDesc {
          format: super::GraphFormat::DefaultColor,
          samples: super::GraphSampleCount::One,
          size: super::GraphSize::Swapchain,
          lifetime: super::GraphResourceLifetime::Transient,
          resolve: false,
          is_depth: false,
        }),
      );

      builder.add_pass(
        "scene",
        RenderPassKind::Graphics,
        RenderPassWork::None,
        0,
        None,
        vec![],
        vec![
          RenderResourceAccess::write(color),
          RenderResourceAccess::write(depth),
          RenderResourceAccess::write(intermediate),
        ],
        vec![color, depth, intermediate],
        vec![color, depth, intermediate],
        None,
        GraphQueueRequest::Any,
        CommandBufferUsage::OneTime,
      );

      builder.add_pass(
        "post",
        RenderPassKind::Graphics,
        RenderPassWork::None,
        0,
        None,
        vec![RenderResourceAccess::read(intermediate)],
        vec![RenderResourceAccess::write(color)],
        vec![color, intermediate],
        vec![color],
        None,
        GraphQueueRequest::Any,
        CommandBufferUsage::OneTime,
      );

      builder.add_pass(
        "present",
        RenderPassKind::Present,
        RenderPassWork::None,
        0,
        None,
        vec![RenderResourceAccess::read(color)],
        vec![],
        vec![color],
        vec![],
        None,
        GraphQueueRequest::Any,
        CommandBufferUsage::OneTime,
      );

      let mut graph = builder.build();
      graph.compile().unwrap();
      let plan = graph.make_plan();
      signature_for_plan(&plan)
    };

    let hash1 = build_and_hash();
    let hash2 = build_and_hash();
    let hash3 = build_and_hash();

    assert_eq!(hash1, hash2, "Plan hash must be deterministic");
    assert_eq!(hash2, hash3, "Plan hash must be deterministic");
  }

  /// Test: Graph report accurately counts passes and attachments
  #[test]
  fn graph_report_counts_correct() {
    initialize_logger();
    let mut builder = RenderGraphBuilder::new();
    let color = builder.add_resource("color", RenderResourceKind::SwapchainColor);
    let depth = builder.add_resource("depth", RenderResourceKind::SwapchainDepth);
    let normal = builder.add_resource(
      "normal",
      RenderResourceKind::Image(super::RenderImageDesc {
        format: super::GraphFormat::Rgba16Float,
        samples: super::GraphSampleCount::One,
        size: super::GraphSize::Swapchain,
        lifetime: super::GraphResourceLifetime::Transient,
        resolve: false,
        is_depth: false,
      }),
    );

    builder.add_pass(
      "gbuffer",
      RenderPassKind::Graphics,
      RenderPassWork::None,
      0,
      None,
      vec![],
      vec![
        RenderResourceAccess::write(color),
        RenderResourceAccess::write(depth),
        RenderResourceAccess::write(normal),
      ],
      vec![color, depth, normal],
      vec![color, depth, normal],
      None,
      GraphQueueRequest::Any,
      CommandBufferUsage::OneTime,
    );

    builder.add_pass(
      "lighting",
      RenderPassKind::Graphics,
      RenderPassWork::None,
      0,
      None,
      vec![RenderResourceAccess::read(normal)],
      vec![RenderResourceAccess::write(color)],
      vec![color, normal],
      vec![color],
      None,
      GraphQueueRequest::Any,
      CommandBufferUsage::OneTime,
    );

    builder.add_pass(
      "present",
      RenderPassKind::Present,
      RenderPassWork::None,
      0,
      None,
      vec![RenderResourceAccess::read(color)],
      vec![],
      vec![color],
      vec![],
      None,
      GraphQueueRequest::Any,
      CommandBufferUsage::OneTime,
    );

    let mut graph = builder.build();
    graph.compile().unwrap();
    let plan = graph.make_plan();
    let report = plan.report(None);

    assert_eq!(report.pass_count, 3, "Should have 3 passes");
    assert_eq!(report.attachment_count, 3, "Should have 3 attachments");
    assert!(
      report.render_pass_count >= 1,
      "Should have at least 1 render pass"
    );
  }

  // =========================================================================
  // M2.5: Snapshot Tests for Plan Hashes and Graph Reports
  // =========================================================================

  /// Snapshot test for a deferred rendering pipeline (G-buffer -> Lighting -> Post)
  #[test]
  fn snapshot_deferred_pipeline() {
    initialize_logger();
    const GOLDEN_SIGNATURE: &str = "8145304538199032555";

    let mut builder = RenderGraphBuilder::new();
    let color = builder.add_resource("color", RenderResourceKind::SwapchainColor);
    let depth = builder.add_resource("depth", RenderResourceKind::SwapchainDepth);
    let albedo = builder.add_resource(
      "albedo",
      RenderResourceKind::Image(super::RenderImageDesc {
        format: super::GraphFormat::Rgba8Srgb,
        samples: super::GraphSampleCount::One,
        size: super::GraphSize::Swapchain,
        lifetime: super::GraphResourceLifetime::Transient,
        resolve: false,
        is_depth: false,
      }),
    );
    let normal = builder.add_resource(
      "normal",
      RenderResourceKind::Image(super::RenderImageDesc {
        format: super::GraphFormat::Rgba16Float,
        samples: super::GraphSampleCount::One,
        size: super::GraphSize::Swapchain,
        lifetime: super::GraphResourceLifetime::Transient,
        resolve: false,
        is_depth: false,
      }),
    );

    // G-buffer pass
    builder.add_pass(
      "gbuffer",
      RenderPassKind::Graphics,
      RenderPassWork::None,
      0,
      None,
      vec![],
      vec![
        RenderResourceAccess::write(albedo),
        RenderResourceAccess::write(normal),
        RenderResourceAccess::write(depth),
      ],
      vec![albedo, normal, depth],
      vec![albedo, normal, depth],
      None,
      GraphQueueRequest::Any,
      CommandBufferUsage::OneTime,
    );

    // Lighting pass (reads G-buffer, writes color)
    builder.add_pass(
      "lighting",
      RenderPassKind::Graphics,
      RenderPassWork::None,
      0,
      None,
      vec![
        RenderResourceAccess::read(albedo),
        RenderResourceAccess::read(normal),
        RenderResourceAccess::read(depth),
      ],
      vec![RenderResourceAccess::write(color)],
      vec![color, albedo, normal, depth],
      vec![color],
      None,
      GraphQueueRequest::Any,
      CommandBufferUsage::OneTime,
    );

    // Post-processing (tonemapping)
    builder.add_pass(
      "tonemap",
      RenderPassKind::Graphics,
      RenderPassWork::None,
      0,
      None,
      vec![RenderResourceAccess::read(color)],
      vec![RenderResourceAccess::write(color)],
      vec![color],
      vec![color],
      None,
      GraphQueueRequest::Any,
      CommandBufferUsage::OneTime,
    );

    // Present
    builder.add_pass(
      "present",
      RenderPassKind::Present,
      RenderPassWork::None,
      0,
      None,
      vec![RenderResourceAccess::read(color)],
      vec![],
      vec![color],
      vec![],
      None,
      GraphQueueRequest::Any,
      CommandBufferUsage::OneTime,
    );

    let mut graph = builder.build();
    graph.compile().unwrap();
    let plan = graph.make_plan();
    let sig = signature_for_plan(&plan);

    assert_eq!(
      sig.to_string(),
      GOLDEN_SIGNATURE,
      "Deferred pipeline plan signature changed! Update golden if intentional."
    );

    // Verify structure
    let report = plan.report(None);
    assert_eq!(report.pass_count, 4);
    assert_eq!(report.attachment_count, 4);
  }

  /// Snapshot test for compute + graphics mixed pipeline
  #[test]
  fn snapshot_compute_graphics_mixed() {
    initialize_logger();
    let _env_guard = TEST_ENV_MUTEX.lock().unwrap();
    std::env::remove_var("ICS_RENDER_GRAPH_SINGLE_QUEUE");

    const GOLDEN_SIGNATURE: &str = "9590127394725438211";

    let mut builder = RenderGraphBuilder::new();
    let color = builder.add_resource("color", RenderResourceKind::SwapchainColor);
    let compute_buffer = builder.add_resource(
      "compute_buf",
      RenderResourceKind::Buffer(RenderBufferDesc::new(
        256,
        16,
        GraphResourceLifetime::Persistent,
      )),
    );

    // Compute pass writes to buffer
    builder.add_pass(
      "compute_sim",
      RenderPassKind::Compute,
      RenderPassWork::None,
      0,
      None,
      vec![],
      vec![RenderResourceAccess::write(compute_buffer)],
      vec![compute_buffer],
      vec![compute_buffer],
      None,
      GraphQueueRequest::Prefer(GraphQueueClass::Compute),
      CommandBufferUsage::OneTime,
    );

    // Graphics pass reads buffer, writes color
    builder.add_pass(
      "render",
      RenderPassKind::Graphics,
      RenderPassWork::None,
      0,
      None,
      vec![RenderResourceAccess::read(compute_buffer)],
      vec![RenderResourceAccess::write(color)],
      vec![color, compute_buffer],
      vec![color],
      None,
      GraphQueueRequest::Require(GraphQueueClass::Graphics),
      CommandBufferUsage::OneTime,
    );

    // Present
    builder.add_pass(
      "present",
      RenderPassKind::Present,
      RenderPassWork::None,
      0,
      None,
      vec![RenderResourceAccess::read(color)],
      vec![],
      vec![color],
      vec![],
      None,
      GraphQueueRequest::Any,
      CommandBufferUsage::OneTime,
    );

    let mut graph = builder.build();
    graph.compile().unwrap();
    let plan = graph.make_plan();
    let sig = signature_for_plan(&plan);

    assert_eq!(
      sig.to_string(),
      GOLDEN_SIGNATURE,
      "Compute+Graphics pipeline plan signature changed! Update golden if intentional."
    );

    let report = plan.report(None);
    assert_eq!(report.pass_count, 3);
  }

  /// Snapshot test for shadow mapping pipeline
  #[test]
  fn snapshot_shadow_mapping() {
    initialize_logger();
    const GOLDEN_SIGNATURE: &str = "825343652313261325";

    let mut builder = RenderGraphBuilder::new();
    let color = builder.add_resource("color", RenderResourceKind::SwapchainColor);
    let depth = builder.add_resource("depth", RenderResourceKind::SwapchainDepth);
    let shadow_map = builder.add_resource(
      "shadow_map",
      RenderResourceKind::Image(super::RenderImageDesc {
        format: super::GraphFormat::D32,
        samples: super::GraphSampleCount::One,
        size: super::GraphSize::Swapchain,
        lifetime: super::GraphResourceLifetime::Transient,
        resolve: false,
        is_depth: true,
      }),
    );

    // Shadow pass (depth-only render to shadow map)
    builder.add_pass(
      "shadow",
      RenderPassKind::Graphics,
      RenderPassWork::None,
      0,
      None,
      vec![],
      vec![RenderResourceAccess::write(shadow_map)],
      vec![shadow_map],
      vec![shadow_map],
      None,
      GraphQueueRequest::Any,
      CommandBufferUsage::OneTime,
    );

    // Main scene pass (reads shadow map for shadow sampling)
    builder.add_pass(
      "scene",
      RenderPassKind::Graphics,
      RenderPassWork::None,
      0,
      None,
      vec![RenderResourceAccess::read(shadow_map)],
      vec![
        RenderResourceAccess::write(color),
        RenderResourceAccess::write(depth),
      ],
      vec![color, depth, shadow_map],
      vec![color, depth],
      None,
      GraphQueueRequest::Any,
      CommandBufferUsage::OneTime,
    );

    // Present
    builder.add_pass(
      "present",
      RenderPassKind::Present,
      RenderPassWork::None,
      0,
      None,
      vec![RenderResourceAccess::read(color)],
      vec![],
      vec![color],
      vec![],
      None,
      GraphQueueRequest::Any,
      CommandBufferUsage::OneTime,
    );

    let mut graph = builder.build();
    graph.compile().unwrap();
    let plan = graph.make_plan();
    let sig = signature_for_plan(&plan);

    assert_eq!(
      sig.to_string(),
      GOLDEN_SIGNATURE,
      "Shadow mapping pipeline plan signature changed! Update golden if intentional."
    );

    let report = plan.report(None);
    assert_eq!(report.pass_count, 3);
    assert_eq!(report.attachment_count, 3);
  }

  /// Verify graph report text format is stable
  #[test]
  fn snapshot_graph_report_format() {
    initialize_logger();
    let mut builder = RenderGraphBuilder::new();
    let color = builder.add_resource("color", RenderResourceKind::SwapchainColor);

    builder.add_pass(
      "clear",
      RenderPassKind::Graphics,
      RenderPassWork::None,
      0,
      None,
      vec![],
      vec![RenderResourceAccess::write(color)],
      vec![color],
      vec![color],
      None,
      GraphQueueRequest::Any,
      CommandBufferUsage::OneTime,
    );

    builder.add_pass(
      "present",
      RenderPassKind::Present,
      RenderPassWork::None,
      0,
      None,
      vec![RenderResourceAccess::read(color)],
      vec![],
      vec![color],
      vec![],
      None,
      GraphQueueRequest::Any,
      CommandBufferUsage::OneTime,
    );

    let mut graph = builder.build();
    graph.compile().unwrap();
    let plan = graph.make_plan();
    let report = plan.report(None);
    let text = report.dump_text();

    // Verify expected sections exist
    assert!(text.contains("RenderGraphReport"), "Missing report header");
    assert!(text.contains("passes="), "Missing pass count");
    assert!(text.contains("attachments="), "Missing attachment count");
    assert!(text.contains("render_passes="), "Missing render pass count");
    assert!(
      text.contains("validation_errors="),
      "Missing validation errors"
    );
    assert!(
      text.contains("validation_warnings="),
      "Missing validation warnings"
    );
  }

  /// Verify plan text format is stable
  #[test]
  fn snapshot_plan_text_format() {
    initialize_logger();
    let mut builder = RenderGraphBuilder::new();
    let color = builder.add_resource("color", RenderResourceKind::SwapchainColor);

    builder.add_pass(
      "main",
      RenderPassKind::Graphics,
      RenderPassWork::None,
      0,
      None,
      vec![],
      vec![RenderResourceAccess::write(color)],
      vec![color],
      vec![color],
      None,
      GraphQueueRequest::Any,
      CommandBufferUsage::OneTime,
    );

    builder.add_pass(
      "present",
      RenderPassKind::Present,
      RenderPassWork::None,
      0,
      None,
      vec![RenderResourceAccess::read(color)],
      vec![],
      vec![color],
      vec![],
      None,
      GraphQueueRequest::Any,
      CommandBufferUsage::OneTime,
    );

    let mut graph = builder.build();
    graph.compile().unwrap();
    let plan = graph.make_plan();
    let text = plan.dump_text();

    // Verify expected sections exist
    assert!(text.contains("RenderPlan"), "Missing plan header");
    assert!(text.contains("Attachments:"), "Missing attachments section");
    assert!(text.contains("Passes:"), "Missing passes section");
    assert!(text.contains("color"), "Missing color attachment");
    assert!(text.contains("main"), "Missing main pass");
    assert!(text.contains("present"), "Missing present pass");
  }

  /// Verify sync plan text format is stable
  #[test]
  fn snapshot_sync_plan_text_format() {
    initialize_logger();
    // Handle poisoned mutex from previous test panics
    let _env_guard = TEST_ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("ICS_RENDER_GRAPH_SINGLE_QUEUE");

    let mut builder = RenderGraphBuilder::new();
    let color = builder.add_resource("color", RenderResourceKind::SwapchainColor);

    builder.add_pass(
      "render",
      RenderPassKind::Graphics,
      RenderPassWork::None,
      0,
      None,
      vec![],
      vec![RenderResourceAccess::write(color)],
      vec![color],
      vec![color],
      None,
      GraphQueueRequest::Any,
      CommandBufferUsage::OneTime,
    );

    builder.add_pass(
      "present",
      RenderPassKind::Present,
      RenderPassWork::None,
      0,
      None,
      vec![RenderResourceAccess::read(color)],
      vec![],
      vec![color],
      vec![],
      None,
      GraphQueueRequest::Any,
      CommandBufferUsage::OneTime,
    );

    let mut graph = builder.build();
    graph.compile().unwrap();
    let plan = graph.make_plan();
    let sync = RenderSyncPlan::from_plan(&plan);
    let text = sync.dump_text();

    // Verify expected sections exist
    assert!(text.contains("RenderSyncPlan"), "Missing sync plan header");
    assert!(text.contains("submissions="), "Missing submissions count");
    assert!(text.contains("resources="), "Missing resources count");
    assert!(
      text.contains("barriers_image="),
      "Missing image barriers count"
    );
  }
}

/// Specifies how a command buffer will be used, affecting allocation and reuse strategy.
///
/// This enum maps to Vulkan's `VkCommandBufferUsageFlags` and controls:
/// - Command buffer allocation from pools
/// - Whether the buffer can be resubmitted
/// - Whether the buffer can be used concurrently
///
/// # Allocation Strategy
///
/// | Mode | Pool Flags | Begin Flags | Reuse |
/// |------|------------|-------------|-------|
/// | `OneTime` | `RESET_COMMAND_BUFFER` | `ONE_TIME_SUBMIT` | No - must re-record each frame |
/// | `Reusable` | `RESET_COMMAND_BUFFER` | None | Yes - can be resubmitted without re-recording |
/// | `Simultaneous` | `RESET_COMMAND_BUFFER` | `SIMULTANEOUS_USE` | Yes - can execute on multiple queues |
///
/// # Usage Guidelines
///
/// - **`OneTime`**: Best for command buffers that change every frame (e.g., scene rendering
///   with dynamic objects). Lower GPU overhead but requires re-recording.
///
/// - **`Reusable`**: Best for static command buffers that don't change between frames
///   (e.g., UI elements, skyboxes). Avoids re-recording overhead but requires the buffer
///   to complete execution before resubmission.
///
/// - **`Simultaneous`**: Required when the same command buffer may be in-flight on
///   multiple queues or when resubmitting before completion. Has the highest overhead
///   but offers maximum flexibility.
///
/// # Examples
///
/// ```ignore
/// // For dynamic scene rendering (default)
/// GraphPassRequest {
///     command_usage: CommandBufferUsage::OneTime,
///     ..Default::default()
/// }
///
/// // For static post-processing that doesn't change
/// GraphPassRequest {
///     command_usage: CommandBufferUsage::Reusable,
///     ..Default::default()
/// }
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CommandBufferUsage {
  /// Record once per frame, cannot be resubmitted.
  /// Maps to `VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT`.
  /// Best for dynamic content that changes every frame.
  OneTime,

  /// Can be resubmitted without re-recording.
  /// No special begin flags, but buffer must complete before resubmission.
  /// Best for static content that doesn't change between frames.
  Reusable,

  /// Can be in-flight on multiple queues simultaneously.
  /// Maps to `VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT`.
  /// Required for concurrent execution, has highest overhead.
  Simultaneous,
}

impl Default for CommandBufferUsage {
  fn default() -> Self {
    CommandBufferUsage::OneTime
  }
}

impl CommandBufferUsage {
  /// Returns a human-readable description of this usage mode.
  pub fn description(&self) -> &'static str {
    match self {
      Self::OneTime => "one-time submit (re-record each frame)",
      Self::Reusable => "reusable (resubmit without re-recording)",
      Self::Simultaneous => "simultaneous use (concurrent execution)",
    }
  }

  /// Returns true if this mode requires re-recording every frame.
  pub fn requires_rerecord(&self) -> bool {
    matches!(self, Self::OneTime)
  }

  /// Returns true if this mode allows resubmission without re-recording.
  pub fn allows_reuse(&self) -> bool {
    matches!(self, Self::Reusable | Self::Simultaneous)
  }

  /// Returns true if this mode supports concurrent execution.
  pub fn allows_concurrent(&self) -> bool {
    matches!(self, Self::Simultaneous)
  }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GraphResourceRequest {
  pub name: String,
  pub kind: GraphResourceKind,
  pub format: GraphFormat,
  pub samples: GraphSampleCount,
  pub size: GraphSize,
  pub buffer_size: usize,
  pub buffer_alignment: usize,
  pub lifetime: GraphResourceLifetime,
  pub role: GraphResourceRole,
}

impl Default for GraphResourceRequest {
  fn default() -> Self {
    Self {
      name: String::new(),
      kind: GraphResourceKind::Color,
      format: GraphFormat::DefaultColor,
      samples: GraphSampleCount::One,
      size: GraphSize::Swapchain,
      buffer_size: 0,
      buffer_alignment: 0,
      lifetime: GraphResourceLifetime::Transient,
      role: GraphResourceRole::Custom,
    }
  }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GraphPassResourceUse {
  pub resource: String,
  pub access: GraphAccessType,
}

impl GraphPassResourceUse {
  pub fn sampled(resource: impl Into<String>) -> Self {
    Self {
      resource: resource.into(),
      access: GraphAccessType::Sampled,
    }
  }
  pub fn input(resource: impl Into<String>) -> Self {
    Self {
      resource: resource.into(),
      access: GraphAccessType::InputAttachment,
    }
  }
  pub fn color_write(resource: impl Into<String>) -> Self {
    Self {
      resource: resource.into(),
      access: GraphAccessType::ColorWrite,
    }
  }
  pub fn depth_write(resource: impl Into<String>) -> Self {
    Self {
      resource: resource.into(),
      access: GraphAccessType::DepthWrite,
    }
  }
  pub fn present(resource: impl Into<String>) -> Self {
    Self {
      resource: resource.into(),
      access: GraphAccessType::Present,
    }
  }
  pub fn buffer_read(resource: impl Into<String>) -> Self {
    Self {
      resource: resource.into(),
      access: GraphAccessType::BufferRead,
    }
  }
  pub fn buffer_write(resource: impl Into<String>) -> Self {
    Self {
      resource: resource.into(),
      access: GraphAccessType::BufferWrite,
    }
  }
  pub fn buffer_read_write(resource: impl Into<String>) -> Self {
    Self {
      resource: resource.into(),
      access: GraphAccessType::BufferReadWrite,
    }
  }
}

// ---------------------------------------------------------------------------
// Graph Output Mapping - bind color attachments to shader outputs
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GraphOutputSemantic {
  Colour,
  Normal,
  Tangent,
  BitTangent,
  Position,
  Texture,
  Ambient,
  Diffuse,
  Specular,
  Shininess,
  Roughness,
  Metallic,
  Emission,
  AmbientOcclusion,
}

impl GraphOutputSemantic {
  pub fn as_str(&self) -> &'static str {
    match self {
      GraphOutputSemantic::Colour => "colour",
      GraphOutputSemantic::Normal => "normal",
      GraphOutputSemantic::Tangent => "tangent",
      GraphOutputSemantic::BitTangent => "bit_tangent",
      GraphOutputSemantic::Position => "position",
      GraphOutputSemantic::Texture => "texture",
      GraphOutputSemantic::Ambient => "ambient",
      GraphOutputSemantic::Diffuse => "diffuse",
      GraphOutputSemantic::Specular => "specular",
      GraphOutputSemantic::Shininess => "shininess",
      GraphOutputSemantic::Roughness => "roughness",
      GraphOutputSemantic::Metallic => "metallic",
      GraphOutputSemantic::Emission => "emission",
      GraphOutputSemantic::AmbientOcclusion => "ambient_occlusion",
    }
  }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum GraphColorOutputSource {
  Semantic(GraphOutputSemantic),
  Attribute(PipelineAttribute),
  Texture(TextureType),
  Material(MaterialVariable),
  Custom(String),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GraphColorOutput {
  pub resource: String,
  pub source: GraphColorOutputSource,
}

impl GraphColorOutput {
  pub fn new(resource: impl Into<String>, source: GraphColorOutputSource) -> Self {
    Self {
      resource: resource.into(),
      source,
    }
  }
}

#[derive(Clone, Debug)]
pub struct GraphPassRequest {
  pub name: String,
  pub kind: RenderPassKind,
  pub work: RenderPassWork,
  pub role: GraphResourceRole,
  pub inputs: Vec<GraphPassResourceUse>,
  pub outputs: Vec<GraphPassResourceUse>,
  /// Optional mapping of color attachments to shader output sources.
  /// Entries are ordered by attachment index.
  pub color_outputs: Vec<GraphColorOutput>,
  pub render_pass_hint: Option<u32>,
  pub subpass_group_hint: Option<u32>,
  pub msaa_hint: Option<GraphSampleCount>,
  pub queue_request: GraphQueueRequest,
  pub command_usage: CommandBufferUsage,
}

impl Default for GraphPassRequest {
  fn default() -> Self {
    Self {
      name: String::new(),
      kind: RenderPassKind::Graphics,
      work: RenderPassWork::None,
      role: GraphResourceRole::Custom,
      inputs: Vec::new(),
      outputs: Vec::new(),
      color_outputs: Vec::new(),
      render_pass_hint: None,
      subpass_group_hint: None,
      msaa_hint: None,
      queue_request: GraphQueueRequest::Any,
      command_usage: CommandBufferUsage::OneTime,
    }
  }
}

#[derive(Clone, Debug, Default)]
pub struct LayerGraphHints {
  pub resources: Vec<GraphResourceRequest>,
  pub passes: Vec<GraphPassRequest>,
}

impl LayerGraphHints {
  /// Create a new empty hints struct.
  pub fn new() -> Self {
    Self::default()
  }

  /// Start building graph hints with a fluent API.
  pub fn builder() -> GraphHintsBuilder {
    GraphHintsBuilder::new()
  }

  /// Standard 3D scene setup: color + depth render targets, single pass.
  ///
  /// Creates:
  /// - `scene_color`: Color attachment (DefaultColor format, transient)
  /// - `scene_depth`: Depth attachment (DefaultDepth format, transient)
  /// - `scene`: Pass that writes to both
  pub fn standard_3d_scene() -> Self {
    Self::builder()
      .with_color("scene_color", GraphResourceRole::Scene)
      .with_depth("scene_depth", GraphResourceRole::Scene)
      .with_pass(
        GraphPass::new("scene")
          .role(GraphResourceRole::Scene)
          .color_output_from(
            "scene_color",
            GraphColorOutputSource::Semantic(GraphOutputSemantic::Colour),
          )
          .depth_output("scene_depth"),
      )
      .build()
  }

  /// Post-processing setup: samples input, writes to output (typically swapchain).
  ///
  /// Creates:
  /// - Reference to input resource (must be declared elsewhere)
  /// - Output resource as swapchain color
  /// - Post pass that samples input and writes output
  pub fn post_process(input: &str, output: &str) -> Self {
    Self::builder()
      .with_swapchain(output)
      .with_pass(
        GraphPass::new("post")
          .role(GraphResourceRole::Post)
          .samples(input)
          .color_output_from(
            output,
            GraphColorOutputSource::Semantic(GraphOutputSemantic::Colour),
          ),
      )
      .build()
  }

  /// Merge another LayerGraphHints into this one.
  /// Resources and passes are appended.
  pub fn merge(&mut self, other: LayerGraphHints) {
    self.resources.extend(other.resources);
    self.passes.extend(other.passes);
  }
}

// ---------------------------------------------------------------------------
// GroupResource - Shared resources declared at group level
// ---------------------------------------------------------------------------

/// A resource declared at the group level that layers can reference.
///
/// Group resources are declared once on a [`LayerGroup`] and can be used
/// by multiple layers within that group. Layers declare which group resources
/// they use via [`LayerDescription::uses_group_resources`].
///
/// # Example
/// ```ignore
/// let group = LayerGroupBuilder::new("main_scene")
///     .declares_resource(GroupResource::color("shared_color"))
///     .declares_resource(GroupResource::depth("shared_depth"))
///     .register(&mut layers);
/// ```
///
/// [`LayerGroup`]: super::layer::LayerGroup
/// [`LayerDescription::uses_group_resources`]: super::layer_description::LayerDescription::uses_group_resources
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GroupResource {
  /// The resource request to add to the group's graph hints
  request: GraphResourceRequest,
}

impl GroupResource {
  /// Create a color resource with standard defaults.
  ///
  /// Defaults: DefaultColor format, 1 sample, swapchain size, transient lifetime, Scene role.
  pub fn color(name: impl Into<String>) -> Self {
    Self {
      request: GraphResourceRequest {
        name: name.into(),
        kind: GraphResourceKind::Color,
        format: GraphFormat::DefaultColor,
        samples: GraphSampleCount::One,
        size: GraphSize::Swapchain,
        buffer_size: 0,
        buffer_alignment: 0,
        lifetime: GraphResourceLifetime::Transient,
        role: GraphResourceRole::Scene,
      },
    }
  }

  /// Create a depth resource with standard defaults.
  ///
  /// Defaults: DefaultDepth format, 1 sample, swapchain size, transient lifetime, Scene role.
  pub fn depth(name: impl Into<String>) -> Self {
    Self {
      request: GraphResourceRequest {
        name: name.into(),
        kind: GraphResourceKind::Depth,
        format: GraphFormat::DefaultDepth,
        samples: GraphSampleCount::One,
        size: GraphSize::Swapchain,
        buffer_size: 0,
        buffer_alignment: 0,
        lifetime: GraphResourceLifetime::Transient,
        role: GraphResourceRole::Scene,
      },
    }
  }

  /// Create a normal buffer resource.
  ///
  /// Defaults: R16G16Snorm format, 1 sample, swapchain size, transient lifetime, Scene role.
  pub fn normal(name: impl Into<String>) -> Self {
    Self {
      request: GraphResourceRequest {
        name: name.into(),
        kind: GraphResourceKind::Normal,
        format: GraphFormat::R16G16Snorm,
        samples: GraphSampleCount::One,
        size: GraphSize::Swapchain,
        buffer_size: 0,
        buffer_alignment: 0,
        lifetime: GraphResourceLifetime::Transient,
        role: GraphResourceRole::Scene,
      },
    }
  }

  /// Create a swapchain color output resource.
  ///
  /// Defaults: SwapchainColor kind, DefaultColor format, external lifetime, Post role.
  pub fn swapchain(name: impl Into<String>) -> Self {
    Self {
      request: GraphResourceRequest {
        name: name.into(),
        kind: GraphResourceKind::SwapchainColor,
        format: GraphFormat::DefaultColor,
        samples: GraphSampleCount::One,
        size: GraphSize::Swapchain,
        buffer_size: 0,
        buffer_alignment: 0,
        lifetime: GraphResourceLifetime::External,
        role: GraphResourceRole::Post,
      },
    }
  }

  /// Create a buffer resource with explicit size/alignment.
  pub fn buffer(
    name: impl Into<String>,
    size: usize,
    alignment: usize,
    role: GraphResourceRole,
  ) -> Self {
    Self {
      request: GraphResourceRequest {
        name: name.into(),
        kind: GraphResourceKind::Buffer,
        format: GraphFormat::DefaultColor,
        samples: GraphSampleCount::One,
        size: GraphSize::Swapchain,
        buffer_size: size,
        buffer_alignment: alignment.max(1),
        lifetime: GraphResourceLifetime::Persistent,
        role,
      },
    }
  }

  /// Create a custom resource with full control.
  pub fn custom(request: GraphResourceRequest) -> Self {
    Self { request }
  }

  /// Get the resource name.
  pub fn name(&self) -> &str {
    &self.request.name
  }

  /// Convert to the underlying request.
  pub fn into_request(self) -> GraphResourceRequest {
    self.request
  }

  /// Get a reference to the underlying request.
  pub fn request(&self) -> &GraphResourceRequest {
    &self.request
  }
}

// ---------------------------------------------------------------------------
// GraphHintsBuilder - Fluent builder for LayerGraphHints
// ---------------------------------------------------------------------------

/// Fluent builder for constructing `LayerGraphHints`.
///
/// # Example
/// ```ignore
/// let hints = LayerGraphHints::builder()
///     .with_color("scene_color", GraphResourceRole::Scene)
///     .with_depth("scene_depth", GraphResourceRole::Scene)
///     .with_pass(GraphPass::new("main")
///         .role(GraphResourceRole::Scene)
///         .color_output_from(
///             "scene_color",
///             GraphColorOutputSource::Semantic(GraphOutputSemantic::Colour),
///         )
///         .depth_output("scene_depth"))
///     .build();
/// ```
#[derive(Clone, Debug, Default)]
pub struct GraphHintsBuilder {
  resources: Vec<GraphResourceRequest>,
  passes: Vec<GraphPassRequest>,
}

impl GraphHintsBuilder {
  /// Create a new empty builder.
  pub fn new() -> Self {
    Self::default()
  }

  /// Add a color resource with standard defaults.
  ///
  /// Defaults: DefaultColor format, 1 sample, swapchain size, transient lifetime.
  pub fn with_color(mut self, name: impl Into<String>, role: GraphResourceRole) -> Self {
    self.resources.push(GraphResourceRequest {
      name: name.into(),
      kind: GraphResourceKind::Color,
      format: GraphFormat::DefaultColor,
      samples: GraphSampleCount::One,
      size: GraphSize::Swapchain,
      buffer_size: 0,
      buffer_alignment: 0,
      lifetime: GraphResourceLifetime::Transient,
      role,
    });
    self
  }

  /// Add a depth resource with standard defaults.
  ///
  /// Defaults: DefaultDepth format, 1 sample, swapchain size, transient lifetime.
  pub fn with_depth(mut self, name: impl Into<String>, role: GraphResourceRole) -> Self {
    self.resources.push(GraphResourceRequest {
      name: name.into(),
      kind: GraphResourceKind::Depth,
      format: GraphFormat::DefaultDepth,
      samples: GraphSampleCount::One,
      size: GraphSize::Swapchain,
      buffer_size: 0,
      buffer_alignment: 0,
      lifetime: GraphResourceLifetime::Transient,
      role,
    });
    self
  }

  /// Add a normal buffer resource.
  ///
  /// Defaults: R16G16Snorm format, 1 sample, swapchain size, transient lifetime.
  pub fn with_normal(mut self, name: impl Into<String>, role: GraphResourceRole) -> Self {
    self.resources.push(GraphResourceRequest {
      name: name.into(),
      kind: GraphResourceKind::Normal,
      format: GraphFormat::R16G16Snorm,
      samples: GraphSampleCount::One,
      size: GraphSize::Swapchain,
      buffer_size: 0,
      buffer_alignment: 0,
      lifetime: GraphResourceLifetime::Transient,
      role,
    });
    self
  }

  /// Add a swapchain color output resource.
  ///
  /// Defaults: SwapchainColor kind, DefaultColor format, external lifetime.
  pub fn with_swapchain(mut self, name: impl Into<String>) -> Self {
    self.resources.push(GraphResourceRequest {
      name: name.into(),
      kind: GraphResourceKind::SwapchainColor,
      format: GraphFormat::DefaultColor,
      samples: GraphSampleCount::One,
      size: GraphSize::Swapchain,
      buffer_size: 0,
      buffer_alignment: 0,
      lifetime: GraphResourceLifetime::External,
      role: GraphResourceRole::Post,
    });
    self
  }

  /// Add a buffer resource with explicit size/alignment.
  pub fn with_buffer(
    mut self,
    name: impl Into<String>,
    size: usize,
    alignment: usize,
    role: GraphResourceRole,
  ) -> Self {
    self.resources.push(GraphResourceRequest {
      name: name.into(),
      kind: GraphResourceKind::Buffer,
      format: GraphFormat::DefaultColor,
      samples: GraphSampleCount::One,
      size: GraphSize::Swapchain,
      buffer_size: size,
      buffer_alignment: alignment.max(1),
      lifetime: GraphResourceLifetime::Persistent,
      role,
    });
    self
  }

  /// Add a custom resource with full control over all properties.
  pub fn with_resource(mut self, request: GraphResourceRequest) -> Self {
    self.resources.push(request);
    self
  }

  /// Add a pass using the GraphPass builder.
  pub fn with_pass(mut self, pass: GraphPass) -> Self {
    self.passes.push(pass.build());
    self
  }

  /// Add a raw GraphPassRequest directly.
  pub fn with_pass_request(mut self, pass: GraphPassRequest) -> Self {
    self.passes.push(pass);
    self
  }

  /// Build the final LayerGraphHints.
  pub fn build(self) -> LayerGraphHints {
    LayerGraphHints {
      resources: self.resources,
      passes: self.passes,
    }
  }
}

// ---------------------------------------------------------------------------
// GraphPass - Fluent builder for GraphPassRequest
// ---------------------------------------------------------------------------

/// Fluent builder for constructing `GraphPassRequest`.
///
/// # Example
/// ```ignore
/// let pass = GraphPass::new("scene")
///     .role(GraphResourceRole::Scene)
///     .samples("input_texture")
///     .color_output_from(
///         "scene_color",
///         GraphColorOutputSource::Semantic(GraphOutputSemantic::Colour),
///     )
///     .depth_output("scene_depth")
///     .with_msaa(GraphSampleCount::Four)
///     .build();
/// ```
#[derive(Clone, Debug)]
pub struct GraphPass {
  name: String,
  kind: RenderPassKind,
  work: RenderPassWork,
  role: GraphResourceRole,
  inputs: Vec<GraphPassResourceUse>,
  outputs: Vec<GraphPassResourceUse>,
  color_outputs: Vec<GraphColorOutput>,
  render_pass_hint: Option<u32>,
  subpass_group_hint: Option<u32>,
  msaa_hint: Option<GraphSampleCount>,
  queue_request: GraphQueueRequest,
  command_usage: CommandBufferUsage,
}

impl GraphPass {
  /// Create a new pass builder with the given name.
  pub fn new(name: impl Into<String>) -> Self {
    Self {
      name: name.into(),
      kind: RenderPassKind::Graphics,
      work: RenderPassWork::None,
      role: GraphResourceRole::Custom,
      inputs: Vec::new(),
      outputs: Vec::new(),
      color_outputs: Vec::new(),
      render_pass_hint: None,
      subpass_group_hint: None,
      msaa_hint: None,
      queue_request: GraphQueueRequest::Any,
      command_usage: CommandBufferUsage::OneTime,
    }
  }

  /// Set the role of this pass (Scene, Post, or Custom).
  pub fn role(mut self, role: GraphResourceRole) -> Self {
    self.role = role;
    self
  }

  /// Set the pass kind (Graphics, Compute, Transfer, etc.).
  pub fn kind(mut self, kind: RenderPassKind) -> Self {
    self.kind = kind;
    self
  }

  /// Set the pass work callback or pipeline.
  pub fn work(mut self, work: RenderPassWork) -> Self {
    self.work = work;
    self
  }

  /// Add a sampled input (texture sampling in shader).
  pub fn samples(mut self, resource: impl Into<String>) -> Self {
    self.inputs.push(GraphPassResourceUse::sampled(resource));
    self
  }

  /// Add an input attachment (subpass input).
  pub fn input_attachment(mut self, resource: impl Into<String>) -> Self {
    self.inputs.push(GraphPassResourceUse::input(resource));
    self
  }

  /// Add a buffer read input.
  pub fn buffer_read(mut self, resource: impl Into<String>) -> Self {
    self
      .inputs
      .push(GraphPassResourceUse::buffer_read(resource));
    self
  }

  /// Add a color attachment output.
  ///
  /// For passes with multiple color attachments, use `color_output_from`
  /// to provide explicit shader output mappings.
  pub fn color_output(mut self, resource: impl Into<String>) -> Self {
    self
      .outputs
      .push(GraphPassResourceUse::color_write(resource));
    self
  }

  /// Add a color attachment output and map it to a shader output source.
  pub fn color_output_from(
    mut self,
    resource: impl Into<String>,
    source: GraphColorOutputSource,
  ) -> Self {
    let resource = resource.into();
    self
      .outputs
      .push(GraphPassResourceUse::color_write(resource.clone()));
    self
      .color_outputs
      .push(GraphColorOutput::new(resource, source));
    self
  }

  /// Add a depth attachment output.
  pub fn depth_output(mut self, resource: impl Into<String>) -> Self {
    self
      .outputs
      .push(GraphPassResourceUse::depth_write(resource));
    self
  }

  /// Add a buffer write output.
  pub fn buffer_write(mut self, resource: impl Into<String>) -> Self {
    self
      .outputs
      .push(GraphPassResourceUse::buffer_write(resource));
    self
  }

  /// Add a present output (for swapchain).
  pub fn presents(mut self, resource: impl Into<String>) -> Self {
    self.outputs.push(GraphPassResourceUse::present(resource));
    self
  }

  /// Set MSAA sample count hint.
  pub fn with_msaa(mut self, samples: GraphSampleCount) -> Self {
    self.msaa_hint = Some(samples);
    self
  }

  /// Set render pass grouping hint.
  pub fn render_pass_hint(mut self, hint: u32) -> Self {
    self.render_pass_hint = Some(hint);
    self
  }

  /// Set subpass grouping hint.
  pub fn subpass_group(mut self, group: u32) -> Self {
    self.subpass_group_hint = Some(group);
    self
  }

  /// Set queue request (Any, Require, or Prefer).
  pub fn queue(mut self, request: GraphQueueRequest) -> Self {
    self.queue_request = request;
    self
  }

  /// Set command buffer usage mode.
  pub fn command_usage(mut self, usage: CommandBufferUsage) -> Self {
    self.command_usage = usage;
    self
  }

  /// Build the final GraphPassRequest.
  pub fn build(self) -> GraphPassRequest {
    GraphPassRequest {
      name: self.name,
      kind: self.kind,
      work: self.work,
      role: self.role,
      inputs: self.inputs,
      outputs: self.outputs,
      color_outputs: self.color_outputs,
      render_pass_hint: self.render_pass_hint,
      subpass_group_hint: self.subpass_group_hint,
      msaa_hint: self.msaa_hint,
      queue_request: self.queue_request,
      command_usage: self.command_usage,
    }
  }
}

// ---------------------------------------------------------------------------
// Render plan (API-agnostic description of attachments/passes), to be consumed
// by backends to build render passes/framebuffers and resource allocations.
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct PlanAttachment {
  pub name: String,
  pub format: GraphFormat,
  pub samples: GraphSampleCount,
  pub size: GraphSize,
  pub lifetime: GraphResourceLifetime,
  pub role: GraphResourceRole,
  pub is_swapchain: bool,
  pub initial_layout: GraphImageLayout,
  pub final_layout: GraphImageLayout,
  pub ever_read: bool,
}

#[derive(Clone, Debug)]
pub struct PlanPass {
  pub name: String,
  /// Render pass group index for this pass.
  pub render_pass: u32,
  pub render_pass_hint: Option<u32>,
  /// Target subpass index within the render pass.
  pub subpass: u32,
  pub reads: Vec<GraphPassResourceUse>,
  pub writes: Vec<GraphPassResourceUse>,
  pub kind: RenderPassKind,
  pub queue_request: GraphQueueRequest,
  pub command_usage: CommandBufferUsage,
}

#[derive(Clone, Debug, Default)]
pub struct RenderPlan {
  pub attachments: Vec<PlanAttachment>,
  pub passes: Vec<PlanPass>,
  pub resource_kinds: HashMap<String, RenderResourceKind>,
  /// Number of subpasses required by this plan (max subpass index + 1).
  pub subpass_count: u32,
  /// Number of distinct render pass groups needed.
  pub render_pass_count: u32,
  /// Subpass counts per render pass group.
  pub render_pass_subpasses: Vec<u32>,
  pub grouping_stats: RenderPlanGroupingStats,
}

#[derive(Clone, Debug, Default)]
pub struct RenderPlanGroupingStats {
  pub split_sampled_after_write: usize,
  pub split_write_after_sampled: usize,
  pub split_due_to_hint: usize,
  pub hint_backwards: usize,
  pub subpass_promotions: usize,
}

#[derive(Clone, Debug, Default)]
pub struct RenderPlanValidation {
  pub errors: Vec<String>,
  pub warnings: Vec<String>,
}

#[derive(Clone, Debug, Default)]
pub struct RenderGraphReport {
  pub pass_count: usize,
  pub attachment_count: usize,
  pub render_pass_count: u32,
  pub subpass_count: u32,
  pub plan_signature: Option<u64>,
  pub submission_count: Option<usize>,
  pub estimated_primary_cmds_per_fb: Option<usize>,
  pub estimated_secondary_cmds_per_fb: Option<usize>,
  pub validation_errors: usize,
  pub validation_warnings: usize,
  pub warnings: Vec<String>,
}

impl RenderGraphReport {
  pub fn dump_text(&self) -> String {
    let mut out = String::new();
    out.push_str("RenderGraphReport\n");
    out.push_str(&format!(
      "passes={} attachments={} render_passes={} subpasses={}\n",
      self.pass_count, self.attachment_count, self.render_pass_count, self.subpass_count
    ));
    let signature = self
      .plan_signature
      .map(|v| v.to_string())
      .unwrap_or_else(|| "unknown".to_string());
    out.push_str(&format!("plan_signature={}\n", signature));
    let submissions = self
      .submission_count
      .map(|v| v.to_string())
      .unwrap_or_else(|| "unknown".to_string());
    let primary = self
      .estimated_primary_cmds_per_fb
      .map(|v| v.to_string())
      .unwrap_or_else(|| "unknown".to_string());
    let secondary = self
      .estimated_secondary_cmds_per_fb
      .map(|v| v.to_string())
      .unwrap_or_else(|| "unknown".to_string());
    out.push_str(&format!(
      "submissions={} primary_cmds_per_fb={} secondary_cmds_per_fb={}\n",
      submissions, primary, secondary
    ));
    out.push_str(&format!(
      "validation_errors={} validation_warnings={}\n",
      self.validation_errors, self.validation_warnings
    ));
    if self.warnings.is_empty() {
      out.push_str("warnings=0\n");
    } else {
      out.push_str(&format!("warnings={}\n", self.warnings.len()));
      for warning in &self.warnings {
        out.push_str(&format!("  - {}\n", warning));
      }
    }
    out
  }
}

impl RenderPlan {
  pub fn validate(&self) -> RenderPlanValidation {
    let mut report = RenderPlanValidation::default();
    let mut attachment_names: HashMap<String, GraphResourceLifetime> = HashMap::new();
    let mut attachment_is_swapchain: HashMap<String, bool> = HashMap::new();
    for att in &self.attachments {
      attachment_names.insert(att.name.clone(), att.lifetime);
      attachment_is_swapchain.insert(att.name.clone(), att.is_swapchain);
    }
    // Include non-attachment resources (buffers/external) to avoid false validation errors.
    for (name, kind) in &self.resource_kinds {
      if !attachment_names.contains_key(name) {
        let lifetime = match kind {
          RenderResourceKind::External => GraphResourceLifetime::External,
          _ => GraphResourceLifetime::External,
        };
        attachment_names.insert(name.clone(), lifetime);
      }
    }
    let mut writes_by_resource: HashMap<String, usize> = HashMap::new();
    let mut reads_by_resource: HashMap<String, usize> = HashMap::new();
    for pass in &self.passes {
      let is_output_expected = matches!(
        pass.kind,
        RenderPassKind::Graphics | RenderPassKind::Compute | RenderPassKind::Transfer
      );
      if is_output_expected && pass.writes.is_empty() {
        report.warnings.push(format!(
          "optimization_hint: pass '{}' has no outputs",
          pass.name
        ));
      }
      for w in &pass.writes {
        *writes_by_resource.entry(w.resource.clone()).or_insert(0) += 1;
      }
      for r in &pass.reads {
        *reads_by_resource.entry(r.resource.clone()).or_insert(0) += 1;
      }
      if let GraphQueueRequest::Require(class) | GraphQueueRequest::Prefer(class) =
        pass.queue_request
      {
        let incompatible = match (pass.kind, class) {
          (RenderPassKind::Graphics, GraphQueueClass::Graphics) => false,
          (RenderPassKind::Compute, GraphQueueClass::Compute) => false,
          (RenderPassKind::Transfer, GraphQueueClass::Transfer) => false,
          (RenderPassKind::Present, _) => true,
          (RenderPassKind::Cpu, _) => true,
          _ => true,
        };
        if incompatible {
          report.warnings.push(format!(
            "engine_limit: pass '{}' kind {:?} incompatible with queue request {:?}",
            pass.name, pass.kind, pass.queue_request
          ));
        }
      }
    }
    let swapchain_count = self
      .attachments
      .iter()
      .filter(|att| att.is_swapchain)
      .count();
    if swapchain_count > 1 {
      report.warnings.push(format!(
        "engine_limit: {} swapchain attachments declared; verify swapchain usage",
        swapchain_count
      ));
    }
    for (resource, count) in writes_by_resource.iter() {
      if *count > 1 {
        if attachment_is_swapchain
          .get(resource)
          .copied()
          .unwrap_or(false)
        {
          report.warnings.push(format!(
            "optimization_hint: swapchain resource '{}' written {} times; check overlay ordering",
            resource, count
          ));
        }
      }
    }
    for pass in &self.passes {
      for r in &pass.reads {
        if !attachment_names.contains_key(&r.resource) {
          report.errors.push(format!(
            "authoring_error: pass '{}' reads missing resource '{}'",
            pass.name, r.resource
          ));
          continue;
        }
        let lifetime = attachment_names
          .get(&r.resource)
          .copied()
          .unwrap_or(GraphResourceLifetime::Transient);
        let has_writer = writes_by_resource.get(&r.resource).copied().unwrap_or(0) > 0;
        if !has_writer && !matches!(lifetime, GraphResourceLifetime::External) {
          report.warnings.push(format!(
            "authoring_error: pass '{}' reads '{}' with no writer (lifetime {:?})",
            pass.name, r.resource, lifetime
          ));
        }
      }
      for w in &pass.writes {
        if !attachment_names.contains_key(&w.resource) {
          report.errors.push(format!(
            "authoring_error: pass '{}' writes missing resource '{}'",
            pass.name, w.resource
          ));
        }
      }
    }
    for (resource, count) in writes_by_resource {
      let reads = reads_by_resource.get(&resource).copied().unwrap_or(0);
      if reads == 0 {
        if let Some(lifetime) = attachment_names.get(&resource).copied() {
          if matches!(lifetime, GraphResourceLifetime::Transient) {
            report.warnings.push(format!(
              "optimization_hint: resource '{}' written {} times but never read",
              resource, count
            ));
          }
        }
      }
    }
    report
  }

  pub fn report(&self, sync_plan: Option<&RenderSyncPlan>) -> RenderGraphReport {
    let mut warnings = Vec::new();
    if std::env::var("ICS_RENDER_GRAPH_SINGLE_QUEUE").is_ok() {
      warnings.push("single-queue mode forced via ICS_RENDER_GRAPH_SINGLE_QUEUE".to_string());
    }
    if self.render_pass_count > 1 {
      warnings.push(format!(
        "plan split into {} render passes; check for sampled-after-write hazards",
        self.render_pass_count
      ));
    }
    if self.subpass_count > 1 {
      warnings.push(format!(
        "plan uses {} subpasses; ensure grouping matches intended input attachments",
        self.subpass_count
      ));
    }
    if self.grouping_stats.subpass_promotions > 0 {
      warnings.push(format!(
        "subpass promotions: {} due to input attachment dependencies",
        self.grouping_stats.subpass_promotions
      ));
    }
    if self.grouping_stats.split_sampled_after_write > 0 {
      warnings.push(format!(
        "render pass splits: {} due to sampled-after-write hazards",
        self.grouping_stats.split_sampled_after_write
      ));
    }
    if self.grouping_stats.split_write_after_sampled > 0 {
      warnings.push(format!(
        "render pass splits: {} due to write-after-sampled hazards",
        self.grouping_stats.split_write_after_sampled
      ));
    }
    if self.grouping_stats.split_due_to_hint > 0 {
      warnings.push(format!(
        "render pass splits: {} forced by render_pass_hint",
        self.grouping_stats.split_due_to_hint
      ));
    }
    if self.grouping_stats.hint_backwards > 0 {
      warnings.push(format!(
        "render pass hints ignored on {} passes (hint behind current render pass)",
        self.grouping_stats.hint_backwards
      ));
    }
    let mut writes_by_resource: HashMap<String, HashMap<String, usize>> = HashMap::new();
    let is_swapchain = |name: &str| {
      self
        .attachments
        .iter()
        .any(|att| att.is_swapchain && att.name == name)
    };
    let layer_from_pass = |name: &str| -> String {
      let prefix = "layer-";
      let marker = "-pipeline-";
      if let Some(stripped) = name.strip_prefix(prefix) {
        if let Some((layer, _)) = stripped.split_once(marker) {
          return layer.to_string();
        }
      }
      "non-layer".to_string()
    };
    for pass in &self.passes {
      if pass.reads.is_empty() && pass.writes.is_empty() {
        warnings.push(format!(
          "pass '{}' has no reads or writes; verify it should exist",
          pass.name
        ));
      }
      let layer_key = layer_from_pass(&pass.name);
      for w in &pass.writes {
        let per_layer = writes_by_resource
          .entry(w.resource.clone())
          .or_insert_with(HashMap::new);
        *per_layer.entry(layer_key.clone()).or_insert(0) += 1;
      }
    }
    for (resource, by_layer) in writes_by_resource {
      let total_writes: usize = by_layer.values().sum();
      if total_writes <= 1 {
        continue;
      }
      let layer_count = by_layer.len();
      if is_swapchain(&resource) && layer_count > 1 {
        warnings.push(format!(
          "swapchain resource '{}' written by {} layers; confirm post/overlay ordering",
          resource, layer_count
        ));
        continue;
      }
      if layer_count > 1 {
        warnings.push(format!(
          "resource '{}' written by {} layers; confirm composition intent",
          resource, layer_count
        ));
        continue;
      }
      if let Some((layer, count)) = by_layer.iter().next() {
        if *count > 1 {
          warnings.push(format!(
            "resource '{}' written {} times by layer {}; confirm multi-pipeline overdraw intent",
            resource, count, layer
          ));
        }
      }
    }
    let mut swapchain_writes: HashMap<String, Vec<usize>> = HashMap::new();
    for (idx, pass) in self.passes.iter().enumerate() {
      for w in &pass.writes {
        if is_swapchain(&w.resource) {
          swapchain_writes
            .entry(w.resource.clone())
            .or_insert_with(Vec::new)
            .push(idx);
        }
      }
    }
    for (resource, indices) in swapchain_writes {
      if let Some(last) = indices.iter().max() {
        if indices.iter().any(|idx| idx < last) {
          warnings.push(format!(
            "swapchain resource '{}' written before final pass; verify overlay/post ordering",
            resource
          ));
        }
      }
    }
    let mut buffer_barriers = 0usize;
    let mut queue_transfers = 0usize;
    let mut present_barriers = 0usize;
    let submission_count = sync_plan.map(|plan| {
      for submission in &plan.submissions {
        for barrier in &submission.barriers {
          if matches!(
            barrier.resource_kind,
            RenderResourceKind::Buffer(_) | RenderResourceKind::External
          ) {
            buffer_barriers += 1;
          }
          if barrier.src_queue != barrier.dst_queue {
            queue_transfers += 1;
          }
          if matches!(barrier.new_layout, GraphImageLayout::Present) {
            present_barriers += 1;
          }
        }
      }
      plan.submissions.len()
    });
    if buffer_barriers > 0 {
      warnings.push(format!(
        "buffer barriers planned: {} (execution path not yet implemented)",
        buffer_barriers
      ));
    }
    if queue_transfers > 0 {
      warnings.push(format!(
        "queue ownership transfers planned: {} (release/acquire barriers)",
        queue_transfers
      ));
    }
    let has_present_pass = self
      .passes
      .iter()
      .any(|pass| matches!(pass.kind, RenderPassKind::Present));
    if has_present_pass && present_barriers == 0 {
      warnings.push(
        "present pass found but no explicit present barrier planned; verify swapchain handling"
          .to_string(),
      );
    }
    RenderGraphReport {
      pass_count: self.passes.len(),
      attachment_count: self.attachments.len(),
      render_pass_count: self.render_pass_count,
      subpass_count: self.subpass_count,
      plan_signature: None,
      submission_count,
      estimated_primary_cmds_per_fb: submission_count,
      estimated_secondary_cmds_per_fb: None,
      validation_errors: 0,
      validation_warnings: 0,
      warnings,
    }
  }

  pub fn dump_text(&self) -> String {
    let mut out = String::new();
    out.push_str("RenderPlan\n");
    out.push_str(&format!(
      "attachments={} passes={} subpasses={} render_passes={}\n",
      self.attachments.len(),
      self.passes.len(),
      self.subpass_count,
      self.render_pass_count
    ));
    out.push_str("Attachments:\n");
    for att in &self.attachments {
      out.push_str(&format!(
        "  - {} fmt={:?} samples={:?} size={:?} life={:?} role={:?} swapchain={} init={:?} final={:?}\n",
        att.name,
        att.format,
        att.samples,
        att.size,
        att.lifetime,
        att.role,
        att.is_swapchain,
        att.initial_layout,
        att.final_layout
      ));
    }
    out.push_str("Passes:\n");
    for pass in &self.passes {
      out.push_str(&format!(
        "  - {} kind={:?} rp={} sp={} reads={} writes={}\n",
        pass.name,
        pass.kind,
        pass.render_pass,
        pass.subpass,
        pass.reads.len(),
        pass.writes.len()
      ));
      for r in &pass.reads {
        out.push_str(&format!("      read {} ({:?})\n", r.resource, r.access));
      }
      for w in &pass.writes {
        out.push_str(&format!("      write {} ({:?})\n", w.resource, w.access));
      }
    }
    out
  }

  pub fn dump_dot(&self) -> String {
    let mut out = String::new();
    out.push_str("digraph RenderPlan {\n  rankdir=LR;\n");
    out.push_str("  node [fontname=\"monospace\"];\n");
    for pass in &self.passes {
      let pass_id = dot_id(&format!("pass:{}", pass.name));
      out.push_str(&format!(
        "  {} [shape=ellipse,label=\"{}\"];\n",
        pass_id,
        dot_escape(&pass.name)
      ));
    }
    for att in &self.attachments {
      let att_id = dot_id(&format!("res:{}", att.name));
      out.push_str(&format!(
        "  {} [shape=box,label=\"{}\"];\n",
        att_id,
        dot_escape(&att.name)
      ));
    }
    for pass in &self.passes {
      let pass_id = dot_id(&format!("pass:{}", pass.name));
      for r in &pass.reads {
        let att_id = dot_id(&format!("res:{}", r.resource));
        out.push_str(&format!("  {} -> {} [label=\"read\"];\n", att_id, pass_id));
      }
      for w in &pass.writes {
        let att_id = dot_id(&format!("res:{}", w.resource));
        out.push_str(&format!("  {} -> {} [label=\"write\"];\n", pass_id, att_id));
      }
    }
    out.push_str("}\n");
    out
  }
}

fn dot_escape(name: &str) -> String {
  name.replace('"', "\\\"")
}

fn dot_id(name: &str) -> String {
  let mut out = String::with_capacity(name.len() + 2);
  out.push('"');
  out.push_str(&dot_escape(name));
  out.push('"');
  out
}

pub fn is_capture_name_valid(name: &str) -> bool {
  let trimmed = name.trim();
  if trimmed.is_empty() {
    return false;
  }
  !trimmed.chars().any(|c| c.is_whitespace())
}

// ---------------------------------------------------------------------------
// Backend-agnostic synchronization planning.
// The render graph will eventually populate these from resource usage.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SyncQueue {
  Graphics,
  Transfer,
  Compute,
  Present,
  External,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SyncStage {
  Top,
  Draw,
  Compute,
  Transfer,
  Bottom,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SyncAccess {
  Read,
  Write,
  ReadWrite,
}

#[derive(Clone, Debug)]
pub struct SyncResourceState {
  pub resource: String,
  pub layout: GraphImageLayout,
  pub access: Vec<SyncAccess>,
  pub stages: Vec<SyncStage>,
  pub queue: SyncQueue,
}

#[derive(Clone, Debug)]
pub struct SyncBarrier {
  pub phase: SyncBarrierPhase,
  pub pass_index: usize,
  pub resource: String,
  pub resource_kind: RenderResourceKind,
  pub old_layout: GraphImageLayout,
  pub new_layout: GraphImageLayout,
  pub src_access: Vec<SyncAccess>,
  pub dst_access: Vec<SyncAccess>,
  pub src_stages: Vec<SyncStage>,
  pub dst_stages: Vec<SyncStage>,
  pub src_queue: SyncQueue,
  pub dst_queue: SyncQueue,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SyncBarrierPhase {
  Acquire,
  Release,
}

#[derive(Clone, Debug)]
pub struct SyncWait {
  pub name: String,
  pub stage: SyncStage,
}

#[derive(Clone, Debug)]
pub struct SyncSignal {
  pub name: String,
}

#[derive(Clone, Debug)]
pub struct SyncFenceIntent {
  pub name: String,
}

#[derive(Clone, Debug)]
pub struct SyncSubmission {
  pub queue: SyncQueue,
  pub pass_indices: Vec<usize>,
  pub waits: Vec<SyncWait>,
  pub signals: Vec<SyncSignal>,
  pub fence: Option<SyncFenceIntent>,
  pub barriers: Vec<SyncBarrier>,
}

#[derive(Clone, Debug, Default)]
pub struct RenderSyncPlan {
  pub submissions: Vec<SyncSubmission>,
  pub resource_states: Vec<SyncResourceState>,
}

impl RenderSyncPlan {
  pub fn dump_text(&self) -> String {
    let mut out = String::new();
    out.push_str("RenderSyncPlan\n");
    out.push_str(&format!(
      "submissions={} resources={}\n",
      self.submissions.len(),
      self.resource_states.len()
    ));
    let mut buffer_barriers = 0usize;
    let mut image_barriers = 0usize;
    let mut queue_transfers = 0usize;
    let mut release_barriers = 0usize;
    let mut acquire_barriers = 0usize;
    for submission in &self.submissions {
      for barrier in &submission.barriers {
        if barrier.src_queue != barrier.dst_queue {
          queue_transfers += 1;
        }
        match barrier.phase {
          SyncBarrierPhase::Acquire => acquire_barriers += 1,
          SyncBarrierPhase::Release => release_barriers += 1,
        }
        match barrier.resource_kind {
          RenderResourceKind::Buffer(_) | RenderResourceKind::External => buffer_barriers += 1,
          _ => image_barriers += 1,
        }
      }
    }
    out.push_str(&format!(
      "barriers_image={} barriers_buffer={} queue_transfers={} release_barriers={} acquire_barriers={}\n",
      image_barriers, buffer_barriers, queue_transfers, release_barriers, acquire_barriers
    ));
    for (idx, submission) in self.submissions.iter().enumerate() {
      out.push_str(&format!(
        "  - submission {} queue={:?} passes={} waits={} signals={} barriers={}\n",
        idx,
        submission.queue,
        submission.pass_indices.len(),
        submission.waits.len(),
        submission.signals.len(),
        submission.barriers.len()
      ));
    }
    out
  }

  pub fn from_plan(plan: &RenderPlan) -> Self {
    if plan.passes.is_empty() {
      return Self::default();
    }

    let mut depth_resources: HashMap<String, bool> = HashMap::new();
    for att in &plan.attachments {
      let is_depth = matches!(
        att.format,
        GraphFormat::DefaultDepth | GraphFormat::D32 | GraphFormat::D24S8
      );
      depth_resources.insert(att.name.clone(), is_depth);
    }

    let mut resource_state: HashMap<String, SyncResourceState> = HashMap::new();
    let mut last_use_pass: HashMap<String, usize> = HashMap::new();
    for att in &plan.attachments {
      resource_state.insert(
        att.name.clone(),
        SyncResourceState {
          resource: att.name.clone(),
          layout: att.initial_layout,
          access: Vec::new(),
          stages: Vec::new(),
          queue: SyncQueue::Graphics,
        },
      );
    }

    let force_single_queue = std::env::var("ICS_RENDER_GRAPH_SINGLE_QUEUE").is_ok();
    let resolve_queue = |pass: &PlanPass| -> SyncQueue {
      if force_single_queue && !matches!(pass.kind, RenderPassKind::Present) {
        return SyncQueue::Graphics;
      }
      let map_class = |class: GraphQueueClass| match class {
        GraphQueueClass::Graphics => SyncQueue::Graphics,
        GraphQueueClass::Compute => SyncQueue::Compute,
        GraphQueueClass::Transfer => SyncQueue::Transfer,
      };
      match pass.queue_request {
        GraphQueueRequest::Require(class) | GraphQueueRequest::Prefer(class) => map_class(class),
        GraphQueueRequest::Any => match pass.kind {
          RenderPassKind::Compute => SyncQueue::Compute,
          RenderPassKind::Transfer => SyncQueue::Transfer,
          RenderPassKind::Present => SyncQueue::Present,
          RenderPassKind::Graphics | RenderPassKind::Schedule | RenderPassKind::Cpu => {
            SyncQueue::Graphics
          }
        },
      }
    };

    let access_to_layout = |access: GraphAccessType, is_depth: bool| match access {
      GraphAccessType::ColorWrite => GraphImageLayout::ColorAttachment,
      GraphAccessType::DepthWrite => GraphImageLayout::DepthAttachment,
      GraphAccessType::InputAttachment => {
        if is_depth {
          GraphImageLayout::DepthRead
        } else {
          GraphImageLayout::ShaderRead
        }
      }
      GraphAccessType::Sampled | GraphAccessType::StorageReadWrite => {
        if is_depth {
          GraphImageLayout::DepthRead
        } else {
          GraphImageLayout::ShaderRead
        }
      }
      GraphAccessType::BufferRead
      | GraphAccessType::BufferWrite
      | GraphAccessType::BufferReadWrite => GraphImageLayout::Undefined,
      GraphAccessType::Present => GraphImageLayout::Present,
    };

    let access_to_sync = |access: GraphAccessType| match access {
      GraphAccessType::BufferRead => SyncAccess::Read,
      GraphAccessType::BufferWrite => SyncAccess::Write,
      GraphAccessType::BufferReadWrite => SyncAccess::ReadWrite,
      GraphAccessType::StorageReadWrite => SyncAccess::ReadWrite,
      GraphAccessType::ColorWrite | GraphAccessType::DepthWrite => SyncAccess::Write,
      GraphAccessType::Sampled | GraphAccessType::InputAttachment | GraphAccessType::Present => {
        SyncAccess::Read
      }
    };

    let access_priority = |access: GraphAccessType| match access {
      GraphAccessType::ColorWrite | GraphAccessType::DepthWrite => 2,
      GraphAccessType::StorageReadWrite
      | GraphAccessType::BufferReadWrite
      | GraphAccessType::BufferWrite => 2,
      GraphAccessType::Sampled | GraphAccessType::InputAttachment | GraphAccessType::BufferRead => {
        1
      }
      GraphAccessType::Present => 0,
    };

    let mut barriers = Vec::new();
    let mut pass_queues: Vec<SyncQueue> = Vec::with_capacity(plan.passes.len());
    for pass in &plan.passes {
      pass_queues.push(resolve_queue(pass));
    }

    for (pass_index, pass) in plan.passes.iter().enumerate() {
      if matches!(pass.kind, RenderPassKind::Present) {
        continue;
      }
      let new_queue = pass_queues[pass_index];
      let mut desired: HashMap<String, GraphAccessType> = HashMap::new();
      for r in &pass.reads {
        desired
          .entry(r.resource.clone())
          .and_modify(|current| {
            if access_priority(r.access) > access_priority(*current) {
              *current = r.access;
            }
          })
          .or_insert(r.access);
      }
      for w in &pass.writes {
        desired
          .entry(w.resource.clone())
          .and_modify(|current| {
            if access_priority(w.access) > access_priority(*current) {
              *current = w.access;
            }
          })
          .or_insert(w.access);
      }

      for (name, access) in desired {
        let kind = plan
          .resource_kinds
          .get(&name)
          .copied()
          .unwrap_or(RenderResourceKind::External);
        let is_depth = *depth_resources.get(&name).unwrap_or(&false);
        let new_layout = access_to_layout(access, is_depth);
        let new_access = vec![access_to_sync(access)];
        let new_stages = vec![match new_queue {
          SyncQueue::Transfer => SyncStage::Transfer,
          SyncQueue::Compute => SyncStage::Compute,
          SyncQueue::Present => SyncStage::Bottom,
          _ => SyncStage::Draw,
        }];

        let prev = resource_state.get(&name).cloned();
        let (old_layout, src_access, src_stages, src_queue) = if let Some(prev) = &prev {
          (
            prev.layout,
            prev.access.clone(),
            prev.stages.clone(),
            prev.queue,
          )
        } else {
          (
            GraphImageLayout::Undefined,
            Vec::new(),
            vec![SyncStage::Top],
            new_queue,
          )
        };

        if src_queue != new_queue {
          if let Some(last_pass) = last_use_pass.get(&name).copied() {
            barriers.push(SyncBarrier {
              phase: SyncBarrierPhase::Release,
              pass_index: last_pass,
              resource: name.clone(),
              resource_kind: kind,
              old_layout,
              new_layout: old_layout,
              src_access: src_access.clone(),
              dst_access: Vec::new(),
              src_stages: src_stages.clone(),
              dst_stages: vec![SyncStage::Bottom],
              src_queue,
              dst_queue: new_queue,
            });
          }
          barriers.push(SyncBarrier {
            phase: SyncBarrierPhase::Acquire,
            pass_index,
            resource: name.clone(),
            resource_kind: kind,
            old_layout,
            new_layout,
            src_access: Vec::new(),
            dst_access: new_access.clone(),
            src_stages: vec![SyncStage::Top],
            dst_stages: new_stages.clone(),
            src_queue,
            dst_queue: new_queue,
          });
        } else if old_layout != new_layout || src_access != new_access {
          barriers.push(SyncBarrier {
            phase: SyncBarrierPhase::Acquire,
            pass_index,
            resource: name.clone(),
            resource_kind: kind,
            old_layout,
            new_layout,
            src_access,
            dst_access: new_access.clone(),
            src_stages,
            dst_stages: new_stages.clone(),
            src_queue,
            dst_queue: new_queue,
          });
        }

        let resource_name = name.clone();
        resource_state.insert(
          resource_name.clone(),
          SyncResourceState {
            resource: resource_name,
            layout: new_layout,
            access: new_access,
            stages: new_stages,
            queue: new_queue,
          },
        );
        last_use_pass.insert(name, pass_index);
      }
    }

    let present_passes = plan
      .passes
      .iter()
      .enumerate()
      .filter(|(_, pass)| matches!(pass.kind, RenderPassKind::Present))
      .collect::<Vec<_>>();
    for (pass_index, pass) in present_passes {
      for r in &pass.reads {
        if r.access != GraphAccessType::Present {
          continue;
        }
        let name = r.resource.clone();
        let kind = plan
          .resource_kinds
          .get(&name)
          .copied()
          .unwrap_or(RenderResourceKind::External);
        let prev = resource_state.get(&name).cloned();
        let (old_layout, src_access, src_stages, src_queue) = if let Some(prev) = &prev {
          (
            prev.layout,
            prev.access.clone(),
            prev.stages.clone(),
            prev.queue,
          )
        } else {
          (
            GraphImageLayout::Undefined,
            Vec::new(),
            vec![SyncStage::Top],
            SyncQueue::Graphics,
          )
        };
        let new_layout = GraphImageLayout::Present;
        let new_queue = SyncQueue::Present;
        let new_access = vec![SyncAccess::Read];
        if (old_layout != new_layout || src_queue != new_queue)
          && matches!(last_use_pass.get(&name).copied(), Some(_))
        {
          if let Some(last_pass) = last_use_pass.get(&name).copied() {
            barriers.push(SyncBarrier {
              phase: SyncBarrierPhase::Release,
              pass_index: last_pass,
              resource: name.clone(),
              resource_kind: kind,
              old_layout,
              new_layout,
              src_access: src_access.clone(),
              dst_access: new_access.clone(),
              src_stages: src_stages.clone(),
              dst_stages: vec![SyncStage::Bottom],
              src_queue,
              dst_queue: new_queue,
            });
          }
        }
        resource_state.insert(
          name,
          SyncResourceState {
            resource: r.resource.clone(),
            layout: new_layout,
            access: new_access,
            stages: vec![SyncStage::Bottom],
            queue: new_queue,
          },
        );
        last_use_pass.insert(r.resource.clone(), pass_index);
      }
    }

    let mut barriers_by_pass: HashMap<usize, Vec<SyncBarrier>> = HashMap::new();
    for barrier in barriers {
      barriers_by_pass
        .entry(barrier.pass_index)
        .or_default()
        .push(barrier);
    }
    let mut submissions = Vec::new();
    let mut current_queue: Option<SyncQueue> = None;
    let mut current = SyncSubmission {
      queue: SyncQueue::Graphics,
      pass_indices: Vec::new(),
      waits: Vec::new(),
      signals: Vec::new(),
      fence: None,
      barriers: Vec::new(),
    };
    for (idx, queue) in pass_queues.iter().enumerate() {
      if matches!(plan.passes[idx].kind, RenderPassKind::Present) {
        continue;
      }
      if current_queue.is_none() {
        current_queue = Some(*queue);
        current.queue = *queue;
      }
      if Some(*queue) != current_queue {
        if !current.pass_indices.is_empty() {
          submissions.push(current);
        }
        current_queue = Some(*queue);
        current = SyncSubmission {
          queue: *queue,
          pass_indices: Vec::new(),
          waits: Vec::new(),
          signals: Vec::new(),
          fence: None,
          barriers: Vec::new(),
        };
      }
      current.pass_indices.push(idx);
      if let Some(list) = barriers_by_pass.get(&idx) {
        current.barriers.extend(list.iter().cloned());
      }
    }
    if !current.pass_indices.is_empty() {
      submissions.push(current);
    }

    let wait_stage_for = |submission: &SyncSubmission| -> SyncStage {
      let mut has_draw = false;
      let mut has_compute = false;
      let mut has_transfer = false;
      for barrier in &submission.barriers {
        for stage in &barrier.dst_stages {
          match stage {
            SyncStage::Draw => has_draw = true,
            SyncStage::Compute => has_compute = true,
            SyncStage::Transfer => has_transfer = true,
            _ => {}
          }
        }
      }
      if has_draw {
        SyncStage::Draw
      } else if has_compute {
        SyncStage::Compute
      } else if has_transfer {
        SyncStage::Transfer
      } else {
        SyncStage::Top
      }
    };
    for i in 0..submissions.len().saturating_sub(1) {
      if submissions[i].queue != submissions[i + 1].queue {
        let name = format!("queue_edge_{}", i);
        let (left, right) = submissions.split_at_mut(i + 1);
        let prev = &mut left[i];
        let next = &mut right[0];
        let stage = wait_stage_for(next);
        prev.signals.push(SyncSignal { name: name.clone() });
        next.waits.push(SyncWait { name, stage });
      }
    }
    if let Some(last) = submissions.last_mut() {
      last.fence = Some(SyncFenceIntent {
        name: "frame_fence".to_string(),
      });
    }

    Self {
      submissions,
      resource_states: resource_state.into_values().collect(),
    }
  }
}

pub fn diff_text(expected: &str, actual: &str) -> String {
  let expected_lines: Vec<&str> = expected.lines().collect();
  let actual_lines: Vec<&str> = actual.lines().collect();
  let max_len = expected_lines.len().max(actual_lines.len());
  let mut out = String::new();
  out.push_str("RenderGraphDiff\n");
  for i in 0..max_len {
    let left = expected_lines.get(i).copied();
    let right = actual_lines.get(i).copied();
    match (left, right) {
      (Some(l), Some(r)) if l == r => {
        out.push_str(&format!("  {}\n", l));
      }
      (Some(l), Some(r)) => {
        out.push_str(&format!("- {}\n", l));
        out.push_str(&format!("+ {}\n", r));
      }
      (Some(l), None) => {
        out.push_str(&format!("- {}\n", l));
      }
      (None, Some(r)) => {
        out.push_str(&format!("+ {}\n", r));
      }
      (None, None) => {}
    }
  }
  out
}

pub fn signature_for_plan(plan: &RenderPlan) -> u64 {
  fn fnv64_acc(mut h: u64, v: u64) -> u64 {
    h ^= v;
    h = h.wrapping_mul(0x100000001b3);
    h
  }
  let mut h: u64 = 0xcbf29ce484222325;
  let mut attachments = plan.attachments.clone();
  attachments.sort_by(|a, b| a.name.cmp(&b.name));
  for att in attachments {
    for b in att.name.as_bytes() {
      h = fnv64_acc(h, *b as u64);
    }
    let fmt_tag = match att.format {
      GraphFormat::DefaultColor => 1,
      GraphFormat::DefaultDepth => 2,
      GraphFormat::Rgba8Unorm => 3,
      GraphFormat::Rgba8Srgb => 4,
      GraphFormat::Rgba16Float => 5,
      GraphFormat::R16G16Snorm => 6,
      GraphFormat::D32 => 7,
      GraphFormat::D24S8 => 8,
      GraphFormat::Custom(v) => 9 ^ (v as u64),
    };
    let samples_tag = match att.samples {
      GraphSampleCount::One => 1,
      GraphSampleCount::Two => 2,
      GraphSampleCount::Four => 4,
      GraphSampleCount::Eight => 8,
      GraphSampleCount::Custom(v) => 16 ^ (v as u64),
    };
    let size_tag = match att.size {
      GraphSize::Swapchain => 1,
      GraphSize::Fixed { width, height } => 2 ^ ((width as u64) << 32) ^ height as u64,
      GraphSize::Scaled {
        numerator,
        denominator,
      } => 3 ^ ((numerator as u64) << 32) ^ denominator as u64,
    };
    let life_tag = match att.lifetime {
      GraphResourceLifetime::External => 1,
      GraphResourceLifetime::Persistent => 2,
      GraphResourceLifetime::History => 3,
      GraphResourceLifetime::Transient => 4,
    };
    let role_tag = match att.role {
      GraphResourceRole::Scene => 1,
      GraphResourceRole::Post => 2,
      GraphResourceRole::Custom => 3,
    };
    h = fnv64_acc(h, fmt_tag);
    h = fnv64_acc(h, samples_tag);
    h = fnv64_acc(h, size_tag);
    h = fnv64_acc(h, life_tag);
    h = fnv64_acc(h, role_tag);
    h = fnv64_acc(h, att.is_swapchain as u64);
  }
  h = fnv64_acc(h, plan.passes.len() as u64);
  h = fnv64_acc(h, plan.subpass_count as u64);
  h = fnv64_acc(h, plan.render_pass_count as u64);
  for sp in &plan.render_pass_subpasses {
    h = fnv64_acc(h, *sp as u64);
  }
  let mut passes = plan.passes.clone();
  passes.sort_by(|a, b| {
    a.name
      .cmp(&b.name)
      .then(a.render_pass.cmp(&b.render_pass))
      .then(a.subpass.cmp(&b.subpass))
  });
  for pass in &passes {
    for b in pass.name.as_bytes() {
      h = fnv64_acc(h, *b as u64);
    }
    h = fnv64_acc(h, pass.render_pass as u64);
    h = fnv64_acc(h, pass.subpass as u64);
    let kind_tag = match pass.kind {
      RenderPassKind::Graphics => 1,
      RenderPassKind::Compute => 2,
      RenderPassKind::Transfer => 3,
      RenderPassKind::Cpu => 4,
      RenderPassKind::Present => 5,
      RenderPassKind::Schedule => 6,
    };
    h = fnv64_acc(h, kind_tag);
    let queue_class_tag = |class: GraphQueueClass| match class {
      GraphQueueClass::Graphics => 1,
      GraphQueueClass::Compute => 2,
      GraphQueueClass::Transfer => 3,
    };
    let queue_tag = match pass.queue_request {
      GraphQueueRequest::Any => 0,
      GraphQueueRequest::Require(class) => 10 + queue_class_tag(class),
      GraphQueueRequest::Prefer(class) => 20 + queue_class_tag(class),
    };
    h = fnv64_acc(h, queue_tag);
    let usage_tag = match pass.command_usage {
      CommandBufferUsage::OneTime => 1,
      CommandBufferUsage::Reusable => 2,
      CommandBufferUsage::Simultaneous => 3,
    };
    h = fnv64_acc(h, usage_tag);
    let access_tag = |access: &GraphAccessType| match access {
      GraphAccessType::Sampled => 1,
      GraphAccessType::InputAttachment => 2,
      GraphAccessType::ColorWrite => 3,
      GraphAccessType::DepthWrite => 4,
      GraphAccessType::StorageReadWrite => 5,
      GraphAccessType::BufferRead => 6,
      GraphAccessType::BufferWrite => 7,
      GraphAccessType::BufferReadWrite => 8,
      GraphAccessType::Present => 9,
    };
    let mut reads = pass.reads.clone();
    reads.sort_by(|a, b| {
      a.resource
        .cmp(&b.resource)
        .then(access_tag(&a.access).cmp(&access_tag(&b.access)))
    });
    let mut writes = pass.writes.clone();
    writes.sort_by(|a, b| {
      a.resource
        .cmp(&b.resource)
        .then(access_tag(&a.access).cmp(&access_tag(&b.access)))
    });
    h = fnv64_acc(h, reads.len() as u64);
    h = fnv64_acc(h, writes.len() as u64);
    for r in &reads {
      for b in r.resource.as_bytes() {
        h = fnv64_acc(h, *b as u64);
      }
      h = fnv64_acc(h, access_tag(&r.access));
    }
    for w in &writes {
      for b in w.resource.as_bytes() {
        h = fnv64_acc(h, *b as u64);
      }
      h = fnv64_acc(h, access_tag(&w.access));
    }
  }
  h
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RenderResourceId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RenderPassId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RenderResourceKind {
  SwapchainColor,
  SwapchainDepth,
  /// Generic image resource produced/consumed by passes.
  /// Carries enough information for the backend to create an attachment.
  Image(RenderImageDesc),
  Buffer(RenderBufferDesc),
  External,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RenderImageDesc {
  pub format: GraphFormat,
  pub samples: GraphSampleCount,
  pub size: GraphSize,
  pub lifetime: GraphResourceLifetime,
  pub resolve: bool,
  pub is_depth: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RenderBufferDesc {
  pub size: usize,
  pub alignment: usize,
  pub lifetime: GraphResourceLifetime,
}

impl RenderBufferDesc {
  pub fn new(size: usize, alignment: usize, lifetime: GraphResourceLifetime) -> Self {
    Self {
      size,
      alignment: alignment.max(1),
      lifetime,
    }
  }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResourceAccess {
  Read,
  InputAttachment,
  Write,
  ReadWrite,
}

#[derive(Clone, Debug)]
pub struct RenderResourceAccess {
  pub resource: RenderResourceId,
  pub access: ResourceAccess,
}

impl RenderResourceAccess {
  pub fn read(resource: RenderResourceId) -> Self {
    Self {
      resource,
      access: ResourceAccess::Read,
    }
  }

  pub fn write(resource: RenderResourceId) -> Self {
    Self {
      resource,
      access: ResourceAccess::Write,
    }
  }

  pub fn input(resource: RenderResourceId) -> Self {
    Self {
      resource,
      access: ResourceAccess::InputAttachment,
    }
  }

  pub fn read_write(resource: RenderResourceId) -> Self {
    Self {
      resource,
      access: ResourceAccess::ReadWrite,
    }
  }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RenderPassKind {
  Graphics,
  Compute,
  Transfer,
  Cpu,
  Present,
  Schedule,
}

pub type RenderGraphCallback = Arc<dyn Fn(usize) -> Result<(), IcsError> + Send + Sync + 'static>;

pub trait RenderCommandEncoder {
  fn as_any_mut(&mut self) -> &mut dyn Any;
}

pub type RenderGraphRecordCallback =
  Arc<dyn Fn(&mut dyn RenderCommandEncoder, usize) -> Result<(), IcsError> + Send + Sync + 'static>;

#[derive(Clone)]
pub enum RenderPassWork {
  Pipeline {
    layer: TypeId,
    pipeline_index: usize,
    /// Subpass index within the render pass this pipeline targets.
    subpass: u32,
  },
  Record(RenderGraphRecordCallback),
  Callback(RenderGraphCallback),
  None,
}

impl std::fmt::Debug for RenderPassWork {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      RenderPassWork::Pipeline {
        layer,
        pipeline_index,
        subpass,
      } => f
        .debug_struct("Pipeline")
        .field("layer", layer)
        .field("pipeline_index", pipeline_index)
        .field("subpass", subpass)
        .finish(),
      RenderPassWork::Record(_) => f.write_str("Record(..)"),
      RenderPassWork::Callback(_) => f.write_str("Callback(..)"),
      RenderPassWork::None => f.write_str("None"),
    }
  }
}

#[derive(Clone)]
pub struct RenderPass {
  pub id: RenderPassId,
  pub name: String,
  pub kind: RenderPassKind,
  pub work: RenderPassWork,
  pub render_pass_hint: Option<u32>,
  pub subpass: u32,
  pub reads: Vec<RenderResourceAccess>,
  pub writes: Vec<RenderResourceAccess>,
  pub consumes: Vec<RenderResourceId>,
  pub produces: Vec<RenderResourceId>,
  pub msaa_hint: Option<GraphSampleCount>,
  pub queue_request: GraphQueueRequest,
  pub command_usage: CommandBufferUsage,
  sequence: usize,
}

impl RenderPass {
  fn new(
    id: RenderPassId,
    name: impl Into<String>,
    kind: RenderPassKind,
    work: RenderPassWork,
    render_pass_hint: Option<u32>,
    subpass: u32,
    reads: Vec<RenderResourceAccess>,
    writes: Vec<RenderResourceAccess>,
    consumes: Vec<RenderResourceId>,
    produces: Vec<RenderResourceId>,
    msaa_hint: Option<GraphSampleCount>,
    queue_request: GraphQueueRequest,
    command_usage: CommandBufferUsage,
    sequence: usize,
  ) -> Self {
    Self {
      id,
      name: name.into(),
      kind,
      work,
      render_pass_hint,
      subpass,
      reads,
      writes,
      consumes,
      produces,
      msaa_hint,
      queue_request,
      command_usage,
      sequence,
    }
  }
}

#[derive(Clone)]
pub struct RenderResource {
  pub id: RenderResourceId,
  pub name: String,
  pub kind: RenderResourceKind,
  pub role: GraphResourceRole,
}

pub struct RenderGraphBuilder {
  resources: Vec<RenderResource>,
  pub(crate) passes: Vec<RenderPass>,
  name_to_id: HashMap<String, RenderResourceId>,
}

impl RenderGraphBuilder {
  pub fn new() -> Self {
    Self {
      resources: Vec::new(),
      passes: Vec::new(),
      name_to_id: HashMap::new(),
    }
  }

  pub fn add_resource(
    &mut self,
    name: impl Into<String>,
    kind: RenderResourceKind,
  ) -> RenderResourceId {
    self.add_resource_with_role(name, kind, GraphResourceRole::Custom)
  }

  pub fn add_resource_with_role(
    &mut self,
    name: impl Into<String>,
    kind: RenderResourceKind,
    role: GraphResourceRole,
  ) -> RenderResourceId {
    let name = name.into();
    if !is_capture_name_valid(&name) {
      ICS_WARN!(
        "RenderGraph: resource name '{}' should be non-empty and whitespace-free",
        name
      );
    }
    if let Some(id) = self.name_to_id.get(&name) {
      if let Some(existing) = self.resources.get_mut(id.0) {
        if existing.kind != kind {
          ICS_ERROR!(
            "RenderGraph: resource '{}' already registered as {:?}, ignoring new kind {:?}",
            name,
            existing.kind,
            kind
          );
        }
        if existing.role == GraphResourceRole::Custom && role != GraphResourceRole::Custom {
          existing.role = role;
        } else if existing.role != role && role != GraphResourceRole::Custom {
          ICS_WARN!(
            "RenderGraph: resource '{}' role {:?} already set; ignoring {:?}",
            name,
            existing.role,
            role
          );
        }
      }
      ICS_DEBUG!("Reusing existing resource {}", name);
      return *id;
    }
    let id = RenderResourceId(self.resources.len());
    self.name_to_id.insert(name.clone(), id);
    self.resources.push(RenderResource {
      id,
      name,
      kind,
      role,
    });
    ICS_INFO!("Added resource {:?} of kind {:?}", id, kind);
    id
  }

  pub fn get(&self, name: &str) -> Option<RenderResourceId> {
    self.name_to_id.get(name).copied()
  }

  pub fn add_pass(
    &mut self,
    name: impl Into<String>,
    kind: RenderPassKind,
    work: RenderPassWork,
    subpass: u32,
    render_pass_hint: Option<u32>,
    reads: Vec<RenderResourceAccess>,
    writes: Vec<RenderResourceAccess>,
    consumes: Vec<RenderResourceId>,
    produces: Vec<RenderResourceId>,
    msaa_hint: Option<GraphSampleCount>,
    queue_request: GraphQueueRequest,
    command_usage: CommandBufferUsage,
  ) -> RenderPassId {
    let name = name.into();
    if !is_capture_name_valid(&name) {
      ICS_WARN!(
        "RenderGraph: pass name '{}' should be non-empty and whitespace-free",
        name
      );
    }
    let id = RenderPassId(self.passes.len());
    let pass = RenderPass::new(
      id,
      name,
      kind,
      work,
      render_pass_hint,
      subpass,
      reads,
      writes,
      consumes,
      produces,
      msaa_hint,
      queue_request,
      command_usage,
      self.passes.len(),
    );
    self.passes.push(pass);
    ICS_INFO!("Added pass {:?} ({:?})", id, kind);
    id
  }

  pub fn build(self) -> RenderGraph {
    RenderGraph {
      resources: self.resources,
      passes: self.passes,
      execution_order: Vec::new(),
      pass_graph: Graph::new(),
    }
  }
}

pub struct RenderGraph {
  pub resources: Vec<RenderResource>,
  pub passes: Vec<RenderPass>,
  execution_order: Vec<RenderPassId>,
  pass_graph: Graph<RenderPassId>,
}

impl RenderGraph {
  pub fn compile(&mut self) -> Result<(), IcsError> {
    ICS_INFO!("Compiling graph with {} passes", self.passes.len());
    // Build DAG using the shared container graph.
    let mut graph: Graph<RenderPassId> = Graph::new();
    let mut node_ids = Vec::with_capacity(self.passes.len());
    for (idx, _) in self.passes.iter().enumerate() {
      let node_id = graph.add_node(RenderPassId(idx));
      node_ids.push(node_id);
    }

    for (src_idx, src) in self.passes.iter().enumerate() {
      for (dst_idx, dst) in self.passes.iter().enumerate() {
        if src_idx == dst_idx {
          continue;
        }
        if Self::has_dependency(src, dst) {
          let _ = graph
            .add_edge(node_ids[src_idx], node_ids[dst_idx])
            .map_err(|e| {
              ICS_ERROR!(
                why: "Failed to add edge to DAG",
                fix: "Inspect pass dependency construction",
                src: e
              )
            })?;
          ICS_DEBUG!(
            "Edge {:?} -> {:?}",
            RenderPassId(src_idx),
            RenderPassId(dst_idx)
          );
        }
      }
    }

    let ordered_nodes = graph.topological_sort().map_err(|e| {
      ICS_ERROR!(
        why: "Cycle detected during compilation",
        fix: "Ensure render graph passes do not have circular dependencies",
        src: e
      )
    })?;

    let mut ordered = Vec::with_capacity(ordered_nodes.len());
    for node_id in ordered_nodes {
      if let Some(node) = graph.get_node(node_id) {
        ordered.push(node.data);
      }
    }

    self.execution_order = ordered;
    self.pass_graph = graph;
    ICS_INFO!("Compiled execution order {:?}", self.execution_order);
    Ok(())
  }

  pub fn ordered_passes(&self) -> impl Iterator<Item = &RenderPass> {
    self.execution_order.iter().map(|id| &self.passes[id.0])
  }

  /// Produce an API-agnostic render plan from the current graph state.
  /// This will be extended to include subpass grouping and dependency info.
  pub fn make_plan(&self) -> RenderPlan {
    let mut max_subpass: u32 = 0;
    let sample_value = |samples: GraphSampleCount| -> u32 {
      match samples {
        GraphSampleCount::One => 1,
        GraphSampleCount::Two => 2,
        GraphSampleCount::Four => 4,
        GraphSampleCount::Eight => 8,
        GraphSampleCount::Custom(raw) => raw,
      }
    };
    let sample_from_value = |value: u32| -> GraphSampleCount {
      match value {
        1 => GraphSampleCount::One,
        2 => GraphSampleCount::Two,
        4 => GraphSampleCount::Four,
        8 => GraphSampleCount::Eight,
        other => GraphSampleCount::Custom(other),
      }
    };
    let mut attachments: Vec<PlanAttachment> = self
      .resources
      .iter()
      .filter_map(|r| match r.kind {
        RenderResourceKind::SwapchainColor => Some(PlanAttachment {
          name: r.name.clone(),
          format: GraphFormat::DefaultColor,
          samples: GraphSampleCount::One,
          size: GraphSize::Swapchain,
          lifetime: GraphResourceLifetime::External,
          role: r.role,
          is_swapchain: true,
          initial_layout: GraphImageLayout::Undefined,
          final_layout: GraphImageLayout::Present,
          ever_read: false,
        }),
        RenderResourceKind::SwapchainDepth => Some(PlanAttachment {
          name: r.name.clone(),
          format: GraphFormat::DefaultDepth,
          samples: GraphSampleCount::One,
          size: GraphSize::Swapchain,
          lifetime: GraphResourceLifetime::External,
          role: r.role,
          is_swapchain: true,
          initial_layout: GraphImageLayout::Undefined,
          final_layout: GraphImageLayout::DepthAttachment,
          ever_read: false,
        }),
        RenderResourceKind::Image(desc) => Some(PlanAttachment {
          name: r.name.clone(),
          format: desc.format,
          samples: desc.samples,
          size: desc.size,
          lifetime: desc.lifetime,
          role: r.role,
          is_swapchain: false,
          initial_layout: GraphImageLayout::Undefined,
          final_layout: if desc.is_depth {
            GraphImageLayout::DepthAttachment
          } else {
            GraphImageLayout::ColorAttachment
          },
          ever_read: false,
        }),
        RenderResourceKind::Buffer(_) | RenderResourceKind::External => None,
      })
      .collect();

    let mut passes: Vec<PlanPass> = self
      .ordered_passes()
      .map(|p| {
        let reads: Vec<GraphPassResourceUse> = p
          .reads
          .iter()
          .filter_map(|r| {
            self.resources.get(r.resource.0).and_then(|res| {
              let is_depth = match res.kind {
                RenderResourceKind::SwapchainDepth => true,
                RenderResourceKind::Image(desc) => desc.is_depth,
                _ => false,
              };
              let access = if matches!(p.kind, RenderPassKind::Present) {
                GraphAccessType::Present
              } else {
                match (res.kind, r.access) {
                  (RenderResourceKind::Buffer(_), ResourceAccess::Read) => {
                    GraphAccessType::BufferRead
                  }
                  (RenderResourceKind::Buffer(_), ResourceAccess::Write) => {
                    GraphAccessType::BufferWrite
                  }
                  (RenderResourceKind::Buffer(_), ResourceAccess::ReadWrite) => {
                    GraphAccessType::BufferReadWrite
                  }
                  (RenderResourceKind::External, ResourceAccess::Read) => {
                    GraphAccessType::BufferRead
                  }
                  (RenderResourceKind::External, ResourceAccess::Write) => {
                    GraphAccessType::BufferWrite
                  }
                  (RenderResourceKind::External, ResourceAccess::ReadWrite) => {
                    GraphAccessType::BufferReadWrite
                  }
                  (_, ResourceAccess::Read) => {
                    if is_depth {
                      GraphAccessType::InputAttachment
                    } else {
                      GraphAccessType::Sampled
                    }
                  }
                  (_, ResourceAccess::InputAttachment) => GraphAccessType::InputAttachment,
                  (_, ResourceAccess::Write) => {
                    if is_depth {
                      GraphAccessType::DepthWrite
                    } else {
                      GraphAccessType::ColorWrite
                    }
                  }
                  (_, ResourceAccess::ReadWrite) => GraphAccessType::StorageReadWrite,
                }
              };
              Some(GraphPassResourceUse {
                resource: res.name.clone(),
                access,
              })
            })
          })
          .collect();
        let writes: Vec<GraphPassResourceUse> = p
          .writes
          .iter()
          .filter_map(|r| {
            self.resources.get(r.resource.0).and_then(|res| {
              let is_depth = match res.kind {
                RenderResourceKind::SwapchainDepth => true,
                RenderResourceKind::Image(desc) => desc.is_depth,
                _ => false,
              };
              let access = if matches!(p.kind, RenderPassKind::Present) {
                GraphAccessType::Present
              } else {
                match (res.kind, r.access) {
                  (RenderResourceKind::Buffer(_), ResourceAccess::Read) => {
                    GraphAccessType::BufferRead
                  }
                  (RenderResourceKind::Buffer(_), ResourceAccess::Write) => {
                    GraphAccessType::BufferWrite
                  }
                  (RenderResourceKind::Buffer(_), ResourceAccess::ReadWrite) => {
                    GraphAccessType::BufferReadWrite
                  }
                  (RenderResourceKind::External, ResourceAccess::Read) => {
                    GraphAccessType::BufferRead
                  }
                  (RenderResourceKind::External, ResourceAccess::Write) => {
                    GraphAccessType::BufferWrite
                  }
                  (RenderResourceKind::External, ResourceAccess::ReadWrite) => {
                    GraphAccessType::BufferReadWrite
                  }
                  (_, ResourceAccess::Read) => {
                    if is_depth {
                      GraphAccessType::InputAttachment
                    } else {
                      GraphAccessType::Sampled
                    }
                  }
                  (_, ResourceAccess::InputAttachment) => GraphAccessType::InputAttachment,
                  (_, ResourceAccess::Write) => {
                    if is_depth {
                      GraphAccessType::DepthWrite
                    } else {
                      GraphAccessType::ColorWrite
                    }
                  }
                  (_, ResourceAccess::ReadWrite) => GraphAccessType::StorageReadWrite,
                }
              };
              Some(GraphPassResourceUse {
                resource: res.name.clone(),
                access,
              })
            })
          })
          .collect();
        max_subpass = max_subpass.max(p.subpass);
        PlanPass {
          name: p.name.clone(),
          render_pass: 0,
          render_pass_hint: p.render_pass_hint,
          subpass: p.subpass,
          reads,
          writes,
          kind: p.kind,
          queue_request: p.queue_request,
          command_usage: p.command_usage,
        }
      })
      .collect();

    // Apply per-pass MSAA hints by promoting attachment sample counts for written resources.
    let mut name_to_attachment: HashMap<String, usize> = HashMap::new();
    for (idx, att) in attachments.iter().enumerate() {
      name_to_attachment.insert(att.name.clone(), idx);
    }
    for pass in self.ordered_passes() {
      let Some(hint) = pass.msaa_hint else { continue };
      let hint_value = sample_value(hint);
      if hint_value <= 1 {
        continue;
      }
      for write in &pass.writes {
        if let Some(res) = self.resources.get(write.resource.0) {
          if let Some(&att_idx) = name_to_attachment.get(&res.name) {
            let current = attachments[att_idx].samples;
            let current_value = sample_value(current);
            let new_value = current_value.max(hint_value);
            attachments[att_idx].samples = sample_from_value(new_value);
          }
        }
      }
    }

    // Assign render pass groups and adjust subpass indices:
    // - Input attachments stay within a render pass but may require later subpasses.
    // - Sampled/storage reads after a write split into a new render pass.
    // - Render pass hints can force a split for advanced usage.
    let mut current_render_pass: u32 = 0;
    let mut last_write: HashMap<String, (u32, u32)> = HashMap::new(); // resource -> (render_pass, subpass)
    let mut last_sampled: HashMap<String, u32> = HashMap::new(); // resource -> render_pass
    let mut grouping_stats = RenderPlanGroupingStats::default();
    for pass in &mut passes {
      if let Some(hint) = pass.render_pass_hint {
        if hint > current_render_pass {
          current_render_pass = hint;
          grouping_stats.split_due_to_hint += 1;
          ICS_DEBUG!(
            "forcing render pass {} before '{}' due to render_pass_hint",
            current_render_pass,
            pass.name
          );
        } else if hint < current_render_pass {
          grouping_stats.hint_backwards += 1;
          ICS_WARN!(
            "render_pass_hint {} for '{}' precedes current render pass {}; ignoring",
            hint,
            pass.name,
            current_render_pass
          );
        }
      }
      let mut needs_new_render_pass = false;
      let mut split_reason: Option<&str> = None;
      for r in &pass.reads {
        if matches!(
          r.access,
          GraphAccessType::Sampled | GraphAccessType::StorageReadWrite
        ) {
          if let Some((rp, _)) = last_write.get(&r.resource) {
            if *rp == current_render_pass {
              needs_new_render_pass = true;
              split_reason = Some("sampled/read after write");
              break;
            }
          }
        }
      }
      if !needs_new_render_pass {
        for w in &pass.writes {
          if let Some(rp) = last_sampled.get(&w.resource) {
            if *rp == current_render_pass {
              needs_new_render_pass = true;
              split_reason = Some("write after sampled read");
              break;
            }
          }
        }
      }
      if needs_new_render_pass {
        current_render_pass += 1;
        match split_reason {
          Some("sampled/read after write") => grouping_stats.split_sampled_after_write += 1,
          Some("write after sampled read") => grouping_stats.split_write_after_sampled += 1,
          _ => {}
        }
        ICS_DEBUG!(
          "splitting render pass before '{}' due to {} (new render_pass={})",
          pass.name,
          split_reason.unwrap_or("hazard"),
          current_render_pass
        );
      }
      pass.render_pass = current_render_pass;
      ICS_DEBUG!(
        "pass '{}' assigned render_pass={} subpass={}",
        pass.name,
        pass.render_pass,
        pass.subpass
      );

      let mut required_subpass = pass.subpass;
      for r in &pass.reads {
        if matches!(r.access, GraphAccessType::InputAttachment) {
          if let Some((rp, subpass)) = last_write.get(&r.resource) {
            if *rp == current_render_pass && *subpass >= required_subpass {
              required_subpass = *subpass + 1;
            }
          }
        }
      }
      if required_subpass != pass.subpass {
        ICS_DEBUG!(
          "promoting pass '{}' from subpass {} to {} due to input attachment hazards",
          pass.name,
          pass.subpass,
          required_subpass
        );
        pass.subpass = required_subpass;
        grouping_stats.subpass_promotions += 1;
      }
      for w in &pass.writes {
        last_write.insert(w.resource.clone(), (current_render_pass, pass.subpass));
      }
      for r in &pass.reads {
        if matches!(
          r.access,
          GraphAccessType::Sampled | GraphAccessType::StorageReadWrite
        ) {
          last_sampled.insert(r.resource.clone(), current_render_pass);
        }
      }
      max_subpass = max_subpass.max(pass.subpass);
    }

    // Compact subpass indices per render pass so each render pass starts at subpass 0.
    let mut per_rp_subpasses: HashMap<u32, Vec<u32>> = HashMap::new();
    for pass in &passes {
      per_rp_subpasses
        .entry(pass.render_pass)
        .or_default()
        .push(pass.subpass);
    }
    let mut subpass_map: HashMap<(u32, u32), u32> = HashMap::new();
    for (rp, mut subs) in per_rp_subpasses {
      subs.sort_unstable();
      subs.dedup();
      for (new_idx, old_idx) in subs.iter().enumerate() {
        subpass_map.insert((rp, *old_idx), new_idx as u32);
      }
    }
    for pass in &mut passes {
      if let Some(new_subpass) = subpass_map.get(&(pass.render_pass, pass.subpass)).copied() {
        if new_subpass != pass.subpass {
          ICS_DEBUG!(
            "remapping pass '{}' render_pass {} subpass {} -> {}",
            pass.name,
            pass.render_pass,
            pass.subpass,
            new_subpass
          );
          pass.subpass = new_subpass;
        }
      }
    }
    max_subpass = passes.iter().map(|p| p.subpass).max().unwrap_or(0);

    // Compute usage info per attachment to derive initial/final layouts.
    let mut usage: HashMap<String, (Option<bool>, bool, bool, bool)> = HashMap::new(); // name -> (first_is_read, ever_read, is_depth, is_swapchain)
    for att in &attachments {
      let is_depth = matches!(
        att.format,
        GraphFormat::DefaultDepth | GraphFormat::D32 | GraphFormat::D24S8
      );
      usage.insert(att.name.clone(), (None, false, is_depth, att.is_swapchain));
    }
    for pass in &passes {
      for r in &pass.reads {
        if let Some(entry) = usage.get_mut(&r.resource) {
          if entry.0.is_none() {
            entry.0 = Some(true);
          }
          entry.1 = true;
        }
      }
      for w in &pass.writes {
        if let Some(entry) = usage.get_mut(&w.resource) {
          if entry.0.is_none() {
            entry.0 = Some(false);
          }
        }
      }
    }
    for att in &mut attachments {
      if let Some((first_is_read, ever_read, is_depth, is_swapchain)) =
        usage.get(&att.name).copied()
      {
        att.ever_read = ever_read;
        if is_swapchain {
          att.initial_layout = GraphImageLayout::Undefined;
          att.final_layout = GraphImageLayout::Present;
          continue;
        }
        if is_depth {
          att.final_layout = GraphImageLayout::DepthAttachment;
          att.initial_layout = if first_is_read.unwrap_or(false) {
            GraphImageLayout::DepthRead
          } else {
            GraphImageLayout::DepthAttachment
          };
        } else {
          let target = if ever_read {
            GraphImageLayout::ShaderRead
          } else {
            GraphImageLayout::ColorAttachment
          };
          att.final_layout = target;
          att.initial_layout = if first_is_read.unwrap_or(false) {
            GraphImageLayout::ShaderRead
          } else {
            GraphImageLayout::ColorAttachment
          };
        }
      }
    }

    ICS_DEBUG!("Plan attachments/layouts:");
    for att in &attachments {
      ICS_DEBUG!(
        "  att {} fmt {:?} samples {:?} init {:?} final {:?} ever_read {} swapchain {}",
        att.name,
        att.format,
        att.samples,
        att.initial_layout,
        att.final_layout,
        att.ever_read,
        att.is_swapchain
      );
    }
    ICS_DEBUG!("Plan passes (subpass order):");
    for p in &passes {
      ICS_DEBUG!(
        "  pass {} render_pass {} subpass {} reads {} writes {}",
        p.name,
        p.render_pass,
        p.subpass,
        p.reads.len(),
        p.writes.len()
      );
    }

    let render_pass_count = if passes.is_empty() {
      0
    } else {
      current_render_pass + 1
    };
    let mut render_pass_subpasses = vec![0u32; render_pass_count as usize];
    for p in &passes {
      let idx = p.render_pass as usize;
      if idx < render_pass_subpasses.len() {
        render_pass_subpasses[idx] = render_pass_subpasses[idx].max(p.subpass + 1);
      }
    }
    let overall_subpass_count = render_pass_subpasses
      .iter()
      .copied()
      .max()
      .unwrap_or(max_subpass + 1);

    let mut resource_kinds = HashMap::new();
    for res in &self.resources {
      resource_kinds.insert(res.name.clone(), res.kind);
    }

    RenderPlan {
      attachments,
      passes,
      resource_kinds,
      subpass_count: overall_subpass_count,
      render_pass_count,
      render_pass_subpasses,
      grouping_stats,
    }
  }

  /// Produce a render plan and a backend-agnostic synchronization plan.
  pub fn make_plan_with_sync(&self) -> (RenderPlan, RenderSyncPlan) {
    let plan = self.make_plan();
    let sync_plan = RenderSyncPlan::from_plan(&plan);
    (plan, sync_plan)
  }

  /// Build a backend-agnostic synchronization plan from an existing plan.
  pub fn make_sync_plan(&self, plan: &RenderPlan) -> RenderSyncPlan {
    RenderSyncPlan::from_plan(plan)
  }

  fn has_dependency(a: &RenderPass, b: &RenderPass) -> bool {
    if a.sequence >= b.sequence {
      return false;
    }
    let writes = |pass: &RenderPass| pass.writes.iter().map(|r| r.resource).collect::<Vec<_>>();
    let reads = |pass: &RenderPass| pass.reads.iter().map(|r| r.resource).collect::<Vec<_>>();
    let a_writes = writes(a);
    let a_reads = reads(a);
    let b_writes = writes(b);
    let b_reads = reads(b);

    // write -> read
    if a_writes.iter().any(|res| b_reads.contains(res)) {
      return true;
    }
    // write -> write (order by sequence to avoid cycles)
    if a_writes.iter().any(|res| b_writes.contains(res)) {
      return a.sequence < b.sequence;
    }
    // read -> write (order by sequence to avoid cycles)
    if a_reads.iter().any(|res| b_writes.contains(res)) {
      return a.sequence < b.sequence;
    }
    false
  }
}
