// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

// Artistic atmospheric scattering for the showcase atmosphere shell. This is a
// *consumer-side* look (it lives in sandbox, not in aether/eidolon) — a cheap
// analytic approximation: limb brightening (the shell glows where it is edge-on
// to the eye) plus forward scattering toward the sun, alpha-blended so the
// surface and ocean meshes beneath show through.

#import bevy_pbr::forward_io::VertexOutput
#import bevy_pbr::mesh_view_bindings::view

struct AtmosphereParams {
  sky_color: vec4<f32>,
  // xyz = world-space direction toward the sun.
  sun_direction: vec4<f32>,
  // x = intensity, y = rim power, z = base alpha, w = sun-glow strength.
  params: vec4<f32>,
}

@group(2) @binding(0) var<uniform> material: AtmosphereParams;

@fragment
fn fragment(mesh: VertexOutput) -> @location(0) vec4<f32> {
  let n = normalize(mesh.world_normal);
  let view_dir = normalize(view.world_position - mesh.world_position.xyz);

  // Limb brightening: thicker apparent air at grazing angles.
  let ndotv = clamp(dot(n, view_dir), 0.0, 1.0);
  let rim = pow(1.0 - ndotv, max(material.params.y, 0.001));

  // Forward scattering: brighten when looking toward the sun through the air.
  let sun = normalize(material.sun_direction.xyz);
  let towards_sun = clamp(dot(view_dir, sun), 0.0, 1.0);
  let glow = pow(towards_sun, 8.0) * material.params.w;

  let intensity = material.params.x;
  let base_alpha = material.params.z;

  let colour =
    material.sky_color.rgb * intensity + vec3<f32>(1.0, 0.95, 0.8) * glow;
  let alpha = clamp(base_alpha + rim * (1.0 - base_alpha) + glow * 0.5, 0.0, 1.0);

  return vec4<f32>(colour * (base_alpha + rim), alpha);
}
