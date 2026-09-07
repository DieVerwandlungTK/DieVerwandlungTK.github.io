// Bond vertices: points[v*2] = clip xy, depth, dash phase; points[v*2+1] = opacity, kind, 0, 0.
// Kind 0 is a constrained intramolecular O-H bond; kind 1 is a geometric hydrogen bond.
struct Params { aspect: f32, warmth: f32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> points: array<vec4f>;

struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0) phase: f32,
  @location(1) depth: f32,
  @location(2) opacity: f32,
  @location(3) kind: f32,
}

@vertex fn vs_main(@builtin(vertex_index) v: u32) -> VertexOut {
  let p = points[v * 2u];
  let attributes = points[v * 2u + 1u];
  var out: VertexOut;
  out.position = vec4f(p.x / params.aspect, p.y, 0.5, 1.0);
  out.phase = p.w;
  out.depth = p.z;
  out.opacity = attributes.x;
  out.kind = attributes.y;
  return out;
}

@fragment fn fs_main(in: VertexOut) -> @location(0) vec4f {
  if (in.kind > 0.5 && fract(in.phase) > 0.55) { discard; }
  let covalent = vec3f(0.31, 0.47, 0.55);
  let hydrogen = mix(vec3f(0.33, 0.55, 0.66), vec3f(0.46, 0.56, 0.62), params.warmth);
  let color = mix(covalent, hydrogen, in.kind);
  let fog = clamp(0.22 - in.depth * 0.24, 0.02, 0.45);
  let alpha = in.opacity * mix(0.72, 0.62, in.kind) * (1.0 - fog);
  return vec4f(mix(color, vec3f(0.96, 0.975, 0.98), fog), alpha);
}
