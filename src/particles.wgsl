// Atoms: points[i*2] = clip xy, depth, radius; points[i*2+1] = opacity, kind, 0, 0.
struct Params { aspect: f32, warmth: f32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> points: array<vec4f>;

struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
  @location(1) depth: f32,
  @location(2) opacity: f32,
  @location(3) kind: f32,
}

@vertex fn vs_main(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> VertexOut {
  var corners = array<vec2f, 6>(vec2f(-1,-1), vec2f(1,-1), vec2f(-1,1), vec2f(-1,1), vec2f(1,-1), vec2f(1,1));
  let p = points[i * 2u];
  let attributes = points[i * 2u + 1u];
  let corner = corners[v];
  let radius = p.w * (1.0 + p.z * 0.16);
  let xy = p.xy + corner * radius;
  var out: VertexOut;
  out.position = vec4f(xy.x / params.aspect, xy.y, 0.5, 1.0);
  out.uv = corner;
  out.depth = p.z;
  out.opacity = attributes.x;
  out.kind = attributes.y;
  return out;
}

@fragment fn fs_main(in: VertexOut) -> @location(0) vec4f {
  let rr = dot(in.uv, in.uv);
  if (rr > 1.0) { discard; }
  let normal = vec3f(in.uv, sqrt(max(0.0, 1.0 - rr)));
  let light = max(0.0, dot(normal, normalize(vec3f(-0.5, 0.7, 1.0))));
  let oxygen = mix(vec3f(0.19, 0.45, 0.60), vec3f(0.26, 0.51, 0.57), params.warmth);
  let hydrogen = vec3f(0.74, 0.83, 0.87);
  let base = mix(oxygen, hydrogen, in.kind);
  let shaded = base * (0.56 + light * 0.44) + vec3f(pow(light, 24.0) * 0.42);
  let fog = clamp(0.25 - in.depth * 0.27, 0.04, 0.5);
  let color = mix(shaded, vec3f(0.96, 0.975, 0.98), fog);
  return vec4f(color, in.opacity * (1.0 - smoothstep(0.85, 1.0, rr)));
}
