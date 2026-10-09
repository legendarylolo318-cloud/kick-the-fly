// Rain simulated entirely on the GPU: no per-drop data exists on the CPU. Each streak's column is a hash of its index; its height is
// `fract(time * fall / height + hash)`, so the drops fall forever inside a box that follows the camera. One mesh, one draw call.
#import bevy_pbr::mesh_view_bindings::view

struct Params {
    count: u32,
    time: f32,
    intensity: f32, // 0..1: the share of the streaks that are drawn
    fall_speed: f32,
    wind: vec4<f32>,   // xyz m/s
    center: vec4<f32>, // xyz: the box follows this point; w: streak length scale
    half_extent: vec4<f32>, // x, z half width; y height; w streak width
    color: vec4<f32>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> params: Params;

struct VertexIn {
    @location(0) corner_id: vec3<f32>,
}

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) alpha: f32,
}

fn hash(n: u32) -> f32 {
    var x = n * 747796405u + 2891336453u;
    x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    x = (x >> 22u) ^ x;
    return f32(x) / 4294967295.0;
}

@vertex
fn vertex(in: VertexIn) -> VertexOut {
    var out: VertexOut;
    out.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    out.uv = vec2<f32>(0.0);
    out.alpha = 0.0;
    let id = u32(in.corner_id.z + 0.5);
    if id >= params.count || hash(id * 7u + 3u) > params.intensity {
        return out;
    }
    let h = params.half_extent;
    // the drop's height cycles with the clock; its column is fixed by its index
    let phase = fract(params.time * params.fall_speed * (0.85 + 0.3 * hash(id * 13u + 1u)) / h.y + hash(id));
    let drift = vec3<f32>(params.wind.x, 0.0, params.wind.z) * (phase * h.y / params.fall_speed);
    let local = vec3<f32>((hash(id * 3u + 11u) * 2.0 - 1.0) * h.x, (0.5 - phase) * h.y, (hash(id * 17u + 5u) * 2.0 - 1.0) * h.z);
    let world = params.center.xyz + local + drift;
    let vel = normalize(vec3<f32>(params.wind.x, -params.fall_speed, params.wind.z));
    let a = view.view_from_world * vec4<f32>(world, 1.0);
    let b = view.view_from_world * vec4<f32>(world - vel * params.center.w, 1.0);
    // the streak runs from `a` to `b` (along the fall) and is `h.w` wide across the screen
    let p = mix(a.xyz, b.xyz, in.corner_id.y * 0.5 + 0.5) + vec3<f32>(in.corner_id.x * h.w, 0.0, 0.0);
    out.clip = view.clip_from_view * vec4<f32>(p, 1.0);
    out.uv = in.corner_id.xy;
    out.alpha = params.color.a;
    return out;
}

@fragment
fn fragment(in: VertexOut) -> @location(0) vec4<f32> {
    let across = 1.0 - abs(in.uv.x);
    let along = 1.0 - abs(in.uv.y);
    let a = in.alpha * across * along; // AlphaMode::Add: the colour is added as it is
    return vec4<f32>(params.color.rgb * a, a);
}
