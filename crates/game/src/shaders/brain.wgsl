// The whole-brain view: every neuron is a billboard pulled from two storage buffers. `soma` (static, uploaded once) holds the cell body
// position and the population; `activity` (one f32 per neuron) is rewritten each frame from the sim snapshot.
#import bevy_pbr::mesh_view_bindings::view

struct Params {
    count: u32,
    size: f32,
    base_glow: f32,
    gain: f32,
    // x, y: rotation about the vertical axis (cos, sin); z, w: scale and vertical shift of the cloud
    rot: vec4<f32>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> params: Params;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var<storage, read> soma: array<vec4<f32>>;
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var<storage, read> activity: array<f32>;

struct VertexIn {
    @location(0) corner_id: vec3<f32>,
}

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
}

fn palette(pop: i32) -> vec3<f32> {
    // photoreceptors, optic lobe, central brain, head sensory, body sensory, ascending, descending, nerve cord, motor neurons
    switch pop {
        case 0: { return vec3<f32>(0.95, 0.85, 0.30); }
        case 1: { return vec3<f32>(0.30, 0.55, 1.00); }
        case 2: { return vec3<f32>(0.75, 0.75, 0.85); }
        case 3: { return vec3<f32>(0.35, 0.90, 0.60); }
        case 4: { return vec3<f32>(0.25, 0.80, 0.80); }
        case 5: { return vec3<f32>(0.80, 0.50, 1.00); }
        case 6: { return vec3<f32>(1.00, 0.45, 0.35); }
        case 7: { return vec3<f32>(0.60, 0.65, 0.40); }
        case 8: { return vec3<f32>(1.00, 0.70, 0.30); }
        default: { return vec3<f32>(0.5, 0.5, 0.5); }
    }
}

@vertex
fn vertex(in: VertexIn) -> VertexOut {
    var out: VertexOut;
    let id = u32(in.corner_id.z + 0.5);
    out.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    out.uv = vec2<f32>(0.0);
    out.color = vec4<f32>(0.0);
    if id >= params.count {
        return out;
    }
    let s = soma[id];
    if s.w < -0.5 {
        return out; // the dataset has no cell body for this neuron
    }
    let a = activity[id];
    let c = params.rot.x;
    let sn = params.rot.y;
    let p = vec3<f32>(s.x * c + s.z * sn, s.y + params.rot.w, -s.x * sn + s.z * c) * params.rot.z;
    let vpos = view.view_from_world * vec4<f32>(p, 1.0);
    let size = params.size * (1.0 + 2.5 * a);
    out.clip = view.clip_from_view * (vpos + vec4<f32>(in.corner_id.xy * size, 0.0, 0.0));
    out.uv = in.corner_id.xy;
    let glow = params.base_glow + params.gain * a * a;
    out.color = vec4<f32>(mix(palette(i32(s.w)), vec3<f32>(1.0), clamp(a, 0.0, 1.0) * 0.6), glow);
    return out;
}

@fragment
fn fragment(in: VertexOut) -> @location(0) vec4<f32> {
    let d = dot(in.uv, in.uv);
    if d > 1.0 {
        discard;
    }
    let g = (1.0 - d);
    // AlphaMode::Add adds the colour as it is (premultiplied), so the brightness is multiplied in here
    let a = g * g * in.color.a;
    return vec4<f32>(in.color.rgb * a, a);
}
