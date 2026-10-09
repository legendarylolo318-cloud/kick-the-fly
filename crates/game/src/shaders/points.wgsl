// Instanced billboards by vertex pulling: ONE mesh of N quads, one draw call. Each quad reads its own record from a storage buffer.
// The quad's corner is mesh POSITION.xy and its index is POSITION.z, so nothing depends on vertex_index/base_vertex.
#import bevy_pbr::mesh_view_bindings::view

struct Params {
    count: u32,
    size_scale: f32,
    time: f32,
    softness: f32,
}

struct Point {
    pos_size: vec4<f32>, // xyz world position, w radius (m)
    color: vec4<f32>,    // rgb, a = brightness
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> params: Params;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var<storage, read> points: array<Point>;

struct VertexIn {
    @location(0) corner_id: vec3<f32>,
}

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
}

@vertex
fn vertex(in: VertexIn) -> VertexOut {
    var out: VertexOut;
    let id = u32(in.corner_id.z + 0.5);
    if id >= params.count {
        out.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0); // outside the clip volume: culled
        out.uv = vec2<f32>(0.0);
        out.color = vec4<f32>(0.0);
        return out;
    }
    let p = points[id];
    let vpos = view.view_from_world * vec4<f32>(p.pos_size.xyz, 1.0);
    let s = p.pos_size.w * params.size_scale;
    out.clip = view.clip_from_view * (vpos + vec4<f32>(in.corner_id.xy * s, 0.0, 0.0));
    out.uv = in.corner_id.xy;
    out.color = p.color;
    return out;
}

@fragment
fn fragment(in: VertexOut) -> @location(0) vec4<f32> {
    let d = dot(in.uv, in.uv);
    if d > 1.0 {
        discard;
    }
    let glow = pow(1.0 - d, params.softness);
    let a = glow * in.color.a; // AlphaMode::Add adds the colour as it is (premultiplied)
    return vec4<f32>(in.color.rgb * a, a);
}
