// The fly's-eye view: the scene a camera at the fly's head sees, re-sampled the way compound eyes do. Each eye is a disc of hexagonal
// ommatidia (one colour sample per hexagon), seen in the fly's spectrum (UV, blue and green; almost no red), with the dark rim and
// vignette of a curved eye. VISUAL ONLY: nothing here is fed to the brain.
#import bevy_ui::ui_vertex_output::UiVertexOutput

struct Params {
    cells: f32,   // hexagons across one eye
    time: f32,
    fade: f32,    // 0..1 how far into the effect the transition is
    pad: f32,
}

@group(1) @binding(0) var<uniform> params: Params;
@group(1) @binding(1) var eye_tex: texture_2d<f32>;
@group(1) @binding(2) var eye_samp: sampler;

// a hexagonal grid: returns (offset from the hexagon's centre, the hexagon's centre in grid units)
fn hex(p: vec2<f32>) -> vec4<f32> {
    let s = vec2<f32>(1.0, 1.7320508);
    let a = (floor(p / s) + 0.5) * s;
    let b = (floor((p - vec2<f32>(0.5, 0.8660254)) / s) + 0.5) * s + vec2<f32>(0.5, 0.8660254);
    let da = p - a;
    let db = p - b;
    if dot(da, da) < dot(db, db) {
        return vec4<f32>(da, a);
    }
    return vec4<f32>(db, b);
}

fn fly_vision(c: vec3<f32>) -> vec3<f32> {
    // photoreceptors R1-R6 are broadband with a UV hump, R7/R8 UV and blue-green: little red. A crude mapping: red is mostly lost,
    // green and blue carry the image, and a UV channel (which we cannot show) is folded into blue.
    let uv = 0.5 * c.b + 0.25 * c.g;
    let g = 0.15 * c.r + 0.65 * c.g + 0.3 * c.b;
    let b = 0.6 * c.b + 0.35 * uv + 0.1 * c.g;
    return vec3<f32>(0.04 * c.r + 0.08 * uv, g, b + 0.06);
}

fn eye(uv: vec2<f32>, centre: vec2<f32>, view_x: vec2<f32>) -> vec4<f32> {
    let radius = 0.47;
    let local = (uv - centre) / radius; // -1..1 across this eye's disc
    let d = length(local);
    if d > 1.0 {
        return vec4<f32>(0.0);
    }
    let cell = hex(local * params.cells * 0.5);
    let centre_g = cell.zw / (params.cells * 0.5);
    let sample_uv = vec2<f32>(mix(view_x.x, view_x.y, centre_g.x * 0.5 + 0.5), 0.5 - centre_g.y * 0.5);
    let col = textureSampleLevel(eye_tex, eye_samp, clamp(sample_uv, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).rgb;
    var rgb = fly_vision(col);
    // the dark gap between ommatidia
    let edge = smoothstep(0.78, 0.98, length(cell.xy) * 1.15);
    rgb = rgb * (1.0 - 0.55 * edge);
    // the eye is a curved surface: darker toward its rim
    rgb = rgb * (1.0 - 0.7 * pow(d, 3.0));
    let aa = 1.0 - smoothstep(0.985, 1.0, d);
    return vec4<f32>(rgb, aa);
}

@fragment
fn fragment(in: UiVertexOutput) -> @location(0) vec4<f32> {
    // the two eyes side by side, each seeing a little more than half of the wide view (they overlap in front)
    let aspect = in.size.x / in.size.y;
    let p = vec2<f32>((in.uv.x - 0.5) * aspect + 0.5, in.uv.y);
    var out = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    let l = eye(p, vec2<f32>(0.5 - 0.27 * 1.0, 0.5), vec2<f32>(0.0, 0.62));
    let r = eye(p, vec2<f32>(0.5 + 0.27 * 1.0, 0.5), vec2<f32>(0.38, 1.0));
    out = mix(out, l, l.a);
    out = mix(out, r, r.a);
    return vec4<f32>(out.rgb, max(l.a, r.a) * params.fade);
}
