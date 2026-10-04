"""Small instanced OpenGL renderer for Kick the Fly 3D (moderngl).

Everything is drawn as instances of a few unit meshes (sphere, cylinder, cube, disk, torus) with a per-instance
model matrix, color, procedural pattern and glow, lit by a window sun, sky/floor ambient and two point lights.
Particles are camera-facing billboards. The HUD is a pygame surface uploaded as a texture each frame.
"""
from __future__ import annotations

import math

import moderngl
import numpy as np

# patterns decided in the fragment shader
P_NONE, P_WOOD, P_WALLPAPER, P_RUG, P_STRIPES, P_EYE, P_PAPER, P_WATER, P_SKY, P_CEIL, P_ICE, P_BOOKS, P_GRASS, \
    P_SKYDOME = range(14)


# --- matrices (column vectors; sent to GL column-major) -------------------------------------------------------
def perspective(fovy: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1 / math.tan(fovy / 2)
    m = np.zeros((4, 4))
    m[0, 0], m[1, 1] = f / aspect, f
    m[2, 2], m[2, 3] = (far + near) / (near - far), 2 * far * near / (near - far)
    m[3, 2] = -1
    return m


def look_at(eye, target, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = r, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def trs(pos, rot: np.ndarray | None = None, scale=(1.0, 1.0, 1.0)) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = (np.eye(3) if rot is None else rot) * np.asarray(scale, float)
    m[:3, 3] = pos
    return m


def frame_from_x(axis) -> np.ndarray:
    """Rotation whose local +x points along axis (local +y stays as close to world up as possible)."""
    x = np.asarray(axis, float)
    n = np.linalg.norm(x)
    x = np.array([1.0, 0, 0]) if n < 1e-9 else x / n
    up = np.array([0.0, 1, 0]) if abs(x[1]) < 0.95 else np.array([1.0, 0, 0])
    z = np.cross(x, up)
    z /= np.linalg.norm(z)
    y = np.cross(z, x)
    return np.stack([x, y, z], 1)


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def segment(a, b, radius: float) -> np.ndarray:
    """Model matrix taking the unit cylinder (y from 0 to 1, radius 1) onto the segment a->b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    L = float(np.linalg.norm(d))
    if L < 1e-9:                    # 3.0 day 3 review: a zero-length segment (a tongue on its first strike frame) gave a NaN matrix
        L, y = 1e-6, np.array([0.0, 1.0, 0.0])
    else:
        y = d / L
    helper = np.array([1.0, 0, 0]) if abs(y[0]) < 0.9 else np.array([0, 0, 1.0])
    x = np.cross(helper, y)
    x /= np.linalg.norm(x)
    z = np.cross(x, y)
    return trs(a, np.stack([x, y, z], 1), (radius, L, radius))


# --- meshes (interleaved position + normal) ---------------------------------------------------------------------
def mesh_sphere(lat: int = 14, lon: int = 22) -> np.ndarray:
    tris = []
    for i in range(lat):
        t0, t1 = math.pi * i / lat - math.pi / 2, math.pi * (i + 1) / lat - math.pi / 2
        for j in range(lon):
            p0, p1 = 2 * math.pi * j / lon, 2 * math.pi * (j + 1) / lon
            q = [(math.cos(t) * math.cos(p), math.sin(t), math.cos(t) * math.sin(p)) for t, p in ((t0, p0), (t0, p1), (t1, p1), (t1, p0))]
            tris += [q[0], q[2], q[1], q[0], q[3], q[2]]
    v = np.array(tris, "f4")
    return np.hstack([v, v]).astype("f4")


def mesh_cylinder(seg: int = 16) -> np.ndarray:
    out = []
    for j in range(seg):
        a0, a1 = 2 * math.pi * j / seg, 2 * math.pi * (j + 1) / seg
        c0, s0, c1, s1 = math.cos(a0), math.sin(a0), math.cos(a1), math.sin(a1)
        n0, n1 = (c0, 0, s0), (c1, 0, s1)
        quad = [((c0, 0, s0), n0), ((c1, 1, s1), n1), ((c1, 0, s1), n1), ((c0, 0, s0), n0), ((c0, 1, s0), n0), ((c1, 1, s1), n1)]
        out += quad
        out += [((0, 1, 0), (0, 1, 0)), ((c1, 1, s1), (0, 1, 0)), ((c0, 1, s0), (0, 1, 0))]      # counter-clockwise
        out += [((0, 0, 0), (0, -1, 0)), ((c0, 0, s0), (0, -1, 0)), ((c1, 0, s1), (0, -1, 0))]    # seen from outside
    return np.array([p + n for p, n in out], "f4")


def mesh_cube() -> np.ndarray:
    out = []
    for axis in range(3):
        for sign in (-1, 1):
            n = [0, 0, 0]
            n[axis] = sign
            u, v = [(axis + 1) % 3, (axis + 2) % 3]
            corners = []
            for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = [0.0, 0.0, 0.0]
                p[axis], p[u], p[v] = 0.5 * sign, 0.5 * a, 0.5 * b
                corners.append(tuple(p))
            order = (0, 1, 2, 0, 2, 3) if sign > 0 else (0, 2, 1, 0, 3, 2)
            out += [corners[k] + tuple(n) for k in order]
    return np.array(out, "f4")


def mesh_disk(seg: int = 28) -> np.ndarray:
    out = []
    for j in range(seg):
        a0, a1 = 2 * math.pi * j / seg, 2 * math.pi * (j + 1) / seg
        out += [(0, 0, 0, 0, 1, 0), (math.cos(a1), 0, math.sin(a1), 0, 1, 0), (math.cos(a0), 0, math.sin(a0), 0, 1, 0)]
    return np.array(out, "f4")


def mesh_torus(major: int = 24, minor: int = 8, r: float = 0.18) -> np.ndarray:
    out = []

    def pt(i, j):
        u, v = 2 * math.pi * i / major, 2 * math.pi * j / minor
        cx, cz = math.cos(u), math.sin(u)
        n = (math.cos(v) * cx, math.sin(v), math.cos(v) * cz)
        return (cx + r * n[0], r * n[1], cz + r * n[2]) + n

    for i in range(major):
        for j in range(minor):
            a, b, c, d = pt(i, j), pt(i + 1, j), pt(i + 1, j + 1), pt(i, j + 1)
            out += [a, c, b, a, d, c]
    return np.array(out, "f4")


LIT_VS = """
#version 330
uniform mat4 u_view;
uniform mat4 u_proj;
in vec3 in_pos;
in vec3 in_norm;
in mat4 in_model;
in vec4 in_color;
in vec2 in_extra;
out vec3 v_world;
out vec3 v_norm;
out vec3 v_local;
out vec4 v_color;
flat out int v_pattern;
out float v_glow;
void main() {
    vec4 w = in_model * vec4(in_pos, 1.0);
    v_world = w.xyz;
    v_norm = normalize(transpose(inverse(mat3(in_model))) * in_norm);
    v_local = in_pos;
    v_color = in_color;
    v_pattern = int(in_extra.x + 0.5);
    v_glow = in_extra.y;
    gl_Position = u_proj * u_view * w;
}
"""

LIT_FS = """
#version 330
uniform vec3 u_cam;
uniform vec3 u_sun_dir;
uniform vec3 u_sun_col;
uniform vec3 u_sky;
uniform vec3 u_ground;
uniform vec3 u_lp0;
uniform vec3 u_lc0;
uniform vec3 u_lp1;
uniform vec3 u_lc1;
uniform float u_time;
uniform vec4 u_fog;          // rgb: horizon colour, w: distance where fog is full (0 = no fog, indoors)
in vec3 v_world;
in vec3 v_norm;
in vec3 v_local;
in vec4 v_color;
flat in int v_pattern;
in float v_glow;
out vec4 f_color;

float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float noise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), u.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), u.x), u.y);
}

void main() {
    vec3 n = normalize(v_norm);
    if (!gl_FrontFacing) n = -n;
    vec3 base = v_color.rgb;
    float alpha = v_color.a;
    float spec_k = 0.15;
    float shin = 24.0;
    if (v_pattern == 1) {                 // wood floor planks running along x
        float row = floor(v_world.z / 0.22);
        float off = hash(vec2(row, 3.0)) * 1.7;
        float plank = floor((v_world.x + off) / 1.1);
        float seam = smoothstep(0.0, 0.012, fract(v_world.z / 0.22) * 0.22) * smoothstep(0.0, 0.01, fract((v_world.x + off) / 1.1) * 1.1);
        float grain = noise(vec2(v_world.x * 3.0, v_world.z * 40.0 + row * 7.0));
        float tone = 0.82 + 0.25 * hash(vec2(row, plank));
        base *= tone * (0.85 + 0.25 * grain) * (0.55 + 0.45 * seam);
        spec_k = 0.35;
    } else if (v_pattern == 2) {          // wallpaper: soft stripes and a small diamond print
        float s = smoothstep(0.35, 0.5, abs(fract((v_world.x + v_world.z) * 2.2) - 0.5));
        vec2 d = fract(vec2(v_world.x + v_world.z, v_world.y) * 6.0) - 0.5;
        float dia = step(abs(d.x) + abs(d.y), 0.12);
        base *= 0.9 + 0.1 * s + 0.08 * dia;
        if (v_world.y < 0.95) base *= 0.78;           // wainscot below the rail
        if (abs(v_world.y - 0.95) < 0.02) base *= 0.6;
    } else if (v_pattern == 3) {          // round rug with rings
        float r = length(v_world.xz);
        float ring = 0.5 + 0.5 * sin(r * 22.0);
        base = mix(base, base * vec3(1.3, 1.1, 0.8), ring * 0.45);
        base *= 0.85 + 0.15 * noise(v_world.xz * 60.0);
        spec_k = 0.02;
    } else if (v_pattern == 4) {          // abdomen: a dark band at the back of each tergite, on top only
        float seg = fract(v_local.x * 2.4 + 0.5);            // local +x points at the thorax, so -x is the tip
        // a wide, soft edge: a hard one follows the sphere's facets and breaks into zigzags up close
        float band = smoothstep(0.42, 0.04, seg);
        float dorsal = smoothstep(-0.12, 0.40, v_local.y);   // the underside stays pale, as on a real fly
        float rear = (0.4 + 0.6 * smoothstep(0.75, -0.05, v_local.x))   // darker toward the tip, and gone at the
                   * smoothstep(-0.92, -0.45, v_local.x);               // tip itself, where the dark tip takes over
        base = mix(base, base * 0.30, band * dorsal * rear);
        base = mix(base, base * vec3(1.10, 1.06, 0.82), (1.0 - dorsal) * 0.45);
        base *= 0.94 + 0.06 * noise(v_local.xy * 40.0);
        spec_k = 0.18; shin = 18.0;
    } else if (v_pattern == 5) {          // compound eye: fine facets, brightest where it catches the light
        vec2 q = v_local.yz * 15.0;
        float fac = length(fract(q) - 0.5);
        base *= 0.80 + 0.30 * smoothstep(0.5, 0.15, fac);
        base *= 1.0 - 0.25 * smoothstep(0.2, 1.0, -v_local.x);
        spec_k = 1.1; shin = 90.0;
    } else if (v_pattern == 6) {          // glossy flypaper with specks
        base *= 0.9 + 0.1 * noise(v_world.xz * 30.0);
        base *= 1.0 - 0.35 * step(0.93, hash(floor(v_world.xz * 40.0)));
        spec_k = 1.0; shin = 90.0;
    } else if (v_pattern == 7) {          // water surface
        float w = sin(v_world.x * 7.0 + u_time * 1.7) * sin(v_world.z * 6.0 - u_time * 1.3);
        n = normalize(n + vec3(0.12 * w, 0.0, 0.1 * cos(v_world.x * 5.0 + u_time)));
        spec_k = 1.6; shin = 120.0;
    } else if (v_pattern == 8) {          // window: sky and distant hills
        float h = v_local.y + 0.5;
        base = mix(vec3(0.55, 0.75, 0.95), vec3(0.95, 0.9, 0.75), 1.0 - h);
        float hill = 0.28 + 0.05 * sin(v_local.x * 9.0) + 0.03 * sin(v_local.x * 23.0);
        if (h < hill) base = vec3(0.35, 0.55, 0.3) * (0.8 + 0.4 * h);
        f_color = vec4(base, 1.0);
        return;
    } else if (v_pattern == 9) {          // ceiling with faint panels
        vec2 c = abs(fract(v_world.xz / 1.4) - 0.5);
        base *= 0.94 + 0.06 * step(0.47, max(c.x, c.y));
    } else if (v_pattern == 10) {         // ice
        spec_k = 1.4; shin = 80.0;
    } else if (v_pattern == 12) {         // outdoor ground: grass and soil patches
        float big = noise(v_world.xz * 0.35);
        float fine = noise(v_world.xz * 7.0);
        base *= 0.78 + 0.3 * big + 0.12 * fine;
        base = mix(base, base * vec3(1.15, 0.95, 0.7), smoothstep(0.62, 0.8, big));
        spec_k = 0.02;
    } else if (v_pattern == 13) {         // sky dome: unlit gradient from horizon haze to blue, with the sun's glow
        vec3 dir = normalize(v_world - u_cam);
        float h = clamp(dir.y, 0.0, 1.0);
        vec3 sky = mix(u_fog.rgb, v_color.rgb, pow(h, 0.55));
        float sun = max(dot(dir, -u_sun_dir), 0.0);
        sky += vec3(1.0, 0.92, 0.75) * (pow(sun, 400.0) * 3.0 + pow(sun, 12.0) * 0.25);
        if (dir.y < 0.0) sky = u_fog.rgb * 0.9;
        f_color = vec4(pow(sky, vec3(1.0 / 1.15)), 1.0);
        return;
    } else if (v_pattern == 11) {         // book spines
        float k = floor(v_world.x * 26.0 + v_world.z * 26.0);
        vec3 c = vec3(hash(vec2(k, 1.0)), hash(vec2(k, 2.0)), hash(vec2(k, 3.0)));
        base = mix(vec3(0.25, 0.12, 0.08), c * 0.8 + 0.1, 0.8);
        base *= 0.85 + 0.3 * step(0.5, fract(v_world.y * 8.0));
    }
    vec3 V = normalize(u_cam - v_world);
    float hemi = 0.5 + 0.5 * n.y;
    vec3 light = mix(u_ground, u_sky, hemi);
    float sd = max(dot(n, -u_sun_dir), 0.0);
    light += u_sun_col * sd;
    vec3 H = normalize(-u_sun_dir + V);
    vec3 spec = u_sun_col * pow(max(dot(n, H), 0.0), shin) * spec_k * sd;
    for (int i = 0; i < 2; i++) {
        vec3 lp = i == 0 ? u_lp0 : u_lp1;
        vec3 lc = i == 0 ? u_lc0 : u_lc1;
        vec3 L = lp - v_world;
        float d2 = dot(L, L);
        L /= sqrt(d2);
        float att = 1.0 / (1.0 + 0.35 * d2);
        float dif = max(dot(n, L), 0.0);
        light += lc * dif * att;
        spec += lc * pow(max(dot(n, normalize(L + V)), 0.0), shin) * spec_k * att;
    }
    vec3 col = base * light + spec + base * v_glow;
    if (v_pattern == 7) alpha = min(1.0, alpha + 0.35 * pow(1.0 - max(dot(n, V), 0.0), 3.0));
    if (u_fog.w > 0.0) {                                   // outdoors: distant things fade into the horizon haze
        float fd = length(v_world - u_cam);
        col = mix(col, u_fog.rgb, smoothstep(u_fog.w * 0.45, u_fog.w, fd));
    }
    col = col / (1.0 + col * 0.15);                        // gentle tone map
    f_color = vec4(pow(col, vec3(1.0 / 1.15)), alpha);
}
"""

PART_VS = """
#version 330
uniform mat4 u_view;
uniform mat4 u_proj;
in vec2 in_corner;
in vec3 in_center;
in float in_size;
in vec4 in_color;
out vec2 v_uv;
out vec4 v_color;
void main() {
    vec4 c = u_view * vec4(in_center, 1.0);
    c.xy += in_corner * in_size;
    v_uv = in_corner;
    v_color = in_color;
    gl_Position = u_proj * c;
}
"""

PART_FS = """
#version 330
in vec2 v_uv;
in vec4 v_color;
out vec4 f_color;
void main() {
    float d = length(v_uv);
    float a = smoothstep(1.0, 0.2, d);
    f_color = vec4(v_color.rgb, v_color.a * a);
}
"""

QUAD_VS = """
#version 330
uniform vec4 u_rect;   // x0, y0, x1, y1 in clip space
in vec2 in_uv;
out vec2 v_uv;
void main() {
    v_uv = in_uv;
    gl_Position = vec4(mix(u_rect.xy, u_rect.zw, in_uv), 0.0, 1.0);
}
"""

QUAD_FS = """
#version 330
uniform sampler2D u_tex;
uniform float u_flip;
in vec2 v_uv;
out vec4 f_color;
void main() {
    vec2 uv = vec2(v_uv.x, u_flip > 0.5 ? 1.0 - v_uv.y : v_uv.y);
    f_color = texture(u_tex, uv);
}
"""

DOF_FS = """
#version 330
uniform sampler2D u_tex;
uniform sampler2D u_depth;
uniform float u_focus;
uniform float u_dof;
uniform float u_near;
uniform float u_far;
uniform vec2 u_res;
uniform float u_flip;

in vec2 v_uv;
out vec4 f_color;

const vec2 SAMPLES[12] = vec2[12](
    vec2( 0.000,  0.300), vec2( 0.285,  0.093),
    vec2( 0.176, -0.243), vec2(-0.176, -0.243),
    vec2(-0.285,  0.093), vec2( 0.000,  0.600),
    vec2( 0.520,  0.300), vec2( 0.520, -0.300),
    vec2( 0.000, -0.600), vec2(-0.520, -0.300),
    vec2(-0.520,  0.300), vec2( 0.000,  1.000)
);

void main() {
    vec2 uv = vec2(v_uv.x, u_flip > 0.5 ? 1.0 - v_uv.y : v_uv.y);
    float d = texture(u_depth, uv).r;
    float z_lin = (u_near * u_far) / max(u_far - d * (u_far - u_near), 0.0001);
    float coc = clamp(abs(z_lin - u_focus) * u_dof * 0.8, 0.0, 1.0);
    if (coc < 0.01) {
        f_color = texture(u_tex, uv);
        return;
    }
    vec2 blur_rad = (coc * 16.0) / u_res;
    vec4 acc = texture(u_tex, uv);
    float total_w = 1.0;
    for (int i = 0; i < 12; i++) {
        vec2 sample_uv = uv + SAMPLES[i] * blur_rad;
        float sd = texture(u_depth, sample_uv).r;
        float sz = (u_near * u_far) / max(u_far - sd * (u_far - u_near), 0.0001);
        float scoc = clamp(abs(sz - u_focus) * u_dof * 0.8, 0.0, 1.0);
        float w = (sz >= z_lin) ? 1.0 : scoc;
        acc += texture(u_tex, sample_uv) * w;
        total_w += w;
    }
    f_color = acc / total_w;
}
"""

LAYERS = ("opaque", "blend", "view", "view_blend")
MESHES = ("sphere", "cylinder", "cube", "disk", "torus")


class Renderer:
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.lit = ctx.program(vertex_shader=LIT_VS, fragment_shader=LIT_FS)
        self.part = ctx.program(vertex_shader=PART_VS, fragment_shader=PART_FS)
        self.quad = ctx.program(vertex_shader=QUAD_VS, fragment_shader=QUAD_FS)
        self.dof = ctx.program(vertex_shader=QUAD_VS, fragment_shader=DOF_FS)
        geo = {"sphere": mesh_sphere(), "cylinder": mesh_cylinder(), "cube": mesh_cube(), "disk": mesh_disk(),
               "torus": mesh_torus()}
        self.vbo = {k: ctx.buffer(v.tobytes()) for k, v in geo.items()}
        self.count = {k: len(v) for k, v in geo.items()}
        self.inst: dict[str, moderngl.Buffer] = {}
        self.inst_cap: dict[str, int] = {}
        self.vao: dict[str, moderngl.VertexArray] = {}
        self.items = {(layer, m): [] for layer in LAYERS for m in MESHES}
        corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, -1], [1, 1], [-1, 1]], "f4")
        self.corner_vbo = ctx.buffer(corners.tobytes())
        self.part_cap, self.part_buf, self.part_vao = 0, None, {}
        self.particles = {"alpha": [], "add": []}
        uv = np.array([[0, 0], [1, 0], [1, 1], [0, 0], [1, 1], [0, 1]], "f4")
        uv_buf = ctx.buffer(uv.tobytes())
        self.quad_vao = ctx.vertex_array(self.quad, [(uv_buf, "2f", "in_uv")])
        self.dof_vao = ctx.vertex_array(self.dof, [(uv_buf, "2f", "in_uv")])
        self.hud_tex: moderngl.Texture | None = None

    # --- queueing -----------------------------------------------------------------------------
    def add(self, mesh: str, model: np.ndarray, color, pattern: int = 0, glow: float = 0.0, layer: str | None = None):
        rgba = tuple(color) + ((1.0,) if len(color) == 3 else ())
        if layer is None:
            layer = "opaque" if rgba[3] >= 0.999 else "blend"
        self.items[(layer, mesh)].append((model, rgba, pattern, glow))

    def particle(self, pos, size: float, color, additive: bool = False):
        self.particles["add" if additive else "alpha"].append((pos, size, color))

    def clear(self):
        for v in self.items.values():
            v.clear()
        for v in self.particles.values():
            v.clear()

    # --- drawing ----------------------------------------------------------------------------------
    def _instances(self, key: str, rows: list, eye) -> tuple[moderngl.VertexArray, int]:
        n = len(rows)
        if eye is not None:                           # back to front for blending
            rows = sorted(rows, key=lambda r: -float(np.linalg.norm(r[0][:3, 3] - eye)))
        data = np.empty((n, 22), "f4")                  # packed in bulk: a per-row loop cost ~3 us an instance
        data[:, :16] = np.stack([r[0] for r in rows]).transpose(0, 2, 1).reshape(n, 16)
        data[:, 16:20] = [r[1] for r in rows]
        data[:, 20:22] = [(r[2], r[3]) for r in rows]
        if self.inst_cap.get(key, 0) < n:
            cap = max(64, 1 << (n - 1).bit_length())
            self.inst[key] = self.ctx.buffer(reserve=cap * 22 * 4, dynamic=True)
            self.inst_cap[key] = cap
            self.vao[key] = self.ctx.vertex_array(self.lit, [(self.vbo[key], "3f 3f", "in_pos", "in_norm"),
                                                             (self.inst[key], "16f 4f 2f/i", "in_model", "in_color", "in_extra")])
        self.inst[key].write(data.tobytes())
        return self.vao[key], n

    def set_scene(self, view: np.ndarray, proj: np.ndarray, cam, lights: dict, t: float):
        self.view, self.proj = view, proj
        for prog in (self.lit, self.part):
            prog["u_view"].write(view.T.astype("f4").tobytes())
            prog["u_proj"].write(proj.T.astype("f4").tobytes())
        self.lit["u_cam"].value = tuple(float(v) for v in cam)
        for name, val in lights.items():
            if name in self.lit:
                self.lit[name].value = tuple(float(v) for v in val) if hasattr(val, "__len__") else float(val)
        if "u_time" in self.lit:
            self.lit["u_time"].value = t
        self.cam = np.asarray(cam, float)

    def draw_layer(self, layer: str):
        ctx = self.ctx
        if layer in ("blend", "view_blend"):
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            ctx.depth_mask = False
        for mesh in MESHES:
            rows = self.items[(layer, mesh)]
            if rows:
                vao, n = self._instances(mesh, rows, self.cam if layer in ("blend", "view_blend") else None)
                vao.render(moderngl.TRIANGLES, instances=n)
        if layer in ("blend", "view_blend"):
            ctx.depth_mask = True
            ctx.disable(moderngl.BLEND)

    def draw_particles(self):
        ctx = self.ctx
        ctx.enable(moderngl.BLEND)
        ctx.depth_mask = False
        for kind in ("alpha", "add"):
            rows = self.particles[kind]
            if not rows:
                continue
            n = len(rows)
            data = np.empty((n, 8), "f4")
            for i, (p, s, c) in enumerate(rows):
                data[i, :3] = p
                data[i, 3] = s
                data[i, 4:8] = c
            if self.part_cap < n:
                self.part_cap = max(256, 1 << (n - 1).bit_length())
                self.part_buf = ctx.buffer(reserve=self.part_cap * 32, dynamic=True)
                self.part_vao = ctx.vertex_array(self.part, [(self.corner_vbo, "2f", "in_corner"),
                                                             (self.part_buf, "3f 1f 4f/i", "in_center", "in_size", "in_color")])
            self.part_buf.write(data.tobytes())
            ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE) if kind == "add" else (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            self.part_vao.render(moderngl.TRIANGLES, instances=n)
        ctx.depth_mask = True
        ctx.disable(moderngl.BLEND)

    def blit_texture(self, tex: moderngl.Texture, rect_px, target_size, flip: bool = False, blend: bool = True):
        """Draw a texture into rect_px = (x, y, w, h) of the bound framebuffer (pixels, y down)."""
        W, H = target_size
        x, y, w, h = rect_px
        x0, x1 = 2 * x / W - 1, 2 * (x + w) / W - 1
        y0, y1 = 1 - 2 * (y + h) / H, 1 - 2 * y / H
        self.quad["u_rect"].value = (x0, y0, x1, y1)
        self.quad["u_flip"].value = 1.0 if flip else 0.0
        tex.use(0)
        if blend:
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.quad_vao.render(moderngl.TRIANGLES)
        self.ctx.enable(moderngl.DEPTH_TEST)
        if blend:
            self.ctx.disable(moderngl.BLEND)

    def blit_dof(self, tex: moderngl.Texture, depth_tex: moderngl.Texture, rect_px, target_size,
                 focus: float = 1.8, dof: float = 0.5, near: float = 0.03, far: float = 40.0,
                 flip: bool = False, blend: bool = False):
        """Draw a texture with depth-of-field blur guided by depth_tex into rect_px."""
        W, H = target_size
        x, y, w, h = rect_px
        x0, x1 = 2 * x / W - 1, 2 * (x + w) / W - 1
        y0, y1 = 1 - 2 * (y + h) / H, 1 - 2 * y / H
        self.dof["u_rect"].value = (x0, y0, x1, y1)
        self.dof["u_flip"].value = 1.0 if flip else 0.0
        self.dof["u_focus"].value = float(focus)
        self.dof["u_dof"].value = float(dof)
        self.dof["u_near"].value = float(near)
        self.dof["u_far"].value = float(far)
        self.dof["u_res"].value = (float(tex.width), float(tex.height))
        self.dof["u_tex"].value = 0
        self.dof["u_depth"].value = 1
        tex.use(0)
        depth_tex.use(1)
        if blend:
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.dof_vao.render(moderngl.TRIANGLES)
        self.ctx.enable(moderngl.DEPTH_TEST)
        if blend:
            self.ctx.disable(moderngl.BLEND)

