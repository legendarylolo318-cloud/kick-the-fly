"""The brain view on the GPU (3.1.0 task 4). GAME RULE (how the brain is drawn); it reads the same activity as the CPU view and draws the
same picture.

The CPU view (`BrainView` in kick_the_fly.py) keeps a sparse matrix from neurons to pixels and multiplies it by each neuron's activity 20
times a second; turning the camera rebuilds the matrix. Here every neuron is an *instance*: its 21 fiber and arbor sample points sit in one
static storage buffer (position, weight) and its colors in another, both uploaded once. Each frame uploads one number per neuron (its
activity, and whether to sparkle it) and draws every sample as a one-pixel point with additive blending into a float image. That image is
what the sparse product was, so the rest of the pipeline (tone map, bloom, HUD, see-through panel alpha, the Imaging recolor) is the same
code. The camera is a uniform: turning, panning and zooming rebuild nothing but the picking table.

Picking is unchanged: `spark_pix` (each neuron's cell-body pixel, read by the neuron inspector, the path overlay and the labels) is still
a CPU array, now computed for the camera without the matrix.

All GL runs on a thread of its own that owns an offscreen (EGL) context, so it never touches the game window's context and works from the
view thread or the main thread alike. If OpenGL 4.3 compute-class features are missing, a driver call fails, or the setting says CPU, the view
falls back to the CPU path (`BrainView`) and says so once. The picture is the same to within float summation order.
"""
from __future__ import annotations

import queue
import threading
from concurrent.futures import Future

import numpy as np

from kickthefly.core.crash import log
from kickthefly.game import kick_the_fly as k2

_VS = """
#version 430
layout(std430, binding = 0) readonly buffer BPts { vec4 pts[]; };      // x, y, z, weight (0: not drawn): static
layout(std430, binding = 1) readonly buffer BCol { vec4 col[]; };      // rgb, hot flag: static per palette
layout(std430, binding = 2) readonly buffer BAct { vec2 act[]; };      // amplitude, sparkle flag: uploaded every frame
uniform int samples;
uniform vec2 size;
uniform mat3 R;
uniform vec3 center;
uniform float zoom;
uniform vec2 pan;
uniform vec2 span;
uniform float zcut;
uniform float gain;
uniform int use_amp;
uniform int sparkle_pass;
uniform vec3 hot_color;
out vec3 v_col;

void main() {
    int nid = sparkle_pass != 0 ? gl_VertexID : gl_InstanceID;
    int k = sparkle_pass != 0 ? 0 : gl_VertexID;
    vec4 p = pts[nid * samples + k];
    vec3 rot = R * (p.xyz - center);
    float rx = rot.x * zoom + pan.x + span.x * 0.5;
    float ry = rot.y * zoom + pan.y + span.y * 0.5;
    float rz = rot.z * zoom + center.z;
    int px = int(rx / span.x * size.x);          // truncation toward zero, as numpy's astype(int32) in the CPU view
    int py = int(ry / span.y * size.y);
    float amp = (use_amp != 0) ? act[nid].x : 1.0;
    bool drawn = p.w > 0.0 && p.z < zcut && px >= 0 && px < int(size.x) && py >= 0 && py < int(size.y);
    if (sparkle_pass != 0) drawn = drawn && act[nid].y > 0.5;
    else drawn = drawn && amp != 0.0;
    if (!drawn) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); v_col = vec3(0.0); gl_PointSize = 1.0; return; }
    if (sparkle_pass != 0) {
        v_col = col[nid].w > 0.5 ? hot_color * 3.0 : vec3(0.7);
    } else {
        float depth = clamp(1.2 - (rz - 5000.0) / 50000.0, 0.35, 1.0);
        v_col = col[nid].rgb * (p.w * depth * amp * gain);
    }
    gl_Position = vec4((float(px) + 0.5) / size.x * 2.0 - 1.0, (float(py) + 0.5) / size.y * 2.0 - 1.0, 0.0, 1.0);
    gl_PointSize = 1.0;
}
"""
_FS = """
#version 430
in vec3 v_col;
out vec4 f_col;
void main() { f_col = vec4(v_col, 1.0); }
"""


_LINE_VS = """
#version 430
in vec3 in_vert;
uniform mat3 R;
uniform vec3 center;
uniform float zoom;
uniform vec2 pan;
uniform vec2 span;
uniform float shade;                    // 1: shade by depth as the brain view does; 0: flat
out float v_depth;
void main() {
    vec3 rot = R * (in_vert - center);
    float rx = rot.x * zoom + pan.x + span.x * 0.5;
    float ry = rot.y * zoom + pan.y + span.y * 0.5;
    float rz = rot.z * zoom + center.z;
    v_depth = shade > 0.5 ? clamp(1.2 - (rz - 5000.0) / 50000.0, 0.35, 1.0) : 1.0;
    gl_Position = vec4(rx / span.x * 2.0 - 1.0, ry / span.y * 2.0 - 1.0, 0.0, 1.0);
}
"""
_LINE_FS = """
#version 430
in float v_depth;
uniform vec3 color;
out vec4 f_col;
void main() { f_col = vec4(color * v_depth, 1.0) * 0.92; }
"""


class _Worker:
    """A thread that owns one offscreen GL context; callers hand it functions and wait (the _GLGroup pattern, in miniature)."""

    def __init__(self) -> None:
        self._q: queue.Queue = queue.Queue()
        self.ctx = None
        self.error: Exception | None = None
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._run, name="gl-brainview", daemon=True)
        self.thread.start()
        self.ready.wait()
        if self.error is not None:
            raise self.error

    def _run(self) -> None:
        try:
            import moderngl

            from kickthefly.sim.connectome.backends import _create_gl_context
            self.ctx = _create_gl_context()
            if self.ctx.version_code < 430:
                raise RuntimeError(f"OpenGL {self.ctx.version_code / 100:.1f} < 4.3: no storage buffers")
            self.moderngl = moderngl
        except Exception as e:
            self.error = e
            self.ready.set()
            return
        self.ready.set()
        while True:
            item = self._q.get()
            if item is None:
                break
            fn, fut = item
            try:
                fut.set_result(fn())
            except Exception as e:
                fut.set_exception(e)
        try:
            self.ctx.release()
        except Exception:
            pass

    def call(self, fn):
        fut: Future = Future()
        self._q.put((fn, fut))
        return fut.result()

    def close(self) -> None:
        self._q.put(None)


def _camera_rotation(yaw: float, pitch: float) -> np.ndarray:
    """The rotation BrainView._recompute_big builds (applied as rot = R @ (p - center))."""
    import math
    ry, rp = math.radians(yaw), math.radians(pitch)
    cy, sy, cp, sp = math.cos(ry), math.sin(ry), math.cos(rp), math.sin(rp)
    return np.array([[cy, 0.0, sy], [sp * sy, cp, -sp * cy], [-cp * sy, sp, cp * cy]], dtype=np.float32)


class GPUBrainView(k2.BrainView):
    """BrainView whose per-frame light and rotated views are drawn on the GPU. `pref` is "auto", "gpu" or "cpu" (the setting)."""

    def __init__(self, *args, **kw):
        self.pref = "auto"
        self._gpu: _Worker | None = None
        self._gpu_failed = False
        self._gpu_state: dict = {}
        self._region_base_cache: dict = {}
        super().__init__(*args, **kw)
        self.engine_note = "CPU (sparse matrices)"

    # --- when the GPU is used --------------------------------------------------------------------------------------------------
    @property
    def on_gpu(self) -> bool:
        return self._gpu is not None and not self._gpu_failed and self.pref != "cpu"

    def _want_gpu(self) -> bool:
        if self.pref == "cpu" or self._gpu_failed:
            return False
        if self._gpu is None:
            try:
                if self.pref == "auto":
                    from kickthefly.sim.connectome import backends
                    if not backends.gpu_capable()[0]:
                        raise RuntimeError("no real GPU with OpenGL 4.3 here")
                self._gpu = _Worker()
                self._gpu.call(self._upload_static)
                self.engine_note = f"GPU ({self._gpu_state['device']}, {self.n:,} neurons as instanced points)"
                log.info("brain view on the GPU: %s", self.engine_note)
            except Exception as e:
                self._fail(e)
                return False
        return True

    def _fail(self, e: Exception) -> None:
        if not self._gpu_failed:
            log.warning("the GPU brain view is off (%s: %s); drawing it on the CPU", type(e).__name__, e)
        self._gpu_failed = True
        self.engine_note = f"CPU (sparse matrices; the GPU view failed: {type(e).__name__})"
        if self._gpu is not None:
            try:
                self._gpu.close()
            except Exception:
                pass
            self._gpu = None
        # the rotated views the GPU made have no sparse matrix: forget them, and rebuild the current one the CPU way
        self._preset_cache = {"front": self._preset_cache["front"]}
        self._dirty_big = True

    # --- the GPU side (these run on the worker thread) --------------------------------------------------------------------------
    def _upload_static(self) -> None:
        ctx, mgl = self._gpu.ctx, self._gpu.moderngl
        n, S = self.n, self.pts.shape[1]
        pts = np.empty((n, S, 4), np.float32)
        pts[..., :3] = self.pts
        pts[..., 3] = (self.wts[None, :] * self.ok[:, None]).astype(np.float32)
        st = self._gpu_state
        st["pts"] = ctx.buffer(pts.tobytes())
        del pts
        st["act"] = ctx.buffer(reserve=n * 8)
        st["prog"] = ctx.program(vertex_shader=_VS, fragment_shader=_FS)
        st["vao"] = ctx.vertex_array(st["prog"], [])
        st["fbo"] = {}
        st["col"] = {}
        st["S"] = S
        st["device"] = ctx.info.get("GL_RENDERER", "OpenGL GPU")
        self._upload_colors("neuron", self.tint)
        self._upload_colors("region", self.col_region * 1.8)
        self._upload_colors("structure", self.col)
        self._upload_colors("region_base", self.col_region * 0.45)

    def _upload_colors(self, name: str, rgb: np.ndarray) -> None:
        ctx = self._gpu.ctx
        c = np.empty((self.n, 4), np.float32)
        c[:, :3] = rgb
        c[:, 3] = self.hot_mask
        st = self._gpu_state
        if name in st["col"]:
            st["col"][name].write(c.tobytes())
        else:
            st["col"][name] = ctx.buffer(c.tobytes())

    def _target(self, key_wh: tuple[int, int]):
        st, ctx = self._gpu_state, self._gpu.ctx
        if key_wh not in st["fbo"]:
            tex = ctx.texture(key_wh, 4, dtype="f4")
            st["fbo"][key_wh] = (tex, ctx.framebuffer(color_attachments=[tex]))
        return st["fbo"][key_wh]

    def _uniforms(self, cam: dict, wh: tuple[int, int], gain: float, use_amp: bool, sparkle: bool) -> None:
        prog = self._gpu_state["prog"]
        prog["samples"] = self._gpu_state["S"]
        prog["size"] = (float(wh[0]), float(wh[1]))
        prog["R"].write(np.ascontiguousarray(cam["R"].T, dtype=np.float32).tobytes())      # GLSL matrices are column-major
        prog["center"] = tuple(float(v) for v in cam["center"])
        prog["zoom"] = float(cam["zoom"])
        prog["pan"] = (float(cam["pan"][0]), float(cam["pan"][1]))
        prog["span"] = (k2.VIEW_X[1] - k2.VIEW_X[0], k2.VIEW_Y[1] - k2.VIEW_Y[0])
        prog["zcut"] = float(cam["zcut"])
        prog["gain"] = float(gain)
        prog["use_amp"] = int(use_amp)
        prog["sparkle_pass"] = int(sparkle)
        prog["hot_color"] = tuple(float(v) for v in self.hot)

    def _draw(self, cam: dict, wh: tuple[int, int], color: str, gain: float, amp: np.ndarray | None, sparks: np.ndarray | None) -> np.ndarray:
        ctx, mgl, st = self._gpu.ctx, self._gpu.moderngl, self._gpu_state
        tex, fbo = self._target(wh)
        fbo.use()
        ctx.viewport = (0, 0, wh[0], wh[1])
        ctx.clear(0.0, 0.0, 0.0, 0.0)
        ctx.enable(mgl.BLEND | mgl.PROGRAM_POINT_SIZE)
        ctx.blend_func = mgl.ONE, mgl.ONE
        ctx.blend_equation = mgl.FUNC_ADD
        st["pts"].bind_to_storage_buffer(0)
        st["col"][color].bind_to_storage_buffer(1)
        if amp is not None or sparks is not None:
            a = np.zeros((self.n, 2), np.float32)
            if amp is not None:
                a[:, 0] = amp
            if sparks is not None and len(sparks):
                a[sparks, 1] = 1.0
            st["act"].write(a.tobytes())
        st["act"].bind_to_storage_buffer(2)
        self._uniforms(cam, wh, gain, amp is not None, False)
        if amp is not None or color in ("structure", "region_base"):
            st["vao"].render(mgl.POINTS, vertices=st["S"], instances=self.n)
        if sparks is not None and len(sparks):
            self._uniforms(cam, wh, gain, True, True)
            st["vao"].render(mgl.POINTS, vertices=self.n)
        ctx.memory_barrier(mgl.BUFFER_UPDATE_BARRIER_BIT)
        raw = np.frombuffer(fbo.read(components=4, dtype="f4"), np.float32).reshape(wh[1], wh[0], 4)
        return raw[..., :3].reshape(-1, 3).copy()

    # --- lines (a real neuron shape: the neuron inspector and the big view) --------------------------------------------------------
    def _draw_lines(self, segs: np.ndarray, cam: dict, wh: tuple[int, int], color, shade: bool) -> np.ndarray:
        ctx, mgl, st = self._gpu.ctx, self._gpu.moderngl, self._gpu_state
        if "lprog" not in st:
            st["lprog"] = ctx.program(vertex_shader=_LINE_VS, fragment_shader=_LINE_FS)
        prog = st["lprog"]
        verts = np.ascontiguousarray(segs.reshape(-1, 3), np.float32)
        vbo = ctx.buffer(verts.tobytes())
        vao = ctx.vertex_array(prog, [(vbo, "3f", "in_vert")])
        tex, fbo = self._target(wh)
        fbo.use()
        ctx.viewport = (0, 0, wh[0], wh[1])
        ctx.clear(0.0, 0.0, 0.0, 0.0)
        ctx.enable(mgl.BLEND)
        ctx.blend_func = mgl.ONE, mgl.ONE_MINUS_SRC_ALPHA
        span = cam.get("span") or (k2.VIEW_X[1] - k2.VIEW_X[0], k2.VIEW_Y[1] - k2.VIEW_Y[0])
        prog["R"].write(np.ascontiguousarray(cam["R"].T, dtype=np.float32).tobytes())
        prog["center"] = tuple(float(v) for v in cam["center"])
        prog["zoom"] = float(cam["zoom"])
        prog["pan"] = (float(cam["pan"][0]), float(cam["pan"][1]))
        prog["span"] = (float(span[0]), float(span[1]))
        prog["shade"] = 1.0 if shade else 0.0
        prog["color"] = tuple(float(c) for c in color)
        vao.render(mgl.LINES, vertices=len(verts))
        raw = np.frombuffer(fbo.read(components=4, dtype="f4"), np.float32).reshape(wh[1], wh[0], 4).copy()
        vao.release()
        vbo.release()
        return raw

    def render_segments(self, segs: np.ndarray, cam: dict, wh: tuple[int, int], color=(1.0, 0.95, 0.6), shade: bool = True) -> np.ndarray | None:
        """(h, w, 4) float RGBA, premultiplied, of line segments (m, 2, 3) in EM space, drawn on the GPU with `cam` (as _camera makes). None if the
        GPU view is off; the caller then draws on the CPU (game/shape_draw.py)."""
        if not self._want_gpu():
            return None
        try:
            return self._gpu.call(lambda: self._draw_lines(segs, cam, wh, color, shade))
        except Exception as e:
            self._fail(e)
            return None

    # --- the camera the next frame uses ----------------------------------------------------------------------------------------------
    def _camera(self, key: str) -> dict:
        if key == "panel" or (key == "big" and self.is_default_view()):
            return dict(R=np.eye(3, dtype=np.float32), center=self.center, zoom=1.0, pan=(0.0, 0.0), zcut=k2.VIEW_ZCUT)
        return dict(R=_camera_rotation(self.yaw, self.pitch), center=self.center, zoom=self.zoom, pan=(self.pan_x, self.pan_y), zcut=1e30)

    def _cam_key(self, key: str) -> tuple:
        c = self._camera(key)
        return (key, round(self.yaw, 3), round(self.pitch, 3), round(c["pan"][0], 2), round(c["pan"][1], 2), round(c["zoom"], 4), c["zcut"])

    # --- BrainView's two hooks ------------------------------------------------------------------------------------------------------
    def _accumulate(self, key: str, amp: np.ndarray, mode: str, sparks: np.ndarray | None = None) -> np.ndarray:
        if not self._want_gpu():
            return super()._accumulate(key, amp, mode, sparks)
        try:
            wh = k2.VIEW_SIZES[key]
            cam = self._camera(key)
            gain = float(self.gain[key])
            amp32 = np.ascontiguousarray(amp, np.float32)
            sp = None if sparks is None else np.asarray(sparks, np.int64)
            return self._gpu.call(lambda: self._draw(cam, wh, mode, gain, amp32, sp))
        except Exception as e:
            self._fail(e)
            if getattr(self, "_dirty_big", False):
                self._recompute_big()
            return super()._accumulate(key, amp, mode, sparks)

    def _region_base(self, key: str) -> np.ndarray:
        if not self._want_gpu():
            return super()._region_base(key)
        ck = self._cam_key(key)
        if ck in self._region_base_cache:
            return self._region_base_cache[ck]
        try:
            wh = k2.VIEW_SIZES[key]
            cam = self._camera(key)
            img = self._gpu.call(lambda: self._draw(cam, wh, "region_base", 1.0, None, None))
        except Exception as e:
            self._fail(e)
            return super()._region_base(key)
        if len(self._region_base_cache) > 12:
            self._region_base_cache.clear()
        self._region_base_cache[ck] = img
        return img

    def set_palette(self, name: str) -> None:
        super().set_palette(name)
        if self._gpu is not None and not self._gpu_failed:
            try:
                self._gpu.call(lambda: self._upload_colors("neuron", self.tint))
            except Exception as e:
                self._fail(e)

    def _recompute_big(self) -> None:
        """Rotated big view: the picking table on the CPU (cell bodies only, n pixels), the structure image and its normalization on the GPU."""
        if self.is_default_view() or not self._want_gpu():
            return super()._recompute_big()
        with self._lock:
            self._dirty_big = False
            cam = self._camera("big")
            w, h = k2.VIEW_SIZES["big"]
            span_x, span_y = k2.VIEW_X[1] - k2.VIEW_X[0], k2.VIEW_Y[1] - k2.VIEW_Y[0]
            rot = (self.pts[:, 0, :] - self.center) @ cam["R"].T                     # the cell-body sample only
            px = ((rot[:, 0] * self.zoom + self.pan_x + span_x * 0.5) / span_x * w).astype(np.int32)
            py = ((rot[:, 1] * self.zoom + self.pan_y + span_y * 0.5) / span_y * h).astype(np.int32)
            m = self.ok & (px >= 0) & (px < w) & (py >= 0) & (py < h)
            spark = np.where(m, py * w + px, -1)
            try:
                struct = self._gpu.call(lambda: self._draw(cam, (w, h), "structure", 1.0, None, None))
            except Exception as e:
                self._fail(e)
                return super()._recompute_big()
            p99 = float(np.percentile(struct.max(1), 99.0)) or 1.0
            self.M["big"] = None                     # no sparse matrix for a camera the GPU made
            self.base["big"] = struct * (0.5 / p99)
            self.gain["big"] = 3.4 / p99
            self.spark_pix["big"] = spark


def make_view(*args, **kw) -> k2.BrainView:
    """The brain view the game uses: the GPU one when ModernGL is importable (it checks for a GPU itself, at the first frame), else the
    CPU one. KICK_THE_FLY_BRAINVIEW=cpu forces the CPU view."""
    import os
    try:
        import moderngl  # noqa: F401
    except Exception:
        return k2.BrainView(*args, **kw)
    if os.environ.get("KICK_THE_FLY_BRAINVIEW", "").lower() == "cpu":
        return k2.BrainView(*args, **kw)
    return GPUBrainView(*args, **kw)
