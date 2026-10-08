"""Drawing a real neuron shape (sim/realshapes.py) in the brain view's own picture, on the GPU where the brain view is on it (3.1.0 task 9).

  camera_for(view, key)         the camera the view draws `key` ("panel" or "big") with, as a dict (works for the CPU and the GPU view)
  overlay(view, skel, ...)      the skeleton as a transparent picture the size of the big view, to lay over it
  thumbnail(view, skel, size)   the skeleton alone, fitted to a small box, for the neuron inspector's card

Both draw with GPUBrainView.render_segments (instanced nothing: one line list in one buffer, one draw call) and fall back to the same projection
in numpy and pygame lines when the GPU view is off or fails; the two give the same picture. Drawing only: the simulation never sees a shape.
"""
from __future__ import annotations

import numpy as np
import pygame

from kickthefly.game import kick_the_fly as k2

SHAPE_COLOR = (1.0, 0.93, 0.55)
SUPERSAMPLE = 2


def camera_for(view, key: str = "big") -> dict:
    from kickthefly.game.gpu_brainview import _camera_rotation

    if key == "panel" or (key == "big" and view.is_default_view()):
        return dict(R=np.eye(3, dtype=np.float32), center=view.center, zoom=1.0, pan=(0.0, 0.0), zcut=k2.VIEW_ZCUT)
    return dict(R=_camera_rotation(view.yaw, view.pitch), center=view.center, zoom=view.zoom, pan=(view.pan_x, view.pan_y), zcut=1e30)


def project(points: np.ndarray, cam: dict, wh: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """(pixel x, pixel y) float arrays and the depth shade for (n, 3) EM-space points, with the view's own transform."""
    span = cam.get("span") or (k2.VIEW_X[1] - k2.VIEW_X[0], k2.VIEW_Y[1] - k2.VIEW_Y[0])
    rot = (np.asarray(points, np.float32) - np.asarray(cam["center"], np.float32)) @ np.asarray(cam["R"], np.float32).T
    rx = rot[:, 0] * cam["zoom"] + cam["pan"][0] + span[0] * 0.5
    ry = rot[:, 1] * cam["zoom"] + cam["pan"][1] + span[1] * 0.5
    rz = rot[:, 2] * cam["zoom"] + cam["center"][2]
    depth = np.clip(1.2 - (rz - 5000.0) / 50000.0, 0.35, 1.0)
    return rx / span[0] * wh[0], ry / span[1] * wh[1], depth


def _to_surface(raw: np.ndarray, wh: tuple[int, int]) -> pygame.Surface:
    """A premultiplied float RGBA image, rows top first as the view's pictures are, as a pygame surface."""
    a = np.clip(raw, 0.0, 1.0)
    alpha = a[..., 3]
    rgb = np.where(alpha[..., None] > 0, a[..., :3] / np.maximum(alpha[..., None], 1e-6), 0.0)
    img = np.concatenate([np.clip(rgb, 0, 1), alpha[..., None]], axis=2)
    surf = pygame.image.frombuffer((img * 255).astype(np.uint8).tobytes(), (raw.shape[1], raw.shape[0]), "RGBA").copy()
    return pygame.transform.smoothscale(surf, wh) if surf.get_size() != wh else surf


def _cpu_lines(segs: np.ndarray, cam: dict, wh: tuple[int, int], color, shade: bool) -> pygame.Surface:
    big = (wh[0] * SUPERSAMPLE, wh[1] * SUPERSAMPLE)
    surf = pygame.Surface(big, pygame.SRCALPHA)
    a, b = segs[:, 0], segs[:, 1]
    ax, ay, da = project(a, cam, big)
    bx, by, db = project(b, cam, big)
    col = np.array(color) * 255
    for i in range(len(segs)):
        d = float((da[i] + db[i]) / 2) if shade else 1.0
        c = tuple(int(v * d) for v in col) + (235,)
        if max(ax[i], bx[i]) < 0 or max(ay[i], by[i]) < 0 or min(ax[i], bx[i]) > big[0] or min(ay[i], by[i]) > big[1]:
            continue
        pygame.draw.line(surf, c, (float(ax[i]), float(ay[i])), (float(bx[i]), float(by[i])), 1)
    return pygame.transform.smoothscale(surf, wh)


def _render(view, segs: np.ndarray, cam: dict, wh: tuple[int, int], color, shade: bool) -> pygame.Surface:
    big = (wh[0] * SUPERSAMPLE, wh[1] * SUPERSAMPLE)
    gpu = getattr(view, "render_segments", None)
    if gpu is not None:
        raw = gpu(segs, cam, big, color, shade)
        if raw is not None:
            return _to_surface(raw, wh)
    return _cpu_lines(segs, cam, wh, color, shade)


def overlay(view, skel, key: str = "big", color=SHAPE_COLOR) -> pygame.Surface:
    """The skeleton laid over the brain view's `key` picture: a transparent surface of that picture's size."""
    wh = k2.VIEW_SIZES[key]
    return _render(view, skel.segments(), camera_for(view, key), wh, color, True)


def fit_camera(skel, wh: tuple[int, int], yaw_deg: float = 25.0, pitch_deg: float = -15.0, margin: float = 0.92) -> dict:
    """A camera that fits the whole skeleton in `wh` pixels from a slightly turned front view."""
    from kickthefly.game.gpu_brainview import _camera_rotation

    lo, hi = skel.bounds()
    center = ((lo + hi) / 2).astype(np.float32)
    R = _camera_rotation(yaw_deg, pitch_deg)
    pts = (skel.xyz - center) @ R.T
    ext = np.maximum(pts[:, :2].max(axis=0) - pts[:, :2].min(axis=0), 1.0)
    zoom = float(margin * min(wh[0] / ext[0], wh[1] / ext[1]))
    mid = (pts[:, :2].max(axis=0) + pts[:, :2].min(axis=0)) / 2
    return dict(R=R, center=center, zoom=zoom, pan=(float(-mid[0] * zoom), float(-mid[1] * zoom)), zcut=1e30, span=(float(wh[0]), float(wh[1])))


def thumbnail(view, skel, size: tuple[int, int], color=SHAPE_COLOR) -> pygame.Surface:
    """The skeleton alone, fitted into `size`, flat colored (no brain-depth shading)."""
    return _render(view, skel.segments(), fit_camera(skel, size), size, color, False)
