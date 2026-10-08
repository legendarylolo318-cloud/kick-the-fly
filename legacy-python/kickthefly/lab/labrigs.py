"""Lab > Behavior rigs (3.0 day 5): a hub and one scene per rig (tethered flight simulator, fly on a ball, Buridan's paradigm, four-field olfactory
arena). A scene's run goes in a background job (a thread that builds its own lockstep brain and steps it; never the game thread), with a progress
bar and a Cancel button; the result is a trace that the scene then replays at real time. The physics and the tags are game/rigs.py's, the protocols,
the export and the pre-registered assay are lab/rigassay.py's; the page repeats the tags next to every number it shows.

Layout is one scrolling column (the scene, the controls, the readout, the tags), so it holds at larger text and at a narrow window: lines wrap
(Menu.wrapped) and buttons are sized from the font."""
from __future__ import annotations

import math
import time

import numpy as np
import pygame

from kickthefly.core.i18n import tr
from kickthefly.game import rigs
from kickthefly.ui import menu as ui
from kickthefly.ui.bgjob import BgJob, draw_progress

SERIES = {"default": ((57, 135, 229), (217, 89, 38)), "blue-yellow": ((30, 100, 200), (240, 190, 20)), "high-contrast": ((255, 255, 255), (255, 200, 0))}
C = {c: i for i, c in enumerate(rigs.COLS)}
PAGES = {"tethered": "lab_rig_tethered", "ball": "lab_rig_ball", "buridan": "lab_rig_buridan", "fourfield": "lab_rig_fourfield"}
BLURB = {
    "tethered": "The fly is held still. A striped panorama rotates around it; the fly's yaw is read from its steering descending neurons DNa01/DNa02 (right minus left). "
                "Open loop: the panorama turns whatever the fly does. Closed loop: the panorama is also turned by the fly's own yaw, so it can cancel the rotation. "
                "The line from the fly is the heading its yaw would give (read out only: the fly never turns).",
    "ball": "A fly on a spherical treadmill in a virtual world driven by the ball: turning comes from DNa01/02, walking from DNp09. In the bar scene a distant bar is tracked "
            "by LC10; in the panorama scene the world drifts and the fly's turning feeds back. Open loop freezes the VR.",
    "buridan": "A round platform with two opposite stripes beyond its edge. The fly walks wherever its steering and walking neurons take it; the trajectory is recorded. "
               "The control has no stripes.",
    "fourfield": "A square arena in four quadrants, one odor in two opposite ones. The odor drives the olfactory neurons of the game's scent; the preference index is the "
                 "time spent in odor against air. The control delivers no odor.",
}
TAGLINE = {
    "tethered": "CONNECTOME: yaw from DNa01/02; the rotation reaches them through T4/T5. GAME RULE: the EMD stage, the yaw gain, the panorama. MODEL PREDICTION: the response.",
    "ball": "CONNECTOME: turning and walking from DNa01/02 and DNp09. GAME RULE: the ball, the VR, the bar (LC10 tracking by the duel's rule). MODEL PREDICTION: the response.",
    "buridan": "CONNECTOME: steering and speed from DNa01/02 and DNp09. GAME RULE: platform, edge reflection, one stripe tracked at a time (LC10), the measures. MODEL PREDICTION: the trajectory.",
    "fourfield": "CONNECTOME: the odor's receptor neurons. GAME RULE: arena, sharp quadrants, the preference index. MODEL PREDICTION: whether the fly prefers the odor (nothing gives it a reason to).",
}


class _St:
    def __init__(self):
        from kickthefly.lab import rigassay

        self.p = {r: dict(rigassay.SCENES[r]["default"]) for r in rigs.RIGS}
        self.res: dict[str, dict] = {}
        self.job: BgJob | None = None
        self.job_rig = ""
        self.t = 0.0
        self.playing = True
        self.last = time.perf_counter()
        self.record = False
        self.error = ""
        self.brains: dict = {}


def st(m) -> _St:
    s = getattr(m.host, "_rigs_state", None)
    if s is None:
        s = m.host._rigs_state = _St()
    return s


def _colors(m):
    return SERIES.get(m.host.cfg["access.palette"], SERIES["default"])


def _mode(m) -> str:
    v = str(m.host.cfg["brain.individuality"]) if hasattr(m.host, "cfg") else "subtle"
    return v if v in ("off", "subtle", "strong") else "subtle"


def _seed(m) -> int:
    try:
        return int(m.host.cfg["brain.seed"])
    except Exception:
        return 0


def _btn_h(m) -> int:
    return max(34, m.f_text.get_linesize() + 14)


# --- the run ---------------------------------------------------------------------------------------------------------------------------
def start(m, rig: str) -> None:
    from kickthefly.lab import recorder, rigassay

    s = st(m)
    if s.job is not None and s.job.running:
        return
    p = dict(s.p[rig])
    seed, mode = _seed(m), _mode(m)
    folder = recorder.exports_dir() / f"{time.strftime('%Y%m%d-%H%M%S')}-rig-{rig}" if s.record else None

    def work(job: BgJob):
        job.update(0.0, tr("building the fly's brain"))
        res = rigassay.scene_run(rig, seed, mode, folder=folder, progress=lambda f: job.update(f, tr("simulating")), cancel=job.cancel, **p)
        res["params_shown"] = p
        return res

    s.error = ""
    s.job_rig = rig
    s.job = BgJob(f"Rig {rig}", work).start()


def collect(s: _St) -> None:
    """Take a finished job's result to ITS rig, by the job's own label (the rig pages share one job slot)."""
    j = s.job
    if j is None or j.running:
        return
    s.job = None
    if j.error:
        s.error = j.error
    elif j.result is not None and not j.cancelled:
        s.res[j.result["rig"]] = j.result
        s.t, s.playing, s.last = 0.0, True, time.perf_counter()


def _advance(s: _St, res: dict | None) -> None:
    now = time.perf_counter()
    if res is not None and s.playing:
        s.t = min(res["seconds"], s.t + (now - s.last))
        if s.t >= res["seconds"]:
            s.playing = False
    s.last = now


# --- drawing -----------------------------------------------------------------------------------------------------------------------------
def _rows(res: dict) -> np.ndarray:
    return np.asarray(res["trace"], float).reshape(-1, len(rigs.COLS))


def _upto(res: dict, t: float) -> np.ndarray:
    a = _rows(res)
    return a[: max(1, int(np.searchsorted(a[:, 0], t, side="right")))]


def _fly(surf, pos, heading_cw: float, size: int, col) -> None:
    """A fly from above: heading_cw is clockwise from straight up."""
    c, s = math.cos(heading_cw), math.sin(heading_cw)
    pts = [(pos[0] + s * size, pos[1] - c * size), (pos[0] - c * size * 0.6 - s * size * 0.6, pos[1] - s * size * 0.6 + c * size * 0.6),
           (pos[0] + c * size * 0.6 - s * size * 0.6, pos[1] + s * size * 0.6 + c * size * 0.6)]
    pygame.draw.polygon(surf, col, pts)


def _empty(m, surf, box, text: str) -> None:
    pygame.draw.rect(surf, (14, 16, 22), box, border_radius=8)
    pygame.draw.rect(surf, (40, 46, 60), box, 1, border_radius=8)
    m.wrapped(surf, text, (box.x + 16, box.centery - 18), box.w - 32, ui.LABEL, m.f_small, max_lines=4)


def _trace_chart(m, surf, box, res: dict, t: float, col_name: str, label: str, colors) -> None:
    pygame.draw.rect(surf, (14, 16, 22), box, border_radius=6)
    m.text(surf, label, (box.x + 8, box.y + 4), ui.LABEL, m.f_small)
    a = _rows(res)
    if len(a) < 2:
        return
    top = m.f_small.get_linesize() + 6
    plot = pygame.Rect(box.x, box.y + top, box.w, box.h - top)
    y = a[:, C[col_name]]
    lo, hi = float(min(y.min(), 0)), float(max(y.max(), 0)) if float(max(y.max(), 0)) > float(min(y.min(), 0)) else float(min(y.min(), 0)) + 1
    pad = (hi - lo) * 0.1 or 1.0
    lo, hi = lo - pad, hi + pad
    xs = plot.x + 4 + (a[:, 0] / max(1e-9, res["seconds"])) * (plot.w - 8)
    ys = plot.bottom - 4 - (y - lo) / (hi - lo) * (plot.h - 8)
    z = plot.bottom - 4 - (0 - lo) / (hi - lo) * (plot.h - 8)
    pygame.draw.line(surf, (60, 66, 80), (plot.x + 4, z), (plot.right - 4, z))
    pygame.draw.lines(surf, colors[0], False, list(zip(xs.tolist(), ys.tolist())), 2)
    cx = plot.x + 4 + t / max(1e-9, res["seconds"]) * (plot.w - 8)
    pygame.draw.line(surf, colors[1], (cx, plot.y + 2), (cx, plot.bottom - 2))


def draw_tethered(m, surf, box, res, t, colors) -> None:
    pygame.draw.rect(surf, (14, 16, 22), box, border_radius=8)
    cx, cy, R = box.centerx, box.centery, min(box.w, box.h) // 2 - 14
    a = _upto(res, t)
    # the panorama's angle relative to the fly: rightward is clockwise; 24 dark stripes. It turns by the slip, which is omega_ext in open loop
    # and omega_ext - gain * yaw rate in closed loop (3.0 release review: this used the yaw rate without the loop gain)
    dt = res["dt"] * rigs.TRACE_EVERY
    pano = panorama_angle(a, dt)
    n = 24
    for i in range(n):
        if i % 2:
            continue
        a0 = pano + i * 2 * math.pi / n
        a1 = a0 + math.pi / n
        pts = [(cx + math.sin(a0) * (R - 16), cy - math.cos(a0) * (R - 16)), (cx + math.sin(a0) * R, cy - math.cos(a0) * R),
               (cx + math.sin(a1) * R, cy - math.cos(a1) * R), (cx + math.sin(a1) * (R - 16), cy - math.cos(a1) * (R - 16))]
        pygame.draw.polygon(surf, (120, 128, 146), pts)
    pygame.draw.circle(surf, (50, 56, 68), (cx, cy), R - 16, 1)
    row = a[-1]
    _fly(surf, (cx, cy), 0.0, 26, colors[0])          # tethered: the fly never turns; its yaw is only read out (the heading below)
    psi = float(row[C["psi"]])
    pygame.draw.line(surf, colors[1], (cx, cy), (cx + math.sin(psi) * (R - 22), cy - math.cos(psi) * (R - 22)), 2)
    om = row[C["omega_ext"]]
    m.text(surf, tr("panorama {w:+.1f} rad/s", w=om), (box.x + 10, box.y + 8), ui.TEXT, m.f_small)
    m.text(surf, tr("slip {s:+.2f} rad/s", s=row[C["slip"]]), (box.x + 10, box.y + 8 + m.f_small.get_linesize()), ui.LABEL, m.f_small)
    # right and left steering rates
    bx, by, bh = box.right - 70, box.bottom - 14, box.h - 60
    for i, (name, col) in enumerate((("R", colors[0]), ("L", colors[1]))):
        v = float(row[C["turn_r" if name == "R" else "turn_l"]])
        hh = int(min(1.0, v / 20.0) * bh)
        pygame.draw.rect(surf, col, (bx + i * 30, by - hh, 22, hh))
        m.text(surf, name, (bx + i * 30 + 11, by + 2), ui.TEXT, m.f_small, "midtop")
    m.text(surf, tr("DNa01/02 Hz"), (bx - 6, box.bottom - 30 - bh - m.f_small.get_linesize()), ui.LABEL, m.f_small)


def panorama_angle(rows: np.ndarray, dt: float) -> float:
    """How far the panorama has turned relative to the fly (rad, clockwise positive): the integral of the slip the fly saw."""
    return float(np.sum(rows[:, C["slip"]]) * dt)


def _arena(box, half: float, top: int = 24):
    """Arena coordinates in the canvas; `top` keeps the caption line above the arena clear (3.0 release review: at larger text the
    caption sat on the arena's frame)."""
    s = min(box.w, box.h - 2 * top) / 2 - 8
    cx, cy = box.centerx, box.centery + (top - 24) // 2
    return (lambda x, y: (cx + x / half * s, cy - y / half * s)), s, cx, cy


def draw_ball(m, surf, box, res, t, colors) -> None:
    pygame.draw.rect(surf, (14, 16, 22), box, border_radius=8)
    a = _upto(res, t)
    p = res["params"]
    xs, ys = a[:, C["x"]], a[:, C["y"]]
    ext = max(1.0, float(np.abs(_rows(res)[:, [C["x"], C["y"]]]).max()) * 1.15)
    f, s, cx, cy = _arena(box, ext, m.f_small.get_linesize() + 14)
    pygame.draw.rect(surf, (22, 26, 34), (cx - s, cy - s, 2 * s, 2 * s))
    if len(a) > 1:
        pygame.draw.lines(surf, colors[0], False, [f(x, y) for x, y in zip(xs, ys)], 2)
    row = a[-1]
    _fly(surf, f(row[C["x"]], row[C["y"]]), float(row[C["psi"]]), 12, colors[1])
    if p["scene"] == "bar":                         # the bar is at a world bearing: a mark on the border
        b = math.radians(float(p["bar_deg"]))
        bx, by = cx + math.sin(b) * (s + 8), cy - math.cos(b) * (s + 8)
        pygame.draw.circle(surf, (230, 230, 240), (int(bx), int(by)), 7)
        m.text(surf, tr("bar"), (bx, by - 16), ui.LABEL, m.f_small, "midbottom")
    m.text(surf, tr("VR view from above: the fly walks, the ball turns under it"), (box.x + 10, box.y + 8), ui.LABEL, m.f_small)


def draw_buridan(m, surf, box, res, t, colors) -> None:
    pygame.draw.rect(surf, (14, 16, 22), box, border_radius=8)
    a = _upto(res, t)
    R = res["geometry"]["platform_radius"]
    f, s, cx, cy = _arena(box, R / 0.62, m.f_small.get_linesize() + 14)            # the platform fills 62% of the view; the stripes are drawn at the border, not to scale
    pygame.draw.circle(surf, (26, 30, 40), (cx, cy), int(0.62 * s))
    pygame.draw.circle(surf, (70, 76, 92), (cx, cy), int(0.62 * s), 1)
    if res["params"]["mode"] == "stripes":
        for sx in (-1, 1):
            x = cx + sx * (s + 8)
            pygame.draw.rect(surf, (230, 230, 240), (x - 5, cy - s * 0.35, 10, s * 0.7))
    if len(a) > 1:
        pygame.draw.lines(surf, colors[0], False, [f(x, y) for x, y in zip(a[:, C["x"]], a[:, C["y"]])], 2)
    row = a[-1]
    _fly(surf, f(row[C["x"]], row[C["y"]]), float(row[C["psi"]]), 10, colors[1])
    m.text(surf, tr("walking trajectory; the stripes lie beyond the edge (not to scale)"), (box.x + 10, box.y + 8), ui.LABEL, m.f_small)


def draw_fourfield(m, surf, box, res, t, colors) -> None:
    pygame.draw.rect(surf, (14, 16, 22), box, border_radius=8)
    a = _upto(res, t)
    half = res["geometry"]["field_half"]
    f, s, cx, cy = _arena(box, half, m.f_small.get_linesize() + 14)
    for q, (qx, qy) in enumerate(((1, 1), (1, -1), (-1, -1), (-1, 1))):
        odor = q in res["geometry"]["odor_quadrants"] and res["params"]["mode"] == "odor"
        col = (colors[1][0] // 4 + 14, colors[1][1] // 4 + 14, colors[1][2] // 4 + 14) if odor else (22, 26, 34)
        pygame.draw.rect(surf, col, (cx if qx > 0 else cx - s, cy - s if qy > 0 else cy, s, s))
    pygame.draw.rect(surf, (70, 76, 92), (cx - s, cy - s, 2 * s, 2 * s), 1)
    if len(a) > 1:
        pygame.draw.lines(surf, colors[0], False, [f(x, y) for x, y in zip(a[:, C["x"]], a[:, C["y"]])], 1)
    row = a[-1]
    _fly(surf, f(row[C["x"]], row[C["y"]]), float(row[C["psi"]]), 9, ui.INK)
    m.text(surf, tr("odor quadrants tinted") if res["params"]["mode"] == "odor" else tr("sham: no odor anywhere (quadrants as in the odor run)"), (box.x + 10, box.y + 8),
           ui.LABEL, m.f_small)


DRAW = {"tethered": draw_tethered, "ball": draw_ball, "buridan": draw_buridan, "fourfield": draw_fourfield}
CHART = {"tethered": ("steer_hz", "DNa01/02 R - L (Hz)"), "ball": ("yaw_rate", "turning (rad/s)"), "buridan": ("yaw_rate", "turning (rad/s)"), "fourfield": ("speed", "speed (m/s)")}


# --- pages -------------------------------------------------------------------------------------------------------------------------------
def _tags(m, surf, x, y, names) -> int:
    for n in names:
        col = ui.TAG_COLORS.get(n, (150, 120, 220))
        img = m.f_small.render(n, True, (10, 12, 16))
        r = pygame.Rect(x, y, img.get_width() + 12, img.get_height() + 4)
        pygame.draw.rect(surf, col, r, border_radius=5)
        surf.blit(img, (r.x + 6, r.y + 2))
        x = r.right + 6
    return x


def page_hub(m, surf, rect, mouse) -> None:
    s = st(m)
    collect(s)
    m.text(surf, tr("BEHAVIOR RIGS"), (rect.x + 24, rect.y + 16), ui.INK, m.f_head)
    y = m.wrapped(surf, tr("Four classic ways to watch a fly, each its own scene. The fly is a real brain; what it steers and walks with are its descending neurons. "
                           "A result the model misses is shown as a miss."), (rect.x + 24, max(m.under_heading(rect), rect.y + 50)), rect.w - 48, ui.LABEL, m.f_small, max_lines=4) + 8
    bh = max(56, m.f_text.get_linesize() * 2 + 24)
    body = pygame.Rect(rect.x + 16, y, rect.w - 32, rect.h - (y - rect.y) - 80)
    key = "lab_rigs"
    off = int(m.scroll.get(key, 0))
    prev = surf.get_clip()
    surf.set_clip(body)
    m.clip = body
    yy = body.y - off
    for rig in rigs.RIGS:
        r = pygame.Rect(body.x, yy, body.w, bh)
        m.button(surf, r, tr(rigs.RIG_TITLE[rig]), (lambda p=PAGES[rig]: m.show(p)), id=("rigs", rig), tip=tr(BLURB[rig]))
        done = rig in s.res
        if done:
            m.text(surf, tr("a run is ready to replay"), (r.right - 14, r.centery), ui.LABEL, m.f_small, "midright")
        yy += bh + 12
    yy = m.wrapped(surf, tr("Headless: --rig NAME (one fly, recorded in the Lab's format), --rig-assay NAME (the pre-registered assay and its off control), and the rig_*.yaml "
                            "protocols. docs/rigs.md has the criteria and the results."), (body.x + 4, yy + 4), body.w - 8, ui.LABEL, m.f_small, max_lines=5)
    m.content_h[key] = max(0, yy + off - body.bottom + 8)
    surf.set_clip(prev)
    m.clip = None
    m.button(surf, (rect.right - 164, rect.bottom - 58, 140, 42), tr("Back"), m.back, style="primary", id=("rigs", "back"))


def _controls(m, surf, x, y, w, rig: str, s: _St) -> int:
    p = s.p[rig]
    busy = s.job is not None and s.job.running
    h = _btn_h(m)
    from kickthefly.lab import rigassay

    modes = rigassay.SCENES[rig]["mode"]
    labels = {"open": tr("Open loop"), "closed": tr("Closed loop"), "stripes": tr("Stripes"), "none": tr("No stripes"), "odor": tr("Odor"), "sham": tr("Sham (no odor)")}
    m.segmented(surf, (x, y, min(w, 200 * len(modes)), h), [labels[k] for k in modes], modes.index(p["mode"]), lambda i: p.__setitem__("mode", modes[i]),
                id=("rig_mode", rig), enabled=not busy, tip=tr("The rig's condition; the other one is its control."))
    y += h + 8
    if rig == "ball":
        m.segmented(surf, (x, y, min(w, 400), h), [tr("Bar"), tr("Panorama")], 0 if p["scene"] == "bar" else 1,
                    lambda i: p.__setitem__("scene", "bar" if i == 0 else "panorama"), id=("rig_scene", rig), enabled=not busy,
                    tip=tr("A distant bar to fixate, or a striped panorama that drifts."))
        y += h + 8
        if p["scene"] == "bar":
            m.segmented(surf, (x, y, min(w, 400), h), [tr("Bar on the right"), tr("Bar on the left")], 0 if p["bar_deg"] > 0 else 1,
                        lambda i: p.__setitem__("bar_deg", 90.0 if i == 0 else -90.0), id=("rig_bar", rig), enabled=not busy)
            y += h + 8
    sliders = {"tethered": [("omega", tr("Rotation (rad/s)"), 0.25, 5.0, 0.25, "{:.2f}"), ("gain", tr("Loop gain"), 0.0, 2.0, 0.25, "{:.2f}"), ("seconds", tr("Seconds"), 9.0, 30.0, 1.0, "{:.0f}")],
               "ball": [("omega", tr("Panorama drift (rad/s)"), -3.0, 3.0, 0.25, "{:+.2f}"), ("seconds", tr("Seconds"), 10.0, 60.0, 5.0, "{:.0f}")],
               "buridan": [("seconds", tr("Seconds"), 20.0, 180.0, 10.0, "{:.0f}")], "fourfield": [("seconds", tr("Seconds"), 20.0, 180.0, 10.0, "{:.0f}")]}[rig]
    for key, label, lo, hi, step, fmt in sliders:
        if rig == "tethered" and key == "gain" and p["mode"] != "closed":
            continue
        if rig == "ball" and key == "omega" and p["scene"] != "panorama":
            continue
        m.text(surf, label, (x, y + h // 2), ui.TEXT, m.f_small, "midleft")
        lw = min(w // 3, 220)
        m.slider(surf, (x + lw, y, max(160, w - lw), h), float(p[key]), lo, hi, step, fmt, lambda v, k=key: p.__setitem__(k, v), lambda: None, id=("rig_sl", rig, key),
                 enabled=not busy)
        y += h + 6
    m.toggle(surf, (x, y, 120, h), s.record, lambda v: setattr(s, "record", v), id=("rig_rec", rig), enabled=not busy,
             tip=tr("Also write the run in the Lab's format (spikes, rates, group rates, kinematics, metadata, trace CSV) into your exports folder."))
    m.text(surf, tr("Record to the exports folder"), (x + 130, y + h // 2), ui.TEXT, m.f_small, "midleft")
    y += h + 8
    m.button(surf, (x, y, 150, h), tr("Run"), lambda: start(m, rig), style="primary", id=("rig_run", rig), enabled=not busy,
             tip=tr("Builds one fly (the Settings random seed and individuality) and runs the rig. Takes about a third of the simulated time in real time, plus a few seconds to build."))
    res = s.res.get(rig)
    m.button(surf, (x + 160, y, 150, h), tr("Replay"), lambda: (setattr(s, "t", 0.0), setattr(s, "playing", True)), id=("rig_replay", rig), enabled=res is not None and not busy)
    m.button(surf, (x + 320, y, 150, h), tr("Pause") if s.playing else tr("Play"), lambda: setattr(s, "playing", not s.playing), id=("rig_pause", rig),
             enabled=res is not None and not busy)
    return y + h + 10


def scene_page(rig: str):
    def page(m, surf, rect, mouse) -> None:
        s = st(m)
        collect(s)
        res = s.res.get(rig)
        _advance(s, res)
        colors = _colors(m)
        key = PAGES[rig]
        m.text(surf, tr(rigs.RIG_TITLE[rig]), (rect.x + 24, rect.y + 16), ui.INK, m.f_head)
        body = pygame.Rect(rect.x + 16, rect.y + 56, rect.w - 32, rect.h - 56 - 80)
        off = int(m.scroll.get(key, 0))
        prev = surf.get_clip()
        surf.set_clip(body)
        m.clip = body
        x, w = body.x + 8, body.w - 40
        y = body.y - off
        y = m.wrapped(surf, tr(BLURB[rig]), (x, y), w, ui.LABEL, m.f_small, max_lines=8) + 6
        wide = w >= 940
        cw = int(min(w * 0.56, 640)) if wide else min(w, 640)
        canvas = pygame.Rect(x, y, cw, min(cw * 3 // 4, max(300, body.h - 60)) + (20 if not wide else 0))
        if res is None:
            _empty(m, surf, canvas, tr("Nothing run yet. Choose a condition and press Run; the run is replayed here at real time."))
        else:
            DRAW[rig](m, surf, canvas, res, s.t, colors)
        yl = canvas.bottom + 6
        if res is not None:
            col, label = CHART[rig]
            chart = pygame.Rect(x, yl, canvas.w, max(80, m.f_small.get_linesize() * 5))
            _trace_chart(m, surf, chart, res, s.t, col, tr(label), colors)
            yl = chart.bottom + 8
        if wide:
            yr = _controls(m, surf, x + cw + 24, y, w - cw - 24, rig, s)
            y = max(yl, yr)
        else:
            y = _controls(m, surf, x, yl, w, rig, s)
        if res is not None:
            m.text(surf, tr("What this run measured") + f" (seed {res['seed']}, individuality {res['individuality']})", (x, y), ui.AMBER, m.f_bold)
            y += m.f_bold.get_linesize() + 2
            for k, v in res["summary"].items():
                if v is None:
                    continue
                txt = f"{k.replace('_', ' ')}: {v:.3g}" if isinstance(v, float) else f"{k.replace('_', ' ')}: {v}"
                y = m.wrapped(surf, txt, (x + 8, y), w - 16, ui.TEXT, m.f_small, max_lines=2)
            if res.get("files"):
                y = m.wrapped(surf, tr("Written to your exports folder: {n} files.", n=len(res["files"])), (x + 8, y + 2), w - 16, ui.LABEL, m.f_small, max_lines=2)
            y += 6
        x2 = _tags(m, surf, x, y, ["CONNECTOME", "GAME RULE", "MODEL PREDICTION"])
        y += m.f_small.get_linesize() + 10
        y = m.wrapped(surf, tr(TAGLINE[rig]), (x, y), w, ui.LABEL, m.f_small, max_lines=6) + 4
        y = m.wrapped(surf, tr("One run is one fly: it shows what the model does, not a result. The pre-registered assay over ten flies and its off control is "
                               "--headless --rig-assay {name}; docs/rigs.md has its result.", name=rig), (x, y), w, ui.LABEL, m.f_small, max_lines=5) + 8
        m.content_h[key] = max(0, y + off - body.bottom + 8)
        surf.set_clip(prev)
        m.clip = None
        if s.error:
            m.text(surf, s.error, (rect.x + 24, rect.bottom - 74), ui.BAD, m.f_small)
        if s.job is not None and s.job.running:
            draw_progress(m, surf, pygame.Rect(rect.x + 24, rect.bottom - 56, rect.w - 220, 12), s.job, ui)
        m.button(surf, (rect.right - 164, rect.bottom - 58, 140, 42), tr("Back"), m.back, style="primary", id=(key, "back"))
    return page


PAGE_FUNCS = {PAGES[r]: scene_page(r) for r in rigs.RIGS}


def install(menu) -> None:
    menu.pages["lab_rigs"] = page_hub
    menu.pages.update(PAGE_FUNCS)


def pad_nav(host, down) -> bool:
    """Gamepad on a rig scene: use runs it, the bumpers change the condition, the kill-cam and big-view buttons change the length, B or Start goes back."""
    m = host.menu
    rig = next((r for r, pg in PAGES.items() if m.screen == pg), None)
    if rig is None:
        return False
    s = st(m)
    if down & {"crouch", "menu"}:
        m.back()
        return True
    busy = s.job is not None and s.job.running
    if busy:
        return False
    from kickthefly.lab import rigassay

    p = s.p[rig]
    used = False
    if down & {"tool_next", "tool_prev"}:
        modes = rigassay.SCENES[rig]["mode"]
        p["mode"] = modes[(modes.index(p["mode"]) + 1) % len(modes)]
        used = True
    if down & {"killcam", "big_view"}:
        d = 10.0 if "killcam" in down else -10.0
        p["seconds"] = float(min(180.0, max(10.0, p["seconds"] + d)))
        used = True
    if "use" in down:
        start(m, rig)
        used = True
    return used
