"""Fly's-eye view (3.1.0 task 11, GAME RULE): the scene as a fly's eyes would sample it. VISUAL ONLY: this picture is never given to the brain.

(The README records that streaming pixels through the simulation's own photoreceptors failed, a looming image stayed inside the brain's random flicker, so the
game drives LPLC2/LC4 directly from how fast an object grows in the fly's view. Nothing here changes that: the view below is drawn from the same scene, for you to look
at, and the panel beside it shows the activity the brain really has; neither feeds the other.)

What it does to the picture:
  - WIDE FIELD. The scene is rendered from the fly's head three times (straight ahead and 90 degrees to each side, 100 degrees each) and the eyes sample the
    union: about 280 degrees across, against the 70 degrees you normally look through. Each eye covers about 165 degrees, overlapping about 30 degrees in front.
  - OMMATIDIA. Each eye is a hexagonal lattice of ommatidia spaced 5 degrees apart (a fly's are about 4.6; about 700 an eye, as here), each seeing one color over a
    5 degree blur, drawn as a hexagon. Fine detail below that is gone, as it is for the fly.
  - SPECTRAL SENSITIVITY, approximately. Flies see UV and blue and green but hardly red. The display's red, green and blue primaries are approximated by Gaussians, and
    each photoreceptor class by a Gaussian at its textbook peak: R1-R6 480 nm (the broad motion/luminance cells; their UV peak at 360 is dark on a screen), R7 345 nm
    (UV: a monitor emits none, so these are always dark here), R8 437 nm (blue) and 520 nm (green). The result is shown in false color (R1-R6 warm, R8 green and blue, R7
    violet), so red things look dim and blue and green ones bright. These are peak wavelengths and widths from the literature, not measured spectra: a sketch, not a measurement.
  - It is tagged GAME RULE on screen, with the reason it is not fed to the brain.

Pure numpy and pygame: no OpenGL here (the 3D game renders the three views; this module samples and draws them).
"""
from __future__ import annotations

import math

import numpy as np
import pygame

TAG = "GAME RULE"
NOTE = "visual only: this picture is not fed to the brain (pixel streaming through the photoreceptors failed; looming drives LPLC2/LC4 directly)"
SPACING_DEG = 5.0
VIEW_FOV_DEG = 100.0
VIEW_YAWS_DEG = (-90.0, 0.0, 90.0)
EYE_AZ_DEG = (-152.0, 15.0)                  # the left eye's azimuth range (azimuth: 0 ahead, negative to the left); the right eye mirrors it
EL_DEG = (-48.0, 48.0)

# --- spectral sensitivity ------------------------------------------------------------------------------------------------------------
WAVES = np.arange(380.0, 701.0, 5.0)
PRIMARIES = {"R": (620.0, 22.0), "G": (535.0, 32.0), "B": (455.0, 20.0)}               # an LCD's emission peaks and widths (approximate)
# class: [(peak nm, sigma nm, weight), ...]  (R1-R6 also has the UV peak of its sensitizing pigment: dark on a monitor)
RECEPTORS = {"R1-6": [(480.0, 58.0, 1.0), (360.0, 26.0, 0.5)], "R7": [(345.0, 22.0, 1.0)], "R8p": [(437.0, 40.0, 1.0)], "R8y": [(520.0, 50.0, 1.0)]}


def _gauss(mu: float, sd: float) -> np.ndarray:
    return np.exp(-0.5 * ((WAVES - mu) / sd) ** 2)


def spectral_matrix() -> np.ndarray:
    """(4, 3): the response of R1-6, R7, R8p, R8y to display R, G, B at unit intensity, scaled so that white (1, 1, 1) gives R1-6 = R8p = R8y = 1."""
    prim = np.stack([_gauss(*PRIMARIES[k]) for k in "RGB"])
    sens = np.stack([sum(w * _gauss(mu, sd) for mu, sd, w in RECEPTORS[c]) for c in ("R1-6", "R7", "R8p", "R8y")])
    m = sens @ prim.T
    white = m.sum(axis=1)
    scale = np.array([white[0], white[0], white[2], white[3]])               # R7 keeps R1-6's scale: it is dark, and stays dark
    return m / scale[:, None]


SPECTRAL = spectral_matrix()


def to_linear(rgb8: np.ndarray) -> np.ndarray:
    """sRGB bytes to linear light (0..1)."""
    c = np.asarray(rgb8, np.float32) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def photoreceptors(rgb8: np.ndarray) -> np.ndarray:
    """(..., 3) display colors to (..., 4) responses R1-6, R7, R8p, R8y (linear)."""
    return to_linear(rgb8) @ SPECTRAL.T


def false_color(resp: np.ndarray) -> np.ndarray:
    """(..., 4) photoreceptor responses to (..., 3) bytes for the display: R1-6 warm, R8y green, R8p blue, R7 violet."""
    r16, r7, r8p, r8y = (resp[..., i] for i in range(4))
    out = np.stack([0.55 * r16 + 0.55 * r7, 0.15 * r16 + 1.0 * r8y, 0.65 * r8p + 1.4 * r7 + 0.1 * r16], axis=-1)
    out = np.clip(out, 0.0, 1.0) ** (1 / 2.2)
    return (out * 255).astype(np.uint8)


def fly_view_color(rgb8: np.ndarray, gain: float = 1.0) -> np.ndarray:
    """Display colors as the fly's photoreceptors would carry them, in false color: red goes dim, blue and green stay bright. `gain` is the light
    adaptation (photoreceptors adapt to the mean light; see adaptation_gain)."""
    return false_color(photoreceptors(rgb8) * gain)


def adaptation_gain(rgb8: np.ndarray, previous: float | None = None, target: float = 0.75) -> float:
    """A slowly following gain that brings the bright parts of the scene (95th percentile of the photoreceptor responses) to `target`: light adaptation,
    so a dim room is not drawn black. Never above 12x, so a dark scene stays dark."""
    resp = photoreceptors(rgb8)
    bright = float(np.percentile(resp[..., [0, 2, 3]].max(axis=-1), 95))
    g = float(np.clip(target / max(bright, 1e-3), 0.5, 12.0))
    return g if previous is None else previous + (g - previous) * 0.15


# --- the lattice -----------------------------------------------------------------------------------------------------------------------
class Lattice:
    """Both eyes' ommatidia: unit view directions in the fly's frame (x ahead, y up, z to its right) and where each is drawn in the panorama."""

    def __init__(self, spacing_deg: float = SPACING_DEG):
        rows = []
        el0, el1 = EL_DEG
        dy = spacing_deg * math.sqrt(3) / 2
        eyes = []
        a0, a1 = EYE_AZ_DEG
        j, el = 0, el0
        while el <= el1 + 1e-6:
            az = a0 + (spacing_deg / 2 if j % 2 else 0.0)
            while az <= a1 + 1e-6:
                eyes.append(("L", az, el))
                az += spacing_deg
            el += dy
            j += 1
        eyes += [("R", -az, el) for _, az, el in list(eyes)]                  # the right eye is the left one mirrored
        self.eye = np.array([e for e, _, _ in eyes])
        self.az = np.array([a for _, a, _ in eyes])
        self.el = np.array([e for _, _, e in eyes])
        self.spacing = spacing_deg
        a, e = np.radians(self.az), np.radians(self.el)
        self.dirs = np.stack([np.cos(e) * np.cos(a), np.sin(e), np.cos(e) * np.sin(a)], axis=1)
        self.n = len(self.az)
        self._views: dict = {}

    @property
    def per_eye(self) -> int:
        return int(np.count_nonzero(self.eye == "L"))

    def view_pixels(self, size: int, fov_deg: float = VIEW_FOV_DEG, yaws=VIEW_YAWS_DEG):
        """For each ommatidium: which of the rendered views sees it (-1: none), and the pixel in that view. Cached per size."""
        key = (size, fov_deg, tuple(yaws))
        if key in self._views:
            return self._views[key]
        which = np.full(self.n, -1, np.int32)
        px = np.zeros(self.n, np.int32)
        py = np.zeros(self.n, np.int32)
        best = np.full(self.n, 1e9)
        f = 1.0 / math.tan(math.radians(fov_deg) / 2)
        for k, yaw in enumerate(yaws):
            c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
            # rotate world directions into the view's frame: its forward is (c, 0, s), its right (-s, 0, c)
            fwd = self.dirs[:, 0] * c + self.dirs[:, 2] * s
            right = -self.dirs[:, 0] * s + self.dirs[:, 2] * c
            up = self.dirs[:, 1]
            ok = fwd > 1e-3
            u = np.where(ok, right / np.maximum(fwd, 1e-3) * f, 9.0)
            v = np.where(ok, up / np.maximum(fwd, 1e-3) * f, 9.0)
            inside = ok & (np.abs(u) <= 1.0) & (np.abs(v) <= 1.0)
            off = np.maximum(np.abs(u), np.abs(v))                        # prefer the view it is most central in
            take = inside & (off < best)
            which[take], best[take] = k, off[take]
            px[take] = np.clip(((u[take] + 1) / 2 * size).astype(np.int32), 0, size - 1)
            py[take] = np.clip(((1 - (v[take] + 1) / 2) * size).astype(np.int32), 0, size - 1)
        self._views[key] = (which, px, py)
        return self._views[key]

    def sample(self, views: list[np.ndarray], size: int) -> np.ndarray:
        """(n, 3) display colors (bytes) each ommatidium sees: each view blurred by the acceptance angle (about the spacing), then read at its pixel.
        Ommatidia no view reaches are black."""
        which, px, py = self.view_pixels(size)
        blur_px = max(1, int(round(size * self.spacing / VIEW_FOV_DEG)))
        out = np.zeros((self.n, 3), np.uint8)
        for k, img in enumerate(views):
            m = which == k
            if not np.any(m):
                continue
            small = _box_blur(img, blur_px)
            out[m] = small[py[m], px[m]]
        return out


def _box_blur(img: np.ndarray, r: int) -> np.ndarray:
    """A box blur of an (h, w, 3) byte image by r pixels each way (an acceptance angle), by integral image."""
    if r <= 1:
        return img
    a = img.astype(np.float32)
    c = np.cumsum(np.cumsum(np.pad(a, ((1, 0), (1, 0), (0, 0))), axis=0), axis=1)
    h, w = a.shape[:2]
    y0, y1 = np.clip(np.arange(h) - r, 0, h), np.clip(np.arange(h) + r + 1, 0, h)
    x0, x1 = np.clip(np.arange(w) - r, 0, w), np.clip(np.arange(w) + r + 1, 0, w)
    s = c[y1][:, x1] - c[y0][:, x1] - c[y1][:, x0] + c[y0][:, x0]
    area = ((y1 - y0)[:, None] * (x1 - x0)[None, :])[..., None]
    return (s / area).astype(np.uint8)


# --- drawing the eyes ------------------------------------------------------------------------------------------------------------------------
class Panorama:
    """The two eyes drawn side by side as hexagons: azimuth across (left eye on the left), elevation up. Precomputes the hexagon pixel map per size."""

    def __init__(self, lattice: Lattice, size: tuple[int, int]):
        self.lattice, self.size = lattice, size
        w, h = size
        la = lattice
        gap = max(6, w // 60)
        ew = (w - gap) / 2
        az_span = EYE_AZ_DEG[1] - EYE_AZ_DEG[0] + la.spacing
        el_span = EL_DEG[1] - EL_DEG[0] + 2 * la.spacing
        s = min(ew / az_span, h / el_span)                  # one scale for both axes, so the hexagons stay regular; the rest is black margin
        self.cx = np.zeros(la.n)
        self.cy = np.zeros(la.n)
        for eye, off, (a0, a1) in (("L", 0.0, EYE_AZ_DEG), ("R", ew + gap, (-EYE_AZ_DEG[1], -EYE_AZ_DEG[0]))):
            m = la.eye == eye
            x0 = off + (ew - az_span * s) / 2
            self.cx[m] = x0 + (la.az[m] - a0 + la.spacing / 2) * s
            self.cy[m] = h / 2 - la.el[m] * s
        self.radius = la.spacing * s / 2 * 1.08
        self.idx = _hex_index(size, self.cx, self.cy, self.radius)

    @classmethod
    def flat(cls, size: tuple[int, int], spacing_px: float) -> "Panorama":
        """A flat hexagonal lattice over a picture (the 2D game: a crop of the arena), spacing_px apart."""
        self = cls.__new__(cls)
        w, h = size
        self.lattice, self.size = None, size
        dy = spacing_px * math.sqrt(3) / 2
        xs, ys = [], []
        j, y = 0, spacing_px / 2
        while y < h:
            x = spacing_px / 2 + (spacing_px / 2 if j % 2 else 0.0)
            while x < w:
                xs.append(x)
                ys.append(y)
                x += spacing_px
            y += dy
            j += 1
        self.cx, self.cy = np.array(xs), np.array(ys)
        self.radius = spacing_px / 2 * 1.08
        self.idx = _hex_index(size, self.cx, self.cy, self.radius)
        return self

    @property
    def n(self) -> int:
        return len(self.cx)

    def sample(self, img: np.ndarray, blur_px: int) -> np.ndarray:
        """(n, 3) display colors at each ommatidium of an (h, w, 3) image of this panorama's size, blurred by the acceptance angle."""
        small = _box_blur(img, blur_px)
        return small[np.clip(self.cy.astype(int), 0, img.shape[0] - 1), np.clip(self.cx.astype(int), 0, img.shape[1] - 1)]

    def image(self, colors: np.ndarray) -> pygame.Surface:
        """(n, 3) bytes to a surface of self.size: each hexagon in its color, black elsewhere."""
        w, h = self.size
        lut = np.vstack([colors, np.zeros((1, 3), np.uint8)])
        img = lut[np.where(self.idx >= 0, self.idx, len(colors))]
        return pygame.image.frombuffer(np.ascontiguousarray(img).tobytes(), (w, h), "RGB").copy()


def _hex_index(size: tuple[int, int], cx: np.ndarray, cy: np.ndarray, r: float) -> np.ndarray:
    """Per pixel: the index of the ommatidium whose pointy-top hexagon (circumradius r) covers it, else -1."""
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    idx = np.full((h, w), -1, np.int32)
    best = np.full((h, w), 1e9, np.float32)
    for i in range(len(cx)):
        x0, x1 = int(max(0, cx[i] - r - 1)), int(min(w, cx[i] + r + 2))
        y0, y1 = int(max(0, cy[i] - r - 1)), int(min(h, cy[i] + r + 2))
        if x0 >= x1 or y0 >= y1:
            continue
        dx, dy = xx[y0:y1, x0:x1] - cx[i], yy[y0:y1, x0:x1] - cy[i]
        adx, ady = np.abs(dx), np.abs(dy)
        inside = (ady <= r) & (adx * math.sqrt(3) / 2 + ady * 0.5 <= r * math.sqrt(3) / 2 * 0.96)
        d = dx * dx + dy * dy
        sub, bsub = idx[y0:y1, x0:x1], best[y0:y1, x0:x1]
        take = inside & (d < bsub)
        sub[take], bsub[take] = i, d[take]
    return idx


# --- the brain's own visual activity, shown beside the view ---------------------------------------------------------------------------------------
VISUAL_GROUPS = (("R1-R6", ("R1-R6",), "photoreceptors, the broad luminance cells"), ("R7 / R8", ("R7", "R8"), "color photoreceptors"),
                 ("L1 / L2", ("L1", "L2"), "lamina: the ON and OFF motion inputs"), ("T4 / T5", ("T4", "T5"), "direction-selective motion neurons"),
                 ("LC4", ("LC4",), "looming, to the giant fiber"), ("LPLC2", ("LPLC2",), "looming, to the giant fiber"),
                 ("LC10", ("LC10",), "target tracking"), ("LC11", ("LC11",), "small-object detectors"))


class VisualActivity:
    """The mean firing of the brain's visual cell types, from the live sim (rows found once per brain, by type prefix). This is what the brain really has:
    the picture next to it does not drive it."""

    def __init__(self):
        self._rows: dict[int, dict[str, np.ndarray]] = {}
        self._calm: dict[int, dict[str, float]] = {}

    def rows_of(self, brain) -> dict[str, np.ndarray]:
        key = id(brain)
        if key not in self._rows:
            types = brain.types.astype(str)
            out = {}
            for label, prefixes, _ in VISUAL_GROUPS:
                m = np.zeros(len(types), bool)
                for p in prefixes:
                    m |= np.char.startswith(types, p)
                out[label] = np.flatnonzero(m)
            self._rows[key] = out
            self._calm[key] = {}
        return self._rows[key]

    def read(self, brain, rates: np.ndarray | None = None) -> list[tuple[str, str, int, float, float]]:
        """[(label, what, neurons, Hz now, Hz calm)]: calm is a slow average of the group's own rate, so a bar shows change."""
        rows = self.rows_of(brain)
        if rates is None:
            rates = brain.sim.activity.rates()
        calm = self._calm[id(brain)]
        out = []
        for label, _, what in VISUAL_GROUPS:
            r = rows[label]
            hz = float(rates[r].mean() / 0.005) if len(r) else 0.0
            calm[label] = hz if label not in calm else calm[label] + (hz - calm[label]) * 0.01
            out.append((label, what, int(len(r)), hz, calm[label]))
        return out


# --- the controller the games hold -----------------------------------------------------------------------------------------------------------
class FlyEyeView:
    """State for one game: on or off, the lattice and panorama caches, the activity reader. `three_d` games call frame_3d(); the 2D game, frame_2d()."""

    VIEW_PX = 192                                 # each of the three renders, square

    def __init__(self):
        self.on = False
        self.lattice = Lattice()
        self.activity = VisualActivity()
        self._pano: dict[tuple[int, int], Panorama] = {}
        self._flat: dict[tuple[tuple[int, int], float], Panorama] = {}
        self.last: pygame.Surface | None = None
        self.last_t = -1.0
        self.gain: float | None = None
        self.min_dt = 1.0 / 20.0                  # redraws 20 times a second (the lattice is a low-resolution eye; the brain view does the same)

    def toggle(self) -> bool:
        self.on = not self.on
        self.last, self.last_t = None, -1.0
        return self.on

    def panorama(self, size: tuple[int, int]) -> Panorama:
        if size not in self._pano:
            self._pano[size] = Panorama(self.lattice, size)
        return self._pano[size]

    def frame_3d(self, game, app, now: float, size: tuple[int, int]) -> pygame.Surface:
        """The fly's eyes on the scene, a surface of `size`: three renders from its head, sampled through the lattice, in false color."""
        import time

        wall = time.perf_counter()
        if self.last is not None and self.last.get_size() == size and wall - self.last_t < self.min_dt:
            return self.last
        from kickthefly.game import kick_the_fly as k2

        slot = game.flies[game.focus]
        head, thx = slot.fly.p[k2.HEAD], slot.fly.p[k2.THX]
        fwd = np.array([head[0] - thx[0], 0.0, head[2] - thx[2]])
        n = float(np.linalg.norm(fwd))
        fwd = fwd / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
        eye = np.array(head, float) + fwd * 0.04 + np.array([0.0, 0.03, 0.0])
        views = []
        for yaw in VIEW_YAWS_DEG:
            c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
            f = np.array([fwd[0] * c - fwd[2] * s, 0.0, fwd[0] * s + fwd[2] * c])    # the fly's own frame: yaw > 0 turns to its right
            views.append(app.render_view(game, now, eye, f, VIEW_FOV_DEG, self.VIEW_PX))
        samples = self.lattice.sample(views, self.VIEW_PX)
        self.gain = adaptation_gain(samples, self.gain)
        colors = fly_view_color(samples, self.gain)
        self.last = self.panorama(size).image(colors)
        self.last_t = wall
        return self.last

    def frame_2d(self, arena: pygame.Surface, center: tuple[float, float], facing: int, size: tuple[int, int] = (520, 260)) -> pygame.Surface:
        """The 2D game's version: the arena ahead of the fly's head (a crop `size` wide, on the side it faces), sampled through a flat hexagonal lattice, in false color."""
        w, h = size
        x0 = int(center[0] - (30 if facing > 0 else w - 30))
        y0 = int(center[1] - h * 0.55)
        x0 = max(0, min(arena.get_width() - w, x0))
        y0 = max(0, min(arena.get_height() - h, y0))
        crop = arena.subsurface(pygame.Rect(x0, y0, w, h)).copy()
        img = np.transpose(pygame.surfarray.array3d(crop), (1, 0, 2))
        key = (size, 14.0)
        if key not in self._flat:
            self._flat[key] = Panorama.flat(size, 14.0)
        pano = self._flat[key]
        samples = pano.sample(img, 5)
        self.gain = adaptation_gain(samples, self.gain)
        out = pano.image(fly_view_color(samples, self.gain))
        return pygame.transform.flip(out, True, False) if facing < 0 else out
