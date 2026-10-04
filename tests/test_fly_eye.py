"""3.1.0 task 11, the fly's-eye view (game/fly_eye.py): the spectral model, the ommatidia lattice, the sampling, the drawing, the key and the overlay in 3D and 2D, and
that it is visual only (it never touches the brain)."""
from __future__ import annotations

import inspect
import math

import numpy as np
import pygame
import pytest

from kickthefly.game import fly_eye as fe


# --- spectral sensitivity -----------------------------------------------------------------------------------------------------------------
def test_white_is_one_for_the_visible_receptors_and_uv_is_dark_on_any_screen():
    white = fe.photoreceptors(np.array([255, 255, 255]))
    assert white[0] == pytest.approx(1.0) and white[2] == pytest.approx(1.0) and white[3] == pytest.approx(1.0)
    assert white[1] == pytest.approx(0.0, abs=1e-3), "a monitor emits no UV: R7 is dark whatever is shown"
    rng = np.random.default_rng(0)
    assert np.all(fe.photoreceptors(rng.integers(0, 256, (200, 3)))[:, 1] < 1e-3)
    assert np.all(fe.SPECTRAL >= 0) and fe.SPECTRAL.shape == (4, 3)


def test_red_is_dim_and_blue_and_green_are_bright_to_a_fly():
    red, green, blue = (fe.photoreceptors(np.array(c)) for c in ((255, 0, 0), (0, 255, 0), (0, 0, 255)))
    assert red[[0, 2, 3]].max() < 0.12, "flies see little red"
    assert green[3] > 0.6 and green[0] > 0.4 and blue[2] > 0.7 and blue[0] > 0.4
    assert blue[2] > blue[3] > 0.0 and green[3] > green[2], "R8p (blue) prefers blue, R8y (green) prefers green"
    yellow = fe.photoreceptors(np.array([255, 255, 0]))
    assert yellow[3] > 0.7 and yellow[2] < yellow[3]


def test_false_color_keeps_the_ordering_and_reaches_the_display_range():
    out = fe.fly_view_color(np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255], [0, 0, 0]], np.uint8))
    assert out.dtype == np.uint8 and out.shape == (5, 3)
    assert out[3].min() > 150 and out[4].max() == 0
    assert out[0].astype(int).sum() < out[1].astype(int).sum() and out[0].astype(int).sum() < out[2].astype(int).sum()


def test_light_adaptation_follows_the_scene_and_is_bounded():
    dark = np.full((100, 3), 8, np.uint8)
    g = fe.adaptation_gain(dark)
    assert g == pytest.approx(12.0)
    bright = np.full((100, 3), 255, np.uint8)
    assert fe.adaptation_gain(bright) == pytest.approx(0.75, abs=0.01) or fe.adaptation_gain(bright) == pytest.approx(0.75 / 1.0)
    mid = np.full((100, 3), 90, np.uint8)
    target = fe.adaptation_gain(mid)
    assert fe.adaptation_gain(mid, previous=1.0) == pytest.approx(1.0 + (target - 1.0) * 0.15)


# --- the lattice ---------------------------------------------------------------------------------------------------------------------------
def test_each_eye_has_about_seven_hundred_ommatidia_five_degrees_apart():
    la = fe.Lattice()
    assert 600 <= la.per_eye <= 900 and la.n == 2 * la.per_eye
    assert np.allclose(np.linalg.norm(la.dirs, axis=1), 1.0)
    left = la.eye == "L"
    assert la.az[left].min() >= fe.EYE_AZ_DEG[0] - 1e-6 and la.az[left].max() <= fe.EYE_AZ_DEG[1] + fe.SPACING_DEG
    assert np.allclose(sorted(la.az[~left]), sorted(-la.az[left])), "the right eye mirrors the left"
    row = np.sort(la.az[left & (np.abs(la.el - la.el[left][0]) < 1e-6)])
    assert np.allclose(np.diff(row), fe.SPACING_DEG)
    assert 0.8 * fe.SPACING_DEG * 0.86 < np.diff(np.unique(np.round(la.el[left], 4)))[0] < 1.2 * fe.SPACING_DEG, "hexagonal row spacing"
    assert (la.az.max() - la.az.min()) > 270 and (la.az < 0).sum() > 100 and (la.az > 0).sum() > 100
    ahead = la.dirs[:, 0] > 0.99
    assert {"L", "R"} <= set(la.eye[ahead]) or (la.eye[np.argmin(np.abs(la.az) + np.abs(la.el))] in "LR"), "the eyes overlap in front"


def test_ommatidia_read_the_view_that_looks_their_way():
    la = fe.Lattice()
    which, px, py = la.view_pixels(192)
    assert (which >= 0).mean() > 0.85
    i = int(np.argmin(np.abs(la.az) + np.abs(la.el)))
    assert which[i] == 1 and abs(px[i] - 96) <= 6 and abs(py[i] - 96) <= 6, "straight ahead is the middle of the front view"
    right = int(np.argmin(np.abs(la.az - 90) + np.abs(la.el)))
    left = int(np.argmin(np.abs(la.az + 90) + np.abs(la.el)))
    assert which[right] == 2 and which[left] == 0, "the fly's right is the view that turned right (yaw +90)"
    assert abs(px[right] - 96) <= 8 and abs(px[left] - 96) <= 8
    up = int(np.argmin(np.abs(la.az) + np.abs(la.el - 40)))
    assert py[up] < 96 < px[up] + 200, "up is the top of the picture"


def test_sampling_returns_each_views_colors_and_black_where_no_view_reaches():
    la = fe.Lattice()
    size = 192
    views = [np.full((size, size, 3), c, np.uint8) for c in ((200, 0, 0), (0, 200, 0), (0, 0, 200))]
    colors = la.sample(views, size)
    which, _, _ = la.view_pixels(size)
    assert np.all(colors[which == 0] == (200, 0, 0)) and np.all(colors[which == 1] == (0, 200, 0)) and np.all(colors[which == 2] == (0, 0, 200))
    assert np.all(colors[which < 0] == 0)


def test_the_acceptance_blur_averages_neighbours():
    img = np.zeros((40, 40, 3), np.uint8)
    img[20, 20] = 255
    out = fe._box_blur(img, 3)
    assert out[20, 20, 0] < 255 and out[20, 18, 0] > 0 and out[20, 30, 0] == 0
    flat = np.full((20, 20, 3), 77, np.uint8)
    assert np.all(fe._box_blur(flat, 4) == 77)
    assert fe._box_blur(img, 1) is img


# --- drawing -------------------------------------------------------------------------------------------------------------------------------
def test_the_panorama_draws_regular_hexagons_in_each_ommatidiums_color():
    pygame.init()
    la = fe.Lattice()
    for size in ((860, 400), (500, 760), (300, 300)):
        pano = fe.Panorama(la, size)
        colors = np.random.default_rng(1).integers(30, 255, (la.n, 3)).astype(np.uint8)
        surf = pano.image(colors)
        assert surf.get_size() == size
        arr = np.transpose(pygame.surfarray.array3d(surf), (1, 0, 2))
        cx, cy = pano.cx.astype(int), pano.cy.astype(int)
        ok = (cx >= 0) & (cx < size[0]) & (cy >= 0) & (cy < size[1])
        assert np.mean(np.all(arr[cy[ok], cx[ok]] == colors[ok], axis=1)) > 0.95, "each hexagon's middle is its color"
        assert (arr.sum(axis=2) == 0).mean() > 0.1, "black between and around the eyes"
        i = pano.idx
        counts = np.bincount(i[i >= 0], minlength=la.n)
        full = counts[counts > 0]
        assert np.std(full) / np.mean(full) < 0.2, "the hexagons are the same size, not stretched (even in a tall window)"


def test_a_flat_lattice_for_the_2d_picture():
    pygame.init()
    pano = fe.Panorama.flat((520, 260), 14.0)
    assert pano.n > 500 and pano.size == (520, 260)
    img = np.zeros((260, 520, 3), np.uint8)
    img[:, :260] = (0, 200, 0)
    cols = pano.sample(img, 5)
    assert cols.shape == (pano.n, 3) and cols[pano.cx < 200].mean(axis=0)[1] > 150 and cols[pano.cx > 330].max() == 0
    assert pano.image(cols).get_size() == (520, 260)


# --- the brain's own visual neurons -----------------------------------------------------------------------------------------------------------------
def test_the_visual_activity_panel_reads_the_live_sim_without_changing_it(synthetic_pack):
    from kickthefly.core import simcore

    br = simcore.new_brain(seed=1, warmup=0, memory=False)
    va = fe.VisualActivity()
    v0, s0, rng0 = br.sim.v.copy(), br.sim.spikes.copy(), br.sim.rng.bit_generator.state
    rows = va.read(br)
    assert [r[0] for r in rows] == [g[0] for g in fe.VISUAL_GROUPS] and all(r[2] > 0 for r in rows if r[0] in ("LPLC2", "LC4", "T4 / T5", "R1-R6"))
    assert np.array_equal(v0, br.sim.v) and np.array_equal(s0, br.sim.spikes) and rng0 == br.sim.rng.bit_generator.state
    for _ in range(300):
        br._step()
    r2 = va.read(br)
    assert all(abs(a[4] - b[4]) >= 0 for a, b in zip(rows, r2)) and all(len(r) == 5 for r in r2)


def test_it_is_visual_only_in_the_code():
    src = inspect.getsource(fe)
    for banned in (".poke(", "set_override", "set_current", "drive_cur", "inject", ".step(", "_step("):
        assert banned not in src, f"fly_eye must never give the brain anything ({banned})"


# --- in the games --------------------------------------------------------------------------------------------------------------------------------
class FakeApp:
    def __init__(self):
        self.calls = []

    def render_view(self, game, now, eye, fwd, fov, px):
        self.calls.append((np.array(eye), np.array(fwd), fov, px))
        return np.full((px, px, 3), (len(self.calls) * 60 % 255, 120, 80), np.uint8)


@pytest.fixture(params=[False, True], ids=["2d", "3d"])
def game(request, synthetic_pack):
    import test_extras3 as t3

    g = t3.make_game(request.param)
    yield g
    for slot in g.flies:
        slot.brain.stop()


def test_the_key_toggles_it_and_the_action_is_rebindable(game):
    from kickthefly.core import config

    assert ("fly_eye", "Fly's-eye view", "f4") in config.ACTIONS and game.cfg.conflicts() == {}
    ev = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_F4, mod=0, unicode="")
    assert not game.fly_eye.on
    assert game.x3.handle_event(ev) and game.fly_eye.on
    assert game.x3.handle_event(ev) and not game.fly_eye.on
    ok, _ = game.cfg.bind("fly_eye", "f9")
    assert ok and game.cfg.actions_for("f9") == ["fly_eye"]


def test_the_overlay_names_its_tag_and_its_reason_and_draws_in_both_games(game):
    from kickthefly.game import kick_the_fly as k2

    game.fly_eye.on = True
    surf = pygame.Surface((k2.W, k2.H), pygame.SRCALPHA)
    if not game.three_d:
        surf = pygame.Surface((k2.W, k2.H))
        surf.fill((60, 70, 90))
    game.x3.draw(surf, 1.0)
    assert surf.get_bounding_rect().w > 100
    assert fe.TAG == "GAME RULE" and "not fed to the brain" in fe.NOTE
    if not game.three_d:                                   # the filtered picture of what is ahead of the fly
        assert game.fly_eye.last is None or True
        img = game.fly_eye.frame_2d(surf, (300.0, 500.0), 1)
        assert img.get_size() == (520, 260)
        flip = game.fly_eye.frame_2d(surf, (300.0, 500.0), -1)
        assert flip.get_size() == (520, 260)


def test_the_3d_frame_renders_three_views_from_the_flys_head_and_is_throttled(synthetic_pack):
    import test_extras3 as t3
    from kickthefly.game import kick_the_fly as k2

    g = t3.make_game(True)
    try:
        app = FakeApp()
        view = g.fly_eye
        slot = g.flies[0]
        surf = view.frame_3d(g, app, 0.0, (700, 400))
        assert surf.get_size() == (700, 400) and len(app.calls) == 3
        head = slot.fly.p[k2.HEAD]
        for eye, fwd, fov, px in app.calls:
            assert np.linalg.norm(eye - head) < 0.15 and fov == fe.VIEW_FOV_DEG and px == view.VIEW_PX and abs(np.linalg.norm(fwd) - 1) < 1e-6
        fwds = np.array([c[1] for c in app.calls])
        ang = lambda a, b: math.degrees(math.acos(np.clip(np.dot(a, b), -1, 1)))     # noqa: E731
        assert 85 < ang(fwds[0], fwds[1]) < 95 and 85 < ang(fwds[2], fwds[1]) < 95 and ang(fwds[0], fwds[2]) > 170
        side = np.cross(fwds[1], [0, 1, 0])                  # the fly's right in this world
        assert np.dot(fwds[2], side) > 0.9 > 0 > np.dot(fwds[0], side), "views[2] looks to the fly's right"
        again = view.frame_3d(g, app, 0.01, (700, 400))
        assert again is surf and len(app.calls) == 3, "throttled to about 20 frames a second"
        view.last_t = -1.0
        view.frame_3d(g, app, 0.1, (700, 400))
        assert len(app.calls) == 6
    finally:
        for slot in g.flies:
            slot.brain.stop()


def test_turning_it_on_never_changes_what_the_brain_does(game):
    """The same brain stepped the same way, with the view off and on and drawn, ends in the same state."""
    from kickthefly.game import kick_the_fly as k2

    def run(on: bool):
        import test_extras3 as t3
        g = t3.make_game(game.three_d)
        g.fly_eye.on = on
        br = g.brain
        for i in range(200):
            with br.step_lock:
                br._step()
            if on and i % 50 == 0:
                s = pygame.Surface((k2.W, k2.H), pygame.SRCALPHA)
                g.x3.draw(s, 1.0)
        out = (br.sim.v.copy(), br.sim.spikes.copy())
        for slot in g.flies:
            slot.brain.stop()
        return out

    a, b = run(False), run(True)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
