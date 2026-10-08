"""3.0 day 3, Rain and storms: the rules (game/weather.py) and the 3D game's use of them in the open field and the orchard.
The game tests run on the synthetic pack: they check the wiring (which neurons get poked, when, how often), not any biology."""
from __future__ import annotations

import gc
import threading

import numpy as np
import pytest

from kickthefly.game import weather as wx


# --- the rules ---------------------------------------------------------------------------------------------------------------
def _run(w, seconds, **kw):
    hits = []
    for _ in range(int(seconds * 60)):
        w.update(1 / 60, **kw)
        hits += w.hits(1 / 60)
    return hits


def test_everything_is_off_by_default_and_off_changes_nothing():
    w = wx.Weather(1)
    assert _run(w, 20) == [] and w.lightning() == 0 and not w.gusts and w.wind(3.0, 180.0) == (3.0, 180.0)
    assert w.darkness() == 0.0 and w.screen_flash() == 0.0 and w.thunder_due() == 0


def test_the_same_seed_gives_the_same_weather():
    a, b = wx.Weather(5), wx.Weather(5)
    assert _run(a, 30, rain=0.6, gust_hz=0.3, storm=True) == _run(b, 30, rain=0.6, gust_hz=0.3, storm=True)
    assert a.wind(2.0, 90.0) == b.wind(2.0, 90.0)


def test_the_hit_rate_scales_with_the_rain_and_parts_follow_the_table():
    n = {r: len(_run(wx.Weather(2), 300, rain=r)) for r in (0.25, 0.5, 1.0)}
    for r, k in n.items():
        assert k == pytest.approx(wx.HITS_PER_S * r * 300, rel=0.08)
    hits = _run(wx.Weather(3), 600, rain=1.0)
    parts = {p: sum(1 for h in hits if h[0] == p) / len(hits) for p, _ in wx.PART_WEIGHTS}
    for p, share in wx.PART_WEIGHTS:
        assert parts[p] == pytest.approx(share, abs=0.03)
    assert all(h[1] in ("L", "R") for h in hits if h[0] in ("wing", "legs")) and all(h[1] is None for h in hits if h[0] in ("body", "head"))
    assert all(0 < h[2] <= wx.HIT_STRENGTH[1] for h in hits)


def test_a_fly_that_is_sheltered_is_not_hit():
    w = wx.Weather(1)
    w.update(1 / 60, rain=1.0)
    assert all(not w.hits(1 / 60, exposed=False) for _ in range(300))


def test_a_gust_is_a_bump_in_the_wind_that_comes_and_goes():
    w = wx.Weather(4)
    seen = []
    for _ in range(int(60 * 60)):
        w.update(1 / 60, gust_hz=0.4)
        seen.append(w.wind(3.0, 180.0)[0])
    assert min(seen) == pytest.approx(3.0) and max(seen) > 3.0 + wx.GUST_AMP[0] * 0.5 and max(seen) <= 3.0 + 3 * wx.GUST_AMP[1]
    w2 = wx.Weather(4)
    for _ in range(600):
        w2.update(1 / 60)
    assert w2.wind(3.0, 180.0) == (3.0, 180.0), "no gusts, no change"


def test_the_storm_preset_is_a_floor_not_a_cap():
    assert wx.effective(0.0, 0.0, True) == (wx.STORM_RAIN, wx.STORM_GUST_HZ)
    assert wx.effective(0.95, 0.4, True) == (0.95, 0.4)
    assert wx.effective(0.0, 0.0, False) == (0.0, 0.0)
    w = wx.Weather(1)
    w.update(1 / 60, storm=True)
    assert w.wind(3.0, 180.0)[0] == pytest.approx(3.0 + wx.STORM_EXTRA_WIND)


def test_lightning_flashes_in_a_storm_only_and_thunder_follows():
    w = wx.Weather(6)
    flashes, claps, t_flash = 0, 0, None
    peak = 0.0
    for i in range(int(120 * 60)):
        w.update(1 / 60, storm=True)
        peak = max(peak, w.lightning())
        if w.flash_at is not None and t_flash != w.flash_at:
            t_flash = w.flash_at
            flashes += 1
        claps += w.thunder_due()
    assert flashes >= 6 and peak == 1.0 and claps >= flashes - 2
    dry = wx.Weather(6)
    assert all((dry.update(1 / 60, rain=1.0), dry.lightning())[1] == 0 for _ in range(int(120 * 60)))


def test_reduced_flashing_turns_the_flash_into_one_slow_swell():
    w = wx.Weather(7)
    seq = []
    for _ in range(int(40 * 60)):
        w.update(1 / 60, storm=True)
        seq.append(w.screen_flash(reduced=True))
    assert max(seq) <= wx.REDUCED_FLASH_PEAK + 1e-9 and max(seq) > 0.05
    steps = np.abs(np.diff(seq))
    assert steps.max() < 0.01, "no frame-to-frame jump: nothing strobes"
    normal = []
    w2 = wx.Weather(7)
    for _ in range(int(40 * 60)):
        w2.update(1 / 60, storm=True)
        normal.append(w2.screen_flash())
    assert np.abs(np.diff(normal)).max() > 0.3, "the normal flash does jump"


def test_wings_fill_with_rain_and_dry_out_of_it():
    level, t = 0.0, 0.0
    while level < 1.0 and t < 100:
        level, t = wx.soak(level, 1 / 60, 1.0), t + 1 / 60
    assert t == pytest.approx(wx.WET_FILL_S, rel=0.02)
    t = 0.0
    while level > 0.0 and t < 100:
        level, t = wx.soak(level, 1 / 60, 0.0), t + 1 / 60
    assert t == pytest.approx(wx.WET_DRY_S, rel=0.02)
    assert wx.soak(0.0, 1.0, 1.0, wing_hits=4) > wx.soak(0.0, 1.0, 1.0, wing_hits=0)


def test_the_lab_parameters_exist_default_to_off_and_are_game_rules():
    from kickthefly.lab import lab

    for name in ("weather.rain", "weather.gust_hz", "weather.storm"):
        row = lab.BY_NAME[name]
        assert row[2] == "rule" and row[3] == 0.0, "off by default: every existing outdoor result is unchanged"
        assert "game rule" in row[8].lower()


# --- the 3D game --------------------------------------------------------------------------------------------------------------
@pytest.fixture
def g3(synthetic_pack):
    import test_extras3 as t3

    g = t3.make_game(True)
    yield g
    while t3._GAMES:
        x = t3._GAMES.pop()
        x.view_stop = True
        for slot in getattr(x, "flies", []):
            slot.brain.stop()
    for th in threading.enumerate():
        if th.name == "brain-view":
            th.join(timeout=2.0)
    gc.collect()


def _arena(g, name):
    g.set_setting("brain.arena", name, save=False)
    g.on_arena_changed(g.clock.now) if hasattr(g, "on_arena_changed") else None


def _record(slot):
    pokes = []
    real = slot.brain.poke
    slot.brain.poke = lambda region, side, strength, recruit=None: (pokes.append((region, side, strength)), real(region, side, strength, recruit))[1]
    return pokes


def _frames(g, seconds):
    t0 = g.clock.now
    n = int(seconds * 60)
    for i in range(n):
        g.frame += 1
        g._environment(t0 + (i + 1) / 60)
    g.clock.now = t0 + seconds


@pytest.mark.parametrize("arena", ["field", "orchard"])
def test_rain_pokes_the_touch_groups_by_part_and_the_humidity_neurons(g3, arena):
    _arena(g3, arena)
    slot = g3.flies[0]
    pokes = _record(slot)
    g3.lab_params["weather.rain"] = 1.0
    _frames(g3, 20)
    regions = {r for r, _, _ in pokes}
    assert {"wing", "body", "head", "legs"} <= regions, "drops land on every part"
    assert "humid" in regions, "the air is wet: the humidity neurons"
    assert slot.fly.wet > 0 and not slot.fly.flying, "soaked wings can't fly (the pool's rule)"
    n_hits = sum(1 for r, _, _ in pokes if r in ("wing", "body", "head", "legs"))
    assert n_hits == pytest.approx(wx.HITS_PER_S * 20, rel=0.25)


@pytest.mark.parametrize("arena", ["field", "orchard"])
def test_no_weather_means_no_rain_pokes_and_no_new_wetness(g3, arena):
    _arena(g3, arena)
    slot = g3.flies[0]
    pokes = _record(slot)
    _frames(g3, 10)
    assert not [p for p in pokes if p[0] in ("head", "legs", "wing", "humid")]
    assert slot.fly.wet <= 0 and getattr(slot, "rain_wet", 0.0) == 0.0


def test_gusts_go_through_the_wind_transduction_in_the_orchard(g3):
    _arena(g3, "orchard")
    slot = g3.flies[0]
    pokes = _record(slot)
    _frames(g3, 10)
    assert not [p for p in pokes if p[0] == "wind"], "the orchard has no steady wind"
    g3.lab_params["weather.gust_hz"] = 0.5
    _frames(g3, 40)
    wind = [p for p in pokes if p[0] == "wind"]
    assert wind and {s for _, s, _ in wind} <= {"L", "R"}, "a gust drives the real JO-C/E neurons through outdoors.wind_drive"


def test_a_storm_flashes_the_photoreceptors_and_darkens_the_scene(g3):
    from kickthefly.game import kick3d

    _arena(g3, "field")
    slot = g3.flies[0]
    pokes = _record(slot)
    before = kick3d.scene_setup(g3)[0]["u_sky"]
    g3.lab_params["weather.storm"] = 1.0
    _frames(g3, 60)
    light = [p for p in pokes if p[0] == "light" and p[2] >= 0.5]
    assert light, "lightning drives both eyes' photoreceptors"
    g3.weather.flash_at = None
    after = kick3d.scene_setup(g3)[0]["u_sky"]
    assert sum(after) < sum(before), "a storm darkens the scene"
    g3.weather.flash_at = g3.weather.t - 0.01
    g3.cfg.set("access.reduced_flashing", True)
    soft = kick3d.scene_setup(g3)[0]["u_sky"]
    g3.cfg.set("access.reduced_flashing", False)
    hard = kick3d.scene_setup(g3)[0]["u_sky"]
    assert sum(hard) > sum(soft) >= sum(after) - 1e-9, "reduced flashing: the screen does not flash"


def test_the_hud_says_what_the_weather_is(g3):
    _arena(g3, "field")
    g3.lab_params["weather.storm"] = 1.0
    _frames(g3, 3)
    assert any("STORM" in s for s in g3.arena_status())


def test_rain_draws_without_error_and_stays_cheap(g3):
    class Rec:
        n = 0

        def add(self, *a, **k):
            self.n += 1

        def particle(self, *a, **k):
            self.n += 1

    _arena(g3, "field")
    g3.lab_params["weather.rain"] = 1.0
    _frames(g3, 1)
    rd = Rec()
    eye = np.array([0.0, 1.0, 0.0])
    g3._draw_rain(rd, eye, 1.0)
    assert 0 < rd.n <= 60, "at most 60 streaks"
