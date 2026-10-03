"""3.0 release review (Opus, final day): a regression test for every bug found in the day 5 review and the release polish.

Each test names the bug it guards against; the code it covers is cited in CHANGELOG.md under "3.0 release review"."""
from __future__ import annotations

import argparse
import math

import numpy as np
import pygame
import pytest

from test_day5_rigs import FakeBrain, FakeTransducer

from kickthefly.game import rigs


@pytest.fixture
def no_transducer(monkeypatch):
    monkeypatch.setattr(rigs.Transducer, "drive", lambda self, br, slip: None)
    monkeypatch.setattr(rigs.Transducer, "release", lambda self, br: None)


# --- rigs ---------------------------------------------------------------------------------------------------------------------------
def test_rig_assay_passes_the_sim_backend_through(monkeypatch, tmp_path):
    """--headless --rig-assay NAME --sim-backend X ignored X: the assay always ran on the default backend."""
    from kickthefly.lab import rigassay

    seen = {}

    def fake_run_assay(rig, seeds, workers=1, backend=None, progress=None, **kw):
        seen["backend"] = backend
        return dict(rig=rig, seeds=list(seeds), title="T", results={})

    monkeypatch.setattr(rigassay, "run_assay", fake_run_assay)
    monkeypatch.setattr(rigassay, "save", lambda res, folder: [])
    monkeypatch.setattr(rigassay, "summary", lambda res: "")
    args = argparse.Namespace(rig_assay="buridan", rig=None, seeds="1000-1001", workers=1, out=str(tmp_path), individuality=None, sim_backend="numba")
    assert rigassay.main(args) == 0
    assert seen["backend"] == "numba"
    args.sim_backend = "auto"
    rigassay.main(args)
    assert seen["backend"] is None


def test_a_long_tethered_scene_is_summarised_over_the_whole_run(no_transducer):
    """scene_run's tethered summary read a 0-99 s window, so a 120 s run (allowed up to 600 s) was summarised on its start only."""
    from kickthefly.lab import rigassay

    class LateBrain(FakeBrain):
        def __init__(self):
            super().__init__(walk_level=1.5)
            self.t = 0

        def _step(self):
            self.t += 1

        def hz(self, name):
            if name == "walk":
                return super().hz(name)
            late = self.t * self.dt > 100.0                    # the fly only responds after 100 s
            d = 10.0 if late else 0.0
            return {"turn_r": 4.0 + d, "turn_l": 4.0}[name]

    res = rigassay.scene_run("tethered", 0, "subtle", br=LateBrain(), seconds=120.0, omega=1.0)
    s = res["summary"]
    # the last right-rotation phase and the left one both lie past 100 s, so the late response must show in the summary
    assert s["response_left_hz"] > 1.0


def test_a_rig_protocol_refuses_params_and_nwb_it_would_have_ignored():
    """A rig protocol accepted `params` (and `nwb`, `assay_options`) and then ran with the defaults: a changed-parameter protocol reported the
    default result as its own."""
    from kickthefly.lab import protocol

    base = dict(name="r", rig=dict(name="buridan", seconds=20))
    protocol.check(dict(base))
    for extra in (dict(params={"noise_std": 0.1}), dict(nwb=True), dict(assay_options={"x": 1})):
        with pytest.raises(protocol.ProtocolError, match="stands alone"):
            protocol.check(dict(base, **extra))


def test_the_tethered_panorama_turns_by_the_slip_including_the_loop_gain():
    """The tethered scene drew the panorama turning by omega - yaw in closed loop, ignoring the loop gain (slip = omega - gain * yaw)."""
    from kickthefly.lab import labrigs

    a = np.zeros((10, len(rigs.COLS)))
    a[:, rigs.COLS.index("omega_ext")] = 1.0
    a[:, rigs.COLS.index("yaw_rate")] = 0.5
    a[:, rigs.COLS.index("slip")] = 1.0 - 2.0 * 0.5                 # gain 2: the panorama stands still relative to the fly
    assert labrigs.panorama_angle(a, 0.1) == pytest.approx(0.0)
    a[:, rigs.COLS.index("slip")] = 1.0                            # open loop
    assert labrigs.panorama_angle(a, 0.1) == pytest.approx(1.0)


# --- mini-papers --------------------------------------------------------------------------------------------------------------------
def _fake_result(pid="von_reyn_2014", reproduced=True):
    from kickthefly.lab import minipapers

    p = minipapers.PAPERS[pid]
    qs = [dict(test=q.test, prompt=q.prompt, expected=q.expected, verdict=dict(reproduced=reproduced, basis="b", measured={}, n=4), pairs=[], note="n")
          for q in p.questions]
    return dict(kind="minipaper", paper=pid, title=p.title, citation=p.citation, doi=p.doi, read=p.read, seeds=[0, 1, 2, 3], full=False, source="live",
                unavailable="", seconds=1.0, questions=qs, created="")


def test_the_hypothesis_is_fixed_when_the_run_starts(monkeypatch):
    """The comparison used whatever hypothesis was selected when it was drawn, so changing it after seeing the result made it 'match'."""
    from kickthefly.ui import minipaper_ui

    class Host:
        pass

    class M:
        host = Host()

    m = M()
    s = minipaper_ui.st(m)
    s.answers["von_reyn_2014"] = {"looming_escape": "down"}
    monkeypatch.setattr(minipaper_ui.BgJob, "start", lambda self: self)
    minipaper_ui.start(m, "von_reyn_2014")                         # commits "down"
    s.answers["von_reyn_2014"]["looming_escape"] = "up"            # changed after the run started
    from kickthefly.lab import minipapers

    rows = minipapers.compare(_fake_result(), minipaper_ui.committed(s, "von_reyn_2014"))
    assert rows[0]["your_hypothesis"] == dict(minipapers.PAPERS["von_reyn_2014"].questions[0].options)["down"]
    assert rows[0]["you_matched_the_model"] is False


def test_the_svg_plot_keeps_negative_values_inside():
    """svg_pairs with all-negative values (a performance index) keeps every dot inside its row (a check; the old scaling only misplaced the zero line)."""
    import re

    from kickthefly.lab import minipapers

    res = dict(questions=[dict(test="mb_conditioning", pairs=[dict(seed=0, drive=-0.4, control=-0.8), dict(seed=1, drive=-0.2, control=-0.6)])])
    svg = minipapers.svg_pairs(res, row_h=160)
    ys = [float(v) for v in re.findall(r'cy="([-0-9.]+)"', svg)]
    assert ys and all(10 <= y <= 10 + 160 for y in ys)


def test_the_quick_verdict_uses_validations_own_thresholds():
    """minipapers kept its own copies of validation's criteria (1.5, 0.01, and 0.5 / 0.25 written inline)."""
    from kickthefly.lab import minipapers, validation

    assert (minipapers.RATIO_MIN, minipapers.P_MAX, minipapers.PI_MIN, minipapers.CONTROL_PI_MAX) == \
           (validation.RATIO_MIN, validation.P_MAX, validation.PI_MIN, validation.CONTROL_PI_MAX)


def test_the_shiu_and_colomb_texts_say_what_their_full_texts_say():
    """Shiu et al. 2024 was listed as paywalled and MN9 as 'this game's choice'; its full text is open (PMC11446845) and uses MN9 itself. The rig's stripe
    deviation was described as Colomb et al.'s measure; theirs is the angle to the front stripe's centre (0-120 degrees)."""
    from kickthefly.lab import minipapers

    shiu = minipapers.PAPERS["shiu_2024"]
    assert "PMC11446845" in shiu.read and "paywall" not in shiu.read
    assert "game's choice" not in shiu.cannot_check and "MN9 is the paper's own readout" in shiu.cannot_check
    colomb = minipapers.PAPERS["colomb_2012"]
    assert "not the paper's metric" in colomb.cannot_check
    assert "not the same" in rigs.stripe_deviation.__doc__


# --- menu widgets -------------------------------------------------------------------------------------------------------------------
@pytest.fixture
def menu():
    pygame.init()
    pygame.display.set_mode((1280, 760))
    from kickthefly.core import config
    from kickthefly.ui import menu as ui

    class Host:
        cfg = config.Config(None)

    m = ui.Menu(Host())
    m.fonts()
    m.mouse = (0, 0)
    m.hits = []
    yield m
    pygame.display.quit()


def test_a_long_button_label_wraps_or_is_cut_inside_its_button(menu):
    """Menu.button drew its label centred whatever its width, so long labels ran past the button onto the next one."""
    surf = pygame.Surface((800, 200))
    drawn = []
    orig = menu.text
    menu.text = lambda s_, t, pos, color=None, font=None, anchor="topleft": drawn.append(orig(s_, t, pos, color or (255, 255, 255), font, anchor)) or drawn[-1]
    r = pygame.Rect(10, 10, 160, 60)                                  # two lines fit
    menu.button(surf, r, "Flip uncertain neurotransmitters", lambda: None, id="a")
    assert len(drawn) == 2 and all(r.contains(d) for d in drawn)
    drawn.clear()
    r2 = pygame.Rect(10, 100, 120, 30)                                # one line only: cut with an ellipsis, full text as the tooltip
    menu.button(surf, r2, "Silence looming detectors (LPLC2, LC4)", lambda: None, id="b")
    assert len(drawn) == 1 and r2.contains(drawn[0])
    hit = [d for _, _, d in menu.hits if d["id"] == "b"][0]
    assert hit["tip"] == "Silence looming detectors (LPLC2, LC4)"


def test_a_short_label_is_drawn_exactly_as_before(menu):
    surf = pygame.Surface((400, 100))
    lines, cut = menu.fit_lines(menu.f_bold, "Back", 128, 1)
    assert lines == ["Back"] and not cut
    menu.button(surf, (10, 10, 140, 42), "Back", lambda: None, id="c")
    assert [d for _, _, d in menu.hits if d["id"] == "c"][0]["tip"] is None


def test_segmented_labels_fit_and_the_toggle_is_translated(menu, monkeypatch):
    surf = pygame.Surface((800, 100))
    menu.segmented(surf, (10, 10, 200, 30), ["Quick: 4 flies, exploration seeds", "Full"], 0, lambda i: None, id="seg")
    tips = {d["id"]: d["tip"] for _, _, d in menu.hits}
    assert tips[("seg", 0)] == "Quick: 4 flies, exploration seeds" and tips[("seg", 1)] is None
    from kickthefly.ui import menu as ui

    monkeypatch.setattr(ui, "tr", lambda s, **k: {"On": "Ein", "Off": "Aus"}.get(s, s))
    drawn = []
    orig = menu.text
    menu.text = lambda s_, t, *a, **k: drawn.append(t) or orig(s_, t, *a, **k)
    menu.toggle(surf, (10, 50, 120, 30), True, lambda v: None, id="t")
    assert "Ein" in drawn


# --- the order-dependent test leak (test_playthrough_extras::test_neurodex_extra_passes) ---------------------------------------------
def test_a_neurodex_table_built_for_one_pack_is_never_returned_for_another(monkeypatch, tmp_path):
    """neurodex.table cached by brain name only: a game's background build for the real pack that finished after a test switched to the
    synthetic pack stored the real table under "adult", and the synthetic brain then indexed it (IndexError: index 2932 ... size 2346, the
    fast suite's order-dependent failure since 3.0 day 1). Reproduced here with a slow build thread."""
    import threading
    import time

    from kickthefly.core import neurodex as nd
    from kickthefly.sim import brainpack

    a, b = tmp_path / "real.npz", tmp_path / "synthetic.npz"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    started = threading.Event()

    def slow_build(path, brain="adult"):
        if path == a:
            started.set()
            time.sleep(0.3)                                  # the real pack takes seconds
        return ("table of", path.name)

    monkeypatch.setattr(nd, "table_from_pack", slow_build)
    nd.reset_cache()
    monkeypatch.setattr(brainpack, "find", lambda brain="adult": a)
    stale = threading.Thread(target=nd.table, args=("adult",))
    stale.start()
    started.wait(2)
    nd.reset_cache()                                         # the next test's fixture
    monkeypatch.setattr(brainpack, "find", lambda brain="adult": b)
    got = nd.table("adult")
    stale.join()
    assert got == ("table of", "synthetic.npz")
    assert nd.table("adult") == ("table of", "synthetic.npz")
    nd.reset_cache()


# --- release polish: pre-existing page bugs -----------------------------------------------------------------------------------------
def test_every_bundled_protocol_gets_a_summary_and_rig_ones_are_runnable():
    """Lab > Protocols showed only the first 12 files (the rig protocols could not be run from it) and a classroom protocol's summary was
    a KeyError ('stimuli')."""
    from kickthefly.lab import lab, protocol

    files = lab.protocol_files()
    assert len(files) > 12
    kinds = {}
    for f in files:
        desc, runnable = lab.protocol_summary(protocol.load(f))
        assert desc and "'" not in desc[:1]
        kinds[f.name] = runnable
    assert all(kinds[n] for n in kinds if n.startswith("rig_"))
    assert not any(kinds[n] for n in kinds if n.startswith("lecture_"))


def test_no_ui_string_uses_a_glyph_the_fallback_font_lacks():
    """The classroom page's buttons and two HUD labels used glyphs (▶ ↺ ⟵ ⟶ ★ ✓) that the Linux fallback font (FreeSans, and its bold) does
    not have: they rendered as empty boxes."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "kickthefly"
    bad = []
    for p in list(root.rglob("*.py")) + list((root / "data").rglob("*.json")) + list((root / "data").glob("*.yaml")):
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#") or "3.0 release review" in line or "fonts don't all have" in line:
                continue
            if any(c in line for c in "▶↺⟵⟶★✓"):
                bad.append(f"{p.name}:{i}")
    assert not bad, bad


def test_the_selftest_children_find_the_package_from_any_directory(monkeypatch):
    """Run from source outside the repo, the self-test's child processes could not import kickthefly: a false FAIL for the gl backend and
    a false OpenGL warning."""
    import os
    from pathlib import Path

    from kickthefly.core import selftest

    monkeypatch.delenv("PYTHONPATH", raising=False)
    env = selftest._child_env(SDL_AUDIODRIVER="dummy")
    root = Path(selftest.__file__).resolve().parents[2]
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(root) and (root / "kickthefly" / "__init__.py").exists()
    assert env["SDL_AUDIODRIVER"] == "dummy"


def test_classroom_tabs_carry_their_lecture_names(monkeypatch):
    """The lecture tabs drew their title, then an empty button over it: five blank tabs."""
    from test_labpages import StubHost

    from kickthefly.lab import labclassroom
    from kickthefly.ui import menu as ui

    pygame.init()
    pygame.display.set_mode((1280, 760))
    m = ui.Menu(StubHost())
    m.fonts()
    m.mouse, m.hits = (0, 0), []
    labclassroom.page(m, pygame.Surface((1280, 760)), pygame.Rect(150, 40, 980, 680), (0, 0))
    labels = {d["id"][1]: d for _, _, d in m.hits if isinstance(d.get("id"), tuple) and d["id"][0] == "class_tab"}
    assert set(labels) == {"looming", "tmaze", "moonwalker", "sugar", "gf_lesion"}
    pygame.display.quit()


def test_the_asymmetry_page_says_partners_and_names_the_giant_fiber_right():
    """The page labelled partner counts as synapses ("In-Syn"), called DNp01 a "braking / backward command" (it is the giant fiber), and
    called the 77,507 bilateral pairs "paired neurons"."""
    import inspect

    from kickthefly.lab import headless, lab

    src = inspect.getsource(lab.page_asymmetry)
    assert "Braking / backward" not in src and "Giant fiber: escape takeoff" in src
    assert "In-Syn" not in src and "In-partners" in src and "77,507 bilateral left/right pairs (155,014 neurons)" in src
    assert "In-Syn" not in inspect.getsource(headless.format_asymmetry_report)
    assert "10,272,125 synapses" not in inspect.getsource(lab.page_benchmark)
