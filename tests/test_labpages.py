"""Every Lab screen draws without a real game behind it.

The Lab pages are the only place several of these features can be reached, and a typo in one of them is a crash in
the pause menu rather than a failing assertion anywhere else. This renders each of them onto an off-screen surface
with a stub host, which catches exactly that.
"""
import os
from types import SimpleNamespace

import pygame
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class StubHost:
    """Just enough of Game for the Lab pages to draw."""

    three_d = False

    def __init__(self):
        from kickthefly.core import config
        from kickthefly.lab import lab
        from kickthefly.sim.wiring import Wiring

        self.cfg = config.Config(None)
        self.cfg.set("brain.mode", "lab")
        self.lab_params = dict(lab.DEFAULTS)
        self.wiring = Wiring()
        self.wiring_busy = ""
        self.flies = []
        self.recording = None
        self.last_export = None
        self.inspect = None
        self.brain = None
        self.arena_i = 0
        self.surgery_modes = []
        self.type_ops = {}
        self.surgery_applied = 0
        self.clock = SimpleNamespace(scale=1.0, now=0.0)
        self.notes = []

    def lab_pages(self):
        from kickthefly.game import kick_the_fly as k

        return k.Game.lab_pages(self)

    def set_lab_param(self, name, value):
        self.lab_params[name] = value

    def set_wiring(self, w, note=True):
        self.wiring = w

    def _apply_surgery(self):
        self.surgery_applied += 1

    def note(self, text, source=None):
        self.notes.append(text)

    def menu_action(self, what):
        pass


@pytest.fixture(scope="module")
def menu():
    pygame.init()
    pygame.display.set_mode((1280, 760))
    from kickthefly.lab import lab
    from kickthefly.ui import menu as ui

    m = ui.Menu(StubHost())
    lab.install(m)
    m.fonts()
    m.mouse = (0, 0)
    yield m
    pygame.display.quit()


LAB_PAGES = ["lab", "lab_params", "lab_assumptions", "lab_asymmetry", "lab_benchmark", "lab_export", "lab_protocols",
             "lab_validation", "lab_assays", "lab_wiring", "lab_critical", "lab_activation", "lab_knockout", "lab_clamp", "lab_diff", "lab_laser", "lab_psych", "lab_classroom"]


@pytest.mark.parametrize("page", LAB_PAGES)
def test_lab_page_draws(menu, page):
    assert page in menu.pages, f"{page} is not installed in the Lab menu"
    surf = pygame.Surface((1280, 760))
    rect = pygame.Rect(40, 40, 1200, 680)
    menu.hits = []
    menu.pages[page](menu, surf, rect, (0, 0))


def test_every_hub_button_has_a_page(menu):
    for label, page, tip in menu.host.lab_pages():
        assert page in menu.pages, f"the Lab hub offers {label!r} but {page} is not a page"
        assert tip, f"{label} has no tooltip"


def test_wiring_tabs_all_draw(menu):
    from kickthefly.lab import labwiring

    surf = pygame.Surface((1280, 760))
    rect = pygame.Rect(40, 40, 1200, 680)
    st = labwiring._st(menu)
    for key, label, _fn in labwiring.TABS:
        st.wiring_tab = key
        menu.hits = []
        labwiring.page(menu, surf, rect, (0, 0))
