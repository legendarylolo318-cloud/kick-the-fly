"""3.0 release review: no text overlap or clipping on the menu pages.

Every page is drawn through the real Menu at the game's own logical sizes (1280x760, and 900x1820, the narrowest a window gets: a narrow
window scales the HUD to at least 900 logical pixels wide with the brain panel hidden, 1280 with it), at normal and larger text, in English
and German, and every piece of text and every button the page draws is recorded. A page fails when a button's label has to be cut, text
leaves the panel or is cut sideways by its clip, two pieces of text overlap, loose text sits on a button, or buttons overlap. The same checks
found the bugs this release fixed (tools: the offscreen sweep described in CHANGELOG.md). Pages are drawn on a stub host and the synthetic
pack, so this is layout only."""
from __future__ import annotations

import pygame
import pytest

from test_labpages import StubHost

SIZES = ((1280, 760), (900, 1820))


@pytest.fixture
def menu(synthetic_pack):
    pygame.init()
    pygame.display.set_mode((1280, 760))
    from kickthefly.core import config, i18n
    from kickthefly.lab import lab
    from kickthefly.ui import arcade_ui, menu as ui, share_ui, whatsnew_ui

    from kickthefly.lab import livelab

    host = StubHost()
    host.cfg = config.Config(None)
    host.thermo_live, host.imaging_live = livelab.ThermoLive(), livelab.ImagingLive()
    host.cfg.set("brain.mode", "lab")
    m = ui.Menu(host)
    lab.install(m)
    whatsnew_ui.install(m)
    m.fonts()
    yield m
    i18n.set_language("en")
    pygame.display.quit()


class Recorder:
    def __init__(self, monkeypatch):
        from kickthefly.ui import menu as ui

        self.rec = []
        rec = self.rec
        t0, b0, s0 = ui.Menu.text, ui.Menu.button, ui.Menu.segmented

        def text(m, surf, s, pos, color=ui.TEXT, font=None, anchor="topleft"):
            r = t0(m, surf, s, pos, color, font, anchor)
            rec.append(("text", pygame.Rect(r), str(s), pygame.Rect(surf.get_clip()), getattr(m, "_in_button", None)))
            return r

        def button(m, surf, rect, label, click, **kw):
            rr = pygame.Rect(rect)
            rec.append(("button", rr, str(label), pygame.Rect(surf.get_clip()), not m.label_fits(label, rr.w, rr.h, kw.get("font") or m.f_bold)))
            m._in_button = rr
            try:
                return b0(m, surf, rect, label, click, **kw)
            finally:
                m._in_button = None

        def segmented(m, surf, rect, labels, current, select, **kw):
            rr = pygame.Rect(rect)
            n = len(labels)
            for i, lab in enumerate(labels):
                r = pygame.Rect(rr.x + i * (rr.w // n), rr.y, rr.w // n - 4, rr.h)
                rec.append(("button", r, str(lab), pygame.Rect(surf.get_clip()), not m.label_fits(lab, r.w, r.h, m.f_small)))
            m._in_button = rr
            try:
                return s0(m, surf, rect, labels, current, select, **kw)
            finally:
                m._in_button = None

        monkeypatch.setattr(ui.Menu, "text", text)
        monkeypatch.setattr(ui.Menu, "button", button)
        monkeypatch.setattr(ui.Menu, "segmented", segmented)

    def problems(self, panel: pygame.Rect) -> list[str]:
        out, texts, buttons = [], [], []
        for kind, r, s, clip, extra in self.rec:
            vis = r.clip(clip)
            if vis.w <= 2 or vis.h <= 2 or not s.strip():
                continue
            if kind == "button":
                buttons.append((vis, s))
                if extra:
                    out.append(f"label cut: {s!r}")
            else:
                texts.append((vis, s, extra))
                if r.left < panel.left - 1 or r.right > panel.right + 1:
                    out.append(f"text outside the panel: {s[:60]!r}")
                elif vis.w < r.w - 2 and vis.h >= r.h - 2:
                    out.append(f"text cut sideways: {s[:60]!r}")
                if vis.bottom > panel.bottom + 1:
                    out.append(f"text below the panel: {s[:60]!r}")
        for i, (a, sa, in_button) in enumerate(texts):
            for b, sb, _b in texts[i + 1:]:
                inter = a.clip(b)
                if inter.w > 3 and inter.h > 3 and inter.w * inter.h > 0.15 * min(a.w * a.h, b.w * b.h) and not (a == b and sa == sb):
                    out.append(f"text overlap: {sa[:40]!r} / {sb[:40]!r}")
            if in_button is None and sa != "a run is ready to replay":
                for bv, bl in buttons:
                    if not bl.strip():
                        continue
                    inter = a.clip(bv)
                    if inter.w > 3 and inter.h > 3:
                        out.append(f"text on a button: {sa[:40]!r} / {bl[:40]!r}")
        for i, (a, sa) in enumerate(buttons):
            for b, sb in buttons[i + 1:]:
                inter = a.clip(b)
                if inter.w > 3 and inter.h > 3 and a != b:
                    out.append(f"buttons overlap: {sa[:30]!r} / {sb[:30]!r}")
        return out


PAGES = ("lab", "lab_params", "lab_assays", "lab_protocols", "lab_assumptions", "lab_asymmetry", "lab_benchmark", "lab_export", "lab_wiring",
         "lab_critical", "lab_activation", "lab_knockout", "lab_clamp", "lab_diff", "lab_laser", "lab_psych", "lab_classroom", "lab_genetics", "lab_thermo", "lab_patch",
         "lab_imaging", "lab_pharm", "lab_netsci", "lab_sleepdep", "lab_sensitivity", "lab_rigs", "lab_minipapers", "whatsnew",
         "settings:Graphics", "settings:Audio", "settings:Brain", "settings:Controls", "settings:Accessibility", "settings:Help", "confirm_quit")


@pytest.mark.parametrize("lang", ["en", "de"])
@pytest.mark.parametrize("larger", [False, True])
def test_no_page_has_overlapping_or_clipped_text(menu, monkeypatch, lang, larger):
    from kickthefly.core import i18n

    i18n.set_language(lang)
    menu.host.cfg.set("access.larger_text", larger)
    menu.fonts()
    rec = Recorder(monkeypatch)
    bad = {}
    for W, H in SIZES:
        surf = pygame.Surface((W, H))
        for name in PAGES:
            if name.startswith("settings:"):
                menu.screen, menu.tab = "settings", name.split(":")[1]
            else:
                if name not in menu.pages and not hasattr(menu, f"_page_{name}"):
                    continue
                menu.screen = name
            menu.stack = []
            key = menu._scroll_key()
            for scroll_pass in range(3):
                ch = menu.content_h.get(key, 0)
                if scroll_pass and ch <= 0:
                    break
                menu.scroll[key] = 0 if scroll_pass == 0 else ch * scroll_pass / 2
                menu.draw(surf, (5, 5))
                rec.rec.clear()
                menu.draw(surf, (5, 5))
                pw, ph = (440, min(650, H - 20)) if menu.screen in ("pause", "confirm_quit") else (min(980, W - 40), min(680 if H < 1100 else 860, H - 30))
                if menu.screen == "confirm_quit":
                    ph = 250
                panel = pygame.Rect((W - pw) // 2, (H - ph) // 2, pw, ph)
                errors = [s for k, _, s, _, _ in rec.rec if k == "text" and s.startswith("This screen hit an error")]
                p = errors + rec.problems(panel)
                if p:
                    bad[f"{name} {W}x{H} scroll{scroll_pass}"] = sorted(set(p))[:5]
    assert not bad, bad
