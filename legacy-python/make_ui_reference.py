#!/usr/bin/env python3
"""Screenshots of the pygame menus for the Rust port to be compared with: python make_ui_reference.py [OUT_DIR] [W H]

Draws each screen of kickthefly/ui/menu.py onto a 1280x720 surface (a flat background under the veil) with a default config and the
mouse off the menu, and saves PNGs. The Rust side (`cargo run -p ktf_ui --example render_menu`) draws the same screens with the same
background; `crates/ktf_ui/tests/reference.rs` compares them numerically and the PNGs can be looked at side by side.
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pygame  # noqa: E402

from kickthefly.core import config  # noqa: E402
from kickthefly.core import loadout as loadout_mod  # noqa: E402
from kickthefly.ui import menu as M  # noqa: E402
from kickthefly.ui import loadout_ui, neurodex_ui, share_ui, whatsnew_ui  # noqa: E402

INSTALL = [whatsnew_ui, share_ui, loadout_ui, neurodex_ui]

BG = (60, 70, 80)


def dex_host_parts():
    """A real type table from the pack and a deterministic progress (the first 400 types, every third by stimulation, plus DNp01 and LPLC2)."""
    import datetime
    import tempfile

    import numpy as np

    from kickthefly.core import neurodex as nd

    pack = Path(__file__).resolve().parent.parent / "data" / "kick_brain.npz"
    z = np.load(pack, allow_pickle=False)
    keep = ("type", "superclass", "region", "nt", "nt_conf", "nt_source", "body_id", "indptr", "indices", "data")
    tab = nd.build_table({k: z[k] for k in keep if k in z.files}, "adult")
    prog = nd.Progress(Path(tempfile.mkdtemp()) / "neurodex.json")
    for i, n in enumerate(list(tab.names[:400]) + ["DNp01", "LPLC2"]):
        prog.mark("adult", str(n), 2.5 + (i % 7) * 0.5, "stimulated" if i % 3 == 0 else "play", "2026-01-02T03:04:05")

    class D(datetime.date):
        @classmethod
        def today(cls):
            return cls(2026, 10, 9)

    neurodex_ui._dt = SimpleNamespace(date=D)
    x3 = SimpleNamespace(ensure_table=lambda: None, ensure_progress=lambda: prog, brain_name="adult", table=tab, table_state="ready", tracker=None,
                         table_error="")
    return x3


def render(screen, tab, out, w=1280, h=720, three_d=True, mutate=None, prep=None, loadout_override=None, x3=None):
    pygame.init()
    cfg = config.Config()
    if mutate:
        mutate(cfg)
    from kickthefly.game import kick_the_fly as k

    host = SimpleNamespace(x3=x3, cfg=cfg, three_d=three_d, set_setting=lambda *a, **k: None, menu_action=lambda *a: None, flies=[], pad=None,
                           surgery_modes=[0] * len(k.SURGERY), type_ops={}, lab_params={}, loadout=loadout_mod.resolve(cfg), tool_name=lambda: "hand", is_larva=False)
    host.loadout.tools = host.loadout.tools if not loadout_override else loadout_override
    m = M.Menu(host)
    for mod in INSTALL:
        mod.install(m)
    m.show(screen)
    m.tab = tab
    if prep:
        prep(m)
    surf = pygame.Surface((w, h))
    surf.fill(BG)
    m.draw(surf, (-1, -1))
    pygame.image.save(surf, str(out))


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "tests" / "ui_reference")
    out.mkdir(parents=True, exist_ok=True)
    render("pause", "Graphics", out / "pause.png")
    render("confirm_quit", "Graphics", out / "confirm_quit.png")
    for tab in ("Graphics", "Audio", "Brain", "Controls", "Accessibility", "Help"):
        render("settings", tab, out / f"settings_{tab.lower()}.png")
    render("settings", "Controls", out / "settings_controls_2d.png", three_d=False)
    render("pause", "Graphics", out / "pause_lab.png", mutate=lambda c: c.set("brain.mode", "lab"))
    render("settings", "Graphics", out / "settings_graphics_big.png", mutate=lambda c: c.set("access.larger_text", True))
    render("whatsnew", "Graphics", out / "page_whatsnew.png")
    x3 = dex_host_parts()
    render("neurodex", "Graphics", out / "page_neurodex.png", x3=x3, prep=lambda m: setattr(neurodex_ui._state(m), "selected", "DNp01"))
    render("neurodex", "Graphics", out / "page_neurodex_unknown.png", x3=x3,
           prep=lambda m: (setattr(neurodex_ui._state(m), "selected", "DNp02"), setattr(neurodex_ui._state(m), "only_found", False)))
    render("loadout", "Graphics", out / "page_loadout_base.png")
    render("loadout", "Graphics", out / "page_loadout_all.png", mutate=lambda c: c.set("controls.loadout_preset", "all"))
    render("share", "Graphics", out / "page_share_surgery.png")
    render("share", "Graphics", out / "page_share_loadout.png", prep=lambda m: setattr(share_ui._state(m), "kind", "loadout"))
    render("share", "Graphics", out / "page_share_import.png", prep=lambda m: setattr(share_ui._state(m), "tab", "import"))
    print("wrote", out)
