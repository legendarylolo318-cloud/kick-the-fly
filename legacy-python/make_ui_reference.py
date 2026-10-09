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
from kickthefly.ui import menu as M  # noqa: E402

BG = (60, 70, 80)


def render(screen, tab, out, w=1280, h=720, three_d=True, mutate=None):
    pygame.init()
    cfg = config.Config()
    if mutate:
        mutate(cfg)
    host = SimpleNamespace(cfg=cfg, three_d=three_d, set_setting=lambda *a, **k: None, menu_action=lambda *a: None, flies=[], pad=None)
    m = M.Menu(host)
    m.show(screen)
    m.tab = tab
    surf = pygame.Surface((w, h))
    surf.fill(BG)
    m.draw(surf, (-1, -1))
    pygame.image.save(surf, str(out))


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "tests" / "ui_reference")
    out.mkdir(parents=True, exist_ok=True)
    render("pause", "Graphics", out / "pause.png")
    render("confirm_quit", "Graphics", out / "confirm_quit.png")
    for tab in ("Graphics", "Audio", "Brain", "Controls", "Accessibility"):
        render("settings", tab, out / f"settings_{tab.lower()}.png")
    render("settings", "Controls", out / "settings_controls_2d.png", three_d=False)
    render("pause", "Graphics", out / "pause_lab.png", mutate=lambda c: c.set("brain.mode", "lab"))
    render("settings", "Graphics", out / "settings_graphics_big.png", mutate=lambda c: c.set("access.larger_text", True))
    print("wrote", out)
