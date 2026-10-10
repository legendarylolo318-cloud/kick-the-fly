#!/usr/bin/env python3
"""Screenshots of the pygame Lab pages (lab/lab.py) for the Rust port: python make_lab_reference.py [OUT_DIR].
Writes lab_fixture.json (the page list the game answers `lab_pages` with) and page_lab_*.png."""
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_ui_reference as R  # noqa: E402
from kickthefly.game import kick_the_fly as k  # noqa: E402
from kickthefly.lab import lab  # noqa: E402

pages = k.Game.lab_pages(SimpleNamespace())


def prep_for(screen, params=None, scroll=0):
    def prep(m):
        lab.install(m)
        m.host.lab_pages = lambda: pages
        m.host.lab_params = dict(params or {})
        m.host.wiring = None
        m.host.brain = None
        m.scroll[screen] = scroll
    return prep


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "tests" / "ui_reference")
    (out / "lab_fixture.json").write_text(json.dumps(dict(pages=pages, params=dict(bias=0.25, **{"thresh.escape": 5.5, "loom_min": 2.0}))))
    R.INSTALL = []
    mod = dict(bias=0.25, **{"thresh.escape": 5.5, "loom_min": 2.0})
    R.render("lab", "Graphics", out / "page_lab_hub.png", prep=prep_for("lab"))
    R.render("lab", "Graphics", out / "page_lab_hub_modified.png", prep=prep_for("lab", mod))
    R.render("lab_params", "Graphics", out / "page_lab_params.png", prep=prep_for("lab_params", mod))
    R.render("lab_params", "Graphics", out / "page_lab_params_scrolled.png", prep=prep_for("lab_params", mod, 600))
    R.render("lab_assumptions", "Graphics", out / "page_lab_assumptions.png", prep=prep_for("lab_assumptions"))
    R.render("lab_assumptions", "Graphics", out / "page_lab_assumptions_scrolled.png", prep=prep_for("lab_assumptions", None, 700))
    print("wrote", out)
