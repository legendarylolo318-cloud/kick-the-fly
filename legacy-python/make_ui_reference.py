#!/usr/bin/env python3
"""Screenshots of the pygame menus for the Rust port to be compared with: python make_ui_reference.py [OUT_DIR] [W H]

Draws each screen of kickthefly/ui/menu.py onto a 1280x720 surface (a flat background under the veil) with a default config and the
mouse off the menu, and saves PNGs. The Rust side (`cargo run -p ktf_ui --example render_menu`) draws the same screens with the same
background; `crates/ktf_ui/tests/reference.rs` compares them numerically and the PNGs can be looked at side by side.
"""
import json
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
from kickthefly.ui import crashscreen, help_ui, loadout_ui, neurodex_ui, share_ui, whatsnew_ui  # noqa: E402
from kickthefly.core import bugreport  # noqa: E402

INSTALL = [whatsnew_ui, share_ui, loadout_ui, neurodex_ui, help_ui]

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


SELFTEST_CHECKS = [
    dict(id="build", name="App and Python", status="PASS", detail="Kick the Fly 3.1.1, Python 3.12.3, source, Ubuntu 24.04, kernel 6.8.0", fix=""),
    dict(id="brainpack_adult", name="Adult brain pack", status="PASS", detail="kick_brain.npz: SHA-256 a65baf8602b48c62..., 36.8 MB, no published checksum next to it; file integrity checked", fix=""),
    dict(id="backend_gl", name="Backend gl", status="WARN",
         detail="total firing differs by 6.3% from the CPU reference, population rates correlate at r=0.874; 3,120 steps/s, device: NVIDIA GeForce RTX 4060",
         fix="GPU kernels add inputs in another order, so small differences are normal, but this is larger than expected. Results may not be comparable with CPU runs; use CPU (NumPy) for science."),
    dict(id="audio", name="Audio device", status="FAIL", detail="no audio device could be opened (ALSA: cannot find card '0')",
         fix="The game plays silently. Check your system's sound output or that another program isn't holding it."),
    dict(id="ffmpeg", name="ffmpeg on PATH", status="PASS", detail="found /usr/bin/ffmpeg", fix=""),
    dict(id="memory", name="Free memory vs per-fly estimate", status="PASS", detail="11,435 MB available; one fly needs about 350 MB on top of ~1.5 GB for the game, so about 28 flies fit", fix=""),
    dict(id="smoke", name="10-second smoke protocol", status="PASS", detail="10 s of simulated time ran; looming detectors 17.0 Hz, giant fiber 17.8 Hz", fix=""),
]


def bugreport_items():
    rep = dict(format="kick-the-fly-selftest", created="2026-10-09 12:00:00", app_version="3.1.1",
               counts=dict(PASS=5, WARN=1, FAIL=1), exit_code=1,
               checks=[dict(c, seconds=0.1, data={}) for c in SELFTEST_CHECKS[:4]])
    log = "".join(f"12:00:{i:02d} INFO frame {i} ok, brain 200.0 steps/s, {i * 7} spikes in the window\n" for i in range(60))
    return [
        bugreport.Item("summary", "Summary", "Kick the Fly 3.1.1\nOS: Test Linux 1.0, kernel 6.8.0\nPython: 3.12.3  frozen: False  AppImage: False\n"
                       "Compute backend: cpu (CPU (NumPy))\nMode: 3d\nDisplay: wayland XDG_SESSION_TYPE=wayland\nGPU: Test GPU  GL: 4.6"),
        bugreport.Item("selftest", "Self-test (JSON)", json.dumps(rep, indent=1), long=True),
        bugreport.Item("log", "Last 500 log lines", log, long=True),
        bugreport.Item("crash", "Crash report", "", long=True, include=False),
    ]


def dump_fixtures(out):
    """The self-test list and the bug-report items as JSON, so the Rust test draws the very same content."""
    items = [dict(id=i.id, title=i.title, text=i.text, long=i.long, include=i.include) for i in bugreport_items()]
    (out / "diag_fixture.json").write_text(json.dumps(dict(
        selftest=dict(running=False, checks=SELFTEST_CHECKS, counts=dict(PASS=5, WARN=1, FAIL=1)), items=items), indent=1), encoding="utf-8")


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
    m.pages["crash"] = crashscreen._page_crash
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
    dump_fixtures(out)
    render("selftest", "Graphics", out / "page_selftest.png", prep=lambda m: m.__dict__.__setitem__(
        "st", dict(started=True, running=False, checks=[dict(c, seconds=0.0) for c in SELFTEST_CHECKS], report=dict(counts=dict(PASS=5, WARN=1, FAIL=1)))))
    for n, sel, name in ((0, 0, "summary"), (1, 2, "log"), (2, 4, "copied")):
        render("bugreport", "Graphics", out / f"page_bugreport_{name}.png", prep=lambda m, sel=sel: m.__dict__.__setitem__(
            "br", dict(items=bugreport_items(), sel=sel, note="The fly froze when I used the bomb", saved=None, msg="")))
    render("bugreport", "Graphics", out / "page_bugreport_saved.png", prep=lambda m: m.__dict__.__setitem__(
        "br", dict(items=bugreport_items(), sel=3, note="", saved="/home/user/.local/state/kick-the-fly/bugreports/kickthefly-bugreport-20261009-120000.txt", msg="")))
    host_paths = [Path("/home/user/.local/state/kick-the-fly/KickTheFly-crash.txt"), Path("/opt/kick/KickTheFly-crash.txt")]
    render("crash", "Graphics", out / "page_crash.png", w=900, h=620, prep=lambda m: setattr(m.host, "paths", host_paths))
    print("wrote", out)
