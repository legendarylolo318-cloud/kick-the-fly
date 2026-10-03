"""Regenerate the README screenshots and the demo GIF from the current build, the same way every time.

    python tools/make_screenshots.py                 every README scene, into docs/ (not demo or portrait)
    python tools/make_screenshots.py duel orchard    only these scenes
    python tools/make_screenshots.py --list          the scenes and what each shows
    python tools/make_screenshots.py demo            the README's animated demo (docs/demo.gif), via the video recorder

Each scene is a short script run inside the real 3D game loop (kick3d.run's `script` hook): a fixed seed, a fixed
arena and camera, and the same inputs a player would give (the tool in hand, panel toggles, menu pages, surgery,
training), then a capture of the whole window, HUD and brain panel included. Nothing is drawn specially for the
pictures, so a scene that stops looking right usually means the game changed.

The Validation scene shows data/validation_results.json (or the file KTF_SHOTS_VALIDATION names), so run
`python kick_the_fly.py --headless --validate --out data/validation_results.json` first.

The game renders with SDL's offscreen driver (a real OpenGL context on your GPU, but no window) at the game's native
1280x760, with a throwaway KICK_THE_FLY_HOME so your settings, training memory and saves are never touched. Needs a GPU
with OpenGL 3.3 and, for the demo, ffmpeg. The brain panel is 'solid' in every scene except the see-through one.
Output PNGs are palette-quantized (256 colours) to keep the README light; docs/ is not bundled in the exe/AppImage.
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DOCS = ROOT / "docs"
SEED = 7

os.environ.setdefault("SDL_VIDEODRIVER", "offscreen")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("KICK_THE_FLY_OFFLINE", "1")              # use the cached neuPrint skeletons, don't fetch
_home = tempfile.mkdtemp(prefix="ktf-shots-")
os.environ["KICK_THE_FLY_HOME"] = _home

import numpy as np  # noqa: E402
import pygame  # noqa: E402

from kickthefly.core import config  # noqa: E402
from kickthefly.game import kick3d, kick_the_fly as k2  # noqa: E402

TOOL = {name: i for i, (name, *_rest) in enumerate(kick3d.TOOLS)}


# --- camera and input helpers --------------------------------------------------------------------------------------
def fly_pos(game, i: int = 0) -> np.ndarray:
    return game.flies[i].fly.p[k2.THX].copy()


def look_at(game, target, dist: float | None = None, bearing: float | None = None, height: float | None = None) -> None:
    """Put your eye `dist` metres from target (horizontally, from compass `bearing` radians), at eye `height`, looking
    at it. Called every frame, so the camera follows a fly that moves."""
    p = game.player
    target = np.asarray(target, float)
    if dist is not None:
        b = math.atan2(p.pos[1] - target[2], p.pos[0] - target[0]) if bearing is None else bearing
        p.pos[:] = (target[0] + dist * math.cos(b), target[2] + dist * math.sin(b))
    if height is not None:
        p.eye_h = height
    d = target - p.eye
    p.yaw = math.atan2(d[2], d[0])
    p.pitch = float(np.clip(math.atan2(d[1], math.hypot(d[0], d[2])), -1.4, 1.4))


def hold(game, tool: str) -> None:
    game.tool = TOOL[tool]


def fire(game, tool: str | None = None) -> None:
    """Pull the trigger of the tool in hand (held tools keep firing while this is called every frame)."""
    if tool:
        hold(game, tool)
    game.use_tool3d(game.clock.now)


def release(game) -> None:
    game.torching = False
    if hasattr(game, "laser_state"):
        game.laser_state.trigger_release()


class Scene:
    """A timeline: `steps` is a list of (seconds after the brain is warm, fn(game)); `every(game, t)` runs each frame;
    the capture happens at `shot_at` seconds, or when `ready(game)` first holds after `shot_at`."""

    def __init__(self, name: str, caption: str, steps=(), every=None, shot_at: float = 6.0, ready=None,
                 timeout: float = 40.0, arena: str = "room", flies: int = 1, lab: bool = False, panel: str = "solid",
                 settle: float = 4.0, settings: dict | None = None):
        self.name, self.caption, self.steps, self.every = name, caption, list(steps), every
        self.shot_at, self.ready, self.timeout = shot_at, ready, timeout
        self.arena, self.flies, self.lab, self.panel, self.settle = arena, flies, lab, panel, settle
        self.settings = dict(settings or {})          # extra config settings for this scene (key -> value)


def near(dist, bearing=None, height=None, i=0):
    return lambda game, t: look_at(game, fly_pos(game, i), dist, bearing, height)


SCENES: dict[str, Scene] = {}


def scene(*a, **kw) -> None:
    s = Scene(*a, **kw)
    SCENES[s.name] = s


# --- the scenes ------------------------------------------------------------------------------------------------------
scene("room3d", "The 3D room: first person, with the fly's live brain on the right",
      every=near(1.5, bearing=math.pi / 2 + 0.5, height=1.25), shot_at=5.0)

scene("swat3d", "Swatting in first person: the touch neurons fire and the leg-touch descending neurons kick",
      steps=[(0.0, lambda g: g.set_setting("brain.immortal", True, save=False)),
             (3.0, lambda g: hold(g, "swatter")), (5.0, lambda g: fire(g)), (6.2, lambda g: fire(g))],
      every=near(1.05, bearing=math.pi / 2 + 0.35, height=1.0), shot_at=6.45)

scene("flypaper3d", "Stuck on flypaper: it struggles through its own touch and descending neurons",
      arena="flypaper", every=near(1.2, bearing=math.pi / 2 + 0.6, height=1.05), shot_at=8.0)

scene("see-through", "The see-through brain panel (V), here in the open field",
      arena="field", panel="see-through",
      steps=[(0.0, lambda g: g.set_setting("brain.immortal", True, save=False))],
      every=near(1.6, bearing=math.pi / 2 + 0.9, height=1.2), shot_at=7.0)


def _torch_every(game, t):
    look_at(game, fly_pos(game), 0.8, math.pi / 2 + 0.4, 0.9)
    if 4.0 <= t < 6.8:
        fire(game, "torch")
    elif t >= 6.8:
        release(game)


scene("brain", "The brain under the blowtorch: its heat-sensing and touch neurons light up the big brain view (B)",
      steps=[(0.0, lambda g: g.set_setting("brain.immortal", True, save=False)),
             (6.8, lambda g: setattr(g, "big_view", True))],
      every=_torch_every, shot_at=7.1)


def _spider_every(game, t):
    look_at(game, fly_pos(game), 1.15, math.pi / 2 + 0.5, 1.0)


scene("spider", "A spider drops, bites and wraps the fly in silk (immortal here, so it will break out)",
      steps=[(0.0, lambda g: g.set_setting("brain.immortal", True, save=False)),
             (3.0, lambda g: fire(g, "spider"))],
      every=_spider_every, shot_at=6.0, ready=lambda g: g.flies[0].fly.wrapped, timeout=40.0)

scene("duel", "1v1 duel (X): the fly aims through LC10 -> DNa02 and shoots when its DNp35 object neurons fire",
      steps=[(0.0, lambda g: g.set_setting("brain.immortal", True, save=False)), (2.0, lambda g: g.toggle_duel())],
      every=near(1.6, bearing=math.pi / 2 + 0.2, height=1.1), shot_at=9.0,
      ready=lambda g: any(r[1] == "SHOOT" for r in list(getattr(g, "reactions", []))[-3:]) or True)


def _train_start(g):
    g.training_open = True
    g.train_scent = "swatter"
    g.train_speed = 3.0
    g.start_training("fear", 10)


scene("training", "Training (T): pairing a smell with a shock weakens its real Kenyon cell -> MBON synapses",
      steps=[(1.0, _train_start)], every=near(1.5, height=1.2), shot_at=8.0, ready=lambda g: g.train is None,
      timeout=240.0)


def _inspect(g):
    g.big_view = True
    dnp01 = int(np.flatnonzero(g.graph.type.astype(str) == "DNp01")[0])
    g.inspect = g._neuron_info(dnp01)


scene("inspect", "Neuron inspector: click any neuron in the big brain view for its type, firing and strongest "
                 "connections; the giant fibre is one of the neurons drawn from its real neuPrint skeleton",
      steps=[(2.0, _inspect)], every=near(1.5, height=1.2), shot_at=5.0)


def _mdn_on(g):
    g.surgery_open = True
    k = [label for label, _ in k2.SURGERY].index("Moonwalker neurons (MDN)")
    g._set_surgery(k, 1)


scene("surgery", "Brain surgery (O): switching the moonwalker neurons (MDN) on makes it back up",
      steps=[(1.0, _mdn_on)], every=near(1.5, height=1.2), shot_at=5.0)

scene("lamp", "The lamp arena: drawn to the light, it flies up to the bulb and singes itself",
      arena="lamp", steps=[(0.0, lambda g: g.set_setting("brain.immortal", True, save=False))],
      every=lambda g, t: look_at(g, kick3d.LAMP3 + (0.0, -0.35, 0.0), 1.5, math.pi / 2 + 0.3, 1.5), shot_at=6.0,
      ready=lambda g: float(np.linalg.norm(fly_pos(g) - kick3d.LAMP3)) < 0.45, timeout=60.0)


def _autopsy_every(game, t):
    if game.report is None:
        look_at(game, fly_pos(game), 0.8, math.pi / 2 + 0.4, 0.9)
        if t >= 3.0:
            fire(game, "torch")
    else:
        release(game)


scene("autopsy", "Autopsy: every brain region's last 2 s alive against its calm baseline, and pain on a timeline",
      every=_autopsy_every, shot_at=4.0, ready=lambda g: g.report is not None, timeout=90.0)


def _settings(g):
    g.open_menu("settings")
    g.menu.tab = "Brain"
    g.menu.scroll["settings:Brain"] = 400.0          # down to Compute backend and State precision


scene("settings", "Settings (Esc > Settings), Brain tab: each setting is tagged Connectome or Game rule, including "
                  "the new compute backend and state precision",
      steps=[(1.0, _settings)], every=near(1.5, height=1.2), shot_at=2.5)

scene("lab", "Lab mode: the research tools, with the simulation engine that's running shown top right",
      lab=True, steps=[(1.0, lambda g: g.open_menu("lab"))], every=near(1.5, height=1.2), shot_at=2.5)

scene("validation", "Lab > Validation: which published fly behaviours this sim reproduces, with numbers, and which "
                    "it doesn't",
      lab=True, steps=[(1.0, lambda g: (g.open_menu("lab"), g.menu.show("lab_validation")))],
      every=near(1.5, height=1.2), shot_at=2.5)

scene("field", "The open field: wind drives its real antennal wind neurons, and escapes carry it away",
      arena="field", steps=[(0.0, lambda g: g.set_setting("brain.immortal", True, save=False))],
      every=lambda g, t: look_at(g, fly_pos(g), 2.6, math.pi / 2 + 1.1, 1.6), shot_at=9.0)


def _orchard_every(game, t):
    """Follow a fly on its way to or on a fruit (switching if it gives up), with the HUD focused on it."""
    lock = getattr(game, "_shot_lock", None)
    if lock is not None and (lock >= len(game.flies) or getattr(game.flies[lock].fly, "perch", None) is None):
        lock = None                                  # it gave up on its fruit (or flew off): pick another
    if lock is None:
        perched = [i for i, s in enumerate(game.flies) if getattr(s.fly, "perch", None) is not None]
        lock = perched[0] if perched else None
        game._shot_lock = lock
    i = 0 if lock is None else lock
    if lock is not None:
        game.focus = lock
        game._manual_focus_until = time.perf_counter() + 1.0
    fp = fly_pos(game, i)
    fruit = getattr(game.flies[i], "fruit", None)
    if fruit is not None and getattr(game, "orchard", None) is not None:
        # stand between the fruit and its trunk, inside the crown (whose back faces aren't drawn), looking out at the fly
        trunk = game.orchard.trees[fruit.tree]["pos"]
        gap = math.hypot(trunk[0] - fp[0], trunk[2] - fp[2])
        look_at(game, fp, 0.55 * gap, math.atan2(trunk[2] - fp[2], trunk[0] - fp[0]), max(0.5, fp[1] - 0.3))
    else:
        look_at(game, fp, 1.6, math.pi / 2 + 0.7, 1.5)


scene("orchard", "The orchard with five flies: they fly to fruit and feed, driving the same sugar and reward "
                 "neurons the sugar tool does",
      arena="orchard", flies=5, every=_orchard_every, shot_at=20.0,
      ready=lambda g: getattr(g, "_shot_lock", None) is not None
      and getattr(g.flies[g._shot_lock].fly, "perch", None) is not None and g.flies[g._shot_lock].fly.eating_until > g.clock.now,
      timeout=120.0)


def _laser_every(game, t):
    look_at(game, fly_pos(game), 1.1, math.pi / 2 + 0.45, 1.05)
    if t >= 3.0:
        fire(game, "laser")


def _laser_setup(g):
    g.laser_state.set_target("dnp01")
    g.laser_state.set_mode("activate")


scene("laser", "The optogenetics laser (Lab, key =): aim it and it drives (or silences) a chosen cell type, here the "
               "giant fibre DNp01",
      lab=True, steps=[(0.0, _laser_setup), (0.0, lambda g: g.set_setting("brain.immortal", True, save=False))],
      every=_laser_every, shot_at=4.2)

scene("escaperoom", "The escape room: fan, flypaper and a hot lamp between the fly and the sugar dish",
      arena="escaperoom", steps=[(0.0, lambda g: g.set_setting("brain.immortal", True, save=False))],
      every=lambda g, t: look_at(g, fly_pos(g), 2.2, math.pi / 2 + 0.3, 1.6), shot_at=8.0)


def _to_hot_side(g):
    """Put the fly on the thermo arena's warm half, where its hot-sensing antennal neurons are driven."""
    for slot in g.flies:
        dx = 2.4 - float(slot.fly.p[k2.THX, 0])
        slot.fly.p[:, 0] += dx
        slot.fly.prev[:, 0] += dx


scene("thermo", "The thermo arena: cold on the left, hot on the right; the warm floor drives its hot-sensing antennal "
                "neurons (TRN_VP2) and their VP2 projection neurons",
      arena="thermo", steps=[(0.0, lambda g: g.set_setting("brain.immortal", True, save=False)), (0.5, _to_hot_side)],
      every=lambda g, t: look_at(g, fly_pos(g), 2.6, math.pi / 2 + 0.25, 1.7), shot_at=6.0)


def _paths(g):
    from kickthefly.lab import neurosearch

    br = g.brain
    g.big_view = True
    src, dst = neurosearch.search(br, "LPLC2")[0], neurosearch.search(br, "DNp01")[0]
    g.set_path_end("from", src)
    g.set_path_end("to", dst)
    g.inspect = g._neuron_info(dst)


scene("paths", "The path tracer in the big brain view (B): the strongest paths from a looming detector (LPLC2) to the "
               "giant fiber, with live spikes running along them while something looms",
      steps=[(1.0, _paths)], every=lambda g, t: (near(1.5, height=1.2)(g, t), g.brain.poke("loom", None, 0.9)),
      shot_at=4.0)


def _decoy_setup(g):
    """One decoy on the floor beside the fly's head, within its forelegs' reach; she stays where she is put."""
    g.set_setting("brain.immortal", True, save=False)
    fly = g.flies[0].fly
    fwd = fly.p[k2.HEAD] - fly.p[k2.THX]
    fwd[1] = 0.0
    norm = float(np.linalg.norm(fwd))
    fwd = fwd / norm if norm > 1e-4 else np.array([1.0, 0.0, 0.0])
    side = np.array([-fwd[2], 0.0, fwd[0]])
    dp = fly.p[k2.HEAD] + fwd * 0.12 + side * 0.28           # beside its head, facing it, bodies apart
    dp[1] = kick3d.STAND3
    g.decoys3 = [{"p": dp, "v": np.zeros(3), "landed": True, "yaw": float(math.atan2(-side[2], -side[0]))}]


def _decoy_every(game, t):
    fly = game.flies[0].fly
    dec = getattr(game, "decoys3", None)
    target = (fly.p[k2.THX] + dec[0]["p"]) / 2 if dec else fly.p[k2.THX]
    look_at(game, target, 1.6, math.pi / 2 + 0.4, 1.2)


scene("decoy", "A decoy female dropped next to the fly: while its forelegs touch her, the HUD shows its LgLG5-8 taste "
               "neurons and P1",
      arena="room", steps=[(0.0, _decoy_setup)], every=_decoy_every, shot_at=0.6,
      ready=lambda g: getattr(g.flies[0].fly, "decoy_contact_until", 0) > g.clock.now, timeout=12.0)


scene("portrait", "Close-up of the fly model itself (not used in the README; for checking the model after a change)",
      steps=[(0.0, lambda g: g.set_setting("brain.immortal", True, save=False))],
      every=lambda g, t: look_at(g, fly_pos(g), 0.8, math.pi / 2 + 2.6, 0.42), shot_at=6.0)


def _demo_every(game, t):
    """~12 s in the open field: walk up, swat it, torch it, and watch it take off, with the brain panel lighting up."""
    fp = fly_pos(game)
    look_at(game, fp, max(0.9, 2.2 - 0.25 * max(0.0, t - 3.0)), math.pi / 2 + 0.6 + 0.04 * t, 1.1)
    if 5.0 <= t < 5.1 or 6.2 <= t < 6.3:
        fire(game, "swatter")
    elif 8.0 <= t < 10.0:
        fire(game, "torch")
    elif t >= 10.0:
        release(game)
        hold(game, "hand")


scene("demo", "Animated demo for the top of the README", arena="field",
      steps=[(0.0, lambda g: g.set_setting("brain.immortal", True, save=False)), (4.5, lambda g: hold(g, "swatter"))],
      every=_demo_every, shot_at=12.5, settle=3.0)


# --- driver ----------------------------------------------------------------------------------------------------------
def run_scene(s: Scene, out: Path) -> Path:
    results = Path(os.environ.get("KTF_SHOTS_VALIDATION", ROOT / "data" / "validation_results.json"))
    if results.exists():                             # the Validation page shows these as "run on this PC"
        (Path(_home) / "data").mkdir(parents=True, exist_ok=True)
        shutil.copy(results, Path(_home) / "data" / "validation_results.json")
    elif s.name == "validation":
        print("  validation: no results to show; run --headless --validate --out data/validation_results.json first")
    cfg = config.Config(None)
    cfg.set("brain.mode", "lab" if s.lab else "play")
    cfg.set("graphics.panel_mode", s.panel)
    cfg.set("graphics.fps_cap", 60)
    cfg.set("brain.arena", s.arena)
    cfg.set("brain.science_popups", False)             # the "Real flies do this too" cards would cover the scene
    # 3.0 release review: a fresh config shows the first-launch tutorial (and the Neuron of the Day card) over every capture
    cfg.first_run.update(tutorial_done=True, loadout_notice=False, neuron_shapes_asked=True, whatsnew_3_0_seen=True)
    cfg.set("brain.neuron_of_day", False)
    for key, value in s.settings.items():
        cfg.set(key, value)
    state = dict(started=None, done=set(), shot=None, video=None)
    demo = s.name == "demo"

    def script(game, app, t, lay):
        if not game.brain.steps or t < 0.5:
            return True
        if state["started"] is None:                         # the brain is running: start the scene's clock
            state["started"] = t + s.settle
            game.set_look(True)
            if s.arena != "room" and k2.ARENAS[game.arena_i] != s.arena:
                game.set_setting("brain.arena", s.arena, save=False)
        st = t - state["started"]
        if s.every:
            s.every(game, max(0.0, st))
        if st < 0:
            return True
        for i, (at, fn) in enumerate(s.steps):
            if i not in state["done"] and st >= at:
                state["done"].add(i)
                fn(game)
        if demo:
            if state["video"] is None:
                state["video"] = out.with_suffix(".mp4")
                game.toggle_video_recording(state["video"])
            if st >= s.shot_at:
                game.toggle_video_recording()
                return False
            return True
        if st >= s.shot_at and state.get("ready_at") is None and (s.ready is None or s.ready(game) or st >= s.timeout):
            if s.ready is not None and not s.ready(game):
                print(f"  {s.name}: timed out waiting for its state; capturing anyway")
            state["ready_at"] = st          # the screen still holds the previous frame: let the state show first
        if state.get("ready_at") is not None and st >= state["ready_at"] + (0.4 if s.ready else 0.0):
            app.ctx.screen.use()
            app.screenshot(out, lay)
            state["shot"] = out
            return False
        return True

    pygame.init()
    real_sizes = pygame.display.get_desktop_sizes
    pygame.display.get_desktop_sizes = lambda: [(1920, 1080)]   # offscreen reports 1024x768; open at 1280x760
    try:
        kick3d.run(seed=SEED, cfg=cfg, flies=s.flies, script=script)
    finally:
        pygame.display.get_desktop_sizes = real_sizes
    return state["video"] if demo else state["shot"]


def shrink_png(path: Path) -> None:
    """256-colour palette PNG: about a third of the size, with no visible change at README scale."""
    from PIL import Image
    im = Image.open(path).convert("RGB")
    im.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG).save(
        path, optimize=True)


def mp4_to_gif(mp4: Path, gif: Path, width: int = 600, fps: int = 10) -> None:
    """A README-sized GIF from the recorder's MP4: one palette for the whole clip, so the colours stay steady."""
    vf = f"fps={fps},scale={width}:-1:flags=lanczos"
    with tempfile.TemporaryDirectory() as td:
        pal = Path(td) / "palette.png"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-vf", f"{vf},palettegen=max_colors=128:stats_mode=diff",
                        str(pal)], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-i", str(pal), "-lavfi",
                        f"{vf} [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
                        "-loop", "0", str(gif)], check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("scenes", nargs="*", help="scene names (default: every screenshot scene, not the demo)")
    ap.add_argument("--out", type=Path, default=DOCS)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--keep-png", action="store_true", help="don't palette-quantize the PNGs")
    ap.add_argument("--keep-mp4", action="store_true", help="keep the demo's MP4 next to the GIF")
    args = ap.parse_args()
    if args.list:
        for s in SCENES.values():
            print(f"{s.name:12s} {s.caption}")
        return 0
    names = args.scenes or [n for n in SCENES if n not in ("demo", "portrait")]
    bad = [n for n in names if n not in SCENES]
    if bad:
        ap.error(f"unknown scene(s): {', '.join(bad)} (see --list)")
    args.out.mkdir(parents=True, exist_ok=True)
    if len(names) > 1:                  # the game tears pygame and GL down when it quits: one process per scene
        failed = []
        for n in names:
            cmd = [sys.executable, __file__, n, "--out", str(args.out)] + (["--keep-png"] if args.keep_png else [])
            if subprocess.run(cmd).returncode != 0:
                failed.append(n)
        if failed:
            print(f"FAILED: {', '.join(failed)}")
        return 1 if failed else 0
    for n in names:
        t0 = time.perf_counter()
        if n == "demo":
            mp4 = run_scene(SCENES[n], args.out / "demo.mp4")
            gif = args.out / "demo.gif"
            mp4_to_gif(mp4, gif)
            if not args.keep_mp4:
                mp4.unlink(missing_ok=True)
            print(f"{n}: {gif} ({gif.stat().st_size / 1e6:.1f} MB, {time.perf_counter() - t0:.0f} s)")
            continue
        out = run_scene(SCENES[n], args.out / f"{n}.png")
        if out is None:
            print(f"{n}: FAILED (no capture)")
            return 1
        if not args.keep_png:
            shrink_png(out)
        print(f"{n}: {out} ({out.stat().st_size // 1024} KB, {time.perf_counter() - t0:.0f} s)")
    shutil.rmtree(_home, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
