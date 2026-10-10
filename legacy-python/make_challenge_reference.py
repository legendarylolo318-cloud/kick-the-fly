#!/usr/bin/env python3
"""Screenshots of the Play-mode challenge panels (`lab/challenges.py`) for the Rust port to be compared with:
python make_challenge_reference.py [OUT_DIR]

Each panel is drawn by the real `Challenge.draw` of the Python game onto a 1280x760 surface (a flat background), with a stand-in `game`
that supplies the fonts (the same FreeSans / FreeMono the Rust HUD bundles), `_text` and the state each panel is put in. Python's Predict
result card reads a `desc` key its readouts lack (a KeyError); the readouts get `desc` = the cue text here, which is what Rust shows."""
import os
import sys
from pathlib import Path


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import pygame  # noqa: E402

from kickthefly.lab import challenges as ch  # noqa: E402

pygame.init()
FONTS = ROOT.parent / "assets" / "fonts"
BG = (60, 70, 80)


def font(name, size):
    return pygame.font.Font(str(FONTS / name), size)


class Game:
    def __init__(self):
        self.f_small = font("FreeMono.otf", 13)
        self.f_text = font("FreeSans.otf", 15)
        self.f_bold = font("FreeSansBold.otf", 16)
        self.f_head = font("FreeSansBold.otf", 20)
        self.f_title = font("FreeSansBold.otf", 34)
        self.frame = 0
        self.mouse = (-1, -1)
        self.sound = SimpleNamespace(play=lambda *a, **k: None)
        self.brain = SimpleNamespace(memory=object(), seed=1, steps=0, hz=lambda n: 0.0, override=np.zeros(10), types=np.array([], dtype="<U1"))
        slot = SimpleNamespace(hits=0, fly=SimpleNamespace(dead=False, escape_until=0.0, p=np.zeros((3, 2))), threat_x=0.0)
        self.flies, self.focus = [slot], 0
        self.clock = SimpleNamespace(now=100.0)
        self.challenge = None

    def _text(self, surf, s, pos, color=ch.TEXT, font=None, anchor="topleft"):
        img = (font or self.f_text).render(s, True, color)
        rect = img.get_rect(**{anchor: pos})
        surf.blit(img, rect)
        return rect

    def sneak_distance(self, slot):
        return 1.7

    def start_challenge(self, key):
        pass

    def note(self, *a):
        pass

    def on_reaction(self, *a):
        pass


def shot(out, name, obj, now=100.0):
    surf = pygame.Surface((1280, 760))
    surf.fill(BG)
    obj.draw(surf, now, (-1, -1))
    pygame.image.save(surf, str(out / f"challenge_{name}.png"))


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT.parent / "tests" / "ui_reference")
    out.mkdir(parents=True, exist_ok=True)
    scores = {"tmaze": 8, "sneak": 1.4, "sweet": 35, "reverse_surgery": 2}
    ch.load_scores = lambda: dict(scores)
    g = Game()

    t = ch.TMaze(g)
    shot(out, "tmaze_choose", t)
    t.zap, t.phase, t.stage, t.cycle = "odor_a", "train", 1, 2
    shot(out, "tmaze_train", t)
    t.phase, t.stage, t.cycle = "naive", 1, 0
    shot(out, "tmaze_naive", t)
    t.phase, t.stage = "test", 0
    t.choices = [("odor_a", False), ("odor_b", True), ("odor_b", True), ("odor_a", False), ("odor_b", True), ("odor_b", True)]
    shot(out, "tmaze_test", t)
    t.phase, t.stage = "done", 3
    t.choices = [("odor_b", True)] * 7 + [("odor_a", False)] * 3
    t.best = False
    shot(out, "tmaze_done", t)
    t.best = True
    shot(out, "tmaze_done_best", t)

    s = ch.Sneak(g)
    shot(out, "sneak_armed", s)
    s.state, s.closest = "sneaking", 1.4
    shot(out, "sneak_sneaking", s)
    s.state, s.result, s.best = "result", ("seen", 1.3), True
    shot(out, "sneak_seen", s)
    s.result, s.best = ("caught", 0.0), False
    shot(out, "sneak_caught", s)

    w = ch.Sweet(g)
    w.dose = 35
    shot(out, "sweet_ready", w)
    w.phase, w.offers, w.shown = "shown", [(50, True, 2.1), (30, False, 1.1), (40, True, 1.8)], 100.0
    shot(out, "sweet_shown", w)
    w.phase, w.offers, w.score, w.best = "done", [(50, True, 2.0)] * 3 + [(30, False, 1.0)] * 5, 25, True
    shot(out, "sweet_done", w)

    g.brain = None
    r = ch.ReverseSurgery(g, choice_idx=0)
    r.hints_revealed = 1
    shot(out, "reverse_testing", r)
    r.state = "guessing"
    shot(out, "reverse_guessing", r)
    r.state, r.guess_result, r.guessed_id = "revealed", True, "jump"
    shot(out, "reverse_correct", r)
    r.guess_result, r.guessed_id = False, "wings"
    shot(out, "reverse_wrong", r)

    for rd in ch.PredictNeuron.READOUTS:
        rd["desc"] = rd["cue_text"]
    p = ch.PredictNeuron(g)
    p.state, p.round_idx, p.score, p.current_readout, p.timer, p.pulse_phase = "cue", 2, 2, ch.PredictNeuron.READOUTS[1], 101.1, 1.0
    shot(out, "predict_cue", p)
    p.state, p.history = "result", [("jump", "jump", True), ("run", "kick", False), ("kick", "kick", True)]
    shot(out, "predict_result", p)
    p.state, p.score = "done", 7
    shot(out, "predict_done", p)
    # the brain surgery screen (Game._draw_surgery) with a few entries switched
    from kickthefly.game import kick_the_fly as k
    counts = [2, 387, 5088, 1846, 14, 2635, 4064, 316, 16, 4, 2, 29, 2810, 105267, 2344, 2372, 79187, 81471, 166700, 2, 86, 5, 140]
    modes = [0] * len(k.SURGERY)
    modes[0], modes[5], modes[14], modes[18] = -1, 1, -1, 0
    stub = SimpleNamespace(f_small=g.f_small, f_text=g.f_text, f_bold=g.f_bold, f_head=g.f_head, f_title=g.f_title, _text=g._text,
                           surgery_rows={label: np.zeros(n) for (label, _), n in zip(k.SURGERY, counts)}, surgery_modes=modes,
                           type_ops={"DNp01": -1})
    surf = pygame.Surface((1280, 760))
    surf.fill(BG)
    k.Game._draw_surgery(stub, surf)
    pygame.image.save(surf, str(out / "challenge_surgery.png"))
    # the brain autopsy card (Game._draw_autopsy) from tests/golden/autopsy.json
    import json
    gold = json.loads((ROOT.parent / "tests" / "golden" / "autopsy.json").read_text())
    pretty = {"head": "touch: head", "body": "touch: body", "legs": "touch: legs", "wing": "touch: wings", "heat": "heat sensors",
              "cold": "cold sensors", "smell": "smell ORNs", "taste": "taste neurons", "reward": "PAM reward DANs",
              "jump": "DNs: jump group", "run": "DNs: run group", "fly": "DNs: DNg02 flight"}
    pops = {n for n, _ in k.POPS}
    rows = [dict(name=pretty.get(r["name"], r["name"]), base=r["base"], final=r["final"], after=r["after"], ratio=r["ratio"],
                 section="pop" if r["name"] in pops or r["name"] == "whole brain" else "detail") for r in gold["rows"]]
    top = max(rows[1:], key=lambda r: r["ratio"])
    rep = dict(rows=rows, series=list(zip(("whole brain", "touch neurons", "descending"), [np.array(s) for s in gold["series"]])),
               death_at=gold["death_at"], n=gold["n"], top=top, pain=np.array(gold["pain"]), pain_last=gold["pain_last"],
               pain_peak=42.0, pain_max_s=1.5, alive_s=95.0, hits=7, by="being kicked")
    stub = SimpleNamespace(f_small=g.f_small, f_text=g.f_text, f_bold=g.f_bold, f_head=g.f_head, f_title=g.f_title, _text=g._text, report=rep,
                           fly=SimpleNamespace(dead_at=0.0), x3=SimpleNamespace(kc_available=lambda: False))
    surf = pygame.Surface((1280, 760))
    surf.fill(BG)
    k.Game._draw_autopsy(stub, surf, 100.0)
    pygame.image.save(surf, str(out / "challenge_autopsy.png"))
    print("wrote", out)
