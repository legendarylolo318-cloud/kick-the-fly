"""Puppeteer mode (3.1.0 task 10, Play): a puzzle mode in which you cannot hit the fly. You steer it only with the optogenetics laser and with
holding neurons on or off, on the fly's real cell types, to make it do ten things. Esc > Challenges > Puppeteer, in the 3D game and in --2d.

The rules (GAME RULE, all of them; every level also says which circuit solves it, with CONNECTOME where a real neuron type does the work):
  - Nothing of yours touches the fly: every tool is locked except the laser, and a touch that would have poked the fly's touch neurons does nothing
    (Game.puppet_active; the walls, wind and the loom the game sends are not you).
  - You choose a cell type from the palette (the same real types the inspector shows) and an effect: activate or silence it. LASER: aim at the fly and
    click, it is on while you hold. LATCH: the type stays on or off until you latch it again. Anything else (the neuron inspector, brain surgery, Lab) is
    allowed too: the puzzle is working out WHICH neurons, not which button.
  - A level is won when the fly does what it asks (a reaction the brain makes: WALK, TURN R, SONG..., or travels some distance, or survives some looms).
  - Score = actions: one for each laser pulse that reached the fly, one for each latch or surgery change, one for each hint. Par is the fewest actions
    the author's own solution needed; stars: par or better 3, up to par + 2 two, otherwise one. Best scores are kept in scores.json like the other challenges.

How the fly "obeys" is not new: the game already reads descending neurons into moves (kick_the_fly.THRESH: DNp09 above 3x calm is WALK, MDN above 3.8x is
BACK UP, DNa01/02 right minus left above 2.1x is a TURN, DNg02 above 1.58x is TAKE OFF, ps1 above 1.8x is SONG, the antennal pathway into aDN1/aDN2 above 4x is
GROOM, FB6/FB7 above 2x is SLEEP, DNp01 above 4x is DODGE). A level only asks for one of those, so every solution is the connectome doing something to the fly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# --- the palette: real cell types, named as the inspector names them (no labels that give the answer away) -----------------------------------------
# (label, laser/latch target spec as lab/laser.py reads it, what it is in plain words for the hover text)
PALETTE: tuple[tuple[str, str], ...] = (
    ("DNp09", "type:DNp09"), ("MDN", "type:MDN"), ("DNa01+DNa02 (right)", "side:R:DNa01,DNa02"), ("DNa01+DNa02 (left)", "side:L:DNa01,DNa02"),
    ("DNg02", "prefix:DNg02"), ("pIP10", "type:pIP10"), ("JO-C + JO-E", "prefix:JO-C,JO-E"), ("DNp01", "type:DNp01"), ("LPLC2 + LC4", "type:LPLC2,LC4"),
    ("FB6 + FB7", "prefix:FB6,FB7"), ("aDN1 + aDN2", "type:DNg62,DNge078"), ("MN9", "type:MN9"), ("PPL1", "prefix:PPL1"), ("PAM", "prefix:PAM"),
    ("Kenyon cells", "prefix:KC"), ("LC10", "prefix:LC10"),
)
SPEC = dict((label, spec) for label, spec in PALETTE)
LABEL_OF = dict((spec, label) for label, spec in PALETTE)

LOOM_EVERY_S, LOOM_FOR_S, LOOM_WINDOW_S = 3.5, 0.6, 1.6        # GAME RULE: the shadow the survive-looms level sends
PAR_SLACK = 2                                                   # two stars up to par + this


@dataclass(frozen=True)
class Level:
    id: str
    title: str
    brief: str                       # what to do, in one sentence
    goal: dict                       # see Tracker
    par: int
    solution: tuple[dict, ...]       # the author's own solution: {"how": "laser"|"latch", "target": spec, "mode": "activate"|"silence", "pulses": n}
    why: tuple[tuple[str, str], ...]  # ("CONNECTOME"|"GAME RULE", text): which circuit solves it and which parts are rules
    start_x: float = 0.0             # metres from the middle of the floor
    facing: int = 1                  # +1 right / forward along +x, -1 the other way
    hint: str = ""


LEVELS: tuple[Level, ...] = (
    Level("first_steps", "First steps", "Make the fly walk.", dict(kind="event", event="WALK"), 1,
          ({"how": "laser", "target": "type:DNp09", "mode": "activate", "pulses": 1},),
          (("CONNECTOME", "DNp09 (P9) is the descending neuron that starts forward walking in flies (Bidaye et al. 2020); here it is a real descending cell type the connectome "
                          "names and the brain reads out."),
           ("GAME RULE", "The fly walks for 1.2 s when DNp09 runs above 3x its calm rate (kick_the_fly.THRESH['walk']); the walk itself is game physics.")),
          start_x=-0.9, hint="A single descending neuron type starts walking."),
    Level("about_face", "About face", "Make the fly turn to its right.", dict(kind="event", event="TURN R"), 1,
          ({"how": "laser", "target": "side:R:DNa01,DNa02", "mode": "activate", "pulses": 1},),
          (("CONNECTOME", "DNa01 and DNa02 are the steering descending neurons; the fly turns toward the side whose pair fires more (right minus left). The validated optomotor "
                          "pathway reads the same neurons (docs/validation.md)."),
           ("GAME RULE", "A turn happens when right minus left is above 2.1x calm; the left and the right neurons are separate cells, so choose a side.")),
          start_x=0.0, facing=-1, hint="Steering neurons come in left and right pairs."),
    Level("long_walk", "The long walk", "Get the fly 0.9 m across the floor.", dict(kind="travel", m=0.9), 1,
          ({"how": "latch", "target": "type:DNp09", "mode": "activate", "pulses": 1},),
          (("CONNECTOME", "One laser pulse on DNp09 buys one 1.2 s step. Holding the same neurons on (a latch) keeps the walking command running, so the fly keeps "
                          "stepping and crosses the floor."),
           ("GAME RULE", "Each WALK step is game physics; the distance is measured along the way the fly faces when you start.")),
          start_x=-1.4, hint="One step is short. What keeps a command on?"),
    Level("reverse_gear", "Reverse gear", "Make the fly back up half a metre.", dict(kind="travel", m=-0.5), 1,
          ({"how": "latch", "target": "type:MDN", "mode": "activate", "pulses": 1},),
          (("CONNECTOME", "MDN, the moonwalker descending neuron, makes flies walk backward (Bidaye et al. 2014). In this model its activity does NOT reach the leg motor "
                          "neurons (docs/validation.md: MDN FAILS), but the game's BACK UP reaction reads MDN directly, so here it still works: a game rule standing in "
                          "for a circuit the model does not reproduce."),
           ("GAME RULE", "BACK UP is MDN above 3.8x calm for 0.8 s; the backward walk is game physics.")),
          start_x=0.5, hint="There is a neuron named for walking backward."),
    Level("lift_off", "Lift off", "Get the fly airborne.", dict(kind="event", event="TAKE OFF"), 1,
          ({"how": "laser", "target": "prefix:DNg02", "mode": "activate", "pulses": 1},),
          (("CONNECTOME", "DNg02 is a family of descending neurons for wing power in flight; the game reads their level as the wish to fly."),
           ("GAME RULE", "TAKE OFF is DNg02 above 1.58x calm; the flight (a hover and wander) is game physics.")),
          start_x=0.0, hint="Flight power comes down a family of descending neurons."),
    Level("serenade", "Serenade", "Make the fly sing.", dict(kind="event", event="SONG"), 1,
          ({"how": "latch", "target": "type:pIP10", "mode": "activate", "pulses": 1},),
          (("CONNECTOME", "pIP10 is the male-specific command neuron for courtship song (von Philipsborn et al. 2011); in the connectome it reaches the ps1 wing motor neurons "
                          "through VNC interneurons, and the validation test for that pathway PASSES (x1.93)."),
           ("GAME RULE", "SONG is the two ps1 wing motor neurons above 1.8x calm, read over about a second, so the neurons need a moment of steady drive: latch them.")),
          start_x=0.0, hint="A command neuron specific to males."),
    Level("clean_antennae", "Clean antennae", "Make the fly groom.", dict(kind="event", event="GROOM"), 1,
          ({"how": "laser", "target": "prefix:JO-C,JO-E", "mode": "activate", "pulses": 1},),
          (("CONNECTOME", "JO-C and JO-E are the Johnston's organ neurons that sense antennal movement; they excite the antennal grooming command neurons aDN1/aDN2 "
                          "(Hampel et al. 2015; the validated pathway x4.87)."),
           ("GAME RULE", "GROOM needs the antennal neurons active AND aDN1/aDN2 above 4x calm, so driving the sensors that feed them does both at once.")),
          start_x=0.0, hint="What does an antenna feel with?"),
    Level("sleep_tight", "Sleep tight", "Put the fly to sleep.", dict(kind="event", event="SLEEP"), 1,
          ({"how": "laser", "target": "prefix:FB6,FB7", "mode": "activate", "pulses": 1},),
          (("CONNECTOME", "The dorsal fan-shaped body layers FB6 and FB7 hold sleep-promoting neurons in the fly (Donlea et al. 2018)."),
           ("GAME RULE", "SLEEP is FB6/FB7 firing above 2x calm; the fly then rests for 2 s (no spontaneous walking or take-off). Nothing in normal play drives them that far.")),
          start_x=0.0, hint="Sleep is in the central complex."),
    Level("nerves_of_steel", "Nerves of steel", "Let three looming shadows pass without the fly dodging.", dict(kind="survive_loom", count=3, forbid="DODGE"), 1,
          ({"how": "latch", "target": "type:DNp01", "mode": "silence", "pulses": 1},),
          (("CONNECTOME", "A shadow that grows in the fly's view drives its looming detectors LPLC2 and LC4, which excite the giant fiber DNp01 (von Reyn et al. 2014; Ache et al. 2019; "
                          "validated x11.8). Silence DNp01, or the detectors, and the dodge command never comes. Silencing works because the giant fiber is the only exit."),
           ("GAME RULE", "The shadows (0.6 s every 3.5 s) are sent through the game's own looming drive onto LPLC2/LC4; DODGE is DNp01 above 4x calm.")),
          start_x=0.0, hint="The dodge comes from one neuron. Switch something OFF."),
    Level("detour", "The detour", "Turn the fly around, then walk it half a metre the way it now faces.",
          dict(kind="sequence", steps=[dict(kind="event", event="TURN R"), dict(kind="travel", m=0.5)]), 2,
          ({"how": "laser", "target": "side:R:DNa01,DNa02", "mode": "activate", "pulses": 1}, {"how": "latch", "target": "type:DNp09", "mode": "activate", "pulses": 1}),
          (("CONNECTOME", "Two circuits in order: the steering pair (a turn) and the walking command (distance), the first one-shot and the second held."),
           ("GAME RULE", "The distance is measured from where the turn happened, along the way the fly then faces.")),
          start_x=-1.2, facing=-1, hint="Steer first, then move."),
)
BY_ID = {lv.id: lv for lv in LEVELS}


# --- the goal logic, with no game in it -------------------------------------------------------------------------------------------------------
class Tracker:
    """Tracks one level's goal from what the game reports: reactions by name, the fly's position and heading each frame, and the time. Pure logic."""

    def __init__(self, goal: dict):
        self.goal = goal
        self.steps = goal["steps"] if goal["kind"] == "sequence" else [goal]
        self.stage = 0
        self.done = False
        self.t0 = 0.0
        self.origin: tuple[float, float, float, float] | None = None       # position and heading when the stage began
        self._last = (0.0, 0.0)
        self.progress = 0.0
        self.message = ""
        # survive_loom
        self.looms_survived = 0
        self.last_loom: float | None = None
        self.next_loom = None
        self.dodged = False

    @property
    def cur(self) -> dict:
        return self.steps[min(self.stage, len(self.steps) - 1)]

    def start(self, t: float) -> None:
        self.t0 = t
        self.next_loom = t + 2.0

    def _advance(self, t: float) -> None:
        self.stage += 1
        self.origin = None
        if self.stage >= len(self.steps):
            self.done = True

    def event(self, name: str, t: float) -> None:
        """A reaction the brain made (the first words of the game's note: WALK, BACK UP, TURN R, TAKE OFF, SONG, GROOM, SLEEP, DODGE...)."""
        if self.done:
            return
        g = self.cur
        if g["kind"] == "event" and name == g["event"]:
            self._advance(t)
        elif g["kind"] == "survive_loom" and name == g["forbid"] and self.last_loom is not None and t - self.last_loom <= LOOM_WINDOW_S + 0.4:
            self.dodged = True

    def position(self, x: float, z: float, hx: float, hz: float, t: float, airborne: bool = False) -> None:
        """Where the fly is (metres) and the unit vector it faces. Distance is what it has walked along the way it faces: each frame's step counts
        forward if it moved the way it faces and backward if it moved against it, so backing up is negative, walking is positive, and a wall that
        turns it round (a game rule) does not turn a walk into a backward distance, or the other way. Flying is not walking: while it is airborne
        no distance is counted."""
        if self.done:
            return
        g = self.cur
        if g["kind"] != "travel":
            self.origin = None
            return
        n = math.hypot(hx, hz) or 1.0
        hx, hz = hx / n, hz / n
        if self.origin is None:
            self.origin = (x, z, hx, hz)                    # where the stage began, and the way it faced (the goal post is drawn from it)
            self._last = (x, z)
            self.progress = 0.0
            return
        lx, lz = self._last
        step = (x - lx) * hx + (z - lz) * hz
        if abs(step) < 0.5 and not airborne:                 # a jump of half a metre in one frame is a teleport, not a step
            self.progress += step
        self._last = (x, z)
        if (g["m"] >= 0 and self.progress >= g["m"]) or (g["m"] < 0 and self.progress <= g["m"]):
            self._advance(t)

    def loom_due(self, t: float) -> bool:
        g = self.cur
        return g["kind"] == "survive_loom" and not self.done and self.next_loom is not None and t >= self.next_loom and self.last_loom is None

    def loom_fired(self, t: float) -> None:
        self.last_loom = t
        self.dodged = False

    def tick(self, t: float) -> None:
        """Settle a loom once its window has passed: survived, or the count starts again."""
        g = self.cur
        if g["kind"] != "survive_loom" or self.done or self.last_loom is None:
            return
        if self.dodged:
            self.looms_survived, self.last_loom, self.next_loom, self.dodged = 0, None, t + LOOM_EVERY_S, False
            self.message = "It saw that one. The count starts again."
        elif t - self.last_loom >= LOOM_WINDOW_S:
            self.looms_survived += 1
            self.last_loom, self.next_loom = None, t + LOOM_EVERY_S - LOOM_WINDOW_S
            self.message = f"It did not flinch ({self.looms_survived}/{g['count']})."
            if self.looms_survived >= g["count"]:
                self._advance(t)

    def describe(self) -> str:
        g = self.cur
        if g["kind"] == "travel":
            return f"{abs(self.progress):.2f} / {abs(g['m']):.2f} m" + (" back" if g["m"] < 0 else "")
        if g["kind"] == "survive_loom":
            return f"{self.looms_survived} / {g['count']} shadows passed"
        if len(self.steps) > 1:
            return f"step {min(self.stage + 1, len(self.steps))} of {len(self.steps)}"
        return ""


def stars_for(actions: int, par: int) -> int:
    return 3 if actions <= par else 2 if actions <= par + PAR_SLACK else 1


def score_key(level_id: str) -> str:
    return f"puppet_{level_id}"


def completed_levels(scores: dict) -> int:
    return sum(1 for lv in LEVELS if score_key(lv.id) in scores)
