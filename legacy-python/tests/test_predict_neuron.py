"""Tests for Predict-the-Neuron challenge (Play mode).

Verifies that the minigame is strictly restricted to descending motor readouts
(jump, run, kick, back up, take off) and correctly scores player predictions.
"""
from types import SimpleNamespace
import numpy as np
import pytest

from conftest import needs_pack


class MockSound:
    def play(self, *a, **k):
        pass


class MockFly:
    def __init__(self):
        self.p = np.zeros((10, 2), dtype=float)
        self.facing = 1.0
        self.walk_until = 0.0
        self.back_until = 0.0
        self.flail_until = 0.0
        self.escape_until = 0.0
        self.run = False

    def escape(self, now, seconds=1.0, wander=False):
        self.escape_until = now + seconds

    def walk(self, *a, **k):
        pass


class MockSlot:
    def __init__(self):
        self.fly = MockFly()
        self.threat_x = 0.0


class MockGame:
    def __init__(self, brain=None):
        self.brain = brain
        self.sound = MockSound()
        self.clock = SimpleNamespace(now=10.0)
        self.flies = [MockSlot()]
        self.focus = 0
        self.challenge = None
        self.f_bold = None
        self.f_small = None
        self.f_head = None
        self.f_text = None

    def start_challenge(self, key):
        from kickthefly.lab import challenges
        self.challenge = challenges.CLASSES[key](self)

    def _text(self, *a, **k):
        pass


def test_predict_neuron_scope_restricted_to_descending_motor():
    from kickthefly.lab.challenges import PredictNeuron

    assert len(PredictNeuron.READOUTS) == 5
    readout_ids = {r["id"] for r in PredictNeuron.READOUTS}
    assert readout_ids == {"jump", "run", "kick", "back", "takeoff"}
    for r in PredictNeuron.READOUTS:
        assert r["target"] in ("jump", "run", "kick", "back", "fly")


@needs_pack
def test_predict_neuron_round_flow_and_scoring():
    from kickthefly.core import simcore
    from kickthefly.lab.challenges import PredictNeuron

    br = simcore.new_brain(seed=456, warmup=10)
    game = MockGame(br)
    game.start_challenge("predict")
    pn = game.challenge

    # Ready -> start round 1
    pn.update(10.0)
    assert pn.state == "cue"
    assert pn.round_idx == 0
    target_id = pn.current_readout["id"]

    # Predict correctly
    assert pn.predict(target_id, 10.5) is True
    assert pn.score == 1
    assert pn.state == "result"

    # Advance timer past result
    pn.update(12.0)
    assert pn.state == "cue"
    assert pn.round_idx == 1

    # Predict incorrectly
    wrong_id = "kick" if target_id != "kick" else "jump"
    assert pn.predict(wrong_id, 12.5) is (wrong_id == pn.current_readout["id"])
    pn.end()


def test_predict_neuron_stars():
    from kickthefly.lab.challenges import stars

    assert stars("predict", 10) == 3
    assert stars("predict", 8) == 3
    assert stars("predict", 7) == 2
    assert stars("predict", 6) == 2
    assert stars("predict", 5) == 1
    assert stars("predict", 4) == 1
    assert stars("predict", 2) == 0
