"""Tests for Reverse Brain Surgery challenge (Play mode)."""
import pytest
from types import SimpleNamespace
import numpy as np

from conftest import needs_pack


class MockSound:
    def play(self, *a, **k):
        pass


class MockGame:
    def __init__(self, brain=None):
        self.brain = brain
        self.sound = MockSound()
        self.clock = SimpleNamespace(now=10.0)
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


@needs_pack
def test_reverse_surgery_circuits_and_silencing():
    from kickthefly.core import simcore
    from kickthefly.lab.challenges import ReverseSurgery

    br = simcore.new_brain(seed=123, warmup=10)
    game = MockGame(br)

    # Test each curated circuit
    for idx, circuit in enumerate(ReverseSurgery.CIRCUITS):
        rev = ReverseSurgery(game, choice_idx=idx)
        assert rev.target_circuit["id"] == circuit["id"]
        assert len(rev.silenced_rows) > 0
        # Verify brain.override for these rows was set to 0.0
        assert np.all(br.override[rev.silenced_rows] == 0.0)

        # Test requesting hints
        h1 = rev.request_hint()
        assert len(h1) > 0
        assert rev.hints_revealed == 1
        h2 = rev.request_hint()
        assert len(h2) > 0
        assert rev.hints_revealed == 2

        # Test wrong guess
        wrong_id = [c["id"] for c in ReverseSurgery.CIRCUITS if c["id"] != circuit["id"]][0]
        # Copy instance to test wrong guess
        rev_wrong = ReverseSurgery(game, choice_idx=idx)
        assert rev_wrong.make_guess(wrong_id) is False
        assert rev_wrong.state == "revealed"
        assert rev_wrong.guess_result is False
        rev_wrong.end()

        # Test correct guess
        assert rev.make_guess(circuit["id"]) is True
        assert rev.state == "revealed"
        assert rev.guess_result is True

        # End restores brain
        rev.end()
        assert len(rev.silenced_rows) == 0


def test_reverse_surgery_stars():
    from kickthefly.lab.challenges import stars

    assert stars("reverse_surgery", 3) == 3
    assert stars("reverse_surgery", 2) == 2
    assert stars("reverse_surgery", 1) == 1
    assert stars("reverse_surgery", 0) == 0
