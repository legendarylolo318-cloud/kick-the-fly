"""Tests for brain stethoscope spike sonification."""

import numpy as np
import pytest
from kickthefly.core import config
from kickthefly.game.kick_the_fly import Sound
from kickthefly.core import simcore


def test_stethoscope_config_settings():
    c = config.Config(None)
    assert c["audio.stethoscope_enabled"] is False
    assert c["audio.stethoscope_vol"] == 0.5
    assert c["audio.stethoscope_target"] == "mushroom_body"

    # Action binding
    assert ("stethoscope", "Brain stethoscope", "k") in config.ACTIONS

    # Coercion
    c.set("audio.stethoscope_enabled", True)
    assert c["audio.stethoscope_enabled"] is True
    c.set("audio.stethoscope_vol", 0.8)
    assert c["audio.stethoscope_vol"] == 0.8
    c.set("audio.stethoscope_target", "antennal_lobe")
    assert c["audio.stethoscope_target"] == "antennal_lobe"


def test_sound_spike_click_synthesis():
    snd = Sound()
    # Check that the synthesized spike click sounds exist in fx dictionary
    for name in ["spike_click1", "spike_click2", "spike_click3", "spike_click_many"]:
        if snd.ok:
            assert name in snd.fx

    # Calling play_spike_click shouldn't raise even if muted or 0 spikes
    snd.play_spike_click(0)
    snd.play_spike_click(-5)
    snd.play_spike_click(1)
    snd.play_spike_click(5)
    snd.play_spike_click(20)


def test_brain_stethoscope_spike_accumulation():
    brain = simcore.new_brain(seed=0, warmup=10)
    # Probe a small set of neurons (e.g. first 500 neurons)
    probe_idx = np.arange(500, dtype=np.int32)
    brain.set_stethoscope_indices(probe_idx)

    assert brain.take_stethoscope_spikes() == 0

    # Advance several steps and verify spike counting
    total = 0
    for _ in range(30):
        brain._step()
    spikes = brain.take_stethoscope_spikes()
    assert isinstance(spikes, int)
    assert spikes >= 0

    # Once taken, counter resets to 0
    assert brain.take_stethoscope_spikes() == 0

    # Clearing probe indices stops accumulation
    brain.set_stethoscope_indices(None)
    for _ in range(10):
        brain._step()
    assert brain.take_stethoscope_spikes() == 0
