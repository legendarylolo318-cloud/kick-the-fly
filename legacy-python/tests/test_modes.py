"""Play/Lab mode, real-vs-rule tagging and Lab parameter plumbing (no brain pack needed)."""
from types import SimpleNamespace

import numpy as np

from kickthefly.core import config
from kickthefly.game import kick_the_fly as k
from kickthefly.lab import lab


def test_mode_defaults_to_play_and_tags_follow_mode():
    c = config.Config(None)
    assert not c.lab and not c.tags_on()
    c.set("brain.mode", "lab")
    assert c.tags_on()
    c.set("brain.real_vs_rule", "off")
    assert not c.tags_on()
    c.set("brain.mode", "play")
    c.set("brain.real_vs_rule", "on")
    assert c.tags_on()


def test_reaction_sources():
    assert k.reaction_source("DODGE    giant fiber DNp01 x7.2") == "real"
    assert k.reaction_source("BACK UP  MDN x4.0") == "real"
    assert k.reaction_source("TURN R   DNa01/02 R-L +2.3") == "real"
    assert k.reaction_source("EATING   sugar: taste + PAM reward") == "rule"
    assert k.reaction_source("AVOID    remembers the swatter (0.50)") == "rule"
    assert k.reaction_source("TO LIGHT flies to the lamp") == "rule"
    assert k.reaction_source("ARENA    fan") is None
    assert k.reaction_source("SURGERY  cleared") is None
    assert k.POPUP_SOURCE["DODGE!"] == "real" and k.POPUP_SOURCE["NOPE!"] == "rule"


def test_lab_params_apply_to_a_sim():
    from kickthefly.sim.connectome.sim import LIFParams
    p = LIFParams()
    rng = np.random.default_rng(0)
    sim = SimpleNamespace(p=p, _noise=rng.standard_normal(1000).astype(np.float32) * p.noise_std, rng=rng,
                          target_p=p.target_rate_hz * p.dt_ms / 1000)
    std0 = float(sim._noise.std())
    lab.apply_to_sim(sim, {"noise_std": 0.1, "target_rate_hz": 10.0})
    assert sim.p.noise_std == 0.1 and abs(float(sim._noise.std()) - 2 * std0) < 1e-4
    assert abs(sim.target_p - 0.05) < 1e-12
    assert lab.modified({"noise_std": 0.1, "bias": 0.2}) == {"noise_std": 0.1}


def test_rule_params_change_thresholds():
    old = dict(k.THRESH), k.LOOM_MIN
    try:
        lab.apply_rules({"thresh.escape": 6.5, "loom_min": 2.0})
        assert k.THRESH["escape"] == 6.5 and k.LOOM_MIN == 2.0
    finally:
        k.THRESH.update(old[0])
        k.LOOM_MIN = old[1]
