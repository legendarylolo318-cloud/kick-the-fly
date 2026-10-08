"""Tests for hemifield and unilateral lesions: blind-side dodge failure, steering asymmetry, duel aiming."""

import numpy as np
import pytest

from conftest import needs_pack


@needs_pack
def test_hemifield_row_selection():
    from kickthefly.core import simcore
    from kickthefly.lab import lesions

    g, _, soma = simcore.pack()
    vis_l = lesions.hemifield_visual_rows(g, "L")
    vis_r = lesions.hemifield_visual_rows(g, "R")
    assert len(vis_l) > 1000
    assert len(vis_r) > 1000
    assert len(set(vis_l).intersection(set(vis_r))) == 0, "left and right visual rows must be disjoint"

    hemi_l = lesions.hemisphere_rows(g, "L", soma=soma)
    hemi_r = lesions.hemisphere_rows(g, "R", soma=soma)
    assert len(hemi_l) > 70000
    assert len(hemi_r) > 70000
    assert len(set(hemi_l).intersection(set(hemi_r))) == 0, "hemispheres must be disjoint"


@needs_pack
def test_hemifield_consequences_verified():
    from kickthefly.lab import lesions

    res = lesions.verify_hemifield_consequences(side="L", seed=1000, steps=100)

    # 1. Blind-side looming dodge failure
    loom = res["looming_dodge"]
    assert loom["failed_from_blind_side"], (
        f"Expected blind-side dodge to fail: blind={loom['blind_side_hz']} vs ctrl={loom['control_hz']}"
    )
    assert loom["intact_side_escapes"], (
        f"Expected intact side to still escape: intact={loom['intact_side_hz']} vs blind={loom['blind_side_hz']}"
    )

    # 2. Asymmetric steering
    steer = res["steering"]
    assert "asymmetry_bias_hz" in steer

    # 3. Duel aiming broken on blind side
    aim = res["duel_aiming"]
    assert aim["aiming_broken_on_blind_side"], (
        f"Expected duel aiming to break on blind side: blind={aim['blind_steer_hz']} vs intact={aim['intact_steer_hz']}"
    )

    # 4. Model interpretation label
    assert "not a model of physical injury" in res["interpretation"]


@needs_pack
def test_game_surgery_hemifield_entries():
    from kickthefly.game import kick_the_fly as k

    labels = [label for label, _ in k.SURGERY]
    assert any("Left visual hemifield" in l for l in labels)
    assert any("Right visual hemifield" in l for l in labels)
    assert any("Left hemisphere" in l for l in labels)
    assert any("Right hemisphere" in l for l in labels)
