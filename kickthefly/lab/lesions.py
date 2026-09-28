"""Hemifield and unilateral lesions: silencing one hemisphere's visual projection neurons or whole hemisphere.

Consequences:
  - Blind-side looming failure: visual projection neurons LPLC2 and LC4 fail to activate the giant fiber (DNp01).
  - Asymmetric steering: turning commands DNa01/DNa02 become imbalanced between left and right.
  - Unilateral duel aiming failure: target-tracking LC10 neurons on the lesioned side cannot drive same-side steering.

Reported strictly as an outcome of the connectome wiring, not as a model of physical tissue injury or trauma.
"""
from __future__ import annotations

import numpy as np

VISUAL_PREFIXES = ("LC", "LPLC", "LPTC", "VS", "HS")


def hemifield_visual_rows(g, side: str = "L") -> np.ndarray:
    """Rows of one hemisphere's visual cell types (LC10, LPLC2, LC4, LPTC/VS/HS, etc.)."""
    types = g.type.astype(str)
    inst = getattr(g, "instance", None)
    inst_str = inst.astype(str) if inst is not None else np.full(g.n, "")
    
    m_vis = np.zeros(g.n, bool)
    for pre in VISUAL_PREFIXES:
        m_vis |= np.char.startswith(types, pre)
    
    target_side = f"_{side.upper()}"
    side_match = np.char.endswith(inst_str, target_side)
    # If some visual neurons lack _L/_R suffix, check soma position if available
    soma = getattr(g, "soma", None)
    if soma is not None and np.any(m_vis & ~side_match):
        midline = float(np.nanmedian(soma[:, 0]))
        has_any_side = np.char.endswith(inst_str, "_L") | np.char.endswith(inst_str, "_R")
        no_side = ~has_any_side & ~np.isnan(soma[:, 0])
        if side.upper() == "L":
            side_match = side_match | (no_side & (soma[:, 0] > midline))
        else:
            side_match = side_match | (no_side & (soma[:, 0] <= midline))
            
    return np.flatnonzero(m_vis & side_match)


def hemisphere_rows(g, side: str = "L", soma: np.ndarray | None = None) -> np.ndarray:
    """Rows of an entire hemisphere (all left or all right neurons)."""
    inst = getattr(g, "instance", None)
    inst_str = inst.astype(str) if inst is not None else np.full(g.n, "")
    target_side = f"_{side.upper()}"
    
    side_match = np.char.endswith(inst_str, target_side)
    has_any_side = np.char.endswith(inst_str, "_L") | np.char.endswith(inst_str, "_R")
    
    soma_arr = soma if soma is not None else getattr(g, "soma", None)
    if soma_arr is None:
        try:
            from kickthefly.core import simcore
            _, _, soma_arr = simcore.pack(brain=getattr(g, "brain_type", "adult"))
        except Exception:
            soma_arr = None

    if soma_arr is not None:
        midline = float(np.nanmedian(soma_arr[:, 0]))
        no_side = ~has_any_side & ~np.isnan(soma_arr[:, 0])
        if side.upper() == "L":
            side_match = side_match | (no_side & (soma_arr[:, 0] > midline))
        else:
            side_match = side_match | (no_side & (soma_arr[:, 0] <= midline))

    return np.flatnonzero(side_match)


def verify_hemifield_consequences(side: str = "L", seed: int = 1000, steps: int = 100) -> dict:
    """Verify the three behavioral consequences of a visual hemifield lesion:
    1. Dodge failure from the blind side (intact side still escapes).
    2. Steering asymmetry in turning commands.
    3. Duel tracking failure when stimulus is in the blind hemifield.
    """
    from kickthefly.core import simcore

    g, _, _ = simcore.pack()
    types = g.type.astype(str)
    inst = g.instance.astype(str)

    blind_side = side.upper()
    intact_side = "R" if blind_side == "L" else "L"

    vis_rows = hemifield_visual_rows(g, blind_side)
    dnp01 = np.flatnonzero(types == "DNp01")

    # 1. Looming Dodge test
    # (a) Unperturbed control dodge from blind side
    br_ctrl = simcore.new_brain(seed=seed, warmup=150)
    simcore.drive(br_ctrl, br_ctrl.sense[("loom", blind_side)], amp=0.5)
    rec_ctrl = simcore.step(br_ctrl, steps, record=dnp01)
    simcore.undrive(br_ctrl, br_ctrl.sense[("loom", blind_side)])
    ctrl_dodge_hz = float(rec_ctrl.mean() / 0.005)

    # (b) Lesioned: stimulus from blind side
    br_lesion = simcore.new_brain(seed=seed, warmup=150)
    br_lesion.set_override(vis_rows, -1)
    simcore.drive(br_lesion, br_lesion.sense[("loom", blind_side)], amp=0.5)
    rec_blind = simcore.step(br_lesion, steps, record=dnp01)
    simcore.undrive(br_lesion, br_lesion.sense[("loom", blind_side)])
    blind_dodge_hz = float(rec_blind.mean() / 0.005)

    # (c) Lesioned: stimulus from intact side
    br_lesion_intact = simcore.new_brain(seed=seed, warmup=150)
    br_lesion_intact.set_override(vis_rows, -1)
    simcore.drive(br_lesion_intact, br_lesion_intact.sense[("loom", intact_side)], amp=0.5)
    rec_intact = simcore.step(br_lesion_intact, steps, record=dnp01)
    simcore.undrive(br_lesion_intact, br_lesion_intact.sense[("loom", intact_side)])
    intact_dodge_hz = float(rec_intact.mean() / 0.005)

    # 2. Steering Asymmetry test
    # Observe baseline turning command rates
    br_steer = simcore.new_brain(seed=seed, warmup=150)
    br_steer.set_override(vis_rows, -1)
    for _ in range(40):
        br_steer._step()
    turn_l_hz = float(br_steer.hz("turn_l"))
    turn_r_hz = float(br_steer.hz("turn_r"))
    steering_bias_hz = turn_r_hz - turn_l_hz

    # 3. Duel Aiming test (LC10 target tracking steering command)
    # Presentation in blind hemifield: calculate steering differential toward target
    br_aim_blind = simcore.new_brain(seed=seed, warmup=150)
    br_aim_blind.set_override(vis_rows, -1)
    br_aim_blind.poke("track", blind_side, 0.5)
    for _ in range(20):
        br_aim_blind._step()
    target_turn_blind_hz = float(br_aim_blind.hz(f"turn_{blind_side.lower()}"))
    opp_turn_blind_hz = float(br_aim_blind.hz(f"turn_{intact_side.lower()}"))
    blind_steer_hz = max(0.0, target_turn_blind_hz - opp_turn_blind_hz)

    # Presentation in intact hemifield
    br_aim_intact = simcore.new_brain(seed=seed, warmup=150)
    br_aim_intact.set_override(vis_rows, -1)
    br_aim_intact.poke("track", intact_side, 0.5)
    for _ in range(20):
        br_aim_intact._step()
    target_turn_intact_hz = float(br_aim_intact.hz(f"turn_{intact_side.lower()}"))
    opp_turn_intact_hz = float(br_aim_intact.hz(f"turn_{blind_side.lower()}"))
    intact_steer_hz = max(0.0, target_turn_intact_hz - opp_turn_intact_hz)

    dodge_failed = blind_dodge_hz < ctrl_dodge_hz * 0.4
    dodge_intact_held = intact_dodge_hz > blind_dodge_hz * 2.0
    duel_aim_broken = blind_steer_hz < intact_steer_hz * 0.6

    return dict(
        side=blind_side,
        intact_side=intact_side,
        silenced_neurons=len(vis_rows),
        looming_dodge=dict(
            control_hz=ctrl_dodge_hz,
            blind_side_hz=blind_dodge_hz,
            intact_side_hz=intact_dodge_hz,
            failed_from_blind_side=bool(dodge_failed),
            intact_side_escapes=bool(dodge_intact_held),
        ),
        steering=dict(
            turn_l_hz=turn_l_hz,
            turn_r_hz=turn_r_hz,
            asymmetry_bias_hz=steering_bias_hz,
        ),
        duel_aiming=dict(
            blind_steer_hz=blind_steer_hz,
            intact_steer_hz=intact_steer_hz,
            aiming_broken_on_blind_side=bool(duel_aim_broken),
        ),
        interpretation="Reported strictly as an outcome of connectome wiring, not a model of physical injury.",
    )
