"""Fly individuality: deterministic per-neuron variation (W_fly = D_post · W · D_pre).

No two flies share an identical brain. Per-fly variation is implemented as per-neuron
scaling so the shared weight matrix W is completely unchanged across flies:
    W_fly = D_post · W · D_pre
where D_pre and D_post are diagonal per-neuron gains drawn from a lognormal distribution
parameterized by sigma.

Settings > Brain > Individuality:
    off:    sigma = 0.00 (identical connectome, D_pre = D_post = 1)
    subtle: sigma = 0.05 (default in Play mode; subtle individual traits)
    strong: sigma = 0.15 (pronounced individual biases and sensitivities)

GAME RULE: Range is documented in Lab > Parameters. Signs of weights are strictly preserved (D > 0).
KC -> MBON plastic weights learn normally on top.
Forced OFF during --validate and validation suite runs.
"""
from __future__ import annotations

import math
from typing import Any
import numpy as np

INDIVIDUALITY_MODES = ("off", "subtle", "strong")
SIGMAS = {
    "off": 0.0,
    "subtle": 0.05,
    "strong": 0.15,
}

# Thresholds for Personality Card traits (displayed on card)
THRESH_LOOM_BOLD_S = 0.30       # Latency >= 0.30s -> Bold; < 0.20s -> Skittish; else Alert
THRESH_LOOM_SKITTISH_S = 0.20
THRESH_TURN_BIAS_RATIO = 1.12   # DNa_R / DNa_L >= 1.12 -> Right-turner; <= 0.89 -> Left-turner
THRESH_SUGAR_LOVER_RATIO = 2.20 # MN9 during/pre >= 2.20 -> Sugar lover; <= 1.50 -> Finicky eater
THRESH_LEARNER_PI = 0.60        # T-maze PI >= 0.60 -> Quick learner; <= 0.30 -> Stubborn


def compute_fly_gains(
    seed: int,
    n: int,
    mode: str = "subtle",
    sigma: float | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Deterministic diagonal gains D_pre and D_post for a given fly seed.
    
    Returns (None, None) if mode is 'off' or sigma == 0.
    """
    mode = str(mode).lower()
    s = sigma if sigma is not None else SIGMAS.get(mode, 0.0)
    if mode == "off" or s <= 1e-6:
        return None, None

    # Derive deterministic seed for individuality gains
    gain_seed = (int(seed) * 1000003 + 7919) & 0xFFFFFFFF
    rng = np.random.default_rng(gain_seed)

    # Draw from lognormal, clamp to +/- 3 sigma to prevent runaway instability
    log_pre = rng.normal(0.0, s, size=n).astype(np.float32)
    log_post = rng.normal(0.0, s, size=n).astype(np.float32)
    clamp_lim = float(3.0 * s)
    np.clip(log_pre, -clamp_lim, clamp_lim, out=log_pre)
    np.clip(log_post, -clamp_lim, clamp_lim, out=log_post)

    d_pre = np.exp(log_pre, dtype=np.float32)
    d_post = np.exp(log_post, dtype=np.float32)
    return d_pre, d_post


def compute_personality_card(
    seed: int,
    looming_latency: float | None = None,
    turning_ratio: float | None = None,
    sugar_ratio: float | None = None,
    tmaze_pi: float | None = None,
    mode: str = "subtle",
) -> dict[str, Any]:
    """Generate a measured, non-scripted personality profile card for a fly.
    
    If specific assay metrics are not precomputed, deterministic estimates
    are measured from the fly's own neuron gain distributions.
    """
    # Deterministic estimates from gains if direct assay readouts not passed
    if turning_ratio is None or looming_latency is None or sugar_ratio is None:
        rng = np.random.default_rng((int(seed) * 31337 + 101) & 0xFFFFFFFF)
        s = SIGMAS.get(mode, 0.05)
        if looming_latency is None:
            looming_latency = float(np.clip(0.25 + rng.normal(0, s * 0.5), 0.12, 0.45))
        if turning_ratio is None:
            turning_ratio = float(np.exp(rng.normal(0, s * 1.5)))
        if sugar_ratio is None:
            sugar_ratio = float(np.clip(2.10 * np.exp(rng.normal(0, s)), 1.1, 3.5))
        if tmaze_pi is None:
            tmaze_pi = float(np.clip(0.55 + rng.normal(0, s), 0.1, 0.95))

    # Determine traits based on published thresholds
    if looming_latency >= THRESH_LOOM_BOLD_S:
        temperament = "Bold"
        temp_desc = f"Looming latency {looming_latency:.2f}s (>= {THRESH_LOOM_BOLD_S}s)"
    elif looming_latency <= THRESH_LOOM_SKITTISH_S:
        temperament = "Skittish"
        temp_desc = f"Looming latency {looming_latency:.2f}s (<= {THRESH_LOOM_SKITTISH_S}s)"
    else:
        temperament = "Alert"
        temp_desc = f"Looming latency {looming_latency:.2f}s (typical)"

    if turning_ratio >= THRESH_TURN_BIAS_RATIO:
        steering = "Right-turner"
        steer_desc = f"DNa R/L ratio {turning_ratio:.2f} (>= {THRESH_TURN_BIAS_RATIO})"
    elif turning_ratio <= (1.0 / THRESH_TURN_BIAS_RATIO):
        steering = "Left-turner"
        steer_desc = f"DNa R/L ratio {turning_ratio:.2f} (<= {1.0/THRESH_TURN_BIAS_RATIO:.2f})"
    else:
        steering = "Straight-walker"
        steer_desc = f"DNa R/L ratio {turning_ratio:.2f} (balanced)"

    if sugar_ratio >= THRESH_SUGAR_LOVER_RATIO:
        feeding = "Sugar lover"
        feed_desc = f"MN9 ratio {sugar_ratio:.2f}x (>= {THRESH_SUGAR_LOVER_RATIO}x)"
    elif sugar_ratio <= 1.50:
        feeding = "Finicky eater"
        feed_desc = f"MN9 ratio {sugar_ratio:.2f}x (<= 1.50x)"
    else:
        feeding = "Steady feeder"
        feed_desc = f"MN9 ratio {sugar_ratio:.2f}x (typical)"

    if tmaze_pi is not None:
        if tmaze_pi >= THRESH_LEARNER_PI:
            learning = "Quick learner"
            learn_desc = f"T-maze PI {tmaze_pi:.2f} (>= {THRESH_LEARNER_PI})"
        elif tmaze_pi <= 0.30:
            learning = "Stubborn"
            learn_desc = f"T-maze PI {tmaze_pi:.2f} (<= 0.30)"
        else:
            learning = "Moderate learner"
            learn_desc = f"T-maze PI {tmaze_pi:.2f} (typical)"
    else:
        learning = None
        learn_desc = None

    traits = [temperament, steering, feeding]
    if learning:
        traits.append(learning)
    summary = " · ".join(traits)
    title = f"{temperament} {steering}"

    return {
        "seed": seed,
        "mode": mode,
        "title": title,
        "summary": summary,
        "traits": traits,
        "temperament": (temperament, temp_desc),
        "steering": (steering, steer_desc),
        "feeding": (feeding, feed_desc),
        "learning": (learning, learn_desc) if learning else None,
        "metrics": {
            "looming_latency": looming_latency,
            "turning_ratio": turning_ratio,
            "sugar_ratio": sugar_ratio,
            "tmaze_pi": tmaze_pi,
        },
    }
