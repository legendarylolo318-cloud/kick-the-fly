"""Measured personality cards for the flies in play and the pet (3.0 day 4 review, decision A). Pure logic, no pygame.

Until 3.0 day 4 the card the game showed for a fly in play and for the pet was compute_personality_card(seed) with no metrics: three
numbers DRAWN from a random generator seeded by the fly's seed, not read from any brain (and the pet's was drawn from the pet file's
own seed, which is not the seed its brain is built from). The tournament and the race measure the card instead
(lab/tournament.measure_card: three CONNECTOME readouts of the individual this seed builds, at rest). Decision: the game shows only
measured cards. A fly whose card has not been measured says so ("card not measured") and shows no trait words.

  - A card is measured on request, in the background (Esc > Fly arcade > "Measure the flies in play"; Fly.card() in Python; every
    tournament and race measures its own flies), at about 10 s of compute per fly.
  - It is cached in <data>/cache/personality_cards.json under everything that decides it: the seed, the individuality mode and its
    sigma, the five LIF parameters, the brain pack's SHA-256 and CARD_VERSION. measure_card is deterministic for that key (bit-exact
    on cpu, numba and torch-cpu), so a card is never stale and never needs migrating: a changed key is simply not measured yet.
  - Save states carry nothing new (the key is in them already: the seed and the settings), so old saves need no migration. The pet
    file stores the pet fly's measured card with its key; a pre-3.0-day-4 pet file's seed-drawn card is kept under
    "legacy_personality_card" (never deleted, never shown as a measurement).
  - With individuality off every fly has the same brain: what differs between their measured cards is their noise and warm-up state,
    and the card says so (see docs/racing.md for why that is not individuality).

GAME RULE: the trait words and their cut-offs (core/individuality.py). CONNECTOME: the three readouts. Nothing here is a MODEL
PREDICTION.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

CARD_VERSION = 2                   # 2: walk_level_calm is read against the race's fixed reference (speed rule version 2)
LIF_KEYS = ("noise_std", "bias", "target_rate_hz", "ext_gain", "gain_adapt")
NOT_MEASURED = "card not measured"
HOW_TO_MEASURE = "Not measured yet: Esc > Fly arcade > Measure the flies in play (about 10 s of compute per fly, in the background)."
OFF_NOTE = ("individuality off: every fly has the same brain, so what differs between these cards is each seed's noise and warm-up "
            "state, not individuality")
_lock = threading.Lock()


def sigma_of(mode: str, sigma: float | None = None) -> float:
    from kickthefly.core import individuality

    mode = str(mode).lower()
    return 0.0 if mode == "off" else float(sigma if sigma is not None else individuality.SIGMAS.get(mode, 0.0))


def lif_params(params: dict | None) -> dict:
    """The five LIF parameters a card depends on, filled in from the Lab defaults."""
    from kickthefly.lab import lab

    params = params or {}
    return {k: float(params.get(k, lab.DEFAULTS[k])) for k in LIF_KEYS}


def pack_sha(brain: str = "adult") -> str | None:
    from kickthefly.core import replay
    from kickthefly.sim import brainpack

    p = brainpack.find(brain=brain)
    return replay.pack_sha256(p) if p is not None else None


def key(seed: int, mode: str, sigma: float | None = None, params: dict | None = None, pack: str | None = None) -> str:
    """Everything a measured card depends on, hashed: seed, individuality mode and sigma, LIF parameters, pack, card version."""
    mode = str(mode).lower()
    d = dict(v=CARD_VERSION, seed=int(seed), mode=mode, sigma=round(sigma_of(mode, sigma), 9),
             lif={k: round(v, 9) for k, v in lif_params(params).items()}, pack=pack if pack is not None else pack_sha())
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:24]


def cache_file() -> Path:
    from kickthefly.core import paths

    return paths.get().data_dir / "cache" / "personality_cards.json"


def _read() -> dict:
    try:
        d = json.loads(cache_file().read_text(encoding="utf-8"))
        if isinstance(d, dict) and d.get("version") == CARD_VERSION and isinstance(d.get("cards"), dict):
            return d["cards"]
    except (OSError, ValueError):
        pass
    return {}


def lookup(seed: int, mode: str, sigma: float | None = None, params: dict | None = None) -> dict | None:
    """The measured card for this fly, or None if it has not been measured under exactly these settings."""
    k = key(seed, mode, sigma, params)
    c = _read().get(k)
    return c if isinstance(c, dict) and c.get("measured") is True and c.get("card_key") == k else None


def stamp(card: dict, sigma: float | None = None, params: dict | None = None) -> dict:
    """The card with its key and version written into it (measure_card does this; a card loaded from a pet file is checked by it)."""
    c = dict(card)
    c["card_version"] = CARD_VERSION
    c["sigma"] = sigma_of(c.get("mode", "subtle"), sigma)
    c["lif_params"] = lif_params(params)
    c["pack_sha256"] = pack_sha()
    c["card_key"] = key(c["seed"], c.get("mode", "subtle"), sigma, params, c["pack_sha256"])
    if str(c.get("mode", "")).lower() == "off":
        c["note"] = OFF_NOTE
    return c


def store(cards) -> int:
    """Add measured cards (each stamped by measure_card) to the cache. Returns how many were stored. Never raises on a read-only
    or damaged cache: the card is then only missing next time."""
    if isinstance(cards, dict):
        cards = [cards]
    good = [c for c in cards if isinstance(c, dict) and c.get("measured") is True and c.get("card_key")]
    if not good:
        return 0
    with _lock:
        try:
            cur = _read()
            for c in good:
                cur[c["card_key"]] = c
            p = cache_file()
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(dict(version=CARD_VERSION, cards=cur), allow_nan=False, default=float), encoding="utf-8")
            os.replace(tmp, p)
        except (OSError, ValueError, TypeError):
            return 0
    return len(good)


def unmeasured(seed: int, mode: str) -> dict:
    """What a fly shows before its card is measured: no numbers and no trait words, and how to measure it."""
    return dict(seed=int(seed), mode=str(mode), measured=False, title=NOT_MEASURED, summary=HOW_TO_MEASURE, traits=[])


def for_fly(seed: int, mode: str, sigma: float | None = None, params: dict | None = None) -> dict:
    """The card to show for a fly in play: its measured card, or the unmeasured placeholder."""
    try:
        return lookup(seed, mode, sigma, params) or unmeasured(seed, mode)
    except Exception:                                   # a missing pack or folder: never a crash in a fly's constructor
        return unmeasured(seed, mode)


def label(card: dict | None) -> str:
    """Short text for the HUD: the measured card's title marked as measured, or 'card not measured'."""
    if not card or card.get("measured") is not True:
        return NOT_MEASURED
    return f"{card['title']} (measured)"


def check_pet_card(card) -> dict | None:
    """A pet file's card if it is a measured, stamped card (3.0 day 4 review and later); None for anything else."""
    if isinstance(card, dict) and card.get("measured") is True and card.get("card_version") == CARD_VERSION and card.get("card_key"):
        return card
    return None
