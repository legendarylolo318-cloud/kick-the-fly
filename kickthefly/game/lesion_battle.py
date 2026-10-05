"""Lesion battles (3.1.0 task 12, Fly arcade): two players each perform surgery on their own fly, within a budget, then the flies fight.

Local and hot-seat (one screen: the second player's choices are made while the first player's are hidden, and both are revealed when the fight starts). The fight is the
Fly arcade's own duel (game/flyduel.py): both sides are brains stepped together, nothing about who wins is scripted, and the surgery is the same silencing and stimulating
the game's brain surgery does (a constant current: kick_the_fly.SURGERY_CURRENT).

Tags: the surgery's effect on the neurons is CONNECTOME (what the wiring then does with it); the budget, the size cap, the arena, the blaster, the number of rounds and who may see
what are GAME RULES; who wins is a MODEL PREDICTION about this model (and about two flies of the same seeds), not about real flies.

The budget is a GAME RULE: at most BUDGET picks per fly, each a cell type (silenced or stimulated) of at most MAX_TYPE_NEURONS neurons, so "switch off the whole optic lobe" is not an
option and a pick costs the same whether the type has two neurons or four hundred. A loadout is saved by name, and shared as an ordinary surgery code (`KTF1-SRG-...`, core/sharecode.py):
a lesion loadout is exactly a surgery that names only cell types, so the existing share, preview and import machinery carries it and nothing new is added to the format.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BUDGET = 3
MAX_TYPE_NEURONS = 500
MAX_SAVED = 20
ROUNDS = (1, 3)
SILENCE, STIMULATE = -1, 1
MODE_WORD = {SILENCE: "silenced", STIMULATE: "stimulated"}
TAG_TEXT = ("CONNECTOME for what the surgery does to the neurons, GAME RULE for the budget, the size cap and the arena, "
            "MODEL PREDICTION for who wins")

# the types the duel itself reads (flyduel): a hint for the picker, never an answer
SUGGESTED = ("DNp35", "DNpe052", "DNa01", "DNa02", "DNp01", "DNp09", "LC10a", "LC10b", "LC11", "LPLC2", "LC4", "MDN", "DNg02_a", "PPL1", "PAM")


class LoadoutError(ValueError):
    """A loadout that cannot be used; str(e) is the reason shown to the player."""


@dataclass
class Loadout:
    name: str = "my fly"
    picks: dict[str, int] = field(default_factory=dict)          # cell type -> -1 silenced / 1 stimulated

    def add(self, cell_type: str, mode: int, sizes: dict[str, int] | None = None, budget: int = BUDGET) -> None:
        """Add (or change) a pick, refusing what the rules do not allow."""
        t = str(cell_type).strip()
        if mode not in (SILENCE, STIMULATE):
            raise LoadoutError("a pick is silenced (-1) or stimulated (1)")
        if sizes is not None:
            if t not in sizes:
                raise LoadoutError(f"{t!r} is not a cell type of this brain")
            if sizes[t] > MAX_TYPE_NEURONS:
                raise LoadoutError(f"{t} has {sizes[t]:,} neurons; a pick may have at most {MAX_TYPE_NEURONS}")
        if t not in self.picks and len(self.picks) >= budget:
            raise LoadoutError(f"the budget is {budget} picks: take one out first")
        self.picks[t] = mode

    def remove(self, cell_type: str) -> None:
        self.picks.pop(cell_type, None)

    def used(self) -> int:
        return len(self.picks)

    def problems(self, sizes: dict[str, int] | None = None, budget: int = BUDGET) -> list[str]:
        out = []
        if len(self.picks) > budget:
            out.append(f"{len(self.picks)} picks is over the budget of {budget}")
        for t, m in self.picks.items():
            if m not in (SILENCE, STIMULATE):
                out.append(f"{t}: not silenced or stimulated")
            if sizes is not None:
                if t not in sizes:
                    out.append(f"{t} is not a cell type of this brain")
                elif sizes[t] > MAX_TYPE_NEURONS:
                    out.append(f"{t} has {sizes[t]:,} neurons (over {MAX_TYPE_NEURONS})")
        return out

    def neurons(self, sizes: dict[str, int]) -> int:
        return sum(sizes.get(t, 0) for t in self.picks)

    def describe(self) -> str:
        return ", ".join(f"{t} {MODE_WORD[m]}" for t, m in self.picks.items()) or "no surgery"

    # --- sharing: an ordinary surgery code ------------------------------------------------------------------------------------------
    def to_surgery_payload(self) -> dict:
        return {"types": {t: int(m) for t, m in self.picks.items()}}

    @classmethod
    def from_surgery_payload(cls, payload: dict, name: str = "shared") -> "Loadout":
        if payload.get("groups"):
            raise LoadoutError("a lesion loadout names cell types only; this surgery code also switches whole groups")
        types = payload.get("types") or {}
        if not types:
            raise LoadoutError("this surgery code names no cell types")
        return cls(name, {str(t): int(m) for t, m in types.items()})

    def code(self) -> str:
        """The share code (`KTF1-SRG-...`): the same one Esc > Share makes for a surgery with these cell types."""
        from kickthefly.core import sharecode

        if not self.picks:
            raise LoadoutError("nothing to share: add a pick first")
        return sharecode.encode("surgery", self.to_surgery_payload())

    @classmethod
    def from_code(cls, text: str, sizes: dict[str, int] | None = None, budget: int = BUDGET, name: str = "shared") -> "Loadout":
        from kickthefly.core import sharecode

        try:
            c = sharecode.decode(text)
        except sharecode.ShareError as e:
            raise LoadoutError(str(e)) from None
        if c.kind != "surgery":
            raise LoadoutError(f"this is a {sharecode.KIND_LABEL[c.kind].lower()} code; a lesion loadout is a surgery code")
        lo = cls.from_surgery_payload(c.payload, name)
        bad = lo.problems(sizes, budget)
        if bad:
            raise LoadoutError("this loadout cannot be used here: " + "; ".join(bad[:3]))
        return lo


# --- saved loadouts ----------------------------------------------------------------------------------------------------------------------
def saved_path() -> Path:
    from kickthefly.core import paths

    return paths.get().data_dir / "lesion_loadouts.json"


def load_saved() -> dict[str, dict[str, int]]:
    try:
        raw = json.loads(saved_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out = {}
    for name, picks in (raw if isinstance(raw, dict) else {}).items():
        if isinstance(name, str) and isinstance(picks, dict) and all(isinstance(t, str) and m in (-1, 1) and not isinstance(m, bool) for t, m in picks.items()):
            out[name[:24]] = {t: int(m) for t, m in picks.items()}
    return out


def save_loadout(lo: Loadout) -> bool:
    """Save under its name (overwriting that name); False if the folder is full of other names."""
    saved = load_saved()
    name = (lo.name or "my fly").strip()[:24] or "my fly"
    if name not in saved and len(saved) >= MAX_SAVED:
        return False
    saved[name] = dict(lo.picks)
    p = saved_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(saved, indent=1), encoding="utf-8")
    tmp.replace(p)
    return True


def delete_saved(name: str) -> None:
    saved = load_saved()
    if saved.pop(name, None) is not None:
        saved_path().write_text(json.dumps(saved, indent=1), encoding="utf-8")


# --- applying and fighting -----------------------------------------------------------------------------------------------------------------
def type_sizes(types) -> dict[str, int]:
    """{cell type: neurons} from a brain's `types` array (untyped neurons left out)."""
    t = np.asarray(types).astype(str)
    keys, counts = np.unique(t[t != ""], return_counts=True)
    return {str(k): int(c) for k, c in zip(keys, counts)}


def apply(brain, loadout: Loadout) -> int:
    """Perform the surgery on a brain with the game's own switch (Brain.set_override). Returns how many neurons it touched."""
    types = np.asarray(brain.types).astype(str)
    n = 0
    for t, m in loadout.picks.items():
        rows = np.flatnonzero(types == t)
        if len(rows):
            brain.set_override(rows, int(m))
            n += len(rows)
    return n


def _fight_task(args: tuple) -> dict:
    """One round in a worker process (module level, so it can be pickled): two brains, each with its loadout, and the duel."""
    seed_a, seed_b, picks_a, picks_b, seconds, mode, backend, battle_seed, rnd = args
    from kickthefly.game import flyduel
    from kickthefly.lab import tournament

    mseed, na, nb = tournament.match_noise(battle_seed, seed_a, seed_b, rnd)
    ba, bb = tournament.build_fighter(seed_a, mode, na, backend), tournament.build_fighter(seed_b, mode, nb, backend)
    apply(ba, Loadout("A", picks_a))
    apply(bb, Loadout("B", picks_b))
    res = flyduel.run_duel(ba, bb, seed=mseed, seconds=seconds, names=("A", "B"))
    res["round"], res["match_seed"] = rnd, mseed
    return res


def fight(seed_a: int, seed_b: int, lo_a: Loadout, lo_b: Loadout, rounds: int = 1, seconds: float = 20.0, mode: str = "subtle",
          backend: str | None = None, battle_seed: int = 0, workers: int = 1, progress=None, cancel=None, task=None) -> dict:
    """A battle of `rounds` duels (1 or 3: best of). Each round has its own match seed and its own noise for both brains (`tournament.match_noise`),
    the same two seeds and surgeries throughout. Returns {winner: 'A'|'B'|None, wins, rounds: [duel results], loadouts, ...}. `task` is for tests."""
    if rounds not in ROUNDS:
        raise LoadoutError(f"a battle is {ROUNDS[0]} or {ROUNDS[1]} rounds")
    task = task or _fight_task
    jobs = [(seed_a, seed_b, dict(lo_a.picks), dict(lo_b.picks), seconds, mode, backend, battle_seed, r) for r in range(rounds)]
    results: list[dict] = []
    wins = {"A": 0, "B": 0}
    t0 = time.time()
    if workers and workers > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        ex = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
        try:
            for r in ex.map(task, jobs):
                results.append(r)
                if progress:
                    progress(len(results), rounds)
        finally:
            ex.shutdown(wait=True, cancel_futures=True)
    else:
        for j in jobs:
            if cancel is not None and cancel.is_set():
                raise RuntimeError("cancelled")
            results.append(task(j))
            if progress:
                progress(len(results), rounds)
    for r in results:
        if r["winner"]:
            wins[r["winner"]] += 1
    need = rounds // 2 + 1
    winner = "A" if wins["A"] >= need else "B" if wins["B"] >= need else (None if wins["A"] == wins["B"] else ("A" if wins["A"] > wins["B"] else "B"))
    return dict(winner=winner, wins=wins, rounds=results, rounds_asked=rounds, seeds=dict(A=seed_a, B=seed_b), seconds_each=seconds,
                loadouts=dict(A=dict(lo_a.picks), B=dict(lo_b.picks)), individuality=mode, took_s=round(time.time() - t0, 1),
                tag="MODEL PREDICTION")


def headline(result: dict) -> str:
    w = result["winner"]
    wins = result["wins"]
    if w is None:
        return f"A draw ({wins['A']}-{wins['B']})."
    side = "Player 1" if w == "A" else "Player 2"
    lo = result["loadouts"][w]
    how = ", ".join(f"{t} {MODE_WORD[m]}" for t, m in lo.items()) or "no surgery at all"
    return f"{side} wins {wins[w]}-{wins['B' if w == 'A' else 'A']} with {how}."
