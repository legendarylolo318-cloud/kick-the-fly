"""Fly tournament (3.0 day 4): a single-elimination bracket of 4, 8 or 16 flies, each a different individuality seed with its
own measured personality card, fighting 1v1 duels where both sides are brains (game/flyduel.py).

CONNECTOME      the neurons that steer, shoot, dodge and run (DNa01/02, DNp35/DNpe052, DNp01, the body-touch group) and the wiring
                that decides when they fire; each fly's personality-card numbers are READOUTS of its own brain (below).
GAME RULE       the arena, the blaster, the start, the bracket (seeds in the order given, paired 1-2, 3-4...), the match length,
                the tie-break (two rematches with new noise, then a seeded coin toss that is labelled as one), the favorite.
MODEL PREDICTION  who wins, which neurons fired before a winner's landed shots, and whether personality predicts winning.

The personality card here is MEASURED (measure_card): the same fly seed builds the same individual (its per-neuron gains come from
the seed, core/individuality.py), and three readouts are taken from that brain at rest: the giant fiber's latency to its dodge
threshold when the looming detectors are driven (temperament), the sugar-pathway -> MN9 drive ratio (feeding), and the right/left
DNa01/02 firing ratio (steering). The card's trait words and their cut-offs are core/individuality.compute_personality_card's.
(The card the game shows for a fly in play is not measured: it is derived from the seed. This one is.)

Pre-registered analysis (written before any run; `analyze`): does personality predict winning?
  Primary, one test per trait: over the matches decided by the brains (coin tosses excluded), how often does the fly with the
  higher trait value win? Exact two-sided binomial test against 1/2, Holm-corrected over the three traits (loom latency, sugar
  ratio, log steering ratio). Secondary, reported but not corrected: Spearman correlation of each trait with the number of rounds
  a fly won. A bracket gives n - 1 matches, so a single 16-fly bracket has little power: pool brackets (--seeds with more than N).
  A null result is a result. Run it with --individuality off as the control: identical brains cannot differ in personality.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from kickthefly.game import flyduel

SIZES = (4, 8, 16)
TRAITS = ("loom_latency_s", "sugar_ratio", "turning_log_ratio")
TRAIT_LABEL = {"loom_latency_s": "looming latency to dodge (s)", "sugar_ratio": "sugar -> MN9 ratio",
               "turning_log_ratio": "log DNa01/02 right/left"}
LATENCY_CAP_S = 1.0
CALM_STEPS, DRIVE_PRE, DRIVE_STIM = 400, 200, 200
DEFAULT_MODE = "subtle"
TAGS = dict(steer_shoot_dodge="CONNECTOME", arena_blaster_bracket_tiebreak="GAME RULE", winner_and_drivers="MODEL PREDICTION")


class TournamentError(ValueError):
    pass


# --- the measured personality card ------------------------------------------------------------------------------------------
def measure_card(seed: int, mode: str = DEFAULT_MODE, backend: str | None = None, warmup: int = 600) -> dict:
    """Build the fly of this individuality seed and read its card numbers from its own brain (see the module docstring)."""
    from kickthefly.core import savestate, simcore
    from kickthefly.core.individuality import compute_personality_card
    from kickthefly.game import kick_the_fly as k
    from kickthefly.lab import assays, validation

    br = simcore.new_brain(seed=int(seed), individuality=mode, backend=backend, warmup=warmup)
    g = assays.groups(br)
    snap: dict = {}
    meta = savestate.brain_state(br, "s_", snap)
    inst, types = br.graph.instance.astype(str), br.types.astype(str)
    dna = (types == "DNa01") | (types == "DNa02")
    right, left = np.flatnonzero(dna & np.char.endswith(inst, "_R")), np.flatnonzero(dna & np.char.endswith(inst, "_L"))
    counts, walk_levels = np.zeros(2), []
    for i in range(CALM_STEPS):
        br._step()
        s = br.sim.spikes
        counts += (np.count_nonzero(s[right]), np.count_nonzero(s[left]))
        if i % 4 == 3:
            walk_levels.append(br.level("walk"))
    hz = counts / (CALM_STEPS * br.dt) / np.array([max(1, len(right)), max(1, len(left))])
    turning_ratio = float((hz[0] + 0.1) / (hz[1] + 0.1))
    savestate.restore_brain(br, meta, snap, "s_")
    b, d = assays.pathway_response(br, g["sweet"], {"mn9": g["mn9"]}, pre=DRIVE_PRE, stim=DRIVE_STIM)["mn9"]
    sugar_ratio = float(validation._ratio(b, d))
    savestate.restore_brain(br, meta, snap, "s_")
    simcore.step(br, DRIVE_PRE)
    simcore.drive(br, g["loom"], 0.5)
    latency, crossed = LATENCY_CAP_S, False
    for i in range(int(LATENCY_CAP_S / br.dt)):
        br._step()
        if br.level("escape") > k.THRESH["escape"]:
            latency, crossed = (i + 1) * br.dt, True
            break
    card = compute_personality_card(int(seed), looming_latency=latency, turning_ratio=turning_ratio, sugar_ratio=sugar_ratio,
                                    tmaze_pi=None, mode=mode)
    return dict(seed=int(seed), mode=mode, loom_latency_s=float(latency), loom_crossed=crossed, sugar_ratio=sugar_ratio,
                turning_ratio=turning_ratio, turning_log_ratio=float(math.log(turning_ratio)),
                walk_level_calm=float(np.mean(walk_levels)),
                title=card["title"], summary=card["summary"], traits=card["traits"], measured=True,
                tmaze_pi="not measured (a T-maze costs about a minute per fly)")


def _card_task(args: tuple) -> dict:
    seed, mode, backend = args
    return measure_card(seed, mode, backend)


def measure_cards(seeds, mode: str, backend: str | None = None, workers: int = 1, pool=None, progress=None, cancel=None) -> dict[int, dict]:
    """The measured cards of several flies, in worker processes when workers > 1 (each takes about ten seconds)."""
    out: dict[int, dict] = {}
    seeds = [int(s) for s in seeds]
    if pool is None or workers <= 1:
        for i, s in enumerate(seeds):
            if cancel is not None and cancel.is_set():
                raise RuntimeError("cancelled")
            if progress:
                progress(i, len(seeds), f"measuring the personality card of fly {s} ({i + 1}/{len(seeds)})")
            out[s] = measure_card(s, mode, backend)
        return out
    futs = {pool.submit(_card_task, (s, mode, backend)): s for s in seeds}
    for f in as_completed(futs):
        if cancel is not None and cancel.is_set():
            raise RuntimeError("cancelled")
        out[futs[f]] = f.result()
        if progress:
            progress(len(out), len(seeds), f"measured the personality card of fly {futs[f]} ({len(out)}/{len(seeds)})")
    return out


def build_fighter(seed: int, mode: str, noise_seed: int | None, backend: str | None = None):
    """The individual of this seed, with a match-specific noise stream when noise_seed is given."""
    from kickthefly.core import simcore

    br = simcore.new_brain(seed=int(seed), individuality=mode, backend=backend)
    if noise_seed is not None:
        br.reseed(int(noise_seed) & 0x7FFFFFFF)
    return br


def match_noise(bracket_seed: int, a: int, b: int, attempt: int) -> tuple[int, int, int]:
    """(match seed, noise of a, noise of b): deterministic from the pair and the attempt (a rematch is another attempt)."""
    h = hashlib.sha256(f"{bracket_seed}:{a}:{b}:{attempt}".encode()).digest()
    return tuple(int.from_bytes(h[i * 4:i * 4 + 4], "little") & 0x7FFFFFFF for i in range(3))


def _match_task(args: tuple) -> dict:
    a, b, bracket_seed, attempt, seconds, mode, backend, drivers = args
    mseed, na, nb = match_noise(bracket_seed, a, b, attempt)
    ba, bb = build_fighter(a, mode, na, backend), build_fighter(b, mode, nb, backend)
    res = flyduel.run_duel(ba, bb, seed=mseed, seconds=seconds, names=(str(a), str(b)), drivers=drivers)
    res["attempt"], res["match_seed"] = attempt, mseed
    return res


def _pairs(fighters: list[int]) -> list[tuple[int, int]]:
    return [(fighters[i], fighters[i + 1]) for i in range(0, len(fighters), 2)]


def run_bracket(seeds, seconds: float = flyduel.DEFAULT_SECONDS, mode: str = DEFAULT_MODE, favorite: int | None = None,
                workers: int = 1, bracket_seed: int = 0, backend: str | None = None, progress=None, cancel=None,
                cards: bool = True, drivers: bool = True, play=None) -> dict:
    """Run a bracket of len(seeds) in (4, 8, 16) flies. Seeds in the order given: 1 v 2, 3 v 4, ... then the winners in order.

    workers > 1 plays each round's matches in worker processes. progress(done_matches, total_matches, label).
    play: a function (task tuple) -> match result, replacing the real brains (tests); runs in this process.
    """
    seeds = [int(s) for s in seeds]
    if len(seeds) not in SIZES:
        raise TournamentError(f"a bracket has {', '.join(map(str, SIZES))} flies, not {len(seeds)}")
    if len(set(seeds)) != len(seeds):
        raise TournamentError("every fly needs its own individuality seed")
    if favorite is not None and int(favorite) not in seeds:
        raise TournamentError("the favorite must be one of the flies")
    t0 = time.time()
    total_matches = len(seeds) - 1
    done = 0
    cards_by_seed: dict[int, dict] = {}
    rounds, drivers_by_seed, wins, fighters = [], {s: [] for s in seeds}, {s: 0 for s in seeds}, list(seeds)
    pool = None
    if workers > 1 and play is None:
        import multiprocessing

        pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
    try:
        if cards:
            if play is not None:
                cards_by_seed = {s: play(("card", s, mode)) for s in seeds}
            else:
                cards_by_seed = measure_cards(seeds, mode, backend, workers, pool, lambda d, n, label: progress and progress(done, total_matches, label), cancel)
        while len(fighters) > 1:
            pairs = _pairs(fighters)
            tasks = {i: (a, b, bracket_seed, 0, seconds, mode, backend, drivers) for i, (a, b) in enumerate(pairs)}
            results: dict[int, list[dict]] = {i: [] for i in tasks}

            def run_all(batch: dict[int, tuple]) -> dict[int, dict]:
                if cancel is not None and cancel.is_set():
                    raise RuntimeError("cancelled")
                if play is not None:
                    return {i: play(t) for i, t in batch.items()}
                if pool is None:
                    out = {}
                    for i, t in batch.items():
                        out[i] = _match_task(t)
                        if progress:
                            progress(done + len(out), total_matches, f"match {t[0]} v {t[1]}")
                    return out
                futs = {pool.submit(_match_task, t): i for i, t in batch.items()}
                out = {}
                for f in as_completed(futs):
                    out[futs[f]] = f.result()
                    if progress:
                        progress(done + len(out), total_matches, f"match {pairs[futs[f]][0]} v {pairs[futs[f]][1]}")
                return out

            first = run_all(tasks)
            pending = {}
            for i, r in first.items():
                results[i].append(r)
                if r["winner"] is None:
                    pending[i] = (*tasks[i][:3], 1, *tasks[i][4:])
            for attempt in range(1, flyduel.TIEBREAK_REMATCHES + 1):
                if not pending:
                    break
                again = run_all({i: (*t[:3], attempt, *t[4:]) for i, t in pending.items()})
                pending = {}
                for i, r in again.items():
                    results[i].append(r)
                    if r["winner"] is None:
                        pending[i] = tasks[i]
            matches = []
            for i, (a, b) in enumerate(pairs):
                rs = results[i]
                final = rs[-1]
                coin = final["winner"] is None
                if coin:                                  # a draw after every rematch: a seeded coin toss, labelled as one
                    h = hashlib.sha256(f"coin:{bracket_seed}:{a}:{b}".encode()).digest()[0]
                    winner = a if h % 2 == 0 else b
                else:
                    winner = int(final["winner"])
                wins[winner] += 1
                for r in rs:                              # the drivers of every match a fly fought, for the champion's report
                    dr = r.pop("_drivers", None) or {}
                    for side, part in dr.items():
                        drivers_by_seed[int(side)].append(part)
                matches.append(dict(a=a, b=b, winner=winner, coin_toss=coin, attempts=len(rs), decided_by=_decided_by(final),
                                    hp=final["hp"], shots=final["shots"], hits=final["hits"], seconds=final["seconds"],
                                    knockout=final["knockout"], max_levels=final["max_levels"], events=final["events"],
                                    frames=final["frames"], match_seed=final["match_seed"]))
            done += len(pairs)
            rounds.append(matches)
            fighters = [m["winner"] for m in matches]
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    champion = fighters[0]
    champ_sums = flyduel.combine_drivers(drivers_by_seed[champion]) if drivers else None
    champ_drivers = flyduel.top_drivers(champ_sums) if drivers else []
    champ_readouts = flyduel.named_drivers(champ_sums) if drivers else []
    return dict(kind="tournament", created=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=round(time.time() - t0, 1), size=len(seeds),
                seeds=seeds, individuality=mode, match_seconds=seconds, bracket_seed=bracket_seed, favorite=favorite,
                favorite_won=(None if favorite is None else int(favorite) == champion),
                favorite_rounds_won=(None if favorite is None else wins[int(favorite)]), cards=cards_by_seed, rounds=rounds,
                champion=champion, rounds_won=wins, champion_drivers=champ_drivers, champion_readouts=champ_readouts,
                champion_drivers_note="cell types whose firing in the 200 ms before the champion's LANDED shots most exceeds their firing "
                                      "over its matches: a correlation, nothing was silenced, and a fly that was just hit also fires its "
                                      "touch neurons, so those lead the list (CONNECTOME readout; the duel is a GAME RULE). "
                                      "champion_readouts: the cell types the duel's own rules read, whatever their rank",
                tags=TAGS)


def _decided_by(r: dict) -> str:
    if r["winner"] is None:
        return "draw"
    return "knockout" if r["knockout"] else "health at the time limit"


# --- does personality predict winning? ----------------------------------------------------------------------------------------
def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    out = [0.0] * len(ps)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        out[i] = run
    return out


def analyze(brackets: list[dict]) -> dict:
    """The pre-registered analysis over one or more brackets (see the module docstring)."""
    from scipy import stats

    flies = {}
    for b in brackets:
        for s in b["seeds"]:
            flies[(id(b), s)] = dict(seed=s, wins=b["rounds_won"][s], **{t: b["cards"][s][t] for t in TRAITS})
    decided = [(b, m) for b in brackets for r in b["rounds"] for m in r if not m["coin_toss"]]
    tests = []
    for t in TRAITS:
        higher_wins = n = 0
        for b, m in decided:
            va, vb = b["cards"][m["a"]][t], b["cards"][m["b"]][t]
            if va == vb:
                continue
            n += 1
            higher = m["a"] if va > vb else m["b"]
            higher_wins += int(m["winner"] == higher)
        p = float(stats.binomtest(higher_wins, n, 0.5).pvalue) if n else 1.0
        xs = [f[t] for f in flies.values()]
        ys = [f["wins"] for f in flies.values()]
        rho, rp = (stats.spearmanr(xs, ys) if len(set(xs)) > 1 and len(set(ys)) > 1 else (float("nan"), float("nan")))
        tests.append(dict(trait=t, label=TRAIT_LABEL[t], matches=n, higher_trait_wins=higher_wins,
                          win_share_of_higher=(higher_wins / n if n else float("nan")), p=p, spearman_rho=float(rho),
                          spearman_p=float(rp)))
    for t, ph in zip(tests, holm([x["p"] for x in tests])):
        t["p_holm"] = ph
        t["significant"] = bool(ph < 0.05)
    return dict(kind="tournament_analysis", brackets=len(brackets), flies=len(flies), decided_matches=len(decided),
                coin_tosses=sum(1 for b in brackets for r in b["rounds"] for m in r if m["coin_toss"]),
                knockouts=sum(1 for b, m in decided if m["knockout"]), tests=tests,
                predicts_winning=any(t["significant"] for t in tests),
                individuality=sorted({b["individuality"] for b in brackets}),
                criteria="exact two-sided binomial test of 'the fly with the higher trait wins' against 1/2, Holm over 3 traits, "
                         "alpha 0.05; Spearman of trait vs rounds won reported, uncorrected")


def summary(brackets: list[dict], an: dict | None = None) -> str:
    from kickthefly.lab import labstats

    lines = []
    for b in brackets:
        lines.append(f"BRACKET of {b['size']} (individuality {b['individuality']}, {b['match_seconds']:.0f} s duels), champion fly "
                     f"{b['champion']} ({b['cards'].get(b['champion'], {}).get('title', '?')})")
        for i, r in enumerate(b["rounds"]):
            for m in r:
                lines.append(f"  round {i + 1}: {m['a']} v {m['b']} -> {m['winner']} ({m['decided_by']}"
                             f"{', coin toss' if m['coin_toss'] else ''}; hits {m['hits'][str(m['a'])]}-{m['hits'][str(m['b'])]})")
        if b.get("favorite") is not None:
            lines.append(f"  your favorite {b['favorite']}: {'WON' if b['favorite_won'] else 'out after ' + str(b['favorite_rounds_won']) + ' win(s)'}")
    an = an or analyze(brackets)
    lines.append(f"DOES PERSONALITY PREDICT WINNING? {an['flies']} flies in {an['brackets']} bracket(s), {an['decided_matches']} brain-decided "
                 f"matches ({an['coin_tosses']} coin tosses excluded)")
    for t in an["tests"]:
        lines.append(f"  {t['label']}: the higher-trait fly won {t['higher_trait_wins']}/{t['matches']}, Holm-corrected "
                     f"{labstats.fmt_p(t['p_holm'])}  ({'significant' if t['significant'] else 'not significant'}); "
                     f"Spearman vs rounds won rho = {t['spearman_rho']:.2f}, {labstats.fmt_p(t['spearman_p'])} (uncorrected)")
    lines.append("  MODEL PREDICTION: this model's duel, rules and noise; not a measurement of real flies")
    return "\n".join(lines)


def save(brackets: list[dict], an: dict, folder: Path | str) -> list[Path]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / "tournament.json"
    p.write_text(json.dumps(dict(brackets=brackets, analysis=an), indent=1), encoding="utf-8")
    out = [p]
    p = folder / "tournament_flies.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bracket", "seed", "title", *TRAITS, "rounds_won", "champion"])
        for bi, b in enumerate(brackets):
            for s in b["seeds"]:
                c = b["cards"][s]
                w.writerow([bi, s, c["title"], *(c[t] for t in TRAITS), b["rounds_won"][s], int(b["champion"] == s)])
    out.append(p)
    p = folder / "tournament_matches.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bracket", "round", "a", "b", "winner", "coin_toss", "decided_by", "attempts", "hp_a", "hp_b", "shots_a", "shots_b",
                    "hits_a", "hits_b"])
        for bi, b in enumerate(brackets):
            for ri, r in enumerate(b["rounds"]):
                for m in r:
                    a, bb = str(m["a"]), str(m["b"])
                    w.writerow([bi, ri + 1, m["a"], m["b"], m["winner"], int(m["coin_toss"]), m["decided_by"], m["attempts"], m["hp"][a],
                                m["hp"][bb], m["shots"][a], m["shots"][bb], m["hits"][a], m["hits"][bb]])
    out.append(p)
    p = folder / "tournament_analysis.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["trait", "matches", "higher_trait_wins", "p", "p_holm", "significant", "spearman_rho", "spearman_p"])
        for t in an["tests"]:
            w.writerow([t["trait"], t["matches"], t["higher_trait_wins"], t["p"], t["p_holm"], int(t["significant"]), t["spearman_rho"],
                        t["spearman_p"]])
    out.append(p)
    return out


def main(args) -> int:
    """--headless --tournament N --seeds A-B [--individuality MODE] [--match-seconds S] [--out DIR]"""
    import sys

    from kickthefly.lab import headless, recorder

    size = int(args.tournament)
    if size not in SIZES:
        print(f"error: --tournament takes {', '.join(map(str, SIZES))}", file=sys.stderr)
        return 2
    seeds = headless.parse_seeds(args.seeds, range(1000, 1000 + size))
    if len(seeds) % size:
        print(f"error: {len(seeds)} seeds do not split into brackets of {size}", file=sys.stderr)
        return 2
    mode = getattr(args, "individuality", None) or DEFAULT_MODE
    seconds = getattr(args, "match_seconds", None) or flyduel.DEFAULT_SECONDS
    t0 = time.time()
    brackets = []
    for bi in range(0, len(seeds), size):
        grp = seeds[bi:bi + size]
        brackets.append(run_bracket(grp, seconds=seconds, mode=mode, workers=max(1, args.workers or 1), bracket_seed=bi // size,
                                    progress=lambda d, n, label: print(f"  {d}/{n} {label} ({time.time() - t0:.0f}s)", flush=True)))
    an = analyze(brackets)
    folder = Path(args.out) if args.out else recorder.exports_dir() / f"{time.strftime('%Y%m%d-%H%M%S')}-tournament"
    save(brackets, an, folder)
    print(summary(brackets, an))
    print(f"results written to {folder}")
    return 0
