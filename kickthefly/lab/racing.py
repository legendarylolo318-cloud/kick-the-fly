"""Fly racing (3.0 day 4): the race, the odds, the bet and the headless race assay. The physics are game/flyrace.py's; read its
docstring for what is CONNECTOME, GAME RULE and MODEL PREDICTION. Points only (core/points.py): no money, no purchases.

The odds (GAME RULE): each fly's FORM is the mean of two z-scores within the field, taken from its measured personality card:
the sugar -> MN9 ratio (how hard the lures' taste drives it) and its calm walking drive (DNp09's level at rest, which is what sets
speed). P(win) = softmax(form), and the payout is the fair price less a 10% take. The odds are a MODEL: nothing here is tuned to
make them right, and the race assay measures whether they are.

Pre-registered analysis (written before any run; `analyze`). Each fly runs the same race REPEATS times (default 2, new noise each
time). Primary family, Holm-corrected, alpha 0.05:
  R1  repeatability: Spearman correlation of a fly's finishing time between its first and second run, across all flies (one-sided,
      positive). It is the direct test of "does a fly's individuality persist": identical brains (--individuality off) should show
      none.
  R2  form: Spearman correlation of the odds' form score with the mean finishing time (one-sided, negative: high form, fast).
  R3  the odds: the Brier score of the win probabilities against the actual winners is lower than a uniform field's, per race,
      one-sided Wilcoxon signed-rank over the races (a floor of p < 0.001 is reported as such).
Secondary, uncorrected: Spearman of each card trait with the mean finishing time. A null result is a result.
"""
from __future__ import annotations

import csv
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from kickthefly.game import flyrace
from kickthefly.lab import tournament

DEFAULT_LANES = 6
DEFAULT_REPEATS = 2
BETA = 1.0                           # softmax temperature of the form score: GAME RULE
TRAITS = ("sugar_ratio", "walk_level_calm", "loom_latency_s", "turning_log_ratio")
TRAIT_LABEL = {"sugar_ratio": "sugar -> MN9 ratio", "walk_level_calm": "calm walking drive (DNp09 level)",
               "loom_latency_s": "looming latency to dodge (s)", "turning_log_ratio": "log DNa01/02 right/left"}
TAGS = dict(speed_from_walking_neurons="CONNECTOME", track_lures_odds_points="GAME RULE", order_and_prediction="MODEL PREDICTION")


def z(x) -> np.ndarray:
    x = np.asarray(x, float)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-12 else np.zeros_like(x)


def form_scores(cards: list[dict]) -> np.ndarray:
    """GAME RULE: the mean of the within-field z-scores of the sugar ratio and the calm walking drive."""
    return (z([c["sugar_ratio"] for c in cards]) + z([c["walk_level_calm"] for c in cards])) / 2.0


def win_probabilities(cards: list[dict]) -> np.ndarray:
    f = form_scores(cards) * BETA
    e = np.exp(f - f.max())
    return e / e.sum()


def lane_noise(race_seed: int, fly: int, repeat: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{race_seed}:{fly}:{repeat}".encode()).digest()[:4], "little") & 0x7FFFFFFF


def _lane_task(args: tuple) -> tuple[int, int, dict]:
    seed, repeat, race_seed, mode, backend, cap = args
    br = tournament.build_fighter(seed, mode, lane_noise(race_seed, seed, repeat), backend)
    return seed, repeat, flyrace.run_lane(br, cap_s=cap)


def measure_field(seeds, mode: str = tournament.DEFAULT_MODE, backend: str | None = None, progress=None, cancel=None, play=None,
                  workers: int = 1) -> list[dict]:
    """The measured personality cards of a field (the pre-race look the odds come from), in worker processes when workers > 1."""
    if play is None and workers > 1:
        import multiprocessing

        ex = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
        try:                                   # 3.0 day 4 review: a `with` block waited for every queued card after Cancel
            got = tournament.measure_cards(seeds, mode, backend, workers, ex, progress, cancel)
        finally:
            ex.shutdown(wait=True, cancel_futures=True)
        return [got[int(s)] for s in seeds]
    out = []
    for i, s in enumerate(seeds):
        if cancel is not None and cancel.is_set():
            raise RuntimeError("cancelled")
        if progress:
            progress(i, len(seeds), f"measuring the personality card of fly {s}")
        out.append(tournament.measure_card(int(s), mode, backend) if play is None else play(("card", int(s), mode)))
    return out


def odds_table(cards: list[dict]) -> list[dict]:
    """The odds a field's cards give: [{fly, title, form, p_win, decimal_odds}]."""
    p = win_probabilities(cards)
    f = form_scores(cards)
    return [dict(fly=c["seed"], title=c["title"], form=float(fi), p_win=float(pi), decimal_odds=_odds(pi)) for c, fi, pi in zip(cards, f, p)]


def run_race(seeds, mode: str = tournament.DEFAULT_MODE, repeats: int = 1, workers: int = 1, race_seed: int = 0,
             backend: str | None = None, cap_s: float = flyrace.TIME_CAP_S, bets: list[dict] | None = None, progress=None,
             cancel=None, cards: bool = True, play=None, card_list: list[dict] | None = None) -> dict:
    """Race the flies (each in its own lane, `repeats` times). bets: [{fly, stake}] are settled by the caller against `winner`.
    card_list: cards measured earlier (measure_field), so a player can bet on the odds before the race.
    play: a function (task) -> (seed, repeat, lane result), replacing the real brains (tests)."""
    seeds = [int(s) for s in seeds]
    if not 2 <= len(seeds) <= flyrace.MAX_LANES:
        raise ValueError(f"a race has 2 to {flyrace.MAX_LANES} lanes")
    if len(set(seeds)) != len(seeds):
        raise ValueError("every fly needs its own individuality seed")
    t0 = time.time()
    if card_list is None:
        card_list = measure_field(seeds, mode, backend, progress, cancel, play, workers) if cards else []
    tasks = [(s, r, race_seed, mode, backend, cap_s) for r in range(repeats) for s in seeds]
    runs: dict[int, dict[int, dict]] = {r: {} for r in range(repeats)}
    done = 0

    def got(seed, repeat, res):
        nonlocal done
        runs[repeat][seed] = res
        done += 1
        if progress:
            progress(done, len(tasks), f"fly {seed} run {repeat + 1}")

    if play is not None:
        for t in tasks:
            got(*play(t))
    elif workers > 1:
        import multiprocessing

        ex = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
        try:
            futs = [ex.submit(_lane_task, t) for t in tasks]
            for f in as_completed(futs):
                if cancel is not None and cancel.is_set():
                    raise RuntimeError("cancelled")
                got(*f.result())
        finally:
            ex.shutdown(wait=True, cancel_futures=True)
    else:
        for t in tasks:
            if cancel is not None and cancel.is_set():
                raise RuntimeError("cancelled")
            got(*_lane_task(t))
    orders = [flyrace.order(runs[r]) for r in range(repeats)]
    out = dict(kind="race", created=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=round(time.time() - t0, 1), seeds=seeds,
               individuality=mode, race_seed=race_seed, repeats=repeats, track_m=flyrace.TRACK_LENGTH, lures=[list(x) for x in flyrace.LURES],
               cards={str(c["seed"]): c for c in card_list}, runs={str(r): {str(s): v for s, v in runs[r].items()} for r in runs},
               orders=orders, winner=orders[0][0], tags=TAGS)
    if card_list:
        p = win_probabilities(card_list)
        out["odds"] = {str(s): dict(p_win=float(pi), decimal_odds=_odds(pi), form=float(f))
                       for s, pi, f in zip(seeds, p, form_scores(card_list))}
    return out


def _odds(p: float) -> float:
    from kickthefly.core import points

    return points.decimal_odds(p)


def settle_bets(race: dict, bets: list[dict], wallet) -> list[dict]:
    """Settle in-game point bets ({fly, stake}) on race['winner'] (the first run). Points only."""
    out = []
    for b in bets:
        odds = race["odds"][str(b["fly"])]["decimal_odds"]
        won = int(b["fly"]) == int(race["winner"])
        delta = wallet.settle(int(b["stake"]), odds, won, label=f"bet on fly {b['fly']}")
        out.append(dict(fly=int(b["fly"]), stake=int(b["stake"]), odds=odds, won=won, delta=delta, points=wallet.points))
    return out


# --- does individuality predict the finishing order? ---------------------------------------------------------------------------
def analyze(races: list[dict]) -> dict:
    from scipy import stats

    flies = []
    for r in races:
        cards = [r["cards"][str(s)] for s in r["seeds"]]
        f = form_scores(cards)
        for i, s in enumerate(r["seeds"]):
            times = [_effective_time(r["runs"][str(k)][str(s)], r["track_m"]) for k in range(r["repeats"])]
            flies.append(dict(seed=s, race=r["race_seed"], form=float(f[i]), times=times, mean_time=float(np.mean(times)),
                              **{t: r["cards"][str(s)][t] for t in TRAITS}))
    tests = []
    if flies and flies[0]["times"] and len(flies[0]["times"]) >= 2:
        a = [f["times"][0] for f in flies]
        b = [f["times"][1] for f in flies]
        rho, p2 = stats.spearmanr(a, b)
        tests.append(dict(id="R1", label="repeatability: finishing time, run 1 vs run 2 (same fly)", n=len(flies), rho=float(rho),
                          p=float(p2 / 2 if rho > 0 else 1 - p2 / 2)))
    mt = [f["mean_time"] for f in flies]
    rho, p2 = stats.spearmanr([f["form"] for f in flies], mt)
    tests.append(dict(id="R2", label="odds form score vs finishing time (high form, fast)", n=len(flies), rho=float(rho),
                      p=float(p2 / 2 if rho < 0 else 1 - p2 / 2)))
    diffs = []
    for r in races:
        p = win_probabilities([r["cards"][str(s)] for s in r["seeds"]])
        w = r["seeds"].index(r["winner"])
        y = np.zeros(len(p))
        y[w] = 1.0
        diffs.append(float(np.sum((p - y) ** 2) - np.sum((1.0 / len(p) - y) ** 2)))
    if len(diffs) >= 1 and np.any(np.asarray(diffs) != 0):
        try:
            pw = float(stats.wilcoxon(diffs, alternative="less", zero_method="wilcox").pvalue)
        except ValueError:
            pw = 1.0
    else:
        pw = 1.0
    tests.append(dict(id="R3", label="odds beat a uniform field (Brier score, per race)", n=len(diffs), mean_brier_diff=float(np.mean(diffs)),
                      p=pw))
    for t, ph in zip(tests, tournament.holm([x["p"] for x in tests])):
        t["p_holm"], t["significant"] = ph, bool(ph < 0.05)
    secondary = []
    for tr in TRAITS:
        xs = [f[tr] for f in flies]
        if len(set(xs)) > 1 and len(set(mt)) > 1:
            rho, p2 = stats.spearmanr(xs, mt)
        else:
            rho, p2 = float("nan"), float("nan")
        secondary.append(dict(trait=tr, label=TRAIT_LABEL[tr], rho=float(rho), p=float(p2)))
    return dict(kind="race_analysis", races=len(races), flies=len(flies), individuality=sorted({r["individuality"] for r in races}),
                tests=tests, secondary=secondary, individuality_predicts_finish=any(t["significant"] for t in tests),
                winners_were_favorite=sum(1 for r in races if r["odds"][str(r["winner"])]["p_win"] == max(v["p_win"] for v in r["odds"].values()))
                if all("odds" in r for r in races) else None,
                criteria="primary family R1-R3, Holm, alpha 0.05, one-sided as stated; secondary uncorrected two-sided Spearman")


def _effective_time(lane: dict, track: float) -> float:
    """Finishing time; a fly that did not finish by the cap is given the time its distance implies at its average speed
    (no censoring trick: it keeps the order it would have had)."""
    if lane["finish_s"] is not None:
        return float(lane["finish_s"])
    return float(lane["seconds"] * track / max(lane["distance"], 1e-3))


def summary(races: list[dict], an: dict | None = None) -> str:
    from kickthefly.lab import labstats

    an = an or analyze(races)
    lines = []
    for r in races:
        lines.append(f"RACE (individuality {r['individuality']}, {len(r['seeds'])} lanes, {r['track_m']:g} m, run 1 order): "
                     + " > ".join(str(s) for s in r["orders"][0]) + (f"   odds: " + ", ".join(
                         f"{s} x{r['odds'][str(s)]['decimal_odds']:g}" for s in r["seeds"]) if "odds" in r else ""))
    lines.append(f"DOES INDIVIDUALITY PREDICT THE FINISH? {an['flies']} flies in {an['races']} race(s) (run twice each: the same fly, new noise)")
    for t in an["tests"]:
        extra = f", rho = {t['rho']:.2f}" if "rho" in t else f", mean Brier difference {t['mean_brier_diff']:+.3f}"
        lines.append(f"  {t['id']} {t['label']}: n = {t['n']}{extra}, Holm-corrected {labstats.fmt_p(t['p_holm'])} "
                     f"({'significant' if t['significant'] else 'not significant'})")
    for s in an["secondary"]:
        lines.append(f"  (uncorrected) {s['label']}: rho = {s['rho']:.2f}, {labstats.fmt_p(s['p'])}")
    lines.append("  MODEL PREDICTION. Points only: no money, no purchases.")
    return "\n".join(lines)


def save(races: list[dict], an: dict, folder: Path | str) -> list[Path]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    p = folder / "race.json"
    p.write_text(json.dumps(dict(races=races, analysis=an), indent=1), encoding="utf-8")
    out.append(p)
    p = folder / "race_flies.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["race", "seed", "title", *TRAITS, "form", "p_win", "odds", *(f"finish_s_run{k + 1}" for k in range(races[0]["repeats"])),
                    "position_run1"])
        for r in races:
            fm = form_scores([r["cards"][str(s)] for s in r["seeds"]])
            for i, s in enumerate(r["seeds"]):
                c, o = r["cards"][str(s)], r.get("odds", {}).get(str(s), {})
                w.writerow([r["race_seed"], s, c["title"], *(c[t] for t in TRAITS), fm[i], o.get("p_win", ""), o.get("decimal_odds", ""),
                            *(r["runs"][str(k)][str(s)]["finish_s"] for k in range(r["repeats"])), r["orders"][0].index(s) + 1])
    out.append(p)
    p = folder / "race_analysis.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "label", "n", "statistic", "p", "p_holm", "significant"])
        for t in an["tests"]:
            w.writerow([t["id"], t["label"], t["n"], t.get("rho", t.get("mean_brier_diff")), t["p"], t["p_holm"], int(t["significant"])])
    out.append(p)
    return out


def main(args) -> int:
    """--headless --race --seeds A-B [--lanes N] [--races K] [--individuality MODE] [--out DIR]"""
    from kickthefly.lab import headless, recorder

    lanes = getattr(args, "lanes", None) or DEFAULT_LANES
    n_races = getattr(args, "races", None) or 1
    seeds = headless.parse_seeds(args.seeds, range(1000, 1000 + lanes * n_races))
    if len(seeds) % lanes:
        print(f"error: {len(seeds)} seeds do not split into races of {lanes}")
        return 2
    mode = getattr(args, "individuality", None) or tournament.DEFAULT_MODE
    t0 = time.time()
    races = []
    for i in range(0, len(seeds), lanes):
        races.append(run_race(seeds[i:i + lanes], mode=mode, repeats=DEFAULT_REPEATS, workers=max(1, args.workers or 1), race_seed=i // lanes,
                              progress=lambda d, n, label: print(f"  {d}/{n} {label} ({time.time() - t0:.0f}s)", flush=True)))
    an = analyze(races)
    folder = Path(args.out) if args.out else recorder.exports_dir() / f"{time.strftime('%Y%m%d-%H%M%S')}-race"
    save(races, an, folder)
    print(summary(races, an))
    print(f"results written to {folder}")
    return 0
