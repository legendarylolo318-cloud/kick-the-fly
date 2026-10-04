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

R1 does not test individuality (found with its control, 3.0 day 4): with --individuality off it is as repeatable (rho 0.92 vs 0.96).
Both runs of a fly start from the same post-warm-up state of its seed (the brain warms up with the seed's noise, then the lane noise
is reseeded), and that state, not the per-neuron gains, carries most of a fly's speed (docs/racing.md; tools/race_diagnosis.py).
R1-R3 and their results are kept as they were.

R4, pre-registered in the 3.0 day 4 review, written here BEFORE it was run on any held-out seed (`run_r4`, `analyze_r4`):
  Design. 8 races x 6 lanes. In race k (k = 0..7) every lane is built from the SAME brain-state seed S_k = 1000 + k (the warm-up and
    noise seed) with its OWN individuality seed I = 1000 + 6k + j (j = 0..5; 1000-1047 over the 8 races), warmed up, then reseeded to
    the lane noise lane_noise(k, I, run) for run 1 and run 2, as R1 does. Two conditions: individuality 'subtle' (the default), and
    the control 'off', in which the six lanes of a race are one brain in one state that differ only by the lane noise.
  Statistic r_w. Within each race, the six finishing times (effective times, as R1) of run 1 and of run 2 are ranked; r_w is the
    Pearson correlation of the (run 1 rank, run 2 rank) pairs over the 48 flies: within-race rank repeatability. The shared brain
    state cannot contribute to it, because it is the same for all six lanes of a race.
  PASS needs both, alpha 0.05:
    (i)  'subtle' r_w > 0: one-sided permutation test, run-2 ranks permuted within each race, 10,000 permutations (generator seed 0),
         p = (1 + #{r_perm >= r_obs}) / 10,001, p < 0.05;
    (ii) r_w('subtle') - r_w('off') > 0: its 95% percentile bootstrap interval lies above 0 (10,000 resamples of the flies within each
         race, with replacement, independently per condition; generator seed 1).
  Reported, not criteria: r_w('off') and its permutation p (a check of the design: it should show nothing), each condition's CI.
  Reading fixed in advance: PASS = the per-neuron individuality gains make a fly's finishing rank repeatable beyond its brain state
    and noise. FAIL = at sigma 0.05 and 48 flies this assay cannot see individuality in race speed; NOT evidence that it has none.
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
TRAIT_LABEL = {"sugar_ratio": "sugar -> MN9 ratio", "walk_level_calm": "calm walking drive (DNp09 / the fixed reference)",
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
        if play is None:
            from kickthefly.core import cards

            out.append(tournament.measure_card(int(s), mode, backend))
            cards.store(out[-1])
        else:
            out.append(play(("card", int(s), mode)))
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
               orders=orders, winner=orders[0][0], tags=TAGS, speed_rule_version=flyrace.RULE_VERSION)
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


# --- R4: individuality with the brain state held fixed (pre-registered in the module docstring before any held-out run) ---------
R4_RACES, R4_LANES, R4_PERMUTATIONS, R4_BOOTSTRAP = 8, 6, 10_000, 10_000
R4_FIRST_SEED = 1000


def r4_design(first: int = R4_FIRST_SEED, races: int = R4_RACES, lanes: int = R4_LANES) -> list[dict]:
    """[{race, state_seed, individuality_seeds}]: race k shares the brain-state seed first + k; its lanes are first + lanes*k + j."""
    return [dict(race=k, state_seed=first + k, individuality_seeds=[first + lanes * k + j for j in range(lanes)]) for k in range(races)]


def _r4_lane_task(args: tuple) -> tuple[int, int, int, dict]:
    race, state_seed, iseed, repeat, mode, backend, cap = args
    from kickthefly.core import simcore

    br = simcore.new_brain(seed=int(state_seed), individuality=mode, individuality_seed=int(iseed), backend=backend)
    tournament.check_individual(br, mode)
    br.reseed(lane_noise(race, iseed, repeat))
    return race, iseed, repeat, flyrace.run_lane(br, cap_s=cap)


def run_r4(mode: str, workers: int = 1, backend: str | None = None, cap_s: float = flyrace.TIME_CAP_S, design: list[dict] | None = None,
           progress=None, cancel=None, play=None) -> dict:
    """Both runs of every lane of the R4 design in one condition (mode). play: (task) -> (race, iseed, repeat, lane), for tests."""
    design = design or r4_design()
    tasks = [(d["race"], d["state_seed"], i, r, mode, backend, cap_s) for d in design for i in d["individuality_seeds"] for r in (0, 1)]
    lanes: dict = {}
    t0 = time.time()

    def got(race, iseed, repeat, lane):
        lanes[(race, iseed, repeat)] = lane
        if progress:
            progress(len(lanes), len(tasks), f"{mode}: race {race} fly {iseed} run {repeat + 1}")

    if play is not None:
        for t in tasks:
            if cancel is not None and cancel.is_set():
                raise RuntimeError("cancelled")
            got(*play(t))
    else:
        import multiprocessing

        ex = ProcessPoolExecutor(max_workers=max(1, workers), mp_context=multiprocessing.get_context("spawn"))
        try:
            futs = [ex.submit(_r4_lane_task, t) for t in tasks]
            for f in as_completed(futs):
                if cancel is not None and cancel.is_set():
                    raise RuntimeError("cancelled")
                got(*f.result())
        finally:
            ex.shutdown(wait=True, cancel_futures=True)
    flies = []
    for d in design:
        for i in d["individuality_seeds"]:
            runs = [lanes[(d["race"], i, r)] for r in (0, 1)]
            flies.append(dict(race=d["race"], state_seed=d["state_seed"], individuality_seed=i,
                              times=[_effective_time(x, flyrace.TRACK_LENGTH) for x in runs], finished=[x["finish_s"] is not None for x in runs]))
    return dict(kind="race_r4", individuality=mode, design=design, flies=flies, seconds=round(time.time() - t0, 1), tags=TAGS)


def _within_ranks(flies: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from scipy import stats

    races = np.array([f["race"] for f in flies])
    r1, r2 = np.zeros(len(flies)), np.zeros(len(flies))
    for k in np.unique(races):
        m = races == k
        r1[m] = stats.rankdata([f["times"][0] for f, mm in zip(flies, m) if mm])
        r2[m] = stats.rankdata([f["times"][1] for f, mm in zip(flies, m) if mm])
    return races, r1, r2


def _r(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else 0.0


def r_within(flies: list[dict]) -> float:
    _, r1, r2 = _within_ranks(flies)
    return _r(r1, r2)


def _perm_p(flies: list[dict], n: int = R4_PERMUTATIONS, seed: int = 0) -> float:
    races, r1, r2 = _within_ranks(flies)
    obs = _r(r1, r2)
    rng = np.random.default_rng(seed)
    idx = [np.flatnonzero(races == k) for k in np.unique(races)]
    hits = 0
    for _ in range(n):
        q = r2.copy()
        for ix in idx:
            q[ix] = r2[rng.permutation(ix)]
        hits += _r(r1, q) >= obs - 1e-12
    return (1 + hits) / (1 + n)


def _boot(flies: list[dict], n: int, rng) -> np.ndarray:
    races, r1, r2 = _within_ranks(flies)
    idx = [np.flatnonzero(races == k) for k in np.unique(races)]
    out = np.empty(n)
    for b in range(n):
        pick = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in idx])
        out[b] = _r(r1[pick], r2[pick])
    return out


def analyze_r4(subtle: dict, off: dict, n_perm: int = R4_PERMUTATIONS, n_boot: int = R4_BOOTSTRAP) -> dict:
    """The pre-registered R4 analysis (module docstring). subtle and off are run_r4 results of the same design."""
    rs, ro = r_within(subtle["flies"]), r_within(off["flies"])
    ps, po = _perm_p(subtle["flies"], n_perm, 0), _perm_p(off["flies"], n_perm, 0)
    rng = np.random.default_rng(1)
    bs, bo = _boot(subtle["flies"], n_boot, rng), _boot(off["flies"], n_boot, rng)
    diff = bs - bo
    lo, hi = (float(x) for x in np.percentile(diff, [2.5, 97.5]))
    crit_i, crit_ii = bool(ps < 0.05), bool(lo > 0)
    return dict(kind="race_r4_analysis", flies=len(subtle["flies"]), races=len({f["race"] for f in subtle["flies"]}),
                r_within_subtle=rs, r_within_off=ro, perm_p_subtle=ps, perm_p_off=po,
                ci_subtle=[float(x) for x in np.percentile(bs, [2.5, 97.5])], ci_off=[float(x) for x in np.percentile(bo, [2.5, 97.5])],
                difference=rs - ro, ci_difference=[lo, hi], criterion_i=crit_i, criterion_ii=crit_ii, passed=crit_i and crit_ii,
                criteria="(i) subtle r_w > 0, within-race permutation p < 0.05 (10,000, seed 0); (ii) 95% bootstrap CI of r_w(subtle) - "
                         "r_w(off) above 0 (10,000 within-race resamples, seed 1). Both needed.",
                reading=("PASS: individuality makes a fly's within-race rank repeatable beyond its brain state and noise" if crit_i and crit_ii
                         else "FAIL: at sigma 0.05 and this n the assay cannot see individuality in race speed (not evidence it has none)"))


def summary_r4(an: dict) -> str:
    from kickthefly.lab import labstats

    lo, hi = an["ci_difference"]
    return "\n".join([
        f"R4 (pre-registered, 3.0 day 4 review): individuality with the brain state held fixed; {an['flies']} flies in {an['races']} races, "
        "each race one brain-state seed, each lane its own individuality seed, each run twice",
        f"  within-race rank repeatability r_w: subtle {an['r_within_subtle']:+.3f} (95% CI {an['ci_subtle'][0]:+.3f} to {an['ci_subtle'][1]:+.3f}), "
        f"permutation {labstats.fmt_p(an['perm_p_subtle'])}",
        f"  control off (identical brains, identical state): {an['r_within_off']:+.3f} (95% CI {an['ci_off'][0]:+.3f} to {an['ci_off'][1]:+.3f}), "
        f"permutation {labstats.fmt_p(an['perm_p_off'])}",
        f"  subtle - off: {an['difference']:+.3f}, 95% CI {lo:+.3f} to {hi:+.3f}",
        f"  (i) {'PASS' if an['criterion_i'] else 'FAIL'}  (ii) {'PASS' if an['criterion_ii'] else 'FAIL'}  ->  R4 {'PASS' if an['passed'] else 'FAIL'}",
        f"  {an['reading']}",
        "  MODEL PREDICTION about this model's individuality rule (per-neuron gains, GAME RULE), not a measurement of real flies"])


def save_r4(subtle: dict, off: dict, an: dict, folder: Path | str) -> list[Path]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / "race_r4.json"
    p.write_text(json.dumps(dict(subtle=subtle, off=off, analysis=an), indent=1), encoding="utf-8")
    q = folder / "race_r4_flies.csv"
    with q.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "race", "state_seed", "individuality_seed", "time_run1_s", "time_run2_s", "finished_run1", "finished_run2"])
        for res in (subtle, off):
            for x in res["flies"]:
                w.writerow([res["individuality"], x["race"], x["state_seed"], x["individuality_seed"], *x["times"], *map(int, x["finished"])])
    r = folder / "race_r4_analysis.csv"
    with r.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "value"])
        for k in ("flies", "races", "r_within_subtle", "r_within_off", "perm_p_subtle", "perm_p_off", "difference", "criterion_i",
                  "criterion_ii", "passed"):
            w.writerow([k, an[k]])
        w.writerow(["ci_difference_low", an["ci_difference"][0]])
        w.writerow(["ci_difference_high", an["ci_difference"][1]])
    return [p, q, r]


def main_r4(args) -> int:
    """--headless --race-r4 [--workers N] [--out DIR]: the pre-registered R4 (both conditions)."""
    from kickthefly.lab import recorder

    t0 = time.time()
    prog = lambda d, n, label: print(f"  {d}/{n} {label} ({time.time() - t0:.0f}s)", flush=True)   # noqa: E731
    sub = run_r4("subtle", workers=max(1, args.workers or 1), progress=prog)
    off = run_r4("off", workers=max(1, args.workers or 1), progress=prog)
    an = analyze_r4(sub, off)
    folder = Path(args.out) if args.out else recorder.exports_dir() / f"{time.strftime('%Y%m%d-%H%M%S')}-race-r4"
    save_r4(sub, off, an, folder)
    print(summary_r4(an))
    print(f"results written to {folder}")
    return 0
