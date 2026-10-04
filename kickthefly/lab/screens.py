"""Whole-brain screens (3.1.0 tasks 6 and 7, Lab): the activation screen and the knockout screen. MODEL PREDICTION.

Two questions asked of every cell type in the connectome, with the same machinery:

  activation screen  Hold one cell type driven at a time (the optogenetic-style current every pathway test uses) and record the descending
                     neurons' response (every DN type's firing change) and what the model's behavior readouts do. "What does this
                     type make the fly do?"
  knockout screen    For each validated behavior, silence cell types one at a time (or in ranked batches of candidates, then singly
                     the batches that mattered) and rank them by how much the behavior's response drops. "What is this behavior
                     built on?"

How a run is made (the same for both, and the reason their numbers can be compared):
  - Held-out seeds. SCREEN_SEEDS (4000-4007) were never used while building or tuning anything. A seed gives one warmed-up brain; its state after
    PRE calm steps is snapshotted, and every condition for that seed starts from that exact snapshot (validation does the same), so a
    condition differs from the unperturbed run only by the perturbation, noise included.
  - An unperturbed control per seed, and matched controls per condition: random neuron sets of the same size and the same superclasses
    (the same count from each), never overlapping the perturbed type, the drive or the readout. A type's effect is its change minus its matched
    controls' change, seed by seed, tested with a paired t-test (paired_p: the Wilcoxon signed-rank test of labstats.py cannot get below
    0.0078 with 8 seeds, so no correction over a whole-brain screen could ever pass it) and corrected over the whole screen with
    Benjamini-Hochberg (q). A result that is not beyond its own matched controls is not reported as an effect.
  - Driving a readout's own neurons (DNp01 for the escape readout, MN9 for proboscis...) is not counted: that readout is excluded from the call for that type, and a
    descending type driven directly is not listed among its own responders.
  - The readouts are the validated behaviors' own neuron sets (assays.groups); a "behavior call" is only made when q < 0.05 AND the
    change is at least MIN_EFFECT_HZ AND at least MIN_SIGN_SHARE of the seeds changed the same way. The calls are MODEL PREDICTIONS: they say what this connectome simulation does, with the model's
    known failures (docs/validation.md), not what a fly does.
  - Resumable. Every finished (condition, seed) is appended to records.jsonl as it completes; running again with the same settings in the
    same folder continues; different settings in the same folder are refused (never mixed).
  - Parallel. Many CPU processes by default; with `batch` > 0 (or the gl backend) that many brains run on threads of one process and
    the GPU steps them together (the group batching of backends.py). GPU engines agree with NumPy statistically, not spike for spike; the
    engine is recorded in meta.json and every comparison is within one engine.

Output (a folder): meta.json, records.jsonl (the raw per-seed records), `<kind>_screen.csv` (one row per type or per behavior x
candidate), `<kind>_screen_responses.csv` (activation: each type's descending-neuron responders), and `.parquet` copies when pyarrow is
installed (it is optional; without it the CSVs are the output and meta.json says so).

From the connectome: which neurons are in each type, the wiring, and everything the simulation does with them. Game choices: the drive
current (0.5, as in every pathway test), the window lengths, the controls' design, the thresholds above, and what counts as a "behavior".
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

SCREEN_SEEDS = tuple(range(4000, 4008))
PRE, STIM, AMP = 400, 400, 0.5
DT = 0.005
MIN_SIGN_SHARE = 0.75               # at least this share of the seeds must change in the direction of the mean for a call
MIN_EFFECT_HZ = 1.0                  # an activation must change a readout by at least this to be called a behavior
DN_Z, DN_MIN_HZ = 4.0, 0.5           # a descending neuron type "responds" when its change is this many noise standard errors and this big
Q_MAX = 0.05
BATCH_DROP, BATCH_Q = 0.10, 0.10     # knockout batches that cut the response by this much (q below this) are opened up
VERSION = 1
TAG = "MODEL PREDICTION"

# (readout group in assays.groups, what a rise in it is called). All are validated or documented readouts.
READOUTS = (
    ("dnp01", "escape (giant fiber)"), ("dnp09", "forward walking command"), ("mdn", "backward walking command (moonwalker)"),
    ("adn", "antennal grooming command"), ("mn9", "proboscis extension"), ("mn_legs", "leg motor output"),
    ("pip10", "courtship song command"), ("ps1", "courtship song motor output"), ("dna_steer_r", "steer right"),
    ("dna_steer_l", "steer left"),
)
READOUT_NAMES = tuple(r for r, _ in READOUTS)
READOUT_LABEL = dict(READOUTS)

# The pathway behaviors that PASS in docs/validation.md (the model reproduces them); the knockout screen's default behaviors. MDN and
# the other failures are left out: a drop in something the model does not do says nothing. A test checks each is a pathway test.
VALIDATED_PATHWAYS = ("looming_escape", "sugar_feeding", "antenna_grooming_circuit", "courtship_song", "bitter_grn_to_dng28",
                      "co2_orn_to_pn", "hot_trn_to_vp2pn", "cold_trn_to_vp3pn", "optomotor_turning", "or67d_to_da1pn", "da1pn_to_lh_asp")


class ScreenError(ValueError):
    pass


# --- progress -----------------------------------------------------------------------------------------------------------------------
class Progress:
    """A one-line progress bar on stderr (a plain line every few seconds when stderr is not a terminal), or a callback(done, total, label)."""

    def __init__(self, total: int, label: str, callback=None, stream=None):
        self.total, self.label, self.callback = max(1, total), label, callback
        self.stream = stream if stream is not None else sys.stderr
        self.done0, self.done, self.t0, self._last = 0, 0, time.time(), 0.0
        self.tty = hasattr(self.stream, "isatty") and self.stream.isatty()

    def start(self, done: int) -> None:
        self.done0 = self.done = done

    def step(self, n: int = 1) -> None:
        self.done += n
        self.draw(final=self.done >= self.total)

    def line(self) -> str:
        frac = min(1.0, self.done / self.total)
        el = time.time() - self.t0
        fresh = self.done - self.done0
        eta = el / fresh * (self.total - self.done) if fresh > 0 else float("nan")
        bar = "#" * int(30 * frac) + "." * (30 - int(30 * frac))
        eta_s = "?" if math.isnan(eta) else _dur(eta)
        return f"[{bar}] {self.done}/{self.total} {frac:4.0%}  {_dur(el)} elapsed, ~{eta_s} left  {self.label}"

    def draw(self, final: bool = False) -> None:
        if self.callback:
            self.callback(self.done, self.total, self.label)
        now = time.time()
        if self.stream is None or (not final and now - self._last < (0.5 if self.tty else 5.0)):
            return
        self._last = now
        end = "\r" if self.tty and not final else "\n"
        print(self.line(), file=self.stream, end=end, flush=True)


def _dur(s: float) -> str:
    s = int(s)
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s" if s >= 60 else f"{s}s"


# --- the resumable record store -----------------------------------------------------------------------------------------------------
class Store:
    """records.jsonl (appended as each job finishes) and meta.json (what the run is: a different run in the same folder is refused)."""

    def __init__(self, folder: Path, signature: dict, restart: bool = False):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.meta_path, self.rec_path = self.folder / "meta.json", self.folder / "records.jsonl"
        self.signature = json.loads(json.dumps(signature, default=str))
        if restart:
            for p in (self.meta_path, self.rec_path):
                p.unlink(missing_ok=True)
        if self.meta_path.exists():
            old = json.loads(self.meta_path.read_text(encoding="utf-8")).get("signature")
            if old != self.signature:
                diff = sorted(k for k in set(old or {}) | set(self.signature) if (old or {}).get(k) != self.signature.get(k))
                raise ScreenError(f"{self.folder} holds a different run (differs in {', '.join(diff)}); use another --out, or restart it")
        else:
            self.meta_path.write_text(json.dumps(dict(signature=self.signature, created=time.strftime("%Y-%m-%d %H:%M:%S"), tag=TAG), indent=1),
                                      encoding="utf-8")
        self.records: list[dict] = []
        if self.rec_path.exists():
            for line in self.rec_path.read_text(encoding="utf-8").splitlines():
                try:
                    self.records.append(json.loads(line))
                except ValueError:
                    pass                                   # a line cut short by an interruption: that job runs again
        self._fh = open(self.rec_path, "a", encoding="utf-8")
        if self.rec_path.stat().st_size and not self.rec_path.read_bytes().endswith(b"\n"):
            self._fh.write("\n")                           # end the line an interruption cut short, or the next record would be glued onto it
            self._fh.flush()

    def add(self, recs: list[dict]) -> None:
        for r in recs:
            self._fh.write(json.dumps(r, separators=(",", ":")) + "\n")
        self._fh.flush()
        self.records += recs

    def update_meta(self, **kw) -> None:
        m = json.loads(self.meta_path.read_text(encoding="utf-8"))
        m.update(kw)
        self.meta_path.write_text(json.dumps(m, indent=1, default=str), encoding="utf-8")

    def close(self) -> None:
        self._fh.close()


# --- catalog and matched controls ---------------------------------------------------------------------------------------------------
def catalog(min_neurons: int = 1, max_neurons: int | None = None, superclasses=None, types=None, max_types: int | None = None) -> list[dict]:
    """The cell types of the adult pack with their sizes and (majority) superclass, in name order."""
    from kickthefly.core import simcore

    g, _, _ = simcore.pack()
    t, sc = np.asarray(g.type).astype(str), np.asarray(g.superclass).astype(str)
    out = []
    for name in sorted({str(x) for x in t[t != ""]}):
        m = t == name
        n = int(m.sum())
        if n < min_neurons or (max_neurons and n > max_neurons):
            continue
        vals, cnt = np.unique(sc[m], return_counts=True)
        s = str(vals[np.argmax(cnt)])
        if superclasses and s not in superclasses:
            continue
        out.append(dict(type=name, neurons=n, superclass=s))
    if types:
        want = list(dict.fromkeys(types))
        have = {c["type"] for c in out}
        missing = [x for x in want if x not in have]
        if missing:
            raise ScreenError(f"unknown or filtered-out cell types: {', '.join(missing[:5])}")
        out = [c for c in out if c["type"] in set(want)]
    if max_types:
        out = out[:max_types]
    return out


def _seed_of(*parts) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "little")


def matched_random(br, rows: np.ndarray, exclude: np.ndarray, seed: int) -> np.ndarray:
    """A random neuron set the size of `rows` with the same count from each superclass, never overlapping `rows` or `exclude`."""
    sc = br.superclass.astype(str)
    taken = np.zeros(br.n, bool)
    taken[rows] = True
    taken[exclude] = True
    rng = np.random.default_rng(seed)
    out = []
    for s in np.unique(sc[rows]):
        need = int(np.count_nonzero(sc[rows] == s))
        pool = np.flatnonzero((sc == s) & ~taken)
        out.append(rng.choice(pool, size=min(need, len(pool)), replace=False))
    return np.sort(np.concatenate(out)) if out else np.zeros(0, np.int64)


# --- one brain: snapshot, then conditions -------------------------------------------------------------------------------------------
class Arena:
    """A warmed brain for one seed, snapshotted after PRE calm steps. `run(...)` restores the snapshot, applies a perturbation and returns
    what the readouts did over a window. Used by both screens."""

    def __init__(self, seed: int, backend: str | None = None, pre: int = PRE):
        from kickthefly.core import savestate, simcore
        from kickthefly.lab import assays

        self.br = simcore.new_brain(seed=seed, individuality="off", backend=backend, memory=False)
        self.g = assays.groups(self.br)
        self.seed, self.pre = seed, pre
        assays.rest(self.br, pre)
        self.snap: dict = {}
        self.meta = savestate.brain_state(self.br, "s_", self.snap)
        self._savestate = savestate
        self.engine = (self.br.sim.backend.name, self.br.sim.backend.device)
        dn = self.g["dn"]
        names = np.asarray(sorted({str(x) for x in self.br.types[dn]}))
        idx = {n: i for i, n in enumerate(names)}
        self.dn_rows = dn
        self.dn_inv = np.array([idx[str(x)] for x in self.br.types[dn]], np.int64)
        self.dn_names = [str(x) for x in names]
        self.dn_sizes = np.bincount(self.dn_inv, minlength=len(names)).astype(float)

    def restore(self) -> None:
        self._savestate.restore_brain(self.br, self.meta, self.snap, "s_")

    def window(self, steps: int, readouts: dict[str, np.ndarray], dn: bool) -> tuple[dict, np.ndarray | None]:
        """Step `steps` and return ({readout: Hz}, DN type rates in Hz or None)."""
        br = self.br
        counts = {k: 0 for k in readouts}
        acc = np.zeros(len(self.dn_rows), np.int64) if dn else None
        for _ in range(steps):
            br._step()
            s = br.sim.spikes
            for k, rows in readouts.items():
                counts[k] += int(np.count_nonzero(s[rows]))
            if dn:
                acc += s[self.dn_rows]
        hz = {k: counts[k] / max(1, len(readouts[k])) / steps / DT for k in readouts}
        rates = None if acc is None else np.bincount(self.dn_inv, weights=acc, minlength=len(self.dn_names)) / np.maximum(self.dn_sizes, 1) / steps / DT
        return hz, rates

    def drive_window(self, rows: np.ndarray, steps: int, readouts, dn: bool, amp: float = AMP):
        from kickthefly.core import simcore

        self.restore()
        if len(rows):
            simcore.drive(self.br, rows, amp)
        try:
            return self.window(steps, readouts, dn)
        finally:
            if len(rows):
                simcore.undrive(self.br, rows)

    def silence_response(self, lesion: np.ndarray, drive: np.ndarray, readout: np.ndarray, pre: int = PRE, stim: int = STIM):
        """validation's pathway test under a lesion: `pre` calm steps, then `drive` held for `stim` steps. Returns (baseline Hz, driven Hz)."""
        from kickthefly.lab import assays

        self.restore()
        if len(lesion):
            self.br.set_override(lesion, -1)
        try:
            r = assays.pathway_response(self.br, drive, {"readout": readout}, pre=pre, stim=stim)["readout"]
        finally:
            if len(lesion):
                self.br.set_override(lesion, 0)
        return float(r[0]), float(r[1])


def _sparse(v: np.ndarray, floor: float = 0.02) -> list:
    nz = np.flatnonzero(np.abs(v) >= floor)
    return [[int(i), round(float(v[i]), 3)] for i in nz]


# --- the activation screen -----------------------------------------------------------------------------------------------------------
def _activation_job(job: dict) -> list[dict]:
    """All of one seed's chunk of cell types. Module level: workers pickle it."""
    seed, names, controls, backend, steps = job["seed"], job["types"], job["controls"], job["backend"], job["steps"]
    ar = Arena(seed, backend)
    br, g = ar.br, ar.g
    readouts = {k: g[k] for k in READOUT_NAMES}
    exclude = np.unique(np.concatenate([g[k] for k in READOUT_NAMES]))
    base_hz, base_dn = ar.drive_window(np.zeros(0, np.int64), steps, readouts, True)
    base_r = np.array([base_hz[k] for k in READOUT_NAMES])
    out = []
    types = br.types.astype(str)
    for name in names:
        rows = np.flatnonzero(types == name)
        hz, dn = ar.drive_window(rows, steps, readouts, True)
        rec = dict(k="act", t=name, s=seed, n=int(len(rows)), r=[round(hz[k] - base_hz[k], 3) for k in READOUT_NAMES],
                   dn=_sparse(dn - base_dn), c=[],
                   ov=[i for i, k in enumerate(READOUT_NAMES) if np.intersect1d(g[k], rows).size])   # readouts the driven neurons are part of
        for j in range(controls):
            crow = matched_random(br, rows, exclude, _seed_of("act", name, seed, j))
            chz, cdn = ar.drive_window(crow, steps, readouts, True)
            rec["c"].append(dict(r=[round(chz[k] - base_hz[k], 3) for k in READOUT_NAMES], dn=_sparse(cdn - base_dn)))
        out.append(rec)
    out.append(dict(k="base", s=seed, r=[round(float(x), 3) for x in base_r], engine=list(ar.engine), n_dn=len(ar.dn_names)))
    return out


def _chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def execute(jobs: list[dict], fn, store: Store, workers: int | None, batch: int, progress: Progress) -> None:
    """Run jobs, appending each one's records to the store as it finishes. workers > 1: processes; batch > 0: that many threads in this
    process (the gl backend steps their brains together); otherwise one after the other."""
    if not jobs:
        return
    if batch and batch > 0:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        ex = ThreadPoolExecutor(max_workers=batch)
    elif workers and workers > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed

        ex = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
    else:
        ex = None
    if ex is not None:
        try:
            futs = [ex.submit(fn, j) for j in jobs]
            for f in as_completed(futs):
                store.add(f.result())
                progress.step()
        finally:                                  # an interrupt (the Lab page's Stop) waits only for the jobs already running
            ex.shutdown(wait=True, cancel_futures=True)
    else:
        for j in jobs:
            store.add(fn(j))
            progress.step()


def run_activation(folder, seeds=SCREEN_SEEDS, types=None, min_neurons: int = 1, max_types: int | None = None, superclasses=None,
                   controls: int = 2, workers: int | None = None, batch: int = 0, backend: str | None = None, steps: int = STIM,
                   chunk: int = 25, progress=None, restart: bool = False, stream=None) -> dict:
    """The activation screen. Returns the aggregated result (also written to `folder`)."""
    seeds = tuple(int(s) for s in seeds)
    if not seeds or controls < 1 or steps < 50:
        raise ScreenError("an activation screen needs at least one seed, one matched control and 50 steps")
    cat = catalog(min_neurons, None, superclasses, types, max_types)
    sig = dict(kind="activation", version=VERSION, seeds=list(seeds), controls=controls, amp=AMP, steps=steps, pre=PRE,
               readouts=list(READOUT_NAMES), backend=backend or "auto")
    store = Store(folder, sig, restart)
    try:
        done = {(r["t"], r["s"]) for r in store.records if r.get("k") == "act"}
        jobs = []
        for s in seeds:
            todo = [c["type"] for c in cat if (c["type"], s) not in done]
            for part in _chunks(todo, chunk):
                jobs.append(dict(seed=s, types=part, controls=controls, backend=backend, steps=steps))
        total_jobs = len(jobs)
        prog = Progress(total_jobs, f"activation screen: {len(cat):,} types x {len(seeds)} seeds, {controls} matched controls each",
                        progress, stream)
        prog.start(0)
        if jobs:
            execute(jobs, _activation_job, store, workers, batch, prog)
        elif progress:
            progress(1, 1, "nothing left to run")
        res = aggregate_activation(store.records, cat, seeds)
        res["meta"] = dict(signature=sig, resumed_jobs=len(jobs) == 0, catalog_types=len(cat))
        engines = sorted({tuple(r["engine"]) for r in store.records if r.get("k") == "base"})
        store.update_meta(engines=[list(e) for e in engines], types=len(cat))
        res["engines"] = [list(e) for e in engines]
        save_activation(res, folder)
        return res
    finally:
        store.close()


def paired_p(treated, control) -> tuple[float, float]:
    """(two-sided paired t-test p, share of seeds whose change has the sign of the mean change). A screen cannot use the Wilcoxon signed-rank
    test that labstats.paired reports for single experiments: with 8 seeds its smallest possible p is 0.0078, which no correction over a
    whole-brain screen (more than 100,000 tests) can ever bring under 0.05, however large and consistent the effect. The t-test has no such
    floor; the sign share guards against one wild seed carrying it (a call needs MIN_SIGN_SHARE)."""
    from scipy import stats

    d = np.asarray(treated, float) - np.asarray(control, float)
    d = d[~np.isnan(d)]
    if len(d) < 2:
        return float("nan"), float("nan")
    m, sd = float(d.mean()), float(d.std(ddof=1))
    share = float(np.mean(np.sign(d) == np.sign(m))) if m != 0 else 0.0
    if sd == 0.0:
        return (0.0 if m != 0 else 1.0), share
    return float(2 * stats.t.sf(abs(m) / (sd / math.sqrt(len(d))), len(d) - 1)), share


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values (NaN stays NaN)."""
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    ok = np.flatnonzero(~np.isnan(p))
    if not len(ok):
        return q
    order = ok[np.argsort(p[ok])]
    m = len(ok)
    raw = p[order] * m / (np.arange(m) + 1)
    q[order] = np.minimum.accumulate(raw[::-1])[::-1].clip(max=1.0)
    return q


def _dense(sparse: list, n: int) -> np.ndarray:
    v = np.zeros(n, float)
    for i, x in sparse:
        v[i] = x
    return v


def aggregate_activation(records: list[dict], cat: list[dict], seeds) -> dict:
    """Per type: effect over matched controls on each readout (paired over seeds), the call, and the descending-neuron responders."""
    from kickthefly.lab import labstats

    acts = [r for r in records if r.get("k") == "act" and r["s"] in set(seeds)]
    bases = [r for r in records if r.get("k") == "base"]
    n_dn = max((b.get("n_dn", 0) for b in bases), default=0)
    dn_names = _dn_names()
    by_type: dict[str, list[dict]] = {}
    for r in acts:
        by_type.setdefault(r["t"], []).append(r)
    # the noise of each readout and each DN type: spread of the matched controls' changes over the whole screen
    ctl_r = np.array([c["r"] for r in acts for c in r["c"]], float).reshape(-1, len(READOUT_NAMES))
    noise_r = np.maximum(ctl_r.std(axis=0, ddof=1) if len(ctl_r) > 1 else np.ones(len(READOUT_NAMES)), 0.05)
    if n_dn:
        sq = np.zeros(n_dn)
        cnt = 0
        for r in acts:
            for c in r["c"]:
                sq += _dense(c["dn"], n_dn) ** 2
                cnt += 1
        noise_dn = np.maximum(np.sqrt(sq / max(cnt, 1)), 0.05)
    else:
        noise_dn = np.zeros(0)
    meta = {c["type"]: c for c in cat}
    rows, p_all, owners = [], [], []
    for name, recs in sorted(by_type.items()):
        recs = sorted(recs, key=lambda r: r["s"])
        a = np.array([r["r"] for r in recs], float)
        c = np.array([np.mean([x["r"] for x in r["c"]], axis=0) for r in recs], float)
        d = a - c
        row = dict(type=name, superclass=meta.get(name, {}).get("superclass", ""), neurons=recs[0]["n"], n_seeds=len(recs))
        overlap = set(recs[0].get("ov", []))                   # driving a readout's own neurons says nothing about the circuit
        row["self_readouts"] = ",".join(READOUT_NAMES[i] for i in sorted(overlap))
        for j, k in enumerate(READOUT_NAMES):
            ci = labstats.mean_ci(d[:, j])
            pv, share = paired_p(a[:, j], c[:, j]) if j not in overlap else (float("nan"), float("nan"))
            row[f"eff_{k}"], row[f"lo_{k}"], row[f"hi_{k}"], row[f"p_{k}"], row[f"sign_{k}"] = ci["mean"], ci["lo"], ci["hi"], pv, share
            row[f"z_{k}"] = ci["mean"] / (noise_r[j] / math.sqrt(len(recs)))
            p_all.append(pv)
            owners.append((len(rows), k))
        if n_dn:
            dn_a = np.array([_dense(r["dn"], n_dn) for r in recs])
            dn_c = np.array([np.mean([_dense(x["dn"], n_dn) for x in r["c"]], axis=0) for r in recs])
            mean_d = (dn_a - dn_c).mean(axis=0)
            z = mean_d / (noise_dn / math.sqrt(len(recs)))
            own = dn_names.index(name) if name in dn_names else -1                  # a DN type driven directly is not a responder
            resp = np.array([i for i in np.flatnonzero((np.abs(z) >= DN_Z) & (np.abs(mean_d) >= DN_MIN_HZ)) if i != own], int)
            order = resp[np.argsort(-np.abs(z[resp]))]
            row["dn_responders"] = int(len(resp))
            row["_dn"] = [(dn_names[i] if i < len(dn_names) else f"DN{i}", float(mean_d[i]), float(z[i])) for i in order[:25]]
            row["top_dn"] = "; ".join(f"{n} {x:+.1f}Hz" for n, x, _ in row["_dn"][:5])
        rows.append(row)
    q = bh(np.array(p_all))
    for (i, k), qv in zip(owners, q):
        rows[i][f"q_{k}"] = float(qv)
    for row in rows:
        best, best_z = None, 0.0
        for k in READOUT_NAMES:
            if (row[f"q_{k}"] < Q_MAX and abs(row[f"eff_{k}"]) >= MIN_EFFECT_HZ and row[f"sign_{k}"] >= MIN_SIGN_SHARE
                    and abs(row[f"z_{k}"]) > best_z):
                best, best_z = k, abs(row[f"z_{k}"])
        row["behavior_readout"] = best or ""
        row["behavior_call"] = (("raises " if row[f"eff_{best}"] > 0 else "lowers ") + READOUT_LABEL[best]) if best else "none"
        row["behavior_effect_hz"] = row[f"eff_{best}"] if best else 0.0
        row["behavior_q"] = row[f"q_{best}"] if best else float("nan")
    rows.sort(key=lambda r: (-abs(r["behavior_effect_hz"]), r["type"]))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return dict(kind="activation", tag=TAG, seeds=list(seeds), readouts=[dict(id=k, label=READOUT_LABEL[k]) for k in READOUT_NAMES],
                rows=rows, noise_readout_hz=dict(zip(READOUT_NAMES, map(float, noise_r))), n_types=len(rows),
                n_called=sum(1 for r in rows if r["behavior_readout"]), n_dn_types=n_dn,
                criteria=f"called when q < {Q_MAX} and |change| >= {MIN_EFFECT_HZ} Hz over matched controls; DN responder when |z| >= {DN_Z} and >= {DN_MIN_HZ} Hz")


_DN_NAMES_CACHE: list[str] | None = None


def _dn_names() -> list[str]:
    """The descending-neuron type names in Arena's order (sorted), from the pack."""
    global _DN_NAMES_CACHE
    if _DN_NAMES_CACHE is None:
        from kickthefly.core import simcore

        g, _, _ = simcore.pack()
        t, sc = np.asarray(g.type).astype(str), np.asarray(g.superclass).astype(str)
        _DN_NAMES_CACHE = sorted({str(x) for x in t[sc == "descending_neuron"]})
    return _DN_NAMES_CACHE


# --- the knockout screen ---------------------------------------------------------------------------------------------------------------
def _knockout_job(job: dict) -> list[dict]:
    seed, bid, specs, controls, backend = job["seed"], job["behavior"], job["specs"], job["controls"], job["backend"]
    from kickthefly.lab import validation

    t = validation.BY_ID[bid]
    ar = Arena(seed, backend)
    br, g = ar.br, ar.g
    drive, readout = g[t["drive"]], g[t["readout"]]
    exclude = np.unique(np.concatenate([drive, readout, *(g[x] for x in t.get("extra_readouts", ()))]))
    types = br.types.astype(str)
    out = []
    base = ar.silence_response(np.zeros(0, np.int64), drive, readout)
    out.append(dict(k="ko_base", b=bid, s=seed, base=base[0], driven=base[1], engine=list(ar.engine)))
    for spec in specs:
        rows = np.flatnonzero(np.isin(types, spec))
        base_h, driven_h = ar.silence_response(rows, drive, readout)
        rec = dict(k="ko", b=bid, s=seed, spec=list(spec), n=int(len(rows)), base=base_h, driven=driven_h, c=[])
        for j in range(controls):
            crow = matched_random(br, rows, exclude, _seed_of("ko", bid, "+".join(spec), seed, j))
            cb, cd = ar.silence_response(crow, drive, readout)
            rec["c"].append([cb, cd])
        out.append(rec)
    return out


def _candidates(bid: str, top: int, backend_seed: int) -> list[dict]:
    """The shortlist for one behavior (criticalpath.shortlist): the types that can reach the readout in one or two hops."""
    from kickthefly.core import simcore
    from kickthefly.lab import criticalpath

    br = simcore.new_brain(seed=backend_seed, warmup=0, memory=False)
    return criticalpath.shortlist(br, bid, top)


def run_knockout(folder, behaviors=None, seeds=SCREEN_SEEDS, top: int = 25, batch_size: int = 0, controls: int = 2,
                 workers: int | None = None, batch: int = 0, backend: str | None = None, types=None, progress=None,
                 restart: bool = False, stream=None) -> dict:
    """The knockout screen. `behaviors`: validation test ids (default: the validated pathways). `batch_size` > 0: silence ranked candidates
    in batches of that many first, then singly the members of the batches that cut the response."""
    from kickthefly.lab import validation

    behaviors = tuple(behaviors) if behaviors else VALIDATED_PATHWAYS
    bad = [b for b in behaviors if b not in VALIDATED_PATHWAYS or b not in validation.BY_ID]
    if bad:
        raise ScreenError(f"not a validated pathway behavior: {', '.join(bad)} (the screen runs {', '.join(VALIDATED_PATHWAYS)}; a behavior "
                          "the model does not reproduce, docs/validation.md, would only show what silencing does to noise)")
    seeds = tuple(int(s) for s in seeds)
    if not seeds or controls < 1 or top < 1 or batch_size < 0:
        raise ScreenError("a knockout screen needs a seed, a matched control, at least one candidate and a non-negative batch size")
    sig = dict(kind="knockout", version=VERSION, seeds=list(seeds), controls=controls, amp=AMP, pre=PRE, stim=STIM, top=top,
               behaviors=list(behaviors), batch_size=batch_size, types=sorted(types) if types else None, backend=backend or "auto")
    store = Store(folder, sig, restart)
    try:
        cands: dict[str, list[dict]] = {}
        saved = store.folder / "candidates.json"
        if saved.exists():
            cands = json.loads(saved.read_text(encoding="utf-8"))
        for b in behaviors:
            if b not in cands:
                cands[b] = ([dict(type=x, neurons=0, influence=float("nan"), drives_the_behavior=False) for x in types] if types
                            else _candidates(b, top, seeds[0]))
        saved.write_text(json.dumps(cands, indent=1, default=str), encoding="utf-8")
        total_est = len(behaviors) * len(seeds)

        def specs_stage1(b):
            names = [c["type"] for c in cands[b]]
            return [names[i:i + batch_size] for i in range(0, len(names), batch_size)] if batch_size > 1 else [[n] for n in names]

        def plan(stage: dict[str, list[list[str]]]) -> list[dict]:
            done = {(r["b"], r["s"], tuple(r["spec"])) for r in store.records if r.get("k") == "ko"}
            jobs = []
            for b, specs in stage.items():
                for s in seeds:
                    todo = [sp for sp in specs if (b, s, tuple(sp)) not in done]
                    for part in _chunks(todo, 10):
                        jobs.append(dict(seed=s, behavior=b, specs=part, controls=controls, backend=backend))
            return jobs

        stage1 = {b: specs_stage1(b) for b in behaviors}
        jobs = plan(stage1)
        prog = Progress(len(jobs) + total_est, f"knockout screen: {len(behaviors)} behaviors x {len(seeds)} seeds", progress, stream)
        prog.start(0)
        execute(jobs, _knockout_job, store, workers, batch, prog)
        if batch_size > 1:                                   # open the batches that cut the response
            opened = _open_batches(store.records, behaviors, seeds, cands)
            stage2 = {b: [[t] for t in members] for b, members in opened.items() if members}
            prog.total += len(plan(stage2))
            execute(plan(stage2), _knockout_job, store, workers, batch, prog)
        res = aggregate_knockout(store.records, behaviors, seeds, cands, batch_size)
        res["meta"] = dict(signature=sig)
        engines = sorted({tuple(r["engine"]) for r in store.records if r.get("k") == "ko_base"})
        store.update_meta(engines=[list(e) for e in engines])
        res["engines"] = [list(e) for e in engines]
        save_knockout(res, folder)
        return res
    finally:
        store.close()


def _evoked(rec: dict) -> float:
    return rec["driven"] - rec["base"]


def _knock_table(records, bid, seeds):
    """{spec tuple: [per-seed (lesion evoked, mean matched-control evoked)]} and the unperturbed evoked response per seed."""
    base = {r["s"]: r["driven"] - r["base"] for r in records if r.get("k") == "ko_base" and r["b"] == bid}
    out: dict[tuple, dict[int, tuple[float, float]]] = {}
    for r in records:
        if r.get("k") != "ko" or r["b"] != bid or r["s"] not in set(seeds):
            continue
        ctl = float(np.mean([cd - cb for cb, cd in r["c"]]))
        out.setdefault(tuple(r["spec"]), {})[r["s"]] = (_evoked(r), ctl)
    return base, out


def _open_batches(records, behaviors, seeds, cands) -> dict[str, list[str]]:
    from kickthefly.lab import labstats

    opened = {}
    for b in behaviors:
        base, table = _knock_table(records, b, seeds)
        names, ps, drops = [], [], []
        for spec, per in table.items():
            if len(spec) < 2 or len(per) < 2:
                continue
            ss = sorted(per)
            les = np.array([per[s][0] for s in ss])
            ctl = np.array([per[s][1] for s in ss])
            b0 = float(np.mean([base[s] for s in ss if s in base])) or float("nan")
            names.append(spec)
            drops.append((float(ctl.mean()) - float(les.mean())) / b0 if b0 else float("nan"))
            ps.append(paired_p(ctl, les)[0])
        q = bh(np.array(ps)) if ps else np.array([])
        members = []
        for spec, d, qv in zip(names, drops, q):
            if d >= BATCH_DROP and qv < BATCH_Q:
                members += list(spec)
        opened[b] = members
    return opened


def aggregate_knockout(records, behaviors, seeds, cands, batch_size: int = 0) -> dict:
    from kickthefly.lab import labstats

    rows, p_all, owners = [], [], []
    summary = {}
    for b in behaviors:
        base, table = _knock_table(records, b, seeds)
        b0s = [base[s] for s in sorted(base)]
        b0 = float(np.mean(b0s)) if b0s else float("nan")
        summary[b] = dict(unperturbed_evoked_hz=b0, seeds=len(b0s))
        cmeta = {c["type"]: c for c in cands.get(b, [])}
        for spec, per in table.items():
            ss = sorted(per)
            les = np.array([per[s][0] for s in ss])
            ctl = np.array([per[s][1] for s in ss])
            ci = labstats.mean_ci(ctl - les)                  # the response the matched controls kept, minus what this lesion kept
            pv, share = paired_p(ctl, les)                   # a drop: the controls kept more than the lesion did
            single = len(spec) == 1
            rows.append(dict(behavior=b, candidate="+".join(spec), is_batch=not single, members=len(spec), neurons=_neurons(records, b, spec),
                             n_seeds=len(ss), unperturbed_evoked_hz=b0, lesion_evoked_hz=float(les.mean()), control_evoked_hz=float(ctl.mean()),
                             drop_hz=ci["mean"], drop_lo=ci["lo"], drop_hi=ci["hi"], drop_share=(ci["mean"] / b0) if b0 else float("nan"),
                             p=pv, sign_share=share, drives_the_behavior=bool(single and cmeta.get(spec[0], {}).get("drives_the_behavior")),
                             influence=cmeta.get(spec[0], {}).get("influence", float("nan")) if single else float("nan")))
            p_all.append(pv)
            owners.append(len(rows) - 1)
    q = bh(np.array(p_all))
    for i, qv in zip(owners, q):
        rows[i]["q"] = float(qv)
    singles = [r for r in rows if not r["is_batch"]]
    for b in behaviors:
        rb = sorted((r for r in singles if r["behavior"] == b), key=lambda r: -(r["drop_share"] if r["drop_share"] == r["drop_share"] else -9))
        for i, r in enumerate(rb, 1):
            r["rank"] = i
        summary[b]["tested"] = len(rb)
        summary[b]["significant"] = sum(1 for r in rb if r["q"] < Q_MAX and r["drop_share"] >= 0.1 and r["sign_share"] >= MIN_SIGN_SHARE)
    for r in rows:
        r.setdefault("rank", None)
    rows.sort(key=lambda r: (r["behavior"], r["is_batch"], -(r["drop_share"] if r["drop_share"] == r["drop_share"] else -9)))
    return dict(kind="knockout", tag=TAG, seeds=list(seeds), behaviors=list(behaviors), rows=rows, summary=summary, batch_size=batch_size,
                criteria=f"a drop is reported when the response is below its matched controls' (paired t-test over seeds, q < {Q_MAX} across the screen, and {MIN_SIGN_SHARE:.0%} of seeds the same way); "
                         f"batches of {batch_size} are opened when they cut the response by >= {BATCH_DROP:.0%} (q < {BATCH_Q})"
                         if batch_size > 1 else f"a drop is reported when the response is below its matched controls' (paired t-test over seeds, q < {Q_MAX} across the screen, and {MIN_SIGN_SHARE:.0%} of seeds the same way)")


def _neurons(records, bid, spec) -> int:
    for r in records:
        if r.get("k") == "ko" and r["b"] == bid and tuple(r["spec"]) == tuple(spec):
            return int(r["n"])
    return 0


# --- output ------------------------------------------------------------------------------------------------------------------------------
def _write_rows(rows: list[dict], path: Path, drop=("_dn",)) -> list[Path]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = [{k: v for k, v in r.items() if k not in drop} for r in rows]
    cols = list(clean[0]) if clean else []
    with open(path.with_suffix(".csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in clean:
            w.writerow([_fmt(r.get(c)) for c in cols])
    written = [path.with_suffix(".csv")]
    pq = _parquet(clean, path.with_suffix(".parquet"))
    if pq:
        written.append(pq)
    return written


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return "" if math.isnan(v) else f"{v:.5g}"
    return v


def _parquet(rows: list[dict], path: Path) -> Path | None:
    """Parquet beside the CSV when pyarrow is installed (optional). Returns the path, or None."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception:
        return None
    if not rows:
        return None
    try:
        pq.write_table(pa.Table.from_pylist(rows), str(path))
    except Exception as e:                       # a column pyarrow cannot type must never lose the run: the CSV is already written
        print(f"note: parquet not written ({type(e).__name__}: {e}); the CSV holds everything", file=sys.stderr)
        return None
    return path


def save_activation(res: dict, folder) -> list[Path]:
    folder = Path(folder)
    written = _write_rows(res["rows"], folder / "activation_screen")
    resp = [dict(type=r["type"], dn_type=n, delta_hz=d, z=z, n_seeds=r["n_seeds"]) for r in res["rows"] for n, d, z in r.get("_dn", [])]
    written += _write_rows(resp, folder / "activation_screen_responses")
    (folder / "activation_screen.json").write_text(json.dumps({k: v for k, v in res.items() if k != "rows"} | dict(
        rows=[{k: v for k, v in r.items()} for r in res["rows"]]), indent=1, default=str), encoding="utf-8")
    note = None if any(p.suffix == ".parquet" for p in written) else "parquet skipped: pyarrow is not installed (pip install pyarrow); the CSVs hold everything"
    if (folder / "meta.json").exists():
        m = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        m.update(outputs=[p.name for p in written], parquet_note=note, tag=TAG)
        (folder / "meta.json").write_text(json.dumps(m, indent=1, default=str), encoding="utf-8")
    return written


def save_knockout(res: dict, folder) -> list[Path]:
    folder = Path(folder)
    written = _write_rows(res["rows"], folder / "knockout_screen")
    (folder / "knockout_screen.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    note = None if any(p.suffix == ".parquet" for p in written) else "parquet skipped: pyarrow is not installed (pip install pyarrow); the CSVs hold everything"
    if (folder / "meta.json").exists():
        m = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        m.update(outputs=[p.name for p in written], parquet_note=note, tag=TAG)
        (folder / "meta.json").write_text(json.dumps(m, indent=1, default=str), encoding="utf-8")
    return written


def load(folder) -> dict | None:
    """A finished screen from its folder (what the Lab page shows), or None."""
    folder = Path(folder)
    for name in ("activation_screen.json", "knockout_screen.json"):
        p = folder / name
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return None
    return None


def summary(res: dict, top: int = 12) -> str:
    if res["kind"] == "activation":
        lines = [f"Activation screen ({TAG}): {res['n_types']:,} cell types, seeds {res['seeds'][0]}-{res['seeds'][-1]}, "
                 f"{res['n_called']} with a behavior call. {res['criteria']}"]
        for r in res["rows"][:top]:
            if r["behavior_readout"]:
                lines.append(f"  {r['rank']:3d}. {r['type']:<18}{r['neurons']:5d} neurons  {r['behavior_call']:<44} {r['behavior_effect_hz']:+7.1f} Hz  "
                             f"q={r['behavior_q']:.3g}  DN responders {r.get('dn_responders', 0)}")
        return "\n".join(lines)
    lines = [f"Knockout screen ({TAG}): {len(res['behaviors'])} behaviors, seeds {res['seeds'][0]}-{res['seeds'][-1]}. {res['criteria']}"]
    for b in res["behaviors"]:
        s = res["summary"][b]
        lines.append(f"  {b}: unperturbed response {s['unperturbed_evoked_hz']:.2f} Hz, {s['tested']} types tested, {s['significant']} cut it by >= 10% (q < {Q_MAX})")
        for r in [x for x in res["rows"] if x["behavior"] == b and not x["is_batch"]][:3]:
            lines.append(f"      {r['rank']}. {r['candidate']:<16} drop {r['drop_share']:+.0%}  q={r['q']:.3g}")
    return "\n".join(lines)
