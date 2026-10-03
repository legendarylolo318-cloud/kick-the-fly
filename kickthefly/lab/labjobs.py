"""Background Lab jobs: assays over many flies (seeds), each fly in a worker process, with summaries and statistics.

Repeated trials: for any surgery (silencing or stimulating neurons), every perturbed fly is paired with an unperturbed
fly of the same seed, and the two are compared with paired tests (labstats.py).
"""
from __future__ import annotations

import multiprocessing
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

import numpy as np

from kickthefly.lab import labstats

ASSAYS = ("tmaze", "looming", "sugar", "orchard", "thermo_escape", "predator_escape", "hum_demo", "sleep_deprivation")
ASSAY_LABEL = {"tmaze": "T-maze conditioning", "looming": "Looming escape", "sugar": "Sugar response",
               "orchard": "Orchard feeding", "thermo_escape": "Thermogenetic escape",
               "predator_escape": "Predator escape", "hum_demo": "Hum demo (JO-A/B)",
               "sleep_deprivation": "Sleep deprivation"}


def default_workers() -> int:
    return max(1, min(4, (os.cpu_count() or 2) - 1))


def assay_task(kind: str, seed: int, options: dict, surgery: dict | None, params: dict | None) -> dict:
    """One fly. Runs in a worker process (top level, so it pickles)."""
    from kickthefly.lab import assays

    opts = dict(options or {})
    if kind == "tmaze":
        a = assays.tmaze_fly(seed, "odor_a", surgery=surgery, params=params, **opts)
        b = assays.tmaze_fly(seed + 50_000, "odor_b", surgery=surgery, params=params, **opts)
        return dict(seed=seed, pi=(a["pi"] + b["pi"]) / 2, fear_plus=(a["fear_plus"] + b["fear_plus"]) / 2,
                    fear_minus=(a["fear_minus"] + b["fear_minus"]) / 2,
                    mbon_plus_hz=(a["mbon_plus_hz"] + b["mbon_plus_hz"]) / 2,
                    mbon_minus_hz=(a["mbon_minus_hz"] + b["mbon_minus_hz"]) / 2, halves=[a, b])
    if kind == "looming":
        return assays.looming_fly(seed, surgery=surgery, params=params, **opts)
    if kind == "sugar":
        return assays.sugar_fly(seed, surgery=surgery, params=params, **opts)
    if kind == "orchard":
        return assays.orchard_fly(seed, surgery=surgery, params=params, **opts)
    if kind == "thermo_escape":
        from kickthefly.lab import thermogenetics

        return thermogenetics.escape_fly(seed, surgery=surgery, params=params, **opts)
    if kind == "predator_escape":
        from kickthefly.lab import predators

        return predators.escape_fly(seed, surgery=surgery, params=params, **opts)
    if kind == "hum_demo":
        from kickthefly.lab import audio

        return audio.demo_fly(seed, params=params, **opts)
    if kind == "sleep_deprivation":
        from kickthefly.lab import sleepdep

        if surgery:
            raise ValueError("the sleep deprivation assay is its own paired design: it takes no surgery")
        return sleepdep.fly_pair(seed, params=params, **opts)
    raise ValueError(kind)


HEADLINE_LABEL = {"tmaze": "performance index", "looming": "escape probability (all speeds)",
                  "sugar": "MN9 ratio (all doses)", "orchard": "MN9 ratio, feeding vs travelling",
                  "thermo_escape": "escape probability (all temperatures, expressing flies)",
                  "predator_escape": "escape probability (all predators)",
                  "hum_demo": "P1 firing ratio, 200 Hz pulse train",
                  "sleep_deprivation": "rebound sleep, deprived minus control (s)"}


def headline(kind: str, fly: dict) -> float:
    """One fly's single headline number, the same one summarize() puts in per_fly."""
    if kind == "tmaze":
        return float(fly["pi"])
    if kind == "looming":
        return float(np.mean([np.mean([t["escaped"] for t in tr]) for tr in fly["trials"].values()]))
    if kind == "sugar":
        return float(np.mean([np.mean([o["ratio"] for o in offs]) for offs in fly["offers"].values()]))
    if kind == "orchard":
        return float(fly["mn9_ratio"])
    if kind == "thermo_escape":
        return float(np.mean([t["expressing"]["escaped"] for t in fly["trials"].values()]))
    if kind == "predator_escape":
        return float(np.mean([np.mean([t["escaped"] for t in tr]) for tr in fly["trials"].values()]))
    if kind == "hum_demo":
        from kickthefly.lab import audio

        return audio.headline(fly)
    if kind == "sleep_deprivation":
        return float(fly["deprived"]["sleep_s"]["recovery"] - fly["control"]["sleep_s"]["recovery"])
    raise ValueError(kind)


def summarize(kind: str, flies: list[dict]) -> dict:
    """Standard metrics with mean and 95% CI across flies."""
    if kind == "tmaze":
        return dict(metric="performance index (Tully & Quinn)", pi=labstats.mean_ci([f["pi"] for f in flies]),
                    fear_cs_plus=labstats.mean_ci([f["fear_plus"] for f in flies]),
                    fear_cs_minus=labstats.mean_ci([f["fear_minus"] for f in flies]),
                    mbon_cs_plus_hz=labstats.mean_ci([f["mbon_plus_hz"] for f in flies]),
                    mbon_cs_minus_hz=labstats.mean_ci([f["mbon_minus_hz"] for f in flies]),
                    per_fly=[f["pi"] for f in flies])
    if kind == "looming":
        speeds = flies[0]["speeds"]
        rows = []
        for v in speeds:
            key = v if v in flies[0]["trials"] else str(v)
            probs = [np.mean([t["escaped"] for t in f["trials"][key]]) for f in flies]
            lat = [t["latency_s"] for f in flies for t in f["trials"][key] if t["escaped"]]
            dist = [t["distance_m"] for f in flies for t in f["trials"][key] if t["escaped"]]
            yes = sum(t["escaped"] for f in flies for t in f["trials"][key])
            n = sum(len(f["trials"][key]) for f in flies)
            rows.append(dict(speed=v, escape_probability=labstats.mean_ci(probs), escapes=yes, approaches=n,
                             latency_s=labstats.mean_ci(lat), distance_m=labstats.mean_ci(dist)))
        return dict(metric="escape probability and latency vs approach speed", rows=rows,
                    per_fly=[float(np.mean([np.mean([t["escaped"] for t in tr]) for tr in f["trials"].values()]))
                             for f in flies])
    if kind == "thermo_escape":
        from kickthefly.lab import thermogenetics

        return thermogenetics.summarize(flies)
    if kind == "predator_escape":
        from kickthefly.lab import predators

        return predators.summarize(flies)
    if kind == "hum_demo":
        from kickthefly.lab import audio

        return audio.summarize(flies)
    if kind == "sleep_deprivation":
        from kickthefly.lab import sleepdep

        return sleepdep.summarize(flies)
    if kind == "orchard":
        return dict(metric="orchard feeding: MN9 and PAM feeding vs travelling, and the orchard's depletion",
                    mn9_ratio=labstats.mean_ci([f["mn9_ratio"] for f in flies]),
                    pam_ratio=labstats.mean_ci([f["pam_ratio"] for f in flies]),
                    feeds=labstats.mean_ci([f["feeds"] for f in flies]),
                    fruit_emptied=labstats.mean_ci([f["fruit_emptied"] for f in flies]),
                    fermented_feeds=labstats.mean_ci([f["fermented_feeds"] for f in flies]),
                    settings=dict(feeds_per_fruit=flies[0]["feeds_per_fruit"], regrow_s=flies[0]["regrow_s"],
                                  cap=flies[0]["cap"], duration_s=flies[0]["duration_s"]),
                    per_fly=[f["mn9_ratio"] for f in flies])
    if kind == "sugar":
        doses = flies[0]["doses"]
        rows = []
        for d in doses:
            key = d if d in flies[0]["offers"] else str(d)
            ratio = [np.mean([o["ratio"] for o in f["offers"][key]]) for f in flies]
            hz = [np.mean([o["mn9_hz"] for o in f["offers"][key]]) for f in flies]
            ext = [np.mean([o["extended"] for o in f["offers"][key]]) for f in flies]
            yes = sum(o["extended"] for f in flies for o in f["offers"][key])
            n = sum(len(f["offers"][key]) for f in flies)
            rows.append(dict(dose=d, mn9_ratio=labstats.mean_ci(ratio), mn9_hz=labstats.mean_ci(hz),
                             extension_probability=labstats.mean_ci(ext), extensions=yes, offers=n))
        return dict(metric="MN9 dose-response", rows=rows,
                    per_fly=[float(np.mean([np.mean([o["ratio"] for o in offs]) for offs in f["offers"].values()]))
                             for f in flies])
    raise ValueError(kind)


def compare(kind: str, treated: dict, control: dict) -> dict:
    """Paired statistics between a surgery and its same-seed unperturbed control."""
    out = dict(overall=labstats.paired(treated["per_fly"], control["per_fly"]))
    if kind in ("looming", "sugar"):
        rows = []
        tk, ck = ("escapes", "approaches") if kind == "looming" else ("extensions", "offers")
        for rt, rc in zip(treated["rows"], control["rows"]):
            rows.append(dict(level=rt.get("speed", rt.get("dose")),
                             fisher_p=labstats.fisher(rt[tk], rt[ck], rc[tk], rc[ck])))
        out["per_level"] = rows
    return out


class Job:
    """Runs an assay over seeds (and the matched controls) in worker processes, off the game's thread."""

    def __init__(self, kind: str, seeds, options: dict | None = None, surgery: dict | None = None,
                 params: dict | None = None, workers: int | None = None):
        self.kind, self.seeds, self.options = kind, list(seeds), dict(options or {})
        self.surgery, self.params = surgery or None, params
        self.workers = workers or default_workers()
        self.total = len(self.seeds) * (2 if self.surgery else 1)
        self.done = 0
        self.result: dict | None = None
        self.error: str | None = None
        self.cancelled = False
        self.started = time.time()
        self.thread = threading.Thread(target=self._run, name=f"lab-{kind}", daemon=True)

    def start(self) -> "Job":
        self.thread.start()
        return self

    @property
    def running(self) -> bool:
        return self.thread.is_alive()

    def _run(self) -> None:
        try:
            self.result = run_sync(self.kind, self.seeds, self.options, self.surgery, self.params, self.workers,
                                   progress=self._progress, cancelled=lambda: self.cancelled)
        except Exception as e:                        # shown on the results screen
            self.error = f"{type(e).__name__}: {e}"

    def _progress(self, done: int, total: int) -> None:
        self.done = done


def run_sync(kind, seeds, options=None, surgery=None, params=None, workers: int = 1, progress=None,
             cancelled=lambda: False) -> dict:
    tasks = [(kind, s, options, surgery, params, "treated") for s in seeds]
    if surgery:
        tasks += [(kind, s, options, None, params, "control") for s in seeds]
    out = {"treated": {}, "control": {}}
    done = 0
    notes = []
    if workers > 1:
        ctx = multiprocessing.get_context("spawn")          # never fork a process that has brain and SDL threads
        try:
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
                futs = {ex.submit(assay_task, k, s, o, sg, p): (group, s) for k, s, o, sg, p, group in tasks}
                for f in as_completed(futs):
                    if cancelled():
                        for other in futs:
                            other.cancel()
                        raise RuntimeError("cancelled")
                    group, s = futs[f]
                    out[group][s] = f.result()
                    done += 1
                    if progress:
                        progress(done, len(tasks))
        except BrokenProcessPool as e:                     # workers couldn't start or died: finish in this process
            from kickthefly.core.crash import log
            log.warning("Lab worker processes failed (%s); running the rest in-process", e)
            notes.append("worker processes failed; ran in one process instead")
    for k, s, o, sg, p, group in tasks:
        if s in out[group]:
            continue
        if cancelled():
            raise RuntimeError("cancelled")
        out[group][s] = assay_task(k, s, o, sg, p)
        done += 1
        if progress:
            progress(done, len(tasks))
    treated = summarize(kind, [out["treated"][s] for s in seeds])
    res = dict(kind=kind, label=ASSAY_LABEL[kind], seeds=list(seeds), options=options or {}, surgery=surgery,
               params=params, treated=treated, flies=[out["treated"][s] for s in seeds], notes=notes)
    if surgery:
        control = summarize(kind, [out["control"][s] for s in seeds])
        res["control"] = control
        res["control_flies"] = [out["control"][s] for s in seeds]
        res["comparison"] = compare(kind, treated, control)
    return res
