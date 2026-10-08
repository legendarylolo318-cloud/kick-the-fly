"""Critical path finder: silence one cell type at a time and rank them by how much the behavior changes.

Brain surgery answers "what happens if I switch this off". This answers the other direction: given a behavior the
simulation reproduces, which cell types is it actually built on? Every candidate type is silenced on its own, the
behavior is re-run, and the types are ranked by effect size against an unperturbed fly of the same seed.

  target       a validated behavior (validation.py: looming_escape, sugar_feeding, ...) or a standard assay
               (tmaze, looming, sugar)
  candidates   cell types ranked by how much of the readout's input they can reach in one or two synaptic hops.
               Testing all ~9,000 annotated types would take days, so the search is narrowed this way, and the
               shortlist is part of the answer: a type that is not in it was never tried.
  effect       mean lesioned minus unperturbed over the seeds, as a share of the unperturbed value, with a 95%
               confidence interval and a paired Wilcoxon signed-rank test over the same seeds (labstats.py)

Runs are resumable: every finished type is appended to the output JSON, and re-running with the same target and
seeds picks up where it stopped.

From the connectome: the types themselves, which neurons are in them, and the wiring the shortlist is computed
from. Game choices: how many types to try, how the shortlist is ranked, and that "effect" means a change in the
game's readout of a behavior.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

DEFAULT_TOP = 25
MIN_TYPE_SIZE = 1


def _readout_rows(br, test_id: str):
    from kickthefly.lab import assays, validation

    t = validation.BY_ID[test_id]
    g = assays.groups(br)
    return g[t["readout"]], g[t["drive"]]


def shortlist(br, test_id: str, top: int = DEFAULT_TOP) -> list[dict]:
    """Cell types ranked by how much of the readout's input they supply, directly or one hop further back.

    score(neuron) = |W[readout, neuron]|.sum()  +  (|W[readout, :]| @ |W|)[neuron]

    W is the rate-normalized signed matrix the simulation runs on, so a type scores highly when its synapses make
    up a real share of what the readout neurons (or the neurons feeding them) receive. Summed over each type's
    neurons, so a large type can outrank a small one: that is the point, a type is silenced as a whole.
    """
    import scipy.sparse as sp

    readout, drive = _readout_rows(br, test_id)
    W = br.sim.W_csr
    absW = sp.csr_array((np.abs(W.data), W.indices, W.indptr), shape=W.shape)
    direct = np.asarray(absW[readout, :].sum(axis=0)).ravel()
    indirect = np.asarray(absW.T @ direct).ravel() if len(readout) else np.zeros(br.n)
    score = direct + indirect
    types = br.types.astype(str)
    out = {}
    for t, s in zip(types, score):
        if not t:
            continue
        e = out.setdefault(t, [0.0, 0])
        e[0] += float(s)
        e[1] += 1
    drive_types = {str(t) for t in types[drive] if t}
    ranked = sorted(out.items(), key=lambda kv: -kv[1][0])
    picked, seen = [], set()
    for name, (s, n) in ranked:
        if n < MIN_TYPE_SIZE or s <= 0:
            continue
        picked.append(dict(type=name, neurons=n, influence=s, drives_the_behavior=name in drive_types))
        seen.add(name)
        if len(picked) >= top:
            break
    for name in sorted(drive_types - seen):            # the type being driven is always worth testing
        n = int(np.count_nonzero(types == name))
        picked.append(dict(type=name, neurons=n, influence=float(out.get(name, [0.0])[0]),
                           drives_the_behavior=True))
    return picked


# --- one measurement ---------------------------------------------------------------------------------------------
def behavior_seed(args) -> dict:
    """One fly of a validated behavior, with an optional lesion. Module level, so workers can pickle it.

    Returns the readout's baseline and driven firing and validation's ratio between them. The ranking uses the
    driven rate: the ratio has a floor under its denominator (it is there so a silent baseline cannot divide by
    zero), and a lesion that quietens the whole readout would show up as a huge ratio rather than as damage.
    """
    seed, test_id, lesion = args
    from kickthefly.core import simcore
    from kickthefly.lab import assays, validation

    t = validation.BY_ID[test_id]
    br = simcore.new_brain(seed=seed)
    if lesion:
        assays.apply_surgery(br, {lesion: -1})
    g = assays.groups(br)
    r = assays.pathway_response(br, g[t["drive"]], {"readout": g[t["readout"]]}, pre=validation.PRE,
                                stim=validation.STIM)["readout"]
    return dict(score=float(r[1]), base_hz=float(r[0]), driven_hz=float(r[1]),
                ratio=float(r[1] / max(r[0], 0.5)))


def assay_seed(args) -> dict:
    """One fly of a standard assay, with an optional lesion. Returns the assay's single headline number."""
    seed, kind, lesion = args
    from kickthefly.lab import labjobs

    res = labjobs.assay_task(kind, seed, None, {lesion: -1} if lesion else None, None)
    return dict(score=float(labjobs.headline(kind, res)))


# --- the sweep ---------------------------------------------------------------------------------------------------
def _run_many(fn, jobs, workers, progress, done0=0, total=1):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    out = {}
    done = done0
    if workers and workers > 1:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs = {ex.submit(fn, j): j for j in jobs}
            for f in as_completed(futs):
                j = futs[f]
                out[j] = f.result()
                done += 1
                if progress:
                    progress(done, total)
    else:
        for j in jobs:
            out[j] = fn(j)
            done += 1
            if progress:
                progress(done, total)
    return out, done


def run(target: str, seeds=None, top: int = DEFAULT_TOP, workers: int | None = None, progress=None,
        resume: Path | None = None, types: list[str] | None = None) -> dict:
    """Silence each candidate type in turn and rank them. `target` is a validation test id or an assay name."""
    from kickthefly.core import simcore
    from kickthefly.lab import labjobs, labstats, validation

    is_assay = target in labjobs.ASSAYS
    seeds = tuple(seeds) if seeds else validation.SEEDS
    t0 = time.time()
    state = _load_resume(resume, target, seeds)
    if types:
        cands = [dict(type=t, neurons=0, influence=float("nan"), drives_the_behavior=False) for t in types]
    elif state and state.get("candidates"):
        cands = state["candidates"]
    else:
        br = simcore.new_brain(seed=seeds[0], warmup=0)
        cands = shortlist(br, "looming_escape" if is_assay else target, top)
        del br
    fn = assay_seed if is_assay else behavior_seed
    todo = [c for c in cands if c["type"] not in state.get("results", {})]
    total = len(seeds) * (len(todo) + (0 if state.get("control") else 1))
    done = 0

    control = state.get("control")
    if not control:
        jobs = [(s, target, None) for s in seeds]
        got, done = _run_many(fn, jobs, workers, lambda d, n: progress and progress(d, total, "unperturbed control"),
                              done, total)
        control = {str(s): got[(s, target, None)] for s in seeds}
        state["control"] = control
        _save_resume(resume, target, seeds, cands, state)

    results = dict(state.get("results", {}))
    base = [control[str(s)] for s in seeds]
    for c in todo:
        spec = f"type:{c['type']}"
        jobs = [(s, target, spec) for s in seeds]
        got, done = _run_many(fn, jobs, workers,
                              lambda d, n, name=c["type"]: progress and progress(d, total, f"silencing {name}"),
                              done, total)
        vals = [got[(s, target, spec)] for s in seeds]
        results[c["type"]] = _score(c, vals, base, labstats)
        state["results"] = results
        _save_resume(resume, target, seeds, cands, state)


    ranked = sorted(results.values(), key=lambda r: -abs(r["effect_share"]))
    for i, r in enumerate(ranked, 1):
        r["rank"] = i
    return dict(kind="critical_path", target=target, is_assay=is_assay, created=time.strftime("%Y-%m-%d %H:%M:%S"),
                seconds=round(time.time() - t0, 1), seeds=list(seeds), top=top,
                metric=labjobs.HEADLINE_LABEL[target] if is_assay else "readout firing while driven (spikes/s)",
                control=dict(mean=float(np.mean([x["score"] for x in base])),
                             per_seed=[x["score"] for x in base]), candidates=cands, ranked=ranked)


def _score(cand: dict, vals, base, labstats) -> dict:
    """One type's effect: lesioned vs the unperturbed fly of the same seed, paired over seeds."""
    v = [x["score"] for x in vals]
    b_ = [x["score"] for x in base]
    ci = labstats.mean_ci(v)
    diff = labstats.paired(v, b_)
    b = float(np.mean(b_))
    extra = {}
    for key in ("base_hz", "driven_hz", "ratio"):
        if key in vals[0]:
            extra[f"lesioned_{key}"] = float(np.mean([x[key] for x in vals]))
            extra[f"control_{key}"] = float(np.mean([x[key] for x in base]))
    return dict(type=cand["type"], neurons=cand.get("neurons", 0), influence=cand.get("influence", float("nan")),
                drives_the_behavior=bool(cand.get("drives_the_behavior")),
                lesioned_mean=ci["mean"], lesioned_ci=[ci["lo"], ci["hi"]], control_mean=b,
                effect=ci["mean"] - b, effect_share=(ci["mean"] - b) / b if b else float("nan"),
                p_value=diff.get("p_value", float("nan")), n=ci["n"], per_seed=v, **extra)


# --- resumable runs ----------------------------------------------------------------------------------------------
def _resume_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    path = Path(path)
    return path if path.suffix == ".json" else path / "critical_path_progress.json"


def _load_resume(path: Path | None, target: str, seeds) -> dict:
    p = _resume_path(path)
    if p is None or not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if data.get("target") != target or list(data.get("seeds", [])) != list(seeds):
        return {}                                      # a different question: start again rather than mix runs
    return data


def _save_resume(path: Path | None, target: str, seeds, candidates, state: dict) -> None:
    p = _resume_path(path)
    if p is None:
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(dict(target=target, seeds=list(seeds), candidates=candidates,
                                   control=state.get("control"), results=state.get("results", {})),
                              indent=1, default=str), encoding="utf-8")
    tmp.replace(p)


# --- output ------------------------------------------------------------------------------------------------------
def to_csv(res: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        extra = [k for k in ("lesioned_base_hz", "control_base_hz", "lesioned_ratio", "control_ratio")
                 if k in (res["ranked"][0] if res["ranked"] else {})]
        w.writerow(["rank", "cell_type", "neurons", "drives_the_behavior", "lesioned_mean", "ci_lo", "ci_hi",
                    "control_mean", "effect", "effect_share", "p_value", "n_seeds"] + extra)
        for r in res["ranked"]:
            w.writerow([r["rank"], r["type"], r["neurons"], int(r["drives_the_behavior"]),
                        f"{r['lesioned_mean']:.4f}", f"{r['lesioned_ci'][0]:.4f}", f"{r['lesioned_ci'][1]:.4f}",
                        f"{r['control_mean']:.4f}", f"{r['effect']:.4f}", f"{r['effect_share']:.4f}",
                        f"{r['p_value']:.5f}", r["n"]] + [f"{r[k]:.4f}" for k in extra])
    return path


def save(res: dict, folder: Path) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "critical_path.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    to_csv(res, folder / "critical_path.csv")
    return folder


def summary(res: dict) -> str:
    lines = [f"Critical path for {res['target']} ({res['metric']}), seeds {res['seeds'][0]}-{res['seeds'][-1]}, "
             f"{res['seconds']:.0f}s. Unperturbed: {res['control']['mean']:.2f}"]
    for r in res["ranked"][:12]:
        star = " *" if r["p_value"] < 0.01 else ""
        lines.append(f"  {r['rank']:2d}. {r['type']:<16} {r['lesioned_mean']:7.2f} "
                     f"({r['effect_share']:+.0%}, p={r['p_value']:.4f}){star}"
                     + ("   [drives it]" if r["drives_the_behavior"] else ""))
    return "\n".join(lines)
