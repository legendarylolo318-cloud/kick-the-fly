"""Failure autopsy (3.1.0 task 8, Lab): for each validation test that FAILS, where does the signal go?

READ-ONLY. It reads the connectome, the simulation's own weights and a run of the brain; it never changes a weight, a time constant, a
threshold or a parameter, and it does not suggest changing one (a test checks the weight matrix is bit-identical after an autopsy). What it
reports is a diagnosis of the model, not a fix. Tag: MODEL PREDICTION for what the run measured; the path numbers are CONNECTOME.

For each failing test the *input* (the neurons the test drives) and the *target* (the readout) are fixed by the test, and the autopsy asks four things:

1. How much of the target's input could the input supply, hop by hop? `layer_budget`: the share of the average target neuron's input
   synapses that arrives along walks of exactly 1, 2 and 3 synapses from the input, split into walks that excite (an even number of inhibitory
   synapses on the way) and walks that inhibit (an odd number). A weight is the simulator's own, the signed synapse count from pre to post
   divided by all of post's input synapses (sim.W_csr[post, pre]), so a share is a fraction of what the target receives. Walks, not simple
   paths: a walk may pass through the input or the target again, which is what current in a recurrent network does.
2. Which cell types carry it? `routes`: the strongest routes through *cell types* (1, 2 or 3 synapses), each with the share it carries, how much
   of it excites and inhibits, and the raw synapse counts between its stages (total synapses from one stage's types onto the next's, split by sign).
3. What did each stage do in the actual run? `stage_rates`: the firing of the input, of every type on the strongest routes, and of the
   target, before and while the input is driven, over several seeds (the same drive as the pathway tests: current 0.5 for 2 s after 2 s of calm).
4. Where does it fade? `verdict`: the first stage along the strongest route whose driven/baseline ratio is under the test's 1.5x after a stage that
   met it, or "never rose at the first hop" when the input did not even lift its first partners.

The drive is the pathway tests' own, not each failing test's special stimulus (the E-PG wedge, the wind arena): the question is whether the
wiring carries a driven signal to the target. Larva failures are not autopsied here: they need the larva pack, which is optional.

Which neurons are the input and the target, for the tests that are not simple drive-and-readout pathways, is fixed in `special_pathways` and written
there with the reason.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

TAG = "MODEL PREDICTION"
RATIO_MIN = 1.5
AUTOPSY_SEEDS = (1000, 1001, 1002)
PRE, STIM, AMP = 400, 400, 0.5
HOPS = 3
TOP_ROUTES = 6
VERSION = 1


class AutopsyError(ValueError):
    pass


# --- which neurons --------------------------------------------------------------------------------------------------------------------------
def _prefix(br, *prefixes):
    t = br.types.astype(str)
    m = np.zeros(br.n, bool)
    for p in prefixes:
        m |= np.char.startswith(t, p)
    return np.flatnonzero(m)


def special_pathways(test_id: str, br) -> list[dict]:
    """The input and target of the failing tests that are not a plain drive-and-readout pathway, with the reason for each choice."""
    from kickthefly.lab import assays

    g = assays.groups(br)
    t = br.types.astype(str)
    if test_id == "grooming_hierarchy":
        return [dict(label="head bristles to the front-leg motor neurons (the head grooming program)", inp=g["groom_anterior"], tgt=g["mn_front"],
                     inp_label="head bristle neurons (BM_*)", tgt_label="front-leg motor neurons"),
                dict(label="abdominal sensory neurons to the hind-leg motor neurons (the abdomen program)", inp=g["groom_posterior"], tgt=g["mn_hind"],
                     inp_label="abdominal mechanosensory neurons", tgt_label="hind-leg motor neurons"),
                dict(label="head bristles to the hind-leg motor neurons (the suppression the hierarchy needs)", inp=g["groom_anterior"], tgt=g["mn_hind"],
                     inp_label="head bristle neurons (BM_*)", tgt_label="hind-leg motor neurons")]
    if test_id in ("epg_compass",):
        epg = np.flatnonzero(t == "EPG")
        return [dict(label="the driven wedge to the rest of the ring (what a bump needs to spread and persist)", inp=epg[:4], tgt=epg[4:],
                     inp_label="4 EPG neurons (the wedge the test drives)", tgt_label=f"the other {max(len(epg) - 4, 0)} EPG neurons")]
    if test_id == "epg_compass_wind":
        return [dict(label="wind neurons to the E-PG compass neurons", inp=g["jo_ce"], tgt=np.flatnonzero(t == "EPG"),
                     inp_label="JO-C/E mechanosensory neurons (the wind's route in the open field)", tgt_label="EPG neurons")]
    if test_id == "mb_extinction":
        return [dict(label="mushroom body output neurons to the reward dopamine neurons (the route an opposing, extinguishing memory needs)",
                     inp=_prefix(br, "MBON"), tgt=_prefix(br, "PAM"), inp_label="MBON types", tgt_label="PAM reward dopamine neurons")]
    if test_id == "mb_second_order":
        return [dict(label="mushroom body output neurons to the punishment dopamine neurons (odor A's memory must drive PPL1 for odor B to gain fear)",
                     inp=_prefix(br, "MBON"), tgt=_prefix(br, "PPL1"), inp_label="MBON types", tgt_label="PPL1 punishment dopamine neurons")]
    raise AutopsyError(f"{test_id} has no autopsy defined")


def pathways(test_id: str, br) -> list[dict]:
    from kickthefly.lab import assays, validation

    t = validation.BY_ID.get(test_id)
    if t is None or t.get("brain") == "larva":
        raise AutopsyError(f"{test_id}: the larva tests are not autopsied here (they need the optional larva pack)")
    if t.get("drive") and t.get("readout") and not t.get("kind") and test_id not in ("epg_compass", "epg_compass_wind"):
        g = assays.groups(br)
        return [dict(label=t["claim"], inp=g[t["drive"]], tgt=g[t["readout"]], inp_label=t["drive_label"], tgt_label=t["readout_label"])]
    return special_pathways(test_id, br)


def failing_tests(results: dict | None = None) -> list[dict]:
    """The adult tests that failed in a validation result (the latest saved one by default), as {id, name, ...}."""
    from kickthefly.lab import validation

    if results is None:
        results, _ = validation.load_results()
    if not results:
        raise AutopsyError("no validation result to read: run --validate first")
    out = []
    for r in results["tests"]:
        if not r.get("passed") and results.get("brain", "adult") == "adult" and r["id"] in validation.BY_ID:
            out.append(r)
    return out


# --- the connectome side (no brain run) -----------------------------------------------------------------------------------------------------
def raw_counts(W: sp.csr_array, inv: np.ndarray) -> sp.csr_array:
    """The signed synapse counts behind sim.W_csr: W is count / (all of post's input synapses), so count = W / inv[post]."""
    scale = np.repeat(np.where(inv > 0, 1.0 / np.maximum(inv, 1e-12), 0.0), np.diff(W.indptr))
    return sp.csr_array((W.data * scale, W.indices, W.indptr), shape=W.shape)


def _split(W: sp.csr_array):
    pos = W.copy()
    pos.data = np.where(pos.data > 0, pos.data, 0.0)
    neg = W.copy()
    neg.data = np.where(neg.data < 0, -neg.data, 0.0)
    pos.eliminate_zeros()
    neg.eliminate_zeros()
    return pos.tocsr(), neg.tocsr()


def layer_budget(W: sp.csr_array, inp: np.ndarray, tgt: np.ndarray, hops: int = HOPS) -> list[dict]:
    """Per hop h = 1..hops: the share of the average target neuron's input synapses that arrives along walks of exactly h synapses from the
    input, split into excitatory walks (an even number of inhibitory synapses) and inhibitory walks (odd)."""
    P, N = _split(W)
    n = W.shape[0]
    p = np.zeros(n)
    p[inp] = 1.0
    q = np.zeros(n)
    out = []
    for h in range(1, hops + 1):
        p, q = P @ p + N @ q, P @ q + N @ p
        e, i = float(p[tgt].mean()) if len(tgt) else 0.0, float(q[tgt].mean()) if len(tgt) else 0.0
        out.append(dict(hop=h, excitatory=e, inhibitory=i, net=e - i, total=e + i,
                        inhibitory_share=(i / (e + i)) if (e + i) > 0 else float("nan")))
    return out


def _by_type(values: np.ndarray, types: np.ndarray) -> dict[str, float]:
    keys, inv = np.unique(types, return_inverse=True)
    sums = np.bincount(inv, weights=values, minlength=len(keys))
    return {str(k): float(s) for k, s in zip(keys, sums) if s > 0}


def routes(W: sp.csr_array, counts: sp.csr_array, types: np.ndarray, inp: np.ndarray, tgt: np.ndarray, k: int = TOP_ROUTES) -> list[dict]:   # noqa: C901
    """The strongest routes through cell types. A route's `share` is the part of the average target neuron's input that arrives along it:
    the sum over every walk input -> x -> ... -> target through neurons of those types of the product of the synapses' |weights|. It is split into
    what excites and what inhibits by the walk's sign. `counts` is the raw signed synapse counts; a stage's counts are the total synapses from
    the types before it onto the types in it."""
    n = W.shape[0]
    P, N = _split(W)
    Wt = W.tocsc()
    ind_i = np.zeros(n)
    ind_i[inp] = 1.0
    ind_t = np.zeros(n)
    ind_t[tgt] = 1.0 / max(len(tgt), 1)
    # forward from the input (v) and backward from the target (b), each as an excitatory and an inhibitory part
    vp, vn = P @ ind_i, N @ ind_i                                  # neuron x: input weight it gets from the input, by sign
    bp, bn = P.T @ ind_t, N.T @ ind_t                              # neuron y: weight it gives the target, by sign
    cand: list[tuple[float, float, float, tuple]] = []             # (share, exc, inh, type chain)
    direct = np.asarray(P[tgt][:, inp].sum()) / max(len(tgt), 1), np.asarray(N[tgt][:, inp].sum()) / max(len(tgt), 1)
    if direct[0] + direct[1] > 0:
        cand.append((float(direct[0] + direct[1]), float(direct[0]), float(direct[1]), ()))
    # one intermediate type
    for name, ex, inh in _merge2(vp, vn, bp, bn, types):
        cand.append((ex + inh, ex, inh, (name,)))
    # two intermediate types, over the neurons that have both a way in and a way out
    xs = np.flatnonzero((vp + vn) > 0)
    ys = np.flatnonzero((bp + bn) > 0)
    if len(xs) and len(ys):
        sub = W[ys][:, xs].tocoo()                                 # W[y, x]: x -> y
        w = sub.data
        x, y = xs[sub.col], ys[sub.row]
        aw = np.abs(w)
        posw = w > 0
        # sign bookkeeping: the walk is excitatory if the number of inhibitory synapses is even
        e_same = np.where(posw, vp[x] * bp[y] + vn[x] * bn[y], vp[x] * bn[y] + vn[x] * bp[y])
        i_same = np.where(posw, vp[x] * bn[y] + vn[x] * bp[y], vp[x] * bp[y] + vn[x] * bn[y])
        ex, inh = aw * e_same, aw * i_same
        keep = np.flatnonzero(ex + inh > 0)
        if len(keep):
            tx, ty = types[x[keep]], types[y[keep]]
            pair = {}
            for t1, t2, e, i in zip(tx, ty, ex[keep], inh[keep]):
                a = pair.setdefault((str(t1), str(t2)), [0.0, 0.0])
                a[0] += e
                a[1] += i
            top = sorted(pair.items(), key=lambda kv: -(kv[1][0] + kv[1][1]))[: k * 4]
            for (t1, t2), (e, i) in top:
                cand.append((e + i, e, i, (t1, t2)))
    cand.sort(key=lambda c: -c[0])
    picked, per = [], {}
    for c in cand:                                    # the strongest few of each length, so a 3-synapse route is not hidden behind 2-synapse ones
        h = len(c[3]) + 1
        if per.get(h, 0) < max(2, k // 3):
            per[h] = per.get(h, 0) + 1
            picked.append(c)
    cand = sorted(picked, key=lambda c: -c[0])[:k + 3]
    out = []
    seen = set()
    for share, e, i, chain in cand:
        if chain in seen:
            continue
        seen.add(chain)
        stages = [inp, *[np.flatnonzero(types == c) for c in chain], tgt]          # the input's and target's own neurons at the ends
        edges = []
        for pre, post in zip(stages[:-1], stages[1:]):
            sub = counts[post][:, pre]
            edges.append(dict(synapses=int(round(np.abs(sub.data).sum())), excitatory=int(round(sub.data[sub.data > 0].sum())),
                              inhibitory=int(round(-sub.data[sub.data < 0].sum()))))
        out.append(dict(hops=len(chain) + 1, types=list(chain), share=share, excitatory=e, inhibitory=i,
                        net_sign=1 if e >= i else -1, inhibitory_share=i / (e + i) if (e + i) else float("nan"), edges=edges))
        if len(out) >= k + 3:
            break
    return out


def _merge2(vp, vn, bp, bn, types) -> list[tuple[str, float, float]]:
    """Walks input -> x -> target through single neurons x, summed by x's type."""
    ex = vp * bp + vn * bn
    inh = vp * bn + vn * bp
    e, i = _by_type(ex, types), _by_type(inh, types)
    names = set(e) | set(i)
    return sorted(((n, e.get(n, 0.0), i.get(n, 0.0)) for n in names), key=lambda r: -(r[1] + r[2]))[:TOP_ROUTES * 4]


# --- the run side ----------------------------------------------------------------------------------------------------------------------------
def _run_stage(seed: int, test_pathway: dict, groups: dict[str, np.ndarray]) -> dict[str, tuple[float, float]]:
    """One seed: PRE calm steps, then the input held driven for STIM steps; (baseline Hz, driven Hz) for each named group."""
    from kickthefly.core import simcore
    from kickthefly.lab import assays

    br = simcore.new_brain(seed=seed, individuality="off", memory=False)
    return assays.pathway_response(br, test_pathway["inp"], groups, pre=PRE, stim=STIM, amp=AMP)


def stage_rates(br, pw: dict, route_types: list[list[str]], seeds=AUTOPSY_SEEDS) -> dict:
    """Firing of the input, each route's types and the target, before and while the input is driven, averaged over seeds."""
    types = br.types.astype(str)
    groups: dict[str, np.ndarray] = {"__input__": pw["inp"], "__target__": pw["tgt"]}
    for chain in route_types:
        for t in chain:
            groups.setdefault(t, np.flatnonzero(types == t))
    acc = {k: [] for k in groups}
    for s in seeds:
        r = _run_stage(s, pw, groups)
        for k, (b, d) in r.items():
            acc[k].append((b, d))
    out = {}
    for k, vals in acc.items():
        b = float(np.mean([v[0] for v in vals]))
        d = float(np.mean([v[1] for v in vals]))
        out[k] = dict(base_hz=b, driven_hz=d, ratio=d / max(b, 0.5), neurons=int(len(groups[k])))
    return out


def verdict(route: dict | None, rates: dict, ratio_min: float = RATIO_MIN) -> dict:
    """Where along the strongest route the driven/baseline ratio first falls below ratio_min after a stage that met it."""
    if route is None:
        return dict(stage=None, text="No route of up to three synapses connects the input to the target: nothing can carry the signal.")
    names = ["input", *route["types"], "target"]
    keys = ["__input__", *route["types"], "__target__"]
    seq = [(n, rates[k]["ratio"]) for n, k in zip(names, keys)]
    lost = None
    for (a, ra), (b, rb) in zip(seq[:-1], seq[1:]):
        if ra >= ratio_min and rb < ratio_min:
            lost = (a, b)
            break
    if lost is None and seq and seq[0][1] < ratio_min:
        text = f"the input itself ran at x{seq[0][1]:.2f}, below {ratio_min}x: the drive never lifted the first stage"
        stage = "input"
    elif lost is None:
        end = seq[-1][1]
        text = (f"every stage along the strongest route met {ratio_min}x (the target ended at x{end:.2f})" if end >= ratio_min
                else f"the ratios never crossed from above to below {ratio_min}x along the strongest route (target x{end:.2f})")
        stage = None
    else:
        stage = f"{lost[0]} -> {lost[1]}"
        text = (f"the signal is lost between {lost[0]} (x{dict(seq)[lost[0]]:.2f}) and {lost[1]} (x{dict(seq)[lost[1]]:.2f}), "
                f"below the test's {ratio_min}x")
    return dict(stage=stage, text=text, ratios=[dict(stage=n, ratio=r) for n, r in seq])


# --- one failing test ---------------------------------------------------------------------------------------------------------------------------
def autopsy(test_id: str, seeds=AUTOPSY_SEEDS, result: dict | None = None, progress=None) -> dict:
    """The autopsy of one failing test: a dict (JSON-safe) with a section per pathway. READ-ONLY."""
    from kickthefly.core import simcore
    from kickthefly.lab import validation
    from kickthefly.sim import brainpack

    t = validation.BY_ID.get(test_id)
    if t is None:
        raise AutopsyError(f"unknown test {test_id!r}")
    br = simcore.new_brain(seed=seeds[0], warmup=0, individuality="off", memory=False)
    W = br.sim.W_csr
    before = _digest(W)
    pws = pathways(test_id, br)
    inv = np.load(brainpack.find("adult"))["inv"]
    counts = raw_counts(W, inv)
    types = br.types.astype(str)
    sections = []
    for pw in pws:
        if not len(pw["inp"]) or not len(pw["tgt"]):
            sections.append(dict(label=pw["label"], error="the input or the target is empty in this pack"))
            continue
        budget = layer_budget(W, pw["inp"], pw["tgt"])
        rt = routes(W, counts, types, pw["inp"], pw["tgt"])
        main = rt[0] if rt else None
        rates = stage_rates(br, pw, [r["types"] for r in rt], seeds)
        for r in rt:
            r["stage_ratios"] = [dict(stage=n, **rates[k]) for n, k in zip(["input", *r["types"], "target"], ["__input__", *r["types"], "__target__"])]
        direct = counts[pw["tgt"]][:, pw["inp"]]
        sections.append(dict(label=pw["label"], input=pw["inp_label"], target=pw["tgt_label"], n_input=int(len(pw["inp"])), n_target=int(len(pw["tgt"])),
                             direct_synapses=int(round(np.abs(direct.data).sum())), budget=budget, routes=rt,
                             input_rate=rates["__input__"], target_rate=rates["__target__"], verdict=verdict(main, rates)))
        if progress:
            progress(len(sections), len(pws), pw["label"])
    assert _digest(W) == before, "an autopsy changed the weights (it must not)"
    return dict(version=VERSION, tag=TAG, test=dict(id=test_id, name=t["name"], claim=t["claim"], citation=t["citation"], note=t.get("note", "")),
                result=_result_line(result), seeds=list(seeds), created=time.strftime("%Y-%m-%d %H:%M:%S"), read_only=True,
                weights_digest=before, sections=sections, ratio_min=RATIO_MIN,
                method="walks of 1-3 synapses, simulator weights (signed count / post's input count); stage rates from "
                       f"{len(seeds)} seeds, the pathway tests' drive (0.5 for 2 s after 2 s calm)")


def _digest(W: sp.csr_array) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(W.data).tobytes())
    h.update(np.ascontiguousarray(W.indices).tobytes())
    h.update(np.ascontiguousarray(W.indptr).tobytes())
    return h.hexdigest()[:16]


def _result_line(result: dict | None) -> dict | None:
    if not result:
        return None
    return dict(passed=bool(result.get("passed")), criteria=result.get("criteria", ""), measured=result.get("measured"))


# --- all the failures + a page per failure --------------------------------------------------------------------------------------------------------
def run_all(folder, results: dict | None = None, only=None, seeds=AUTOPSY_SEEDS, progress=None) -> list[dict]:
    """Autopsy every failing adult test of a validation result into `folder`: <id>.json and <id>.md each, and index.md."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fails = failing_tests(results)
    if only:
        fails = [f for f in fails if f["id"] in only]
    out = []
    for i, f in enumerate(fails, 1):
        try:
            a = autopsy(f["id"], seeds, f)
        except AutopsyError as e:
            a = dict(test=dict(id=f["id"], name=f.get("name", f["id"])), error=str(e), tag=TAG, sections=[], read_only=True)
        (folder / f"{f['id']}.json").write_text(json.dumps(a, indent=1, default=str), encoding="utf-8")
        (folder / f"{f['id']}.md").write_text(to_markdown(a), encoding="utf-8")
        out.append(a)
        if progress:
            progress(i, len(fails), f["id"])
    (folder / "index.md").write_text(index_markdown(out), encoding="utf-8")
    return out


def index_markdown(autopsies: list[dict]) -> str:
    lines = [f"# Failure autopsy ({TAG}, read-only)", "",
             "One page per failing validation test: where the driven signal goes and where it fades. Nothing here changes the model or suggests a change.", ""]
    for a in autopsies:
        v = [s.get("verdict", {}).get("text", s.get("error", "")) for s in a.get("sections", [])]
        lines.append(f"- [{a['test']['name']}]({a['test']['id']}.md): " + (a.get("error") or " / ".join(x for x in v if x)))
    return "\n".join(lines) + "\n"


def pct(x: float) -> str:
    """A share as a percentage with enough digits for the very small ones these are (0.0031%, not 0.00%)."""
    return "n/a" if x != x else ("0%" if x == 0 else f"{100 * x:.3g}%")


def to_markdown(a: dict) -> str:
    t = a["test"]
    lines = [f"# Autopsy: {t['name']}", "", f"**{a['tag']}** for the run numbers; **CONNECTOME** for the path numbers. Read-only: no weight, constant or threshold is "
             "changed or suggested.", ""]
    if a.get("error"):
        return "\n".join(lines + [f"Not autopsied: {a['error']}", ""])
    lines += [f"Claim: {t['claim']}", f"Source: {t['citation']}", ""]
    r = a.get("result")
    if r:
        m = r.get("measured") or {}
        lines += [f"Result: **FAIL**. {r.get('criteria', '')}", ""]
        if m:
            lines.append("Measured: " + ", ".join(f"{k} {v:.3g}" for k, v in m.items() if isinstance(v, (int, float))) + ".")
            lines.append("")
    if t.get("note"):
        lines += [f"The test's own note: {t['note']}", ""]
    for s in a["sections"]:
        lines += [f"## {s['label']}", ""]
        if s.get("error"):
            lines += [s["error"], ""]
            continue
        lines += [f"Input: {s['input']} ({s['n_input']} neurons). Target: {s['target']} ({s['n_target']} neurons). "
                  f"Direct synapses from input to target: {s['direct_synapses']}.", "",
                  "### How much of the target's input the input can supply", "",
                  "| synapses | excitatory walks | inhibitory walks | net | inhibitory share |", "|---|---|---|---|---|"]
        for b in s["budget"]:
            ish = b["inhibitory_share"]
            ish = "n/a" if ish != ish else format(ish, ".0%")
            lines.append(f"| {b['hop']} | {pct(b['excitatory'])} | {pct(b['inhibitory'])} | {pct(b['net'])} | {ish} |")
        lines += ["", "Shares of the average target neuron's input synapses, along walks of exactly that many synapses.", "",
                  "### The strongest routes, through cell types", ""]
        for i, rt in enumerate(s["routes"], 1):
            chain = " -> ".join(["input", *rt["types"], "target"])
            lines.append(f"{i}. **{chain}**: {pct(rt['share'])} of the target's input "
                         f"({pct(rt['excitatory'])} excitatory, {pct(rt['inhibitory'])} inhibitory; net {'excites' if rt['net_sign'] > 0 else 'inhibits'}).")
            for st, e in zip(rt["stage_ratios"][1:], rt["edges"]):
                lines.append(f"   - into **{st['stage']}**: {e['synapses']:,} synapses ({e['excitatory']:,} excitatory, {e['inhibitory']:,} inhibitory); in the run "
                             f"{st['base_hz']:.1f} Hz at rest, {st['driven_hz']:.1f} Hz driven (x{st['ratio']:.2f}, {st['neurons']} neurons)")
        lines += ["", "### Where it fades", "", f"{s['verdict']['text']}.", ""]
        if s["verdict"].get("ratios"):
            lines.append("Driven / baseline along the strongest route: " + ", ".join(f"{x['stage']} x{x['ratio']:.2f}" for x in s["verdict"]["ratios"]) + ".")
            lines.append("")
    lines += [f"Method: {a['method']}.", f"Weights digest (unchanged by the autopsy): {a['weights_digest']}.", ""]
    return "\n".join(lines)
