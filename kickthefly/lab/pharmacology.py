"""Pharmacology: drugs as synaptic scaling by predicted neurotransmitter, with the confidence of each prediction on show.

MODEL PREDICTION. A drug here is one rule: multiply the weight of every synapse whose presynaptic neuron the dataset
predicts releases transmitter X by a factor that follows the dose. Nothing else about the drug is modeled.

  CONNECTOME   which synapses: each neuron's transmitter (acetylcholine, GABA, glutamate...) and the dataset's confidence in
               it come from the MaleCNS v1.0 neurotransmitter annotations stored in the brain pack. 85,484 of 166,700
               neurons have a measured transmitter (`ground_truth`); the rest are predictions with a confidence.
  MODEL        the drug acts on the synaptic weight only. It does not know about receptor subtypes, where on the cell a
               synapse sits, which neurons actually express the targeted receptor, co-transmission, extrasynaptic or
               neuromodulatory action, dose kinetics, washout, or side effects. The simulator also holds the brain's average
               firing near a target with a slow global gain (sim/connectome/sim.py), which works against any drug that changes
               overall synaptic strength: over seconds it pushes back. Effects are therefore read soon after the drug goes on.
  GAME RULE    the dose-to-scale mapping: a blocker scales by (1 - dose); the GABA-A agonist by (1 + dose x (max - 1)) with
               max = 2.0; the cut that makes a prediction "low confidence" (default 0.7, adjustable); the confidence bands
               shown; the drugs on offer.

Drugs (each scales the listed transmitters' synapses):
  picrotoxin         GABA and glutamate synapses, down. This is the game's existing "inhibition block" (the same synapses, the
                     same arithmetic as Wiring.inhibition_scale), now a drug here. Picrotoxin blocks GABA-A receptors and, in
                     insects, glutamate-gated chloride channels; the model treats both as one scale.
  cholinergic block  acetylcholine synapses, down (a nicotinic-receptor antagonist in spirit; no particular compound is
                     modeled).
  glutamate-Cl block glutamate synapses, down (the glutamate-gated chloride channel, the inhibitory glutamate receptor in
                     flies; this model treats every glutamate synapse as inhibitory, as the dataset's sign rule does).
  GABA-A agonist     GABA synapses, up.

Left out on purpose: octopamine and dopamine modulation. In this simulation the synapses of octopamine, dopamine, serotonin
and 'unclear' neurons carry no sign and are not in the synaptic matrix at all (sim/wiring.py), and there are no existing
neuromodulatory gains to scale; any knob would be invented. The plastic dopamine -> mushroom body learning rule is a separate
game rule (core/memory.py) and is not touched here.

    from kickthefly.lab import pharmacology as ph
    w = ph.wiring_for({"picrotoxin": 0.5}, include_low_confidence=False)   # a sim.wiring.Wiring
    ph.affected(g, "cholinergic", 0.5)                                      # synapses hit, at each confidence level
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

AGONIST_MAX = 2.0                 # GAME RULE: a full-dose GABA-A agonist doubles GABA synapse weights
DEFAULT_CUT = 0.7                 # GAME RULE: a prediction below this confidence is "low confidence"
TAG_TEXT = ("MODEL PREDICTION: scales synapses by the dataset's predicted transmitter. No receptor, kinetic or "
            "neuromodulatory model.")
LEVELS = ("measured", "predicted >= 0.9", "predicted 0.7-0.9", "predicted 0.5-0.7", "predicted < 0.5", "no confidence")


@dataclass(frozen=True)
class Drug:
    key: str
    label: str
    targets: tuple                # transmitters whose presynaptic synapses are scaled
    kind: str                     # "block" | "agonist"
    text: str


DRUGS = {
    "picrotoxin": Drug("picrotoxin", "Picrotoxin", ("gaba", "glutamate"), "block",
                       "Blocks inhibition: GABA and glutamate synapses scaled down by the dose (the game's original inhibition "
                       "block). Runaway excitation, if it happens, is what the recurrent network does, not a scripted seizure."),
    "cholinergic": Drug("cholinergic", "Cholinergic block", ("acetylcholine",), "block",
                        "Scales acetylcholine synapses down: most of the brain's excitation."),
    "glucl": Drug("glucl", "Glutamate-Cl block", ("glutamate",), "block",
                  "Scales glutamate synapses down (every one counts as inhibitory here, following the dataset's sign rule)."),
    "gabaa_agonist": Drug("gabaa_agonist", "GABA-A agonist", ("gaba",), "agonist",
                          f"Scales GABA synapses up, to x{AGONIST_MAX:g} at full dose (the x{AGONIST_MAX:g} is a game rule)."),
}
ALIASES = {"picrotoxin": "picrotoxin", "ptx": "picrotoxin", "cholinergic": "cholinergic", "cholinergic_block": "cholinergic",
           "ach_block": "cholinergic", "glucl": "glucl", "glutamate_cl": "glucl", "glutamate-cl": "glucl",
           "glucl_block": "glucl", "gabaa_agonist": "gabaa_agonist", "gaba_agonist": "gabaa_agonist",
           "gabaa": "gabaa_agonist"}


class PharmError(ValueError):
    pass


def drug(name: str) -> Drug:
    key = ALIASES.get(str(name).strip().lower().replace(" ", "_"))
    if key is None:
        raise PharmError(f"unknown drug {name!r}; use one of {sorted(DRUGS)}")
    return DRUGS[key]


def scale_for(name: str, dose: float) -> float:
    """The synaptic scale a dose (0-1) gives."""
    d = drug(name)
    dose = float(dose)
    if not 0.0 <= dose <= 1.0:
        raise PharmError("dose must be between 0 and 1")
    return 1.0 - dose if d.kind == "block" else 1.0 + dose * (AGONIST_MAX - 1.0)


def wiring_for(doses: dict, include_low_confidence: bool = True, cut: float = DEFAULT_CUT, base=None):
    """A sim.wiring.Wiring that applies these drugs ({name: dose 0-1}). Two drugs that act on the same transmitter multiply.
    include_low_confidence=False leaves out neurons whose transmitter is a prediction below `cut`. base: keep its other
    wiring changes (synapse threshold, sign flips)."""
    from kickthefly.sim.wiring import Wiring

    if not 0.0 < float(cut) <= 1.0:
        raise PharmError("the confidence cut must be in (0, 1]")
    scales: dict = {}
    for name, dose in (doses or {}).items():
        d = drug(name)
        f = scale_for(name, dose)
        for nt in d.targets:
            scales[nt] = scales.get(nt, 1.0) * f
    nts = tuple(sorted((nt, float(f)) for nt, f in scales.items() if f != 1.0))
    base = base or Wiring()
    return Wiring(base.min_synapses, base.flip_rows, base.inhibition_scale, nts, 0.0 if include_low_confidence else float(cut))


# --- how many synapses are affected, by confidence level --------------------------------------------------------------------------
def _level(nt_conf: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Confidence level index (0-5, see LEVELS) of every neuron's transmitter."""
    lvl = np.full(len(nt_conf), 5, np.int8)
    c = np.nan_to_num(nt_conf, nan=-1.0)
    lvl[(c >= 0) & (c < 0.5)] = 4
    lvl[(c >= 0.5) & (c < 0.7)] = 3
    lvl[(c >= 0.7) & (c < 0.9)] = 2
    lvl[c >= 0.9] = 1
    lvl[source == "ground_truth"] = 0
    return lvl


def _counts(g, cut: float = 0.0) -> dict:
    """{transmitter: {level: (neurons, connections, synapses)}} counting only neurons the confidence cut lets a drug scale
    (cut 0: all). Computed once per (connectome, cut) and kept on the connectome."""
    store = getattr(g, "_pharm_counts", None)
    if store is None:
        store = {}
        try:
            g._pharm_counts = store
        except Exception:
            pass
    key = round(float(cut), 4)
    if key in store:
        return store[key]
    from kickthefly.sim import wiring as W

    nt, conf, source = W.transmitters(g)
    lvl = _level(conf, source)
    ok = W.confidence_ok(g, cut)
    pre, syn = W.edge_pre(), W.synapse_counts()
    e_lvl, e_ok = lvl[pre], ok[pre]
    out = {}
    for t in np.unique(nt):
        if not t:
            continue
        is_t = nt == t
        e_is = is_t[pre] & e_ok
        per = {}
        for i in range(len(LEVELS)):
            sel = e_is & (e_lvl == i)
            per[i] = (int(np.count_nonzero(is_t & ok & (lvl == i))), int(np.count_nonzero(sel)), int(syn[sel].sum()))
        out[str(t)] = per
    store[key] = out
    return out


def affected(g, name: str, dose: float = 1.0, include_low_confidence: bool = True, cut: float = DEFAULT_CUT) -> dict:
    """What a drug touches, at each confidence level. Counts are presynaptic neurons, connections (entries of the synaptic
    matrix) and synapses (the dataset's integer counts). The plain columns are everything the drug could touch; the
    `*_applied` ones are what the current setting scales (low-confidence predictions left out when include_low_confidence is
    False: only measured transmitters and predictions at least `cut` sure)."""
    d = drug(name)
    full, kept = _counts(g, 0.0), _counts(g, 0.0 if include_low_confidence else cut)
    rows = []
    for i, label in enumerate(LEVELS):
        r = dict(level=label)
        for tag, table in (("", full), ("_applied", kept)):
            n = c = z = 0
            for t in d.targets:
                a, b, y = table.get(t, {}).get(i, (0, 0, 0))
                n, c, z = n + a, c + b, z + y
            r["neurons" + tag], r["connections" + tag], r["synapses" + tag] = n, c, z
        r["applied"] = r["synapses_applied"] > 0 or r["synapses"] == 0
        rows.append(r)
    tot = lambda key: sum(r[key] for r in rows)   # noqa: E731
    return dict(drug=d.key, label=d.label, dose=float(dose), scale=scale_for(name, dose), targets=list(d.targets),
                rows=rows, neurons=tot("neurons"), connections=tot("connections"), synapses=tot("synapses"),
                neurons_applied=tot("neurons_applied"), connections_applied=tot("connections_applied"),
                synapses_applied=tot("synapses_applied"), include_low_confidence=bool(include_low_confidence),
                cut=float(cut), tag=TAG_TEXT)


# --- measuring a drug's effect -------------------------------------------------------------------------------------------------
def response(doses: dict, include_low_confidence: bool = True, cut: float = DEFAULT_CUT, seeds=(1000,), steps: int = 200,
             base=None) -> dict:
    """Whole-brain firing with the drug on, against the same seeds without it. A short window: the simulator's global gain
    slowly counteracts a drug that changes overall synaptic strength, so a long window measures the drug plus that
    compensation. Uses robustness.measure_firing_distribution (the original inhibition-block measurement)."""
    from kickthefly.lab import robustness
    from kickthefly.sim.wiring import Wiring

    w = wiring_for(doses, include_low_confidence, cut, base)
    before = robustness.measure_firing_distribution(base or Wiring(), seeds=seeds, steps=steps)
    after = robustness.measure_firing_distribution(w, seeds=seeds, steps=steps)
    return dict(doses=dict(doses), wiring=w.as_dict(), include_low_confidence=include_low_confidence, cut=cut, seeds=list(seeds),
                steps=int(steps), before=before, after=after, tag=TAG_TEXT)


def dose_response(name: str, doses=(0.0, 0.25, 0.5, 0.75, 1.0), include_low_confidence: bool = True,
                  cut: float = DEFAULT_CUT, seeds=(1000,), steps: int = 200) -> dict:
    """Mean whole-brain firing (Hz) at each dose of one drug."""
    from kickthefly.lab import robustness
    from kickthefly.sim.wiring import Wiring

    rows = []
    for dose in doses:
        w = wiring_for({name: dose}, include_low_confidence, cut) if dose else Wiring()
        dist = robustness.measure_firing_distribution(w, seeds=seeds, steps=steps)
        rows.append(dict(dose=float(dose), scale=scale_for(name, dose), mean_hz=float(dist["mean"]),
                         median_hz=float(dist["median"]), p95_hz=float(dist["p95"])))
    return dict(drug=drug(name).key, rows=rows, include_low_confidence=include_low_confidence, cut=cut, seeds=list(seeds),
                steps=int(steps), tag=TAG_TEXT)
