"""Thermogenetics: TrpA1 (activates the neurons that express it above a temperature) and shibire-ts (silences them).

In the lab these are tools of a fly line: a driver (a cell type, or a split-GAL4 line from lab/genetics.py) makes the
neurons carry the effector, and the temperature of the room does the rest. Here the expression is a neuron spec, the
temperature is a number you set, and the effector turns into a current on those neurons.

What is what:
  LITERATURE  (cited, approximate)
    TrpA1 turns on near 25 C: Pulver et al. 2009 (J Neurophysiol 101:3075, doi:10.1152/jn.00071.2009) expressed dTrpA1 in larval
    motor neurons and "heat ramps from 21 to 27 degrees C evoked tonic spiking at approximately 25 degrees C that showed
    little adaptation over many minutes". Hamada et al. 2008 (Nature 454:217, doi:10.1038/nature07001) identify dTrpA1 as a
    warmth sensor in the fly's anterior cell neurons.
    shibire-ts blocks synaptic transmission at high temperature: Kitamoto 2001 (J Neurobiol 47:81, doi:10.1002/neu.1018) found adults
    expressing shi-ts in cholinergic neurons "becoming motionless within 2 min at 30 degrees C", and walking again within
    about 1 min after the return to the permissive temperature; the shi product is needed for synaptic vesicle recycling.
  GAME RULE  (everything the papers do not give)
    - the activation curve: 0 below the onset temperature, rising linearly to full at `full_c` (TrpA1 25 -> 29 C, shibire-ts
      28 -> 30 C; the 29 C and 28 C ends are this game's choices, not measurements);
    - the kinetics: first-order approach to the target with time constants TrpA1 1 s on and off (no published constant is
      used); shibire-ts 40 s on and 20 s off, read from "within 2 min" and "about 1 min" as roughly three time constants;
    - the size of the effect: TrpA1 is the same activation current the validation suite uses for optogenetic-style drive
      (0.5, the `amp` of protocol `mode: drive`); shibire-ts is brain surgery's silencing current (-0.6);
    - the thermo arena's temperature map: 15 C at the cold wall to 35 C at the hot wall, linear in x;
    - `time_scale` (protocols): compresses the kinetics so a 120 s experiment fits in seconds. Off by default in the game.
  MODEL  (what the simulation then does)
    The fly's neurons are leaky point neurons. A driven neuron fires; a silenced one does not. Nothing else in the model
    depends on temperature (no Q10, no change of synaptic or membrane rates), so ONLY expressing neurons respond.
    shibire-ts really blocks synapses at the terminal; the model cannot do that for one neuron's output alone, so it silences
    the neuron, which is a coarser intervention (it also stops the neuron from spiking). Expression is all-or-nothing in the
    neurons of the chosen cell types: there is no expression-level variation and no leaky expression below threshold.

    from kickthefly.lab import thermogenetics as tg
    th = tg.Thermogenetics([tg.Expression("trpa1", "type:DNp01")])
    th.update(br, temperature_c=30.0, dt_s=0.05)            # call every 10 brain steps
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CITATIONS = {
    "trpa1": "Pulver et al. 2009, J Neurophysiol 101:3075 (doi:10.1152/jn.00071.2009); Hamada et al. 2008, Nature 454:217 "
             "(doi:10.1038/nature07001)",
    "shibire": "Kitamoto 2001, J Neurobiol 47:81 (doi:10.1002/neu.1018)",
}
ARENA_COLD_C, ARENA_HOT_C = 15.0, 35.0           # GAME RULE: the thermo arena's walls, -1 and +1
TEMP_RANGE_C = (10.0, 45.0)
ACTIVATE_CURRENT = 0.5                           # GAME RULE: validation's activation current (protocol mode: drive)
SILENCE_CURRENT = -0.6                           # GAME RULE: brain surgery's silencing current


@dataclass(frozen=True)
class Effector:
    key: str
    label: str
    kind: str                     # "activate" | "silence"
    onset_c: float                # below this the effector does nothing
    full_c: float                 # at and above this it is fully on
    tau_on_s: float
    tau_off_s: float
    current: float
    cited: str                    # what the numbers above are backed by
    rule: str                     # what is a game rule

    def target_fraction(self, temperature_c: float) -> float:
        if temperature_c <= self.onset_c:
            return 0.0
        if temperature_c >= self.full_c:
            return 1.0
        return (temperature_c - self.onset_c) / (self.full_c - self.onset_c)


EFFECTORS = {
    "trpa1": Effector("trpa1", "TrpA1 (activates above a threshold temperature)", "activate", 25.0, 29.0, 1.0, 1.0,
                      ACTIVATE_CURRENT,
                      "onset near 25 C (Pulver 2009: tonic spiking at about 25 C on a 21-27 C ramp; little adaptation)",
                      "full activation at 29 C, the 1 s kinetics, and the current size are game rules"),
    "shibire": Effector("shibire", "shibire-ts (silences above a threshold temperature)", "silence", 28.0, 30.0, 40.0, 20.0,
                        SILENCE_CURRENT,
                        "restrictive temperature 30 C: motionless within 2 min, walking again within about 1 min of "
                        "cooling (Kitamoto 2001)",
                        "the 28 C start of the ramp, the 40 s / 20 s time constants and the silencing current are game rules"),
}
ALIASES = {"trpa1": "trpa1", "dtrpa1": "trpa1", "shibire": "shibire", "shibire-ts": "shibire", "shi": "shibire",
           "shits": "shibire", "shi-ts": "shibire", "shibire_ts": "shibire"}


class ThermoError(ValueError):
    pass


def effector(name: str) -> Effector:
    key = ALIASES.get(str(name).strip().lower())
    if key is None:
        raise ThermoError(f"unknown effector {name!r}; use one of {sorted(EFFECTORS)}")
    return EFFECTORS[key]


def arena_temperature(t: float) -> float:
    """GAME RULE: the thermo arena's temperature at x position t in [-1 (cold wall), +1 (hot wall)], 15-35 C."""
    mid, half = (ARENA_HOT_C + ARENA_COLD_C) / 2, (ARENA_HOT_C - ARENA_COLD_C) / 2
    return float(mid + half * max(-1.0, min(1.0, float(t))))


@dataclass
class Expression:
    """An effector expressed in some neurons: `target` is any neuron spec (a type, a prefix, `line:SS00727`...)."""
    effector: str
    target: str
    strength: float = 1.0            # scales the effector's current (GAME RULE slider)
    rows: np.ndarray | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        self.effector = effector(self.effector).key
        self.target = str(self.target)
        self.strength = float(max(0.0, min(3.0, self.strength)))


class Thermogenetics:
    """A set of expressions and their live state on one brain. update() moves each effector's activation toward what the
    temperature asks for and puts the resulting current on the brain as the named source "thermo"."""

    SOURCE = "thermo"

    def __init__(self, expressions=(), kinetics: str = "real", time_scale: float = 1.0):
        if kinetics not in ("real", "steady"):
            raise ThermoError("kinetics must be 'real' or 'steady'")
        self.expressions = [e if isinstance(e, Expression) else Expression(**e) for e in expressions]
        self.kinetics = kinetics
        self.time_scale = float(max(1e-3, time_scale))
        self.activation = [0.0] * len(self.expressions)
        self.temperature_c = 22.0
        self._written: np.ndarray | None = None

    # -- setup ---------------------------------------------------------------------------------------------------------
    def resolve(self, br) -> None:
        from kickthefly.core import simcore

        for e in self.expressions:
            if e.rows is None:
                try:
                    e.rows = np.asarray(simcore.rows_of(br, e.target), np.int64)
                except ValueError as err:
                    raise ThermoError(str(err)) from None
                if not len(e.rows):
                    raise ThermoError(f"{e.target!r} selects no neurons in this brain")

    def neurons(self) -> int:
        rows = [e.rows for e in self.expressions if e.rows is not None and len(e.rows)]
        return int(len(np.unique(np.concatenate(rows)))) if rows else 0

    # -- live ------------------------------------------------------------------------------------------------------------
    def update(self, br, temperature_c: float, dt_s: float) -> float:
        """Advance the activations by dt_s at this temperature and write the currents. Returns the largest activation."""
        self.resolve(br)
        self.temperature_c = float(temperature_c)
        cur = np.zeros(br.n, np.float32) if self.expressions else None
        for i, e in enumerate(self.expressions):
            ef = EFFECTORS[e.effector]
            goal = ef.target_fraction(self.temperature_c)
            if self.kinetics == "steady":
                self.activation[i] = goal
            else:
                tau = (ef.tau_on_s if goal > self.activation[i] else ef.tau_off_s) / self.time_scale
                self.activation[i] += (goal - self.activation[i]) * (1.0 - float(np.exp(-dt_s / tau)))
                if abs(goal - self.activation[i]) < 1e-4:
                    self.activation[i] = goal
            if self.activation[i] > 0:
                cur[e.rows] += np.float32(ef.current * e.strength * self.activation[i])
        self._apply(br, cur)
        return max(self.activation, default=0.0)

    def _apply(self, br, cur) -> None:
        if cur is None or not np.any(cur):
            if self._written is not None:
                br.clear_current(self.SOURCE)
                self._written = None
            return
        if self._written is not None and np.array_equal(cur, self._written):
            return
        rows = np.flatnonzero(cur)
        br.set_current(self.SOURCE, rows, cur[rows])
        self._written = cur

    def clear(self, br) -> None:
        br.clear_current(self.SOURCE)
        self._written = None
        self.activation = [0.0] * len(self.expressions)

    def describe(self) -> list[dict]:
        out = []
        for e, a in zip(self.expressions, self.activation):
            ef = EFFECTORS[e.effector]
            out.append(dict(effector=ef.label, target=e.target, neurons=0 if e.rows is None else int(len(e.rows)),
                            activation=a, onset_c=ef.onset_c, full_c=ef.full_c, cited=ef.cited, rule=ef.rule))
        return out


# --- protocol and assay plumbing --------------------------------------------------------------------------------------
SPEC_KEYS = {"expression", "temperature_c", "kinetics", "time_scale"}


def check_spec(spec, where: str = "protocol") -> dict:
    """Validate a protocol's `thermogenetics:` block. Returns it normalized."""
    if not isinstance(spec, dict):
        raise ThermoError(f"{where}: thermogenetics must be a mapping")
    bad = set(spec) - SPEC_KEYS
    if bad:
        raise ThermoError(f"{where}: thermogenetics has unknown keys {sorted(bad)}; allowed: {sorted(SPEC_KEYS)}")
    ex = spec.get("expression")
    if not isinstance(ex, list) or not ex:
        raise ThermoError(f"{where}: thermogenetics.expression must be a list of {{effector, target}}")
    out = dict(spec)
    clean = []
    for i, e in enumerate(ex):
        if not isinstance(e, dict) or set(e) - {"effector", "target", "strength"} or "effector" not in e or "target" not in e:
            raise ThermoError(f"{where}: expression {i + 1} needs effector and target (and optionally strength)")
        try:
            effector(e["effector"])
            float(e.get("strength", 1.0))
        except (ThermoError, TypeError, ValueError) as err:
            raise ThermoError(f"{where}: expression {i + 1}: {err}") from None
        clean.append(dict(e))
    out["expression"] = clean
    t = spec.get("temperature_c", 22.0)
    if isinstance(t, (int, float)):
        pts = [(0.0, float(t))]
    elif isinstance(t, list) and t and all(isinstance(p, dict) and set(p) == {"at_s", "c"} for p in t):
        pts = sorted((float(p["at_s"]), float(p["c"])) for p in t)
    else:
        raise ThermoError(f"{where}: temperature_c must be a number or a list of {{at_s, c}}")
    if any(not (TEMP_RANGE_C[0] <= c <= TEMP_RANGE_C[1]) for _, c in pts):
        raise ThermoError(f"{where}: temperatures must be within {TEMP_RANGE_C[0]:g}-{TEMP_RANGE_C[1]:g} C")
    out["temperature_c"] = t
    if spec.get("kinetics", "real") not in ("real", "steady"):
        raise ThermoError(f"{where}: kinetics must be real or steady")
    try:
        ts = float(spec.get("time_scale", 1.0))
    except (TypeError, ValueError):
        raise ThermoError(f"{where}: time_scale must be a number") from None
    if not 1e-3 <= ts <= 1e3:
        raise ThermoError(f"{where}: time_scale must be between 0.001 and 1000")
    return out


def temperature_at(spec: dict, t_s: float) -> float:
    """The protocol's temperature at time t_s: a constant, or a schedule held until the next point (linear ramps are
    written as several points)."""
    t = spec.get("temperature_c", 22.0)
    if isinstance(t, (int, float)):
        return float(t)
    pts = sorted((float(p["at_s"]), float(p["c"])) for p in t)
    cur = pts[0][1]
    for at, c in pts:
        if t_s >= at:
            cur = c
    return cur


def from_spec(spec: dict) -> Thermogenetics:
    return Thermogenetics([Expression(e["effector"], e["target"], float(e.get("strength", 1.0))) for e in spec["expression"]],
                          spec.get("kinetics", "real"), float(spec.get("time_scale", 1.0)))


# --- the assay: thermogenetic activation of DNp01, escape vs temperature ----------------------------------------------
ASSAY_TEMPS = (18.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 36.0)
HOLD_STEPS = 200                     # 1 s at each temperature (steady-state kinetics: the activation is set, not ramped)

# Pass criteria, fixed before the first run (validation seeds 1000-1009), not adjusted after seeing results:
ASSAY_CRITERIA = (
    "C1  at 18, 22 and 24 C (below the TrpA1 onset) the expressing flies' escape rate equals the control flies' (Fisher exact p > 0.05)",
    "C2  at 30, 32 and 36 C (fully on) at least 9 of 10 expressing flies escape and at most 1 of 10 control flies does",
    "C3  escape rate never falls as the temperature rises (Spearman rho >= 0.8 across the 8 temperatures, expressing flies)",
)


def escape_fly(seed: int, temps=ASSAY_TEMPS, target: str = "type:DNp01", effector_name: str = "trpa1",
               params: dict | None = None, brain=None, surgery: dict | None = None) -> dict:
    """One fly. At each temperature (in an order shuffled by the seed, like a real session) the fly is tested twice: once
    with the effector expressed in `target` and once as its own genotype control (no expression). An escape is the game's
    own rule: DNp01 above THRESH['escape'] times its calm rate. Steady-state kinetics: this measures the threshold curve,
    not the time course."""
    from kickthefly.core import simcore
    from kickthefly.game import kick_the_fly as k
    from kickthefly.lab import assays

    br = brain or simcore.new_brain(seed=seed, params=params)
    assays.apply_surgery(br, surgery)
    assays.rest(br, 400)
    dn = assays.groups(br)["dnp01"]
    order = [(t, expr) for t in temps for expr in (True, False)]
    np.random.default_rng(seed + 11).shuffle(order)
    out = {t: {} for t in temps}
    for t, expr in order:
        th = Thermogenetics([Expression(effector_name, target)] if expr else [], "steady")
        th.update(br, t, 0.05)
        peak, spikes = 0.0, 0
        for _ in range(HOLD_STEPS):
            br._step()
            peak = max(peak, br.level("escape"))
            spikes += int(np.count_nonzero(br.sim.spikes[dn]))
        th.clear(br)
        out[t]["expressing" if expr else "control"] = dict(
            escaped=bool(peak > k.THRESH["escape"]), peak_level=float(peak),
            dnp01_hz=spikes / max(1, len(dn)) / (HOLD_STEPS * 0.005))
        assays.rest(br, 400)
    return dict(seed=seed, temps=list(temps), target=target, effector=effector_name, trials=out)


def summarize(flies: list[dict]) -> dict:
    """Escape rate vs temperature, expressing vs control, with Fisher exact p per temperature."""
    from kickthefly.lab import labstats

    temps = flies[0]["temps"]
    rows = []
    for t in temps:
        key = t if t in flies[0]["trials"] else str(t)
        e = [bool(f["trials"][key]["expressing"]["escaped"]) for f in flies]
        c = [bool(f["trials"][key]["control"]["escaped"]) for f in flies]
        rows.append(dict(temperature_c=t, escapes=int(sum(e)), control_escapes=int(sum(c)), flies=len(flies),
                         escape_rate=float(np.mean(e)), control_rate=float(np.mean(c)),
                         dnp01_hz=labstats.mean_ci([f["trials"][key]["expressing"]["dnp01_hz"] for f in flies]),
                         control_dnp01_hz=labstats.mean_ci([f["trials"][key]["control"]["dnp01_hz"] for f in flies]),
                         fisher_p=labstats.fisher(int(sum(e)), len(e), int(sum(c)), len(c))))
    return dict(metric="thermogenetic activation: escape rate vs temperature (expressing vs control)", rows=rows,
                per_fly=[float(np.mean([f["trials"][t if t in f["trials"] else str(t)]["expressing"]["escaped"]
                                        for t in temps])) for f in flies])


def verdict(summary: dict) -> dict:
    """Score the three pre-registered criteria (ASSAY_CRITERIA) on a summary. Meaningful for 10 flies (seeds 1000-1009)."""
    from scipy import stats

    rows = {r["temperature_c"]: r for r in summary["rows"]}
    n = rows[min(rows)]["flies"]
    below = [rows[t] for t in (18.0, 22.0, 24.0) if t in rows]
    full = [rows[t] for t in (30.0, 32.0, 36.0) if t in rows]
    c1 = all(r["fisher_p"] > 0.05 for r in below)
    need = int(np.ceil(0.9 * n))
    c2 = all(r["escapes"] >= need and r["control_escapes"] <= int(np.floor(0.1 * n)) for r in full)
    rates = [rows[t]["escape_rate"] for t in sorted(rows)]
    rho = float(stats.spearmanr(sorted(rows), rates).statistic) if len(set(rates)) > 1 else float("nan")
    c3 = bool(rho >= 0.8)
    return dict(C1=bool(c1), C2=bool(c2), C3=c3, spearman_rho=rho, passed=bool(c1 and c2 and c3), flies=n)
