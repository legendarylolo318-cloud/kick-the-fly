"""Protocol files: an experiment described in YAML, run from Lab > Protocols or headless.

    KickTheFly --headless --protocol my-experiment.yaml [--out folder]

Two kinds of protocol:

1. A stimulus protocol: drive neurons on a schedule and record spikes.

    name: giant-fiber-looming
    seed: 1                      # first seed; flies N use seed .. seed+N-1
    flies: 3
    warmup_s: 3                  # extra calm time after the standard warm-up, before t = 0
    duration_s: 4
    params: {noise_std: 0.05}    # optional Lab model parameters (lab.py names)
    nwb: true                    # also write each fly's recording as NWB (needs pynwb)
    surgery: {"type:DNp01": -1}  # optional: neuron spec -> -1 silence / +1 stimulate
    control: true                # with surgery: also run each seed unperturbed (default true)
    stimuli:
      - {at_s: 1.0, for_s: 0.5, target: loom, strength: 0.8}                 # like a game stimulus, renewed every 50 ms
      - {at_s: 2.5, for_s: 1.0, target: "type:LPLC2,LC4", mode: drive, amp: 0.5}   # constant current (activation)
    recordings:
      - {name: giant_fiber, neurons: escape}
      - {name: looming, neurons: loom}

2. An assay protocol: one of the standard assays over many flies (assays.py, labjobs.py).

    name: kc-silencing
    assay: tmaze                 # tmaze | looming | sugar
    seed: 2000
    flies: 8
    surgery: {"prefix:KC": -1}   # optional; a same-seed control always runs with it
    assay_options: {cycles: 6}

3.0 day 2 additions (each is optional; the existing keys and files are unchanged):

    thermogenetics:                          # TrpA1 / shibire-ts in some neurons, and a temperature (lab/thermogenetics.py)
      expression: [{effector: trpa1, target: "type:DNp01"}]      # target: any neuron spec, including "line:SS00727"
      temperature_c: 32                      # or a schedule: [{at_s: 0, c: 22}, {at_s: 2, c: 32}]
      kinetics: real                         # real (default) | steady ;  time_scale: 1  compresses the time constants
    drug:                                    # synaptic scaling by predicted transmitter (lab/pharmacology.py)
      doses: {picrotoxin: 0.5}               # picrotoxin | cholinergic | glucl | gabaa_agonist, each 0-1
      include_low_confidence: true           # false leaves out predictions below `cut` (default 0.7)
    imaging:                                 # simulated GCaMP imaging alongside the run (lab/imaging.py), MODEL
      indicator: gcamp6s                     # gcamp6s | gcamp6f | gcamp8m
      fps: 20
      rois: regions                          # or a list of neuron specs, one ROI each
      tiff: false                            # also write the rendered brain view as a TIFF stack (slow: builds the view)
    # a third kind of protocol, a virtual patch clamp (lab/patchclamp.py), MODEL:
    patch: {neuron: "type:DNp01", index: 0, mode: embedded, amplitudes: [0, 0.05, 0.1], duration_ms: 500, repeats: 3}

3.0 day 5: a fourth stand-alone kind, a classic behavior rig (game/rigs.py, lab/rigassay.py), one fly per seed, each recorded in the Lab's format:

    rig: {name: buridan, mode: stripes, seconds: 60}     # name: tethered | ball | buridan | fourfield
    #   tethered: mode open|closed, omega (rad/s), seconds, gain      ball: mode closed|open, scene panorama|bar, seconds, omega, bar_deg
    #   buridan: mode stripes|none, seconds                           fourfield: mode odor|sham, seconds
    #   any of them: individuality: off|subtle|strong (default subtle)

3.1.0 task 6 and 7: a fifth stand-alone kind, a whole-brain screen (lab/screens.py), MODEL PREDICTION. Resumable (run the same protocol into the same
--out again); the protocol's `seeds`/`seed`+`flies` are the held-out seeds (default 4000-4007):

    screen: {kind: activation, controls: 2, min_neurons: 1, max_types: 200, types: [LC4, LPLC2]}     # hold each cell type driven in turn
    screen: {kind: knockout, behaviors: [looming_escape], top: 25, candidate_batch: 5, controls: 2}   # silence candidates, rank by the drop

3.0 day 3 additions (stimulus protocols only; each is optional):

    weather:                                 # rain, gusts, lightning (game/weather.py), GAME RULE on the real touch / humidity / wind / light neurons
      {rain: 0.6, gust_hz: 0.2, storm: true, wind_speed: 3, wind_dir: 180}
    audio:                                   # a SYNTHETIC hum through the microphone's analysis onto JO-A/B (core/mic.py); never the microphone
      {hz: 200, ipi_ms: 35, amp: 0.1, at_s: 1, seconds: 3, sensitivity: 1}     # ipi_ms omitted: a steady hum
    predator: {kind: frog, at_s: 1.0}        # a frog | dragonfly | mantis attack as looming onto LPLC2/LC4 (game/predators.py)
    # assay kinds: predator_escape (lab/predators.py) and hum_demo (lab/audio.py), with assay_options as for the others

Neuron specs (simcore.rows_of and assays.groups): a game group ("loom", "escape", "head", "reward", "sweet"...), an
assay group ("dnp01", "mn9", "adn", "jo_ce", "mn_front"...), "type:A,B", "prefix:KC", "superclass:descending_neuron"
or "rows:1,2,3".

Output (a new folder per run): per seed the recorder's CSV, npz and metadata files (recorder.py), plus summary.json with
mean firing per recording group, and, with surgery, the paired comparison against the unperturbed controls. With
`nwb: true` (or --nwb on the command line) each fly's recording is also written as one NWB file (nwbexport.py).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

TOP_KEYS = {"name", "description", "title", "classroom", "steps", "seed", "seeds", "flies", "warmup_s", "duration_s", "params", "surgery", "control",
            "stimuli", "recordings", "assay", "assay_options", "workers", "nwb", "thermogenetics", "drug", "imaging", "patch",
            "weather", "audio", "predator", "rig", "screen"}
STIM_KEYS = {"at_s", "for_s", "target", "strength", "recruit", "mode", "amp", "side"}


class ProtocolError(ValueError):
    pass


MAX_FLIES = 10_000              # 3.0 review: checking flies: 10**9 built a billion-seed list (tens of GB) before anything ran


DAY2_KEYS = ("thermogenetics", "drug", "imaging", "patch")
DAY3_KEYS = ("weather", "audio", "predator")


def _check_day2(p: dict, where: str) -> None:
    """Validate the 3.0 day 2 blocks (thermogenetics, drug, imaging, patch); raise ProtocolError with the reason."""
    from kickthefly.lab import imaging, patchclamp, pharmacology, thermogenetics

    if "screen" in p:
        ignored = [k for k in ("assay", "classroom", "thermogenetics", "drug", "imaging", "stimuli", "recordings", "surgery", "control",
                               "warmup_s", "duration_s", "patch", "params", "nwb", "assay_options", "rig", *DAY3_KEYS) if k in p]
        if ignored:
            raise ProtocolError(f"{where}: a screen protocol stands alone; it can't use {', '.join(ignored)} (a screen sets its own timeline and "
                                "builds its flies with the default parameters)")
        _check_screen(p, where)
        return
    if "rig" in p:
        # 3.0 release review: params, nwb and assay_options were accepted and then silently ignored (a rig fly is built with the default
        # Lab parameters and writes no NWB), so a protocol asking for changed parameters would have reported the defaults' result as its own
        ignored = [k for k in ("assay", "classroom", "thermogenetics", "drug", "imaging", "stimuli", "recordings", "surgery", "control",
                               "warmup_s", "duration_s", "patch", "params", "nwb", "assay_options", *DAY3_KEYS) if k in p]
        if ignored:
            raise ProtocolError(f"{where}: a rig protocol stands alone; it can't use {', '.join(ignored)} (the rig sets its own timeline "
                                "and builds its flies with the default parameters)")
        _check_rig(p, where)
        return
    if "patch" in p:
        # 3.0 day 2 review: surgery, recordings, control and a top-level warmup/duration were accepted and then ignored
        ignored = [k for k in ("assay", "classroom", "thermogenetics", "drug", "imaging", "stimuli", "recordings", "surgery",
                               "control", "warmup_s", "duration_s", *DAY3_KEYS) if k in p]
        if ignored:
            raise ProtocolError(f"{where}: a patch protocol stands alone; it can't use {', '.join(ignored)} (its own warm-up is "
                                f"patch.warmup_s)")
        pa = p["patch"]
        keys = {"neuron", "index", "mode", "amplitudes", "duration_ms", "repeats", "warmup_s"}
        if not isinstance(pa, dict) or set(pa) - keys or "neuron" not in pa:
            raise ProtocolError(f"{where}: patch needs {{neuron, ...}} with only {sorted(keys)}")
        if not isinstance(pa["neuron"], str) or not pa["neuron"].strip():
            raise ProtocolError(f"{where}: patch.neuron must be a neuron spec such as type:DNp01")
        try:
            proto = patchclamp.ClampProtocol(pa.get("amplitudes", [0.0, 0.05, 0.1]), pa.get("duration_ms", 500),
                                             repeats=pa.get("repeats", 3))
            if pa.get("mode", "embedded") not in patchclamp.MODES:
                raise patchclamp.PatchError(f"mode must be one of {patchclamp.MODES}")
            index, warmup = int(pa.get("index", 0)), float(pa.get("warmup_s", 1.0))
            if index < 0 or not 0 <= warmup <= 3600:
                raise patchclamp.PatchError("index must be >= 0 and warmup_s within 0-3600")
        except (patchclamp.PatchError, TypeError, ValueError) as e:
            raise ProtocolError(f"{where}: patch: {e}") from None
        # numbers from here on (3.0 day 2 review: amplitudes written as text passed and broke the summary after the run)
        p["patch"] = dict(pa, amplitudes=list(proto.amplitudes), duration_ms=proto.duration_ms, repeats=proto.repeats,
                          index=index, warmup_s=warmup)
        return
    if "assay" in p or p.get("classroom"):
        used = [k for k in DAY2_KEYS + DAY3_KEYS if k in p]
        if used:
            kind = "an assay" if "assay" in p else "a classroom"
            raise ProtocolError(f"{where}: {', '.join(used)} can't be used in {kind} protocol yet (stimulus protocols only)")
        if p.get("assay") == "thermo_escape":
            _check_thermo_escape(p.get("assay_options") or {}, where)
        if p.get("assay") == "predator_escape":
            _check_predator_escape(p.get("assay_options") or {}, where)
        if p.get("assay") == "hum_demo":
            _check_hum_demo(p.get("assay_options") or {}, where)
        if p.get("assay") == "sleep_deprivation":
            _check_sleep_deprivation(p, where)
        return
    _check_day3(p, where)
    if "thermogenetics" in p:
        try:
            p["thermogenetics"] = thermogenetics.check_spec(p["thermogenetics"], where)
        except thermogenetics.ThermoError as e:
            raise ProtocolError(str(e)) from None
    if "drug" in p:
        d = p["drug"]
        if not isinstance(d, dict) or set(d) - {"doses", "include_low_confidence", "cut"} or not isinstance(d.get("doses"), dict) \
                or not d["doses"]:
            raise ProtocolError(f"{where}: drug needs doses: {{name: 0-1}} (and optionally include_low_confidence, cut)")
        if not isinstance(d.get("include_low_confidence", True), bool):
            raise ProtocolError(f"{where}: drug.include_low_confidence must be true or false")
        try:
            pharmacology.wiring_for(d["doses"], d.get("include_low_confidence", True), float(d.get("cut", pharmacology.DEFAULT_CUT)))
        except (pharmacology.PharmError, TypeError, ValueError) as e:
            raise ProtocolError(f"{where}: drug: {e}") from None
    if "imaging" in p:
        im = p["imaging"]
        keys = {"indicator", "fps", "rois", "tiff", "shot_noise", "f0_photons", "dff_per_spike", "f0_tau_s"}
        if not isinstance(im, dict) or set(im) - keys:
            raise ProtocolError(f"{where}: imaging has unknown keys; allowed: {sorted(keys)}")
        try:
            imaging.indicator(im.get("indicator", imaging.DEFAULT_INDICATOR))
            if not 1 <= float(im.get("fps", 20)) <= 200:
                raise imaging.ImagingError("fps must be between 1 and 200")
            for key, default in (("f0_photons", 100.0), ("dff_per_spike", 0.2), ("f0_tau_s", 30.0)):   # 3.0 day 2 review: were checked mid-run
                v = im.get(key, default)
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
                    raise imaging.ImagingError(f"{key} must be a positive number")
            for key in ("shot_noise", "tiff"):
                if key in im and not isinstance(im[key], bool):
                    raise imaging.ImagingError(f"{key} must be true or false")
        except (imaging.ImagingError, TypeError, ValueError) as e:
            raise ProtocolError(f"{where}: imaging: {e}") from None
        rois = im.get("rois", "regions")
        if not (rois == "regions" or (isinstance(rois, list) and rois and all(isinstance(r, str) for r in rois))):
            raise ProtocolError(f"{where}: imaging.rois must be 'regions' or a list of neuron specs")
        if isinstance(rois, list) and len(rois) > imaging.MAX_ROIS:
            raise ProtocolError(f"{where}: imaging: at most {imaging.MAX_ROIS} ROIs")


def _check_thermo_escape(opts: dict, where: str) -> None:
    """The thermo_escape assay's options become keyword arguments of thermogenetics.escape_fly. 3.0 day 2 review: any key
    went through (brain: 1 crashed every worker; temps: [] crashed the summary)."""
    from kickthefly.lab import thermogenetics as tg

    bad = set(opts) - {"temps", "target", "effector_name"}
    if bad:
        raise ProtocolError(f"{where}: thermo_escape options are temps, target and effector_name, not {sorted(bad)}")
    temps = opts.get("temps", list(tg.ASSAY_TEMPS))
    if (not isinstance(temps, list) or not 1 <= len(temps) <= 50
            or not all(isinstance(t, (int, float)) and not isinstance(t, bool) for t in temps)
            or not all(tg.TEMP_RANGE_C[0] <= t <= tg.TEMP_RANGE_C[1] for t in temps)):
        raise ProtocolError(f"{where}: thermo_escape temps must be a list of 1-50 temperatures within "
                            f"{tg.TEMP_RANGE_C[0]:g}-{tg.TEMP_RANGE_C[1]:g} C")
    if not isinstance(opts.get("target", "type:DNp01"), str):
        raise ProtocolError(f"{where}: thermo_escape target must be a neuron spec")
    try:
        tg.effector(opts.get("effector_name", "trpa1"))
    except tg.ThermoError as e:
        raise ProtocolError(f"{where}: thermo_escape: {e}") from None


def _num(v, lo, hi, what, where, allow_none=False):
    if v is None and allow_none:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not lo <= v <= hi:
        raise ProtocolError(f"{where}: {what} must be a number from {lo:g} to {hi:g}")
    return float(v)


def _check_day3(p: dict, where: str) -> None:
    """Validate the 3.0 day 3 stimulus blocks (weather, audio, predator): unknown keys and out-of-range values are refused."""
    from kickthefly.core import mic
    from kickthefly.game import predators as pr

    if "weather" in p:
        w = p["weather"]
        keys = {"rain", "gust_hz", "storm", "wind_speed", "wind_dir"}
        if not isinstance(w, dict) or set(w) - keys:
            raise ProtocolError(f"{where}: weather has unknown keys; allowed: {sorted(keys)}")
        if "storm" in w and not isinstance(w["storm"], bool):
            raise ProtocolError(f"{where}: weather.storm must be true or false")
        p["weather"] = dict(rain=_num(w.get("rain", 0.0), 0, 1, "weather.rain", where),
                            gust_hz=_num(w.get("gust_hz", 0.0), 0, 0.5, "weather.gust_hz", where), storm=bool(w.get("storm", False)),
                            wind_speed=_num(w.get("wind_speed", 0.0), 0, 20, "weather.wind_speed", where),
                            wind_dir=_num(w.get("wind_dir", 180.0), 0, 360, "weather.wind_dir", where))
    if "audio" in p:
        a = p["audio"]
        keys = {"hz", "ipi_ms", "amp", "at_s", "seconds", "sensitivity"}
        if not isinstance(a, dict) or set(a) - keys:
            raise ProtocolError(f"{where}: audio has unknown keys (it is a synthetic hum, never the microphone); allowed: {sorted(keys)}")
        p["audio"] = dict(hz=_num(a.get("hz", 200.0), 10, 2000, "audio.hz", where),
                          ipi_ms=_num(a.get("ipi_ms"), 10, 500, "audio.ipi_ms", where, allow_none=True),
                          amp=_num(a.get("amp", 0.1), 0, 1, "audio.amp", where), at_s=_num(a.get("at_s", 0.0), 0, 3600, "audio.at_s", where),
                          seconds=_num(a.get("seconds", 1.0), 0.05, 600, "audio.seconds", where),
                          sensitivity=_num(a.get("sensitivity", 1.0), mic.SENSITIVITY[0], mic.SENSITIVITY[1], "audio.sensitivity", where))
    if "predator" in p:
        d = p["predator"]
        if not isinstance(d, dict) or set(d) - {"kind", "at_s"} or not isinstance(d.get("kind"), str) or d["kind"] not in pr.SPECS:
            raise ProtocolError(f"{where}: predator needs kind: one of {', '.join(pr.KINDS)} (and optionally at_s)")
        p["predator"] = dict(kind=d["kind"], at_s=_num(d.get("at_s", 0.0), 0, 3600, "predator.at_s", where))


RIG_KEYS = {"name", "mode", "seconds", "omega", "gain", "scene", "bar_deg", "individuality"}


SCREEN_KEYS = {"kind", "controls", "min_neurons", "max_types", "types", "behaviors", "top", "candidate_batch", "batch", "backend"}


def _check_screen(p: dict, where: str) -> None:
    """The screen block (3.1.0): numbers become numbers, names are checked against what the screens can run, before anything runs."""
    from kickthefly.lab import screens, validation

    sc = p["screen"]
    if not isinstance(sc, dict) or set(sc) - SCREEN_KEYS or sc.get("kind") not in ("activation", "knockout"):
        raise ProtocolError(f"{where}: screen needs {{kind: activation | knockout, ...}} with only {sorted(SCREEN_KEYS)}")
    out = dict(sc)
    for key, lo, hi in (("controls", 1, 20), ("min_neurons", 1, 100000), ("max_types", 1, 100000), ("top", 1, 500), ("candidate_batch", 0, 100),
                        ("batch", 0, 64)):
        if key in sc:
            out[key] = int(_num(sc[key], lo, hi, f"screen.{key}", where))
    for key in ("types", "behaviors"):
        if key in sc and (not isinstance(sc[key], list) or not sc[key] or not all(isinstance(x, str) for x in sc[key]) or len(sc[key]) > 20000):
            raise ProtocolError(f"{where}: screen.{key} must be a non-empty list of names")
    if sc["kind"] == "activation" and set(sc) & {"behaviors", "top", "candidate_batch"}:
        raise ProtocolError(f"{where}: behaviors, top and candidate_batch belong to a knockout screen")
    if sc["kind"] == "knockout" and set(sc) & {"min_neurons", "max_types"}:
        raise ProtocolError(f"{where}: min_neurons and max_types belong to an activation screen")
    for b in sc.get("behaviors", []):
        if b not in screens.VALIDATED_PATHWAYS or b not in validation.BY_ID:
            raise ProtocolError(f"{where}: screen.behaviors: {b!r} is not a validated pathway behavior; use {', '.join(screens.VALIDATED_PATHWAYS)}")
    p["screen"] = out


def _check_rig(p: dict, where: str) -> None:
    """The rig block (3.0 day 5): checked against rigassay.SCENES; numbers become numbers before anything runs."""
    from kickthefly.lab import rigassay

    r = p["rig"]
    if not isinstance(r, dict) or set(r) - RIG_KEYS or not isinstance(r.get("name"), str) or r["name"] not in rigassay.SCENES:      # a list name is unhashable
        raise ProtocolError(f"{where}: rig needs {{name: {' | '.join(rigassay.SCENES)}, ...}} with only {sorted(RIG_KEYS)}")
    sc = rigassay.SCENES[r["name"]]
    if r.get("mode", sc["default"]["mode"]) not in sc["mode"]:
        raise ProtocolError(f"{where}: rig {r['name']} mode must be one of {', '.join(sc['mode'])}")
    if r.get("scene", "bar") not in ("panorama", "bar"):
        raise ProtocolError(f"{where}: rig.scene must be panorama or bar")
    if r.get("individuality", "subtle") not in ("off", "subtle", "strong"):
        raise ProtocolError(f"{where}: rig.individuality must be off, subtle or strong")
    out = dict(r)
    for key, lo, hi in (("seconds", 5, 600), ("omega", -10, 10), ("gain", 0, 3), ("bar_deg", -180, 180)):
        if key in r:
            out[key] = _num(r[key], lo, hi, f"rig.{key}", where)
    p["rig"] = out


def _check_sleep_deprivation(p: dict, where: str) -> None:
    """The sleep_deprivation assay (3.0 day 4): options become keyword arguments of sleepdep.fly_pair. It is a paired design (each
    seed is its own control), so it takes no surgery."""
    opts = p.get("assay_options") or {}
    if not isinstance(opts, dict):
        raise ProtocolError(f"{where}: sleep_deprivation assay_options must be a mapping (deprive_s, recover_s, mode)")
    bad = set(opts) - {"deprive_s", "recover_s", "mode"}
    if bad:
        raise ProtocolError(f"{where}: sleep_deprivation options are deprive_s, recover_s and mode, not {sorted(bad)}")
    _num(opts.get("deprive_s", 45.0), 5, 120, "sleep_deprivation deprive_s", where)
    _num(opts.get("recover_s", 60.0), 5, 180, "sleep_deprivation recover_s", where)
    if opts.get("mode", "off") not in ("off", "subtle", "strong"):
        raise ProtocolError(f"{where}: sleep_deprivation mode must be off, subtle or strong")
    if p.get("surgery"):
        raise ProtocolError(f"{where}: sleep_deprivation is a paired design (each seed is its own control) and takes no surgery")


def _check_hum_demo(opts: dict, where: str) -> None:
    """The hum_demo assay's options become keyword arguments of audio.demo_fly."""
    from kickthefly.lab import audio

    if not isinstance(opts, dict):
        raise ProtocolError(f"{where}: hum_demo assay_options must be a mapping (conditions, seconds, amp)")
    bad = set(opts) - {"conditions", "seconds", "amp"}
    if bad:
        raise ProtocolError(f"{where}: hum_demo options are conditions, seconds and amp, not {sorted(bad)}")
    c = opts.get("conditions", list(audio.CONDITIONS))
    if not isinstance(c, list) or not 1 <= len(c) <= len(audio.CONDITIONS) or not all(isinstance(x, str) and x in audio.CONDITIONS for x in c):
        raise ProtocolError(f"{where}: hum_demo conditions must be a list of {', '.join(audio.CONDITIONS)}")
    _num(opts.get("seconds", audio.SECONDS), 0.2, 30, "hum_demo seconds", where)
    _num(opts.get("amp", audio.HUM_AMP), 0.0, 1.0, "hum_demo amp", where)


def _check_predator_escape(opts: dict, where: str) -> None:
    """The predator_escape assay's options become keyword arguments of predators.escape_fly (3.0 day 3)."""
    from kickthefly.game import predators as pr

    if not isinstance(opts, dict):
        raise ProtocolError(f"{where}: predator_escape assay_options must be a mapping (kinds, trials)")
    bad = set(opts) - {"kinds", "trials"}
    if bad:
        raise ProtocolError(f"{where}: predator_escape options are kinds and trials, not {sorted(bad)}")
    kinds = opts.get("kinds", list(pr.KINDS))
    if not isinstance(kinds, list) or not 1 <= len(kinds) <= len(pr.KINDS) or not all(isinstance(k, str) and k in pr.SPECS for k in kinds):
        raise ProtocolError(f"{where}: predator_escape kinds must be a list of {', '.join(pr.KINDS)}")
    trials = opts.get("trials", 3)
    if isinstance(trials, bool) or not isinstance(trials, int) or not 1 <= trials <= 20:
        raise ProtocolError(f"{where}: predator_escape trials must be a whole number from 1 to 20")


def folder_name(name) -> str:
    """A protocol's name as one safe path component, for its output folder. 3.0 review: a name such as '../../x' (which a
    share code or a downloaded protocol can carry) put the run's files outside the exports folder."""
    import re

    return re.sub(r"[^A-Za-z0-9._+-]+", "-", str(name)).strip("-.")[:60] or "protocol"


def load(path: Path) -> dict:
    import yaml

    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, yaml.YAMLError) as e:
        raise ProtocolError(f"{path}: not valid YAML ({e})") from None
    return check(data, str(path))


def check(data, where: str = "protocol") -> dict:
    from kickthefly.lab import labjobs
    from kickthefly.lab import lab

    if not isinstance(data, dict):
        raise ProtocolError(f"{where}: the top level must be a mapping of keys")
    unknown = set(data) - TOP_KEYS
    if unknown:
        raise ProtocolError(f"{where}: unknown keys {sorted(unknown)}; allowed: {sorted(TOP_KEYS)}")
    p = dict(data)
    p.setdefault("name", Path(where).stem if where != "protocol" else "protocol")
    if "screen" in p and not any(k in p for k in ("seeds", "seed", "flies")):
        from kickthefly.lab import screens

        p["seeds"] = list(screens.SCREEN_SEEDS)                 # a screen's own held-out seeds
    if "seeds" in p:
        if not isinstance(p["seeds"], list) or not all(isinstance(s, int) for s in p["seeds"]):
            raise ProtocolError(f"{where}: seeds must be a list of integers")
        if len(p["seeds"]) > MAX_FLIES:
            raise ProtocolError(f"{where}: at most {MAX_FLIES:,} seeds")
    else:
        seed, flies = p.get("seed", 0), p.get("flies", 1)
        if not isinstance(seed, int) or not isinstance(flies, int) or flies < 1:
            raise ProtocolError(f"{where}: seed must be an integer and flies a positive integer")
        if flies > MAX_FLIES:
            raise ProtocolError(f"{where}: flies must be at most {MAX_FLIES:,}")
        p["seeds"] = list(range(seed, seed + flies))
    for key in ("params", "surgery", "assay_options"):
        if key in p and not isinstance(p[key], dict):
            raise ProtocolError(f"{where}: {key} must be a mapping")
    if "nwb" in p and not isinstance(p["nwb"], bool):
        raise ProtocolError(f"{where}: nwb must be true or false")
    bad = set(p.get("params", {})) - set(lab.DEFAULTS)
    if bad:
        raise ProtocolError(f"{where}: unknown params {sorted(bad)}; allowed: {sorted(lab.DEFAULTS)}")
    for spec, mode in p.get("surgery", {}).items():
        if mode not in (-1, 1):
            raise ProtocolError(f"{where}: surgery {spec!r} must be -1 (silence) or 1 (stimulate)")
    if "assay" in p:
        if p["assay"] not in labjobs.ASSAYS:
            raise ProtocolError(f"{where}: assay must be one of {list(labjobs.ASSAYS)}")
        _check_day2(p, where)
        return p
    if p.get("classroom"):
        _check_day2(p, where)
        return p
    if "patch" in p or "rig" in p or "screen" in p:
        _check_day2(p, where)
        return p
    for key, default in (("warmup_s", 1.0), ("duration_s", 2.0)):
        v = p.setdefault(key, default)
        if not isinstance(v, (int, float)) or v < 0 or v > 3600:
            raise ProtocolError(f"{where}: {key} must be a number of seconds (0-3600)")
    for i, s in enumerate(p.setdefault("stimuli", [])):
        if not isinstance(s, dict) or "target" not in s or "at_s" not in s:
            raise ProtocolError(f"{where}: stimulus {i + 1} needs at least 'at_s' and 'target'")
        if set(s) - STIM_KEYS:
            raise ProtocolError(f"{where}: stimulus {i + 1} has unknown keys {sorted(set(s) - STIM_KEYS)}")
        if s.get("mode", "poke") not in ("poke", "drive"):
            raise ProtocolError(f"{where}: stimulus {i + 1} mode must be poke or drive")
    recs = p.setdefault("recordings", [])
    if not isinstance(recs, list) or not all(isinstance(r, dict) and "neurons" in r for r in recs):
        raise ProtocolError(f"{where}: recordings must be a list of {{name, neurons}}")
    _check_day2(p, where)
    return p


def _rows(br, spec):
    from kickthefly.lab import assays
    from kickthefly.core import simcore

    g = assays.groups(br)
    if isinstance(spec, str) and spec in g:
        return g[spec]
    try:
        return simcore.rows_of(br, spec)
    except ValueError as e:
        raise ProtocolError(str(e)) from None


def run_seed(p: dict, seed: int, surgery: dict | None, folder: Path, tag: str, replay_to: Path | None = None) -> dict:
    """One fly of a stimulus protocol. Returns mean firing per recording group. replay_to: also record the run as a
    .ktfreplay (kickthefly/core/replay.py) there."""
    from kickthefly.lab import assays
    from kickthefly.lab import bundle as bundle_mod
    from kickthefly.lab import recorder
    from kickthefly.core import simcore

    wiring = None
    if p.get("drug"):
        from kickthefly.lab import pharmacology

        d = p["drug"]
        wiring = pharmacology.wiring_for(d["doses"], d.get("include_low_confidence", True),
                                         float(d.get("cut", pharmacology.DEFAULT_CUT)))
    br = simcore.new_brain(seed=seed, params=p.get("params"), wiring=wiring)
    if surgery:
        assays.apply_surgery(br, surgery)
    replay_rec = None
    if replay_to is not None:
        from kickthefly.core import replay

        replay_rec = replay.ReplayRecorder(
            seed=seed, backend=br.sim.backend.name, dtype=str(br.sim.p.dtype), signature=replay.pack_signature(br),
            arena="headless", settings=dict(params=p.get("params") or {}, surgery=surgery or {}, protocol=p["name"]),
        ).attach(br)
    assays.rest(br, int(round(p["warmup_s"] / 0.005)))
    groups = {r.get("name", f"rec{i}"): _rows(br, r["neurons"]) for i, r in enumerate(p["recordings"])}
    stims = []
    for s in p["stimuli"]:
        rows = _rows(br, s["target"])
        start = int(round(float(s["at_s"]) / 0.005))
        stims.append(dict(s, rows=rows, start=start, stop=start + max(1, int(round(float(s.get("for_s", 0.5)) / 0.005)))))
    rec = recorder.Recorder(br, groups).start()
    n = int(round(p["duration_s"] / 0.005))
    driving: set[int] = set()
    th = th_spec = session = frames = view = None
    if p.get("thermogenetics"):
        from kickthefly.lab import thermogenetics

        th_spec = p["thermogenetics"]
        th = thermogenetics.from_spec(th_spec)
    wx = wx_args = hum = hum_rows = loom_rates = None
    if p.get("weather"):
        from kickthefly.game import weather

        wx_args = p["weather"]
        wx = weather.Weather(seed)
    if p.get("audio"):
        from kickthefly.core import mic

        ad = p["audio"]
        an = mic.Analyzer(mic.RATE, ad["sensitivity"])
        wave = mic.hum(ad["hz"], ad["seconds"], mic.RATE, ad["amp"], ad["ipi_ms"])
        hum = [an.push(ch) for ch in mic.chunks(wave)]               # the synthetic hum, analysed once, deterministically
        hum_rows = mic.jo_rows(br)
    if p.get("predator"):
        from kickthefly.game import predators as pr
        from kickthefly.lab import predators as lp

        pd = p["predator"]
        loom_rates = lp.loom_rates(pr.trace(pd["kind"], seed))
    if p.get("imaging"):
        from kickthefly.lab import imaging

        im = p["imaging"]
        rois = imaging.rois_by_region(br) if im.get("rois", "regions") == "regions" else imaging.rois_from_specs(br, im["rois"])
        session = imaging.ImagingSession(br.n, rois, im.get("indicator", imaging.DEFAULT_INDICATOR), float(im.get("fps", 20)),
                                         float(im.get("f0_photons", 100.0)), float(im.get("dff_per_spike", 0.2)),
                                         bool(im.get("shot_noise", True)), seed=seed,
                                         baseline_tau_s=float(im.get("f0_tau_s", 30.0)))
        if im.get("tiff"):
            view, frames = imaging.make_view(br), []
    for t in range(n):
        if th is not None and t % 10 == 0:                       # every 50 ms: the temperature, then the effectors
            th.update(br, thermogenetics.temperature_at(th_spec, t * 0.005), 0.05)
        if wx is not None and t % 10 == 0:                       # every 50 ms: rain, gusts and lightning onto the real neurons
            _weather_step(br, wx, wx_args)
        if hum is not None:
            _hum_step(br, hum, hum_rows, p["audio"], t)
        if loom_rates is not None:
            _predator_step(br, loom_rates, p["predator"], t)
        for k, s in enumerate(stims):
            active = s["start"] <= t < s["stop"]
            if s.get("mode", "poke") == "drive":
                if active and k not in driving:
                    simcore.drive(br, s["rows"], float(s.get("amp", 0.5)))
                    driving.add(k)
                elif not active and k in driving:
                    simcore.undrive(br, s["rows"])
                    driving.discard(k)
            elif active and (t - s["start"]) % 10 == 0:
                _poke_rows(br, s)
        br._step()
        if session is not None and session.push(np.flatnonzero(br.sim.spikes)) and frames is not None:
            frames.append(imaging.render_frame(view, session, "default", "panel")[0])
    if th is not None:
        th.clear(br)
    if hum is not None:
        from kickthefly.core import mic

        mic.release(br)
    rec.stop()
    if replay_rec is not None:
        replay_rec.detach()
        replay_rec.save(replay_to)
    stem = folder / f"{tag}-seed{seed}"
    if session is not None:
        res = session.result(dict(protocol=p["name"], seed=seed))
        imaging.export_csv(res, stem.with_name(stem.name + "-imaging.csv"))
        if frames:
            imaging.export_tiff(frames, stem.with_name(stem.name + "-imaging.tif"), res.meta)
            res.frames = frames
        if p.get("nwb"):
            imaging.export_nwb(res, stem.with_name(stem.name + "-imaging.nwb"))
    a = rec.arrays()
    # 3.0: what a bundle (lab/bundle.py) needs to check a rerun: the backend and precision that really ran, and a
    # fingerprint of every recorded spike. Extra keys in the per-fly metadata; nothing the simulation reads.
    extra = dict(protocol=p["name"], protocol_spec={k: v for k, v in p.items() if k != "stimuli_rows"},
                 condition=tag, surgery=surgery, sim_backend=br.sim.backend.name, sim_dtype=str(br.sim.p.dtype),
                 spike_sha256=bundle_mod.spike_sha256(a["spike_steps"], a["spike_index"], a["n_steps"]))
    rec.save(stem, extra)
    if p.get("nwb"):
        from kickthefly.lab import nwbexport

        reason = nwbexport.available()
        if reason:
            raise ProtocolError(reason)
        nwbexport.write(rec, stem.with_name(stem.name + ".nwb"), recorder.metadata(br, None, extra))
    out = {}
    for name, rows in groups.items():
        members = np.searchsorted(rec.rows, rows)
        spikes = int(np.isin(a["spike_index"], members).sum())
        out[name] = spikes / max(1, len(rows)) / max(1e-9, a["n_steps"] * 0.005)
    return out


def _weather_step(br, wx, a: dict) -> None:
    """One 50 ms tick of the weather block, the same transduction as the game's (game/weather.py, game/outdoors.py)."""
    from kickthefly.game import outdoors

    wx.update(0.05, a["rain"], a["gust_hz"], a["storm"])
    for region, side, s in wx.hits(0.05):
        br.poke(region, side, s)
    h = wx.humid(0.05)
    if h > 0:
        br.poke("humid", None, h)
    ws, wd = wx.wind(a["wind_speed"], a["wind_dir"])
    if ws > 0:
        left, right = outdoors.wind_drive(0.0, wd, ws)
        if left > 0.02:
            br.poke("wind", "L", left)
        if right > 0.02:
            br.poke("wind", "R", right)
    light = wx.lightning()
    if light > 0:
        br.poke("light", "L", light, recruit=0.6 * light)
        br.poke("light", "R", light, recruit=0.6 * light)


def _hum_step(br, readings: list, rows, a: dict, t: int) -> None:
    """The synthetic hum's analysed current on JO-A/B, one reading per 256 samples (11.6 ms), released when it ends."""
    from kickthefly.core import mic

    sec = t * 0.005 - a["at_s"]
    if sec < 0:
        return
    i = int(sec * mic.RATE / mic.CHUNK)
    if i >= len(readings):
        if i == len(readings):
            mic.release(br)
        return
    mic.apply(br, readings[i], rows)


def _predator_step(br, rates: list, a: dict, t: int) -> None:
    """The predator's loom measure, frame by frame at 60 Hz, through the game's own transduction onto LPLC2/LC4."""
    from kickthefly.game import kick_the_fly as k

    f = int((t * 0.005 - a["at_s"]) * 60)
    if f < 0 or f >= len(rates):
        return
    if t > 0 and f == int(((t - 1) * 0.005 - a["at_s"]) * 60):      # still inside the same 1/60 s frame: already poked
        return
    strength = float(np.clip((rates[f] - k.LOOM_MIN) / k.LOOM_FULL, 0, 1))
    if strength > 0:
        br.poke("loom", None, strength, recruit=0.6 * strength)


def _poke_rows(br, s) -> None:
    """A game-style stimulus on arbitrary rows: a share of them driven for a short, renewed pulse."""
    strength = float(np.clip(s.get("strength", 0.8), 0, 1))
    key = ("protocol", str(s["target"]) + str(s["at_s"]))
    br.sense[key] = s["rows"]
    br.poke("protocol", key[1], strength, recruit=s.get("recruit"))


def run(p: dict, out: Path | None = None, workers: int | None = None, progress=None) -> Path:
    from kickthefly.lab import labjobs
    from kickthefly.lab import recorder
    from kickthefly.lab import labstats

    folder = (out or recorder.exports_dir()) / f"{time.strftime('%Y%m%d-%H%M%S')}-{folder_name(p['name'])}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "protocol.json").write_text(json.dumps(p, indent=1, default=str), encoding="utf-8")
    t0 = time.time()
    if "screen" in p:
        summary = run_screen(p, folder, progress)
        summary["seconds"] = round(time.time() - t0, 1)
    elif "rig" in p:
        summary = run_rig(p, folder, progress)
        summary["seconds"] = round(time.time() - t0, 1)
    elif "patch" in p:
        summary = run_patch(p, folder, progress)
        summary["seconds"] = round(time.time() - t0, 1)
    elif "assay" in p:
        res = labjobs.run_sync(p["assay"], p["seeds"], p.get("assay_options"), p.get("surgery"), p.get("params"),
                               workers or labjobs.default_workers(), progress=progress)
        recorder.export_result(res, None, folder)
        summary = dict(protocol=p["name"], assay=p["assay"], seeds=p["seeds"], seconds=round(time.time() - t0, 1),
                       treated=res["treated"], control=res.get("control"), comparison=res.get("comparison"))
    else:
        surgery = p.get("surgery") or None
        conditions = [("treated" if surgery else "run", surgery)]
        if surgery and p.get("control", True):
            conditions.append(("control", None))
        rates = {tag: {} for tag, _ in conditions}
        total, done = len(conditions) * len(p["seeds"]), 0
        for seed in p["seeds"]:
            for tag, sg in conditions:
                rates[tag][seed] = run_seed(p, seed, sg, folder, tag)
                done += 1
                if progress:
                    progress(done, total)
        groups = [r.get("name", f"rec{i}") for i, r in enumerate(p["recordings"])]
        summary = dict(protocol=p["name"], seeds=p["seeds"], seconds=round(time.time() - t0, 1), surgery=surgery,
                       mean_rate_hz={tag: {g: labstats.mean_ci([rates[tag][s][g] for s in p["seeds"]]) for g in groups}
                                     for tag in rates})
        if len(conditions) == 2:
            summary["comparison"] = {g: labstats.paired([rates["treated"][s][g] for s in p["seeds"]],
                                                        [rates["control"][s][g] for s in p["seeds"]]) for g in groups}
    (folder / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    return folder


def run_patch(p: dict, folder: Path, progress=None) -> dict:
    """A patch protocol: each seed's fly gets one neuron current-clamped (lab/patchclamp.py). MODEL, not electrophysiology."""
    from kickthefly.core import simcore
    from kickthefly.lab import assays, labstats
    from kickthefly.lab import patchclamp as pc

    pa = p["patch"]
    amps = pa.get("amplitudes", [0.0, 0.05, 0.1])
    per_seed, rheo, neuron = {}, {}, None
    for i, seed in enumerate(p["seeds"]):
        br = simcore.new_brain(seed=seed, params=p.get("params"))
        assays.rest(br, int(round(float(pa.get("warmup_s", 1.0)) / 0.005)))
        try:
            row = pc.pick_neuron(br, pa["neuron"], int(pa.get("index", 0)))
            curve = pc.if_curve(br, row, amps, float(pa.get("duration_ms", 500)), int(pa.get("repeats", 3)),
                                pa.get("mode", "embedded"), seed, getattr(br.sim, "p", None))
        except pc.PatchError as e:
            raise ProtocolError(str(e)) from None
        neuron = pc.describe_neuron(br, row)
        pc.export_csv(curve["recording"], folder / f"patch-seed{seed}-trace.csv")
        pc.export_if_csv(curve, folder / f"patch-seed{seed}-if.csv")
        if p.get("nwb"):
            pc.export_nwb(curve["recording"], folder / f"patch-seed{seed}.nwb")
        per_seed[seed] = dict(zip(curve["amplitudes"], curve["rate_hz"]))
        rheo[seed] = curve["rheobase"]
        if progress:
            progress(i + 1, len(p["seeds"]))
    by_amp = {f"{a:g}": labstats.mean_ci([per_seed[s][a] for s in p["seeds"]]) for a in sorted(set(amps))}
    return dict(protocol=p["name"], seeds=p["seeds"], patch=dict(pa), tag=pc.TAG_TEXT, units=pc.UNITS_NOTE, neuron=neuron,
                rate_hz_by_current=by_amp, rheobase_by_seed={str(k): v for k, v in rheo.items()})


def run_screen(p: dict, folder: Path, progress=None) -> dict:
    """A screen protocol (lab/screens.py): the whole-brain activation or knockout screen over the protocol's seeds, into folder/screen. To
    resume an interrupted one run the same protocol with --out pointing at the same folder name (the screen folder is fixed inside it)."""
    from kickthefly.lab import screens

    sc = dict(p["screen"])
    kind = sc.pop("kind")
    out = folder / "screen"
    cb = (lambda d, n, label: progress(d, n)) if progress else None
    common = dict(seeds=p["seeds"], controls=sc.get("controls", 2), workers=p.get("workers"), batch=sc.get("batch", 0),
                  backend=sc.get("backend"), types=sc.get("types"), progress=cb, stream=None)
    try:
        if kind == "activation":
            res = screens.run_activation(out, min_neurons=sc.get("min_neurons", 1), max_types=sc.get("max_types"), **common)
        else:
            res = screens.run_knockout(out, behaviors=sc.get("behaviors"), top=sc.get("top", 25), batch_size=sc.get("candidate_batch", 0), **common)
    except screens.ScreenError as e:
        raise ProtocolError(str(e)) from None
    head = res["rows"][:10]
    return dict(protocol=p["name"], seeds=p["seeds"], screen=dict(p["screen"]), tag=screens.TAG, criteria=res["criteria"],
                text=screens.summary(res), files=sorted(x.name for x in out.iterdir()), top_rows=[{k: v for k, v in r.items() if not k.startswith(("lo_", "hi_", "z_", "sign_", "p_", "q_", "eff_", "_"))} for r in head])


def run_rig(p: dict, folder: Path, progress=None) -> dict:
    """A rig protocol: each seed's fly in the rig (lab/rigassay.scene_run), recorded in the Lab's format. Tags: see rigs.RIG_TAGS."""
    from kickthefly.game import rigs
    from kickthefly.lab import rigassay

    r = dict(p["rig"])
    name, mode_ind = r.pop("name"), r.pop("individuality", rigassay.DEFAULT_MODE)
    per_seed = {}
    for i, seed in enumerate(p["seeds"]):
        res = rigassay.scene_run(name, seed, mode_ind, folder=folder, **r)
        per_seed[str(seed)] = res["summary"]
        if progress:
            progress(i + 1, len(p["seeds"]))
    return dict(protocol=p["name"], seeds=p["seeds"], rig=dict(p["rig"]), title=rigs.RIG_TITLE[name], tags=rigs.RIG_TAGS[name], rig_summary=per_seed)


def find(path: Path) -> Path:
    """A protocol file by path, or by name from your protocols folder or the bundled examples."""
    if Path(path).exists():
        return Path(path)
    from kickthefly.lab import lab

    for f in lab.protocol_files():
        if f.name == Path(path).name or f.stem == Path(path).name:
            return f
    raise FileNotFoundError(f"protocol file {path} not found (bundled: {', '.join(f.name for f in lab.protocol_files())})")


def record_replay(path: Path, dest: Path, out: Path | None = None) -> int:
    """--headless --protocol FILE --record-replay DEST: the protocol's first fly, recorded as a .ktfreplay."""
    path = find(path)
    try:
        p = load(path)
        if "assay" in p:
            raise ProtocolError("--record-replay records a stimulus protocol; assay protocols aren't supported")
        # 3.0 day 3 review: audio is a current too (Brain.set_current), so a replay of it did not reproduce the spikes; weather and
        # predator blocks are pokes, which are replay events (checked: both replay spike for spike)
        used = [k for k in DAY2_KEYS + ("audio",) if k in p]
        if used:
            raise ProtocolError(f"--record-replay does not record {', '.join(used)} (their currents are not replay events)")
    except ProtocolError as e:
        print(f"error: {e}")
        return 2
    from kickthefly.lab import recorder

    seed = p["seeds"][0]
    folder = (out or recorder.exports_dir()) / f"{time.strftime('%Y%m%d-%H%M%S')}-{folder_name(p['name'])}-replay"
    folder.mkdir(parents=True, exist_ok=True)
    run_seed(p, seed, p.get("surgery") or None, folder, "replay", replay_to=Path(dest))
    from kickthefly.core import replay

    rp = replay.ReplayPlayer.load(Path(dest))
    print(f"recorded {p['name']} seed {seed}: {rp.total_steps} steps on {rp.backend}, "
          f"{len(rp.meta['events'])} input events, spikes SHA-256 {rp.spike_sha256}")
    print(f"replay written to {dest}")
    return 0


def run_file(path: Path, out: Path | None = None, workers: int | None = None, nwb: bool = False,
             bundle_to: Path | None = None) -> int:
    path = find(path)
    try:
        p = load(path)
        if nwb:
            p["nwb"] = True
        if p.get("nwb"):
            from kickthefly.lab import nwbexport

            reason = nwbexport.available()
            if reason:
                raise ProtocolError(reason)
    except ProtocolError as e:
        print(f"error: {e}")
        return 2
    t0 = time.time()
    print(f"protocol {p['name']}: {len(p['seeds'])} fly(s)" + (f", assay {p['assay']}" if "assay" in p else "")
          + (", patch clamp (MODEL)" if "patch" in p else "") + (f", rig {p['rig']['name']}" if "rig" in p else "")
          + (f", {p['screen']['kind']} screen (MODEL PREDICTION)" if "screen" in p else ""), flush=True)
    folder = run(p, out, workers, progress=lambda d, n: print(f"  {d}/{n} ({time.time() - t0:.0f}s)", flush=True))
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    for tag, groups in summary.get("mean_rate_hz", {}).items():
        for g, c in groups.items():
            print(f"  {tag:8s} {g:20s} {c['mean']:8.2f} Hz  (n={c['n']})")
    if summary.get("text"):
        print(summary["text"])
    for sd_, sm in (summary.get("rig_summary") or {}).items():
        print(f"  seed {sd_}: " + ", ".join(f"{k} {v:.3g}" if isinstance(v, float) else f"{k} {v}" for k, v in sm.items() if v is not None))
    for a_, c in (summary.get("rate_hz_by_current") or {}).items():
        print(f"  current {a_:>6s}  {c['mean']:8.2f} Hz  (n={c['n']})   [{summary['tag']}]")
    for g, c in (summary.get("comparison") or {}).items():
        if isinstance(c, dict) and "p_value" in c:
            print(f"  {g}: surgery - control = {c['mean_difference']:+.2f}, p = {c['p_value']:.3g}")
    print(f"done in {time.time() - t0:.0f}s; results in {folder}")
    if bundle_to is not None:
        from kickthefly.lab import bundle

        try:
            z = bundle.create(folder, bundle_to)
        except bundle.BundleError as e:
            print(f"error: can't bundle this run: {e}")
            return 2
        print(f"bundle written to {z} (rerun it with --headless --rerun-bundle {z} --out DIR)")
    return 0
