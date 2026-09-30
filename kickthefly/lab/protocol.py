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

Neuron specs (simcore.rows_of and assays.groups): a game group ("loom", "escape", "head", "reward", "sweet"...), an
assay group ("dnp01", "mn9", "adn", "jo_ce", "mn_front"...), "type:A,B", "prefix:KC", "superclass:descending_neuron"
or "rows:1,2,3".

Output (a new folder per run): per seed the recorder's CSV, npz and metadata files (recorder.py), plus summary.json with
mean firing per recording group, and, with surgery, the paired comparison against the unperturbed controls. With
`nwb: true` (or --nwb on the command line) each fly's recording is also written as one NWB file (nwbexport.py).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

TOP_KEYS = {"name", "description", "title", "classroom", "steps", "seed", "seeds", "flies", "warmup_s", "duration_s", "params", "surgery", "control",
            "stimuli", "recordings", "assay", "assay_options", "workers", "nwb"}
STIM_KEYS = {"at_s", "for_s", "target", "strength", "recruit", "mode", "amp", "side"}


class ProtocolError(ValueError):
    pass


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
    if "seeds" in p:
        if not isinstance(p["seeds"], list) or not all(isinstance(s, int) for s in p["seeds"]):
            raise ProtocolError(f"{where}: seeds must be a list of integers")
    else:
        seed, flies = p.get("seed", 0), p.get("flies", 1)
        if not isinstance(seed, int) or not isinstance(flies, int) or flies < 1:
            raise ProtocolError(f"{where}: seed must be an integer and flies a positive integer")
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
        return p
    if p.get("classroom"):
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

    br = simcore.new_brain(seed=seed, params=p.get("params"))
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
    for t in range(n):
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
    rec.stop()
    if replay_rec is not None:
        replay_rec.detach()
        replay_rec.save(replay_to)
    stem = folder / f"{tag}-seed{seed}"
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

    folder = (out or recorder.exports_dir()) / f"{time.strftime('%Y%m%d-%H%M%S')}-{p['name']}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "protocol.json").write_text(json.dumps(p, indent=1, default=str), encoding="utf-8")
    t0 = time.time()
    if "assay" in p:
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
    except ProtocolError as e:
        print(f"error: {e}")
        return 2
    from kickthefly.lab import recorder

    seed = p["seeds"][0]
    folder = (out or recorder.exports_dir()) / f"{time.strftime('%Y%m%d-%H%M%S')}-{p['name']}-replay"
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
    print(f"protocol {p['name']}: {len(p['seeds'])} fly(s)" + (f", assay {p['assay']}" if "assay" in p else ""), flush=True)
    folder = run(p, out, workers, progress=lambda d, n: print(f"  {d}/{n} ({time.time() - t0:.0f}s)", flush=True))
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    for tag, groups in summary.get("mean_rate_hz", {}).items():
        for g, c in groups.items():
            print(f"  {tag:8s} {g:20s} {c['mean']:8.2f} Hz  (n={c['n']})")
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
