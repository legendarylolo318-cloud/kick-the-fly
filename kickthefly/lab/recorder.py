"""Recording and export: spike times and firing rates to CSV and npz, always with a metadata JSON.

A Recorder attaches to one brain and keeps every spike of the neurons it watches (row indices into the connectome),
cheaply, on the brain's own thread. Alongside the spikes it keeps the run's context: every stimulus the brain was
given (Brain.poke), every tool use, hit, reaction, arena and surgery change the game logged, where the fly was and
how it was doing, and the learned KC -> MBON weights as they were when recording started. CSV and npz are the quick
export; nwbexport.py writes the same run as one NWB file.

Saving writes, next to each other:
  <name>-spikes.csv      time_ms, row, body_id, type, instance, group       one line per spike
  <name>-rates.csv       row, body_id, type, instance, group, spikes, rate_hz
  <name>-group-rates.csv time_ms, one column per group: spikes/s per neuron in 50 ms bins
  <name>.npz             the same data as arrays (spike_steps, spike_rows, rows, body_ids, types, ...)
  <name>-events.csv      time_ms, kind (stimulus | tool | hit | note), label, detail, value
  <name>-kinematics.csv  time_ms, x, y, z, speed, health, action, arena
  <name>-metadata.json   app version, seed, parameters, thresholds, connectome version, brain pack, surgery, what was
                         recorded and when

Exports go to the "exports" folder in the data folder (Documents\\Kick the Fly\\exports on Windows,
~/.local/share/kickthefly/exports on Linux) unless a protocol or command line names another.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import threading
import time
from pathlib import Path

import numpy as np

from kickthefly.core.version import __version__

CONNECTOME = "Janelia FlyEM MaleCNS v1.0 (minconf 0.5, synapse count >= 3, signed by predicted neurotransmitter)"
BIN_STEPS = 10


def exports_dir() -> Path:
    from kickthefly.core import paths

    return paths.ensure_dir(paths.get().data_dir / "exports")


def pack_info() -> dict:
    from kickthefly.sim import brainpack

    p = brainpack.find()
    if p is None:
        return {}
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return dict(file=p.name, sha256=h.hexdigest(), size_bytes=p.stat().st_size)


_pack_info_cache: dict | None = None


def metadata(brain=None, game=None, extra: dict | None = None) -> dict:
    """Everything needed to know how a recording or result was made."""
    global _pack_info_cache
    from kickthefly.game import kick_the_fly as k
    from kickthefly.lab import lab

    if _pack_info_cache is None:
        _pack_info_cache = pack_info()
    meta = dict(app="Kick the Fly", app_version=__version__, created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                platform=f"{platform.system()} {platform.release()}", python=platform.python_version(),
                connectome=CONNECTOME, brain_pack=_pack_info_cache, dt_ms=5.0,
                reaction_thresholds=dict(k.THRESH), looming=dict(loom_min=k.LOOM_MIN, loom_full=k.LOOM_FULL))
    if brain is not None:
        sim = brain.sim
        meta.update(seed=int(getattr(brain, "seed", 0)), n_neurons=int(brain.n), synapses=int(sim.W_csr.nnz),
                    lif_params={kk: (list(v) if isinstance(v, tuple) else v) for kk, v in vars(sim.p).items()},
                    surgery_active=bool(brain.surgery),
                    surgery_neurons=dict(silenced=int(np.count_nonzero(brain.override < 0)),
                                         stimulated=int(np.count_nonzero(brain.override > 0))),
                    pain_level=int(brain.pain_level))
    if game is not None:
        w = getattr(game, "wiring", None)
        if w is not None:
            meta["wiring"] = w.as_dict()
        arena = k.ARENAS[game.arena_i]
        if arena in k.THREE_D_ONLY:                    # the weather and fruit settings the run was made under
            meta["arena_params"] = {n: v for n, v in game.lab_params.items()
                                    if n.startswith(("field.", "outdoor.", "orchard.", "weather."))}
        meta.update(mode="3d" if game.three_d else "2d", arena=k.ARENAS[game.arena_i],
                    lab_params=dict(game.lab_params), lab_params_modified=lab.modified(game.lab_params),
                    surgery={label: mode for (label, _), mode in zip(k.SURGERY, game.surgery_modes) if mode},
                    surgery_by_type=dict(game.type_ops), sim_speed=game.clock.scale)
    meta.update(extra or {})
    return meta


class Recorder:
    def __init__(self, brain, groups: dict[str, np.ndarray]):
        self.brain = brain
        self.groups = {name: np.asarray(rows, np.int64) for name, rows in groups.items() if len(rows)}
        rows = np.unique(np.concatenate(list(self.groups.values()))) if self.groups else np.zeros(0, np.int64)
        self.rows = rows
        self.group_of = np.full(len(rows), "", dtype=object)
        for name, r in self.groups.items():
            idx = np.searchsorted(rows, r)
            for i in idx:
                self.group_of[i] = name if not self.group_of[i] else self.group_of[i] + "|" + name
        self.lock = threading.Lock()
        self.steps: list[np.ndarray] = []
        self.idx: list[np.ndarray] = []
        self.start_step = int(brain.steps)
        self.end_step = self.start_step
        self.started = time.time()
        self.started_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.active = False
        # run context, each entry stamped with the brain step so it lines up with the spikes
        self.events: list[tuple[int, str, str, str, float]] = []      # step, kind, label, detail, value
        self.kinematics: list[tuple[int, float, float, float, float, float, str, str]] = []
        self.kc_mbon_before: np.ndarray | None = None
        self.arena = ""

    def start(self) -> "Recorder":
        self.start_step = self.end_step = int(self.brain.steps)
        self.active = True
        mem = getattr(self.brain, "memory", None)
        if mem is not None:
            with mem.lock:
                self.kc_mbon_before = np.asarray(mem.w, np.float32).copy()
        self.brain.recorder = self
        return self

    # --- context the brain and the game hand in while a recording runs --------------------------------------------
    def log_event(self, kind: str, label: str, detail: str = "", value: float = 0.0) -> None:
        """A stimulus, tool use, hit, reaction, arena or surgery change, at the brain step it happened on."""
        if not self.active:
            return
        with self.lock:
            if len(self.events) < 200_000:                      # a runaway log never eats the recording
                self.events.append((int(self.brain.steps), kind, str(label), str(detail), float(value)))

    def log_kinematics(self, x: float, y: float, z: float, speed: float, health: float, action: str,
                       arena: str) -> None:
        """Where the fly is and how it is doing, sampled by the game loop (never by the brain thread)."""
        if not self.active:
            return
        self.arena = arena
        with self.lock:
            if len(self.kinematics) < 200_000:
                self.kinematics.append((int(self.brain.steps), float(x), float(y), float(z), float(speed),
                                        float(health), str(action), str(arena)))

    def stop(self) -> None:
        self.active = False
        if getattr(self.brain, "recorder", None) is self:
            self.brain.recorder = None

    def push(self, step: int, spikes: np.ndarray) -> None:
        hit = np.flatnonzero(spikes[self.rows])
        with self.lock:
            self.end_step = step + 1
            if len(hit):
                self.idx.append(hit.astype(np.int32))
                self.steps.append(np.full(len(hit), step - self.start_step, np.int32))

    @property
    def seconds(self) -> float:
        return (self.end_step - self.start_step) * 0.005

    def arrays(self) -> dict:
        with self.lock:
            idx = np.concatenate(self.idx) if self.idx else np.zeros(0, np.int32)
            st = np.concatenate(self.steps) if self.steps else np.zeros(0, np.int32)
            n_steps = self.end_step - self.start_step
        return dict(spike_steps=st, spike_index=idx, n_steps=n_steps)

    def context(self) -> dict:
        """The run's events, kinematics and learned KC -> MBON weights, with times relative to the recording."""
        with self.lock:
            events = [(s - self.start_step, k, lb, d, v) for s, k, lb, d, v in self.events]
            kin = [(s - self.start_step, *rest) for s, *rest in self.kinematics]
        mem = getattr(self.brain, "memory", None)
        before = self.kc_mbon_before
        after = None
        if mem is not None:
            with mem.lock:
                after = np.asarray(mem.w, np.float32).copy()
                baseline = np.asarray(mem.w0, np.float32).copy()
                kc_rows = np.asarray(mem.kc, np.int64)[mem.syn_kc]
                mbon_rows = np.asarray(mem.mbon, np.int64)[mem.syn_mbon]
                punish = np.asarray(mem.punish_syn, bool).copy()
        else:
            baseline = kc_rows = mbon_rows = punish = None
        return dict(events=events, kinematics=kin, kc_mbon_before=before, kc_mbon_after=after,
                    kc_mbon_connectome=baseline, kc_rows=kc_rows, mbon_rows=mbon_rows, punish_compartment=punish)

    def region_of_rows(self) -> np.ndarray:
        """The dataset's neuropil region label for each recorded neuron ('unassigned' where the dataset has none)."""
        region = getattr(self.brain, "region", None)
        if region is None or not len(self.rows):
            return np.full(len(self.rows), "unassigned")
        return np.asarray(region)[self.rows].astype(str)

    def per_region_rates(self, st: np.ndarray, idx: np.ndarray, n_steps: int) -> dict[str, np.ndarray]:
        """Spikes/s per neuron in 50 ms bins for each dataset region the recorded neurons fall in."""
        if not len(self.rows):
            return {}
        names = self.region_of_rows()
        n_bins = max(1, int(np.ceil(n_steps / BIN_STEPS)))
        out = {}
        for name in sorted(set(names.tolist())):
            members = np.flatnonzero(names == name)
            mask = np.isin(idx, members)
            c = np.bincount(st[mask] // BIN_STEPS, minlength=n_bins)[:n_bins]
            out[name] = c / len(members) / (BIN_STEPS * 0.005)
        return out

    def save(self, stem: Path, meta_extra: dict | None = None, game=None) -> list[Path]:
        br = self.brain
        a = self.arrays()
        st, idx, n_steps = a["spike_steps"], a["spike_index"], max(1, a["n_steps"])
        rows = self.rows
        body = getattr(br, "body_id", None)
        body_ids = body[rows] if body is not None else np.full(len(rows), -1, np.int64)
        types, inst = br.types[rows], br.instance[rows]
        counts = np.bincount(idx, minlength=len(rows)) if len(rows) else np.zeros(0, int)
        stem.parent.mkdir(parents=True, exist_ok=True)
        out = []
        p = stem.with_name(stem.name + "-spikes.csv")
        with open(p, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["time_ms", "row", "body_id", "type", "instance", "group"])
            order = np.lexsort((idx, st))
            for s_, i in zip(st[order], idx[order]):
                w.writerow([f"{s_ * 5.0:.1f}", int(rows[i]), int(body_ids[i]), types[i], inst[i], self.group_of[i]])
        out.append(p)
        p = stem.with_name(stem.name + "-rates.csv")
        with open(p, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["row", "body_id", "type", "instance", "group", "spikes", "rate_hz"])
            for i in range(len(rows)):
                w.writerow([int(rows[i]), int(body_ids[i]), types[i], inst[i], self.group_of[i], int(counts[i]),
                            f"{counts[i] / (n_steps * 0.005):.3f}"])
        out.append(p)
        n_bins = int(np.ceil(n_steps / BIN_STEPS))
        binned = {}
        for name, grows in self.groups.items():
            members = np.searchsorted(rows, grows)
            mask = np.isin(idx, members)
            c = np.bincount(st[mask] // BIN_STEPS, minlength=n_bins)[:n_bins]
            binned[name] = c / len(grows) / (BIN_STEPS * 0.005)
        p = stem.with_name(stem.name + "-group-rates.csv")
        with open(p, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["time_ms"] + list(binned))
            for b in range(n_bins):
                w.writerow([f"{b * BIN_STEPS * 5.0:.0f}"] + [f"{binned[g][b]:.3f}" for g in binned])
        out.append(p)
        ctx = self.context()
        if ctx["events"]:
            p = stem.with_name(stem.name + "-events.csv")
            with open(p, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["time_ms", "kind", "label", "detail", "value"])
                for st_, kind, label, detail, value in ctx["events"]:
                    w.writerow([f"{st_ * 5.0:.1f}", kind, label, detail, f"{value:.4f}"])
            out.append(p)
        if ctx["kinematics"]:
            p = stem.with_name(stem.name + "-kinematics.csv")
            with open(p, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["time_ms", "x", "y", "z", "speed", "health", "action", "arena"])
                for st_, x, y, z, sp, hp, act, arena in ctx["kinematics"]:
                    w.writerow([f"{st_ * 5.0:.1f}", f"{x:.3f}", f"{y:.3f}", f"{z:.3f}", f"{sp:.3f}", f"{hp:.1f}",
                                act, arena])
            out.append(p)
        p = stem.with_name(stem.name + ".npz")
        np.savez_compressed(p, spike_steps=st, spike_rows=rows[idx] if len(idx) else np.zeros(0, np.int64),
                            rows=rows, body_ids=body_ids, types=types.astype(str), instances=inst.astype(str),
                            groups=self.group_of.astype(str), spike_counts=counts, dt_ms=5.0, n_steps=n_steps,
                            group_names=np.array(list(binned)), group_rates=np.array([binned[g] for g in binned])
                            if binned else np.zeros((0, 0)))
        out.append(p)
        meta = metadata(br, game, dict(recording=dict(start_brain_step=self.start_step, steps=n_steps,
                                                      seconds=n_steps * 0.005, neurons=int(len(rows)),
                                                      spikes=int(len(idx)),
                                                      groups={g: int(len(r)) for g, r in self.groups.items()},
                                                      events=len(ctx["events"]),
                                                      kinematics_samples=len(ctx["kinematics"]),
                                                      files=[q.name for q in out])))
        meta.update(meta_extra or {})
        p = stem.with_name(stem.name + "-metadata.json")
        p.write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
        out.append(p)
        return out


def export_result(res: dict, game=None, folder: Path | None = None) -> Path:
    """An assay or trials result: result.json, per-fly CSV and metadata.json in a new folder."""
    folder = folder or exports_dir() / f"{time.strftime('%Y%m%d-%H%M%S')}-{res.get('kind', 'result')}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "result.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    with open(folder / "per_fly.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "seed", "score"])
        for group, key in (("treated", "treated"), ("control", "control")):
            if key in res:
                for s, v in zip(res["seeds"], res[key]["per_fly"]):
                    w.writerow([group, s, v])
    meta = metadata(game.brain if game is not None else None, game,
                    dict(assay=res.get("kind"), seeds=res.get("seeds"), surgery=res.get("surgery"),
                         assay_options=res.get("options"), assay_params=res.get("params")))
    (folder / "metadata.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    return folder
