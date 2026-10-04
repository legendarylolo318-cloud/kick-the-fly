"""Neurodata Without Borders export: one .nwb file holding everything a recording captured.

NWB (https://nwb.org) is the standard file format for neurophysiology data, so a run exported this way opens in
pynwb, the NWB inspector, NWB Explorer, MatNWB and the rest of the ecosystem without this game's code.

What goes in the file:
  units                      one row per recorded neuron: its spike times, plus the connectome's body id, cell type,
                             instance, superclass, region and the recording group it was in
  processing/ecephys         per-group and per-region firing rates in 50 ms bins (spikes/s per neuron)
  stimuli (TimeIntervals)    every sensory stimulus the brain was given: which sensory group, which side, how hard
  events (TimeIntervals)     tool uses, hits, reactions, arena and surgery changes, with their timing
  processing/behavior        the fly's position over time, its speed, and its health
  scratch/kc_mbon_weights    the plastic KC -> MBON synapses: connectome weight, weight when recording started and
                             weight when it stopped, with both neurons' rows and body ids
  scratch/surgery, /arena    the surgery state and arena the run was made in
  general + lab_meta         app version, seed, all LIF parameters, all game-rule thresholds, connectome version,
                             brain pack checksum and the MaleCNS v1.0 / CC BY 4.0 citation

Times are seconds from the start of the recording; the simulation steps at 5 ms, so spike times are exact multiples
of 0.005 s. Nothing here is a measurement from a living fly: it is the output of this simulation, and the file says
so in its session description.

pynwb is an optional dependency (it pulls in h5py and pandas). Without it the Lab export screen says so and the
CSV/npz export, which needs nothing extra, still works.

    pip install pynwb
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from kickthefly.core.version import __version__

CITATION = ("Janelia FlyEM MaleCNS v1.0 connectome (HHMI Janelia, University of Cambridge, MRC Laboratory of "
            "Molecular Biology, Google Research), licensed CC BY 4.0, https://male-cns.janelia.org/download/")
DOI = "https://doi.org/10.1101/2025.05.30.656568"
SPECIES = "Drosophila melanogaster"
BIN_S = 0.05                         # the rate bins the recorder writes (10 sim steps)
DT = 0.005


def available() -> str | None:
    """None when NWB export can run, else a one-line reason to show the user."""
    try:
        import pynwb  # noqa: F401
    except Exception as e:                                       # ImportError, or a broken h5py build
        return f"NWB export needs pynwb ({type(e).__name__}: {e}). Install it with: pip install pynwb"
    return None


def _clean(obj):
    """JSON-safe metadata: numpy scalars and arrays, Paths and anything else fall back to str."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _session_start(meta: dict) -> datetime:
    for key in ("recording_started", "created"):
        v = meta.get(key)
        if isinstance(v, str):
            try:
                d = datetime.fromisoformat(v)
                return d if d.tzinfo else d.astimezone()
            except ValueError:
                pass
    return datetime.now(timezone.utc)


def _intervals(name: str, description: str, columns: dict[str, str]):
    from pynwb.epoch import TimeIntervals

    t = TimeIntervals(name=name, description=description)
    for col, desc in columns.items():
        t.add_column(name=col, description=desc)
    return t


def write(rec, path: Path, meta: dict | None = None, game=None) -> Path:
    """Write one recording as NWB. `rec` is a recorder.Recorder, already stopped."""
    reason = available()
    if reason:
        raise RuntimeError(reason)
    from pynwb import NWBHDF5IO, NWBFile, TimeSeries
    from pynwb.behavior import BehavioralTimeSeries, Position, SpatialSeries
    from pynwb.file import Subject
    from pynwb.misc import Units

    from kickthefly.lab import recorder as rc

    meta = dict(meta or rc.metadata(rec.brain, game))
    meta.setdefault("recording_started", getattr(rec, "started_iso", None))
    a = rec.arrays()
    ctx = rec.context()
    st, idx = a["spike_steps"], a["spike_index"]
    n_steps = max(1, a["n_steps"])
    rows = rec.rows
    br = rec.brain
    body = getattr(br, "body_id", None)
    body_ids = body[rows] if body is not None else np.full(len(rows), -1, np.int64)
    regions = rec.region_of_rows()
    duration = n_steps * DT

    nwb = NWBFile(
        session_description=(
            "Kick the Fly: a leaky integrate-and-fire simulation of the Janelia FlyEM MaleCNS v1.0 connectome, "
            "recorded while the simulated fly was played with. Simulated data, not a recording from a living "
            "animal. What is read from the connectome and what is a game rule is listed in this file's "
            "game_rules metadata and in the project's README."),
        identifier=str(uuid.uuid4()),
        session_start_time=_session_start(meta),
        file_create_date=datetime.now(timezone.utc),
        lab="Kick the Fly (simulation)",
        institution="none (simulation)",
        experiment_description=f"Connectome simulation, app version {meta.get('app_version', __version__)}, "
                               f"seed {meta.get('seed', 'unknown')}.",
        session_id=str(meta.get("session_id", Path(path).stem)),
        keywords=["connectome", "Drosophila", "MaleCNS", "simulation", "leaky integrate-and-fire"],
        related_publications=[DOI],
        source_script="kickthefly/lab/nwbexport.py",
        source_script_file_name="nwbexport.py",
        notes=json.dumps(_clean(meta), indent=1),
        data_collection=(f"Simulated with {meta.get('app', 'Kick the Fly')} {meta.get('app_version', __version__)}; "
                         f"backend: {meta.get('sim_backend', getattr(getattr(br.sim, 'backend', None), 'name', 'unknown'))} "
                         f"({meta.get('sim_device', getattr(getattr(br.sim, 'backend', None), 'device', 'unknown'))}); "
                         f"timestep {DT * 1000:.0f} ms; connectome: {rc.CONNECTOME}. Data citation: {CITATION}"),
        stimulus_notes=("Stimuli are currents injected into the sensory neuron groups the connectome annotates; "
                        "their timing and strength come from the game and are listed in the 'stimuli' table."),
    )
    nwb.subject = Subject(
        subject_id=str(meta.get("seed", "0")), species=SPECIES, sex="M", age="P5D/P15D",
        description=("One simulated adult male fly brain: every neuron of MaleCNS v1.0 "
                     f"({meta.get('n_neurons', br.n):,} neurons, {meta.get('synapses', br.sim.W_csr.nnz):,} signed "
                     "synapses), run as leaky integrate-and-fire point neurons. " + CITATION),
    )

    # --- units: one row per recorded neuron -------------------------------------------------------------------
    nwb.units = Units(name="units", resolution=DT,
                      description="One row per recorded neuron. Spike times are exact multiples of the 5 ms "
                                  "simulation step, so the resolution is 0.005 s.")
    order = np.argsort(idx, kind="stable")
    idx_sorted, st_sorted = idx[order], st[order]
    bounds = np.searchsorted(idx_sorted, np.arange(len(rows) + 1))
    for col, desc in (("row", "neuron's row in the brain pack / connectome adjacency matrix"),
                      ("body_id", "FlyEM MaleCNS v1.0 bodyId, -1 if the brain pack predates body ids"),
                      ("cell_type", "MaleCNS v1.0 `type` annotation"),
                      ("instance", "MaleCNS v1.0 `instance` annotation (carries the side)"),
                      ("superclass", "MaleCNS v1.0 `superclass` annotation"),
                      ("region", "neuropil region derived from the dataset's class/somaNeuromere annotations"),
                      ("recording_group", "the group(s) this neuron was recorded as part of"),
                      ("mean_rate_hz", "spikes per second over the whole recording")):
        nwb.add_unit_column(name=col, description=desc)
    for i in range(len(rows)):
        times = st_sorted[bounds[i]:bounds[i + 1]].astype(np.float64) * DT
        nwb.add_unit(spike_times=times, row=int(rows[i]), body_id=int(body_ids[i]),
                     cell_type=str(br.types[rows[i]]), instance=str(br.instance[rows[i]]),
                     superclass=str(br.superclass[rows[i]]), region=str(regions[i]),
                     recording_group=str(rec.group_of[i]), mean_rate_hz=float(len(times) / duration))

    # --- firing rates: per group and per region ---------------------------------------------------------------
    ecephys = nwb.create_processing_module(
        name="ecephys", description="Firing rates in 50 ms bins, spikes/s per neuron, for the recorded groups and "
                                    "for each neuropil region they fall in.")
    n_bins = max(1, int(np.ceil(n_steps / (BIN_S / DT))))
    for label, series in (("group", _group_rates(rec, st, idx, n_steps, n_bins)),
                          ("region", rec.per_region_rates(st, idx, n_steps))):
        for name, values in series.items():
            safe = f"{label}_rate_{name}".replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
            ecephys.add(TimeSeries(name=safe, data=np.asarray(values, np.float32), unit="spikes/s",
                                   starting_time=0.0, rate=float(1.0 / BIN_S),
                                   description=f"Mean firing rate of the {name} {label}, {len(values)} bins of "
                                               f"{BIN_S * 1000:.0f} ms."))

    # --- stimuli and events -----------------------------------------------------------------------------------
    stim = _intervals("stimuli", "Sensory stimuli given to the simulated brain: a current into a share of the "
                                 "neurons of one annotated sensory group. Which group each hit drives comes from "
                                 "the connectome; the strength and timing are the game's.",
                      {"sensory_group": "the Brain.poke group driven (head, body, legs, wings, heat, cold, smell...)",
                       "side": "which side, where the group has one",
                       "strength": "0-1 drive strength passed to Brain.poke"})
    events = _intervals("events", "Tool uses, hits, and everything the game logged while recording: reactions read "
                                  "from descending neurons, arena changes and brain surgery.",
                        {"kind": "tool | hit | note", "label": "tool name, body region, or the reaction's first word",
                         "detail": "free text: the full reaction line, the tool used, or where the hit landed",
                         "value": "strength for hits, 0 otherwise"})
    n_stim = n_event = 0
    for step, kind, label, detail, value in ctx["events"]:
        t = float(step) * DT
        if kind == "stimulus":
            # Brain.poke holds the drive for 8-48 steps depending on strength; 40 ms is its floor.
            stim.add_row(start_time=t, stop_time=t + 0.04 + 0.2 * float(value), sensory_group=label,
                         side=detail or "both", strength=float(value))
            n_stim += 1
        else:
            # an event is instantaneous; its interval is the one 5 ms step it was logged during
            events.add_row(start_time=t, stop_time=t + DT, kind=kind, label=label, detail=detail, value=float(value))
            n_event += 1
    if n_stim:
        nwb.add_time_intervals(stim)
    if n_event:
        nwb.add_time_intervals(events)

    # --- behavior: where the fly was and how it was doing ------------------------------------------------------
    kin = ctx["kinematics"]
    if kin:
        t = np.array([k[0] for k in kin], np.float64) * DT
        xyz = np.array([[k[1], k[2], k[3]] for k in kin], np.float32)
        speed = np.array([k[4] for k in kin], np.float32)
        health = np.array([k[5] for k in kin], np.float32)
        unit = "meters" if meta.get("mode") == "3d" else "pixels"
        behavior = nwb.create_processing_module(
            name="behavior", description="The fly's body in the game world. The body, its physics and its damage "
                                         "are game rules; what triggers a movement is read from the neurons.")
        pos = Position(spatial_series=SpatialSeries(
            name="body_position", description=f"Centre of the fly's ragdoll, in game {unit} "
                                              f"({'x, y, z' if unit == 'meters' else 'x, y (z is always 0)'}). "
                                              "Sampled once per drawn frame, so the timestamps are irregular "
                                              "whenever the frame rate varies or the sim clock is slowed or paused.",
            data=xyz, unit=unit, timestamps=t, reference_frame="game world origin"))
        behavior.add(pos)
        behavior.add(BehavioralTimeSeries(time_series=[
            TimeSeries(name="speed", data=speed, unit=f"{unit}/s", timestamps=t,
                       description="Speed of the body centre between physics ticks (game rule)."),
            TimeSeries(name="health", data=health, unit="health points", timestamps=t,
                       description="Game-rule health, 0-100; not a physiological measure."),
        ], name="body_state"))

    # --- learned synapses, surgery, arena ----------------------------------------------------------------------
    _add_kc_mbon(nwb, ctx)
    _add_state_tables(nwb, meta, rec, ctx)
    nwb.add_scratch(json.dumps(_clean(meta), indent=1), name="kickthefly_metadata",
                    description="The recording's full metadata JSON: app version, seed, every LIF parameter, every "
                                "game-rule threshold, connectome version, brain pack checksum and surgery state.")
    nwb.add_scratch(json.dumps(_clean(_provenance(meta, rec, n_stim, n_event)), indent=1), name="game_rules",
                    description="What in this file comes from the connectome and what is a rule the game adds.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwb)
    return path


def _group_rates(rec, st, idx, n_steps, n_bins) -> dict[str, np.ndarray]:
    out = {}
    for name, grows in rec.groups.items():
        members = np.searchsorted(rec.rows, grows)
        mask = np.isin(idx, members)
        c = np.bincount(st[mask] // int(BIN_S / DT), minlength=n_bins)[:n_bins]
        out[name] = c / len(grows) / BIN_S
    return out


def _add_kc_mbon(nwb, ctx) -> None:
    """The plastic mushroom body synapses, before and after: the learning is on the connectome's own synapses."""
    from hdmf.common import DynamicTable, VectorData

    before, after = ctx["kc_mbon_before"], ctx["kc_mbon_after"]
    if before is None or after is None or ctx["kc_rows"] is None or len(before) != len(after):
        return
    cols = [
        VectorData(name="kc_row", description="Kenyon cell's row in the connectome", data=ctx["kc_rows"]),
        VectorData(name="mbon_row", description="mushroom body output neuron's row", data=ctx["mbon_rows"]),
        VectorData(name="compartment", description="True where PPL1 (punishment) dopamine gates this synapse, "
                                                   "False where PAM (reward) does",
                   data=np.asarray(ctx["punish_compartment"], bool)),
        VectorData(name="weight_connectome", description="the untrained weight from MaleCNS v1.0",
                   data=np.asarray(ctx["kc_mbon_connectome"], np.float32)),
        VectorData(name="weight_before", description="weight when the recording started",
                   data=np.asarray(before, np.float32)),
        VectorData(name="weight_after", description="weight when the recording stopped",
                   data=np.asarray(after, np.float32)),
    ]
    table = DynamicTable(
        name="kc_mbon_weights",
        description=("Every plastic Kenyon cell -> MBON synapse in the connectome, with its untrained weight and its "
                     "weight before and after this recording. The synapses and the dopamine gating are the "
                     "connectome's; the learning rule (dopamine + KC activity weakens the synapse), the learning "
                     "rate and what drives the dopamine neurons are game rules."),
        columns=cols)
    nwb.add_scratch(table)


def _add_state_tables(nwb, meta: dict, rec, ctx) -> None:
    from hdmf.common import DynamicTable, VectorData

    surgery = meta.get("surgery") or {}
    by_type = meta.get("surgery_by_type") or {}
    labels = list(surgery) + list(by_type)
    modes = [surgery[k] for k in surgery] + [by_type[k] for k in by_type]
    kinds = ["group"] * len(surgery) + ["cell type"] * len(by_type)
    table = DynamicTable(
        name="surgery",
        description=("Brain surgery in force during the recording: neuron groups or cell types held silenced (-1) "
                     "or stimulated (+1) with a constant current. The neurons are the connectome's; holding them "
                     "at a fixed current is the manipulation."),
        columns=[VectorData(name="target", description="neuron group or cell type", data=labels or [""]),
                 VectorData(name="kind", description="what the target names", data=kinds or [""]),
                 VectorData(name="mode", description="-1 silenced, +1 stimulated",
                            data=np.asarray(modes or [0], np.int8))])
    nwb.add_scratch(table)
    nwb.add_scratch(json.dumps(_clean(dict(arena=meta.get("arena", rec.arena or "room"),
                                           mode=meta.get("mode", "2d"),
                                           pain_level=meta.get("pain_level"),
                                           sim_speed=meta.get("sim_speed", 1.0))), indent=1),
                    name="arena_and_mode",
                    description="The arena the run happened in, 2D or 3D, the pain-neuron setting and the sim speed. "
                                "Arenas drive real sensory neurons (wind, humidity, light, heat); the arena physics "
                                "is a game rule.")


def _provenance(meta: dict, rec, n_stim: int, n_event: int) -> dict:
    return {
        "from_the_connectome": [
            "every neuron and synapse: MaleCNS v1.0, filtered to synapse count >= 3 and signed by the dataset's "
            "neurotransmitter predictions",
            "which sensory neurons each stimulus drives, and the descending neurons every reaction is read from",
            "the plastic KC -> MBON synapses and which dopamine neurons gate which compartment",
            "the cell type, instance, superclass, region and body id on every unit",
        ],
        "game_rules": [
            "which movement each neuron group's firing triggers, and the thresholds (see reaction_thresholds)",
            "the fly's body, its physics, its health and its damage",
            "the pain index, death, and how looming is converted into drive on LPLC2/LC4",
            "the learning rate, the forgetting speed, and what makes the dopamine neurons fire",
        ],
        "counts": {"units": int(len(rec.rows)), "stimuli": n_stim, "events": n_event,
                   "kinematics_samples": len(rec.kinematics)},
        "citation": CITATION,
        "license": "Connectome data CC BY 4.0. This file is simulation output, not a recording from an animal.",
        "app_version": meta.get("app_version", __version__),
        "backend": meta.get("sim_backend", getattr(getattr(rec.brain.sim, "backend", None), "name", "unknown")),
        "device": meta.get("sim_device", getattr(getattr(rec.brain.sim, "backend", None), "device", "unknown")),
    }


def inspect(path: Path) -> list[str]:
    """Run the NWB inspector over a file if it is installed; returns its messages (empty list = clean)."""
    try:
        from nwbinspector import inspect_nwbfile
    except Exception:
        return ["nwbinspector not installed (pip install nwbinspector)"]
    return [f"{m.importance.name}: {m.check_function_name} - {m.message}" for m in inspect_nwbfile(str(path))]
