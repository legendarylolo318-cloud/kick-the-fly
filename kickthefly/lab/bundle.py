"""Experiment bundles (3.0): one zip that holds an experiment and everything needed to check it, and to run it again.

    Lab > Record and export > Bundle       bundle the last protocol run (or, failing that, the last live recording)
    KickTheFly --headless --protocol FILE --bundle OUT.zip   run a protocol and bundle it
    KickTheFly --headless --rerun-bundle BUNDLE.zip --out DIR [--backend NAME]

What is in it:
    protocol.yaml            the experiment (for a live recording: a description only, marked not rerunnable)
    results/                 summary.json (and, for assays, result.json and per_fly.csv)
    raw/                     the recorder's CSV / npz / NWB files and per-fly metadata, untouched
    metadata.json            app version, simulation backend and state precision as they actually ran, every seed, the
                             brain pack's SHA-256, Lab parameters, surgery, individuality, arena, platform, the spike
                             SHA-256 of every fly, and whether (and why not) it can be rerun
    ro-crate-metadata.json   an RO-Crate 1.1 description of the above (https://w3id.org/ro/crate/1.1), each file with its
                             SHA-256, so the bundle can be checked without this program
    README.txt               the same in words

GAME RULE / not science: a bundle is a container and a provenance record. It adds nothing to the simulation and claims
no biological result. The connectome it was computed from is named in the crate with that dataset's own license
(CC BY 4.0, from the brain pack's documentation); the bundle's own license is left for its author to choose, so none is
stated.

Rerun (`rerun`): refuses a bundle from another brain pack or a newer bundle format, and a damaged one (a file whose hash
no longer matches). Otherwise it runs the protocol again with the recorded state precision and (unless --backend says
otherwise) the recorded backend, and judges:
  bit-exact   when both the recorded and the rerun backend are CPU-side (cpu, numba, torch-cpu; tests/test_backends.py
              holds them bit-identical) and the precision is the same: every fly's spike SHA-256 must be equal. Any
              difference is a MISMATCH.
  statistical when either side is a GPU backend (torch-cuda, torch-rocm, gl: held only to a statistical tolerance of
              NumPy). Spikes are not compared. Each recording group's across-seed mean rate must lie within the
              original's 95% confidence interval widened by STAT_SLACK_REL of the original mean (with one seed, which has
              no interval, within STAT_SINGLE_REL of it plus STAT_SINGLE_ABS_HZ). Assays: every mean with a confidence
              interval must satisfy the same rule. These three numbers were fixed before any rerun and are the
              game's choice, not a measured tolerance.
Nothing here uses the network or the microphone, and a rerun is never part of --validate.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import platform
import re
import time
import zipfile
from pathlib import Path

import numpy as np

from kickthefly.core.version import __version__

FORMAT = "kick-the-fly-bundle"
FORMAT_VERSION = 1
BIT_EXACT_BACKENDS = ("cpu", "numba", "torch-cpu")
STAT_SLACK_REL = 0.10          # widen the original 95% CI by this share of its mean
STAT_SINGLE_REL = 0.25         # one seed: no CI, so within this share of the mean ...
STAT_SINGLE_ABS_HZ = 0.5       # ... plus this many Hz
MAX_META_BYTES = 16 << 20
MAX_FILE_BYTES = 8 << 30        # no single file in a bundle is read past this (a zip bomb is refused, not expanded)
CRATE_CONTEXT = "https://w3id.org/ro/crate/1.1/context"
CRATE_SPEC = "https://w3id.org/ro/crate/1.1"
CC_BY_4 = "https://creativecommons.org/licenses/by/4.0/"
FIXED_ZIP_TIME = (2000, 1, 1, 0, 0, 0)


class BundleError(Exception):
    """A bundle that can't be read, is damaged, or can't be rerun here; str(e) is the reason."""


# --- helpers ---------------------------------------------------------------------------------------------------------------
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def spike_sha256(spike_steps: np.ndarray, spike_index: np.ndarray, n_steps: int) -> str:
    """A fingerprint of a recording's spikes: (step, recorded-neuron index) pairs in step order, and the step count."""
    st = np.asarray(spike_steps, np.int64)
    ix = np.asarray(spike_index, np.int64)
    order = np.lexsort((ix, st))
    h = hashlib.sha256()
    h.update(np.int64(n_steps).tobytes())
    h.update(st[order].tobytes())
    h.update(ix[order].tobytes())
    return h.hexdigest()


def _safe_name(name: str) -> bool:
    p = Path(name)
    return not (p.is_absolute() or ".." in p.parts or name.startswith(("/", "\\")) or "\\" in name or ":" in name)


def probe_backend() -> str:
    """The simulation backend a fresh brain actually gets on this machine (the name 'auto' resolves to)."""
    from kickthefly.core import simcore

    br = simcore.new_brain(seed=0, memory=False, warmup=0)
    try:
        return str(br.sim.backend.name)
    finally:
        br.stop()


@contextlib.contextmanager
def _env(**kv):
    old = {k: os.environ.get(k) for k in kv}
    try:
        for k, v in kv.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# --- creating -------------------------------------------------------------------------------------------------------------
def _protocol_yaml(p: dict) -> str:
    import yaml

    clean = {k: v for k, v in p.items()}
    return yaml.safe_dump(json.loads(json.dumps(clean, default=str)), sort_keys=False, allow_unicode=False)


def _run_manifest(run_folder: Path, p: dict) -> dict:
    """Per-fly facts the run wrote (spike hashes, backend, precision) gathered from its metadata files."""
    flies = {}
    for f in sorted(run_folder.glob("*-seed*-metadata.json")):
        try:
            m = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        key = f.name[: -len("-metadata.json")]
        flies[key] = {k: m.get(k) for k in ("seed", "sim_backend", "sim_dtype", "spike_sha256", "n_neurons", "synapses",
                                            "lab_params_modified", "surgery", "condition")}
        flies[key]["lif_params"] = m.get("lif_params")
    return flies


def create(run_folder: Path, out_zip: Path, extra: dict | None = None) -> Path:
    """Bundle a protocol run (the folder `protocol.run` made). Returns out_zip."""
    run_folder, out_zip = Path(run_folder), Path(out_zip)
    if not (run_folder / "protocol.json").exists() or not (run_folder / "summary.json").exists():
        raise BundleError(f"{run_folder} is not a protocol run folder (no protocol.json and summary.json)")
    p = json.loads((run_folder / "protocol.json").read_text(encoding="utf-8"))
    summary = json.loads((run_folder / "summary.json").read_text(encoding="utf-8"))
    from kickthefly.core import replay

    flies = _run_manifest(run_folder, p)
    backends = {f.get("sim_backend") for f in flies.values() if f.get("sim_backend")}
    dtypes = {f.get("sim_dtype") for f in flies.values() if f.get("sim_dtype")}
    if "assay" in p:
        backend = probe_backend()                   # assay workers don't record theirs: what a fresh brain gets here
        dtype = os.environ.get("KICK_THE_FLY_SIM_DTYPE", "float32")
        rerunnable, why = True, None
    else:
        if len(backends) > 1 or len(dtypes) > 1:
            raise BundleError(f"the run mixed backends {sorted(backends)} / precisions {sorted(dtypes)}; it can't be bundled")
        backend = next(iter(backends), None) or probe_backend()
        dtype = next(iter(dtypes), None) or "float32"
        rerunnable = bool(flies) and all(f.get("spike_sha256") for f in flies.values())
        why = None if rerunnable else "the run predates spike fingerprints (made by a version before 3.0)"
    meta = {
        "format": FORMAT, "format_version": FORMAT_VERSION, "kind": "assay" if "assay" in p else "protocol",
        "app": "Kick the Fly", "app_version": __version__, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": f"{platform.system()} {platform.release()}", "python": platform.python_version(),
        "protocol_name": p.get("name"), "seeds": p.get("seeds"), "backend": backend, "dtype": dtype,
        "brain_pack_sha256": replay.pack_sha256(), "parameters": p.get("params") or {}, "surgery": p.get("surgery") or {},
        "individuality": "off", "arena": "headless", "rerunnable": rerunnable, "not_rerunnable_reason": why,
        "flies": flies, "tags": {"container": "GAME RULE", "numbers": "CONNECTOME (the simulation's own output)"},
    }
    meta.update(extra or {})
    files: dict[str, bytes | Path] = {"protocol.yaml": _protocol_yaml(p).encode("utf-8"),
                                      "results/summary.json": run_folder / "summary.json"}
    for f in sorted(run_folder.iterdir()):
        if f.name in ("protocol.json", "summary.json") or not f.is_file():
            continue
        dest = "results/" + f.name if f.name in ("result.json", "per_fly.csv", "metadata.json") else "raw/" + f.name
        files[dest] = f
    return _write(out_zip, meta, files, p)


def create_live(files: list[Path], meta_extra: dict, out_zip: Path, description: dict) -> Path:
    """Bundle a live Lab recording (what the Lab's Record and export page writes). It is a record, not a rerunnable
    experiment: the stimuli were delivered by hand, so the bundle says so and `rerun` refuses with that reason."""
    from kickthefly.core import replay

    meta = {
        "format": FORMAT, "format_version": FORMAT_VERSION, "kind": "live-recording", "app": "Kick the Fly",
        "app_version": __version__, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": f"{platform.system()} {platform.release()}", "python": platform.python_version(),
        "brain_pack_sha256": replay.pack_sha256(), "individuality": "off", "rerunnable": False,
        "not_rerunnable_reason": "recorded live: the stimuli were delivered by hand (see the events file in raw/), "
                                 "so there is no script to run again",
        "flies": {}, "tags": {"container": "GAME RULE", "numbers": "CONNECTOME (the simulation's own output)"},
    }
    meta.update(meta_extra)
    blob = {}
    for f in files:
        f = Path(f)
        if f.is_file():
            blob["raw/" + f.name] = f
    return _write(Path(out_zip), meta, blob, description)


def _crate(meta: dict, entries: dict[str, dict], p: dict | None) -> dict:
    from kickthefly.lab import recorder

    parts = [{"@id": n} for n in sorted(entries) if n != "ro-crate-metadata.json"]
    graph = [
        {"@type": "CreativeWork", "@id": "ro-crate-metadata.json", "conformsTo": {"@id": CRATE_SPEC},
         "about": {"@id": "./"}},
        {"@id": "./", "@type": "Dataset",
         "name": f"Kick the Fly experiment: {meta.get('protocol_name') or meta.get('kind')}",
         "description": "An experiment run with the Kick the Fly connectome simulation, with its protocol, results, raw "
                        "exports and the provenance needed to run it again. A container: it makes no biological claim.",
         "datePublished": meta["created"], "hasPart": parts, "isBasedOn": {"@id": "#malecns-v1.0"},
         "mentions": {"@id": "#kick-the-fly"}},
        {"@id": "#kick-the-fly", "@type": "SoftwareApplication", "name": "Kick the Fly", "version": meta["app_version"],
         "description": f"simulation backend {meta.get('backend')}, state precision {meta.get('dtype')}"},
        {"@id": "#malecns-v1.0", "@type": "Dataset", "name": "MaleCNS v1.0 connectome (Janelia FlyEM)",
         "description": recorder.CONNECTOME, "license": CC_BY_4,
         "identifier": {"@type": "PropertyValue", "name": "brain pack SHA-256", "value": meta.get("brain_pack_sha256")}},
    ]
    for name, e in sorted(entries.items()):
        if name == "ro-crate-metadata.json":
            continue
        fmt = {".yaml": "application/yaml", ".json": "application/json", ".csv": "text/csv", ".npz": "application/zip",
               ".txt": "text/plain", ".nwb": "application/x-hdf5"}.get(Path(name).suffix, "application/octet-stream")
        graph.append({"@id": name, "@type": "File", "name": name, "encodingFormat": fmt, "contentSize": e["size"],
                      "sha256": e["sha256"]})
    if "protocol.yaml" in entries:
        graph.append({"@id": "#run", "@type": "CreateAction", "name": "Run the protocol", "instrument": {"@id": "#kick-the-fly"},
                      "object": {"@id": "protocol.yaml"}, "endTime": meta["created"],
                      "result": [{"@id": n} for n in sorted(entries) if n.startswith(("results/", "raw/"))]})
    return {"@context": CRATE_CONTEXT, "@graph": graph}


def _readme(meta: dict) -> str:
    lines = [f"Kick the Fly experiment bundle ({meta.get('kind')}), format {FORMAT_VERSION}",
             f"made by Kick the Fly {meta['app_version']} on {meta['platform']}, {meta['created']}", "",
             f"backend {meta.get('backend')}, precision {meta.get('dtype')}, brain pack SHA-256 "
             f"{meta.get('brain_pack_sha256')}", ""]
    if meta.get("rerunnable"):
        lines += ["Run it again:",
                  "    KickTheFly --headless --rerun-bundle THIS.zip --out DIR",
                  "CPU backends must reproduce every fly's spikes exactly; GPU backends are compared statistically "
                  "(see kickthefly/lab/bundle.py).", ""]
    else:
        lines += [f"This bundle can't be rerun: {meta.get('not_rerunnable_reason')}", ""]
    lines += ["metadata.json lists every parameter; ro-crate-metadata.json (RO-Crate 1.1) describes each file with its "
              "SHA-256.", "A bundle is a container. It is not a claim that the result is biologically true."]
    return "\n".join(lines) + "\n"


def _write(out_zip: Path, meta: dict, files: dict, protocol: dict | None) -> Path:
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, bytes] = {}
    for name, src in files.items():
        data[name] = src if isinstance(src, bytes) else Path(src).read_bytes()
    data["README.txt"] = _readme(meta).encode("utf-8")
    data["metadata.json"] = json.dumps(meta, indent=1, default=str, sort_keys=True).encode("utf-8")
    entries = {n: {"size": len(b), "sha256": sha256_bytes(b)} for n, b in data.items()}
    entries["ro-crate-metadata.json"] = {}
    data["ro-crate-metadata.json"] = json.dumps(_crate(meta, entries, protocol), indent=1, default=str).encode("utf-8")
    tmp = out_zip.with_name(out_zip.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for name in sorted(data):
            zi = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            z.writestr(zi, data[name])
    os.replace(tmp, out_zip)
    return out_zip


# --- reading and checking ----------------------------------------------------------------------------------------------------
class Bundle:
    def __init__(self, path: Path):
        self.path = Path(path)
        try:
            self.zip = zipfile.ZipFile(self.path)
        except (OSError, zipfile.BadZipFile) as e:
            raise BundleError(f"{self.path.name} is not a readable zip file ({e})") from None
        names = self.zip.namelist()
        bad = [n for n in names if not _safe_name(n)]
        if bad:
            raise BundleError(f"refusing a bundle with unsafe file names: {bad[:3]}")
        for need in ("metadata.json", "ro-crate-metadata.json"):
            if need not in names:
                raise BundleError(f"not a Kick the Fly bundle: {need} is missing")
        for n in ("metadata.json", "ro-crate-metadata.json", "protocol.yaml"):
            if n in names and self.zip.getinfo(n).file_size > MAX_META_BYTES:
                raise BundleError(f"{n} is implausibly large")
        try:
            self.meta = json.loads(self.zip.read("metadata.json"))
            self.crate = json.loads(self.zip.read("ro-crate-metadata.json"))
        except ValueError as e:
            raise BundleError(f"the bundle's metadata is not readable ({e})") from None
        if not isinstance(self.meta, dict) or not isinstance(self.crate, dict):
            raise BundleError("not a Kick the Fly bundle (its metadata is not a JSON object)")
        if self.meta.get("format") != FORMAT:
            raise BundleError("not a Kick the Fly bundle (metadata.json has another format)")
        v = self.meta.get("format_version")
        if not isinstance(v, int) or v > FORMAT_VERSION:
            raise BundleError(f"the bundle is a newer format ({v}; this version reads up to {FORMAT_VERSION}): update "
                              "Kick the Fly")

    def names(self) -> list[str]:
        return self.zip.namelist()

    def read(self, name: str) -> bytes:
        return self.zip.read(name)

    def protocol(self) -> dict:
        import yaml

        try:
            return yaml.safe_load(self.zip.read("protocol.yaml").decode("utf-8"))
        except (KeyError, ValueError, yaml.YAMLError) as e:
            raise BundleError(f"the bundle's protocol.yaml can't be read ({e})") from None

    def _hash_member(self, n: str, limit: int) -> str | None:
        """SHA-256 of one member, streamed; None if it expands past `limit` bytes (a zip bomb, or a lying header)."""
        h, total = hashlib.sha256(), 0
        with self.zip.open(n) as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                total += len(chunk)
                if total > limit:
                    return None
                h.update(chunk)
        return h.hexdigest()

    def verify(self) -> list[str]:
        """Problems found checking every file against the crate's SHA-256 (empty list = intact). Streams each file and
        stops at the size the crate declares for it, so memory stays flat whatever a bundle claims (review, Day 1)."""
        problems = []
        graph = self.crate.get("@graph", []) if isinstance(self.crate, dict) else []
        listed = {e["@id"]: e for e in graph if isinstance(e, dict) and e.get("@type") == "File" and "@id" in e}
        for n in self.names():
            if n == "ro-crate-metadata.json" or n.endswith("/"):
                continue
            e = listed.get(n)
            if e is None:
                problems.append(f"{n} is in the zip but not in the crate")
                continue
            size = e.get("contentSize")
            if not isinstance(size, int) or size < 0 or size > MAX_FILE_BYTES or self.zip.getinfo(n).file_size != size:
                problems.append(f"{n} doesn't have the size the crate records for it")
                continue
            got = self._hash_member(n, size)
            if got is None or got != e.get("sha256"):
                problems.append(f"{n} has changed since the bundle was made (its SHA-256 doesn't match)")
        for n in listed:
            if n not in self.names():
                problems.append(f"{n} is listed in the crate but missing from the zip")
        return problems

    def close(self) -> None:
        self.zip.close()


def inspect(path: Path) -> Bundle:
    return Bundle(path)


# --- rerunning ---------------------------------------------------------------------------------------------------------------------
def _ci_ok(orig: dict, new_mean: float) -> tuple[bool, str]:
    m = orig["mean"]
    if any(isinstance(orig.get(k), float) and math.isnan(orig[k]) for k in ("lo", "hi")) or "lo" not in orig:
        tol = STAT_SINGLE_REL * abs(m) + STAT_SINGLE_ABS_HZ
    else:
        tol = (orig["hi"] - orig["lo"]) / 2 + STAT_SLACK_REL * abs(m)
    return abs(new_mean - m) <= tol, f"{m:.3f} -> {new_mean:.3f} (allowed +-{tol:.3f})"


def _walk_means(a, b, path=""):
    """Yield (path, original {mean,...}, rerun mean) for every mean-with-interval leaf present in both."""
    if isinstance(a, dict) and "mean" in a and "n" in a:
        if isinstance(b, dict) and isinstance(b.get("mean"), (int, float)) and not (math.isnan(a["mean"]) and math.isnan(b["mean"])):
            yield path, a, float(b["mean"])
    elif isinstance(a, dict) and isinstance(b, dict):
        for k in a:
            if k in b:
                yield from _walk_means(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            yield from _walk_means(x, y, f"{path}[{i}]")


def _strip_volatile(summary: dict) -> dict:
    return {k: v for k, v in summary.items() if k not in ("seconds",)}


def rerun(bundle_path: Path, out: Path, backend: str | None = None, dtype: str | None = None, workers: int | None = None,
          progress=None) -> dict:
    """Rerun a bundle's protocol into `out` and judge whether the results match (see the module docstring). Returns the
    report dict; raises BundleError for a bundle that can't be used. Writes out/rerun_report.json."""
    from kickthefly.core import replay
    from kickthefly.lab import protocol as protocol_mod

    b = Bundle(bundle_path)
    try:
        problems = b.verify()
        if problems:
            raise BundleError("the bundle is damaged: " + "; ".join(problems[:3]))
        meta = b.meta
        if not meta.get("rerunnable"):
            raise BundleError(f"this bundle can't be rerun: {meta.get('not_rerunnable_reason') or 'no reason given'}")
        mine = replay.pack_sha256()
        theirs = meta.get("brain_pack_sha256")
        if mine is None:
            raise BundleError("no brain pack is built here, so the experiment can't be rerun")
        if not theirs:
            raise BundleError("the bundle doesn't record which brain pack it used, so it can't be matched to this one")
        if theirs != mine:
            raise BundleError(f"made with a different brain pack (SHA-256 {theirs[:12]}... vs this one {mine[:12]}...)")
        try:
            p = protocol_mod.check(b.protocol(), "the bundle's protocol")
        except protocol_mod.ProtocolError as e:
            raise BundleError(f"the bundle's protocol isn't valid here: {e}") from None
        if "results/summary.json" not in b.names():
            raise BundleError("the bundle has no results/summary.json, so there is nothing to compare a rerun with")
        if b.zip.getinfo("results/summary.json").file_size > MAX_META_BYTES:
            raise BundleError("results/summary.json is implausibly large")
        orig_summary = json.loads(b.read("results/summary.json"))
        orig_backend = meta.get("backend")
        want_dtype = dtype or meta.get("dtype") or "float32"
        want_backend = backend or orig_backend or "cpu"
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        if "assay" not in p:
            p = dict(p, nwb=False)                                  # the NWB export is raw data, not needed to compare
        with _env(KICK_THE_FLY_SIM_BACKEND=want_backend, KICK_THE_FLY_SIM_DTYPE=want_dtype):
            folder = protocol_mod.run(p, out / "rerun", workers=workers, progress=progress)
            ran_backend = probe_backend()
        new_summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        exact = (orig_backend in BIT_EXACT_BACKENDS and ran_backend in BIT_EXACT_BACKENDS
                 and want_dtype == meta.get("dtype"))
        report = {"bundle": str(bundle_path), "protocol": p.get("name"), "app_version_bundle": meta.get("app_version"),
                  "app_version_now": __version__, "recorded_backend": orig_backend, "rerun_backend": ran_backend,
                  "recorded_dtype": meta.get("dtype"), "rerun_dtype": want_dtype,
                  "mode": "bit-exact" if exact else "statistical", "results_folder": str(folder), "details": []}
        if exact:
            ok = _compare_exact(meta, p, folder, orig_summary, new_summary, report)
        else:
            ok = _compare_statistical(p, orig_summary, new_summary, report)
        report["match"] = bool(ok)
        report["verdict"] = (("MATCH (bit-exact)" if exact else "MATCH (statistically consistent)") if ok
                             else ("MISMATCH (spikes differ on backends that must be bit-identical)" if exact
                                   else "DIFFERS (outside the pre-declared statistical tolerance)"))
        (out / "rerun_report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
        return report
    finally:
        b.close()


def _compare_exact(meta, p, folder: Path, orig_summary, new_summary, report) -> bool:
    ok = True
    if "assay" in p:
        for key in ("treated", "control", "comparison", "surgery"):
            same = json.dumps(orig_summary.get(key), sort_keys=True, default=str) == json.dumps(new_summary.get(key), sort_keys=True, default=str)
            report["details"].append({"what": f"assay {key}", "identical": same})
            ok &= same
        return ok
    new = _run_manifest(folder, p)
    for key, rec in sorted(meta["flies"].items()):
        got = (new.get(key) or {}).get("spike_sha256")
        same = got is not None and got == rec["spike_sha256"]
        report["details"].append({"what": f"{key} spikes", "recorded": rec["spike_sha256"][:16], "rerun": (got or "missing")[:16],
                                  "identical": same})
        ok &= same
    extra = set(new) - set(meta["flies"])
    if extra:
        report["details"].append({"what": "flies in the rerun but not the bundle", "flies": sorted(extra), "identical": False})
        ok = False
    return ok


def _compare_statistical(p, orig_summary, new_summary, report) -> bool:
    ok = True
    a, bnew = _strip_volatile(orig_summary), _strip_volatile(new_summary)
    if "assay" in p:
        pairs = list(_walk_means(a.get("treated"), bnew.get("treated"), "treated")) + \
            list(_walk_means(a.get("control"), bnew.get("control"), "control"))
    else:
        pairs = [(f"{tag}/{g}", c, float(bnew.get("mean_rate_hz", {}).get(tag, {}).get(g, {}).get("mean", float("nan"))))
                 for tag, groups in a.get("mean_rate_hz", {}).items() for g, c in groups.items()]
    if not pairs:
        report["details"].append({"what": "nothing to compare", "ok": False})
        return False
    for path, orig, new_mean in pairs:
        if math.isnan(new_mean):
            good, txt = False, "missing in the rerun"
        else:
            good, txt = _ci_ok(orig, new_mean)
        report["details"].append({"what": path, "ok": good, "detail": txt})
        ok &= good
    return ok


def format_report(r: dict) -> str:
    lines = [f"rerun of {r['protocol']}: {r['verdict']}",
             f"  recorded on {r['recorded_backend']} ({r['recorded_dtype']}), rerun on {r['rerun_backend']} "
             f"({r['rerun_dtype']}); judged {r['mode']}"]
    bad = [d for d in r["details"] if not d.get("identical", d.get("ok", True))]
    n = len(r["details"])
    lines.append(f"  {n - len(bad)} of {n} checks passed")
    for d in bad[:8]:
        lines.append("  FAIL " + d["what"] + (f": {d['detail']}" if "detail" in d else ""))
    return "\n".join(lines)


def main(args) -> int:
    """--headless --rerun-bundle ZIP --out DIR"""
    import sys

    try:
        if not args.out:
            print("error: --rerun-bundle needs --out DIR for the rerun's files", file=sys.stderr)
            return 2
        t0 = time.time()
        rep = rerun(Path(args.rerun_bundle), Path(args.out), backend=getattr(args, "backend", None) or getattr(args, "sim_backend", None),
                    dtype=getattr(args, "dtype", None), workers=getattr(args, "workers", None),
                    progress=lambda d, n: print(f"  {d}/{n} ({time.time() - t0:.0f}s)", flush=True))
    except BundleError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(format_report(rep))
    print(f"report: {Path(args.out) / 'rerun_report.json'}")
    return 0 if rep["match"] else 1
