"""MaleCNS v1.0 connectome loader.

inspect: fetch the three flat-connectome Arrow IPC tables into data/ with
integrity checks, read them with pyarrow.feather.read_table, and print schema,
row count, and sample rows.

build: filter neurons and edges, sign edges by presynaptic neurotransmitter,
and pickle a CSR adjacency plus body_id -> row index to data/graph.pkl.

    python -m kickthefly.sim.connectome.loader inspect
    python -m kickthefly.sim.connectome.loader build
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import pickle
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather

from kickthefly import DATA_DIR, SOURCE_ROOT

PROJECT_ROOT = SOURCE_ROOT
LOCKFILE = DATA_DIR / "SHA256SUMS"
GRAPH_PATH = DATA_DIR / "graph.pkl"

# The objects live under flat-connectome/; the same paths without that
# segment return 404.
_BASE_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
_CHUNK = 1 << 20


class IntegrityError(RuntimeError):
    pass


@dataclass(frozen=True)
class Dataset:
    key: str
    filename: str
    size: int
    # md5Hash from the GCS object metadata (storage/v1 JSON API), base64 as GCS reports it.
    md5_b64: str
    # No sha256 is published for these files. Pin one here if an authoritative
    # value appears; otherwise the first md5-verified copy is pinned in data/SHA256SUMS.
    sha256: str | None = None

    @property
    def url(self) -> str:
        return _BASE_URL + self.filename

    @property
    def path(self) -> Path:
        return DATA_DIR / self.filename


DATASETS: dict[str, Dataset] = {
    d.key: d
    for d in (
        Dataset("annotations", "body-annotations-male-cns-v1.0-minconf-0.5.feather",
                14_483_314, "UKdxh3DFciDxYLpPQxq4ng=="),
        Dataset("neurotransmitters", "body-neurotransmitters-male-cns-v1.0.feather",
                43_282_834, "PYQrEv5cSe763lKNfdJKHw=="),
        Dataset("weights", "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
                1_051_241_946, "8w6dzKJc/QIb8eez2XVZng=="),
    )
}


# --- integrity -------------------------------------------------------------

def _read_lock() -> dict[str, str]:
    if not LOCKFILE.exists():
        return {}
    entries = {}
    for line in LOCKFILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            entries[name.strip()] = digest
    return entries


def _write_lock(entries: dict[str, str]) -> None:
    LOCKFILE.write_text("".join(f"{d}  {n}\n" for n, d in sorted(entries.items())), encoding="utf-8")


class _Digests:
    def __init__(self) -> None:
        self.size = 0
        self._md5 = hashlib.md5()
        self._sha = hashlib.sha256()

    def update(self, chunk: bytes) -> None:
        self.size += len(chunk)
        self._md5.update(chunk)
        self._sha.update(chunk)

    @property
    def md5_b64(self) -> str:
        return base64.b64encode(self._md5.digest()).decode()

    @property
    def sha256(self) -> str:
        return self._sha.hexdigest()


def _verify(ds: Dataset, got: _Digests, lock: dict[str, str]) -> None:
    problems = []
    if got.size != ds.size:
        problems.append(f"size {got.size:,} != expected {ds.size:,}")
    if got.md5_b64 != ds.md5_b64:
        problems.append(f"md5 {got.md5_b64} != GCS md5 {ds.md5_b64}")
    expected_sha = ds.sha256 or lock.get(ds.filename)
    if expected_sha and got.sha256 != expected_sha:
        problems.append(f"sha256 {got.sha256} != pinned {expected_sha}")
    if problems:
        raise IntegrityError(f"{ds.filename}: " + "; ".join(problems))


def _hash_file(path: Path) -> _Digests:
    d = _Digests()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            d.update(chunk)
    return d


def _download(ds: Dataset, lock: dict[str, str]) -> _Digests:
    part = ds.path.with_name(ds.path.name + ".part")
    d = _Digests()
    print(f"[download] {ds.url}\n           -> {ds.path} ({ds.size / 1e6:,.1f} MB)", flush=True)
    try:
        with urllib.request.urlopen(ds.url, timeout=60) as resp, open(part, "wb") as fh:
            next_report = 0.1
            while chunk := resp.read(_CHUNK):
                fh.write(chunk)
                d.update(chunk)
                if d.size >= next_report * ds.size:
                    print(f"           {d.size / ds.size:4.0%}", flush=True)
                    next_report += 0.1
        _verify(ds, d, lock)
    except urllib.error.HTTPError as e:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"download failed: HTTP {e.code} for {ds.url}") from e
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    part.replace(ds.path)
    return d


def ensure(ds: Dataset) -> Path:
    """Return a verified local path for ds, downloading it if absent."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    lock = _read_lock()
    if ds.path.exists():
        print(f"[verify] {ds.path.name} ...", flush=True)
        d = _hash_file(ds.path)
        try:
            _verify(ds, d, lock)
        except IntegrityError as e:
            raise IntegrityError(f"{e}\nExisting file failed verification; delete {ds.path} to re-download.") from None
    else:
        d = _download(ds, lock)
    if lock.get(ds.filename) != d.sha256:
        lock[ds.filename] = d.sha256
        _write_lock(lock)
        print(f"[verify] pinned sha256 {d.sha256} for {ds.filename} (size and GCS md5 matched)")
    return ds.path


# --- step 1: inspection ----------------------------------------------------

def load_table(key: str) -> pa.Table:
    return feather.read_table(ensure(DATASETS[key]), memory_map=True)


def print_summary(key: str, table: pa.Table, n_sample: int = 5) -> None:
    import pandas as pd

    print(f"\n=== {key}: {DATASETS[key].filename}")
    print(f"rows: {table.num_rows:,}    columns: {table.num_columns}")
    print("schema:")
    print(table.schema.to_string(show_schema_metadata=False))
    # Transposed so wide tables read as one column per sample row.
    print(f"\nfirst {n_sample} rows (transposed):")
    with pd.option_context("display.max_rows", None, "display.max_columns", None,
                           "display.width", 250, "display.max_colwidth", 32):
        print(table.slice(0, n_sample).to_pandas().T)


# --- build: filtered, signed graph -----------------------------------------

EXPECTED_NEURONS = 165_000
EXPECTED_EDGES = 10_200_000
TOLERANCE = 0.15
# weight >= 3 reproduces the ~10.2M edge expectation (+3.1%); >= 5 gave 6.24M (-38.8%).
MIN_WEIGHT = 3
# Histamine is -1: every R1-R8 photoreceptor is histaminergic (HisCl1-gated, hyperpolarizing
# at the photoreceptor -> lamina synapse); at 0 no visual input would leave the retina.
# Everything else (dopamine, serotonin, octopamine, unclear): 0, contact kept.
NT_SIGN = {"acetylcholine": 1, "gaba": -1, "glutamate": -1, "histamine": -1}
# consensus_nt is never null in v1.0; unresolved neurons carry the string 'unclear'.
NT_UNRESOLVED = {None, "unclear"}


@dataclass
class Graph:
    """Signed connectome.

    adjacency[i, j] = sign(nt of presynaptic i) * synapse count, i -> j, float32.
    Zero-sign contacts are stored as explicit zeros; `weights` has the same
    sparsity structure with the unsigned counts, so contacts survive any
    downstream eliminate_zeros().
    """
    adjacency: "sp.csr_array"
    weights: "sp.csr_array"
    body_ids: np.ndarray          # row -> bodyId (sorted int64)
    index: dict[int, int]         # bodyId -> row
    nt: np.ndarray                # row -> neurotransmitter string used for the sign
    nt_source: np.ndarray         # row -> "consensus_nt" | "predicted_nt" | "missing"
    sign: np.ndarray              # row -> int8
    type: np.ndarray              # row -> annotation `type` (object, may be None)
    superclass: np.ndarray        # row -> annotation `superclass`
    instance: np.ndarray          # row -> annotation `instance`
    meta: dict

    @property
    def n(self) -> int:
        return len(self.body_ids)

    def rows(self, body_ids) -> np.ndarray:
        return np.searchsorted(self.body_ids, np.asarray(body_ids, dtype=np.int64))


def _delta(label: str, actual: int, expected: int) -> bool:
    rel = (actual - expected) / expected
    ok = abs(rel) <= TOLERANCE
    print(f"  {label}: actual {actual:,} vs expected ~{expected:,} -> {rel:+.1%} "
          f"({'within' if ok else 'OUTSIDE'} {TOLERANCE:.0%})")
    return ok


def _n_touched(pre: pa.ChunkedArray, post: pa.ChunkedArray) -> int:
    return len(np.union1d(pc.unique(pre).to_numpy(), pc.unique(post).to_numpy()))


def build_graph(accept_out_of_range: bool = False) -> Graph | None:
    import scipy.sparse as sp

    t0 = time.perf_counter()
    ann = load_table("annotations")
    nt_tab = load_table("neurotransmitters")
    w_tab = load_table("weights")
    print(f"[build] loaded tables in {time.perf_counter() - t0:.1f}s")

    # 1. neurons: superclass non-null and != 'glia'
    keep = pc.and_kleene(pc.is_valid(ann["superclass"]), pc.not_equal(ann["superclass"], "glia"))
    ann = ann.filter(keep).sort_by("bodyId")
    body_ids = ann["bodyId"].to_numpy()
    if len(np.unique(body_ids)) != len(body_ids):
        raise RuntimeError("duplicate bodyId in filtered annotations")
    n = len(body_ids)
    glia_status = pc.sum(pc.equal(ann["status"], "Glia")).as_py() or 0
    print(f"[filter 1] superclass non-null and != 'glia': neurons {n:,}   edges {w_tab.num_rows:,} (unfiltered)"
          f"   [status=='Glia' among survivors: {glia_status:,}]")

    # 2. edges: both endpoints in the neuron set
    t = time.perf_counter()
    value_set = pa.array(body_ids)
    both = pc.and_(pc.is_in(w_tab["body_pre"], value_set=value_set),
                   pc.is_in(w_tab["body_post"], value_set=value_set))
    w_tab = w_tab.filter(both)
    print(f"[filter 2] both endpoints in set:            neurons {n:,}   edges {w_tab.num_rows:,}"
          f"   [neurons with >=1 edge: {_n_touched(w_tab['body_pre'], w_tab['body_post']):,}]"
          f"  ({time.perf_counter() - t:.1f}s)")

    # 3. edges: weight >= MIN_WEIGHT
    w_tab = w_tab.filter(pc.greater_equal(w_tab["weight"], MIN_WEIGHT))
    n_edges = w_tab.num_rows
    print(f"[filter 3] weight >= {MIN_WEIGHT}:                        neurons {n:,}   edges {n_edges:,}"
          f"   [neurons with >=1 edge: {_n_touched(w_tab['body_pre'], w_tab['body_post']):,}]")

    print("[check] magnitudes")
    ok = _delta("neurons", n, EXPECTED_NEURONS) & _delta("edges", n_edges, EXPECTED_EDGES)
    if not ok and not accept_out_of_range:
        print("[check] counts outside tolerance; not writing graph.pkl. Re-run with --accept-out-of-range to override.")
        return None

    # Neurotransmitter per neuron: consensus_nt, falling back to predicted_nt when null or 'unclear'.
    nt_tab = nt_tab.filter(pc.is_in(nt_tab["body"], value_set=value_set)).sort_by("body")
    nt_body = nt_tab["body"].to_numpy()
    pos = np.searchsorted(nt_body, body_ids)
    has_row = (pos < len(nt_body)) & (nt_body[np.minimum(pos, len(nt_body) - 1)] == body_ids)
    consensus = np.full(n, None, dtype=object)
    predicted = np.full(n, None, dtype=object)
    consensus[has_row] = np.asarray(nt_tab["consensus_nt"].to_pylist(), dtype=object)[pos[has_row]]
    predicted[has_row] = np.asarray(nt_tab["predicted_nt"].to_pylist(), dtype=object)[pos[has_row]]
    c_unres = np.array([v in NT_UNRESOLVED for v in consensus])
    p_unres = np.array([v in NT_UNRESOLVED for v in predicted])
    use_pred = c_unres & ~p_unres
    nt = np.where(use_pred, predicted, consensus)
    nt_source = np.where(use_pred, "predicted_nt", np.where(~has_row, "missing", "consensus_nt"))
    sign = np.array([NT_SIGN.get(v, 0) for v in nt], dtype=np.int8)

    vals, counts = np.unique(nt.astype(str), return_counts=True)
    print(f"[nt] neurons without an nt row: {int((~has_row).sum()):,}   "
          f"consensus unresolved: {int(c_unres.sum()):,}   predicted_nt fallback used: {int(use_pred.sum()):,}")
    print("[nt] per-neuron nt: " + ", ".join(f"{v}={c:,}" for v, c in sorted(zip(vals, counts), key=lambda x: -x[1])))
    print(f"[nt] per-neuron sign: +1={int((sign == 1).sum()):,}  -1={int((sign == -1).sum()):,}  0={int((sign == 0).sum()):,}")

    # CSR built directly from sorted (pre, post) so zero-sign contacts stay as explicit entries.
    pre = np.searchsorted(body_ids, w_tab["body_pre"].to_numpy())
    post = np.searchsorted(body_ids, w_tab["body_post"].to_numpy())
    cnt = w_tab["weight"].to_numpy()
    del w_tab
    order = np.lexsort((post, pre))
    pre, post, cnt = pre[order], post[order], cnt[order]
    dup = (np.diff(pre) == 0) & (np.diff(post) == 0)
    if dup.any():
        raise RuntimeError(f"{int(dup.sum()):,} duplicate (pre, post) pairs in weights table")
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(pre, minlength=n), out=indptr[1:])
    indices = post.astype(np.int32)
    weights = sp.csr_array((cnt.astype(np.int32), indices, indptr), shape=(n, n))
    signed = sp.csr_array(((sign[pre] * cnt).astype(np.float32), indices, indptr), shape=(n, n))
    for m in (weights, signed):
        m.has_sorted_indices = True
    edge_sign = sign[pre]
    print(f"[edges] +1={int((edge_sign == 1).sum()):,}  -1={int((edge_sign == -1).sum()):,}  "
          f"0 (kept as explicit zeros)={int((edge_sign == 0).sum()):,}   adjacency.nnz={signed.nnz:,}")

    def col(name):
        return np.asarray(ann[name].to_pylist(), dtype=object)

    graph = Graph(
        adjacency=signed, weights=weights, body_ids=body_ids,
        index={int(b): i for i, b in enumerate(body_ids)},
        nt=nt, nt_source=nt_source, sign=sign,
        type=col("type"), superclass=col("superclass"), instance=col("instance"),
        meta={
            "dataset": "MaleCNS v1.0 minconf 0.5", "min_weight": MIN_WEIGHT, "nt_sign": NT_SIGN,
            "orientation": "adjacency[pre, post]", "sha256": _read_lock(),
            "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    with open(GRAPH_PATH, "wb") as fh:
        pickle.dump(graph, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[build] wrote {GRAPH_PATH} ({GRAPH_PATH.stat().st_size / 1e6:,.0f} MB) in {time.perf_counter() - t0:.1f}s total")
    return graph


# graph.pkl files written before 2.7 recorded this module as "connectome.loader" (it lived in the repo root).
# They are 260 MB and take a 1.1 GB download to rebuild, so they are still read as they are.
_LEGACY_MODULES = {"connectome.loader": __name__, "connectome.sim": __name__.replace("loader", "sim")}


class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return super().find_class(_LEGACY_MODULES.get(module, module), name)


def load_graph() -> Graph:
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(f"{GRAPH_PATH} not found. Build it once with: "
                                "python -m kickthefly.sim.connectome.loader build")
    with open(GRAPH_PATH, "rb") as fh:
        return _CompatUnpickler(fh).load()


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="python -m kickthefly.sim.connectome.loader")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_inspect = sub.add_parser("inspect", help="download/verify, then print schema, row count, and sample rows")
    p_inspect.add_argument("tables", nargs="*", metavar="TABLE", help=f"subset of {', '.join(DATASETS)} (default: all)")
    p_build = sub.add_parser("build", help="filter, sign, and pickle the graph to data/graph.pkl")
    p_build.add_argument("--accept-out-of-range", action="store_true",
                         help=f"write graph.pkl even if counts differ from expectations by more than {TOLERANCE:.0%}".replace("%", "%%"))
    args = parser.parse_args(argv)

    if args.cmd == "inspect":
        keys = args.tables or list(DATASETS)
        unknown = sorted(set(keys) - set(DATASETS))
        if unknown:
            parser.error(f"unknown table(s): {', '.join(unknown)}")
        for key in keys:
            print_summary(key, load_table(key))
    elif args.cmd == "build":
        return 0 if build_graph(args.accept_out_of_range) is not None else 2
    return 0


if __name__ == "__main__":
    # Re-import so pickled classes are recorded as kickthefly.sim.connectome.loader.Graph, not __main__.Graph.
    from kickthefly.sim.connectome.loader import main as _main
    raise SystemExit(_main())
