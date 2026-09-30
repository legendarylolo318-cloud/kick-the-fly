"""Neurodex (3.0): a collectible encyclopedia of cell types, filled in by playing.

No pygame here: the game, the Python API, the Neuron of the Day card and the tests all read this module.

What is what (the canonical map is the docstring of kickthefly/game/kick_the_fly.py):
  CONNECTOME   every number in an entry: the type's neuron count, superclass and regions as the brain pack annotates
               them, the transmitter the dataset predicts (with the dataset's own confidence), and the strongest
               partner types by synapse count. Nothing is measured by the game; nothing is estimated.
  GAME RULE    "discovered". A type is discovered the first time the mean firing of its neurons is at least
               DISCOVER_MIN_HZ and at least DISCOVER_FACTOR x its own calm rate (never below DISCOVER_CALM_FLOOR_HZ),
               for DISCOVER_SUSTAIN consecutive checks (50 ms apart), after the brain has settled. Those four numbers
               are the game's choice; they were fixed before anything was played and are not tuned. The collection, its
               progress bars and the "discovered by stimulation" tag are game rules too. Discovery only runs in the
               windowed game: never in --validate, assays, protocols or tests.
  LITERATURE   the one-line fact and citation on a curated type (kickthefly/data/neurodex_facts.yaml). Hand-written,
               checked against the cited paper, and only for the types in that table.

Two honest limits. The rule reads the type's MEAN rate, so a large, sparsely coding type (the ~4,000 Kenyon cells, where
an odor activates a few percent) barely moves its mean and is hard to discover by smell alone; stimulating the type in
brain surgery or the Lab laser discovers it, and says so. And the pack's regions are the coarse ones the game derives
from the dataset's class and soma-neuromere annotations (Antennal Lobe, Mushroom Body, ...), not neuropil ROIs.

Saved per player next to the training memory (memory_dir()/neurodex.json), one list per brain (adult, larva).
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CONNECTOME, GAME_RULE, LITERATURE = "CONNECTOME", "GAME RULE", "LITERATURE"
FILE_NAME = "neurodex.json"
FORMAT_VERSION = 1

# GAME RULE: the discovery rule (see the module docstring). Fixed, not tuned.
DISCOVER_MIN_HZ = 6.0            # mean firing of the type's neurons, spikes/s
DISCOVER_FACTOR = 3.0            # ... and this many times its own calm rate
DISCOVER_CALM_FLOOR_HZ = 2.0     # the calm rate is never taken below this (as Brain.level does)
DISCOVER_SUSTAIN = 3             # consecutive checks
SETTLE_CHECKS = 100              # checks (5 s) of learning calm before any discovery
CHECK_STEPS = 10                 # sim steps (5 ms) between checks
CALM_K_QUIET, CALM_K_BUSY = 0.02, 0.0005     # calm-rate tracking, as core/memory.py does

TOP_PARTNERS = 6
UNNAMED = "(no type)"


# --- the type table (CONNECTOME) ------------------------------------------------------------------------------------------
@dataclass
class TypeTable:
    brain: str
    names: np.ndarray                    # (T,) type names, sorted
    count: np.ndarray                    # (T,) neurons per type
    superclass: list[str]
    region: list[str]                    # primary region (most common; "unassigned" only if nothing else)
    regions: list[list[tuple[str, int]]]  # per type: [(region, neurons)], most common first
    nt: list[str]                        # most common predicted transmitter ("" = none in the dataset)
    nt_conf: np.ndarray                  # mean confidence of that call (NaN = none)
    nt_truth: np.ndarray                 # share of the type's neurons whose transmitter was measured (0..1)
    type_id: np.ndarray                  # (n,) type index per neuron, -1 for a neuron without a type
    partners: object = None              # scipy csr (T+1, T+1) synapse counts [post, pre]; last index = no type
    body_id: np.ndarray | None = None    # (n,) for skeleton lookup
    pack_sha256: str | None = None
    _rows: dict = field(default_factory=dict, repr=False)
    _csc: object = field(default=None, repr=False)
    _index: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._index = {str(n): i for i, n in enumerate(self.names)}
        order = np.argsort(self.type_id, kind="stable")
        tid = self.type_id[order]
        self._order = order
        self._start = np.searchsorted(tid, np.arange(len(self.names) + 1))   # start[i]:start[i+1] rows of type i

    def __len__(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int | None:
        return self._index.get(name)

    def rows(self, name: str) -> np.ndarray:
        i = self._index[name]
        return self._order[self._start[i]:self._start[i + 1]]

    def superclasses(self) -> list[str]:
        return sorted(set(self.superclass))

    def region_names(self) -> list[str]:
        return sorted(set(self.region), key=lambda r: (r == "unassigned", r))

    def top_partners(self, name: str, direction: str, k: int = TOP_PARTNERS) -> list[tuple[str, int, float]]:
        """[(partner type, synapses, share)] strongest first. direction "in": partners that synapse onto this type;
        "out": partners this type synapses onto. share is of this type's total input (in) or output (out) synapses,
        counting the ones on neurons without a type. Synapses are the pack's signed connection counts."""
        i = self._index.get(name)
        if i is None or self.partners is None:
            return []
        if direction == "in":
            C = self.partners.tocsr()
            lo, hi = C.indptr[i], C.indptr[i + 1]
            idx, val = C.indices[lo:hi], C.data[lo:hi]
        else:
            if self._csc is None:
                self._csc = self.partners.tocsc()
            C = self._csc
            lo, hi = C.indptr[i], C.indptr[i + 1]
            idx, val = C.indices[lo:hi], C.data[lo:hi]
        if len(val) == 0:
            return []
        total = float(val.sum())
        top = np.argsort(-val, kind="stable")[:k]
        names = list(self.names) + [UNNAMED]
        return [(names[int(idx[j])], int(val[j]), float(val[j]) / total) for j in top]


def _mode(values) -> str:
    c = Counter(values)
    return c.most_common(1)[0][0] if c else ""


def _partner_matrix(tid: np.ndarray, T: int, indptr: np.ndarray, indices: np.ndarray, data: np.ndarray, chunk: int = 1 << 21):
    """(T+1, T+1) synapse counts [post type, pre type] from the signed neuron CSR, summed a few million connections at a
    time so the real pack (tens of millions of them) never needs several full-size index arrays at once."""
    import scipy.sparse as sp

    n = len(indptr) - 1
    type_or_last = np.where(tid >= 0, tid, T).astype(np.int32)
    C = sp.csr_array((T + 1, T + 1), dtype=np.int64)
    row = 0
    while row < n:
        end = int(np.searchsorted(indptr, indptr[row] + chunk, side="left"))
        end = min(max(end, row + 1), n)
        lo, hi = int(indptr[row]), int(indptr[end])
        post = np.repeat(np.arange(row, end, dtype=np.int32), np.diff(indptr[row:end + 1]))
        pre = indices[lo:hi]
        w = np.abs(data[lo:hi].astype(np.int64))
        C = C + sp.coo_array((w, (type_or_last[post], type_or_last[pre])), shape=(T + 1, T + 1)).tocsr()
        row = end
    C.sum_duplicates()
    return C


def build_table(arrays: dict, brain: str = "adult", partners: bool = True) -> TypeTable:
    """A TypeTable from a brain pack's arrays (type, superclass, region, nt, nt_conf, nt_source, body_id and the
    signed synapse CSR indptr / indices / data). Missing optional arrays (a larva pack has no transmitters) leave
    that part of the entry empty instead of guessing."""
    import scipy.sparse as sp

    types = np.asarray(arrays["type"]).astype(str)
    n = len(types)
    named = types != ""
    names, inv = np.unique(types[named], return_inverse=True)
    tid = np.full(n, -1, np.int32)
    tid[named] = inv
    T = len(names)
    count = np.bincount(tid[named], minlength=T)
    sc = np.asarray(arrays["superclass"]).astype(str) if "superclass" in arrays else np.full(n, "")
    reg = np.asarray(arrays["region"]).astype(str) if "region" in arrays else np.full(n, "unassigned")
    nt = np.asarray(arrays["nt"]).astype(str) if arrays.get("nt") is not None else None
    conf = np.asarray(arrays["nt_conf"], np.float32) if arrays.get("nt_conf") is not None else None
    src = np.asarray(arrays["nt_source"]).astype(str) if arrays.get("nt_source") is not None else None

    order = np.argsort(tid, kind="stable")
    start = np.searchsorted(tid[order], np.arange(T + 1))
    superclass, region, regions, nts = [], [], [], []
    nt_conf = np.full(T, np.nan, np.float32)
    nt_truth = np.zeros(T, np.float32)
    for i in range(T):
        rows = order[start[i]:start[i + 1]]
        superclass.append(_mode(sc[rows]))
        rc = Counter(reg[rows]).most_common()
        regions.append([(r, int(c)) for r, c in rc])
        real = [r for r, _ in rc if r != "unassigned"]
        region.append(real[0] if real else "unassigned")
        if nt is None:
            nts.append("")
            continue
        label = _mode([v for v in nt[rows] if v])
        nts.append(label)
        if label:
            mine = rows[nt[rows] == label]
            if conf is not None:
                ok = conf[mine][~np.isnan(conf[mine])]
                if len(ok):
                    nt_conf[i] = float(ok.mean())
            if src is not None:
                nt_truth[i] = float(np.mean(src[rows] == "ground_truth"))
    C = None
    if partners and "indptr" in arrays:
        C = _partner_matrix(tid, T, np.asarray(arrays["indptr"]), np.asarray(arrays["indices"]), np.asarray(arrays["data"]))
    bid = np.asarray(arrays["body_id"]) if "body_id" in arrays else None
    return TypeTable(brain, names, count, superclass, region, regions, nts, nt_conf, nt_truth, tid, C, bid)


_TABLES: dict[str, TypeTable] = {}
_TABLE_LOCK = threading.Lock()


def table_from_pack(path: Path, brain: str = "adult") -> TypeTable:
    """The type table of a brain pack file (stamped with the pack's SHA-256)."""
    from kickthefly.core import replay

    sha = replay.pack_sha256(path)
    z = np.load(path, allow_pickle=False)
    keep = ("type", "superclass", "region", "nt", "nt_conf", "nt_source", "body_id", "indptr", "indices", "data")
    arrays = {k: z[k] for k in keep if k in z.files}
    t = build_table(arrays, brain)
    t.pack_sha256 = sha
    return t


def table(brain: str = "adult") -> TypeTable | None:
    """The process-wide type table for `brain`, built on first use (a few seconds on the real pack). None if there is
    no brain pack. Safe to call from several threads."""
    from kickthefly.sim import brainpack

    with _TABLE_LOCK:
        if brain in _TABLES:
            return _TABLES[brain]
        p = brainpack.find(brain=brain)
        if p is None:
            return None
        _TABLES[brain] = table_from_pack(p, brain)
        return _TABLES[brain]


def reset_cache() -> None:
    _TABLES.clear()


# --- curated facts (LITERATURE) ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Fact:
    id: str
    text: str
    cite: str
    doi: str
    checked: str
    types: tuple = ()
    prefix: tuple = ()
    game_note: str = ""

    def matches(self, type_name: str) -> bool:
        return type_name in self.types or any(type_name.startswith(p) for p in self.prefix)


FACT_KEYS = {"id", "types", "prefix", "text", "cite", "doi", "checked", "game_note"}
MAX_FACT_CHARS = 260


class FactsError(ValueError):
    pass


def facts_path() -> Path:
    from kickthefly import data

    return data.path("neurodex_facts.yaml")


def load_facts(path: Path | None = None) -> list[Fact]:
    """The curated table. A malformed file raises FactsError naming the entry (tests/test_neurodex.py keeps the shipped
    file valid); the game treats an unreadable file as "no curated facts" rather than crashing."""
    import yaml

    p = Path(path) if path else facts_path()
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        raise FactsError(f"{p.name}: {e}") from None
    if not isinstance(data, dict) or not isinstance(data.get("facts"), list):
        raise FactsError(f"{p.name}: expected a mapping with a 'facts' list")
    out, seen = [], set()
    for i, f in enumerate(data["facts"]):
        where = f"{p.name} entry {i + 1}"
        if not isinstance(f, dict):
            raise FactsError(f"{where}: not a mapping")
        where = f"{p.name} entry '{f.get('id', i + 1)}'"
        if set(f) - FACT_KEYS:
            raise FactsError(f"{where}: unknown keys {sorted(set(f) - FACT_KEYS)}")
        for k in ("id", "text", "cite", "doi", "checked"):
            if not isinstance(f.get(k), str) or not f[k].strip():
                raise FactsError(f"{where}: '{k}' is required (a fact needs its citation and how it was checked)")
        if f["id"] in seen:
            raise FactsError(f"{where}: duplicate id")
        seen.add(f["id"])
        if len(f["text"]) > MAX_FACT_CHARS:
            raise FactsError(f"{where}: text is {len(f['text'])} characters; facts are short (max {MAX_FACT_CHARS})")
        types, prefix = f.get("types") or [], f.get("prefix") or []
        if not types and not prefix:
            raise FactsError(f"{where}: needs 'types' or 'prefix'")
        if not all(isinstance(x, str) and x for x in [*types, *prefix]):
            raise FactsError(f"{where}: types and prefixes must be non-empty strings")
        out.append(Fact(f["id"], f["text"].strip(), f["cite"].strip(), str(f["doi"]).strip(), f["checked"].strip(),
                        tuple(types), tuple(prefix), (f.get("game_note") or "").strip()))
    return out


_FACTS: list[Fact] | None = None


def facts() -> list[Fact]:
    global _FACTS
    if _FACTS is None:
        try:
            _FACTS = load_facts()
        except (FactsError, ImportError) as e:
            from kickthefly.core.crash import log

            log.warning("Neurodex facts unavailable: %s", e)
            _FACTS = []
    return _FACTS


def fact_for(type_name: str, brain: str = "adult") -> Fact | None:
    """The curated fact for a type, or None. The table is about the adult fly, so the larva dex has none."""
    if brain != "adult":
        return None
    for f in facts():
        if f.matches(type_name):
            return f
    return None


def curated_types(tab: TypeTable) -> list[str]:
    """The types of this brain that have a curated fact, sorted."""
    return [str(n) for n in tab.names if fact_for(str(n), tab.brain) is not None]


# --- progress (GAME RULE), saved like the training memory ---------------------------------------------------------------------
def progress_path() -> Path:
    from kickthefly.core import memory

    return memory.memory_dir() / FILE_NAME


class Progress:
    """Which types this player has discovered, per brain. Corrupt or unreadable files are kept as .bad and the dex starts
    empty (a dex is never worth a crash); a file from a newer version is left alone and not overwritten."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else progress_path()
        self.data: dict = {"version": FORMAT_VERSION, "brains": {}}
        self.warnings: list[str] = []
        self.read_only = False
        self.dirty = False
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or not isinstance(raw.get("brains", {}), dict):
                raise ValueError("unexpected layout")
        except (OSError, ValueError) as e:
            self.warnings.append(f"{self.path.name} could not be read ({e}); starting an empty Neurodex")
            try:
                self.path.replace(self.path.with_name(self.path.name + ".bad"))
                self.warnings.append(f"the unreadable file was kept as {self.path.name}.bad")
            except OSError:
                pass
            return
        v = raw.get("version", 1)
        if not isinstance(v, int) or v > FORMAT_VERSION:
            self.read_only = True
            self.warnings.append(f"{self.path.name} is from a newer version; this one won't change it")
            return
        for brain, rec in raw["brains"].items():
            types = rec.get("types") if isinstance(rec, dict) else None
            if isinstance(types, dict):
                self.data["brains"][str(brain)] = {"types": {
                    str(t): {"t": str(v.get("t", "")), "x": float(v.get("x", 0.0)), "how": str(v.get("how", "play"))}
                    for t, v in types.items() if isinstance(v, dict)}}

    def types(self, brain: str) -> dict:
        return self.data["brains"].setdefault(brain, {"types": {}})["types"]

    def discovered(self, brain: str, name: str) -> bool:
        return name in self.types(brain)

    def n_discovered(self, brain: str) -> int:
        return len(self.types(brain))

    def mark(self, brain: str, name: str, ratio: float = 0.0, how: str = "play", when: str | None = None) -> bool:
        """Record a discovery. False if it was already known (the first record stays)."""
        t = self.types(brain)
        if name in t:
            return False
        t[name] = {"t": when or time.strftime("%Y-%m-%dT%H:%M:%S"), "x": round(float(ratio), 2), "how": how}
        self.dirty = True
        return True

    def save(self) -> bool:
        if self.read_only or self.path is None:
            return False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(json.dumps(self.data, indent=1, sort_keys=True), encoding="utf-8")
            os.replace(tmp, self.path)
            self.dirty = False
            return True
        except OSError as e:
            from kickthefly.core.crash import log

            log.warning("could not save the Neurodex to %s: %s", self.path, e)
            return False

    def wipe(self, brain: str | None = None) -> None:
        if brain is None:
            self.data["brains"] = {}
        else:
            self.data["brains"].pop(brain, None)
        self.dirty = True


def region_progress(tab: TypeTable, prog: Progress) -> list[tuple[str, int, int]]:
    """[(region, discovered, total)] by each type's primary region, regions with types only, largest first, and a
    "(all)" row last."""
    known = prog.types(tab.brain)
    tot, got = Counter(), Counter()
    for name, reg in zip(tab.names, tab.region):
        tot[reg] += 1
        got[reg] += str(name) in known
    rows = sorted(tot, key=lambda r: (r == "unassigned", -tot[r], r))
    out = [(r, got[r], tot[r]) for r in rows]
    out.append(("(all)", sum(got.values()), sum(tot.values())))
    return out


# --- the discovery rule (GAME RULE) ---------------------------------------------------------------------------------------------
class Tracker:
    """Watches a running brain and records discoveries. One per game; feed it each brain's per-neuron rates.

    observe() takes the simulator's per-neuron rate EMA in spikes per step (sim.activity.rates()), the step length in
    seconds, whether the brain is calm (nothing touched it, no surgery, no drive: the game's own test for "calm"), and
    how it is being driven. It returns the names of types discovered just now. A separate calm baseline is kept per
    brain, because each fly's brain settles on its own."""

    def __init__(self, tab: TypeTable, prog: Progress):
        self.tab, self.prog = tab, prog
        self.valid = tab.type_id >= 0
        self.tid = np.where(self.valid, tab.type_id, 0)
        self.count = np.maximum(tab.count, 1).astype(np.float64)
        self._state: dict = {}
        self.known_mask = np.zeros(len(tab), bool)
        self.sync()

    def sync(self) -> None:
        """Re-read which types are known (after the progress file changed)."""
        known = self.prog.types(self.tab.brain)
        self.known_mask = np.fromiter((str(n) in known for n in self.tab.names), bool, len(self.tab))

    def type_rates_hz(self, rates: np.ndarray, dt: float) -> np.ndarray:
        r = np.where(self.valid, rates, 0.0)
        return np.bincount(self.tid, weights=r, minlength=len(self.tab)) / self.count / dt

    def observe(self, key, rates: np.ndarray, dt: float, calm: bool, driven: bool = False) -> list[str]:
        st = self._state.get(key)
        hz = self.type_rates_hz(np.asarray(rates, np.float64), dt)
        if st is None:
            st = self._state[key] = {"calm": hz.copy(), "n": 0, "run": np.zeros(len(hz), np.int16)}
        st["n"] += 1
        k = CALM_K_QUIET if (calm or st["n"] < SETTLE_CHECKS) else CALM_K_BUSY
        if st["n"] > 1:
            st["calm"] += (hz - st["calm"]) * k
        if st["n"] < SETTLE_CHECKS:
            return []
        ratio = hz / np.maximum(st["calm"], DISCOVER_CALM_FLOOR_HZ)
        hot = (hz >= DISCOVER_MIN_HZ) & (ratio >= DISCOVER_FACTOR) & ~self.known_mask
        run = st["run"]
        run[hot] += 1
        run[~hot] = 0
        found = np.flatnonzero(run >= DISCOVER_SUSTAIN)
        out = []
        for i in found:
            name = str(self.tab.names[i])
            if self.prog.mark(self.tab.brain, name, float(ratio[i]), "stimulated" if driven else "play"):
                out.append(name)
            self.known_mask[i] = True
            run[i] = 0
        return out


# --- one entry, for the panel, the API and the tests ----------------------------------------------------------------------------
def entry(tab: TypeTable, name: str, prog: Progress | None = None) -> dict | None:
    """Everything the Neurodex shows about a type. `discovered` False means the panel shows only a silhouette; this
    function still returns the data (the Python API and tests use it), the panel decides what to reveal."""
    i = tab.index(name)
    if i is None:
        return None
    rec = prog.types(tab.brain).get(name) if prog is not None else None
    nt = tab.nt[i]
    conf = None if np.isnan(tab.nt_conf[i]) else float(tab.nt_conf[i])
    f = fact_for(name, tab.brain)
    return {
        "type": name, "brain": tab.brain, "discovered": rec is not None, "how": (rec or {}).get("how"),
        "when": (rec or {}).get("t"), "peak_x": (rec or {}).get("x"),
        "count": int(tab.count[i]), "superclass": tab.superclass[i],
        "region": tab.region[i], "regions": tab.regions[i][:4],
        "transmitter": nt or None, "transmitter_confidence": conf, "transmitter_measured_share": float(tab.nt_truth[i]),
        "inputs": tab.top_partners(name, "in"), "outputs": tab.top_partners(name, "out"),
        "curated": None if f is None else {"text": f.text, "cite": f.cite, "doi": f.doi, "game_note": f.game_note,
                                           "tag": LITERATURE},
        "tags": {"data": CONNECTOME, "discovery": GAME_RULE},
    }


def skeleton_points(tab: TypeTable, name: str, n_samples: int = 80) -> np.ndarray | None:
    """(n_samples, 3) points along a cached EM skeleton of one neuron of this type, or None. Reads the cache only: the
    Neurodex never contacts neuPrint (the game's own skeleton fetch is the one place that can, and it is opt-out)."""
    if tab.body_id is None or name not in tab._index:
        return None
    try:
        from kickthefly.sim import morphology

        cache = morphology.default_cache_dir()
        for row in tab.rows(name)[:40]:
            f = cache / f"{int(tab.body_id[row])}.swc"
            if f.exists():
                pts = morphology.parse_swc(f.read_text(encoding="utf-8", errors="ignore"), n_samples)
                if pts is not None:
                    return np.asarray(pts, np.float32)
    except Exception:
        return None
    return None


def list_entries(tab: TypeTable, prog: Progress, region: str | None = None, query: str = "", only_found: bool = False,
                 only_curated: bool = False) -> list[int]:
    """Type indices for the panel list: filtered by primary region, by a substring of the name, and sorted by name.
    The query only searches names the player has discovered (an undiscovered type's name isn't revealed by search)."""
    known = prog.types(tab.brain)
    q = query.strip().lower()
    out = []
    for i, n in enumerate(tab.names):
        n = str(n)
        if region and region != "(all)" and tab.region[i] != region:
            continue
        found = n in known
        if only_found and not found:
            continue
        if only_curated and fact_for(n, tab.brain) is None:
            continue
        if q and not (found and q in n.lower()):
            continue
        out.append(i)
    return out
