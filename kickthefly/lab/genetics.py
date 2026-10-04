"""Genetic toolkit: choose neurons by driver line instead of by cell type.

A split-GAL4 line labels a few cell types. Surgery, the optogenetics laser, thermogenetics, recordings and protocols can name
a line with the neuron spec `line:SS00727` (or several, `line:SS00727,SS02299`); the line is turned into the connectome's
neurons of the cell types it is reported to label.

Where the mapping comes from (CONNECTOME vs LITERATURE vs GAME RULE):
  LITERATURE   kickthefly/data/driver_lines.yaml: line -> cell types and an expression-quality score, copied unchanged from
               Meissner et al. 2025 (eLife 13:RP98405, doi:10.7554/eLife.98405, CC BY 4.0), built by
               tools/build_driver_lines.py. Adult split-GAL4 lines only. It is the light-microscopy literature's claim about
               which cells a line labels, not a measurement on this connectome.
  CONNECTOME   how many neurons carry a type, and which ones, is read from the brain pack (MaleCNS v1.0).
  GAME RULE    a line's cell-type name is matched to a MaleCNS type only when the spelling is identical (no fuzzy matching,
               no prefixes, no synonyms). A name that matches nothing is listed as unmatched, never guessed. A line none of
               whose names match selects no neurons and says so.

What this does NOT know, and the UI says so: what a line expresses in a living fly beyond what its authors scored (see
`off_target`), expression strength, developmental timing, or GAL4 (non-split) lines, for which no redistributable mapping to
cell types was found. The scoring text below is the paper's own wording; other quality values are shown as written.

    from kickthefly.lab import genetics
    genetics.describe("SS00727", br.types)        # line -> cell types -> neuron count, plus the off-target note
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data" / "driver_lines.yaml"

# The paper's own words for its quality scores (Figure 2 legend). Anything else is reported as written, unexplained.
QUALITY_TEXT = {
    "1": "one cell type, strongly and specifically; occasional weak expression may be seen in other cells",
    "2": "two cell types; occasional weak expression may be seen in other cells",
    "3": "three or more cell types",
    "4": "specific expression but weak or variable labeling",
    "5": "not stabilized: groups of neurons are visible but the target cell type was not labeled cleanly",
}


class GeneticsError(ValueError):
    pass


@dataclass(frozen=True)
class Line:
    name: str
    quality: str
    sex_difference: str
    vnc_only: str
    brain_only: str
    cell_types: tuple
    doi: str = ""
    cite: str = ""


@dataclass
class Match:
    """One line against one connectome."""
    line: Line
    matched: dict = field(default_factory=dict)        # MaleCNS type -> neuron count
    unmatched: list = field(default_factory=list)      # names in the table with no identically spelled MaleCNS type
    rows: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))

    @property
    def neurons(self) -> int:
        return int(len(self.rows))


@lru_cache(maxsize=1)
def source() -> dict:
    """The provenance block of the data file (title, doi, license, table hash)."""
    return _load()[0]


@lru_cache(maxsize=1)
def table() -> dict:
    """{line name: Line}."""
    return _load()[1]


def _load():
    import yaml

    try:
        loader = yaml.CSafeLoader
    except AttributeError:
        loader = yaml.SafeLoader
    try:
        doc = yaml.load(DATA.read_text(encoding="utf-8"), Loader=loader)
    except (OSError, yaml.YAMLError) as e:
        raise GeneticsError(f"the driver-line table can't be read ({e})") from None
    lines = {}
    for d in doc.get("lines", []):
        lines[d["line"]] = Line(d["line"], str(d.get("quality", "")), str(d.get("sex_difference", "")),
                                str(d.get("vnc_only", "")), str(d.get("brain_only", "")), tuple(d.get("cell_types", ())),
                                str(d.get("doi", "")), str(d.get("cite", "")))
    return doc.get("source", {}), lines


def get(name: str) -> Line:
    t = table()
    key = str(name).strip()
    if key in t:
        return t[key]
    low = {k.lower(): v for k, v in t.items()}
    if key.lower() in low:
        return low[key.lower()]
    raise GeneticsError(f"unknown driver line {name!r} (the table lists {len(t):,} adult split-GAL4 lines; Lab > Genetic "
                        f"toolkit searches them)")


def search(query: str, limit: int = 50) -> list[Line]:
    """Lines whose name or one of whose cell types contains the text (case-insensitive), best first: exact line name,
    line-name prefix, then cell-type hits."""
    q = str(query).strip().lower()
    if not q:
        return []
    exact, prefix, other = [], [], []
    for ln in table().values():
        n = ln.name.lower()
        if n == q:
            exact.append(ln)
        elif n.startswith(q):
            prefix.append(ln)
        elif any(q in t.lower() for t in ln.cell_types):
            other.append(ln)
    return (exact + prefix + other)[:limit]


def lines_for_type(cell_type: str) -> list[Line]:
    """Every listed line that names exactly this cell type."""
    return [ln for ln in table().values() if cell_type in ln.cell_types]


def type_counts(types) -> dict:
    """{MaleCNS type: neuron count}, cached on the array object (a brain's type labels never change)."""
    cache = getattr(type_counts, "_cache", None)
    if cache is None or cache[0] is not types:
        u, c = np.unique(np.asarray(types).astype(str), return_counts=True)
        cache = type_counts._cache = (types, dict(zip(u.tolist(), c.tolist())))
    return cache[1]


def match(line: str | Line, types) -> Match:
    ln = get(line) if isinstance(line, str) else line
    counts = type_counts(types)
    m = Match(ln)
    for t in ln.cell_types:
        if t in counts:
            m.matched[t] = counts[t]
        else:
            m.unmatched.append(t)
    if m.matched:
        m.rows = np.flatnonzero(np.isin(np.asarray(types).astype(str), list(m.matched)))
    return m


def rows_of_lines(types, spec: str) -> np.ndarray:
    """The rows for a neuron spec `line:A,B` (simcore.rows_of calls this)."""
    names = [s.strip() for s in str(spec).split(":", 1)[1].split(",") if s.strip()]
    if not names:
        raise GeneticsError("line: needs at least one driver line name, for example line:SS00727")
    rows, problems = [], []
    for n in names:
        m = match(n, types)
        if not m.neurons:
            problems.append(f"{m.line.name} labels {', '.join(m.line.cell_types) or 'no listed cell type'}; none of those is "
                            f"spelled like a cell type in this connectome, so it selects no neurons")
        rows.append(m.rows)
    if problems:
        raise GeneticsError("; ".join(problems))
    return np.unique(np.concatenate(rows))


def off_target(line: Line) -> tuple[str, str]:
    """(level, sentence): whether the source says this line's expression is off-target.

    level is "minimal" (score 1), "some" (2: the paper says two cell types; if the table lists just one, the other is
    off-target), "yes" (3: three or more cell types), "weak" (4), "unstable" (5) or "unknown" when the score is not one the
    paper's text explains. The sentence never says more than the paper does."""
    q = line.quality.strip()
    base = QUALITY_TEXT.get(q)
    if q == "1":
        return "minimal", f"quality 1: {base}."
    if q == "2":
        n = len(line.cell_types)
        extra = (" The table lists one cell type for this line, so the second labeled type is off-target."
                 if n == 1 else " Both labeled types are the ones the table lists.")
        return "some", f"quality 2: {base}.{extra}"
    if q == "3":
        return "yes", f"quality 3: {base}; expression beyond the listed types is expected."
    if q == "4":
        return "weak", f"quality 4: {base}."
    if q == "5":
        return "unstable", f"quality 5: {base}."
    return "unknown", (f"quality \"{q}\" is recorded in the source table, but its meaning isn't given in the text this table "
                       f"was built from, so nothing is claimed about off-target expression.")


def describe(line: str | Line, types) -> dict:
    """Everything the UI shows for a line: cell types, what matched the connectome, neuron count, off-target note."""
    m = match(line, types)
    level, text = off_target(m.line)
    return dict(line=m.line.name, quality=m.line.quality, cell_types=list(m.line.cell_types), matched=dict(m.matched),
                unmatched=list(m.unmatched), neurons=m.neurons, off_target=level, off_target_text=text,
                cite=m.line.cite, doi=m.line.doi, sex_difference=m.line.sex_difference, brain_only=m.line.brain_only,
                vnc_only=m.line.vnc_only)


def coverage(types) -> dict:
    """How much of the table can be used on this connectome: lines with at least one matching type, and the names that
    never match. A number for the docs and the handoff, not a claim about biology."""
    counts = type_counts(types)
    usable = sum(1 for ln in table().values() if any(t in counts for t in ln.cell_types))
    all_names = {t for ln in table().values() for t in ln.cell_types}
    hit = {t for t in all_names if t in counts}
    return dict(lines=len(table()), lines_with_a_match=usable, cell_type_names=len(all_names),
                cell_type_names_matched=len(hit))
