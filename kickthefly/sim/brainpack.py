"""Compact brain pack for Kick the Fly: everything the game needs from the connectome in one ~40 MB file.

Holds the signed synapse matrix [post, pre] as int16 counts plus each neuron's 1 / total input synapses, the
neuron type, superclass, subclass and instance labels, FlyEM body IDs, each neuron's cell-body position, and the
dopamine-neuron ->
mushroom body output neuron synapse counts (dopamine synapses have no sign, so the signed matrix drops them; the
game's learning needs to know which dopamine neurons reach which output neurons). The game rebuilds the
simulator's rate-normalized matrix from it in about a second, so a packaged build skips the 1.1 GB download.

Derived from Janelia FlyEM MaleCNS v1.0 (CC BY 4.0).

    python -m kickthefly.sim.brainpack build    # data/graph.pkl + body annotations -> data/kick_brain.npz
"""
from __future__ import annotations

import sys
from pathlib import Path

from kickthefly import DATA_DIR as _DATA_DIR
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

PACK_NAME = "kick_brain.npz"
DATA_DIR = _DATA_DIR                                    # the repo-root data/ folder, unchanged since 2.2
ANNOTATIONS = "body-annotations-male-cns-v1.0-minconf-0.5.feather"


def soma_positions(g) -> np.ndarray:
    """(n, 3) float32 cell-body position per graph row, NaN where the annotation has none (e.g. sensory neurons)."""
    import pyarrow.feather as feather

    t = feather.read_table(DATA_DIR / ANNOTATIONS, columns=["bodyId", "somaLocation", "tosomaLocation"])
    rows = np.array([g.index.get(int(b), -1) for b in t["bodyId"].to_numpy()])
    pos = np.full((g.n, 3), np.nan, np.float32)
    for col in ("tosomaLocation", "somaLocation"):          # a real soma wins over the soma-tract estimate
        for r, v in zip(rows, t[col].to_pylist()):
            if r >= 0 and v is not None:
                pos[r] = v
    return pos


def subclasses(g) -> np.ndarray:
    """Annotation `subclass` per graph row (e.g. fl/ml/hl for front/middle/hind leg motor neurons), "" where none."""
    import pyarrow.feather as feather

    t = feather.read_table(DATA_DIR / ANNOTATIONS, columns=["bodyId", "subclass"])
    out = np.full(g.n, "", dtype=object)
    for b, v in zip(t["bodyId"].to_numpy(), t["subclass"].to_pylist()):
        r = g.index.get(int(b), -1)
        if r >= 0 and v is not None:
            out[r] = str(v)
    return out.astype(str)


def regions(g) -> np.ndarray:
    """Dataset neuropil region annotations per graph row, 'unassigned' where none.

    Packs built since 2.6 carry these, so only a pack older than the annotations file ever lands here; a packaged
    build has neither pyarrow nor the feather table, and then every neuron stays 'unassigned'.
    """
    out = np.full(g.n, "unassigned", dtype=object)
    path = DATA_DIR / ANNOTATIONS
    if not path.exists():
        return out.astype(str)
    try:
        import pyarrow.feather as feather
    except ImportError:
        return out.astype(str)
    t = feather.read_table(path, columns=["bodyId", "class", "somaNeuromere", "superclass"])
    bids = t["bodyId"].to_numpy()
    classes = t["class"].to_pylist()
    neuromeres = t["somaNeuromere"].to_pylist()
    superclasses = t["superclass"].to_pylist()
    if getattr(g, "index", None) is not None:
        b_to_idx = g.index
    elif getattr(g, "body_id", None) is not None:
        b_to_idx = {int(b): i for i, b in enumerate(g.body_id)}
    else:
        b_to_idx = {}

    for b, cls, nm, sc in zip(bids, classes, neuromeres, superclasses):
        idx = b_to_idx.get(int(b), -1)
        if idx < 0:
            continue
        reg = None
        if cls in ("ALPN", "ALLN", "ALIN", "ALON"):
            reg = "Antennal Lobe"
        elif cls in ("Kenyon_Cell", "MBON"):
            reg = "Mushroom Body"
        elif cls == "CX":
            reg = "Central Complex"
        elif cls in ("visual", "ol_bilateral") or (sc and (sc.startswith("ol_") or sc.startswith("visual_"))):
            reg = "Optic Lobe"
        elif nm in ("T1", "T2", "T3"):
            reg = f"VNC ({nm})"
        elif nm and nm.startswith("A") and nm[1:].isdigit():
            reg = "VNC (Abdomen)"
        elif nm in ("GNG", "LB", "MX", "MD"):
            reg = "Gnathal (GNG)"
        elif nm in ("CG", "DC", "TC"):
            reg = "Central Brain"
        elif sc and sc.startswith("vnc_"):
            reg = "VNC (Other)"
        if reg:
            out[idx] = reg
    return out.astype(str)


NEUROTRANSMITTERS = "body-neurotransmitters-male-cns-v1.0.feather"


def neurotransmitters(g) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(nt, confidence, source) per graph row, from the dataset's own neurotransmitter predictions.

    `nt` is the transmitter the loader signed each neuron's synapses with (consensus_nt, or predicted_nt where the
    consensus is 'unclear'). `source` says where it came from and `confidence` is the dataset's number for exactly
    that call:

      ground_truth   1.0      the dataset has a measured transmitter for this body
      consensus_nt   the per-body predicted_nt_confidence when the per-body prediction agrees with the consensus,
                     else the cell type's celltype_predicted_nt_confidence when that one does
      predicted_nt   the per-body predicted_nt_confidence
      missing/unknown NaN     no row for this body, or neither prediction matches the consensus

    Confidence is the model's, not an error bar on biology: a neuron the dataset is 95% sure about can still be
    wrong. What the Lab does with it (flip everything below a cutoff) is this game's choice.
    """
    import pyarrow.feather as feather

    n = g.n
    nt = np.array(["" if v is None else str(v) for v in getattr(g, "nt", np.full(n, ""))])
    source = np.array([str(v) for v in getattr(g, "nt_source", np.full(n, "missing"))])
    conf = np.full(n, np.nan, np.float32)
    path = DATA_DIR / NEUROTRANSMITTERS
    if not path.exists():
        return nt, conf, source
    t = feather.read_table(path, columns=["body", "predicted_nt", "predicted_nt_confidence", "ground_truth",
                                          "celltype_predicted_nt", "celltype_predicted_nt_confidence"])
    body = t["body"].to_numpy()
    order = np.argsort(body)
    sorted_body = body[order]
    bids = np.asarray(g.body_ids, np.int64)
    pos = np.searchsorted(sorted_body, bids)
    has = (pos < len(sorted_body)) & (sorted_body[np.minimum(pos, len(sorted_body) - 1)] == bids)
    idx = order[pos[has]]
    pred = np.array([("" if v is None else str(v)) for v in t["predicted_nt"].to_pylist()], dtype=object)[idx]
    ct = np.array([("" if v is None else str(v)) for v in t["celltype_predicted_nt"].to_pylist()], dtype=object)[idx]
    truth = np.array([v is not None for v in t["ground_truth"].to_pylist()])[idx]
    pconf = t["predicted_nt_confidence"].to_numpy()[idx]
    cconf = t["celltype_predicted_nt_confidence"].to_numpy()[idx]
    here = nt[has]
    c = np.where(pred == here, pconf, np.where(ct == here, cconf, np.nan))
    c = np.where(truth, 1.0, c)
    conf[has] = c.astype(np.float32)
    source[has] = np.where(truth, "ground_truth", source[has])
    return nt, conf, source


def _build_adult(out: Path = DATA_DIR / PACK_NAME) -> Path:
    from kickthefly.sim.connectome.loader import load_graph

    g = load_graph()
    signed = g.adjacency.T.tocsr()                          # [post, pre]
    counts_in = np.asarray(g.weights.sum(axis=0)).ravel()
    inv = np.where(counts_in > 0, 1.0 / np.maximum(counts_in, 1), 0).astype(np.float32)
    signed.eliminate_zeros()
    if np.abs(signed.data).max() > np.iinfo(np.int16).max:
        raise ValueError("synapse count does not fit int16")

    def labels(a):
        return np.array(["" if x is None else str(x) for x in a])

    soma = soma_positions(g)
    types = labels(g.type)
    dan = np.flatnonzero(np.char.startswith(types, "PAM") | np.char.startswith(types, "PPL1"))
    mbon = np.flatnonzero(np.char.startswith(types, "MBON"))
    dan_mbon = g.weights.tocsr()[dan][:, mbon].toarray().astype(np.int16)        # [DAN, MBON] synapse counts
    reg = regions(g)
    nt, nt_conf, nt_source = neurotransmitters(g)
    np.savez_compressed(
        out, nt=nt, nt_conf=nt_conf, nt_source=nt_source,
        indptr=signed.indptr.astype(np.int32), indices=signed.indices.astype(np.int32),
        data=signed.data.astype(np.int16), inv=inv, type=types, superclass=labels(g.superclass),
        instance=labels(g.instance), soma=soma, dan=dan.astype(np.int32), mbon=mbon.astype(np.int32), dan_mbon=dan_mbon,
        subclass=subclasses(g), body_id=np.asarray(g.body_ids, np.int64), region=reg, brain_type=np.array("adult"),
    )
    print(f"[brainpack] wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {g.n:,} neurons, {signed.nnz:,} synapse pairs, "
          f"{int((~np.isnan(soma[:, 0])).sum()):,} cell bodies, "
          f"{int(np.count_nonzero(nt_source == 'ground_truth')):,} measured transmitters)")
    return out


LARVA_PACK_NAME = "kick_larva_brain.npz"


def find(brain: str = "adult") -> Path | None:
    """The pack next to a PyInstaller bundle, next to the exe, or in data/."""
    pack_filename = LARVA_PACK_NAME if brain == "larva" else PACK_NAME
    roots = [
        Path(getattr(sys, "_MEIPASS", "")),
        Path(sys.executable).resolve().parent,
        DATA_DIR,
        DATA_DIR / "larva",
    ]
    for root in roots:
        if str(root) and (root / pack_filename).exists():
            return root / pack_filename
    return None


def build(brain: str = "adult", out: Path | None = None) -> Path:
    """Build the adult pack (from the MaleCNS download) or the larva pack (from Winding et al. 2023 Data S1)."""
    if brain == "larva":
        from kickthefly.sim.connectome.larva_loader import build_larva_pack
        return build_larva_pack(out or (DATA_DIR / LARVA_PACK_NAME))
    return _build_adult(out or (DATA_DIR / PACK_NAME))


def load(path: Path):
    """Returns (graph-like namespace with n/type/superclass/instance, W_in [post, pre] float32, soma (n, 3))."""
    z = np.load(path)
    inv = z["inv"]
    n = len(inv)
    data = z["data"].astype(np.float32)
    indptr = z["indptr"]
    data *= np.repeat(inv, np.diff(indptr))                 # row-normalize by the post neuron's input count
    W = sp.csr_array((data, z["indices"], indptr), shape=(n, n))
    inst = z["instance"]
    g = SimpleNamespace(n=n, type=z["type"], superclass=z["superclass"], instance=np.where(inst == "", None, inst))
    if "dan_mbon" in z:
        g.dan, g.mbon, g.dan_mbon = z["dan"], z["mbon"], z["dan_mbon"]
    # packs from before 2.6 have neither: leg motor neuron groups and body IDs in exports are then unavailable
    g.subclass = z["subclass"] if "subclass" in z else np.full(n, "", dtype="<U1")
    g.subtype = z["subtype"] if "subtype" in z else g.subclass
    g.body_id = z["body_id"] if "body_id" in z else None
    g.region = z["region"] if "region" in z else regions(g)
    # packs from before 2.7 have no transmitter predictions: the Lab's sign-flip and inhibition controls then say
    # the pack needs rebuilding instead of guessing one.
    g.nt = z["nt"] if "nt" in z else None
    g.nt_conf = z["nt_conf"] if "nt_conf" in z else None
    g.nt_source = z["nt_source"] if "nt_source" in z else None
    g.brain_type = str(z["brain_type"]) if "brain_type" in z else "adult"
    g.citation = str(z["citation"]) if "citation" in z else ""
    g.license = str(z["license"]) if "license" in z else ""
    return g, W, z["soma"]


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        build()
    else:
        print(__doc__)
