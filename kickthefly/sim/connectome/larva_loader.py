"""Drosophila larva connectome loader and brain pack builder.

Loads the synaptic-resolution connectome of the Drosophila larva brain from
Winding et al. 2023 (Science 379, eadd9330, DOI: 10.1126/science.add9330).

Reads neuron count, synapse count, cell types and annotations dynamically from
the official released dataset (Supplementary-Data-S1). Builds the compact
kick_larva_brain.npz brain pack.

    python -m kickthefly.sim.connectome.larva_loader build
"""
from __future__ import annotations

import csv
import io
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

from kickthefly import DATA_DIR

LARVA_DIR = DATA_DIR / "larva"
LARVA_ZIP = LARVA_DIR / "Supplementary-Data-S1.zip"
LARVA_PACK_NAME = "kick_larva_brain.npz"
LARVA_URL = "https://raw.githubusercontent.com/brain-networks/larval-drosophila-connectome/main/Supplementary-Data-S1.zip"
LARVA_SHA256 = "8c1f43809ed5d527ba61b154e377cc21da26383a75eda8aab85ce05607a72a4c"

CITATION = "Winding et al. 2023, 'The connectome of an insect brain', Science 379:eadd9330. DOI: 10.1126/science.add9330"
# Neither the Science supplement nor the mirror above states a license, so the pack is never bundled: it is built on the
# player's machine from the downloaded Data S1 (see docs/larva.md).
LICENSE = "Science Data S1 (Winding et al. 2023); no explicit license stated. not redistributed: built locally on first use"


def larva_groups(subtype) -> dict:
    """Row indices of the annotated larval groups the game and the validation use, from the Data S1 annotations
    (the pack's `subtype`, i.e. annotations.csv 'additional_annotations'; `type` only holds the broad class).

    noci:    ascending neurons annotated 'noci' (incl. A00c_a4/a5/a6). These are nociceptive ascending neurons from
             the nerve cord; the class IV md sensory neurons themselves are not in the brain dataset.
    chordo:  ascending neurons annotated 'mechano-Ch' (chordotonal pathway).
    noci_pn / chordo_pn: brain neurons annotated 'noci 2nd_order PN' / 'mechano-Ch 2nd_order PN'. The Basin
             interneurons of Ohyama et al. 2015 sit in the nerve cord and are not in this dataset.
    goro:    neurons annotated '_telegoro-1' (DN-VNC).
    """
    st = np.asarray(subtype).astype(str)
    parts = [set(p.strip() for p in x.split(";")) for x in st]
    has = lambda tag: np.array([tag in p for p in parts], bool)      # noqa: E731
    return dict(
        noci=np.flatnonzero(has("noci")),
        chordo=np.flatnonzero(has("mechano-Ch")),
        mechano=np.flatnonzero(has("mechano-Ch") | has("mechano-II/III")),
        noci_pn=np.flatnonzero(has("noci 2nd_order PN")),
        chordo_pn=np.flatnonzero(has("mechano-Ch 2nd_order PN")),
        goro=np.flatnonzero(has("_telegoro-1")),
    )


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dataset(path: Path = LARVA_ZIP) -> Path:
    """Download the larval connectome archive if not present."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[larva_loader] downloading {LARVA_URL} -> {path}...")
    req = urllib.request.Request(LARVA_URL, headers={"User-Agent": "Mozilla/5.0 (KickTheFly)"})
    tmp = path.with_name(path.name + ".part")
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as f:
        f.write(resp.read())
    got = _sha256(tmp)
    if got != LARVA_SHA256:        # the mirror is not the publisher: only accept the exact Data S1 archive
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"larva dataset checksum mismatch (got {got}, expected {LARVA_SHA256})")
    tmp.replace(path)
    print(f"[larva_loader] downloaded {path.stat().st_size:,} bytes (sha256 ok)")
    return path


def load_raw_data(zip_path: Path | None = None) -> tuple[list[int], np.ndarray, dict[int, dict]]:
    """Load matrix and annotations directly from Supplementary-Data-S1.zip.
    
    Returns:
        body_ids: list of int body IDs in matrix order
        matrix: (n, n) float32 adjacency matrix where matrix[pre, post] is synapse count
        annotations: dict mapping body_id -> dict of metadata
    """
    path = ensure_dataset(zip_path or LARVA_ZIP)
    with zipfile.ZipFile(path) as z:
        # 1. Connectivity matrix
        with io.TextIOWrapper(z.open("Supplementary-Data-S1/all-all_connectivity_matrix.csv"), encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            body_ids = [int(x) for x in header[1:]]
            n = len(body_ids)
            mat = np.zeros((n, n), dtype=np.float32)
            for i, r in enumerate(reader):
                mat[i] = [float(x) for x in r[1:]]

        id_to_idx = {b: i for i, b in enumerate(body_ids)}

        # 2. Annotations
        ann_map: dict[int, dict] = {}
        with io.TextIOWrapper(z.open("Supplementary-Data-S1/annotations.csv"), encoding="utf-8") as f:
            ann_reader = csv.reader(f)
            next(ann_reader)
            for r in ann_reader:
                lid, rid, ctype, add_annot, cluster = r
                c_val = int(cluster) if cluster and cluster != "no cluster" else -1
                for bid_str, side in [(lid, "left"), (rid, "right")]:
                    if bid_str != "no pair":
                        b = int(bid_str)
                        other_pair = int(rid if side == "left" else lid) if (rid if side == "left" else lid) != "no pair" else None
                        ann_map[b] = {
                            "celltype": ctype,
                            "additional_annotations": add_annot,
                            "hemisphere": side,
                            "homologue": other_pair,
                            "cluster": c_val,
                        }

    return body_ids, mat, ann_map


def build_larva_pack(out: Path = DATA_DIR / LARVA_PACK_NAME) -> Path:
    """Build compact kick_larva_brain.npz from the larval connectome data."""
    body_ids, mat, ann_map = load_raw_data()
    n = len(body_ids)
    id_to_idx = {b: i for i, b in enumerate(body_ids)}

    types = np.full(n, "unassigned", dtype=object)
    subtypes = np.full(n, "", dtype=object)
    superclasses = np.full(n, "central_brain", dtype=object)
    instances = np.full(n, "", dtype=object)
    regions = np.full(n, "central_brain", dtype=object)
    nt = np.full(n, "acetylcholine", dtype=object)
    nt_source = np.full(n, "inferred", dtype=object)
    nt_conf = np.full(n, 0.8, dtype=np.float32)
    soma = np.zeros((n, 3), dtype=np.float32)

    signs = np.ones(n, dtype=np.float32)

    for i, b in enumerate(body_ids):
        ann = ann_map.get(b)
        if ann:
            ct = ann["celltype"]
            sub = ann["additional_annotations"]
            side = ann["hemisphere"]
            types[i] = ct
            subtypes[i] = sub
            instances[i] = f"{sub or ct}_{'L' if side == 'left' else 'R'}_{b}"

            if ct == "sensory":
                superclasses[i] = "sensory"
                regions[i] = "Sensory"
            elif "DN" in ct:
                superclasses[i] = "descending_neuron"
                regions[i] = "SEZ" if "SEZ" in ct else "VNC"
            elif ct == "ascending":
                superclasses[i] = "ascending_neuron"
                regions[i] = "VNC"
            elif ct == "LN":
                superclasses[i] = "local_interneuron"
                regions[i] = "Antennal Lobe"
                signs[i] = -1.0
                nt[i] = "GABA"
            elif ct in ("KC", "MBON", "MBIN", "MB-FBN", "MB-FFN"):
                superclasses[i] = "mushroom_body"
                regions[i] = "Mushroom Body"
                if ct == "MBON":
                    signs[i] = -1.0
                    nt[i] = "GABA"
            elif "PN" in ct:
                superclasses[i] = "projection_neuron"
                regions[i] = "Antennal Lobe"
            else:
                superclasses[i] = "central_brain"
                regions[i] = "Central Brain"

            # Layout coordinates: hemisphere determines X, cluster and type determine Y and Z
            x_sign = -1.0 if side == "left" else 1.0
            cid = ann["cluster"]
            # Spread somas bilaterally around center
            soma[i, 0] = x_sign * (100.0 + (abs(cid) % 15) * 12.0 + (i % 7) * 3.0)
            soma[i, 1] = 200.0 + ((abs(cid) * 7) % 250) + (i % 11) * 2.0
            soma[i, 2] = 50.0 + ((i * 13) % 150)
        else:
            instances[i] = f"unannotated_{b}"
            soma[i, 0] = ((i % 2) * 2 - 1) * (80.0 + (i % 20) * 8.0)
            soma[i, 1] = 180.0 + (i % 100) * 2.5
            soma[i, 2] = 75.0 + (i % 50) * 2.0

    # Signed matrix [post, pre]
    # Presynaptic neuron i scales its outgoing synapses by signs[i]
    signed_mat = mat * signs[:, None]
    signed_W = signed_mat.T.astype(np.float32) # [post, pre]
    signed_csr = sp.csr_array(signed_W)
    signed_csr.eliminate_zeros()

    counts_in = mat.sum(axis=0).astype(np.float32)
    inv = np.where(counts_in > 0, 1.0 / np.maximum(counts_in, 1), 0).astype(np.float32)

    types_arr = types.astype(str)
    dan = np.flatnonzero(np.char.find(types_arr, "MBIN") >= 0)
    mbon = np.flatnonzero(np.char.find(types_arr, "MBON") >= 0)
    dan_mbon = np.zeros((len(dan), len(mbon)), dtype=np.int16)
    if len(dan) and len(mbon):
        dan_mbon = mat[dan][:, mbon].astype(np.int16)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        indptr=signed_csr.indptr.astype(np.int32),
        indices=signed_csr.indices.astype(np.int32),
        data=signed_csr.data.astype(np.int16),
        inv=inv,
        type=types_arr,
        subtype=subtypes.astype(str),
        superclass=superclasses.astype(str),
        instance=instances.astype(str),
        soma=soma,
        body_id=np.asarray(body_ids, np.int64),
        region=regions.astype(str),
        dan=dan.astype(np.int32),
        mbon=mbon.astype(np.int32),
        dan_mbon=dan_mbon,
        nt=nt.astype(str),
        nt_conf=nt_conf,
        nt_source=nt_source.astype(str),
        subclass=subtypes.astype(str),
        brain_type=np.array("larva"),
        citation=np.array(CITATION),
        license=np.array(LICENSE),
    )
    print(f"[larva_loader] built {out}: {n:,} neurons, {signed_csr.nnz:,} synapse pairs, {int(mat.sum()):,} total synapses")
    return out


def ensure_larva_brain_pack(out_path: Path | None = None) -> Path:
    out = out_path or (DATA_DIR / LARVA_PACK_NAME)
    if not out.exists():
        build_larva_pack(out)
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build_larva_pack()
    else:
        print(__doc__)
