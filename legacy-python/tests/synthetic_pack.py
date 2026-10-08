"""A tiny SYNTHETIC brain pack for tests that exercise the game's plumbing (UI, saves, exports) without the 1.1 GB connectome.

It is random wiring with real cell-type names, made here from a fixed seed. It is NOT the connectome and says nothing
about biology: never use it for validation, never ship it, and never read a result off a simulation that runs on it.
It exists so code that needs "a Brain with the game's groups" can run in CI and on a machine without `data/`.
Tests that use it carry the `synthetic` marker's fixture (`synthetic_pack` in conftest.py), which points
`brainpack.find` at this file for the test only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# (type, superclass, how many neurons, region, transmitter). Names are real MaleCNS type names the game already uses.
TYPES = (
    ("BM_InOm", "cb_sensory", 40, "Central Brain", "acetylcholine"), ("JO-C", "cb_sensory", 20, "Central Brain", "acetylcholine"),
    ("JO-E", "cb_sensory", 20, "Central Brain", "acetylcholine"), ("JO-A", "cb_sensory", 10, "Central Brain", "acetylcholine"),
    ("SNta01", "vnc_sensory", 40, "VNC (T1)", "acetylcholine"), ("SNpp01", "vnc_sensory", 40, "VNC (T2)", "acetylcholine"),
    ("WG1", "vnc_sensory", 20, "VNC (T2)", "acetylcholine"),
    ("TRN_VP2", "cb_sensory", 12, "Antennal Lobe", "acetylcholine"), ("TRN_VP3a", "cb_sensory", 12, "Antennal Lobe", "acetylcholine"),
    ("HRN_VP4", "cb_sensory", 10, "Antennal Lobe", "acetylcholine"),
    ("ORN_DM1", "cb_sensory", 30, "Antennal Lobe", "acetylcholine"), ("ORN_DM2", "cb_sensory", 30, "Antennal Lobe", "acetylcholine"),
    ("ORN_DA1", "cb_sensory", 40, "Antennal Lobe", "acetylcholine"), ("ORN_DP1m", "cb_sensory", 30, "Antennal Lobe", "acetylcholine"),
    ("ORN_V", "cb_sensory", 20, "Antennal Lobe", "acetylcholine"),
    *((f"ORN_G{i}", "cb_sensory", 12, "Antennal Lobe", "acetylcholine") for i in range(60)),
    ("LgLG5", "cb_sensory", 8, "Gnathal (GNG)", "acetylcholine"), ("LgLG6", "cb_sensory", 8, "Gnathal (GNG)", "acetylcholine"),
    ("LB1", "cb_sensory", 20, "Gnathal (GNG)", "acetylcholine"), ("PhG1", "cb_sensory", 10, "Gnathal (GNG)", "acetylcholine"),
    ("R1-R6", "ol_sensory", 60, "Optic Lobe", "histamine"), ("R7", "ol_sensory", 20, "Optic Lobe", "histamine"),
    ("R8", "ol_sensory", 20, "Optic Lobe", "histamine"),
    ("LPLC2", "visual_projection", 60, "Optic Lobe", "acetylcholine"), ("LC4", "visual_projection", 40, "Optic Lobe", "acetylcholine"),
    ("LC10a", "visual_projection", 30, "Optic Lobe", "acetylcholine"), ("LC11", "visual_projection", 10, "Optic Lobe", "acetylcholine"),
    ("T4a", "ol_intrinsic", 40, "Optic Lobe", "acetylcholine"), ("T5a", "ol_intrinsic", 40, "Optic Lobe", "acetylcholine"),
    ("Mi1", "ol_intrinsic", 60, "Optic Lobe", "acetylcholine"),
    ("DNp01", "descending_neuron", 2, "Central Brain", "acetylcholine"), ("DNg85", "descending_neuron", 2, "Central Brain", "acetylcholine"),
    ("DNg48", "descending_neuron", 2, "Central Brain", "acetylcholine"), ("DNge122", "descending_neuron", 2, "Central Brain", "acetylcholine"),
    ("DNge074", "descending_neuron", 2, "Central Brain", "acetylcholine"), ("DNp09", "descending_neuron", 2, "Central Brain", "acetylcholine"),
    ("MDN", "descending_neuron", 2, "Central Brain", "acetylcholine"), ("DNa01", "descending_neuron", 2, "Central Brain", "acetylcholine"),
    ("DNa02", "descending_neuron", 2, "Central Brain", "acetylcholine"),
    *((f"DNg02_{c}", "descending_neuron", 2, "Central Brain", "acetylcholine") for c in "abcdefg"),
    ("DNp35", "descending_neuron", 2, "Central Brain", "acetylcholine"), ("DNg62", "descending_neuron", 2, "Central Brain", "gaba"),
    ("DNge078", "descending_neuron", 2, "Central Brain", "acetylcholine"), ("DNg28", "descending_neuron", 2, "Central Brain", "acetylcholine"),
    ("pIP10", "descending_neuron", 2, "Central Brain", "acetylcholine"), ("DNpe052", "descending_neuron", 2, "Central Brain", "gaba"),
    ("MN9", "cb_motor", 2, "Gnathal (GNG)", "acetylcholine"), ("ps1 MN", "vnc_motor", 2, "VNC (T2)", "acetylcholine"),
    ("MNfl", "vnc_motor", 8, "VNC (T1)", "acetylcholine"),
    ("AVLP727m", "cb_intrinsic", 6, "Central Brain", "acetylcholine"), ("pC1a", "cb_intrinsic", 6, "Central Brain", "acetylcholine"),
    ("FB6A", "cb_intrinsic", 8, "Central Complex", "acetylcholine"), ("FB7A", "cb_intrinsic", 8, "Central Complex", "acetylcholine"),
    ("EPG", "cb_intrinsic", 20, "Central Complex", "acetylcholine"),
    ("V_ilPN", "cb_intrinsic", 4, "Antennal Lobe", "acetylcholine"), ("VP2_adPN", "cb_intrinsic", 4, "Antennal Lobe", "acetylcholine"),
    ("VP3+_vPN", "cb_intrinsic", 4, "Antennal Lobe", "acetylcholine"), ("DA1_lPN", "cb_intrinsic", 6, "Antennal Lobe", "acetylcholine"),
    ("LHAV4a4", "cb_intrinsic", 4, "Central Brain", "acetylcholine"), ("GNG540", "cb_intrinsic", 4, "Gnathal (GNG)", "acetylcholine"),
    ("GNG550", "cb_intrinsic", 4, "Gnathal (GNG)", "acetylcholine"),
    ("l-LNv", "cb_intrinsic", 4, "Central Brain", "acetylcholine"), ("s-LNv", "cb_intrinsic", 4, "Central Brain", "acetylcholine"),
    ("KCg-m", "cb_intrinsic", 120, "Mushroom Body", "acetylcholine"), ("KCab-c", "cb_intrinsic", 60, "Mushroom Body", "acetylcholine"),
    *((f"MBON{i:02d}", "cb_intrinsic", 2, "Mushroom Body", "gaba" if i % 2 else "glutamate") for i in range(1, 25)),
    *((f"PAM{i:02d}", "cb_intrinsic", 8, "Mushroom Body", "dopamine") for i in range(1, 16)),
    *((f"PPL1{i:02d}", "cb_intrinsic", 2, "Mushroom Body", "dopamine") for i in range(1, 9)),
    ("AN_asc", "ascending_neuron", 40, "Gnathal (GNG)", "acetylcholine"),
    ("IN_vnc", "vnc_intrinsic", 60, "VNC (T2)", "gaba"), ("CB_misc", "cb_intrinsic", 160, "Central Brain", "glutamate"),
    ("CB_unassigned", "cb_intrinsic", 60, "unassigned", "acetylcholine"),
)
SEED = 20260930
PACK_NAME = "kick_brain.npz"


def build(out: Path, seed: int = SEED, fan_in: int = 18) -> Path:
    """Write the synthetic pack (same arrays as brainpack._build_adult) and return its path."""
    import scipy.sparse as sp

    rng = np.random.default_rng(seed)
    types, sc, reg, nt = [], [], [], []
    for t, s, k, r, n in TYPES:
        types += [t] * k
        sc += [s] * k
        reg += [r] * k
        nt += [n] * k
    n = len(types)
    types, sc, reg, nt = map(np.array, (types, sc, reg, nt))
    inst = np.array([f"{t}_{'LR'[i % 2]}" for i, t in enumerate(types)])
    # random sparse wiring, excitatory 70%; integer synapse counts like the real pack
    post = np.repeat(np.arange(n), fan_in)
    pre = rng.integers(0, n, size=n * fan_in)
    cnt = rng.integers(1, 12, size=n * fan_in) * np.where(rng.random(n * fan_in) < 0.7, 1, -1)
    m = post != pre
    signed = sp.coo_array((cnt[m].astype(np.int16), (post[m], pre[m])), shape=(n, n)).tocsr()
    signed.sum_duplicates()
    tot = np.asarray(abs(signed).sum(axis=1)).ravel()
    inv = np.where(tot > 0, 1.0 / np.maximum(tot, 1), 0).astype(np.float32)
    soma = np.stack([rng.uniform(5000, 90000, n), rng.uniform(4000, 52000, n), rng.uniform(2000, 55000, n)], 1).astype(np.float32)
    soma[np.char.find(sc, "sensory") >= 0] = np.nan
    dan = np.flatnonzero(np.char.startswith(types, "PAM") | np.char.startswith(types, "PPL1"))
    mbon = np.flatnonzero(np.char.startswith(types, "MBON"))
    dan_mbon = rng.integers(0, 6, size=(len(dan), len(mbon))).astype(np.int16)
    conf = np.clip(rng.normal(0.85, 0.1, n), 0.3, 0.99).astype(np.float32)
    sub = np.where(sc == "vnc_motor", "fl", "")
    out = Path(out)
    np.savez_compressed(
        out, nt=nt, nt_conf=conf, nt_source=np.array(["predicted_nt"] * n),
        indptr=signed.indptr.astype(np.int32), indices=signed.indices.astype(np.int32), data=signed.data.astype(np.int16),
        inv=inv, type=types, superclass=sc, instance=inst, soma=soma, dan=dan.astype(np.int32), mbon=mbon.astype(np.int32),
        dan_mbon=dan_mbon, subclass=sub, body_id=np.arange(1000, 1000 + n, dtype=np.int64), region=reg,
        brain_type=np.array("adult"))
    return out
