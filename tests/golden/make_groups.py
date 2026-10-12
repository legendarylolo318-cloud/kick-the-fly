"""Writes tests/golden/groups.json: group sizes and scent-glomerulus deals computed with the same NumPy code as
legacy-python/kickthefly/game/kick_the_fly.py:Brain.__init__ (adult brain). Run: python3 tests/golden/make_groups.py [pack.npz]"""
import json, sys
import numpy as np

pack = sys.argv[1] if len(sys.argv) > 1 else "data/kick_brain.npz"
z = np.load(pack, allow_pickle=False)
types = z["type"].astype(str); sc = z["superclass"].astype(str); inst = z["instance"].astype(str)
n = len(types)
sides = np.array([(i or "")[-2:] for i in inst])
is_dn = sc == "descending_neuron"
SENSE = {"wind": ("JO-C", "JO-E"), "head": ("BM_", "JO-"), "body": ("SNta",), "legs": ("SNpp",), "wing": ("WG",),
         "heat": ("TRN_VP2",), "cold": ("TRN_VP3",), "humid": ("HRN_",), "smell": ("ORN_",),
         "taste": ("LgLG", "LgAG", "LB", "PhG", "claw_"), "light": ("R1-R6", "R7", "R8"), "loom": ("LPLC2", "LC4"),
         "track": ("LC10",), "small": ("LC11", "LC18", "LC21", "LC26")}
NOT_SENSORY = {"loom", "track", "small"}
MOTOR = (("jump", ("DNg85", "DNg48", "DNg37", "DNge067", "DNg29", "DNge132"), None), ("run", ("DNge122", "DNge104", "DNge102", "DNxl114", "DNge182", "DNge048"), None),
         ("kick", ("DNge074", "DNge075", "DNge096", "DNg30", "DNg34", "DNp38"), None), ("walk", ("DNp09",), None), ("back", ("MDN",), None),
         ("turn_l", ("DNa01", "DNa02"), "L"), ("turn_r", ("DNa01", "DNa02"), "R"), ("fly", tuple(f"DNg02_{c}" for c in "abcdefg"), None),
         ("escape", ("DNp01",), None), ("fire", ("DNp35", "DNpe052"), None))
det = np.full(n, -1); sizes = {}; sense = {}
def add(name, m):
    m = m & (det < 0); det[m] = len(sizes); sizes[name] = int(m.sum()); return m
for region, prefixes in SENSE.items():
    m = np.zeros(n, bool)
    for p in prefixes: m |= np.char.startswith(types, p)
    m = add(region, m if region in NOT_SENSORY else m & (np.char.find(sc, "sensory") >= 0))
    sense[region] = np.flatnonzero(m)
    for s in "LR":
        rows = np.flatnonzero(m & (sides == f"_{s}"))
        sense[region + "_" + s] = rows if len(rows) > 0.2 * m.sum() else np.flatnonzero(m)
for name, tys, side in MOTOR:
    m = is_dn & np.isin(types, tys)
    if side: m &= sides == f"_{side}"
    add(name, m)
add("groom", is_dn & np.isin(types, ("DNg62", "DNge078"))); add("proboscis", np.isin(types, ("MN9",)))
add("song", (sc == "vnc_motor") & (types == "ps1 MN"))
add("aggression", np.isin(types, ("AVLP727m",)) | (np.char.startswith(types, "pC1") & (sc == "cb_intrinsic")))
add("sleep", np.char.startswith(types, "FB6") | np.char.startswith(types, "FB7"))
add("co2", np.isin(types, ("V_ilPN", "V_l2PN")))
add("hot_pn", np.isin(types, ("VP2_adPN", "VP2_l2PN", "VP2+_adPN", "VP1m+VP2_lvPN1", "VP1m+VP2_lvPN2")))
add("cold_pn", np.isin(types, ("VP3+_l2PN", "VP3+_vPN", "VP1l+VP3_ilPN", "VP3+VP1l_ivPN", "VP5+VP3_l2PN")))
add("clock", np.isin(types, ("l-LNv", "s-LNv")))
add("body_extra", np.isin(sc, ("vnc_sensory", "vnc_sensory_tbc", "sensory_ascending", "sensory_ascending_tbc")))
add("reward", np.char.startswith(types, "PAM")); add("punish", np.char.startswith(types, "PPL1"))
TOOLS = ("hand", "flick", "swatter", "bomb", "torch", "cleaner", "zapper", "freeze", "spider", "sugar", "alcohol", "laser", "cva", "decoy", "fruit", "frog", "dragonfly", "mantis")
orn = sense["smell"]
glom = np.array([t.split("_", 1)[1] if "_" in t else t for t in types[orn]])
order = np.random.default_rng(11).permutation(np.unique(glom))
scent = {}
for k, name in enumerate(TOOLS):
    if name == "alcohol": r = orn[np.isin(glom, ["DM1", "DM2", "DP1m"])]
    elif name == "laser": r = orn[:0]
    elif name == "cva": r = orn[np.isin(glom, ["DA1"])]
    elif name == "decoy": r = orn[np.isin(glom, ["DA1", "VA1v"])]
    else: r = orn[np.isin(glom, order[k * 5:(k + 1) * 5])] if len(order) >= (k + 1) * 5 else orn[:5]
    scent[name] = r
scent["player"] = orn[np.isin(glom, order[50:])] if len(order) >= 53 else orn[:3]
odor_order = np.random.default_rng(1985).permutation(np.unique(glom))
for k, key in enumerate(("odor_a", "odor_b", "odor_c")):
    scent[key] = orn[np.isin(glom, odor_order[k * 6:(k + 1) * 6])] if len(odor_order) >= (k + 1) * 6 else orn[:6]
import hashlib
def h(a): return hashlib.sha256(np.asarray(a, "<u4").tobytes()).hexdigest()[:16]
out = {"n": n, "sizes": sizes, "sense_sha": {k: h(v) for k, v in sense.items()}, "scent_sha": {k: h(v) for k, v in scent.items()},
       "scent_len": {k: int(len(v)) for k, v in scent.items()}, "n_glomeruli": int(len(order))}
json.dump(out, open("tests/golden/groups.json", "w"), indent=1, sort_keys=True)
print(len(sizes), "groups;", out["n_glomeruli"], "glomeruli;", {k: sizes[k] for k in ("walk", "fly", "reward", "punish", "loom", "turn_l", "turn_r")})
