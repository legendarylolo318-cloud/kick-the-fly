"""Writes tests/golden/neurodex.json: the type table of the real pack as `core/neurodex.py:build_table` makes it (sampled entries, region
progress, list filters) and a deterministic scenario through `Tracker.observe`. Run (needs numpy, scipy):
    python3 tests/golden/make_neurodex.py [pack.npz]"""
import json, sys, tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "legacy-python"))
from kickthefly.core import neurodex as nd  # noqa: E402

pack = sys.argv[1] if len(sys.argv) > 1 else "data/kick_brain.npz"
z = np.load(pack, allow_pickle=False)
keep = ("type", "superclass", "region", "nt", "nt_conf", "nt_source", "body_id", "indptr", "indices", "data")
tab = nd.build_table({k: z[k] for k in keep if k in z.files}, "adult")
T = len(tab)
out = {"n_types": T, "names_sha": __import__("hashlib").sha256("\n".join(tab.names).encode()).hexdigest()[:16]}
pick = list(range(0, T, max(1, T // 60)))
ent = []
for i in pick:
    name = str(tab.names[i])
    ent.append(dict(name=name, count=int(tab.count[i]), superclass=tab.superclass[i], region=tab.region[i], regions=[[r, c] for r, c in tab.regions[i]],
                    nt=tab.nt[i], nt_conf=None if np.isnan(tab.nt_conf[i]) else float(tab.nt_conf[i]), nt_truth=float(tab.nt_truth[i]),
                    inputs=[[n, v, s] for n, v, s in tab.top_partners(name, "in")], outputs=[[n, v, s] for n, v, s in tab.top_partners(name, "out")]))
out["entries"] = ent
tmp = Path(tempfile.mkdtemp()) / "neurodex.json"
prog = nd.Progress(tmp)
known_names = [str(n) for n in tab.names[:500]] + [str(n) for n in tab.names[T // 2:T // 2 + 300]]
for n in known_names:
    prog.mark("adult", n, 1.0, "play", "t")
out["known"] = known_names
out["region_progress"] = [[r, int(a), int(b)] for r, a, b in nd.region_progress(tab, prog)]
lists = {}
for key, kw in {"all": dict(), "found": dict(only_found=True), "region_found": dict(region=tab.region[5], only_found=True),
                "query": dict(query="DNp", only_found=True), "query_unknown": dict(query="DNp"), "region": dict(region=tab.region[100])}.items():
    lists[key] = [int(i) for i in nd.list_entries(tab, prog, **kw)]
out["lists"] = lists
out["region_names"] = tab.region_names()

# --- tracker scenario (integer arithmetic only, so the Rust test builds the same spikes)
chosen = [i for i in range(T) if 3 <= tab.count[i] <= 60][:25]
rows_of = {i: tab.rows(str(tab.names[i])) for i in chosen}
prog2 = nd.Progress(Path(tempfile.mkdtemp()) / "n2.json")
trk = nd.Tracker(tab, prog2)
found = []
for c in range(400):
    spikes = []
    for j, i in enumerate(chosen):
        level = 1
        if j % 4 == 0 and 150 <= c < 260:
            level = 12
        if j % 5 == 1 and 300 <= c < 400:
            level = 8
        for r in rows_of[i]:
            w = (level + (int(r) * 7 + c * 13) % 3) // 3
            spikes.extend([int(r)] * w)
    calm = c < 60
    driven = 150 <= c < 170
    for name in trk.observe(0, np.array(spikes, np.int64), 0.005, calm, driven):
        rec = prog2.types("adult")[name]
        found.append(dict(check=c, name=name, how=rec["how"], x=rec["x"]))
out["tracker"] = dict(chosen=chosen, found=found)
# --- curated facts and the Neuron of the Day
from kickthefly.core import neuron_of_day as notd  # noqa: E402
import datetime as dt  # noqa: E402

facts = nd.facts()
Path("crates/ktf_data/data").mkdir(parents=True, exist_ok=True)
json.dump([dict(id=f.id, text=f.text, cite=f.cite, doi=f.doi, checked=f.checked, types=list(f.types), prefix=list(f.prefix), game_note=f.game_note)
           for f in facts], open("crates/ktf_data/data/neurodex_facts.json", "w"), indent=0, ensure_ascii=False)
out["facts_for"] = {str(tab.names[i]): (nd.fact_for(str(tab.names[i]), "adult").id if nd.fact_for(str(tab.names[i]), "adult") else None) for i in range(0, T, 7)}
cards = []
for d in range(0, 420, 3):
    day = dt.date(2026, 1, 1) + dt.timedelta(days=d)
    for lab in (False, True):
        c = notd.card(day, tab, lab=lab)
        cards.append(dict(day=day.isoformat(), lab=lab, fact_id=c["fact_id"], types=c["types"], plan=c["plan"]))
out["notd"] = cards
out["notd_candidates"] = [f.id for f in notd.candidates(tab)]
json.dump(out, open("tests/golden/neurodex.json", "w"), indent=0)
print(T, "types;", len(found), "discoveries")
