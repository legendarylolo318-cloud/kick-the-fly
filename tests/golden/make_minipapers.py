"""Goldens for fly_sim::minipapers: the paper table (written to crates/fly_sim/data/minipapers.json) and quick runs of the four adult papers
on the real pack (result, comparison table, text, SVG, saved files)."""
import json, sys, dataclasses, copy
sys.path.insert(0, "legacy-python")
from pathlib import Path
from kickthefly.lab import minipapers as mp

papers = []
for pid in mp.ORDER:
    d = dataclasses.asdict(mp.PAPERS[pid])
    d["questions"] = [dict(q, options=[list(o) for o in q["options"]]) for q in d["questions"]]
    d["steps"] = [list(s) for s in d["steps"]]
    papers.append(d)
Path("crates/fly_sim/data/minipapers.json").write_text(json.dumps(dict(
    papers=papers, quick_seeds=list(mp.QUICK_SEEDS), full_seeds=list(mp.FULL_SEEDS)), indent=1, ensure_ascii=False))

out = {}
answers = {"looming_escape": "up", "mb_conditioning": "none", "sugar_feeding": "up", "antenna_grooming_circuit": "up", "adn_grooming_motor": "small"}
d = Path("tests/golden/minipapers_files"); d.mkdir(exist_ok=True)
for pid in ("von_reyn_2014", "tully_quinn_1985", "shiu_2024", "hampel_2015", "ohyama_2015"):
    res = mp.run_paper(pid, mp.QUICK_SEEDS, workers=1)
    res["seconds"] = 0.0
    out[pid] = dict(result=res, compare=mp.compare(res, answers), text=mp.render(res, answers),
                    key_order={q["test"]: list(q["verdict"]["measured"]) for q in res["questions"]})
    sub = d / pid; sub.mkdir(exist_ok=True)
    for p in mp.save(res, sub, answers):
        pass
json.dump(out, open("tests/golden/minipapers.json", "w"), default=str)
print({k: [q["verdict"]["reproduced"] for q in v["result"]["questions"]] for k, v in out.items()})
