#!/usr/bin/env python3
"""Screenshots of the pygame Lab pages (lab/lab.py) for the Rust port: python make_lab_reference.py [OUT_DIR].
Writes lab_fixture.json (the page list the game answers `lab_pages` with) and page_lab_*.png."""
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_ui_reference as R  # noqa: E402
from kickthefly.game import kick_the_fly as k  # noqa: E402
from kickthefly.lab import lab  # noqa: E402

pages = k.Game.lab_pages(SimpleNamespace())


def prep_for(screen, params=None, scroll=0):
    def prep(m):
        lab.install(m)
        m.host.lab_pages = lambda: pages
        m.host.lab_params = dict(params or {})
        m.host.wiring = None
        m.host.brain = None
        m.scroll[screen] = scroll
    return prep


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "tests" / "ui_reference")
    (out / "lab_fixture.json").write_text(json.dumps(dict(pages=pages, params=dict(bias=0.25, **{"thresh.escape": 5.5, "loom_min": 2.0}))))
    R.INSTALL = []
    mod = dict(bias=0.25, **{"thresh.escape": 5.5, "loom_min": 2.0})
    R.render("lab", "Graphics", out / "page_lab_hub.png", prep=prep_for("lab"))
    R.render("lab", "Graphics", out / "page_lab_hub_modified.png", prep=prep_for("lab", mod))
    R.render("lab_params", "Graphics", out / "page_lab_params.png", prep=prep_for("lab_params", mod))
    R.render("lab_params", "Graphics", out / "page_lab_params_scrolled.png", prep=prep_for("lab_params", mod, 600))
    R.render("lab_assumptions", "Graphics", out / "page_lab_assumptions.png", prep=prep_for("lab_assumptions"))
    R.render("lab_assumptions", "Graphics", out / "page_lab_assumptions_scrolled.png", prep=prep_for("lab_assumptions", None, 700))
    from kickthefly.lab import validation as val_mod
    published = json.loads((Path(__file__).resolve().parent.parent / "docs" / "results" / "3.0-release" / "validate_after.json").read_text())
    val_mod.load_results = lambda: (published, "from this build")
    R.render("lab_validation", "Graphics", out / "page_lab_validation.png", prep=prep_for("lab_validation"))
    R.render("lab_validation", "Graphics", out / "page_lab_validation_scrolled.png", prep=prep_for("lab_validation", None, 700))
    gold = json.loads((Path(__file__).resolve().parent.parent / "tests" / "golden" / "assay_jobs.json").read_text())

    def assay_prep(kind, name, surgery_i=0, flies=6, stimulate=False):
        def prep(m):
            prep_for("lab_assays")(m)
            st = lab._state(m)
            st.kind, st.flies, st.surgery_i, st.stimulate = kind, flies, surgery_i, stimulate
            st.job, st.result = None, gold[name] if name else None
        return prep

    R.render("lab_assays", "Graphics", out / "page_lab_assays_empty.png", prep=assay_prep("tmaze", None))
    R.render("lab_assays", "Graphics", out / "page_lab_assays_tmaze.png", prep=assay_prep("tmaze", "tmaze"))
    R.render("lab_assays", "Graphics", out / "page_lab_assays_looming_surgery.png", prep=assay_prep("looming", "looming_surgery", surgery_i=7))
    R.render("lab_assays", "Graphics", out / "page_lab_assays_sugar.png", prep=assay_prep("sugar", "sugar", flies=2))
    from kickthefly.lab import labjobs
    opts = lab.surgery_options()
    (out / "lab_assays_fixture.json").write_text(json.dumps(dict(
        kinds=[[k, labjobs.ASSAY_LABEL[k]] for k in labjobs.ASSAYS], options=[[lbl, list(sp) if sp else []] for lbl, sp in opts],
        results={k: gold[k] for k in ("tmaze", "looming_surgery", "sugar")}, surgery_label_7=opts[7][0])))
    R.render("lab_asymmetry", "Graphics", out / "page_lab_asymmetry.png", prep=prep_for("lab_asymmetry"))
    R.render("lab_asymmetry", "Graphics", out / "page_lab_asymmetry_mirror.png", prep=prep_for("lab_asymmetry"),
             mutate=lambda c: c.set("brain.mirror_weights", True))
    bench = {"timestamp": "2026-10-10 12:00:00", "system": {"os": "Linux-test", "cpu_count": 16, "python": "3.12.3", "cpu_model": "Test CPU @ 3.0GHz"},
             "backend": "CPU (NumPy)", "device": "CPU", "connectome": {"neurons": 166700, "synapses": 10272125}, "seconds_per_run": 2.5,
             "records": [dict(flies=1, paced_steps_per_s=200.0, paced_min_steps_per_s=200.0, paced_realtime_ratio=1.0, uncapped_steps_per_s=412.5,
                              uncapped_latency_ms=2.424, uncapped_realtime_ratio=2.06, agg_uncapped_steps_per_s=412.5, neurons_per_sec=68770000.0,
                              active_fraction=0.0042, synapses_per_sec=1.2e8, memory_mb=1500.2),
                         dict(flies=8, paced_steps_per_s=171.3, paced_min_steps_per_s=160.0, paced_realtime_ratio=0.857, uncapped_steps_per_s=60.1,
                              uncapped_latency_ms=16.639, uncapped_realtime_ratio=0.3, agg_uncapped_steps_per_s=480.8, neurons_per_sec=80150000.0,
                              active_fraction=0.0042, synapses_per_sec=1.3e8, memory_mb=3100.7),
                         dict(flies=16, paced_steps_per_s=80.0, paced_min_steps_per_s=70.0, paced_realtime_ratio=0.4, uncapped_steps_per_s=30.0,
                              uncapped_latency_ms=33.333, uncapped_realtime_ratio=0.15, agg_uncapped_steps_per_s=480.0, neurons_per_sec=80000000.0,
                              active_fraction=0.0042, synapses_per_sec=1.3e8, memory_mb=5000.0)]}
    (out / "lab_benchmark_fixture.json").write_text(json.dumps(bench))

    def bench_prep(res):
        def prep(m):
            prep_for("lab_benchmark")(m)
            m.host._benchmark_results = res
            from kickthefly.lab import benchmark as bm
            bm.load_benchmark_results = lambda *a, **k: res
        return prep

    R.render("lab_benchmark", "Graphics", out / "page_lab_benchmark.png", prep=bench_prep(bench))
    print("wrote", out)
