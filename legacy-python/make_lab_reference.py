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
    from kickthefly.lab import labday4
    gd = Path(__file__).resolve().parent.parent / "tests" / "golden"

    def day4_prep(screen, net=None, sleep=None, sens=None, cached=False):
        def prep(m):
            prep_for(screen)(m)
            st = labday4._St()
            if net:
                st.net["adult"] = net
            st.cached["adult"] = cached
            st.sleep, st.sens = sleep, sens
            m.host._day4_lab = st
        return prep

    net = json.loads((gd / "netsci.json").read_text())
    sleep = json.loads((gd / "sleepdep.json").read_text())
    sens = json.loads((gd / "sensitivity.json").read_text())
    R.render("lab_netsci", "Graphics", out / "page_lab_netsci_empty.png", prep=day4_prep("lab_netsci"))
    R.render("lab_netsci", "Graphics", out / "page_lab_netsci.png", prep=day4_prep("lab_netsci", net=net, cached=True))
    R.render("lab_sleepdep", "Graphics", out / "page_lab_sleepdep_empty.png", prep=day4_prep("lab_sleepdep"))
    R.render("lab_sleepdep", "Graphics", out / "page_lab_sleepdep.png", prep=day4_prep("lab_sleepdep", sleep=sleep))
    R.render("lab_sensitivity", "Graphics", out / "page_lab_sensitivity_empty.png", prep=day4_prep("lab_sensitivity"))
    R.render("lab_sensitivity", "Graphics", out / "page_lab_sensitivity.png", prep=day4_prep("lab_sensitivity", sens=sens))
    from kickthefly.lab import labwiring
    from kickthefly.sim.wiring import Wiring
    rob = json.loads((gd / "robustness.json").read_text())

    def wiring_prep(tab, min_syn=1, conf=0.5, sweep=None, flips=None, live=None):
        def prep(m):
            prep_for("lab_wiring")(m)
            m.host.wiring = live or Wiring()
            m.host.wiring_busy = ""
            st = labwiring._st(m)
            st.wiring_tab, st.min_syn, st.flip_conf = tab, min_syn, conf
            st.sweep_result, st.flip_result = sweep, flips
        return prep

    R.render("lab_wiring", "Graphics", out / "page_lab_wiring_threshold.png", prep=wiring_prep("threshold", 5))
    R.render("lab_wiring", "Graphics", out / "page_lab_wiring_sweep.png",
             prep=wiring_prep("threshold", 5, sweep=rob["sweep"], live=Wiring(min_synapses=5)))
    R.render("lab_wiring", "Graphics", out / "page_lab_wiring_signflip.png", prep=wiring_prep("signflip", 5, 0.7))
    R.render("lab_wiring", "Graphics", out / "page_lab_wiring_flips.png", prep=wiring_prep("signflip", 5, 0.7, flips=rob["flips"]))
    from kickthefly.lab import labcritical
    crit = json.loads((gd / "critical.json").read_text().replace("NaN", "null"))

    def crit_prep(target, res=None, applied=()):
        def prep(m):
            prep_for("lab_critical")(m)
            st = labcritical._st(m)
            st.cp_target = target
            st.cp_results = {target: res} if res else {}
            m.host.type_ops = {t: -1 for t in applied}
        return prep

    R.render("lab_critical", "Graphics", out / "page_lab_critical_empty.png", prep=crit_prep("looming_escape"))
    R.render("lab_critical", "Graphics", out / "page_lab_critical.png",
             prep=crit_prep("looming_escape", crit["behavior"], applied=("LC4",)))
    from kickthefly.lab import labclamp
    clamp_gold = json.loads((gd / "clamp.json").read_text())

    def clamp_prep(res=None, scenario="looming", mod="lesion_gf"):
        def prep(m):
            prep_for("lab_clamp")(m)
            st = labclamp._st(m)
            st.clamp_scenario, st.clamp_mod, st.clamp_result = scenario, mod, res
        return prep

    R.render("lab_clamp", "Graphics", out / "page_lab_clamp_empty.png", prep=clamp_prep())
    R.render("lab_clamp", "Graphics", out / "page_lab_clamp.png", prep=clamp_prep(clamp_gold["thresh"], "looming", "thresh_10"))
    from kickthefly.lab import labdiff
    diff_gold = json.loads((gd / "diff.json").read_text())

    def diff_prep(res=None, scenario="looming", mod="lesion_loom"):
        def prep(m):
            prep_for("lab_diff")(m)
            st = labdiff._st(m)
            st.diff_scenario, st.diff_mod, st.diff_result = scenario, mod, res
        return prep

    R.render("lab_diff", "Graphics", out / "page_lab_diff_empty.png", prep=diff_prep())
    R.render("lab_diff", "Graphics", out / "page_lab_diff.png", prep=diff_prep(diff_gold["looming"], "looming", "hemi_vis"))
    from kickthefly.lab.laser import LaserState

    def laser_prep(ls=None, equipped=False):
        def prep(m):
            prep_for("lab_laser")(m)
            m.host.laser_state = ls or LaserState()
            if equipped:
                from kickthefly.game.kick_the_fly import TOOLS
                m.host.tool = [i for i, t in enumerate(TOOLS) if t[0] == "laser"][0]
        return prep

    R.render("lab_laser", "Graphics", out / "page_lab_laser.png", prep=laser_prep())
    R.render("lab_laser", "Graphics", out / "page_lab_laser_pulse.png",
             prep=laser_prep(LaserState("MDN", "silence", 1.5, "pulse", 0.3), equipped=True))
    from kickthefly.lab import labpsych
    psych_gold = json.loads((gd / "psych.json").read_text().replace("NaN", "null"))

    def psych_prep(res=None, target="looming", flies=3):
        def prep(m):
            prep_for("lab_psych")(m)
            st = labpsych._st(m)
            st.psych_target, st.psych_flies, st.psych_result = target, flies, res
        return prep

    R.render("lab_psych", "Graphics", out / "page_lab_psych_empty.png", prep=psych_prep())
    R.render("lab_psych", "Graphics", out / "page_lab_psych.png", prep=psych_prep(psych_gold["tmaze"], "tmaze", 2))
    from kickthefly.lab.classroom import ClassroomSession

    def class_prep(lecture, step, status=None):
        def prep(m):
            prep_for("lab_classroom")(m)
            sess = ClassroomSession(lecture_id=lecture, brain=None)
            sess.step_idx, sess.last_action_applied = step, status
            m.host.classroom_session = sess
        return prep

    R.render("lab_classroom", "Graphics", out / "page_lab_classroom.png", prep=class_prep("looming", 0))
    R.render("lab_classroom", "Graphics", out / "page_lab_classroom_step.png",
             prep=class_prep("tmaze", 1, "Stimulus delivered to 30 neurons (odor_a)"))
    R.render("lab_classroom", "Graphics", out / "page_lab_classroom_last.png", prep=class_prep("gf_lesion", 3))
    from kickthefly.lab import labautopsy, failure_autopsy as fa_mod
    aut_gold = json.loads((gd / "autopsy.json").read_text().replace("NaN", "null"))
    fails = fa_mod.failing_tests(published)
    (out / "lab_autopsy_fixture.json").write_text(json.dumps([dict(id=f["id"], name=f["name"]) for f in fails]))

    def aut_prep(sel, done):
        def prep(m):
            prep_for("lab_autopsy")(m)
            m.host.lab_params = {}
            st = labautopsy._st(m)
            st["fails"], st["sel"], st["done"], st["error"] = fails, sel, done, None
        return prep

    R.render("lab_autopsy", "Graphics", out / "page_lab_autopsy_empty.png", prep=aut_prep("mdn_backward", {}))
    R.render("lab_autopsy", "Graphics", out / "page_lab_autopsy.png", prep=aut_prep("grooming_hierarchy", dict(aut_gold)))
    R.render("lab_autopsy", "Graphics", out / "page_lab_autopsy_scrolled.png", prep=aut_prep("mdn_backward", dict(aut_gold)))
    from kickthefly.lab import genetics as gen_mod, labtoolkit
    from kickthefly.core import simcore as sc_mod
    pack_g = sc_mod.pack()[0]

    def gen_prep(query, selected, type_ops=None, note=""):
        def prep(m):
            prep_for("lab_genetics")(m)
            m.host.brain = SimpleNamespace(types=pack_g.type, graph=pack_g)
            m.host.type_ops = dict(type_ops or {})
            st = labtoolkit._st(m)
            st.query, st.selected, st.note = query, selected, note
            st.results = gen_mod.search(query)
        return prep

    R.render("lab_genetics", "Graphics", out / "page_lab_genetics.png", prep=gen_prep("DNp01", "SS00727"))
    R.render("lab_genetics", "Graphics", out / "page_lab_genetics_ops.png",
             prep=gen_prep("LC10", None, {"DNp01": -1}, "line SS00727: silenced"))
    R.render("lab_genetics", "Graphics", out / "page_lab_genetics_empty.png", prep=gen_prep("zzz", None))
    gen_fx = {q: [dict(name=l.name, quality=l.quality, cell_types=list(l.cell_types)) for l in gen_mod.search(q)] for q in ("DNp01", "LC10", "zzz")}
    gen_fx["describe"] = {n: gen_mod.describe(n, pack_g.type) for n in ("SS00727",)}
    gen_fx["coverage"] = gen_mod.coverage(pack_g.type)
    (out / "lab_genetics_fixture.json").write_text(json.dumps(gen_fx))
    from kickthefly.lab import livelab

    def thermo_prep(exprs=(), source="slider", temp=22.0, kinetics="real", note="", eff=0, target="type:DNp01"):
        def prep(m):
            prep_for("lab_thermo")(m)
            tl = livelab.ThermoLive()
            tl.expressions = [dict(effector=e, target=t, strength=1.0) for e, t in exprs]
            tl.source, tl.temperature_c, tl.kinetics = source, temp, kinetics
            m.host.thermo_live = tl
            m.host.flies, m.host.arena_i = [], 0
            m.host.brain = SimpleNamespace(types=pack_g.type, graph=pack_g)
            st = labtoolkit._st(m)
            st.note, st.t_effector, st.t_target = note, eff, target
        return prep

    R.render("lab_thermo", "Graphics", out / "page_lab_thermo.png", prep=thermo_prep())
    R.render("lab_thermo", "Graphics", out / "page_lab_thermo_expr.png",
             prep=thermo_prep([("trpa1", "type:DNp01"), ("shibire", "line:SS00727")], "arena", 30.5, "steady",
                              "added; it applies to every fly on the next frames", 1, "prefix:KC"))
    from kickthefly.lab import sleepdep as sd_mod, thermogenetics as th_mod
    thermo_gold = json.loads((gd / "thermo.json").read_text())
    sd_gold = json.loads((gd / "sleepdep.json").read_text().replace("NaN", "null"))
    sd_flies = [dict(seed=c["seed"], deprive_s=45.0, recover_s=60.0, individuality="off", control=c, deprived=d)
                for c, d in zip(sd_gold["control"], sd_gold["deprived"])]
    ext_res = {
        "thermo_escape": dict(kind="thermo_escape", label="Thermogenetic escape", seeds=[1000, 1001], surgery=None, control=None,
                              treated=thermo_gold["summary"]),
        "sleep_deprivation": dict(kind="sleep_deprivation", label="Sleep deprivation", seeds=[1000, 1001, 1002], surgery=None, control=None,
                                  treated=sd_mod.summarize(sd_flies)),
    }
    (out / "lab_assays_ext_fixture.json").write_text(json.dumps(ext_res))

    def ext_prep(kind):
        def prep(m):
            prep_for("lab_assays")(m)
            st = lab._state(m)
            st.kind, st.flies, st.surgery_i, st.stimulate = kind, 6, 0, False
            st.job, st.result = None, ext_res[kind]
        return prep

    R.render("lab_assays", "Graphics", out / "page_lab_assays_thermo.png", prep=ext_prep("thermo_escape"))
    R.render("lab_assays", "Graphics", out / "page_lab_assays_sleep.png", prep=ext_prep("sleep_deprivation"))
    from kickthefly.lab import pharmacology as ph_mod
    pharm_gold = json.loads((gd / "pharm.json").read_text())

    def pharm_prep(drug="picrotoxin", dose=0.5, low=True, cut=0.7, live=None, dr=None):
        def prep(m):
            prep_for("lab_pharm")(m)
            m.host.wiring = live or Wiring()
            m.host.wiring_busy = ""
            st = labtoolkit._st(m)
            st.d_drug, st.d_dose, st.d_low, st.d_cut, st.d_dr = drug, dose, low, cut, dr
        return prep

    R.render("lab_pharm", "Graphics", out / "page_lab_pharm.png", prep=pharm_prep())
    R.render("lab_pharm", "Graphics", out / "page_lab_pharm_agonist.png",
             prep=pharm_prep("gabaa_agonist", 1.0, False, 0.9, live=ph_mod.wiring_for({"gabaa_agonist": 1.0}, False, 0.9),
                             dr=pharm_gold["dose_response"]))
    (out / "lab_pharm_fixture.json").write_text(json.dumps({k: v for k, v in pharm_gold.items() if ":" in k}))
    from kickthefly.lab import patchclamp as pc_mod
    from kickthefly.sim.connectome.sim import LIFParams as _LP
    import numpy as _np
    p_rows = _np.flatnonzero(_np.asarray(pack_g.type) == "DNp01")
    p_row = int(p_rows[0])
    p_br = SimpleNamespace(types=pack_g.type, superclass=getattr(pack_g, "super_class", getattr(pack_g, "superclass", None)),
                           instance=getattr(pack_g, "instance", []), graph=pack_g, n=len(pack_g.type), remote=False, seed=1000,
                           sim=SimpleNamespace(p=_LP()), step_requests=0)
    p_proto = pc_mod.ClampProtocol([0.1], duration_ms=300, repeats=2, holding=0.0)
    p_rec = pc_mod.run_current_clamp(p_br, p_row, p_proto, "isolated", seed=3, params=_LP())
    p_curve = pc_mod.if_curve(None, p_row, [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5], 500.0, 3, "isolated", 3, _LP())
    p_win = pc_mod.isolated(_np.r_[_np.zeros(200), _np.full(300, 0.2), _np.zeros(100)].astype(_np.float32), _LP(), seed=5)

    class _Ele:
        row, br, n, done, recording = p_row, p_br, 600, False, None

        def window(self):
            return p_win.v, p_win.spikes, p_win.current

        def set_hold(self, a):
            pass

        def detach(self):
            pass

    def patch_prep(chosen=True, rec=False, curve=False, live=False, mode=0):
        def prep(m):
            prep_for("lab_patch")(m)
            m.host.brain = p_br
            m.host.graph = pack_g
            m.host.inspect = None
            st = labtoolkit._st(m)
            st.p_row = p_row if chosen else None
            st.p_type, st.p_index, st.p_mode = "DNp01", 0, mode
            st.p_rec = p_rec if rec else None
            st.p_curve = p_curve if curve else None
            st.p_live = live
            st.p_electrode = _Ele() if live else None
            st.p_last_frame = 0.0
        return prep

    R.render("lab_patch", "Graphics", out / "page_lab_patch_empty.png", prep=patch_prep(False))
    R.render("lab_patch", "Graphics", out / "page_lab_patch_rec.png", prep=patch_prep(True, True, True, False, 1))
    R.render("lab_patch", "Graphics", out / "page_lab_patch_live.png", prep=patch_prep(True, False, False, True, 0))
    f32 = lambda a: [float(x) for x in a]
    (out / "lab_patch_fixture.json").write_text(json.dumps(dict(
        row=p_row, desc=pc_mod.describe_neuron(p_br, p_row),
        rec=dict(v=f32(p_rec.v), spikes=[bool(x) for x in p_rec.spikes], current=f32(p_rec.current)),
        curve={k: v for k, v in p_curve.items() if k not in ("recording", "sweeps")},
        win=dict(v=f32(p_win.v), spikes=[bool(x) for x in p_win.spikes], current=f32(p_win.current)))))
    from kickthefly.lab import imaging as im_mod, livelab as ll_mod

    def imaging_prep(on=False, indicator="gcamp6s", roi_mode="regions", noise=True, error="", msg=""):
        def prep(m):
            prep_for("lab_imaging")(m)
            il = ll_mod.ImagingLive()
            il.indicator, il.shot_noise, il.roi_mode, il.error = indicator, noise, roi_mode, error
            if on:
                rng = _np.random.default_rng(11)
                rois = {"antennal lobe": _np.arange(0, 400), "mushroom body": _np.arange(400, 900), "optic lobe": _np.arange(900, 1600),
                        "central complex": _np.arange(1600, 1800)}
                s = im_mod.ImagingSession(2000, rois, indicator, 20.0, shot_noise=noise, seed=2, max_frames=600)
                for step in range(1200):
                    boost = 4.0 if 400 <= step < 700 else 1.0
                    p = _np.r_[_np.full(900, 0.01 * boost), _np.full(1100, 0.004)]
                    s.push(_np.flatnonzero(rng.random(2000) < p))
                il.session, il.on = s, True
            m.host.imaging_live = il
            labtoolkit._st(m).i_msg = msg
        return prep

    R.render("lab_imaging", "Graphics", out / "page_lab_imaging_off.png", prep=imaging_prep())
    R.render("lab_imaging", "Graphics", out / "page_lab_imaging_on.png", prep=imaging_prep(True, "gcamp6f"))
    R.render("lab_imaging", "Graphics", out / "page_lab_imaging_custom.png",
             prep=imaging_prep(True, "jgcamp8m", "type:DNp01;prefix:KC", False, "", "saved to 20260101-120000-imaging"))
    _rng = _np.random.default_rng(11)
    _rois = {"antennal lobe": _np.arange(0, 400), "mushroom body": _np.arange(400, 900), "optic lobe": _np.arange(900, 1600),
             "central complex": _np.arange(1600, 1800)}
    _fx = {}
    for _k, (_ind, _noise) in {"on": ("gcamp6f", True), "custom": ("jgcamp8m", False)}.items():
        _rng = _np.random.default_rng(11)
        _s = im_mod.ImagingSession(2000, _rois, _ind, 20.0, shot_noise=_noise, seed=2, max_frames=600)
        for _step in range(1200):
            _b = 4.0 if 400 <= _step < 700 else 1.0
            _p = _np.r_[_np.full(900, 0.01 * _b), _np.full(1100, 0.004)]
            _s.push(_np.flatnonzero(_rng.random(2000) < _p))
        _fx[_k] = dict(names=list(_s.roi_names), t=[float(x) for x in _s.t], dff=_np.array(_s.dff)[:, :7].tolist(),
                       true_dff=_np.array(_s.true_dff)[:, :7].tolist())
    (out / "lab_imaging_fixture.json").write_text(json.dumps(_fx))
    from kickthefly.ui import minipaper_ui as mpu
    import kickthefly.lab.minipapers as _mp

    mp_gold = json.loads((gd / "minipapers.json").read_text())
    _mp_ans = {"looming_escape": "up", "mb_conditioning": "none", "sugar_feeding": "up", "antenna_grooming_circuit": "up", "adn_grooming_motor": "small"}

    def mp_prep(pid=None, step=0, answers=None, result=None, full=False):
        def prep(m):
            prep_for("lab_minipapers")(m)
            s = mpu.st(m)
            s.paper, s.full = pid, full
            if pid:
                s.sess.set_lecture(f"paper_{pid}")
                s.sess.goto_step(step)
                s.answers[pid] = dict(answers or {})
                s.committed[pid] = dict(answers or {})
                if result:
                    s.results[pid] = result
        return prep

    R.render("lab_minipapers", "Graphics", out / "page_lab_minipapers_list.png", prep=mp_prep())
    R.render("lab_minipapers", "Graphics", out / "page_lab_minipapers_read.png", prep=mp_prep("von_reyn_2014", 0))
    R.render("lab_minipapers", "Graphics", out / "page_lab_minipapers_hyp.png", prep=mp_prep("hampel_2015", 1, {"antenna_grooming_circuit": "up"}))
    R.render("lab_minipapers", "Graphics", out / "page_lab_minipapers_run.png",
             prep=mp_prep("von_reyn_2014", 2, {"looming_escape": "up"}, mp_gold["von_reyn_2014"]["result"]))
    R.render("lab_minipapers", "Graphics", out / "page_lab_minipapers_plot.png",
             prep=mp_prep("hampel_2015", 3, {k: _mp_ans[k] for k in ("antenna_grooming_circuit", "adn_grooming_motor")}, mp_gold["hampel_2015"]["result"]))
    R.render("lab_minipapers", "Graphics", out / "page_lab_minipapers_cmp.png",
             prep=mp_prep("hampel_2015", 4, {k: _mp_ans[k] for k in ("antenna_grooming_circuit", "adn_grooming_motor")}, mp_gold["hampel_2015"]["result"]))
    R.render("lab_minipapers", "Graphics", out / "page_lab_minipapers_cmp_larva.png",
             prep=mp_prep("ohyama_2015", 4, {"larva_noci_to_goro_rolling": "up", "larva_chordotonal_to_basin": "same"}, mp_gold["ohyama_2015"]["result"]))
    from kickthefly.lab import labscreen as ls_mod
    scr_gold = json.loads((gd / "screens.json").read_text().replace("NaN", "null"))

    def scr_prep(kind, res, sel=None, sort=None, desc=True, search="", ops=None):
        def prep(m):
            prep_for("lab_activation" if kind == "activation" else "lab_knockout")(m)
            st_ = ls_mod._st(m, kind)
            st_.update(res=res, loaded=True, sel=sel, sort=sort, desc=desc, search=search)
            m.host.type_ops = dict(ops or {})
        return prep

    R.render("lab_activation", "Graphics", out / "page_lab_activation.png", prep=scr_prep("activation", scr_gold["act"]))
    R.render("lab_activation", "Graphics", out / "page_lab_activation_sel.png",
             prep=scr_prep("activation", scr_gold["act"], "DNp09", "type", False, "", {"DNp09": 1}))
    R.render("lab_activation", "Graphics", out / "page_lab_activation_empty.png", prep=scr_prep("activation", None))
    _ko_rows = scr_gold["ko"]["rows"]
    _ko_sel = next((f"{r['behavior']}|{r['candidate']}" for r in _ko_rows if r["is_batch"]), None)
    R.render("lab_knockout", "Graphics", out / "page_lab_knockout.png", prep=scr_prep("knockout", scr_gold["ko"]))
    R.render("lab_knockout", "Graphics", out / "page_lab_knockout_sel.png", prep=scr_prep("knockout", scr_gold["ko"], _ko_sel, "drop_share", True))
    from kickthefly.lab import labrigs as lr_mod, rigassay as ra_mod

    rig_opts = {"tethered": dict(mode="closed", omega=2.0, seconds=9.0, gain=1.0), "ball": dict(scene="bar", seconds=10.0, bar_deg=-90.0),
                "buridan": dict(seconds=15.0), "fourfield": dict(seconds=15.0, mode="odor")}
    rig_res = {r: ra_mod.scene_run(r, 1000, "off", **o) for r, o in rig_opts.items()}
    (out / "lab_rigs_fixture.json").write_text(json.dumps(rig_res, default=str))

    def rig_prep(rig=None, t=0.0, with_res=True):
        def prep(m):
            prep_for(lr_mod.PAGES[rig] if rig else "lab_rigs")(m)
            s = lr_mod.st(m)
            if with_res:
                s.res = {k: v for k, v in rig_res.items()}
            s.playing = False
            s.t = t
            if rig:
                s.p[rig].update(rig_opts[rig] if with_res else {})
        return prep

    R.render("lab_rigs", "Graphics", out / "page_lab_rigs_hub.png", prep=rig_prep(None, 0.0, True))
    R.render("lab_rig_tethered", "Graphics", out / "page_lab_rig_tethered.png", prep=rig_prep("tethered", 5.0))
    R.render("lab_rig_ball", "Graphics", out / "page_lab_rig_ball.png", prep=rig_prep("ball", 6.0))
    R.render("lab_rig_buridan", "Graphics", out / "page_lab_rig_buridan.png", prep=rig_prep("buridan", 12.0))
    R.render("lab_rig_fourfield", "Graphics", out / "page_lab_rig_fourfield.png", prep=rig_prep("fourfield", 12.0))
    R.render("lab_rig_buridan", "Graphics", out / "page_lab_rig_empty.png", prep=rig_prep("buridan", 0.0, False))
    def export_prep(last=None, insp=None, bundle=None):
        def prep(m):
            prep_for("lab_export")(m)
            m.host.recording = None
            m.host.inspect = insp
            m.host.last_export = last
            if bundle:
                lab._state(m).bundle_msg = bundle
        return prep

    R.render("lab_export", "Graphics", out / "page_lab_export.png", prep=export_prep())
    R.render("lab_export", "Graphics", out / "page_lab_export_last.png",
             prep=export_prep("/home/user/.local/share/kickthefly/exports/20260101-120000-recording", dict(type="DNp01", i=3),
                              ("Bundled the last live recording (a record: it can't be rerun, and says why).", "/home/user/exports/bundles/x.zip")))
    from kickthefly.lab import nwbexport
    (out / "lab_export_fixture.json").write_text(json.dumps(dict(nwb=nwbexport.available())))
    R.render("lab_protocols", "Graphics", out / "page_lab_protocols.png", prep=prep_for("lab_protocols"))
    R.render("lab_protocols", "Graphics", out / "page_lab_protocols_scrolled.png", prep=prep_for("lab_protocols", None, 600))
    from kickthefly.lab import labjobs as lj_mod, assays as as_mod, predators as pd_mod

    _of = [as_mod.orchard_fly(s_, duration_s=20.0) for s_ in (1000, 1001)]
    _pf = [pd_mod.escape_fly(s_, kinds=("frog", "mantis"), trials=1) for s_ in (1000, 1001)]
    ext2 = {
        "orchard": dict(kind="orchard", label="Orchard feeding", seeds=[1000, 1001], surgery=None, control=None, treated=lj_mod.summarize("orchard", _of)),
        "predator_escape": dict(kind="predator_escape", label="Predator escape", seeds=[1000, 1001], surgery=None, control=None,
                                treated=lj_mod.summarize("predator_escape", _pf)),
    }
    (out / "lab_assays_ext2_fixture.json").write_text(json.dumps(ext2, default=float))

    def ext2_prep(kind):
        def prep(m):
            prep_for("lab_assays")(m)
            st = lab._state(m)
            st.kind, st.flies, st.surgery_i, st.stimulate = kind, 6, 0, False
            st.job, st.result = None, ext2[kind]
        return prep

    R.render("lab_assays", "Graphics", out / "page_lab_assays_orchard.png", prep=ext2_prep("orchard"))
    R.render("lab_assays", "Graphics", out / "page_lab_assays_predator.png", prep=ext2_prep("predator_escape"))
    from kickthefly.core import mic as micmod
    from kickthefly.game import live_inputs as li

    micmod.availability = lambda: (True, "an input device exists (test)")
    _bins = tuple(min(1.0, 0.04 + 0.6 * max(0.0, 1 - abs(i - 7) / 4)) for i in range(24))

    def live_prep(mic_on=False, stream_on=False):
        def prep(m):
            prep_for("lab_params")(m)
            m.pages["live_inputs"] = li.page
            live = SimpleNamespace(mic_on=mic_on, mic_error="", stream_on=stream_on, stream_error="", channel="mychannel",
                                   chat=SimpleNamespace(state="live", describe=lambda: "irc.chat.twitch.tv:6697 #mychannel (read-only, anonymous)",
                                                         commands_read=12) if stream_on else None,
                                   reading=micmod.Reading(peak_hz=83.0, drive_a=0.12, drive_b=0.4, bins=_bins) if mic_on else micmod.Reading(),
                                   jo_firing=(3.0, 11.0, 7, 40) if mic_on else (0.0, 0.0, 0, 0),
                                   mic=SimpleNamespace(set_sensitivity=lambda v: None),
                                   lab_locked=lambda: False, set_mic=lambda v: None, set_stream=lambda v, c=None: None)
            m.host.live = live
        return prep

    R.render("live_inputs", "Graphics", out / "page_live_inputs.png", prep=live_prep())
    R.render("live_inputs", "Graphics", out / "page_live_inputs_on.png", prep=live_prep(True, True))
    print("wrote", out)
