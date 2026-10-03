"""Lab screen for Neural Clamp: isolating wiring effects from input variation via dynamic clamping.

DYNAMIC CLAMP NOTICE:
Forced spikes override the network's own state and break feedback loops such as proprioception.
This is dynamic clamping, not an autonomous run.
"""
from __future__ import annotations

import threading
import time

import pygame

from kickthefly.ui import menu as ui

SCENARIOS = (
    ("looming", "Looming escape"),
    ("sugar", "Sugar feeding"),
    ("antenna", "Antennal touch"),
)

MODS = (
    ("lesion_loom", "Silence looming detectors (LPLC2, LC4)"),
    ("lesion_gf", "Silence giant fiber (DNp01)"),
    ("thresh_10", "Threshold: drop <10 synapses"),
    ("flips", "Flip uncertain neurotransmitters (50% @ 0.5)"),
    ("inhibition_0", "Picrotoxin: 100% inhibition block"),
)


def _st(m):
    from kickthefly.lab import lab

    st = lab._state(m)
    if not hasattr(st, "clamp_scenario"):
        st.clamp_scenario = "looming"
        st.clamp_mod = "lesion_gf"
        st.clamp_job = None
        st.clamp_result = None
    return st


def page(m: ui.Menu, surf, rect, mouse) -> None:
    st, host = _st(m), m.host
    m.text(surf, "NEURAL CLAMP (DYNAMIC CLAMPING)", (rect.x + 24, rect.y + 16), ui.INK, m.f_head)
    y = m.subtitle(surf, rect, "Record a run's exact spike train and replay it into a modified connectome to isolate wiring effects "
                               "from behavioral input variation.")

    # Dynamic clamp prominent notice (3.0 release review: one fixed-height line that ran past the panel; now wrapped and measured)
    note = ("DYNAMIC CLAMP: Forced spikes override the network's own state and break feedback loops (such as "
            "proprioception). This is dynamic clamping, not an autonomous run.")
    n = len(ui.Menu.fit_lines(m.f_small, note, rect.w - 68, 4)[0])
    banner = pygame.Rect(rect.x + 24, y, rect.w - 48, n * m.f_small.get_linesize() + 10)
    pygame.draw.rect(surf, (40, 24, 20), banner, border_radius=6)
    pygame.draw.rect(surf, (220, 110, 60), banner, 1, border_radius=6)
    m.wrapped(surf, note, (banner.x + 10, banner.y + 5), banner.w - 20, (255, 180, 120), m.f_small, max_lines=4)

    y = banner.bottom + 10
    right = rect.right - 24
    y = m.flow_buttons(surf, rect.x + 24, y, right, [(label, (lambda k=key: setattr(st, "clamp_scenario", k)), ("c_scen", key),
                                                      st.clamp_scenario == key, None) for key, label in SCENARIOS], label="Reference scenario:") + 8
    y = m.flow_buttons(surf, rect.x + 24, y, right, [(label, (lambda k=key: setattr(st, "clamp_mod", k)), ("c_mod", key),
                                                      st.clamp_mod == key, None) for key, label in MODS], label="Connectome modification:") - 30
    y += 44
    job = st.clamp_job
    if job is not None and not job["thread"].is_alive():
        if job.get("result"):
            st.clamp_result = job["result"]
        if job.get("error"):
            m.text(surf, f"error: {job['error']}", (rect.x + 24, y + 10), ui.BAD, m.f_small)
        st.clamp_job = job = None

    if job is not None:
        frac = job["done"] / max(1, job["total"])
        pygame.draw.rect(surf, (30, 36, 48), (rect.x + 24, y + 10, rect.w - 300, 12), border_radius=6)
        pygame.draw.rect(surf, ui.AMBER, (rect.x + 24, y + 10, max(8, int((rect.w - 300) * frac)), 12),
                         border_radius=6)
        m.text(surf, f"{job['label']} …", (rect.x + 24, y + 26), ui.LABEL, m.f_small)
    else:
        m.button(surf, (rect.x + 24, y, 280, 38), "Run Neural Clamp", lambda: _start(m, st),
                 id="run_clamp", style="primary",
                 tip="Records reference run with the chosen stimulus, then clamps those spikes into the modified "
                     "connectome to directly observe downstream divergence.")

    res = st.clamp_result
    body = pygame.Rect(rect.x + 24, y + 48, rect.w - 48, rect.bottom - y - 48 - 66)
    if res:
        _draw_results(m, surf, body, res)
    else:
        m.text(surf, "Click 'Run Neural Clamp' to compare free-running vs clamped activity side-by-side." if job is None else "",
               (body.x, body.y + 12), ui.LABEL, m.f_small)

    if res:
        m.button(surf, (rect.x + 24, rect.bottom - 58, 200, 42), "Export CSV + JSON",
                 lambda: _export(m, res), id="clamp_export")
    m.button(surf, (rect.right - 164, rect.bottom - 58, 140, 42), "Back", m.back, style="primary", id=("clamp", "back"))


def _start(m, st) -> None:
    from kickthefly.lab import clamp
    from kickthefly.core import simcore
    from kickthefly.sim.wiring import Wiring, random_flip

    job = dict(done=0, total=2, label="recording reference run", result=None, error=None)

    def work():
        try:
            # 1. Record reference
            job["done"] = 1
            job["label"] = "recording reference free-running spike train"
            ref = clamp.record_reference_run(stimulus=st.clamp_scenario, steps=200, seed=1000)

            # 2. Configure perturbation
            g = simcore.pack()[0]
            types = g.type.astype(str)
            wiring = Wiring()
            lesion_rows = None

            if st.clamp_mod == "lesion_loom":
                lesion_rows = np.flatnonzero(np.isin(types, ("LPLC2", "LC4")))
            elif st.clamp_mod == "lesion_gf":
                lesion_rows = np.flatnonzero(types == "DNp01")
            elif st.clamp_mod == "thresh_10":
                wiring = Wiring(min_synapses=10)
            elif st.clamp_mod == "flips":
                wiring = Wiring(flip_rows=random_flip(g, 0.5, 0.5, seed=1))
            elif st.clamp_mod == "inhibition_0":
                wiring = Wiring(inhibition_scale=0.0)

            job["done"] = 2
            job["label"] = "replaying into modified connectome under dynamic clamp"
            res = clamp.run_neural_clamp(ref, wiring=wiring, lesion_rows=lesion_rows, seed=1000)
            job["result"] = res
        except Exception as e:
            job["error"] = f"{type(e).__name__}: {e}"

    job["thread"] = threading.Thread(target=work, name="neural_clamp", daemon=True)
    job["thread"].start()
    st.clamp_job = job


def _draw_results(m, surf, area, res: dict) -> None:
    m.text(surf, f"Side-by-side Diff: Free-Running vs Clamped  ·  Stimulus: {res['stimulus']}  ·  "
                 f"{res['steps']} steps ({res['steps']*0.005:.1f} s)  ·  clamped {res['clamped_neurons']:,} neurons",
           (area.x, area.y), ui.INK, m.f_small)

    y = area.y + 24
    # Readouts summary card
    head = ["Key Readout", "Free-running (Hz)", "Clamped (Hz)", "Wiring Effect Δ (Hz)"]
    xs = [area.x, area.x + 220, area.x + 380, area.x + 540]
    for hx, label in zip(xs, head):
        m.text(surf, label, (hx, y), ui.LABEL, m.f_small)
    y += 18

    readouts = res.get("readouts", {})
    for r_name, vals in readouts.items():
        diff = vals["diff_hz"]
        col = ui.BAD if diff < -1.0 else ui.AMBER if diff > 1.0 else ui.GOOD
        m.text(surf, r_name.replace("_", " "), (xs[0], y), ui.TEXT, m.f_small)
        m.text(surf, f"{vals['free_hz']:.1f}", (xs[1], y), ui.LABEL, m.f_small)
        m.text(surf, f"{vals['clamped_hz']:.1f}", (xs[2], y), ui.INK, m.f_small)
        m.text(surf, f"{diff:+.1f}", (xs[3], y), col, m.f_small)
        y += 18

    y += 12
    m.text(surf, "Neuropil Region Rates (spikes/s per neuron):", (area.x, y), ui.INK, m.f_small)
    y += 18
    head_reg = ["Neuropil Region", "Free-running (Hz)", "Clamped (Hz)", "Diff Δ (Hz)"]
    for hx, label in zip(xs, head_reg):
        m.text(surf, label, (hx, y), ui.LABEL, m.f_small)
    y += 18

    f_rates = res.get("free_region_rates", {})
    c_rates = res.get("clamped_region_rates", {})
    d_rates = res.get("diff_region_rates", {})

    top_regs = sorted(d_rates.keys(), key=lambda k: abs(d_rates[k]), reverse=True)[:8]
    for reg in top_regs:
        diff = d_rates.get(reg, 0.0)
        col = ui.BAD if diff < -1.0 else ui.AMBER if diff > 1.0 else ui.GOOD
        m.text(surf, reg[:26], (xs[0], y), ui.TEXT, m.f_small)
        m.text(surf, f"{f_rates.get(reg, 0.0):.1f}", (xs[1], y), ui.LABEL, m.f_small)
        m.text(surf, f"{c_rates.get(reg, 0.0):.1f}", (xs[2], y), ui.INK, m.f_small)
        m.text(surf, f"{diff:+.1f}", (xs[3], y), col, m.f_small)
        y += 18


def _export(m, res: dict) -> None:
    from kickthefly.lab import clamp, recorder

    folder = recorder.exports_dir() / f"{time.strftime('%Y%m%d-%H%M%S')}-neural-clamp"
    clamp.save(res, folder)
    m.host.last_export = str(folder)
    m.flash(f"saved to {folder.name} in exports", ui.GOOD)
