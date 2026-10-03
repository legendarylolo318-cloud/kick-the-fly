"""Lab screens that change the connectome itself rather than driving neurons.

  Synapse threshold    drop every connection reconstructed with fewer than N synapses, live, and report the
                       threshold at which each validated behavior breaks
  Sign flips           flip the sign of neurons whose neurotransmitter prediction the dataset is least sure of
  Inhibition block     scale every predicted-inhibitory synapse, 0-100%

The numbers each one acts on (synapse counts, neurotransmitter predictions and their confidence) are the dataset's.
The sliders, the trial counts and how a severity maps onto them are this game's choices, and both are said so on
screen.
"""
from __future__ import annotations

import threading

import pygame

from kickthefly.ui import menu as ui

# (key, label, draw function). Each feature adds its own tab here.
TABS: list[tuple[str, str, object]] = []


def tab(key: str, label: str):
    def wrap(fn):
        TABS.append((key, label, fn))
        return fn
    return wrap


def _st(m):
    from kickthefly.lab import lab

    st = lab._state(m)
    if not hasattr(st, "wiring_tab"):
        st.wiring_tab = "threshold"
        st.min_syn = 1
        st.sweep_job = None
        st.sweep_result = None
        st.flip_conf = 0.5
        st.flip_job = None
        st.flip_result = None
        st.flip_trials = 10
        st.flip_seeds = 10
        st.inhibition = 100
        st.inhibition_severity = 100
        st.inhibition_before = None
        st.inhibition_job = None
        st.inhibition_result = None
    return st


def page(m: ui.Menu, surf, rect, mouse) -> None:
    st, host = _st(m), m.host
    m.text(surf, "CONNECTOME ROBUSTNESS", (rect.x + 24, rect.y + 16), ui.INK, m.f_head)
    ty = max(rect.y + 72, m.subtitle(surf, rect, "Change the wiring itself and re-run the behaviors. Everything here acts on the dataset's own "
                                                 "synapse counts and neurotransmitter predictions.") - 4)
    x = rect.x + 24
    for key, label, _ in TABS:
        r = pygame.Rect(x, ty, m.bw(label, 200), 34)
        m.button(surf, r, label, (lambda k=key: setattr(st, "wiring_tab", k)), id=("wtab", key),
                 active=st.wiring_tab == key, style="normal")
        x += r.w + 10
    body = pygame.Rect(rect.x + 16, ty + 44, rect.w - 32, rect.bottom - 70 - (ty + 44))
    busy = getattr(host, "wiring_busy", "")
    if busy:
        m.text(surf, f"applying to every fly: {busy} …", (rect.right - 24, rect.y + 84), ui.AMBER, m.f_small,
               "topright")
    draw = next((fn for key, _, fn in TABS if key == st.wiring_tab), TABS[0][2] if TABS else None)
    if draw is not None:
        draw(m, surf, body, st, host)
    w = getattr(host, "wiring", None)
    if w is not None and not w.is_identity:
        m.text(surf, f"live: {w.label()}", (rect.x + 24, rect.bottom - 48), ui.AMBER, m.f_small)
        m.button(surf, (rect.x + 24, rect.bottom - 34, m.bw("Restore the connectome", 200), 28), "Restore the connectome",
                 lambda: _restore(host), id="wire_reset", style="danger")
    m.button(surf, (rect.right - 164, rect.bottom - 58, 140, 42), "Back", m.back, style="primary", id=("wire", "back"))


def _restore(host) -> None:
    from kickthefly.sim.wiring import Wiring

    host.set_wiring(Wiring())


# --- synapse threshold ------------------------------------------------------------------------------------------
def _intro(m, surf, body, y: int, lead: str, more: str, chip: str, chip_text: str) -> int:
    """A tab's two description paragraphs and its tag line, wrapped and measured (3.0 release review: they were single lines at fixed
    offsets that ran off a 860 px menu and overlapped each other at larger text). Returns the y below."""
    w = body.w - 16
    y = m.wrapped(surf, lead, (body.x + 8, y), w, ui.TEXT, m.f_small, max_lines=3)
    y = m.wrapped(surf, more, (body.x + 8, y + 2), w, ui.LABEL, m.f_small, max_lines=3) + 6
    r = m.chip(surf, (body.x + 8, y), chip)
    y = m.wrapped(surf, chip_text, (r.right + 10, y + 1), body.right - 8 - r.right - 10, ui.LABEL, m.f_small, max_lines=2)
    return max(y, r.bottom) + 10


@tab("threshold", "Synapse threshold")
def _tab_threshold(m, surf, body, st, host) -> None:
    from kickthefly.lab import robustness
    from kickthefly.sim import wiring as wiring_mod
    from kickthefly.sim.wiring import Wiring

    y = body.y
    y = _intro(m, surf, body, y, "Drop every connection the reconstruction found fewer than N synapses for, then watch the fly and "
                                 "re-run the behaviors.",
               f"Weak contacts are the ones most likely to be reconstruction noise. The brain pack already drops "
               f"anything below {robustness.MIN_PACK_SYNAPSES} synapses, so 1-{robustness.MIN_PACK_SYNAPSES} change nothing.",
               "CONNECTOME", "the synapse counts are the dataset's; which thresholds to try is a game choice")
    m.text(surf, "Minimum synapses", (body.x + 8, y + 15), ui.TEXT, m.f_text, "midleft")
    m.slider(surf, (body.x + 200, y, body.w - 420, 30), st.min_syn, 1, 15, 1, "{:.0f}",
             lambda v: setattr(st, "min_syn", int(v)), lambda: None, id="min_syn",
             tip="Connections with fewer synapses than this are set to zero weight. The neurons stay; only the weak "
                 "wiring between them goes.")
    y += 44
    try:
        stats = wiring_mod.threshold_stats(int(st.min_syn))
    except FileNotFoundError:
        m.text(surf, "needs the brain pack", (body.x + 8, y), ui.BAD, m.f_small)
        return
    m.text(surf, f"At >= {stats['threshold']} synapses: {stats['connections_dropped']:,} of "
                 f"{stats['connections']:,} connections dropped ({stats['connections_dropped_share']:.1%}), "
                 f"{stats['synapses_dropped']:,} of {stats['synapses']:,} synapses",
           (body.x + 8, y), ui.INK, m.f_text)
    m.text(surf, f"{stats['neurons_touched']:,} of {stats['neurons']:,} neurons lose at least one connection; "
                 f"{stats['neurons_cut_off']:,} lose every input, {stats['neurons_silenced_output']:,} lose every "
                 f"output", (body.x + 8, y + 24), ui.TEXT, m.f_small)
    y += 56
    live = getattr(host, "wiring", None)
    applied = live is not None and live.min_synapses == int(st.min_syn)
    m.button(surf, (body.x + 8, y, 260, 40), "Apply to every fly" if not applied else "Applied",
             lambda: host.set_wiring(Wiring(min_synapses=int(st.min_syn), flip_rows=live.flip_rows if live else (),
                                            inhibition_scale=live.inhibition_scale if live else 1.0)),
             id="apply_thresh", style="primary", enabled=not applied and not getattr(host, "wiring_busy", ""),
             tip="Rebuilds every live fly's synapse matrix. Takes about a second per fly.")
    job, res = st.sweep_job, st.sweep_result
    if job is not None and not job["thread"].is_alive():
        st.sweep_result = res = job.get("result") or res
        if job.get("error"):
            m.text(surf, f"error: {job['error']}", (body.x + 290, y + 20), ui.BAD, m.f_small, "midleft")
        st.sweep_job = job = None
    if job is not None:
        frac = job["done"] / max(1, job["total"])
        pygame.draw.rect(surf, (30, 36, 48), (body.x + 290, y + 14, body.w - 320, 12), border_radius=6)
        pygame.draw.rect(surf, ui.AMBER, (body.x + 290, y + 14, max(8, int((body.w - 320) * frac)), 12),
                         border_radius=6)
        m.text(surf, job["label"], (body.x + 290, y + 30), ui.LABEL, m.f_small)
    else:
        m.button(surf, (body.x + 290, y, 300, 40), "Run the threshold report", lambda: _start_sweep(m, st),
                 id="run_sweep", tip="Re-runs each validated behavior at thresholds 1, 4, 5, 6, 8 and 10 on the "
                                     "validation seeds, with the validation suite's own pass criteria, and reports "
                                     "the threshold at which each one breaks. Takes several minutes.")
    y += 52
    if res:
        _draw_sweep(m, surf, pygame.Rect(body.x + 8, y, body.w - 16, body.bottom - y), st, res)


def _start_sweep(m, st) -> None:
    from kickthefly.lab import labjobs, robustness

    job = dict(done=0, total=1, label="starting", result=None, error=None)

    def work():
        try:
            job["result"] = robustness.threshold_sweep(
                workers=labjobs.default_workers(),
                progress=lambda d, n, label: job.update(done=d, total=n, label=label))
        except Exception as e:
            job["error"] = f"{type(e).__name__}: {e}"

    job["thread"] = threading.Thread(target=work, name="threshold-sweep", daemon=True)
    job["thread"].start()
    st.sweep_job = job


def _draw_sweep(m, surf, area, st, res) -> None:
    m.text(surf, f"Threshold report  ·  seeds {res['seeds'][0]}-{res['seeds'][-1]}  ·  {res['seconds']:.0f}s  ·  "
                 f"validation's own pass criteria", (area.x, area.y), ui.INK, m.f_small)
    y = area.y + 22
    head = ["behavior"] + [f">={t}" for t in res["thresholds"]] + ["breaks at"]
    xs = [area.x] + [area.x + 300 + i * 56 for i in range(len(res["thresholds"]))] + [area.right - 120]
    for hx, label in zip(xs, head):
        m.text(surf, label, (hx, y), ui.LABEL, m.f_small)
    y += 20
    for test_id, b in res["breaks"].items():
        m.text(surf, b["name"][:44], (area.x, y), ui.TEXT, m.f_small)
        for i, t in enumerate(res["thresholds"]):
            step = next((s for s in res["steps"] if s["threshold"] == t), None)
            ok = step and step["behaviors"].get(test_id, {}).get("passed")
            m.text(surf, "pass" if ok else "fail", (xs[i + 1], y), ui.GOOD if ok else ui.BAD, m.f_small)
        m.text(surf, f">= {b['breaks_at']}" if b["breaks_at"] else "holds", (xs[-1], y),
               ui.BAD if b["breaks_at"] else ui.GOOD, m.f_small)
        y += 20
    y += 6
    y = m.wrapped(surf, "A behavior that breaks at a low threshold depends on connections the reconstruction saw only a few "
                 "synapses of.", (area.x, y), area.w, ui.LABEL, m.f_small, max_lines=3)
    m.button(surf, (area.x, min(y + 4, area.bottom - 34), 200, 30), "Export CSV + JSON",
             lambda: _export(m, res), id="sweep_export")


# --- sign flips -------------------------------------------------------------------------------------------------
@tab("signflip", "Sign flips")
def _tab_signflip(m, surf, body, st, host) -> None:
    from kickthefly.core import simcore
    from kickthefly.sim import wiring as wiring_mod
    from kickthefly.sim.wiring import Wiring

    y = body.y
    y = _intro(m, surf, body, y, "MaleCNS v1.0 predicts every neuron's neurotransmitter and says how sure it is. This flips the "
                                 "least certain ones from excitatory to inhibitory, or back.",
               "A neuron with the wrong sign in the dataset has the wrong sign here too. The stress test asks "
               "which validated behaviors survive that being true.",
               "CONNECTOME", "the predictions and confidences are the dataset's; the cutoff, the share and the trial count are game choices")
    try:
        g = simcore.pack()[0]
        stats = wiring_mod.confidence_stats(g, st.flip_conf)
    except Exception as e:
        m.text(surf, f"needs a brain pack with transmitter predictions: {e}", (body.x + 8, y), ui.BAD, m.f_small)
        return
    m.text(surf, "Confidence below", (body.x + 8, y + 15), ui.TEXT, m.f_text, "midleft")
    m.slider(surf, (body.x + 200, y, 300, 30), st.flip_conf, 0.3, 1.0, 0.05, "{:.2f}",
             lambda v: setattr(st, "flip_conf", float(v)), lambda: None, id="flip_conf",
             tip="Neurons the dataset is less sure than this about are the ones a trial may flip. 1.00 includes "
                 "everything except the neurons with a measured transmitter.")
    sx = body.x + 530
    if sx + 120 + 220 > body.right - 8:                       # a narrow menu: the second slider goes on its own row
        y += 40
        sx = body.x + 8
    m.text(surf, "Share flipped", (sx, y + 15), ui.TEXT, m.f_text, "midleft")
    m.slider(surf, (sx + max(120, m.f_text.size("Share flipped")[0] + 12), y, 220, 30), st.flip_share if hasattr(st, "flip_share") else 0.5, 0.1, 1.0, 0.1,
             "{:.0%}", lambda v: setattr(st, "flip_share", float(v)), lambda: None, id="flip_share",
             tip="Each trial flips this share of the candidates, chosen at random. Which of them are actually wrong "
                 "is unknowable, so a trial samples one possible world and the trials together give the answer.")
    y += 40
    m.text(surf, f"{stats['candidates']:,} of {stats['signed']:,} signed neurons are below the cutoff "
                 f"({stats['unknown_confidence']:,} of those have no confidence at all)", (body.x + 8, y), ui.INK,
           m.f_text)
    m.text(surf, f"{stats['measured']:,} neurons have a measured transmitter and are never flipped  ·  "
                 f"{stats['excitatory']:,} excitatory, {stats['inhibitory']:,} inhibitory in the model",
           (body.x + 8, y + 22), ui.TEXT, m.f_small)
    live = getattr(host, "wiring", None)
    if live is not None and live.flip_rows:
        m.text(surf, f"live: {len(live.flip_rows):,} neurons flipped (click a neuron in the big brain view to flip "
                     f"one by hand)", (body.x + 8, y + 40), ui.AMBER, m.f_small)
    y += 62
    share = getattr(st, "flip_share", 0.5)
    m.button(surf, (body.x + 8, y, m.bw("Flip them on every fly", 260), 40), "Flip them on every fly",
             lambda: host.set_wiring(Wiring(min_synapses=live.min_synapses if live else 1,
                                            flip_rows=wiring_mod.random_flip(g, st.flip_conf, share, 0),
                                            inhibition_scale=live.inhibition_scale if live else 1.0)),
             id="apply_flip", style="primary", enabled=not getattr(host, "wiring_busy", ""),
             tip="Applies one sampled flip set to the live flies so you can watch what it does. The stress test "
                 "below is the measured version.")
    job, res = st.flip_job, st.flip_result
    if job is not None and not job["thread"].is_alive():
        st.flip_result = res = job.get("result") or res
        if job.get("error"):
            m.text(surf, f"error: {job['error']}", (body.x + 290, y + 48), ui.BAD, m.f_small)
        st.flip_job = job = None
    if job is not None:
        frac = job["done"] / max(1, job["total"])
        pygame.draw.rect(surf, (30, 36, 48), (body.x + 290, y + 14, body.w - 320, 12), border_radius=6)
        pygame.draw.rect(surf, ui.AMBER, (body.x + 290, y + 14, max(8, int((body.w - 320) * frac)), 12),
                         border_radius=6)
        m.text(surf, job["label"], (body.x + 290, y + 30), ui.LABEL, m.f_small)
    else:
        m.button(surf, (body.x + 290, y, 300, 40),
                 f"Run {st.flip_trials} randomized trials", lambda: _start_flips(m, st),
                 id="run_flips", tip="Each trial flips a different random sample, then re-runs the validated "
                                     "behaviors over the validation seeds with their own pass criteria. The "
                                     "unperturbed run is the control.")
        m.slider(surf, (body.x + 610, y + 5, 200, 30), st.flip_trials, 2, 20, 1, "{:.0f} trials",
                 lambda v: setattr(st, "flip_trials", int(v)), lambda: None, id="flip_trials")
    y += 52
    if res:
        _draw_flips(m, surf, pygame.Rect(body.x + 8, y, body.w - 16, body.bottom - y), res)


def _start_flips(m, st) -> None:
    from kickthefly.lab import labjobs, robustness

    job = dict(done=0, total=1, label="starting", result=None, error=None)
    share = getattr(st, "flip_share", 0.5)

    def work():
        try:
            job["result"] = robustness.signflip_trials(
                cutoff=float(st.flip_conf), share=share, trials=int(st.flip_trials),
                workers=labjobs.default_workers(),
                progress=lambda d, n, label: job.update(done=d, total=n, label=label))
        except Exception as e:
            job["error"] = f"{type(e).__name__}: {e}"

    job["thread"] = threading.Thread(target=work, name="signflip", daemon=True)
    job["thread"].start()
    st.flip_job = job


def _draw_flips(m, surf, area, res) -> None:
    c = res["candidates"]
    m.text(surf, f"{res['trials']} trials x {res['share']:.0%} of {c['candidates']:,} uncertain neurons  ·  seeds "
                 f"{res['seeds'][0]}-{res['seeds'][-1]}  ·  {res['seconds']:.0f}s  ·  validation's own criteria",
           (area.x, area.y), ui.INK, m.f_small)
    y = area.y + 22
    for label, x in (("behavior", area.x), ("survived", area.x + 320), ("95% CI", area.x + 420),
                     ("effect (flipped)", area.x + 540), ("unperturbed", area.x + 690)):
        m.text(surf, label, (x, y), ui.LABEL, m.f_small)
    y += 20
    for test_id, b in res["behaviors"].items():
        col = ui.GOOD if b["verdict"] == "survives" else ui.BAD if b["verdict"] == "breaks" else ui.AMBER
        m.text(surf, b["name"][:46], (area.x, y), ui.TEXT, m.f_small)
        m.text(surf, f"{b['survived']}/{b['trials']}", (area.x + 320, y), col, m.f_small)
        m.text(surf, f"{b['survival_ci'][0]:.0%}-{b['survival_ci'][1]:.0%}", (area.x + 420, y), ui.TEXT, m.f_small)
        m.text(surf, f"{b['effect_mean']:.2f} [{b['effect_ci'][0]:.2f}, {b['effect_ci'][1]:.2f}]",
               (area.x + 540, y), ui.TEXT, m.f_small)
        m.text(surf, f"{b['control_effect']:.2f}", (area.x + 690, y), ui.LABEL, m.f_small)
        y += 20
    y += 4
    y = m.wrapped(surf, "A behavior that survives every trial does not depend on the signs the dataset is unsure of.", (area.x, y), area.w, ui.LABEL, m.f_small, max_lines=3)
    m.button(surf, (area.x, min(y + 4, area.bottom - 34), 200, 30), "Export CSV + JSON",
             lambda: _export(m, res), id="flips_export")


def _export(m, res) -> None:
    import time

    from kickthefly.lab import recorder, robustness

    folder = recorder.exports_dir() / f"{time.strftime('%Y%m%d-%H%M%S')}-{res['kind']}"
    robustness.save(res, folder)
    m.host.last_export = str(folder)
    m.flash(f"saved to {folder.name} in exports", ui.GOOD)


# --- picrotoxin / global inhibition block ------------------------------------------------------------------------
# 3.0 day 2: picrotoxin and the inhibition block moved to Lab > Pharmacology (labtoolkit.py), which calls the helpers below. The
# tab is no longer registered here; the drawing function stays because the Pharmacology page reuses its report.
def _tab_inhibition(m, surf, body, st, host) -> None:
    from kickthefly.core import simcore
    from kickthefly.sim import wiring as wiring_mod
    from kickthefly.sim.wiring import Wiring

    y = body.y
    m.text(surf, "Picrotoxin / global inhibition block: scales or zeroes inhibitory synapse weights based on MaleCNS v1.0 "
                 "neurotransmitter predictions.", (body.x + 8, y), ui.TEXT, m.f_small)
    m.text(surf, "Whether removing inhibition produces runaway activity is an observed outcome of the recurrent "
                 "network, not a scripted seizure.", (body.x + 8, y + 18), ui.LABEL, m.f_small)
    m.chip(surf, (body.x + 8, y + 40), "CONNECTOME")
    m.text(surf, "acts on real predicted-inhibitory synapses (GABA and glutamate)",
           (body.x + 120, y + 48), ui.LABEL, m.f_small, "midleft")
    m.chip(surf, (body.x + 540, y + 40), "RULE")
    m.text(surf, "severity mapping and physical convulsing are game rules; receptor expression is unmodeled",
           (body.x + 600, y + 48), ui.LABEL, m.f_small, "midleft")
    y += 72
    try:
        g = simcore.pack()[0]
        severity = getattr(st, "inhibition_severity", 100)
        scale = max(0.0, min(1.0, 1.0 - severity / 100.0))
        stats = wiring_mod.inhibition_stats(g, scale)
    except Exception as e:
        m.text(surf, f"needs a brain pack with transmitter predictions: {e}", (body.x + 8, y), ui.BAD, m.f_small)
        return

    m.text(surf, "Block severity", (body.x + 8, y + 15), ui.TEXT, m.f_text, "midleft")
    m.slider(surf, (body.x + 160, y, 260, 30), severity, 0, 100, 5, "{:.0f}%",
             lambda v: setattr(st, "inhibition_severity", int(v)), lambda: None, id="inhib_sev",
             tip="0% leaves all inhibitory synapses unmodified (x1.00). 100% zeroes all predicted GABA and glutamate "
                 "synapses (x0.00, full picrotoxin block).")
    m.text(surf, f"Synapse weight scale: x{scale:.2f}", (body.x + 440, y + 15), ui.INK, m.f_text, "midleft")
    y += 40

    m.text(surf, f"{stats['inhibitory_connections']:,} of {stats['total_connections']:,} connections affected "
                 f"({stats['inhibitory_share']:.1%})  ·  {stats['inhibitory_neurons']:,} presynaptic inhibitory neurons "
                 f"({stats['gaba_neurons']:,} GABA, {stats['glutamate_neurons']:,} glutamate)",
           (body.x + 8, y), ui.INK, m.f_text)
    live = getattr(host, "wiring", None)
    applied = live is not None and abs(live.inhibition_scale - scale) < 1e-4
    if live is not None and live.inhibition_scale != 1.0:
        m.text(surf, f"live on fly: inhibition scaled x{live.inhibition_scale:.2f}",
               (body.x + 8, y + 20), ui.AMBER, m.f_small)
    y += 46

    m.button(surf, (body.x + 8, y, 240, 38), "Applied" if applied else "Apply to every fly",
             lambda: host.set_wiring(Wiring(min_synapses=live.min_synapses if live else 1,
                                            flip_rows=live.flip_rows if live else (),
                                            inhibition_scale=scale)),
             id="apply_inhib", style="primary", enabled=not applied and not getattr(host, "wiring_busy", ""),
             tip="Reversibly scales all predicted inhibitory synapses on the live flies. Watch the fly convulse "
                 "naturally as runaway excitation recruits motor circuits.")

    job, res = st.inhibition_job, st.inhibition_result
    if job is not None and not job["thread"].is_alive():
        st.inhibition_result = res = job.get("result") or res
        if job.get("error"):
            m.text(surf, f"error: {job['error']}", (body.x + 270, y + 19), ui.BAD, m.f_small, "midleft")
        st.inhibition_job = job = None

    if job is not None:
        frac = job["done"] / max(1, job["total"])
        pygame.draw.rect(surf, (30, 36, 48), (body.x + 270, y + 13, body.w - 300, 12), border_radius=6)
        pygame.draw.rect(surf, ui.AMBER, (body.x + 270, y + 13, max(8, int((body.w - 300) * frac)), 12),
                         border_radius=6)
        m.text(surf, job["label"], (body.x + 270, y + 29), ui.LABEL, m.f_small)
    else:
        m.button(surf, (body.x + 270, y, m.bw("Measure firing rate distribution", 300), 38), "Measure firing rate distribution",
                 lambda: _start_inhibition(m, st, scale), id="run_inhib_dist",
                 tip="Simulates the whole connectome before (x1.00) and after (current scale) and measures the "
                     "firing rate distribution across all 166.7k neurons to document runaway excitation.")
    y += 50
    if res:
        _draw_inhibition(m, surf, pygame.Rect(body.x + 8, y, body.w - 16, body.bottom - y), st, res)


def _start_inhibition(m, st, scale: float) -> None:
    import threading
    from kickthefly.lab import robustness

    job = dict(done=0, total=2, label="starting", result=None, error=None)

    def work():
        try:
            job["label"] = "measuring before (control x1.00) …"
            job["done"] = 1
            res = robustness.inhibition_report(scale=scale, seeds=(1000,), steps=200)
            job["done"] = 2
            job["label"] = "done"
            job["result"] = res
        except Exception as e:
            job["error"] = f"{type(e).__name__}: {e}"

    job["thread"] = threading.Thread(target=work, name="inhibition_block", daemon=True)
    job["thread"].start()
    st.inhibition_job = job


def _draw_inhibition(m, surf, area, st, res) -> None:
    b, a = res["before"], res["after"]
    m.text(surf, f"Firing Rate Distribution Report  ·  Severity: {res['severity']:.0%} (scale x{res['scale']:.2f})  ·  "
                 f"seeds {res['seeds']}  ·  {res['seconds']:.1f}s", (area.x, area.y), ui.INK, m.f_small)
    y = area.y + 20

    m.text(surf, f"Mean rate: {b['mean']:.1f} Hz -> {a['mean']:.1f} Hz ({a['mean'] / max(b['mean'], 0.01):.1f}x)   "
                 f"Median: {b['median']:.1f} Hz -> {a['median']:.1f} Hz   "
                 f"95th percentile: {b['p95']:.1f} Hz -> {a['p95']:.1f} Hz   "
                 f"Max: {b['max']:.1f} Hz -> {a['max']:.1f} Hz",
           (area.x, y), ui.BAD if a['mean'] > b['mean'] * 2 else ui.GOOD, m.f_small)
    y += 24

    head = ["Rate bin", "Control (100% inhib)", f"Treated (x{res['scale']:.2f} inhib)", "Shift"]
    xs = [area.x, area.x + 130, area.x + 310, area.x + 500]
    for hx, label in zip(xs, head):
        m.text(surf, label, (hx, y), ui.LABEL, m.f_small)
    y += 18

    for lbl, b_cnt, b_sh, a_cnt, a_sh in zip(b["bin_labels"], b["histogram"], b["bin_shares"],
                                             a["histogram"], a["bin_shares"]):
        m.text(surf, lbl, (xs[0], y), ui.TEXT, m.f_small)
        m.text(surf, f"{b_cnt:,} ({b_sh:.1%})", (xs[1], y), ui.LABEL, m.f_small)
        m.text(surf, f"{a_cnt:,} ({a_sh:.1%})", (xs[2], y), ui.INK, m.f_small)
        diff = a_sh - b_sh
        col = ui.BAD if (diff > 0 and (">" in lbl or "40" in lbl or "20" in lbl)) else ui.GOOD if diff < 0 else ui.TEXT
        m.text(surf, f"{diff:+.1%}", (xs[3], y), col, m.f_small)
        y += 18

    y += 6
    y = m.wrapped(surf, "Runaway activity is an observed emergent outcome of uninhibited recurrent excitation across 166.7k "
                 "neurons, not a scripted seizure.", (area.x, y), area.w, ui.LABEL, m.f_small, max_lines=3)
    m.button(surf, (area.x, min(y + 4, area.bottom - 34), 200, 30), "Export CSV + JSON",
             lambda: _export(m, res), id="inhib_export")

