"""Lab screen for the critical path finder: pick a behavior, rank the cell types it depends on, apply a lesion.

The run itself lives in criticalpath.py and is the same one --critical-path does headless; this is the front end,
with a progress bar, a ranked table and a button that puts any row's lesion on the live flies so you can watch it.
"""
from __future__ import annotations

import threading
import time

import pygame

from kickthefly.ui import menu as ui

TARGETS = (("looming_escape", "Looming -> giant fiber"), ("sugar_feeding", "Sugar -> MN9"),
           ("antenna_grooming_circuit", "Antennal touch -> aDN"), ("mb_conditioning", "T-maze conditioning"),
           ("looming", "Assay: looming escape"), ("sugar", "Assay: sugar response"),
           ("tmaze", "Assay: T-maze"))


def _st(m):
    from kickthefly.lab import lab

    st = lab._state(m)
    if not hasattr(st, "cp_target"):
        st.cp_target = "looming_escape"
        st.cp_top = 15
        st.cp_job = None
        st.cp_results = {}
        st.cp_scroll = 0
    return st


def page(m: ui.Menu, surf, rect, mouse) -> None:
    from kickthefly.lab import criticalpath

    st, host = _st(m), m.host
    m.text(surf, "CRITICAL PATH FINDER", (rect.x + 24, rect.y + 16), ui.INK, m.f_head)
    y = m.subtitle(surf, rect, "Silence one cell type at a time, re-run the behavior, and rank the types by what it did. Every "
                               "lesion is compared with an unperturbed fly of the same seed.")
    y = m.flow_buttons(surf, rect.x + 24, y, rect.right - 24, [(label, (lambda k=key: setattr(st, "cp_target", k)), ("cp", key),
                                                               st.cp_target == key, None) for key, label in TARGETS]) - 30
    y += 42
    job = st.cp_job
    if job is not None and not job["thread"].is_alive():
        if job.get("result"):
            st.cp_results[job["target"]] = job["result"]
        if job.get("error"):
            m.text(surf, f"error: {job['error']}", (rect.x + 24, y + 48), ui.BAD, m.f_small)
        st.cp_job = job = None
    if job is not None:
        frac = job["done"] / max(1, job["total"])
        pygame.draw.rect(surf, (30, 36, 48), (rect.x + 24, y + 10, rect.w - 300, 12), border_radius=6)
        pygame.draw.rect(surf, ui.AMBER, (rect.x + 24, y + 10, max(8, int((rect.w - 300) * frac)), 12),
                         border_radius=6)
        m.text(surf, f"{job['label']}  ·  {job['done']}/{job['total']} flies  ·  {time.time() - job['t0']:.0f}s",
               (rect.x + 24, y + 26), ui.LABEL, m.f_small)
        m.text(surf, "interrupting is safe: finished types are written as they go and --resume picks them up",
               (rect.x + 24, y + 44), ui.DIM, m.f_small)
    else:
        m.button(surf, (rect.x + 24, y, 250, 36), f"Run on the top {st.cp_top} types",
                 lambda: _start(m, st), id="cp_run", style="primary",
                 tip="Each type is silenced on its own over the validation seeds, against a same-seed unperturbed "
                     "control. Minutes, not seconds. Results are written as they finish, so it can be resumed.")
        m.slider(surf, (rect.x + 290, y + 3, 220, 30), st.cp_top, 5, 60, 5, "{:.0f} types",
                 lambda v: setattr(st, "cp_top", int(v)), lambda: None, id="cp_top",
                 tip="How many candidate types to try. They are ranked by how much of the readout's input they "
                     "supply within two synaptic hops; anything outside the shortlist is not tested at all.")
    res = st.cp_results.get(st.cp_target)
    body = pygame.Rect(rect.x + 16, y + 48, rect.w - 32, rect.bottom - y - 48 - 66)
    if res:
        _draw_table(m, surf, body, st, host, res)
    else:
        m.text(surf, "No run for this target yet." if job is None else "", (body.x + 8, body.y + 8), ui.LABEL,
               m.f_small)
    if res:
        m.button(surf, (rect.x + 24, rect.bottom - 58, 200, 42), "Export CSV + JSON",
                 lambda: _export(m, res), id="cp_export")
    m.button(surf, (rect.right - 164, rect.bottom - 58, 140, 42), "Back", m.back, style="primary", id=("cp", "back"))


def _start(m, st) -> None:
    from kickthefly.lab import criticalpath, labjobs, recorder

    job = dict(done=0, total=1, label="starting", result=None, error=None, target=st.cp_target, t0=time.time())
    folder = recorder.exports_dir() / f"critical-path-{st.cp_target}"

    def work():
        try:
            job["result"] = criticalpath.run(
                job["target"], top=int(st.cp_top), workers=labjobs.default_workers(), resume=folder,
                progress=lambda d, n, label: job.update(done=d, total=n, label=label))
            criticalpath.save(job["result"], folder)
        except Exception as e:
            job["error"] = f"{type(e).__name__}: {e}"

    job["thread"] = threading.Thread(target=work, name="critical-path", daemon=True)
    job["thread"].start()
    st.cp_job = job


def _draw_table(m, surf, area, st, host, res) -> None:
    m.text(surf, f"{res['metric']}  ·  unperturbed {res['control']['mean']:.2f}  ·  seeds {res['seeds'][0]}-"
                 f"{res['seeds'][-1]}  ·  {res['seconds']:.0f}s  ·  {len(res['ranked'])} types tried",
           (area.x + 8, area.y), ui.INK, m.f_small)
    cols = ((8, "#"), (44, "cell type"), (180, "neurons"), (250, "lesioned"), (360, "95% CI"), (470, "change"),
            (560, "p"), (650, ""))
    y = area.y + 22
    for dx, label in cols:
        m.text(surf, label, (area.x + dx, y), ui.LABEL, m.f_small)
    y += 18
    view = pygame.Rect(area.x, y, area.w, area.bottom - y)
    off = int(m.scroll.get("lab_critical", 0))
    m.clip = view
    prev = surf.get_clip()
    surf.set_clip(view)
    yy = y - off
    for r in res["ranked"]:
        strong = r["p_value"] < 0.01 and abs(r["effect_share"]) >= 0.1
        col = ui.BAD if strong and r["effect_share"] < 0 else ui.GOOD if strong else ui.TEXT
        m.text(surf, str(r["rank"]), (area.x + 8, yy), ui.DIM, m.f_small)
        m.text(surf, r["type"] + ("  *" if r["drives_the_behavior"] else ""), (area.x + 44, yy), col, m.f_small)
        m.text(surf, f"{r['neurons']:,}", (area.x + 180, yy), ui.LABEL, m.f_small)
        m.text(surf, f"{r['lesioned_mean']:.2f}", (area.x + 250, yy), ui.TEXT, m.f_small)
        m.text(surf, f"[{r['lesioned_ci'][0]:.2f}, {r['lesioned_ci'][1]:.2f}]", (area.x + 360, yy), ui.DIM,
               m.f_small)
        m.text(surf, f"{r['effect_share']:+.0%}", (area.x + 470, yy), col, m.f_small)
        m.text(surf, f"{r['p_value']:.4f}", (area.x + 560, yy), ui.LABEL, m.f_small)
        live = f"type:{r['type']}" in {f"type:{t}" for t in getattr(host, "type_ops", {})
                                       if host.type_ops.get(t) == -1}
        m.button(surf, (area.x + 650, yy - 3, 150, 22), "lesion applied" if live else "Apply this lesion",
                 (lambda t=r["type"]: _apply(host, t)), id=("cp_apply", r["type"]), active=live,
                 tip=f"Silences every {r['type']} neuron on the live flies, the same way brain surgery does, so you "
                     f"can watch what this row means in the game.")
        yy += 24
    m.content_h["lab_critical"] = max(0, yy + off - view.bottom + 8)
    surf.set_clip(prev)
    m.clip = None


def _apply(host, cell_type: str) -> None:
    """One click: silence that type on every live fly, exactly as brain surgery does."""
    on = host.type_ops.get(cell_type) == -1
    if on:
        host.type_ops.pop(cell_type, None)
    else:
        host.type_ops[cell_type] = -1
    host._apply_surgery()
    host.note(f"SURGERY  {cell_type}: {'normal' if on else 'off'}")


def _export(m, res) -> None:
    from kickthefly.lab import criticalpath, recorder

    folder = recorder.exports_dir() / f"{time.strftime('%Y%m%d-%H%M%S')}-critical-path-{res['target']}"
    criticalpath.save(res, folder)
    m.host.last_export = str(folder)
    m.flash(f"saved to {folder.name} in exports", ui.GOOD)
