"""Mini-papers (3.0 day 5; Esc > Mini-papers and Lab > Mini-papers): short guided experiments that reproduce a classic paper step by step.

A paper is a lecture protocol (lab/classroom.py) of five steps: the paper, YOUR HYPOTHESIS, the run, the plot, and your result next to the
paper's. The experiments are validation tests and the Buridan rig (lab/minipapers.py); the run is a background job (worker processes) with a
progress bar and a Cancel button, never on the game thread. What the paper found is written from what the paper itself states, cited, with
how much of it was read; "your result" is a MODEL PREDICTION.

One scrolling column, so it holds at larger text and a narrow window: lines wrap (Menu.wrapped) and buttons are sized from the font."""
from __future__ import annotations

import pygame

from kickthefly.core.i18n import tr
from kickthefly.lab import classroom, minipapers
from kickthefly.ui import menu as ui
from kickthefly.ui.bgjob import BgJob, draw_progress

SERIES = {"default": ((57, 135, 229), (217, 89, 38)), "blue-yellow": ((30, 100, 200), (240, 190, 20)), "high-contrast": ((255, 255, 255), (255, 200, 0))}
REFS = {"mb_conditioning": [(0.5, "PI 0.5 (validation's bar)")], "rig:buridan": [(45.0, "45 deg: chance")],
        "default": [(1.0, "calm (x1)"), (1.5, "x1.5: validation's bar")]}
AXIS = {"mb_conditioning": "performance index", "rig:buridan": "stripe deviation (deg)", "default": "firing ratio to calm"}
LEGEND = {"mb_conditioning": ("unpaired", "paired"), "rig:buridan": ("no stripes", "stripes"), "default": ("control", "drive")}


class _St:
    def __init__(self):
        self.paper: str | None = None
        self.answers: dict[str, dict[str, str]] = {}
        self.committed: dict[str, dict[str, str]] = {}   # the hypothesis as it was when the shown result's run started
        self.sess = classroom.ClassroomSession("looming")
        self.results: dict[str, dict] = {}
        self.job: BgJob | None = None
        self.full = False
        self.q_cursor = 0                      # the gamepad's question on the hypothesis step
        self.error = ""


def st(m) -> _St:
    s = getattr(m.host, "_minipaper_state", None)
    if s is None:
        s = m.host._minipaper_state = _St()
    return s


def _colors(m):
    return SERIES.get(m.host.cfg["access.palette"], SERIES["default"])


def _h(m) -> int:
    return max(34, m.f_text.get_linesize() + 14)


def _workers() -> int:
    from kickthefly.lab import labjobs

    return max(1, min(3, labjobs.default_workers()))


def open_paper(m, pid: str) -> None:
    s = st(m)
    s.paper = pid
    s.sess.set_lecture(f"paper_{pid}")
    s.sess.goto_step(0)
    s.q_cursor = 0
    s.error = ""


def start(m, pid: str) -> None:
    s = st(m)
    if s.job is not None and s.job.running:
        return
    seeds = minipapers.FULL_SEEDS if s.full else minipapers.QUICK_SEEDS
    commit(s, pid)

    def work(job: BgJob):
        return minipapers.run_paper(pid, seeds, workers=_workers(), cancel=job.cancel,
                                    progress=lambda d, n, label="": job.update(d / max(1, n), label or tr("running")))

    s.error = ""
    s.job = BgJob(f"Mini-paper {pid}", work).start()


def commit(s: _St, pid: str) -> None:
    """Freeze the hypothesis for the run that starts now (3.0 release review: the comparison used whatever was selected when it was drawn, so a
    hypothesis changed after seeing the result was reported as having matched it)."""
    s.committed[pid] = dict(s.answers.get(pid, {}))


def committed(s: _St, pid: str) -> dict[str, str]:
    return s.committed.get(pid, s.answers.get(pid, {}))


def collect(s: _St) -> None:
    """A finished job's result goes to the paper whose label it carries (a job outlives the page that started it)."""
    j = s.job
    if j is None or j.running:
        return
    s.job = None
    if j.error:
        s.error = j.error
    elif j.result is not None and not j.cancelled:
        s.results[j.result["paper"]] = j.result


def _tags(m, surf, x, y, names) -> int:
    for n in names:
        col = ui.TAG_COLORS.get(n, (150, 120, 220))
        img = m.f_small.render(n, True, (10, 12, 16))
        r = pygame.Rect(x, y, img.get_width() + 12, img.get_height() + 4)
        pygame.draw.rect(surf, col, r, border_radius=5)
        surf.blit(img, (r.x + 6, r.y + 2))
        x = r.right + 6
    return x


def _para(m, surf, x, y, w, text, color=None, font=None, lines=12) -> int:
    return m.wrapped(surf, text, (x, y), w, color or ui.TEXT, font or m.f_text, max_lines=lines) + 6


def _pairs_plot(m, surf, box, q: dict, colors) -> None:
    pygame.draw.rect(surf, (14, 16, 22), box, border_radius=8)
    pairs = q["pairs"]
    test = q["test"]
    refs = REFS.get(test, REFS["default"])
    leg = LEGEND.get(test, LEGEND["default"])
    if not pairs:
        m.wrapped(surf, tr("Recorded numbers only (no per-fly data): see the comparison."), (box.x + 12, box.y + 12), box.w - 24, ui.LABEL, m.f_small, max_lines=3)
        return
    vals = [v for p in pairs for v in (p["drive"], p["control"])] + [r for r, _ in refs]
    lo, hi = min(0.0, min(vals)), max(vals) * 1.12
    if hi - lo < 1e-9:
        hi = lo + 1.0
    top, bottom = box.y + 28, box.bottom - 30
    left, right = box.x + 70, box.right - 16

    def Y(v):
        return bottom - (v - lo) / (hi - lo) * (bottom - top)

    for r, label in refs:
        pygame.draw.line(surf, (70, 76, 92), (left, Y(r)), (right, Y(r)))
        m.text(surf, label, (right, Y(r) - 2), ui.LABEL, m.f_small, "bottomright")
    xs = (left + (right - left) // 4, left + 3 * (right - left) // 4)
    for p in pairs:
        pygame.draw.line(surf, (110, 116, 130), (xs[0], Y(p["control"])), (xs[1], Y(p["drive"])), 1)
    for p in pairs:
        pygame.draw.circle(surf, colors[1], (xs[0], int(Y(p["control"]))), 5)
        pygame.draw.circle(surf, colors[0], (xs[1], int(Y(p["drive"]))), 5)
    m.text(surf, leg[0], (xs[0], box.bottom - 6), ui.TEXT, m.f_small, "midbottom")
    m.text(surf, leg[1], (xs[1], box.bottom - 6), ui.TEXT, m.f_small, "midbottom")
    m.text(surf, tr(AXIS.get(test, AXIS["default"])), (box.x + 8, box.y + 6), ui.LABEL, m.f_small)
    m.text(surf, f"{lo:g}", (left - 6, bottom), ui.LABEL, m.f_small, "midright")
    m.text(surf, f"{hi:.3g}", (left - 6, top), ui.LABEL, m.f_small, "midright")


# --- steps -------------------------------------------------------------------------------------------------------------------------------
def _step_read(m, surf, x, y, w, p) -> int:
    y = _para(m, surf, x, y, w, tr(p.summary))
    m.text(surf, tr("The paper"), (x, y), ui.AMBER, m.f_bold)
    y += m.f_bold.get_linesize() + 2
    y = _para(m, surf, x, y, w, p.citation, ui.TEXT, m.f_small)
    y = _para(m, surf, x, y, w, "doi:" + p.doi, ui.LABEL, m.f_small)
    y = _para(m, surf, x, y, w, tr("What was read: {r}", r=p.read), ui.LABEL, m.f_small)
    m.text(surf, tr("What the paper reports"), (x, y), ui.AMBER, m.f_bold)
    y += m.f_bold.get_linesize() + 2
    y = _para(m, surf, x, y, w, tr(p.found))
    y = _para(m, surf, x, y, w, tr("Numbers the paper states: {n}", n=p.found_numbers), ui.LABEL, m.f_small)
    _tags(m, surf, x, y, ["LITERATURE"])
    return y + m.f_small.get_linesize() + 12


def _step_hypothesis(m, surf, x, y, w, p, s) -> int:
    y = _para(m, surf, x, y, w, tr("Commit to a hypothesis before anything runs. It is fixed when you press Run; a change after that applies to your next run."),
              ui.LABEL, m.f_small)
    ans = s.answers.setdefault(p.id, {})
    h = _h(m)
    for q in p.questions:
        y = _para(m, surf, x, y, w, tr(q.prompt))
        for key, label in q.options:
            r = pygame.Rect(x + 8, y, min(w - 8, 760), h)
            m.button(surf, r, tr(label), (lambda k=key, t=q.test: ans.__setitem__(t, k)), id=("mp_opt", p.id, q.test, key), active=ans.get(q.test) == key)
            y += h + 6
        y += 8
    return y


def _step_run(m, surf, x, y, w, p, s) -> int:
    ok, why = minipapers.available(p)
    ans = s.answers.get(p.id, {})
    ready = all(q.test in ans for q in p.questions)
    busy = s.job is not None and s.job.running
    h = _h(m)
    if not ready:
        y = _para(m, surf, x, y, w, tr("Choose a hypothesis on the previous step first: the comparison needs it."), ui.AMBER, m.f_small)
    if not ok:
        y = _para(m, surf, x, y, w, tr("Cannot run live here: {why}.", why=why), ui.AMBER, m.f_small)
    else:
        m.segmented(surf, (x, y, min(w, 560), h), [tr("Quick: 4 flies, exploration seeds"), tr("Full: 10 flies, validation seeds")], 1 if s.full else 0,
                    lambda i: setattr(s, "full", i == 1), id=("mp_size", p.id), enabled=not busy,
                    tip=tr("Quick is a first look (seeds 0-3, too few flies for the validation's p < 0.01 bar). Full is the validation seeds 1000-1009 and takes several times longer."))
        y += h + 8
    res = s.results.get(p.id)
    m.button(surf, (x, y, 220, h), tr("Run the experiment") if ok else tr("Show recorded numbers"), lambda: _run_or_record(m, p, s), style="primary",
             id=("mp_run", p.id), enabled=ready and not busy)
    y += h + 8
    if res is not None:
        y = _para(m, surf, x, y, w, tr("Done: {n} flies in {sec:.0f} s ({src}). Go on to the plot.", n=len(res["seeds"]) if res["source"] == "live" else 10, sec=res["seconds"],
                                        src=tr("run now") if res["source"] == "live" else tr("recorded validation numbers")), ui.TEXT, m.f_small)
    y = _para(m, surf, x, y, w, tr("The experiment is a validation test, run from one brain snapshot per fly with a matched control, in worker processes."), ui.LABEL, m.f_small)
    return y


def _run_or_record(m, p, s) -> None:
    ok, why = minipapers.available(p)
    if ok:
        start(m, p.id)
    elif p.recorded:
        commit(s, p.id)
        s.results[p.id] = minipapers.run_paper(p.id, minipapers.FULL_SEEDS)       # recorded numbers: instant, no brain
    else:
        s.error = why


def _step_plot(m, surf, x, y, w, p, s, colors) -> int:
    res = s.results.get(p.id)
    if res is None:
        return _para(m, surf, x, y, w, tr("Run the experiment first."), ui.AMBER, m.f_small)
    for q in res["questions"]:
        y = _para(m, surf, x, y, w, tr(q["prompt"]), ui.TEXT, m.f_small, 6)
        box = pygame.Rect(x, y, min(w, 640), max(150, m.f_small.get_linesize() * 9))
        _pairs_plot(m, surf, box, q, colors)
        y = box.bottom + 8
    _tags(m, surf, x, y, ["MODEL PREDICTION"])
    return y + m.f_small.get_linesize() + 12


def _fmt(m_: dict) -> str:
    bits = []
    for k, v in m_.items():
        if k in ("passed",):
            continue
        bits.append(f"{k.replace('_', ' ')} {v:.3g}" if isinstance(v, float) else f"{k.replace('_', ' ')} {v}")
    return ", ".join(bits)


def _step_compare(m, surf, x, y, w, p, s) -> int:
    res = s.results.get(p.id)
    if res is None:
        return _para(m, surf, x, y, w, tr("Run the experiment first."), ui.AMBER, m.f_small)
    rows = minipapers.compare(res, committed(s, p.id))
    wide = w > 880
    cw = (w - 20) // 2 if wide else w
    y0 = y
    m.text(surf, tr("Your result (the model)"), (x, y), ui.AMBER, m.f_bold)
    y += m.f_bold.get_linesize() + 2
    src = tr("recorded validation numbers (not run now)") if res["source"] == "recorded" else \
        tr("run now on {n} flies{kind}", n=len(res["seeds"]), kind="" if res["full"] else tr(" (exploration seeds, a first look)"))
    y = _para(m, surf, x, y, cw, src, ui.LABEL, m.f_small)
    for r in rows:
        y = _para(m, surf, x, y, cw, tr("The model {v} the direction the paper's finding points to.", v=tr("REPRODUCES") if r["model_reproduces_paper"] else tr("DOES NOT REPRODUCE")),
                  ui.GOOD if r["model_reproduces_paper"] else ui.BAD, m.f_text, 3)
        y = _para(m, surf, x, y, cw, _fmt(r["measured"]), ui.TEXT, m.f_small, 4)
        y = _para(m, surf, x, y, cw, tr(r["basis"]), ui.LABEL, m.f_small, 5)
        y = _para(m, surf, x, y, cw, tr(r["note"]), ui.TEXT, m.f_small, 12)
        if r["your_hypothesis"]:
            mt = tr("You expected: {a}. The paper's finding points to: {b}.", a=tr(r["your_hypothesis"]), b=tr(r["paper_direction"]))
            y = _para(m, surf, x, y, cw, mt + " " + (tr("Your hypothesis matched the model.") if r["you_matched_the_model"] else tr("The model went the other way from your hypothesis.")),
                      ui.TEXT, m.f_small, 6)
        y += 6
    ya = y
    if wide:
        x2, y = x + cw + 20, y0
    else:
        x2, y = x, y + 8
    m.text(surf, tr("The paper found"), (x2, y), ui.AMBER, m.f_bold)
    y += m.f_bold.get_linesize() + 2
    y = _para(m, surf, x2, y, cw, tr(p.found), ui.TEXT, m.f_small, 14)
    y = _para(m, surf, x2, y, cw, tr("Numbers the paper states: {n}", n=p.found_numbers), ui.LABEL, m.f_small, 5)
    y = _para(m, surf, x2, y, cw, p.citation + "  doi:" + p.doi, ui.LABEL, m.f_small, 8)
    y = _para(m, surf, x2, y, cw, tr("What was read: {r}", r=p.read), ui.LABEL, m.f_small, 4)
    _tags(m, surf, x2, y, ["LITERATURE"])
    y += m.f_small.get_linesize() + 10
    y = max(y, ya)
    m.text(surf, tr("What this model cannot check"), (x, y), ui.AMBER, m.f_bold)
    y += m.f_bold.get_linesize() + 2
    return _para(m, surf, x, y, w, tr(p.cannot_check), ui.TEXT, m.f_small, 14) + 6


def page(m, surf, rect, mouse) -> None:
    s = st(m)
    collect(s)
    colors = _colors(m)
    key = "lab_minipapers"
    m.text(surf, tr("MINI-PAPERS"), (rect.x + 24, rect.y + 16), ui.INK, m.f_head)
    body = pygame.Rect(rect.x + 16, rect.y + 56, rect.w - 32, rect.h - 56 - 80)
    off = int(m.scroll.get(key, 0))
    prev = surf.get_clip()
    surf.set_clip(body)
    m.clip = body
    x, w = body.x + 8, body.w - 40
    y = body.y - off
    h = _h(m)
    if s.paper is None:
        y = _para(m, surf, x, y, w, tr("Each mini-paper reproduces a classic paper in five steps: read it, state a hypothesis, run it on the model, plot it, and see your "
                                       "result next to what the paper found. Where the model misses the paper, it says so and why."), ui.LABEL, m.f_small)
        for pid in minipapers.ORDER:
            p = minipapers.PAPERS[pid]
            r = pygame.Rect(x, y, min(w, 900), max(h, m.f_text.get_linesize() + m.f_small.get_linesize() + 18))
            m.button(surf, r, tr(p.title), (lambda i=pid: open_paper(m, i)), id=("mp_paper", pid), tip=tr(p.summary))
            m.text(surf, tr(p.read.split(" (")[0]), (r.right - 10, r.centery), ui.LABEL, m.f_small, "midright")
            y += r.h + 8
        y = _para(m, surf, x, y + 4, w, tr("Headless: --minipaper list, then --minipaper ID [--seeds A-B]. docs/minipapers.md."), ui.LABEL, m.f_small)
    else:
        p = minipapers.PAPERS[s.paper]
        sess = s.sess
        kind = p.steps[sess.step_idx][2]
        m.text(surf, tr(p.title), (x, y), ui.INK, m.f_bold)
        y += m.f_bold.get_linesize() + 2
        m.text(surf, tr("Step {i} of {n}: {t}", i=sess.step_idx + 1, n=sess.total_steps, t=tr(p.steps[sess.step_idx][0])), (x, y), ui.AMBER, m.f_text)
        y += m.f_text.get_linesize() + 8
        if kind == "read":
            y = _step_read(m, surf, x, y, w, p)
        elif kind == "hypothesis":
            y = _step_hypothesis(m, surf, x, y, w, p, s)
        elif kind == "run":
            y = _step_run(m, surf, x, y, w, p, s)
        elif kind == "plot":
            y = _step_plot(m, surf, x, y, w, p, s, colors)
        else:
            y = _step_compare(m, surf, x, y, w, p, s)
        last = sess.step_idx >= sess.total_steps - 1
        nb = pygame.Rect(x, y + 8, 170, h)
        m.button(surf, (x, y + 8, 150, h), tr("Previous"), sess.prev_step, id=("mp_prev", p.id), enabled=sess.step_idx > 0)
        m.button(surf, (x + 160, y + 8, 150, h), tr("Next") if not last else tr("All papers"), (sess.next_step if not last else (lambda: setattr(s, "paper", None))),
                 style="primary", id=("mp_next", p.id))
        y = nb.bottom + 14
    m.content_h[key] = max(0, y + off - body.bottom + 8)
    surf.set_clip(prev)
    m.clip = None
    if s.error:
        m.text(surf, s.error, (rect.x + 24, rect.bottom - 74), ui.BAD, m.f_small)
    if s.job is not None and s.job.running:
        draw_progress(m, surf, pygame.Rect(rect.x + 24, rect.bottom - 56, rect.w - 220, 12), s.job, ui)
    back = (lambda: setattr(s, "paper", None)) if s.paper is not None else m.back
    m.button(surf, (rect.right - 164, rect.bottom - 58, 140, 42), tr("Back") if s.paper is None else tr("Papers"), back, style="primary", id=(key, "back"))


def install(menu) -> None:
    menu.pages["lab_minipapers"] = page


def pad_nav(host, down) -> bool:
    """Gamepad on the Mini-papers page: the bumpers step back and forward, the kill-cam and big-view buttons change the hypothesis of the current
    question, use goes to the next question (then the next step) or runs the experiment, B or Start closes the paper, then the page."""
    m = host.menu
    if m.screen != "lab_minipapers":
        return False
    s = st(m)
    if down & {"crouch", "menu"}:
        if s.paper is not None:
            s.paper = None
        else:
            m.back()
        return True
    if s.paper is None:
        return False
    p = minipapers.PAPERS[s.paper]
    sess = s.sess
    kind = p.steps[sess.step_idx][2]
    busy = s.job is not None and s.job.running
    used = False
    if "tool_next" in down:
        used = sess.next_step() or used
    if "tool_prev" in down:
        used = sess.prev_step() or used
    if kind == "hypothesis" and down & {"killcam", "big_view"}:
        ans = s.answers.setdefault(p.id, {})
        q = p.questions[min(s.q_cursor, len(p.questions) - 1)]
        keys = [k for k, _ in q.options]
        fwd = "killcam" in down
        ans[q.test] = keys[(keys.index(ans[q.test]) + (1 if fwd else -1)) % len(keys)] if q.test in ans else keys[0 if fwd else -1]
        used = True
    if "use" in down and not busy:
        if kind == "run" and all(q.test in s.answers.get(p.id, {}) for q in p.questions):
            _run_or_record(m, p, s)
        elif kind == "hypothesis" and s.q_cursor < len(p.questions) - 1:
            s.q_cursor += 1                                    # the next question of this paper
        else:
            sess.next_step()
            s.q_cursor = 0
        used = True
    return used
