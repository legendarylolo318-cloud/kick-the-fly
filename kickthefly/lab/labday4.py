"""Lab pages added in 3.0 day 4: Network science, Sleep deprivation and Sensitivity analysis. Each runs its analysis in a background
job with a progress bar and a Cancel button (never on the game thread); the analysis itself is netsci.py, sleepdep.py and
sensitivity.py, the same code the headless flags run."""
from __future__ import annotations

import pygame

from kickthefly.core.i18n import tr
from kickthefly.ui import menu as ui
from kickthefly.ui.bgjob import BgJob, draw_progress


class _St:
    def __init__(self):
        self.job: BgJob | None = None
        self.brain = "adult"
        self.net: dict[str, dict] = {}
        self.cached: dict[str, bool] = {}              # whether a valid cache exists per brain: looked up once, not every frame
        self.error = ""
        self.sleep: dict | None = None
        self.seeds = 5
        self.sens: dict | None = None
        self.sens_params = {p: True for p in ()}
        self.sens_smoke = True
        self.msg = ""


def _st(m) -> _St:
    st = getattr(m.host, "_day4_lab", None)
    if st is None:
        st = m.host._day4_lab = _St()
    return st


def _workers() -> int:
    from kickthefly.lab import labjobs

    return max(1, min(2, labjobs.default_workers()))        # each worker holds a brain (about 1 GB): two keeps a small machine usable


def _chrome(m, surf, rect, title: str, sub: str, st: _St, key: str):
    m.text(surf, title, (rect.x + 24, rect.y + 16), ui.INK, m.f_head)
    m.text(surf, sub, (rect.x + 24, rect.y + 50), ui.LABEL, m.f_small)
    body = pygame.Rect(rect.x + 16, rect.y + 80, rect.w - 32, rect.h - 80 - 80)
    off = int(m.scroll.get(key, 0))
    return body, off


def _end(m, surf, rect, body, key: str, y: int, off: int, st: _St) -> None:
    m.content_h[key] = max(0, y + off - body.bottom + 8)
    if st.error:
        m.text(surf, st.error, (rect.x + 24, rect.bottom - 74), ui.BAD, m.f_small)
    if st.job is not None and st.job.running:
        draw_progress(m, surf, pygame.Rect(rect.x + 24, rect.bottom - 56, rect.w - 220, 12), st.job, ui)
    m.button(surf, (rect.right - 164, rect.bottom - 58, 140, 42), tr("Back"), m.back, style="primary", id=(key, "back"))


def _tags(m, surf, x, y, names) -> int:
    for n in names:
        col = ui.TAG_COLORS.get(n, (150, 120, 220))
        img = m.f_small.render(n, True, (10, 12, 16))
        r = pygame.Rect(x, y, img.get_width() + 12, img.get_height() + 4)
        pygame.draw.rect(surf, col, r, border_radius=5)
        surf.blit(img, (r.x + 6, r.y + 2))
        x = r.right + 6
    return x


# --- network science -----------------------------------------------------------------------------------------------------------
def _start_net(st: _St, brain: str, force: bool) -> None:
    from kickthefly.lab import netsci

    def work(job: BgJob):
        return netsci.compute(brain, progress=lambda f, label: job.update(f, label), cancel=job.cancel, force=force)

    st.error = ""
    st.job = BgJob("Network science", work).start()


_KIND = {"Network science": "net", "Sleep deprivation": "sleep", "Sensitivity": "sens"}


def _collect(st: _St, kind: str | None = None) -> None:
    """Take a finished job's result to ITS page. The three pages share one job slot, so the page that happens to be open may not be
    the one that started it (3.0 day 4 review: a sleep result collected on the network-science page raised KeyError 'brain'). The
    job's own label decides; `kind`, the calling page, is not used for that."""
    job = st.job
    if job is None or job.running:
        return
    st.job = None
    if job.error:
        st.error = job.error
        return
    if job.cancelled or job.result is None:
        return
    got = _KIND.get(job.label)
    if got == "net":
        st.net[job.result["brain"]] = job.result
        st.cached.pop(job.result["brain"], None)
    elif got == "sleep":
        st.sleep = job.result
    elif got == "sens":
        st.sens = job.result


def page_netsci(m: ui.Menu, surf, rect, mouse) -> None:
    from kickthefly.lab import netsci

    st = _st(m)
    _collect(st, "net")
    key = "lab_netsci"
    body, off = _chrome(m, surf, rect, tr("NETWORK SCIENCE"), tr("Degrees, reciprocity, motifs against a null model, rich club, communities and regions of the brain pack."), st, key)
    busy = st.job is not None and st.job.running
    x, y = body.x + 8, body.y + 4
    for b in ("adult", "larva"):
        m.button(surf, (x, y, 110, 32), tr(b.capitalize()), (lambda b=b: setattr(st, "brain", b)), id=("ns_brain", b), active=st.brain == b, enabled=not busy)
        x += 118
    if st.brain not in st.cached:
        st.cached[st.brain] = netsci.load_cached(st.brain) is not None
    cached = st.cached[st.brain]
    m.button(surf, (x + 10, y, 220, 32), tr("Compute (cached)") if cached else tr("Compute"), lambda: _start_net(st, st.brain, False), style="primary",
             id="ns_run", enabled=not busy, tip=tr("Adult takes minutes (10 million connections), larva seconds. Cached next to your data with a checksum; recomputed if the pack or settings change."))
    m.button(surf, (x + 240, y, 150, 32), tr("Recompute"), lambda: _start_net(st, st.brain, True), id="ns_force", enabled=not busy)
    res = st.net.get(st.brain)
    m.button(surf, (x + 400, y, 190, 32), tr("Export CSV"), lambda: _export_net(m, st, res), id="ns_csv", enabled=res is not None and not busy,
             tip=tr("Writes one CSV per table into your exports folder."))
    m.clip = body
    prev = surf.get_clip()
    surf.set_clip(body)
    y += 44 - off
    x = body.x + 8
    xx = _tags(m, surf, x, y, ["CONNECTOME"])
    m.text(surf, tr("computed from the wiring (no simulation); the analysis choices (null graphs, samples, seed) are GAME RULE; nothing here is a MODEL PREDICTION"), (xx + 4, y + 2), ui.LABEL, m.f_small)
    y += 28
    if res is None:
        m.text(surf, tr("Not computed yet."), (x, y), ui.LABEL, m.f_text)
        y += 30
    else:
        d, r, c = res["degrees"], res["reciprocity"], res["communities"]
        lines = [
            f"{d['n']:,} {tr('neurons')}, {d['connections']:,} {tr('connections')}, {d['synapses']:,.0f} {tr('synapses')}  ·  {res['pack_name']}  ·  {'cache' if res.get('from_cache') else tr('computed')} {res['created']}",
            f"{tr('In-degree')}: {tr('mean')} {d['in_degree']['mean']:.1f}, {tr('median')} {d['in_degree']['median']:.0f}, p99 {d['in_degree']['p99']:.0f}, {tr('max')} {d['in_degree']['max']:.0f}   ·   "
            f"{tr('Out-degree')}: {tr('mean')} {d['out_degree']['mean']:.1f}, {tr('median')} {d['out_degree']['median']:.0f}, p99 {d['out_degree']['p99']:.0f}, {tr('max')} {d['out_degree']['max']:.0f}",
            f"{tr('Reciprocity')}: {r['binary']:.3f} ({tr('null')} {r['null_binary_mean']:.3f}, x{r['enrichment']:.1f}); {tr('synapse-weighted')} {r['weighted']:.3f}",
            f"{tr('Communities')}: {c['communities']:,}, {tr('modularity')} Q = {c['modularity']:.3f}, NMI {tr('with regions')} {c['nmi_with_regions']:.2f}"]
        for ln in lines:
            m.text(surf, ln, (x, y), ui.TEXT, m.f_small)
            y += 22
        y += 6
        m.text(surf, tr("In-degree histogram (log2 bins)"), (x, y), ui.INK, m.f_bold)
        y += 24
        y = _hist(m, surf, x, y, body.w - 40, d["in_degree"]["histogram"], d["out_degree"]["histogram"])
        m.text(surf, tr("3-node motifs (sampled estimate; enrichment over the degree-preserving null)"), (x, y), ui.INK, m.f_bold)
        y += 24
        for mo in res["motifs"]:
            zc = ui.GOOD if mo["enrichment"] == mo["enrichment"] and mo["enrichment"] > 1.5 else (ui.BAD if mo["enrichment"] == mo["enrichment"] and mo["enrichment"] < 0.67 else ui.TEXT)
            m.text(surf, f"{mo['motif']:<5} {mo['count_estimate']:>14,.0f}   x{mo['enrichment']:>8.2f}   z {mo['z']:>7.1f}   {mo['description']}", (x + 8, y), zc, m.f_small)
            y += 19
        y += 8
        m.text(surf, tr("Rich club (rho = phi / phi_null; above 1 means more than degrees alone give)"), (x, y), ui.INK, m.f_bold)
        y += 24
        m.text(surf, "   ".join(f"k>{q['k']:.0f}: {q['rho']:.2f}" for q in res["rich_club"]), (x + 8, y), ui.TEXT, m.f_small)
        y += 28
        m.text(surf, tr("Regions"), (x, y), ui.INK, m.f_bold)
        y += 24
        for q in res["regions"]:
            m.text(surf, f"{q['region']:<16} {q['neurons']:>8,} {tr('neurons')}   {tr('in')} {q['mean_in_degree']:.1f} / {tr('out')} {q['mean_out_degree']:.1f}   "
                         f"{q['share_out_staying_inside'] * 100:.0f}% {tr('of output stays inside')}   {q['inhibitory_share_of_out_synapses'] * 100:.0f}% {tr('inhibitory')}",
                   (x + 8, y), ui.TEXT, m.f_small)
            y += 19
        y += 6
        if st.msg:
            m.text(surf, st.msg, (x, y), ui.GOOD, m.f_small)
            y += 22
    surf.set_clip(prev)
    m.clip = None
    _end(m, surf, rect, body, key, y + off, off, st)


def _hist(m, surf, x, y, w, hin, hout) -> int:
    bars = max(len(hin), len(hout))
    h = 70
    top = max(max(b["count"] for b in hin), max(b["count"] for b in hout), 1)
    bw = max(6, min(36, (w - 20) // max(1, bars)))
    pygame.draw.rect(surf, (18, 22, 30), (x, y, bars * bw * 2 + 10, h + 8), border_radius=6)
    for i, (a, b) in enumerate(zip(hin + [dict(count=0)] * (bars - len(hin)), hout + [dict(count=0)] * (bars - len(hout)))):
        for k, (bb, col) in enumerate(((a, (57, 135, 229)), (b, (217, 89, 38)))):
            hh = int(h * bb["count"] / top)
            pygame.draw.rect(surf, col, (x + 5 + i * bw * 2 + k * bw, y + 4 + h - hh, bw - 1, hh))
    m.text(surf, tr("blue: in-degree, orange: out-degree; bins 0, 1, 2-3, 4-7, ..."), (x + bars * bw * 2 + 18, y + 4), ui.LABEL, m.f_small)
    return y + h + 20


def _export_net(m, st: _St, res) -> None:
    from kickthefly.lab import netsci, recorder

    try:
        folder = recorder.exports_dir() / f"netsci-{res['brain']}"
        files = netsci.export_csv(res, folder)
        st.msg = tr("Wrote {n} CSV files to {p}").format(n=len(files), p=folder)
    except OSError as e:
        st.error = str(e)


# --- sleep deprivation -------------------------------------------------------------------------------------------------------------
def _start_sleep(st: _St) -> None:
    from kickthefly.lab import sleepdep

    seeds = list(range(1000, 1000 + st.seeds))

    def work(job: BgJob):
        return sleepdep.run(seeds, workers=_workers(), cancel=job.cancel, progress=lambda d, n, label: job.update(d / max(1, n), label))

    st.error = ""
    st.job = BgJob("Sleep deprivation", work).start()


def page_sleepdep(m: ui.Menu, surf, rect, mouse) -> None:
    from kickthefly.lab import labstats, sleepdep

    st = _st(m)
    _collect(st, "sleep")
    key = "lab_sleepdep"
    body, off = _chrome(m, surf, rect, tr("SLEEP DEPRIVATION"), tr("Keep a fly awake through the night, then measure its rebound sleep against an undisturbed control of the same seed."), st, key)
    busy = st.job is not None and st.job.running
    x, y = body.x + 8, body.y + 4
    m.slider(surf, (x, y, 260, 32), st.seeds, 3, 10, 1, "{:.0f} flies", lambda v: setattr(st, "seeds", int(v)), lambda: None, id="sd_n", enabled=not busy,
             tip=tr("Seeds 1000 to 1000+n-1 (the validation seeds). A paired test needs at least 7 to reach p < 0.01."))
    m.button(surf, (x + 290, y, 190, 32), tr("Run the assay"), lambda: _start_sleep(st), style="primary", id="sd_run", enabled=not busy,
             tip=tr("Each fly runs twice (deprived and control), about 2 minutes of simulated time each, in worker processes."))
    m.clip = body
    prev = surf.get_clip()
    surf.set_clip(body)
    y += 44 - off
    m.text(surf, tr("What is what in this assay:"), (x, y), ui.INK, m.f_bold)
    y += 24
    for label, tag in sleepdep.COMPONENTS:
        xx = _tags(m, surf, x + 8, y, [next(t for t in ("GAME RULE", "CONNECTOME", "MODEL PREDICTION") if tag.startswith(t))])
        m.text(surf, label, (xx + 4, y + 2), ui.TEXT, m.f_small)
        y += 24
    y += 8
    res = st.sleep
    if res is None:
        m.text(surf, tr("Not run yet."), (x, y), ui.LABEL, m.f_text)
        y += 30
    else:
        mm = res["mean"]
        for ln in (f"{tr('Sleep in the deprivation window')}: {tr('control')} {mm['deprivation_sleep_control']:.1f} s, {tr('deprived')} {mm['deprivation_sleep_deprived']:.1f} s",
                   f"{tr('Sleep pressure at its end')} (GAME RULE): {mm['pressure_control']:.2f} vs {mm['pressure_deprived']:.2f}   ·   {tr('dFB level')} (CONNECTOME): {mm['dfb_control']:.2f}x vs {mm['dfb_deprived']:.2f}x",
                   f"{tr('Sleep in the recovery window')}: {tr('control')} {mm['recovery_sleep_control']:.1f} s, {tr('deprived')} {mm['recovery_sleep_deprived']:.1f} s"):
            m.text(surf, ln, (x, y), ui.TEXT, m.f_small)
            y += 22
        for cid, c in res["criteria"].items():
            extra = f"  {labstats.fmt_p(c['p'])}" if "p" in c else ""
            m.text(surf, f"{cid} {'PASS' if c['passed'] else 'FAIL'}  {c['label']}{extra}", (x + 8, y), ui.GOOD if c["passed"] else ui.BAD, m.f_small)
            y += 20
        m.text(surf, tr("A rebound is expected from the pressure rule (a game rule): this shows it works through the real dFB neurons."), (x, y + 6), ui.LABEL, m.f_small)
        y += 34
    surf.set_clip(prev)
    m.clip = None
    _end(m, surf, rect, body, key, y + off, off, st)


# --- sensitivity analysis ----------------------------------------------------------------------------------------------------------
def _start_sens(st: _St) -> None:
    from kickthefly.lab import recorder, sensitivity, validation

    smoke = st.sens_smoke
    folder = recorder.exports_dir() / ("sensitivity-smoke" if smoke else "sensitivity")

    def work(job: BgJob):
        res = sensitivity.run(params=["noise_std"] if smoke else None, tests=["looming_escape", "sugar_feeding"] if smoke else None,
                              seeds=validation.SEEDS[:3] if smoke else None, values={"noise_std": [0.0375, 0.075]} if smoke else None,
                              workers=_workers(), folder=folder, resume=True, progress=lambda d, n, label: job.update(d / max(1, n), label),
                              cancel=job.cancel)
        sensitivity.save(res, folder)
        return res

    st.error = ""
    st.job = BgJob("Sensitivity", work).start()


def page_sensitivity(m: ui.Menu, surf, rect, mouse) -> None:
    from kickthefly.lab import sensitivity

    st = _st(m)
    _collect(st, "sens")
    key = "lab_sensitivity"
    body, off = _chrome(m, surf, rect, tr("SENSITIVITY ANALYSIS"), tr("Vary each LIF parameter and re-run the validated behaviors with validation's own criteria. Analysis only: the defaults never change."), st, key)
    busy = st.job is not None and st.job.running
    x, y = body.x + 8, body.y + 4
    m.button(surf, (x, y, 250, 32), tr("Quick check (3 seeds)") if st.sens_smoke else tr("Full grid (10 seeds)"), lambda: setattr(st, "sens_smoke", not st.sens_smoke),
             id="se_mode", enabled=not busy, tip=tr("The quick check varies one parameter at two values on 2 behaviors and 3 seeds (underpowered: a FAIL means 'cannot pass at n = 3'). The full grid is hours; run it headless with --sensitivity."))
    m.button(surf, (x + 260, y, 190, 32), tr("Run"), lambda: _start_sens(st), style="primary", id="se_run", enabled=not busy,
             tip=tr("Resumable: finished cells are kept in your exports folder."))
    m.clip = body
    prev = surf.get_clip()
    surf.set_clip(body)
    y += 44 - off
    xx = _tags(m, surf, x, y, ["CONNECTOME", "GAME RULE", "MODEL PREDICTION"])
    m.text(surf, tr("pathways (CONNECTOME); ranges and criteria (GAME RULE); every cell (MODEL PREDICTION)"), (xx + 4, y + 2), ui.LABEL, m.f_small)
    y += 30
    res = st.sens
    if res is None:
        m.text(surf, tr("Not run yet. Parameters:") + " " + ", ".join(p["label"] for p in sensitivity.PARAMETERS), (x, y), ui.LABEL, m.f_small)
        y += 28
    else:
        y = _heatmap(m, surf, x, y, body.w - 30, res)
    surf.set_clip(prev)
    m.clip = None
    _end(m, surf, rect, body, key, y + off, off, st)


def _heatmap(m, surf, x, y, w, res) -> int:
    from kickthefly.lab import sensitivity

    pal = sensitivity.PALETTES.get(m.host.cfg["access.palette"], sensitivity.PALETTES["default"])
    ok, bad, ok_t, bad_t, _ = pal
    cols, grid = sensitivity.heatmap_cells(res)
    lw = 200
    cw = max(64, min(110, (w - lw) // max(1, len(cols))))
    if res.get("underpowered"):
        m.text(surf, tr("UNDERPOWERED: fewer than 7 seeds cannot reach p < 0.01, so a FAIL only means 'cannot pass at this n'."), (x, y), ui.AMBER, m.f_small)
        y += 22
    for j, c in enumerate(cols):
        m.text(surf, c.replace("_", " ")[:cw // 7], (x + lw + j * cw + 2, y), ui.LABEL, m.f_small)
    y += 20
    for label, by in grid:
        m.text(surf, label, (x, y + 3), ui.TEXT, m.f_small)
        for j, c in enumerate(cols):
            r = by.get(c)
            box = pygame.Rect(x + lw + j * cw, y, cw - 2, 20)
            if r is None:
                pygame.draw.rect(surf, (40, 44, 54), box)
                continue
            pygame.draw.rect(surf, ok if r["passed"] else bad, box)
            m.text(surf, f"{'PASS' if r['passed'] else 'FAIL'} {r['effect']:.2f}", box.center, ok_t if r["passed"] else bad_t, m.f_small, "center")
            m._register(box, "label", id=("se_cell", label, c), tip=f"{c}: {'PASS' if r['passed'] else 'FAIL'}, {r['metric']} {r['effect']:.2f} (control {r['control']:.2f}), "
                                                                 f"dz {r['effect_size_dz']:.2f}, p {r['p_value']:.4f}, n {r['n']}")
        y += 22
    m.text(surf, tr("Each cell: PASS or FAIL by the validation criteria, and the effect (drive ratio, or T-maze PI). Hover for the control, effect size and p."), (x, y + 6), ui.LABEL, m.f_small)
    return y + 34
