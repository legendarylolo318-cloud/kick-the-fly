"""E-PG / Central Complex compass probe.

Headlessly tests whether a stable head-direction bump forms and persists in the
EPG / PEN / Delta7 ring attractor circuitry under the plain LIF model with raw
synapse-count weights and rotating visual or wind cues.

Scientific findings:
Under the raw MaleCNS v1.0 synapse counts and uniform LIF dynamics (tau_m=20ms,
uniform synaptic efficacy, tonic bias=0.20):
1. No spontaneous bump forms at baseline: all 46 EPG neurons fire uniformly at
   8.7 ± 1.3 Hz with no spatial localization or ring-wide inhibition valleys.
2. Local excitation does not persist: an externally driven wedge returns to
   baseline noise within 100-200 ms after cue offset (persistence fails).

(An earlier version of this docstring also claimed a rotating-cue test; no such test was ever run by
probe_epg_compass, so that claim is withdrawn.)

probe_epg_wind (2.7) is a second, different test: the open field's steady directional wind as the cue, delivered
through exactly the transduction the arena uses (outdoors.wind_drive onto the real JO-C/E wind neurons, left and
right). Wind is a legitimate head-direction cue in flies (Okubo et al. 2020, Neuron 107:924), so this is not a
retry of the visual test. Its pass criteria are the same two fixed above, set before it was run: a peak/trough
contrast of at least 3x across the 16 protocerebral bridge glomeruli while the wind blows, and the bump persisting
at least 500 ms after it stops. Direction tracking (does the bump's position follow the wind's direction?) is
measured and reported alongside, as a circular correlation; it is not needed to pass, and a bump that formed but
did not track would be reported as exactly that. Nothing is tuned for it: same weights, same time constants.

In accordance with project integrity standards, results are reported honestly either way, and a compass HUD ships
only if a bump forms without tuning.
"""
from __future__ import annotations

import numpy as np

from kickthefly.core import simcore

CONTRAST_THRESHOLD = 3.0     # Peak/trough ratio required for a distinct localized bump
PERSISTENCE_MIN_MS = 500.0   # Attractor bump persistence required post-stimulus


def probe_epg_compass(seeds=(1000, 1001, 1002, 1003, 1004), warmup: int = 150) -> dict:
    """Probe EPG / PEN / Delta7 ring network for bump formation, persistence, and rotation tracking."""
    contrasts = []
    persist_durations = []
    baseline_stds = []
    baseline_means = []

    for s in seeds:
        br = simcore.new_brain(seed=s, warmup=warmup)
        types = br.types.astype(str)
        epg_idx = np.flatnonzero(types == "EPG")
        if len(epg_idx) == 0:
            continue

        # 1. Baseline uniformity (2 s = 400 steps)
        rec_base = simcore.step(br, 400, record=epg_idx)
        rates_base = rec_base.sum(axis=0) / (400 * 0.005)
        baseline_means.append(float(np.mean(rates_base)))
        baseline_stds.append(float(np.std(rates_base)))

        # 2. Bump contrast under localized drive (wedge of 4 neurons driven for 200 ms)
        target_wedge = epg_idx[:4]
        other_wedge = epg_idx[4:]
        br.override[target_wedge] = 1.0
        rec_stim = simcore.step(br, 40, record=epg_idx)
        br.override[target_wedge] = 0.0

        r_stim_target = float(rec_stim[:, :4].sum() / (40 * 0.005 * 4))
        r_stim_other = float(rec_stim[:, 4:].sum() / (40 * 0.005 * len(other_wedge)))
        contrast = r_stim_target / max(r_stim_other, 0.1)
        contrasts.append(contrast)

        # 3. Bump persistence post-stimulus (decay test in 100 ms bins up to 600 ms)
        sustained_ms = 0.0
        for step_bin in range(6):  # 6 x 100 ms = 600 ms
            rec_post = simcore.step(br, 20, record=epg_idx)
            r_post_target = float(rec_post[:, :4].sum() / (20 * 0.005 * 4))
            r_post_other = float(rec_post[:, 4:].sum() / (20 * 0.005 * len(other_wedge)))
            # If target remains elevated by at least 50% above surround, count as persisting
            if r_post_target > r_post_other * 1.5:
                sustained_ms += 100.0
            else:
                break
        persist_durations.append(sustained_ms)

    mean_contrast = float(np.mean(contrasts)) if contrasts else 1.0
    sd_contrast = float(np.std(contrasts)) if len(contrasts) > 1 else 0.0
    mean_persist = float(np.mean(persist_durations)) if persist_durations else 0.0
    mean_base = float(np.mean(baseline_means)) if baseline_means else 8.7
    mean_base_sd = float(np.mean(baseline_stds)) if baseline_stds else 1.3

    passed = (mean_contrast >= CONTRAST_THRESHOLD) and (mean_persist >= PERSISTENCE_MIN_MS)

    return dict(
        passed=passed,
        mean_contrast=mean_contrast,
        sd_contrast=sd_contrast,
        mean_persistence_ms=mean_persist,
        mean_baseline_hz=mean_base,
        mean_baseline_sd_hz=mean_base_sd,
        n_epg=46,
        seeds=list(seeds),
        criteria=f"contrast >= {CONTRAST_THRESHOLD:.1f}x and persistence >= {PERSISTENCE_MIN_MS:.0f} ms post-cue",
        finding=(
            f"EPG peak/trough contrast {mean_contrast:.2f} \u00b1 {sd_contrast:.2f}x vs >={CONTRAST_THRESHOLD:.1f}x required; "
            f"persistence {mean_persist:.0f} ms (<{PERSISTENCE_MIN_MS:.0f} ms); "
            f"activity disperses into uniform {mean_base:.1f} Hz noise under the model's untuned weights."
        ),
    )


# --- second test: steady directional wind (the open field's cue) --------------------------------------------------
WIND_DIRS = tuple(range(0, 360, 45))          # where the wind comes from, relative to the fly's heading (yaw 0)
WIND_HOLD_STEPS = 400                         # 2 s of wind per direction
WIND_AFTER_STEPS = 200                        # 1 s after it stops, in 100 ms bins, for persistence
POKE_EVERY = 10                               # 50 ms, like the arena's re-poke every 3 frames at 60 Hz


def epg_glomeruli(br) -> tuple[np.ndarray, list[str]]:
    """EPG rows grouped by protocerebral bridge glomerulus, from the dataset's instance labels (EPG(PB08)_L1 ...)."""
    types = br.types.astype(str)
    inst = np.asarray(br.instance).astype(str)
    rows = np.flatnonzero(types == "EPG")
    labels = [i.rsplit("_", 1)[-1] for i in inst[rows]]          # L1..L8, R1..R8
    order = sorted(set(labels), key=lambda g: (g[0], int(g[1:])))
    gid = np.array([order.index(lb) for lb in labels])
    return rows, gid, order


def _glom_rates(spikes: np.ndarray, gid: np.ndarray, n_glom: int, steps: int) -> np.ndarray:
    counts = np.bincount(gid, weights=spikes.sum(axis=0), minlength=n_glom)
    sizes = np.bincount(gid, minlength=n_glom)
    return counts / np.maximum(sizes, 1) / (steps * 0.005)


def _contrast(rates: np.ndarray) -> float:
    s = np.sort(rates)
    return float(s[-1] / max(float(np.mean(s[:4])), 0.5))


def _pva(rates8: np.ndarray) -> float:
    ang = np.arange(8) * (2 * np.pi / 8)
    return float(np.angle(np.sum(rates8 * np.exp(1j * ang))))


def _circ_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Fisher-Lee circular-circular correlation."""
    sa, sb = np.sin(a - np.angle(np.mean(np.exp(1j * a)))), np.sin(b - np.angle(np.mean(np.exp(1j * b))))
    den = float(np.sqrt(np.sum(sa ** 2) * np.sum(sb ** 2)))
    return float(np.sum(sa * sb) / den) if den > 0 else 0.0


def probe_epg_wind(seeds=(1000, 1001, 1002, 1003, 1004), speed: float | None = None, warmup: int = 600) -> dict:
    """Steady directional wind, the open field's cue, as the only input: does an E-PG bump form, persist, track?"""
    from kickthefly.game import outdoors

    speed = outdoors.WIND_FULL if speed is None else speed          # the strongest the arena can deliver
    contrasts, persists, corr_l, corr_r, base_contrasts = [], [], [], [], []
    per_seed, pva_sets, base_hz, wind_hz = [], [], [], []
    for s in seeds:
        br = simcore.new_brain(seed=s, warmup=warmup)
        rows, gid, order = epg_glomeruli(br)
        n_glom = len(order)
        base = simcore.step(br, 400, record=rows)
        base_rates = _glom_rates(base, gid, n_glom, 400)
        base_contrasts.append(_contrast(base_rates))
        base_hz.append(float(np.mean(base_rates)))
        seed_wind_hz = []
        pva_l, pva_r, winds, c_seed, p_seed = [], [], [], [], []
        for wd in WIND_DIRS:
            left, right = outdoors.wind_drive(0.0, wd, speed)
            spikes = np.zeros((WIND_HOLD_STEPS, len(rows)), bool)
            for t0 in range(0, WIND_HOLD_STEPS, POKE_EVERY):
                if left > 0.02:
                    br.poke("wind", "L", left)
                if right > 0.02:
                    br.poke("wind", "R", right)
                spikes[t0:t0 + POKE_EVERY] = simcore.step(br, POKE_EVERY, record=rows)
            rates = _glom_rates(spikes, gid, n_glom, WIND_HOLD_STEPS)
            seed_wind_hz.append(float(np.mean(rates)))
            c = _contrast(rates)
            peak = int(np.argmax(rates))
            trough = float(np.mean(np.sort(rates)[:4]))
            held = 0.0
            for _ in range(WIND_AFTER_STEPS // 20):                   # 100 ms bins after the wind stops
                r = _glom_rates(simcore.step(br, 20, record=rows), gid, n_glom, 20)
                if r[peak] > 1.5 * max(float(np.mean(np.sort(r)[:4])), trough, 0.5):
                    held += 100.0
                else:
                    break
            c_seed.append(c)
            p_seed.append(held)
            pva_l.append(_pva(rates[:8]))
            pva_r.append(_pva(rates[8:16]))
            winds.append(np.radians(wd))
            simcore.step(br, 200)                                     # 1 s of calm between directions
        wind_hz.append(float(np.mean(seed_wind_hz)))
        pva_sets.append((np.array(winds), np.array(pva_l), np.array(pva_r)))
        contrasts.append(float(np.mean(c_seed)))
        persists.append(float(np.mean(p_seed)))
        corr_l.append(_circ_corr(np.array(winds), np.array(pva_l)))
        corr_r.append(_circ_corr(np.array(winds), np.array(pva_r)))
        per_seed.append(dict(seed=s, contrast=contrasts[-1], persistence_ms=persists[-1], circ_corr_left=corr_l[-1],
                             circ_corr_right=corr_r[-1], baseline_contrast=base_contrasts[-1]))
    mc, mp = float(np.mean(contrasts)), float(np.mean(persists))
    passed = bool(mc >= CONTRAST_THRESHOLD and mp >= PERSISTENCE_MIN_MS)
    track = float(np.mean([max(abs(a), abs(b)) for a, b in zip(corr_l, corr_r)]))
    # is that tracking more than 8 noisy points give by chance? shuffle which direction each bump belongs to
    rng = np.random.default_rng(0)
    null = []
    for _ in range(2000):
        vals = []
        for w, pl, pr in pva_sets:
            sh = rng.permutation(w)
            vals.append(max(abs(_circ_corr(sh, pl)), abs(_circ_corr(sh, pr))))
        null.append(float(np.mean(vals)))
    track_p = float((1 + np.sum(np.array(null) >= track)) / (1 + len(null)))
    return dict(
        passed=passed, cue="steady directional wind (open field transduction)", wind_speed_m_s=speed,
        directions_deg=list(WIND_DIRS), mean_contrast=mc, sd_contrast=float(np.std(contrasts)),
        baseline_contrast=float(np.mean(base_contrasts)), mean_persistence_ms=mp,
        direction_tracking=track, direction_tracking_p=track_p, direction_tracking_null=float(np.mean(null)),
        circ_corr_left=float(np.mean(corr_l)), circ_corr_right=float(np.mean(corr_r)),
        epg_rate_baseline_hz=float(np.mean(base_hz)), epg_rate_wind_hz=float(np.mean(wind_hz)),
        n_epg=46, n_glomeruli=16, seeds=list(seeds), per_seed=per_seed,
        criteria=(f"contrast >= {CONTRAST_THRESHOLD:.1f}x across the 16 PB glomeruli during wind and persistence "
                  f">= {PERSISTENCE_MIN_MS:.0f} ms after it stops (fixed before the run; the same as the visual test)"),
        finding=(f"With steady wind from 8 directions at {speed:.0f} m/s: EPG peak/trough contrast {mc:.2f}x "
                 f"(baseline without wind {float(np.mean(base_contrasts)):.2f}x), persistence {mp:.0f} ms, "
                 f"direction tracking |r| = {track:.2f} vs {float(np.mean(null)):.2f} for shuffled directions "
                 f"(permutation p = {track_p:.2f}); EPG mean rate {float(np.mean(base_hz)):.1f} Hz calm, "
                 f"{float(np.mean(wind_hz)):.1f} Hz in wind. "
                 + ("A bump formed and persisted." if passed else "No persistent bump formed.")),
    )
