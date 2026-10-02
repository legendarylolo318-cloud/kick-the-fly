"""The hum demo (3.0 day 3): what does humming do to the fly's brain, and does it reach the courtship song pathway?

A synthetic hum (no microphone, no device, deterministic) goes through the same Analyzer that the microphone uses, becomes
current on the real JO-A and JO-B neurons (core/mic.py), and the demo reads out the courtship pathway: the P1 cluster (the pC1
neurons the dataset calls P1), pIP10 and the ps1 wing motor neurons that the game's SONG reaction listens to.

What is what:
  CONNECTOME      the neurons and everything the simulation does between JO-A/B and the readouts.
  GAME RULE       the sound-to-current transduction (core/mic.py), the demo's sounds, and THRESH["song"] (the multiple of calm
                  at which the game calls the ps1 firing SONG).
  LITERATURE      the song's rhythm the demo's pulse train imitates. Zhou et al. 2015, eLife 4:e08477 ("Central neural circuitry
                  mediating courtship song perception in male Drosophila"; full text read by the 3.0 day 3 review, PMC4575990):
                  D. melanogaster pulse song has a ~35 ms interpulse interval and the sine song is ~160 Hz (introduction, citing
                  Bennet-Clark & Ewing 1967 and others); their synthetic pulse song used a 220 Hz carrier and their sine stimulus
                  140 Hz (methods). The same paper reports that pC1 neurons respond to pulse song with band-pass tuning to
                  interpulse intervals of about 35-65 ms, and only above 80 dB (Figure 6): a selectivity this model does NOT
                  reproduce (see the demo's result). Lillvis et al. 2024, Curr Biol 34 (doi 10.1016/j.cub.2024.01.015) is about
                  the song motor circuits; its abstract gives no song numbers and its full text was not read.
  MODEL PREDICTION the response ratios this prints. They are what this model does with this model of a sound, not what a real
                  fly's circuit does.

Pre-registered before the validation run (fixed here; design chosen on exploration seeds 0-9, scored once on seeds 1000-1009):
  H1  the 200 Hz pulse train at a 35 ms interpulse interval drives the P1 cluster to at least 1.2x its calm rate in at least 8
      of 10 flies
  H2  silence does not (P1 below 1.2x its calm rate) in at least 9 of 10 flies
  H3  the ps1 song motor neurons stay below THRESH["song"] (1.8x calm) during that pulse train in at least 8 of 10 flies: the
      expectation from exploration seeds 0-2 is that the wiring carries a hum to P1 but not on to the song motor neurons
Nothing else is a criterion; every other number is reported. If H3 fails, the answer to "does humming drive the song?" is yes
in this model, and the handoff says so.
"""
from __future__ import annotations

import numpy as np

from kickthefly.core import mic, simcore

CRITERIA = (
    "H1  the 200 Hz pulse train at a 35 ms interpulse interval drives the P1 cluster to >= 1.2x its calm rate in >= 8 of 10 flies",
    "H2  silence does not (P1 < 1.2x calm) in >= 9 of 10 flies",
    "H3  the ps1 song motor neurons stay below THRESH['song'] (1.8x calm) during that pulse train in >= 8 of 10 flies",
)
P1_RATIO, H1_FLIES, H2_FLIES, H3_FLIES = 1.2, 8, 9, 8
HUM_AMP = 0.1                       # waveform amplitude (full scale 1.0): a steady hum at this level drives JO-A fully
SECONDS = 3.0
PRE_STEPS = 600
READOUTS = ("p1", "pip10", "ps1")
STEP_S = 0.005
CONDITIONS = {
    "silence": None,
    "steady_200": dict(hz=200.0, ipi_ms=None),
    "pulses_200_ipi35": dict(hz=200.0, ipi_ms=35.0),
    "pulses_200_ipi70": dict(hz=200.0, ipi_ms=70.0),
    "steady_50": dict(hz=50.0, ipi_ms=None),
    "steady_600": dict(hz=600.0, ipi_ms=None),
}


def sound(cond: str, seconds: float = SECONDS, amp: float = HUM_AMP, rate: float = mic.RATE) -> np.ndarray:
    spec = CONDITIONS[cond]
    if spec is None:
        return np.zeros(int(seconds * rate), np.float32)
    return mic.hum(spec["hz"], seconds, rate, amp, spec["ipi_ms"])


def hear(br, x: np.ndarray, rate: float = mic.RATE, sensitivity: float = 1.0, watch: dict | None = None) -> dict:
    """Play a waveform to a brain: analyse each chunk, put its drive on JO-A/B, and step the brain for that chunk's time
    (fractions of a step carried over). Returns what the analysis saw and the spikes of the `watch` row sets."""
    from kickthefly.core import mic as m

    an = m.Analyzer(rate, sensitivity)
    rows = m.jo_rows(br)
    counts = {k: 0 for k in (watch or {})}
    steps_done, carry = 0, 0.0
    peaks, drive_a, drive_b = [], [], []
    for ch in m.chunks(np.asarray(x, np.float32)):
        r = an.push(ch)
        m.apply(br, r, rows)
        carry += len(ch) / rate / STEP_S
        n = int(carry)
        carry -= n
        for _ in range(n):
            br._step()
            sp = br.sim.spikes
            for k, rr in (watch or {}).items():
                counts[k] += int(np.count_nonzero(sp[rr]))
        steps_done += n
        if r.peak_hz:
            peaks.append(r.peak_hz)
        drive_a.append(r.drive_a)
        drive_b.append(r.drive_b)
    m.release(br)
    return dict(steps=steps_done, counts=counts, peak_hz=float(np.median(peaks)) if peaks else 0.0,
                mean_drive_a=float(np.mean(drive_a)) if drive_a else 0.0, mean_drive_b=float(np.mean(drive_b)) if drive_b else 0.0)


def demo_fly(seed: int, conditions=tuple(CONDITIONS), seconds: float = SECONDS, amp: float = HUM_AMP, brain=None,
             wiring=None, params: dict | None = None) -> dict:
    """One fly hears each condition in turn (calm, then the sound, then calm again). Returns per condition, per readout:
    calm Hz, Hz during the sound and the ratio."""
    from kickthefly.lab import assays

    bad = [c for c in conditions if c not in CONDITIONS]
    if bad:
        raise ValueError(f"unknown condition {bad[0]!r}; use one of {', '.join(CONDITIONS)}")
    br = brain or simcore.new_brain(seed=seed, params=params, wiring=wiring)
    g = assays.groups(br)
    watch = {k: g[k] for k in READOUTS}
    a, b = mic.jo_rows(br)
    watch_jo = dict(watch, jo_a=a, jo_b=b)
    assays.rest(br, 400)
    out = {}
    for cond in conditions:
        base = {k: 0 for k in watch_jo}
        for _ in range(PRE_STEPS):
            br._step()
            for k, rr in watch_jo.items():
                base[k] += int(np.count_nonzero(br.sim.spikes[rr]))
        res = hear(br, sound(cond, seconds, amp), watch=watch_jo)
        row = {}
        for k, rr in watch_jo.items():
            b_hz = assays.hz(base[k], len(rr), PRE_STEPS)
            d_hz = assays.hz(res["counts"][k], len(rr), max(1, res["steps"]))
            row[k] = dict(calm_hz=b_hz, hz=d_hz, ratio=d_hz / max(b_hz, 0.05))
        row["analysis"] = dict(peak_hz=res["peak_hz"], drive_a=res["mean_drive_a"], drive_b=res["mean_drive_b"])
        out[cond] = row
        assays.rest(br, PRE_STEPS)
    return dict(seed=seed, seconds=seconds, amp=amp, conditions=out, n_jo_a=int(len(a)), n_jo_b=int(len(b)))


def summarize(flies: list[dict]) -> dict:
    """Per condition and readout: mean ratio with a 95% CI across flies, and how many flies cleared 1.2x."""
    from kickthefly.lab import labstats

    rows = []
    for cond in flies[0]["conditions"]:
        r = {"condition": cond}
        for k in READOUTS + ("jo_a", "jo_b"):
            ratios = [f["conditions"][cond][k]["ratio"] for f in flies]
            r[k] = dict(ratio=labstats.mean_ci(ratios), flies_over_1_2=int(sum(x >= P1_RATIO for x in ratios)))
        r["peak_hz"] = float(np.median([f["conditions"][cond]["analysis"]["peak_hz"] for f in flies]))
        rows.append(r)
    # 3.0 day 3 review: the pre-registered verdict goes into the run's own summary (it was only ever computed by hand and in tests)
    scored = all(c in flies[0]["conditions"] for c in ("pulses_200_ipi35", "silence"))
    return dict(metric="hum -> JO-A/B -> courtship pathway: firing ratio vs calm (MODEL PREDICTION)", rows=rows, n=len(flies),
                criteria=list(CRITERIA), verdict=verdict(flies) if scored else None,
                per_fly=[headline(f) for f in flies])        # 3.0 day 3 review: the export needs it (the protocol crashed there)


def headline(fly: dict) -> float:
    """One fly's headline: P1's firing ratio during the 200 Hz pulse train (NaN when that condition was not run)."""
    c = fly["conditions"].get("pulses_200_ipi35")
    return float(c["p1"]["ratio"]) if c is not None else float("nan")


def verdict(flies: list[dict]) -> dict:
    """Score H1-H3 (CRITERIA). Meaningful for ten flies, seeds 1000-1009."""
    from kickthefly.game import kick_the_fly as k

    n = len(flies)
    p = [f["conditions"]["pulses_200_ipi35"] for f in flies]
    s = [f["conditions"]["silence"] for f in flies]
    h1n = sum(c["p1"]["ratio"] >= P1_RATIO for c in p)
    h2n = sum(c["p1"]["ratio"] < P1_RATIO for c in s)
    h3n = sum(c["ps1"]["ratio"] < k.THRESH["song"] for c in p)
    need = lambda k_, of=10: int(np.ceil(k_ * n / of))  # noqa: E731
    return dict(H1=bool(h1n >= need(H1_FLIES)), H2=bool(h2n >= need(H2_FLIES)), H3=bool(h3n >= need(H3_FLIES)),
                p1_flies_over=int(h1n), silence_flies_under=int(h2n), ps1_flies_under_song_threshold=int(h3n), flies=n,
                passed=bool(h1n >= need(H1_FLIES) and h2n >= need(H2_FLIES) and h3n >= need(H3_FLIES)))
