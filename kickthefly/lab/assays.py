"""Assays: T-maze olfactory conditioning, looming escape, and sugar response, run headless in lockstep.

The in-game challenges (challenges.py), Lab results, repeated trials, protocols and the validation suite all use
these functions. Every run steps brains synchronously (simcore.py), so the same seed gives the same result.

What comes from the connectome and what is a game rule, per assay:

T-maze (Tully & Quinn 1985)
  Connectome: the odors reach Kenyon cells through the real olfactory pathway; the plastic synapses are the
    connectome's 41,495 KC -> MBON synapses, gated by its own PPL1/PAM dopamine -> MBON wiring (memory.py).
  Game rules: each odor is a fixed set of 6 olfactory glomeruli (not a real chemical's receptor profile); the shock
    drives PPL1 dopamine neurons and leg touch neurons directly; the plasticity rule and its rate (memory.py); and the
    choice. The sim has no T-maze body, so at the choice point the fly smells each arm and picks the one whose learned
    approach drive (liking minus fear, read from the synapses) is higher, plus decision noise. The standard metric is
    computed exactly as in the paper: PI = (flies choosing CS- minus flies choosing CS+) / all choices, averaged over
    reciprocal halves (each odor serving once as CS+). The control is explicitly unpaired: the same odors and the same
    shocks, but the shocks fall in the rest periods.

Looming escape (von Reyn et al. 2014; Ache et al. 2019)
  Connectome: LPLC2/LC4 -> giant fiber DNp01 and everything the sim does in between.
  Game rules: how an approaching object becomes LPLC2/LC4 drive (angular expansion speed above LOOM_MIN, full at
    LOOM_MIN + LOOM_FULL; the same code path as the game), and escaping when DNp01 exceeds THRESH["escape"] x calm.
    Because of that transduction threshold, part of the speed dependence is a game rule, not a measurement.

Sugar response (Shiu et al. 2024; sugar SEL projection neurons from Yao & Scott 2022)
  Connectome: which neurons are "sugar pathway" GRNs, and MN9's response. The dataset doesn't label GRNs by taste
    quality, so the sugar set is defined from wiring: the 30 head gustatory neurons (LB*, PhG*, claw_tpGRN,
    dorsal_tpGRN) with the most one- and two-synapse input to the sugar SEL PNs (GNG540, GNG550, annotated "Yao & Scott
    2022: Sugar SEL PN") minus their input to the bitter SEL neuron DNg28 ("Yao & Scott 2022: Bitter-SEL"). The bitter
    set is the 30 with the opposite preference. MN9 (a proboscis motor neuron) is never used to choose them.
  Game rules: dose is the share of sugar-pathway GRNs driven; a "proboscis extension" is MN9 firing at least
    PER_RATIO x its rate in the second before.
"""
from __future__ import annotations

import math

import numpy as np

from kickthefly.core import simcore

STEPS_PER_TICK = (4, 3, 3)          # 10 brain steps (50 ms) per 3 game ticks (1/20 s)
SWEET_N = 30
PER_RATIO = 1.5
ODOR_GLOMERULI_SEED = 1985           # which 6 glomeruli each T-maze odor uses (game rule)
P1_TYPES = ("pC1_1a", "pC1_1b", "pC1_2a", "pC1_2a/2b", "pC1_2b", "pC1_2c", "pC1_3a", "pC1_3b", "pC1_3c", "pC1_5a",
            "pC1_5b", "pC1_6a", "pC1_7a", "pC1_7b", "pC1_12a", "pC1_13a", "pC1_14a", "pC1_14b", "pC1_15a", "pC1_15b",
            "pC1_15c", "pC1_16a", "pC1_16b", "pC1_17a", "pC1_17b")     # synonym "Cachero 2010: pMP-e; Yu 2010: pMP4"
OPTOMOTOR_YAW = 3.0                  # rad/s, rightward: the validation's wide-field rotation (~170 deg/s)


# --- neuron groups -------------------------------------------------------------------------------------------------
def groups(br) -> dict[str, np.ndarray]:
    """Named neuron sets for assays and validation, computed once per brain from the pack's labels and wiring."""
    cached = getattr(br, "_assay_groups", None)
    if cached is not None:
        return cached
    t, sc = br.types, br.superclass
    sub = getattr(br, "subclass", np.full(br.n, "", dtype="<U1"))

    def T(*names):
        return np.flatnonzero(np.isin(t, names))

    motor = sc == "vnc_motor"
    head_gust = np.flatnonzero((np.char.startswith(t, "LB") | np.char.startswith(t, "PhG") |
                                np.char.startswith(t, "claw_tpGRN") | np.char.startswith(t, "dorsal_tpGRN"))
                               & (sc == "cb_sensory"))
    g = dict(
        dnp01=T("DNp01"), loom=np.concatenate([T("LPLC2"), T("LC4")]), mdn=T("MDN"), dnp09=T("DNp09"),
        adn=T("DNg62", "DNge078"),                       # "Hampel 2015: aDN1" / "aDN2" in the dataset's synonyms
        jo_ce=np.flatnonzero((np.char.startswith(t, "JO-C") | np.char.startswith(t, "JO-E")) & (np.char.find(sc, "sensory") >= 0)),
        mn9=T("MN9"), sugar_pn=T("GNG540", "GNG550"), bitter_sel=T("DNg28"),
        mn_front=np.flatnonzero(motor & (sub == "fl")), mn_mid=np.flatnonzero(motor & (sub == "ml")),
        mn_hind=np.flatnonzero(motor & (sub == "hl")), head_gust=head_gust,
        dn=np.flatnonzero(sc == "descending_neuron"), sensory=np.flatnonzero(np.char.find(sc, "sensory") >= 0),
        vpn=np.flatnonzero(sc == "visual_projection"),
        pip10=T("pIP10"), ps1=np.flatnonzero(motor & (t == "ps1 MN")),
        song_wm=np.flatnonzero(motor & (sub == "wm")),
        dng28=T("DNg28"),
        co2_orn=np.flatnonzero((t == "ORN_V") & (np.char.find(sc, "sensory") >= 0)),
        co2_pn=np.flatnonzero(np.isin(t, ("V_ilPN", "V_l2PN"))),
        trn_vp2=np.flatnonzero((t == "TRN_VP2") & (np.char.find(sc, "sensory") >= 0)),
        vp2_pn=np.flatnonzero(np.isin(t, ("VP2_adPN", "VP2_l2PN", "VP2+_adPN", "VP1m+VP2_lvPN1", "VP1m+VP2_lvPN2"))),
        trn_vp3=np.flatnonzero(np.isin(t, ("TRN_VP3a", "TRN_VP3b")) & (np.char.find(sc, "sensory") >= 0)),
        vp3_pn=np.flatnonzero(np.isin(t, ("VP3+_l2PN", "VP3+_vPN", "VP1l+VP3_ilPN", "VP3+VP1l_ivPN", "VP5+VP3_l2PN"))),
        # cVA pheromone circuit (Task 1): Or67d ORNs -> DA1 glomerulus projection neurons -> lateral horn / aSP targets
        or67d_orn=np.flatnonzero((t == "ORN_DA1") & (np.char.find(sc, "sensory") >= 0)),
        da1_pn=np.flatnonzero(np.isin(t, ("DA1_lPN", "DA1_vPN", "M_lvPNm43", "M_lvPNm45"))),
        da1_lh_asp=np.flatnonzero(np.isin(t, ("LHAV4a4", "LHAV4c1", "LH008m"))),
        # Foreleg sex pheromone circuit (Task 2): putative ppk23/ppk25 foreleg GRNs (LgLG5..8) -> vAB3/PPN1
        foreleg_pheromone_grn=T("LgLG5", "LgLG6", "LgLG7", "LgLG8"),
        vab3_ppn1=T("AN09B017e", "AN09B017f", "AN09B017g", "AN05B102a"),
        orn=np.flatnonzero(np.char.startswith(t, "ORN_") & (np.char.find(sc, "sensory") >= 0)),
        orn_not_da1=np.flatnonzero(np.char.startswith(t, "ORN_") & (t != "ORN_DA1") & (np.char.find(sc, "sensory") >= 0)),
        alpn=np.flatnonzero(np.char.endswith(t, "PN") & ~np.isin(t, ("DA1_lPN", "DA1_vPN", "M_lvPNm43", "M_lvPNm45"))),
    )
    # P1: the male-specific fru/dsx cluster of Kimura et al. 2008. MaleCNS v1.0 has no "P1" type or synonym, but P1 is
    # the clone Cachero 2010 calls pMP-e and Yu 2010 pMP4, and exactly these 25 pC1 types (86 neurons) carry that
    # synonym; no pC1 type mixes neurons with and without it. The pack keeps no synonyms, hence the fixed list.
    g["p1"] = T(*P1_TYPES)
    g["cb_intrinsic"] = np.flatnonzero(sc == "cb_intrinsic")
    # Seeds et al. 2014 grooming hierarchy: head bristle mechanosensory neurons (BM_*, not the taste ones) are the
    # anterior stimulus, the abdomen's sensory neurons (subclass "abdomen") minus its chemosensory (SNch) and
    # proprioceptive (SNpp) types the posterior one
    g["groom_anterior"] = np.flatnonzero(np.char.startswith(t, "BM_") & (t != "BM_Taste") & (np.char.find(sc, "sensory") >= 0))
    g["groom_posterior"] = np.flatnonzero((sub == "abdomen") & (np.char.find(sc, "sensory") >= 0)
                                          & ~np.char.startswith(t, "SNch") & ~np.char.startswith(t, "SNpp"))
    # Optomotor. T4/T5 subtype a prefers front-to-back (progressive) motion, b back-to-front (Maisak et al. 2013). A
    # world rotating rightward (clockwise from above) moves front-to-back across the right eye and back-to-front across
    # the left, so a rightward yaw stimulus is T4a/T5a_R + T4b/T5b_L, and the fly follows it by turning right.
    inst = np.array([("" if x is None else str(x)) for x in getattr(br, "instance", np.full(br.n, ""))])

    def side(types, s):
        return np.flatnonzero(np.isin(t, types) & np.char.endswith(inst, "_" + s))

    g["t45_prog_r"], g["t45_prog_l"] = side(("T4a", "T5a"), "R"), side(("T4a", "T5a"), "L")
    g["t45_regr_r"], g["t45_regr_l"] = side(("T4b", "T5b"), "R"), side(("T4b", "T5b"), "L")
    g["optomotor_right"] = np.concatenate([g["t45_prog_r"], g["t45_regr_l"]])
    g["dna_steer_r"], g["dna_steer_l"] = side(("DNa01", "DNa02"), "R"), side(("DNa01", "DNa02"), "L")
    # the optomotor control pool: optic-lobe intrinsic neurons like T4/T5, minus every T4/T5
    t45 = np.char.startswith(t, "T4") | np.char.startswith(t, "T5")
    g["ol_intrinsic_not_t45"] = np.flatnonzero((sc == "ol_intrinsic") & ~t45)
    g["mn_legs"] = np.concatenate([g["mn_front"], g["mn_mid"], g["mn_hind"]])
    W = abs(br.sim.W_csr)

    def reach(targets):                               # one- plus two-synapse input from each head GRN onto targets
        into = np.asarray(W[targets].sum(axis=0)).ravel()
        one = into[head_gust]
        two = np.asarray(W[:, head_gust].T @ into).ravel()
        return one + two

    pref = reach(g["sugar_pn"]) - reach(g["bitter_sel"])
    order = np.argsort(-pref, kind="stable")
    g["sweet"], g["bitter"] = head_gust[order[:SWEET_N]], head_gust[order[-SWEET_N:]]
    br._assay_groups = g
    return g


def random_like(pool: np.ndarray, n: int, exclude: np.ndarray, seed: int) -> np.ndarray:
    """A random control set of n neurons from pool, never overlapping exclude (seeded, so reproducible)."""
    cand = np.setdiff1d(pool, exclude)
    return np.sort(np.random.default_rng(seed).choice(cand, size=min(n, len(cand)), replace=False))


def hz(spike_counts: int, n_neurons: int, steps: int, dt: float = 0.005) -> float:
    return spike_counts / max(1, n_neurons) / max(1, steps) / dt


# --- pathway response (validation and Lab) ----------------------------------------------------------------------------
def pathway_response(br, drive_rows, readouts: dict[str, np.ndarray], pre: int = 400, stim: int = 400,
                     amp: float = 0.5) -> dict[str, tuple[float, float]]:
    """Hold drive_rows driven for `stim` steps after `pre` calm steps. Returns {readout: (baseline Hz, driven Hz)}.
    Driven neurons get current every step, the way optogenetic activation does in the experiments.
    drive_rows may also be a list of (rows, amp) pairs, each held at its own current (the optomotor EMD stage)."""
    if isinstance(drive_rows, list):
        sets = [(np.asarray(r), float(a)) for r, a in drive_rows if len(r) and a > 0]
    else:
        sets = [(drive_rows, amp)] if drive_rows is not None and len(drive_rows) else []
    counts = {k: [0, 0] for k in readouts}
    for phase, n in ((0, pre), (1, stim)):
        if phase == 1:
            for rows, a in sets:
                simcore.drive(br, rows, a)
        for _ in range(n):
            br._step()
            s = br.sim.spikes
            for k, rows in readouts.items():
                counts[k][phase] += int(np.count_nonzero(s[rows]))
    for rows, _ in sets:
        simcore.undrive(br, rows)
    return {k: (hz(c[0], len(readouts[k]), pre), hz(c[1], len(readouts[k]), stim)) for k, c in counts.items()}


def emd_stage(g: dict, yaw_rate: float, amp: float = 0.5) -> list[tuple[np.ndarray, float]]:
    """The optomotor transduction (GAME RULE): a wide-field yaw rotation becomes current on the T4/T5 subtypes whose
    preferred direction it matches in each eye, through outdoors.emd_motion_drive (a saturating Reichardt-style
    detector output). Full motion gets the same current as every other pathway test's drive. The sim has no
    photoreceptor-to-medulla motion computation that works, so the detector is replaced by this rule."""
    from kickthefly.game import outdoors

    e = outdoors.emd_motion_drive(yaw_rate)
    return [(g["t45_prog_r"], amp * e["prog_r"]), (g["t45_prog_l"], amp * e["prog_l"]),
            (g["t45_regr_r"], amp * e["regr_r"]), (g["t45_regr_l"], amp * e["regr_l"])]


# --- T-maze olfactory conditioning ------------------------------------------------------------------------------------
def _tick_steps(i: int) -> int:
    return STEPS_PER_TICK[i % 3]


def present(br, odors, steps: int, observe: bool = True, shock: bool = False, record_mbon: list | None = None) -> None:
    """Deliver odors (and optionally shock) for `steps` brain steps, re-poking every tick like the game does."""
    mem = br.memory
    done, tick = 0, 0
    while done < steps:
        for o in odors:
            br.poke("scent", o, 0.5)
        if shock:
            br.poke("punish", None, 1.0)
            br.poke("legs", "L", 0.6)
            br.poke("legs", "R", 0.6)
        n = min(_tick_steps(tick), steps - done)
        for _ in range(n):
            br._step()
        done += n
        tick += 1
        if observe and mem is not None and tick % 3 == 0 and len(odors) == 1:
            r = br.sim.activity.rates()
            mem.observe(odors[0], r)
            if record_mbon is not None:
                record_mbon.append(mem.mbon_response(r))


def rest(br, steps: int) -> None:
    for _ in range(steps):
        br._step()


def tmaze_fly(seed: int, cs_plus: str = "odor_a", cycles: int = 6, test_trials: int = 20, paired: bool = True,
              decision_noise: float = 0.08, surgery: dict | None = None, params: dict | None = None,
              brain=None, wiring=None) -> dict:
    """One fly: train CS+ with shock (paired) or with shocks in the rests (unpaired control), then T-maze choices."""
    br = brain or simcore.new_brain(seed=seed, params=params, wiring=wiring)
    cs_minus = "odor_b" if cs_plus == "odor_a" else "odor_a"
    apply_surgery(br, surgery)
    rest(br, 500)                                    # memory.py learns resting activity before any plasticity
    for odor in (cs_plus, cs_minus):                 # naive exposure so each odor has a Kenyon cell template
        present(br, [odor], 240)
        rest(br, 160)
    naive_plus, naive_minus = br.memory.memory_of(cs_plus), br.memory.memory_of(cs_minus)
    for _ in range(cycles):
        present(br, [cs_plus], 240)                  # 1.2 s odor
        present(br, [cs_plus], 200, shock=paired)    # 1.0 s odor + shock (paired) or odor alone
        rest(br, 60)
        if not paired:
            present(br, [], 200, observe=False, shock=True)     # unpaired: the same shock, with no odor
        rest(br, 200)
        present(br, [cs_minus], 440)                 # CS- for as long, never with shock
        rest(br, 260)
    rest(br, 200)
    rng = np.random.default_rng(seed * 7919 + 17)
    mbon = {cs_plus: [], cs_minus: []}
    choices_plus = choices_minus = 0
    for trial in range(test_trials):
        first = (cs_plus, cs_minus) if trial % 2 == 0 else (cs_minus, cs_plus)
        drive = {}
        for odor in first:                           # the fly samples each arm's odor at the choice point
            present(br, [odor], 60, record_mbon=mbon[odor])
            fear, like = br.memory.memory_of(odor)
            drive[odor] = like - fear
            rest(br, 40)
        pick_plus = drive[cs_plus] + rng.normal(0, decision_noise) > drive[cs_minus] + rng.normal(0, decision_noise)
        choices_plus += int(pick_plus)
        choices_minus += int(not pick_plus)
        rest(br, 60)
    fear_plus, _ = br.memory.memory_of(cs_plus)
    fear_minus, _ = br.memory.memory_of(cs_minus)
    n = choices_plus + choices_minus
    return dict(seed=seed, cs_plus=cs_plus, paired=paired, cycles=cycles, pi=(choices_minus - choices_plus) / max(1, n),
                chose_plus=choices_plus, chose_minus=choices_minus, fear_plus=fear_plus, fear_minus=fear_minus,
                naive_fear_plus=naive_plus[0], naive_fear_minus=naive_minus[0],
                mbon_plus_hz=float(np.mean(mbon[cs_plus])) if mbon[cs_plus] else float("nan"),
                mbon_minus_hz=float(np.mean(mbon[cs_minus])) if mbon[cs_minus] else float("nan"))


def tmaze_group(seeds, paired: bool = True, **kw) -> dict:
    """Reciprocal design: half the flies learn odor A as CS+, half odor B. PI per reciprocal pair and overall."""
    flies = []
    for s in seeds:
        a = tmaze_fly(s, "odor_a", paired=paired, **kw)
        b = tmaze_fly(s + 50_000, "odor_b", paired=paired, **kw)
        flies.append(dict(seed=s, pi=(a["pi"] + b["pi"]) / 2, halves=[a, b]))
    return dict(paired=paired, flies=flies, pi=[f["pi"] for f in flies])


# --- looming escape ---------------------------------------------------------------------------------------------------
LOOM_SPEEDS = (0.7, 1.4, 3.0, 5.5, 9.0)      # m/s: creeping, crouch-walking, walking, sprinting, a lunge (the 3D game)


def looming_fly(seed: int, speeds=LOOM_SPEEDS, approaches: int = 3, radius: float = 0.28, start: float = 3.0,
                contact: float | None = None, surgery: dict | None = None, params: dict | None = None,
                brain=None, wiring=None) -> dict:
    """An object of `radius` m (default: your body in the 3D game) approaches the fly's head straight on at each speed,
    from `start` m to contact. Returns per speed whether DNp01 crossed the escape threshold before contact, the latency
    from the start of the approach, and how far away the object still was."""
    contact = radius + 0.02 if contact is None else contact
    from kickthefly.game import kick_the_fly as k

    br = brain or simcore.new_brain(seed=seed, params=params, wiring=wiring)
    apply_surgery(br, surgery)
    rest(br, 400)
    out = {}
    for v in speeds:
        trials = []
        for _ in range(approaches):
            d, prev_theta, tick, t = start, None, 0, 0.0
            escaped_at, peak = None, 0.0
            while d > contact:
                theta = 2 * math.atan(radius / d)
                if prev_theta is not None:
                    rate = (theta - prev_theta) * 60.0             # rad/s, the game's measure (kick_the_fly._vision)
                    strength = float(np.clip((rate - k.LOOM_MIN) / k.LOOM_FULL, 0, 1))
                    if strength > 0:
                        br.poke("loom", None, strength, recruit=0.6 * strength)
                prev_theta = theta
                for _ in range(_tick_steps(tick)):
                    br._step()
                lvl = br.level("escape")
                peak = max(peak, lvl)
                if escaped_at is None and lvl > k.THRESH["escape"]:
                    escaped_at = t
                    break
                tick += 1
                t += 1 / 60
                d -= v / 60
            trials.append(dict(escaped=escaped_at is not None, latency_s=escaped_at, distance_m=d if escaped_at is not None
                               else None, time_to_contact_s=None if escaped_at is None else (d - contact) / v,
                               peak_level=peak))
            rest(br, 400)
        out[v] = trials
    return dict(seed=seed, radius=radius, start=start, speeds=list(speeds), trials=out)


# --- sugar dose response ----------------------------------------------------------------------------------------------
def offer_sugar(br, dose: float, steps: int = 200, baseline: int = 200) -> dict:
    """Drive `dose` (0..1) of the sugar-pathway GRNs for `steps`; MN9 firing before vs during."""
    g = groups(br)
    mn9 = g["mn9"]
    before = 0
    for _ in range(baseline):
        br._step()
        before += int(np.count_nonzero(br.sim.spikes[mn9]))
    during = 0
    for i in range(steps):
        if dose > 0 and i % 3 == 0:
            br.poke("sweet", None, 0.5, recruit=dose)
        br._step()
        during += int(np.count_nonzero(br.sim.spikes[mn9]))
    b, d = hz(before, len(mn9), baseline), hz(during, len(mn9), steps)
    return dict(dose=dose, mn9_before_hz=b, mn9_hz=d, ratio=d / max(b, 1.0), extended=d >= PER_RATIO * max(b, 1.0))


def sugar_fly(seed: int, doses=(0.0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0), repeats: int = 2, surgery: dict | None = None,
              params: dict | None = None, brain=None, wiring=None) -> dict:
    br = brain or simcore.new_brain(seed=seed, params=params, wiring=wiring)
    apply_surgery(br, surgery)
    rest(br, 200)
    out = {d: [] for d in doses}
    rng = np.random.default_rng(seed + 3)
    order = [d for d in doses for _ in range(repeats)]
    rng.shuffle(order)                                  # interleaved doses, like a real dose-response session
    for d in order:
        out[d].append(offer_sugar(br, d))
        rest(br, 200)
    return dict(seed=seed, doses=list(doses), offers=out)


# --- surgery ----------------------------------------------------------------------------------------------------------
def apply_surgery(br, surgery: dict | None) -> None:
    """surgery: {neuron spec: -1 silence | +1 stimulate}, specs as in simcore.rows_of or an assays group name."""
    if not surgery:
        return
    from kickthefly.game import kick_the_fly as k

    g = groups(br)
    for spec, mode in surgery.items():
        rows = g[spec] if spec in g else simcore.rows_of(br, spec)
        br.set_override(rows, int(mode))
    br.surgery = bool(np.any(br.override))
    assert k.SURGERY_CURRENT


# --- orchard feeding (the Orchard arena, headless) ----------------------------------------------------------------------
ORCHARD_TRAVEL_S = 2.0              # time between feeds: flying to the next fruit (game rule)


def orchard_fly(seed: int, feeds: int | None = None, regrow_s: float | None = None, cap: int | None = None,
                duration_s: float = 120.0, surgery: dict | None = None, params: dict | None = None,
                brain=None, wiring=None) -> dict:
    """One fly feeding in the Orchard for `duration_s`, in lockstep, so a reward schedule is reproducible.

    The orchard (trees, fruit, how many feeds each holds, regrowth, the cap) and the fly's schedule (fly to the nearest
    ripe fruit, feed one bout, fly on) are GAME RULES, taken unchanged from kickthefly/game/outdoors.py. Feeding drives
    the same real neurons the game does: the sugar-pathway taste neurons and the PAM reward neurons (fermented fruit a
    little harder, as the alcohol tool does). Readouts: PAM and MN9 firing while feeding vs while travelling, and the
    orchard's state over time. PAM being driven by feeding is itself the game rule the sugar tool uses (in this sim
    taste alone doesn't reach PAM), so its ratio says how hard the schedule rewards, not that the fly finds fruit
    rewarding; MN9's response to the sugar pathway is the validated part.
    """
    from kickthefly.game import outdoors

    params = params or {}
    feeds = int(feeds if feeds is not None else params.get("orchard.feeds", outdoors.DEFAULT_FEEDS))
    regrow_s = float(regrow_s if regrow_s is not None else params.get("orchard.regrow_s", outdoors.DEFAULT_REGROW_S))
    cap = int(cap if cap is not None else params.get("orchard.cap", outdoors.DEFAULT_CAP))
    br = brain or simcore.new_brain(seed=seed, params=params, wiring=wiring)
    apply_surgery(br, surgery)
    rest(br, 200)
    g = groups(br)
    pam = br.sense[("reward", None)]
    mn9 = g["mn9"]
    orch = outdoors.Orchard(outdoors.scenery("orchard")["trees"], feeds=feeds, regrow_s=regrow_s, cap=cap, seed=seed)
    pos = np.array([0.0, 1.0, 0.0])
    t = 0.0
    dt = 0.005
    counts = {"feed": [0, 0, 0], "travel": [0, 0, 0]}     # PAM spikes, MN9 spikes, steps
    n_feeds = emptied = fermented_feeds = 0
    timeline = []
    next_sample = 0.0

    def run(steps, phase, fruit=None):
        nonlocal t
        for i in range(steps):
            if fruit is not None and i % 3 == 0:
                br.poke("taste", None, 0.5)
                br.poke("sweet", None, 0.6 if fruit.fermented else 0.5, recruit=0.6)
                br.poke("reward", None, 0.6 if fruit.fermented else 0.4)
                if fruit.fermented and i % 10 == 0:
                    br.poke("scent", "alcohol", 0.35)
            br._step()
            s = br.sim.spikes
            c = counts[phase]
            c[0] += int(np.count_nonzero(s[pam]))
            c[1] += int(np.count_nonzero(s[mn9]))
            c[2] += 1
            t += dt

    while t < duration_s:
        orch.step(t)
        if t >= next_sample:
            timeline.append(dict(t=round(t, 2), **orch.counts()))
            next_sample += 5.0
        f = orch.nearest_ripe(pos)
        run(int(ORCHARD_TRAVEL_S / dt), "travel")
        if f is None or not f.ripe:
            continue
        pos = f.pos.copy()
        run(int(outdoors.FEED_BOUT_S / dt), "feed", f)
        n_feeds += 1
        fermented_feeds += int(f.fermented)
        emptied += int(orch.feed(f, t))
    fe, tr = counts["feed"], counts["travel"]
    pam_feed, pam_travel = hz(fe[0], len(pam), fe[2]), hz(tr[0], len(pam), tr[2])
    mn9_feed, mn9_travel = hz(fe[1], len(mn9), fe[2]), hz(tr[1], len(mn9), tr[2])
    return dict(seed=seed, feeds_per_fruit=feeds, regrow_s=regrow_s, cap=cap, duration_s=duration_s,
                feeds=n_feeds, fermented_feeds=fermented_feeds, fruit_emptied=emptied,
                pam_feed_hz=pam_feed, pam_travel_hz=pam_travel, pam_ratio=pam_feed / max(pam_travel, 0.5),
                mn9_feed_hz=mn9_feed, mn9_travel_hz=mn9_travel, mn9_ratio=mn9_feed / max(mn9_travel, 0.5),
                timeline=timeline)


# --- mushroom body learning extensions (Task 4) -------------------------------------------------------------------
def extinction_fly(seed: int, cs_plus: str = "odor_a", cycles: int = 6, ext_cycles: int = 8,
                   test_trials: int = 20, decision_noise: float = 0.08, brain=None, wiring=None) -> dict:
    """Extinction of learned fear (Felsenberg et al. 2018): train CS+ with shock (cycles),
    then apply repeated unreinforced CS+ alone exposures (ext_cycles).
    Runs both the extinction fly and an unextinguished control fly (same conditioned brain snapshot)."""
    from kickthefly.core import savestate

    br = brain or simcore.new_brain(seed=seed, wiring=wiring)
    cs_minus = "odor_b" if cs_plus == "odor_a" else "odor_a"
    rest(br, 500)
    for odor in (cs_plus, cs_minus):
        present(br, [odor], 240)
        rest(br, 160)

    # Conditioning phase: CS+ paired with shock
    for _ in range(cycles):
        present(br, [cs_plus], 240)
        present(br, [cs_plus], 200, shock=True)
        rest(br, 60)
        rest(br, 200)
        present(br, [cs_minus], 440)
        rest(br, 260)
    rest(br, 200)
    fear_cond, _ = br.memory.memory_of(cs_plus)

    # Save conditioned snapshot to run paired extinction vs unextinguished control
    snap = {}
    meta = savestate.brain_state(br, "ext_", snap)

    # Branch 1: Extinction (repeated CS+ alone exposures)
    for _ in range(ext_cycles):
        present(br, [cs_plus], 440)
        rest(br, 260)
    rest(br, 200)
    fear_ext, _ = br.memory.memory_of(cs_plus)

    # T-maze test choices for extinguished fly
    rng_ext = np.random.default_rng(seed * 7919 + 23)
    choices_plus_ext = choices_minus_ext = 0
    for trial in range(test_trials):
        first = (cs_plus, cs_minus) if trial % 2 == 0 else (cs_minus, cs_plus)
        drive = {}
        for odor in first:
            present(br, [odor], 60)
            fear, like = br.memory.memory_of(odor)
            drive[odor] = like - fear
            rest(br, 40)
        pick_plus = drive[cs_plus] + rng_ext.normal(0, decision_noise) > drive[cs_minus] + rng_ext.normal(0, decision_noise)
        choices_plus_ext += int(pick_plus)
        choices_minus_ext += int(not pick_plus)
        rest(br, 60)
    pi_ext = (choices_minus_ext - choices_plus_ext) / max(1, choices_plus_ext + choices_minus_ext)

    # Branch 2: Unextinguished control (restore conditioned snapshot, equal rest time)
    savestate.restore_brain(br, meta, snap, "ext_")
    for _ in range(ext_cycles):
        rest(br, 700)
    rest(br, 200)
    fear_unext, _ = br.memory.memory_of(cs_plus)

    rng_unext = np.random.default_rng(seed * 7919 + 23)
    choices_plus_unext = choices_minus_unext = 0
    for trial in range(test_trials):
        first = (cs_plus, cs_minus) if trial % 2 == 0 else (cs_minus, cs_plus)
        drive = {}
        for odor in first:
            present(br, [odor], 60)
            fear, like = br.memory.memory_of(odor)
            drive[odor] = like - fear
            rest(br, 40)
        pick_plus = drive[cs_plus] + rng_unext.normal(0, decision_noise) > drive[cs_minus] + rng_unext.normal(0, decision_noise)
        choices_plus_unext += int(pick_plus)
        choices_minus_unext += int(not pick_plus)
        rest(br, 60)
    pi_unext = (choices_minus_unext - choices_plus_unext) / max(1, choices_plus_unext + choices_minus_unext)

    return dict(seed=seed, cs_plus=cs_plus, cycles=cycles, ext_cycles=ext_cycles,
                fear_cond=fear_cond, fear_ext=fear_ext, fear_unext=fear_unext,
                pi_ext=pi_ext, pi_unext=pi_unext,
                chose_plus_ext=choices_plus_ext, chose_minus_ext=choices_minus_ext,
                chose_plus_unext=choices_plus_unext, chose_minus_unext=choices_minus_unext)


def second_order_fly(seed: int, paired: bool = True, cycles: int = 6, test_trials: int = 20,
                     decision_noise: float = 0.08, brain=None, wiring=None) -> dict:
    """Second-order olfactory conditioning (Tabone & de Belle 2011; Felsenberg et al. 2017):
    Phase 1: CS1 (odor_a) + shock conditioning.
    Phase 2: CS2 (odor_b) paired with CS1 (odor_a) without shock. In the paired condition,
    the conditioned fear of odor_a triggers PPL1 punishment reinforcement during compound presentation.
    In the unpaired condition, odor_b and odor_a are presented explicitly unpaired.
    Test: T-maze choice between odor_b (CS2) and naive odor_c."""
    br = brain or simcore.new_brain(seed=seed, wiring=wiring)
    if ("scent", "odor_c") not in br.sense:
        orn = br.sense[("smell", None)]
        glom = np.array([t.split("_", 1)[1] for t in br.types[orn]])
        order = np.random.default_rng(ODOR_GLOMERULI_SEED).permutation(np.unique(glom))
        br.sense[("scent", "odor_c")] = orn[np.isin(glom, order[12:18])]

    rest(br, 500)
    for odor in ("odor_a", "odor_b", "odor_c"):
        present(br, [odor], 240)
        rest(br, 160)

    # Phase 1: train odor_a + shock (6 cycles)
    for _ in range(cycles):
        present(br, ["odor_a"], 240)
        present(br, ["odor_a"], 200, shock=True)
        rest(br, 60)
        rest(br, 200)

    fear_a, _ = br.memory.memory_of("odor_a")

    # Phase 2: second-order pairing of odor_b with odor_a
    for _ in range(cycles):
        if paired:
            present(br, ["odor_b"], 240)
            # Compound presentation: conditioned fear of odor_a drives punishment dopamine
            present(br, ["odor_b", "odor_a"], 200, shock=True)
            rest(br, 60)
            rest(br, 200)
        else:
            # Unpaired: odor_b alone, then shock/odor_a in rest
            present(br, ["odor_b"], 240)
            rest(br, 260)
            present(br, ["odor_a"], 200, shock=True)
            rest(br, 200)

    fear_b, _ = br.memory.memory_of("odor_b")
    fear_c, _ = br.memory.memory_of("odor_c")

    # Test choice between odor_b (CS2) and odor_c (control)
    rng = np.random.default_rng(seed * 7919 + 31)
    choices_b = choices_c = 0
    for trial in range(test_trials):
        first = ("odor_b", "odor_c") if trial % 2 == 0 else ("odor_c", "odor_b")
        drive = {}
        for odor in first:
            present(br, [odor], 60)
            fear, like = br.memory.memory_of(odor)
            drive[odor] = like - fear
            rest(br, 40)
        pick_b = drive["odor_b"] + rng.normal(0, decision_noise) > drive["odor_c"] + rng.normal(0, decision_noise)
        choices_b += int(pick_b)
        choices_c += int(not pick_b)
        rest(br, 60)
    pi = (choices_c - choices_b) / max(1, choices_b + choices_c)
    return dict(seed=seed, paired=paired, cycles=cycles, pi=pi,
                fear_a=fear_a, fear_b=fear_b, fear_c=fear_c,
                chose_b=choices_b, chose_c=choices_c)


# --- odor plume tracking (Task 3) ---------------------------------------------------------------------------------
def plume_tracking_fly(seed: int, duration_s: float = 25.0, wind_speed: float = 2.0,
                       source_pos=(0.0, 10.0), start_pos=(0.0, -5.0), brain=None) -> dict:
    """Plume tracking in open field (GAME RULE navigation + real ORN activation):
    Intermittent filaments carried by ambient wind drive the real ORNs (ORN_DM1).
    Fly navigates by surging upwind upon odor contact and crosswind casting during blanks."""
    br = brain or simcore.new_brain(seed=seed)
    orn_dm1 = np.where(br.types == "ORN_DM1")[0]
    rng = np.random.default_rng(seed * 10007 + 3)
    pos = np.array([start_pos[0] + rng.uniform(-1.5, 1.5), start_pos[1]], dtype=float)
    source = np.array(source_pos, dtype=float)
    heading = np.pi / 2.0 + rng.uniform(-0.5, 0.5)
    dt = 0.05
    reached = False
    time_taken = duration_s
    path_len = 0.0

    steps = int(duration_s / dt)
    for step in range(steps):
        t = step * dt
        dist_to_src = float(np.linalg.norm(pos - source))
        if dist_to_src < 1.5:
            reached = True
            time_taken = t
            break

        downwind = source[1] - pos[1]
        crosswind = abs(pos[0] - source[0])
        sigma = 0.9 + 0.12 * max(downwind, 0.0) ** 0.65
        filament = (crosswind < 2.2 * sigma) and (np.sin(t * 2.5 * np.pi) > -0.35)
        c = float(np.exp(-0.5 * (crosswind / sigma) ** 2)) if (downwind > 0 and filament) else 0.0

        # Drive real ORN_DM1 sensory neurons
        if c > 0.05:
            simcore.drive(br, orn_dm1, c * 0.5)
        else:
            simcore.undrive(br, orn_dm1)
        br._step()

        # Navigation (GAME RULE): upwind surge vs crosswind cast
        if c > 0.1:
            target = np.pi / 2.0 + rng.normal(0, 0.06)
            speed = 2.0
        else:
            target = (0.0 if pos[0] < 0 else np.pi) + rng.normal(0, 0.06)
            speed = 1.3

        diff = np.arctan2(np.sin(target - heading), np.cos(target - heading))
        heading += np.clip(diff, -10.0 * dt, 10.0 * dt)
        step_vec = np.array([np.cos(heading), np.sin(heading)]) * speed * dt
        pos += step_vec
        path_len += speed * dt

    simcore.undrive(br, orn_dm1)
    return dict(seed=seed, reached=reached, time_s=time_taken, path_m=path_len,
                final_dist=float(np.linalg.norm(pos - source)))


def plume_tracking_assay(seeds=tuple(range(1000, 1020)), duration_s: float = 25.0, wind_speed: float = 2.0) -> dict:
    """Run plume tracking assay across seeds and compute success rate and tracking metrics with 95% CI."""
    from kickthefly.lab import labstats

    flies = [plume_tracking_fly(s, duration_s=duration_s, wind_speed=wind_speed) for s in seeds]
    successes = [1.0 if f["reached"] else 0.0 for f in flies]
    success_ci = labstats.mean_ci(successes)
    times = [f["time_s"] for f in flies if f["reached"]]
    time_ci = labstats.mean_ci(times)
    paths = [f["path_m"] for f in flies if f["reached"]]
    path_ci = labstats.mean_ci(paths)
    return dict(n_seeds=len(seeds), duration_s=duration_s, wind_speed=wind_speed,
                success_ci=success_ci, time_ci=time_ci, path_ci=path_ci, flies=flies)

