"""Lab mode screens in the pause menu: the hub, and live parameter controls.

Parameters come in two kinds, tagged on screen:
  MODEL      the LIF simulation's own parameters (kickthefly/sim/connectome/sim.py LIFParams). The wiring is untouched, but the
             validation results were measured at the defaults, so anything changed here is flagged as "modified" in
             the validation dashboard, exports and save states.
  GAME RULE  thresholds the game uses to turn neuron firing into moves, and how looming is converted to drive.
"""
from __future__ import annotations

import time

import pygame

from kickthefly.ui import menu as ui

# name, label, kind (model | rule), default, lo, hi, step, fmt, tooltip
PARAMS = (
    ("noise_std", "Membrane noise", "model", 0.05, 0.0, 0.2, 0.005, "{:.3f}",
     "Random current added to every neuron each 5 ms step. Higher makes the brain flicker more on its own."),
    ("bias", "Tonic drive", "model", 0.20, 0.0, 0.3, 0.005, "{:.3f}",
     "Constant input to every neuron; at 0.20 a neuron rests at 80% of its firing threshold."),
    ("target_rate_hz", "Target rate (Hz)", "model", 5.0, 1.0, 20.0, 0.5, "{:.1f}",
     "The whole-brain firing rate the slow gain controller steers toward."),
    ("ext_gain", "Sensory input gain", "model", 4.0, 1.0, 8.0, 0.1, "{:.1f}",
     "How strongly hits, smells and other stimuli drive the sensory neurons they reach."),
    ("gain_adapt", "Gain adaptation rate", "model", 0.002, 0.0, 0.01, 0.0005, "{:.4f}",
     "How fast the gain controller reacts. 0 freezes synaptic gain where it is."),
    ("thresh.escape", "Dodge: DNp01 above", "rule", 4.0, 1.5, 10.0, 0.1, "{:.1f}x",
     "The giant fiber's firing, as a multiple of its calm rate, that makes the fly dodge."),
    ("thresh.groom", "Groom: aDN1/aDN2 above", "rule", 4.0, 1.5, 10.0, 0.1, "{:.1f}x",
     "Antennal grooming command neurons' firing that logs a GROOM reaction (no body movement)."),
    ("thresh.jump", "Fly away: head-touch DNs above", "rule", 3.0, 1.2, 8.0, 0.1, "{:.1f}x", "Head-touch DN threshold."),
    ("thresh.run", "Run: body-touch DNs above", "rule", 2.4, 1.2, 8.0, 0.1, "{:.1f}x", "Body-touch DN threshold."),
    ("thresh.kick", "Kick: leg-touch DNs above", "rule", 2.0, 1.2, 8.0, 0.1, "{:.1f}x", "Leg-touch DN threshold."),
    ("thresh.walk", "Walk: DNp09 above", "rule", 3.0, 1.2, 8.0, 0.1, "{:.1f}x", "Forward walking threshold."),
    ("thresh.back", "Back up: MDN above", "rule", 3.8, 1.2, 8.0, 0.1, "{:.1f}x", "Moonwalker threshold."),
    ("thresh.turn", "Turn: DNa01/02 R-L above", "rule", 2.1, 0.5, 8.0, 0.1, "{:.1f}", "Steering difference threshold."),
    ("thresh.fly", "Take off: DNg02 above", "rule", 1.58, 1.1, 4.0, 0.02, "{:.2f}x", "Wing-power threshold."),
    ("thresh.fire", "Shoot: DNp35 above", "rule", 3.0, 1.2, 8.0, 0.1, "{:.1f}x", "Duel trigger threshold."),
    ("thresh.song", "Song: ps1 wing MNs (1 s) above", "rule", 1.8, 1.1, 5.0, 0.05, "{:.2f}x",
     "The two ps1 wing motor neurons' firing, averaged over about a second, that plays the courtship buzz (SONG)."),
    ("thresh.aggression", "Lunge: AVLP727m/pC1 above", "rule", 2.0, 1.1, 8.0, 0.1, "{:.1f}x",
     "Aggression neurons' firing that throws the fly at the nearest other fly (LUNGE; needs 2+ flies)."),
    ("thresh.sleep", "Sleep: dFB FB6/FB7 above", "rule", 2.0, 1.1, 8.0, 0.1, "{:.1f}x",
     "Dorsal fan-shaped body firing that puts the fly to rest (SLEEP): no spontaneous take-off or walking."),
    ("thresh.co2", "CO2 log: V glomerulus PNs above", "rule", 2.0, 1.1, 8.0, 0.1, "{:.1f}x",
     "V glomerulus projection neurons' firing that logs a CO2 reaction (no body movement)."),
    ("thresh.hot_pn", "Heat log: VP2 PNs above", "rule", 1.6, 1.1, 8.0, 0.05, "{:.2f}x",
     "Hot-pathway projection neurons' firing that logs a HEAT reaction (no body movement)."),
    ("thresh.cold_pn", "Cold log: VP3 PNs above", "rule", 1.7, 1.1, 8.0, 0.05, "{:.2f}x",
     "Cold-pathway projection neurons' firing that logs a COLD reaction (no body movement)."),
    ("loom_min", "Looming: ignored below (rad/s)", "rule", 1.5, 0.0, 6.0, 0.1, "{:.1f}",
     "Angular expansion speed below which approaching objects don't drive LPLC2/LC4 at all."),
    ("loom_full", "Looming: full drive span (rad/s)", "rule", 8.0, 1.0, 20.0, 0.5, "{:.1f}",
     "How much faster than the minimum an object must grow to drive the looming detectors fully."),
    ("field.wind_dir", "Open field: wind from (deg)", "rule", 180.0, 0.0, 355.0, 5.0, "{:.0f}",
     "Where the open field's steady wind comes from, in degrees (0 = +x). It drives the real JO-C/E wind neurons; "
     "how the two antennae split it by heading is a game rule."),
    ("field.wind_speed", "Open field: wind speed (m/s)", "rule", 3.0, 0.0, 10.0, 0.5, "{:.1f}",
     "Steady wind strength in the open field. 6 m/s and above drives the wind neurons fully (game rule)."),
    ("weather.rain", "Weather: rain intensity", "rule", 0.0, 0.0, 1.0, 0.05, "{:.2f}",
     "Rain in the open field and the orchard (0 = none). Drops hit the body by part and fire the real touch neurons (head, "
     "body, legs, wings), the air drives the humidity neurons, and enough rain wets the wings so the fly can't take off, "
     "as after the pool. The hit rate, strengths, parts and wetting are game rules."),
    ("weather.gust_hz", "Weather: gusts per second", "rule", 0.0, 0.0, 0.5, 0.02, "{:.2f}",
     "How often a gust blows in the open field and the orchard. A gust is extra wind speed (2-6 m/s for 1-3 s) fed through "
     "the existing wind -> Johnston's organ JO-C/E transduction. The gust's size, length and turn are game rules."),
    ("weather.storm", "Weather: storm (0 off, 1 on)", "rule", 0.0, 0.0, 1.0, 1.0, "{:.0f}",
     "A storm is at least 70% rain, 0.2 gusts a second and +3 m/s of wind, a darker scene, and lightning every 5-14 s that "
     "drives the photoreceptors (the screen swells slowly instead of flashing with Reduced flashing on). Thunder follows "
     "the flash. The storm preset is a game rule."),
    ("outdoor.sun_az", "Outdoors: sun azimuth (deg)", "rule", 135.0, 0.0, 355.0, 5.0, "{:.0f}",
     "Where the sun stands. Sunlight drives the real photoreceptors, split between the eyes by heading (game rule)."),
    ("outdoor.sun_el", "Outdoors: sun elevation (deg)", "rule", 45.0, -10.0, 90.0, 5.0, "{:.0f}",
     "How high the sun is: brightness of the light drive, and the scene's lighting. Below 0 it is night."),
    ("outdoor.day_s", "Outdoors: day length (s, 0 = off)", "rule", 0.0, 0.0, 1800.0, 30.0, "{:.0f}",
     "Day/night cycle: the sun circles once in this many seconds (noon +60 deg, midnight -30 deg), replacing the "
     "fixed sun above. Daylight drives the photoreceptors and the morning clock neurons l-LNv/s-LNv (game rule)."),
    ("orchard.feeds", "Orchard: feeds per fruit", "rule", 4.0, 1.0, 10.0, 1.0, "{:.0f}",
     "How many feeding bouts one fruit supports before it drops. Applies to fruit that grow from now on."),
    ("orchard.regrow_s", "Orchard: regrow time (s)", "rule", 75.0, 10.0, 300.0, 5.0, "{:.0f}",
     "Mean time for a dropped fruit to grow back, jittered +-25% so regrowth is staggered."),
    ("orchard.cap", "Orchard: fruit per tree (cap)", "rule", 4.0, 1.0, 6.0, 1.0, "{:.0f}",
     "Most fruit one tree carries at a time."),
    ("indiv.sigma_subtle", "Individuality: subtle sigma", "rule", 0.05, 0.0, 0.5, 0.01, "{:.2f}",
     "Spread (log-normal sigma) of the per-neuron gains D_pre and D_post in W_fly = D_post·W·D_pre at Individuality "
     "Subtle. Signs never change. Applies to brains built from now on; validation always runs with individuality off."),
    ("indiv.sigma_strong", "Individuality: strong sigma", "rule", 0.15, 0.0, 0.5, 0.01, "{:.2f}",
     "The same at Individuality Strong."),
    ("pet.hunger_h", "Pet: hours from full to starving", "rule", 24.0, 1.0, 168.0, 1.0, "{:.0f}",
     "How fast the pet's hunger rises, on the wall clock and in the catch-up when you come back."),
    ("pet.awake_h", "Pet: hours awake until exhausted", "rule", 16.0, 1.0, 72.0, 1.0, "{:.0f}",
     "How fast sleep pressure builds while the pet is awake."),
    ("pet.sleep_h", "Pet: hours of sleep to recover", "rule", 8.0, 1.0, 24.0, 1.0, "{:.0f}",
     "How fast sleep pressure falls while the pet sleeps."),
    ("pet.sugar_gain", "Pet: hunger boost to sugar taste", "rule", 2.0, 0.0, 5.0, 0.1, "{:.1f}",
     "Sugar drives the real taste neurons at (1 + this x hunger) times the normal strength."),
    ("pet.pam_gain", "Pet: hunger boost to PAM reward", "rule", 1.5, 0.0, 5.0, 0.1, "{:.1f}",
     "Eating drives the real PAM reward dopamine neurons at (1 + this x hunger) times the normal strength."),
    ("pet.dfb_drive", "Pet: sleep pressure to dFB drive", "rule", 0.5, 0.0, 2.0, 0.05, "{:.2f}",
     "Sleep pressure drives the real dorsal fan-shaped body sleep neurons (FB6/FB7) with this x pressure."),
    ("pet.max_catchup_d", "Pet: longest catch-up (days)", "rule", 7.0, 1.0, 30.0, 1.0, "{:.0f}",
     "Time away longer than this counts as this long, so a clock jump cannot starve the pet. Read at launch."),
)
DEFAULTS = {p[0]: p[3] for p in PARAMS}
BY_NAME = {p[0]: p for p in PARAMS}


def apply_to_sim(sim, params: dict) -> None:
    """Model parameters onto one LIFSim (also used for newly spawned flies and headless runs)."""
    for name in ("noise_std", "bias", "target_rate_hz", "ext_gain", "gain_adapt"):
        v = float(params.get(name, DEFAULTS[name]))
        if name == "noise_std":
            old = float(sim.p.noise_std)
            if old > 0:
                sim._noise *= v / old          # the noise bank is pre-scaled at construction
            elif v > 0:
                import numpy as np
                sim._noise = sim.rng.standard_normal(sim._noise.size, dtype=np.float32) * np.float32(v)
        if name == "target_rate_hz":
            sim.target_p = v * sim.p.dt_ms / 1000.0
        setattr(sim.p, name, v)


def apply_rules(params: dict) -> None:
    from kickthefly.game import kick_the_fly as k2

    for name, value in params.items():
        if name.startswith("thresh."):
            k2.THRESH[name.split(".", 1)[1]] = float(value)
    k2.LOOM_MIN = float(params.get("loom_min", DEFAULTS["loom_min"]))
    k2.LOOM_FULL = float(params.get("loom_full", DEFAULTS["loom_full"]))
    from kickthefly.core import individuality, pet

    val = lambda k: float(params.get(k, DEFAULTS[k]))      # noqa: E731
    individuality.SIGMAS["subtle"] = val("indiv.sigma_subtle")
    individuality.SIGMAS["strong"] = val("indiv.sigma_strong")
    pet.HUNGER_RATE_PER_SEC = 1.0 / (val("pet.hunger_h") * 3600.0)
    pet.SLEEP_RATE_PER_SEC = 1.0 / (val("pet.awake_h") * 3600.0)
    pet.SLEEP_RECOVERY_RATE = 1.0 / (val("pet.sleep_h") * 3600.0)
    pet.HUNGER_SUGAR_GAIN = val("pet.sugar_gain")
    pet.HUNGER_PAM_GAIN = val("pet.pam_gain")
    pet.SLEEP_DFB_DRIVE = val("pet.dfb_drive")
    pet.MAX_CATCHUP_SECONDS = val("pet.max_catchup_d") * 86400.0


def modified(params: dict) -> dict:
    return {k: v for k, v in params.items() if k in DEFAULTS and abs(float(v) - DEFAULTS[k]) > 1e-9}


def page_hub(m: ui.Menu, surf, rect, mouse) -> None:
    host = m.host
    m.text(surf, "LAB", (rect.x + 24, rect.y + 16), ui.INK, m.f_head)
    m.text(surf, "Research tools. Everything here runs on the same connectome sim as the game.", (rect.x + 24, rect.y + 50),
           ui.LABEL, m.f_small)
    b_obj = getattr(getattr(getattr(host, "brain", None), "sim", None), "backend", None)
    b_name = getattr(b_obj, "name", "CPU (NumPy)")
    b_dev = getattr(b_obj, "device", "CPU")
    m.text(surf, f"Engine: {b_name} · {b_dev}", (rect.right - 24, rect.y + 20), (120, 220, 240), m.f_small, "topright")
    items = [(label, page, tip) for label, page, tip in host.lab_pages() if page in m.pages]
    body = pygame.Rect(rect.x + 8, rect.y + 84, rect.w - 16, rect.h - 84 - 96)
    off = int(m.scroll.get("lab", 0))
    m.clip = body
    prev = surf.get_clip()
    surf.set_clip(body)
    bw, bh = (rect.w - 72) // 2, 64
    for i, (label, page, tip) in enumerate(items):
        x = rect.x + 24 + (i % 2) * (bw + 24)
        y = body.y + 6 + (i // 2) * (bh + 16) - off
        m.button(surf, (x, y, bw, bh), label, (lambda p=page: m.show(p)), id=("lab", page), tip=tip)
    rows = (len(items) + 1) // 2
    m.content_h["lab"] = max(0, body.y + 6 + rows * (bh + 16) - body.bottom + 8)
    surf.set_clip(prev)
    m.clip = None
    mod = modified(host.lab_params)
    notes = []
    if mod:
        notes.append(f"Modified parameters: {', '.join(BY_NAME[k][1] for k in mod)}")
    w = getattr(host, "wiring", None)
    if w is not None and not w.is_identity:
        notes.append(f"Modified connectome: {w.label()}")
    for i, note in enumerate(notes):
        m.text(surf, note, (rect.x + 24, rect.bottom - 100 + i * 18), ui.AMBER, m.f_small)
    m.button(surf, (rect.right - 164, rect.bottom - 58, 140, 42), "Back", m.back, style="primary", id=("lab", "back"))


def page_params(m: ui.Menu, surf, rect, mouse) -> None:
    host = m.host
    m.text(surf, "PARAMETERS", (rect.x + 24, rect.y + 16), ui.INK, m.f_head)
    m.text(surf, "Applies live to every fly. Validation results were measured at the defaults.", (rect.x + 24, rect.y + 50),
           ui.LABEL, m.f_small)
    body = pygame.Rect(rect.x + 16, rect.y + 80, rect.w - 32, rect.h - 80 - 70)
    key = "lab_params"
    off = int(m.scroll.get(key, 0))
    m.clip = body
    prev = surf.get_clip()
    surf.set_clip(body)
    y = body.y + 4 - off
    for name, label, kind, default, lo, hi, step, fmt, tip in PARAMS:
        value = float(host.lab_params.get(name, default))
        row = pygame.Rect(body.x, y, body.w, 40)
        changed = abs(value - default) > 1e-9
        m.text(surf, label, (row.x + 12, row.centery), ui.AMBER if changed else ui.TEXT, m.f_text, "midleft")
        m.chip(surf, (row.x + 330, row.centery - 10), "MODEL" if kind == "model" else "GAME RULE")
        m._register(pygame.Rect(row.x, row.y, 440, row.h), "label", id=("plabel", name),
                    tip=tip + f"  Default {fmt.format(default)}.")
        m.slider(surf, (row.x + 450, row.y + 6, row.w - 470, 28), value, lo, hi, step, fmt,
                 lambda v, n=name: host.set_lab_param(n, v), lambda: None, id=("param", name), tip=tip)
        y += 44
    m.content_h[key] = max(0, y + off - body.bottom + 8)
    surf.set_clip(prev)
    m.clip = None
    m.button(surf, (rect.x + 24, rect.bottom - 58, 200, 42), "Reset to defaults",
             lambda: [host.set_lab_param(n, d) for n, d in DEFAULTS.items()], id=("params", "reset"))
    m.button(surf, (rect.right - 164, rect.bottom - 58, 140, 42), "Back", m.back, style="primary", id=("params", "back"))


ASSUMPTIONS = (
    ("Raw synapse counts as functional strength proxy",
     "SYNAPSE",
     "The sim treats raw EM synapse counts between neuron pairs as directly proportional to synaptic conductance.",
     "Biological synapses vary widely in vesicle pool size, neurotransmitter release probability, post-synaptic receptor density, and phosphorylation state. Real connection efficacy does not linearly track anatomical contact count.",
     "README.md § Connectome vs Game Rule · kickthefly/sim/connectome/sim.py:LIFParams"),

    ("Uniform synaptic efficacy per connection type",
     "SYNAPSE",
     "All excitatory and inhibitory synapses share fixed base efficacy constants across the whole connectome.",
     "Drosophila synapses exhibit diverse quantal sizes and kinetics across cell types (cholinergic, GABAergic, glutamatergic). Here, sign is assigned from neurotransmitter annotations with uniform base weights.",
     "kickthefly/sim/connectome/sim.py:LIFSim · kickthefly/lab/validation.py"),

    ("Leaky integrate-and-fire (LIF) point neurons",
     "BIOPHYSICS",
     "Each cell body and its entire arbor is condensed into a single isopotential point compartment with tau = 20 ms.",
     "Drosophila neurons have complex non-spiking local computations, passive cable filtering along fine neurites, and compartmentalized local dendritic processing (e.g. in mushroom body lobes and optic lobes) that point-LIF collapses.",
     "kickthefly/sim/connectome/sim.py:LIFParams (tau_m=20ms, dt=5ms)"),

    ("No neurotransmitter or receptor kinetics",
     "DYNAMICS",
     "Synaptic current transfers instantaneously within the discrete 5 ms simulation time-step.",
     "Real ligand-gated and metabotropic receptors have finite activation, desensitization, and clearance timescales (AMPA/nAChR vs slow GABA_B / metabotropic receptors). Slow receptor dynamics are absent.",
     "kickthefly/sim/connectome/sim.py:step()"),

    ("No slow NMDA-like or neuromodulatory states",
     "DYNAMICS",
     "No voltage-dependent ion channel gating, NMDA slow kinetics, or broad volumetric neuromodulator wash.",
     "Neuropeptides and biogenic amines (octopamine, serotonin, dopamine) set global arousal, hunger, and sleep states. Except for modeled reward-driven plasticity, broad state transitions are simplified.",
     "kickthefly/sim/connectome/sim.py · README.md § Limitations"),

    ("Tonic depolarizing bias (0.20 threshold)",
     "TUNING",
     "A constant current bias of 0.20 (80% of threshold) is injected into every neuron to sustain basal activity.",
     "Without background excitation or unmodeled inputs, resting connectome simulations fall completely silent. The tonic bias maintains the biological ~5 Hz spontaneous brain-wide firing rate.",
     "kickthefly/lab/lab.py:PARAMS (bias=0.20) · kickthefly/sim/connectome/sim.py"),

    ("Gaussian membrane noise (sigma = 0.05)",
     "TUNING",
     "Zero-mean Gaussian current noise is added to every neuron at each 5 ms simulation step.",
     "Stochasticity mimics thermal channel noise, spontaneous miniature EPSPs, and unmodeled inputs from sensory organs, preventing artificial deterministic synchronization across identical network paths.",
     "kickthefly/lab/lab.py:PARAMS (noise_std=0.05) · kickthefly/sim/connectome/sim.py"),

    ("Alcohol inebriation as scripted motor degradation",
     "GAME RULE",
     "Sipping drives the real sweet taste pathway and PAM reward neurons, but the drunkenness is scripted: an "
     "inebriation level rises 0 to 1 and decays over ~45 s, scaling tremors, wobbly flight and slower escape reflexes.",
     "Ethanol acts pharmacologically across the whole nervous system (channel gating, dopaminergic and octopaminergic "
     "signalling), none of which is modelled. No simulated neuron is drunk: only the body's movement is degraded.",
     "README.md § Connectome vs Game Rule · kickthefly/game/kick_the_fly.py:Game._alcohol · kickthefly/game/kick3d.py:Game3D._alcohol3d"),

    ("Weak connections are kept as real wiring",
     "DATASET",
     "Every connection the reconstruction found at least 3 synapses for is simulated as a real synapse, at full "
     "strength for its count. The Lab's synapse threshold drops connections below any count you choose and re-runs "
     "the validated behaviors, so you can see which results depend on them.",
     "A contact reconstructed from a handful of synapses can be a mis-assigned fragment, an artefact of proofreading "
     "depth, or a real but negligible connection. At a threshold of 5 synapses, 40% of this connectome's connections "
     "and 2,828 neurons' entire input go; the four validated behaviors were re-run at 1, 4, 5, 6, 8 and 10.",
     "kickthefly/sim/wiring.py · kickthefly/lab/robustness.py · Lab > Connectome robustness"),

    ("Synapse sign from predicted neurotransmitters",
     "DATASET",
     "Each neuron's synapses are excitatory or inhibitory according to the transmitter MaleCNS v1.0 gives it: "
     "acetylcholine +1, GABA, glutamate and histamine -1, everything else 0. 85,484 of the 166,700 neurons have a "
     "measured transmitter; the rest are a machine-learning prediction with a confidence, and 4,366 signed neurons "
     "come with no confidence at all.",
     "A wrong prediction puts the wrong sign on every synapse that neuron makes. Lab > Connectome robustness > Sign "
     "flips flips the least certain ones over randomized trials and reports which validated behaviors survive: in "
     "this release the looming and antennal grooming pathways survive and the sugar -> MN9 pathway does not.",
     "kickthefly/sim/connectome/loader.py:NT_SIGN · kickthefly/sim/wiring.py · Lab > Connectome robustness"),

    ("Outdoor wind and sun reach the neurons through a scripted transduction",
     "GAME RULE",
     "In the Open field, steady wind drives the real wind-sensing JO-C/E neurons of each antenna, split by the cosine "
     "of where the wind comes from relative to the heading; sunlight drives the real photoreceptors, bright by the "
     "sun's elevation and split between the eyes by its azimuth. Both arenas' neurons are real; the split is a rule.",
     "Real Johnston's organs sense wind through antennal deflection mechanics that aren't modelled, and real "
     "photoreceptors see an image, not a luminance number. Wind also drives the head-touch escape DNs through JO "
     "(as in the fan arena), so a windy field launches the fly repeatedly: that part is the wiring's.",
     "kickthefly/game/outdoors.py:wind_drive, sun_light · Lab > Parameters (Open field, Outdoors)"),

    ("The orchard: fruit, trees and foraging are game rules",
     "GAME RULE",
     "Feeding on a fruit drives the real sugar-pathway taste neurons and the PAM reward neurons exactly as the sugar "
     "tool does (fermented fruit as the alcohol tool does) and heals the fly. The trees, the fruit, how many feeds a "
     "fruit holds, regrowth, the per-tree cap, the fly flying to a fruit, one fly per fruit, and landing are rules.",
     "The fly does not forage through its own circuitry: the game steers it to the nearest ripe fruit. Its neurons "
     "can still override that (a real escape or take-off abandons the fruit), and several flies notice each other "
     "only through the existing looming and touch pathways. No competition behaviour is scripted.",
     "kickthefly/game/outdoors.py:Orchard · kickthefly/game/kick3d.py:_orchard_tick · assay: orchard"),

    ("Alcohol's scent shares glomeruli with the zapper's",
     "DATASET",
     "The alcohol tool and fermented fruit smell through the real fermentation glomeruli DM1, DM2 and DP1m. Each "
     "other tool's scent is 5 glomeruli picked at random (game rule), and DM2 and DP1m happen to be in the zapper's.",
     "Mushroom-body training on alcohol (or fermented fruit) therefore partly generalises to the zapper and back. "
     "That follows from using the real glomeruli, not a bug, but anyone running feeding or training experiments in "
     "the orchard needs to know it.",
     "kickthefly/game/kick_the_fly.py:Brain (scent sets) · tests/test_outdoors.py"),

    ("No head-direction compass forms, from vision or from wind",
     "DYNAMICS",
     "Two tests of the E-PG ring: a driven visual-like wedge, and (2.7) the Open field's steady wind from 8 "
     "directions. Neither forms a bump: with wind the peak/trough contrast is 1.81x (1.71x without wind; 3.0x needed), "
     "it persists 101 ms (500 ms needed) and its position doesn't follow the wind beyond chance (p = 0.17).",
     "Nothing was tuned to make one appear, and no compass HUD ships. The ring's recurrent weights here are raw "
     "synapse counts with uniform efficacy; whether tuned weights would support a bump isn't tested.",
     "kickthefly/lab/compass.py:probe_epg_wind · validation: epg_compass_wind"),

    ("Left/Right asymmetry as EM reconstruction artifact risk",
     "DATASET",
     "Asymmetries in synaptic weights or firing between left and right hemibrains reflect both biology and reconstruction noise.",
     "MaleCNS v1.0 EM tracing has variable proofreading depth, staining artifacts, and truncation near slice boundaries. L/R differences may stem from incomplete reconstruction rather than true lateralization.",
     "README.md § Connectome Data · kickthefly/lab/headless.py"),

    ("Dynamic neural clamp breaks closed-loop sensorimotor feedback",
     "DYNAMICS",
     "Replaying recorded reference spikes forces activity on target neurons, overriding endogenous membrane state and breaking feedback loops (proprioception, visual flow, collision).",
     "In vivo closed-loop dynamics depend on continuous sensorimotor recurrence. Dynamic clamping isolates connectome structural changes from sensory input variation, but converts an autonomous organism into a driven open-loop circuit.",
     "kickthefly/lab/clamp.py · kickthefly/lab/labclamp.py · Lab > Neural clamp"),

    ("Global inhibition block (picrotoxin) as synaptic scale",
     "SYNAPSE",
     "A 0-100% severity slider scales inhibitory synaptic weights (W_inh * (1 - s)). Runaway excitation (>30 Hz) is an emergent recurrent network outcome; visual convulsion twitching is a game rule.",
     "Picrotoxin pharmacologically blocks ionotropic GABA_A (Rdl) chloride channels with non-uniform subunit affinities and dose kinetics. Receptor-dependent washes (octopamine, dopamine antagonists) and metabotropic cascades are omitted.",
     "kickthefly/sim/wiring.py · kickthefly/lab/robustness.py · Lab > Pharmacology (picrotoxin; moved from Robustness in 3.0)"),

    ("Driver lines: a literature table matched to connectome types by exact name",
     "DATASET",
     "Lab > Genetic toolkit picks neurons by split-GAL4 line. The line -> cell type mapping is copied from Meissner et al. 2025 (eLife, "
     "CC BY 4.0; 2,667 adult lines with cell types); neuron counts come from the pack. A cell-type name counts only if it is spelled "
     "exactly like a MaleCNS type (about half of the table's names are, since it uses the light-microscopy literature's names); the "
     "rest are shown as unmatched, never guessed.",
     "The table is the authors' claim about what a line labels, not a measurement on this connectome, and says nothing about expression "
     "strength or timing. The off-target note is only the paper's quality score. No GAL4 (non-split) mapping was found that could be "
     "redistributed, so those lines are not offered.",
     "kickthefly/lab/genetics.py · kickthefly/data/driver_lines.yaml · tools/build_driver_lines.py"),

    ("Thermogenetics: a temperature-gated current, nothing more",
     "BIOPHYSICS",
     "TrpA1 adds a depolarizing current above ~25 C (Pulver 2009) and shibire-ts a silencing current near 30 C (Kitamoto 2001) to the "
     "neurons that express them. The curve's other end, the 1 s / 40 s / 20 s kinetics and the current size are game rules. Escape vs "
     "temperature in the DNp01 assay is the model's response to that game-rule current.",
     "Temperature changes nothing else (no Q10 on any neuron or synapse). shibire-ts really blocks synaptic vesicle recycling at the "
     "terminal; the model silences the whole neuron, which also stops its spiking. Expression is all-or-nothing in the chosen types.",
     "kickthefly/lab/thermogenetics.py · Lab > Thermogenetics · assay thermo_escape"),

    ("Virtual patch clamp on a point-neuron model",
     "BIOPHYSICS",
     "Current-clamp steps and I-F curves of one simulated neuron. Every neuron is the same leaky integrate-and-fire unit, so in isolation "
     "every neuron has the same I-F curve; in the wired brain the membrane potential also carries the real synaptic input around it.",
     "Potential is in model units (threshold 1.0, reset 0.0), never millivolts; no spike waveform, dendrite, ion channel, adaptation or "
     "cell-specific property; 5 ms resolution. This is not electrophysiology and is not calibrated to any recording.",
     "kickthefly/lab/patchclamp.py · Lab > Patch clamp · inspector PATCH button"),

    ("Simulated calcium imaging is a forward model on spikes",
     "BIOPHYSICS",
     "Spikes are convolved with a two-exponential GCaMP kernel, averaged over an ROI, and given Poisson photon shot noise; dF/F is "
     "against a running mean (30 s by default, a setting). Kernel speeds: Chen 2013 Supplementary Table 3 (GCaMP6s/6f, mouse V1, 1 action "
     "potential) and Zhang 2023 (jGCaMP8m, fly visual responses), each checked in the paper.",
     "Linear in spikes, equal brightness for every neuron in an ROI, dF/F per spike and the photon budget are game parameters; no "
     "subthreshold calcium, bleaching, motion, scattering or neuropil; a real pipeline estimates F0 and segments cells.",
     "kickthefly/lab/imaging.py · Lab > Calcium imaging"),

    ("Pharmacology scales synapses by predicted transmitter",
     "SYNAPSE",
     "Picrotoxin, a cholinergic block, a glutamate-Cl block and a GABA-A agonist multiply the weights of every synapse whose presynaptic "
     "neuron is predicted to release that transmitter. The panel counts the affected synapses at each confidence level and can leave "
     "out low-confidence predictions. Octopamine and dopamine modulation are left out: those synapses are not in the simulated matrix.",
     "No receptor subtypes, subunit affinities, location on the cell, kinetics, washout or side effects. The simulator's slow global "
     "gain works against any drug that changes overall synaptic strength, so effects are read soon after a drug goes on.",
     "kickthefly/lab/pharmacology.py · kickthefly/sim/wiring.py · Lab > Pharmacology"),

    ("Predators: a game rule that reaches the fly only as something growing in its view",
     "GAME RULE",
     "The frog (a tongue that takes 0.07 s), the dragonfly (a chase from above that only takes a fly in the air) and the mantis (a "
     "10 cm/s creep, then a 0.06 s strike) are state machines with numbers chosen for play. Each shows up to the fly only as a growing "
     "circle, through the game's existing looming transduction onto LPLC2/LC4 -> DNp01; a capture fires the real touch neurons by body "
     "part. Whether the mantis's creep is noticed follows from the looming threshold (itself a game rule). The escape probabilities of "
     "the Predator escape assay are MODEL PREDICTIONS with 95% Wilson intervals.",
     "No real frog, dragonfly or mantis is measured or implied: no published speed, reach or timing is used. The fly in the assay does "
     "not move, so an 'escape' is DNp01 crossing the game's escape threshold before the capture, not a flight path that clears the "
     "tongue. The larva has no looming detectors in this model, so it has no predators.",
     "kickthefly/game/predators.py · kickthefly/game/predator_play.py · kickthefly/lab/predators.py · Lab > Assays > Predator escape"),

    ("Rain, gusts and storms: touch, humidity, wind and light pokes with game-rule numbers",
     "GAME RULE",
     "Raindrop hits poke the real touch neurons by body part (wings, body, head, legs), the wet air pokes the humidity neurons, a gust "
     "adds wind speed to the existing wind -> JO-C/E transduction, and lightning pokes the photoreceptors. How often drops hit, which "
     "part (by exposed area), how hard, when the wings count as wet (the pool's rule), the gusts' size and length, the lightning's "
     "timing and the storm preset are game rules. Reduced flashing turns the screen's lightning into one slow swell.",
     "Raindrops are not simulated: a hit is a touch pulse of a set strength. No shelter under the trees, no evaporation, no cooling, "
     "no wind shear. The humidity neurons idle at about 18 Hz in this model, so rain moves their mean rate only about 14%, though "
     "their 100 ms peaks rise clearly.",
     "kickthefly/game/weather.py · kickthefly/game/kick3d.py · Lab > Parameters (weather.*) · protocol weather:"),

    ("The kitchen: every part is a game rule on the neurons the fly already has",
     "GAME RULE",
     "Fruit in the bowl is the orchard's feeding (taste and PAM reward neurons). The vinegar trap pokes the fermentation glomeruli "
     "DM1/DM2/DP1m, and the fly flies to it only if those neurons' own firing is well above its calm rate; a fly that hovers over the "
     "mouth falls in and is stuck (trap physics are a game rule). The sink is the pool's water in a basin (humidity neurons, wet wings, "
     "drowning); the burner is the lamp's heat. The cook swats every 10-20 s at where a fly was, slowly enough to be seen coming.",
     "The counter is the floor; the room is the 3D room with different furniture. No real vinegar chemistry, no real trap catch rates, "
     "no stove heat transfer. Whether a given fly dodges a swat is a MODEL PREDICTION, not a measurement.",
     "kickthefly/game/kitchen.py · kickthefly/game/kick3d.py · arena E > kitchen (3D only)"),

    ("Microphone to Johnston's organ: air pressure treated as antennal vibration",
     "GAME RULE",
     "The microphone's sound is band-passed into JO-B (10-100 Hz) and JO-A (100-1,000 Hz) and becomes current on those neurons (50 and "
     "88 in the dataset); JO-C/E are left to the wind. A 2019 review says JO-B prefers low and JO-A higher frequencies, and that "
     "JO-A/B are the sound-sensitive groups (Kamikouchi 2009). Everything else, the filters, the loudness mapping, the noise gate and "
     "the equal current per neuron, is a game rule. The hum demo's P1 and song-motor-neuron ratios are MODEL PREDICTIONS.",
     "A microphone is not an antenna; the real organ is a mechanical resonator and each neuron has its own tuning. Humming drives the "
     "P1 courtship cluster here but not the song motor neurons, and the model is not tuned to the song's 35 ms rhythm (a 70 ms rhythm "
     "drives P1 about as much). The song numbers (about 220 Hz pulses every ~35 ms) were read in search-result summaries, not the "
     "papers. Opt-in, off at every launch; nothing is recorded or sent.",
     "kickthefly/core/mic.py · kickthefly/lab/audio.py · Esc > Mic and streamer · assay hum_demo"),

    ("Streamer mode: chat votes through the game's own actions",
     "GAME RULE",
     "Viewers of a Twitch channel vote (!tool, !arena, !surgery) and the winner runs the same action a player would. The streamer "
     "chooses which commands count; a round, a cooldown, a minimum vote count and rate limits are all game rules. It only reads chat "
     "anonymously (no login, no token), keeps no names, and shows its connection on screen.",
     "Nothing here touches the connectome. The anonymous login is described in Twitch developer-forum threads, not the current "
     "official documentation, so Twitch may stop allowing it. Off at every launch and never during validation, protocols or tests.",
     "kickthefly/core/streamer.py · kickthefly/core/netguard.py · Esc > Mic and streamer"),

    ("Hemifield lesion as static connectome wiring ablation",
     "DATASET",
     "One-click surgery silences visual pathways (LC10, LPLC2, LC4, LPTC, VS, HS) on one hemifield or an entire hemibrain, reported strictly as a wiring outcome.",
     "Physical brain lesions in Drosophila trigger axotomy, Wallerian degeneration, glial immune responses, and homeostatic synaptic compensation. The simulation models pure static silencing of cell rows without injury pathology.",
     "kickthefly/lab/lesions.py · kickthefly/game/kick_the_fly.py · Lab > Brain surgery"),

    ("Central complex head-direction ring attractor requires fine-tuned E/I balance",
     "BIOPHYSICS",
     "The sim tests the EPG/PEN/Delta7 compass network directly under raw connectome weights, without weight-tuning.",
     "Biological head-direction tracking in the central complex relies on precisely balanced recurrent excitation and broad Delta7 lateral inhibition to sustain a localized activity bump and track rotational visual/wind cues. With the connectome's signed synapse counts (normalized by each neuron's total input) and no ring weights tuned, bump contrast and persistence fail; reported as a negative validation result rather than tuned.",
     "kickthefly/lab/compass.py · kickthefly/lab/validation.py:epg_compass"),

    ("Neuron morphology: drawn, not simulated; real skeletons for ten neurons only",
     "DATASET",
     "Ten neurons (two each of DNp01, DNa02, MBON01, MBON14 and KCg) are drawn from 21 points sampled along their real "
     "EM skeletons from neuPrint (MaleCNS v1.0), cached after one download. Every other neuron is drawn as an estimated "
     "fiber from its real cell body toward its synaptic partners. Either way the simulation treats each neuron as a "
     "single point.",
     "Real Drosophila neurons have extensive arbors, and where on the arbor a synapse sits shapes its effect. The LIF "
     "point-neuron model ignores geometry entirely, so the drawn shape never changes what a neuron does.",
     "kickthefly/sim/morphology.py · kickthefly/game/kick_the_fly.py:BrainView"),

    ("Compute backends and float precision",
     "DYNAMICS",
     "NumPy (cpu) is the reference. numba and torch-cpu reproduce it bit for bit (same float32 operations in the same "
     "order), so results are identical on them. GPU backends (torch-cuda, torch-rocm) may sum a neuron's inputs in a "
     "different order; single-precision rounding then differs and, the network being chaotic, individual spikes diverge "
     "after a few hundred steps while firing statistics agree (within 2% brain-wide in the tests). State precision "
     "float32 (default) or float64 changes the numbers the same way.",
     "Real neurons are not deterministic at all; the point of bit-exactness is reproducibility of this model, not "
     "biological fidelity. The backend that ran is recorded with every result.",
     "kickthefly/sim/connectome/backends.py · tests/test_backends.py"),

    ("Courtship song: the buzz is synthesized",
     "GAME RULE",
     "When the two ps1 wing motor neurons, averaged over about a second, pass the song threshold (Lab > Parameters), "
     "the game plays a pulse-song buzz (220 Hz pulses every 35 ms) and logs SONG. pIP10 -> ps1 is validated; P1 -> "
     "ps1 is consistent but too weak. Nothing in play drives pIP10: stimulate it in brain surgery (strong wing hits "
     "can also fire ps1 past the threshold).",
     "Real pulse song comes from the wing's vibration, shaped by thoracic mechanics and feedback. The sim has no wing, "
     "and pIP10 reaches ps1 only through VNC interneurons, so the sound is the game's, not the fly's.",
     "kickthefly/game/kick_the_fly.py:Game.readouts · validation: courtship_song, p1_courtship_song"),

    ("Aggression lunge from the TK-FruM and pC1 neurons",
     "GAME RULE",
     "With two or more flies, the aggression neurons (AVLP727m, \"Asahina 2014: TK-FruM\", and the pC1 cluster, "
     "P1 included) above the lunge threshold throw the fly at the nearest fly within reach (2D and 3D). Flies only "
     "notice each other through their looming and touch neurons, which in testing never took the aggression neurons "
     "past ~1.5x; stimulate P1 in brain surgery to see a lunge (AVLP727m alone is too few neurons to move it).",
     "Real aggression depends on pheromones (e.g. 7-tricosene through Gr32a), octopamine, social experience and a "
     "sequenced motor program. None of that is modelled; the lunge itself is scripted.",
     "kickthefly/game/kick_the_fly.py:Game._aggression · protocols/male_aggression.yaml"),

    ("Optomotor EMD stage in front of T4/T5",
     "GAME RULE",
     "A wide-field yaw rotation becomes current on the T4/T5 subtypes whose preferred direction it matches in each "
     "eye (a saturating Reichardt-style detector output). Everything after T4/T5 is the connectome: rightward "
     "rotation excites DNa01_R/DNa02_R (validated). Used by validation and protocols; the game loop doesn't feed it "
     "the fly's own view.",
     "Real motion detection is computed from photoreceptors through lamina and medulla neurons (L1-L3, Mi1, Tm3, "
     "Tm1/2/4/9) onto T4/T5 dendrites. Pixel input through the sim's photoreceptors carries no usable motion signal, "
     "so this stage replaces that computation.",
     "kickthefly/lab/assays.py:emd_stage · kickthefly/game/outdoors.py:emd_motion_drive · validation: optomotor_turning"),

    ("Thermo arena temperature gradient",
     "GAME RULE",
     "The thermo arena (2D and 3D) runs from 15 C at the left wall to 35 C at the right. The fly's distance from the "
     "comfortable middle drives its cold (TRN_VP3) or hot (TRN_VP2) antennal neurons, and the extremes hurt it on the "
     "floor. The fly doesn't seek the comfortable middle: nothing it does comes from the temperature.",
     "Real flies sense temperature through thermosensitive channels (Gr28b in the hot cells, Brv1 and IR21a/25a/93a "
     "in the cold cells) that adapt, and they navigate gradients with steering behavior. The hot- and cold-cell to "
     "projection-neuron steps are validated here; thermotaxis is not.",
     "kickthefly/game/kick_the_fly.py:Game._thermo_tick · validation: hot_trn_to_vp2pn, cold_trn_to_vp3pn"),

    ("Day/night cycle and the dFB sleep readout",
     "GAME RULE",
     "With Outdoors: day length above 0 the sun circles (noon +60, midnight -30 deg), daylight drives the "
     "photoreceptors and the morning clock neurons l-LNv/s-LNv directly, and the scene darkens at night. SLEEP is "
     "logged when the dorsal fan-shaped body (FB6/FB7) passes its threshold, and the fly then stops taking off and "
     "walking on its own for 2 s. In testing a full day never took dFB past ~1.5x, so it only sleeps when FB6/FB7 are "
     "stimulated in brain surgery.",
     "Real circadian timing is a molecular clock (per/tim/Clk/cyc) and slow PDF signalling over hours, and sleep "
     "pressure builds up over waking. The sim is point neurons over seconds: it has no clock and no sleep drive.",
     "kickthefly/game/outdoors.py:sun_now · kickthefly/game/kick_the_fly.py:Game._sleep"),

    ("cVA pheromone puff",
     "GAME RULE",
     "The puff (its visible cloud showing its reach, 250 px in 2D and 2.5 m in 3D, and its strength) is a game rule. What it drives is real: the "
     "204 DA1 olfactory receptor neurons, and from there whatever the connectome does. Nothing ties cVA to aggression "
     "beyond that wiring.",
     "cVA is a male pheromone sensed by Or67d neurons (DA1). Real puffs spread and fade; here a puff drives the DA1 "
     "ORNs in proportion to distance, with a short-lived visual cloud indicating reach.",
     "kickthefly/game/kick_the_fly.py:Game.use_tool · kickthefly/lab/validation.py:or67d_to_da1pn"),

    ("Decoy female and the COURTSHIP tag",
     "GAME RULE",
     "The decoy's body, the contact distance, and the COURTSHIP tag, which contact triggers without reading any neuron. "
     "Contact drives the real foreleg GRNs annotated putative ppk23/ppk25 (LgLG5-8); a live HUD line displays real "
     "LgLG5-8 and P1 firing during contact (REAL). How much of that reaches P1 is the connectome's (weak: foreleg_grn_to_p1 fails).",
     "Real males taste female cuticular pheromones (7,11-HD, 7,11-ND) through foreleg ppk23/ppk25 neurons and court. "
     "Nothing in play reads P1 to decide courtship.",
     "kickthefly/game/kick_the_fly.py:Game._decoy · kickthefly/lab/validation.py:foreleg_grn_to_p1"),

    ("Plume tracking (headless assay)",
     "GAME RULE",
     "assays.plume_tracking_fly: the plume (Gaussian, downwind of a source, on and off in time) and the navigation "
     "(surge upwind in odor, cast crosswind without) are computed from the geometry. ORN_DM1 are driven while the fly "
     "is in odor, but their firing doesn't steer it. The game's open field has no plume.",
     "Real plume tracking is closed-loop between smell, wind sensing and steering. Whether the connectome's own "
     "steering neurons would do it wasn't tested.",
     "kickthefly/lab/assays.py:plume_tracking_fly"),

    ("Mushroom body extinction and second-order conditioning",
     "SYNAPSE",
     "No plasticity rule was added for either: they run on the existing learning rule (dopamine x Kenyon cell "
     "eligibility, with its reversal term), so the brain has to produce them through its own dopamine neurons. Both "
     "are validation tests (mb_extinction, mb_second_order); see the Validation page for whether it does.",
     "In real flies extinction forms a parallel opposing memory through reward dopamine neurons (Felsenberg et al. "
     "2018), and second-order conditioning needs MBON-to-dopamine-neuron feedback.",
     "kickthefly/lab/assays.py:extinction_fly, second_order_fly · kickthefly/core/memory.py"),

    ("Neurodex: what counts as discovered (3.0)",
     "GAME RULE",
     "Discovered = 6+ spikes/s, 3x its calm rate and a Poisson count test (alpha ~1e-11) for 150 ms. A calm fly "
     "discovers nothing (3.0 Day 2 decision); only play or stimulation counts.",
     "Nothing in a real fly is 'discovered'. The rule reads the type as a whole, so sparse big types (Kenyon cells) are "
     "mostly found by stimulating them, and the entry says so.",
     "kickthefly/core/neurodex.py · docs/neurodex.md"),

    ("Neurodex literature facts (3.0)",
     "DATASET",
     "One sentence and a citation for about 30 curated types, written from the paper's abstract or full text and checked "
     "against it. Every other type shows dataset data only.",
     "A fact is what the paper reported in a real fly, not what this simulation does. That a dataset type is the paper's "
     "named neuron is the dataset's annotation (aDN1/aDN2 = DNg62/DNge078).",
     "kickthefly/data/neurodex_facts.yaml"),

    ("Kill cam (3.0)",
     "GAME RULE",
     "On death, the last 6 s of each neuron's own firing rate is replayed at 0.25x on the brain panel, with the 12 neurons "
     "that rose most highlighted. Frames in memory: nothing feeds back, the body is not re-simulated.",
     "Death is a game rule (the drive is cancelled over 1.5 s), so the replay ends at that cut-off. A rising rate shows "
     "what was active, not what caused the death.",
     "kickthefly/core/killcam.py · docs/killcam.md"),

    ("Share codes and experiment bundles (3.0)",
     "GAME RULE",
     "Containers for settings and results the game already has; they add nothing to the simulation. A bundle rerun is "
     "bit-exact on CPU backends, and statistical (three numbers fixed beforehand) on a GPU backend.",
     "A matching rerun shows the software is deterministic on that backend. It does not show the result is biologically "
     "true.",
     "kickthefly/core/sharecode.py · kickthefly/lab/bundle.py · docs/share-codes.md · docs/bundles.md"),

    ("Neuron of the Day (3.0)",
     "GAME RULE",
     "A launch card picks one curated type by date and shows its fact. Try it stimulates that type in brain surgery or "
     "the Lab laser. Its own setting, default on; nothing is sent or remembered about you.",
     "Constant current is not how a real neuron is activated in a fly, and experimental driver lines are not modelled.",
     "kickthefly/core/neuron_of_day.py · docs/neurodex.md"),
)


def page_assumptions(m: ui.Menu, surf, rect, mouse) -> None:
    m.text(surf, "MODEL ASSUMPTIONS & SIMPLIFICATIONS", (rect.x + 24, rect.y + 16), ui.INK, m.f_head)
    m.text(surf, "Scientific caveats and approximations distinguishing the simulation from living biology.",
           (rect.x + 24, rect.y + 50), ui.LABEL, m.f_small)
    body = pygame.Rect(rect.x + 16, rect.y + 80, rect.w - 32, rect.h - 80 - 70)
    key = "lab_assumptions"
    off = int(m.scroll.get(key, 0))
    m.clip = body
    prev = surf.get_clip()
    surf.set_clip(body)
    y = body.y + 4 - off

    category_colors = {
        "SYNAPSE": (100, 180, 240),
        "BIOPHYSICS": (150, 120, 220),
        "DYNAMICS": (240, 160, 60),
        "TUNING": (80, 200, 140),
        "DATASET": (240, 100, 100),
        "GAME RULE": (220, 150, 50),
    }

    card_h = 136
    max_w = body.w - 74
    for title, cat, sim_rule, bio_reality, docs in ASSUMPTIONS:
        card = pygame.Rect(body.x, y, body.w - 12, card_h)
        pygame.draw.rect(surf, (24, 28, 38), card, border_radius=8)
        pygame.draw.rect(surf, (45, 52, 68), card, 1, border_radius=8)

        # Category chip
        col = category_colors.get(cat, ui.LABEL)
        chip = pygame.Rect(card.x + 12, card.y + 10, 84, 20)
        pygame.draw.rect(surf, (15, 18, 26), chip, border_radius=4)
        pygame.draw.rect(surf, col, chip, 1, border_radius=4)
        m.text(surf, cat, chip.center, col, m.f_small, "center")

        # Title
        m.text(surf, title, (card.x + 106, card.y + 10), ui.INK, m.f_bold)

        # Sim rule
        m.text(surf, "Sim:", (card.x + 14, card.y + 36), (140, 180, 220), m.f_small)
        m.wrapped(surf, sim_rule, (card.x + 50, card.y + 36), max_w, ui.TEXT, m.f_small, max_lines=2)

        # Biology reality
        m.text(surf, "Bio:", (card.x + 14, card.y + 68), (220, 150, 100), m.f_small)
        m.wrapped(surf, bio_reality, (card.x + 50, card.y + 68), max_w, (185, 190, 200), m.f_small, max_lines=2)

        # Documentation reference
        m.text(surf, "Doc:", (card.x + 14, card.y + 104), ui.LABEL, m.f_small)
        m.text(surf, docs, (card.x + 50, card.y + 104), (130, 160, 210), m.f_small)

        y += card_h + 12

    m.content_h[key] = max(0, y + off - body.bottom + 8)
    surf.set_clip(prev)
    m.clip = None

    m.button(surf, (rect.right - 164, rect.bottom - 58, 140, 42), "Back", m.back, style="primary", id=("assumptions", "back"))


def page_asymmetry(m: ui.Menu, surf, rect, mouse) -> None:
    host = m.host
    m.text(surf, "LEFT / RIGHT ASYMMETRY AUDIT", (rect.x + 24, rect.y + 16), ui.INK, m.f_head)
    m.text(surf, "Audit bilateral differences in connectome structure and spontaneous turning bias.",
           (rect.x + 24, rect.y + 50), ui.LABEL, m.f_small)

    body = pygame.Rect(rect.x + 16, rect.y + 80, rect.w - 32, rect.h - 80 - 70)
    key = "lab_asymmetry"
    off = int(m.scroll.get(key, 0))
    m.clip = body
    prev = surf.get_clip()
    surf.set_clip(body)
    y = body.y + 4 - off

    mirror = bool(host.cfg["brain.mirror_weights"]) if hasattr(host, "cfg") else False

    # Status & Context Card
    card_h = 130
    card = pygame.Rect(body.x, y, body.w - 12, card_h)
    pygame.draw.rect(surf, (24, 28, 38), card, border_radius=8)
    pygame.draw.rect(surf, (45, 52, 68), card, 1, border_radius=8)

    chip_col = (240, 160, 60) if mirror else (100, 180, 240)
    chip_txt = "GAME RULE: MIRROR-AVERAGED" if mirror else "CONNECTOME: RAW DATA"
    chip = pygame.Rect(card.x + 12, card.y + 10, 210, 22)
    pygame.draw.rect(surf, (15, 18, 26), chip, border_radius=4)
    pygame.draw.rect(surf, chip_col, chip, 1, border_radius=4)
    m.text(surf, chip_txt, chip.center, chip_col, m.f_small, "center")

    bias_txt = "+1.00 Hz (symmetric weights)" if mirror else "+0.10 Hz (right turn bias)"
    m.text(surf, f"Baseline Turning Bias: {bias_txt}", (card.x + 235, card.y + 12), ui.INK, m.f_bold)

    expl = (
        "In the raw MaleCNS v1.0 connectome, bilateral asymmetries arise from both true biology and uneven electron "
        "microscopy (EM) reconstruction and proofreading depth between hemispheres. Descending steering neurons DNa01 and "
        "DNa02 drive a mild spontaneous rightward turning bias in quiet walking.\n"
        "Mirror-averaging synaptic weights across 77,507 paired bilateral neurons enforces exact structural symmetry. "
        "Because this alters real connectome data, it is tagged strictly as a Game Rule."
    )
    m.wrapped(surf, expl, (card.x + 14, card.y + 40), card.w - 28, ui.TEXT, m.f_small, max_lines=4)
    y += card_h + 14

    # Table Header Card
    hdr_h = 32
    hdr = pygame.Rect(body.x, y, body.w - 12, hdr_h)
    pygame.draw.rect(surf, (32, 38, 52), hdr, border_radius=6)
    m.text(surf, "Key Cell Type", (hdr.x + 14, hdr.centery), ui.INK, m.f_bold, "midleft")
    m.text(surf, "Functional Role", (hdr.x + 120, hdr.centery), ui.LABEL, m.f_small, "midleft")
    m.text(surf, "Count L/R", (hdr.x + 370, hdr.centery), ui.LABEL, m.f_small, "midleft")
    m.text(surf, "In-Syn L/R", (hdr.x + 470, hdr.centery), ui.LABEL, m.f_small, "midleft")
    m.text(surf, "Out-Syn L/R", (hdr.x + 580, hdr.centery), ui.LABEL, m.f_small, "midleft")
    m.text(surf, "Calm Rate L/R", (hdr.x + 690, hdr.centery), ui.LABEL, m.f_small, "midleft")
    m.text(surf, "Diff (R-L)", (hdr.right - 14, hdr.centery), ui.LABEL, m.f_small, "midright")
    y += hdr_h + 6

    rows_data = [
        ("DNa01", "Steering descending command", "1 / 1", "393 / 402", "577 / 577", "316 / 311", "532 / 532", "2.4 / 3.0 Hz", "2.2 / 3.2 Hz", "+0.60 Hz", "+1.00 Hz"),
        ("DNa02", "Sharp steering command", "1 / 1", "695 / 716", "1084 / 1084", "339 / 322", "534 / 534", "6.0 / 5.6 Hz", "5.6 / 6.6 Hz", "-0.40 Hz", "+1.00 Hz"),
        ("LC10", "Courtship tracking / fixation", "479 / 481", "78 / 97", "86 / 106", "54 / 60", "72 / 80", "3.8 / 3.9 Hz", "4.0 / 4.1 Hz", "+0.18 Hz", "+0.07 Hz"),
        ("LPLC2", "Rapid looming escape", "94 / 91", "232 / 283", "243 / 295", "93 / 97", "125 / 131", "10.2 / 6.5 Hz", "10.5 / 7.7 Hz", "-3.66 Hz", "-2.79 Hz"),
        ("LC4", "Collision looming avoidance", "112 / 90", "191 / 224", "210 / 244", "79 / 85", "116 / 125", "0.7 / 1.1 Hz", "0.8 / 0.9 Hz", "+0.38 Hz", "+0.13 Hz"),
        ("DNp01", "Braking / backward command", "1 / 1", "558 / 482", "836 / 836", "136 / 125", "200 / 200", "6.6 / 6.2 Hz", "7.6 / 8.6 Hz", "-0.40 Hz", "+1.00 Hz"),
    ]

    row_h = 36
    for t, role, counts, in_raw, in_mir, out_raw, out_mir, r_raw, r_mir, d_raw, d_mir in rows_data:
        rbox = pygame.Rect(body.x, y, body.w - 12, row_h)
        pygame.draw.rect(surf, (20, 24, 33), rbox, border_radius=6)
        m.text(surf, t, (rbox.x + 14, rbox.centery), ui.INK, m.f_bold, "midleft")
        m.text(surf, role, (rbox.x + 120, rbox.centery), (150, 180, 220), m.f_small, "midleft")
        m.text(surf, counts, (rbox.x + 370, rbox.centery), ui.TEXT, m.f_small, "midleft")
        m.text(surf, in_mir if mirror else in_raw, (rbox.x + 470, rbox.centery), ui.TEXT, m.f_small, "midleft")
        m.text(surf, out_mir if mirror else out_raw, (rbox.x + 580, rbox.centery), ui.TEXT, m.f_small, "midleft")
        m.text(surf, r_mir if mirror else r_raw, (rbox.x + 690, rbox.centery), ui.TEXT, m.f_small, "midleft")
        diff_str = d_mir if mirror else d_raw
        diff_val = abs(float(diff_str.replace(" Hz", "")))
        diff_col = ui.GOOD if diff_val < 0.2 else (240, 160, 60)
        m.text(surf, diff_str, (rbox.right - 14, rbox.centery), diff_col, m.f_small, "midright")
        y += row_h + 6

    m.content_h[key] = max(0, y + off - body.bottom + 8)
    surf.set_clip(prev)
    m.clip = None

    btn_txt = "Mirror weights: ON [Rule]" if mirror else "Mirror weights: OFF [Raw]"
    def toggle():
        if hasattr(host, "toggle_mirror_weights"):
            host.toggle_mirror_weights()
    m.button(surf, (rect.x + 24, rect.bottom - 58, 250, 42), btn_txt, toggle,
             id=("asymmetry", "toggle_mirror"), tip="Toggle bilateral weight symmetrization [GAME RULE]")
    m.button(surf, (rect.right - 164, rect.bottom - 58, 140, 42), "Back", m.back, style="primary", id=("asymmetry", "back"))


_bench_job = None


def page_benchmark(m: ui.Menu, surf, rect, mouse) -> None:
    from kickthefly.lab import benchmark
    global _bench_job
    host = m.host
    m.text(surf, "SIMULATION BENCHMARK", (rect.x + 24, rect.y + 16), ui.INK, m.f_head)
    m.text(surf, "Throughput, real-time pace, neurons/sec and memory footprint across 1, 8, and 16 flies.",
           (rect.x + 24, rect.y + 50), ui.LABEL, m.f_small)

    body = pygame.Rect(rect.x + 16, rect.y + 80, rect.w - 32, rect.h - 80 - 70)
    key = "lab_benchmark"
    off = int(m.scroll.get(key, 0))
    m.clip = body
    prev = surf.get_clip()
    surf.set_clip(body)
    y = body.y + 4 - off

    res = getattr(host, "_benchmark_results", None)
    if res is None:
        res = benchmark.load_benchmark_results()
        host._benchmark_results = res

    # System card
    card_h = 94
    card = pygame.Rect(body.x, y, body.w - 12, card_h)
    pygame.draw.rect(surf, (24, 28, 38), card, border_radius=8)
    pygame.draw.rect(surf, (45, 52, 68), card, 1, border_radius=8)

    sys_info = (res or {}).get("system") or benchmark.get_system_info()
    b_name = (res or {}).get("backend") or getattr(getattr(getattr(host, "brain", None), "sim", None), "backend", None)
    b_name = getattr(b_name, "name", str(b_name or "CPU (NumPy)"))
    b_dev = (res or {}).get("device") or getattr(getattr(getattr(host, "brain", None), "sim", None), "backend", None)
    b_dev = getattr(b_dev, "device", str(b_dev or "CPU"))
    m.text(surf, f"CPU: {sys_info.get('cpu_model')} ({sys_info.get('cpu_count')} threads)", (card.x + 14, card.y + 12), ui.INK, m.f_bold)
    m.text(surf, f"OS: {sys_info.get('os')}  ·  Python: {sys_info.get('python')}", (card.x + 14, card.y + 32), ui.TEXT, m.f_small)
    m.text(surf, f"Sim Backend: {b_name}  ·  Device: {b_dev}", (card.x + 14, card.y + 50), (120, 220, 240), m.f_small)
    m.text(surf, "Connectome: Janelia MaleCNS v1.0 (166,700 neurons, 10,272,125 synapses)", (card.x + 14, card.y + 70), (140, 180, 220), m.f_small)
    y += card_h + 16

    # Results Table
    if res and "records" in res:
        hdr_h = 32
        hdr = pygame.Rect(body.x, y, body.w - 12, hdr_h)
        pygame.draw.rect(surf, (32, 38, 52), hdr, border_radius=6)
        m.text(surf, "Flies", (hdr.x + 14, hdr.centery), ui.INK, m.f_bold, "midleft")
        m.text(surf, "Paced Rate", (hdr.x + 80, hdr.centery), ui.LABEL, m.f_small, "midleft")
        m.text(surf, "Real-Time", (hdr.x + 230, hdr.centery), ui.LABEL, m.f_small, "midleft")
        m.text(surf, "Uncapped Rate", (hdr.x + 340, hdr.centery), ui.LABEL, m.f_small, "midleft")
        m.text(surf, "Max Speedup", (hdr.x + 490, hdr.centery), ui.LABEL, m.f_small, "midleft")
        m.text(surf, "Throughput", (hdr.x + 620, hdr.centery), ui.LABEL, m.f_small, "midleft")
        m.text(surf, "Memory", (hdr.right - 14, hdr.centery), ui.LABEL, m.f_small, "midright")
        y += hdr_h + 6

        row_h = 36
        for r in res["records"]:
            rbox = pygame.Rect(body.x, y, body.w - 12, row_h)
            pygame.draw.rect(surf, (20, 24, 33), rbox, border_radius=6)
            m.text(surf, f"{r['flies']} flies", (rbox.x + 14, rbox.centery), ui.INK, m.f_bold, "midleft")
            paced_s = f"{r['paced_steps_per_s']:.1f} steps/s"
            m.text(surf, paced_s, (rbox.x + 80, rbox.centery), ui.TEXT, m.f_small, "midleft")
            rt_ratio = r["paced_realtime_ratio"]
            rt_col = ui.GOOD if rt_ratio >= 0.99 else (240, 160, 60) if rt_ratio >= 0.8 else ui.BAD
            m.text(surf, f"{rt_ratio:.2f}x", (rbox.x + 230, rbox.centery), rt_col, m.f_bold, "midleft")
            lat_ms = r.get('uncapped_latency_ms', round(1000.0 / max(r['uncapped_steps_per_s'], 1e-6), 2))
            uncap_s = f"{r['uncapped_steps_per_s']:.1f} st/s ({lat_ms:.2f}ms)"
            m.text(surf, uncap_s, (rbox.x + 340, rbox.centery), ui.TEXT, m.f_small, "midleft")
            m.text(surf, f"{r['uncapped_realtime_ratio']:.2f}x", (rbox.x + 490, rbox.centery), (140, 200, 240), m.f_small, "midleft")
            tp_s = f"{r['neurons_per_sec'] / 1e6:.1f} M neurons/s"
            m.text(surf, tp_s, (rbox.x + 620, rbox.centery), (170, 230, 190), m.f_small, "midleft")
            m.text(surf, f"{r['memory_mb']:.0f} MB", (rbox.right - 14, rbox.centery), ui.LABEL, m.f_small, "midright")
            y += row_h + 6

        m.text(surf, f"Measured at {res.get('timestamp')}  ·  {res.get('seconds_per_run', 3):.0f}s per condition  ·  Target: 200 steps/s",
               (body.x + 12, y + 10), ui.LABEL, m.f_small)
        y += 36
    else:
        m.text(surf, "No benchmark results found. Click 'Run Benchmark' to measure multi-fly simulation throughput.",
               (body.x + 14, y + 10), ui.TEXT, m.f_text)
        y += 40

    m.content_h[key] = max(0, y + off - body.bottom + 8)
    surf.set_clip(prev)
    m.clip = None

    # Benchmark Execution / Status
    busy = _bench_job is not None and _bench_job["thread"].is_alive()
    if busy:
        status_txt = f"Benchmarking... {_bench_job.get('status', '')}"
        m.text(surf, status_txt, (rect.x + 24, rect.bottom - 48), ui.AMBER, m.f_bold)
    else:
        if _bench_job is not None and _bench_job.get("result"):
            host._benchmark_results = _bench_job["result"]
            _bench_job = None

        def start_bench():
            global _bench_job
            import threading
            job_state = {"status": "starting", "result": None}
            def worker():
                def progress(done, total, label):
                    job_state["status"] = f"{label} ({done}/{total})"
                b_choice = host.cfg.get("brain.backend", "auto") if hasattr(host, "cfg") and host.cfg is not None else "auto"
                r = benchmark.run_benchmark(fly_counts=(1, 8, 16), seconds=2.5, progress_cb=progress, backend=b_choice)
                benchmark.save_benchmark_results(r)
                job_state["result"] = r
            t = threading.Thread(target=worker, name="benchmark-worker", daemon=True)
            job_state["thread"] = t
            _bench_job = job_state
            t.start()

        m.button(surf, (rect.x + 24, rect.bottom - 58, 220, 42), "Run Benchmark", start_bench,
                 id=("benchmark", "run"), tip="Measure simulation throughput across 1, 8, 16 flies")

    m.button(surf, (rect.right - 164, rect.bottom - 58, 140, 42), "Back", m.back, style="primary", id=("benchmark", "back"))


def install(menu: ui.Menu) -> None:
    menu.pages["lab"] = page_hub
    menu.pages["lab_params"] = page_params
    menu.pages["lab_assays"] = lambda *a: page_assays(*a)
    menu.pages["lab_validation"] = lambda *a: page_validation(*a)
    menu.pages["lab_export"] = lambda *a: page_export(*a)
    menu.pages["lab_protocols"] = lambda *a: page_protocols(*a)
    menu.pages["lab_assumptions"] = page_assumptions
    menu.pages["lab_asymmetry"] = page_asymmetry
    menu.pages["lab_benchmark"] = page_benchmark
    from kickthefly.lab import labclassroom, labclamp, labcritical, labdiff, lablaser, labpsych, labtoolkit, labwiring
    menu.pages.update(labtoolkit.PAGES)                 # 3.0 day 2: genetic toolkit, thermogenetics, patch, imaging, pharmacology
    menu.pages["lab_wiring"] = labwiring.page
    menu.pages["lab_critical"] = labcritical.page
    menu.pages["lab_clamp"] = labclamp.page
    menu.pages["lab_diff"] = labdiff.page
    menu.pages["lab_laser"] = lablaser.page
    menu.pages["lab_psych"] = labpsych.page
    menu.pages["lab_classroom"] = labclassroom.page
    ui.TAG_COLORS.setdefault("MODEL", (150, 120, 220))


def short(path, n: int = 70) -> str:
    """A long path shortened in the middle so it fits on one line."""
    t = str(path)
    return t if len(t) <= n else t[: n // 2 - 2] + "…" + t[-(n // 2 - 1):]


# --- charts ----------------------------------------------------------------------------------------------------------
def draw_chart(m: ui.Menu, surf, rect: pygame.Rect, xs, series, x_label: str, y_label: str, y_max: float | None = None,
               x_fmt="{:g}", connect: bool = True) -> None:
    """Line chart with 95% CI bands. series: [(label, color, [mean_ci dicts per x])]."""
    import math

    pygame.draw.rect(surf, (12, 14, 20), rect, border_radius=8)
    plot = pygame.Rect(rect.x + 52, rect.y + 44, rect.w - 70, rect.h - 82)
    vals = [c["hi"] if not math.isnan(c.get("hi", float("nan"))) else c["mean"] for _, _, cs in series for c in cs
            if not math.isnan(c["mean"])]
    top = y_max if y_max is not None else max(vals + [1e-6]) * 1.1
    for k in range(5):
        y = plot.bottom - plot.h * k / 4
        pygame.draw.line(surf, (36, 40, 50), (plot.x, y), (plot.right, y))
        m.text(surf, f"{top * k / 4:.2g}", (plot.x - 6, y), ui.LABEL, m.f_small, "midright")
    n = len(xs)

    def px(i):
        return plot.x + (plot.w * (i + 0.5) / n)

    def py(v):
        return plot.bottom - plot.h * min(max(v / top, 0), 1)

    for i, x in enumerate(xs):
        m.text(surf, x_fmt.format(x) if isinstance(x, (int, float)) else str(x), (px(i), plot.bottom + 6), ui.LABEL,
               m.f_small, "midtop")
    for li, (label, col, cs) in enumerate(series):
        pts = []
        for i, c in enumerate(cs):
            if math.isnan(c["mean"]):
                continue
            if not math.isnan(c.get("lo", float("nan"))):
                pygame.draw.line(surf, col, (px(i), py(c["lo"])), (px(i), py(c["hi"])), 2)
            pts.append((px(i), py(c["mean"])))
            pygame.draw.circle(surf, col, (int(px(i)), int(py(c["mean"]))), 4)
        if len(pts) > 1 and connect:
            pygame.draw.lines(surf, col, False, pts, 2)
        m.text(surf, label, (rect.right - 12 - li * 190, rect.y + 6), col, m.f_small, "topright")
    m.text(surf, x_label, (plot.centerx, rect.bottom - 16), ui.TEXT, m.f_small, "midtop")
    m.text(surf, y_label, (rect.x + 6, rect.y + 2), ui.TEXT, m.f_small)


def _ci(c: dict, fmt="{:.2f}") -> str:
    import math

    if c["n"] == 0 or math.isnan(c["mean"]):
        return "n/a"
    if math.isnan(c["lo"]):
        return fmt.format(c["mean"])
    return f"{fmt.format(c['mean'])} ± {fmt.format((c['hi'] - c['lo']) / 2)}"


# --- assays and repeated trials ---------------------------------------------------------------------------------------
def surgery_options() -> list[tuple[str, dict | None]]:
    from kickthefly.game import kick_the_fly as k

    out = [("none", None)]
    for label, (kind, names) in k.SURGERY:
        if kind == "type":
            spec = {"type:" + ",".join(names): -1}
        elif kind == "group":
            spec = {n: -1 for n in names}
        elif kind == "prefix":
            spec = {"prefix:" + ",".join(names): -1}
        elif kind == "pop":
            spec = {n: -1 for n in names}
        else:
            continue
        out.append((label, spec))
    return out


class LabState:
    def __init__(self):
        self.kind = "tmaze"
        self.flies = 6
        self.surgery_i = 0
        self.stimulate = False
        self.job = None
        self.result = None
        self.validation_job = None


def _state(m) -> LabState:
    if not hasattr(m, "lab_state"):
        m.lab_state = LabState()
    return m.lab_state


def page_assays(m: ui.Menu, surf, rect, mouse) -> None:
    from kickthefly.lab import labjobs
    from kickthefly.lab import labstats

    st, host = _state(m), m.host
    m.text(surf, "ASSAYS AND REPEATED TRIALS", (rect.x + 24, rect.y + 16), ui.INK, m.f_head)
    m.text(surf, "Each fly is a fresh, untrained brain with its own seed. Your saved training memory isn't used or "
                 "changed.", (rect.x + 24, rect.y + 48), ui.LABEL, m.f_small)
    x0, y = rect.x + 24, rect.y + 78
    busy = st.job is not None and st.job.running
    m.text(surf, "Assay", (x0, y + 15), ui.TEXT, m.f_text, "midleft")
    m.segmented(surf, (x0 + 110, y, 700, 30), [labjobs.ASSAY_LABEL[k] for k in labjobs.ASSAYS],
                labjobs.ASSAYS.index(st.kind), lambda i: setattr(st, "kind", labjobs.ASSAYS[i]), id="assay_kind",
                enabled=not busy)
    y += 40
    m.text(surf, "Flies", (x0, y + 15), ui.TEXT, m.f_text, "midleft")
    m.slider(surf, (x0 + 110, y, 300, 30), st.flies, 2, 30, 1, "{:.0f}", lambda v: setattr(st, "flies", int(v)),
             lambda: None, id="assay_flies", enabled=not busy,
             tip="How many flies (seeds) to run. More gives tighter confidence intervals and takes longer.")
    base = int(host.cfg["brain.seed"])
    m.text(surf, f"seeds {base + 2000}-{base + 2000 + st.flies - 1}", (x0 + 430, y + 15), ui.LABEL, m.f_small, "midleft")
    y += 40
    opts = surgery_options()
    label, spec = opts[st.surgery_i]
    m.text(surf, "Surgery", (x0, y + 15), ui.TEXT, m.f_text, "midleft")
    m.button(surf, (x0 + 110, y, 44, 30), "<", lambda: setattr(st, "surgery_i", (st.surgery_i - 1) % len(opts)),
             id="surg<", enabled=not busy)
    m.text(surf, label, (x0 + 170, y + 15), ui.INK if spec else ui.LABEL, m.f_bold, "midleft")
    m.button(surf, (x0 + 520, y, 44, 30), ">", lambda: setattr(st, "surgery_i", (st.surgery_i + 1) % len(opts)),
             id="surg>", enabled=not busy)
    if spec:
        m.segmented(surf, (x0 + 580, y, 220, 30), ["Silence", "Stimulate"], int(st.stimulate),
                    lambda i: setattr(st, "stimulate", bool(i)), id="surg_mode", enabled=not busy)
    y += 40
    if spec:
        m.text(surf, "Each fly also runs unperturbed with the same seed as its control; the results show both and a "
                     "paired test.", (x0, y), ui.LABEL, m.f_small)
    y += 24

    def start():
        surgery = {k_: (1 if st.stimulate else -1) for k_ in spec} if spec else None
        seeds = [base + 2000 + i for i in range(st.flies)]
        params = modified(host.lab_params) and dict(host.lab_params) or None
        st.result = None
        st.job = labjobs.Job(st.kind, seeds, surgery=surgery, params=params).start()
        st.job.surgery_label = label

    if busy:
        j = st.job
        frac = j.done / max(1, j.total)
        pygame.draw.rect(surf, (30, 36, 48), (x0, y, rect.w - 220, 14), border_radius=7)
        pygame.draw.rect(surf, ui.AMBER, (x0, y, max(8, int((rect.w - 220) * frac)), 14), border_radius=7)
        m.text(surf, f"{j.done}/{j.total} flies  ·  {time.time() - j.started:.0f}s  ·  {j.workers} worker processes",
               (x0, y + 20), ui.TEXT, m.f_small)
        m.button(surf, (rect.right - 170, y - 8, 146, 36), "Cancel", lambda: setattr(j, "cancelled", True),
                 id="assay_cancel", style="danger")
    else:
        if st.job is not None and st.job.result is not None and st.result is not st.job.result:
            st.result = st.job.result
            host.last_lab_result = st.result
        m.button(surf, (x0, y - 6, 180, 40), "Run", start, id="assay_run", style="primary",
                 tip="Runs in background worker processes while the game stays paused.")
        if st.job is not None and st.job.error:
            m.text(surf, st.job.error, (x0 + 200, y + 14), ui.BAD, m.f_small, "midleft")
        if st.result is not None:
            m.button(surf, (x0 + 200, y - 6, 200, 40), "Export results", lambda: host.export_lab_result(st.result),
                     id="assay_export", tip="Save these results as JSON and CSV in the exports folder.")
    y += 40
    if st.result is not None and not busy:
        draw_assay_result(m, surf, pygame.Rect(rect.x + 24, y, rect.w - 48, rect.bottom - 70 - y), st.result)
    m.button(surf, (rect.right - 164, rect.bottom - 58, 140, 42), "Back", m.back, style="primary", id=("assays", "back"))


def draw_assay_result(m: ui.Menu, surf, area: pygame.Rect, res: dict) -> None:
    from kickthefly.lab import labstats

    kind, t, c = res["kind"], res["treated"], res.get("control")
    surg = res.get("surgery")
    tl = "with surgery" if surg else "flies"
    chart = pygame.Rect(area.x, area.y, area.w // 2 - 10, area.h)
    tx = area.x + area.w // 2 + 10
    y = area.y
    m.text(surf, f"{res['label']}  ·  n = {len(res['seeds'])} flies", (tx, y), ui.INK, m.f_bold)
    y += 26
    if kind == "tmaze":
        xs, pts = (["with surgery", "control"], [t["pi"], c["pi"]]) if c else (["all flies"], [t["pi"]])
        draw_chart(m, surf, chart, xs, [("mean and 95% CI", ui.ACCENT, pts)], "", "performance index", y_max=1.0,
                   connect=False)
        rows = [("performance index", _ci(t["pi"]), _ci(c["pi"]) if c else ""),
                ("fear of CS+", _ci(t["fear_cs_plus"]), _ci(c["fear_cs_plus"]) if c else ""),
                ("fear of CS-", _ci(t["fear_cs_minus"]), _ci(c["fear_cs_minus"]) if c else ""),
                ("approach MBONs to CS+ (Hz)", _ci(t["mbon_cs_plus_hz"], "{:.1f}"), _ci(c["mbon_cs_plus_hz"], "{:.1f}") if c else ""),
                ("approach MBONs to CS- (Hz)", _ci(t["mbon_cs_minus_hz"], "{:.1f}"), _ci(c["mbon_cs_minus_hz"], "{:.1f}") if c else "")]
    elif kind == "orchard":
        st_ = t["settings"]
        draw_chart(m, surf, chart, ["MN9", "PAM"], [("feeding / travelling", ui.ACCENT, [t["mn9_ratio"], t["pam_ratio"]])],
                   "", "firing x travelling", connect=False)
        rows = [("MN9 feeding / travelling", f"x{_ci(t['mn9_ratio'])}", f"x{_ci(c['mn9_ratio'])}" if c else ""),
                ("PAM feeding / travelling", f"x{_ci(t['pam_ratio'])}", f"x{_ci(c['pam_ratio'])}" if c else ""),
                ("feeds", _ci(t["feeds"], "{:.1f}"), _ci(c["feeds"], "{:.1f}") if c else ""),
                ("fruit emptied", _ci(t["fruit_emptied"], "{:.1f}"), _ci(c["fruit_emptied"], "{:.1f}") if c else ""),
                ("fermented feeds", _ci(t["fermented_feeds"], "{:.1f}"), ""),
                ("settings (game rules)", f"{st_['feeds_per_fruit']} feeds/fruit, regrow {st_['regrow_s']:.0f} s, "
                                          f"cap {st_['cap']}, {st_['duration_s']:.0f} s", "")]
    elif kind == "looming":
        xs = [r["speed"] for r in t["rows"]]
        series = [(tl, ui.ACCENT, [r["escape_probability"] for r in t["rows"]])]
        if c:
            series.append(("control", (200, 200, 200), [r["escape_probability"] for r in c["rows"]]))
        draw_chart(m, surf, chart, xs, series, "approach speed (m/s)", "escape probability", y_max=1.0)
        rows = [(f"{r['speed']:g} m/s", f"{r['escapes']}/{r['approaches']} escaped, latency {_ci(r['latency_s'])} s",
                 (f"{cr['escapes']}/{cr['approaches']}" if c else "")) for r, cr in zip(t["rows"], c["rows"] if c else t["rows"])]
    elif kind == "predator_escape":
        xs = [r["predator"] for r in t["rows"]]
        ci = [dict(mean=r["escape_probability"], lo=r["ci95"][0], hi=r["ci95"][1]) for r in t["rows"]]
        draw_chart(m, surf, chart, xs, [("escape probability (Wilson 95% CI)", ui.ACCENT, ci)], "predator",
                   "escape probability", y_max=1.0, connect=False)
        rows = [(r["predator"], f"{r['escapes']}/{r['trials']} escaped, 95% CI {r['ci95'][0]:.2f}-{r['ci95'][1]:.2f}",
                 f"{r['escapes']}/{r['trials']}") for r in t["rows"]]
        rows.append(("MODEL PREDICTION", "the attack is a game rule; an escape = DNp01 above the game's escape rule before the capture", ""))
    elif kind == "thermo_escape":
        xs = [r["temperature_c"] for r in t["rows"]]
        mk = lambda key: [dict(mean=r[key], lo=float("nan"), hi=float("nan")) for r in t["rows"]]   # noqa: E731
        draw_chart(m, surf, chart, xs, [("TrpA1 in DNp01", ui.ACCENT, mk("escape_rate")),
                                        ("control (no expression)", (200, 200, 200), mk("control_rate"))],
                   "temperature (C)", "escape rate", y_max=1.0, x_fmt="{:g}")
        rows = [(f"{r['temperature_c']:g} C", f"{r['escapes']}/{r['flies']} escaped (control {r['control_escapes']}), "
                 f"DNp01 {_ci(r['dnp01_hz'], '{:.0f}')} Hz", f"{r['escapes']}/{r['flies']}") for r in t["rows"]]
        rows.append(("MODEL", "the temperature curve is a game rule; escape = the game's DNp01 rule", ""))
    else:
        xs = [r["dose"] for r in t["rows"]]
        series = [(tl, ui.ACCENT, [r["mn9_ratio"] for r in t["rows"]])]
        if c:
            series.append(("control", (200, 200, 200), [r["mn9_ratio"] for r in c["rows"]]))
        draw_chart(m, surf, chart, xs, series, "sugar dose (share of sugar-pathway GRNs)", "MN9 firing x before",
                   x_fmt="{:.0%}")
        rows = [(f"{r['dose']:.0%}", f"MN9 x{_ci(r['mn9_ratio'])}, extended {r['extensions']}/{r['offers']}",
                 (f"x{cr['mn9_ratio']['mean']:.2f}" if c else "")) for r, cr in zip(t["rows"], c["rows"] if c else t["rows"])]
    col_a = tx + 190
    col_b = tx + (area.w // 2 - 10) - 120
    if c:
        m.text(surf, "surgery", (col_a, y), ui.LABEL, m.f_small)
        m.text(surf, "control", (col_b, y), ui.LABEL, m.f_small)
        y += 18
    else:
        m.text(surf, "mean ± 95% CI half-width", (col_a, y), ui.LABEL, m.f_small)
        y += 18
    for label, a, b in rows:
        m.text(surf, label, (tx, y), ui.TEXT, m.f_small)
        if c:
            m.wrapped(surf, a, (col_a, y), col_b - col_a - 10, ui.INK, m.f_small, 1)
            m.text(surf, b, (col_b, y), ui.INK, m.f_small)
        else:
            m.wrapped(surf, a, (col_a, y), area.right - col_a, ui.INK, m.f_small, 1)
        y += 20
    if c:
        cmp_ = res["comparison"]["overall"]
        y += 8
        m.text(surf, f"Surgery vs same-seed control: mean difference {cmp_['mean_difference']:+.3f} (n={cmp_['n']} pairs)",
               (tx, y), ui.INK, m.f_small)
        m.text(surf, f"{cmp_['test']}: {labstats.fmt_p(cmp_['p_value'])}   (paired t: {labstats.fmt_p(cmp_.get('t_p_value'))})",
               (tx, y + 18), ui.AMBER if cmp_["p_value"] < 0.05 else ui.TEXT, m.f_small)
        m.text(surf, f"surgery: {', '.join(f'{k} {v:+d}' for k, v in res['surgery'].items())}", (tx, y + 36), ui.LABEL, m.f_small)


# --- validation dashboard ---------------------------------------------------------------------------------------------
def page_validation(m: ui.Menu, surf, rect, mouse) -> None:
    from kickthefly.lab import labstats
    from kickthefly.lab import validation

    st, host = _state(m), m.host
    m.text(surf, "VALIDATION", (rect.x + 24, rect.y + 16), ui.INK, m.f_head)
    job = st.validation_job
    if job is not None and not job["thread"].is_alive() and job.get("result"):
        host.refresh_validation()
        st.validation_job = job = None
    res, source = validation.load_results()
    if res is None:
        m.text(surf, "No results yet. Run the suite (a few minutes).", (rect.x + 24, rect.y + 50), ui.LABEL, m.f_small)
    else:
        m.text(surf, f"Kick the Fly {res['app_version']}, {res['created']}, {source}  ·  seeds {res['seeds'][0]}-"
                     f"{res['seeds'][-1]} (n={len(res['seeds'])})  ·  thresholds chosen for this release, not from the papers",
               (rect.x + 24, rect.y + 50), ui.LABEL, m.f_small)
    if modified(host.lab_params):
        m.text(surf, "Parameters are modified: these results were measured at the defaults.", (rect.right - 24, rect.y + 20),
               ui.AMBER, m.f_small, "topright")
    body = pygame.Rect(rect.x + 16, rect.y + 76, rect.w - 32, rect.h - 76 - 70)
    off = int(m.scroll.get("lab_validation", 0))
    m.clip = body
    prev = surf.get_clip()
    surf.set_clip(body)
    y = body.y - off
    for t in (res or {}).get("tests", []):
        card = pygame.Rect(body.x, y, body.w - 12, 156)
        pygame.draw.rect(surf, (26, 30, 40), card, border_radius=10)
        ok = t["passed"]
        chip = pygame.Rect(card.x + 12, card.y + 12, 64, 24)
        pygame.draw.rect(surf, ui.GOOD if ok else ui.BAD, chip, border_radius=6)
        m.text(surf, "PASS" if ok else "FAIL", chip.center, (10, 12, 16), m.f_bold, "center")
        m.text(surf, t["name"], (card.x + 90, card.y + 12), ui.INK, m.f_bold)
        m.text(surf, t["claim"][:140], (card.x + 90, card.y + 36), ui.TEXT, m.f_small)
        mm = t["measured"]
        if "drive_ratio_mean" in mm:
            meas = (f"{t['readout_label']}: x{mm['drive_ratio_mean']:.2f} ± {mm['drive_ratio_sd']:.2f} driving "
                    f"{t['drive_label']}  vs  x{mm['control_ratio_mean']:.2f} ± {mm['control_ratio_sd']:.2f} for "
                    f"{t['control_label']}  ·  {labstats.fmt_p(mm['p_value'])}")
        elif "extinguished_pi_mean" in mm:
            meas = (f"PI after extinction {mm['extinguished_pi_mean']:.2f} ± {mm['extinguished_pi_sd']:.2f} vs "
                    f"{mm['unextinguished_pi_mean']:.2f} ± {mm['unextinguished_pi_sd']:.2f} without  ·  fear of CS+ "
                    f"{mm['fear_extinguished']:.2f} vs {mm['fear_unextinguished']:.2f}  ·  {labstats.fmt_p(mm['p_value'])}")
        elif "fear_b_paired" in mm:
            meas = (f"PI for odor B {mm['pi_mean']:.2f} ± {mm['pi_sd']:.2f} paired vs {mm['control_pi_mean']:.2f} ± "
                    f"{mm['control_pi_sd']:.2f} unpaired  ·  fear of odor B {mm['fear_b_paired']:.2f} vs "
                    f"{mm['fear_b_unpaired']:.2f}  ·  {labstats.fmt_p(mm['p_value'])}")
        elif "pi_mean" in mm:
            meas = (f"PI {mm['pi_mean']:.2f} ± {mm['pi_sd']:.2f} vs unpaired {mm['control_pi_mean']:.2f} ± "
                    f"{mm['control_pi_sd']:.2f}  ·  fear CS+ {mm['fear_cs_plus']:.2f} vs CS- {mm['fear_cs_minus']:.2f}  ·  "
                    f"approach MBONs {mm['approach_mbon_cs_plus_hz']:.1f} vs {mm['approach_mbon_cs_minus_hz']:.1f} Hz  ·  "
                    f"{labstats.fmt_p(mm['p_value'])}")
        elif "direction_tracking" in mm:
            meas = (f"EPG contrast in wind {mm['mean_contrast']:.2f}x (no wind {mm['baseline_contrast']:.2f}x, need "
                    f">= 3.0x)  ·  persistence {mm['mean_persistence_ms']:.0f} ms (need >= 500)  ·  tracking |r| "
                    f"{mm['direction_tracking']:.2f} vs {mm['direction_tracking_null']:.2f} shuffled, "
                    f"p = {mm['direction_tracking_p']:.2f}  ·  EPG {mm['epg_rate_baseline_hz']:.1f} -> "
                    f"{mm['epg_rate_wind_hz']:.1f} Hz")
        elif "mean_contrast" in mm:
            meas = (f"EPG peak/trough contrast: {mm['mean_contrast']:.2f} ± {mm.get('sd_contrast', 0):.2f}x "
                    f"(required >= 3.0x)  ·  persistence: {mm.get('mean_persistence_ms', 0):.0f} ms (< 500 ms)  ·  "
                    f"baseline: {mm.get('mean_baseline_hz', 0):.1f} ± {mm.get('mean_baseline_sd_hz', 0):.1f} Hz")
        else:
            meas = t.get("finding", str(mm))
        yy = m.wrapped(surf, meas, (card.x + 90, card.y + 58), card.w - 110, ui.INK, m.f_small, 2)
        m.text(surf, f"Pass if: {t['criteria']}", (card.x + 90, yy + 4), ui.LABEL, m.f_small)
        m.text(surf, t["citation"], (card.x + 90, yy + 24), (150, 180, 220), m.f_small)
        if t.get("note"):
            m._register(pygame.Rect(card.x, card.y, card.w, card.h), "label", id=("vnote", t["id"]), tip=t["note"])
            m.text(surf, "hover for notes", (card.right - 12, card.y + 12), ui.DIM, m.f_small, "topright")
        y += 164
    m.content_h["lab_validation"] = max(0, y + off - body.bottom)
    surf.set_clip(prev)
    m.clip = None
    if job is not None:
        frac = job["done"] / max(1, job["total"])
        pygame.draw.rect(surf, (30, 36, 48), (rect.x + 24, rect.bottom - 44, rect.w - 420, 12), border_radius=6)
        pygame.draw.rect(surf, ui.AMBER, (rect.x + 24, rect.bottom - 44, max(8, int((rect.w - 420) * frac)), 12), border_radius=6)
        m.text(surf, f"running: {job['done']}/{job['total']}" + (f"  error: {job['error']}" if job.get("error") else ""),
               (rect.x + 24, rect.bottom - 28), ui.TEXT, m.f_small)
    else:
        m.button(surf, (rect.x + 24, rect.bottom - 58, 240, 42), "Run validation now", lambda: start_validation(m),
                 id="val_run", tip="Runs every test on this PC with the default parameters (a few minutes). The result "
                                   "replaces the build's bundled one on this PC.")
    m.button(surf, (rect.right - 164, rect.bottom - 58, 140, 42), "Back", m.back, style="primary", id=("val", "back"))


def start_validation(m: ui.Menu) -> None:
    import threading

    from kickthefly.lab import labjobs
    from kickthefly.lab import validation

    st = _state(m)
    job = dict(done=0, total=1, result=None, error=None)

    def work():
        try:
            res = validation.run(workers=labjobs.default_workers(),
                                 progress=lambda d, n, label: job.update(done=d, total=n))
            validation.save_results(res)
            job["result"] = res
        except Exception as e:
            job["error"] = f"{type(e).__name__}: {e}"

    job["thread"] = threading.Thread(target=work, name="validation", daemon=True)
    job["thread"].start()
    st.validation_job = job




# --- record and export ------------------------------------------------------------------------------------------------
RECORD_GROUPS = (
    ("Giant fiber DNp01", "dnp01"), ("Looming detectors LPLC2, LC4", "loom"), ("Touch neurons", "touch"),
    ("Reaction descending neurons", "reaction_dns"), ("All descending neurons", "superclass:descending_neuron"),
    ("Proboscis motor neuron MN9", "mn9"), ("Grooming command aDN1/aDN2", "adn"), ("Antennal JO-C/E", "jo_ce"),
    ("Sugar-pathway taste neurons", "sweet"), ("Kenyon cells", "prefix:KC"), ("Mushroom body output neurons", "prefix:MBON"),
    ("Reward dopamine PAM", "reward"), ("Punishment dopamine PPL1", "punish"), ("Leg motor neurons", "mn_legs"),
    ("Whole brain (large files)", "whole brain"),
)


def resolve_group(br, spec: str):
    import numpy as np

    from kickthefly.lab import assays
    from kickthefly.game import kick_the_fly as k
    from kickthefly.core import simcore

    g = assays.groups(br)
    if spec in g:
        return g[spec]
    if spec == "touch":
        return np.concatenate([simcore.rows_of(br, n) for n in k.TOUCH])
    if spec == "reaction_dns":
        return np.concatenate([simcore.rows_of(br, n) for n, *_ in k.MOTOR])
    return simcore.rows_of(br, spec)


def make_bundle(host, st) -> tuple[str | None, str]:
    """Lab > Record and export > Bundle (also on the Protocols page): zip the last protocol run (rerunnable with
    --rerun-bundle), or else the last live recording (a record, not rerunnable: the stimuli were delivered by hand).
    Returns (path, message). See kickthefly/lab/bundle.py for what is in it."""
    import json
    import time
    from pathlib import Path

    from kickthefly.lab import bundle, recorder

    out_dir = recorder.exports_dir() / "bundles"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    try:
        job = getattr(st, "proto_job", None)
        folder = job.get("folder") if job else None
        if folder and (Path(folder) / "protocol.json").exists():
            z = bundle.create(Path(folder), out_dir / f"{stamp}-{Path(folder).name}.zip")
            return str(z), "Bundled the last protocol run: it can be rerun with --headless --rerun-bundle."
        last = getattr(host, "last_export", None)
        if last and Path(last).is_dir():
            files = [f for f in sorted(Path(last).iterdir()) if f.is_file()]
            meta_file = next((f for f in files if f.name.endswith("-metadata.json")), None)
            meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file else {}
            sim = host.brain.sim
            extra = dict(seeds=[meta.get("seed", getattr(host.brain, "seed", 0))], arena=meta.get("arena"),
                         parameters=meta.get("lab_params_modified", {}), surgery=meta.get("surgery", {}),
                         surgery_by_type=meta.get("surgery_by_type", {}), backend=sim.backend.name,
                         dtype=str(sim.p.dtype), individuality=str(host.cfg["brain.individuality"]))
            z = bundle.create_live(files, extra, out_dir / f"{stamp}-{Path(last).name}.zip",
                                   {"name": "live-recording", "note": "recorded live in the Lab; stimuli by hand"})
            return str(z), "Bundled the last live recording (a record: it can't be rerun, and says why)."
        return None, "Nothing to bundle yet: run a protocol or make a recording first."
    except Exception as e:                        # a bundle is never worth a crash
        from kickthefly.core.crash import log

        log.exception("bundling failed")
        return None, f"Couldn't make the bundle: {e}"


def page_export(m: ui.Menu, surf, rect, mouse) -> None:
    from kickthefly.lab import recorder

    st, host = _state(m), m.host
    if not hasattr(st, "rec_pick"):
        st.rec_pick, st.rec_seconds = {"dnp01", "loom", "reaction_dns"}, 10
    m.text(surf, "RECORD AND EXPORT", (rect.x + 24, rect.y + 16), ui.INK, m.f_head)
    m.text(surf, "Records spike times from the fly your brain panel shows, live, while you play. Files: spikes and rates "
                 "as CSV and npz, plus a metadata JSON.", (rect.x + 24, rect.y + 48), ui.LABEL, m.f_small)
    if not hasattr(st, "rec_nwb"):
        st.rec_nwb = True
    y = rect.y + 84
    options = list(RECORD_GROUPS)
    insp = getattr(host, "inspect", None)
    if insp:
        options.append((f"Inspected type {insp['type']}", "type:" + insp["type"]))
    col_w = (rect.w - 48) // 2
    for i, (label, spec) in enumerate(options):
        x = rect.x + 24 + (i % 2) * col_w
        yy = y + (i // 2) * 38
        on = spec in st.rec_pick
        m.toggle(surf, (x, yy, 90, 30), on, lambda v, s=spec: (st.rec_pick.add(s) if v else st.rec_pick.discard(s)),
                 id=("rec", spec))
        m.text(surf, label, (x + 100, yy + 15), ui.TEXT, m.f_text, "midleft")
    y += ((len(options) + 1) // 2) * 38 + 12
    m.text(surf, "Duration", (rect.x + 24, y + 15), ui.TEXT, m.f_text, "midleft")
    m.slider(surf, (rect.x + 140, y, 360, 30), st.rec_seconds, 1, 60, 1, "{:.0f} s",
             lambda v: setattr(st, "rec_seconds", int(v)), lambda: None, id="rec_seconds")
    y += 46
    from kickthefly.lab import nwbexport

    why = nwbexport.available()
    m.toggle(surf, (rect.x + 24, y, 90, 30), st.rec_nwb and not why,
             lambda v: setattr(st, "rec_nwb", bool(v)), id="rec_nwb", enabled=not why)
    m.text(surf, "Also write NWB (Neurodata Without Borders)", (rect.x + 130, y + 15),
           ui.LABEL if why else ui.TEXT, m.f_text, "midleft")
    m._register(pygame.Rect(rect.x + 24, y, rect.w - 48, 30), "label", id=("nwbtip",),
                tip=why or "One .nwb file with spike times, per-neuron and per-region rates, stimuli, tool events, "
                           "the fly's movement, surgery, arena and the learned KC->MBON weights before and after, "
                           "plus the full metadata and the MaleCNS v1.0 citation. Opens in pynwb, the NWB inspector "
                           "and NWB Explorer. CSV and npz are always written as the quick option.")
    if why:
        m.text(surf, why, (rect.x + 24, y + 34), ui.AMBER, m.f_small)
        y += 20
    y += 44
    rec = getattr(host, "recording", None)
    if rec is None:
        m.button(surf, (rect.x + 24, y, 260, 44), "Start recording and resume", lambda: host.start_recording(
            [(lbl, s) for lbl, s in options if s in st.rec_pick], st.rec_seconds,
            nwb=bool(st.rec_nwb) and not why), style="primary", id="rec_start",
            enabled=bool(st.rec_pick), tip="Closes the menu; the recording stops by itself after the duration.")
    else:
        m.button(surf, (rect.x + 24, y, 200, 44), "Stop and save", host.stop_recording, style="danger", id="rec_stop")
    m.text(surf, f"Saved to {short(recorder.exports_dir(), 110)}", (rect.x + 24, y + 56), ui.LABEL, m.f_small)
    last = getattr(host, "last_export", None)
    if last:
        m.text(surf, f"Last: {short(last, 110)}", (rect.x + 24, y + 76), ui.GOOD, m.f_small)

    def bundle_now():
        path, msg = make_bundle(host, st)
        st.bundle_msg = (msg, path)

    m.button(surf, (rect.x + 24, y + 98, 200, 40), "Bundle", bundle_now, id="export_bundle",
             tip="One zip with the protocol YAML, results, raw exports, metadata (version, backend, precision, seeds, brain "
                 "pack checksum, parameters, surgery, individuality, arena) and an RO-Crate description. Bundles the "
                 "last protocol run, or else the last recording. A protocol run's bundle can be rerun with "
                 "--headless --rerun-bundle ZIP --out DIR.")
    bm = getattr(st, "bundle_msg", None)
    if bm:
        m.text(surf, bm[0], (rect.x + 236, y + 103), ui.GOOD if bm[1] else ui.AMBER, m.f_small)
        if bm[1]:
            m.text(surf, short(bm[1], 120), (rect.x + 236, y + 121), ui.LABEL, m.f_small)
    m.button(surf, (rect.right - 164, rect.bottom - 58, 140, 42), "Back", m.back, style="primary", id=("export", "back"))


# --- protocols --------------------------------------------------------------------------------------------------------
def protocol_files() -> list:
    import sys
    from pathlib import Path

    import kickthefly
    from kickthefly.core import paths

    user = paths.get().data_dir / "protocols"
    roots = [user, Path(getattr(sys, "_MEIPASS", "")) / "protocols", kickthefly.PROTOCOLS_DIR]
    seen, out = set(), []
    for root in roots:
        if str(root) and root.is_dir():
            for f in sorted(root.glob("*.y*ml")):
                if f.name not in seen:
                    seen.add(f.name)
                    out.append(f)
    return out


def page_protocols(m: ui.Menu, surf, rect, mouse) -> None:
    import threading

    from kickthefly.lab import labjobs
    from kickthefly.core import paths
    from kickthefly.lab import protocol

    st = _state(m)
    m.text(surf, "PROTOCOLS", (rect.x + 24, rect.y + 16), ui.INK, m.f_head)
    m.text(surf, f"YAML experiment files. Put your own in {short(paths.get().data_dir / 'protocols', 60)}. Headless: "
                 "KickTheFly --headless --protocol FILE", (rect.x + 24, rect.y + 48), ui.LABEL, m.f_small)
    job = getattr(st, "proto_job", None)
    busy = job is not None and job["thread"].is_alive()
    y = rect.y + 84
    for f in protocol_files()[:12]:
        try:
            p = protocol.load(f)
            desc = (f"assay {p['assay']}, " if "assay" in p else f"{len(p['stimuli'])} stimuli, {len(p['recordings'])} "
                    f"recordings, ") + f"{len(p['seeds'])} fly(s)" + (", with surgery + control" if p.get("surgery") else "")
            err = None
        except Exception as e:
            desc, err, p = str(e), True, None
        row = pygame.Rect(rect.x + 24, y, rect.w - 48, 42)
        pygame.draw.rect(surf, (26, 30, 40), row, border_radius=8)
        m.text(surf, f.name, (row.x + 12, row.y + 4), ui.INK, m.f_bold)
        m.text(surf, desc[:120], (row.x + 12, row.y + 23), ui.BAD if err else ui.LABEL, m.f_small)

        def start(p=p):
            j = dict(done=0, total=1, folder=None, error=None, name=p["name"])

            def work():
                try:
                    j["folder"] = protocol.run(p, workers=labjobs.default_workers(),
                                               progress=lambda d, n: j.update(done=d, total=n))
                except Exception as e:
                    j["error"] = f"{type(e).__name__}: {e}"

            j["thread"] = threading.Thread(target=work, name="protocol", daemon=True)
            j["thread"].start()
            st.proto_job = j

        m.button(surf, (row.right - 110, row.y + 6, 96, 30), "Run", start, id=("proto", f.name),
                 enabled=not err and not busy, font=m.f_small)
        y += 50
    if job is not None:
        if busy:
            frac = job["done"] / max(1, job["total"])
            pygame.draw.rect(surf, (30, 36, 48), (rect.x + 24, rect.bottom - 100, rect.w - 220, 12), border_radius=6)
            pygame.draw.rect(surf, ui.AMBER, (rect.x + 24, rect.bottom - 100, max(8, int((rect.w - 220) * frac)), 12),
                             border_radius=6)
            m.text(surf, f"running {job['name']}: {job['done']}/{job['total']}", (rect.x + 24, rect.bottom - 84), ui.TEXT, m.f_small)
        elif job["error"]:
            m.text(surf, job["error"], (rect.x + 24, rect.bottom - 90), ui.BAD, m.f_small)
        elif job["folder"]:
            m.text(surf, f"Done: {short(job['folder'], 110)}", (rect.x + 24, rect.bottom - 90), ui.GOOD, m.f_small)

            def bundle_run():
                path, msg = make_bundle(m.host, st)
                st.bundle_msg = (msg, path)

            m.button(surf, (rect.right - 320, rect.bottom - 58, 140, 42), "Bundle", bundle_run, id=("proto", "bundle"),
                     tip="Zip this run with its protocol, metadata and an RO-Crate description, so it can be rerun and checked.")
            bm = getattr(st, "bundle_msg", None)
            if bm:
                m.text(surf, bm[0], (rect.x + 24, rect.bottom - 70), ui.GOOD if bm[1] else ui.AMBER, m.f_small)
    m.button(surf, (rect.right - 164, rect.bottom - 58, 140, 42), "Back", m.back, style="primary", id=("proto", "back"))
