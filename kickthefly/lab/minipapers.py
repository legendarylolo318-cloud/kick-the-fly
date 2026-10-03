"""Guided mini-papers (3.0 day 5): short experiments that reproduce a classic paper step by step.

The flow is the lecture protocol system's (lab/classroom.py: LectureProtocol, LectureStep, ClassroomSession), with four kinds of step the page
(ui/minipaper_ui.py) draws itself: the paper and its question, YOUR HYPOTHESIS (you commit before anything runs), RUN (on the model, in a
background job), PLOT (your flies, each drive next to its control) and COMPARE (your result next to what the paper found).

Nothing is re-implemented. Every adult experiment is a validation test run through `validation.run(seeds, include=[id])`, so its numbers are
the validation suite's own; the T-maze is `mb_conditioning`, the larva pair is the larva tests, and the rig paper is lab/rigassay.py.

What the paper found is written from what the paper itself states, and only that: the direction and an approximate effect where the paper
gives one, with the full citation and which text was read. Where a paper states no number, none is shown. `read` says how much of each
paper was read: "abstract" (Europe PMC / PubMed record) or "full text" (an open-access full text, read through a page summary), and any
claim about a paper that rests on a different paper is marked as not read.

Where the model fails a paper's result, the mini-paper says so, and says why, from docs/validation.md.

Tags: the experiment is CONNECTOME (the wiring driven and read) plus the validation test's GAME RULE inputs; "your result" is a MODEL
PREDICTION; "what the paper found" is LITERATURE.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

QUICK_SEEDS = (0, 1, 2, 3)                       # exploration seeds: a first look, never a validation result
FULL_SEEDS = tuple(range(1000, 1010))            # the validation seeds
RATIO_MIN, P_MAX = 1.5, 0.01                     # validation.py's own criteria


@dataclass(frozen=True)
class Question:
    test: str                                    # a validation test id ("rig:buridan" for the rig)
    prompt: str
    options: tuple                               # ((key, label), ...); the first is the direction the paper reports
    expected: str                                # the key of what the paper reports
    pass_note: str                               # what it means when the model reproduces it
    fail_why: str                                # why it does not, from docs/validation.md


@dataclass(frozen=True)
class Paper:
    id: str
    title: str
    short: str
    citation: str
    doi: str
    read: str                                    # what was read
    summary: str                                 # what the experiment is, in a sentence
    found: str                                   # what the paper itself states (direction and approximate effect)
    found_numbers: str                           # the numbers the paper states, or "none stated in the part read"
    cannot_check: str                            # what this model cannot test of the paper
    questions: tuple
    brain: str = "adult"
    recorded: dict = field(default_factory=dict)  # recorded validation numbers, used when the brain pack is missing (larva)
    steps: tuple = ()                            # (title, text, kind) for the guided steps; kinds: read, hypothesis, run, plot, compare


UNEXPECTED_MISS = ("Unexpected: this test passes in the validation (n = 10, held-out seeds). A miss here most likely means too few flies or a different seed set; "
                   "run the Full set before reading anything into it.")
UNEXPECTED_HIT = ("Unexpected: this test does not pass in the validation (docs/validation.md). A pass on a small run is a first look at most, not a validation "
                  "result; run the Full set.")


def _q(test, prompt, options, expected, ok, why):
    return Question(test, prompt, tuple(options), expected, ok, why)


PAPERS: dict[str, Paper] = {}


def _add(p: Paper) -> None:
    PAPERS[p.id] = p


_add(Paper(
    id="von_reyn_2014", title="Looming drives the giant fiber (von Reyn et al. 2014)", short="Looming and the giant fiber",
    citation="von Reyn CR, Breads P, Peek MY, Zheng GZ, Williamson WR, Yee AL, Leonardo A, Card GM (2014) A spike-timing mechanism for action selection. "
             "Nature Neuroscience 17:962-970",
    doi="10.1038/nn.3741", read="abstract only (Europe PMC); the full text is behind a paywall and was not read",
    summary="Drive the looming detectors LPLC2 and LC4, and see whether the giant fiber DNp01, the neuron that starts the fast escape takeoff, fires.",
    found="A looming stimulus makes a fly escape in one of two ways: a long escape sequence that starts stable flight, or a short one that gives up "
          "flight stability for speed. Which one depends on when the giant fiber spikes relative to parallel escape circuits. A simple model in which the "
          "giant fiber has a higher activation threshold than the parallel circuits describes it.",
    found_numbers="none stated in the abstract",
    cannot_check="Spike timing, the two takeoff modes and the parallel circuits: the model has no legs or wings. It checks only the premise, that the looming "
                 "detectors excite the giant fiber. The abstract does not name the detectors; LPLC2 and LC4 driving the giant fiber comes from later work "
                 "(Ache et al. 2019; von Reyn et al. 2017) that was not read here.",
    questions=(_q("looming_escape", "You drive all 311 looming detectors (LPLC2 and LC4) for 2 s. What does the giant fiber DNp01 do, compared with driving 311 "
                  "random other visual projection neurons?",
                  (("up", "fires much more (at least 1.5x its calm rate)"), ("same", "about the same"), ("down", "fires less")), "up",
                  "The wiring from the looming detectors to the giant fiber is strong enough to drive it: the premise of the paper's escape model holds in this connectome.",
                  UNEXPECTED_MISS),),
    steps=(("The paper", "A fly sees a dark object grow on its retina and, in a few milliseconds, one of two escape sequences starts. The giant fiber is a pair of "
            "descending neurons whose spike timing the authors recorded from.", "read"),
           ("Your hypothesis", "Before anything runs, say what the connectome will do when the looming detectors are driven.", "hypothesis"),
           ("Run it", "The model drives the detectors for 2 s after 2 s of calm and counts the giant fiber's spikes, against a matched control set of neurons.", "run"),
           ("Your result", "Each fly (a seed) gives one dot for the drive and one for its control.", "plot"),
           ("Next to the paper", "What the paper reports, what the model gave, and what the model cannot say.", "compare")),
))

_add(Paper(
    id="tully_quinn_1985", title="Odor + shock gives a T-maze memory (Tully & Quinn 1985)", short="T-maze conditioning",
    citation="Tully T, Quinn WG (1985) Classical conditioning and retention in normal and mutant Drosophila melanogaster. "
             "Journal of Comparative Physiology A 157:263-277",
    doi="10.1007/BF01350033", read="abstract only (Europe PMC record of PubMed 3939242); the full text was not read",
    summary="Pair an odor with shock six times and let the fly choose between that odor and another at the T-maze's choice point; compare with unpaired training.",
    found="Twelve shock pulses in the presence of the first odor (and none with the second) were given and flies were tested in a T-maze. 95% of trained flies "
          "avoided the shock-associated odor. Learning reached a plateau within one training cycle and was resistant to extinction. Backward conditioning "
          "produced no learning, and nonassociative controls slightly reduced avoidance equally for both odors without affecting the associative learning index. "
          "Memory in wild-type flies decayed gradually over the first seven hours and was still present 24 h later.",
    found_numbers="95% of trained flies avoided the shock-associated odor (a share of about 150 flies per test, not a per-fly index)",
    cannot_check="Retention over hours, shock intensity, odor concentration, delay versus backward conditioning and the amnesiac, rutabaga and dunce "
                 "mutants: none of them is in the model. Only learning (paired versus unpaired) is tested. The sim's per-fly performance index PI = "
                 "(choices of the other odor - choices of the shocked odor) / choices; (1 + PI) / 2 is the share of choices that avoid the shocked odor, "
                 "which is the closest thing to the paper's 95% (derived here, not stated by the paper).",
    questions=(_q("mb_conditioning", "You train a fly with odor A + shock (six cycles) and odor B without. At the choice point, what is its performance index "
                  "(1 = always picks the odor that was not shocked)? And an unpaired control, with the same odors and shocks never together?",
                  (("learns", "paired flies avoid the shocked odor (PI of 0.5 or more); unpaired flies don't"), ("none", "no difference between paired and unpaired"),
                   ("prefers", "paired flies prefer the shocked odor")), "learns",
                  "The game's learning rule on the connectome's own Kenyon-cell to output-neuron synapses gives odor-specific memory. The sim's PI is higher than real flies' "
                  "(the paper's 95% avoiding is a PI of about 0.9 if every fly did it): it was not tuned to match.",
                  UNEXPECTED_MISS),),
    steps=(("The paper", "Flies learn that an odor predicts shock; at the T-maze's choice point they walk away from it. The plasticity rule, the shock's link to "
            "dopamine neurons and the choice are GAME RULES on the real wiring.", "read"),
           ("Your hypothesis", "Commit to what training does to the fly's choice.", "hypothesis"),
           ("Run it", "Each fly is trained six cycles (about a minute of simulated time) and then makes 20 choices; reciprocal odors, paired and unpaired.", "run"),
           ("Your result", "Performance index per fly, paired next to unpaired.", "plot"),
           ("Next to the paper", "The paper's 95% avoiding, your PI, and what is not tested.", "compare")),
))

_add(Paper(
    id="shiu_2024", title="Sugar taste reaches the feeding motor neuron (Shiu et al. 2024)", short="Sugar to MN9",
    citation="Shiu PK, Sterne GR, Spiller N, Franconville R, Sandoval A, Zhou J, Simha N, Kang CH, Yu S, Kim JS, Dorkenwald S, Matsliah A, Schlegel P, Yu SC, "
             "McKellar CE, Sterling A, Costa M, Eichler K, Bates AS, Eckstein N, Funke J, Jefferis GSXE, Murthy M, Bidaye SS, Hampel S, Seeds AM, Scott K (2024) "
             "A Drosophila computational brain model reveals sensorimotor processing. Nature 634:210-219",
    doi="10.1038/s41586-024-07763-9", read="abstract only (Europe PMC); the full text is behind a paywall and was not read",
    summary="Drive 30 sugar-pathway taste neurons and see whether the proboscis motor neuron MN9 fires more than for 30 bitter-pathway taste neurons.",
    found="A leaky integrate-and-fire model of a whole adult fly central brain, built from connectivity and predicted neurotransmitter identity, predicts "
          "which neurons respond to sugar or water taste and are required for feeding initiation; activating neurons in the feeding region predicts which of "
          "them make motor neurons fire, which the authors confirmed with optogenetic activation and behavior; activating different classes of taste neurons "
          "predicts how several taste modalities interact.",
    found_numbers="none stated in the abstract (it names no motor neuron and no effect size)",
    cannot_check="The paper's model is built on a different connectome (a central brain of more than 125,000 neurons and 50 million synapses, per the abstract); this game's "
                 "is the MaleCNS v1.0 with 166,700 neurons and its own LIF parameters, so this is the same kind of model on other wiring, not the paper's. "
                 "The abstract does not name MN9, so MN9 as the readout is this game's choice (the proboscis motor neuron), and the sugar and bitter sets are "
                 "chosen from their wiring to the sugar and bitter SEL neurons, not by taste labels (the dataset has none).",
    questions=(_q("sugar_feeding", "You drive 30 sugar-pathway taste neurons for 2 s. What happens to MN9 (the proboscis motor neuron), compared with driving 30 "
                  "bitter-pathway taste neurons?",
                  (("up", "MN9 fires clearly more for sugar (at least 1.5x calm, and more than for bitter)"), ("same", "no difference between sugar and bitter"),
                   ("down", "MN9 fires less for sugar")), "up",
                  "The sugar-pathway neurons reach the proboscis motor neuron and the bitter-pathway neurons mostly don't, as the paper's model leads you to expect.",
                  UNEXPECTED_MISS),),
    steps=(("The paper", "The authors built a brain-wide LIF model from connectivity and tested its predictions about feeding with optogenetics.", "read"),
           ("Your hypothesis", "Say what driving sugar-sensing taste neurons will do to the proboscis motor neuron.", "hypothesis"),
           ("Run it", "The model drives the two sets of 30 neurons from the same brain state and counts MN9's spikes.", "run"),
           ("Your result", "MN9's firing ratio per fly, sugar next to bitter.", "plot"),
           ("Next to the paper", "What the abstract says, and what differs between that model and this one.", "compare")),
))

_add(Paper(
    id="hampel_2015", title="From antennal touch to grooming (Hampel et al. 2015)", short="Antennal grooming circuit",
    citation="Hampel S, Franconville R, Simpson JH, Seeds AM (2015) A neural command circuit for grooming movement control. eLife 4:e08758",
    doi="10.7554/eLife.08758", read="full text (open access, PMC4599031), read through a page summary, and the abstract",
    summary="Drive the antennal mechanosensory neurons JO-C/E and see whether the antennal-grooming command neurons aDN1/aDN2 fire; then drive aDN1/aDN2 and see whether "
            "front-leg motor neurons follow.",
    found="Mechanosensory chordotonal neurons detect antenna displacements and activate three functionally connected classes of interneurons, including brain "
          "interneurons and descending neurons, and each level can trigger antennal grooming. In the Results, activating the antennal Johnston's-organ neurons gave "
          "calcium responses in aDN1 and only a weak response in aDN2 even at high light intensity, so aDN1 is likely downstream and aDN2 weakly or indirectly; "
          "activating aDN1 or aDN2 each elicited antennal grooming.",
    found_numbers="none stated in the parts read (no fraction of flies or grooming duration is given for single-aDN activation)",
    cannot_check="The model pools aDN1 and aDN2 (the dataset's DNg62 and DNge078), so the paper's aDN1 versus aDN2 difference cannot be seen; it has no legs, so grooming "
                 "movements cannot be seen at all, only the front-leg motor neurons' firing as a stand-in.",
    questions=(_q("antenna_grooming_circuit", "You drive the 335 JO-C/E antennal mechanosensory neurons for 2 s. What do the grooming command neurons aDN1/aDN2 do, "
                  "compared with 335 random other sensory neurons?",
                  (("up", "aDN1/aDN2 fire much more (at least 1.5x calm)"), ("same", "about the same"), ("down", "fire less")), "up",
                  "The upstream half of the grooming circuit, antennal touch to the descending command neurons, reproduces strongly.",
                  UNEXPECTED_MISS),
               _q("adn_grooming_motor", "You drive aDN1/aDN2 for 2 s. What do the front-leg motor neurons do, compared with 4 random descending neurons?",
                  (("up", "front-leg motor neurons fire much more (at least 1.5x calm)"), ("small", "a small rise, well below 1.5x"), ("same", "no change")), "up",
                  UNEXPECTED_HIT,
                  "The motor half of the circuit does not reproduce: the front-leg motor neurons rise only to about x1.13 against x0.95 (consistent in direction, far below 1.5x). "
                  "The model has no legs, so nothing it does can be a grooming movement; the signal is lost in the nerve cord's interneurons, which the dataset "
                  "contains but the LIF model drives with a coarse sign rule (docs/validation.md).")),
    steps=(("The paper", "Touching the antennae drives a chain of neurons that ends in a stereotyped leg movement.", "read"),
           ("Your hypothesis", "Two predictions: antennal touch to the command neurons, and the command neurons to the leg motor neurons.", "hypothesis"),
           ("Run it", "Two validation tests, each from one brain snapshot, with a matched control.", "run"),
           ("Your result", "Each fly's firing ratio, drive next to control, for both halves.", "plot"),
           ("Next to the paper", "The first half reproduces; see what the second does and why.", "compare")),
))

_add(Paper(
    id="ohyama_2015", title="Larval rolling escape (Ohyama et al. 2015)", short="Larva rolling",
    citation="Ohyama T, Schneider-Mizell CM, Fetter RD, Aleman JV, Franconville R, Rivera-Alba M, Mensh BD, Branson KM, Simpson JH, Truman JW, Cardona A, "
             "Zlatic M (2015) A multilevel multimodal circuit enhances action selection in Drosophila. Nature 520:633-639",
    doi="10.1038/nature14297", read="abstract only (Europe PMC); the full text is behind a paywall and was not read",
    summary="In the larva brain, drive the nociceptive (and chordotonal) ascending neurons and see whether their downstream targets and the neurons annotated as the "
            "rolling command pair fire more.",
    found="Combining mechanosensory and nociceptive cues synergistically enhances the selection of the fastest mode of escape locomotion in Drosophila larvae. From an "
          "electron microscopy volume the authors reconstructed the multisensory circuit behind the synergy, found a multilevel multimodal convergence architecture, "
          "and identified functionally connected nodes that trigger the fastest locomotor mode and others that facilitate it.",
    found_numbers="none stated in the abstract",
    cannot_check="The dataset used here (Winding et al. 2023, Data S1) does not contain the class IV md sensory neurons, the Basin interneurons or Goro itself (they are in the "
                 "nerve cord). The tests drive the annotated nociceptive and chordotonal ascending neurons and read the annotated second-order PNs and the "
                 "`_telegoro-1` DN-VNC pair, which is not verified as Goro. The synergy of two modalities is not tested.",
    brain="larva",
    recorded=dict(source="docs/validation.md, 3.0 validation, larva brain, seeds 1000-1009",
                  larva_noci_to_goro_rolling=dict(drive_ratio_mean=0.66, drive_ratio_sd=0.01, control_ratio_mean=0.65, control_ratio_sd=0.01, p_value=0.001, n=10, passed=False),
                  larva_chordotonal_to_basin=dict(drive_ratio_mean=0.68, drive_ratio_sd=0.01, control_ratio_mean=0.64, control_ratio_sd=0.01, p_value=0.001, n=10, passed=False)),
    questions=(_q("larva_noci_to_goro_rolling", "You drive the 12 nociceptive ascending neurons of the larva brain for 2 s. What do the `_telegoro-1` neurons (2, the "
                  "rolling command pair in the annotations) do, compared with driving 12 random other sensory neurons?",
                  (("up", "they fire clearly more (at least 1.5x calm)"), ("same", "no change"), ("down", "they fire less")), "up",
                  UNEXPECTED_HIT,
                  "No excitation: the readout drops, for the drive and the control alike (x0.66 versus x0.65). Data S1 has no transmitter identities, so the pack makes local "
                  "neurons and MBONs inhibitory and everything else excitatory, a guess; the network then idles at about 60 Hz and any extra sensory drive lowers these "
                  "readouts whichever neurons are driven. Nothing was tuned (docs/validation.md, docs/larva.md)."),
               _q("larva_chordotonal_to_basin", "You drive the 12 chordotonal ascending neurons for 2 s. What do the second-order noci / mechano PNs do, compared with 12 random "
                  "other sensory neurons?",
                  (("up", "they fire clearly more (at least 1.5x calm)"), ("same", "no change"), ("down", "they fire less")), "up",
                  UNEXPECTED_HIT,
                  "No excitation, for the same reason: the readout drops for drive and control alike (x0.68 versus x0.64).")),
    steps=(("The paper", "Larvae escape a wasp's sting by rolling; mechanosensory and nociceptive cues together make rolling more likely.", "read"),
           ("Your hypothesis", "Predict what the larva connectome does when nociceptive neurons are driven.", "hypothesis"),
           ("Run it", "Needs the larva brain pack (built locally, docs/larva.md). Without it, the recorded validation numbers are shown, marked as recorded.", "run"),
           ("Your result", "Firing ratios per fly, drive next to control.", "plot"),
           ("Next to the paper", "The model does not reproduce this; the comparison says why.", "compare")),
))

_add(Paper(
    id="colomb_2012", title="Walking between two stripes: Buridan's paradigm (Colomb et al. 2012)", short="Buridan's paradigm",
    citation="Colomb J, Reiter L, Blaszkiewicz J, Wessnitzer J, Brembs B (2012) Open source tracking and analysis of adult Drosophila locomotion in Buridan's "
             "paradigm with and without visual targets. PLoS ONE 7(8):e42247",
    doi="10.1371/journal.pone.0042247", read="full text (open access), read through a page summary",
    summary="Put the fly on a round platform with two opposite stripes beyond its edge (and, as the control, without them) and measure how its walking is oriented to the stripe axis.",
    found="Flies show fixation and antifixation of two inaccessible visual targets: with narrow stripes they walk directly back and forth between them, with wide stripes the "
          "organisation is weaker. The chance level of the median stripe deviation for a random walk is 45 degrees, and flies' median stripe deviation was lower with narrow "
          "stripes. Narrow stripes eliminated centrophobism while moving; about a third of the experiment time was spent active.",
    found_numbers="45 degrees (chance level of stripe deviation); about 33% of the time active",
    cannot_check="Antifixation, stripe width, the water moat, centrophobism and pauses are not modeled. The sim's fly walks steadily at one speed set by DNp09. The stripe is tracked "
                 "by LC10 through the duel's rule (GAME RULE) and the platform edge reflects it (GAME RULE), so the transits are produced by those rules acting on the real "
                 "steering and walking neurons; what the connectome contributes is how accurately and how fast the steering neurons carry the fly to the stripe.",
    questions=(_q("rig:buridan", "You put a fly on the platform for two minutes, with the two stripes and without. What happens to its stripe deviation (the angle between its "
                  "walking direction and the line joining the stripes; 45 degrees is chance)?",
                  (("lower", "lower with the stripes than without, well below chance"), ("same", "about the same with and without"), ("higher", "higher with the stripes")), "lower",
                  "The model's fly orients its walking along the stripe axis when the stripes are there and wanders at about chance when they are not, the direction the paper reports. "
                  "Its deviation is far lower and far less variable than real flies', because the tracking rule is strong and the sim has no pauses or antifixation.",
                  "At least one of the rig's pre-registered criteria (B1 stripe deviation, B2 transits) did not pass in this run; docs/rigs.md has the held-out result."),),
    steps=(("The paper", "The platform is surrounded by water, so the stripes cannot be reached; the fly is free to walk anywhere.", "read"),
           ("Your hypothesis", "Say how having stripes changes the direction the fly walks in.", "hypothesis"),
           ("Run it", "Two runs per fly (stripes, then none) in the Buridan rig, two simulated minutes each.", "run"),
           ("Your result", "Median stripe deviation per fly, stripes next to none.", "plot"),
           ("Next to the paper", "The paper's chance level and direction against the model's.", "compare")),
))

ORDER = ("von_reyn_2014", "tully_quinn_1985", "shiu_2024", "hampel_2015", "ohyama_2015", "colomb_2012")


# --- running ------------------------------------------------------------------------------------------------------------------------
def available(paper: Paper) -> tuple[bool, str]:
    """Whether the paper can be run live on this machine, and if not why not (the larva pack is built locally and optional)."""
    from kickthefly.sim import brainpack

    if brainpack.find(brain=paper.brain) is None:
        if paper.brain == "larva":
            return False, ("the larva brain pack is not built (python -m kickthefly.sim.connectome.larva_loader build downloads Data S1 once; docs/larva.md); "
                           "the recorded validation numbers are shown instead")
        return False, "the adult brain pack is not built (python -m kickthefly.sim.brainpack build)"
    return True, ""


def _ratio_pairs(t: dict) -> list[dict]:
    out = []
    for s in t.get("per_seed", []):
        if "drive" in s:                                       # validation's {ratio: ...} dicts, or plain numbers (the rig's deviations)
            d, c = s["drive"], s["control"]
            out.append(dict(seed=s["seed"], drive=d["ratio"] if isinstance(d, dict) else float(d), control=c["ratio"] if isinstance(c, dict) else float(c)))
        else:                                                  # the T-maze: PI paired and unpaired
            out.append(dict(seed=s["seed"], drive=s["pi"], control=s["control_pi"]))
    return out


def _verdict(test_id: str, t: dict, full: bool) -> dict:
    """reproduced or not, by validation's criteria on a full run; on a quick run by effect and direction only (n too small for p < 0.01)."""
    m = t["measured"]
    if test_id == "mb_conditioning":
        eff = m["pi_mean"] >= 0.5 and abs(m["control_pi_mean"]) <= 0.25 and m["pi_mean"] > m["control_pi_mean"]
    else:
        eff = m["drive_ratio_mean"] >= RATIO_MIN and m["drive_ratio_mean"] > m["control_ratio_mean"]
    n = int(m.get("n", len(t.get("per_seed", []))))
    p = m.get("p_value")
    if full:
        ok = bool(t.get("passed", eff and p is not None and p < P_MAX))
        basis = "validation's criteria on the held-out seeds (n = 10, p < 0.01)"
    else:
        ok = bool(eff)
        basis = f"effect size and direction only: n = {n} is too few flies for the validation's p < 0.01 (the smallest one-sided Wilcoxon p at n = {n} is {1 / 2 ** n:.3g})"
    return dict(reproduced=ok, basis=basis, measured=m, n=n)


def run_paper(paper_id: str, seeds=QUICK_SEEDS, workers: int = 1, progress=None, cancel=None, play=None) -> dict:
    """Run a mini-paper's experiment(s) on the model. play: a function (paper, seeds) -> {test_id: validation-style test dict}, replacing the brains (tests).
    A paper that cannot run live here (the larva pack) returns the recorded numbers, marked source = 'recorded'."""
    p = PAPERS[paper_id]
    seeds = [int(s) for s in seeds]
    full = tuple(seeds) == FULL_SEEDS
    t0 = time.time()
    ok, why = available(p)
    got: dict[str, dict] = {}
    source = "live"
    if play is not None:
        got = play(p, seeds)
    elif not ok and not p.recorded:
        raise FileNotFoundError(why)                    # an adult paper has no recorded numbers to fall back on
    elif not ok:
        source = "recorded"
        for q in p.questions:
            rec = p.recorded[q.test]
            got[q.test] = dict(measured=dict(rec), passed=rec["passed"], per_seed=[], id=q.test)
        full = True                                    # the recorded numbers are the held-out run's
    elif p.questions[0].test.startswith("rig:"):
        from kickthefly.lab import rigassay

        name = p.questions[0].test.split(":", 1)[1]
        flies = rigassay.run_flies(name, seeds, rigassay.DEFAULT_MODE, workers, None, progress, cancel)
        an = rigassay.analyze(name, flies)
        got[p.questions[0].test] = dict(rig=name, flies=flies, analysis=an,
                                        measured=dict(deviation_stripes_deg=float(np.mean([f["stripes"]["deviation_deg"] for f in flies])),
                                                      deviation_none_deg=float(np.mean([f["none"]["deviation_deg"] for f in flies])),
                                                      transits_stripes=float(np.mean([f["stripes"]["transits"] for f in flies])),
                                                      transits_none=float(np.mean([f["none"]["transits"] for f in flies])), n=len(flies)),
                                        passed=an["all_passed"], effects_met=all(c["mean_effect"] >= c["min_effect"] for c in an["criteria"]),
                                        per_seed=[dict(seed=f["seed"], drive=f["stripes"]["deviation_deg"], control=f["none"]["deviation_deg"]) for f in flies])
    else:
        from kickthefly.lab import validation

        res = validation.run(seeds=tuple(seeds), workers=max(1, workers), progress=progress, include=[q.test for q in p.questions], cancel=cancel)
        got = {t["id"]: t for t in res["tests"] if t["id"] in {q.test for q in p.questions}}
    qs = []
    for q in p.questions:
        t = got[q.test]
        if q.test.startswith("rig:"):
            m = t["measured"]
            if full:
                v = dict(reproduced=bool(t["passed"]), basis="the rig's pre-registered criteria B1 and B2 (minimum effects and one-sided Wilcoxon p < 0.01, n = 10)",
                         measured=m, n=m["n"])
            else:                                                   # n = 4 cannot reach p < 0.01: the effects' minimums only, like the validation papers' quick runs
                v = dict(reproduced=bool(t["effects_met"]), measured=m, n=m["n"],
                         basis=f"the rig's minimum effects (B1 10 degrees, B2 3 transits) only: n = {m['n']} is too few flies for p < 0.01 (the smallest one-sided Wilcoxon p at "
                               f"n = {m['n']} is {1 / 2 ** m['n']:.3g})")
        else:
            v = _verdict(q.test, t, full)
        qs.append(dict(test=q.test, prompt=q.prompt, expected=q.expected, verdict=v, pairs=_ratio_pairs(t) if "per_seed" in t else [],
                       note=q.pass_note if v["reproduced"] else q.fail_why))
    return dict(kind="minipaper", paper=paper_id, title=p.title, citation=p.citation, doi=p.doi, read=p.read, seeds=seeds, full=full, source=source,
                unavailable=why, seconds=round(time.time() - t0, 1), questions=qs, created=time.strftime("%Y-%m-%d %H:%M:%S"))


def hypothesis_text(paper: Paper, answers: dict[str, str] | None) -> list[str]:
    """For each question, whether the player's choice matched what the paper found and what the model gave: (kept for the page and --minipaper)."""
    return []


def compare(res: dict, answers: dict[str, str] | None = None) -> list[dict]:
    """The comparison table: per question, your hypothesis, the paper's direction, the model's result and whether they agree."""
    p = PAPERS[res["paper"]]
    rows = []
    for q, r in zip(p.questions, res["questions"]):
        mine = (answers or {}).get(q.test)
        lab = dict(q.options)
        reproduced = r["verdict"]["reproduced"]
        rows.append(dict(test=q.test, question=q.prompt, your_hypothesis=lab.get(mine) if mine else None, paper_direction=lab[q.expected],
                         model_reproduces_paper=reproduced, you_matched_the_paper=(mine == q.expected) if mine else None,
                         you_matched_the_model=(((mine == q.expected) == reproduced) if mine else None), basis=r["verdict"]["basis"], note=r["note"],
                         measured=r["verdict"]["measured"]))
    return rows


def guided_protocol(paper: Paper):
    """The paper as a lecture protocol (lab/classroom.py), so Lab > Classroom mode style stepping and ClassroomSession work on it."""
    from kickthefly.lab.classroom import LectureProtocol, LectureStep, NeuronInvolved

    steps = []
    for title, text, kind in paper.steps:
        steps.append(LectureStep(title=f"{len(steps) + 1}. {title}", explanation=text, key_takeaway=paper.summary if kind == "read" else text,
                                 citation=paper.citation, neurons=[NeuronInvolved(paper.short, "", "see the experiment's neurons on the Run step", 0)],
                                 action=dict(kind="minipaper_" + kind), grounding="LITERATURE (the paper) next to a MODEL PREDICTION (your run)"))
    return LectureProtocol(id=f"paper_{paper.id}", title=paper.title, description=paper.summary, steps=steps)


def register() -> None:
    from kickthefly.lab import classroom

    for pid in ORDER:
        classroom.EXTRA_LECTURES[f"paper_{pid}"] = guided_protocol(PAPERS[pid])


register()


# --- text, plot and files --------------------------------------------------------------------------------------------------------------
def render(res: dict, answers: dict[str, str] | None = None) -> str:
    p = PAPERS[res["paper"]]
    lines = [f"MINI-PAPER: {p.title}", f"  {p.citation}", f"  doi:{p.doi}   read: {p.read}", "",
             f"  The paper reports: {p.found}", f"  Numbers the paper states: {p.found_numbers}", ""]
    src = "RECORDED validation numbers (not run now): " + res["unavailable"] if res["source"] == "recorded" else \
        f"RUN NOW on {len(res['seeds'])} flies (seeds {res['seeds'][0]}-{res['seeds'][-1]}{'' if res['full'] else ', exploration seeds: a first look'})"
    lines.append(f"  Your result: {src}")
    for row in compare(res, answers):
        m = row["measured"]
        nums = ", ".join(f"{k} {v:.3g}" if isinstance(v, float) else f"{k} {v}" for k, v in m.items() if k != "passed")
        lines += ["", f"  Q ({row['test']}): {row['question']}", f"    the direction the paper's finding points to: {row['paper_direction']}",
                  f"    your hypothesis: {row['your_hypothesis'] or 'not given'}",
                  f"    the model {'REPRODUCES' if row['model_reproduces_paper'] else 'DOES NOT REPRODUCE'} that ({row['basis']})", f"    numbers: {nums}",
                  f"    {row['note']}"]
    lines += ["", f"  What this model cannot check: {p.cannot_check}"]
    return "\n".join(lines)


def svg_pairs(res: dict, width: int = 560, row_h: int = 160) -> str:
    """A paired dot plot per question: each fly's drive and its control joined by a line. Plain SVG, no dependencies."""
    qs = [q for q in res["questions"] if q["pairs"]]
    h = row_h * max(1, len(qs)) + 20
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{h}" font-family="sans-serif" font-size="11">']
    for i, q in enumerate(qs):
        top = 10 + i * row_h
        vals = [v for pr in q["pairs"] for v in (pr["drive"], pr["control"])]
        lo, hi = min(0.0, min(vals)), max(vals) * 1.1 or 1.0
        def y(v):
            return top + row_h - 40 - (v - lo) / (hi - lo) * (row_h - 70)
        out.append(f'<text x="10" y="{top + 12}">{q["test"]}</text>')
        out.append(f'<line x1="150" y1="{y(0):.1f}" x2="{width - 20}" y2="{y(0):.1f}" stroke="#888"/>')
        xs = (260, 440)
        for pr in q["pairs"]:
            out.append(f'<line x1="{xs[0]}" y1="{y(pr["control"]):.1f}" x2="{xs[1]}" y2="{y(pr["drive"]):.1f}" stroke="#999"/>')
            out.append(f'<circle cx="{xs[0]}" cy="{y(pr["control"]):.1f}" r="4" fill="#6b7a8f"/><circle cx="{xs[1]}" cy="{y(pr["drive"]):.1f}" r="4" fill="#d9822b"/>')
        out.append(f'<text x="{xs[0] - 20}" y="{top + row_h - 20}">control</text><text x="{xs[1] - 14}" y="{top + row_h - 20}">drive</text>')
    out.append("</svg>")
    return "\n".join(out)


def save(res: dict, folder: Path, answers: dict[str, str] | None = None) -> list[Path]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    p = folder / f"minipaper_{res['paper']}.json"
    p.write_text(json.dumps(dict(result=res, comparison=compare(res, answers), hypotheses=answers or {}), indent=1, default=str), encoding="utf-8")
    out.append(p)
    p = folder / f"minipaper_{res['paper']}.txt"
    p.write_text(render(res, answers) + "\n", encoding="utf-8")
    out.append(p)
    if any(q["pairs"] for q in res["questions"]):
        p = folder / f"minipaper_{res['paper']}.svg"
        p.write_text(svg_pairs(res), encoding="utf-8")
        out.append(p)
    return out


def main(args) -> int:
    """--headless --minipaper ID [--seeds A-B] [--workers N] [--out DIR]; ID 'list' lists them. Default seeds: the quick run (exploration seeds 0-3)."""
    from kickthefly.lab import headless, recorder

    pid = args.minipaper
    if pid == "list":
        for k in ORDER:
            p = PAPERS[k]
            print(f"{k:18s} {p.short:28s} {p.citation.split('(')[0].strip()} ({p.read.split(' (')[0]})")
        return 0
    if pid not in PAPERS:
        print(f"error: unknown mini-paper {pid!r}; use --minipaper list", flush=True)
        return 2
    seeds = headless.parse_seeds(args.seeds, QUICK_SEEDS)
    t0 = time.time()
    res = run_paper(pid, seeds, workers=max(1, args.workers or 1), progress=lambda d, n, label="": print(f"  {d}/{n} {label}", flush=True))
    folder = Path(args.out) if args.out else recorder.exports_dir() / f"{time.strftime('%Y%m%d-%H%M%S')}-minipaper-{pid}"
    save(res, folder)
    print(render(res))
    print(f"\nwritten to {folder} ({time.time() - t0:.0f} s)")
    return 0
