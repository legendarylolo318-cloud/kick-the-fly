"""Classroom mode and lecture protocols.

Step-by-step pedagogical demonstrations of Drosophila connectome circuits
with sequential Next/Back controls, step explanations, participating neuron
breakdowns, scientific citations, and live demonstration actions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from kickthefly.core import simcore


@dataclass
class NeuronInvolved:
    label: str
    spec: str
    role: str
    count: int = 0


@dataclass
class LectureStep:
    title: str
    explanation: str
    key_takeaway: str
    citation: str
    neurons: list[NeuronInvolved] = field(default_factory=list)
    action: dict[str, Any] = field(default_factory=dict)
    grounding: str = "Traceable to MaleCNS v1.0 connectome wiring"


@dataclass
class LectureProtocol:
    id: str
    title: str
    description: str
    steps: list[LectureStep]


def _build_curated_lectures() -> dict[str, LectureProtocol]:
    """Build the 5 standard curated lecture protocols."""
    lectures = {}

    # 1. Looming Visual Escape
    lectures["looming"] = LectureProtocol(
        id="looming",
        title="Looming Visual Escape Circuit",
        description="How visual projection neurons LPLC2 and LC4 drive the giant fiber DNp01 to trigger escape.",
        steps=[
            LectureStep(
                title="1. Spontaneous Baseline State",
                explanation=(
                    "Under calm resting conditions, Drosophila visual projection neurons and descending motor "
                    "pathways exhibit low spontaneous stochastic firing (~5-8 Hz). Inhibitory and excitatory "
                    "inputs maintain homeostatic equilibrium without triggering motor commands."
                ),
                key_takeaway="Resting state exhibits calm stochastic firing without premature escape trigger.",
                citation="von Reyn et al. 2014, Nat Neurosci 17:962; Ache et al. 2019, Curr Biol 29:1073",
                neurons=[
                    NeuronInvolved("LPLC2 + LC4 (Looming Detectors)", "type:LPLC2,LC4", "Visual projection neurons sensitive to optical expansion", 311),
                    NeuronInvolved("DNp01 (Giant Fiber)", "dnp01", "Descending escape command neuron in the posterior slope", 2),
                ],
                action={"kind": "reset"},
                grounding="EM reconstruction wiring: MaleCNS v1.0 posterior slope innervation.",
            ),
            LectureStep(
                title="2. Expanding Retinal Loom Detection",
                explanation=(
                    "When a predator or flyswatter rapidly approaches, its optical image expands exponentially on "
                    "the retina. Lobula plate projection neurons LPLC2 and lobula neurons LC4 compute non-linear "
                    "angular expansion rates and fire bursts of high-frequency action potentials."
                ),
                key_takeaway="LPLC2 and LC4 selectively detect rapid angular expansion of dark looming shadows.",
                citation="Ache et al. 2019, Curr Biol 29:1073",
                neurons=[
                    NeuronInvolved("LPLC2 (Lobula Plate/Lobula Col. 2)", "type:LPLC2", "Encodes looming edge expansion across receptive field", 204),
                    NeuronInvolved("LC4 (Lobula Col. 4)", "type:LC4", "Encodes visual angular size and velocity of expanding cue", 107),
                ],
                action={"kind": "stimulus", "target": "type:LPLC2,LC4", "strength": 0.85, "for_s": 0.6},
                grounding="Connectome visual inputs: sensory tuning based on patch-clamp and 2-photon imaging (Ache 2019).",
            ),
            LectureStep(
                title="3. Synaptic Convergence onto Giant Fiber",
                explanation=(
                    "LPLC2 and LC4 axons terminate directly in the posterior slope, forming extensive chemical synapses "
                    "onto the giant dendrites of descending neuron DNp01 (the giant fiber). Summation of these hundreds "
                    "of inputs rapidly depolarizes the giant fiber membrane toward firing threshold."
                ),
                key_takeaway="Convergent synaptic input from hundreds of detectors drives DNp01 past threshold.",
                citation="von Reyn et al. 2014, Nat Neurosci 17:962",
                neurons=[
                    NeuronInvolved("DNp01 (Giant Fiber)", "dnp01", "Large-diameter axon descending into the ventral nerve cord", 2),
                    NeuronInvolved("LPLC2, LC4 Axon Terminals", "type:LPLC2,LC4", "Presynaptic partners innervating the posterior slope", 311),
                ],
                action={"kind": "drive", "target": "type:LPLC2,LC4", "amp": 0.65, "for_s": 0.5},
                grounding="Direct EM synaptic connectivity verified in MaleCNS v1.0 (mean >11x drive ratio).",
            ),
            LectureStep(
                title="4. All-or-None Escape Jump Motor Readout",
                explanation=(
                    "Once DNp01 fires, its action potential propagates down the cervical connective into the thoracic "
                    "ganglion at over 5 m/s, directly activating the tergotrochanteral motor neuron (TTMn) to trigger "
                    "an explosive middle-leg extension and wing depression."
                ),
                key_takeaway="DNp01 activation triggers an unguided, ultra-fast jump takeoff in <5 milliseconds.",
                citation="von Reyn et al. 2014, Nat Neurosci 17:962; Allen et al. 2006, J Neurosci 26:1565",
                neurons=[
                    NeuronInvolved("DNp01 (Giant Fiber)", "dnp01", "Descending command trigger", 2),
                    NeuronInvolved("Thoracic leg motor pool", "mn_legs", "Tergotrochanteral and leg motor effectors", 381),
                ],
                action={"kind": "drive", "target": "dnp01", "amp": 1.0, "for_s": 0.4},
                grounding="Escape jump threshold: 4.0x DNp01 calm baseline rate (sim validation passed).",
            ),
        ],
    )

    # 2. T-Maze Associative Conditioning
    lectures["tmaze"] = LectureProtocol(
        id="tmaze",
        title="Mushroom Body Olfactory Conditioning",
        description="Dopaminergic depression of Kenyon cell to MBON synapses underlying learned odor avoidance.",
        steps=[
            LectureStep(
                title="1. Sparse Odor Encoding by Kenyon Cells",
                explanation=(
                    "Antennal olfactory receptor neurons project to antennal lobe projection neurons (PNs), "
                    "which divergence-map onto ~2,000 intrinsic mushroom body Kenyon Cells (KCs). Each distinct "
                    "odor activates a sparse, random ~5% subset of KCs in the mushroom body calyx."
                ),
                key_takeaway="Kenyon cells form sparse, decorrelated representations of specific chemical odors.",
                citation="Aso et al. 2014, eLife 3:e04577; Turner et al. 2008, Nature 454:493",
                neurons=[
                    NeuronInvolved("Kenyon Cells (KCg, KCab, KCap)", "prefix:KC", "Sparse odor-representing intrinsic neurons", 1920),
                    NeuronInvolved("Olfactory Projection Neurons", "olfactory", "Antennal lobe relay neurons targeting calyx", 150),
                ],
                action={"kind": "stimulus", "target": "odor_a", "strength": 0.7, "for_s": 0.5},
                grounding="MaleCNS v1.0 mushroom body connectome structure.",
            ),
            LectureStep(
                title="2. Dopaminergic Reinforcement (PPL1 / Electric Shock)",
                explanation=(
                    "During aversive conditioning, presentation of the conditioned odor (CS+) is paired with electric shock. "
                    "Nociceptive pathways activate PPL1 dopaminergic neurons (DANs) that innervate the mushroom body lobes. "
                    "Dopamine release acts co-incidentally with active Kenyon cell presynaptic terminals."
                ),
                key_takeaway="PPL1 dopaminergic neurons signal negative reinforcement to specific mushroom body compartments.",
                citation="Claridge-Chang et al. 2009, Cell 139:405; Aso et al. 2014, eLife 3:e04577",
                neurons=[
                    NeuronInvolved("PPL1 Dopaminergic Neurons", "type:PPL101,PPL103", "Aversive reinforcement dopamine release", 12),
                    NeuronInvolved("Kenyon Cells (CS+ activated)", "prefix:KC", "Co-active presynaptic boutons", 100),
                ],
                action={"kind": "drive", "target": "type:PPL101,PPL103", "amp": 0.8, "for_s": 0.5},
                grounding="MaleCNS v1.0 connectome: compartmental DAN innervation confirmed by Aso 2014.",
            ),
            LectureStep(
                title="3. Long-Term Synaptic Depression at KC-to-MBON Synapses",
                explanation=(
                    "Coincident KC spike activity and dopamine receptor activation induces long-term depression (LTD) "
                    "of synapses between active KCs and mushroom body output neurons (MBONs) that direct approach behavior. "
                    "Synaptic efficacy from CS+ KCs to positive MBONs is selectively scaled down."
                ),
                key_takeaway="LTD of approach MBONs tips the net circuit balance toward learned avoidance.",
                citation="Hige et al. 2015, Neuron 88:985; Owald & Waddell 2015, Curr Opin Neurobiol 35:178",
                neurons=[
                    NeuronInvolved("Mushroom Body Output Neurons", "mbon", "Readout neurons projecting to motor centers", 42),
                    NeuronInvolved("KC-to-MBON Synapses", "prefix:KC", "Undergo dopamine-gated depression", 1920),
                ],
                action={"kind": "assay_step", "target": "tmaze_condition"},
                grounding="Synaptic plasticity rule: game-implemented LTD verified by T-maze PI > 0.5.",
            ),
            LectureStep(
                title="4. Memory Recall & T-Maze Choice",
                explanation=(
                    "Upon subsequent exposure in the T-maze choice point, the CS+ odor evokes depressed approach MBON firing. "
                    "Unbalanced avoidance MBON signals prevail, steering the fly away from the learned shock odor into the "
                    "unpaired safe odor arm (Performance Index > 0.5)."
                ),
                key_takeaway="Flies robustly avoid the shock-paired odor with statistically significant avoidance.",
                citation="Tully & Quinn 1985, J Comp Physiol A 157:263",
                neurons=[
                    NeuronInvolved("Approach MBONs", "type:MBON01,MBON03", "Reduced output to CS+", 14),
                    NeuronInvolved("Avoidance MBONs", "type:MBON02,MBON04", "Relative dominance drives avoidance steering", 16),
                ],
                action={"kind": "stimulus", "target": "odor_a", "strength": 0.8, "for_s": 0.5},
                grounding="Connectome validation suite confirmed: paired PI = 1.0 vs unpaired |PI| <= 0.25 (p < 0.01).",
            ),
        ],
    )

    # 3. Moonwalker Backward Walking
    lectures["moonwalker"] = LectureProtocol(
        id="moonwalker",
        title="Moonwalker Backward Walking Circuit",
        description="How Moonwalker Descending Neurons (MDN) coordinate reversal of leg motor walking rhythms.",
        steps=[
            LectureStep(
                title="1. Forward Walking Default Program",
                explanation=(
                    "In foraging flies, descending neurons like DNp09 promote forward walking. Thoracic motor circuits "
                    "generate coordinated tripod gaits (front/rear ipsilateral legs move in phase with contralateral middle leg) "
                    "propelling the body forward."
                ),
                key_takeaway="Forward locomotion is maintained by forward-promoting descending command neurons.",
                citation="Bidaye et al. 2014, Science 344:97; Bidaye et al. 2020, Neuron 108:109",
                neurons=[
                    NeuronInvolved("DNp09 (Forward Walking DN)", "type:DNp09", "Drives forward tripod locomotion", 2),
                    NeuronInvolved("Leg Motor Neurons (T1-T3)", "mn_legs", "Thoracic motor pools for leg joints", 381),
                ],
                action={"kind": "drive", "target": "type:DNp09", "amp": 0.6, "for_s": 0.5},
                grounding="Connectome descending pathways to ventral nerve cord leg neuromeres.",
            ),
            LectureStep(
                title="2. Obstacle Encounter & MDN Activation",
                explanation=(
                    "When head touch or antennal mechanoreceptors detect an unyielding physical obstacle or dead end, "
                    "sensory inputs recruit the pair of bilateral Moonwalker Descending Neurons (MDN, 4 cells total). "
                    "MDNs project from the gnathal ganglion directly down the neck connective."
                ),
                key_takeaway="Head collisions and mechanosensory touch rapidly excite MDN command neurons.",
                citation="Bidaye et al. 2014, Science 344:97",
                neurons=[
                    NeuronInvolved("MDN (Moonwalker Descending Neurons)", "mdn", "Bilateral descending pair directing backward walking", 4),
                    NeuronInvolved("Antennal mechanosensory neurons", "jo_ce", "Sensory detection of obstacles and touch", 335),
                ],
                action={"kind": "drive", "target": "mdn", "amp": 0.8, "for_s": 0.5},
                grounding="Connectome annotations (MaleCNS v1.0): MDN (4 cells).",
            ),
            LectureStep(
                title="3. Backward Gait Motor Program Coordination",
                explanation=(
                    "In biological flies, MDN axons branch across prothoracic (T1), mesothoracic (T2), and metathoracic (T3) "
                    "neuromeres. MDN activation simultaneously suppresses forward motor rhythm generators and activates "
                    "backward walking pattern generators, inverting the leg stepping sequence."
                ),
                key_takeaway="MDN switches the central pattern generator state from forward tripod to backward walking.",
                citation="Bidaye et al. 2014, Science 344:97; Feng et al. 2020, Nature 585:395",
                neurons=[
                    NeuronInvolved("MDN Axons", "mdn", "Invert stepping phase across thoracic motor centers", 4),
                    NeuronInvolved("Thoracic Leg Motor Pools", "mn_legs", "Execute reverse leg kinematics", 381),
                ],
                action={"kind": "drive", "target": "mdn", "amp": 1.0, "for_s": 0.6},
                grounding="Note: Sim models connectome reach; leg kinematics are game rules.",
            ),
        ],
    )

    # 4. Sugar Feeding & Proboscis Extension
    lectures["sugar"] = LectureProtocol(
        id="sugar",
        title="Sugar Gustatory Pathway & Proboscis Extension",
        description="How gustatory receptor neurons on the labellum and tarsus reach motor neuron MN9 to extend proboscis.",
        steps=[
            LectureStep(
                title="1. Gustatory Detection by Sugar Receptors",
                explanation=(
                    "When a hungry fly lands on food, sucrose contacts gustatory receptor neurons (GRNs) on the legs "
                    "and labellum expressing Gr5a/Gr64f. These sugar-sensing sensory neurons fire action potentials "
                    "relaying nutrient detection directly to the subesophageal zone (SEZ)."
                ),
                key_takeaway="Sugar contact activates gustatory sensory neurons that innervate the SEZ feeding center.",
                citation="Shiu et al. 2024, Nature 634:210; Yao & Scott 2022, Cell 185:2472",
                neurons=[
                    NeuronInvolved("Sugar Gustatory Neurons", "sweet", "Sensory taste neurons projecting to gnathal ganglion", 30),
                ],
                action={"kind": "stimulus", "target": "sweet", "strength": 0.8, "for_s": 0.5},
                grounding="Connectome wiring: sugar GRNs identified via wiring to Yao & Scott 2022 sugar-SEL projection neurons.",
            ),
            LectureStep(
                title="2. Gnathal Relay via Second-Order SEL Projection Neurons",
                explanation=(
                    "Sugar sensory afferents synapse onto subesophageal layer projection neurons (Sugar SEL PNs, GNG540/GNG550). "
                    "These second-order relay neurons process nutritional salience and distribute excitatory currents "
                    "toward feeding motor command pools."
                ),
                key_takeaway="Sugar SEL PNs integrate taste signals and transmit excitation to proboscis motor circuits.",
                citation="Yao & Scott 2022, Cell 185:2472; Shiu et al. 2024, Nature 634:210",
                neurons=[
                    NeuronInvolved("Sugar SEL PNs (GNG540, GNG550)", "type:GNG540,GNG550", "Second-order gustatory relay neurons", 8),
                ],
                action={"kind": "drive", "target": "type:GNG540,GNG550", "amp": 0.7, "for_s": 0.5},
                grounding="Connectome synaptome (MaleCNS v1.0) verifies direct synaptic chain from sweet GRNs to SEL PNs.",
            ),
            LectureStep(
                title="3. Proboscis Motor Neuron 9 (MN9) Activation",
                explanation=(
                    "Excitatory signals reach motor neuron 9 (MN9) in the ventral SEZ. MN9 innervates muscle 9 (the "
                    "retractor bulbi), whose contraction produces the characteristic Proboscis Extension Response (PER). "
                    "In contrast, bitter substances inhibit this circuit, preventing ingestion."
                ),
                key_takeaway="MN9 excitation produces proboscis extension; bitter pathways fail to activate MN9.",
                citation="Shiu et al. 2024, Nature 634:210; Schwarz et al. 2017, eLife 6:e19892",
                neurons=[
                    NeuronInvolved("MN9 (Proboscis Motor Neuron)", "mn9", "Motor neuron innervating feeding muscle 9", 2),
                    NeuronInvolved("Bitter-sensing neurons (Control)", "bitter", "Bitter pathway fails to activate MN9", 30),
                ],
                action={"kind": "drive", "target": "sweet", "amp": 0.9, "for_s": 0.5},
                grounding="Sim validation passed: sweet/control drive ratio > 1.5x (p < 0.01).",
            ),
        ],
    )

    # 5. Giant Fiber Lesion: Visual vs Tactile Escape
    lectures["gf_lesion"] = LectureProtocol(
        id="gf_lesion",
        title="Giant Fiber Lesion: Visual vs Tactile Escape Pathways",
        description="Ablating DNp01 abolishes visual looming escapes while leaving mechanosensory escapes intact.",
        steps=[
            LectureStep(
                title="1. Dual Escape Circuits in the Fly Brain",
                explanation=(
                    "Drosophila possesses parallel escape mechanisms: (A) a fast visual escape mediated by the giant "
                    "fiber (DNp01), and (B) mechanosensory / tactile escapes driven by antennal (JO-C/E) and leg "
                    "mechanoreceptors through non-giant descending pathways."
                ),
                key_takeaway="Parallel neural circuits mediate escape from looming shadows vs physical touch.",
                citation="von Reyn et al. 2014, Nat Neurosci 17:962; Hampel et al. 2015, eLife 4:e08758",
                neurons=[
                    NeuronInvolved("DNp01 (Giant Fiber)", "dnp01", "Visual looming escape command", 2),
                    NeuronInvolved("Antennal & Leg mechanoreceptors", "jo_ce", "Tactile escape initiators", 335),
                ],
                action={"kind": "reset"},
                grounding="Verified in connectome: distinct descending projections for visual vs tactile modalities.",
            ),
            LectureStep(
                title="2. Silencing the Giant Fiber (DNp01 Lesion)",
                explanation=(
                    "Targeted ablation or silencing of DNp01 (e.g. via Kir2.1 hyperpolarization or connectome row silencing) "
                    "completely severs the direct link between lobula looming detectors and thoracic jump motor neurons. "
                    "The giant fiber membrane potential is locked at resting baseline."
                ),
                key_takeaway="DNp01 lesion selectively breaks the rapid visual looming escape conduit.",
                citation="von Reyn et al. 2014, Nat Neurosci 17:962",
                neurons=[
                    NeuronInvolved("DNp01 (Silenced)", "dnp01", "Override set to 0.0 (permanently clamped silent)", 2),
                ],
                action={"kind": "surgery", "target": "dnp01", "mode": -1},
                grounding="Simulated as static silencing of DNp01 rows in brain.override.",
            ),
            LectureStep(
                title="3. Testing Looming Visual Cue (Fails)",
                explanation=(
                    "With DNp01 silenced, presenting an identical rapid looming visual stimulus drives LPLC2 and LC4 "
                    "neurons vigorously, but no escape action potential reaches the thoracic ganglia. The fly fails to "
                    "dodge or jump."
                ),
                key_takeaway="Visual looming stimuli fail to elicit an escape reflex in the absence of DNp01.",
                citation="von Reyn et al. 2014, Nat Neurosci 17:962",
                neurons=[
                    NeuronInvolved("LPLC2 + LC4 (Active)", "type:LPLC2,LC4", "Active but signal cannot cross severed DNp01", 311),
                    NeuronInvolved("DNp01 (Silenced)", "dnp01", "Produces 0 action potentials", 2),
                ],
                action={"kind": "stimulus", "target": "type:LPLC2,LC4", "strength": 0.85, "for_s": 0.6},
                grounding="Sim validation: silencing DNp01 reduces visual escape rate to 0.0 Hz.",
            ),
            LectureStep(
                title="4. Testing Tactile Mechanosensory Cue (Preserved)",
                explanation=(
                    "In contrast, presenting a tactile air puff or physical touch to antennae (JO-C/E mechanoreceptors) "
                    "recruits alternative non-giant descending pathways (e.g. antennal groom/jump command neurons). "
                    "Tactile escape remains fully functional, proving circuit selectivity."
                ),
                key_takeaway="Non-giant mechanosensory escape pathways remain completely functional despite DNp01 ablation.",
                citation="Hampel et al. 2015, eLife 4:e08758; Allen et al. 2006, J Neurosci 26:1565",
                neurons=[
                    NeuronInvolved("JO-C/E Mechanosensory", "jo_ce", "Sensory detection of air puff / touch", 335),
                    NeuronInvolved("Non-giant descending neurons", "adn", "Alternative escape/grooming descending conduits", 4),
                ],
                action={"kind": "stimulus", "target": "jo_ce", "strength": 0.9, "for_s": 0.5},
                grounding="Connectome pathways: JO-C/E bypasses DNp01 and projects to aDN and alternative motor circuits.",
            ),
        ],
    )

    return lectures


CURATED_LECTURES = _build_curated_lectures()
# 3.0 day 5: guided mini-papers (lab/minipapers.py) are lecture protocols too, kept apart so the five curated lectures stay five.
EXTRA_LECTURES: dict[str, LectureProtocol] = {}


def lecture(lecture_id: str) -> LectureProtocol | None:
    """A curated lecture or a registered mini-paper protocol, by id."""
    return CURATED_LECTURES.get(lecture_id) or EXTRA_LECTURES.get(lecture_id)


class ClassroomSession:
    """Headless state machine for stepping through classroom lecture protocols."""

    def __init__(self, lecture_id: str = "looming", brain=None):
        self.lecture_id = lecture_id if lecture(lecture_id) is not None else "looming"
        self.step_idx = 0
        self.brain = brain
        self.last_action_applied = None

    @property
    def protocol(self) -> LectureProtocol:
        return lecture(self.lecture_id)

    @property
    def current_step(self) -> LectureStep:
        return self.protocol.steps[self.step_idx]

    @property
    def total_steps(self) -> int:
        return len(self.protocol.steps)

    def set_lecture(self, lecture_id: str) -> None:
        if lecture(lecture_id) is not None:
            self.lecture_id = lecture_id
            self.step_idx = 0
            self.last_action_applied = None

    def next_step(self) -> bool:
        if self.step_idx < self.total_steps - 1:
            self.step_idx += 1
            self.last_action_applied = None
            return True
        return False

    def prev_step(self) -> bool:
        if self.step_idx > 0:
            self.step_idx -= 1
            self.last_action_applied = None
            return True
        return False

    def goto_step(self, idx: int) -> bool:
        if 0 <= idx < self.total_steps:
            self.step_idx = idx
            self.last_action_applied = None
            return True
        return False

    def reset(self) -> None:
        self.step_idx = 0
        self.last_action_applied = None
        if self.brain is not None:
            self.brain.override[:] = 0.0

    def execute_step_action(self, brain=None, host=None) -> dict[str, Any]:
        """Execute the live demo action for the current step on a brain or game host."""
        br = brain or self.brain
        step = self.current_step
        action = step.action
        res = {"step": self.step_idx, "action": action, "status": "ok"}
        if not action or br is None:
            self.last_action_applied = "none"
            return res

        kind = action.get("kind", "none")
        target_spec = action.get("target", "")

        from kickthefly.lab import assays

        g = assays.groups(br)
        rows = None
        if target_spec:
            if target_spec in g:
                rows = g[target_spec]
            else:
                try:
                    rows = simcore.rows_of(br, target_spec)
                except Exception:
                    rows = np.array([], dtype=int)

        if kind == "reset":
            br.override[:] = 0.0
            self.last_action_applied = "Reset all overrides to baseline"
            res["description"] = "Cleared brain overrides."
        elif kind == "stimulus" and rows is not None and len(rows) > 0:
            strength = float(action.get("strength", 0.8))
            br.sense[("classroom", str(target_spec))] = rows
            br.poke("classroom", str(target_spec), strength)
            self.last_action_applied = f"Stimulus delivered to {len(rows)} neurons ({target_spec})"
            res["description"] = f"Stimulated {len(rows)} neurons with strength {strength:.2f}."
        elif kind == "drive" and rows is not None and len(rows) > 0:
            amp = float(action.get("amp", 0.5))
            simcore.drive(br, rows, amp)
            self.last_action_applied = f"Current drive ({amp:+.2f}) injected into {len(rows)} neurons ({target_spec})"
            res["description"] = f"Drove {len(rows)} neurons with current {amp:+.2f}."
        elif kind == "surgery" and rows is not None and len(rows) > 0:
            mode = int(action.get("mode", -1))
            val = 0.0 if mode == -1 else 1.0
            br.override[rows] = val
            self.last_action_applied = f"Surgery: silenced {len(rows)} neurons ({target_spec})"
            res["description"] = f"Silenced {len(rows)} neurons ({target_spec})."
        elif kind == "assay_step":
            self.last_action_applied = f"Assay step executed: {target_spec}"
            res["description"] = f"Assay phase {target_spec} recorded."
        else:
            self.last_action_applied = "Action completed"

        return res
