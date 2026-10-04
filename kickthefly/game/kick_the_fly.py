"""Kick the Fly: a kick-the-buddy toy whose buddy is the MaleCNS v1.0 connectome.

This module's docstring is the canonical map of what comes from the connectome and what is a game rule; the
launcher shim kick_the_fly.py in the repo root points here, and README.md says the same in prose.

Every hit lands on real touch-sensing neurons, and the live LIF sim from
kickthefly/sim/connectome/sim.py decides how the fly reacts:

  where you hit          sensory neurons driven
  head                   head bristles (BM_*) + Johnston's organ (JO-*)
  thorax / abdomen       tactile neurons SNta*
  legs                   proprioceptive neurons SNpp*
  wings                  wing sensory neurons WG*

  descending neurons                   what the fly does
  DNg85 DNg48 DNg37 DNge067 DNg29 ...  jump away
  DNge122 DNge104 DNge102 DNxl114 ...  run away
  DNge074 DNge075 DNge096 DNg30 ...    kick its legs
  DNp09 (P9, forward walking)          walk
  MDN (moonwalker)                     back up
  DNa01 + DNa02, right minus left      turn around

The three touch groups are the DN types that responded most to each touch site
when the sim was probed (40 hits per site, 400 ms before vs after). Which move
each group triggers is a game choice. The giant fiber DNp01 barely reacts to
touch in this model, so it isn't used. Wing touches mostly raise DNp09.

Brain view: a front view built from the neurons' real cell-body positions. Almost
every neuron is drawn as an estimated fiber from its cell body toward the center of
its synaptic partners, colored by direction and shaded by depth. Ten of them (two
each of the giant fiber DNp01, the steering neuron DNa02, MBON01, MBON14 and KCg,
the first two of each type in the pack) are drawn instead from 21 points sampled
along their real EM skeletons, downloaded from neuPrint (MaleCNS v1.0) and cached
(sim/morphology.py); offline and uncached, they fall back to estimated fibers. The
shapes are only drawn: the simulation is point neurons either way. Neurons firing
above their calm rate glow. B toggles the big view.

Compute backends (sim/connectome/backends.py): the same LIF step runs on NumPy
(cpu, the reference), Numba (numba), PyTorch (torch-cpu, torch-cuda, torch-rocm) or
OpenGL 4.3 compute shaders (gl, any vendor's GPU); auto picks a PyTorch GPU, then
Numba, then NumPy (gl only when asked for), and a missing library falls back to NumPy
with a logged reason. On gl, a process's brains step together, up to 32 on one GL context
with the weights read once per step for all of them, and learning uploads only the
synapses it changed. numba and torch-cpu are bit-exact with cpu (same float32
operations in the same order; tested over 1000 steps in float32 and float64), so
validation, assays and save states give identical spikes on any of them. GPU sparse
kernels may add a neuron's inputs in another order, so torch-cuda/torch-rocm are only
held to a statistical tolerance. The backend that actually ran is recorded in
benchmarks, validation results, exports, save states and crash reports. The fly cap
(N) depends on it. The exe and AppImage bundle neither Numba nor PyTorch.

Video (Shift+R, --record-video): MP4 through ffmpeg (WebM by name), else a capped,
downscaled GIF, paced by the wall clock so it plays back at real speed.

Pain neurons (P): the connectome can't be given extra neurons, so this setting
listens to more of the fly's real ones and makes each hit fire more of them.
  normal  9,080 neurons   touch, heat/cold, smell and taste groups; a light touch
                          fires 30% of a region's neurons
  more   11,392 neurons   + the rest of the body's sensory neurons (campaniform
                          sensilla, hair plates, chordotonal organs...), which
                          also fire at half strength on every touch; 60%
  max    13,238 neurons   + the 1,846 ascending neurons that carry body signals
                          to the brain (the "body relay" pain part); 100%

Immortal (I): it still feels everything, but health stops at 1 and it heals when
left alone, melting and freezing stop short and wear off, and it breaks out of
spider silk.

It sees you coming: the game measures how fast each object (your tool, the
swatter, the spider, bombs) grows in the fly's view and drives its real looming
detectors LPLC2 and LC4 with that. Streaming pixels through the sim's own
photoreceptors didn't work: a looming image stayed inside the brain's random
flicker. Driven directly, LPLC2/LC4 excite the giant fiber DNp01 to 7-12x calm
(its calm max is 2.6x), and DNp01 above 4x makes the fly DODGE. Move slowly and
it won't see you.

Brain surgery (O): silence or stimulate real neuron groups with a constant
current. The inspector (click a neuron in the big view) shows its type, firing,
and strongest partners from the connectome, and can silence or stimulate its type.

Learning (a rule added on top of the fixed connectome): each tool carries a
scent, 6 of the 53 olfactory glomeruli, that the fly smells within 330 px. The
smell reaches its Kenyon cells for real (repeat scents give correlated Kenyon
cell patterns, 0.82 vs 0.38 between scents). Pain drives the PPL1 punishment
dopamine neurons (game rule), and while they fire, active Kenyon cells gain
"fear"; PAM reward neurons give "liking" the same way. When a remembered
tool's scent comes close, the fly runs or flies away. Silence PPL1 or the
Kenyon cells in surgery and it can't learn.

Arenas (E): the fan's wind drives Johnston's organ wind neurons (JO-C/E); the
pool's water drives humidity neurons (HRN) and it floats, gets wet wings and
can drown; flypaper glues any part that touches it, and the stuck fly's leg
neurons fire; the lamp's light drives photoreceptors and it is drawn to it
(phototaxis is a game rule), and the hot bulb fires heat sensors.

Outdoor arenas (3D only; outdoors.py). The 2D game stays indoors and says so.
  Open field  30 m x 30 m of ground you can walk, open sky above. Connectome: a steady
              wind drives the real JO-C/E wind neurons of each antenna, and sunlight
              drives the photoreceptors. Game rules: how the two antennae and eyes split
              that drive (by heading), the wind's push on the body, the ground, rocks
              and grass. Wind also reaches the head-touch escape DNs through JO (as in
              the fan arena), so in wind the fly keeps flying off: that is the wiring.
              Escapes last 2.5x longer outdoors (game rule) so a flight goes somewhere;
              a fly more than 26 m from you or 15 m up is LOST, and J recalls it.
  Orchard     24 fruit trees. Connectome: feeding drives the sugar-pathway taste
              neurons and the PAM reward neurons exactly as sugar does (fermented fruit
              exactly as alcohol does) and heals. Game rules: the trees and fruit, feeds
              per fruit (default 4), regrowth (default 75 s, +-25%, dormant slots wait
              for room), the per-tree cap (default 4), the fly flying to the nearest
              ripe fruit, one fly per fruit, landing. Its own escapes override the trip.
              Several flies only notice each other through looming and touch; nothing
              about competing is scripted.
  Alcohol's scent is the real fermentation glomeruli DM1/DM2/DP1m, and DM2/DP1m are
  also two of the zapper's five scent glomeruli, so mushroom-body training on one
  partly generalises to the other.

Tracking: every neuron's spikes are counted into its group each 5 ms step. Each
group's rate is compared with its own calm baseline, a 10 s average taken only
while nothing has touched the fly for 2 s, so hits don't inflate "normal".

Pain: the adult connectome has no neurons annotated as nociceptors, so the PAIN
index is a game estimate built from real signals, not a measurement of what the
fly feels. It adds these, capped at 100:
  55%  touch overload  all touch neurons above calm, saturating at +35 spikes/s
  25%  heat / cold     hot cells TRN_VP2 or cold cells TRN_VP3 above calm, +30
  55%  chemical        smell ORNs and taste neurons above calm, +35
  20%  DN alarm        all descending neurons more than 6% above calm, saturating
                       at +36%, held at its peak for ~1 s
  30%  body relay      (max pain setting only) ascending neurons above calm, the
                       same way
The blowtorch drives every touch and heat neuron at full strength, which pins it.

More ways to kill it (game rules around real sensory input):
  zapper   every touch neuron plus current into a random 30% of all neurons for 40 ms
  freeze   cold cells TRN_VP3 and a little touch; cooling damps the brain; frozen solid,
           then any hit shatters it
  spider   hunts it; bites hit body and leg touch neurons, venom paralyses and damps the
           brain, two bites wrap it in silk

Reward: sugar makes the fly walk over and eat (game rule). Eating drives taste
neurons and the PAM dopaminergic neurons (316), which signal reward in flies. In
this sim taste input alone doesn't reach PAM, so sugar drives them directly, as in
PAM activation experiments. The REWARD meter is PAM activity above calm.

Alcohol (-): a droplet of fermented fruit the fly walks over and sips. Connectome:
drinking drives the sugar-pathway taste neurons and the PAM reward neurons, as sugar
does, and the droplet's smell drives the real ORNs of the fermentation glomeruli DM1,
DM2 and DP1m. Game rule: the drunkenness. Each sip raises an inebriation level (0..1)
that decays over ~45 s of game time, and the game scales tremors, a stumbling gait,
wobbly flight and a delayed escape reflex by it. The simulated neurons are not
intoxicated: ethanol's pharmacology isn't modelled, only the body's movement is
degraded. The HUD card, the REAL/RULE tags and Lab > Model Assumptions all say so.

Flight: DNg02 (29 wing-power descending neurons). It takes off when DNg02 fires
above 1.58x calm, and flies away when the head-touch escape group fires. Flight
speed follows DNg02. The flight path itself is a game rule.

Brake cleaner: the spray drives the fly's smell neurons (ORN_*, 2,635) and taste
neurons (leg, labellar and pharyngeal GRNs) at full strength. Solvents like this
depress nervous systems, so as the fly dissolves an inhibitory current grows on
every neuron (game rule; its strength is not measured) and activity fades before
it melts into a puddle.

Death: health runs out from hits. A sim can't die on its own, so on death the
tonic drive is cancelled with an inhibitory ramp over 1.5 s and the global gain
controller is frozen. The autopsy compares each population's last 2 s alive
with its calm baseline.

Also game rules, not the connectome: the ragdoll physics, standing back up,
the direction of jumps and runs (away from the last hit), being stunned, damage.

Antennal grooming (GROOM): the fan's wind drives the antennal JO-C/E neurons, which
in the connectome excite aDN1/aDN2 (DNg62, DNge078; "Hampel 2015: aDN1/aDN2" in the
dataset). While JO-C/E fire, aDN above 4x calm (never above 3.6x over 60 s calm) logs
GROOM. Validated (x4.9 vs x0.85). The fly doesn't move to groom: aDN activation barely
reaches its front-leg motor neurons in this sim (x1.13), which fails validation.

Proboscis (PROBOSCIS): eating sugar also drives the 30 sugar-pathway taste neurons,
chosen from their wiring to the sugar SEL projection neurons (assays.py). When MN9,
a proboscis motor neuron, fires 1.5x its pre-meal rate one second into eating, the
proboscis comes out (drawn by the game). Sugar-pathway neurons -> MN9 is validated
(x2.1 vs x1.25 for bitter-pathway neurons).

Real vs rule (REACTION_SOURCE): reactions triggered by live descending-neuron firing
are tagged REAL; ones the game decides (eating, the lamp, memory-driven avoidance,
silk, death, duel hits, flying to fruit, getting lost, recall, LUNGE, SLEEP) are tagged RULE. The movement itself
is always game physics.

Courtship song (SONG). CONNECTOME: the male-specific descending neuron pIP10 excites the two ps1 wing motor
neurons, through VNC interneurons (no direct synapses): validated, x1.93 vs x0.94. P1 (the pC1 types annotated
"Cachero 2010: pMP-e; Yu 2010: pMP4") sends pIP10 1,503 synapses, but P1 -> ps1 is too weak (x1.43, FAIL). SONG
reads ps1 averaged over ~1 s (two neurons are noisy). GAME RULE: the threshold, and the synthesized pulse-song buzz
(220 Hz pulses every 35 ms; Settings > Brain > Courtship song buzz). Nothing in play drives pIP10: stimulate it in
brain surgery (strong wing hits can also fire ps1 past the threshold now and then).

Aggression (LUNGE), 2D and 3D. CONNECTOME: the readout is AVLP727m ("Asahina 2014: TK-FruM") plus the pC1 cluster,
P1 included; flies notice each other only through looming and touch. GAME RULE: with 2+ flies, that readout above
THRESH["aggression"] throws the fly at the nearest one (Settings > Brain > Aggression lunges). In testing, touch and
looming took it to ~1.5x at most, so lunges need P1 stimulated in surgery (~9x; AVLP727m alone only ~1.3x).

Optomotor. CONNECTOME: T4a/T5a (front-to-back) and T4b/T5b (back-to-front) and everything after them: rightward
wide-field rotation (T4a/T5a_R + T4b/T5b_L) excites DNa01_R/DNa02_R, x2.72 vs x0.80 for matched optic-lobe neurons,
while DNa01_L/02_L stay near calm (x1.09): validated. GAME RULE: the EMD stage (outdoors.emd_motion_drive,
assays.emd_stage) that turns a rotation into T4/T5 current. Lab only: the game loop doesn't feed it the fly's view.

Thermo arena (2D and 3D). GAME RULE: a floor from 15 C (left) to 35 C (right); distance from the middle drives the
cold TRN_VP3 or hot TRN_VP2 antennal neurons (thermo_gradient), and the extremes hurt it on the floor. CONNECTOME:
what that drive reaches; TRN_VP2 -> VP2 PNs (x2.00) and TRN_VP3 -> VP3 PNs (x1.86) are validated as one-synapse
activation, not as avoidance. HEAT, COLD and CO2 log the VP2, VP3 and V glomerulus PNs (thresholds in THRESH); the
fly doesn't move toward comfort or away from CO2.

Day/night and sleep (3D outdoors). GAME RULE: the cycle (Settings > Brain > Day/night cycle, or Lab
outdoor.day_s; off by default), daylight driving the morning clock neurons l-LNv/s-LNv directly, the scene darkening,
and SLEEP: dorsal fan-shaped body FB6/FB7 above THRESH["sleep"] stops spontaneous take-off and walking for 2 s.
CONNECTOME: the photoreceptors, the LNvs and everything downstream. A full simulated day never took dFB past 1.46x,
so the fly doesn't sleep on its own; stimulate FB6/FB7 in surgery.

Fly tournament and fly racing (3.0 day 4; Esc > Fly arcade). CONNECTOME: in a duel both sides are brains: turning is DNa01/02 right minus
left, shooting DNp35/DNpe052 above THRESH["fire"], the dodge DNp01 above THRESH["escape"], the run the body-touch group; in a race the
speed is DNp09's firing against one fixed reference rate (speed rule version 2; version 1 used each fly's own warm-up baseline). A fly's individuality (per-neuron gains from its seed) and its personality card, MEASURED from
its brain (looming latency to its dodge threshold, sugar -> MN9 ratio, DNa01/02 right/left ratio), are its own. GAME RULE: the arena, the
blaster, the pairing, the tie-break, the track, the lures, the odds (softmax of a form score) and the points (in-game points only: no
money, nothing to buy). MODEL PREDICTION: who wins or finishes first, which neurons fired before a winner's landed shots, whether
personality or individuality predicts the result (headless --tournament N, --race; the criteria are written in lab/tournament.py and
lab/racing.py before any run).

Network science (Lab; --netsci). CONNECTOME only: degrees, reciprocity, 3-node motifs against a degree-preserving null, rich club,
communities and per-region summaries computed from the pack's synapse counts, cached with a checksum; the analysis choices are GAME RULE.

Sleep deprivation assay (Lab; --sleep-deprivation). GAME RULE: sleep pressure (rises awake, falls asleep), the disturbances, what counts as
sleep and the current pressure becomes on FB6/FB7. CONNECTOME: the dFB firing that current produces (the SLEEP readout), daylight on the
photoreceptors and clock neurons. MODEL PREDICTION: the rebound sleep, which the pressure rule makes expected.

Sensitivity analysis (Lab; --sensitivity). Each LIF parameter and the synapse threshold varied across a documented range, every validated
behavior re-run with validation's own criteria: a heatmap. Analysis only: no default changes. Every cell is a MODEL PREDICTION.

Behavior rigs (3.0 day 5; Lab > Behavior rigs, --rig, --rig-assay, protocols/rig_*.yaml; game/rigs.py, lab/rigassay.py). CONNECTOME: yaw is read
from DNa01 + DNa02 right minus left and walking from DNp09 (against the race's fixed reference rate); wide-field rotation reaches the steering neurons
through T4/T5 (validated: optomotor_turning). GAME RULE: how steering Hz becomes a yaw rate (the duel's: 1.5 Hz dead zone, 0.24 rad/s per Hz, at most
2.1 rad/s), the EMD stage in front of T4/T5, LC10 tracking of a stripe or bar on the side it is on (the duel's rule; one stripe at a time in Buridan's
arena), the platform and its reflecting edge, the ball and the VR, the four-field arena, its sharp odor quadrants and the preference index. MODEL
PREDICTION: what the fly does. Criteria were committed before any held-out run (seeds 1000-1009, with an --individuality off control): the tethered
rig, the ball (the optomotor response, closed-loop bar fixation) and Buridan's paradigm (stripe deviation about 3 degrees against 42 without stripes,
far tighter than real flies', docs/rigs.md) PASS; the four-field arena FAILS (no preference for the odor: nothing connects an odor to steering or
walking in this model).

Guided mini-papers (3.0 day 5; Esc > Mini-papers, Lab > Mini-papers, --minipaper; lab/minipapers.py). Lecture protocols (lab/classroom.py) whose
experiments are validation tests (or the Buridan rig): you state a hypothesis, run it, see the plot, and read your result (MODEL PREDICTION) next to what
the paper itself states (LITERATURE: direction and any numbers the paper gives, cited, with how much of it was read: abstract or full text). Where the
model fails a paper (the aDN to leg motor neurons, the larva rolling pair) the page says so and why.

What's New in 3.0 (3.0 day 5; ui/whatsnew_ui.py). One screen, once, after an upgrade: [first_run] whatsnew_3_0_seen in config.toml (missing = not seen; a
fresh install starts seen). No simulation behavior.

Validation (validation.py): which published results this sim reproduces, on held-out seeds with pass criteria fixed
beforehand. Pass: looming -> giant fiber, sugar -> MN9, antennal touch -> aDN, T-maze conditioning, pIP10 -> ps1,
bitter GRNs -> DNg28, CO2 ORNs -> V PNs, TRN_VP2 -> VP2 PNs, TRN_VP3 -> VP3 PNs, optomotor (T4/T5 -> DNa_R),
cVA: DA1 ORNs -> DA1 PNs -> lateral horn / aSP (2.10).
Fail: MDN -> leg motor neurons (MDN's targets send them nearly balanced excitation and inhibition, 27,560 vs 24,186
synapses), aDN -> front-leg motor neurons, P1 -> ps1 (too weak), the Seeds et al. 2014 grooming hierarchy, and no
E-PG head-direction bump forms, neither from a driven wedge nor (2.7) from the open field's steady wind. 2.10 adds three
fails: foreleg pheromone GRNs -> P1 (too weak), mushroom body extinction and second-order conditioning. The
pathway tests on CO2, bitter, hot and cold are one- or two-synapse activation, not avoidance behavior.
Real-science cards in Play mode come only from passing tests; the BACK UP reaction (MDN) is a game rule.

Assays and challenges (assays.py, challenges.py): T-maze conditioning (the choice at
the fork and each odor's glomeruli are game rules), looming escape (uses the game's
looming transduction, so part of its speed dependence is a rule) and sugar response
(dose = share of sugar-pathway neurons driven). Lab mode runs them over many seeds
with statistics and same-seed controls for surgery (labjobs.py, labstats.py).

Connectome robustness and research findings (wiring.py, robustness.py, labwiring.py):
  - Synapse threshold sweep: pruning connections with fewer than 10 synapses drops 73.6%
    of all connections and 12,234 neurons' entire input, yet all 4 validated behaviors
    survive.
  - Predicted transmitter sign flips: flipping the least confident neurotransmitter
    predictions leaves looming -> giant fiber (10.9x) and antennal -> aDN (4.8x) intact,
    while sugar -> MN9 fails in 100% of trials (ratio drops from 2.22 to 1.06).
  - Critical path (looming): single-group silencing reveals LC4 (-50%) and LPLC2 (-43%)
    carry nearly all drive; silencing any other visual group has near-zero effect.

Neural clamp (clamp.py, labclamp.py): records reference spike trains from a calibrated
run and replays forced spikes into an altered connectome (lesion, threshold, sign flips)
to isolate structural wiring changes from sensory feedback variations. Explicit notice:
dynamic clamping overrides intrinsic membrane potential dynamics and severs closed-loop
sensorimotor feedback loops (e.g. proprioception and visual flow).

Connectome diff mode (diffmode.py, labdiff.py): runs two flies (Fly A reference vs
Fly B perturbed) from identical initial states, seeds, and sensory inputs in lockstep.
Visualizes activity divergence region-by-region in real-time with an autopsy-style
diverging bar chart and a divergence timeline showing the precise moment trajectories split.

Hemifield lesion (lesions.py, surgery): one-click silencing of unilateral visual pathways
(LC10, LPLC2, LC4, LPTC, VS, HS) or an entire hemibrain. Validated consequences: failure
to dodge looming stimuli approaching from the blind side, asymmetric steering biases, and
broken 1v1 duel tracking when targets cross into the blind hemifield. Reported strictly
as a connectome wiring ablation outcome, not a physical trauma model.

Global inhibition block / Picrotoxin (wiring.py, robustness.py, labwiring.py):
a 0-100% severity slider scales inhibitory synaptic weights (inhibition_scale = 1 - severity).
Disinhibition triggers emergent, non-scripted runaway firing (>30 Hz brain-wide mean) as an
unconstrained property of the recurrent network. Reports before/after firing rate distributions.
Convulsion animations and severity levels are game-level rules; the runaway activity is a
direct connectome manipulation outcome.

cVA pheromone tool (2.10; mouse wheel or toolbar). CONNECTOME: the puff drives the fly's DA1 olfactory receptor
neurons (ORN_DA1, 204; the Or67d cVA sensors, Kurtovic et al. 2007), which excite the DA1 projection neurons
(DA1_lPN, DA1_vPN, M_lvPNm43/45: x2.58 vs x0.80, validated), which excite their lateral horn / aSP targets
(LHAV4a4, LHAV4c1, LH008m = aSP-f: x2.22 vs x0.96, validated). Nothing reads a cVA behavior from them, and nothing
ties cVA to aggression beyond the wiring. GAME RULE: the puff's visible cloud, reach (250 px, 2.5 m in 3D) and strength.

Decoy female (2.10; mouse wheel or toolbar). GAME RULE: the decoy's body (female morphology, drops to the floor; three at most), the
contact distance, and COURTSHIP, a RULE tag contact triggers without reading any neuron. CONNECTOME: contact drives
LgLG5-8, the 64 foreleg gustatory neurons MaleCNS v1.0 annotates as putative ppk23/ppk25 (prothoracic leg nerve).
A live HUD readout reports real LgLG5-8 and P1 firing during contact (REAL).
Their drive reaches P1 only weakly (x1.20 vs x0.92, FAIL); SONG is still read from ps1 as before.

Plume tracking (2.10; assays.plume_tracking_fly, headless and Python only; the open field has no plume). GAME RULE,
all of it: the plume and the navigation (surge upwind in odor, cast crosswind without) are computed from the
geometry. ORN_DM1 are driven while the fly is in odor, but their firing doesn't steer it.

Extinction and second-order conditioning (2.10; assays.extinction_fly, second_order_fly; validation mb_extinction,
mb_second_order). No rule is added for either: they run on the existing learning rule, so the brain would have to
make them through its own dopamine neurons. Neither emerges: re-exposing a trained odor without shock leaves the
T-maze choice unchanged (PI 1.00 vs 1.00), and odor B paired with a trained odor A, without shock, gains no fear
(PI 0.00 vs -0.03 unpaired). Both FAIL.

Replay files (.ktfreplay, 2.10; core/replay.py). Headless only: `--headless --protocol FILE --record-replay OUT`
records a protocol's first fly (seed, Lab parameters, surgery, the brain pack's SHA-256, every poke, drive and
surgery change at its step, and a checksum of the spikes); `--headless --replay FILE --out DIR` runs it again. On
NumPy, Numba and torch-cpu the spikes must match exactly; on GPU backends a replay is reported, not judged. Another
brain pack is refused. The windowed game doesn't record or play replays yet.

Localization (2.10; core/i18n.py, data/locales/). JSON catalogs with English fallback; Settings > Accessibility >
Language. So far the pause menu and the Settings labels go through it; German is machine-translated and incomplete.
Cell types, gene names and citations are never translated.

Real-science popups (Settings > Brain) are OFF by default in Play and Lab from 2.10; configs from before migrate to
off once, and turning them on afterwards sticks.

Drosophila larva connectome (2.11; Winding et al. 2023, Science 379:eadd9330; docs/larva.md).
Selected with --brain adult|larva and Settings > Brain > Brain (default adult). Headless only in 2.11: the windowed
game runs the adult brain (playable_brain) because LarvaBody does not implement the Fly interface yet.
- CONNECTOME: the brain+SEZ connectome (2,952 neurons, 352,611 synapses) from Data S1, built locally, not shipped
  (no license stated). Groups by exact annotation (larva_loader.larva_groups): touch -> mechano-Ch / mechano-II/III
  ascending neurons, blowtorch -> nociceptive ascending neurons ('noci', incl. A00c; the class IV md sensory neurons
  are not in the dataset), sugar -> gustatory, cold -> thermo-cold. The pain meter reads the 'noci' group.
- GAME RULE: transmitter signs (Data S1 has none: LN and MBON inhibitory, the rest excitatory), soma layout, the
  segmented body, crawling, head casting and rolling. Neither larva validation test passes, so ROLL is a rule.

Tool loadouts (2.13; core/loadout.py, ui/loadout_ui.py, docs/loadouts.md). GAME RULE, all of it, and none of it read by
any neuron: the hotbar (keys 1-9 and 0, wheel, gamepad; - and = turn the page of a loadout longer than 10), the presets (Base,
Chaos, Chemist, Lab, All, Pet, custom), the loadout editor (Q), the tool wheel (hold `), the hand always being in, the laser
only in Lab mode, and larva mode hiding tools with no larval sensory mapping. A loadout only decides which tools are one key
away; validation and replays don't depend on it. Every tool's ToolInfo says which neurons it drives and whether that is
CONNECTOME or GAME RULE, and its `probes` are what the playthrough bot checks fire.
- fruit tool. CONNECTOME: exactly what sugar drives, the sugar-pathway taste neurons and the PAM reward neurons. GAME RULE: the
  item, the fly walking to it and eating it (the same rules as sugar; it even smells like sugar).

First-launch tutorial, self-test, bug report (2.13; ui/tutorial.py, core/selftest.py, core/bugreport.py). Interface and
diagnostics; nothing touches the simulation. The self-test only looks (it never installs or configures anything) and the bug
report never sends anything: it hands a URL to the player's own browser when asked.

Playthrough bot (2.13; lab/playthrough.py). Drives the real games headless; asserts, per tool, that the neurons the tool
documents fire above their own calm baseline. It changes no rule and tunes nothing.

Neurodex (3.0; core/neurodex.py, ui/neurodex_ui.py, data/neurodex_facts.yaml, docs/neurodex.md). A collectible encyclopedia of cell
types. CONNECTOME: every number in an entry, read from the dataset through the brain pack: neuron count, superclass, regions (the
game's coarse regions, derived from class and soma neuromere), the predicted transmitter and the dataset's own confidence, and the
strongest input and output partner types by synapse count. GAME RULE: "discovered" (over 150 ms windows, 3 checks in a row, the type's mean
firing is at least 6 spikes/s and 3x its own calm rate, and its spike count is a one-sided Poisson event below alpha ~1.2e-11 at the
calm rate; after 5 s of settling), the collection, its progress per region, and the "by stimulation" / "at rest" / "in play" tags.
On the real pack a calm fly would still discover 25-35 sensory types a minute through correlated spontaneous bursts; since 3.0 Day 2
a calm, untouched fly discovers nothing (a design decision, docs/neurodex.md), and "at rest" survives only on old saved entries. LITERATURE: the one-line fact and its citation, for about 30 curated types only; each was checked
against the paper it cites. Saved per player next to the training memory; the larva has its own list. Never runs in --validate,
assays, protocols or tests.
Neuron of the Day (3.0; core/neuron_of_day.py). GAME RULE: a launch card that picks one curated Neurodex type by date and offers Try it
(the Lab laser or brain surgery on that type). Its own setting (default on), separate from the real-science popups. The fact on it is
LITERATURE.
Kill cam (3.0; core/killcam.py). CONNECTOME: the replay is each neuron's own simulated firing rate for the last 6 s of the brain's
life, and the highlighted neurons are the ones whose rate rose most. GAME RULE: the offer, the window, the slow motion, the count, and
death itself (so the moment the replay ends). Nothing feeds back into the simulation.
Share codes and experiment bundles (3.0; core/sharecode.py, ui/share_ui.py, lab/bundle.py). GAME RULE, both: containers for settings
and results the game already has; they add nothing to the simulation. A bundle rerun is judged bit-exact on the CPU backends and,
on a GPU backend, statistically by a rule fixed before any rerun (lab/bundle.py).

Genetic toolkit (3.0 day 2; lab/genetics.py, data/driver_lines.yaml, docs/genetics.md). LITERATURE: which cell types a split-GAL4
line labels, and its expression-quality score, copied from Meissner et al. 2025 (eLife, CC BY 4.0; 2,667 adult lines). CONNECTOME: how
many neurons carry those types. GAME RULE: a cell-type name is matched to a MaleCNS type only when spelled identically (no fuzzy
matching); `line:SS00727` is a neuron spec that surgery, the laser, recordings, protocols and thermogenetics accept. The off-target
note is only the paper's own score. No GAL4 (non-split) mapping was found that could be redistributed, so none is offered.

Thermogenetics (3.0 day 2; lab/thermogenetics.py, docs/thermogenetics.md). LITERATURE: TrpA1 turns on near 25 C (Pulver et al. 2009);
shibire-ts blocks synaptic transmission, paralysing within 2 min at 30 C and recovering in about 1 min (Kitamoto 2001). GAME RULE: the
full-on ends of the curves (29 C, 30 C, a 28 C start for shibire-ts), the kinetics (TrpA1 1 s; shibire-ts 40 s on, 20 s off), the current
sizes (the validation drive 0.5; surgery's silencing -0.6), the thermo arena's 15-35 C map, the Lab temperature slider. MODEL: only the
expressing neurons respond (no Q10 anywhere), and shibire-ts silences the neuron rather than blocking its terminal. Assay thermo_escape:
thermogenetic activation of DNp01, escape rate vs temperature, against a no-expression control (criteria in the module, fixed first).

Virtual patch clamp (3.0 day 2; lab/patchclamp.py, docs/patchclamp.md). MODEL: a point-neuron LIF unit, NOT electrophysiology. The
potential is in model units (threshold 1.0, reset 0.0) and never millivolts; every neuron is the same unit, so the isolated I-F curve is
the same for all of them. CONNECTOME: embedded mode keeps the real synaptic input around the neuron. GAME RULE: the step protocol,
hold current and I-F amplitudes. Exports CSV and NWB (in model units, not a volts series). Lab > Patch clamp, or PATCH in the inspector.

Simulated calcium imaging (3.0 day 2; lab/imaging.py, docs/imaging.md). MODEL: spikes convolved with a two-exponential GCaMP kernel,
averaged over an ROI, with Poisson shot noise; dF/F against a running mean (30 s by default, a setting) of the neuron's own fluorescence. LITERATURE: kernel
speeds from Chen et al. 2013 (GCaMP6s/6f: Supplementary Table 3, mouse V1 in vivo, 1 action potential) and Zhang et al. 2023
(jGCaMP8m), each checked in the paper (3.0 day 2 review); the UI gives the source of each. GAME RULE: dF/F per spike, photon
budget, ROI sets, colors (Settings > Accessibility palettes; Reduced flashing smooths the view). The brain view's Imaging mode
colors neurons by dF/F (its firing counters and legend give way to an IMAGING (MODEL) label). Exports CSV, NWB (RoiResponseSeries, ImageSeries) and TIFF stacks.

Pharmacology (3.0 day 2; lab/pharmacology.py, docs/pharmacology.md). MODEL PREDICTION: a drug multiplies the weight of every synapse whose
presynaptic neuron is predicted to release a transmitter (picrotoxin: GABA and glutamate down; cholinergic block; glutamate-Cl block;
GABA-A agonist, GABA up to x2). CONNECTOME: the transmitter predictions and their confidence (measured, or predicted with a confidence);
the panel counts the synapses affected at each level and can leave out low-confidence predictions. GAME RULE: the dose mapping and the
confidence cut. Octopamine and dopamine are left out: those synapses are not in the simulated matrix. The simulator's slow global gain
works against any drug that changes overall synaptic strength, so effects are read soon after a drug goes on.

Predators (3.0 day 3; game/predators.py, game/predator_play.py, lab/predators.py, docs/predators.md). Tools in the Creatures category,
next to the spider: FROG (a very fast tongue), DRAGONFLY (a chase from above that takes only a fly in the air), MANTIS (a 10 cm/s creep, then
a strike). GAME RULE: all of the predators' behaviour: when they hop, creep, chase, aim and strike, how fast, how far, how many strikes
(no real animal is measured and no published value is used). CONNECTOME: detection goes only through the existing looming transduction
(LPLC2/LC4 -> DNp01 -> DODGE), so "sneak up slowly and it won't notice" is a consequence of the looming threshold (a game rule), and a capture
fires the real touch neurons by body part (head BM_*/JO-*, body SNta*, legs SNpp*, wings WG*). Assay predator_escape: escape probability by
predator over seeds with a Wilson 95% CI (MODEL PREDICTION; the fly does not move, so "escape" is DNp01 above the game's escape threshold
before the capture). 2D and 3D.

Rain, gusts and storms (3.0 day 3; game/weather.py, docs/weather.md; open field and orchard, 3D). CONNECTOME: what is driven: raindrop hits fire
the touch neurons by body part, wet air the humidity neurons (HRN), a gust the wind neurons JO-C/E through the existing wind
transduction, lightning the photoreceptors. GAME RULE: everything else (the hit rate, strength and part table, when wings count as wet, the
gusts, the lightning's timing, the storm preset, the darkening). Lab parameters weather.rain, weather.gust_hz, weather.storm (all off by
default, so every existing outdoor result is unchanged). Reduced flashing: the screen's lightning is one slow swell, never a flash.

Kitchen arena (3.0 day 3; game/kitchen.py, docs/kitchen.md; 3D only, E cycles to it; the counter is the floor). CONNECTOME: the fruit bowl
is the orchard's feeding (taste + PAM reward neurons), the vinegar trap's scent is the fermentation glomeruli DM1/DM2/DP1m (and the fly flies to
the jar only if its own DM1/DM2/DP1m firing is far above its calm rate), the sink is the pool's water (humidity neurons), the burner is the
lamp's heat (heat sensors), the cook's swat is seen as looming. GAME RULE: the layout, the trap physics (hover over the mouth, fall in, stuck,
drowning), the basin, the burner, the cook (10-20 s between swats, a 0.9 s steady swing at where a fly was) and every number.

Microphone to Johnston's organ (3.0 day 3; core/mic.py, lab/audio.py, docs/microphone.md). OPT-IN, off at every launch, a red MIC ON pill on screen
while on, nothing recorded or sent. CONNECTOME: JO-A (50) and JO-B (88) neurons. LITERATURE: A and B are the sound-sensitive groups (Kamikouchi
2009); B prefers below ~100 Hz and A higher (Ishikawa et al. 2019, citing Matsuo 2014, Patella & Wilson 2018). GAME RULE: the whole transduction (band-passes, loudness mapping, gate, equal
current per neuron). MODEL PREDICTION: the hum demo (a synthetic 200 Hz hum, steady and in 35 ms pulses): it drives P1, not the song motor neurons, and not
selectively at 35 ms, unlike real pC1 neurons (Zhou et al. 2015 measured 35-65 ms band-pass tuning): a disagreement with a measurement.

Streamer mode (3.0 day 3; core/streamer.py, core/netguard.py, docs/streamer.md). OPT-IN, off at every launch, never during tests, --validate or
any headless run. GAME RULE: all of it: viewers of a Twitch channel (read-only, anonymous, no token, no names kept) vote !tool, !arena, !surgery;
the streamer picks which commands count; a winner runs the game's own action. Every connection is shown on screen. No telemetry, ever.

Fly individuality (2.11; Kain et al. 2012, Linneweber et al. 2020).
- GAME RULE: Per-fly variation, deterministic from each fly's seed, implemented as per-neuron scaling
  so the shared weight matrix is unchanged: W_fly = D_post · W · D_pre, where D_pre and D_post are
  diagonal per-neuron gains drawn from a lognormal distribution. Sigma is 0.05 (subtle) and 0.15 (strong),
  both in Lab > Parameters. Signs of weights are strictly preserved. Not implemented on gl (logged, runs off).
  Plastic KC->MBON weights learn normally on top. Shared weight matrix structure preserves streaming
  SpMM on batched torch backends. NumPy and Numba stay bit-exact; torch-cpu only with it off. Settings > Brain > Individuality
  (off / subtle / strong, default subtle in Play). Forced OFF for --validate.
- Personality cards: short profiles per fly (temperament, steering bias, feeding drive) with their thresholds shown. CONNECTOME: the
  numbers, MEASURED from the fly's own brain (lab/tournament.measure_card). GAME RULE: the words and cut-offs. Since the 3.0 day 4
  review the game shows only measured cards (core/cards.py): a fly in play or the pet says "card not measured" until Esc > Fly arcade >
  Measure the flies in play reads it (cached by seed, individuality mode and sigma, LIF parameters and pack). Before, these were
  numbers drawn from the seed, not measured. Displayed in the focus note (F), pet mode and the neuron inspector header.

Pet mode (2.11; core/pet.py).
- New mode in Esc > Mode alongside Play and Lab. One fly (adult or larva) persists across real days,
  saved to XDG data dir / Documents\\Kick the Fly\\pet (pet.ktfsave with pet.ktfsave.bak backup).
- NO background process, daemon, service, autostart entry or timer, ever. The simulation only runs
  while the game window is open. Time advances upon next launch via a deterministic wall-clock catch-up rule.
  Clock changes: negative jumps clamped to 0, jumps > 7 days clamped to 7 days, both logged.
- GAME RULE needs feeding real neurons:
  - Hunger rises over time; scales sugar taste sensory gain and PAM reward dopamine drive. Feeding lowers it.
  - Sleep pressure rises while awake; drives dFB sleep neurons (FB6/FB7 in adult). The fly sleeps at night in
    the day/night cycle.
  - Death: off by default (immortal). Optional "real stakes" toggle allows starvation or trauma death with autopsy.

Settings, time and saves: settings live in config.toml (config.py, menu.py). The game
runs on a virtual clock (simclock.py): pause, 0.1-1x slow motion and single steps
slow the room and every brain together. Save states (savestate.py) hold every
neuron's state, the learned synapses, surgery, bodies and the seed.

    .venv/bin/python kick_the_fly.py
    python kick_the_fly.py --headless --validate          (no window)
    python kick_the_fly.py --headless --protocol smoke.yaml
"""
from __future__ import annotations

import math
import os
import random
import sys
import threading
import weakref
import time
from collections import deque
from pathlib import Path

import numpy as np
import pygame
import scipy.sparse as sp
from pygame import gfxdraw

from kickthefly.core import config
from kickthefly.core import crash
from kickthefly.core import loadout as loadout_mod
from kickthefly.ui import menu as menu_ui
from kickthefly.core import paths
from kickthefly.core import profiler as _profiler
from kickthefly.core.profiler import PROF
from kickthefly.game.predator_anim import SpiderCycle
from kickthefly.core import platform_env
from kickthefly.core.simclock import SimClock
from kickthefly.core.crash import log
from kickthefly.core.i18n import tr
from kickthefly.core.version import __version__
from kickthefly.lab.assays import P1_TYPES

W, H = 1280, 760
PLAY_W = 890                 # left part is the arena, right part the brain panel
FLOOR = 640
CEIL = 40
# the spider drops on its thread as in the 3D game (0.04 m a frame down to 0.12 m above the floor, at kick3d's 0.006 m a
# pixel), so it looms the same way in both: game rules, like its hunting
SPIDER_DROP = 0.04 / 0.006
SPIDER_DROP_TO = FLOOR - 0.12 / 0.006
BG = (9, 11, 15)
PANEL_BG = (13, 16, 22)
CARD = (20, 24, 32)
BORDER = (40, 47, 60)
LABEL = (120, 132, 150)
TEXT = (205, 212, 224)
INK = (240, 243, 248)
DIM = (70, 78, 92)
ACCENT = (86, 214, 255)
AMBER = (255, 176, 64)
RED = (235, 70, 60)
UP = (230, 103, 103)         # diverging pair (red <-> blue) with a gray midpoint
DOWN = (57, 135, 229)
MID = (56, 56, 53)
S_GOOD, S_WARN, S_CRIT = (12, 163, 12), (250, 178, 25), (208, 59, 59)
SERIES = ((57, 135, 229), (217, 89, 38), (25, 158, 112))   # blue, orange, aqua

# --- brain -------------------------------------------------------------------
SENSE = {"wind": ("JO-C", "JO-E"), "head": ("BM_", "JO-"), "body": ("SNta",), "legs": ("SNpp",), "wing": ("WG",),
         "heat": ("TRN_VP2",), "cold": ("TRN_VP3",), "humid": ("HRN_",), "smell": ("ORN_",),
         "taste": ("LgLG", "LgAG", "LB", "PhG", "claw_"), "light": ("R1-R6", "R7", "R8"),
         "loom": ("LPLC2", "LC4"),               # wind goes first so it keeps JO-C/E; head keeps the rest of JO
         "track": ("LC10",),                     # target tracking (LC10a-e): drives same-side DNa02 steering
         "small": ("LC11", "LC18", "LC21", "LC26")}   # small-object detectors: drive same-side DNp35
NOT_SENSORY = {"loom", "track", "small"}        # visual projection neurons, not sensory neurons
TOUCH = ("head", "body", "legs", "wing")
MOTOR = (  # name, types, side, label
    ("jump", ("DNg85", "DNg48", "DNg37", "DNge067", "DNg29", "DNge132"), None, "head-touch DNs  > JUMP"),
    ("run", ("DNge122", "DNge104", "DNge102", "DNxl114", "DNge182", "DNge048"), None, "body-touch DNs  > RUN"),
    ("kick", ("DNge074", "DNge075", "DNge096", "DNg30", "DNg34", "DNp38"), None, "leg-touch DNs   > KICK"),
    ("walk", ("DNp09",), None, "DNp09 (P9)      > WALK"),
    ("back", ("MDN",), None, "MDN moonwalker  > BACK UP"),
    ("turn_l", ("DNa01", "DNa02"), "L", "DNa01/02 left   > TURN"),
    ("turn_r", ("DNa01", "DNa02"), "R", "DNa01/02 right  > TURN"),
    ("fly", tuple(f"DNg02_{c}" for c in "abcdefg"), None, "DNg02 wing power> FLY"),
    ("escape", ("DNp01",), None, "DNp01 giant fiber> DODGE"),
    ("fire", ("DNp35", "DNpe052"), None, "DNp35 object DNs> SHOOT"),
)
POPS = (  # population name, superclasses
    ("photoreceptors", ("ol_sensory",)),
    ("optic lobe", ("ol_intrinsic", "visual_projection", "visual_centrifugal", "visual_projection_tbc")),
    ("central brain", ("cb_intrinsic",)),
    ("head sensory", ("cb_sensory", "cb_sensory_tbc", "sensory_descending")),
    ("body sensory", ("vnc_sensory", "vnc_sensory_tbc", "sensory_ascending", "sensory_ascending_tbc")),
    ("ascending", ("ascending_neuron",)),
    ("descending", ("descending_neuron", "descending_neuron_tbc")),
    ("nerve cord", ("vnc_intrinsic", "vnc_tbc")),
    ("motor neurons", ("vnc_motor", "cb_motor", "vnc_efferent", "cb_efferent", "efferent_ascending", "efferent_descending")),
)
# groom (aDN1/aDN2): over 60 s calm its level never passed 3.6x (two seeds: 2.96, 3.63); fan-strength wind on the antennal
# JO-C/E neurons took it to 5.2-5.6x.
# x calm baseline. Probed over 60 s with no touch the maxima were jump 2.6, run 1.8, kick 1.7; typical hits reach
# jump 4.6 (head), run 2.9-3.8 (body), kick 2.0-2.8 (legs). walk/back/turn sit between their spontaneous p99 and
# p99.9 so the fly wanders on its own every so often.
# DNg02 (29 wing-power DNs) rests at ~7 spikes/s; over 30 s of play its level never passed 1.64x (p99.9 1.60), so 1.58x
# takes off now and then.
# DNp01, the giant fiber: over 60 s calm its level never passed 2.64x; driving the looming detectors took it to 7-12x.
# 2.9 readouts, probed the same way on seeds 21 and 33 (exploration seeds). song is ps1's level averaged over ~1 s:
# over 120 s calm it never passed 1.61x, wing touch took it to 1.86x, and pIP10 stimulated in brain surgery held it at
# a median 1.78x on one fly and 2.85x on the other, so 1.8x sings with pIP10 on and now and then under wing hits. co2 (V PNs),
# hot_pn (VP2 PNs) and cold_pn (VP3 PNs) never passed 1.44x calm; brake cleaner's smell took co2 to 2.5x, full heat
# hot_pn to 2.5x (half heat 1.75x), full cold cold_pn to 2.05x. aggression (AVLP727m + pC1) and sleep (FB6/FB7) never
# passed 1.57x calm or under touch, looming, heat, cold or light; stimulating P1 in surgery takes aggression to ~9x
# (AVLP727m alone only ~1.3x: 5 of its 161 neurons), FB6/FB7 takes sleep to ~6.6x.
THRESH = {"groom": 4.0, "jump": 3.0, "run": 2.4, "kick": 2.0, "walk": 3.0, "back": 3.8, "turn": 2.1, "fly": 1.58, "escape": 4.0, "fire": 3.0,
          "song": 1.8, "aggression": 2.0, "sleep": 2.0, "co2": 2.0, "hot_pn": 1.6, "cold_pn": 1.7}
PAIN_WEIGHTS = np.array([0.55, 0.25, 0.55, 0.20, 0.30])   # touch, thermal, chemical, DN alarm, body relay; cap 100
PAIN_LEVELS = (  # name, share of a region's neurons a light touch recruits, how hard the rest of the body's sensors join
    ("normal", 0.3, 0.0), ("more", 0.6, 0.5), ("max", 1.0, 1.0),
)
SURGERY_CURRENT = {-1: -0.6, 0: 0.0, 1: 0.12}   # x ext_gain 4: silenced -2.4 per step (beats any touch), stimulated +0.48
TOOL_NAMES = ("hand", "flick", "swatter", "bomb", "torch", "cleaner", "zapper", "freeze", "spider", "sugar", "alcohol", "laser", "cva", "decoy", "fruit", "frog", "dragonfly", "mantis")
STIM_AMP = 0.5              # x ext_gain 4 = 2.0 per step: a driven neuron fires every refractory cycle
HIST = 1500                  # history samples, one per 20 ms = 30 s
CALM_STEPS = 400             # 2 s without a touch before the baseline learns again


def get_max_flies(backend: str | None = None, brain: str = "adult") -> int:
    """How many flies N may spawn, dynamically adapted to the active backend (measured in docs/performance.md).

    NumPy (cpu) and torch-cpu: 16 (adult) / 64 (larva); brains share Python's GIL.
    Numba: 16 to 32 (adult) / 128 (larva) (releases GIL).
    PyTorch GPU (torch-cuda, torch-rocm): 32/64 (adult) / 128 (larva).
    gl: 32 (adult) / 128 (larva).
    Every spawn also needs free memory (BRAIN_MB each), checked when you press N."""
    brain = str(brain).lower()
    if brain == "larva":
        backend_str = str(backend).lower() if backend else "cpu"
        if backend_str in ("numba", "torch-cuda", "torch-rocm", "gl"):
            return 128
        return 64
    if backend is None or backend == "auto":
        from kickthefly.sim.connectome.backends import detect_available_backends
        avail = detect_available_backends()
        from kickthefly.sim.connectome.backends import auto_chain
        backend = auto_chain(avail)[0]          # the same chain create_backend's 'auto' walks, under the same policy
    backend = str(backend).lower()
    if backend == "gl":
        return 32                    # a 33rd fly would start a second group, streaming the weights twice per step
    if backend in ("torch-cuda", "torch-rocm"):
        return 64 if os.environ.get("KICK_THE_FLY_EXPANDED_SWARM") else 32
    if backend == "numba":
        return int(min(32, max(16, os.cpu_count() or 16)))
    return 16


def save_image(surf: "pygame.Surface", path) -> None:
    """pygame.image.save, falling back to Pillow when this pygame was built without PNG/JPEG support (some distro
    packages are)."""
    try:
        pygame.image.save(surf, str(path))
    except (NotImplementedError, pygame.error):
        from PIL import Image
        Image.frombytes("RGB", surf.get_size(), pygame.image.tobytes(surf, "RGB")).save(str(path))


def drop_item(items: list, item) -> None:
    """Remove `item` from `items` by identity. These lists hold dicts of numpy arrays, and
    list.remove() compares with ==, which raises "truth value of an array is ambiguous" as
    soon as it has to compare against an earlier element."""
    for i, other in enumerate(items):
        if other is item:
            del items[i]
            return


MAX_FLIES = get_max_flies()
BRAIN_MB = 350                # resident memory per extra brain (310-390 MB measured; its own weight matrices)
FLY_TOUCH_RADIUS = 40.0       # how close two flies' thoraxes get before they bump (game rule, not a measurement)


class Brain:
    """Runs the LIF sim in real time on its own thread; hits queue sensory current, the game reads group rates."""

    def __init__(self, g, sim, seed: int = 0):
        self.sim, self.n = sim, g.n
        self.dt = sim.p.dt_ms / 1000.0
        types = g.type.astype(str)
        sc = g.superclass.astype(str)
        sides = np.array([(i or "")[-2:] for i in g.instance])
        is_dn = sc == "descending_neuron"
        self.rng = np.random.default_rng(seed)
        self.seed = seed

        # detail groups (touch sites + reaction DNs) are disjoint; population groups cover the whole connectome
        self.det_id = np.full(g.n, -1, np.int32)
        self.pop_id = np.full(g.n, -1, np.int32)
        self.names: list[str] = []
        sizes: list[int] = []
        self.sense: dict[tuple[str, str | None], np.ndarray] = {}

        def add_detail(name, m):
            m = m & (self.det_id < 0)
            self.det_id[m] = len(self.names)
            self.names.append(name)
            sizes.append(int(m.sum()))
            return m

        for region, prefixes in SENSE.items():
            m = np.zeros(g.n, bool)
            for p in prefixes:
                m |= np.char.startswith(types, p)
            m = add_detail(region, m if region in NOT_SENSORY else m & (np.char.find(sc, "sensory") >= 0))
            self.sense[(region, None)] = np.flatnonzero(m)
            for s in "LR":
                rows = np.flatnonzero(m & (sides == f"_{s}"))
                self.sense[(region, s)] = rows if len(rows) > 0.2 * m.sum() else np.flatnonzero(m)
        for name, tys, side, _ in MOTOR:
            m = is_dn & np.isin(types, tys)
            if side:
                m &= sides == f"_{side}"
            add_detail(name, m)
        # readouts for validated behaviors (validation.py): antennal grooming command neurons aDN1/aDN2 (the dataset's
        # DNg62 and DNge078, "Hampel 2015: aDN1/aDN2") and the proboscis motor neuron MN9
        add_detail("groom", is_dn & np.isin(types, ("DNg62", "DNge078")))
        add_detail("proboscis", np.isin(types, ("MN9",)))
        # 2.9 readouts. song: the ps1 wing motor neurons that pIP10 and P1 drive (validated). aggression: the
        # tachykinin/fru neurons AVLP727m ("Asahina 2014: TK-FruM") and the male pC1 cluster, P1 included. sleep: the
        # dorsal fan-shaped body layers FB6/FB7. co2, hot_pn, cold_pn: the second-order neurons of the CO2 ORNs and the
        # hot and cold antennal neurons (V, VP2 and VP3 projection neurons; each pathway validated).
        add_detail("song", (sc == "vnc_motor") & (types == "ps1 MN"))
        add_detail("aggression", np.isin(types, ("AVLP727m",)) | (np.char.startswith(types, "pC1") & (sc == "cb_intrinsic")))
        # pet mode's sleep-pressure coupling (core/pet.py) drives these, so they are also a stimulus site
        self.sense[("sleep", None)] = np.flatnonzero(
            add_detail("sleep", np.char.startswith(types, "FB6") | np.char.startswith(types, "FB7")))
        add_detail("co2", np.isin(types, ("V_ilPN", "V_l2PN")))
        add_detail("hot_pn", np.isin(types, ("VP2_adPN", "VP2_l2PN", "VP2+_adPN", "VP1m+VP2_lvPN1", "VP1m+VP2_lvPN2")))
        add_detail("cold_pn", np.isin(types, ("VP3+_l2PN", "VP3+_vPN", "VP1l+VP3_ilPN", "VP3+VP1l_ivPN", "VP5+VP3_l2PN")))
        # the morning clock neurons l-LNv and s-LNv: daylight drives them directly outdoors when the day/night cycle
        # is on (game rule: real LNvs see light through the Hofbauer-Buchner eyelet and CRY, neither modelled)
        self.sense[("clock", None)] = np.flatnonzero(add_detail("clock", np.isin(types, ("l-LNv", "s-LNv"))))
        # the rest of the body's sensory neurons (campaniform sensilla, hair plates, chordotonal organs, unnamed SN*):
        # only counted and driven when the pain setting asks for more neurons
        body_extra = add_detail("body_extra", np.isin(sc, ("vnc_sensory", "vnc_sensory_tbc", "sensory_ascending",
                                                            "sensory_ascending_tbc")))
        self.sense[("body_extra", None)] = np.flatnonzero(body_extra)
        self.recruit, self.spill, self.pain_level = 0.3, 0.0, 0
        # PAM dopaminergic neurons (316) signal reward in flies; sugar drives them directly (see the docstring)
        self.sense[("reward", None)] = np.flatnonzero(add_detail("reward", np.char.startswith(types, "PAM")))
        # PPL1 dopaminergic neurons (16) signal punishment; pain drives them (game rule), learning reads them
        self.sense[("punish", None)] = np.flatnonzero(add_detail("punish", np.char.startswith(types, "PPL1")))
        self.kc = np.flatnonzero(np.char.startswith(types, "KC"))        # Kenyon cells, the mushroom body
        self.brain_type = getattr(g, "brain_type", "adult")
        self.is_larva = (self.brain_type == "larva") or (g.n < 10000)
        self.subclass = np.asarray(getattr(g, "subclass", np.full(g.n, ""))).astype(str)
        from kickthefly.lab import assays

        if self.is_larva:
            # Annotated larval sensory and motor groups (Winding et al. 2023)
            # Class IV md nociceptors
            from kickthefly.sim.connectome.larva_loader import larva_groups
            lg = larva_groups(self.subclass)          # exact annotation tags, so 2nd-order PNs are not counted as input
            noci_idx, chord_idx, mech_idx = lg["noci"], lg["chordo"], lg["mechano"]
            # its own detail group, so the pain meter below can read the nociceptive neurons' rates
            add_detail("noci", np.isin(np.arange(g.n), noci_idx))
            gust_idx = np.flatnonzero(np.char.find(self.subclass, "gustatory-") >= 0)
            cold_idx = np.flatnonzero(np.char.find(self.subclass, "thermo-cold") >= 0)
            olf_idx = np.flatnonzero(np.char.find(self.subclass, "olfactory") >= 0)
            goro_idx = lg["goro"]

            self.noci_idx = noci_idx
            self.chord_idx = chord_idx
            self.goro_idx = goro_idx

            # Touch -> chordotonal and mechanosensory neurons
            touch_all = np.unique(np.concatenate([chord_idx, mech_idx])) if (len(chord_idx) or len(mech_idx)) else np.arange(min(12, g.n))
            self.sense[("head", None)] = chord_idx if len(chord_idx) else touch_all
            self.sense[("body", None)] = mech_idx if len(mech_idx) else touch_all
            self.sense[("legs", None)] = mech_idx if len(mech_idx) else touch_all
            self.sense[("wing", None)] = mech_idx if len(mech_idx) else touch_all
            # Blowtorch / heat -> class IV md nociceptors
            self.sense[("heat", None)] = noci_idx if len(noci_idx) else touch_all
            self.sense[("cold", None)] = cold_idx if len(cold_idx) else touch_all
            # Sugar -> gustatory neurons
            self.sense[("taste", None)] = gust_idx if len(gust_idx) else np.arange(min(10, g.n))
            self.sense[("punish", None)] = noci_idx
            self.sense[("goro", None)] = goro_idx

            for k, name in enumerate(TOOL_NAMES):
                self.sense[("scent", name)] = olf_idx[:5] if len(olf_idx) >= 5 else olf_idx
            self.sense[("scent", "player")] = olf_idx[-3:] if len(olf_idx) >= 3 else olf_idx
            self.sense[("scent", "cva")] = olf_idx[:2] if len(olf_idx) >= 2 else olf_idx
            self.sense[("pheromone", "foreleg")] = np.array([], dtype=int)
            self.lglg_indices = np.array([], dtype=int)
            self.p1_indices = np.array([], dtype=int)
            self.sense[("scent", "odor_a")] = olf_idx[:4] if len(olf_idx) >= 4 else olf_idx
            self.sense[("scent", "odor_b")] = olf_idx[4:8] if len(olf_idx) >= 8 else olf_idx
            self.sense[("scent", "odor_c")] = olf_idx[8:12] if len(olf_idx) >= 12 else olf_idx
        else:
            self.noci_idx = np.array([], dtype=int)
            self.chord_idx = np.array([], dtype=int)
            self.goro_idx = np.array([], dtype=int)
            orn = self.sense[("smell", None)]
            glom = np.array([t.split("_", 1)[1] if "_" in t else t for t in types[orn]])
            order = np.random.default_rng(11).permutation(np.unique(glom)) if len(glom) else np.array([])
            for k, name in enumerate(TOOL_NAMES):        # each tool's scent: its own 5 of the 53 olfactory glomeruli
                if name == "alcohol":
                    # Fermented fruit / ethanol odor activates canonical food attraction glomeruli (DM1, DM2, DP1m)
                    self.sense[("scent", name)] = orn[np.isin(glom, ["DM1", "DM2", "DP1m"])]
                elif name == "laser":
                    self.sense[("scent", name)] = orn[:0]   # pure optical beam: carries no odor
                elif name == "cva":
                    self.sense[("scent", name)] = orn[np.isin(glom, ["DA1"])]
                elif name == "decoy":
                    self.sense[("scent", name)] = orn[np.isin(glom, ["DA1", "VA1v"])]
                else:
                    self.sense[("scent", name)] = orn[np.isin(glom, order[k * 5:(k + 1) * 5])] if len(order) >= (k + 1) * 5 else orn[:5]
            self.sense[("scent", "cva")] = orn[np.isin(glom, ["DA1"])]
            self.sense[("pheromone", "foreleg")] = np.flatnonzero(np.isin(types, ["LgLG5", "LgLG6", "LgLG7", "LgLG8"]))
            self.sense[("scent", "player")] = orn[np.isin(glom, order[50:])] if len(order) >= 53 else orn[:3]
            from kickthefly.lab import assays
            self.lglg_indices = np.flatnonzero(np.isin(types, ["LgLG5", "LgLG6", "LgLG7", "LgLG8"]))
            self.p1_indices = np.flatnonzero(np.isin(types, assays.P1_TYPES))
            odor_order = np.random.default_rng(assays.ODOR_GLOMERULI_SEED).permutation(np.unique(glom)) if len(glom) else np.array([])
            self.sense[("scent", "odor_a")] = orn[np.isin(glom, odor_order[:6])] if len(odor_order) >= 6 else orn[:6]
            self.sense[("scent", "odor_b")] = orn[np.isin(glom, odor_order[6:12])] if len(odor_order) >= 12 else orn[:6]
            self.sense[("scent", "odor_c")] = orn[np.isin(glom, odor_order[12:18])] if len(odor_order) >= 18 else orn[:6]
        self.types, self.superclass = types, sc
        self.subclass = np.asarray(getattr(g, "subclass", np.full(g.n, ""))).astype(str)
        self.instance = np.array([i or "" for i in g.instance])
        self.override = np.zeros(g.n, np.float32)    # brain surgery: per-neuron silencing / stimulating current
        self.surgery = False
        # Experimental activation (assays, validation, protocols) is its own current, added to surgery rather than
        # replacing it: a neuron you silenced does not start firing because the experiment drives its type.
        self.drive_cur = np.zeros(g.n, np.float32)
        self.driving = False
        # 3.0 day 2: named extra currents (thermogenetics, the patch electrode): each feature owns one entry, the sum is
        # added to the drive. Nothing is added while none is set, so every existing run is bit-identical.
        self.currents: dict[str, np.ndarray] = {}
        self.probe = None               # 3.0 day 2: optional callable(brain) run after every step (the patch electrode)
        self.inject = np.zeros(g.n, np.float32)
        self.injecting = False
        self.sense[("all", None)] = np.arange(g.n)   # the zapper's shock
        self.n_det = len(self.names)
        for k, (name, classes) in enumerate(POPS):
            m = np.isin(sc, classes)
            self.pop_id[m] = k
            self.names.append(name)
            sizes.append(int(m.sum()))
        self.names.append("whole brain")
        sizes.append(g.n)

        G = len(self.names)
        self.col = {n: i for i, n in enumerate(self.names)}
        self.g_size = np.maximum(np.array(sizes, float), 1)
        self.fast = np.zeros(G)                      # Hz per neuron, EMA ~150 ms
        self.base = np.zeros(G)                      # calm baseline, Hz per neuron
        self.lglg_fast = 0.0
        self.lglg_base = 0.0
        self.p1_fast = 0.0
        self.p1_base = 0.0
        self.k_fast = 1 - math.exp(-1 / 30)
        self.k_base = 1 - math.exp(-1 / 2000)
        self.hist = np.zeros((HIST, G), np.float32)
        self.hist_n = 0                              # samples written so far
        self.last_poke = -10**9
        self._lock = threading.Lock()
        self._pending: dict[tuple[str, str | None], list] = {}
        self._cur = np.zeros(g.n, np.float32)
        self._inhib = np.full(g.n, -0.4, np.float32)  # x ext_gain 4 = -1.6 per step, far past bias + noise
        self._stop = False
        self.death_step: int | None = None
        self.sedation = 0.0                          # 0..1 share of the death inhibition, set by the game
        self.death_sample = 0
        self.death_base: np.ndarray | None = None
        self._revive = False
        self._gain_adapt = sim.p.gain_adapt
        self._gain = sim.gain
        self.steps_per_s = 0.0
        self.steps = 0
        g_assay = assays.groups(self)                # sugar- and bitter-pathway taste neurons, from the wiring
        self.sense[("sweet", None)], self.sense[("bitter", None)] = g_assay["sweet"], g_assay["bitter"]
        self.speed = 1.0                             # x real time: >1 training, <1 slow motion, 0 paused
        self.step_requests = 0                       # brain steps owed while paused (single-step)
        self.step_lock = threading.Lock()            # held for each step, so save states never see half a step
        self.meter_reset = False
        self.memory = None                           # memory.Memory: learning on the real KC -> MBON synapses
        self.recorder = None                         # recorder.Recorder while recording spikes for export
        self.body_id = getattr(g, "body_id", None)
        self.region = np.asarray(getattr(g, "region", np.full(g.n, "unassigned"))).astype(str)
        self.stethoscope_indices: np.ndarray | None = None
        self.stethoscope_spikes = 0

    def set_stethoscope_indices(self, indices: np.ndarray | None) -> None:
        with self._lock:
            self.stethoscope_indices = indices
            self.stethoscope_spikes = 0

    def take_stethoscope_spikes(self) -> int:
        with self._lock:
            s = self.stethoscope_spikes
            self.stethoscope_spikes = 0
            return s

    @property
    def dead(self) -> bool:
        return self.death_step is not None

    def poke(self, region: str, side: str | None, strength: float, recruit: float | None = None) -> None:
        """Drive a random share of a region's neurons; harder hits recruit more of them for longer."""
        if self.dead:
            return
        s = float(np.clip(strength, 0, 1))
        pop = self.sense[(region, side)]
        if not len(pop):                             # e.g. the laser's scent: it carries none
            return
        share = self.recruit + (1 - self.recruit) * s if recruit is None else recruit
        rows = self.rng.choice(pop, size=max(1, int(len(pop) * share)), replace=False)
        steps = 8 + int(40 * s)
        with self._lock:
            self.last_poke = self.steps
            old = self._pending.get((region, side))
            if old is None or steps > old[1]:
                self._pending[(region, side)] = [rows, steps]
        rec = self.recorder                          # a Lab recording keeps every stimulus with its timing
        if rec is not None:
            rec.log_event("stimulus", region, side or "both", s)
        if region in TOUCH and self.spill > 0:       # more pain neurons: the rest of the body's sensors fire too
            self.poke("body_extra", None, s * self.spill)

    def set_override(self, rows: np.ndarray, mode: int) -> None:
        """Brain surgery on these neurons: -1 silence, 0 leave alone, +1 stimulate."""
        self.override[rows] = SURGERY_CURRENT[mode]
        self.surgery = bool(np.any(self.override))

    def set_current(self, name: str, rows, values) -> None:
        """Give the named source `values` (a scalar or one per row) on `rows`, replacing what that source had. The
        sources add up; one source never overwrites another (unlike drive_cur, which assays own). Safe from any thread:
        the Lab tick calls it on the game thread while the brain thread steps (3.0 day 2 review)."""
        arr = np.zeros(len(self.inject), np.float32)
        arr[np.asarray(rows, np.int64)] = values
        with self._lock:
            self.currents[name] = arr
            self._sum_currents()

    def clear_current(self, name: str) -> None:
        with self._lock:
            if self.currents.pop(name, None) is not None:
                self._sum_currents()

    def _sum_currents(self) -> None:
        # Build the sum in a new array and swap it in: a step on the brain thread may be reading the old one, and zeroing it
        # in place let that step see a half-built sum (3.0 day 2 review). The array is set before the flag.
        total = np.zeros(len(self.inject), np.float32)
        for arr in self.currents.values():
            total += arr
        self.inject = total
        self.injecting = bool(np.any(total))

    def clear_overrides(self) -> None:
        self.override[:] = 0
        self.surgery = False

    def set_pain_level(self, level: int) -> None:
        _, self.recruit, self.spill = PAIN_LEVELS[level]
        self.pain_level = level

    def pain_neurons(self, level: int | None = None) -> int:
        """How many real neurons the pain index listens to at this setting."""
        level = self.pain_level if level is None else level
        names = list(TOUCH) + ["heat", "cold", "smell", "taste"] + (["body_extra"] if level >= 1 else [])
        names += ["ascending"] if level >= 2 else []
        return int(sum(self.g_size[self.col[n]] for n in names))

    def kill(self) -> None:
        with self._lock:
            self._pending.clear()
            self._gain = self.sim.gain
            self.death_base = self.base.copy()
            self.death_sample = self.hist_n
            self.death_step = self.steps
        self.sim.p.gain_adapt = 0.0

    def revive(self) -> None:
        self._revive = True

    def _step(self) -> None:
        with self._lock:
            active = list(self._pending.items())
            for key, item in active:
                item[1] -= 1
                if item[1] <= 0:
                    del self._pending[key]
        for _, (rows, _) in active:
            self._cur[rows] = STIM_AMP
        if self.death_step is not None:                  # start from any damping already on, or it would rebound
            ramp = min(1.0, max(self.sedation, (self.steps - self.death_step) / 300))
            spikes = self.sim.step(self._inhib * np.float32(ramp))
        else:
            drive = self._cur
            if self.sedation > 0:                    # solvent, cold or venom depression
                drive = drive + self._inhib * np.float32(self.sedation)
            if self.surgery:
                drive = drive + self.override
            if self.driving:
                drive = drive + self.drive_cur
            if self.injecting:
                drive = drive + self.inject
            spikes = self.sim.step(drive if (active or self.sedation > 0 or self.surgery or self.driving
                                             or self.injecting) else None)
        for _, (rows, _) in active:
            self._cur[rows] = 0

        if self.recorder is not None:
            self.recorder.push(self.steps, spikes)
        if self.probe is not None:
            try:
                self.probe(self)
            except Exception:                            # 3.0 day 2 review: on the brain thread an exception ends the loop
                log.exception("brain probe failed; removed")
                self.probe = None
        if self.stethoscope_indices is not None and len(self.stethoscope_indices) > 0:
            self.stethoscope_spikes += int(np.count_nonzero(spikes[self.stethoscope_indices]))
        on = np.flatnonzero(spikes)
        G = len(self.names)
        counts = np.zeros(G)
        d = self.det_id[on]
        counts[:self.n_det] = np.bincount(d[d >= 0], minlength=self.n_det)
        p = self.pop_id[on]
        counts[self.n_det:G - 1] = np.bincount(p[p >= 0], minlength=len(POPS))
        counts[-1] = len(on)
        inst = counts / self.g_size / self.dt
        self.fast += (inst - self.fast) * self.k_fast
        if len(self.lglg_indices) > 0:
            lglg_cnt = int(np.count_nonzero(spikes[self.lglg_indices]))
            lglg_inst = (lglg_cnt / len(self.lglg_indices)) / self.dt
            self.lglg_fast += (lglg_inst - self.lglg_fast) * self.k_fast
        if len(self.p1_indices) > 0:
            p1_cnt = int(np.count_nonzero(spikes[self.p1_indices]))
            p1_inst = (p1_cnt / len(self.p1_indices)) / self.dt
            self.p1_fast += (p1_inst - self.p1_fast) * self.k_fast
        if (self.death_step is None and self.sedation == 0 and not self.surgery and not self.driving
                and not self.injecting and self.steps - self.last_poke > CALM_STEPS):
            self.base += (self.fast - self.base) * self.k_base
            self.lglg_base += (self.lglg_fast - self.lglg_base) * self.k_base
            self.p1_base += (self.p1_fast - self.p1_base) * self.k_base
        self.steps += 1
        if self.steps % 4 == 0:
            self.hist[self.hist_n % HIST] = self.fast
            self.hist_n += 1
        if self.memory is not None and self.steps % MEMORY_STEPS == 0 and self.death_step is None:
            calm = (self.sedation == 0 and not self.surgery and not self.driving and not self.injecting
                    and self.steps - self.last_poke > CALM_STEPS)
            self.memory.step(self.sim.activity.rates(), calm, self.steps)

    def reseed(self, seed: int) -> None:
        """New random seed for this brain's noise and hit sampling (Settings > Brain > Random seed, applied on R)."""
        sim = self.sim
        with self._lock:
            self.seed = seed
            self.rng = np.random.default_rng(seed)
            sim.rng = np.random.default_rng(seed)
            sim._noise = sim.rng.standard_normal(sim.n * 16, dtype=np.float32) * np.float32(sim.p.noise_std)

    def request_steps(self, n: int) -> None:
        with self._lock:
            self.step_requests += n

    def warmup(self, steps: int = 600) -> None:
        k = self.k_base
        self.k_base = 1 - math.exp(-1 / 60)          # let the baseline settle quickly
        for _ in range(steps):
            self._step()
        self.k_base = k

    SWITCH_S = 0.0002        # 3.1.0 task 3, see start()

    def start(self) -> None:
        # The brain runs on a thread of its own so a slow frame cannot slow it, but CPython hands the GIL to a waiting thread only every
        # sys.getswitchinterval() (5 ms by default) while another thread runs Python code. A render thread that is busy for a frame then
        # starved the brain: measured with a main thread that never yields, 27 steps/s of the 200 (0.001 s gave 78, 0.0005 s 163, 0.0002 s
        # 200). Waiting costs nothing; the brain's own numeric work releases the GIL. Spike for spike the output is unchanged.
        if sys.getswitchinterval() > self.SWITCH_S:
            sys.setswitchinterval(self.SWITCH_S)
        threading.Thread(target=self._loop, name="brain", daemon=True).start()

    def stop(self) -> None:
        self._stop = True

    def _loop(self) -> None:
        t0 = time.perf_counter()
        done, last_t, last_steps = 0, t0, self.steps
        speed = self.speed
        while not self._stop:
            if self.speed != speed:                  # re-anchor the clock when the speed changes (training, slow-mo, pause)
                speed, t0, done = self.speed, time.perf_counter(), 0
            if self.step_requests > 0:               # single steps while time is paused
                with self._lock:
                    n, self.step_requests = self.step_requests, 0
                for _ in range(n):
                    with self.step_lock:
                        self._step()
                t0, done = time.perf_counter(), 0
                continue
            if self._revive:
                self._revive = False
                self.death_step = None
                self.sim.p.gain_adapt = self._gain_adapt
                self.sim.gain = self._gain
                with self.step_lock:
                    for _ in range(300):
                        self._step()
                t0, done = time.perf_counter(), 0
            now = time.perf_counter()
            if self.meter_reset:                     # a save state replaced the step count
                self.meter_reset = False
                last_t, last_steps = now, self.steps
            due = int((now - t0) / self.dt * speed)
            if due - done > 40:                      # fell behind: drop the backlog instead of racing
                done = due - 40
            if done >= due:
                time.sleep(0.001)
                continue
            with self.step_lock:
                self._step()
            done += 1
            if now - last_t > 1.0:
                self.steps_per_s = (self.steps - last_steps) / (now - last_t)
                last_t, last_steps = now, self.steps

    def level(self, name: str) -> float:
        i = self.col[name]
        return float(self.fast[i] / max(self.base[i], 2.0))

    def hz(self, name: str) -> float:
        return float(self.fast[self.col[name]])

    def lglg_level(self) -> float:
        return float(self.lglg_fast / max(self.lglg_base, 0.5))

    def p1_level(self) -> float:
        return float(self.p1_fast / max(self.p1_base, 0.5))

    def step(self) -> None:
        self._step()

    def pain_parts(self, rates: np.ndarray, base: np.ndarray) -> np.ndarray:
        """(touch overload, thermal, chemical, DN alarm, body relay) in 0..1 per row of group rates. See the docstring."""
        rates = np.atleast_2d(rates)

        if getattr(self, "is_larva", False) and "noci" in self.col:
            ni = self.col["noci"]
            noci_diff = rates[:, ni] - base[ni]
            p_val = np.clip(noci_diff / 25.0, 0.0, 1.0)
            d = self.col.get("descending")
            alarm = (rates[:, d] / max(float(base[d]), 1.0) - 1.06) / 0.3 if d is not None else np.zeros(len(rates))
            return np.clip(np.stack([p_val, p_val, p_val * 0.5, alarm, np.zeros(len(rates))], 1), 0, 1)

        def above(names, full):
            idx = [self.col[n] for n in names]
            sz = self.g_size[idx]
            return ((rates[:, idx] - base[idx]) * sz).sum(1) / sz.sum() / full

        touch = above(list(TOUCH) + (["body_extra"] if self.pain_level else []), 35.0)
        thermal = np.maximum(above(["heat"], 30.0), above(["cold"], 30.0))   # thermosensors fire ~24 spikes/s at rest
        chemical = (above(["smell", "taste"], 1.0) - 8.0) / 27.0   # 8 spikes/s deadzone: a tool's faint scent isn't pain
        d = self.col["descending"]
        alarm = (rates[:, d] / max(float(base[d]), 1.0) - 1.06) / 0.3        # 6% deadzone: resting DN flicker
        a = self.col["ascending"]                    # ascending neurons carry body signals up to the brain
        relay = (rates[:, a] / max(float(base[a]), 1.0) - 1.06) / 0.35 if self.pain_level >= 2 else np.zeros(len(rates))
        return np.clip(np.stack([touch, thermal, chemical, alarm, relay], 1), 0, 1)

    @staticmethod
    def hold_alarm(parts: np.ndarray, decay: float = 0.95) -> np.ndarray:
        """Peak-hold the DN alarm (~1 s at one sample per 20 ms): the gain controller makes the raw surge flicker."""
        out = parts.copy()
        for k in range(1, len(out)):
            out[k, 3] = max(out[k, 3], out[k - 1, 3] * decay)
        return out

    @staticmethod
    def pain_index(parts: np.ndarray) -> np.ndarray:
        return np.minimum(100.0, 100.0 * (parts @ PAIN_WEIGHTS))

    def history(self, first: int, last: int) -> np.ndarray:
        """Samples [first, last) by absolute sample index, clipped to what the ring buffer still holds."""
        first = max(first, self.hist_n - HIST, 0)
        last = min(last, self.hist_n)
        if last <= first:
            return np.zeros((0, len(self.names)), np.float32)
        return self.hist[np.arange(first, last) % HIST].copy()


# --- brain view ----------------------------------------------------------------
VIEW_X, VIEW_Y, VIEW_ZCUT = (0.0, 96000.0), (2000.0, 54000.0), 60000.0   # front view of the brain; +y is ventral
VIEW_SIZES = {"panel": (356, 193), "big": (860, 466)}


REGION_COLORS = {
    "Antennal Lobe": (255, 130, 40),
    "Mushroom Body": (255, 215, 0),
    "Central Complex": (50, 220, 100),
    "Optic Lobe": (0, 180, 255),
    "Central Brain": (160, 100, 240),
    "Gnathal (GNG)": (0, 210, 190),
    "VNC (T1)": (255, 90, 90),
    "VNC (T2)": (230, 60, 140),
    "VNC (T3)": (180, 50, 200),
    "VNC (Abdomen)": (240, 110, 180),
    "VNC (Other)": (140, 140, 190),
    "unassigned": (110, 115, 125),
}


class BrainView:
    """Front view of the brain that lights up with the live sim.

    Every neuron is a fiber from its real cell body (MaleCNS soma positions) to the synapse-weighted center of its
    partners' cell bodies, with a tuft of points there for its arbor. Neurons without a soma in the data (most sensory
    neurons) are just the tuft. Real neuron shapes aren't in the pack, so fibers are estimates. The wiring is colored by
    fiber direction (red left-right, green up-down, blue front-back) and shaded by depth, front brightest. Firing above
    a neuron's own calm rate glows: pain-sensing neurons (touch, heat, cold, chemical, and the ascending relay) glow
    hot orange, everything else cool white, and firing neurons that spiked on the latest step sparkle at their cell
    body. The nerve cord is cut off at the neck, so ascending fibers enter through it.
    """

    FIBER, ARBOR = 18, 3
    HOT = np.array([1.0, 0.34, 0.07], np.float32)
    # Accessibility palettes: (pain-sensing neurons, everything else). Blue/yellow stays distinct with red-green color
    # blindness; high contrast is magenta on white.
    PALETTES = {"default": ((1.0, 0.34, 0.07), (0.3, 0.75, 1.0)),
                "blue-yellow": ((1.0, 0.8, 0.08), (0.18, 0.42, 1.0)),
                "high-contrast": ((1.0, 0.12, 0.85), (0.92, 0.92, 0.92))}

    def __init__(self, soma: np.ndarray, W, pain_mask: np.ndarray, seed: int = 1, regions: np.ndarray | None = None, graph=None):
        rng = np.random.default_rng(seed)
        n = len(soma)
        has = ~np.isnan(soma[:, 0])
        S = np.nan_to_num(soma).astype(np.float32)
        M = abs(W).astype(np.float32)
        M = (M + M.T).tocsr()
        tot = M @ has.astype(np.float32)
        ok = tot > 0
        A = (M @ (S * has[:, None])) / np.maximum(tot, 1e-6)[:, None]
        start = np.where(has[:, None], S, A)
        d = A - start
        length = np.linalg.norm(d, axis=1)
        rand = np.abs(rng.normal(size=(n, 3))).astype(np.float32)
        dirv = np.where((length > 1)[:, None], np.abs(d) / np.maximum(length, 1)[:, None], rand / np.linalg.norm(rand, axis=1)[:, None])
        col = dirv ** 2
        col = col / col.max(axis=1, keepdims=True)                   # saturated direction color
        self.col = (0.06 + 0.94 * col).astype(np.float32)
        self.hot_mask = np.asarray(pain_mask, bool)
        self.sparkle = True
        self.set_palette("default")

        t = np.linspace(0, 1, self.FIBER, dtype=np.float32)[None, :, None]
        bend = rng.normal(size=(n, 3)).astype(np.float32) * (0.12 * length)[:, None]
        mid = (start + A) / 2 + bend
        fiber = (1 - t) ** 2 * start[:, None] + 2 * (1 - t) * t * mid[:, None] + t ** 2 * A[:, None]
        r = np.clip(0.05 * length, 400, 1600)
        arbor = A[:, None] + rng.normal(size=(n, self.ARBOR, 3)).astype(np.float32) * r[:, None, None]
        pts = np.concatenate([fiber, arbor], 1)                      # (n, samples, 3)

        self.skeleton_status = "Skeletons offline - using synthetic fibers"
        if graph is not None:
            try:
                from kickthefly.sim import morphology
                skels, status = morphology.load_key_skeletons(graph, n_samples=self.FIBER + self.ARBOR)
                self.skeleton_status = status
                for idx, coords in skels.items():
                    if 0 <= idx < n and len(coords) == (self.FIBER + self.ARBOR):
                        pts[idx] = coords
            except Exception:
                self.skeleton_status = "Skeletons offline - using synthetic fibers"

        wts = np.concatenate([np.ones(self.FIBER), np.full(self.ARBOR, 2.0)]).astype(np.float32)
        depth = np.clip(1.2 - (pts[..., 2] - 5000.0) / 50000.0, 0.35, 1.0)   # the front of the brain is brighter
        nid = np.broadcast_to(np.arange(n)[:, None], pts.shape[:2])
        keep = ok[:, None] & (pts[..., 2] < VIEW_ZCUT)

        self.M, self.base, self.gain, self.spark_pix = {}, {}, {}, {}
        for key, (w, h) in VIEW_SIZES.items():
            px = ((pts[..., 0] - VIEW_X[0]) / (VIEW_X[1] - VIEW_X[0]) * w).astype(np.int32)
            py = ((pts[..., 1] - VIEW_Y[0]) / (VIEW_Y[1] - VIEW_Y[0]) * h).astype(np.int32)
            m = keep & (px >= 0) & (px < w) & (py >= 0) & (py < h)
            P = sp.coo_array(((wts * depth)[m], ((py * w + px)[m], nid[m])), shape=(w * h, n)).tocsr()
            self.M[key] = P
            struct = P @ self.col
            p99 = float(np.percentile(struct.max(1), 99.0)) or 1.0
            self.base[key] = struct * (0.5 / p99)
            self.gain[key] = 3.4 / p99
            self.spark_pix[key] = np.where(m[:, 0], py[:, 0] * w + px[:, 0], -1)   # cell body (or arbor) pixel
        self.pts, self.wts, self.nid, self.ok, self.n = pts, wts, nid, ok, n
        self.center = np.array([48000.0, 28000.0, 30000.0], np.float32)
        self.yaw, self.pitch = 0.0, 0.0
        self.pan_x, self.pan_y = 0.0, 0.0
        self.zoom = 1.0
        self.preset = "front"
        if regions is not None:
            self.regions = np.asarray(regions, dtype=str)
        else:
            self.regions = np.full(n, "unassigned", dtype=str)
        self.region_names = list(REGION_COLORS.keys())
        self.region_to_id = {name: i for i, name in enumerate(self.region_names)}
        self.region_id = np.array([self.region_to_id.get(r, self.region_to_id["unassigned"]) for r in self.regions], dtype=np.int32)
        self.region_counts = np.bincount(self.region_id, minlength=len(self.region_names))
        self.region_rates = np.zeros(len(self.region_names), np.float32)
        palette_rgb = np.array([REGION_COLORS[name] for name in self.region_names], dtype=np.float32) / 255.0
        self.col_region = (0.15 + 0.85 * palette_rgb[self.region_id]).astype(np.float32)
        self.view_mode = "neuron"  # "neuron" | "region"
        self.calm = np.full(n, 0.025, np.float32)                   # per-neuron calm rate, spikes per step
        self.firing = self.hot_firing = 0
        self._cache_front = (self.M["big"], self.base["big"], self.gain["big"], self.spark_pix["big"])
        self._preset_cache = {"front": self._cache_front}
        self._dirty_big = False
        self._lock = threading.Lock()

    def toggle_view_mode(self) -> str:
        self.view_mode = "region" if self.view_mode == "neuron" else "neuron"
        return self.view_mode

    def set_camera(self, yaw: float, pitch: float, pan_x: float = 0.0, pan_y: float = 0.0, zoom: float = 1.0, preset: str | None = None, recompute: bool = True) -> None:
        self.yaw = float(yaw)
        self.pitch = float(np.clip(pitch, -90.0, 90.0))
        self.pan_x = float(pan_x)
        self.pan_y = float(pan_y)
        self.zoom = float(np.clip(zoom, 0.2, 5.0))
        self.preset = preset or ("front" if self.is_default_view() else "custom")
        self._dirty_big = True
        if recompute:
            self._recompute_big()

    def orbit(self, dyaw: float, dpitch: float) -> None:
        self.set_camera(self.yaw + dyaw, self.pitch + dpitch, self.pan_x, self.pan_y, self.zoom, preset=None)

    def pan(self, dpan_x: float, dpan_y: float) -> None:
        self.set_camera(self.yaw, self.pitch, self.pan_x + dpan_x, self.pan_y + dpan_y, self.zoom, preset=None)

    def zoom_by(self, factor: float) -> None:
        self.set_camera(self.yaw, self.pitch, self.pan_x, self.pan_y, self.zoom * factor, preset=None)

    def set_preset(self, name: str) -> None:
        if name in ("front", "reset"):
            self.yaw, self.pitch, self.pan_x, self.pan_y, self.zoom = 0.0, 0.0, 0.0, 0.0, 1.0
            self.preset = "front"
            M, b, g, spx = self._preset_cache["front"]
            self.M["big"], self.base["big"], self.gain["big"], self.spark_pix["big"] = M, b, g, spx
        elif name == "side":
            if "side" in self._preset_cache:
                self.yaw, self.pitch, self.pan_x, self.pan_y, self.zoom = 90.0, 0.0, 0.0, 0.0, 1.0
                self.preset = "side"
                M, b, g, spx = self._preset_cache["side"]
                self.M["big"], self.base["big"], self.gain["big"], self.spark_pix["big"] = M, b, g, spx
            else:
                self.set_camera(90.0, 0.0, 0.0, 0.0, 1.0, preset="side")
                self._preset_cache["side"] = (self.M["big"], self.base["big"], self.gain["big"], self.spark_pix["big"])
        elif name == "top":
            if "top" in self._preset_cache:
                self.yaw, self.pitch, self.pan_x, self.pan_y, self.zoom = 0.0, -90.0, 0.0, 0.0, 1.0
                self.preset = "top"
                M, b, g, spx = self._preset_cache["top"]
                self.M["big"], self.base["big"], self.gain["big"], self.spark_pix["big"] = M, b, g, spx
            else:
                self.set_camera(0.0, -90.0, 0.0, 0.0, 1.0, preset="top")
                self._preset_cache["top"] = (self.M["big"], self.base["big"], self.gain["big"], self.spark_pix["big"])

    def is_default_view(self) -> bool:
        return abs(self.yaw) < 1e-4 and abs(self.pitch) < 1e-4 and abs(self.pan_x) < 1e-4 and abs(self.pan_y) < 1e-4 and abs(self.zoom - 1.0) < 1e-4

    def _recompute_big(self) -> None:
        with getattr(self, "_lock", threading.Lock()):
            self._dirty_big = False
            if self.is_default_view():
                M, b, g, spx = self._cache_front
                self.M["big"], self.base["big"], self.gain["big"], self.spark_pix["big"] = M, b, g, spx
                return
            rad_y = math.radians(self.yaw)
            rad_p = math.radians(self.pitch)
            cy, sy = math.cos(rad_y), math.sin(rad_y)
            cp, sp_ = math.cos(rad_p), math.sin(rad_p)
            R = np.array([[cy, 0.0, sy], [sp_ * sy, cp, -sp_ * cy], [-cp * sy, sp_, cp * cy]], dtype=np.float32)

            rot = (self.pts - self.center) @ R.T
            w, h = VIEW_SIZES["big"]
            span_x = VIEW_X[1] - VIEW_X[0]
            span_y = VIEW_Y[1] - VIEW_Y[0]

            rx = rot[..., 0] * self.zoom + self.pan_x + span_x * 0.5
            ry = rot[..., 1] * self.zoom + self.pan_y + span_y * 0.5
            rz = rot[..., 2] * self.zoom + self.center[2]

            px = (rx / span_x * w).astype(np.int32)
            py = (ry / span_y * h).astype(np.int32)

            depth = np.clip(1.2 - (rz - 5000.0) / 50000.0, 0.35, 1.0)
            m = self.ok[:, None] & (px >= 0) & (px < w) & (py >= 0) & (py < h)

            P = sp.coo_array(((self.wts * depth)[m], ((py * w + px)[m], self.nid[m])), shape=(w * h, self.n)).tocsr()
            self.M["big"] = P
            struct = P @ self.col
            p99 = float(np.percentile(struct.max(1), 99.0)) or 1.0
            self.base["big"] = struct * (0.5 / p99)
            self.gain["big"] = 3.4 / p99
            self.spark_pix["big"] = np.where(m[:, 0], py[:, 0] * w + px[:, 0], -1)

    def set_palette(self, name: str) -> None:
        hot, cool_base = self.PALETTES.get(name, self.PALETTES["default"])
        self.hot = np.array(hot, np.float32)
        cool = (0.35 * self.col + 0.65 * np.array(cool_base, np.float32)) * 0.55   # everything else: dim tint
        self.tint = np.where(self.hot_mask[:, None], self.hot * 2.2, cool).astype(np.float32)
        self.legend = (tuple(int(255 * c) for c in hot), tuple(int(min(255, 255 * c * 1.1)) for c in cool_base))

    # --- the two pieces of render() that a GPU view replaces (kickthefly/game/gpu_brainview.py) ----------------------------------
    def _accumulate(self, key: str, amp: np.ndarray, mode: str, sparks: np.ndarray | None = None) -> np.ndarray:
        """The light of every neuron on the (h*w, 3) image: sum over a neuron's fiber and arbor samples of weight * depth * amp * color,
        times the view's gain, plus a sparkle at the cell body of each neuron in `sparks`. `mode` picks the color: the neuron tints, or
        the region colors."""
        color = self.tint if mode == "neuron" else self.col_region * 1.8
        light = (self.M[key] @ (amp[:, None] * color)) * self.gain[key]
        if sparks is not None and len(sparks):
            pix = self.spark_pix[key][sparks]
            s, pix = sparks[pix >= 0], pix[pix >= 0]
            np.add.at(light, pix, np.where(self.hot_mask[s, None], self.hot * 3.0, np.float32(0.7)))
        return light

    def _region_base(self, key: str) -> np.ndarray:
        """The dim, depth-shaded structure of the brain in region colors."""
        return self.M[key] @ (self.col_region * 0.45)

    def render(self, key: str, rates: np.ndarray, spiked: np.ndarray, t: float, learn: bool) -> pygame.Surface:
        if key == "big" and getattr(self, "_dirty_big", False):
            self._recompute_big()
        if learn:
            self.calm += (rates - self.calm) * 0.01
        excess = np.maximum(rates / 0.025 - self.calm / 0.025 - np.where(self.hot_mask, 1.0, 0.6), 0)
        act = (2.2 * excess).astype(np.float32)
        firing = act > 0.05
        self.firing, self.hot_firing = int(firing.sum()), int((firing & self.hot_mask).sum())

        rates_hz = rates * 200.0
        self.region_rates = np.bincount(self.region_id, weights=rates_hz, minlength=len(self.region_names)) / np.maximum(self.region_counts, 1)

        w, h = VIEW_SIZES[key]
        if self.view_mode == "region":
            reg_heat = self.region_rates[self.region_id]
            heat_act = np.clip((reg_heat - 2.0) / 4.0, 0.0, 3.0).astype(np.float32) * 1.5 + 0.4 * act
            light = self._accumulate(key, heat_act, "region")
            base_col = self._region_base(key)
            img = (255 * (1 - np.exp(-(base_col + light)))).astype(np.uint8)
            surf = pygame.image.frombuffer(img.tobytes(), (w, h), "RGB").copy()
            hot = (255 * (1 - np.exp(-0.7 * light))).astype(np.uint8)
            glow = pygame.image.frombuffer(hot.tobytes(), (w, h), "RGB")
            glow = pygame.transform.smoothscale(pygame.transform.smoothscale(glow, (w // 6, h // 6)), (w, h))
            surf.blit(glow, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
            return surf

        sparks = None
        if len(spiked) and self.sparkle:              # sparkles: firing neurons that spiked on the latest step
            sparks = spiked[act[spiked] > 2.0]
        light = self._accumulate(key, act, "neuron", sparks)
        img = (255 * (1 - np.exp(-(self.base[key] + light)))).astype(np.uint8)
        surf = pygame.image.frombuffer(img.tobytes(), (w, h), "RGB").copy()
        hot = (255 * (1 - np.exp(-0.7 * light))).astype(np.uint8)          # bloom from the firing only
        glow = pygame.image.frombuffer(hot.tobytes(), (w, h), "RGB")
        glow = pygame.transform.smoothscale(pygame.transform.smoothscale(glow, (w // 6, h // 6)), (w, h))
        surf.blit(glow, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
        return surf


# --- fly ragdoll ---------------------------------------------------------------
HEAD, THX, ABD = 0, 1, 2
KNEE = (3, 5, 7, 9, 11, 13)       # near front, mid, hind, then far front, mid, hind
FOOT = (4, 6, 8, 10, 12, 14)
WING = (15, 16)                   # near, far
N_P = 17
STAND = 64                        # thorax height above the floor when standing

REST = np.zeros((N_P, 2))         # facing right, thorax at origin, +y down
REST[HEAD], REST[ABD] = (42, -6), (-50, 6)
for i, (k, f) in enumerate((((44, 22), (58, 60)), ((8, 28), (10, 60)), ((-30, 24), (-44, 60)))):
    REST[KNEE[i]], REST[FOOT[i]] = k, f
    REST[KNEE[i + 3]], REST[FOOT[i + 3]] = (k[0] + 8, k[1] - 4), (f[0] + 10, f[1] - 4)
REST[WING[0]], REST[WING[1]] = (-62, -22), (-54, -30)
RADIUS = np.array([16, 19, 24] + [4] * 12 + [3, 3], float)
PULL = np.array([0.10, 0.10, 0.10] + [0.12, 0.22] * 6 + [0.15, 0.15])
TRIPOD = (0, 1, 0, 1, 0, 1)       # near front/hind + far mid step together

# (a, b, stiffness, shape): shape links only hold legs and wings in place while the fly is a limp ragdoll. They form
# triangles that can't un-flip, so while standing the pose pull shapes the fly instead and they are skipped.
LINKS = [(HEAD, THX, 1.0, False), (THX, ABD, 1.0, False), (HEAD, ABD, 1.0, False)]
for i in range(6):
    LINKS += [(THX, KNEE[i], 0.9, False), (KNEE[i], FOOT[i], 0.9, False), (HEAD if i % 3 == 0 else ABD, KNEE[i], 0.25, True)]
for wtip in WING:
    LINKS += [(THX, wtip, 0.9, False), (ABD, wtip, 0.6, True), (HEAD, wtip, 0.3, True)]
LINK_LEN = [float(np.hypot(*(REST[a] - REST[b]))) for a, b, _, _ in LINKS]
MAX_HEALTH = 100.0
# New arenas are only ever appended: a save state from before 2.7 names its arena by index alone (savestate.py).
ARENAS = ("room", "fan", "flypaper", "pool", "lamp", "escaperoom", "field", "orchard", "thermo", "kitchen")
OUTDOOR_ARENAS = ("field", "orchard")            # 3D only: large open worlds (kickthefly/game/outdoors.py)
THREE_D_ONLY = OUTDOOR_ARENAS + ("kitchen",)     # 3.0 day 3: the kitchen (kickthefly/game/kitchen.py) is 3D only too
# Thermo arena (GAME RULE): the temperature runs linearly from 15 C at the left wall to 35 C at the right, and the
# fly's cold (TRN_VP3) or hot (TRN_VP2) antennal neurons are driven by how far it is from the comfortable middle.
THERMO_MID_X, THERMO_HALF_X = 445.0, 365.0     # 2D pixels; the 3D room uses its own width (kick3d.py)
THERMO_DEADBAND = 0.05


def thermo_gradient(t: float) -> tuple[float, float]:
    """(cold, hot) drive in 0..1 at position t, -1 at the cold wall and +1 at the hot wall."""
    t = float(np.clip(t, -1.0, 1.0))
    return max(0.0, -t), max(0.0, t)
WATER_Y = FLOOR - 140                 # pool surface
PAPER_X = (230.0, 660.0)              # flypaper strip on the floor
LAMP = (445.0, 196.0)                 # bulb center
FAN = (70.0, 548.0)                   # fan hub, low on the left so its wind hits a standing fly
SUGAR_GOAL = (800.0, FLOOR - 14.0)    # escaperoom target sugar dish


def particle_region(i: int) -> tuple[str, str | None]:
    if i == HEAD:
        return "head", None
    if i in (THX, ABD):
        return "body", None
    if i in WING:
        return "wing", "L" if i == WING[0] else "R"
    return "legs", "L" if (i - 3) // 2 < 3 else "R"


class Fly:
    def __init__(self, x: float):
        self.facing = 1
        self.p = REST + (x, FLOOR - STAND)
        self.prev = self.p.copy()
        self.anchor_x = x
        self.phase = 0.0
        self.grabbed: int | None = None
        self.stun_until = 0.0
        self.escape_until = 0.0
        self.walk_until = 0.0
        self.back_until = 0.0
        self.flail_until = 0.0
        self.run = False
        self.turn_ready = 0.0
        self.escape_ready = 0.0
        self.recover = 1.0
        self.hurt = 0.0
        self.last_hit_x = x
        self.action = "idle"
        self.health = MAX_HEALTH
        self.dead_at: float | None = None
        self.char = 0.0                   # scorch from the blowtorch
        self.burn_until = 0.0
        self.soak = 0.0                   # brake cleaner on the body, 0..1
        self.melt = 0.0                   # how dissolved it is; 1 = puddle
        self.dissolved_at: float | None = None
        self.hover = np.array([x, 300.0])   # flight: current hover point and where it's heading
        self.fly_target = np.array([x, 300.0])
        self.wander = False
        self.power = 1.0                  # DNg02 level while flying
        self.frost = 0.0                  # freeze spray, 1 = frozen solid
        self.frozen_at: float | None = None
        self.shattered_at: float | None = None
        self.venom = 0.0                  # spider venom, paralyses
        self.wrapped = False              # in the spider's silk
        self.zap_until = 0.0
        self.eating_until = 0.0
        self.arena = "room"
        self.wind = 0.0                   # fan push, px/frame^2
        self.wet = 0.0                    # seconds until its wings dry
        self.inebriation = 0.0            # alcohol intoxication (0..1, GAME RULE: motor degradation)
        self.stuck: dict[int, np.ndarray] = {}   # flypaper: particle -> glued position

    @property
    def flying(self) -> bool:
        return self.escape_until > 0 and self.grabbed is None and not self.dead and not self.wrapped \
            and self.melt < 0.3 and self.frost < 0.5 and self.venom < 0.5 and self.wet <= 0 and len(self.stuck) < 2

    @property
    def dead(self) -> bool:
        return self.dead_at is not None

    def nearest(self, pos, max_d: float) -> int | None:
        d = np.hypot(*(self.p - pos).T) - RADIUS
        i = int(np.argmin(d))
        return i if d[i] < max_d else None

    def impulse(self, i: int, v) -> None:
        self.prev[i] -= v

    def stun(self, now: float, s: float) -> None:
        self.stun_until = max(self.stun_until, now + s)
        self.hurt = 1.0
        self.escape_until = min(self.escape_until, now)     # knocked out of the air

    def escape(self, now: float, seconds: float = 2.4, wander: bool = False) -> None:
        """Take off. Escapes fly up and away from the last hit; wandering flights drift around the room."""
        away = 1.0 if self.p[THX, 0] >= self.last_hit_x else -1.0
        launch = np.array([away * 4.0, -9.0])
        if getattr(self, "inebriation", 0.0) > 0:
            launch += np.random.normal(0, 2.5 * self.inebriation, 2)
        self.prev[:] = self.p - launch
        self.hover = self.p[THX].copy()
        self.wander = wander
        self._new_target(away)
        self.escape_until = now + seconds
        delay = 2.0 * getattr(self, "inebriation", 0.0)
        self.escape_ready = now + seconds + 1.0 + delay

    def _new_target(self, away: float | None = None) -> None:
        x = self.p[THX, 0]
        if away is None:
            tx = random.uniform(120, PLAY_W - 120)
        else:
            tx = x + away * random.uniform(220, 380)
            if not 100 < tx < PLAY_W - 100:
                tx = x - away * random.uniform(220, 380)
        self.fly_target = np.array([float(np.clip(tx, 100, PLAY_W - 100)), random.uniform(150, 430)])

    def step(self, now: float, mouse) -> list[tuple[int, float]]:
        """One 60 Hz physics frame. Returns (particle, impact speed) for hard contacts with the arena."""
        if self.frozen_at is not None:                       # a block of ice: nothing moves until it shatters
            self.prev = self.p.copy()
            return []
        flying = now < self.escape_until and self.flying
        if flying:
            self.recover = 1.0
        elif self.grabbed is not None or now < self.stun_until or self.dead or self.wrapped:
            self.recover = 0.0
        else:
            self.recover = min(1.0, self.recover + 1 / 30)
        height = (FLOOR - STAND) - self.p[THX, 1]
        stiff = (1 - self.melt) ** 0.5 * (1 - self.frost) * (1 - self.venom)   # dissolving, freezing, paralysed
        strength = self.recover * (1.0 if flying else float(np.clip(1 - height / 160, 0, 1)) * stiff)
        if self.grabbed is not None or (self.arena == "pool" and not flying and self.p[THX, 1] > WATER_Y - 30):
            strength = 0.0                                   # held, or floating: nothing to stand on
        shrink = 1 - 0.35 * self.melt                        # dissolving: the body shrinks and slumps

        v = (self.p - self.prev) * (0.992 - 0.12 * strength)  # standing legs also damp the body
        speed = np.hypot(*v.T)
        v *= np.minimum(1.0, 60.0 / np.maximum(speed, 1e-6))[:, None]
        self.prev = self.p.copy()
        self.p += v
        self.p[:, 1] += 0.9 * (1 - strength)                 # standing or flying: legs or wings carry the weight
        if self.arena == "pool":                             # water: buoyancy and drag, so it floats at the surface
            sub = self.p[:, 1] > WATER_Y
            if sub.any():
                self.p[sub, 1] -= 1.35
                self.p[sub] -= (self.p[sub] - self.prev[sub]) * 0.12
        if self.wind:                                        # the fan pushes wings hardest
            push = np.full(N_P, self.wind * 0.35)
            push[list(WING)] *= 2.5
            self.p[:, 0] += push
            if flying:
                self.hover[0] += self.wind * 3.0

        if flying:
            self._fly(now)
        elif strength > 0:
            self._pose(now, strength)
        else:
            self.anchor_x = float(self.p[THX, 0])
        if self.dead:                                        # legs slowly curl in toward the body
            idx = list(KNEE + FOOT)
            self.p[idx] += (self.p[THX] - self.p[idx]) * 0.02
        elif now < self.flail_until or (self.grabbed is not None and random.random() < 0.5):
            idx = list(KNEE + FOOT)
            self.p[idx] += np.random.normal(0, 2.5, (12, 2))

        if getattr(self, "inebriation", 0.0) > 0 and not self.dead:
            # Uncoordinated motor tremors (GAME RULE: alcohol inebriation)
            self.p += np.random.normal(0, 1.8 * self.inebriation, (N_P, 2))

        posed = strength > 0.5
        for _ in range(6):
            if self.grabbed is not None:
                self.p[self.grabbed] = mouse
            for i, at in self.stuck.items():
                self.p[i] = at
            for (a, b, stiff, shape), rest in zip(LINKS, LINK_LEN):
                if shape and posed:
                    continue
                d = self.p[b] - self.p[a]
                dist = math.hypot(d[0], d[1]) or 1e-6
                corr = d * (0.5 * stiff * (dist - rest * shrink) / dist)
                self.p[a] += corr
                self.p[b] -= corr
            self._clamp()
        if self.grabbed is not None:
            self.p[self.grabbed] = mouse
        for i, at in self.stuck.items():
            self.p[i] = at
            self.prev[i] = at
        if self.arena in ("flypaper", "escaperoom") and self.grabbed is None:   # anything touching the strip sticks
            px0, px1 = (280.0, 560.0) if self.arena == "escaperoom" else PAPER_X
            for i in range(N_P):
                if i not in self.stuck and px0 < self.p[i, 0] < px1 and self.p[i, 1] >= FLOOR - RADIUS[i] - 1.5:
                    self.stuck[i] = self.p[i].copy()
        self.hurt = max(0.0, self.hurt - 1 / 20)
        return self._contacts()

    def _fly(self, now: float) -> None:
        """Hovering flight: the body is held in shape around a hover point that chases a target, legs tucked."""
        d = self.fly_target - self.hover
        dist = float(np.hypot(*d))
        speed = 2.5 + 2.5 * min(self.power, 2.0)
        if dist < 25:
            self._new_target()
        else:
            self.hover += d / dist * min(speed, dist)
        if abs(d[0]) > 8:
            self.facing = 1 if d[0] > 0 else -1
        self.anchor_x = float(self.hover[0])
        self.action = "flying"
        bob = 4 * math.sin(now * 9)
        if getattr(self, "inebriation", 0.0) > 0:
            # Erratic drunk flight wobble (GAME RULE: alcohol inebriation)
            self.hover += np.array([math.sin(now * 3.7) * 4.5 * self.inebriation, math.cos(now * 2.9) * 3.0 * self.inebriation])
        off = REST.copy()
        off[:, 0] *= self.facing
        tgt = off + (self.hover[0], self.hover[1] + bob)
        for i in range(6):
            tgt[FOOT[i]] = tgt[KNEE[i]] + (-6 * self.facing, 14)          # legs hang tucked under the body
        d = tgt - self.p
        self.p += d * 0.25
        self.prev += d * 0.2

    def _pose(self, now: float, strength: float) -> None:
        walking = now < self.walk_until
        backing = now < self.back_until
        speed = 5.0 if self.run else 2.3
        vx = -1.7 * self.facing if backing else speed * self.facing if walking else 0.0
        if vx and not 80 < self.anchor_x + vx < PLAY_W - 80:     # walked into a wall: turn around (game rule)
            self.facing = -self.facing
            vx = -vx
        self.anchor_x = float(np.clip(self.anchor_x + vx, 70, PLAY_W - 70))
        self.anchor_x += 0.05 * (self.p[THX, 0] - self.anchor_x)   # don't drag a body that got knocked away
        self.action = "back up" if backing else ("run" if self.run else "walk") if walking else "idle"
        if vx:
            self.phase += 0.12 * abs(vx)
        bob = 1.5 * math.sin(now * 2.2)
        s = 1 - 0.35 * self.melt
        off = REST * s
        off[:, 0] *= self.facing
        tgt = off + (self.anchor_x, FLOOR - STAND * s + bob)
        if getattr(self, "inebriation", 0.0) > 0:
            # Stumbling gait and wobbling drift (GAME RULE: alcohol inebriation)
            wobble = math.sin(now * 3.8) * 6.0 * self.inebriation
            bob += math.cos(now * 4.5) * 3.0 * self.inebriation
            tgt[:, 0] += wobble
            if vx:
                self.phase += 0.06 * self.inebriation * math.sin(now * 7.0)
        for leg in range(6):
            ph = self.phase + math.pi * TRIPOD[leg]
            if vx:
                tgt[FOOT[leg], 0] += 14 * math.cos(ph) * np.sign(vx)
                tgt[FOOT[leg], 1] -= 12 * max(0.0, math.sin(ph))
        d = tgt - self.p
        self.p += d * (PULL * strength)[:, None]
        self.prev += d * (PULL * strength * 0.85)[:, None]

    def _clamp(self) -> None:
        r = RADIUS
        np.clip(self.p[:, 0], r, PLAY_W - r, out=self.p[:, 0])
        np.clip(self.p[:, 1], CEIL + r, FLOOR - r, out=self.p[:, 1])

    def _contacts(self) -> list[tuple[int, float]]:
        hits = []
        r = RADIUS
        v = self.p - self.prev
        for i in range(N_P):
            x, y = self.p[i]
            if y >= FLOOR - r[i] - 0.5 and v[i, 1] > 0:
                if v[i, 1] > 9:
                    hits.append((i, v[i, 1]))
                self.prev[i, 1] = y + v[i, 1] * 0.35
                self.prev[i, 0] = x - v[i, 0] * 0.75
            elif y <= CEIL + r[i] + 0.5 and v[i, 1] < 0:
                if -v[i, 1] > 9:
                    hits.append((i, -v[i, 1]))
                self.prev[i, 1] = y + v[i, 1] * 0.35
            if (x <= r[i] + 0.5 and v[i, 0] < 0) or (x >= PLAY_W - r[i] - 0.5 and v[i, 0] > 0):
                if abs(v[i, 0]) > 9:
                    hits.append((i, abs(v[i, 0])))
                self.prev[i, 0] = x + v[i, 0] * 0.4
        return hits


def pet_card(pet, slot) -> dict | None:
    """The pet's personality card: the MEASURED card of the pet fly (the original fly, built from Settings > Brain > Random seed), never
    the pet file's own seed (3.0 day 4 review: the card was drawn from pet.seed, a different number from the seed the pet's brain is
    built from). A measured card the pet file carries for that same fly and settings is put back into the card cache; the slot's
    measured card is written into the pet file, so the pet keeps it."""
    from kickthefly.core import cards

    if slot is None:
        return None
    card = slot.personality if (slot.personality or {}).get("measured") is True else None
    stored = cards.check_pet_card(getattr(pet, "personality_card", None)) if pet is not None else None
    if card is None and stored is not None:
        mode, sigma, params = slot.card_settings()
        if stored.get("card_key") == cards.key(slot.seed, mode, sigma, params):
            cards.store(stored)
            slot.personality = card = stored
    if pet is not None and card is not None and getattr(pet, "personality_card", None) is not card:
        pet.personality_card = card
    return card


class FlySlot:
    """One spawned fly: its ragdoll body, its own independent Brain/LIFSim thread, and the episode-scoped bookkeeping
    (hits, pain, reward, loom state...) that used to live directly on Game when there was only ever one fly."""

    def __init__(self, fly: Fly, brain: Brain, seed: int, primary: bool = False):
        self.fly, self.brain, self.seed, self.primary = fly, brain, seed, primary
        self.persist_memory = primary          # only the original fly's learning is saved to disk (see README)
        self.hue = (seed * 0.6180339887) % 1.0  # golden-ratio spread so several flies look visually distinct
        self.refresh_card()
        self.reset_episode()

    def card_settings(self) -> tuple[str, float | None, dict]:
        """(individuality mode, sigma, LIF parameters) this fly's brain really runs with: what its measured card must match. A brain
        that asked for individuality but has no gains (the gl backend) runs the shared connectome, so its card is the 'off' one."""
        sim = getattr(self.brain, "sim", None)
        if sim is None:
            return "off", None, {}
        mode = str(getattr(sim, "individuality", "off") or "off")
        if getattr(sim, "d_pre", None) is None and getattr(sim, "d_post", None) is None:
            mode = "off"
        p = getattr(sim, "p", None)
        params = {k: float(getattr(p, k)) for k in ("noise_std", "bias", "target_rate_hz", "ext_gain", "gain_adapt") if hasattr(p, k)}
        return mode, getattr(sim, "individuality_sigma", None), params

    def refresh_card(self) -> None:
        """This fly's MEASURED personality card from the card cache, or 'card not measured' (3.0 day 4 review, decision A: the card
        used to be numbers drawn from the seed, shown as if they described this fly; core/cards.py)."""
        from kickthefly.core import cards

        mode, sigma, params = self.card_settings()
        self.personality = cards.for_fly(self.seed, mode, sigma, params)

    def reset_episode(self) -> None:
        self.hits = 0
        self.pending_hits: dict[tuple[str, str | None], float] = {}
        self.pending_damage = 0.0
        self.damage_src = ""
        self.born = time.perf_counter()
        self.reward = 0.0
        self.pain = 0.0
        self.pain_parts = np.zeros(5)
        self.last_damage = 0.0
        self.pain_peak = 0.0
        self.pain_max_s = 0.0
        self.pain_trace: list[float] = []
        self.fear_now = self.like_now = 0.0
        self.avoid_ready = 0.0
        self.loom, self.loom_prev, self.threat_x = 0.0, {}, 0.0
        self.scent_now: str | None = None
        self.sugar_scent = False
        self.photo_ready = 0.0


# --- drawing -------------------------------------------------------------------
def ellipse_pts(c, a: float, b: float, ang: float, n: int = 24):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ca, sa = math.cos(ang), math.sin(ang)
    x, y = a * np.cos(t), b * np.sin(t)
    return [(c[0] + x[k] * ca - y[k] * sa, c[1] + x[k] * sa + y[k] * ca) for k in range(n)]


def aapoly(surf, pts, col) -> None:
    ip = [(int(round(x)), int(round(y))) for x, y in pts]
    gfxdraw.filled_polygon(surf, ip, col)
    gfxdraw.aapolygon(surf, ip, col)


def aacircle(surf, c, r: float, col) -> None:
    x, y, r = int(round(c[0])), int(round(c[1])), max(1, int(round(r)))
    gfxdraw.filled_circle(surf, x, y, r, col)
    gfxdraw.aacircle(surf, x, y, r, col)


def thick_line(surf, a, b, w: float, col) -> None:
    d = np.array(b, float) - np.array(a, float)
    L = math.hypot(*d) or 1
    n = np.array([-d[1], d[0]]) / L * w / 2
    aapoly(surf, [a + n, b + n, b - n, a - n], col)
    aacircle(surf, b, w / 2, col)


def shade(col, hurt: float, dead: bool, char: float = 0.0, melt: float = 0.0, frost: float = 0.0):
    if char:
        col = tuple(c * (1 - 0.75 * char) + 22 * char for c in col)
    if frost:
        col = tuple(c * (1 - 0.7 * frost) + t * 0.7 * frost for c, t in zip(col, (175, 215, 245)))
    if dead:
        g = sum(col) / 3
        col = tuple(0.45 * c + 0.55 * g * 0.8 for c in col)
    if melt:                                          # dissolving: sickly, glassy, see-through
        col = tuple(c * (1 - 0.6 * melt) + t * 0.6 * melt for c, t in zip(col, (150, 160, 115)))
    rgb = tuple(int(min(255, c + (255 - c) * hurt * 0.7)) for c in col)
    return (*rgb, int(255 * (1 - 0.6 * melt))) if melt else rgb


def draw_puddle(surf: pygame.Surface, fly: Fly, now: float) -> None:
    """What's left after the brake cleaner: a spreading puddle, a wing and two red eye specks."""
    x = float(fly.p[THX, 0])
    e = min(1.0, (now - fly.dissolved_at) / 1.5)
    w = 70 + 60 * e
    gfxdraw.filled_ellipse(surf, int(x), FLOOR - 3, int(w), int(9 + 3 * e), (110, 105, 60, 170))
    gfxdraw.aaellipse(surf, int(x), FLOOR - 3, int(w), int(9 + 3 * e), (160, 150, 90, 200))
    gfxdraw.filled_ellipse(surf, int(x - w * 0.3), FLOOR - 5, int(w * 0.35), 4, (185, 175, 120, 120))
    wing = ellipse_pts((x + 22, FLOOR - 8), 26, 6, 0.15)
    gfxdraw.filled_polygon(surf, [(int(a), int(b)) for a, b in wing], (205, 218, 238, 90))
    for dx in (-8, 5):
        aacircle(surf, (x + dx, FLOOR - 6), 2.5, (180, 30, 26))
    for k in range(3):                               # slow bubbles popping on the surface
        ph = (now * 0.8 + k / 3) % 1.0
        bx = x + (k - 1) * w * 0.45
        gfxdraw.aacircle(surf, int(bx), int(FLOOR - 6 - 10 * ph), int(2 + 3 * ph), (200, 200, 160, int(200 * (1 - ph))))


def draw_proboscis(surf: pygame.Surface, fly: Fly, now: float) -> None:
    if now < getattr(fly, "proboscis_until", 0.0) and not fly.dead:
        head, abd = fly.p[HEAD], fly.p[ABD]
        fwd = (head - abd) / max(1.0, float(np.hypot(*(head - abd))))
        tip = head + fwd * 14 + np.array([0.0, 22.0])
        thick_line(surf, head + (0, 8), tip, 4, (130, 95, 60))


def draw_fly(surf: pygame.Surface, fly: Fly, now: float) -> None:
    from kickthefly.game.larva import LarvaBody, draw_larva
    if isinstance(fly, LarvaBody):
        draw_larva(surf, fly, now)
        return
    if fly.dissolved_at is not None:
        draw_puddle(surf, fly, now)
        return
    if fly.shattered_at is not None:
        return                                        # the shards are drawn by the game
    _draw_fly_body(surf, fly, now)
    draw_proboscis(surf, fly, now)
    p = fly.p
    if fly.wrapped:                                   # spider silk
        axis = p[HEAD] - p[ABD]
        ang = math.atan2(axis[1], axis[0])
        mid = (p[HEAD] + p[ABD]) / 2
        L = float(np.hypot(*axis))
        gfxdraw.filled_polygon(surf, [(int(x), int(y)) for x, y in ellipse_pts(mid, L / 2 + 26, 26, ang)], (235, 235, 230, 170))
        for k in range(-3, 4):
            q = mid + np.array([math.cos(ang), math.sin(ang)]) * k * 11
            pts = ellipse_pts(q, 5, 27, ang, 16)
            gfxdraw.aapolygon(surf, [(int(x), int(y)) for x, y in pts], (250, 250, 245, 220))
    if fly.frozen_at is not None:                     # ice block
        lo, hi = p.min(0) - 14, p.max(0) + 14
        box = [(lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1])]
        gfxdraw.filled_polygon(surf, [(int(x), int(y)) for x, y in box], (170, 215, 245, 90))
        gfxdraw.aapolygon(surf, [(int(x), int(y)) for x, y in box], (225, 245, 255, 230))
        gfxdraw.line(surf, int(lo[0] + 8), int(lo[1] + 6), int(lo[0] + 30), int(lo[1] + 6), (255, 255, 255, 220))
    elif fly.frost > 0.1:                             # frost crystals
        rng = random.Random(7)
        for _ in range(int(12 * fly.frost)):
            i = rng.choice((HEAD, THX, ABD, ABD))
            x, y = p[i] + (rng.uniform(-24, 24), rng.uniform(-22, 18))
            for a in (0, 1.05, 2.1):
                gfxdraw.line(surf, int(x - 4 * math.cos(a)), int(y - 4 * math.sin(a)), int(x + 4 * math.cos(a)),
                             int(y + 4 * math.sin(a)), (235, 250, 255, 220))
    if now < fly.eating_until:                        # happy hearts
        for k in range(3):
            ph = (now * 0.9 + k / 3) % 1.0
            hx, hy = p[HEAD, 0] + (k - 1) * 18, p[HEAD, 1] - 30 - 40 * ph
            col = (255, 110, 150, int(255 * (1 - ph)))
            aacircle(surf, (hx - 3, hy), 4, col)
            aacircle(surf, (hx + 3, hy), 4, col)
            aapoly(surf, [(hx - 7, hy + 1), (hx + 7, hy + 1), (hx, hy + 9)], col)


def _draw_fly_body(surf: pygame.Surface, fly: Fly, now: float) -> None:
    p = fly.p
    axis = p[HEAD] - p[ABD]
    ang = math.atan2(axis[1], axis[0])
    fwd = axis / (np.hypot(*axis) or 1)
    up = np.array([fwd[1], -fwd[0]])
    if fwd[0] < 0:                                   # keep "up" above the body when it faces left
        up = -up
    hurt, dead = fly.hurt, fly.dead
    flap = not dead and (now < fly.escape_until or (fly.grabbed is not None and random.random() < 0.3))
    k_ = 1 - 0.4 * fly.melt                           # parts shrink as it dissolves

    zapped = now < fly.zap_until and int(now * 40) % 2 == 0

    def c(col):
        if zapped:                                    # electrocution flash
            return (200, 235, 255)
        return shade(col, hurt, dead, fly.char, fly.melt, fly.frost)

    def hip(k):
        return p[THX] + fwd * (14 - 12 * (k % 3)) + up * -10

    def leg(k, col, w):
        h, kn, ft = hip(k), p[KNEE[k]], p[FOOT[k]]
        thick_line(surf, h, kn, w, col)
        thick_line(surf, kn, ft, w * 0.65, col)

    def wing(i, alpha):
        base = p[THX] + up * 12
        tip = p[WING[i]]
        if flap:
            tip = base + (tip - base) * 0.8 + up * (38 * math.sin(now * 90 + i))
        mid = (base + tip) / 2
        span = np.hypot(*(tip - base))
        pts = ellipse_pts(mid, span / 2 + 6, 13, math.atan2(*(tip - base)[::-1]))
        ip = [(int(x), int(y)) for x, y in pts]
        gfxdraw.filled_polygon(surf, ip, (205, 218, 238, alpha))
        gfxdraw.aapolygon(surf, ip, (235, 240, 252, min(255, alpha + 90)))
        for off in (-4, 4):
            q = tip + up * off
            gfxdraw.line(surf, int(base[0]), int(base[1]), int(q[0]), int(q[1]), (170, 180, 200, alpha + 30))

    wing(1, 55)
    for k in (3, 4, 5):
        leg(k, c((58, 38, 18)), 4.5)
    ac = p[ABD]
    aang = math.atan2(*(p[THX] - p[ABD])[::-1])
    along = np.array([math.cos(aang), math.sin(aang)])
    aapoly(surf, ellipse_pts(ac, 30 * k_, 21 * k_, aang), c((176, 116, 44)))
    for s in (-15, -4, 7):
        aapoly(surf, ellipse_pts(ac + along * s * k_, 4 * k_, 19 * k_, aang, 14), c((82, 52, 24)))
    aapoly(surf, ellipse_pts(ac + (up * 7 + along * 4) * k_, 16 * k_, 6 * k_, aang, 16), c((214, 158, 80)))   # sheen
    aapoly(surf, ellipse_pts(p[THX], 23 * k_, 19 * k_, ang), c((156, 102, 40)))
    aapoly(surf, ellipse_pts(p[THX] + up * 8 * k_, 13 * k_, 6 * k_, ang, 16), c((196, 140, 70)))
    for k in range(5):                                # bristles
        b0 = p[THX] + up * 17 + fwd * (k * 6 - 12)
        b1 = b0 + up * 7 - fwd * 3
        gfxdraw.line(surf, int(b0[0]), int(b0[1]), int(b1[0]), int(b1[1]), c((60, 40, 18)))
    hc = p[HEAD]
    aacircle(surf, hc, 15 * k_, c((146, 96, 38)))
    eye = hc + (fwd * 2 + up * 2) * k_
    aapoly(surf, ellipse_pts(eye, 11 * k_, 12 * k_, ang), c((196, 30, 26)))
    if dead:
        for sgn in (-1, 1):
            a0 = eye + np.array([-6, -6 * sgn])
            a1 = eye + np.array([6, 6 * sgn])
            thick_line(surf, a0, a1, 3, (30, 20, 20))
    else:
        for fx, fy in ((-4, 3), (0, 5), (4, 2), (-2, -1), (3, -3)):
            gfxdraw.pixel(surf, int(eye[0] + fx), int(eye[1] + fy), (120, 16, 14))
        aacircle(surf, eye + (-3, -4), 3, (255, 170, 160))
    for da in (0.0, 0.3):
        a0 = hc + fwd * 12 + up * 8
        a1 = a0 + fwd * 10 + up * (10 - 7 * da)
        thick_line(surf, a0, a1, 2, c((70, 45, 20)))
    thick_line(surf, hc + fwd * 8 - up * 10, hc + fwd * 14 - up * 20, 3, c((70, 45, 20)))
    for k in (0, 1, 2):
        leg(k, c((86, 58, 26)), 5.5)
    wing(0, 75)

    if fly.soak > 0.05:                              # foam where the solvent is eating in
        rng = random.Random(int(now * 8))
        for _ in range(int(4 + 10 * fly.soak)):
            i = rng.choice((HEAD, THX, ABD, ABD))
            fx, fy = p[i] + (rng.uniform(-22, 22) * k_, rng.uniform(-20, 16) * k_)
            gfxdraw.aacircle(surf, int(fx), int(fy), rng.randint(2, 5), (235, 240, 220, int(90 + 120 * fly.soak)))
    if dead:
        e = min(1.0, (now - fly.dead_at) / 2.0)
        halo = hc + np.array([0, -34 - 10 * e])
        gfxdraw.aaellipse(surf, int(halo[0]), int(halo[1]), 16, 5, (255, 225, 120))
        gfxdraw.aaellipse(surf, int(halo[0]), int(halo[1]), 15, 4, (255, 225, 120))
    elif now < fly.stun_until:                       # dizzy stars
        for k in range(4):
            a = now * 6 + k * math.pi / 2
            sx, sy = hc[0] + 28 * math.cos(a), hc[1] - 30 + 7 * math.sin(a)
            pts = [(sx + (7 if j % 2 == 0 else 3) * math.cos(j * math.pi / 5 - math.pi / 2),
                    sy + (7 if j % 2 == 0 else 3) * math.sin(j * math.pi / 5 - math.pi / 2)) for j in range(10)]
            aapoly(surf, pts, (255, 215, 60))


def draw_swatter(surf, pos, ph: float) -> None:
    """ph 0..1: 0-0.3 swing down, 0.3-0.55 pressed, 0.55-1 lift and fade."""
    if ph < 0.3:
        angle, alpha = -0.9 * (1 - (ph / 0.3) ** 2), 255
    elif ph < 0.55:
        angle, alpha = 0.0, 255
    else:
        e = (ph - 0.55) / 0.45
        angle, alpha = -1.1 * e, int(255 * (1 - e))
    pivot = np.array([pos[0] + 260, pos[1] + 330])
    d = np.array(pos, float) - pivot
    L = float(np.hypot(*d))
    base = math.atan2(d[1], d[0]) + angle
    u = np.array([math.cos(base), math.sin(base)])
    v = np.array([-u[1], u[0]])
    head = pivot + u * L
    hw, hl = 70, 80
    corners = [head + u * hl * a + v * hw * b for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    thick_line(surf, pivot, head - u * hl, 12, (120, 78, 40, alpha))
    aapoly(surf, corners, (210, 40, 45, min(alpha, 220)))
    for i in np.linspace(-0.8, 0.8, 7):
        for a0, a1 in ((head - u * hl + v * hw * i, head + u * hl + v * hw * i), (head + u * hl * i - v * hw, head + u * hl * i + v * hw)):
            gfxdraw.line(surf, int(a0[0]), int(a0[1]), int(a1[0]), int(a1[1]), (250, 130, 130, min(alpha, 150)))
    gfxdraw.aapolygon(surf, [(int(x), int(y)) for x, y in corners], (110, 18, 22, alpha))


def draw_icon(surf, name: str, c, col) -> None:
    x, y = c
    if name == "hand":
        aacircle(surf, (x, y + 4), 9, col)
        for k in range(4):
            thick_line(surf, (x - 7 + k * 4.5, y), (x - 8 + k * 5, y - 11 + abs(k - 1.5) * 2), 4, col)
    elif name == "flick":
        aacircle(surf, (x - 4, y + 3), 7, col)
        thick_line(surf, (x - 2, y - 1), (x + 10, y - 9), 4, col)
        for k in (-1, 0, 1):
            thick_line(surf, (x + 13, y - 12 + k * 6), (x + 18, y - 14 + k * 8), 2, col)
    elif name == "swatter":
        aapoly(surf, [(x - 3, y - 13), (x + 11, y - 13), (x + 11, y + 1), (x - 3, y + 1)], col)
        thick_line(surf, (x + 2, y + 1), (x - 8, y + 12), 3, col)
    elif name == "zapper":
        gfxdraw.aaellipse(surf, int(x), int(y - 4), 9, 11, col)
        thick_line(surf, (x, y + 7), (x, y + 14), 3, col)
        aapoly(surf, [(x + 1, y - 12), (x - 5, y - 2), (x, y - 2), (x - 2, y + 5), (x + 5, y - 5), (x, y - 5)], (255, 230, 90))
    elif name == "freeze":
        for a in (0, 1.047, 2.094):
            thick_line(surf, (x - 11 * math.cos(a), y - 11 * math.sin(a)), (x + 11 * math.cos(a), y + 11 * math.sin(a)), 2.5, (170, 220, 255))
    elif name == "spider":
        for sgn in (-1, 1):
            for k in range(4):
                a = (k - 1.5) * 0.45
                thick_line(surf, (x, y), (x + sgn * 13 * math.cos(a), y + 9 * math.sin(a) - 3 + k), 1.6, col)
        aacircle(surf, (x, y + 1), 6, col)
        aacircle(surf, (x, y - 6), 4, col)
    elif name == "frog":
        aacircle(surf, (x, y + 2), 8, (70, 140, 58))
        aacircle(surf, (x - 4, y - 5), 3, (240, 220, 80))
        aacircle(surf, (x + 4, y - 5), 3, (240, 220, 80))
        thick_line(surf, (x + 6, y + 4), (x + 14, y + 9), 3, (230, 90, 100))
    elif name == "dragonfly":
        thick_line(surf, (x - 12, y), (x + 8, y), 3, (38, 116, 190))
        aacircle(surf, (x + 10, y), 4, (30, 150, 90))
        gfxdraw.filled_ellipse(surf, int(x - 2), int(y - 7), 9, 3, (200, 215, 235, 190))
        gfxdraw.filled_ellipse(surf, int(x - 2), int(y + 7), 9, 3, (200, 215, 235, 190))
    elif name == "mantis":
        thick_line(surf, (x - 8, y + 6), (x, y - 2), 4, (116, 158, 56))
        thick_line(surf, (x, y - 2), (x + 3, y - 12), 3, (116, 158, 56))
        aacircle(surf, (x + 4, y - 14), 3, (78, 116, 38))
        thick_line(surf, (x + 3, y - 10), (x + 11, y - 6), 2, (78, 116, 38))
    elif name == "sugar":
        aapoly(surf, [(x - 8, y - 4), (x + 2, y - 9), (x + 11, y - 4), (x + 1, y + 1)], (250, 250, 255))
        aapoly(surf, [(x - 8, y - 4), (x + 1, y + 1), (x + 1, y + 12), (x - 8, y + 7)], (215, 215, 225))
        aapoly(surf, [(x + 1, y + 1), (x + 11, y - 4), (x + 11, y + 7), (x + 1, y + 12)], (190, 190, 205))
    elif name == "fruit":
        aacircle(surf, (x, y + 2), 9, (214, 64, 48))
        aacircle(surf, (x - 3, y - 1), 3, (245, 150, 130))
        thick_line(surf, (x, y - 6), (x + 2, y - 12), 2, (90, 60, 30))
        aapoly(surf, [(x + 2, y - 10), (x + 10, y - 13), (x + 7, y - 6)], (90, 170, 80))
    elif name == "cleaner":
        aapoly(surf, [(x - 8, y - 6), (x + 2, y - 6), (x + 2, y + 13), (x - 8, y + 13)], col)
        aapoly(surf, [(x - 6, y - 11), (x, y - 11), (x, y - 6), (x - 6, y - 6)], col)
        for k in range(3):
            aacircle(surf, (x + 8 + k * 4, y - 12 + (k - 1) * 4), 2, (160, 220, 255))
    elif name == "torch":
        aapoly(surf, [(x - 12, y + 2), (x + 2, y - 4), (x + 5, y + 2), (x - 9, y + 8)], col)
        aapoly(surf, [(x + 4, y - 4), (x + 16, y - 14), (x + 11, y - 1)], (255, 140, 40))
        aapoly(surf, [(x + 5, y - 3), (x + 12, y - 9), (x + 9, y - 2)], (255, 230, 120))
    elif name == "alcohol":
        aapoly(surf, [(x - 6, y + 11), (x + 6, y + 11), (x + 8, y + 3), (x + 3, y - 3), (x + 3, y - 10), (x - 3, y - 10), (x - 3, y - 3), (x - 8, y + 3)], col)
        aapoly(surf, [(x - 5, y + 10), (x + 5, y + 10), (x + 6, y + 4), (x - 6, y + 4)], (225, 80, 130))
        aacircle(surf, (x, y - 1), 2, (255, 180, 210))
    elif name == "laser":
        aapoly(surf, [(x - 10, y + 8), (x + 2, y - 4), (x + 6, y - 2), (x - 6, y + 10)], col)
        aacircle(surf, (x + 4, y - 3), 3, (80, 200, 255))
        thick_line(surf, (x + 5, y - 4), (x + 14, y - 11), 2, (100, 220, 255))
    elif name == "cva":
        aacircle(surf, (x - 4, y), 5, (255, 180, 100))
        aacircle(surf, (x + 3, y - 2), 6, (255, 160, 80))
        aacircle(surf, (x, y + 4), 4, (255, 200, 120))
    elif name == "decoy":
        aacircle(surf, (x, y - 6), 3, (180, 180, 200))
        aacircle(surf, (x, y - 1), 4, (140, 140, 160))
        gfxdraw.aaellipse(surf, int(x), int(y + 6), 5, 8, (120, 120, 140))
    else:
        aacircle(surf, (x - 2, y + 3), 10, col)
        thick_line(surf, (x + 5, y - 5), (x + 10, y - 11), 3, col)
        aacircle(surf, (x + 11, y - 13), 3, AMBER)


# --- game ----------------------------------------------------------------------
TOOLS = (("hand", "HAND", "drag the fly and throw it"), ("flick", "FLICK", "click near the fly"),
         ("swatter", "SWAT", "click on the fly"), ("bomb", "BOMB", "click to drop a bomb, 1.5 s fuse"),
         ("torch", "TORCH", "hold: burns it, maxes out pain"), ("cleaner", "CLEANER", "hold: brake cleaner dissolves it"),
         ("zapper", "ZAP", "click: electric shock through its body"), ("freeze", "FREEZE", "hold: freezes it solid, then smash the ice"),
         ("spider", "SPIDER", "click: drop a spider that hunts it"), ("sugar", "SUGAR", "click: drop sugar to reward it"),
         ("alcohol", "ALCOHOL", "click: drop alcohol, sweet PAM reward but escalating drunkenness"),
         ("laser", "LASER", "targeted laser: hold/click to stimulate or silence cell types in real time"),
         ("cva", "CVA", "puffs cVA pheromone: activates Or67d/DA1 glomerulus"),
         ("decoy", "DECOY", "spawns a decoy female target to evoke courtship"),
         ("fruit", "FRUIT", "click: drop ripe fruit, eaten like sugar (as in the orchard)"),
         ("frog", "FROG", "click: put down a frog; its tongue is very fast"),
         ("dragonfly", "DRAGONFLY", "click: a dragonfly chases flying flies from above"),
         ("mantis", "MANTIS", "click: a mantis creeps up slowly, then strikes"))
assert tuple(t[0] for t in TOOLS) == TOOL_NAMES
# Real vs rule (the on-screen tags, Settings > Brain): which reactions are triggered by the connectome sim's own neurons
# and which by a rule the game adds. REAL means live descending-neuron firing crossed a threshold; how the body then
# moves is always game physics. Everything not listed (your tools' effects, settings) is untagged.
REACTION_SOURCE = {
    "DODGE": "real", "FLY AWAY": "real", "TAKE OFF": "real", "RUN": "real", "KICK": "real", "BACK UP": "real",
    "WALK": "real", "TURN": "real", "SHOOT": "real", "GROOM": "real", "PROBOSCIS": "real",
    "SONG": "real", "CO2": "real", "HEAT": "real", "COLD": "real",
    "ROLL": "rule", "ROLLING": "rule", "PAIN (LARVA)": "rule",   # larva pathways not validated (docs/larva.md)
    "CRAWL": "rule", "HEAD CAST": "rule",
    "EATING": "rule", "TO LIGHT": "rule", "AVOID": "rule", "APPROACH": "rule", "FLEE": "rule", "WRAPPED": "rule",
    "BROKE FREE": "rule", "DIED": "rule", "HIT YOU": "rule", "YOU DIED": "rule", "AUTOPILOT": "rule",
    "PHOTO MODE": "rule", "ALCOHOL": "rule", "INEBRIATED": "rule", "STUMBLE": "rule", "SIP": "rule", "DRINKING": "rule",
    "TO FRUIT": "rule", "LOST": "rule", "RECALL": "rule", "LUNGE": "rule", "FIGHT": "rule", "SLEEP": "rule",
    "cVA PUFF": "rule", "DECOY FEMALE": "rule", "COURTSHIP": "rule",
    # 3.0 day 3: predators, the kitchen, the microphone and Streamer mode are all game rules on the connectome's own neurons
    "CAUGHT": "rule", "TRAPPED": "rule", "TO VINEGAR": "rule", "MIC": "rule", "STREAM": "rule", "CHAT VOTED": "rule",
}
POPUP_SOURCE = {"DODGE!": "real", "YIKES!": "real", "ROLL!": "rule", "NOPE!": "rule", "RUN AWAY!": "rule", "YUM!": "rule",
                "SWEET!": "rule", "NOM NOM": "rule", "K.O.!": "rule", "BROKE FREE!": "rule", "FLY WINS!": "rule",
                "GOTCHA!": "rule", "PEW PEW!": "rule", "TAKE THAT!": "rule", "AUTOPILOT": "rule", "SPECTATOR": "rule",
                "PHOTO MODE": "rule", "*HIC*": "rule", "SIP...": "rule", "GLUG!": "rule", "STUMBLE!": "rule",
                "ALL GONE": "rule", "♪ BUZZ ♪": "rule", "LUNGE!": "rule", "ZZZ": "rule",
                "COURTSHIP": "rule", "cVA PUFF": "rule", "DECOY FEMALE": "rule",
                "A FROG!": "rule", "A DRAGONFLY!": "rule", "A MANTIS!": "rule", "SNAP!": "rule", "ZAP!": "rule", "GULP!": "rule",
                "GRABBED!": "rule", "SNATCH!": "rule", "STRIKE!": "rule", "SWAT!": "rule", "STUCK IN THE TRAP!": "rule",
                "THE COOK RAISES A SWATTER": "rule"}
SOURCE_TIP = {"real": "REAL: triggered by the connectome sim's own neurons firing above a threshold.",
              "rule": "RULE: a game rule, not something the connectome sim produced."}


def reaction_source(text: str) -> str | None:
    head = text.split("  ")[0].strip()
    for key, src in REACTION_SOURCE.items():
        if head == key or head.startswith(key + " ") or text.startswith(key + " "):
            return src
    return None


def draw_source_chip(surf, pos, source: str, font, anchor: str = "midtop", alpha: int = 255) -> pygame.Rect:
    col = (60, 170, 220) if source == "real" else (220, 150, 50)
    alpha = int(min(255, max(0, alpha)))
    img = font.render("REAL" if source == "real" else "RULE", True, (10, 12, 16))
    r = pygame.Rect(0, 0, img.get_width() + 10, img.get_height() + 2)
    setattr(r, anchor, (int(pos[0]), int(pos[1])))
    chip = pygame.Surface(r.size, pygame.SRCALPHA)
    pygame.draw.rect(chip, (*col, alpha), chip.get_rect(), border_radius=5)
    img.set_alpha(alpha)
    chip.blit(img, img.get_rect(center=chip.get_rect().center))
    surf.blit(chip, r)
    return r


# cVA and the decoy have no key: [ and ] are slow motion (and photo mode's field of view). Wheel or toolbar.
# 2.13: the number keys are the rebindable actions slot1..slot10 (config.py) and the hotbar is a loadout (core/loadout.py);
# TOOL_KEYS stays only as the default 1-9, 0 key codes for code that still wants them
TOOL_KEYS = (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9, pygame.K_0, pygame.K_MINUS, pygame.K_EQUALS)
TOOL_KEY_LABELS = ("1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "-", "=", "", "")      # what the toolbar shows for each
TORCH_KEYS = (("head", None), ("body", None), ("legs", "L"), ("legs", "R"), ("wing", "L"), ("wing", "R"), ("heat", None))
OUCH = ("BONK!", "OOF!", "SPLAT!", "THWACK!", "BZZT!", "OW!")
CURSOR_SIZE = {"flick": 12, "swatter": 38, "bomb": 16, "torch": 18, "cleaner": 22, "zapper": 16, "freeze": 22, "spider": 20, "laser": 14, "frog": 20, "dragonfly": 20, "mantis": 20}
LOOM_MIN, LOOM_FULL = 1.5, 8.0        # rad/s of angular expansion: below LOOM_MIN nothing, LOOM_MIN + LOOM_FULL = full drive
SCENT_RANGE = 330.0                   # px: how close a tool must be for the fly to smell it
FEAR_ACT, LIKE_ACT = 0.35, 0.35       # learned memory (memory.py) that changes behavior
MEMORY_STEPS = 10                     # sim steps between plasticity updates (memory.UPDATE_STEPS)
GIF_SIZE, GIF_FRAMES = (400, 238), 90  # 15 fps x 6 s
AUTOPSY_DELAY = 3.0          # seconds of flatline shown before the report


def make_background() -> pygame.Surface:
    y = np.linspace(0, 1, H)[None, :, None]
    x = np.linspace(-1, 1, PLAY_W)[:, None, None]
    top, bottom = np.array([28, 34, 48]), np.array([14, 16, 22])
    img = top * (1 - y) + bottom * y
    yy = np.linspace(-1, 1, H)[None, :, None]
    spot = np.exp(-(x ** 2 * 1.6 + (yy - 0.35) ** 2 * 2.2))    # soft light pooled over the floor
    img = img + spot * np.array([16, 18, 22])
    floor_y = np.arange(H)[None, :, None]
    fl = floor_y >= FLOOR
    wood = np.array([46, 38, 32]) - (floor_y - FLOOR) * 0.12
    img = np.where(fl, wood + spot * 20, img)
    surf = pygame.surfarray.make_surface(np.clip(img, 0, 255).astype(np.uint8))
    for k in range(-12, 14):                         # floor boards in perspective
        x0 = PLAY_W / 2 + k * 70
        pygame.draw.aaline(surf, (38, 31, 26), (x0, FLOOR), (PLAY_W / 2 + k * 150, H))
    for yy_ in (FLOOR + 22, FLOOR + 58, FLOOR + 104):
        pygame.draw.line(surf, (40, 33, 28), (0, yy_), (PLAY_W, yy_))
    pygame.draw.line(surf, (92, 78, 64), (0, FLOOR), (PLAY_W, FLOOR), 2)
    return surf


def make_shadow() -> pygame.Surface:
    w, h = 220, 40
    x = np.linspace(-1, 1, w)[:, None]
    y = np.linspace(-1, 1, h)[None, :]
    a = np.clip(1 - (x ** 2 + y ** 2), 0, 1) ** 1.6 * 150
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.surfarray.pixels_alpha(s)[:] = a.astype(np.uint8)
    return s


class Sound:
    """Every sound is synthesized here with numpy at startup, so the game ships no audio files. M mutes."""

    RATE = 22050

    def __init__(self):
        self.ok, self.muted = False, False
        self.master = self.sfx = self.buzz = 1.0      # Settings > Audio
        self.loops: dict[str, tuple[str, object]] = {}
        try:
            if not pygame.mixer.get_init():
                try:
                    pygame.mixer.init(self.RATE, -16, 1, 512)
                except pygame.error as e:            # e.g. SDL_AUDIODRIVER=pipewire, which pygame's SDL lacks
                    for drv in ("pulseaudio", "alsa", "wasapi", "directsound"):
                        os.environ["SDL_AUDIODRIVER"] = drv
                        try:
                            pygame.mixer.init(self.RATE, -16, 1, 512)
                            log.info("audio: %s failed (%s), using %s", "default driver", e, drv)
                            break
                        except pygame.error:
                            continue
                    else:
                        raise
            self.channels = pygame.mixer.get_init()[2]
            pygame.mixer.set_num_channels(20)
            self.fx = self._make()
            self.ok = True
        except Exception:                            # no audio device: play silently
            self.fx = {}

    def _snd(self, x: np.ndarray, vol: float = 0.6):
        a = (np.clip(x, -1, 1) * 32767 * vol).astype(np.int16)
        if self.channels == 2:
            a = np.repeat(a[:, None], 2, axis=1)
        return pygame.sndarray.make_sound(np.ascontiguousarray(a))

    def _make(self) -> dict:
        R = self.RATE
        rng = np.random.default_rng(5)

        def t(d):
            return np.arange(int(R * d)) / R

        def noise(d):
            return rng.uniform(-1, 1, int(R * d))

        def low(x, k):                               # moving-average lowpass
            return np.convolve(x, np.ones(k) / k, mode="same")

        def sweep(f0, f1, d):
            tt = t(d)
            f = f0 + (f1 - f0) * tt / d
            return np.sin(2 * np.pi * np.cumsum(f) / R)

        fx = {}
        for k, f in enumerate((150, 190, 240, 300)):   # wing buzz loops; 0.5 s holds whole cycles, so they loop cleanly
            tt = t(0.5)
            saw = 2 * ((tt * f) % 1) - 1
            fx[f"buzz{k}"] = self._snd(0.45 * saw * (0.8 + 0.2 * np.sin(2 * np.pi * 2 * tt)), 0.35)
        tt = t(0.18)
        fx["whack"] = self._snd(low(noise(0.18), 6) * np.exp(-tt * 28) * 2 + np.sin(2 * np.pi * 85 * tt) * np.exp(-tt * 20))
        tt = t(0.3)
        fx["squish"] = self._snd(low(noise(0.3), 30) * np.exp(-tt * 10) * 3 + sweep(220, 50, 0.3) * np.exp(-tt * 9) * 0.6)
        tt = t(0.06)
        fx["flick"] = self._snd(noise(0.06) * np.exp(-tt * 90))
        tt = t(1.0)
        fx["boom"] = self._snd(low(noise(1.0), 60) * np.exp(-tt * 4) * 5 + np.sin(2 * np.pi * 45 * tt) * np.exp(-tt * 5), 0.8)
        tt = t(0.4)
        crackle = noise(0.4) * (rng.random(len(tt)) < 0.25)
        fx["zap"] = self._snd((np.sign(np.sin(2 * np.pi * 70 * tt)) * 0.4 + crackle) * np.exp(-tt * 6), 0.5)
        fx["sizzle"] = self._snd((noise(1.0) - low(noise(1.0), 4)) * (0.5 + 0.5 * (rng.random(R) < 0.08)), 0.35)
        fx["hiss"] = self._snd(noise(1.0) - low(noise(1.0), 3), 0.25)
        tt = t(0.2)
        bite = low(noise(0.2), 8) * 2.5 * (np.exp(-tt * 40) + np.exp(-np.maximum(tt - 0.1, 0) * 40) * (tt > 0.1))
        fx["chomp"] = self._snd(bite)
        fx["yum"] = self._snd(np.concatenate([np.sin(2 * np.pi * 520 * t(0.09)) * np.exp(-t(0.09) * 20),
                                              np.sin(2 * np.pi * 780 * t(0.16)) * np.exp(-t(0.16) * 12)]), 0.4)
        tt = t(0.6)
        pings = sum(np.sin(2 * np.pi * f * tt) * np.exp(-tt * 9) for f in (2200, 2950, 3700, 4400))
        fx["shatter"] = self._snd((noise(0.6) - low(noise(0.6), 3)) * np.exp(-tt * 7) + 0.25 * pings, 0.5)
        tt = t(0.6)
        fx["splash"] = self._snd(low(noise(0.6), 12) * np.exp(-tt * 6) * 2 + 0.3 * sweep(300, 900, 0.6) * np.exp(-tt * 10))
        tt = t(1.0)
        fx["whoosh"] = self._snd(low(noise(1.0), 40) * 4 * (0.7 + 0.3 * np.sin(2 * np.pi * tt)), 0.35)
        tt = t(0.9)
        fx["death"] = self._snd(sweep(440, 90, 0.9) * np.exp(-tt * 2.5), 0.45)
        tt = t(0.18)
        fx["dodge"] = self._snd(sweep(180, 950, 0.18) * np.exp(-tt * 6), 0.45)
        tt = t(0.07)
        fx["pop"] = self._snd(np.sin(2 * np.pi * 900 * tt) * np.exp(-tt * 50), 0.4)
        tt = t(0.14)
        fx["bonk"] = self._snd(np.sin(2 * np.pi * 150 * tt) * np.exp(-tt * 25) + noise(0.14) * np.exp(-tt * 120) * 0.5)
        tt = t(0.3)
        fx["drop"] = self._snd(sweep(1400, 500, 0.3) * np.exp(-tt * 5), 0.25)
        tt = t(0.5)
        fx["hum"] = self._snd(np.sin(2 * np.pi * 120 * tt) * 0.5 + np.sin(2 * np.pi * 240 * tt) * 0.2, 0.2)
        tt = t(0.25)
        fx["click"] = self._snd(np.sin(2 * np.pi * 1300 * tt) * np.exp(-tt * 40), 0.3)
        tt = t(0.16)
        fx["pew"] = self._snd(np.sign(sweep(1500, 260, 0.16)) * 0.5 * np.exp(-tt * 14) + noise(0.16) * np.exp(-tt * 60) * 0.3, 0.35)
        tt = t(0.25)
        fx["hurt"] = self._snd(low(noise(0.25), 20) * 3 * np.exp(-tt * 14) + np.sin(2 * np.pi * 70 * tt) * np.exp(-tt * 10), 0.7)
        t1, t2 = t(0.04), t(0.06)
        c1 = np.sin(2 * np.pi * 1800 * t1) * np.exp(-t1 * 120) + noise(0.04) * np.exp(-t1 * 150) * 0.4
        c2 = np.sin(2 * np.pi * 1100 * t2) * np.exp(-t2 * 80) + noise(0.06) * np.exp(-t2 * 100) * 0.5
        pause = np.zeros(int(R * 0.02))
        fx["shutter"] = self._snd(np.concatenate([c1, pause, c2]), 0.6)

        # Drosophila courtship pulse song buzz (GAME RULE audio): 220 Hz oscillation with 35 ms IPI
        tt_song = t(0.42)
        song_wave = np.zeros_like(tt_song)
        n_pulses = int(0.42 / 0.035)
        for pi in range(n_pulses):
            p_start = pi * 0.035
            p_mask = (tt_song >= p_start) & (tt_song < p_start + 0.012)
            tau = tt_song[p_mask] - p_start
            song_wave[p_mask] = np.sin(2 * np.pi * 220 * tau) * np.sin(np.pi * tau / 0.012)
        fx["pulse_song"] = self._snd(song_wave * 0.6, 0.4)

        # Extracellular biphasic action potential clicks (~1.5 to 15 ms) for brain stethoscope sonification
        t_spk = t(0.002)
        spk1 = -np.sin(2 * np.pi * 950 * t_spk) * np.exp(-t_spk * 2500) + 0.15 * noise(0.002) * np.exp(-t_spk * 2000)
        fx["spike_click1"] = self._snd(spk1, 0.45)

        t_spk2 = t(0.005)
        s1 = -np.sin(2 * np.pi * 950 * t_spk2) * np.exp(-t_spk2 * 2500)
        t_shift = np.maximum(t_spk2 - 0.0018, 0)
        s2 = -np.sin(2 * np.pi * 880 * t_shift) * np.exp(-t_shift * 2500) * (t_spk2 >= 0.0018)
        fx["spike_click2"] = self._snd(s1 + 0.85 * s2 + 0.1 * noise(0.005) * np.exp(-t_spk2 * 1500), 0.5)

        t_spk3 = t(0.009)
        s_burst = np.zeros_like(t_spk3)
        for dt_s, f_hz, amp in [(0.0, 950, 1.0), (0.0021, 1020, 0.9), (0.0045, 890, 0.8)]:
            sub_t = np.maximum(t_spk3 - dt_s, 0)
            s_burst += -np.sin(2 * np.pi * f_hz * sub_t) * np.exp(-sub_t * 2400) * (t_spk3 >= dt_s) * amp
        fx["spike_click3"] = self._snd(s_burst + 0.1 * noise(0.009) * np.exp(-t_spk3 * 1000), 0.55)

        t_many = t(0.015)
        s_many = np.zeros_like(t_many)
        for dt_s, f_hz in zip([0.0, 0.0015, 0.0032, 0.0051, 0.0078, 0.0102], [950, 1100, 850, 1050, 920, 980]):
            sub_t = np.maximum(t_many - dt_s, 0)
            s_many += -np.sin(2 * np.pi * f_hz * sub_t) * np.exp(-sub_t * 2200) * (t_many >= dt_s) * 0.6
        fx["spike_click_many"] = self._snd(s_many + 0.12 * noise(0.015) * np.exp(-t_many * 500), 0.6)

        return fx

    def configure(self, cfg) -> None:
        self.master, self.sfx, self.buzz = cfg["audio.master"], cfg["audio.sfx"], cfg["audio.buzz"]
        self.stethoscope_vol = cfg.get("audio.stethoscope_vol", 0.5)
        if cfg["audio.mute"] != self.muted:
            self.muted = cfg["audio.mute"]
            for slot in list(self.loops):
                self.loop(slot, None)
        for slot, (name, ch) in list(self.loops.items()):
            ch.set_volume(self._vol(slot, 1.0))

    def play_spike_click(self, n_spikes: int) -> None:
        if not self.ok or self.muted or n_spikes <= 0:
            return
        vol = float(min(1.0, self.master * getattr(self, "stethoscope_vol", 0.5)))
        if vol <= 0.001:
            return
        if n_spikes == 1:
            snd_name = "spike_click1"
        elif n_spikes <= 4:
            snd_name = "spike_click2"
        elif n_spikes <= 12:
            snd_name = "spike_click3"
        else:
            snd_name = "spike_click_many"
        snd = self.fx.get(snd_name)
        if snd:
            ch = snd.play()
            if ch:
                ch.set_volume(vol)

    def _vol(self, slot: str | None, vol: float) -> float:
        return float(min(1.0, vol * self.master * (self.buzz if slot == "wings" else self.sfx)))

    def play(self, name: str, vol: float = 1.0) -> None:
        if self.ok and not self.muted and name in self.fx:
            ch = self.fx[name].play()
            if ch:
                ch.set_volume(self._vol(None, vol))

    def loop(self, slot: str, name: str | None, vol: float = 1.0) -> None:
        """Keep one looping sound per slot; None (or muted) stops it."""
        if not self.ok:
            return
        cur = self.loops.get(slot)
        if name is None or self.muted:
            if cur:
                cur[1].stop()
                del self.loops[slot]
            return
        if cur and cur[0] == name:
            cur[1].set_volume(self._vol(slot, vol))
            return
        if cur:
            cur[1].stop()
        ch = self.fx[name].play(loops=-1)
        if ch:
            ch.set_volume(self._vol(slot, vol))
            self.loops[slot] = (name, ch)


SURGERY = (  # label, how to find the neurons (see Game._surgery_rows)
    ("Giant fiber escape neuron (DNp01)", ("type", ("DNp01",))),
    ("Looming detectors (LPLC2, LC4)", ("group", ("loom",))),
    ("Touch neurons", ("group", TOUCH)),
    ("Pain relay: ascending neurons", ("pop", ("ascending",))),
    ("Heat and cold sensors", ("group", ("heat", "cold"))),
    ("Smell neurons (ORNs)", ("group", ("smell",))),
    ("Mushroom bodies: Kenyon cells", ("prefix", ("KC",))),
    ("Reward dopamine neurons (PAM)", ("group", ("reward",))),
    ("Punishment dopamine neurons (PPL1)", ("group", ("punish",))),
    ("Moonwalker neurons (MDN)", ("group", ("back",))),
    ("Walking neurons (DNp09)", ("group", ("walk",))),
    ("Flight power neurons (DNg02)", ("group", ("fly",))),
    ("Central complex (compass, steering)", ("prefix", ("EPG", "PEN", "PEG", "PFL", "PFN", "PFG", "PFR", "ER", "FB",
                                                         "hDelta", "vDelta", "FC", "FS", "EL", "ExR"))),
    ("Optic lobes and photoreceptors", ("pop", ("optic lobe", "photoreceptors"))),
    ("Left visual hemifield (LC10, LPLC2, LC4, LPTC)", ("hemifield_visual", "L")),
    ("Right visual hemifield (LC10, LPLC2, LC4, LPTC)", ("hemifield_visual", "R")),
    ("Left hemisphere (all left neurons)", ("hemisphere", "L")),
    ("Right hemisphere (all right neurons)", ("hemisphere", "R")),
    ("Every neuron", ("all", ())),
    # 2.9, appended so older saves' surgery settings keep pointing at the same groups
    ("Song command neuron (pIP10)", ("type", ("pIP10",))),
    ("Courtship neurons P1 (pC1, pMP-e/pMP4)", ("type", P1_TYPES)),
    ("Aggression neurons TK-FruM (AVLP727m)", ("type", ("AVLP727m",))),
    ("Sleep neurons: dorsal fan-shaped body (FB6, FB7)", ("prefix", ("FB6", "FB7"))),
)
HELP = (
    ("1-9, 0, -, =", "pick a tool (or click the toolbar; = is the laser)"),
    ("B", "big live brain view; click a neuron to inspect it"),
    ("O", "brain surgery: silence or stimulate neuron groups"),
    ("E", "change arena: room, fan, flypaper, pool, lamp, escaperoom, thermo"),
    ("P", "pain neurons: normal, more, max"),
    ("I", "immortal mode"),
    ("K", "brain stethoscope (spike sonification clicks)"),
    ("L", "time-lapse record (2x-20x to GIF/MP4)"),
    ("Shift+R", "record a video (MP4 with ffmpeg, else GIF; again to stop)"),
    ("M", "mute sound"),
    ("S / G", "save a screenshot / a GIF of the last 6 seconds"),
    ("N", "spawn another fly, each with its own brain (16; one per CPU core, up to 32, on Numba)"),
    ("F", "cycle focused fly (brain panel / surgery / training)"),
    ("R", "reset to a single fresh fly"),
    ("F11", "fullscreen (or Alt+Enter); drag the window edge to resize"),
    ("T", "training: teach it to fear or like a smell (saved between sessions)"),
    ("D", "the Neurodex: cell types you have discovered (in 3D only with the mouse free, Tab)"),
    (";", "kill cam: slow-motion replay of the fly's last seconds, after it dies"),
    ("Z [ ] .", "pause time, slower, faster, single step"),
    ("F3", "profiler overlay: sim, physics, render and UI milliseconds, FPS, engine (off at every launch)"),
    ("H", "this help"),
    ("Esc", "close a panel, or open the menu (settings, save, quit)"),
)


class Game:
    three_d = False

    def __init__(self, screen, brain: Brain, view: BrainView, graph=None, weights=None, cfg: "config.Config | None" = None):
        self.screen = screen
        self.cfg = cfg if cfg is not None else config.Config(None)
        self.clock = SimClock()
        self.menu = menu_ui.Menu(self)
        from kickthefly.lab import lab
        lab.install(self.menu)
        self.menu.pages["load_state"] = page_load_state
        from kickthefly.ui import help_ui, loadout_ui, neurodex_ui, share_ui
        loadout_ui.install(self.menu)
        help_ui.install(self.menu)
        neurodex_ui.install(self.menu)                # 3.0: the Neurodex panel and Esc > Share
        share_ui.install(self.menu)
        from kickthefly.lab import challenges
        from kickthefly.lab import validation
        self.menu.pages["challenges"] = challenges.page_challenges
        self.challenge = None
        self.recording: dict | None = None
        self.last_export: str | None = None
        self.science_card: tuple[dict, float] | None = None
        self.science_seen = self._load_seen()
        results, self.validation_source = validation.load_results()
        W = brain.sim.W_csr
        self.science_events = validation.passing_events(results, brain.n, int(W.nnz))
        self.lab_params: dict[str, float] = dict(lab.DEFAULTS)
        self.want_quit = False
        self.train_active_speed = 1.0
        self.graph, self.weights = graph, weights     # the loaded connectome, kept so spawn_fly() can build more brains
        from kickthefly.sim.wiring import Wiring
        self.wiring = Wiring()                        # Lab: changes to the connectome itself, off by default
        self.wiring_busy = ""
        self._flipped_set: frozenset = frozenset()    # rows the inspector marks as sign-flipped
        self._next_seed = 1
        self._spawning = False
        self._new_slot: FlySlot | None = None
        self._reset_gen = 0    # bumped by new_fly(), so a spawn_fly() build in flight during an R can't reappear after
        self.brain_type = getattr(brain, "brain_type", self.cfg.get("brain.brain", "adult"))
        running = getattr(getattr(getattr(brain, "sim", None), "backend", None), "name", None)
        self.max_flies = get_max_flies(running or self.cfg.get("brain.backend", "auto"), brain=self.brain_type)
        from kickthefly.core.pet import PetManager
        self.pet = PetManager() if self.cfg.pet else None
        if self.pet:
            self.pet.load_or_create(brain_type=self.brain_type)
        self._manual_focus_until = 0.0
        self.tool = 0
        self.tool_uses = 0                # counted for the tutorial (ui/tutorial.py)
        self.last_tool_used = ""
        self.tutorial = None              # ui/tutorial.py, while the first-launch tutorial runs
        self.kwheel_open = False          # the keyboard/mouse tool wheel (hold the wheel key); the pad has its own
        self.kwheel_vec = [0.0, 0.0]
        self.loadout = loadout_mod.resolve(self.cfg, larva=self.is_larva)
        self.kills = 0
        self.make_fonts()
        self.bg = make_background()
        self.shadow = make_shadow()
        self.view = view
        if hasattr(view, "pref"):
            view.pref = str(self.cfg.get("graphics.brain_view", "auto"))
        self.view_surf: dict[str, pygame.Surface] = {}
        self.view_rect = pygame.Rect(0, 0, 0, 0)
        self.big_view = False
        self.view_stop = False
        view.calm[:] = brain.sim.activity.rates()   # the warmed-up brain's own resting rates
        self._primary_brain = brain
        self.born_view = time.perf_counter()
        self.immortal = False
        self.pain_level = 0
        self.sound = Sound()
        self.arena_i = 0
        self.surgery_open = self.help_open = False
        self.training_open = False
        self.train: dict | None = None
        self.train_scent = "swatter"
        self.train_speed = 1.0
        self.train_buttons: list = []
        self.wipe_armed = 0.0
        self.last_save = time.perf_counter()
        self.surgery_modes = [0] * len(SURGERY)
        self.type_ops: dict[str, int] = {}
        from kickthefly.lab import livelab
        self.thermo_live = livelab.ThermoLive()       # 3.0 day 2: Lab > Thermogenetics (off until an expression is added)
        self.imaging_live = livelab.ImagingLive()     # 3.0 day 2: Lab > Imaging (off until turned on)
        self.imaging_live.indicator = str(self.cfg["brain.imaging_indicator"])
        self.imaging_live.fps = float(self.cfg["brain.imaging_fps"])
        self.imaging_live.baseline_tau_s = float(self.cfg["brain.imaging_f0_tau_s"])
        self.patch_row: int | None = None             # 3.0 day 2: the neuron the inspector's Patch button chose
        self.surgery_buttons: list = []
        self.inspect: dict | None = None
        from kickthefly.lab.laser import LaserState
        self.laser_state = LaserState()
        self.escaperoom_start_t = self.clock.now          # the game clock runs on perf_counter, not from 0
        self.escaperoom_seed = 0
        self.escaperoom_completed = False
        self.escaperoom_finish_t = 0.0
        self.escaperoom_code = ""
        self.inspect_buttons: list = []
        self.big_rect = pygame.Rect(0, 0, 0, 0)
        self.frames: deque = deque(maxlen=GIF_FRAMES)
        self.frame = 0
        self.timelapse_recording = False
        self.timelapse_frames: list[bytes] = []
        self.timelapse_start_t = 0.0
        self.timelapse_size = (600, 340)
        from kickthefly.game.video_recorder import VideoRecorder
        self.video_recorder = VideoRecorder()
        self.saved_msg: tuple[str, float] | None = None
        self.mouse = (0, 0)
        self.new_fly()                              # self.fly/self.brain (below) proxy to self.flies; build it first
        self.surgery_rows = {label: self._surgery_rows(spec) for label, spec in SURGERY}
        for s in config.SETTINGS:                   # settings from config.toml take effect before the first frame
            if s.key != "graphics.fullscreen":
                self.apply_setting(s.key)
        from kickthefly.game import extras3
        self.x3 = extras3.Extras(self)               # 3.0: Neurodex tracking, kill cam, Neuron of the Day
        from kickthefly.game import live_inputs
        self.live = live_inputs.LiveInputs(self)     # 3.0 day 3: microphone and streamer mode (both off at every launch)
        self.menu.pages["live_inputs"] = live_inputs.page
        from kickthefly.ui import arcade_ui

        self.menu.pages["arcade"] = arcade_ui.page             # 3.0 day 4: fly tournament and fly racing
        from kickthefly.ui import whatsnew_ui

        whatsnew_ui.install(self.menu)                          # 3.0 day 5: What's New in 3.0 (once, then Settings > Help)
        Game.LIVE.add(self)
        threading.Thread(target=self._view_loop, name="brain-view", daemon=True).start()

    # every Game that has been built and not shut down (3.0 day 4 review: the brain-view thread holds its game, and through it every
    # brain, until view_stop; test files that never set it kept 1-1.8 GB each. tests/conftest.py shuts these down after each test)
    LIVE: "weakref.WeakSet[Game]" = weakref.WeakSet()

    def shutdown(self, join: float = 2.0) -> None:
        """Stop this game's background threads (the brain-view renderer and every fly's brain thread) so it and its brains can be
        freed. The game is not usable afterwards. Safe to call twice."""
        self.view_stop = True
        for slot in list(getattr(self, "flies", []) or []):
            try:
                slot.brain.stop()
            except Exception:
                pass
        new = getattr(self, "_new_slot", None)
        if new is not None:
            new.brain.stop()
        Game.LIVE.discard(self)
        if join:
            for th in threading.enumerate():
                if th.name == "brain-view" and getattr(getattr(th, "_target", None), "__self__", None) is self:
                    th.join(timeout=join)

    # --- settings, menu and time ----------------------------------------------------------------------------------
    def make_fonts(self) -> None:
        k = 1.2 if self.cfg["access.larger_text"] else 1.0
        self.f_small = pygame.font.SysFont("consolas", 13)       # dense panels keep their size so rows still fit
        self.f_text = pygame.font.SysFont("segoeui,consolas", round(15 * k))
        self.f_bold = pygame.font.SysFont("segoeuisemibold,segoeui,consolas", round(16 * k), bold=True)
        self.f_head = pygame.font.SysFont("segoeuiblack,segoeui,consolas", round(20 * k), bold=True)
        self.f_title = pygame.font.SysFont("segoeuiblack,segoeui,consolas", round(34 * k), bold=True)
        self.f_big = pygame.font.Font(pygame.font.match_font("impact,arialblack,arial"), round(40 * k))

    @property
    def calm_fx(self) -> bool:
        """Reduced flashing (Accessibility)."""
        return bool(self.cfg["access.reduced_flashing"])

    def set_setting(self, key: str | None, value, save: bool = True, force: bool = False) -> None:
        """The one way settings change, from the menu or a hotkey: validate, apply live, save config.toml."""
        if key == "keys":
            pass
        elif key is not None:
            if self.cfg.set(key, value) or force:
                self.apply_setting(key)
        if save:
            self.cfg.save()

    def apply_setting(self, key: str) -> None:
        c = self.cfg
        if key.startswith("audio."):
            self.sound.configure(c)
            if hasattr(self, "flies") and len(self.flies) > 0:
                self.update_stethoscope_target()
        elif key == "brain.arena":
            name = c[key]
            if getattr(self, "is_larva", False):
                from kickthefly.game.larva import is_arena_allowed_for_larva
                allowed, reason = is_arena_allowed_for_larva(name)
                if not allowed:
                    self.note(f"ARENA    {name} not suitable for larva; staying in {ARENAS[self.arena_i]}")
                    if hasattr(self, "menu"):
                        self.menu.flash(reason, menu_ui.AMBER)
                    return
            if not self.three_d and name in THREE_D_ONLY:
                self.note(f"ARENA    {name} needs the 3D game; staying in the room")
                if hasattr(self, "menu"):
                    self.menu.flash("Open field, Orchard and Kitchen need the 3D game (OpenGL 3.3). The 2D game stays "
                                    "indoors.", menu_ui.AMBER)
                name = "room"
            i = ARENAS.index(name) if name in ARENAS else 0
            changed = i != getattr(self, "arena_i", 0)
            self.arena_i = i
            if hasattr(self, "flies"):
                for slot in self.flies:
                    slot.fly.stuck.clear()
                self.on_arena_changed(self.clock.now)
                if changed:
                    self.note(f"ARENA    {ARENAS[self.arena_i]}")
        elif key == "brain.individuality":
            mode = c[key]
            for slot in getattr(self, "flies", []):
                sim = slot.brain.sim
                sim.p.individuality = mode
                from kickthefly.core.individuality import compute_fly_gains
                sim.d_pre, sim.d_post = compute_fly_gains(slot.seed, sim.n, mode=mode, sigma=sim.p.individuality_sigma)
                sim.backend.sync_to_host()      # device backends keep their own copy of the gains
                sim.backend.sync_from_host()
        elif key == "controls.loadout_preset":
            if hasattr(self, "loadout"):
                self.refresh_loadout()
        elif key == "brain.mode":
            if hasattr(self, "loadout"):
                self.refresh_loadout()
            mode = c[key]
            if mode == "pet" and not getattr(self, "pet", None):
                from kickthefly.core.pet import PetManager
                self.pet = PetManager()
                self.pet.load_or_create(brain_type=self.brain_type)
        elif key == "brain.neuron_shapes":
            from kickthefly.sim import morphology
            morphology.set_opt_in(bool(c[key]))
        elif key == "brain.pet_real_stakes":
            if getattr(self, "pet", None):
                self.pet.real_stakes = bool(c[key])
        elif key == "brain.pain_level":
            self.pain_level = c[key]
            for slot in self.flies:
                slot.brain.set_pain_level(self.pain_level)
        elif key == "brain.immortal":
            self.immortal = c[key]
        elif key == "brain.sim_speed":
            self.clock.scale = c[key]
        elif key == "graphics.fps_cap":
            self.clock.fixed_per_frame = c[key] == 60
        elif key == "graphics.brain_view" and hasattr(self.view, "pref"):
            self.view.pref = str(c[key])
        elif key == "graphics.fullscreen":
            if bool(c[key]) != self.is_fullscreen():
                pygame.display.toggle_fullscreen()
        elif key == "access.palette":
            self.view.set_palette(c[key])
        elif key == "access.language":
            from kickthefly.core import i18n
            i18n.set_language(c[key])
        elif key == "access.larger_text":
            self.make_fonts()
        elif key == "brain.mirror_weights":
            from kickthefly.core import simcore
            mirror = bool(c[key]) and not self.is_larva       # mirroring pairs adult L/R instances only
            g, orig_w, _ = simcore.pack(brain=self.brain_type)
            w_new = simcore.symmetrize_weights(g, orig_w) if mirror else orig_w
            self.weights = w_new
            w_csr = w_new.astype(np.float32).tocsr()
            w_csc = w_csr.tocsc()
            for slot in self.flies:
                slot.brain.sim.W_csr = w_csr
                slot.brain.sim.W_csc = w_csc
        elif key == "brain.dtype":
            dt_str = str(c[key])
            dt = np.float64 if dt_str == "float64" else np.float32
            for slot in getattr(self, "flies", []):
                sim = slot.brain.sim
                sim.p.dtype = dt_str
                sim.dtype = dt
                sim.v = sim.v.astype(dt)
                sim._drive = sim._drive.astype(dt)
                sim._noise = sim._noise.astype(dt)
                sim._sfloat = sim._sfloat.astype(dt)
                sim._zeros = sim._zeros.astype(dt)
                sim.leak = dt(sim.p.dt_ms / sim.p.tau_ms)
        elif key == "brain.backend":
            from kickthefly.sim.connectome import backends
            b_choice = str(c[key])
            for slot in getattr(self, "flies", []):
                sim = slot.brain.sim
                sim.p.backend = b_choice
                sim.backend_choice = b_choice
                sim.backend = backends.create_backend(sim, b_choice)
            if getattr(self, "flies", None):
                self.max_flies = get_max_flies(self.flies[0].brain.sim.backend.name)

    def toggle_mirror_weights(self) -> None:
        val = not bool(self.cfg["brain.mirror_weights"])
        self.cfg.set("brain.mirror_weights", val)
        self.apply_setting("brain.mirror_weights")
        self.cfg.save()

    @staticmethod
    def is_fullscreen() -> bool:
        try:
            return bool(pygame.display.get_surface() and pygame.display.get_surface().get_flags() & pygame.FULLSCREEN) \
                or bool(pygame.display.is_fullscreen())
        except (AttributeError, pygame.error):
            return False

    def toggle_fullscreen(self) -> None:
        if not pygame.display.toggle_fullscreen():
            surf = pygame.display.get_surface()
            if surf is not None:
                is_fs = bool(surf.get_flags() & pygame.FULLSCREEN)
                new_flags = pygame.SCALED | (pygame.RESIZABLE if is_fs else pygame.FULLSCREEN)
                self.screen = pygame.display.set_mode((W, H), new_flags)
        self.cfg.set("graphics.fullscreen", self.is_fullscreen())
        self.cfg.save()

    def hotbar_click(self, pos) -> bool:
        for r, d in getattr(self, "page_rects", []):
            if r.collidepoint(pos):
                self.turn_page(d)
                return True
        for k, r in enumerate(getattr(self, "tool_rects", [])):
            if r.collidepoint(pos):
                self.select_slot(k)
                return True
        return False

    def tutorial_event(self, ev) -> bool:
        tut = getattr(self, "tutorial", None)
        return bool(tut is not None and tut.active and tut.handle(ev))

    def open_menu(self, screen: str = "pause") -> None:
        self.torching = False
        self.kwheel_open = False
        self.menu.show(screen)
        self.clock.menu_paused = True

    # --- tool loadout, hotbar and wheel (2.13; core/loadout.py, ui/loadout_ui.py) ---------------------------------------
    def tool_name(self) -> str:
        return TOOLS[self.tool][0]

    def refresh_loadout(self) -> None:
        """Rebuild the hotbar from the config and the current mode and brain (a preset, mode or brain change).
        A tool that is no longer allowed (the laser after leaving Lab) is put down for the hand."""
        prev = self.tool_name()
        self.loadout = loadout_mod.resolve(self.cfg, larva=self.is_larva)
        if not loadout_mod.available(prev, lab=self.cfg.lab, larva=self.is_larva):
            self.tool = TOOL_NAMES.index("hand")
            self.torching = False
        else:
            self.loadout.show(prev)

    def select_tool(self, name: str) -> bool:
        """Take a tool in hand by name, from the hotbar, the wheel, the mouse wheel or the gamepad."""
        if name not in TOOL_NAMES or not loadout_mod.available(name, lab=self.cfg.lab, larva=self.is_larva):
            return False
        self.tool = TOOL_NAMES.index(name)
        self.loadout.show(name)
        self.torching = False
        return True

    def select_slot(self, slot: int) -> bool:
        """Hotbar slot 0-9 (keys 1-9, 0) on the page showing."""
        name = self.loadout.slot_tool(slot)
        return self.select_tool(name) if name else False

    def cycle_tool(self, d: int) -> None:
        """Mouse wheel and gamepad bumpers: the next (+1) or previous (-1) tool of the loadout."""
        self.select_tool(self.loadout.neighbor(self.tool_name(), 1 if d > 0 else -1))

    def turn_page(self, d: int) -> None:
        if self.loadout.turn_page(d):
            self.note_loadout(f"{tr('Hotbar page')} {self.loadout.page + 1}/{self.loadout.n_pages}")

    def note_loadout(self, text: str) -> None:
        self.saved_msg = (text, time.perf_counter())

    def open_loadout_editor(self) -> None:
        from kickthefly.ui import loadout_ui
        loadout_ui.install(self.menu)
        self.open_menu("loadout")

    def apply_loadout_setting(self) -> None:
        self.refresh_loadout()

    def _wheel_tools(self) -> list[str]:
        return loadout_mod.available_tools(lab=self.cfg.lab, larva=self.is_larva)

    def open_wheel(self) -> None:
        if self.menu.open or self._overlay_open() or getattr(self, "photo_mode", False):
            return
        self.kwheel_open = True
        self.kwheel_vec = [0.0, 0.0]
        self.torching = False

    def close_wheel(self, apply: bool = True) -> None:
        """Let go of the wheel key: take the tool the pointer is on (nothing if it never left the middle)."""
        if not self.kwheel_open:
            return
        self.kwheel_open = False
        if apply:
            from kickthefly.ui import loadout_ui
            pick = loadout_ui.wheel_pick(self.kwheel_vec, len(self._wheel_tools()))
            if pick is not None:
                self.select_tool(self._wheel_tools()[pick])
        self.kwheel_vec = [0.0, 0.0]

    def wheel_event(self, ev) -> bool:
        """Events while the keyboard/mouse wheel is open. Returns True if consumed. In 2D the pointer's place picks;
        the 3D game feeds mouse movement in through wheel_move() instead (the mouse is captured for looking)."""
        if not self.kwheel_open:
            return False
        if ev.type == pygame.KEYUP and self.cfg.action_for(pygame.key.name(ev.key)) == "tool_wheel":
            self.close_wheel(True)
            return True
        if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
            self.close_wheel(False)
            return True
        if ev.type == pygame.MOUSEMOTION and (not self.three_d or not getattr(self, "look", False)):
            from kickthefly.ui import loadout_ui
            cx, cy = loadout_ui.wheel_center(self)
            self.kwheel_vec = [ev.pos[0] - cx, ev.pos[1] - cy]
            return True
        if ev.type == pygame.MOUSEBUTTONDOWN:
            self.close_wheel(True)
            return True
        return ev.type in (pygame.MOUSEBUTTONUP, pygame.MOUSEWHEEL, pygame.MOUSEMOTION)

    def wheel_move(self, rel) -> None:
        """3D: mouse movement while the wheel is open moves the pointer instead of turning the view."""
        v = self.kwheel_vec
        v[0] += rel[0]
        v[1] += rel[1]
        m = math.hypot(*v)
        if m > 150:
            v[0], v[1] = v[0] * 150 / m, v[1] * 150 / m

    def show_first_run_notices(self) -> None:
        """Once per install, from the run loops (never from Game.__init__, so tests and headless runs skip them): the
        loadout notice a migrated config gets, else the first-launch tutorial."""
        fr = self.cfg.first_run
        if fr.get("loadout_notice"):
            fr["loadout_notice"] = False
            self.cfg.dirty = True
            self.cfg.save()
            from kickthefly.ui import loadout_ui
            loadout_ui.install(self.menu)
            self.open_menu("loadout_notice")
        elif not fr.get("tutorial_done"):
            self.start_tutorial()
        elif not fr.get("whatsnew_3_0_seen"):               # 3.0 day 5: once, then never again (a migrated [first_run] key)
            fr["whatsnew_3_0_seen"] = True
            self.cfg.dirty = True
            self.cfg.save()
            self.open_whatsnew()
        elif not fr.get("neuron_shapes_asked") and not self.cfg["brain.neuron_shapes"]:
            fr["neuron_shapes_asked"] = True             # asked once; Settings > Brain changes it later
            self.cfg.dirty = True
            self.cfg.save()
            self.menu.pages["neuron_shapes_ask"] = page_neuron_shapes
            self.open_menu("neuron_shapes_ask")
        else:
            self.x3.maybe_show_notd()                 # 3.0: the Neuron of the Day card (its own setting, default on)

    def open_whatsnew(self) -> None:
        self.open_menu("whatsnew")

    def start_tutorial(self, replay: bool = False) -> None:
        from kickthefly.ui import tutorial
        self.tutorial = tutorial.Tutorial(self, replay=replay)

    def menu_action(self, name: str) -> None:
        if name == "resume":
            self.menu.close()
        elif name == "closed":
            self.clock.menu_paused = False
        elif name == "click_sound":
            self.sound.play("click")
        elif name == "settings":
            self.menu.show("settings")
        elif name == "quit":
            self.menu.show("confirm_quit")
        elif name == "quit_now":
            self.live.stop_all()
            self.want_quit = True
        elif name == "lab":
            self.menu.show("lab")
        elif name == "challenges":
            self.menu.show("challenges")
        elif name == "neurodex":
            self.x3.open_neurodex()
        elif name == "share":
            self.menu.show("share")
        elif name == "live_inputs":
            self.menu.show("live_inputs")
        elif name == "arcade":
            self.menu.show("arcade")
        elif name == "minipapers":
            self.menu.show("lab_minipapers")           # 3.0 day 5: the same page as Lab > Mini-papers
        elif name == "save_state":
            self.save_state()
        elif name == "load_state":
            self.menu._saves_cache = None
            self.menu.show("load_state")
        elif name == "toggle_mode":
            curr = self.cfg.get("brain.mode", "play")
            next_mode = {"play": "lab", "lab": "pet", "pet": "play"}.get(curr, "play")
            self.set_setting("brain.mode", next_mode)
            self.menu.flash(f"{next_mode.capitalize()} mode", menu_ui.GOOD)
        else:
            self.menu.flash("coming soon", menu_ui.AMBER)

    # --- challenges and real-science popups -------------------------------------------------------------------------
    def start_challenge(self, key: str) -> None:
        from kickthefly.lab import challenges

        if self.challenge is not None:
            self.challenge.end()
        if self.train is not None:
            self.stop_training()
        self.training_open = self.surgery_open = self.big_view = self.help_open = False
        self.challenge = challenges.CLASSES[key](self)
        if self.menu.open:
            self.menu.close()

    def sneak_distance(self, slot) -> float:
        """How far your cursor is from the fly's head, in fly lengths (the 3D game measures your body and tool)."""
        return float(np.hypot(*(np.asarray(self.mouse, float) - slot.fly.p[HEAD]))) / 92.0

    def on_reaction(self, kind: str, slot) -> None:
        """A reaction happened (DODGE, AVOID, GROOM, PROBOSCIS): challenges and the real-science popup hear about it."""
        if self.challenge is not None:
            self.challenge.on_reaction(kind, slot)
            return                                    # popups are for normal play only
        if not self.cfg["brain.science_popups"] or kind in self.science_seen:
            return
        test = self.science_events.get(kind)
        if test is None:
            return
        self.science_seen.add(kind)
        self.science_card = (test, self.clock.now)
        try:
            p = paths.get().data_dir / "science_seen.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            import json
            p.write_text(json.dumps(sorted(self.science_seen)), encoding="utf-8")
        except OSError:
            pass

    @staticmethod
    def _load_seen() -> set:
        import json
        try:
            return set(json.loads((paths.get().data_dir / "science_seen.json").read_text(encoding="utf-8")))
        except (OSError, ValueError):
            return set()

    def science_rect(self) -> pygame.Rect:
        return pygame.Rect(PLAY_W // 2 - 300, H - 250, 600, 104)

    def draw_science_card(self, surf, now: float) -> None:
        """Both games call this last: the 3.0 overlays (Neurodex toasts, the kill cam, the launch card) draw with it."""
        self.x3.draw(surf, now)
        self._draw_science_card(surf, now)

    def _draw_science_card(self, surf, now: float) -> None:
        if self.science_card is None:
            return
        test, t0 = self.science_card
        if now - t0 > 12.0 or self.cfg.lab:
            self.science_card = None
            return
        r = self.science_rect()
        card = pygame.Surface(r.size, pygame.SRCALPHA)
        pygame.draw.rect(card, (12, 30, 22, 235), card.get_rect(), border_radius=14)
        pygame.draw.rect(card, (90, 200, 120, 255), card.get_rect(), 2, border_radius=14)
        surf.blit(card, r)
        menu_ui.draw_check(surf, (r.x + 30, r.y + 30), 22, (90, 220, 130))
        self._text(surf, "Real flies do this too", (r.x + 54, r.y + 14), (170, 240, 190), self.f_head)
        words, lines, line = test["play"].split(), [], ""
        for w_ in words:
            if self.f_text.size(line + " " + w_)[0] > r.w - 80 and line:
                lines.append(line)
                line = w_
            else:
                line = (line + " " + w_).strip()
        lines.append(line)
        for i, ln in enumerate(lines[:2]):
            self._text(surf, ln, (r.x + 54, r.y + 46 + i * 20), TEXT, self.f_text)
        self._text(surf, "click to dismiss", (r.right - 14, r.y + 12), DIM, self.f_small, "topright")

    def lab_pages(self) -> list[tuple[str, str, str]]:
        """(label, menu page, tooltip) for the Lab hub; later features add their pages to the menu."""
        return [("Validation", "lab_validation", "Which published fly behaviors this sim reproduces, with numbers."),
                ("Assays and repeated trials", "lab_assays", "T-maze conditioning, looming escape and sugar response over "
                 "many flies: standard metrics, mean and 95% CI, and a same-seed control for any surgery."),
                ("Model assumptions", "lab_assumptions", "Transparent disclosure of biophysical simplifications and EM reconstruction caveats."),
                ("Asymmetry audit", "lab_asymmetry", "Measure baseline turning bias and bilateral L vs R synapse & firing asymmetries."),
                ("Connectome robustness", "lab_wiring", "Drop weak connections, flip uncertain neurotransmitter signs "
                 "or block inhibition, and see which validated behaviors survive."),
                ("Critical path finder", "lab_critical", "Silence each cell type in turn and rank them by what it "
                 "does to a behavior, against a same-seed unperturbed control."),
                ("Simulation benchmark", "lab_benchmark", "Measure simulation throughput (neurons/s, synapses/s, sim vs real time) for 1, 8, 16 flies."),
                ("Parameters", "lab_params", "Model parameters and game-rule thresholds, live."),
                ("Record and export", "lab_export", "Record spike times and firing rates live to CSV and npz, with "
                 "metadata."),
                ("Neural clamp", "lab_clamp", "Replay exact spike trains into a modified connectome under dynamic clamping "
                 "to isolate wiring effects from input variation."),
                ("Connectome diff mode", "lab_diff", "Run two flies with different configurations from the same seed "
                 "and inputs and diff regional activity live."),
                ("Optogenetics laser", "lab_laser", "Aimable in-world laser to activate or silence cell types live."),
                ("Psychometric curves", "lab_psych", "Sweep parameters across trials and export publication-ready SVG/PDF."),
                ("Classroom mode", "lab_classroom", "Sequential step-by-step lecture walkthroughs of connectome circuits with citations."),
                ("Protocols", "lab_protocols", "Load and run YAML protocol files."),
                ("Genetic toolkit", "lab_genetics", "Pick neurons by split-GAL4 driver line: line, cell types, neuron count, and what the "
                 "source says about off-target expression."),
                ("Thermogenetics", "lab_thermo", "TrpA1 (activates) and shibire-ts (silences) above a threshold temperature, by cell type "
                 "or driver line."),
                ("Patch clamp", "lab_patch", "Virtual current clamp of one simulated neuron: potential, spikes, current steps, I-F curve "
                 "(a point-neuron model, not electrophysiology)."),
                ("Calcium imaging", "lab_imaging", "GCaMP dF/F simulated from spikes (MODEL): brain view Imaging mode, ROI traces, "
                 "CSV / NWB / TIFF export."),
                ("Pharmacology", "lab_pharm", "Picrotoxin, cholinergic block, glutamate-Cl block and a GABA-A agonist as synaptic "
                 "scaling by predicted transmitter, with the confidence of each prediction shown."),
                ("Network science", "lab_netsci", "Degree distributions, reciprocity, 3-node motifs against a degree-preserving null, rich club, "
                 "communities and per-region summaries of the brain pack (adult or larva), cached and exportable to CSV."),
                ("Sleep deprivation", "lab_sleepdep", "Keep a fly awake through the night with timed disturbances, then measure rebound sleep "
                 "against an undisturbed control (paired). Sleep pressure is a game rule; the dFB readout is the connectome's."),
                ("Behavior rigs", "lab_rigs", "Four classic rigs, each its own scene: a tethered flight simulator, a fly on a ball, Buridan's paradigm and a "
                 "four-field olfactory arena. Steering and walking are read from real descending neurons; each has a pre-registered assay."),
                ("Mini-papers", "lab_minipapers", "Short guided experiments that reproduce a classic paper: state a hypothesis, run it, plot it, and see your "
                 "result next to what the paper found (and why the model misses where it does)."),
                ("Sensitivity analysis", "lab_sensitivity", "Vary each LIF parameter across a documented range and re-run the validated behaviors "
                 "with validation's own criteria: a parameter x behavior heatmap. Analysis only; defaults never change.")]

    def start_recording(self, groups: list[tuple[str, str]], seconds: float, nwb: bool = False) -> None:
        from kickthefly.lab import lab
        from kickthefly.lab import recorder

        br = self.brain
        named = {}
        for label, spec in groups:
            try:
                named[label] = lab.resolve_group(br, spec)
            except ValueError as e:
                self.menu.flash(str(e), menu_ui.BAD)
                return
        rec = recorder.Recorder(br, named).start()
        self.recording = dict(rec=rec, until=br.steps + int(seconds / 0.005), seconds=seconds,
                              slot=self.flies[self.focus], nwb=bool(nwb))
        self.note(f"RECORD   {len(rec.rows):,} neurons for {seconds:g} s{' (+NWB)' if nwb else ''}")
        self.menu.close()

    def stop_recording(self, wait: bool = False) -> None:
        from kickthefly.lab import recorder

        job, self.recording = self.recording, None
        if job is None:
            return
        rec = job["rec"]
        rec.stop()
        stem = recorder.exports_dir() / f"{time.strftime('%Y%m%d-%H%M%S')}-recording" / "recording"

        def save():
            try:
                extra = dict(recorded_live=True, requested_seconds=job["seconds"])
                rec.save(stem, extra, game=self)
                if job.get("nwb"):
                    from kickthefly.lab import nwbexport

                    nwbexport.write(rec, stem.with_name(stem.name + ".nwb"),
                                    recorder.metadata(rec.brain, self, extra), game=self)
                self.last_export = str(stem.parent)
                self.saved_msg = (f"recording saved to {stem.parent.name} in exports", time.perf_counter())
            except Exception as e:
                log.exception("saving the recording failed")
                self.saved_msg = (f"recording failed: {e}", time.perf_counter())

        if wait:
            save()
        else:
            threading.Thread(target=save, name="save-recording", daemon=True).start()

    def draw_recording(self, surf, x: int, y: int) -> None:
        job = getattr(self, "recording", None)
        if job is None:
            return
        rec = job["rec"]
        img = self.f_bold.render(f"REC  {rec.seconds:4.1f} / {job['seconds']:g} s", True, INK)
        box = img.get_rect(midtop=(x, y)).inflate(28, 10)
        pygame.draw.rect(surf, (140, 30, 30), box, border_radius=8)
        pygame.draw.circle(surf, (255, 90, 80) if self.calm_fx or int(time.perf_counter() * 2) % 2 else (180, 60, 60),
                           (box.x + 12, box.centery), 5)
        surf.blit(img, img.get_rect(center=(box.centerx + 6, box.centery)))

    def refresh_validation(self) -> None:
        from kickthefly.lab import validation

        results, self.validation_source = validation.load_results()
        self.science_events = validation.passing_events(results, self.brain.n, int(self.brain.sim.W_csr.nnz))

    def export_lab_result(self, res: dict) -> None:
        from kickthefly.lab import recorder

        try:
            path = recorder.export_result(res, self)
            self.menu.flash(f"Exported to {path}", menu_ui.GOOD, 6)
        except Exception as e:
            log.exception("export failed")
            self.menu.flash(f"Export failed: {e}", menu_ui.BAD, 6)

    def set_lab_param(self, name: str, value: float) -> None:
        from kickthefly.lab import lab

        p = lab.BY_NAME[name]
        v = float(min(max(value, p[4]), p[5]))
        self.lab_params[name] = v
        if p[2] == "model":
            for slot in self.flies:
                br = slot.brain
                lab.apply_to_sim(br.sim, {name: v})
                if name == "gain_adapt":
                    br._gain_adapt = v
                    if br.dead:
                        br.sim.p.gain_adapt = 0.0            # a dead brain's gain stays frozen until it revives
        else:
            lab.apply_rules(self.lab_params)

    def sync_time(self) -> None:
        """Brain threads follow game time: paused, slow motion, training speed-up; single steps are fed in."""
        self.poll_load()
        base = self.clock.brain_speed()
        steps = self.clock.take_brain_steps()
        rec = getattr(self, "recording", None)
        if rec is not None and (rec["slot"].brain.steps >= rec["until"] or rec["slot"] not in self.flies):
            self.stop_recording()
        elif rec is not None:
            self.sample_kinematics(rec)
        training = self.flies[self.focus] if (self.train is not None or self.challenge is not None) else None
        fast = self.train_active_speed if self.train is not None else getattr(self.challenge, "speed", 1.0)
        for slot in self.flies:
            want = base * (fast if slot is training else 1.0)
            if slot.brain.speed != want:
                slot.brain.speed = want
            if steps:
                slot.brain.request_steps(steps)

    def sample_kinematics(self, job: dict) -> None:
        """Where the recorded fly is, in game units (2D pixels, 3D metres), sampled once per frame."""
        fly = job["slot"].fly
        pos = np.asarray(fly.p, float).mean(axis=0)
        prev = np.asarray(fly.prev, float).mean(axis=0)
        speed = float(np.linalg.norm(pos - prev)) * 60.0        # the ragdoll integrates in fixed 1/60 s ticks
        x, y, z = (pos[0], pos[1], 0.0) if len(pos) == 2 else (pos[0], pos[1], pos[2])
        job["rec"].log_kinematics(x, y, z, speed, float(fly.health), str(fly.action), ARENAS[self.arena_i])

    def time_action(self, action: str) -> None:
        c = self.clock
        if action == "time_pause":
            c.user_paused = not c.user_paused
            self.note("TIME     paused" if c.user_paused else "TIME     running")
        elif action == "time_slower":
            self.set_setting("brain.sim_speed", c.slower())
        elif action == "time_faster":
            self.set_setting("brain.sim_speed", c.faster())
        elif action == "time_step":
            if not c.user_paused:
                c.user_paused = True
            c.step()
        if not c.label():                            # the badge under the health bar shows pause and slow motion
            self.saved_msg = ("time: normal speed", time.perf_counter())

    # --- fly properties: proxy to whichever FlySlot is currently focused --------------------------------------------
    @property
    def fly(self) -> Fly:
        return self.flies[self.focus].fly

    @property
    def brain(self) -> Brain:
        return self.flies[self.focus].brain

    @property
    def hits(self) -> int:
        return self.flies[self.focus].hits

    @property
    def reward(self) -> float:
        return self.flies[self.focus].reward

    @property
    def pain(self) -> float:
        return self.flies[self.focus].pain

    @property
    def pain_parts(self) -> np.ndarray:
        return self.flies[self.focus].pain_parts

    @property
    def pain_peak(self) -> float:
        return self.flies[self.focus].pain_peak

    @property
    def pain_max_s(self) -> float:
        return self.flies[self.focus].pain_max_s

    @property
    def pain_trace(self) -> list:
        return self.flies[self.focus].pain_trace

    @property
    def is_larva(self) -> bool:
        return getattr(self, "brain_type", "adult") == "larva"      # the brain loaded, not the setting (playable_brain)

    def _new_primary_fly(self) -> Fly:
        """Overridden by Game3D to build a Fly3D positioned in front of the player instead."""
        if self.is_larva:
            from kickthefly.game.larva import LarvaBody
            return LarvaBody(PLAY_W / 2)
        return Fly(PLAY_W / 2)

    def _new_spawn_fly(self) -> Fly:
        """Overridden by Game3D to build a Fly3D. Called on spawn_fly()'s background thread."""
        if self.is_larva:
            from kickthefly.game.larva import LarvaBody
            return LarvaBody(random.uniform(120, PLAY_W - 120))
        return Fly(random.uniform(120, PLAY_W - 120))

    def new_fly(self) -> None:
        """R: stop every fly spawned with N and give the original, persistent-brain fly a fresh body."""
        for slot in getattr(self, "flies", [])[1:]:
            slot.brain.stop()
        primary = self.flies[0].brain if getattr(self, "flies", None) else self._primary_brain
        base_seed = int(self.cfg["brain.seed"])
        if getattr(primary, "seed", base_seed) != base_seed:
            primary.reseed(base_seed)
        random.seed(base_seed)                        # the game's own randomness follows the seed too
        np.random.seed(base_seed % 2**32)
        crash.info["seed"] = str(base_seed)
        primary.sedation = 0.0
        primary.clear_overrides()
        if primary.dead:
            primary.revive()
        self.flies: list[FlySlot] = [FlySlot(self._new_primary_fly(), primary, seed=base_seed, primary=True)]
        self.focus = 0
        self._next_seed = base_seed + 1
        self._spawning = False
        self._new_slot = None
        self._reset_gen += 1
        self.log: list[tuple[float, str, str | None]] = []
        self.report: dict | None = None
        self.death_frames: list = []
        self.clear_transients()
        self.sugars: list[dict] = []
        self.alcohols: list[dict] = []
        self.decoys: list[dict] = []
        self.grabbed_decoy = None
        self.surgery_modes = [0] * len(SURGERY)
        self.type_ops = {}
        if getattr(self, "arena_i", 0) < len(ARENAS) and ARENAS[self.arena_i] == "escaperoom":
            self.reset_escaperoom()

    def arena_status(self) -> list[str]:
        """Extra HUD notes for the arena (the 3D game adds wind, fruit and lost flies outdoors)."""
        return []

    def recall_flies(self, now: float) -> None:
        """The 2D game has walls: no fly can get lost in it. The 3D game overrides this for its outdoor arenas."""
        self.note("RECALL   only outdoor arenas (3D) can lose a fly", source="rule")

    def on_arena_changed(self, now: float) -> None:
        """Whatever a new arena needs set up. The 3D game overrides this to build its outdoor worlds."""
        self.decoys = []
        self.grabbed_decoy = None
        if ARENAS[self.arena_i] == "escaperoom":
            self.reset_escaperoom(now=now)

    def reset_escaperoom(self, now: float | None = None, seed: int | None = None) -> None:
        # the game clock doesn't start at 0: without a time the timer used to show the machine's uptime
        self.escaperoom_start_t = self.clock.now if now is None else now
        self.escaperoom_seed = seed if seed is not None else int(self.cfg.get("brain.seed", 0))
        self.escaperoom_completed = False
        self.escaperoom_finish_t = 0.0
        self.escaperoom_code = ""
        if getattr(self, "flies", None):
            fly = self.flies[0].fly
            fly.stuck.clear()
            if getattr(fly.p, "shape", [0, 0])[1] == 3:
                shift_x = (-1.8 + 0.9) - fly.p[HEAD, 0]
            else:
                shift_x = 130.0 - fly.p[HEAD, 0]
            fly.p[:, 0] += shift_x
            fly.prev[:, 0] += shift_x
            fly.hover[0] += shift_x
            fly.fly_target[0] += shift_x

    def clear_transients(self) -> None:
        """Things in flight and effects, which save states don't keep."""
        self.popups: list[list] = []
        self.bombs: list[dict] = []
        self.flashes: list[list] = []
        self.swats: list[list] = []
        self.dust: list[list] = []
        self.shake_until = 0.0
        self.killed_by = ""
        self.torching = False
        self.flames: list[list] = []
        self.mist: list[list] = []
        self.shards: list[list] = []
        self.bolts: list[list] = []
        self.spider: dict | None = None
        from kickthefly.game.predator_play import PredatorPlay
        self.preds = PredatorPlay(self, False, FLOOR, CEIL, PLAY_W)       # 3.0 day 3: frog, dragonfly, mantis
        self.zap_ready = 0.0
        self.streaks: list[list] = []

    def spawn_fly(self) -> None:
        """N: add another fly, each with its own fully independent connectome brain thread, up to max_flies. The
        brain build + warm-up (~600 steps) runs on a background thread so it never hitches a frame."""
        max_f = getattr(self, "max_flies", MAX_FLIES)
        if self._spawning or len(self.flies) >= max_f or self.graph is None or self.weights is None:
            return
        free = platform_env.available_memory_mb()
        if free is not None and free < 2 * BRAIN_MB:     # each fly holds its own copy of the connectome
            self.note(f"N        no room for another brain: each needs about {BRAIN_MB} MB, {free:,.0f} MB free",
                      source="rule")
            return
        self._spawning = True
        seed = self._next_seed
        self._next_seed += 1
        gen = self._reset_gen

        def build() -> None:
            try:
                new_brain = self.build_brain(seed)
            except Exception:                   # never leave N dead for the rest of the session
                log.exception("building fly #%d's brain failed", seed)
                self._spawning = False
                return
            if gen != self._reset_gen:          # R was pressed while this build was in flight: discard it
                new_brain.stop()
                return
            new_brain.start()
            self._new_slot = FlySlot(self._new_spawn_fly(), new_brain, seed=seed, primary=False)

        threading.Thread(target=build, name=f"brain-{seed}", daemon=True).start()

    def build_brain(self, seed: int) -> "Brain":
        """A fresh, warmed-up, not-yet-started brain for another fly (its own LIFSim and mushroom body)."""
        from kickthefly.lab import lab
        from kickthefly.sim.connectome.sim import LIFParams, LIFSim

        lif_params = LIFParams()
        if hasattr(self, "cfg") and self.cfg:
            lif_params.backend = str(self.cfg.get("brain.backend", "auto"))
            lif_params.dtype = str(self.cfg.get("brain.dtype", "float32"))
            lif_params.individuality = str(self.cfg.get("brain.individuality", "off"))
        sim = LIFSim(None, lif_params, W_in=self.weights, seed=seed)
        lab.apply_to_sim(sim, self.lab_params)
        new_brain = Brain(self.graph, sim, seed=seed)
        new_brain.set_pain_level(self.pain_level)
        new_brain.graph = self.graph
        if getattr(self.graph, "dan_mbon", None) is not None:
            from kickthefly.core import memory
            new_brain.memory = memory.Memory(self.graph, sim)
            # it still loads the primary's saved weights as a starting point and learns live from there, but
            # never writes back to that shared file (no matter which code path calls .save(), now or later)
            new_brain.memory.save = lambda: None
        if not self.wiring.is_identity:               # a fly spawned under a Lab wiring gets the same connectome
            from kickthefly.sim import wiring as wiring_mod

            wiring_mod.apply(new_brain, self.wiring, self.graph)
        new_brain.warmup()
        return new_brain

    # --- Lab: changing the connectome itself -----------------------------------------------------------------------
    def set_wiring(self, w, note: bool = True) -> None:
        """Rebuild every live fly's synapse matrix under a new wiring (threshold, sign flips, inhibition block).

        Runs on a worker thread: rebuilding the column copy of 10.3M synapses takes about a second per fly, and the
        brains keep stepping meanwhile, each one pausing only for its own swap.
        """
        from kickthefly.sim import wiring as wiring_mod

        if self.wiring_busy:
            return
        self.wiring = w
        self.wiring_busy = w.label()
        self._flipped_set = frozenset(w.flip_rows)

        def work():
            try:
                for slot in list(self.flies):
                    br = slot.brain
                    with br.step_lock:
                        wiring_mod.apply(br, w, self.graph)
                if note:
                    self.note(f"WIRING   {w.label()}")   # like SURGERY and ARENA: a setting, not a reaction
            except Exception as e:
                log.exception("applying the wiring failed")
                self.menu.flash(f"wiring failed: {e}", menu_ui.BAD)
            finally:
                self.wiring_busy = ""

        threading.Thread(target=work, name="wiring", daemon=True).start()

    # --- save states -------------------------------------------------------------------------------------------------
    def save_extra(self, arrays: dict, now: float) -> dict:
        return dict(sugars=[dict(p=[float(x) for x in sg["p"]], v=float(sg.get("v", 0.0)), left=float(sg["left"]))
                            for sg in self.sugars],
                    alcohols=[dict(p=[float(x) for x in al["p"]], v=float(al.get("v", 0.0)), left=float(al["left"]))
                              for al in self.alcohols],
                    decoys=[dict(p=[float(x) for x in dec["p"]], v=[float(x) for x in dec.get("v", [0.0, 0.0])],
                                 t=float(dec.get("t", now)))
                            for dec in getattr(self, "decoys", [])])

    def load_extra(self, extra: dict, z, now: float) -> None:
        self.sugars = [dict(p=np.array(sg["p"]), v=sg["v"], left=sg["left"]) for sg in extra.get("sugars", [])]
        self.alcohols = [dict(p=np.array(al["p"]), v=al["v"], left=al["left"]) for al in extra.get("alcohols", [])]
        self.decoys = [dict(p=np.array(dec["p"], float), v=np.array(dec.get("v", [0.0, 0.0]), float),
                            t=float(dec.get("t", now)))
                       for dec in extra.get("decoys", [])]
        self.grabbed_decoy = None

    def prepare_load(self, n: int, seeds: list[int]) -> None:
        """Match the number of flies to a save (using brains built for it in the background) and clear effects."""
        if self.train is not None:
            self.stop_training()
        self._reset_gen += 1                       # a spawn still building is discarded
        self._spawning, self._new_slot = False, None
        while len(self.flies) > n:
            self.flies.pop().brain.stop()
        built = list(getattr(self, "_load_brains", []))
        for seed in seeds[len(self.flies):]:
            br = built.pop(0) if built else self.build_brain(seed)
            br.speed = 0.0
            br.start()
            self.flies.append(FlySlot(self._new_spawn_fly(), br, seed=seed, primary=False))
        for br in built:
            br.stop()
        self._load_brains = []
        self.report = None
        self.death_frames = []
        self.clear_transients()

    def save_state(self) -> Path | None:
        from kickthefly.core import savestate
        d = paths.ensure_dir(paths.get().saves_dir)
        stem = f"save-{time.strftime('%Y%m%d-%H%M%S')}"
        path, n = d / f"{stem}{savestate.SUFFIX}", 2
        while path.exists():
            path, n = d / f"{stem}-{n}{savestate.SUFFIX}", n + 1
        try:
            savestate.save_game(self, path)
        except Exception as e:
            log.exception("save state failed")
            self.menu.flash(f"Couldn't save: {e}", menu_ui.BAD)
            return None
        self.menu.flash(f"Saved {path.name}", menu_ui.GOOD)
        log.info("saved state %s", path)
        return path

    def load_state(self, path: Path, wait: bool = False) -> bool:
        """Load a save. Brains for extra flies are built on a background thread first (a few seconds each), with
        the menu showing progress; wait=True does it all now (tests, headless)."""
        from kickthefly.core import savestate
        try:
            meta = savestate.read_meta(path)
            why = savestate.compatible(meta, self)
            if why:
                raise savestate.SaveError(f"can't load this save: {why}")
        except savestate.SaveError as e:
            self.menu.flash(str(e), menu_ui.BAD, 6)
            return False
        need = [f["seed"] for f in meta["flies"]][len(self.flies):]
        if need and not wait:
            job = dict(path=path, brains=[], total=len(need), error=None)
            self._load_job = job
            self.menu.flash(f"Loading: building {len(need)} brain{'s' if len(need) > 1 else ''}...", menu_ui.AMBER, 60)

            def build():
                try:
                    for seed in need:
                        job["brains"].append(self.build_brain(seed))
                except Exception as e:
                    job["error"] = e
                job["done"] = True

            threading.Thread(target=build, name="load-brains", daemon=True).start()
            return True
        return self._finish_load(path)

    def poll_load(self) -> None:
        job = getattr(self, "_load_job", None)
        if job is None or not job.get("done"):
            return
        self._load_job = None
        if job["error"] is not None:
            self.menu.flash(f"Couldn't load: {job['error']}", menu_ui.BAD, 6)
            return
        self._load_brains = job["brains"]
        self._finish_load(job["path"])

    def _finish_load(self, path: Path) -> bool:
        from kickthefly.core import savestate
        try:
            savestate.load_game(self, path)
        except savestate.SaveError as e:
            self.menu.flash(str(e), menu_ui.BAD, 6)
            return False
        except Exception as e:
            log.exception("load state failed")
            self.menu.flash(f"Couldn't load: {type(e).__name__}: {e}", menu_ui.BAD, 6)
            return False
        self.note(f"LOADED   {path.name}")
        for slot in self.flies:
            slot.brain.meter_reset = True
        self.menu.message = None
        if self.menu.open:
            self.menu.close()
        return True

    def _poll_spawn(self) -> None:
        if self._new_slot is None:
            return
        slot, self._new_slot, self._spawning = self._new_slot, None, False
        self.flies.append(slot)
        self.focus = len(self.flies) - 1
        self.popup(slot.fly.p[HEAD] + (0, -60), "NEW FLY!", (170, 255, 200), force=True)
        self.note(f"SPAWNED  fly #{len(self.flies)} (seed {slot.seed})")

    def _you_pos(self) -> np.ndarray:
        """Reference point for 'closest to you'; Game3D overrides this with the player's eye."""
        return np.asarray(self.mouse, float)

    def cycle_focus(self, step: int = 1) -> None:
        """Cycle focus to the next/prev fly. Overrides auto-nearest-tracking for 5 seconds."""
        if len(self.flies) > 1:
            self.focus = (self.focus + step) % len(self.flies)
            self._manual_focus_until = time.perf_counter() + 5.0
            slot = self.flies[self.focus]
            from kickthefly.core import cards

            self.note(f"FOCUS    fly #{self.focus + 1}/{len(self.flies)} [{cards.label(getattr(slot, 'personality', None))}]")

    def _update_focus(self) -> None:
        """The brain panel (and whichever fly training/surgery act on) always follows the fly nearest to you,
        except while a panel is open or mid-training/duel, where switching brains under the player would be
        confusing or apply an action to the wrong fly."""
        if len(self.flies) <= 1 or getattr(self, "duel", False) or self._overlay_open() or self.train is not None \
                or getattr(self, "challenge", None) is not None \
                or all(s.fly.dead for s in self.flies) \
                or getattr(self, "_manual_focus_until", 0.0) > time.perf_counter():
            return
        you = self._you_pos()
        self.focus = min(range(len(self.flies)),
                          key=lambda i: float(np.linalg.norm(self.flies[i].fly.p[THX] - you)))

    def _view_loop(self) -> None:
        """Renders the brain view at ~20 Hz on its own thread (10-25 ms per render; the sparse math releases the GIL)."""
        while not self.view_stop:
            t0 = time.perf_counter()
            key = "big" if self.big_view else "panel"
            br = self.brain
            calm = not br.dead and br.sedation == 0 and br.steps - br.last_poke > CALM_STEPS
            raster = br.sim.activity.raster()
            spiked = raster[-1] if raster else np.zeros(0, np.int64)
            replay = self.x3.view_rates()            # 3.0: while the kill cam plays, the recorded frame, not the live brain
            img = self.imaging_live
            if replay is not None:
                self.view_surf[key] = self.view.render(key, replay[0], replay[1], t0 - self.born_view, learn=False)
            elif img.on:                             # 3.0 day 2: Imaging mode, the view shows simulated dF/F (MODEL)
                try:
                    img.feed(br)
                    rates = img.view_rates(self.view, bool(self.cfg["access.reduced_flashing"]))
                    if rates is None:
                        raise RuntimeError("no imaging session")
                    surf = self.view.render(key, rates, np.zeros(0, np.int64), t0 - self.born_view, learn=False)
                    self.view_surf[key] = img.recolor(self.view, surf, self.cfg["access.palette"],
                                                      bool(self.cfg["access.reduced_flashing"]))
                except Exception as e:
                    img.on, img.error = False, f"imaging stopped: {e}"
            else:
                self.view_surf[key] = self.view.render(key, br.sim.activity.rates(), spiked, t0 - self.born_view, learn=calm)
            time.sleep(max(0.005, 0.05 - (time.perf_counter() - t0)))

    def _view_surface(self, key: str) -> pygame.Surface:
        surf = self.view_surf.get(key)
        if surf is None:
            surf = pygame.Surface(VIEW_SIZES[key])
            surf.fill((4, 5, 8))
        return surf

    def _hud_overlay(self, surf, rect: pygame.Rect, now: float, small: bool) -> None:
        """Sci-fi dressing over a brain view: a sweeping scan band, corner brackets and live counters."""
        self.view.sparkle = not self.calm_fx
        ph = (now * 0.22) % 1.0
        band_y = rect.y + int(ph * rect.h)
        band = pygame.Surface((rect.w, 34))
        for k in range(34):                          # additive glow that fades in toward the scan line
            a = 0.16 * (k / 33) ** 2
            band.fill((int(80 * a), int(200 * a), int(255 * a)), (0, k, rect.w, 1))
        clip = surf.get_clip()
        surf.set_clip(rect)
        if not self.calm_fx:
            surf.blit(band, (rect.x, band_y - 34), special_flags=pygame.BLEND_RGB_ADD)
            pygame.draw.line(surf, (60, 120, 150), (rect.x, band_y), (rect.right, band_y), 1)
        surf.set_clip(clip)
        L = 10 if small else 22
        for cx, cy, sx, sy in ((rect.x, rect.y, 1, 1), (rect.right - 1, rect.y, -1, 1),
                               (rect.x, rect.bottom - 1, 1, -1), (rect.right - 1, rect.bottom - 1, -1, -1)):
            pygame.draw.line(surf, ACCENT, (cx, cy), (cx + sx * L, cy), 2)
            pygame.draw.line(surf, ACCENT, (cx, cy), (cx, cy + sy * L), 2)
        v = self.view
        if self.imaging_live.on:
            ind = self.imaging_live.session.ind.name if self.imaging_live.session else ""
            self._text(surf, f"IMAGING (MODEL)  {ind.split(' (')[0]}  dF/F" if not small else "IMAGING (MODEL)",
                       (rect.x + 6, rect.bottom - 18), (170, 255, 190), self.f_small)
        if small and not self.imaging_live.on:     # 3.0 day 2 review: in Imaging mode the view is fed dF/F, so these would
            self._text(surf, f"{v.firing:,} firing", (rect.x + 6, rect.y + 4), (150, 215, 240), self.f_small)  # count pixels
            self._text(surf, f"{v.hot_firing:,} pain", (rect.right - 6, rect.y + 4), (255, 150, 70), self.f_small, "topright")

    def _draw_big_view(self, surf) -> None:
        w, h = VIEW_SIZES["big"]
        now = time.perf_counter()
        veil = pygame.Surface((PLAY_W, h + 120), pygame.SRCALPHA)
        pygame.draw.rect(veil, (3, 4, 7, 252), veil.get_rect(), border_radius=14)
        surf.blit(veil, (0, 0))
        rect = pygame.Rect((PLAY_W - w) // 2, 58, w, h)
        self.big_rect = rect
        surf.blit(self._view_surface("big"), rect)
        self._hud_overlay(surf, rect, now, small=False)
        self.x3.draw_killcam_rings(surf, rect, "big", now)
        self._clean_frame = surf.copy()
        if getattr(self, "photo_mode", False):
            b_txt = self.f_small.render("PHOTO MODE  |  S / F12: Clean Snap  |  F10: Exit", True, (240, 240, 240))
            box = b_txt.get_rect(midbottom=(rect.centerx, rect.bottom - 16)).inflate(24, 8)
            pygame.draw.rect(surf, (12, 16, 24, 200), box, border_radius=6)
            pygame.draw.rect(surf, (70, 80, 100, 180), box, 1, border_radius=6)
            surf.blit(b_txt, b_txt.get_rect(center=box.center))
            return
        self._draw_paths(surf, rect, now)
        if self.inspect is not None:
            self._draw_inspect(surf, rect, now)
        self._draw_search(surf, rect)
        labels = (("optic lobe", 0.10, 0.18), ("optic lobe", 0.90, 0.18), ("mushroom bodies", 0.50, 0.06),
                  ("central brain", 0.50, 0.42), ("to nerve cord", 0.50, 0.93))
        for label, fx, fy in labels if (self.inspect is None and self.view.is_default_view()
                                        and not getattr(self, "path_ends", None)) else ():
            self._text(surf, label.upper(), (rect.x + int(fx * w), rect.y + int(fy * h)), (120, 170, 190), self.f_small, "center")
        self._text(surf, "LIVE CONNECTOME", (22, 16), INK, self.f_head)
        v = self.view
        pulse = 1.0 if self.calm_fx else 0.5 + 0.5 * math.sin(now * 6)
        # Header: title and the recording/stethoscope/mode controls on the first row, the live counts and the camera
        # presets on the second; status and help lines go under the brain so nothing overlaps.
        if self.imaging_live.on:                     # 3.0 day 2 review: not firing counts while the view shows dF/F
            r = self._text(surf, "dF/F (MODEL)", (24, 38), (170, 255, 190), self.f_bold)
        else:
            r = self._text(surf, f"{v.firing:,} firing", (24, 38), (150, 215, 240), self.f_bold)
            self._text(surf, f"{v.hot_firing:,} pain", (r.right + 18, 38), tuple(int(c * (0.7 + 0.3 * pulse)) for c in (255, 150, 70)), self.f_bold)

        # Time-lapse recording button:
        tl_btn = pygame.Rect(232, 13, 105, 22)
        rec_active = getattr(self, "timelapse_recording", False)
        sp = int(self.cfg.get("graphics.timelapse_speedup", 5))
        pygame.draw.rect(surf, (65, 25, 25) if rec_active else (24, 28, 38), tl_btn, border_radius=4)
        pygame.draw.rect(surf, (255, 90, 90) if rec_active else BORDER, tl_btn, 1, border_radius=4)
        tl_txt = "■ STOP (L)" if rec_active else f"● REC {sp}x"
        self._text(surf, tl_txt, tl_btn.center, (255, 120, 120) if rec_active else (230, 180, 180), self.f_small, "center")
        self.big_timelapse_button = tl_btn

        # Stethoscope buttons:
        steth_target_btn = pygame.Rect(tl_btn.right + 8, 13, 145, 22)
        steth_tgt_lbl = self.stethoscope_target_label()
        if len(steth_tgt_lbl) > 13:
            steth_tgt_lbl = steth_tgt_lbl[:12] + "…"
        pygame.draw.rect(surf, (24, 28, 38), steth_target_btn, border_radius=4)
        pygame.draw.rect(surf, BORDER, steth_target_btn, 1, border_radius=4)
        self._text(surf, f"Probe: {steth_tgt_lbl} ▾", steth_target_btn.center, (170, 205, 235), self.f_small, "center")
        self.big_steth_target_button = steth_target_btn

        steth_btn = pygame.Rect(steth_target_btn.right + 8, 13, 125, 22)
        steth_active = bool(self.cfg["audio.stethoscope_enabled"])
        pygame.draw.rect(surf, (30, 65, 40) if steth_active else (24, 28, 38), steth_btn, border_radius=4)
        pygame.draw.rect(surf, (80, 220, 120) if steth_active else BORDER, steth_btn, 1, border_radius=4)
        steth_txt = f"Steth: {'ON' if steth_active else 'OFF'} (K)"
        self._text(surf, steth_txt, steth_btn.center, (140, 255, 170) if steth_active else TEXT, self.f_small, "center")
        self.big_steth_button = steth_btn

        # Camera presets & status:
        self.big_preset_buttons = []
        bx = PLAY_W - 22 - (4 * 58 + 3 * 6)
        for name, key_label in (("front", "1:Front"), ("side", "2:Side"), ("top", "3:Top"), ("reset", "0:Reset")):
            btn_rect = pygame.Rect(bx, 37, 58, 20)
            active = (v.preset == name) if name != "reset" else False
            bg = (45, 65, 95) if active else (24, 28, 38)
            border_c = ACCENT if active else BORDER
            pygame.draw.rect(surf, bg, btn_rect, border_radius=4)
            pygame.draw.rect(surf, border_c, btn_rect, 1, border_radius=4)
            self._text(surf, key_label, btn_rect.center, INK if active else TEXT, self.f_small, "center")
            self.big_preset_buttons.append((btn_rect, name))
            bx += 64
        if self.cfg["brain.autopilot"] and self.cfg["brain.autopilot_orbit"] and not getattr(self, "big_drag", None):
            t = time.perf_counter()
            dt = t - getattr(self, "_last_big_orbit", t)
            self._last_big_orbit = t
            if 0 < dt < 0.2:
                self.view.orbit(dt * 12.0, 0.0)
        else:
            self._last_big_orbit = time.perf_counter()

        cam_info = f"{v.preset.upper()}  yaw {v.yaw:+.0f}°  pitch {v.pitch:+.0f}°  zoom {v.zoom:.1f}x"
        if self.cfg["brain.autopilot"]:
            cam_info += "  AUTOPILOT"
        self._text(surf, cam_info, (r.right + 150, 40), (150, 200, 225), self.f_small)

        # Mode toggle button:
        mode_btn = pygame.Rect(steth_btn.right + 8, 13, 130, 22)
        mode_label = "Mode: Per-Region" if v.view_mode == "region" else "Mode: Per-Neuron"
        mode_active = (v.view_mode == "region")
        pygame.draw.rect(surf, (50, 75, 110) if mode_active else (24, 28, 38), mode_btn, border_radius=4)
        pygame.draw.rect(surf, ACCENT if mode_active else BORDER, mode_btn, 1, border_radius=4)
        self._text(surf, mode_label, mode_btn.center, INK if mode_active else TEXT, self.f_small, "center")
        self.big_mode_button = mode_btn

        self.big_region_buttons = []
        if v.view_mode == "region":
            leg_w, leg_h = 206, 316
            leg_rect = pygame.Rect(rect.right - leg_w - 6, rect.y + 6, leg_w, leg_h)
            card = pygame.Surface((leg_w, leg_h), pygame.SRCALPHA)
            pygame.draw.rect(card, (10, 12, 18, 220), card.get_rect(), border_radius=8)
            pygame.draw.rect(card, BORDER, card.get_rect(), 1, border_radius=8)
            surf.blit(card, leg_rect)
            self._text(surf, "NEUROPIL REGIONS", (leg_rect.x + 8, leg_rect.y + 6), INK, self.f_bold)
            self._text(surf, "click to view neuron list", (leg_rect.x + 8, leg_rect.y + 22), DIM, self.f_small)

            for idx, rname in enumerate(v.region_names):
                ry = leg_rect.y + 38 + idx * 22
                row_r = pygame.Rect(leg_rect.x + 4, ry, leg_w - 8, 20)
                col = REGION_COLORS.get(rname, (180, 180, 180))
                pygame.draw.circle(surf, col, (row_r.x + 8, row_r.centery), 4)
                rate = v.region_rates[idx]
                rshort = rname if len(rname) <= 14 else rname[:13] + "…"
                self._text(surf, rshort, (row_r.x + 18, ry + 2), TEXT, self.f_small)
                self._text(surf, f"{rate:4.1f} Hz", (row_r.right - 4, ry + 2), (160, 220, 255), self.f_small, "topright")
                self.big_region_buttons.append((row_r, rname))

        if getattr(self, "selected_region", None):
            self._draw_region_neuron_list(surf, rect, now)

        ly = rect.bottom + 6
        if v.view_mode == "region":
            self._text(surf, "click a region above to browse neurons  |  colors: dataset region labels  |  glow: live region activity heat",
                       (30, ly), TEXT, self.f_small)
            self._text(surf, "Neurons without region annotations in MaleCNS v1.0 are kept in an explicit 'unassigned' bucket.",
                       (30, ly + 18), DIM, self.f_small)
        elif self.imaging_live.on:                   # 3.0 day 2 review: the firing legend's colors are not what Imaging shows
            self._text(surf, "IMAGING (MODEL): brightness = simulated GCaMP dF/F of the ROI neurons; the rest stay dark",
                       (30, ly), TEXT, self.f_small)
            self._text(surf, "a forward model on the simulation's spikes, not a measurement", (30, ly + 18), DIM, self.f_small)
        else:
            hot_c, cool_c = self.view.legend
            aacircle(surf, (30, ly + 7), 5, hot_c)
            r = self._text(surf, "pain-sensing neurons firing", (42, ly), TEXT, self.f_small)
            aacircle(surf, (r.right + 22, ly + 7), 5, cool_c)
            r = self._text(surf, "other neurons firing", (r.right + 34, ly), TEXT, self.f_small)
            self._text(surf, "dim wiring: fiber direction, front brighter", (r.right + 22, ly), LABEL, self.f_small)
            skel_status = getattr(v, "skeleton_status", "Skeletons offline - using synthetic fibers")
            real = skel_status.startswith("Real")
            self._text(surf, skel_status + ("; every other fiber is estimated" if real else ""), (24, ly + 18),
                       (130, 220, 180) if real else (210, 160, 120), self.f_small)
        self._text(surf, "click: inspect, then PATH FROM/TO · drag: orbit · Shift+drag: pan · wheel: zoom · K: stethoscope "
                   "(GAME RULE) · B: close", (24, ly + 36), LABEL, self.f_small)

    def _draw_region_neuron_list(self, surf, rect: pygame.Rect, now: float) -> None:
        rname = self.selected_region
        v = self.view
        br = self.brain
        if rname not in v.region_to_id:
            self.selected_region = None
            return
        rid = v.region_to_id[rname]
        indices = np.flatnonzero(v.region_id == rid)
        rates = br.sim.activity.rates() * 200.0

        mw, mh = min(600, rect.w - 30), min(420, rect.h - 30)
        mrect = pygame.Rect(rect.centerx - mw // 2, rect.centery - mh // 2, mw, mh)
        self.region_modal_rect = mrect
        modal = pygame.Surface((mw, mh), pygame.SRCALPHA)
        pygame.draw.rect(modal, (10, 14, 22, 252), modal.get_rect(), border_radius=10)
        pygame.draw.rect(modal, ACCENT, modal.get_rect(), 1, border_radius=10)
        surf.blit(modal, mrect)

        col = REGION_COLORS.get(rname, (200, 200, 200))
        pygame.draw.circle(surf, col, (mrect.x + 20, mrect.y + 20), 6)
        self._text(surf, f"{rname.upper()} ({len(indices):,} neurons, mean: {v.region_rates[rid]:.1f} Hz)",
                   (mrect.x + 34, mrect.y + 12), INK, self.f_bold)

        close_r = pygame.Rect(mrect.right - 80, mrect.y + 10, 70, 22)
        pygame.draw.rect(surf, (40, 44, 56), close_r, border_radius=4)
        pygame.draw.rect(surf, BORDER, close_r, 1, border_radius=4)
        self._text(surf, "Close", close_r.center, TEXT, self.f_small, "center")
        self.region_close_button = close_r

        page = getattr(self, "region_page", 0)
        per_page = 11
        max_page = max(0, (len(indices) - 1) // per_page)
        page = min(page, max_page)
        self.region_page = page

        prev_r = pygame.Rect(mrect.x + 20, mrect.bottom - 34, 64, 22)
        next_r = pygame.Rect(mrect.x + 94, mrect.bottom - 34, 64, 22)
        pygame.draw.rect(surf, (35, 40, 52), prev_r, border_radius=4)
        pygame.draw.rect(surf, (35, 40, 52), next_r, border_radius=4)
        self._text(surf, "Prev", prev_r.center, TEXT if page > 0 else DIM, self.f_small, "center")
        self._text(surf, "Next", next_r.center, TEXT if page < max_page else DIM, self.f_small, "center")
        self.region_prev_btn, self.region_next_btn = prev_r, next_r
        self._text(surf, f"Page {page + 1} / {max_page + 1}", (mrect.right - 20, mrect.bottom - 28), LABEL, self.f_small, "topright")

        y = mrect.y + 44
        self._text(surf, "Neuron ID", (mrect.x + 20, y), LABEL, self.f_small)
        self._text(surf, "Type", (mrect.x + 130, y), LABEL, self.f_small)
        self._text(surf, "Instance", (mrect.x + 290, y), LABEL, self.f_small)
        self._text(surf, "Rate", (mrect.right - 30, y), LABEL, self.f_small, "topright")
        pygame.draw.line(surf, (40, 45, 58), (mrect.x + 15, y + 16), (mrect.right - 15, y + 16))

        self.region_neuron_buttons = []
        start_i = page * per_page
        for row_k, idx in enumerate(indices[start_i:start_i + per_page]):
            ry = y + 22 + row_k * 24
            row_rect = pygame.Rect(mrect.x + 15, ry - 2, mrect.w - 30, 22)
            bg = (24, 28, 40) if row_k % 2 == 0 else (18, 22, 32)
            pygame.draw.rect(surf, bg, row_rect, border_radius=3)
            ntype = br.types[idx] or "untyped"
            ninst = br.instance[idx] or "-"
            nrate = rates[idx]
            self._text(surf, f"#{idx}", (row_rect.x + 6, ry + 2), (180, 210, 240), self.f_small)
            self._text(surf, ntype[:18], (row_rect.x + 115, ry + 2), TEXT, self.f_small)
            self._text(surf, ninst[:20], (row_rect.x + 275, ry + 2), DIM, self.f_small)
            self._text(surf, f"{nrate:5.1f} Hz", (row_rect.right - 6, ry + 2), (140, 220, 180), self.f_small, "topright")
            self.region_neuron_buttons.append((row_rect, int(idx)))

    # --- seeing, smelling, learning ---------------------------------------------
    def _overlay_open(self) -> bool:
        if getattr(self, "training_open", False):
            return True
        ch = getattr(self, "challenge", None)
        if ch is not None and ch.overlay:
            return True
        return self.report is not None or self.big_view or self.surgery_open or self.help_open or self.x3.kc_playing()

    def _threats(self, slot: "FlySlot", now: float, mouse) -> list:
        out = []
        name = TOOLS[self.tool][0]
        if not self.cfg["brain.autopilot"] and not self._overlay_open() and mouse[0] < PLAY_W and mouse[1] < FLOOR and name not in ("hand", "sugar", "fruit", "alcohol"):
            out.append(("cursor", np.array(mouse, float), CURSOR_SIZE.get(name, 16)))
        for sw in self.swats:
            ph = (now - sw[1]) / 0.55
            if ph < 0.3:                              # the swatter head on its way down (same path as draw_swatter)
                pos = np.array(sw[0], float)
                pivot = pos + (260.0, 330.0)
                d = pos - pivot
                ang = math.atan2(d[1], d[0]) - 0.9 * (1 - (ph / 0.3) ** 2)
                out.append((("swat", id(sw)), pivot + np.array([math.cos(ang), math.sin(ang)]) * float(np.hypot(*d)), 80.0))
        if self.spider is not None and self.spider["state"] in ("drop", "hunt"):     # as in the 3D game
            out.append(("spider", self.spider["p"].copy(), 22.0))
        out.extend(self.preds.threats())                                   # 3.0 day 3: the same looming measure
        for b in self.bombs:
            out.append((("bomb", id(b)), b["p"].copy(), 16.0))
        for other in self.flies:                      # other flies loom too: a real, symmetric dodge reaction
            if other is slot or other.fly.dead:
                continue
            out.append((("fly", id(other.fly)), other.fly.p[THX].copy(), FLY_TOUCH_RADIUS))
        return out

    def _vision(self, slot: "FlySlot", now: float, mouse) -> None:
        """Looming: how fast each object's angular size grows as seen from the fly's head. The game computes this and
        drives the real looming detectors LPLC2/LC4, which excite the giant fiber in the connectome (see docstring)."""
        fly = slot.fly
        if fly.dead or fly.frozen_at is not None:
            slot.loom = 0.0
            return
        head = fly.p[HEAD]
        best, best_pos, seen = 0.0, None, {}
        for key, pos, r in self._threats(slot, now, mouse):
            d = max(float(np.hypot(*(pos - head))), r + 4.0)
            theta = 2 * math.atan(r / d)
            prev = slot.loom_prev.get(key)
            seen[key] = theta
            if prev is not None and (theta - prev) * 60.0 > best:
                best, best_pos = (theta - prev) * 60.0, pos
        slot.loom_prev = seen
        slot.loom += (best - slot.loom) * 0.5
        strength = float(np.clip((best - LOOM_MIN) / LOOM_FULL, 0, 1))
        if strength > 0 and best_pos is not None:
            slot.brain.poke("loom", None, strength, recruit=0.6 * strength)
            slot.threat_x = float(best_pos[0])

    def _scents(self, slot: "FlySlot", now: float, mouse) -> None:
        """Each tool carries its own scent (6 of the 53 olfactory glomeruli; a game rule) that the fly smells up close."""
        fly = slot.fly
        slot.scent_now, slot.sugar_scent = None, False
        if fly.dead:
            return
        name = TOOLS[self.tool][0]
        name = "sugar" if name == "fruit" else name           # fruit smells like sugar: same glomeruli, same memory
        near = np.hypot(*(np.array(mouse, float) - fly.p[HEAD])) < SCENT_RANGE
        if not self._overlay_open() and mouse[0] < PLAY_W and near:
            slot.scent_now = name
            slot.brain.poke("scent", name, 0.3)
        if any(abs(sg["p"][0] - fly.p[HEAD, 0]) < 400 for sg in self.sugars):
            slot.sugar_scent = True
            slot.brain.poke("scent", "sugar", 0.3)
        if any(abs(al["p"][0] - fly.p[HEAD, 0]) < 400 for al in self.alcohols):
            slot.brain.poke("scent", "alcohol", 0.35)

    # --- real training (memory.py) --------------------------------------------------------------------------------
    def _learn(self, slot: "FlySlot", now: float) -> None:
        """Learning happens inside the brain thread on the real KC -> MBON synapses (memory.py). Here: remember what
        each smell's Kenyon cell pattern looks like, read the current memory, and autosave."""
        br, fly = slot.brain, slot.fly
        mem = br.memory
        if mem is None:
            slot.fear_now = slot.like_now = 0.0
            return
        if self.frame % 3 == 0:
            r = br.sim.activity.rates()
            for key in ([slot.scent_now] if slot.scent_now else []) + (["sugar"] if slot.sugar_scent else []):
                mem.observe(key, r)
            if (slot.scent_now or slot.sugar_scent) and not fly.dead:
                # recognise the smell by its stored Kenyon cell pattern and read that pattern's synapses
                slot.fear_now, slot.like_now = mem.memory_of(slot.scent_now) if slot.scent_now else mem.memory_of("sugar")
            else:
                slot.fear_now = slot.like_now = 0.0
        borrowed = getattr(self.challenge, "borrows_memory", False)
        if mem.dirty and slot.persist_memory and not borrowed and now - self.last_save > 30:
            self.last_save = now
            threading.Thread(target=mem.save, daemon=True).start()

    def memory_of(self, key: str) -> tuple[float, float]:
        return (0.0, 0.0) if self.brain.memory is None else self.brain.memory.memory_of(key)

    def start_training(self, kind: str, trials: int = 10) -> None:
        mem = self.brain.memory
        if mem is None or self.train is not None:
            return
        scent = self.train_scent
        if kind != "test" and scent not in mem.naive_mbon and not mem.log.get(scent):
            kind_first = "naive"                       # measure the untrained response first
        else:
            kind_first = None
        self.train = dict(scent=scent, kind=kind, trials=1 if kind == "test" else trials, done=0,
                          phase="naive" if kind_first else "odor", until=self.brain.steps + (400 if kind_first else 240),
                          hz=[], first=kind_first)
        self.train_active_speed = self.train_speed
        self.note(f"TRAINING {kind} on {scent}")

    def stop_training(self) -> None:
        if self.train is not None:
            self.note(f"TRAINING stopped ({self.train['done']}/{self.train['trials']})")
        self.train = None
        self.train_active_speed = 1.0
        if self.brain.memory is not None and self.flies[self.focus].persist_memory:
            threading.Thread(target=self.brain.memory.save, daemon=True).start()

    def _training_tick(self, now: float) -> None:
        """Lab-style conditioning, timed in brain steps (5 ms each) so it works at any sim speed:
        smell 1.2 s -> smell + shock or sugar 1.0 s -> rest 1.3 s, repeated; tests are 2 s of smell alone."""
        t, br = self.train, self.brain
        if t is None:
            return
        mem = br.memory
        scent, step = t["scent"], br.steps
        phase = t["phase"]
        if phase in ("naive", "odor", "pair", "test"):
            br.poke("scent", scent, 0.5)
            if self.frame % 3 == 0:
                r = br.sim.activity.rates()
                mem.observe(scent, r)
                if phase in ("naive", "test", "odor") and step > t["until"] - (300 if phase != "odor" else 180):
                    t["hz"].append(mem.mbon_response(r))
        if phase == "pair":
            if t["kind"] == "fear":
                br.poke("punish", None, 1.0)                      # the punishment dopamine neurons
                br.poke("legs", "L", 0.6)                         # and an electric shock through the legs
                br.poke("legs", "R", 0.6)
            else:
                br.poke("reward", None, 0.8)                      # the reward dopamine neurons
                br.poke("taste", None, 0.6)                       # and the taste of sugar
        if step < t["until"]:
            return
        hz = float(np.mean(t["hz"])) if t["hz"] else None
        if phase == "naive":
            if hz is not None:
                mem.naive_mbon[scent] = hz
            mem.record(scent, "naive", hz)
            t.update(phase="rest", until=step + 260, hz=[])
        elif phase == "odor":
            t.update(phase="pair", until=step + 200, odor_hz=hz, hz=[])
        elif phase == "pair":
            t["done"] += 1
            mem.record(scent, t["kind"], t.get("odor_hz"))
            t.update(phase="rest", until=step + 260, hz=[])
        elif phase == "test":
            mem.record(scent, "test", hz)
            t["done"] = 1
            t.update(phase="rest", until=step + 1, hz=[])
        elif phase == "rest":
            if t["first"]:
                t["first"] = None
                if t["kind"] == "test":
                    self.stop_training()
                    return
                t.update(phase="test" if t["kind"] == "test" else "odor", until=step + (400 if t["kind"] == "test" else 240))
            elif t["done"] >= t["trials"]:
                self.stop_training()
            else:
                t.update(phase="test" if t["kind"] == "test" else "odor", until=step + (400 if t["kind"] == "test" else 240))

    def _draw_training(self, surf) -> None:
        mem = self.brain.memory
        panel = pygame.Rect(40, 30, min(PLAY_W - 80, 820), min(H - 60, 610))
        veil = pygame.Surface((W, H), pygame.SRCALPHA)
        veil.fill((4, 5, 8, 150))
        surf.blit(veil, (0, 0))
        pygame.draw.rect(surf, (18, 21, 28), panel, border_radius=16)
        pygame.draw.rect(surf, BORDER, panel, 1, border_radius=16)
        x, y = panel.x + 22, panel.y + 14
        self._text(surf, "TRAINING", (x, y), INK, self.f_title)
        self._text(surf, "Pair a smell with a shock or with sugar. Dopamine then weakens the fly's real Kenyon cell to "
                         "output neuron synapses.", (x + 2, y + 46), LABEL, self.f_small)
        self.train_buttons = []
        if mem is None:
            self._text(surf, "Training needs the updated brain pack.", (x, y + 90), S_CRIT, self.f_bold)
            return
        # scent list
        ly = y + 72
        names = list(TOOL_NAMES)
        self._text(surf, "SMELL", (x, ly), LABEL, self.f_small)
        self._text(surf, "fear", (x + 150, ly), (240, 150, 60), self.f_small)
        self._text(surf, "liking", (x + 225, ly), (255, 150, 190), self.f_small)
        self._text(surf, "trials", (x + 300, ly), LABEL, self.f_small)
        ly += 18
        for name in names:
            fear, like = mem.memory_of(name)
            r = pygame.Rect(x - 6, ly - 3, 350, 22)
            sel = name == self.train_scent
            if sel:
                pygame.draw.rect(surf, (44, 50, 64), r, border_radius=6)
            self._text(surf, name, (x, ly), INK if sel else TEXT, self.f_text if sel else self.f_small)
            self._bar(surf, x + 150, ly + 4, 64, fear, (240, 150, 60))
            self._bar(surf, x + 225, ly + 4, 64, like, (255, 150, 190))
            n = sum(1 for e in mem.log.get(name, []) if e[1] in ("fear", "like"))
            self._text(surf, str(n), (x + 330, ly), LABEL, self.f_small, "topright")
            self.train_buttons.append((r, "select", name))
            ly += 22
        # learning curve for the selected smell
        cx, cy, cw, ch = x + 380, y + 90, panel.right - x - 380 - 22, 230
        pygame.draw.rect(surf, (12, 14, 20), (cx, cy, cw, ch), border_radius=8)
        log = [e for e in mem.log.get(self.train_scent, []) if e[1] != "naive"]
        self._text(surf, f"{self.train_scent}: memory after each trial", (cx + 10, cy + 8), INK, self.f_bold)
        lx = cx + 12
        for label, col in (("fear", (240, 150, 60)), ("liking", (255, 150, 190))):
            pygame.draw.rect(surf, col, (lx, cy + 35, 14, 4), border_radius=2)
            lx = self._text(surf, label, (lx + 20, cy + 28), TEXT, self.f_small).right + 16
        px0, py0, pw, ph = cx + 36, cy + 54, cw - 50, ch - 80
        for v in (0, 0.5, 1):
            gy = py0 + ph - v * ph
            pygame.draw.line(surf, (40, 44, 54), (px0, gy), (px0 + pw, gy))
            self._text(surf, f"{v:g}", (px0 - 8, gy - 7), LABEL, self.f_small, "topright")
        if len(log) >= 1:
            n = max(len(log), 2)
            for j, col in ((2, (240, 150, 60)), (3, (255, 150, 190))):
                pts = [(px0 + k * pw / (n - 1), py0 + ph - float(e[j]) * ph) for k, e in enumerate(log)]
                if len(pts) > 1:
                    pygame.draw.lines(surf, col, False, pts, 2)
                for p in pts:
                    aacircle(surf, p, 3, col)
            self._text(surf, f"{len(log)} trials and tests", (px0 + pw, py0 + ph + 6), LABEL, self.f_small, "topright")
        else:
            self._text(surf, "no trials yet", (px0 + pw // 2, py0 + ph // 2), DIM, self.f_small, "center")
        naive = mem.naive_mbon.get(self.train_scent)
        tests = [e for e in mem.log.get(self.train_scent, []) if e[4] is not None]
        sy = cy + ch + 12
        if naive is not None and tests:
            self._text(surf, "approach output neurons (MBONs) to this smell:", (cx, sy), TEXT, self.f_small)
            self._text(surf, f"{naive:.1f} spikes/s untrained  >  {tests[-1][4]:.1f} now", (cx, sy + 16), INK, self.f_small)
        self._text(surf, f"{mem.weakened_share():.1%} of {len(mem.w0):,} Kenyon cell synapses weakened",
                   (cx, sy + 36), LABEL, self.f_small)
        # buttons
        by = panel.bottom - 104
        t = self.train
        buttons = (("TRAIN FEAR x10", "fear", (170, 90, 40)), ("TRAIN LIKING x10", "like", (170, 70, 110)),
                   ("TEST", "test", (60, 90, 140)), (f"SPEED x{self.train_speed:g}", "speed", (60, 64, 76)),
                   ("WIPE MEMORY" if now_wipe_armed(self) else "wipe memory", "wipe", (130, 40, 40) if now_wipe_armed(self) else (60, 40, 44)))
        bx = x
        for label, what, col in buttons:
            img = self.f_bold.render(label if not (t and what in ("fear", "like", "test")) else label, True, INK)
            r = pygame.Rect(bx, by, img.get_width() + 24, 34)
            dim = t is not None and what in ("fear", "like", "test", "wipe")
            pygame.draw.rect(surf, tuple(c // 2 for c in col) if dim else col, r, border_radius=8)
            surf.blit(img, img.get_rect(center=r.center))
            self.train_buttons.append((r, what, None))
            bx = r.right + 10
        if t is not None:
            stop = pygame.Rect(panel.right - 110, by, 88, 34)
            pygame.draw.rect(surf, (90, 90, 96), stop, border_radius=8)
            self._text(surf, "STOP", stop.center, INK, self.f_bold, "center")
            self.train_buttons.append((stop, "stop", None))
            what = {"naive": "measuring the untrained response", "odor": "smell", "pair": "smell + " +
                    ("shock" if t["kind"] == "fear" else "sugar"), "rest": "rest", "test": "testing: smell alone"}[t["phase"]]
            done = t["done"] / max(t["trials"], 1)
            pygame.draw.rect(surf, (30, 36, 48), (x, by + 44, panel.w - 44, 10), border_radius=5)
            pygame.draw.rect(surf, AMBER, (x, by + 44, max(8, int((panel.w - 44) * done)), 10), border_radius=5)
            self._text(surf, f"{t['kind']} training on {t['scent']}: trial {min(t['done'] + 1, t['trials'])}/{t['trials']}   "
                             f"{what}   brain {self.brain.steps_per_s:.0f} steps/s", (x, by + 60), TEXT, self.f_small)
        else:
            self._text(surf, "Kept across new flies and restarts, fades slowly (~30 min). Hurting it near a tool "
                             "trains it too.", (x, by + 44), LABEL, self.f_small)
            where = str(mem.path.parent)
            if len(where) > 80:
                where = "..." + where[-77:]
            self._text(surf, f"saved in {where}", (x, by + 62), DIM, self.f_small)

    def _training_click(self, pos) -> None:
        mem = self.brain.memory
        for r, what, arg in getattr(self, "train_buttons", []):
            if not r.collidepoint(pos):
                continue
            if what == "select" and self.train is None:
                self.train_scent = arg
            elif what in ("fear", "like", "test") and self.train is None:
                self.start_training(what)
            elif what == "speed":
                self.train_speed = {1.0: 2.0, 2.0: 3.0}.get(self.train_speed, 1.0)
                if self.train is not None:
                    self.train_active_speed = self.train_speed
            elif what == "stop":
                self.stop_training()
            elif what == "wipe" and self.train is None and mem is not None:
                if now_wipe_armed(self):
                    mem.wipe()
                    self.wipe_armed = 0.0
                    self.note("MEMORY   wiped")
                else:
                    self.wipe_armed = time.perf_counter()
            self.sound.play("click")
            return


    def readouts(self, slot: "FlySlot", now: float) -> None:
        """Validated readouts with no body animation of their own (validation.py):
        GROOM      aDN1/aDN2 (DNg62, DNge078) above THRESH["groom"] x calm while its antennal JO-C/E neurons are
                   active (wind in the fan arena)
        PROBOSCIS  while eating sugar, MN9 fires at least assays.PER_RATIO x its rate from before the fly started eating"""
        from kickthefly.lab import assays

        br, fly = slot.brain, slot.fly
        lvl = br.level("groom")
        antennae = br.level("wind") > 2.0             # the JO-C/E antennal neurons the validated pathway starts from
        if antennae and lvl > THRESH["groom"] and now >= getattr(slot, "groom_ready", 0.0):
            slot.groom_ready = now + 4.0
            self.note(f"GROOM    aDN1/aDN2 x{lvl:.1f}")
            self.on_reaction("GROOM", slot)
        mn9 = br.hz("proboscis")
        if now < fly.eating_until:
            slot.mn9_bout = getattr(slot, "mn9_bout", []) + [mn9]
            if len(slot.mn9_bout) == 60:                       # one second into an eating bout
                ratio = float(np.mean(slot.mn9_bout)) / max(getattr(slot, "mn9_calm", mn9), 1.0)
                if ratio >= assays.PER_RATIO:
                    fly.proboscis_until = now + 1.5
                    self.note(f"PROBOSCIS MN9 x{ratio:.1f}")
                    self.on_reaction("PROBOSCIS", slot)
        else:
            slot.mn9_bout = []
            slot.mn9_calm = getattr(slot, "mn9_calm", mn9) + (mn9 - getattr(slot, "mn9_calm", mn9)) * 0.02
        # SONG: the ps1 wing motor neurons that pIP10 and P1 drive. There are only two of them, so their level is read
        # over about a second (the fast level of two neurons passes 2.5x at rest).
        slot.song_slow = getattr(slot, "song_slow", 1.0) + (br.level("song") - getattr(slot, "song_slow", 1.0)) / 60
        if slot.song_slow > THRESH["song"] and now >= getattr(slot, "song_ready", 0.0):
            slot.song_ready = now + 1.8
            self.note(f"SONG     ps1 wing MNs x{slot.song_slow:.1f}")
            self.on_reaction("SONG", slot)
            if self.cfg["brain.song_buzz"]:
                self.sound.play("pulse_song")
                head_offset = np.array([0.0, 0.4, 0.0]) if len(fly.p[HEAD]) == 3 else np.array([0.0, -60.0])
                self.popup(fly.p[HEAD] + head_offset, "♪ BUZZ ♪", (255, 180, 220))
        # second-order neurons of the CO2, hot and cold sensors: logged only, the fly doesn't act on them
        for kind, group, label in (("CO2", "co2", "V glomerulus PNs"), ("HEAT", "hot_pn", "VP2 PNs (hot)"),
                                   ("COLD", "cold_pn", "VP3 PNs (cold)")):
            lvl = br.level(group)
            if lvl > THRESH[group] and now >= getattr(slot, f"{group}_ready", 0.0):
                setattr(slot, f"{group}_ready", now + 3.0)
                self.note(f"{kind:<8} {label} x{lvl:.1f}")
                self.on_reaction(kind, slot)

    def _memory_behavior(self, slot: "FlySlot", now: float, free: bool, can_fly: bool) -> None:
        fly = slot.fly
        if not slot.scent_now or not free or now < slot.avoid_ready or fly.frozen_at is not None or now < fly.stun_until:
            return
        mx = self.mouse[0]
        if slot.scent_now != "sugar" and slot.fear_now > FEAR_ACT:
            slot.avoid_ready = now + 2.5
            fly.last_hit_x = mx
            if can_fly and slot.fear_now > 0.6:
                fly.escape(now)
            else:
                fly.facing = 1 if fly.p[THX, 0] >= mx else -1
                fly.walk_until, fly.back_until, fly.run = now + 1.2, 0.0, True
            self.note(f"AVOID    remembers the {slot.scent_now} ({slot.fear_now:.2f})")
            self.on_reaction("AVOID", slot)
            self.popup(fly.p[HEAD] + (0, -60), "NOPE!", (255, 220, 120))
        elif slot.scent_now == "sugar" and slot.like_now > LIKE_ACT and now >= fly.walk_until:
            slot.avoid_ready = now + 1.5
            fly.facing = 1 if mx > fly.p[THX, 0] else -1
            fly.walk_until, fly.run = now + 1.0, False
            self.note(f"APPROACH remembers sugar ({slot.like_now:.2f})")

    # --- arenas --------------------------------------------------------------------
    def _environment(self, now: float, mouse) -> None:
        arena = ARENAS[self.arena_i]
        for slot in self.flies:
            self._environment_one(slot, arena, now, mouse)
        self.streaks = [st for st in self.streaks if now - st[3] < 1.2]

    def _environment_one(self, slot: "FlySlot", arena: str, now: float, mouse) -> None:
        fly, br = slot.fly, slot.brain
        fly.arena, fly.wind = arena, 0.0
        if arena != "flypaper":
            fly.stuck.clear()
        if arena == "fan":
            gust = 0.75 + 0.25 * math.sin(now * 1.3) + 0.15 * math.sin(now * 4.1)
            fly.wind = 0.55 * gust * float(np.clip(1.15 - fly.p[THX, 0] / 900.0, 0.25, 1.0))
            if not fly.dead and self.frame % 3 == 0:
                br.poke("wind", None, min(1.0, fly.wind * 1.4))     # Johnston's organ wind neurons
            if random.random() < 0.7:
                self.streaks.append([FAN[0] + 50, FAN[1] + random.uniform(-80, 80), random.uniform(10, 18), now])
        elif arena == "flypaper" and fly.stuck:
            kicking = now < fly.flail_until
            for i in list(fly.stuck):
                pull = fly.grabbed is not None and np.hypot(*(np.array(mouse, float) - fly.stuck[i])) > 90
                if random.random() < 0.0012 + (0.012 if kicking else 0) + (0.06 if pull else 0):
                    del fly.stuck[i]
            if not fly.dead:
                if self.frame % 6 == 0:
                    br.poke("legs", "L", 0.4)
                    br.poke("legs", "R", 0.4)
                    br.poke("body", None, 0.2)
                if len(fly.stuck) >= 3:
                    self.damage(slot, 0.012, "the flypaper")
        elif arena == "pool":
            sub = fly.p[:, 1] > WATER_Y
            if sub.any():
                if fly.wet <= 0 and float(np.max((fly.p - fly.prev)[sub, 1])) > 4:
                    self.sound.play("splash")
                    self.puff((fly.p[THX, 0], WATER_Y), 10, 3)
                fly.wet = 3.0
                if not fly.dead:
                    if self.frame % 4 == 0:
                        br.poke("humid", None, 0.9)             # hygrosensory neurons
                        br.poke("body", None, 0.25)
                    if fly.p[HEAD, 1] > WATER_Y + 6:
                        self.damage(slot, 0.05, "drowning")
        elif arena == "lamp":
            d = float(np.hypot(*(fly.p[HEAD] - LAMP)))
            if not fly.dead:
                light = float(np.clip(1.2 - d / 500.0, 0.15, 1.0))
                if self.frame % 2 == 0:
                    br.poke("light", None, light, recruit=0.25 * light)   # photoreceptors
                if d < 58:
                    br.poke("heat", None, 0.8)
                    self.damage(slot, 0.08, "the hot lamp")
                    away = (fly.p[HEAD] - LAMP) / max(d, 1.0)
                    for i in (HEAD, THX, ABD):
                        fly.impulse(i, away * 1.5)
            if now < fly.escape_until and np.hypot(*(fly.fly_target - LAMP)) > 130:
                fly.fly_target = np.array(LAMP) + (random.uniform(-110, 110), random.uniform(50, 130))
        elif arena == "escaperoom":
            # 1. Fan wind from left
            gust = 0.75 + 0.25 * math.sin(now * 1.3) + 0.15 * math.sin(now * 4.1)
            fly.wind = 0.45 * gust * float(np.clip(1.15 - fly.p[THX, 0] / 900.0, 0.15, 1.0))
            if not fly.dead and self.frame % 3 == 0:
                br.poke("wind", None, min(1.0, fly.wind * 1.4))
            if random.random() < 0.6:
                self.streaks.append([FAN[0] + 50, FAN[1] + random.uniform(-80, 80), random.uniform(10, 18), now])

            # 2. Flypaper stickiness in middle
            if fly.stuck:
                kicking = now < fly.flail_until
                for i in list(fly.stuck):
                    pull = fly.grabbed is not None and np.hypot(*(np.array(mouse, float) - fly.stuck[i])) > 90
                    if random.random() < 0.0012 + (0.012 if kicking else 0) + (0.06 if pull else 0):
                        del fly.stuck[i]
                if not fly.dead:
                    if self.frame % 6 == 0:
                        br.poke("legs", "L", 0.4)
                        br.poke("legs", "R", 0.4)
                        br.poke("body", None, 0.2)
                    if len(fly.stuck) >= 3:
                        self.damage(slot, 0.012, "the flypaper")

            # 3. Hot Lamp overhead
            d_lamp = float(np.hypot(*(fly.p[HEAD] - LAMP)))
            if not fly.dead:
                light = float(np.clip(1.2 - d_lamp / 500.0, 0.15, 1.0))
                if self.frame % 2 == 0:
                    br.poke("light", None, light, recruit=0.25 * light)
                if d_lamp < 58:
                    br.poke("heat", None, 0.8)
                    self.damage(slot, 0.08, "the hot lamp")
                    away = (fly.p[HEAD] - LAMP) / max(d_lamp, 1.0)
                    for i in (HEAD, THX, ABD):
                        fly.impulse(i, away * 1.5)
            if now < fly.escape_until and np.hypot(*(fly.fly_target - LAMP)) > 130:
                fly.fly_target = np.array(LAMP) + (random.uniform(-110, 110), random.uniform(50, 130))

            # 4. Sugar Goal on the right
            d_sugar = float(np.hypot(*(fly.p[HEAD] - SUGAR_GOAL)))
            if d_sugar < 45.0 and not fly.dead:
                if self.frame % 3 == 0:
                    br.poke("sweet", None, 1.0)
                if not self.escaperoom_completed:
                    self.escaperoom_completed = True
                    self.escaperoom_finish_t = now
                    run_time = round(max(0.1, now - self.escaperoom_start_t), 2)
                    from kickthefly.game.speedrun import make_speedrun_code
                    self.escaperoom_code = make_speedrun_code(self.escaperoom_seed, run_time)
                    from kickthefly.lab.challenges import record_score
                    record_score("escaperoom_speedrun", run_time, "low")
                    self.sound.play("yum")
                    self.note(f"ESCAPEROOM CLEAR {run_time:.2f}s ({self.escaperoom_code})")
        elif arena == "thermo":
            self._thermo_tick(slot, (float(fly.p[THX, 0]) - THERMO_MID_X) / THERMO_HALF_X, fly.p[THX, 1] > FLOOR - STAND - 20)
        if arena != "pool":
            fly.wet = max(0.0, fly.wet - 1 / 60)
        elif not (fly.p[:, 1] > WATER_Y).any():
            fly.wet = max(0.0, fly.wet - 1 / 60)

    def _draw_arena_back(self, surf, now: float) -> None:
        arena = ARENAS[self.arena_i]
        if arena == "fan":
            cx, cy = FAN
            thick_line(surf, (cx - 20, cy), (cx - 20, FLOOR), 10, (70, 74, 84))
            aacircle(surf, (cx, cy), 78, (50, 54, 64))
            aacircle(surf, (cx, cy), 70, (26, 28, 34))
            for k in range(3):
                a = now * 25 + k * 2.094
                tip = (cx + 62 * math.cos(a), cy + 62 * math.sin(a))
                aapoly(surf, ellipse_pts(((cx + tip[0]) / 2, (cy + tip[1]) / 2), 34, 13, a, 16), (150, 160, 175))
            aacircle(surf, (cx, cy), 12, (90, 96, 108))
            for k in range(-3, 4):
                gfxdraw.line(surf, int(cx - 70), int(cy + k * 18), int(cx + 70), int(cy + k * 18), (95, 100, 112))
            for st in self.streaks:
                e = (now - st[3]) / 1.2
                x = st[0] + e * 1100 * (st[2] / 14)
                gfxdraw.line(surf, int(x), int(st[1]), int(x + 40), int(st[1]), (200, 220, 235, int(110 * (1 - e))))
        elif arena == "flypaper":
            x0, x1 = PAPER_X
            pygame.draw.rect(surf, (214, 176, 50), (x0, FLOOR - 5, x1 - x0, 9), border_radius=3)
            pygame.draw.rect(surf, (245, 214, 95), (x0 + 6, FLOOR - 4, x1 - x0 - 12, 2))
            for k in range(12):
                gfxdraw.filled_circle(surf, int(x0 + 20 + k * 35), FLOOR + 1, 2, (170, 130, 40))
            self._text(surf, "FLYPAPER", ((x0 + x1) // 2, FLOOR + 10), (190, 160, 70), self.f_small, "midtop")
        elif arena == "lamp":
            lx, ly = LAMP
            pygame.draw.line(surf, (60, 60, 64), (lx, CEIL - 40), (lx, ly - 40), 3)
            aapoly(surf, [(lx - 22, ly - 42), (lx + 22, ly - 42), (lx + 46, ly - 8), (lx - 46, ly - 8)], (60, 64, 74))
            pulse = 0.9 + 0.1 * math.sin(now * 3)
            for r, a in ((190, 18), (120, 30), (70, 60)):
                aacircle(surf, (lx, ly), r * pulse, (255, 230, 150, a))
            aacircle(surf, (lx, ly), 24, (255, 246, 205))
        elif arena == "escaperoom":
            # 1. Fan
            cx, cy = FAN
            thick_line(surf, (cx - 20, cy), (cx - 20, FLOOR), 10, (70, 74, 84))
            aacircle(surf, (cx, cy), 78, (50, 54, 64))
            aacircle(surf, (cx, cy), 70, (26, 28, 34))
            for k in range(3):
                a = now * 25 + k * 2.094
                tip = (cx + 62 * math.cos(a), cy + 62 * math.sin(a))
                aapoly(surf, ellipse_pts(((cx + tip[0]) / 2, (cy + tip[1]) / 2), 34, 13, a, 16), (150, 160, 175))
            aacircle(surf, (cx, cy), 12, (90, 96, 108))
            for k in range(-3, 4):
                gfxdraw.line(surf, int(cx - 70), int(cy + k * 18), int(cx + 70), int(cy + k * 18), (95, 100, 112))
            for st in self.streaks:
                e = (now - st[3]) / 1.2
                x = st[0] + e * 1100 * (st[2] / 14)
                gfxdraw.line(surf, int(x), int(st[1]), int(x + 40), int(st[1]), (200, 220, 235, int(110 * (1 - e))))

            # 2. Flypaper strip in middle
            px0, px1 = (280.0, 560.0)
            pygame.draw.rect(surf, (214, 176, 50), (px0, FLOOR - 5, px1 - px0, 9), border_radius=3)
            pygame.draw.rect(surf, (245, 214, 95), (px0 + 6, FLOOR - 4, px1 - px0 - 12, 2))
            for k in range(8):
                gfxdraw.filled_circle(surf, int(px0 + 20 + k * 35), FLOOR + 1, 2, (170, 130, 40))
            self._text(surf, "FLYPAPER HAZARD", ((px0 + px1) // 2, FLOOR + 10), (190, 160, 70), self.f_small, "midtop")

            # 3. Hot Lamp overhead
            lx, ly = LAMP
            pygame.draw.line(surf, (60, 60, 64), (lx, CEIL - 40), (lx, ly - 40), 3)
            aapoly(surf, [(lx - 22, ly - 42), (lx + 22, ly - 42), (lx + 46, ly - 8), (lx - 46, ly - 8)], (60, 64, 74))
            pulse = 0.9 + 0.1 * math.sin(now * 3)
            for r, a in ((190, 18), (120, 30), (70, 60)):
                aacircle(surf, (lx, ly), r * pulse, (255, 230, 150, a))
            aacircle(surf, (lx, ly), 24, (255, 246, 205))

            # 4. Sugar Goal on the right
            gx, gy = SUGAR_GOAL
            pygame.draw.rect(surf, (180, 190, 205), (gx - 28, FLOOR - 6, 56, 8), border_radius=3)
            sparkle = 0.8 + 0.2 * math.sin(now * 6)
            pygame.draw.circle(surf, (255, 245, 220), (int(gx), int(gy)), 14)
            pygame.draw.circle(surf, (255, 255, 255), (int(gx), int(gy)), int(10 * sparkle))
            self._text(surf, "SUGAR GOAL", (int(gx), FLOOR + 10), (255, 220, 100), self.f_small, "midtop")
        elif arena == "thermo":
            bar_y = FLOOR - 6
            bar_h = 10
            n_segs = 40
            seg_w = PLAY_W / n_segs
            for s in range(n_segs):
                frac = s / (n_segs - 1)
                if frac < 0.5:
                    t_val = frac / 0.5
                    r = int(50 * (1 - t_val) + 40 * t_val)
                    g = int(120 * (1 - t_val) + 180 * t_val)
                    b = int(230 * (1 - t_val) + 110 * t_val)
                else:
                    t_val = (frac - 0.5) / 0.5
                    r = int(40 * (1 - t_val) + 230 * t_val)
                    g = int(180 * (1 - t_val) + 60 * t_val)
                    b = int(110 * (1 - t_val) + 40 * t_val)
                pygame.draw.rect(surf, (r, g, b), (int(s * seg_w), bar_y, int(seg_w) + 1, bar_h))
            self._text(surf, "COLD (15°C)", (100, FLOOR + 10), (70, 150, 240), self.f_small, "midtop")
            self._text(surf, "COMFORT (24°C)", (int(PLAY_W / 2), FLOOR + 10), (80, 200, 140), self.f_small, "midtop")
            self._text(surf, "HOT (35°C)", (PLAY_W - 100, FLOOR + 10), (240, 90, 60), self.f_small, "midtop")

    def _aggression(self, slot: "FlySlot", now: float, free: bool) -> None:
        """LUNGE (GAME RULE, 2D and 3D): with other flies around, the aggression neurons (AVLP727m and pC1, P1
        included) above THRESH["aggression"] throw the fly at the nearest one. In play nothing takes them there on its
        own (flies bumping or looming at each other reach ~1.5x): stimulate them in brain surgery (P1, AVLP727m)."""
        fly, br = slot.fly, slot.brain
        if (len(self.flies) < 2 or not free or now < getattr(slot, "lunge_ready", 0.0)
                or not self.cfg["brain.lunge"]):
            return
        lvl = br.level("aggression")
        if lvl <= THRESH["aggression"]:
            return
        scale = 0.006 if self.three_d else 1.0          # kick3d's meters per 2D pixel
        others = [o for o in self.flies if o is not slot and not o.fly.dead]
        if not others:
            return
        near = min(others, key=lambda o: float(np.linalg.norm(o.fly.p[THX] - fly.p[THX])))
        d = float(np.linalg.norm(near.fly.p[THX] - fly.p[THX]))
        if d > 180.0 * scale:
            return
        slot.lunge_ready = now + 1.8
        vec = (near.fly.p[THX] - fly.p[THX]) / max(d, 1e-6)
        if self.three_d:
            fly.yaw_target = math.atan2(vec[0], vec[2])
            fly.impulse(THX, (vec * 6.0 + np.array([0.0, 2.5, 0.0])) * scale)
        else:
            fly.facing = 1 if vec[0] > 0 else -1
            fly.impulse(THX, vec * 6.0 + np.array([0.0, -2.5]))
        self.note(f"LUNGE    AVLP727m/pC1 x{lvl:.1f}")
        self.on_reaction("LUNGE", slot)
        up = np.array([0.0, 0.4, 0.0]) if self.three_d else np.array([0.0, -60.0])
        self.popup(fly.p[HEAD] + up, "LUNGE!", (255, 120, 80))
        self.sound.play("whack")

    def _sleep(self, slot: "FlySlot", now: float, free: bool) -> None:
        """SLEEP (GAME RULE readout): dorsal fan-shaped body (FB6/FB7) above THRESH["sleep"] puts the fly to rest for
        2 s: no spontaneous take-off or walking (its reflexes still work). Nothing in play drives dFB that far on its
        own, daylight and darkness included; stimulate FB6/FB7 in brain surgery to see it."""
        lvl = slot.brain.level("sleep")
        if lvl <= THRESH["sleep"] or not free:
            return
        if now >= getattr(slot, "asleep_until", 0.0):
            self.note(f"SLEEP    dFB FB6/FB7 x{lvl:.1f}")
            self.on_reaction("SLEEP", slot)
            up = np.array([0.0, 0.4, 0.0]) if self.three_d else np.array([0.0, -60.0])
            self.popup(slot.fly.p[HEAD] + up, "ZZZ", (170, 190, 255))
        slot.asleep_until = now + 2.0

    def _lab_tick(self) -> None:
        """3.0 day 2: the Lab's live thermogenetics (every few frames). Does nothing until an expression is added. Also takes
        the patch electrode off once its page is closed."""
        if getattr(self.menu, "toolkit", None) is not None:
            from kickthefly.lab import labtoolkit
            labtoolkit.leave_patch(self.menu)
        tl = self.thermo_live
        if (tl.active or tl.per_brain) and self.frame % 3 == 0:
            try:
                tl.tick(self.flies, ARENAS[self.arena_i] == "thermo")
            except Exception as e:                       # a target this brain can't resolve: that expression is dropped
                if not isinstance(e, ValueError):        # (ThermoLive did it); anything else stops them all
                    tl.clear([s.brain for s in self.flies])
                self.menu.flash(f"thermogenetics: {e}", menu_ui.BAD)

    def patch_neuron(self, row: int) -> None:
        """The inspector's Patch button: open Lab > Patch on this neuron."""
        self.patch_row = int(row)
        self.menu.show("lab_patch")

    def _thermo_tick(self, slot: "FlySlot", t: float, on_floor: bool) -> None:
        """The thermo arena, shared by the 2D and 3D games. t is where the fly is between the cold wall (-1) and the
        hot wall (+1). GAME RULE: the gradient and the damage at the extremes. CONNECTOME: what the drive reaches."""
        fly, br = slot.fly, slot.brain
        cold, hot = thermo_gradient(t)
        from kickthefly.lab.thermogenetics import arena_temperature
        slot.arena_temp_c = arena_temperature(t)           # 3.0 day 2: what TrpA1 / shibire-ts expressed in this fly sense
        if not fly.dead and self.frame % 3 == 0:
            if cold > THERMO_DEADBAND:
                br.poke("cold", None, cold)
            if hot > THERMO_DEADBAND:
                br.poke("heat", None, hot)
        if on_floor and not fly.dead:
            if hot > 0.85:
                self.damage(slot, 0.02, "extreme heat")
            elif cold > 0.85:
                self.damage(slot, 0.02, "extreme cold")

    def _draw_arena_front(self, surf, now: float) -> None:
        if ARENAS[self.arena_i] != "pool":
            return
        water = pygame.Surface((PLAY_W, FLOOR - WATER_Y), pygame.SRCALPHA)
        water.fill((40, 120, 190, 95))
        surf.blit(water, (0, WATER_Y))
        pts = [(x, WATER_Y + 3 * math.sin(x * 0.03 + now * 2.2)) for x in range(0, PLAY_W + 20, 20)]
        pygame.draw.aalines(surf, (170, 220, 255), False, pts)
        for k in range(6):
            x = (k * 157 + now * 30) % PLAY_W
            gfxdraw.line(surf, int(x), WATER_Y + 20 + k * 17, int(x + 50), WATER_Y + 20 + k * 17, (150, 210, 250, 70))

    # --- brain surgery ---------------------------------------------------------------
    def _surgery_rows(self, spec) -> np.ndarray:
        kind, names = spec
        br = self.brain
        if kind == "group":
            return np.flatnonzero(np.isin(br.det_id, [br.col[n] for n in names]))
        if kind == "pop":
            pops = [n for n, _ in POPS]
            return np.flatnonzero(np.isin(br.pop_id, [pops.index(n) for n in names]))
        if kind == "type":
            return np.flatnonzero(np.isin(br.types, names))
        if kind == "prefix":
            m = np.zeros(br.n, bool)
            for pre in names:
                m |= np.char.startswith(br.types, pre)
            return np.flatnonzero(m)
        if kind == "hemifield_visual":
            from kickthefly.lab import lesions
            g = self.graph if self.graph is not None else getattr(self.brain, "graph", None)
            if g is None:
                from kickthefly.core import simcore
                g = simcore.pack()[0]
            return lesions.hemifield_visual_rows(g, side=names)
        if kind == "hemisphere":
            from kickthefly.lab import lesions
            g = self.graph if self.graph is not None else getattr(self.brain, "graph", None)
            if g is None:
                from kickthefly.core import simcore
                g = simcore.pack()[0]
            return lesions.hemisphere_rows(g, side=names)
        return np.arange(br.n)

    def silence_hemifield(self, side: str = "L", visual_only: bool = True) -> None:
        """One-click silencing of visual hemifield or whole hemisphere."""
        target = f"{'Left' if side.upper() == 'L' else 'Right'} {'visual hemifield' if visual_only else 'hemisphere'}"
        for k, (label, _) in enumerate(SURGERY):
            if label.startswith(target):
                self._set_surgery(k, -1)
                break

    def _apply_surgery(self) -> None:
        br = self.brain
        br.clear_overrides()
        for (label, _), mode in zip(SURGERY, self.surgery_modes):
            if mode:
                br.set_override(self.surgery_rows[label], mode)
        for t, mode in self.type_ops.items():
            if mode:
                br.set_override(np.flatnonzero(br.types == t), mode)

    def _set_surgery(self, k: int, mode: int) -> None:
        self.surgery_modes[k] = mode
        self._apply_surgery()
        label = SURGERY[k][0]
        self.note(f"SURGERY  {label.split(' (')[0]}: {'off' if mode < 0 else 'on' if mode > 0 else 'normal'}")
        self.sound.play("click")

    def _draw_surgery(self, surf) -> None:
        panel_h = min(H - 24, 690)
        panel = pygame.Rect(75, 12, 740, panel_h)
        veil = pygame.Surface((PLAY_W, H), pygame.SRCALPHA)
        veil.fill((4, 5, 8, 170))
        surf.blit(veil, (0, 0))
        pygame.draw.rect(surf, (18, 21, 28), panel, border_radius=16)
        pygame.draw.rect(surf, BORDER, panel, 1, border_radius=16)
        x = panel.x + 24
        self._text(surf, "BRAIN SURGERY", (x, panel.y + 14), INK, self.f_title)
        self._text(surf, "Silence (OFF) or stimulate (ON) real neuron groups. Hemifield lesions test wiring outcomes, not injury.",
                   (x + 2, panel.y + 50), LABEL, self.f_small)
        y = panel.y + 76
        self.surgery_buttons = []
        row_step = 26
        for k, (label, _) in enumerate(SURGERY):
            n = len(self.surgery_rows[label])
            mode = self.surgery_modes[k]
            is_hemi = "hemifield" in label.lower() or "hemisphere" in label.lower()
            txt_col = AMBER if mode else (180, 210, 240) if is_hemi else TEXT
            self._text(surf, label, (x, y), txt_col, self.f_small if len(label) > 34 else self.f_text)
            self._text(surf, f"{n:,}", (panel.right - 240, y + 2), LABEL, self.f_small, "topright")
            for j, (m, text) in enumerate(((-1, "OFF"), (0, "-"), (1, "ON"))):
                r = pygame.Rect(panel.right - 216 + j * 64, y - 1, 56, 22)
                on = mode == m
                fill = ((60, 110, 200) if m < 0 else (230, 130, 50) if m > 0 else (70, 76, 90)) if on else (30, 34, 44)
                pygame.draw.rect(surf, fill, r, border_radius=6)
                self._text(surf, text, r.center, INK if on else LABEL, self.f_small, "center")
                self.surgery_buttons.append((r, k, m))
            y += row_step
        if self.type_ops:
            ops = ", ".join(f"{t} {'off' if m < 0 else 'on'}" for t, m in self.type_ops.items() if m)
            self._text(surf, f"From the inspector: {ops}", (x, y + 2), AMBER, self.f_small)
        clear = pygame.Rect(panel.right - 150, panel.bottom - 36, 126, 26)
        pygame.draw.rect(surf, (60, 64, 76), clear, border_radius=8)
        self._text(surf, "CLEAR ALL", clear.center, INK, self.f_bold, "center")
        self.surgery_buttons.append((clear, -1, 0))
        self._text(surf, "OFF adds -2.4 per step (silences). ON adds +0.48. Reported as wiring outcomes, not physical trauma.",
                   (x, panel.bottom - 30), DIM, self.f_small)

    # --- neuron inspector ----------------------------------------------------------------
    def _inspect_at(self, pos) -> None:
        if getattr(self.view, "_dirty_big", False):
            self.view._recompute_big()
        w, h = VIEW_SIZES["big"]
        px, py = pos[0] - self.big_rect.x, pos[1] - self.big_rect.y
        pix = self.view.spark_pix["big"]
        cx, cy = pix % w, pix // w
        near = np.flatnonzero((pix >= 0) & (np.abs(cx - px) <= 7) & (np.abs(cy - py) <= 7))
        if not len(near):
            self.inspect = None
            return
        rates = self.brain.sim.activity.rates()[near]
        dist = np.hypot(cx[near] - px, cy[near] - py)
        self.inspect = self._neuron_info(int(near[np.argmax(rates / 0.025 * 0.5 - dist)]))   # nearest, then busiest
        self.sound.play("click")

    def _neuron_info(self, i: int) -> dict:
        br = self.brain
        Wr, Wc = br.sim.W_csr, br.sim.W_csc

        def top(idx, val):
            order = np.argsort(-np.abs(val))[:6]
            return [(int(idx[k]), float(val[k])) for k in order]

        ins = top(Wr.indices[Wr.indptr[i]:Wr.indptr[i + 1]], Wr.data[Wr.indptr[i]:Wr.indptr[i + 1]])
        outs = top(Wc.indices[Wc.indptr[i]:Wc.indptr[i + 1]], Wc.data[Wc.indptr[i]:Wc.indptr[i + 1]])
        pop = br.pop_id[i] if hasattr(br, "pop_id") and i < len(br.pop_id) else -1
        pop_str = [n for n, _ in POPS][pop] if 0 <= pop < len(POPS) else (
            str(br.superclass[i]) if getattr(br, "superclass", None) is not None and i < len(br.superclass) and br.superclass[i] else "unknown"
        )
        inst_str = str(br.instance[i]) if hasattr(br, "instance") and i < len(br.instance) and br.instance[i] else ""
        return {"i": i, "type": br.types[i] or "untyped", "instance": inst_str,
                "pop": pop_str, "ins": ins, "outs": outs,
                "n_in": int(Wr.indptr[i + 1] - Wr.indptr[i]), "n_out": int(Wc.indptr[i + 1] - Wc.indptr[i])}

    def _draw_inspect(self, surf, rect: pygame.Rect, now: float) -> None:
        info = self.inspect
        br, w = self.brain, VIEW_SIZES["big"][0]
        pix = self.view.spark_pix["big"]

        def at(j):
            if j is None or j < 0 or j >= len(pix):
                return None
            q = pix[j]
            return None if q < 0 else (rect.x + q % w, rect.y + q // w)

        me = at(info["i"])
        for lst, col in ((info["ins"], (110, 200, 255)), (info["outs"], (255, 160, 80))):
            for j, _ in lst:
                q = at(j)
                if me and q:
                    pygame.draw.aaline(surf, col, me, q)
                    aacircle(surf, q, 3, col)
        if me:
            r = max(1.0, 8.0 + 3.0 * math.sin(now * 8))
            gfxdraw.aacircle(surf, int(me[0]), int(me[1]), int(r), (255, 255, 255))
            gfxdraw.aacircle(surf, int(me[0]), int(me[1]), int(r + 4), (255, 255, 255, 120))
        card = pygame.Rect(rect.right - 318, rect.y + 10, 308, 348)
        bg = pygame.Surface(card.size, pygame.SRCALPHA)
        pygame.draw.rect(bg, (8, 10, 16, 225), bg.get_rect(), border_radius=10)
        surf.blit(bg, card)
        x, y = card.x + 12, card.y + 10
        i = info["i"]
        rate = float(br.sim.activity.rates()[i]) / 0.005
        calm = float(self.view.calm[i]) / 0.005
        slot = self.flies[self.focus] if self.focus < len(self.flies) else None
        pers = getattr(slot, "personality", None)
        if pers:
            from kickthefly.core import cards

            self._text(surf, f"Fly #{self.focus + 1} · {cards.label(pers)}", (x, y), ACCENT, self.f_small)
            self._text(surf, info["type"], (x, y + 16), INK, self.f_head)
            inst_label = f"   {info['instance']}" if info.get("instance") else ""
            self._text(surf, f"{info['pop']}{inst_label}", (x, y + 38), LABEL, self.f_small)
            self._text(surf, f"firing {rate:5.1f} spikes/s   (calm {calm:4.1f})", (x, y + 54),
                       (255, 170, 90) if rate > calm + 2 else TEXT, self.f_small)
            self._text(surf, f"{info['n_in']} input partners, {info['n_out']} output partners", (x, y + 68), LABEL, self.f_small)
            yy = y + 84
        else:
            self._text(surf, info["type"], (x, y), INK, self.f_head)
            inst_label = f"   {info['instance']}" if info.get("instance") else ""
            self._text(surf, f"{info['pop']}{inst_label}", (x, y + 26), LABEL, self.f_small)
            self._text(surf, f"firing {rate:5.1f} spikes/s   (calm {calm:4.1f})", (x, y + 44),
                       (255, 170, 90) if rate > calm + 2 else TEXT, self.f_small)
            self._text(surf, f"{info['n_in']} input partners, {info['n_out']} output partners", (x, y + 60), LABEL, self.f_small)
            yy = y + 76
        nt, conf, src = self.transmitter_of(i)
        if nt:
            sign = {"acetylcholine": "excitatory (+)", "gaba": "inhibitory (-)", "glutamate": "inhibitory (-)",
                    "histamine": "inhibitory (-)"}.get(nt, "no sign in this model (0)")
            flipped = i in getattr(self, "_flipped_set", ())
            how = {"ground_truth": "measured", "consensus_nt": "predicted (consensus)",
                   "predicted_nt": "predicted (per body)"}.get(src, src or "unknown")
            conf_txt = "no confidence given" if conf != conf else f"{conf:.0%} confident"
            self._text(surf, f"{nt}: {sign}", (x, yy), (255, 140, 140) if flipped else TEXT, self.f_small)
            self._text(surf, f"{how}, {conf_txt}" + ("   SIGN FLIPPED IN LAB" if flipped else ""), (x, yy + 14),
                       (255, 140, 140) if flipped else LABEL, self.f_small)
            yy += 34
        yy += 4
        for title, lst, col in (("strongest inputs (% of its input)", info["ins"], (110, 200, 255)),
                                ("strongest outputs (% of target's input)", info["outs"], (255, 160, 80))):
            self._text(surf, title, (x, yy), col, self.f_small)
            yy += 15
            for j, v in lst[:4]:
                name = br.types[j] or "untyped"
                inst = br.instance[j] or "" if j < len(br.instance) else ""
                side = inst[-2:] if (isinstance(inst, str) and inst.endswith(("_L", "_R"))) else ""
                sign = "+" if v > 0 else "-"
                self._text(surf, f"{sign}{abs(v) * 100:4.1f}%  {name}{side}", (x + 6, yy), TEXT, self.f_small)
                yy += 14
            yy += 4
        self.inspect_buttons = []
        t = info["type"]
        for k, (label, mode) in enumerate((("SILENCE TYPE", -1), ("STIMULATE", 1), ("CLEAR", 0))):
            r = pygame.Rect(card.x + 12 + k * 98, card.bottom - 62, 90, 24)
            active = self.type_ops.get(t, 0) == mode and mode != 0
            pygame.draw.rect(surf, (60, 110, 200) if (active and mode < 0) else (230, 130, 50) if active else (40, 46, 58), r, border_radius=6)
            self._text(surf, label, r.center, INK, self.f_small, "center")
            self.inspect_buttons.append((r, mode))
        self.inspect_flip_button = None
        if self.cfg.lab and nt in ("acetylcholine", "gaba", "glutamate", "histamine"):
            r = pygame.Rect(card.x + 12, card.bottom - 32, 188, 24)
            on = i in getattr(self, "_flipped_set", ())
            pygame.draw.rect(surf, (190, 70, 70) if on else (40, 46, 58), r, border_radius=6)
            self._text(surf, "UNFLIP THIS NEURON" if on else "FLIP THIS NEURON'S SIGN", r.center, INK, self.f_small,
                       "center")
            self.inspect_flip_button = (r, i)
            self._text(surf, "Lab: sign flip", (card.right - 12, card.bottom - 26), DIM, self.f_small, "topright")
        self.patch_button = None
        if self.cfg.lab:                             # 3.0 day 2: virtual patch clamp on this neuron (Lab > Patch)
            # a row of its own under PATH FROM/TO (3.0 day 2 review: at card.x + 312 it fell off the card and the view's clip)
            r = pygame.Rect(card.x, card.bottom + 36, 306, 24)
            pygame.draw.rect(surf, (90, 70, 140), r, border_radius=6)
            self._text(surf, "PATCH CLAMP THIS NEURON (MODEL)", r.center, INK, self.f_small, "center")
            self.patch_button = (r, i)
        self.path_buttons = []                       # path tracer: this neuron as the start or the end
        for k, (label, what) in enumerate((("PATH FROM HERE", "from"), ("PATH TO HERE", "to"))):
            r = pygame.Rect(card.x + k * 156, card.bottom + 6, 150, 24)
            on = getattr(self, "path_ends", {}).get(what) == i
            pygame.draw.rect(surf, (40, 120, 90) if on else (40, 46, 58), r, border_radius=6)
            self._text(surf, label, r.center, INK, self.f_small, "center")
            self.path_buttons.append((r, what, i))

    # --- big view: neuron search and path tracer (kickthefly/lab/neurosearch.py) ---------------------------------
    def search_box_rect(self) -> pygame.Rect:
        r = getattr(self, "big_rect", None) or pygame.Rect(0, 58, *VIEW_SIZES["big"])
        return pygame.Rect(r.x + 8, r.y + 8, 250, 24)

    def search_key(self, ev) -> bool:
        """Typing into the big view's search box. Returns True if the event was used (hotkeys wait meanwhile)."""
        st = getattr(self, "nsearch", None)
        if not (self.big_view and st and st["active"]) or self.menu.open:
            return False
        if ev.type == pygame.TEXTINPUT:
            st["text"] = (st["text"] + ev.text)[:40]
        elif ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_ESCAPE:
                st["active"] = False
            elif ev.key == pygame.K_BACKSPACE:
                st["text"] = st["text"][:-1]
            elif ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                if st["results"]:
                    self.pick_neuron(st["results"][0])
                st["active"] = False
            else:
                return True
        else:
            return False
        from kickthefly.lab import neurosearch
        st["results"] = neurosearch.search(self.brain, st["text"]) if st["text"].strip() else []
        return True

    def search_click(self, pos) -> bool:
        """Clicks on the search box, its results and the path tracer. Returns True if used."""
        st = self.__dict__.setdefault("nsearch", dict(active=False, text="", results=[]))
        if self.search_box_rect().collidepoint(pos):
            st["active"] = True
            return True
        for r, i in getattr(self, "search_result_rects", []):
            if st["active"] and r.collidepoint(pos):
                self.pick_neuron(i)
                st["active"] = False
                return True
        was = st["active"]
        st["active"] = False
        for r, what, i in getattr(self, "path_buttons", []) if self.inspect is not None else ():
            if r.collidepoint(pos):
                self.set_path_end(what, i)
                return True
        clear = getattr(self, "path_clear_rect", None)
        if clear is not None and clear.collidepoint(pos):
            self.path_ends, self.paths = {}, []
            return True
        return was                                    # a click outside only closes the result list

    def pick_neuron(self, i: int) -> None:
        self.inspect = self._neuron_info(int(i))
        self.update_stethoscope_target()
        self.sound.play("click")

    def set_path_end(self, what: str, i: int, k: int = 5) -> None:
        """Mark a path end; with both ends set, trace the k strongest paths (up to 3 synapses) between them."""
        from kickthefly.lab import neurosearch

        ends = self.__dict__.setdefault("path_ends", {})
        ends[what] = int(i)
        self.paths = []
        if "from" in ends and "to" in ends:
            sim = self.brain.sim
            self.paths = neurosearch.top_paths(sim.W_csr, ends["from"], ends["to"], k=k, Wc=sim.W_csc)
            self.note(f"PATHS    {self.brain.types[ends['from']] or '#%d' % ends['from']} -> "
                      f"{self.brain.types[ends['to']] or '#%d' % ends['to']}: {len(self.paths)} found (<= 3 synapses)")

    def _draw_search(self, surf, rect: pygame.Rect) -> None:
        st = self.__dict__.setdefault("nsearch", dict(active=False, text="", results=[]))
        box = self.search_box_rect()
        pygame.draw.rect(surf, (18, 24, 36), box, border_radius=5)
        pygame.draw.rect(surf, ACCENT if st["active"] else BORDER, box, 1, border_radius=5)
        caret = "|" if st["active"] and int(time.perf_counter() * 2) % 2 == 0 else ""
        label = (st["text"] + caret) if (st["text"] or st["active"]) else "Search neurons (type, body ID)"
        self._text(surf, label, (box.x + 8, box.y + 4), INK if st["text"] else LABEL, self.f_small)
        self.search_result_rects = []
        if not st["active"] or not st["results"]:
            return
        br, rates = self.brain, self.brain.sim.activity.rates()
        for k, i in enumerate(st["results"]):
            r = pygame.Rect(box.x, box.bottom + 2 + k * 20, 330, 20)
            pygame.draw.rect(surf, (14, 18, 28) if k % 2 else (20, 26, 38), r)
            bid = f"  {int(br.body_id[i])}" if getattr(br, "body_id", None) is not None else ""
            self._text(surf, f"{br.types[i] or 'untyped'}  {br.instance[i] or ''}{bid}"[:48], (r.x + 6, r.y + 2), TEXT,
                       self.f_small)
            self._text(surf, f"{rates[i] / 0.005:4.0f} Hz", (r.right - 6, r.y + 2), (140, 220, 180), self.f_small,
                       "topright")
            self.search_result_rects.append((r, i))

    def _draw_paths(self, surf, rect: pygame.Rect, now: float) -> None:
        """The traced paths over the brain: brighter for stronger paths, and a dot runs along each synapse whenever
        its presynaptic neuron fires above its calm rate (the live activity)."""
        ends = getattr(self, "path_ends", {})
        self.path_clear_rect = None
        if not ends:
            return
        w = VIEW_SIZES["big"][0]
        pix = self.view.spark_pix["big"]
        rates = self.brain.sim.activity.rates()
        calm = self.view.calm

        def at(j):
            q = pix[j] if 0 <= j < len(pix) else -1
            return None if q < 0 else (rect.x + q % w, rect.y + q // w)

        cols = ((120, 255, 170), (110, 200, 255), (255, 210, 110), (230, 140, 255), (255, 140, 140))
        top = max((p["strength"] for p in self.paths), default=1.0)
        for rank, pth in reversed(list(enumerate(self.paths))):
            col = cols[rank % len(cols)]
            fade = 0.35 + 0.65 * (pth["strength"] / top) ** 0.25
            c = tuple(int(v * fade) for v in col)
            pts = [at(j) for j in pth["nodes"]]
            for (a, b), pre, wgt in zip(zip(pts, pts[1:]), pth["nodes"], pth["weights"]):
                if a is None or b is None:
                    continue
                pygame.draw.line(surf, c, a, b, 3 if rank == 0 else 2)
                act = float(rates[pre] / max(calm[pre], 1e-4))
                if act > 1.2 and not self.calm_fx:        # live: spikes travelling along the synapse
                    for f in ((now * 1.6 + rank * 0.2) % 1.0, (now * 1.6 + rank * 0.2 + 0.5) % 1.0):
                        q = (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
                        aacircle(surf, q, 3, (255, 255, 255) if wgt > 0 else (255, 110, 110))
            for j, q in zip(pth["nodes"], pts):
                if q is not None:
                    glow = min(1.0, float(rates[j] / max(calm[j], 1e-4)) / 3)
                    aacircle(surf, q, 4 + int(3 * glow), tuple(int(v * (0.5 + 0.5 * glow)) for v in col))
        for what, label in (("from", "FROM"), ("to", "TO")):
            q = at(ends[what]) if what in ends else None
            if q:
                self._text(surf, label, (q[0] + 8, q[1] - 16), INK, self.f_small)
        # the list of paths, bottom left
        br = self.brain
        lines = []
        if "from" in ends and "to" in ends:
            lines.append(f"{br.types[ends['from']] or '#%d' % ends['from']} -> {br.types[ends['to']] or '#%d' % ends['to']}"
                         f": {'the strongest paths, up to 3 synapses' if self.paths else 'no path within 3 synapses'}")
            for rank, pth in enumerate(self.paths):
                names = " > ".join((br.types[j] or f"#{j}") for j in pth["nodes"])
                lines.append((rank, f"{'+' if pth['sign'] > 0 else '-'} {pth['strength']:.1e}  {names}"))
        else:
            lines.append("Path tracer: now pick the other end (inspect a neuron, PATH FROM / TO HERE)")
        box = pygame.Rect(rect.x + 8, rect.bottom - 14 - 18 * len(lines) - 8, min(rect.w - 16, 520), 18 * len(lines) + 12)
        bg = pygame.Surface(box.size, pygame.SRCALPHA)
        pygame.draw.rect(bg, (8, 10, 16, 215), bg.get_rect(), border_radius=8)
        surf.blit(bg, box)
        for n, line in enumerate(lines):
            if isinstance(line, tuple):
                rank, text = line
                aacircle(surf, (box.x + 12, box.y + 14 + n * 18), 4, cols[rank % len(cols)])
                self._text(surf, text[:70], (box.x + 22, box.y + 6 + n * 18), TEXT, self.f_small)
            else:
                self._text(surf, line, (box.x + 8, box.y + 6 + n * 18), INK, self.f_small)
        self.path_clear_rect = pygame.Rect(box.right - 60, box.y + 4, 54, 18)
        pygame.draw.rect(surf, (40, 46, 58), self.path_clear_rect, border_radius=4)
        self._text(surf, "CLEAR", self.path_clear_rect.center, INK, self.f_small, "center")

    def flip_neuron(self, i: int) -> None:
        """Flip (or unflip) one neuron's excitatory/inhibitory sign, from the inspector."""
        rows = set(self.wiring.flip_rows)
        rows.symmetric_difference_update({int(i)})
        nt, conf, _ = self.transmitter_of(i)
        self.set_wiring(self.wiring.with_rows(sorted(rows)), note=False)
        self.note(f"WIRING   {self.types_of(i)} sign {'flipped' if int(i) in rows else 'restored'} "
                  f"({nt}, {'no confidence' if conf != conf else f'{conf:.0%} confident'})")

    def types_of(self, i: int) -> str:
        return str(self.brain.types[i]) or "untyped"

    def transmitter_of(self, i: int) -> tuple[str, float, str]:
        """(transmitter, the dataset's confidence in it, where it came from) for one neuron, from the brain pack."""
        g = self.graph
        nt = getattr(g, "nt", None)
        if nt is None or i is None or i >= len(nt):
            return "", float("nan"), ""
        conf = getattr(g, "nt_conf", None)
        src = getattr(g, "nt_source", None)
        return (str(nt[i]), float(conf[i]) if conf is not None else float("nan"),
                str(src[i]) if src is not None else "")

    # --- sound, saving, help ---------------------------------------------------------------
    def update_stethoscope_target(self) -> None:
        if not self.cfg["audio.stethoscope_enabled"]:
            for slot in getattr(self, "flies", []):
                slot.brain.set_stethoscope_indices(None)
            return

        target = self.cfg["audio.stethoscope_target"]
        indices = None

        # 1. Selected region in big brain view
        if getattr(self, "selected_region", None):
            rname = self.selected_region
            v = self.view
            if hasattr(v, "region_to_id") and rname in v.region_to_id:
                rid = v.region_to_id[rname]
                indices = np.flatnonzero(v.region_id == rid)
        # 2. Inspected neuron in brain view
        elif getattr(self, "inspect", None) is not None:
            info = self.inspect
            i = info.get("i") if isinstance(info, dict) else None
            if i is not None:
                indices = np.array([i], dtype=np.int32)
        # 3. Target setting
        if indices is None:
            v = self.view
            br = self.brain
            if target == "antennal_lobe":
                if hasattr(v, "region_to_id") and "antennal_lobe" in v.region_to_id:
                    indices = np.flatnonzero(v.region_id == v.region_to_id["antennal_lobe"])
                else:
                    indices = br.sense.get(("smell", None), np.array([], np.int32))
            elif target == "mushroom_body":
                if hasattr(v, "region_to_id") and "mushroom_body" in v.region_to_id:
                    indices = np.flatnonzero(v.region_id == v.region_to_id["mushroom_body"])
                else:
                    indices = br.kc
            elif target == "central_complex":
                if hasattr(v, "region_to_id") and "central_complex" in v.region_to_id:
                    indices = np.flatnonzero(v.region_id == v.region_to_id["central_complex"])
            elif target == "optic_lobes":
                if hasattr(v, "region_to_id") and "optic_lobes" in v.region_to_id:
                    indices = np.flatnonzero(v.region_id == v.region_to_id["optic_lobes"])
            elif target == "motor":
                indices = np.flatnonzero(np.isin(br.superclass, ("descending_neuron", "motor")))
            elif target == "whole_brain":
                indices = np.arange(br.n, dtype=np.int32)

        idx_arr = indices if (indices is not None and len(indices) > 0) else None
        for slot in getattr(self, "flies", []):
            slot.brain.set_stethoscope_indices(idx_arr)

    def stethoscope_target_label(self) -> str:
        if getattr(self, "selected_region", None):
            return self.selected_region.replace("_", " ").title()
        if getattr(self, "inspect", None) is not None:
            info = self.inspect
            t = info.get("type", "Neuron") if isinstance(info, dict) else str(info)
            return t
        tgt = self.cfg["audio.stethoscope_target"]
        return tgt.replace("_", " ").title()

    def toggle_stethoscope(self) -> None:
        val = not bool(self.cfg["audio.stethoscope_enabled"])
        self.cfg.set("audio.stethoscope_enabled", val)
        self.cfg.save()
        self.update_stethoscope_target()
        status = "ON" if val else "OFF"
        tgt = self.stethoscope_target_label()
        self.note(f"STETHOSCOPE {status} ({tgt})", source="rule")

    def cycle_stethoscope_target(self) -> None:
        targets = ("mushroom_body", "antennal_lobe", "central_complex", "optic_lobes", "motor", "whole_brain")
        cur = self.cfg["audio.stethoscope_target"]
        nxt = targets[(targets.index(cur) + 1) % len(targets)] if cur in targets else targets[0]
        self.cfg.set("audio.stethoscope_target", nxt)
        self.cfg.save()
        self.selected_region = None
        self.update_stethoscope_target()
        tgt = self.stethoscope_target_label()
        self.note(f"STETHOSCOPE TARGET {tgt}", source="rule")

    def _sound_update(self, now: float) -> None:
        snd, fly = self.sound, self.fly
        flying = fly.flying and now < fly.escape_until
        snd.loop("wings", f"buzz{min(3, max(0, int((fly.power - 0.6) * 3)))}" if flying else None, 0.55)
        name = TOOLS[self.tool][0]
        tool = None
        if self.torching and self.report is None:
            tool = "sizzle" if name == "torch" else "hiss" if name in ("cleaner", "freeze") else None
        snd.loop("tool", tool, 0.6)
        arena = ARENAS[self.arena_i]
        snd.loop("arena", "whoosh" if arena == "fan" else "hum" if arena == "lamp" else None, 0.5)
        if self.cfg["audio.stethoscope_enabled"]:
            spikes_count = self.brain.take_stethoscope_spikes()
            if spikes_count > 0:
                snd.play_spike_click(spikes_count)

    def capture(self) -> None:
        small = pygame.transform.smoothscale(self.screen, GIF_SIZE)
        self.frames.append(pygame.image.tobytes(small, "RGB"))

    def _save_dir(self) -> Path:
        """Pictures\\Kick the Fly on Windows, <XDG Pictures>/Kick the Fly on Linux (see paths.py)."""
        return paths.ensure_dir(paths.get().pictures_dir, Path.cwd() / "Kick the Fly saves")

    def media_path(self, ext: str) -> Path:
        """A new file in the pictures folder; two saves in the same second no longer overwrite each other."""
        d = self._save_dir()
        stem = f"kick-the-fly-{time.strftime('%Y%m%d-%H%M%S')}"
        path, n = d / f"{stem}.{ext}", 2
        while path.exists():
            path, n = d / f"{stem}-{n}.{ext}", n + 1
        return path

    def saved_note(self, path: Path) -> None:
        self.saved_msg = (f"saved {path.name} in {path.parent.name}", time.perf_counter())

    def save_png(self, clean: bool | None = None) -> None:
        if clean is None:
            clean = bool(self.cfg.get("graphics.clean_capture", True))
        scale = int(self.cfg.get("graphics.photo_scale", 2))
        path = self.media_path("png")
        surf = self.screen
        if clean and getattr(self, "_clean_frame", None) is not None:
            surf = self._clean_frame
        if scale > 1:
            w, h = surf.get_size()
            surf = pygame.transform.smoothscale(surf, (w * scale, h * scale))
        save_image(surf, path)
        self.saved_note(path)
        self.sound.play("shutter")

    def save_gif(self, frames: list | None = None) -> None:
        frames = list(self.frames) if frames is None else list(frames)
        if len(frames) < 3:
            self.saved_msg = ("not enough footage yet", time.perf_counter())
            return
        path = self.media_path("gif")
        self.saved_msg = ("saving GIF...", time.perf_counter())

        def work():
            try:
                from PIL import Image

                imgs = [Image.frombytes("RGB", GIF_SIZE, f).convert("P", palette=Image.ADAPTIVE, colors=160) for f in frames]
                imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=66, loop=0)
                self.saved_note(path)
            except Exception as e:                   # e.g. Pillow missing in a source checkout
                self.saved_msg = (f"GIF failed: {e}", time.perf_counter())

        threading.Thread(target=work, daemon=True).start()
        self.sound.play("click")

    def toggle_timelapse(self) -> None:
        if not getattr(self, "timelapse_recording", False):
            self.timelapse_recording = True
            self.timelapse_frames = []
            self.timelapse_start_t = time.perf_counter()
            sp = int(self.cfg.get("graphics.timelapse_speedup", 5))
            fmt = str(self.cfg.get("graphics.timelapse_format", "mp4")).upper()
            tgt = str(self.cfg.get("graphics.timelapse_target", "brain")).title()
            self.note(f"TIME-LAPSE RECORDING ({tgt}, {sp}x, {fmt}) - L to stop", source="rule")
            self.sound.play("click")
        else:
            self.timelapse_recording = False
            self.save_timelapse()

    def capture_timelapse_frame(self) -> None:
        if not getattr(self, "timelapse_recording", False):
            return
        target = self.cfg.get("graphics.timelapse_target", "brain")
        w, h = getattr(self, "timelapse_size", (600, 340))
        if target == "brain" and hasattr(self, "_view_surface"):
            surf = self._view_surface("big")
            scaled = pygame.transform.smoothscale(surf, (w, h))
        else:
            surf = self.screen
            scaled = pygame.transform.smoothscale(surf, (w, h))
        self.timelapse_frames.append(pygame.image.tobytes(scaled, "RGB"))

    @property
    def video_recording(self) -> bool:
        return getattr(self, "video_recorder", None) is not None and self.video_recorder.is_recording

    def toggle_video_recording(self, output_path: str | Path | None = None) -> None:
        """Shift+R: start or stop a video of any length (MP4 through ffmpeg, else a GIF), saved with the screenshots
        unless output_path (--record-video PATH) says otherwise."""
        if not getattr(self, "video_recorder", None):
            from kickthefly.game.video_recorder import VideoRecorder
            self.video_recorder = VideoRecorder()
        rec = self.video_recorder
        if not rec.is_recording:
            from kickthefly.game.video_recorder import VIDEO_FPS
            disp = pygame.display.get_surface()
            w, h = disp.get_size() if disp else self.screen.get_size()
            out = rec.start(output_path or self.media_path("mp4"), width=w, height=h, fps=VIDEO_FPS)
            fmt = out.suffix.upper().lstrip(".")
            self.note(f"VIDEO RECORDING ({fmt}) -> {out.name}  (Shift+R: stop)", source="rule")
            self.sound.play("click")
        else:
            out = rec.stop()
            self.sound.play("shutter")
            if out:
                self.saved_note(out)
                self.note(f"VIDEO SAVED: {out.name}", source="rule")
            else:
                self.note("VIDEO FAILED: nothing was saved (see the log)", source="rule")

    def capture_video_frame(self) -> None:
        if getattr(self, "video_recorder", None) and self.video_recorder.is_recording:
            self.video_recorder.write_frame_surface(self.screen)

    def save_timelapse(self) -> None:
        frames = list(getattr(self, "timelapse_frames", []))
        if len(frames) < 3:
            self.saved_msg = ("not enough time-lapse footage", time.perf_counter())
            return
        speedup = int(self.cfg.get("graphics.timelapse_speedup", 5))
        fmt = str(self.cfg.get("graphics.timelapse_format", "mp4")).lower()
        size = getattr(self, "timelapse_size", (600, 340))
        path = self.media_path(fmt)
        self.saved_msg = (f"exporting {speedup}x time-lapse ({fmt.upper()})...", time.perf_counter())

        def work():
            try:
                import shutil
                import subprocess
                stride = max(1, speedup // 2) if speedup >= 5 else 1
                sub_frames = frames[::stride]
                actual_fps = min(60, max(15, int(15 * (speedup / stride))))
                exported_path = path

                if fmt == "mp4" and shutil.which("ffmpeg"):
                    cmd = [
                        "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
                        "-s", f"{size[0]}x{size[1]}", "-pix_fmt", "rgb24", "-r", str(actual_fps),
                        "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        str(path)
                    ]
                    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    for f in sub_frames:
                        proc.stdin.write(f)
                    proc.stdin.close()
                    proc.wait()
                    if proc.returncode != 0:
                        raise RuntimeError(f"ffmpeg returned {proc.returncode}")
                else:
                    from PIL import Image
                    gif_path = path.with_suffix(".gif")
                    exported_path = gif_path
                    duration = max(20, int(66 / (speedup / stride)))
                    imgs = [Image.frombytes("RGB", size, f).convert("P", palette=Image.ADAPTIVE, colors=160) for f in sub_frames]
                    imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=duration, loop=0)

                self.saved_note(exported_path)
            except Exception as e:
                self.saved_msg = (f"Time-lapse export failed: {e}", time.perf_counter())

        threading.Thread(target=work, daemon=True).start()
        self.sound.play("shutter")

    def _draw_help(self, surf) -> None:
        step = min(30, (H - 40 - 64) // len(HELP))                # 3.0: two more rows; squeeze the spacing, never run off-screen
        height = 64 + step * len(HELP)
        panel = pygame.Rect(165, max(12, min(110, (H - height) // 2)), 560, height)
        pygame.draw.rect(surf, (18, 21, 28), panel, border_radius=16)
        pygame.draw.rect(surf, BORDER, panel, 1, border_radius=16)
        self._text(surf, "CONTROLS", (panel.x + 24, panel.y + 16), INK, self.f_head)
        for k, (key, what) in enumerate(HELP):
            y = panel.y + 54 + k * step
            self._text(surf, key, (panel.x + 30, y), AMBER, self.f_bold)
            self._text(surf, what, (panel.x + 120, y), TEXT, self.f_text)

    def _draw_memory(self, surf) -> None:
        mem = self.brain.memory
        learned = [(k, *self.memory_of(k)) for k in (mem.templates if mem is not None else {})]
        x, y, w = 10, 380, 236
        rows = sorted((r for r in learned if max(r[1], r[2]) >= 0.02), key=lambda r: -max(r[1], r[2]))[:4]
        h = 46 + 16 * max(1, len(rows))
        card = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(card, (10, 12, 18, 180), card.get_rect(), border_radius=10)
        surf.blit(card, (x, y))
        self._text(surf, "MEMORY", (x + 12, y + 8), LABEL, self.f_small)
        self._text(surf, "mushroom body", (x + w - 12, y + 8), DIM, self.f_small, "topright")
        yy = y + 28
        if not rows:
            self._text(surf, "nothing learned yet", (x + 12, yy), DIM, self.f_small)
        for key, fear, like in rows:
            good = like > fear
            v = like if good else fear
            self._text(surf, f"{key}", (x + 12, yy - 3), TEXT, self.f_small)
            self._bar(surf, x + 84, yy, w - 150, v, (255, 150, 190) if good else (240, 150, 60))
            self._text(surf, f"{'likes' if good else 'fears'} {v:.2f}", (x + w - 12, yy - 3), TEXT, self.f_small, "topright")
            yy += 16
        self._text(surf, "real synapses · T: train", (x + 12, y + h - 16), DIM, self.f_small)

    def _above_head(self):
        """Where a popup over the fly goes (the 3D game overrides this)."""
        return self.fly.p[HEAD] + (0, -70)

    def _panel_image(self, surf: pygame.Surface) -> pygame.Surface:
        """In see-through panel modes the brain image's black background turns transparent; bright neurons stay."""
        pa = getattr(self, "panel_alpha", 255)
        if pa >= 255:
            return surf
        out = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        out.blit(surf, (0, 0))
        rgb = pygame.surfarray.pixels3d(out)
        alpha = pygame.surfarray.pixels_alpha(out)
        alpha[:] = np.clip(rgb.max(axis=2).astype(np.int16) * 2 + pa // 3, 0, 255)
        del rgb, alpha
        return out

    def hit(self, slot: "FlySlot", i: int, strength: float) -> None:
        key = particle_region(i)
        slot.pending_hits[key] = max(slot.pending_hits.get(key, 0.0), strength)
        self.record_event("hit", key[0], TOOLS[self.tool][0], strength, slot)

    def damage(self, slot: "FlySlot", amount: float, source: str) -> None:
        if amount > slot.pending_damage:
            slot.damage_src = source
        slot.pending_damage += amount

    def _nearest_fly(self, pos, max_d: float) -> tuple["FlySlot | None", "int | None"]:
        """The living fly (and its nearest particle) closest to a point, across every spawned fly."""
        best_slot, best_i, best_d = None, None, max_d
        for slot in self.flies:
            fly = slot.fly
            d = np.hypot(*(fly.p - pos).T) - RADIUS
            i = int(np.argmin(d))
            if d[i] < best_d:
                best_slot, best_i, best_d = slot, i, float(d[i])
        return best_slot, best_i

    def _flies_within(self, pos, radius: float) -> list["FlySlot"]:
        """Every fly with at least one particle within radius of a point (for area-effect tools)."""
        out = []
        for slot in self.flies:
            if np.any(np.hypot(*(slot.fly.p - pos).T) < radius):
                out.append(slot)
        return out

    def popup(self, pos, text: str, color=(255, 245, 235), force=False) -> None:
        now = self.clock.now
        if self.popups and now - self.popups[-1][3] < 0.3 and not force:
            return
        x = float(np.clip(pos[0], 90, PLAY_W - 90))
        y = float(np.clip(pos[1], CEIL + 60, FLOOR - 20))
        self.popups.append([x, y, text, now, color, POPUP_SOURCE.get(text)])

    def puff(self, pos, n: int, spread: float = 3.0) -> None:
        for _ in range(n):
            self.dust.append([pos[0], pos[1], random.uniform(-spread, spread), random.uniform(-spread, 0.3),
                              self.clock.now, random.uniform(0.35, 0.8), random.uniform(3, 7)])

    def note(self, text: str, source: str | None = None) -> None:
        src = source or reaction_source(text)
        self.log.append((self.clock.now, text, src))
        self.log = self.log[-7:]
        self.record_event("note", text.split()[0] if text.split() else "note", text.strip(), 0.0,
                          extra=dict(source=src))

    def record_event(self, kind: str, label: str, detail: str = "", value: float = 0.0, slot=None,
                     extra: dict | None = None) -> None:
        """Log a tool use, hit, reaction, arena or surgery change into a running Lab recording (else a no-op)."""
        job = getattr(self, "recording", None)
        if job is None:
            return
        if slot is not None and slot is not job["slot"]:          # only the fly being recorded
            return
        detail = detail if not extra else f"{detail} [{extra.get('source', '')}]".strip()
        job["rec"].log_event(kind, label, detail, value)

    def _nearest_decoy(self, pos, radius: float = 45):
        if not hasattr(self, "decoys") or not self.decoys:
            return None
        p = np.array(pos, float)
        close = [d for d in self.decoys if float(np.hypot(*(d["p"] - p))) < radius]
        if not close:
            return None
        return min(close, key=lambda d: float(np.hypot(*(d["p"] - p))))

    def remove_decoy(self, dec) -> None:
        if hasattr(self, "decoys") and dec in self.decoys:
            drop_item(self.decoys, dec)
            if getattr(self, "grabbed_decoy", None) is dec:
                self.grabbed_decoy = None
            self.popup(dec["p"], "DECOY REMOVED", (200, 180, 220))
            self.sound.play("pop")

    # --- tools ---
    def use_tool(self, pos, now: float) -> None:
        if self.cfg["brain.autopilot"]:
            return
        name = TOOLS[self.tool][0]
        self.tool_uses += 1
        self.last_tool_used = name
        self.record_event("tool", name, f"x={pos[0]:.0f} y={pos[1]:.0f}")
        if name in ("hand", "flick", "swatter", "zapper"):
            slot, _ = self._nearest_fly(pos, 60)
            if slot is not None and slot.fly.frozen_at is not None and slot.fly.shattered_at is None:
                self._shatter(slot, now)
                return
        if name == "hand":
            dec = self._nearest_decoy(pos, 45)
            slot, i = self._nearest_fly(pos, 40)
            if dec is not None and slot is None:
                self.grabbed_decoy = dec
                return
            if slot is not None and i is not None and slot.fly.frozen_at is None:
                slot.fly.grabbed = i
                slot.fly.last_hit_x = pos[0]
                self.hit(slot, i, 0.25)
                self.focus = self.flies.index(slot)
            elif dec is not None:
                self.grabbed_decoy = dec
                return
        elif name == "flick":
            slot, _ = self._nearest_fly(pos, 60)
            if slot is not None:
                fly = slot.fly
                d = np.hypot(*(fly.p - pos).T) - RADIUS
                near = np.flatnonzero(d < 60)
                fly.last_hit_x = pos[0]
                for i in near:
                    dirv = fly.p[i] - pos
                    dirv = dirv / (np.hypot(*dirv) or 1) + (0, -0.6)
                    s = 1 - max(d[i], 0) / 60
                    fly.impulse(i, dirv * (10 + 16 * s))
                    self.hit(slot, i, 0.35 + 0.4 * s)
                fly.stun(now, 0.35)
                self.damage(slot, 4, "a flick")
                self.popup(pos, "FLICK!")
                self.sound.play("flick")
                self.focus = self.flies.index(slot)
        elif name == "swatter":
            if not self.swats or now - self.swats[-1][1] > 0.35:
                self.swats.append([pos, now, False])
        elif name == "bomb" and len(self.bombs) < 3:
            self.bombs.append({"p": np.array(pos, float), "v": np.zeros(2), "t": now})
        elif name in ("torch", "cleaner", "freeze"):
            self.torching = True
        elif name == "zapper" and now >= self.zap_ready:
            self._zap(pos, now)
        elif name == "spider" and self.spider is None and any(not s.fly.dead for s in self.flies):
            # it drops on its thread to the floor under the click, then hunts: the 3D spider's drop (kick3d._spider3d)
            self.spider = {"p": np.array([pos[0], CEIL + 4.0]), "state": "drop", "bite_at": 0.0, "bites": 0, "t": now}
            self.sound.play("drop")
            self.popup((pos[0], CEIL + 70), "A SPIDER!", (200, 200, 210))
        elif name in ("frog", "dragonfly", "mantis"):
            if self.preds.spawn(name, pos):
                self.sound.play("drop")
        elif name in ("sugar", "fruit") and len(self.sugars) < 3:
            # fruit is eaten exactly as sugar is (same neurons, same rules): only the item and its look differ
            self.sugars.append({"p": np.array(pos, float), "v": 0.0, "left": 1.0, "fruit": name == "fruit"})
            self.sound.play("pop")
        elif name == "alcohol" and len(self.alcohols) < 3:
            self.alcohols.append({"p": np.array(pos, float), "v": 0.0, "left": 1.0})
            self.sound.play("drop")
        elif name == "laser":
            if not hasattr(self, "laser_state"):
                from kickthefly.lab.laser import LaserState
                self.laser_state = LaserState()
            self.laser_state.trigger_press(now)
            self.torching = True
        elif name == "cva":
            self.sound.play("pop")
            self.popup(pos, "cVA PUFF", (255, 180, 100))
            if not hasattr(self, "cva_puffs"):
                self.cva_puffs = []
            self.cva_puffs.append({"p": np.array(pos, float), "t": now, "duration": 0.65, "reach": 250.0})
            for slot in self.flies:
                dist = float(np.hypot(*(slot.fly.p[HEAD] - pos)))
                if dist < 250:
                    strength = 0.8 * (1.0 - dist / 250)
                    slot.brain.poke("scent", "cva", strength)
        elif name == "decoy":
            if not hasattr(self, "decoys"):
                self.decoys = []
            if len(self.decoys) < 3:
                self.sound.play("drop")
                self.popup(pos, "DECOY FEMALE", (200, 180, 220))
                self.decoys.append({"p": np.array(pos, float), "v": np.zeros(2), "t": now})
            else:
                self.popup(pos, "MAX 3 DECOYS", (200, 180, 220), force=True)
                self.note("DECOY    limit reached (max 3)")

    def _spray(self, mouse, now: float, kind: str) -> None:
        """Mist toward the nearest fly, but drenches every fly it passes over. Brake cleaner soaks (dissolves; smell
        and taste neurons); freeze spray chills (cold-sensing neurons)."""
        nozzle = np.array(mouse, float)
        aim_slot, _ = self._nearest_fly(mouse, 1e9)
        to_fly = (aim_slot.fly.p[THX] - nozzle) if aim_slot is not None else np.array([1.0, 0.0])
        dist = float(np.hypot(*to_fly))
        aim = to_fly / dist if dist > 1 else np.array([1.0, 0.0])
        self.torch_aim = aim
        for _ in range(5):
            a = math.atan2(aim[1], aim[0]) + random.uniform(-0.3, 0.3)
            sp = random.uniform(6, 10)
            self.mist.append([nozzle[0], nozzle[1], math.cos(a) * sp, math.sin(a) * sp, now, random.uniform(0.35, 0.6), kind])
        for slot in self.flies:
            fly = slot.fly
            rel = fly.p - nozzle
            d = np.hypot(*rel.T)
            cosang = (rel @ aim) / np.maximum(d, 1e-6)
            if fly.dissolved_at is not None or fly.frozen_at is not None or not np.any((d < 230) & (cosang > 0.8)):
                continue
            fly.last_hit_x = nozzle[0]
            if kind == "cleaner":
                fly.soak = min(1.0, fly.soak + 0.04)
            else:
                fly.frost = min(0.9 if self.immortal else 1.0, fly.frost + 0.006)
            if fly.dead:
                continue
            if kind == "cleaner":
                slot.brain.poke("smell", None, 1.0)
                slot.brain.poke("taste", None, 0.8)
                self.damage(slot, 0.12, "brake cleaner")
                words, col = ("FSSSSH!", "MELTING!", "IT BURNS!"), (170, 230, 255)
            else:
                slot.brain.poke("cold", None, 1.0)
                for region, side in TORCH_KEYS[:-1]:       # ice on the cuticle
                    slot.brain.poke(region, side, 0.35)
                self.damage(slot, 0.1, "freezing")
                words, col = ("SO COLD!", "BRRRR!", "ICING!"), (190, 230, 255)
            if int(now * 2) != int((now - 1 / 60) * 2):
                slot.hits += 1
            if random.random() < 0.02:
                self.popup(fly.p[HEAD] + (0, -60), random.choice(words), col)

    def _zap(self, pos, now: float) -> None:
        """Electric shock: every touch neuron plus current into a random 30% of all neurons for 40 ms (the sim can't
        place a current path, so the shocked neurons are random)."""
        self.zap_ready = now + 0.3
        self.sound.play("zap")
        slot, _ = self._nearest_fly(pos, 280 + float(RADIUS.max()))
        fly = slot.fly if slot is not None else None
        if fly is None or np.hypot(*(fly.p[THX] - pos)) > 280 or fly.dissolved_at is not None or fly.shattered_at is not None:
            self.bolts.append([np.array(pos, float), np.array(pos, float) + (0, 60), now])
            return
        target = fly.p[THX]
        self.bolts.append([np.array(pos, float), target.copy(), now])
        fly.zap_until = now + 0.3
        fly.char = min(1.0, fly.char + 0.06)
        fly.last_hit_x = pos[0]
        for i in range(N_P):
            fly.impulse(i, np.random.normal(0, 3, 2))
        self.shake_until = now + 0.12
        if fly.dead:
            return
        for region, side in TORCH_KEYS[:-1]:
            slot.brain.poke(region, side, 1.0)
        slot.brain.poke("all", None, 0.0)
        fly.stun(now, 1.0)
        self.damage(slot, 16, "the zapper")
        self.popup(target + (0, -70), random.choice(("BZZZT!", "ZAP!", "KRZZT!")), (200, 235, 255))

    def _shatter(self, slot: "FlySlot", now: float) -> None:
        fly = slot.fly
        fly.shattered_at = now
        self.shake_until = now + 0.25
        self.sound.play("shatter")
        for i in range(N_P):
            for _ in range(3):
                self.shards.append([fly.p[i, 0], fly.p[i, 1], random.uniform(-9, 9), random.uniform(-14, 2),
                                    random.uniform(0, 6.28), random.uniform(5, 13), random.random() < 0.35])
        self.popup(fly.p[THX] + (0, -60), "SHATTERED!", (200, 235, 255), force=True)

    def _effects(self, now: float) -> None:
        """Brake cleaner, freezing, venom and sugar over time. Their brain damping is a game rule (see docstring)."""
        for slot in self.flies:
            self._effects_one(slot, now)
        self._spider(now)
        self.preds.step(now)
        self._sugar(now)
        self._alcohol(now)
        self._decoy(now)

    def _effects_one(self, slot: "FlySlot", now: float) -> None:
        fly = slot.fly
        if getattr(fly, "inebriation", 0.0) > 0:
            fly.inebriation = max(0.0, fly.inebriation - 0.00035)   # ethanol metabolism (~45s)
        if self.immortal:                                    # melting and venom stop short and wear off
            fly.melt = min(fly.melt, 0.85)
            if fly.soak < 0.05:
                fly.melt = max(0.0, fly.melt - 0.002)
            fly.venom = max(0.0, fly.venom - 0.002)
        if fly.dissolved_at is None and fly.soak > 0.02:
            fly.melt = min(0.85 if self.immortal else 1.0, fly.melt + 0.0045 * fly.soak)
            fly.soak *= 0.985 if self.immortal else 0.997     # immortal: the solvent evaporates fast
            if not fly.dead:
                self.damage(slot, 0.25 * fly.soak, "brake cleaner")
                for region, side in TORCH_KEYS[:-1]:  # the solvent eating the cuticle hits the touch neurons too
                    slot.brain.poke(region, side, 0.5 * fly.soak)
        if fly.melt >= 1.0 and fly.dissolved_at is None:
            fly.dissolved_at = now
            self.puff((fly.p[THX, 0], FLOOR - 10), 14, 3)
            self.popup(fly.p[THX] + (0, -60), "DISSOLVED", (170, 230, 255), force=True)
            self.sound.play("squish")
            if not fly.dead:
                self.damage(slot, MAX_HEALTH, "brake cleaner")
        if fly.frozen_at is None and fly.frost > 0:
            if not (self.torching and TOOLS[self.tool][0] == "freeze"):
                fly.frost = max(0.0, fly.frost - 0.0015)   # thawing
            if fly.frost >= 1.0:
                fly.frozen_at = now
                self.popup(fly.p[THX] + (0, -70), "FROZEN SOLID", (190, 230, 255), force=True)
                if not fly.dead:
                    self.damage(slot, MAX_HEALTH, "freezing")
        if not fly.dead:
            slot.brain.sedation = max(0.25 * fly.melt ** 2.5,       # solvent, mild at first so it can still flee
                                      0.2 * fly.frost ** 2,         # cold slows neurons
                                      0.3 * fly.venom ** 1.5)       # spider venom

    def _spider(self, now: float) -> None:
        sp = self.spider
        if sp is None:
            return
        if sp["state"] == "drop":                          # the 3D drop in 2D pixels: 0.04 m a frame to 0.12 m up
            sp["p"][1] += SPIDER_DROP
            if sp["p"][1] >= SPIDER_DROP_TO:
                sp["p"][1], sp["state"] = SPIDER_DROP_TO, "hunt"
            return
        candidates = [s for s in self.flies if not (s.fly.dead or s.fly.dissolved_at is not None or s.fly.shattered_at is not None)]
        if sp["state"] == "hunt":
            if not candidates:
                sp["state"] = "leave"
                return
            slot = min(candidates, key=lambda s: float(np.hypot(*(s.fly.p[THX] - sp["p"]))))
            fly = slot.fly
            d = fly.p[THX] + (0, -18) - sp["p"]
            dist = float(np.hypot(*d))
            cyc = sp.setdefault("cycle", SpiderCycle())      # 3.1.0 task 1: walk -> windup -> strike -> recover (GAME RULE)
            if cyc.state in ("walk", "windup") and abs(d[0]) > 1.0:
                sp["face_x"] = 1.0 if d[0] > 0 else -1.0
            evs = cyc.step(1.0 / 60.0, dist, 26.0)
            if cyc.walking and dist > 26:
                sp["p"] += d / dist * min(3.4, dist)
            if "bite" in evs:
                sp["bite_at"] = now
                sp["bites"] += 1
                fly.venom = min(1.0, fly.venom + 0.3)
                fly.last_hit_x = sp["p"][0]
                fly.hurt = 1.0
                slot.brain.poke("body", None, 0.9)
                slot.brain.poke("legs", "L", 0.6)
                slot.brain.poke("legs", "R", 0.6)
                self.damage(slot, 16, "a spider")
                self.sound.play("chomp")
                self.popup(sp["p"] + (0, -40), random.choice(("CHOMP!", "BITE!", "SLURP!")), (230, 120, 120))
                if sp["bites"] >= 2 and not fly.wrapped:
                    fly.wrapped = True
                    sp["target"] = slot
                    self.note("WRAPPED  in spider silk")
                if self.immortal and sp["bites"] >= 5:          # it can't be eaten: it breaks out and the spider leaves
                    fly.wrapped, fly.grabbed = False, None
                    sp["state"] = "leave"
                    self.popup(fly.p[HEAD] + (0, -60), "BROKE FREE!", (255, 225, 120), force=True)
                    self.note("BROKE FREE of the silk")
                    return
            if fly.wrapped:
                fly.grabbed = THX
        elif sp["state"] in ("leave", "carry"):
            wrapped = sp.get("target")
            if wrapped is not None and wrapped.fly.wrapped and wrapped.fly.dead and wrapped.fly.frozen_at is None:
                sp["state"] = "carry"
            sp["p"][1] -= 2.2
            if sp["state"] == "carry" and wrapped is not None:
                wrapped.fly.grabbed = THX
            if sp["p"][1] < CEIL - 60:
                self.spider = None
                if wrapped is not None and wrapped.fly.wrapped:
                    wrapped.fly.grabbed = None

    def _sugar(self, now: float) -> None:
        """Sugar: each fly walks over and eats the nearest pile (game rule). Eating drives taste neurons and the PAM
        reward neurons and heals it."""
        for s in self.sugars:
            if s["p"][1] < FLOOR - 8:
                s["v"] += 0.9
                s["p"][1] = min(FLOOR - 8, s["p"][1] + s["v"])
        if not self.sugars:
            return
        for slot in self.flies:
            fly = slot.fly
            if fly.dead or fly.wrapped or fly.frozen_at is not None or fly.grabbed is not None:
                continue
            s = min(self.sugars, key=lambda s: abs(s["p"][0] - fly.p[THX, 0]))
            dx = s["p"][0] - fly.p[HEAD, 0]
            on_floor = fly.p[THX, 1] > FLOOR - STAND - 30 and now >= fly.escape_until
            if abs(dx) > 30:
                if on_floor and now >= fly.stun_until:
                    fly.facing = 1 if dx > 0 else -1
                    fly.walk_until, fly.run = now + 0.2, False
                continue
            if not on_floor or s["p"][1] < FLOOR - 10:
                continue
            if now >= fly.eating_until:
                self.popup(fly.p[HEAD] + (0, -60), random.choice(("YUM!", "SWEET!", "NOM NOM")), (255, 160, 190))
                self.sound.play("yum")
                self.note("EATING   sugar: taste + PAM reward")
            fly.eating_until = now + 0.4
            s["left"] -= 1 / 240
            t_gain = 0.5 * (self.pet.get_sugar_gain_multiplier() if self.pet else 1.0)
            r_gain = 0.4 * (self.pet.get_pam_reward_multiplier() if self.pet else 1.0)
            slot.brain.poke("taste", None, t_gain)
            slot.brain.poke("sweet", None, t_gain, recruit=0.6)          # the sugar-pathway taste neurons (assays.py)
            slot.brain.poke("reward", None, r_gain)
            if self.pet:
                self.pet.feed(amount=0.003, source="sugar", save=False)   # every eating frame: saved on quit
            fly.health = min(MAX_HEALTH, fly.health + 0.15)
            if s["left"] <= 0:
                drop_item(self.sugars, s)
                break

    def _alcohol(self, now: float) -> None:
        """Alcohol drop: each fly walks over and sips the nearest droplet (fermented fruit / ethanol).
        Connectome component: PAM dopaminergic reward excitation (316 neurons) + sweet taste pathway.
        Game rule component: escalating motor degradation (inebriation: uncoordinated tremors, wobbly drift, stumbling gait, delayed jump reflexes)."""
        for a in self.alcohols:
            if a["p"][1] < FLOOR - 8:
                a["v"] += 0.9
                a["p"][1] = min(FLOOR - 8, a["p"][1] + a["v"])
        if not self.alcohols:
            return
        for slot in self.flies:
            fly = slot.fly
            if fly.dead or fly.wrapped or fly.frozen_at is not None or fly.grabbed is not None:
                continue
            a = min(self.alcohols, key=lambda a: abs(a["p"][0] - fly.p[THX, 0]))
            dx = a["p"][0] - fly.p[HEAD, 0]
            on_floor = fly.p[THX, 1] > FLOOR - STAND - 30 and now >= fly.escape_until
            if abs(dx) > 30:
                if on_floor and now >= fly.stun_until:
                    fly.facing = 1 if dx > 0 else -1
                    fly.walk_until, fly.run = now + 0.2, False
                continue
            if not on_floor or a["p"][1] < FLOOR - 10:
                continue
            if now >= fly.eating_until:
                self.popup(fly.p[HEAD] + (0, -60), random.choice(("SIP...", "GLUG!", "*HIC*")), (255, 140, 210))
                self.sound.play("yum")
                self.note("ALCOHOL  drinking: sweet + PAM reward [GAME RULE: inebriation]", source="rule")
            fly.eating_until = now + 0.4
            a["left"] -= 1 / 240
            slot.brain.poke("taste", None, 0.5)
            slot.brain.poke("sweet", None, 0.6, recruit=0.6)
            slot.brain.poke("reward", None, 0.6)
            fly.inebriation = min(1.0, getattr(fly, "inebriation", 0.0) + 0.008)
            if a["left"] <= 0:
                drop_item(self.alcohols, a)
                break

    def _decoy(self, now: float) -> None:
        """Decoy female (GAME RULE: the body, the contact distance and the COURTSHIP tag, which contact triggers; no
        neuron is read for it). CONNECTOME: contact drives the foreleg GRNs annotated putative ppk23/ppk25 (LgLG5-8)."""
        if not hasattr(self, "decoys") or not self.decoys:
            return
        for dec in self.decoys:
            if getattr(self, "grabbed_decoy", None) is dec:
                continue
            if not isinstance(dec.get("v"), np.ndarray):
                dec["v"] = np.array(dec.get("v", [0.0, 0.0]), float)
            dec["v"][1] += 0.5  # gravity
            dec["p"] += dec["v"]
            if dec["p"][1] >= FLOOR - 8:
                dec["p"][1] = FLOOR - 8
                dec["v"][:] = 0.0
        for slot in self.flies:
            fly = slot.fly
            if fly.dead or fly.wrapped or fly.frozen_at is not None or fly.grabbed is not None:
                continue
            dec = min(self.decoys, key=lambda d: abs(d["p"][0] - fly.p[THX, 0]))
            dist = float(np.hypot(*(dec["p"] - fly.p[HEAD])))
            if dist < 45:
                # Forelegs contact the decoy: drives real putative ppk23/ppk25 foreleg GRNs
                slot.brain.poke("pheromone", "foreleg", 0.6)
                fly.decoy_contact_until = now + 0.35
                if now >= getattr(fly, "court_until", 0.0):
                    fly.court_until = now + 0.5
                    self.note("COURTSHIP   foreleg contact (rule): drives the putative ppk23/ppk25 GRNs")
                    self.popup(fly.p[HEAD] + (0, -50), "COURTSHIP", (255, 180, 220))

    def _torch(self, mouse, now: float) -> None:
        """Flame jet from the cursor, aimed at the nearest fly but scorching every fly it passes over. Heat reaches
        the whole body: every touch and heat neuron, full strength."""
        nozzle = np.array(mouse, float)
        aim_slot, _ = self._nearest_fly(mouse, 1e9)
        to_fly = (aim_slot.fly.p[THX] - nozzle) if aim_slot is not None else np.array([1.0, 0.0])
        dist = float(np.hypot(*to_fly))
        aim = to_fly / dist if dist > 1 else np.array([1.0, 0.0])
        self.torch_aim = aim
        for _ in range(7):
            a = math.atan2(aim[1], aim[0]) + random.uniform(-0.22, 0.22)
            sp = random.uniform(8, 13)
            self.flames.append([nozzle[0], nozzle[1], math.cos(a) * sp, math.sin(a) * sp, now, random.uniform(0.22, 0.4)])
        for slot in self.flies:
            fly = slot.fly
            rel = fly.p - nozzle
            d = np.hypot(*rel.T)
            cosang = (rel @ aim) / np.maximum(d, 1e-6)
            inside = np.flatnonzero((d < 200) & (cosang > 0.85))
            if len(inside):
                fly.burn_until = now + 0.9                  # it catches fire, so dodging the jet doesn't cool it
                for i in inside:
                    fly.impulse(i, aim * 0.5 + (0, -0.2))
            if now >= fly.burn_until:
                continue
            fly.char = min(1.0, fly.char + 0.004)
            fly.hurt = max(fly.hurt, 0.5)
            fly.last_hit_x = nozzle[0]
            if fly.dead:
                continue
            for region, side in TORCH_KEYS:
                slot.brain.poke(region, side, 1.0)
            self.damage(slot, 0.35, "the blowtorch")
            if int(now * 2) != int((now - 1 / 60) * 2):
                slot.hits += 1
            if random.random() < 0.02:
                self.popup(fly.p[HEAD] + (0, -60), random.choice(("SIZZLE!", "TSSSS!", "HOT HOT!")), (255, 150, 60))

    def _laser_step(self, mouse, now: float) -> None:
        if not hasattr(self, "laser_state"):
            return
        hit_any = False
        m_pos = np.array(mouse, float)
        for slot in self.flies:
            fly = slot.fly
            if fly.dead:
                self.laser_state.apply(slot.brain, now, is_hitting=False)
                continue
            d = np.hypot(*(fly.p - m_pos).T) - RADIUS
            if np.any(d < 45):
                hit_any = True
                cur = self.laser_state.apply(slot.brain, now, is_hitting=True)
                if random.random() < 0.04:
                    act_txt = "STIM" if self.laser_state.mode == "activate" else "SILENCE"
                    self.popup(fly.p[HEAD] + (0, -40), f"LASER {act_txt} {self.laser_state.target_type}",
                               (255, 180, 80) if self.laser_state.mode == "activate" else (80, 200, 255))
            else:
                self.laser_state.apply(slot.brain, now, is_hitting=False)
        self.laser_state.hit_fly = hit_any

    def _draw_laser(self, surf, now: float, mouse) -> None:
        if not hasattr(self, "laser_state") or not self.laser_state.is_active(now):
            return
        ls = self.laser_state
        emitter = (PLAY_W - 30, FLOOR + 20)
        col = (255, 140, 50) if ls.mode == "activate" else (60, 190, 255)
        core_col = (255, 255, 220) if ls.mode == "activate" else (220, 250, 255)
        thick_line(surf, emitter, mouse, 5, col)
        thick_line(surf, emitter, mouse, 2, core_col)
        spot_r = 9 + int(3 * math.sin(now * 24))
        aacircle(surf, mouse, spot_r, col)
        aacircle(surf, mouse, max(2, spot_r - 4), core_col)
        badge = f"LASER: {ls.target_type} ({'STIM' if ls.mode == 'activate' else 'SILENCE'})"
        self._text(surf, badge, (mouse[0] + 14, mouse[1] - 16), col, self.f_small)

    def _swat_impact(self, pos, now: float) -> None:
        self.shake_until = now + 0.18
        self.sound.play("whack")
        for slot in self._flies_within(pos, 95):
            fly = slot.fly
            d = np.hypot(*(fly.p - pos).T) - RADIUS
            near = np.flatnonzero(d < 95)
            fly.last_hit_x = pos[0]
            for i in near:
                fly.impulse(i, np.array([(fly.p[i, 0] - pos[0]) * 0.15, 32.0]))
                self.hit(slot, i, 1.0)
            fly.stun(now, 1.8)
            self.damage(slot, 14, "the swatter")
        self.popup((pos[0], pos[1] - 90), "SWAT!", (255, 230, 120))
        self.puff((pos[0], min(pos[1] + 40, FLOOR)), 10, 4)

    def _explode(self, b: dict, now: float) -> None:
        pos = b["p"]
        self.flashes.append([pos.copy(), now])
        self.shake_until = now + 0.4
        self.sound.play("boom")
        self.puff(pos, 24, 7)
        for slot in self._flies_within(pos, 330):
            fly = slot.fly
            d = np.hypot(*(fly.p - pos).T)
            worst = 0.0
            for i in np.flatnonzero(d < 330):
                f = 48 * (1 - d[i] / 330) ** 1.3
                dirv = (fly.p[i] - pos) / (d[i] or 1) + (0, -0.8)
                fly.impulse(i, dirv * f)
                if f > 4:
                    self.hit(slot, i, f / 40)
                    worst = max(worst, f)
            if fly.frozen_at is not None and fly.shattered_at is None and worst:
                self._shatter(slot, now)
            elif worst:
                fly.last_hit_x = pos[0]
                fly.stun(now, 2.4)
                self.damage(slot, 32 * worst / 48, "a bomb")
        self.popup((pos[0], pos[1] - 70), "KABOOM!", (255, 160, 60), force=True)

    # --- per frame ---
    def update(self, now: float, mouse) -> None:
        for slot in self.flies:
            slot.fly.p_tick = slot.fly.p.copy()
        if self.challenge is not None:
            self.challenge.update(now)
        self.frame += 1
        self.mouse = mouse
        self._poll_spawn()
        self._lab_tick()
        self._environment(now, mouse)
        for slot in list(self.flies):
            fly = slot.fly
            pin = np.array(mouse, float)
            if fly.wrapped and self.spider is not None and self.spider.get("target") is slot:
                pin = self.spider["p"] + (0, 26)
            held = self.preds.pin_for(slot)
            if held is not None:
                pin = held
            for i, sp in fly.step(now, pin):
                s = float(np.clip((sp - 9) / 35, 0.05, 1))
                self.hit(slot, i, s)
                if sp > 18:
                    self.damage(slot, min(8.0, (sp - 18) * 0.35), "the wall" if fly.p[i, 1] < FLOOR - 30 else "the floor")
                if sp > 14 and fly.p[i, 1] > FLOOR - 30:
                    self.puff((fly.p[i, 0], FLOOR - 2), 3)
                if sp > 24 and not fly.dead:
                    self.sound.play("bonk", 0.4 + 0.6 * s)
                    fly.stun(now, 0.8 * s + 0.3)
                    if random.random() < 0.5:
                        self.popup(fly.p[i] + (0, -40), random.choice(OUCH))
        for sw in self.swats:
            if not sw[2] and now - sw[1] > 0.1:
                sw[2] = True
                self._swat_impact(sw[0], now)
        self.swats = [s for s in self.swats if now - s[1] < 0.55]
        for b in list(self.bombs):
            b["v"][1] += 0.9
            b["p"] += b["v"]
            if b["p"][1] > FLOOR - 16:
                b["p"][1], b["v"][:] = FLOOR - 16, 0
            if now - b["t"] > 1.5:
                drop_item(self.bombs, b)
                self._explode(b, now)
        for d in self.dust:
            d[0] += d[2]
            d[1] += d[3]
            d[3] -= 0.05
        self.dust = [d for d in self.dust if now - d[4] < d[5]]
        if self.torching and TOOLS[self.tool][0] == "torch" and self.report is None:
            self._torch(mouse, now)
        if self.torching and TOOLS[self.tool][0] in ("cleaner", "freeze") and self.report is None:
            self._spray(mouse, now, TOOLS[self.tool][0])
        if (self.torching or getattr(self, "laser_state", None) and self.laser_state.is_active(now)) and TOOLS[self.tool][0] == "laser" and self.report is None:
            self._laser_step(mouse, now)
        elif hasattr(self, "laser_state"):
            for slot in self.flies:
                if len(getattr(slot.brain, "_laser_rows", [])):
                    self.laser_state.apply(slot.brain, now, is_hitting=False)
        if getattr(self, "grabbed_decoy", None) is not None:
            self.grabbed_decoy["p"] = np.array(mouse, float)
            self.grabbed_decoy["v"] = np.zeros(2)
        self._effects(now)
        for sh in self.shards:
            sh[0] += sh[2]
            sh[1] = min(FLOOR - 3, sh[1] + sh[3])
            sh[3] += 0.8
            sh[2] *= 0.98 if sh[1] < FLOOR - 3 else 0.8
            sh[4] += sh[2] * 0.05
        self.bolts = [b for b in self.bolts if now - b[2] < 0.25]
        for fl in self.flames:
            fl[0] += fl[2]
            fl[1] += fl[3]
            fl[2] *= 0.96
            fl[3] = fl[3] * 0.96 - 0.25
        self.flames = [fl for fl in self.flames if now - fl[4] < fl[5]]
        for m in self.mist:
            m[0] += m[2]
            m[1] += m[3]
            m[2] *= 0.94
            m[3] = m[3] * 0.94 + 0.05
        self.mist = [m for m in self.mist if now - m[4] < m[5]]

        self._fly_collisions(now)

        all_dead = all(s.fly.dead for s in self.flies)
        for slot in list(self.flies):
            fly, br = slot.fly, slot.brain
            parts = br.pain_parts(br.fast, br.base)[0]
            parts[3] = max(parts[3], slot.pain_parts[3] * 0.96)
            slot.pain_parts += (parts - slot.pain_parts) * 0.15
            slot.pain += (float(br.pain_index(parts)) - slot.pain) * 0.15
            slot.reward += (float(np.clip((br.level("reward") - 1.0) / 0.6, 0, 1)) * 100 - slot.reward) * 0.1
            if not fly.dead and slot.pain > 25 and self.frame % 3 == 0:
                br.poke("punish", None, slot.pain / 100)    # pain drives the punishment dopamine neurons (game rule)
            if self.pet and slot is self.flies[0]:
                self.pet.update_live(dt=0.016, pam_rate=br.level("reward"), ppl1_rate=br.level("punish") if "punish" in br.col else 0.0)
                sd = self.pet.get_dfb_sleep_drive()
                if sd > 0.05 and len(br.sense.get(("sleep", None), ())):
                    br.poke("sleep", None, sd)
                if self.pet.real_stakes and self.pet.is_dead and not fly.dead:
                    slot.pending_damage = 999.0
            self._vision(slot, now, mouse)
            self._scents(slot, now, mouse)
            self._learn(slot, now)
            if not fly.dead:
                slot.pain_peak = max(slot.pain_peak, slot.pain)
                if slot.pain >= 99:
                    slot.pain_max_s += 1 / 60
            slot.pain_trace.append(slot.pain)
            slot.pain_trace = slot.pain_trace[-720:]

            if not fly.dead:
                for (region, side), s in slot.pending_hits.items():
                    br.poke(region, side, s)
                    slot.hits += 1
                if slot.pending_damage:
                    floor = 1.0 if self.immortal else 0.0          # immortal: it can be hurt, never killed
                    fly.health = max(floor, fly.health - slot.pending_damage)
                    slot.last_damage = now
                    if fly.health <= 0:
                        self._die(slot, now)
                elif self.immortal and now - slot.last_damage > 1.5:
                    fly.health = min(MAX_HEALTH, fly.health + 0.1)   # heals ~6 health/s once you stop
            slot.pending_hits.clear()
            slot.pending_damage = 0.0
            if fly.dead:
                if (all_dead and self.report is None and slot is self.flies[self.focus]
                        and now - fly.dead_at > AUTOPSY_DELAY):
                    self.report = self._autopsy(slot, now)
                    self.death_frames = list(self.frames)
                continue

            # reactions read from the descending neurons
            can_fly = fly.grabbed is None and not fly.wrapped and fly.frost < 0.5 and fly.melt < 0.3 and fly.venom < 0.5
            free = fly.grabbed is None and now >= fly.escape_until
            can_fly = can_fly and fly.wet <= 0 and len(fly.stuck) < 2
            lv = {n: br.level(n) for n in ("jump", "run", "kick", "walk", "back", "turn_l", "turn_r", "fly", "escape")}
            fly.power = lv["fly"]
            away = 1 if fly.p[THX, 0] >= fly.last_hit_x else -1
            if lv["escape"] > THRESH["escape"] and now >= fly.escape_ready and fly.grabbed is None and can_fly:
                fly.stun_until = 0.0                             # giant fiber escape: it saw something coming
                fly.last_hit_x = slot.threat_x
                fly.escape(now)
                self.note(f"DODGE    giant fiber DNp01 x{lv['escape']:.1f}")
                self.on_reaction("DODGE", slot)
                self.popup(fly.p[HEAD] + (0, -60), "DODGE!", (170, 255, 200))
                self.sound.play("dodge")
            elif lv["jump"] > THRESH["jump"] and now >= fly.escape_ready and free and can_fly:
                fly.stun_until = 0.0                             # the reflex beats the dizziness
                fly.escape(now)
                self.note(f"FLY AWAY head-touch DNs x{lv['jump']:.1f}")
                self.popup(fly.p[HEAD] + (0, -60), "YIKES!", (160, 230, 255))
            elif (lv["fly"] > THRESH["fly"] and now >= fly.escape_ready and free and can_fly and now >= fly.stun_until
                  and now >= getattr(slot, "asleep_until", 0.0)
                  and now >= fly.eating_until):
                fly.escape(now, seconds=random.uniform(2.5, 4.0), wander=True)
                self.note(f"TAKE OFF DNg02 x{lv['fly']:.2f}")
            if lv["run"] > THRESH["run"] and now >= fly.walk_until and free:
                fly.stun_until = min(fly.stun_until, now + 0.2)
                fly.facing, fly.walk_until, fly.back_until, fly.run = away, now + 1.1, 0.0, True
                self.note(f"RUN      body-touch DNs x{lv['run']:.1f}")
            if lv["kick"] > THRESH["kick"] and now >= fly.flail_until:
                fly.flail_until = now + 0.6
                self.note(f"KICK     leg-touch DNs x{lv['kick']:.1f}")
            if lv["back"] > THRESH["back"] and now >= fly.back_until and now >= fly.walk_until:
                fly.back_until = now + 0.8
                self.note(f"BACK UP  MDN x{lv['back']:.1f}")
            elif (lv["walk"] > THRESH["walk"] and now >= fly.walk_until and now >= fly.back_until
                  and now >= getattr(slot, "asleep_until", 0.0)):
                fly.walk_until, fly.run = now + 1.2, False
                self.note(f"WALK     DNp09 x{lv['walk']:.1f}")
            self._memory_behavior(slot, now, free, can_fly)
            self.readouts(slot, now)
            lamp_idle = free and now >= fly.escape_until and now >= fly.stun_until and now >= fly.walk_until
            if ARENAS[self.arena_i] == "lamp" and lamp_idle and now >= slot.photo_ready:
                slot.photo_ready = now + random.uniform(3.0, 6.0)   # drawn to the light (game rule)
                fly.facing = 1 if LAMP[0] > fly.p[THX, 0] else -1
                if can_fly and random.random() < 0.6:
                    fly.escape(now, seconds=random.uniform(3.0, 5.0), wander=True)
                    self.note("TO LIGHT flies to the lamp")
                else:
                    fly.walk_until, fly.run = now + 1.5, False
            turn = lv["turn_r"] - lv["turn_l"]
            if abs(turn) > THRESH["turn"] and now >= fly.turn_ready and free and now >= fly.walk_until:
                new = 1 if turn > 0 else -1
                fly.turn_ready = now + 1.5
                if new != fly.facing:
                    fly.facing = new
                    self.note(f"TURN {'R' if new > 0 else 'L'}   DNa01/02 R-L {turn:+.1f}")

            self._aggression(slot, now, free)
            self._sleep(slot, now, free)

        self._update_focus()
        self._training_tick(now)
        self._sound_update(now)
        self.x3.tick(now)

    def _fly_collisions(self, now: float) -> None:
        """Two flies that bump: a soft push-apart always, and if the impact is hard enough, a real touch-neuron poke
        on both brains (the same body/legs regions a swat or a wall bonk fires)."""
        flies = self.flies
        for a in range(len(flies)):
            sa = flies[a]
            if sa.fly.dead or sa.fly.dissolved_at is not None or sa.fly.shattered_at is not None:
                continue
            for b in range(a + 1, len(flies)):
                sb = flies[b]
                if sb.fly.dead or sb.fly.dissolved_at is not None or sb.fly.shattered_at is not None:
                    continue
                fa, fb = sa.fly, sb.fly
                d = fb.p[THX] - fa.p[THX]
                dist = float(np.hypot(*d))
                overlap = FLY_TOUCH_RADIUS * 2 - dist
                if overlap <= 0:
                    continue
                n = d / dist if dist > 1e-6 else np.array([1.0, 0.0])
                closing = float(np.dot((fb.p[THX] - fb.prev[THX]) - (fa.p[THX] - fa.prev[THX]), n))
                for i in range(N_P):
                    fa.p[i] -= n * overlap * 0.5
                    fb.p[i] += n * overlap * 0.5
                if closing < -3.0:                        # a real bump, not just jostling
                    s = float(np.clip(-closing / 20.0, 0.15, 1.0))
                    for slot, fly, side in ((sa, fa, -n), (sb, fb, n)):
                        fly.impulse(THX, side * -4.0)
                        if not fly.dead:
                            slot.brain.poke("body", None, 0.5 * s)
                            slot.brain.poke("legs", None, 0.3 * s)
                            fly.hurt = max(fly.hurt, 0.3)
                    if sa.brain.level("aggression") > THRESH["aggression"] or sb.brain.level("aggression") > THRESH["aggression"]:
                        self.note("FIGHT    aggression collision")

    def _die(self, slot: "FlySlot", now: float) -> None:
        fly = slot.fly
        fly.dead_at = now
        fly.grabbed = None
        self.kills += 1
        self.x3.on_die(slot)                         # 3.0: keep the last seconds for the kill cam (before the drive is cut)
        slot.brain.kill()
        self.sound.play("death")
        self.note("DIED     brain drive cut, activity fading")
        self.popup(fly.p[HEAD] + (0, -70), "K.O.!", (255, 90, 80), force=True)
        self.shake_until = now + 0.3
        if all(s.fly.dead for s in self.flies):
            self.killed_by = slot.damage_src or "being kicked"
            self.focus = self.flies.index(slot)    # auto-focus whoever ended the game, for the autopsy screen

    def _autopsy(self, slot: "FlySlot", now: float) -> dict:
        br = slot.brain
        ds = br.death_sample
        base = br.death_base
        last = br.history(ds - 100, ds)                  # last 2 s alive
        after = br.history(ds + 50, ds + 150)            # 1-3 s after death
        rows = []
        names = ["whole brain"] + [n for n, _ in POPS] + ["head", "body", "legs", "wing", "heat", "cold", "smell", "taste", "reward", "jump", "run", "fly"]
        pretty = {"head": "touch: head", "body": "touch: body", "legs": "touch: legs", "wing": "touch: wings", "heat": "heat sensors",
                  "cold": "cold sensors", "smell": "smell ORNs", "taste": "taste neurons", "reward": "PAM reward DANs",
                  "jump": "DNs: jump group", "run": "DNs: run group", "fly": "DNs: DNg02 flight"}
        for n in names:
            i = br.col[n]
            b = float(base[i])
            f = float(last[:, i].mean()) if len(last) else 0.0
            a = float(after[:, i].mean()) if len(after) else 0.0
            rows.append({"name": pretty.get(n, n), "base": b, "final": f, "after": a,
                         "ratio": (f + 0.05) / (b + 0.05), "section": "pop" if n in dict(POPS) or n == "whole brain" else "detail"})
        tl = br.history(ds - 1000, ds + 150)             # 20 s before death to 3 s after
        start = max(ds - 1000, br.hist_n - HIST, 0)
        touch_idx = [br.col[n] for n in ("head", "body", "legs", "wing")]
        tsz = br.g_size[touch_idx]
        series = []
        for label, vals, b in (
            ("whole brain", tl[:, br.col["whole brain"]], base[br.col["whole brain"]]),
            ("touch neurons", (tl[:, touch_idx] * tsz).sum(1) / tsz.sum(), float((base[touch_idx] * tsz).sum() / tsz.sum())),
            ("descending", tl[:, br.col["descending"]], base[br.col["descending"]]),
        ):
            series.append((label, vals / max(float(b), 1e-3)))
        top = max(rows[1:], key=lambda r: r["ratio"])
        pain = br.pain_index(br.hold_alarm(br.pain_parts(tl, base))) if len(tl) else np.zeros(0)
        last_pain = pain[max(0, ds - start - 100):ds - start].mean() if len(pain) else 0.0
        return {"rows": rows, "series": series, "death_at": ds - start, "n": len(tl), "top": top, "pain": pain,
                "pain_last": float(last_pain), "pain_peak": slot.pain_peak, "pain_max_s": slot.pain_max_s,
                "alive_s": slot.fly.dead_at - slot.born, "hits": slot.hits, "by": self.killed_by}

    # --- render ---
    def draw(self, now: float, mouse) -> None:
        with PROF.section("render"):
            self._draw(now, mouse)

    def _draw(self, now: float, mouse) -> None:
        scr = self.screen
        scr.fill(BG)
        arena = self.bg.copy()
        self._draw_arena_back(arena, now)

        for b in self.bombs:
            bx, by = b["p"]
            aacircle(arena, (bx, by), 16, (34, 34, 40))
            aacircle(arena, (bx - 5, by - 5), 4, (100, 100, 112))
            thick_line(arena, (bx + 8, by - 12), (bx + 14, by - 22), 3, (180, 150, 90))
            if int(now * 12) % 2:
                aacircle(arena, (bx + 15, by - 24), 5, (255, 220, 80))
        for k, slot in enumerate(self.flies):
            fly = slot.fly
            lift = float(np.clip((FLOOR - fly.p[:, 1].max()) / 300, 0, 1))
            sh = pygame.transform.smoothscale(self.shadow, (int(200 * (1 - 0.5 * lift)), int(26 * (1 - 0.5 * lift))))
            sh.set_alpha(int(255 * (1 - 0.7 * lift)))
            arena.blit(sh, sh.get_rect(center=(int(fly.p[THX, 0]), FLOOR + 2)))
            prev = getattr(fly, "p_tick", None)
            if self.clock.alpha < 1.0 and prev is not None:        # slow motion: draw between the last two ticks
                real_p = fly.p
                fly.p = prev + (real_p - prev) * self.clock.alpha
                draw_fly(arena, fly, now)
                fly.p = real_p
            else:
                draw_fly(arena, fly, now)
            if now < fly.burn_until:                         # small flames licking off the body
                for i in (HEAD, THX, ABD):
                    fx, fy = fly.p[i] + (random.uniform(-14, 14), random.uniform(-18, -4))
                    aacircle(arena, (fx, fy), random.uniform(4, 9), (255, random.randint(110, 200), 40, 170))
            if len(self.flies) > 1:                           # tell several flies apart: a numbered, focus-lit badge
                bx, by = fly.p[HEAD] + (0, -34)
                lit = k == self.focus
                aacircle(arena, (bx, by), 9, ACCENT if lit else (60, 66, 78))
                self._text(arena, str(k + 1), (bx, by), BG if lit else TEXT, self.f_small, "center")
        for sw in self.swats:
            draw_swatter(arena, sw[0], (now - sw[1]) / 0.55)
        for d in self.dust:
            e = (now - d[4]) / d[5]
            aacircle(arena, (d[0], d[1]), d[6] * (0.6 + e), (170, 160, 150, int(120 * (1 - e))))
        for fl in self.flames:
            e = (now - fl[4]) / fl[5]
            col = (255, 240, 170) if e < 0.25 else (255, 160, 40) if e < 0.55 else (220, 60, 30) if e < 0.8 else (90, 80, 80)
            aacircle(arena, (fl[0], fl[1]), 4 + 16 * e, (*col, int(210 * (1 - e ** 2))))
        for m in self.mist:
            e = (now - m[4]) / m[5]
            col = (200, 230, 250) if m[6] == "cleaner" else (225, 245, 255)
            aacircle(arena, (m[0], m[1]), 5 + 22 * e, (*col, int((90 if m[6] == "cleaner" else 130) * (1 - e))))
        if self.torching and TOOLS[self.tool][0] in ("cleaner", "freeze") and self.report is None and mouse[0] < PLAY_W:
            aim = getattr(self, "torch_aim", np.array([1.0, 0.0]))
            m = np.array(mouse, float)
            body = (200, 40, 40) if TOOLS[self.tool][0] == "cleaner" else (60, 130, 210)
            thick_line(arena, m - aim * 78, m - aim * 12, 26, body)                     # the can
            thick_line(arena, m - aim * 60, m - aim * 40, 27, (235, 235, 240))         # label band
            thick_line(arena, m - aim * 12, m - aim * 2, 8, (60, 60, 66))              # nozzle
        for sh in self.shards:                           # ice and fly fragments
            c, s_ = math.cos(sh[4]), math.sin(sh[4])
            r = sh[5]
            pts = [(sh[0] + r * c, sh[1] + r * s_), (sh[0] - r * 0.6 * s_, sh[1] + r * 0.6 * c), (sh[0] - r * c * 0.8, sh[1] - r * s_ * 0.5)]
            aapoly(arena, pts, (150, 100, 50) if sh[6] else (200, 235, 255, 200))
        for b in self.bolts:                             # lightning
            a, z = b[0], b[1]
            pts = [a]
            for k in range(1, 7):
                q = a + (z - a) * k / 7
                pts.append(q + np.random.normal(0, 12, 2))
            pts.append(z)
            for w_, col in ((7, (120, 170, 255, 120)), (3, (235, 245, 255))):
                pygame.draw.lines(arena, col[:3], False, [tuple(p) for p in pts], w_)
        if self.spider is not None:
            self._draw_spider(arena, now)
        if self.preds.list:
            from kickthefly.game import predator_play
            predator_play.draw2d(self.preds, arena, now)
        for s in self.sugars:
            k_ = max(0.35, s["left"])
            x, y = s["p"]
            e = 11 * k_
            if s.get("fruit"):
                aacircle(arena, (x, y), e * 0.8, (214, 64, 48))
                aacircle(arena, (x - e * 0.25, y - e * 0.25), e * 0.25, (245, 150, 130))
                continue
            aapoly(arena, [(x - e, y - e * 0.4), (x, y - e), (x + e, y - e * 0.4), (x, y + e * 0.2)], (250, 250, 255))
            aapoly(arena, [(x - e, y - e * 0.4), (x, y + e * 0.2), (x, y + e), (x - e, y + e * 0.5)], (215, 215, 228))
            aapoly(arena, [(x, y + e * 0.2), (x + e, y - e * 0.4), (x + e, y + e * 0.5), (x, y + e)], (185, 185, 205))
        for a in self.alcohols:
            k_ = max(0.35, a["left"])
            x, y = a["p"]
            r = int(9 * k_)
            aapoly(arena, [(x - r, y), (x, y - int(r * 0.8)), (x + r, y), (x, y + int(r * 0.8))], (215, 75, 125))
            aacircle(arena, (int(x), int(y)), r, (235, 90, 140))
        for d in getattr(self, "decoys", []):
            x, y = d["p"]
            self._draw_decoy_female_2d(arena, float(x), float(y), now)
        active_puffs = []
        for pf in getattr(self, "cva_puffs", []):
            dt = now - pf["t"]
            if dt < pf.get("duration", 0.65):
                active_puffs.append(pf)
                progress = dt / pf.get("duration", 0.65)
                reach_r = int(250.0 * (progress ** 0.5))
                alpha = int(140.0 * (1.0 - progress))
                px, py = int(pf["p"][0]), int(pf["p"][1])
                if reach_r > 5:
                    aacircle(arena, (px, py), reach_r, (255, 180, 100, alpha))
                    aacircle(arena, (px, py), int(reach_r * 0.7), (255, 160, 80, int(alpha * 0.7)))
                    aacircle(arena, (px, py), int(reach_r * 0.4), (255, 200, 120, int(alpha * 0.5)))
        self.cva_puffs = active_puffs
        self._draw_arena_front(arena, now)
        self._clean_frame = arena.copy()
        if self.torching and TOOLS[self.tool][0] == "torch" and self.report is None and mouse[0] < PLAY_W:
            aim = getattr(self, "torch_aim", np.array([1.0, 0.0]))
            m = np.array(mouse, float)
            thick_line(arena, m - aim * 70, m - aim * 8, 16, (110, 116, 128))
            thick_line(arena, m - aim * 70, m - aim * 50, 18, (200, 60, 50))
            aacircle(arena, m, 5, (255, 250, 220))
        if (self.torching or getattr(self, "laser_state", None) and self.laser_state.is_active(now)) and TOOLS[self.tool][0] == "laser" and self.report is None and mouse[0] < PLAY_W:
            self._draw_laser(arena, now, mouse)
        for fl in list(self.flashes):
            e = (now - fl[1]) / 0.35
            if e >= 1 or self.calm_fx:
                drop_item(self.flashes, fl)
                continue
            aacircle(arena, fl[0], 40 + 260 * e, (255, 200, 90, int(170 * (1 - e))))
        for pu in list(self.popups):
            e = (now - pu[3]) / 0.9
            if e >= 1:
                drop_item(self.popups, pu)
                continue
            txt = self.f_big.render(pu[2], True, pu[4])
            shd = self.f_big.render(pu[2], True, (120, 16, 22))
            a = int(255 * (1 - e ** 3))
            txt.set_alpha(a)
            shd.set_alpha(a)
            x, y = pu[0] - txt.get_width() / 2, pu[1] - 50 * e
            arena.blit(shd, (x + 3, y + 3))
            arena.blit(txt, (x, y))
            if pu[5] and self.cfg.tags_on():
                draw_source_chip(arena, (pu[0], y + txt.get_height()), pu[5], self.f_small, alpha=a)
        _ui = PROF.section("ui")                       # the HUD, the brain panel and the menus (3.1.0 profiler)
        _ui.__enter__()
        if getattr(self, "photo_mode", False):
            b_txt = self.f_small.render("PHOTO MODE  |  S / F12: Clean Snap  |  F10: Exit", True, (240, 240, 240))
            box = b_txt.get_rect(midbottom=(PLAY_W // 2, FLOOR + 30)).inflate(24, 8)
            pygame.draw.rect(arena, (12, 16, 24, 200), box, border_radius=6)
            pygame.draw.rect(arena, (70, 80, 100, 180), box, 1, border_radius=6)
            arena.blit(b_txt, b_txt.get_rect(center=box.center))
            if self.cfg.tags_on():
                draw_source_chip(arena, (box.right + 6, box.y + 2), "rule", self.f_small)
        else:
            self._draw_toolbar(arena)
            self._draw_hud(arena, now)
            if not (self.cfg["brain.autopilot"] and self.cfg["brain.autopilot_hide_hud"]) and not self._overlay_open() and TOOLS[self.tool][0] != "hand" and mouse[0] < PLAY_W and mouse[1] < FLOOR:
                gfxdraw.aacircle(arena, mouse[0], mouse[1], 10, (255, 255, 255))
                pygame.draw.line(arena, (255, 255, 255), (mouse[0] - 14, mouse[1]), (mouse[0] + 14, mouse[1]))
                pygame.draw.line(arena, (255, 255, 255), (mouse[0], mouse[1] - 14), (mouse[0], mouse[1] + 14))
        if self.report is not None:
            self._draw_autopsy(arena, now)
        elif self.big_view:
            self._draw_big_view(arena)
        if self.training_open:
            self._draw_training(arena)
        if self.surgery_open:
            self._draw_surgery(arena)
        if self.help_open:
            self._draw_help(arena)
        if self.challenge is not None:
            self.challenge.draw(arena, now, pygame.mouse.get_pos())
        self.draw_science_card(arena, now)

        shake = (0, 0)
        if now < self.shake_until:
            shake = (random.randint(-7, 7), random.randint(-5, 5))
        if self.calm_fx:
            shake = (0, 0)
        scr.blit(arena, shake)
        self._draw_brain(now)
        self.draw_time_indicator(scr, PLAY_W // 2, 92)
        self.draw_recording(scr, PLAY_W // 2, 130)
        if self.kwheel_open and not self.menu.open:
            from kickthefly.ui import loadout_ui
            loadout_ui.draw_wheel(self, scr)
        if self.tutorial is not None and self.tutorial.active and not self.menu.open and not self.x3.kc_playing():
            self.tutorial.draw(scr)                   # review: it drew over the kill cam
        if self.menu.open:
            self.menu.draw(scr, pygame.mouse.get_pos(), now)
        _ui.__exit__()

    def draw_time_indicator(self, surf, cx: int, y: int) -> None:
        label = self.clock.label()
        if not label:
            return
        img = self.f_bold.render(label, True, (20, 16, 8))
        box = img.get_rect(midtop=(cx, y)).inflate(22, 10)
        pygame.draw.rect(surf, AMBER if self.clock.user_paused else ACCENT, box, border_radius=8)
        surf.blit(img, img.get_rect(center=box.center))

    def _draw_spider(self, surf, now: float) -> None:
        sp = self.spider
        x, y = sp["p"]
        pygame.draw.aaline(surf, (220, 220, 225), (x, CEIL), (x, y - 8))
        from kickthefly.game import predator_play
        predator_play.draw_spider2d(sp, surf, now)

    def _draw_decoy_female_2d(self, surf: pygame.Surface, x: float, y: float, now: float) -> None:
        """Draw 2D decoy as a female fly: larger, rounder abdomen with the female band pattern across all
        tergites without the dark male tip cap, no sex combs, clearly a fly, tinted slightly so it reads as a decoy."""
        def tc(col):
            k = 0.22
            tr, tg, tb = 200, 180, 220
            r = int(col[0] * (1 - k) + tr * k)
            g = int(col[1] * (1 - k) + tg * k)
            b = int(col[2] * (1 - k) + tb * k)
            if len(col) == 4:
                return (r, g, b, col[3])
            return (r, g, b)

        thx = np.array([x, y - 18], float)
        # 1. Shadow on floor
        gfxdraw.filled_ellipse(surf, int(x), int(y + 2), 24, 5, (20, 20, 30, 85))
        # 2. Far legs (3 legs, no sex combs)
        far_col = tc((58, 38, 18))
        for hx, hy, kx, ky, fx, fy in [
            (thx[0] + 10, thx[1] - 4, thx[0] + 18, thx[1] + 6, thx[0] + 26, y),
            (thx[0] - 2, thx[1] - 4, thx[0] - 4, thx[1] + 8, thx[0] + 4, y),
            (thx[0] - 12, thx[1] - 4, thx[0] - 22, thx[1] + 6, thx[0] - 28, y),
        ]:
            thick_line(surf, (hx, hy), (kx, ky), 3.5, far_col)
            thick_line(surf, (kx, ky), (fx, fy), 2.5, far_col)

        # 3. Wings resting along the back
        w_base = (thx[0] - 4, thx[1] - 10)
        w_tip = (thx[0] - 38, thx[1] - 4)
        w_mid = ((w_base[0] + w_tip[0]) / 2, (w_base[1] + w_tip[1]) / 2)
        w_span = math.hypot(w_tip[0] - w_base[0], w_tip[1] - w_base[1])
        w_pts = ellipse_pts(w_mid, w_span / 2 + 5, 10, math.atan2(w_tip[1] - w_base[1], w_tip[0] - w_base[0]))
        w_ip = [(int(px), int(py)) for px, py in w_pts]
        gfxdraw.filled_polygon(surf, w_ip, tc((205, 218, 238, 65)))
        gfxdraw.aapolygon(surf, w_ip, tc((230, 238, 250, 150)))
        gfxdraw.line(surf, int(w_base[0]), int(w_base[1]), int(w_tip[0]), int(w_tip[1]), tc((170, 180, 200, 100)))

        # 4. Abdomen: larger, rounder female abdomen (length 32, radius 21), full female banding, no dark male cap
        ac = thx + np.array([-24.0, 1.0])
        aang = 0.12
        along = np.array([math.cos(aang), math.sin(aang)])
        aapoly(surf, ellipse_pts(ac, 32, 21, aang), tc((180, 122, 48)))
        # Female banded pattern across all tergites
        for s in (-18, -10, -2, 6, 14):
            aapoly(surf, ellipse_pts(ac + along * s, 3.5, 19, aang, 14), tc((82, 52, 24)))
        # Sheen
        aapoly(surf, ellipse_pts(ac + np.array([4.0, -7.0]), 16, 6, aang, 16), tc((218, 164, 88)))
        # Pointed lighter ovipositor tip (female specific, replacing the dark male cap)
        aapoly(surf, ellipse_pts(ac - along * 24, 7, 7, aang), tc((215, 188, 138)))

        # 5. Thorax
        aapoly(surf, ellipse_pts(thx, 22, 18, 0.0), tc((160, 106, 44)))
        aapoly(surf, ellipse_pts(thx + np.array([0.0, -7.0]), 12, 6, 0.0, 16), tc((200, 145, 75)))
        # Bristles on thorax
        b_col = tc((60, 40, 18))
        for bx in (-10, -4, 2, 8):
            gfxdraw.line(surf, int(thx[0] + bx), int(thx[1] - 16), int(thx[0] + bx - 3), int(thx[1] - 22), b_col)

        # 6. Head
        hc = thx + np.array([24.0, -3.0])
        aacircle(surf, hc, 14, tc((150, 100, 42)))
        # Large compound eye with highlight
        eye = hc + np.array([2.0, -2.0])
        aapoly(surf, ellipse_pts(eye, 10, 11, 0.0), tc((196, 30, 26)))
        for fx, fy in ((-3, 2), (0, 4), (3, 1), (-1, -1), (2, -2)):
            gfxdraw.pixel(surf, int(eye[0] + fx), int(eye[1] + fy), tc((120, 16, 14)))
        aacircle(surf, eye + (-2, -3), 3, (255, 185, 175))
        # Antennae & proboscis
        thick_line(surf, hc + (10, -6), hc + (18, -14), 2, tc((70, 45, 20)))
        thick_line(surf, hc + (12, -4), hc + (21, -9), 2, tc((70, 45, 20)))
        thick_line(surf, hc + (6, 8), hc + (12, 16), 3, tc((70, 45, 20)))

        # 7. Near legs (3 legs, no sex combs on forelegs)
        near_col = tc((86, 58, 26))
        for hx, hy, kx, ky, fx, fy in [
            (thx[0] + 12, thx[1] - 2, thx[0] + 24, thx[1] + 8, thx[0] + 34, y),
            (thx[0], thx[1] - 2, thx[0] + 2, thx[1] + 10, thx[0] + 10, y),
            (thx[0] - 10, thx[1] - 2, thx[0] - 18, thx[1] + 8, thx[0] - 22, y),
        ]:
            thick_line(surf, (hx, hy), (kx, ky), 4.5, near_col)
            thick_line(surf, (kx, ky), (fx, fy), 3.0, near_col)

    def _fly_state(self, now: float) -> str:
        f = self.fly
        if f.shattered_at is not None:
            return "shattered"
        if f.dissolved_at is not None:
            return "dissolved"
        if f.frozen_at is not None:
            return "frozen solid"
        if f.dead:
            return "dead"
        if f.stuck:
            return "stuck on flypaper"
        if f.arena == "pool" and (f.p[:, 1] > WATER_Y).any() and now >= f.escape_until:
            return "swimming"
        if f.wrapped:
            return "wrapped in silk"
        if f.grabbed is not None:
            return "grabbed"
        if now < f.escape_until:
            return "flying"
        if now < f.eating_until:
            return "eating sugar"
        if now < f.stun_until:
            return "stunned"
        return f.action

    def _text(self, surf, s, pos, color=TEXT, font=None, anchor="topleft"):
        img = (font or self.f_text).render(s, True, color)
        rect = img.get_rect(**{anchor: pos})
        surf.blit(img, rect)
        return rect

    def _draw_hud(self, surf, now: float) -> None:
        self.live.draw(surf, self.f_bold, PLAY_W)             # 3.0 day 3: the red MIC ON / TWITCH CHAT pills, the vote tally (Larger text applies)
        if self.cfg["brain.autopilot"] and self.cfg["brain.autopilot_hide_hud"]:
            badge = self.f_small.render("AUTOPILOT / SPECTATOR   (Y: exit)", True, (130, 160, 190))
            box = badge.get_rect(midtop=(PLAY_W // 2, 14)).inflate(16, 6)
            pygame.draw.rect(surf, (10, 12, 18, 160), box, border_radius=6)
            surf.blit(badge, badge.get_rect(center=box.center))
            return
        fly = self.fly
        card = pygame.Surface((236, 62), pygame.SRCALPHA)
        pygame.draw.rect(card, (10, 12, 18, 170), card.get_rect(), border_radius=10)
        surf.blit(card, (10, 8))
        self._text(surf, self._fly_state(now).upper(), (22, 12), AMBER, self.f_head)
        max_f = getattr(self, "max_flies", MAX_FLIES)
        flies_txt = f"   flies {len(self.flies)}/{max_f} · #{self.focus + 1}" if len(self.flies) > 1 \
            else f"   flies 1/{max_f} (N)"
        self._text(surf, f"hits {self.hits}   kills {self.kills}{flies_txt}", (22, 42), TEXT, self.f_text)
        # health
        bw, bx, by = 300, PLAY_W // 2 - 150, 16
        frac = fly.health / MAX_HEALTH
        col = (230, 185, 60) if self.immortal else S_GOOD if frac > 0.5 else S_WARN if frac > 0.25 else S_CRIT
        pygame.draw.rect(surf, (10, 12, 18), (bx - 4, by - 4, bw + 8, 26), border_radius=13)
        if frac > 0:
            pygame.draw.rect(surf, col, (bx, by, max(18, int(bw * frac)), 18), border_radius=9)
            pygame.draw.rect(surf, tuple(min(255, c + 60) for c in col), (bx + 6, by + 3, max(6, int(bw * frac) - 12), 4), border_radius=2)
        label = "DEAD" if fly.dead else f"HEALTH {fly.health:.0f}" + ("  IMMORTAL" if self.immortal else "")
        self._text(surf, label, (PLAY_W // 2, by + 9), INK, self.f_bold, "center")
        bits = [f"arena: {ARENAS[self.arena_i]} (E)"]
        status = self.arena_status()
        bits += status
        if self.brain.surgery:
            bits.append("SURGERY ON (O)")
        if self.sound.muted:
            bits.append("muted (M)")
        bits.append("H help" if status else getattr(self, "hint_extra", "F11 fullscreen   H help"))
        hint = self.f_small.render("   ".join(bits), True, AMBER if self.brain.surgery else TEXT)
        box = hint.get_rect(topright=(PLAY_W - 14, 44)).inflate(14, 6)
        pygame.draw.rect(surf, (10, 12, 18, 170), box, border_radius=6)
        surf.blit(hint, hint.get_rect(center=box.center))
        if self.saved_msg and now - self.saved_msg[1] < 4:
            self._text(surf, self.saved_msg[0], (PLAY_W // 2, 66), (170, 230, 190), self.f_small, "midtop")
        if getattr(self, "timelapse_recording", False):
            elapsed = int(now - getattr(self, "timelapse_start_t", now))
            m_s, s_s = divmod(elapsed, 60)
            sp = int(self.cfg.get("graphics.timelapse_speedup", 5))
            fmt = str(self.cfg.get("graphics.timelapse_format", "mp4")).upper()
            pulse = 0.5 + 0.5 * math.sin(now * 8)
            col = (int(255 * (0.6 + 0.4 * pulse)), 40, 40)
            rec_str = f"● REC {m_s:02d}:{s_s:02d} ({sp}x {fmt}) [L: stop]"
            rbox = pygame.Rect(PLAY_W // 2 - 110, 36, 220, 24)
            pygame.draw.rect(surf, (15, 18, 26, 220), rbox, border_radius=6)
            pygame.draw.rect(surf, col, rbox, 1, border_radius=6)
            self._text(surf, rec_str, rbox.center, col, self.f_small, "center")
        if getattr(self, "video_recorder", None) and self.video_recorder.is_recording:
            badge_txt = self.video_recorder.badge_text(now)
            pulse = 0.5 + 0.5 * math.sin(now * 8)
            col = (int(255 * (0.6 + 0.4 * pulse)), 40, 40)
            rec_str = f"● {badge_txt} [Shift+R]"
            vbox = pygame.Rect(PLAY_W // 2 - 100, 62, 200, 24)
            pygame.draw.rect(surf, (15, 18, 26, 220), vbox, border_radius=6)
            pygame.draw.rect(surf, col, vbox, 1, border_radius=6)
            self._text(surf, rec_str, vbox.center, col, self.f_small, "center")
        if now < getattr(fly, "decoy_contact_until", 0.0):
            self._draw_decoy_contact_hud(surf, now)
        self._draw_pain(surf)
        self._draw_reward(surf)
        self._draw_memory(surf)
        if ARENAS[self.arena_i] == "escaperoom":
            self._draw_escaperoom_hud(surf, now)

    def _draw_decoy_contact_hud(self, surf: pygame.Surface, now: float) -> None:
        slot = self.flies[self.focus]
        br = slot.brain
        # LgLG5-8 are all but silent at rest, so a multiple of calm is meaningless (any contact read "x99"): their rate
        # in spikes/s against calm; P1 fires at rest, so its multiple of calm is fine
        p1_x = br.p1_level()
        txt = f"DECOY  LgLG5-8 {br.lglg_fast:.0f}/s (calm {br.lglg_base:.1f})  P1 x{p1_x:.2f}"
        rendered = self.f_small.render(txt, True, (230, 230, 240))
        chip_pad = 48
        total_w = rendered.get_width() + chip_pad + 20
        box = pygame.Rect(PLAY_W // 2 - total_w // 2, 88, total_w, 24)
        card = pygame.Surface(box.size, pygame.SRCALPHA)
        pygame.draw.rect(card, (10, 14, 22, 215), card.get_rect(), border_radius=6)
        pygame.draw.rect(card, (80, 95, 120, 190), card.get_rect(), 1, border_radius=6)
        surf.blit(card, box.topleft)
        surf.blit(rendered, (box.x + 10, box.y + 3))
        draw_source_chip(surf, (box.x + 16 + rendered.get_width(), box.y + 3), "real", self.f_small, anchor="topleft")
        if getattr(self, "pet", None):
            self._draw_pet_widget(surf)

    def _draw_pet_widget(self, surf) -> None:
        """Casual-friendly HUD widget for pet mode: hunger, sleep, mood, personality."""
        if not getattr(self, "pet", None):
            return
        pet = self.pet
        w, h = 236, 88
        x, y = 10, 310
        card = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(card, (10, 12, 18, 180), card.get_rect(), border_radius=10)
        surf.blit(card, (x, y))

        from kickthefly.core import cards

        slot = self.flies[0] if getattr(self, "flies", None) else None
        self._text(surf, f"PET: {cards.label(pet_card(pet, slot))}", (x + 12, y + 8), ACCENT, self.f_small)

        bx, bw = x + 12, w - 24
        self._text(surf, f"hunger {int(pet.hunger * 100)}%", (bx, y + 26), TEXT, self.f_small)
        self._bar(surf, bx + 80, y + 29, bw - 80, float(pet.hunger), (240, 120, 80) if pet.hunger > 0.7 else (170, 176, 188))

        self._text(surf, f"sleep {int(pet.sleep_pressure * 100)}%", (bx, y + 44), TEXT, self.f_small)
        self._bar(surf, bx + 80, y + 47, bw - 80, float(pet.sleep_pressure), (120, 160, 240) if pet.sleep_pressure > 0.7 else (170, 176, 188))

        mood_str = "content" if pet.mood > 0.6 else "happy" if pet.mood > 0.4 else "grumpy"
        if pet.is_sleeping:
            mood_str = "sleeping (ZZZ)"
        self._text(surf, f"mood: {mood_str}", (bx, y + 64), LABEL, self.f_small)
        age_days = (time.time() - pet.born_time) / 86400.0
        self._text(surf, f"age {age_days:.1f}d", (bx + bw, y + 64), LABEL, self.f_small, "topright")

    def _draw_escaperoom_hud(self, surf, now: float) -> None:
        cw, ch = 380, 84
        cx, cy = PLAY_W // 2 - cw // 2, 70
        card = pygame.Surface((cw, ch), pygame.SRCALPHA)
        pygame.draw.rect(card, (14, 18, 26, 220), card.get_rect(), border_radius=10)
        border_col = (255, 214, 95) if self.escaperoom_completed else (86, 180, 240)
        pygame.draw.rect(card, border_col, card.get_rect(), 2, border_radius=10)
        surf.blit(card, (cx, cy))

        elapsed = (self.escaperoom_finish_t if self.escaperoom_completed else now) - self.escaperoom_start_t
        elapsed = max(0.0, elapsed)

        status_txt = "GOAL REACHED!" if self.escaperoom_completed else "ESCAPE ROOM: SPEEDRUN"
        col = (255, 220, 100) if self.escaperoom_completed else (120, 210, 255)
        self._text(surf, status_txt, (cx + 14, cy + 8), col, self.f_small)

        timer_txt = f"{elapsed:.2f}s"
        self._text(surf, timer_txt, (cx + 14, cy + 24), INK, self.f_head)

        from kickthefly.lab.challenges import load_scores
        best = load_scores().get("escaperoom_speedrun")
        best_txt = f"BEST: {best:.2f}s" if best is not None else "BEST: --"
        self._text(surf, f"SEED: {self.escaperoom_seed}   {best_txt}", (cx + cw - 14, cy + 10), LABEL, self.f_small, "topright")

        if self.escaperoom_completed and self.escaperoom_code:
            self._text(surf, f"CODE: {self.escaperoom_code}", (cx + 14, cy + 58), (200, 245, 180), self.f_small)
            self._text(surf, "(R to run again)", (cx + cw - 14, cy + 58), LABEL, self.f_small, "topright")
        else:
            self._text(surf, "Navigate fan wind, flypaper & heat to reach sugar! (R: reset)", (cx + 14, cy + 58), LABEL, self.f_small)

    def _draw_reward(self, surf) -> None:
        x, y, w, h = 10, 308, 236, 64
        card = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(card, (10, 12, 18, 180), card.get_rect(), border_radius=10)
        surf.blit(card, (x, y))
        r = self.reward
        word, col = ("bliss", (255, 120, 170)) if r >= 70 else ("happy", (255, 160, 190)) if r >= 30 else \
            ("pleased", (220, 190, 200)) if r >= 8 else ("neutral", LABEL)
        self._text(surf, "REWARD", (x + 12, y + 8), LABEL, self.f_small)
        self._text(surf, f"{r:.0f}", (x + 72, y + 3), INK, self.f_head)
        self._text(surf, word, (x + w - 12, y + 6), col, self.f_bold, "topright")
        pygame.draw.rect(surf, (30, 36, 48), (x + 12, y + 32, w - 24, 10), border_radius=5)
        if r > 0.5:
            pygame.draw.rect(surf, col, (x + 12, y + 32, max(8, int((w - 24) * r / 100)), 10), border_radius=5)
        lvl = self.brain.level("reward")
        self._text(surf, f"PAM dopamine neurons x{lvl:.2f} calm", (x + 12, y + 46), LABEL, self.f_small)

    def _draw_pain(self, surf) -> None:
        x, y, w, h = 10, 78, 236, 222
        card = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(card, (10, 12, 18, 180), card.get_rect(), border_radius=10)
        surf.blit(card, (x, y))
        p = self.pain
        word, col = (("MAXED OUT", S_CRIT) if p >= 99 else ("agony", S_CRIT) if p >= 70 else ("severe", (236, 131, 90))
                     if p >= 35 else ("mild", S_WARN) if p >= 10 else ("calm", S_GOOD))
        self._text(surf, "PAIN", (x + 12, y + 8), LABEL, self.f_small)
        if "noci" in getattr(self.brain, "col", {}):
            self._text(surf, "CONNECTOME (noci)", (x + 46, y + 8), (50, 220, 100), self.f_small)
        else:
            self._text(surf, "ESTIMATE (RULE)", (x + 46, y + 8), DIM, self.f_small)
        r = self._text(surf, f"{p:.0f}", (x + 12, y + 20), INK, self.f_title)
        self._text(surf, "/100", (r.right + 4, r.bottom - 22), LABEL, self.f_small)
        self._text(surf, word, (x + w - 12, y + 30), col, self.f_bold, "topright")
        bx, bw = x + 12, w - 24
        pygame.draw.rect(surf, (30, 36, 48), (bx, y + 64, bw, 10), border_radius=5)
        if p > 0.5:
            pygame.draw.rect(surf, col, (bx, y + 64, max(8, int(bw * p / 100)), 10), border_radius=5)
        yy = y + 84
        for label, v in zip(("touch overload", "heat / cold", "chemical", "DN alarm", "body relay"), self.pain_parts):
            self._text(surf, label, (bx, yy - 3), TEXT, self.f_small)
            self._bar(surf, bx + 112, yy, bw - 146, float(v), (170, 176, 188))
            self._text(surf, f"{v * 100:3.0f}", (bx + bw, yy - 3), TEXT, self.f_small, "topright")
            yy += 15
        tr = self.pain_trace
        sy, sh = y + 162, 22
        pygame.draw.rect(surf, (22, 26, 34), (bx, sy, bw, sh), border_radius=3)
        if len(tr) > 2:
            pts = [(bx + k * bw / 719, sy + sh - tr[k] / 100 * (sh - 2)) for k in range(0, len(tr), 3)]
            if len(pts) > 1:
                pygame.draw.aalines(surf, S_CRIT, False, pts)
        self._text(surf, f"peak {self.pain_peak:.0f}   maxed {self.pain_max_s:.1f}s", (bx, y + h - 34), LABEL, self.f_small)
        lvl = PAIN_LEVELS[self.pain_level][0]
        self._text(surf, f"P  {lvl} ({self.brain.pain_neurons():,} neurons)", (bx, y + h - 18),
                   AMBER if self.pain_level else TEXT, self.f_small)

    def _draw_toolbar(self, surf) -> None:
        if self.cfg["brain.autopilot"] and self.cfg["brain.autopilot_hide_hud"]:
            self.tool_rects = []
            self.page_rects = []
            return
        lo = self.loadout
        tools = lo.page_tools()
        bw, gap = 58, 4
        x0 = (PLAY_W - bw * len(tools) - gap * (len(tools) - 1)) // 2
        self.tool_rects = []
        cur = self.tool_name()
        _, label, hint = TOOLS[self.tool]
        slot = lo.slot_of(cur)
        if slot is not None:
            key = self.cfg.keys.get(f"slot{slot + 1}", "")
        elif cur in lo.tools:                               # on the other page of a long loadout
            key = f"{tr('page')} {lo.tools.index(cur) // loadout_mod.PAGE_SIZE + 1}"
        else:
            key = tr("wheel")                               # picked from the tool wheel: not on the hotbar
        self._text(surf, f"{key.upper()}  {label}: {hint}", (PLAY_W // 2, FLOOR + 10), INK, self.f_bold, "midtop")
        for k, name in enumerate(tools):
            _, label, _ = TOOLS[TOOL_NAMES.index(name)]
            r = pygame.Rect(x0 + k * (bw + gap), FLOOR + 38, bw, 64)
            self.tool_rects.append(r)
            on = name == cur
            card = pygame.Surface(r.size, pygame.SRCALPHA)
            pygame.draw.rect(card, (60, 46, 22, 230) if on else (16, 18, 24, 210), card.get_rect(), border_radius=10)
            surf.blit(card, r)
            pygame.draw.rect(surf, AMBER if on else BORDER, r, 2, border_radius=10)
            draw_icon(surf, name, (r.centerx, r.y + 24), AMBER if on else TEXT)
            k_lbl = self.cfg.keys.get(f"slot{k + 1}", "").upper()
            self._text(surf, k_lbl, (r.x + 7, r.y + 4), LABEL, self.f_small)
            self._text(surf, label[:9], (r.centerx, r.y + 44), AMBER if on else INK, self.f_small, "midtop")
        self.page_rects = []
        if lo.n_pages > 1:                                     # page arrows either side, and which page this is
            y = FLOOR + 38 + 18
            for d, x, sym in ((-1, x0 - 30, "<"), (1, x0 + len(tools) * (bw + gap) + 2, ">")):
                r = pygame.Rect(x, y, 28, 28)
                self.page_rects.append((r, d))
                pygame.draw.rect(surf, (16, 18, 24, 210), r, border_radius=8)
                pygame.draw.rect(surf, BORDER, r, 1, border_radius=8)
                self._text(surf, sym, r.center, INK, self.f_bold, "center")
            self._text(surf, f"{lo.page + 1}/{lo.n_pages}", (x0 - 16, y + 34), LABEL, self.f_small, "center")

    def _draw_brain(self, now: float) -> None:
        scr, br = self.screen, self.brain
        x0 = PLAY_W
        pa = getattr(self, "panel_alpha", 255)
        pygame.draw.rect(scr, (*PANEL_BG, pa), (x0, 0, W - x0, H))
        pygame.draw.line(scr, BORDER, (x0, 0), (x0, H))
        x = x0 + 12
        self._text(scr, "THE FLY'S BRAIN", (x, 8), ACCENT, self.f_head)
        status, scol = ("FLATLINE", S_CRIT) if br.dead else ("LIVE", S_GOOD)
        r = self._text(scr, status, (W - 14, 13), scol, self.f_bold, "topright")
        aacircle(scr, (r.x - 10, r.centery), 4, scol if br.dead or self.calm_fx or int(now * 2) % 2 else DIM)
        self._text(scr, f"MaleCNS v1.0 connectome, {br.n:,} neurons", (x, 34), LABEL, self.f_small)

        nw, nh = VIEW_SIZES["panel"]
        self.view_rect = pygame.Rect(x, 54, nw, nh)
        if self.big_view:                            # the big view is showing it; don't pay for both renders
            pygame.draw.rect(scr, (4, 5, 8), self.view_rect)
            self._text(scr, "shown in big view", self.view_rect.center, DIM, self.f_small, "center")
        else:
            scr.blit(self._panel_image(self._view_surface("panel")), (x, 54))
            self._hud_overlay(scr, self.view_rect, now, small=True)
            self.x3.draw_killcam_rings(scr, self.view_rect, "panel", now)
            self._text(scr, "B: big view", (x + nw - 6, 54 + nh - 16), LABEL, self.f_small, "topright")
        y = 54 + nh + 12

        bw = W - x - 12
        y = self._card(scr, x, y, bw, "TOUCH NEURONS", "spikes/s per neuron", 4)
        fast, base = self.x3.panel_fast(br)          # the recorded group rates while the kill cam plays
        for region, label in (("head", "head BM/JO"), ("body", "body SNta"), ("legs", "legs SNpp"), ("wing", "wing WG")):
            hz = float(fast[br.col[region]])
            self._text(scr, label, (x + 8, y - 3), TEXT, self.f_small)
            self._bar(scr, x + 104, y, bw - 150, hz / 60.0, (90, 200, 120))
            self._text(scr, f"{hz:4.0f}", (x + bw - 8, y - 3), TEXT, self.f_small, "topright")
            y += 16
        y += 10
        y = self._card(scr, x, y, bw, "DESCENDING NEURONS", "x calm baseline", len(MOTOR))
        for name, _, _, label in MOTOR:
            lvl = float(fast[br.col[name]] / max(base[br.col[name]], 2.0))
            th = THRESH.get(name)
            over = th is not None and lvl > th
            self._text(scr, label, (x + 8, y - 3), AMBER if over else TEXT, self.f_small)
            self._bar(scr, x + 200, y, bw - 246, lvl / 5.0, AMBER if over else ACCENT, None if th is None else th / 5.0)
            self._text(scr, f"{lvl:4.1f}", (x + bw - 8, y - 3), TEXT, self.f_small, "topright")
            y += 16
        y += 10
        y = self._card(scr, x, y, bw, "WHOLE BRAIN", "spikes/s, last 12 s", 3)
        tr = br.history(br.hist_n - 600, br.hist_n)[:, br.col["whole brain"]]
        th_h = 40
        if len(tr) > 2:
            top = max(float(tr.max()), 8.0)
            pts = [(x + 8 + k * (bw - 16) / (len(tr) - 1), y + th_h - float(tr[k]) / top * (th_h - 4))   # float32 breaks
                   for k in range(0, len(tr), 2)]                                                        # pygame-ce aalines
            pygame.draw.aalines(scr, ACCENT, False, pts)
            self._text(scr, f"{tr[-1]:.1f}", (x + bw - 8, y - 6), TEXT, self.f_small, "topright")
        y += th_h + 12

        self._text(scr, "REACTIONS", (x, y), LABEL, self.f_small)
        y += 16
        tags = self.cfg.tags_on()
        for t, msg, src in reversed(self.log[-max(1, min(4, (H - 36 - y) // 15)):]):
            age = now - t
            col = AMBER if age < 1.0 else TEXT if age < 5 else DIM
            r = self._text(scr, f"{age:4.1f}s  {msg}", (x, y), col, self.f_small)
            if tags and src:
                draw_source_chip(scr, (W - 14, y), src, self.f_small, anchor="topright")
            y += 15
        self._text(scr, f"sim {br.steps_per_s:4.0f} steps/s  {br.sim.last_step_ms:4.1f} ms/step",
                   (x, H - 18), DIM, self.f_small)

    def _card(self, scr, x, y, w, title, unit, rows) -> int:
        h = 24 + rows * 16
        pygame.draw.rect(scr, (*CARD, min(255, getattr(self, "panel_alpha", 255) + 40)), (x - 4, y - 4, w + 8, h + 4), border_radius=8)
        self._text(scr, title, (x + 4, y), LABEL, self.f_small)
        self._text(scr, unit, (x + w - 6, y), DIM, self.f_small, "topright")
        return y + 22

    def _bar(self, scr, x, y, w, frac, color, mark=None) -> None:
        pygame.draw.rect(scr, (30, 36, 48), (x, y, w, 8), border_radius=4)
        fw = int(w * max(0.0, min(frac, 1.0)))
        if fw > 0:
            pygame.draw.rect(scr, color, (x, y, max(fw, 4), 8), border_radius=4)
        if mark is not None:
            tx = x + int(w * min(mark, 1.0))
            pygame.draw.line(scr, INK, (tx, y - 2), (tx, y + 9))

    def _draw_autopsy(self, surf, now: float) -> None:
        rep = self.report
        e = float(np.clip((now - self.fly.dead_at - AUTOPSY_DELAY) / 0.3, 0.0, 1.0))
        veil = pygame.Surface((PLAY_W, H), pygame.SRCALPHA)
        veil.fill((6, 7, 10, int(200 * e)))
        surf.blit(veil, (0, 0))
        card = pygame.Rect(28, 26, PLAY_W - 56, H - 52)
        pygame.draw.rect(surf, (20, 22, 28), card, border_radius=16)
        pygame.draw.rect(surf, BORDER, card, 1, border_radius=16)
        x, y = card.x + 26, card.y + 18
        self._text(surf, "BRAIN AUTOPSY", (x, y), INK, self.f_title)
        m, s = divmod(int(rep["alive_s"]), 60)
        self._text(surf, f"survived {m}:{s:02d}   {rep['hits']} hits   killed by {rep['by']}", (x + 2, y + 44), LABEL, self.f_text)
        top = rep["top"]
        self._text(surf, f"Biggest change before death: {top['name']} at {top['ratio']:.1f}x its calm rate "
                         f"({top['base']:.1f} > {top['final']:.1f} spikes/s)", (x + 2, y + 68), TEXT, self.f_text)
        pr = self._text(surf, "PAIN", (x + 2, y + 92), S_CRIT, self.f_bold)
        self._text(surf, f"peak {rep['pain_peak']:.0f}/100    maxed out for {rep['pain_max_s']:.1f} s    "
                         f"last 2 s alive {rep['pain_last']:.0f}/100", (pr.right + 10, y + 92), TEXT, self.f_text)
        r_btn = pygame.Rect(card.right - 196, card.y + 22, 170, 42)
        pygame.draw.rect(surf, AMBER, r_btn, border_radius=10)
        self._text(surf, "NEW FLY  (R)", r_btn.center, (30, 20, 8), self.f_bold, "center")
        self.new_fly_rect = r_btn
        self.save_rects = []
        if self.x3.kc_available():                   # 3.0: the kill cam, offered on the autopsy card
            r = pygame.Rect(card.right - 196, card.y + 102, 198, 26)
            pygame.draw.rect(surf, (40, 110, 150), r, border_radius=7)
            self._text(surf, "KILL CAM (slow motion)", r.center, INK, self.f_small, "center")
            self.save_rects.append((r, "killcam"))
        for k, (label, what) in enumerate((("SAVE IMAGE", "png"), ("SAVE DEATH GIF", "gif"))):
            r = pygame.Rect(card.right - 196 + k * 88 - (0 if k == 0 else 4), card.y + 70, 84 if k == 0 else 110, 26)
            pygame.draw.rect(surf, (44, 50, 64), r, border_radius=7)
            self._text(surf, label, r.center, INK, self.f_small, "center")
            self.save_rects.append((r, what))

        # diverging bars: last 2 s alive vs calm baseline, log2 scale
        y0 = card.y + 142
        lab_w, col_w = 170, 64
        cx0 = x + lab_w
        chart_w = card.w - 52 - lab_w - col_w * 3 - 40      # 40 px keeps a full-length bar's label off the columns
        mid = cx0 + chart_w // 2
        self._text(surf, "Last 2 s alive vs its calm baseline", (x, y0 - 8), INK, self.f_bold)
        hdr_y = y0 + 18
        self._text(surf, "calm", (cx0 + chart_w + col_w, hdr_y), LABEL, self.f_small, "topright")
        self._text(surf, "final", (cx0 + chart_w + col_w * 2, hdr_y), LABEL, self.f_small, "topright")
        self._text(surf, "dead", (cx0 + chart_w + col_w * 3, hdr_y), LABEL, self.f_small, "topright")
        for tick, lab in ((-4, "1/16x"), (-2, "1/4x"), (0, "same"), (2, "4x"), (4, "16x")):
            tx = mid + tick / 4 * (chart_w / 2 - 8)
            self._text(surf, lab, (tx, hdr_y), LABEL, self.f_small, "midtop")
        ry = hdr_y + 20
        row_h = 15
        chart_top = ry
        n_rows = len(rep["rows"])
        for k in (-4, -2, 2, 4):
            tx = int(mid + k / 4 * (chart_w / 2 - 8))
            pygame.draw.line(surf, (32, 35, 42), (tx, chart_top - 2), (tx, chart_top + n_rows * row_h + 8))
        pygame.draw.line(surf, (90, 90, 86), (mid, chart_top - 2), (mid, chart_top + n_rows * row_h + 8), 2)
        prev_section = None
        for row in rep["rows"]:
            if prev_section and row["section"] != prev_section:
                ry += 8
            prev_section = row["section"]
            lr = math.log2(max(row["ratio"], 1e-3))
            frac = max(-1.0, min(1.0, lr / 4))
            bl = int(abs(frac) * (chart_w / 2 - 8))
            col = UP if frac > 0 else DOWN
            if bl >= 2:
                rx = mid + 2 if frac > 0 else mid - 2 - bl
                pygame.draw.rect(surf, col, (rx, ry + 2, bl, row_h - 5), border_radius=3)
            else:
                aacircle(surf, (mid, ry + row_h // 2), 3, MID)
            self._text(surf, row["name"], (x, ry), TEXT, self.f_small)
            pct = (row["ratio"] - 1) * 100
            lab = f"{row['ratio']:.1f}x" if row["ratio"] >= 2 else f"{pct:+.0f}%"
            lx = mid + 2 + bl + 6 if frac > 0 else mid - 2 - bl - 6
            self._text(surf, lab, (lx, ry), INK, self.f_small, "topleft" if frac > 0 else "topright")
            self._text(surf, f"{row['base']:.1f}", (cx0 + chart_w + col_w, ry), LABEL, self.f_small, "topright")
            self._text(surf, f"{row['final']:.1f}", (cx0 + chart_w + col_w * 2, ry), INK, self.f_small, "topright")
            self._text(surf, f"{row['after']:.1f}", (cx0 + chart_w + col_w * 3, ry), DIM, self.f_small, "topright")
            ry += row_h
        self._text(surf, "spikes/s per neuron. Red = higher than calm, blue = lower.", (x, ry + 6), LABEL, self.f_small)

        # timeline: x calm baseline, one shared axis
        ty = ry + 30
        th = card.bottom - ty - 48
        tx0, tw = x + 40, card.w - 52 - 40
        self._text(surf, "Activity vs calm, 20 s before death to 3 s after", (x, ty - 4), INK, self.f_bold)
        lx = x + 380
        for (label, _), col in zip(rep["series"], SERIES):
            pygame.draw.rect(surf, col, (lx, ty + 3, 14, 4), border_radius=2)
            lx = self._text(surf, label, (lx + 20, ty - 2), TEXT, self.f_small).right + 18
        ty += 22
        th -= 22
        lo, hi = -3.0, 5.0                               # log2 axis: 1/8x (and silence) up to 32x

        def yof(v):
            return ty + th - (min(max(math.log2(max(float(v), 1e-6)), lo), hi) - lo) / (hi - lo) * th

        for v, lab in ((0.125, "0"), (0.25, "1/4x"), (1, "1x"), (4, "4x"), (16, "16x")):
            gy = yof(v)
            pygame.draw.line(surf, (70, 70, 66) if v == 1 else (32, 35, 42), (tx0, gy), (tx0 + tw, gy))
            self._text(surf, lab, (tx0 - 6, gy - 7), LABEL, self.f_small, "topright")
        n = max(rep["n"], 2)
        for (label, vals), col in zip(rep["series"], SERIES):
            if len(vals) < 2:
                continue
            pts = [(tx0 + k * tw / (n - 1), yof(v)) for k, v in enumerate(vals)]
            pygame.draw.lines(surf, col, False, pts, 2)
        # pain strip on the same time axis: one hue, dark = none, bright red = maxed
        sy = ty + th + 8
        self._text(surf, "pain", (tx0 - 6, sy), LABEL, self.f_small, "topright")
        pain = rep["pain"]
        if len(pain):
            step = max(1, len(pain) // 300)
            cw = tw / len(pain) * step
            for k in range(0, len(pain), step):
                t = float(np.clip(pain[k] / 100, 0, 1))
                col = tuple(int(MID[j] + (UP[j] - MID[j]) * t) for j in range(3))
                pygame.draw.rect(surf, col, (tx0 + k * tw / len(pain), sy, math.ceil(cw) + 1, 14))
        dx = tx0 + rep["death_at"] * tw / (n - 1)
        pygame.draw.line(surf, S_CRIT, (dx, ty - 2), (dx, sy + 14), 2)
        self._text(surf, "died", (dx + 6, ty), S_CRIT, self.f_small)

    def menu_first(self, ev, mouse_pos) -> bool:
        """The pause menu, quitting and Esc. Returns True if the event was used here."""
        if ev.type == pygame.QUIT:
            if self.menu.screen == "confirm_quit":
                self.want_quit = True                    # a second close request quits
            else:
                self.open_menu("confirm_quit")
            return True
        if self.menu.open:
            self.menu.handle(ev, getattr(ev, "pos", mouse_pos))
            return True
        if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
            if getattr(self, "kwheel_open", False):                   # Esc cancels the tool wheel, and only that
                self.close_wheel(False)
            elif self._overlay_open() and self.report is None:        # Esc closes an overlay first
                self.help_open = self.surgery_open = self.big_view = self.training_open = False
            else:
                self.open_menu("pause")
            return True
        return False

    def do_action(self, action: str, now: float) -> None:
        """Hotkeys shared by the 2D and 3D games. Settings hotkeys go through set_setting so the menu stays in sync."""
        if action.startswith("slot") and action[4:].isdigit():
            self.select_slot(int(action[4:]) - 1)
        elif action == "page_prev":
            self.turn_page(-1)
        elif action == "page_next":
            self.turn_page(1)
        elif action == "loadout":
            if not getattr(self, "photo_mode", False):        # in photo mode Q flies the camera down
                self.open_loadout_editor()
        elif action == "tool_wheel":
            self.open_wheel()
        elif action == "help":
            self.help_open = not self.help_open
        elif action == "training":
            self.training_open = not self.training_open
        elif action == "surgery":
            self.surgery_open = not self.surgery_open
        elif action == "arena":
            i = (self.arena_i + 1) % len(ARENAS)
            while not self.three_d and ARENAS[i] in THREE_D_ONLY:     # the 2D game has no outdoor worlds or kitchen
                i = (i + 1) % len(ARENAS)
            self.set_setting("brain.arena", ARENAS[i])
        elif action == "spawn":
            self.spawn_fly()
        elif action == "cycle_fly":
            self.cycle_focus()
        elif action == "recall":
            self.recall_flies(now)
        elif action == "mute":
            self.set_setting("audio.mute", not self.cfg["audio.mute"])
        elif action == "screenshot":
            self.save_png()
        elif action == "gif":
            self.save_gif()
        elif action == "reset":
            self.new_fly()
            self.on_arena_changed(now)
        elif action == "big_view":
            self.big_view = not self.big_view
        elif action == "autopilot":
            new_val = not self.cfg["brain.autopilot"]
            self.set_setting("brain.autopilot", new_val)
            self.note(f"AUTOPILOT {'on: spectator mode' if new_val else 'off'}", source="rule")
        elif action == "photo_mode":
            self.photo_mode = not getattr(self, "photo_mode", False)
            self.note(f"PHOTO MODE {'on (clean preview)' if self.photo_mode else 'off'}", source="rule")
        elif action == "stethoscope":
            self.toggle_stethoscope()
        elif action == "timelapse":
            self.toggle_timelapse()
        elif action == "record_video":
            self.toggle_video_recording()
        elif action == "immortal":
            self.set_setting("brain.immortal", not self.immortal)
            self.note(f"IMMORTAL {'on: it can feel pain but never die' if self.immortal else 'off'}")
            self.popup(self._above_head(), "IMMORTAL!" if self.immortal else "MORTAL", (255, 225, 120), force=True)
        elif action == "pain":
            self.set_setting("brain.pain_level", (self.pain_level + 1) % len(PAIN_LEVELS))
            self.note(f"PAIN     {PAIN_LEVELS[self.pain_level][0]}: {self.brain.pain_neurons():,} neurons")
        elif action == "fullscreen":
            self.toggle_fullscreen()
        elif action.startswith("time_"):
            self.time_action(action)

    def handle(self, ev, now: float) -> bool:
        """2D input. Returns False to quit. Keys go through the rebindable actions in config.py."""
        if self.search_key(ev):
            return True
        if self.menu_first(ev, pygame.mouse.get_pos()):
            return not self.want_quit
        if self.x3.kc_playing() and self.x3.handle_event(ev):   # review: the kill cam's keys first, before the tutorial's
            return True
        if self.tutorial_event(ev) or self.wheel_event(ev):
            return True
        if self.x3.handle_event(ev):                 # 3.0: kill cam keys and clicks, the launch card, the Neurodex key
            return True
        if ev.type == pygame.KEYDOWN:
            # 1/2/3/0 pick the big view's camera, unless the player bound that key to the big view itself (2.13: digits
            # are rebindable), which must still close it
            if self.big_view and self.cfg.action_for(pygame.key.name(ev.key)) != "big_view":
                if ev.key in (pygame.K_1, pygame.K_KP1):
                    self.view.set_preset("front")
                    return True
                if ev.key in (pygame.K_2, pygame.K_KP2):
                    self.view.set_preset("side")
                    return True
                if ev.key in (pygame.K_3, pygame.K_KP3):
                    self.view.set_preset("top")
                    return True
                if ev.key in (pygame.K_0, pygame.K_KP0):
                    self.view.set_preset("reset")
                    return True
            if (ev.key == pygame.K_r and (ev.mod & pygame.KMOD_SHIFT)) or getattr(ev, "unicode", "") == "R":
                self.toggle_video_recording()
                return True
            if ev.key == pygame.K_s and self.cfg.action_for("s") in (None, *config.MOVEMENT_3D_ONLY):
                self.save_png()                          # S has always saved a screenshot in the 2D game
                return True
            action = self.cfg.action_for(pygame.key.name(ev.key))
            if action is not None:
                self.do_action(action, now)
            return True
        if ev.type == pygame.MOUSEWHEEL and self.big_view:
            mpos = getattr(ev, "pos", pygame.mouse.get_pos())
            if getattr(self, "big_rect", None) and self.big_rect.collidepoint(mpos):
                self.view.zoom_by(1.15 if ev.y > 0 else 0.87)
                return True
        if ev.type == pygame.MOUSEWHEEL and not self._overlay_open() and ev.y:
            self.cycle_tool(-ev.y)                       # the wheel steps through the loadout, as in the 3D game
            return True
        if ev.type == pygame.MOUSEMOTION and self.big_view and getattr(self, "big_drag", None):
            start_pos, btn, y0, p0, px0, py0 = self.big_drag
            dx = ev.pos[0] - start_pos[0]
            dy = ev.pos[1] - start_pos[1]
            if (pygame.key.get_mods() & pygame.KMOD_SHIFT) or btn == 2:
                self.view.set_camera(y0, p0, px0 + dx * 120.0, py0 + dy * 120.0, self.view.zoom, recompute=False)
            else:
                self.view.set_camera(y0 + dx * 0.4, p0 - dy * 0.4, px0, py0, self.view.zoom, recompute=False)
            return True
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button in (1, 2, 3):
            if self.science_card is not None and self.science_rect().collidepoint(ev.pos):
                self.science_card = None
                return True
            if self.challenge is not None and self.challenge.click(ev.pos):
                return True
            if self.help_open:
                self.help_open = False
                return True
            if self.training_open:
                self._training_click(ev.pos)
                return True
            if self.surgery_open:
                for r, k, mode in self.surgery_buttons:
                    if r.collidepoint(ev.pos):
                        if k < 0:
                            self.surgery_modes = [0] * len(SURGERY)
                            self.type_ops = {}
                            self._apply_surgery()
                            self.note("SURGERY  cleared")
                        else:
                            self._set_surgery(k, mode)
                return True
            if self.report is not None:
                if getattr(self, "new_fly_rect", None) and self.new_fly_rect.collidepoint(ev.pos):
                    self.new_fly()
                for r, what in getattr(self, "save_rects", []):
                    if r.collidepoint(ev.pos):
                        if what == "killcam":
                            self.x3.kc_start()
                        else:
                            self.save_png() if what == "png" else self.save_gif(self.death_frames)
                return True
            if self.view_rect.collidepoint(ev.pos):
                self.big_view = not self.big_view
                return True
            if self.big_view:
                if ev.button == 1 and self.search_click(ev.pos):         # search box, results, path tracer
                    return True
                # Region neuron list modal interaction:
                if getattr(self, "selected_region", None):
                    if getattr(self, "region_close_button", None) and self.region_close_button.collidepoint(ev.pos):
                        self.selected_region = None
                        return True
                    if getattr(self, "region_prev_btn", None) and self.region_prev_btn.collidepoint(ev.pos):
                        self.region_page = max(0, getattr(self, "region_page", 0) - 1)
                        return True
                    if getattr(self, "region_next_btn", None) and self.region_next_btn.collidepoint(ev.pos):
                        self.region_page = getattr(self, "region_page", 0) + 1
                        return True
                    for nr, nid in getattr(self, "region_neuron_buttons", []):
                        if nr.collidepoint(ev.pos):
                            self.inspect = self._neuron_info(nid)
                            self.selected_region = None
                            self.update_stethoscope_target()
                            self.sound.play("click")
                            return True
                    if getattr(self, "region_modal_rect", None) and not self.region_modal_rect.collidepoint(ev.pos):
                        self.selected_region = None
                        self.update_stethoscope_target()
                        return True
                    return True

                # Time-lapse recording button:
                if getattr(self, "big_timelapse_button", None) and self.big_timelapse_button.collidepoint(ev.pos):
                    self.toggle_timelapse()
                    return True

                # Stethoscope buttons:
                if getattr(self, "big_steth_button", None) and self.big_steth_button.collidepoint(ev.pos):
                    self.toggle_stethoscope()
                    self.sound.play("click")
                    return True
                if getattr(self, "big_steth_target_button", None) and self.big_steth_target_button.collidepoint(ev.pos):
                    self.cycle_stethoscope_target()
                    self.sound.play("click")
                    return True

                # Mode toggle:
                if getattr(self, "big_mode_button", None) and self.big_mode_button.collidepoint(ev.pos):
                    mode = self.view.toggle_view_mode()
                    self.note(f"VIEW     {'per-region heatmap' if mode == 'region' else 'per-neuron'}")
                    self.sound.play("click")
                    return True

                # Region list click-through:
                if self.view.view_mode == "region":
                    for rr, rname in getattr(self, "big_region_buttons", []):
                        if rr.collidepoint(ev.pos):
                            self.selected_region = rname
                            self.region_page = 0
                            self.update_stethoscope_target()
                            self.sound.play("click")
                            return True

                for r, preset_name in getattr(self, "big_preset_buttons", []):
                    if r.collidepoint(ev.pos):
                        self.view.set_preset(preset_name)
                        return True
                if getattr(self, "big_rect", None) and self.big_rect.collidepoint(ev.pos):
                    self.big_drag = (ev.pos, ev.button, self.view.yaw, self.view.pitch, self.view.pan_x, self.view.pan_y)
                    return True
                return True
            if self.hotbar_click(ev.pos):
                return True
            if ev.button == 3 and TOOLS[self.tool][0] == "hand" and ev.pos[0] < PLAY_W:
                dec = self._nearest_decoy(ev.pos, 45)
                if dec is not None:
                    self.remove_decoy(dec)
                    return True
            if ev.pos[0] < PLAY_W:
                self.use_tool(ev.pos, now)
        elif ev.type == pygame.MOUSEBUTTONUP:
            if self.big_view and getattr(self, "big_drag", None):
                start_pos, btn, _, _, _, _ = self.big_drag
                self.big_drag = None
                if getattr(self.view, "_dirty_big", False):
                    self.view._recompute_big()
                if btn == 1 and math.hypot(ev.pos[0] - start_pos[0], ev.pos[1] - start_pos[1]) < 6:
                    if self.inspect is not None:
                        flip = getattr(self, "inspect_flip_button", None)
                        if flip is not None and flip[0].collidepoint(ev.pos):
                            self.flip_neuron(flip[1])
                            return True
                        pb = getattr(self, "patch_button", None)
                        if pb is not None and pb[0].collidepoint(ev.pos):
                            self.patch_neuron(pb[1])
                            return True
                        for r, mode in getattr(self, "inspect_buttons", []):
                            if r.collidepoint(ev.pos):
                                self.type_ops[self.inspect["type"]] = mode
                                self._apply_surgery()
                                self.note(f"SURGERY  {self.inspect['type']}: {'off' if mode < 0 else 'on' if mode > 0 else 'normal'}")
                                return True
                    if self.big_rect.collidepoint(ev.pos):
                        self._inspect_at(ev.pos)
                return True
            if ev.button == 1:
                if getattr(self, "grabbed_decoy", None) is not None:
                    self.grabbed_decoy["v"] = np.array([0.0, -3.0])
                    self.grabbed_decoy = None
                if not self.fly.wrapped:
                    self.fly.grabbed = None
                self.torching = False
                if hasattr(self, "laser_state"):
                    self.laser_state.trigger_release()
        return True


def page_load_state(m, surf, rect, mouse) -> None:
    """Pause menu > Load State: saves in the saves folder, newest first; ones that can't load say why."""
    from kickthefly.core import savestate

    game = m.host
    m.text(surf, "LOAD STATE", (rect.x + 24, rect.y + 16), menu_ui.INK, m.f_head)
    folder = paths.get().saves_dir
    m.text(surf, str(folder), (rect.x + 24, rect.y + 50), menu_ui.LABEL, m.f_small)
    now = time.perf_counter()
    cache = getattr(m, "_saves_cache", None)
    if cache is None or now - cache[0] > 2.0:
        entries = []
        files = sorted(folder.glob("*" + savestate.SUFFIX), key=lambda f: f.stat().st_mtime, reverse=True) \
            if folder.is_dir() else []
        for f in files[:40]:
            try:
                meta = savestate.read_meta(f)
                entries.append((f, meta, savestate.compatible(meta, game)))
            except savestate.SaveError as e:
                entries.append((f, None, str(e)))
        m._saves_cache = cache = (now, entries)
    entries = cache[1]
    body = pygame.Rect(rect.x + 16, rect.y + 80, rect.w - 32, rect.h - 150)
    off = int(m.scroll.get("load_state", 0))
    m.clip = body
    prev = surf.get_clip()
    surf.set_clip(body)
    y = body.y - off
    if not entries:
        m.text(surf, "No saves yet. Use Save State in the pause menu.", body.center, menu_ui.LABEL, m.f_text, "center")
    for f, meta, why in entries:
        row = pygame.Rect(body.x, y, body.w - 12, 56)
        if meta is not None:
            nfl = len(meta["flies"])
            label = (f"{meta['created']}   {meta['mode'].upper()}   {nfl} {'fly' if nfl == 1 else 'flies'}   "
                     f"{meta.get('arena') or ARENAS[min(meta['arena_i'], len(ARENAS) - 1)]}   seed {meta['seed']}")
            sub = f"{f.name}   v{meta['app_version']} on {meta['platform']}"
        else:
            label, sub = f.name, ""
        m.button(surf, row, "", (lambda f=f: game.load_state(f)), id=("load", f.name), enabled=why is None,
                 tip=why and f"Can't load: {why}")
        m.text(surf, label, (row.x + 14, row.y + 8), menu_ui.INK if why is None else menu_ui.DIM, m.f_bold)
        m.text(surf, f"can't load: {why}" if why else sub, (row.x + 14, row.y + 32),
               menu_ui.BAD if why else menu_ui.LABEL, m.f_small)
        y += 62
    m.content_h["load_state"] = max(0, y + off - body.bottom)
    surf.set_clip(prev)
    m.clip = None
    m.button(surf, (rect.right - 164, rect.bottom - 58, 140, 42), "Back", m.back, style="primary", id=("load", "back"))


def now_wipe_armed(game) -> bool:
    """Wiping memory takes a second click within 3 seconds."""
    return time.perf_counter() - game.wipe_armed < 3.0


def playable_brain(cfg) -> str:
    """The connectome the windowed game loads. The larva body (game/larva.py) does not implement the Fly interface
    the game loop uses yet, so the windowed game runs the adult brain and says so; the larva brain is available
    headless (--headless --validate --brain larva, protocols, benchmarks)."""
    if cfg.get("brain.brain", "adult") == "larva":
        log.warning("the larva brain is headless-only in this build (the windowed larva game is not finished); "
                    "starting with the adult brain. Use --headless --validate --brain larva for the larva connectome.")
    return "adult"


def page_neuron_shapes(menu, surf, rect, mouse) -> None:
    """The one-time question (3.0 day 3 review): may the brain view download ten real neuron shapes from neuPrint?"""
    cx = rect.centerx
    menu.text(surf, tr("Download real neuron shapes?"), (cx, rect.y + 40), menu_ui.INK, menu.f_head, "midtop")
    menu.wrapped(surf, tr("The brain view can draw ten neurons (the giant fiber DNp01, DNa02, MBON01, MBON14 and a Kenyon cell type) "
                          "from their real electron-microscopy skeletons, downloaded once from Janelia's neuPrint and kept in a "
                          "cache. Without them it draws estimated fibers. Nothing about the simulation changes either way: every "
                          "neuron is simulated as a point. This is the game's only network use besides Streamer mode, so it is "
                          "off unless you say yes. Settings > Brain changes it later; it applies on the next launch."),
                 (rect.x + 60, rect.y + 96), rect.w - 120, menu_ui.TEXT, menu.f_text, max_lines=9)

    def answer(yes: bool) -> None:
        menu.host.set_setting("brain.neuron_shapes", yes)
        menu.back()

    bw = max(menu.bw(tr("Yes, download them"), 240), menu.bw(tr("No thanks"), 240))
    menu.button(surf, (cx - 10 - bw, rect.bottom - 90, bw, 50), tr("Yes, download them"), lambda: answer(True), id=("ns", "yes"))
    menu.button(surf, (cx + 10, rect.bottom - 90, bw, 50), tr("No thanks"), lambda: answer(False), style="primary",
                id=("ns", "no"))


def load_brain(out: dict) -> None:
    try:
        from kickthefly.sim import brainpack
        from kickthefly.sim.connectome.sim import LIFParams, LIFSim

        brain_type = str(out.get("brain_type", "adult")).lower()
        pack = brainpack.find(brain=brain_type)
        if pack is None:                             # source checkout: pack the connectome once (~30 s)
            out["stage"] = f"building the {brain_type} brain pack (first run only)"
            pack = brainpack.build(brain=brain_type)
        out["stage"] = f"unpacking the {brain_type}'s brain"
        g, weights, soma = brainpack.load(pack)
        if out.get("mirror_weights", False) and brain_type == "adult":
            from kickthefly.core import simcore
            weights = simcore.symmetrize_weights(g, weights)
        out["stage"] = f"wiring {g.n:,} neurons"
        seed = int(out.get("seed", 0))
        p = LIFParams()
        if "backend" in out:
            p.backend = out["backend"]
        if "dtype" in out:
            p.dtype = out["dtype"]
        p.individuality = out.get("individuality", "off")   # callers that want individuality pass it (main, kick3d)
        sim = LIFSim(None, p, W_in=weights, seed=seed)
        brain = Brain(g, sim, seed=seed)
        brain.brain_type = brain_type
        out["stage"] = "placing neurons"
        if brain_type == "larva":
            pain_mask = np.zeros(g.n, bool)
            from kickthefly.sim.connectome.larva_loader import larva_groups
            pain_mask[larva_groups(g.subtype)["noci"]] = True
        else:
            pain_groups = [brain.col[n] for n in (*TOUCH, "heat", "cold", "smell", "taste", "body_extra")]
            pain_mask = np.isin(brain.det_id, pain_groups) | (brain.pop_id == [n for n, _ in POPS].index("ascending"))
        from kickthefly.game.gpu_brainview import make_view          # 3.1.0 task 4: the GPU view, or this file's CPU one
        out["view"] = make_view(soma, weights, pain_mask, regions=getattr(g, "region", None), graph=g)
        if getattr(g, "dan_mbon", None) is not None:
            from kickthefly.core import memory
            out["stage"] = "loading the fly's memory"
            brain.memory = memory.Memory(g, sim)
        out["stage"] = "waking the fly up"
        brain.warmup()
        out["brain"] = brain
        out["brain_type"] = brain_type
        out["graph"], out["weights"] = g, weights   # kept so spawning more flies never re-touches disk
    except Exception as e:  # shown on the loading screen
        out["error"] = f"{type(e).__name__}: {e}"


def parse_args(argv: list[str] | None = None):
    import argparse

    ap = argparse.ArgumentParser(prog="KickTheFly", description="Kick the Fly: a live MaleCNS v1.0 fly connectome.")
    ap.add_argument("--2d", dest="two_d", action="store_true", help="the original 2D game")
    ap.add_argument("--fullscreen", action="store_true")
    from kickthefly.sim.connectome import backends as sim_backends
    ap.add_argument("--backend", help="Compute backend (auto, cpu, numba, torch-cuda, torch-rocm, torch-cpu, gl) or Linux display backend (wayland, x11)")
    ap.add_argument("--brain", choices=("adult", "larva"), default=None, help="which connectome to simulate: adult or larva")
    ap.add_argument("--individuality", choices=("off", "subtle", "strong"), default=None, help="fly individuality setting")
    ap.add_argument("--pet", action="store_true", help="start in pet mode")
    ap.add_argument("--sim-backend", choices=sim_backends.BACKEND_NAMES, help="Simulation compute backend")
    ap.add_argument("--dtype", choices=("float32", "float64"), help="simulation state precision (default float32)")
    ap.add_argument("--seed", type=int, help="random seed for the brains and the game")
    ap.add_argument("--smoke", nargs="+", metavar="ARG", help="build check: SECONDS [SCREENSHOT.png]")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--headless", action="store_true", help="no window: run --validate or --protocol and exit")
    ap.add_argument("--validate", action="store_true", help="run the validation suite (implies --headless)")
    ap.add_argument("--protocol", metavar="FILE", help="run a YAML protocol file (implies --headless)")
    ap.add_argument("--out", metavar="PATH", help="where headless results go")
    ap.add_argument("--arena", choices=ARENAS, help="start in this arena (saved to config.toml like pressing E): room, fan, "
                    "flypaper, pool, lamp, escaperoom, thermo, field or orchard; field and orchard need the 3D game")
    ap.add_argument("--nwb", action="store_true", help="also write each recording as NWB (needs pynwb)")
    ap.add_argument("--workers", type=int, help="worker processes for headless runs (default: up to 4)")
    ap.add_argument("--seeds", help="validation seeds, e.g. 1000-1009")
    ap.add_argument("--autopilot", "--spectator", dest="autopilot", action="store_true", help="spectator mode: hands-off simulation with auto-orbiting brain view")
    ap.add_argument("--audit-asymmetry", dest="audit_asymmetry", action="store_true", help="run bilateral asymmetry audit and exit")
    ap.add_argument("--mirror-weights", dest="mirror_weights", action="store_true", help="mirror-average synaptic weights (game rule: data modification)")
    ap.add_argument("--benchmark", action="store_true", help="run simulation throughput benchmark (1, 8, 16 flies), then a frame profile "
                    "of fixed rendered scenes (sim, physics, render, UI milliseconds, FPS, engine), and exit")
    ap.add_argument("--no-render-bench", dest="no_render_bench", action="store_true", help="with --benchmark: skip the rendered scenes")
    ap.add_argument("--threshold-sweep", dest="threshold_sweep", action="store_true",
                    help="re-run the validated behaviors at a range of minimum-synapse thresholds and exit")
    ap.add_argument("--thresholds", type=int, nargs="+", help="minimum-synapse thresholds for --threshold-sweep")
    ap.add_argument("--signflip-test", dest="signflip_test", action="store_true",
                    help="flip low-confidence neurotransmitter signs over N trials and report which behaviors survive")
    ap.add_argument("--flip-confidence", dest="flip_confidence", type=float,
                    help="confidence cutoff for --signflip-test (default 0.7)")
    ap.add_argument("--flip-share", dest="flip_share", type=float,
                    help="share of the candidates each trial flips (default 0.5)")
    ap.add_argument("--trials", type=int, help="randomized trials for --signflip-test (default 5)")
    ap.add_argument("--critical-path", dest="critical_path", metavar="TARGET",
                    help="rank cell types by what silencing each does to a behavior (looming_escape, sugar_feeding, "
                         "antenna_grooming_circuit, mb_conditioning) or an assay (tmaze, looming, sugar)")
    ap.add_argument("--top", type=int, help="how many candidate cell types --critical-path tries (default 25)")
    ap.add_argument("--types", nargs="+", help="test exactly these cell types instead of the shortlist")
    ap.add_argument("--resume", action="store_true", help="continue an interrupted --critical-path run in --out")
    ap.add_argument("--sensitivity", action="store_true",
                    help="3.0 day 4: vary each LIF parameter and re-run the validated behaviors with validation's own criteria "
                         "(headless; resumable with --resume; writes a heatmap, CSV and JSON into --out)")
    ap.add_argument("--sens-params", dest="sens_params", nargs="+", metavar="PARAM",
                    help="with --sensitivity: only these parameters (noise_std bias target_rate_hz ext_gain gain_adapt min_synapses)")
    ap.add_argument("--sens-tests", dest="sens_tests", nargs="+", metavar="TEST",
                    help="with --sensitivity: only these validated behaviors (default: every one validation reproduces)")
    ap.add_argument("--sens-values", dest="sens_values", nargs="+", metavar="PARAM=V1,V2",
                    help="with --sensitivity: replace a parameter's documented range, e.g. noise_std=0.03,0.07")
    ap.add_argument("--tournament", type=int, metavar="N",
                    help="3.0 day 4, headless: a single-elimination bracket of N (4, 8 or 16) flies per --seeds group of N "
                         "individuality seeds; reports whether personality predicts winning")
    ap.add_argument("--match-seconds", dest="match_seconds", type=float,
                    help="with --tournament: the length of a duel in game seconds (default 20)")
    ap.add_argument("--race-r4", action="store_true",
                    help="3.0 day 4 review, headless: the pre-registered race test R4 (individuality with the brain state held fixed: "
                         "8 races of 6, seeds 1000-1047, subtle and the off control; lab/racing.py)")
    ap.add_argument("--race", action="store_true",
                    help="3.0 day 4, headless: the race assay (flies race through their own brains; does individuality "
                         "predict the finishing order?)")
    ap.add_argument("--races", type=int, metavar="K", help="with --race: how many races (each --seeds group of --lanes flies)")
    ap.add_argument("--lanes", type=int, metavar="N", help="with --race: flies per race (default 6)")
    ap.add_argument("--rig", choices=("tethered", "ball", "buridan", "fourfield"),
                    help="3.0 day 5, headless: one fly (the first of --seeds) in a classic behavior rig, recorded in the Lab's format into --out "
                         "(tethered flight simulator, fly on a ball, Buridan's paradigm, four-field olfactory arena)")
    ap.add_argument("--rig-assay", dest="rig_assay", choices=("tethered", "ball", "buridan", "fourfield"),
                    help="3.0 day 5, headless: the rig's pre-registered assay over --seeds (default 1000-1009), default individuality and the "
                         "off control; JSON and CSV into --out")
    ap.add_argument("--minipaper", metavar="ID", help="3.0 day 5, headless: run a guided mini-paper (--seeds sets the flies; the default is "
                    "its quick run) and print your result next to what the paper found; ID 'list' lists them")
    ap.add_argument("--netsci", nargs="?", const="adult", choices=("adult", "larva", "both"), metavar="BRAIN",
                    help="3.0 day 4, headless: network science of the brain pack (degrees, reciprocity, motifs, rich club, "
                         "communities, regions); cached; CSV into --out")
    ap.add_argument("--sleep-deprivation", dest="sleep_deprivation", action="store_true",
                    help="3.0 day 4, headless: the sleep-deprivation assay (rebound sleep vs undisturbed controls, paired)")
    ap.add_argument("--flies", type=int, nargs="+", help="flies count list for benchmark (default: 1 8 16)")
    ap.add_argument("--seconds", type=float, help="duration per benchmark condition in seconds")
    ap.add_argument("--strict", action="store_true", help="exit 1 if validation differs from the expected results")
    ap.add_argument("--record-video", dest="record_video", nargs="?", const="default", metavar="PATH",
                    help="start recording a video at launch, to PATH or the screenshots folder (MP4/WebM with ffmpeg, else GIF)")
    ap.add_argument("--replay", metavar="FILE", help="with --headless: run a .ktfreplay again and compare its spikes")
    ap.add_argument("--selftest", action="store_true",
                    help="check this install (build, brain packs, backends, OpenGL, audio, folders, memory, a 10 s smoke "
                         "run) and exit: 0 all pass, 1 any FAIL, 3 warnings only; --out FILE.json also writes the JSON")
    ap.add_argument("--selftest-child", dest="selftest_child", nargs=2, metavar=("WHAT", "STEPS"), help=argparse.SUPPRESS)
    ap.add_argument("--playthrough", nargs="?", const="all", choices=("adult", "larva", "all"), metavar="BRAIN",
                    help="with --headless: the playthrough bot: every arena x tool x brain (adult, larva or all), report "
                         "into --out DIR")
    ap.add_argument("--playthrough-quick", dest="playthrough_quick", action="store_true",
                    help="with --playthrough: fewer combos (each tool once, two arenas), for CI on pull requests")
    ap.add_argument("--bugreport", action="store_true",
                    help="print exactly what Settings > Help > Report a bug would include, and write it to --out DIR; "
                         "nothing is ever sent")
    ap.add_argument("--record-replay", dest="record_replay", metavar="FILE",
                    help="with --protocol: record the protocol's first fly as a .ktfreplay")
    ap.add_argument("--bundle", metavar="ZIP",
                    help="with --protocol: also write the run as an experiment bundle (protocol, results, raw files, "
                         "metadata, RO-Crate) that --rerun-bundle can check")
    ap.add_argument("--rerun-bundle", dest="rerun_bundle", metavar="ZIP",
                    help="with --headless --out DIR: run a bundle's protocol again and report whether the results "
                         "match (bit-exact on CPU backends, statistical on GPU backends)")
    ap.add_argument("--share-decode", dest="share_decode", metavar="CODE",
                    help="print what a share code (or a .ktfshare file) contains and whether this version accepts it; "
                         "changes nothing")
    args, unknown = ap.parse_known_args(argv)
    if unknown:
        log.warning("ignoring unknown arguments: %s", " ".join(unknown))
    args.smoke_s = float(args.smoke[0]) if args.smoke else 0.0
    args.shot = args.smoke[1] if args.smoke and len(args.smoke) > 1 else None
    return args


def main(argv: list[str] | None = None) -> int:
    for name in ("stdout", "stderr"):                   # the windowed exe has neither
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w"))
    args = parse_args(argv)
    p = paths.get()
    crash.setup_logging(p.state_dir, args.verbose)
    log.info("Kick the Fly %s on %s", __version__, crash.os_description())
    for n in p.notes:
        log.info(n)
    if getattr(args, "selftest_child", None):
        from kickthefly.core import selftest
        return selftest.child_main(args.selftest_child[0], int(args.selftest_child[1]))
    if getattr(args, "selftest", False):
        from kickthefly.core import selftest
        return selftest.main(args)
    if getattr(args, "bugreport", False):
        from kickthefly.core import bugreport
        return bugreport.main(args)
    if getattr(args, "playthrough", None):
        args.headless = True
    if (args.headless or args.validate or args.protocol or getattr(args, "audit_asymmetry", False)
            or getattr(args, "benchmark", False) or getattr(args, "threshold_sweep", False)
            or getattr(args, "signflip_test", False) or getattr(args, "critical_path", None)
            or getattr(args, "sensitivity", False) or getattr(args, "tournament", None) or getattr(args, "race", False)
            or getattr(args, "netsci", None) or getattr(args, "sleep_deprivation", False)
            or getattr(args, "rig", None) or getattr(args, "rig_assay", None) or getattr(args, "minipaper", None)
            or getattr(args, "replay", None) or getattr(args, "record_replay", None)
            or getattr(args, "rerun_bundle", None) or getattr(args, "share_decode", None)):
        if getattr(args, "replay", None) and not args.headless:
            print("--replay runs headless (--headless --replay FILE): the windowed game doesn't play replays yet",
                  file=sys.stderr)
            return 2
        from kickthefly.lab import headless

        return headless.main(args)
    cfg = config.Config.load(p.config_file)
    from kickthefly.sim import morphology
    morphology.set_opt_in(bool(cfg["brain.neuron_shapes"]))   # 3.0 day 3 review: the neuPrint download is opt-in
    from kickthefly.core import i18n
    i18n.set_language(cfg["access.language"])
    # Disambiguate --backend: if it matches a sim backend, apply to brain.backend; if display backend, use for video
    sim_backend_choice = getattr(args, "sim_backend", None)
    display_backend_choice = None
    if args.backend:
        if args.backend in ("wayland", "x11"):
            display_backend_choice = args.backend
        elif args.backend in ("cpu", "numba", "torch-cuda", "torch-rocm", "torch-cpu", "gl"):
            sim_backend_choice = args.backend
        elif args.backend == "auto":
            sim_backend_choice = "auto"
    if sim_backend_choice:
        cfg.set("brain.backend", sim_backend_choice)
    if os.environ.get("KICK_THE_FLY_AUTO", "").lower() != "exact":           # 3.1.0 task 2: the game's 'auto' is the fastest working engine
        from kickthefly.sim.connectome import backends as _backends
        _backends.set_auto_policy("fastest")
    if getattr(args, "dtype", None):
        cfg.set("brain.dtype", args.dtype)
    args.display_backend = display_backend_choice
    args.sim_backend = sim_backend_choice
    if args.autopilot:
        cfg.set("brain.autopilot", True)
    if getattr(args, "brain", None):
        cfg.set("brain.brain", args.brain)
    if getattr(args, "individuality", None):
        cfg.set("brain.individuality", args.individuality)
    if getattr(args, "pet", False):
        cfg.set("brain.mode", "pet")
    if getattr(args, "arena", None):
        cfg.set("brain.arena", args.arena)
    if getattr(args, "mirror_weights", False):
        cfg.set("brain.mirror_weights", True)
    seed = args.seed if args.seed is not None else cfg["brain.seed"]
    crash.info["seed"] = str(seed)
    smoke, shot = args.smoke_s, args.shot
    platform_env.windows_dpi_aware()
    try:
        pygame.mixer.pre_init(Sound.RATE, -16, 1, 512)
    except Exception as e:                               # pygame built without mixer: the game plays silently
        log.warning("sound unavailable: %s", e)
    if not args.two_d:                                   # first person 3D by default; 2D if OpenGL 3.3 isn't there
        try:
            from kickthefly.game import kick3d
        except Exception as e:                           # e.g. moderngl missing in a source checkout
            log.warning("3D unavailable (%s); starting the 2D game", e)
        else:
            platform_env.init_video(args.display_backend, cfg["graphics.backend"])
            pygame.init()
            for attempt in range(2):
                try:
                    crash.info["mode"] = "3d"
                    return kick3d.run(smoke, shot, args.fullscreen or cfg["graphics.fullscreen"], seed=seed, cfg=cfg,
                                      flies=(args.flies[0] if args.flies else 1),
                                      record_video=getattr(args, "record_video", None))
                except kick3d.GLUnavailable as e:
                    if attempt == 0 and platform_env.reset_to_x11():
                        continue
                    log.warning("OpenGL 3.3 is not available on this PC (%s). Using the 2D game instead; "
                                "start with --2d to skip this check.", e)
                    pygame.display.quit()
                    break
    crash.info["mode"] = "2d"
    if not pygame.display.get_init():
        platform_env.init_video(args.display_backend, cfg["graphics.backend"])
    os.environ.setdefault("SDL_RENDER_SCALE_QUALITY", "linear")   # smooth when scaled, not blocky
    pygame.init()
    pygame.display.set_caption("Kick the Fly")
    # SCALED: the game always draws at 1280x760 and SDL scales that to the window or the whole screen, keeping the
    # aspect ratio (black bars if needed) and mapping the mouse back, so it can go fullscreen at any resolution.
    desk = pygame.display.get_desktop_sizes()[0] if pygame.display.get_desktop_sizes() else (W, H)
    fs = bool(args.fullscreen or cfg["graphics.fullscreen"] or desk[0] < W or desk[1] < H + 60)
    flags = pygame.SCALED | (pygame.FULLSCREEN if fs else pygame.RESIZABLE)
    screen = pygame.display.set_mode((W, H), flags)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("segoeui,consolas", 22)
    state: dict = {"stage": "starting"}
    state["seed"] = seed
    state["brain_type"] = playable_brain(cfg)   # not "brain": the loop below waits for that key
    state["individuality"] = cfg.get("brain.individuality", "subtle")
    state["mirror_weights"] = bool(cfg["brain.mirror_weights"])
    state["backend"] = getattr(args, "sim_backend", None) or cfg["brain.backend"]
    state["dtype"] = cfg["brain.dtype"]
    threading.Thread(target=load_brain, args=(state,), daemon=True).start()
    t0 = time.perf_counter()
    while "brain" not in state:
        for ev in pygame.event.get():
            if ev.type == pygame.KEYDOWN and (ev.key == pygame.K_F11 or
                                              (ev.key == pygame.K_RETURN and ev.mod & pygame.KMOD_ALT)):
                pygame.display.toggle_fullscreen()
                continue
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                return 0
        screen.fill(BG)
        msg = state.get("error") or f"{state['stage']}{'.' * (int((time.perf_counter() - t0) * 3) % 4)}"
        img = font.render(msg, True, RED if "error" in state else TEXT)
        screen.blit(img, img.get_rect(center=(W // 2, H // 2)))
        pygame.display.flip()
        clock.tick(30)

    brain = state["brain"]
    brain.start()
    game = Game(screen, brain, state["view"], state.get("graph"), state.get("weights"), cfg=cfg)
    if not smoke:
        game.show_first_run_notices()
    if cfg["brain.autopilot"]:
        game.big_view = True
    if getattr(args, "record_video", None):
        game.toggle_video_recording(None if args.record_video == "default" else args.record_video)
    running = True
    t_game = time.perf_counter()
    _profiler.bench_start()
    warm_done = False
    while running:
        real = time.perf_counter()
        if smoke and not warm_done and real - t_game > smoke / 2:
            warm_done = True
            _profiler.bench_warm_done()
        if smoke and real - t_game > smoke:
            try:
                from PIL import Image  # noqa: F401  (GIF saving works in this build)
                gif = "gif ok"
            except ImportError:
                gif = "no gif"
            status = f"smoke ok: {brain.n:,} neurons, {brain.steps_per_s:.0f} steps/s, sound {game.sound.ok}, {gif}"
            print(status)
            _profiler.bench_finish(game, status)
            if shot:                                      # optional screenshot path; the exe has no console
                save_image(screen, shot)
                with open(shot + ".txt", "w") as f:
                    f.write(status)
            break
        mouse = pygame.mouse.get_pos()
        ticks = game.clock.frame(real)
        crash.maybe_debug_crash(real - t_game)
        for ev in pygame.event.get():
            if ev.type == pygame.KEYDOWN and (ev.key == pygame.K_F11 or
                                              (ev.key == pygame.K_RETURN and ev.mod & pygame.KMOD_ALT)):
                game.toggle_fullscreen()
                continue
            running = game.handle(ev, game.clock.now) and running
        game.sync_time()
        with PROF.section("physics"):
            for dt in ticks:
                game.clock.now += dt
                game.update(game.clock.now, (min(mouse[0], PLAY_W - 5), mouse[1]))
        game.draw(game.clock.now, mouse)
        if ticks and game.frame % 4 == 0:            # rolling footage for G / the death GIF
            game.capture()
        if ticks and getattr(game, "timelapse_recording", False) and game.frame % 2 == 0:
            game.capture_timelapse_frame()
        if ticks and getattr(game, "video_recorder", None) and game.video_recorder.is_recording:
            game.capture_video_frame()
        with PROF.section("present"):
            pygame.display.flip()
        if PROF.on:
            PROF.sim_busy(sum(sl.brain.sim.busy_s for sl in game.flies))
            PROF.end_frame()
        clock.tick(cfg["graphics.fps_cap"])
    shutdown(game)
    return 0


def shutdown(game) -> None:
    if getattr(game, "video_recorder", None) and game.video_recorder.is_recording:
        game.video_recorder.stop()
    if getattr(game, "timelapse_recording", False):
        game.timelapse_recording = False
        game.save_timelapse()
    if getattr(game, "recording", None) is not None:
        game.stop_recording(wait=True)
    if game.challenge is not None:                   # puts back any memory a challenge borrowed before saving
        game.challenge.end()
    for slot in game.flies:
        slot.brain.stop()
        if slot.persist_memory and slot.brain.memory is not None:
            slot.brain.memory.save()
    if getattr(game, "pet", None) is not None:      # live hunger/sleep/mood since the last meal
        game.pet.save()
    if getattr(game, "x3", None) is not None and game.x3.progress is not None and game.x3.progress.dirty:
        game.x3.progress.save()                     # 3.0: the Neurodex is saved at each discovery; this is the backstop
    if game.cfg.dirty:
        game.cfg.save()
    pygame.quit()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()                 # Lab worker processes in the exe and AppImage start here
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        crash.handle_crash()
        raise
