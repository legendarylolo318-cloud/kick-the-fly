# Kick the Fly

A kick-the-buddy game where the buddy is a real fruit fly brain: the
**MaleCNS v1.0** connectome ([Google Research blog](https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/)),
all 166,700 neurons simulated live while you throw, swat, bomb, burn, dissolve, zap, freeze and feed it to a spider. Or reward it with sugar.

**It's first person:** walk around a 3D living room and use your tools on the fly up close. Walk into it and you kick it. Or take it outside: an **open field** with wind and sun, and an **orchard** where it flies to fruit and feeds. The original 2D version is still there with `--2d`.

**Play or Lab:** Play is the game, with challenges and scores. Lab mode adds research tools: a validation dashboard showing which published fly behaviors this simulation reproduces (and which it doesn't), repeated trials with statistics, data export and protocol files that also run headless.

![Kick the Fly in the open field: swatting and torching the fly while its live brain lights up on the right](docs/demo.gif)

*12 seconds in the open field, recorded with the in-game video recorder (Shift+R): a swat and the blowtorch fire its real touch and heat neurons, and the brain panel on the right shows all 166,700 neurons responding.*

## Download

### Windows

**[Download KickTheFly.exe](https://github.com/legendarylolo318-cloud/kick-the-fly/releases/latest/download/KickTheFly.exe)** (about 90 MB) and double-click it. You don't need to install anything, and the fly's whole brain is inside the exe.

- **Startup:** the first launch takes a few seconds while the exe unpacks.
- **Windows warning:** the exe isn't code-signed, so Windows SmartScreen may say "Windows protected your PC". Click **More info**, then **Run anyway**.
- **Your fly remembers:** training memory from earlier versions is still read from `Documents\Kick the Fly\memory`, including a Documents folder moved to OneDrive. Nothing is moved or deleted.
- **Fullscreen and sharp scaling:** press **F11**, or start it with `KickTheFly.exe --fullscreen`. The game is DPI aware, so 125% and 150% displays draw at full resolution instead of blurry.
- **Headless:** `start /wait KickTheFly.exe --headless --protocol smoke.yaml --out results` runs without a window (bundled protocols can be named without a folder; see Lab tools).
- **If it crashes:** it writes `KickTheFly-crash.txt` next to the exe and in `%LOCALAPPDATA%\Kick the Fly`.
- **Checksums:** each release has `SHA256SUMS`; `Get-FileHash KickTheFly.exe` should match.

### Linux

**[Download KickTheFly-x86_64.AppImage](https://github.com/legendarylolo318-cloud/kick-the-fly/releases/latest/download/KickTheFly-x86_64.AppImage)** (about 110 MB), `chmod +x` it, and run it. No install needed, and the fly's whole brain is inside it. It's built on Ubuntu 22.04, so it runs on most distros from then on.

- **Startup:** the first launch takes a few seconds while it unpacks.
- **Flatpak:** a manifest for building it yourself is in `packaging/flatpak/` (`flatpak-builder --user --install ...`, see its README). It isn't on Flathub. It keeps its files in the same folders as the AppImage.
- **Wayland and X11:** it uses native Wayland when `WAYLAND_DISPLAY` is set and falls back to X11/XWayland by itself if that fails. Force one with `--backend wayland` or `--backend x11`, or in Settings > Graphics (applies on restart). An `SDL_VIDEODRIVER` you set yourself always wins. Mouse look uses relative pointer mode on both.
- **GPU:** needs OpenGL 3.3 for the 3D room. Without it the game logs why and starts the 2D game (`--2d` skips the check). Software rendering (`LIBGL_ALWAYS_SOFTWARE=1`) works but is slow.
- **If it won't start (FUSE):** AppImages mount themselves with FUSE. Without FUSE (no `libfuse2`/`fusermount`, containers, some minimal distros) run `./KickTheFly-x86_64.AppImage --appimage-extract-and-run`, or set `APPIMAGE_EXTRACT_AND_RUN=1`. `sudo apt install libfuse2` fixes it on Debian/Ubuntu.
- **Headless over SSH:** `./KickTheFly-x86_64.AppImage --headless --protocol smoke.yaml` works with no `DISPLAY` or `WAYLAND_DISPLAY`.
- **Sound:** plays through PulseAudio (including PipeWire's PulseAudio server) or ALSA.
- **If it crashes:** it writes `KickTheFly-crash.txt` next to the AppImage and in `~/.local/state/kickthefly/`.

### From source

Any OS with Python 3.11+: see [Run from source](#run-from-source). That's also how you get the faster Numba and GPU
backends, the Python API and the Lab's optional NWB export.

## Contents

- [Download](#download)
- [What you can do](#what-you-can-do)
- [Controls](#controls) and [Settings](#settings)
- [Play: challenges and real-science cards](#play-challenges-and-real-science-cards)
- [Screenshots](#screenshots)
- [Lab mode](#lab-mode), [Validation](#validation), [Performance](#performance), [Python API](#python-api)
- [Run from source](#run-from-source)
- [File locations](#file-locations)
- [Time controls and save states](#time-controls-and-save-states)
- [What is the connectome and what is a game rule](#what-is-the-connectome-and-what-is-a-game-rule)
- [Repo layout](#repo-layout) and [Credits](#credits)

## What you can do

- **Hits fire real sensory neurons:**
  - head: head bristles and Johnston's organ
  - body: tactile neurons
  - legs: proprioceptive neurons
  - wings: wing sensory neurons
  - blowtorch: heat-sensing neurons
  - brake cleaner: smell and taste neurons (and it dissolves the fly)
  - freeze spray: cold-sensing neurons
  - zapper: every touch neuron plus a shock through its brain
  - spider bites: body and leg touch neurons
- **Its reactions come from its descending neurons:** running, kicking, walking, backing up and turning.
- **It can fly:** it takes off when its DNg02 wing-power neurons fire above normal, and flies away when its head-touch escape neurons fire.
- **Pain meter:** built from touch overload, heat and cold sensors, chemical senses and descending-neuron alarm. The **blowtorch** and **brake cleaner** max it out.
- **More pain neurons (P):** the wiring can't gain neurons, so the pain setting listens to more of the fly's real ones. **Normal** uses 9,080. **More** uses 11,392 and adds the rest of the body's sensory neurons. **Max** uses 13,238 and adds the ascending neurons that relay body signals to the brain. Higher settings also make each hit fire more of them.
- **Immortal mode (I):** it feels everything but can't die. It heals when you stop, and breaks out of spider silk.
- **Reward:** drop **sugar** and it walks over to eat. That lights up its PAM dopamine reward neurons and heals it. Its sugar-pathway taste neurons fire its proboscis motor neuron MN9, and when MN9 responds its proboscis comes out.
- **Alcohol (-):** drop a droplet of fermented fruit and the fly walks over and sips it. Drinking drives its real sweet taste pathway and its PAM dopamine reward neurons, the same ones sugar does, and its smell comes through the real fermentation glomeruli (DM1, DM2, DP1m). Getting drunk is a **game rule**: an inebriation level builds up with every sip and wears off over about 45 seconds, and while it lasts the game gives the fly tremors, a stumbling gait, wobbly flight and slower escape reflexes. No neuron in the simulation is actually intoxicated.
- **Death and autopsy:** it can die. The autopsy compares every brain region's last 2 s alive with its calm baseline, and shows pain on a timeline.
- **It sees you coming:** move a weapon at it fast and its real looming detectors (LPLC2 and LC4) fire its giant fiber escape neuron, so it dodges. Sneak up slowly and it won't notice.
- **Brain surgery (O):** silence or stimulate real neuron groups and watch what happens. Switch on the moonwalker neurons and it backs up; silence the giant fiber and it can't dodge.
- **Neuron inspector:** in the big brain view, click any neuron to see its type, how fast it's firing, and its strongest connections in the connectome.
- **1v1 duel (X):** the fly gets a blaster and can kill you, and every part of the fight runs through its brain:
  - it sees you through its real target-tracking neurons (LC10), which steer it toward you through its steering neurons (DNa02);
  - it shoots when its small-object detectors fire its DNp35 neurons;
  - landing a hit fires its reward dopamine neurons, so it learns to like hunting you;
  - hurting it fires its punishment dopamine neurons, so it learns to fear you, and it runs away and stops shooting.

  Silence its tracking neurons in brain surgery and it can't aim.
- **Multiple flies (N):** press N to spawn another fly, each running its own complete, independent connectome — 166,700 neurons apiece. Up to 16 flies on plain NumPy (the exe and AppImage), one per CPU core (16-32) with the optional Numba backend, or 32 on a GPU backend (see [Performance](docs/performance.md)); each needs about 350 MB of free memory. They notice each other for real: a fly closing in fast fires another's actual looming detectors (LPLC2/LC4) and makes it dodge, and bumping into each other fires real touch neurons. Press **F** to pick which fly the brain panel, surgery and training follow (for 5 s; otherwise they follow the fly nearest you). Only the original fly's mushroom-body learning is saved between sessions. Every fly is its own brain thread, so with many flies the brains can fall behind real time (see Performance).
- **Real training (T):** the fly learns with its actual mushroom body. Pair a smell with a shock or with sugar and dopamine weakens the real Kenyon cell to output neuron synapses for that smell, just like in real flies. The Training panel runs lab-style conditioning and graphs the learning curve. Memory is saved between flies and sessions (see File locations). Hurting the fly while it smells a tool trains it too.
- **Arenas (E):**
  - **fan:** wind that fires its wind-sensing neurons, which excite its antennal grooming command neurons
  - **flypaper:** it gets stuck and struggles
  - **pool:** it floats, gets wet wings, and can drown
  - **lamp:** it's drawn to the light and singes itself on the bulb
  - **thermo** (new in 2.9, 2D and 3D): the floor runs from cold (15 °C, left) to hot (35 °C, right). Its real cold- and hot-sensing antennal neurons fire more the further it is from the comfortable middle, and the extremes hurt it. It doesn't seek the middle on its own: the temperature is a **game rule** reaching real neurons, and the neurons it reaches are validated only as one-synapse activation.
  - **escaperoom:** multi-hazard gauntlet combining fan wind, flypaper strip, and hot lamp overhead; reach the sugar dish to stop the speedrun timer and generate a tamper-evident verification code (`KTF-<SEED>-<TIME>-<SIG>`).
  - **open field** (3D): 30 m by 30 m of grass, rocks and open sky. A steady wind drives its real wind-sensing antennal neurons and the sun drives its photoreceptors; wind direction and strength and the sun's position are Lab parameters. Wind also reaches its head-touch escape neurons, so in a breeze it keeps flying off, and outdoors an escape really goes somewhere: fly out of sight (26 m from you, or 15 m up) and it's **lost**. **J** calls it back.
  - **orchard** (3D): a grove of 24 fruit trees. The fly flies to a ripe fruit, lands and feeds, which drives the same real taste and PAM reward neurons sugar does and heals it. Each fruit holds a few feeds and shrinks and browns as it's eaten, then drops; it grows back after about 75 s, staggered, with a cap per tree. Some fruit are fermented and act like the alcohol tool. The fruit, the trees and the flying to them are **game rules**: the fly doesn't forage through its own circuitry. With several flies they end up competing for fruit, but only through the looming and touch neurons they already have; nothing about competing is scripted.

  - **day and night** (new in 2.9, outdoors, off by default: Settings > Brain > Day/night cycle): the sun circles, night falls, and daylight drives its photoreceptors and its morning clock neurons (l-LNv, s-LNv). The cycle and that direct link are game rules. There's a **SLEEP** readout on its dorsal fan-shaped body (FB6/FB7), but in testing a whole day never moved those neurons enough, so it only sleeps when you stimulate them in brain surgery.

  Open field and orchard need the 3D game; the 2D game stays indoors and says so. The arena you pick is saved in `config.toml` (Settings > Brain > Arena, or **E**), in save states and in every export's metadata.
- **Save and share:** **F12** (3D) or **S** (2D) saves a screenshot and **G** saves a GIF of the last 6 seconds. **Shift+R** starts a **video** of any length and Shift+R again stops it: an MP4 if [ffmpeg](https://ffmpeg.org/) is installed (on your PATH), otherwise a GIF (smaller, up to 60 s). It plays back at real speed however fast the game draws, and a red badge shows how long you've been recording. `--record-video [PATH]` starts one at launch. **L** toggles time-lapse recording (2x, 5x, 10x or 20x speed-up, to MP4 or GIF). The autopsy can save a GIF of the death. Everything goes to your screenshots folder (see File locations).
- **Slow motion and save states:** pause time, slow everything to 0.1x, step it 1/60 s at a time, and save or load the whole simulation (see Time controls).
- **Live brain view:** a front view of the brain built from the neurons' real cell-body positions, shaded by depth. Almost every neuron is drawn as an estimated fiber toward its synaptic partners; ten of them (two each of the giant fiber DNp01, the DNa02 steering neurons, MBON01, MBON14 and a Kenyon cell type) are drawn from their **real reconstructed shapes**, sampled from their EM skeletons on Janelia's neuPrint and downloaded once, then cached (offline and uncached they fall back to estimated fibers, and the big view says which). Pain-sensing neurons glow orange and everything else glows cyan when firing (blue/yellow and high-contrast palettes in Settings > Accessibility). Press **B** for the big view.
- **Courtship song (new in 2.9):** switch on its song command neuron pIP10 in brain surgery and its ps1 wing motor neurons fire and it buzzes a pulse song. pIP10 -> ps1 is validated; the buzz itself is synthesized (a game rule, Settings > Brain > Courtship song buzz). Hard hits on the wings can set it off too.
- **Aggression (new in 2.9):** with several flies, stimulate P1 (the male courtship and aggression cluster) in brain surgery and its aggression neurons fire and it lunges at the nearest fly. The neurons are real (AVLP727m, "TK-FruM", and the pC1 cluster); the lunge is a game rule (Settings > Brain > Aggression lunges). Flies bumping into each other don't fire them hard enough on their own.
- **Brain search and path tracer (new in 2.9):** in the big brain view, search any neuron by type, instance or body ID, then trace the strongest paths (up to 3 synapses) between two neurons and watch live spikes run along them.
- **Gamepad (new in 2.9, 3D):** sticks walk and look, the right trigger uses the tool, bumpers or a tool wheel (hold Y) pick tools. Rebind everything in Settings > Controls.
- **cVA pheromone tool (new in 2.10):** puffs the male pheromone 11-cis-vaccenyl acetate (cVA), driving the real Or67d olfactory receptor neurons (ORN_DA1) to the DA1 glomerulus and downstream lateral horn/aSP targets (LH008m/aSP-f). In multi-fly play, cVA drives the fly's aggression circuitry through the connectome; any direct shortcut is tagged as a game rule.
- **Decoy female (new in 2.10):** spawns a stationary or slowly walking female-shaped decoy (a game rule body). Foreleg contact drives the male's real foreleg pheromone GRNs (LgLG5..8, putative ppk23/ppk25) -> vAB3/PPN1 ascending neurons -> P1 courtship hub -> pIP10 song command neuron -> ps1 wing motor neurons. Male courtship reactions (orienting, wing extension, pulse song buzz) read from these real neurons.
- **Odor plume in the open field (new in 2.10):** open-field odor plumes model intermittent filaments advected downwind by the arena wind, driving real ORNs (e.g. DM1). Plume navigation (surge upwind on odor encounter, cast crosswind on loss) is tagged as a game rule, evaluated in the plume tracking assay.
- **Mushroom body learning extensions (new in 2.10):**
  - **Extinction:** unreinforced exposure to a previously shock-paired odor triggers depotentiation of learned KC->MBON synapses (Felsenberg et al. 2018), reducing conditioned avoidance. The unreinforced plasticity rule is a game rule.
  - **Second-order conditioning:** pairing odor A with shock, then pairing odor B with odor A without shock, transfers learned avoidance to odor B via fear-driven PPL1 dopaminergic reinforcement.


## Controls

Every key below can be rebound in Settings > Controls (a key that's already taken swaps with that action). Esc and the tool keys (1-9, 0, - and =) are fixed.

| key | what it does |
|---|---|
| Esc | close a panel, or open the pause menu: Resume, Challenges (Play) or Lab tools (Lab), Settings, Save State, Load State, Mode, Quit |
| WASD | walk (Shift sprint, Ctrl or C crouch); walk into the fly to kick it |
| Mouse | look around; left click uses the tool in your hand |
| 1-9, 0, -, =, C, D or mouse wheel | pick a tool: hand, flick, swatter, bomb, blowtorch, brake cleaner, zapper, freeze spray, spider, sugar, alcohol, laser, cVA pheromone, decoy female |
| Tab | free the mouse to click the brain panel and menus (click the room to look again) |
| B | big live brain view; click a neuron to inspect it, search neurons, trace paths between two |
| O | brain surgery |
| T | training: teach it to fear or like a smell (saved between sessions) |
| X | 1v1 duel: the fly gets a blaster and can kill you (R respawns you) |
| E | arena: room, fan, flypaper, pool, lamp, escape room, open field, orchard (these two 3D only), thermo |
| J | outdoors: call back a fly that flew out of sight |
| P / I | pain neurons / immortal mode |
| K | brain stethoscope (spike sonification clicks in big brain view / body parts) |
| L | time-lapse record (2x-20x speedup to MP4/GIF; toggle on/off) |
| Shift+R | start or stop a video (MP4 with ffmpeg, else GIF) |
| Y | autopilot / spectator mode (hands-off orbit camera) |
| F10 | photo mode / free camera with depth of field |
| M | mute |
| F12 (S in 2D) / G | save a screenshot / a GIF of the last 6 seconds |
| V | brain panel: solid, see-through, faint, hidden (hidden gives the room the whole screen) |
| U | menu size: crisp (sharp whole-pixel scaling, the default) or large |
| F11 or Alt+Enter | fullscreen; the game fills any screen with no black bars |
| N | spawn another fly, each with its own independent brain (up to 16 on NumPy; one per CPU core, 16-32, on Numba; 32 on a GPU backend, 64 with `KICK_THE_FLY_EXPANDED_SWARM=1`) |
| F | pick which fly the brain panel, surgery and training follow (for 5 s, then back to the nearest) |
| R | reset to a single fresh fly |
| Z | pause or resume time |
| [ / ] | slower / faster: 0.1x, 0.25x, 0.5x, 1x |
| . | single step while paused (1/60 s of the room and the matching brain steps) |
| H | controls help |

**Gamepad (3D):** left stick walks, right stick looks, right trigger uses the tool (pull it to start looking around), LB / RB pick the previous / next tool, hold Y for the tool wheel (point with the right stick, release to pick), B crouches, left-stick click sprints, Start opens the menu, Back the big brain view. Button names are the pad's own labels (on a Switch Pro: ZR uses, L / R pick tools, + is Start and − is Back). Pads SDL knows (Xbox, PlayStation, Switch Pro and many more) work without setup; every button and stick can still be rebound in Settings > Controls (click a binding, then press the button or push the stick). Keyboard and mouse work as always alongside it. The 2D game has no gamepad support.

Command line: `--2d`, `--fullscreen`, `--backend NAME` (a simulation backend: `auto`, `cpu`, `numba`, `gl`, `torch-cpu`, `torch-cuda`, `torch-rocm`; or, on Linux, the display backend `wayland` or `x11` as before), `--sim-backend NAME` (the simulation backend only, same choices), `--dtype float32|float64`, `--record-video [PATH]`, `--seed N`, `--arena NAME` (room, fan, flypaper, pool, lamp, escaperoom, thermo, field, orchard), `--flies N` (start with N flies), `--replay FILE` (windowed deterministic replay), and for headless runs `--headless`, `--replay FILE --out DIR` (re-exports recording), `--validate`, `--protocol FILE`, `--nwb`, `--out PATH`, `--workers N`, `--seeds 1000-1009`, `--strict`, `--threshold-sweep`, `--signflip-test`, `--critical-path TARGET`, `--benchmark`.

## Settings

Esc > Settings. Changes apply right away and are saved to `config.toml`; hover any setting for a plain explanation.

- **Graphics:** fullscreen, resolution scale (3D drawn smaller and stretched, for weak GPUs), FPS cap, VSync (restart), display backend (Linux only, restart), brain panel style, menu size, UI scale.
- **Audio:** master, wing buzz and sound effects volume, brain stethoscope (spike sonification clicks, hotkey K), mute.
- **Brain:** Play/Lab mode, arena, pain neurons, immortal, sim speed, random seed (applies on R), real vs rule tags, real-science popups (default OFF; see below), courtship song buzz, aggression lunges, day/night cycle (outdoors), compute backend (see [Optional: faster simulation](#optional-faster-simulation-with-numba-pytorch-or-opengl)) and state precision (float32, the default, or float64). Brain settings are tagged **Connectome** (changes how the simulation runs) or **Game rule** (a rule the game adds on top).
- **Controls:** mouse sensitivity, invert Y, field of view, key bindings, and the gamepad: on/off, look speed, dead zone, invert Y and every button binding.
- **Accessibility:** language (Settings > Accessibility; English, German, or custom community translations via [docs/translating.md](docs/translating.md)), colorblind-safe brain view colors (blue/yellow) and a high-contrast palette, reduced flashing (no screen shake, flashes, sparkles, scanning band or blinking), larger text.

## Play: challenges and real-science cards

**Play** (the default) is the game plus **challenges** in the pause menu, each built on a real experiment or neural readout:

- **Teach it to pick the right door:** pick which of two smelly doors zaps. The fly is trained on its real mushroom body, then chooses a door 10 times. Score: right choices. The practice memory is put back afterwards, so it never changes how the fly treats your tools.
- **How close can you sneak?:** creep up on the fly. When its giant fiber fires it dodges. Score: how close you got, in fly lengths.
- **Find its sweet tooth:** offer sugar at different strengths and find the weakest one its proboscis motor neuron still responds to, in 8 tries.
- **Mystery defect (Reverse brain surgery):** one circuit is turned off at random (curated, unambiguous circuits). Test the fly with tools, request hints, and deduce what is missing without neuroscience jargon.
- **Predict the move (Motor readouts):** test your reflexes predicting motor readouts from real descending neuron spike surges (jump, run, kick, back up, take off) before the fly moves.

In Play mode a short **"Real flies do this too"** card appears the first time the fly does something that passed this game's validation (dodging, reaching for sugar with its proboscis, its antennal grooming neurons firing in the fan's wind, avoiding a smell it learned to fear, its song motor neurons firing). Behaviors that failed validation never get one. **Real-science cards default to OFF** in Play and Lab (enable them in Settings > Brain > Real-science popups; hover text: "Show a short card when the fly does something real flies were shown to do."). Existing configurations migrate to OFF once upon upgrade, preserving any subsequent user choices.

## Screenshots

All from the current build at 1280x760, made by `tools/make_screenshots.py` (see CONTRIBUTING.md to remake them).

**The room.** First person, with the fly's live brain on the right.

![first person in the 3D room, the fly on the rug and its brain panel on the right](docs/room3d.png)

**Swatting.** Its touch neurons fire, and the head-, body- and leg-touch descending neurons that drive jumping, running and kicking light up.

![swatting the fly in first person; touch neurons and descending neurons light up in the brain panel](docs/swat3d.png)

**Flypaper.** Stuck, it struggles through its own body- and leg-touch neurons and their descending neurons.

![the fly stuck on flypaper, leg-touch and body-touch descending neurons active](docs/flypaper3d.png)

**The open field.** Grass, rocks and sky; the wind drives its real antennal wind neurons, and escapes carry it away.

![the fly flying over the open field](docs/field.png)

**The see-through brain panel (V),** here in the open field: the world shows through the brain.

![the see-through brain panel over the open field sky](docs/see-through.png)

**The orchard,** from inside a fruit tree's crown: one of five flies feeding on a fruit. Feeding drives the same taste and PAM reward neurons as the sugar tool, so its REWARD meter reads happy.

![a fly feeding on a fruit inside an orchard tree's crown](docs/orchard.png)

**The blowtorch in the big brain view (B).** Every touch and heat neuron is pinned and the brain lights up.

![big brain view lit up while the fly is torched](docs/brain.png)

**A spider** drops, bites twice and wraps the fly in silk (immortal here, so it breaks free after five bites).

![a spider wrapping the fly in silk](docs/spider.png)

**1v1 duel (X).** The fly aims with its LC10 -> DNa02 steering pathway and fires when its DNp35 object neurons do. Here it has hit you 6 times out of 7 shots; every hit fires its reward dopamine neurons, so its REWARD meter reads bliss and its mushroom body now likes you 0.49.

![the fly shooting at you in the 1v1 duel, its reward meter at bliss](docs/duel.png)

**Training (T).** Ten shock pairings with the swatter's smell weaken its real Kenyon cell -> MBON synapses; the curve is its fear after each trial.

![the training panel with a fear learning curve](docs/training.png)

**Neuron inspector.** Click a neuron in the big view for its type, firing, transmitter and strongest connections. The giant fiber shown here is one of the neurons drawn from its real neuPrint skeleton.

![the neuron inspector showing the giant fiber DNp01](docs/inspect.png)

**Brain surgery (O).** Switching the moonwalker neurons (MDN) on makes it back up.

![the brain surgery panel with the moonwalker neurons switched on](docs/surgery.png)

**The lamp.** It's drawn to the light: here it flies up to the bulb (touching the bulb singes it).

![the fly flying up to the lamp's bulb](docs/lamp.png)

**The thermo arena (new in 2.9).** Cold on the left, hot on the right: the warm floor drives its hot-sensing antennal neurons.

![the thermo arena in 3D, the fly on the warm side of a blue-to-red floor](docs/thermo.png)

**The path tracer (new in 2.9, B).** The strongest paths from a looming detector to the giant fiber, with live spikes running along them.

![the big brain view tracing paths from LPLC2 to DNp01](docs/paths.png)

**The escape room.** Fan, flypaper and a hot lamp stand between the fly and the sugar dish.

![the escape room arena](docs/escaperoom.png)

**Autopsy.** Every brain region's last 2 s alive against its calm baseline, and pain on a timeline.

![the brain autopsy after the fly died under the blowtorch](docs/autopsy.png)

**Settings (Esc > Settings), Brain tab.** Each setting is tagged Connectome (changes the simulation) or Game rule, including the compute backend and state precision.

![the settings menu, Brain tab](docs/settings.png)

**Lab mode.** The research tools, with the simulation engine that's running shown top right.

![the Lab tools menu](docs/lab.png)

**Lab > Validation.** Which published fly behaviors this simulation reproduces, with the numbers and criteria, and which it doesn't.

![the validation dashboard with PASS and FAIL results](docs/validation.png)

**The optogenetics laser (Lab, key =).** Aim it and it drives, or silences, a chosen cell type; here it drives the giant fiber DNp01.

![the optogenetics laser driving the giant fiber](docs/laser.png)

## Lab mode

**Lab** (Esc > Mode, or Settings > Brain) replaces the challenges with research tools: the validation dashboard,
assays over many flies with same-seed controls and statistics, psychometric sweeps, the optogenetics laser, recording
and NWB export, the critical path finder, connectome robustness tests (synapse threshold sweeps, transmitter sign
flips, inhibition block), neural clamp, diff mode, hemifield lesions, classroom lectures, and YAML protocols that also
run headless (`--headless --protocol FILE`). Its Parameters page holds the model's parameters and every game-rule
threshold. All of it, with examples: **[docs/lab.md](docs/lab.md)**.

## Validation

Lab > Validation asks whether this simulation reproduces published fly results, on held-out seeds with pass criteria
fixed before the run (at least 1.5x the calm rate and above a matched control, one-sided Wilcoxon p < 0.01, n = 10).
Results of this release:

| passes | fails |
|---|---|
| looming detectors -> giant fiber (x11.81 vs x0.80) | MDN -> leg motor neurons, i.e. backward walking (x0.96 vs x0.96) |
| sugar taste neurons -> proboscis motor neuron MN9 (x2.11 vs x1.25) | aDN -> front-leg motor neurons (x1.13, too weak) |
| antennal touch -> grooming command neurons aDN (x4.87 vs x0.85) | E-PG head-direction bump, from a wedge or from wind |
| T-maze odor + shock conditioning (PI 1.00 vs -0.03) | P1 -> ps1 wing motor neurons (x1.43, too weak) |
| song neuron pIP10 -> ps1 wing motor neurons (x1.93 vs x0.94) | grooming hierarchy, head over abdomen (Seeds et al. 2014) |
| optomotor: rightward motion -> right steering DNs (x2.72 vs x0.80), through a game-rule motion stage | foreleg pheromone GRNs -> P1 cluster (x1.20, too weak) |
| Or67d cVA ORNs -> DA1 projection neurons (x2.58 vs x0.80) | MB extinction depotentiation (PI 0.90 vs <= 0.65 threshold) |
| DA1 PNs -> lateral horn / aSP targets (x2.22 vs x0.96) | |
| MB second-order conditioning (PI 1.00 vs -0.10) | |
| one-synapse activation (not avoidance): bitter -> DNg28, CO2 -> V PNs, hot -> VP2 PNs, cold -> VP3 PNs | |

Only passing behaviors get "Real flies do this too" cards. The full table with SDs, controls, citations and what each
failure means: **[docs/validation.md](docs/validation.md)**.

## Performance

A brain keeps real time (200 steps/s) on any backend with one fly. With many flies, Numba scales best on the CPU
(16 brains in real time on 24 cores) and PyTorch GPU backends run them batched (32 brains at 15x real time in 2.8).
The OpenGL `gl` backend runs on any vendor's GPU but is slower than NumPy here, so `auto` doesn't pick it; since 2.9
its learning uploads only the synapses that changed (68 KB instead of 41 MB per update, issue #2). Tables and
methods: **[docs/performance.md](docs/performance.md)**.

## Python API

From a source checkout, the brain is a library: the same headless machinery the Lab and protocols use.

```python
from kickthefly import Fly

fly = Fly(seed=1000)
rec = fly.record({"giant fiber": "dnp01"})
fly.step(2.0)
fly.drive("loom")           # LPLC2 + LC4, like optogenetic activation
fly.step(2.0)
print(rec.rates(0, 2), rec.rates(2, 4))
fly.export("results/looming")
```

`drive()`, `silence()`, `stimulate()`, `poke()`, `step()`, `record()` and `export()` (CSV, npz, NWB):
**[docs/api.md](docs/api.md)**, with a worked notebook in [docs/api_example.ipynb](docs/api_example.ipynb).

## Run from source

Needs Python 3.11, a GPU with OpenGL 3.3 for 3D, and about 1.5 GB of disk for the connectome.

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m kickthefly.sim.connectome.loader build   # downloads the connectome (~1.1 GB) and builds data/graph.pkl
.venv\Scripts\python kick_the_fly.py                # the first run packs data/kick_brain.npz (~30 s)
```

On Linux/macOS, use `python3` and forward slashes instead:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m kickthefly.sim.connectome.loader build   # downloads the connectome (~1.1 GB) and builds data/graph.pkl
.venv/bin/python kick_the_fly.py                    # the first run packs data/kick_brain.npz (~30 s)
```

### Optional: faster simulation with Numba, PyTorch or OpenGL

The brain simulation runs on plain NumPy by default. Optional backends can run it instead; install one and the
game uses it by itself (`auto`), or pick one in Settings > Brain > Compute backend or with `--backend NAME`:

| backend | install | what it does |
|---|---|---|
| `numba` | `pip install numba` | JIT-compiled CPU kernels that release Python's interpreter lock, so several flies' brains run in parallel ([numbers](docs/performance.md)) |
| `torch-cuda` | PyTorch with CUDA, from the selector on [pytorch.org](https://pytorch.org/get-started/locally/) | NVIDIA GPU |
| `torch-rocm` | PyTorch with ROCm (Linux), from the same selector | AMD GPU |
| `torch-cpu` | any PyTorch | PyTorch on the CPU; mainly for checking the torch code path |
| `gl` | nothing extra (ModernGL is already a requirement); needs OpenGL 4.3 | OpenGL compute shaders on any vendor's GPU (AMD, NVIDIA, Intel). Only when you pick it: it is slower than NumPy on the machines tested and doesn't run many flies in parallel ([numbers](docs/performance.md)) |

`auto` picks a PyTorch GPU, then Numba, then NumPy (never `gl`). A backend that can't start (library missing, no GPU visible to that
PyTorch build) falls back to NumPy and logs why; the backend that actually ran is what the Lab header, benchmarks,
validation results, exports, save states and crash reports record. Numba and `torch-cpu` give **exactly** the same
spikes as NumPy, so every result in this README is the same on them; GPU backends (`torch-cuda`, `torch-rocm`, `gl`) agree statistically but not spike for
spike (see [Deterministic runs](#time-controls-and-save-states)). The number of flies you can spawn depends on the
backend (see Controls: N).

**The exe and the AppImage include neither.** They always run the NumPy backend and don't pick up a Numba or PyTorch
you've installed on your system (a frozen app can't safely load another Python's packages); choosing another backend
there falls back to NumPy with a note in the log. For Numba or a GPU, run from source.

Tests (the validation suite takes a few minutes; `-m "not validation"` skips it):

```bash
.venv/bin/pip install pytest
.venv/bin/python -m pytest
```

To build the exe yourself (after one run from source, so the brain pack exists; add `data/validation_results.json` from `--validate` for the real-science popups):

```powershell
powershell -ExecutionPolicy Bypass -File build_exe.ps1
```

To build the AppImage yourself (same prerequisite):

```bash
./build_appimage.sh
```

Releases are built by `.github/workflows/release.yml` on a tag push: the brain pack is built from the public connectome, all tests (including validation) run on Ubuntu 22.04 and Windows, the AppImage is built on Ubuntu 22.04 and the exe on Windows, and the release is published with `SHA256SUMS` only if everything passes.

## File locations

| | Windows | Linux |
|---|---|---|
| settings (`config.toml`) | `%APPDATA%\Kick the Fly` | `$XDG_CONFIG_HOME/kickthefly` (`~/.config/kickthefly`) |
| training memory | `Documents\Kick the Fly\memory` | `$XDG_DATA_HOME/kickthefly/memory` (`~/.local/share/kickthefly/memory`) |
| save states | `Documents\Kick the Fly\saves` | `~/.local/share/kickthefly/saves` |
| exports, scores, validation runs | `Documents\Kick the Fly` | `~/.local/share/kickthefly` |
| your protocol files | `Documents\Kick the Fly\protocols` | `~/.local/share/kickthefly/protocols` |
| screenshots, GIFs and videos | `Pictures\Kick the Fly` | `<xdg-user-dir PICTURES>/Kick the Fly` (`~/Pictures/Kick the Fly`) |
| neuPrint skeleton cache | `Documents\Kick the Fly\skeletons` | `~/.local/share/kickthefly/skeletons` (from source: `data/skeletons/`) |
| crash reports and log | `%LOCALAPPDATA%\Kick the Fly` | `$XDG_STATE_HOME/kickthefly` (`~/.local/state/kickthefly`) |

Documents and Pictures on Windows come from the Known Folders API, so redirected and OneDrive folders work. On Linux, versions before 2.6 used `~/Documents/Kick the Fly/memory` and `~/Pictures/Kick the Fly`; on first launch the memory and any screenshots are copied to the new locations and the originals are left alone. A broken or missing `config.toml` falls back to default settings with a warning (a broken one is kept as `config.toml.bad`). `KICK_THE_FLY_HOME=/some/folder` keeps everything in one folder (portable use, tests).

## Time controls and save states

- **Pause (Z), slow motion ([ and ]) and single step (.):** the room and every brain slow down together, so spikes and the reactions they cause stay lined up. An on-screen badge shows the state. You can still look and walk around at full speed.
- **Save State / Load State** (pause menu) saves the whole simulation: every neuron's membrane potential and refractory state, synaptic gain, the random generators, the learned Kenyon cell to MBON weights, surgery, Lab parameters, the arena, every fly's body and timers, sugar piles, the seed and (3D) you. The `.ktfsave` format is versioned and platform independent, so a save made on Linux loads on Windows and the other way round. Saves from a newer version, from the other (2D/3D) game or from a different brain pack are refused with a reason. Things in flight (bombs, sprays, the spider) aren't saved.
- **Deterministic replay files (.ktfreplay, new in 2.10):** record the initial seed, backend, dtype, brain pack checksum, arena, settings, and step-exact input events (tool actions, tool selection, brain surgery, laser, spawns, and time controls). Launch replaying in windowed mode with `--replay FILE` or headless with `--headless --replay FILE --out DIR` (which re-exports recording data), or select "Replay" from the pause menu. When played back on NumPy, Numba, or torch-cpu backends, replays reproduce every spike bit-for-bit across all 166,700 neurons; on GPU backends reproduction is statistical. Replays generated with a mismatched brain pack checksum or incompatible version are refused with an informative error message.
- **Deterministic runs:** with the same seed and the same inputs a lockstep run (headless, protocols, validation, the tests) replays spike for spike. That holds across the NumPy, Numba and PyTorch-CPU backends too: they're bit-exact with each other (`tests/test_backends.py` compares every spike over 1000 steps in float32 and float64, and the full validation suite gives identical numbers on all three). GPU backends (`torch-cuda`, `torch-rocm`) are not bit-exact: a GPU may add up a neuron's inputs in a different order, the last bit of a float32 sum differs, and because the network is chaotic, individual spikes then diverge within a few hundred steps. Their tolerance is statistical: brain-wide firing within 2% of NumPy's and per-population rates correlated at r > 0.95 over 5 s. Results record the backend they ran on. The live game runs each brain on its own real-time thread, so play itself isn't bit-for-bit repeatable.

## What is the connectome and what is a game rule

**Connectome**
- The spiking model (leaky integrate-and-fire over 10.5M signed synapses) and all the neuron firing.
- Which sensory neurons each hit drives.
- The descending neurons read out for reactions. The jump, run and kick groups are the DN types that responded most to head, body and leg touch when the sim was probed.
- Take-off and flight speed come from DNg02, the wing-power descending neurons.
- In the 1v1 duel: aiming comes from the steering neurons DNa02 and DNa01 (right minus left), shooting from DNp35 and DNpe052, and walking toward you from DNp09. The pathways from target-tracking LC10 to DNa02 and from the small-object detectors LC11/18/21/26 to DNp35 are the connectome's own wiring. In testing, driving LC10 on one side took that side's DNa02 from about 1 to about 19 spikes/s. Whether it fights or flees is read from its mushroom body synapses for your smell.
- Dodging: the looming detectors LPLC2 and LC4 exciting the giant fiber DNp01 is the connectome's own wiring (validated: x11.8 vs x0.8 for a control).
- The wind, humidity and light neurons each arena fires, and the Kenyon cell patterns each tool's scent produces.
- The REWARD meter reads the PAM dopaminergic neurons.
- Drinking alcohol drives the same real pathways sugar does (the sugar-pathway taste neurons and the PAM reward neurons), and the droplet's smell drives the real olfactory neurons of the fermentation glomeruli DM1, DM2 and DP1m.
- Antennal wind excites the antennal grooming command neurons aDN1/aDN2 (validated); the GROOM reaction reads them.
- Sugar reaching the proboscis motor neuron MN9 (validated); the PROBOSCIS reaction reads MN9 during an eating bout. Which taste neurons count as sugar-pathway ones is chosen from the connectome's wiring to the annotated sugar and bitter SEL neurons.
- Everything inside the assays and protocols: how drive spreads, which neurons respond, and what silencing a group does.
- **Global inhibition block (Picrotoxin):** Scaling down inhibitory synapses unmasks recurrent excitation, driving brain-wide firing rate from 6.7 Hz calm mean up to 33.8 Hz mean (one seed, 1 s) as an emergent property of connectome recurrence.
- **Hemifield visual lesions:** Unilateral visual silencing (LC10, LPLC2, LC4, LPTC, VS, HS) causes lateralized behavioral failure: intact escapes for contralateral looming (10.9x GF drive) vs complete failure for ipsilateral looming (1.02x drive), biased spontaneous steering (-1.8 Hz vs +0.10 Hz baseline), and loss of 1v1 duel aim when the opponent is in the blind hemifield (turn differential drops to 0.85 Hz, below steering deadzone).
- **Connectome robustness sweeps:** Dropping connections below 10 synapses (73.6% of connections) preserves all four validated behaviors. Sign flips of low-confidence predictions break sugar -> MN9 in 3 of 3 trials while looming -> GF and JO -> aDN survive. Looming critical path depends primarily on LC4 (-50%) and LPLC2 (-43%).
- **Outdoor senses:** the open field's wind drives the real JO-C/E wind neurons of each antenna and the sun drives the photoreceptors, as the fan and lamp arenas do. In wind, JO also drives the head-touch escape DNs, so the fly flies off repeatedly; that is the wiring, not a scripted behavior. Take-off still reads from DNg02 and escape from the head-touch DNs outdoors, with no ceiling (tested).
- **Orchard feeding:** landing on a fruit drives the sugar-pathway taste neurons and the PAM reward neurons exactly as the sugar tool does (fermented fruit as the alcohol tool does). The MN9 response to it is the validated sugar -> MN9 pathway.
- **Several flies in the orchard** notice each other only through the looming detectors and touch neurons they always had.
- **Neural clamp:** Isolates structural wiring perturbations by forcing identical reference spike trains onto target neurons across different connectome variants.
- **Courtship song (2.9):** pIP10 driving the ps1 wing motor neurons (validated, through VNC interneurons) and what SONG reads: ps1's firing.
- **Aggression (2.9):** the aggression neurons LUNGE reads (AVLP727m and the pC1 cluster, P1 included), and how flies notice each other (looming and touch only).
- **Optomotor (2.9):** everything after T4/T5: rightward motion excites the right steering neurons DNa01/DNa02 (validated).
- **Thermo arena and day/night (2.9):** which neurons the temperature and the daylight reach (the hot and cold antennal neurons TRN_VP2/TRN_VP3, the photoreceptors, the morning clock neurons l-LNv/s-LNv) and everything downstream, including the dorsal fan-shaped body the SLEEP readout watches. The hot, cold and CO2 sensory-to-projection-neuron steps are validated as activation only; HEAT, COLD and CO2 in the log read those projection neurons.
- **Search and path tracer (2.9):** the paths are the connectome's own synapses, ranked by their weights.
- **cVA pheromone pathway (2.10):** Or67d ORNs (`ORN_DA1`) driving DA1 PNs (`DA1_lPN`, `DA1_vPN`, `M_lvPNm43`, `M_lvPNm45`) and downstream lateral horn/aSP targets (`LH008m` / aSP-f, `LHAV4a4`, `LHAV4c1`).
- **Foreleg contact courtship circuitry (2.10):** Foreleg contact activates real foreleg pheromone GRNs (`LgLG5..8`, putative ppk23/ppk25) which synapse onto ascending projection neurons `vAB3` (`AN09B017e/f/g`) and `PPN1` (`AN05B102a`), relaying drive to central courtship hub P1 and descending song command neuron pIP10.
- **Second-order conditioning (2.10):** Fear acquired by CS1 (odor A) drives PPL1 dopaminergic neurons during unreinforced CS2-CS1 pairing, reinforcing learned avoidance to CS2 via connectome wiring.

**Game rules**
- Which move each neuron group triggers, and the thresholds (all adjustable in Lab > Parameters).
- The ragdoll physics and standing back up.
- Jump direction, stun and damage.
- The pain index. The adult connectome has no neurons annotated as nociceptors, so pain is an estimate built from real signals, not a measurement of what the fly feels.
- Death. A sim can't die on its own, so on death its tonic drive is switched off and activity fades out.
- Brake cleaner dissolving the fly, and the brain slowing as it dissolves. Solvents depress nervous systems, so an inhibitory current grows on every neuron as the fly melts. How strong it is was picked for the game, not measured. The smell and taste neurons it fires are real.
- Freezing and spider venom damping the brain, and the zapper's shock going into a random 30% of neurons (the sim has no current path to place it).
- Alcohol inebriation. Drinking raises a scripted inebriation level (0 to 1, decaying over ~45 s) that the game turns into tremors, a stumbling gait, wobbly flight and delayed escape reflexes. Ethanol's real pharmacology is not modelled: the simulated neurons are unaffected, and only the body's movement is degraded. The Lab mode Model Assumptions page lists this too.
- Sugar switching on the PAM reward neurons directly. In this sim taste input alone doesn't reach them, so sugar drives them the way PAM activation experiments do. The fly walking to the sugar and the proboscis coming out are also game rules (what triggers the proboscis is MN9).
- How looming reaches the fly. The game measures how fast an object grows in its view and drives LPLC2/LC4 directly. Streaming pixels through the sim's own photoreceptors didn't work: the looming signal stayed inside the brain's random flicker. The same transduction runs in the looming assay, so part of its speed dependence is this rule, not a measurement.
- Learning. The plasticity happens on the connectome's own synapses: all 41,495 Kenyon cell to MBON connections that dopamine neurons reach. Which dopamine neurons gate which output neurons comes from the connectome's 37,909 dopamine to output neuron synapses: PPL1 punishment dopamine for MBON11-20 and 30-35, and PAM reward dopamine for MBON01-10, 21, 24 and 26-29. That matches the published map. The rule is the one found in real flies: dopamine plus Kenyon cell activity weakens the synapse. Game rules: pain driving PPL1, sugar driving PAM, each tool having a smell, and the learning rate and forgetting speed. In testing, 10 pairings raised fear of the trained smell from 0 to 0.65 while an untrained smell stayed at 0.01, and the trained smell's approach output neurons dropped from 32 to 30 spikes/s.
- The T-maze (challenge and assay): each odor being a fixed set of 6 glomeruli, the shock driving PPL1 and leg touch neurons, and the choice at the fork. The fly smells each arm and picks the one whose learned drive (liking minus fear, read from its synapses) is higher, plus decision noise. The performance index is computed as in Tully & Quinn 1985.
- The looming assay's escape rule (DNp01 above its threshold before contact) and the sneak challenge's scoring.
- The sugar assay's dose (the share of sugar-pathway taste neurons driven) and what counts as a proboscis extension (MN9 at least 1.5x its rate just before).
- Real-science cards: which reactions get one is decided by the validation results, and only passing tests count.
- Slow motion: the room's physics still advances in 1/60 s ticks; bodies are drawn in between.
- Being drawn to the lamp, and the arena physics.
- The outdoor worlds: the ground, sky, rocks, grass, trees and fruit; the open field's size and where a fly counts as lost; the wind's push on the body; escapes lasting 2.5x longer outdoors; recall (J).
- How wind and sun reach the neurons: each antenna's share of the wind drive is the cosine of where the wind comes from relative to the heading, and the sun's drive is its elevation split between the eyes by azimuth. Real antennae sense wind by being deflected and real eyes see an image; neither is modelled.
- The orchard: each fruit's feeds (default 4), regrowth (default 75 s, +-25%, with slots above the per-tree cap waiting until the tree loses a fruit), the cap (default 4), the share of fermented fruit, the fly flying to the nearest ripe fruit, landing, one fly per fruit, and the feeding bout length (1.5 s). The fly does not forage through its own circuitry; a real escape or take-off abandons the trip.
- Alcohol's scent overlaps another tool's. Alcohol and fermented fruit smell through the real fermentation glomeruli DM1, DM2 and DP1m; every other tool's scent is 5 randomly chosen glomeruli (game rule), and DM2 and DP1m are two of the zapper's. So mushroom-body training on alcohol partly generalises to the zapper and back. That follows from using the real glomeruli, and anyone running feeding or training experiments in the orchard needs to know it.
- In the 1v1 duel:
  - that it has a blaster at all;
  - where you appear in its view, which the game computes;
  - the gun's automatic up/down aim toward your chest;
  - hits firing its reward dopamine neurons, and getting hurt near you firing its punishment ones;
  - how learned fear switches its attention off you;
  - reversal learning: new opposite dopamine restores that smell's weakened synapses in the other compartment, so fear can overturn liking.

  The overall fly-brain firing of those output neurons was too noisy in this sim to read a decision from, so the choice is read from the learned synapses themselves.
- The flight path.
- Everything about the 3D room: the fly's 3D body, physics, walking and flight, and your tools. They use the 2D game's tuned physics scaled to meters, so the brain gets the same kinds of hits as before. What triggers take-off is from the neurons, but where it flies is not.
- Fiber shapes in the brain view. Cell-body positions are real, but almost every neuron's shape is estimated: it's drawn from its cell body toward the center of its synaptic partners. Color is the fiber's direction: red left-right, green up-down, blue front-back. Ten neurons (two each of DNp01, DNa02, MBON01, MBON14 and KCg) are drawn instead from 21 points sampled along their real EM skeletons from neuPrint; that's real data, but how few are drawn, and how coarsely, is a display choice. Either way the simulation treats every neuron as a single point.
- Bilateral symmetry and mirror-averaging. In the raw connectome, bilateral asymmetries arise from both true biology and uneven EM reconstruction/proofreading depth between hemispheres, producing a small spontaneous turning bias in quiet walking (~+0.10 Hz DNa steering bias). The headless audit command (`--audit-asymmetry`) and the Lab Asymmetry page measure L vs R synapse counts and firing rates for key cell types (DNa01, DNa02, LC10, LPLC2, LC4, DNp01). An optional setting (`brain.mirror_weights` or `--mirror-weights`) averages synaptic weights across 77,507 paired bilateral neurons ($W_{sym} = 0.5(W + P W P^T)$). Because this modifies the raw connectome dataset, it is tagged strictly as a Game Rule.
- Brain stethoscope (spike sonification). Synthetic audio clicks triggered when neurons spike in a user-probed neuropil region (mushroom body, antennal lobe, central complex, optic lobes, motor neurons) or inspected neuron group. Hotkey K or button in the big brain view. Tagged strictly as a Game Rule: this is synthetic audio sonification for intuitive listening, not a biophysical local field potential (LFP) or extracellular microelectrode recording.
- Dynamic neural clamp override. Forcing recorded reference spike trains overrides target neurons' natural membrane potentials and severs closed-loop sensorimotor feedback (proprioception and visual flow are open-loop).
- Picrotoxin convulsion animation and severity levels. Scaling inhibitory synapses produces emergent runaway excitation in the connectome; the 0-100% severity slider, convulsion twitching, and clinical seizure labels are game-level rules.
- Hemifield lesion surgery presets. Grouping unilateral cell types into one-click surgical options is a user interface preset; all resulting behavioral consequences are connectome wiring outcomes.
- The courtship song's sound (a synthesized pulse-song buzz) and the SONG threshold; ps1 is read over about a second.
- The aggression lunge: with two or more flies, the aggression neurons above a threshold throw the fly at the nearest one.
- The optomotor EMD stage: how a wide-field rotation becomes current on the T4/T5 subtypes that prefer it. It stands in for the lamina and medulla motion computation, and is used by validation and protocols, not fed the fly's view.
- The thermo arena's temperature gradient (15 to 35 °C), how it maps to the antennal neurons' drive, and the damage at the extremes. The fly doesn't seek comfort through its own circuitry.
- The day/night cycle, daylight driving the LNv clock neurons directly (real ones see light through the H-B eyelet and CRY), the scene darkening, and the SLEEP threshold on the dorsal fan-shaped body. The sim has no molecular clock or sleep pressure.
- The HEAT, COLD and CO2 log thresholds (they log, the fly doesn't act on them).
- Gamepad controls, and the brain view's search box and path drawing (display only).
- Decoy female body physics, animation, and male foreleg proximity detection.
- Any direct shortcut driving aggression from cVA in multi-fly play bypassing connectome wiring.
- Plume filament transport and dispersion by arena wind; upwind surge on odor detection and crosswind cast on odor loss.
- Mushroom body extinction plasticity rule (`EXTINCTION_RATE = 0.005`) depotentiating learned KC->MBON synapses during unreinforced odor exposure.
- Deterministic replay serialization (`.ktfreplay`) and JSON-based UI translation framework with automatic locale fallback.

The full mapping is in the docstring at the top of `kickthefly/game/kick_the_fly.py` (the `kick_the_fly.py` shim in the repo root points at it), and per assay in `kickthefly/lab/assays.py`.

## Repo layout

```
kick_the_fly.py            launcher shim: `python kick_the_fly.py ...` works exactly as before
kickthefly/                the package everything lives in (`python -m kickthefly` runs the same game)
  core/                    clock, save states, settings, user folders, crash reports, version
  sim/                     the connectome: loader, brain pack, the LIF simulator
  game/                    the 2D game, the 3D room, physics, tools, arenas
  ui/                      menu framework and settings screens
  lab/                     validation, assays, challenges, statistics, protocols, recording and export, Lab tools
  data/                    non-code assets bundled inside the package
tests/  protocols/  docs/  packaging/  tools/
data/                      not in git: the connectome download, graph.pkl and the brain pack
```

The canonical map of what comes from the connectome and what is a game rule is the module docstring at the top of
`kickthefly/game/kick_the_fly.py`; the shim in the repo root points at it. `CONTRIBUTING.md` says where new code
goes. This layout arrived in 2.7; nothing user-facing moved, so existing saves, training memory, settings, protocol
files and command lines are unchanged.

## Credits

The connectome data is Janelia FlyEM MaleCNS v1.0, a collaboration between HHMI Janelia, the University of Cambridge, the MRC Laboratory of Molecular Biology and Google Research. It is licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and available at [male-cns.janelia.org](https://male-cns.janelia.org/download/).

The exe and AppImage bundle a compact pack derived from that data. The pack keeps the signed synapse counts, the neuron labels (type, superclass, subclass, instance), body IDs and the cell-body positions, and is otherwise unmodified.
