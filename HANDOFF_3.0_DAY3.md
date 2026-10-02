# Handoff: 3.0 day 3 (written by Sonnet on branch `sonnet/3.0-day3`, from `release/3.0` at e55adf8, for Opus to test and merge)

9 commits, nothing pushed. 53 files, +5,944 lines. Everything below was run on this machine on the real adult pack (166,700 neurons), Python 3.14.7,
the existing GPU driver untouched (the only GPU use was SDL's offscreen GL for the benchmark and the screenshots; nothing failed, nothing installed or changed).

## Read this first

1. **Validation is unchanged.** `--headless --validate --sim-backend cpu --workers 6`, baseline `release/3.0` (e55adf8) vs this branch: 2,020 values compared, **all 21
   tests identical (12 PASS / 9 FAIL, the documented results)**. The only differences are 3 new keys in the recorded Lab-parameter metadata: `weather.rain`, `weather.gust_hz`,
   `weather.storm` at their off defaults. No result value moved.
2. **Full pytest: 992 passed, 22 skipped, 0 failed** (two halves run one after the other: 519 + 473 passed; 41 min each; run on `bcc2cf6`). After that I changed only the HUD layout, the
   bot's dragonfly rule, the handoff and docs, and reran `tests/test_live_inputs.py` (17 passed) and the full playthrough. Running the two halves *at once* made the
   validation tests' workers exhaust RAM and my guard stopped both (no failures in the parts that finished); sequentially it is fine.
3. **Playthrough (`--headless --playthrough adult|larva --sim-backend cpu`): adult 335 passed, 0 failed, 4 gated, 12 skipped; larva 55 passed, 0 failed, 100 gated** (the new
   tools are hidden in larva mode, hence the extra gated rows). The new checks all pass: `extra:weather`, `extra:mic`, `extra:predators`, `extra:kitchen`, `extra:live-inputs`, and
   frog / dragonfly / mantis as tools in the room in both games. Of the 12 skips, 10 are the usual "the item never reached the fly" ones (hand on flypaper, sugar/alcohol/fruit in
   the pool...) and 2 are mine: the 2D dragonfly in flypaper and in the pool. By the game's own rule the dragonfly only takes a fly that is in the air, and a fly stuck on paper or
   floating can't fly, so the bot now reports "inconclusive" there (it judges the dragonfly only when it actually chased).
   **A first full run had 3 FAILs** (those two dragonfly legs, before that fix, and `game2d:adult:flypaper:spider`). The spider one did not recur: I ran that leg alone 6 times on
   this branch (looming peak 26-48 Hz against 7-9 needed, 6/6 pass) and 6 times on `release/3.0` (6/6 pass), and the full rerun passed it. I was rendering frames and running pytest
   at the same time as that first run, so I believe it was load noise, but **I did not prove that**; treat it as a possible flaky leg.
4. **Weather costs the brain about 10% of its real-time speed, and I did not make that go away.** Clear weather is unchanged from before (field 1.009 -> 0.99-1.01, orchard
   0.984 -> 0.96-1.02; screen fps 61 both). With rain or a storm the brain runs at about 0.85-1.0 of real time in the field and 0.8-0.9 in the orchard (run-to-run noise is
   about +-0.1). The first version of the rain drawing (90 translucent streaks built with `segment()`) cost 30-50% (0.60 field, 0.45 orchard); it is now 60 opaque
   streaks. Switching the drawing off leaves the orchard at 0.88-0.90 against 0.96-1.0 clear, so the rest is mostly the real extra neural activity that rain causes.
   Table below. Decide whether that is inside "the existing outdoor budget".
5. **Two things I found by looking at the results, not by design:** the cook's first swat could not be dodged (the fly's looming neurons barely fired before impact), and a
   fly flying to the vinegar jar wandered off before it could fall in. Both fixed, both explained under "Bugs found along the way".
6. **The hum demo says something you may not have expected** (below): humming drives the P1 courtship cluster (about 1.4x calm) but not the song motor neurons, and it is
   **not tuned to the song's 35 ms rhythm**.

## What was built

| # | feature | where | how it reaches the fly |
|---|---|---|---|
| 1 | **Predators**: FROG, DRAGONFLY, MANTIS tools (Creatures, next to the spider), 2D and 3D, plus the `predator_escape` Lab assay | `game/predators.py` (engine, pure), `game/predator_play.py` (adapter for both games), `lab/predators.py` (assay) | only through the existing looming transduction (LPLC2/LC4 -> DNp01); captures fire the touch neurons by body part. Behaviour is GAME RULE, no published value used |
| 2 | **Rain, gusts, storms** (field and orchard) | `game/weather.py`, `kick3d._weather`, Lab params `weather.rain` / `weather.gust_hz` / `weather.storm` (off by default) | drops -> touch neurons by part; wet air -> humidity neurons; gusts -> the existing wind -> JO-C/E transduction; lightning -> photoreceptors; wet wings reuse the pool's rule; reduced flashing = one slow swell |
| 3 | **Kitchen arena** (3D; E, `--arena kitchen`, saves, exports) | `game/kitchen.py`, `kick3d` (`_kitchen_tick`, `_cook_tick`, room and drawing) | bowl = the orchard's fruit code; vinegar trap = DM1/DM2/DP1m scent poke, the fly goes only if those neurons' own firing is >= 2.5x calm; sink = the pool's water in a basin; burner = the lamp's heat; cook = a swatter that is looming |
| 4 | **Microphone -> JO-A/B** (opt-in, off at every launch, red MIC ON pill, nothing recorded or sent), hum test, `hum_demo` assay | `core/mic.py`, `lab/audio.py`, `game/live_inputs.py` | band-passed into JO-B (10-100 Hz) and JO-A (100-1,000 Hz) as current on 88 / 50 real neurons; every number GAME RULE |
| 5 | **Streamer mode** (opt-in, off at every launch, read-only anonymous Twitch chat, `!tool` `!arena` `!surgery`, on-screen tally, every connection shown) | `core/streamer.py`, `core/netguard.py`, `game/live_inputs.py` | nothing touches the connectome; winners run the game's own actions |

Esc > **Mic and streamer** is the page for 4 and 5 (one new pause-menu button; the page also has the hum test and the vote settings). Settings > Brain has the saved tuning
(microphone sensitivity; which chat commands count; vote window, cooldown, minimum votes). **The mic and stream switches are session-only and never written to `config.toml`.**
Also: Python API `fly.attack(kind)`, `fly.weather(...)`, `fly.hear(...)` (synthetic, never a device); protocol blocks `weather:`, `audio:` (a synthetic hum: there is no key
that can name a device), `predator:` and assay kinds `predator_escape`, `hum_demo`; five example protocols; `--selftest` gained `day3`, `microphone` (optional, looks for a
device and never opens it) and `network` (optional, never connects); five docs pages (`docs/predators.md`, `weather.md`, `kitchen.md`, `microphone.md`, `streamer.md`),
README entries, five Lab > Model assumptions entries, the main docstring, tags in `REACTION_SOURCE` / `POPUP_SOURCE`, i18n catalog.

## Pre-registered assays, scored once on seeds 1000-1009 (criteria written into the modules before the run; design chosen on exploration seeds 0-9)

**Predator escape** (`lab/predators.py`; 10 flies x 3 attacks per predator). **P1 PASS, P2 PASS.**

| predator | escaped | probability | 95% CI (Wilson) | max looming before the strike |
|---|---|---|---|---|
| frog | 0 / 30 | 0.00 | 0.00 - 0.11 | 5.0 rad/s |
| dragonfly | 2 / 30 | 0.07 | 0.02 - 0.21 | 14.8 rad/s |
| mantis | 0 / 30 | 0.00 | 0.00 - 0.11 | 0.09 rad/s (below the 1.5 looming threshold) |

P1 = the mantis creep never expands faster than LOOM_MIN (geometry, all 30). P2 = DNp01 never crossed the escape threshold during the creep (10/10 flies). MODEL PREDICTION: a frog
(0.07 s) or mantis (0.06 s) strike gives DNp01 too little time; the dragonfly's dive gives a little. The fly does not move in the assay, so "escape" = the brain's decision.
Design question for you: **in the assay a held-still fly is essentially never "dodging" a frog or a mantis**; in the game the fly can move, but a strike that is already out
cannot be avoided. Is that the difficulty you want?

**Hum demo** (`lab/audio.py`): **H1 PASS (exactly 8 of 10 flies, the minimum), H2 PASS 10/10, H3 PASS 10/10.** P1 firing as a multiple of calm, mean [95% CI]:

| sound | P1 | ps1 (song motor neurons) |
|---|---|---|
| silence | 0.94 [0.88, 1.00] | 0.87 |
| steady 200 Hz | 1.40 [1.32, 1.47] | 1.09 |
| 200 Hz pulses, 35 ms | 1.39 [1.29, 1.48] | 1.05 |
| 200 Hz pulses, 70 ms | 1.32 [1.21, 1.42] | 1.03 |
| steady 50 Hz / 600 Hz | 1.30 / 1.20 | 1.09 / 1.06 |

Answer to "does humming near the song's rhythm drive the song pathway": **it reaches P1 modestly and does not reach the song motor neurons, and it is not selective for the 35 ms
interval** (steady, 35 ms and 70 ms overlap; 50 Hz and 600 Hz also raise P1). The wiring "supports" the hum->P1 step only, and the model has no song-rhythm filter. pIP10 is
two neurons; its ratio is noise and is reported, not judged.

## Outdoor benchmark (`tools/bench_outdoor.py`, the real 3D game, SDL offscreen, 25 s each, one fly)

| arena | build | weather | fps | sim/real |
|---|---|---|---|---|
| field | before (e55adf8) | clear | 61.6 | 1.009 |
| field | after | clear / rain / storm | 61.6 / 61.4 / 61.7 | 0.988 / 1.152* / 1.024 |
| orchard | before | clear | 61.2 | 0.984 |
| orchard | after | clear / rain / storm | 61.3 / 60.6 / 60.6 | 0.959 / 0.840 / 0.810 |

\* a noisy high. Repeated field runs after the last drawing change: rain 0.92 and 0.88, storm 0.96 and 0.95; orchard rain 0.69-0.85 with drawing on, 0.88-0.90 with it off.
Run-to-run noise on this machine is about +-0.1. The first version of the rain drawing measured 0.60 (field) and 0.45 (orchard) rain; 0.70 and 0.50 storm.

## Bugs found along the way (all fixed, each with a test or a playthrough check)

1. **The cook's swat could not be dodged.** The playthrough's looming probe for the swatter peaked at 5 Hz where 8 was needed: a 0.28 s ease-in swing of a small swatter only
   exceeds the looming threshold in its last three frames. Now a 0.9 s steady swing and a 0.5 m looming radius (swatter plus forearm). A design fix for "a swat you can dodge",
   not a threshold tuned to pass; whether a given fly dodges is a MODEL PREDICTION (a held fly's giant fiber reached 3.5x calm against the 4x rule).
2. **A fly flying to the vinegar jar wandered off** before the 0.4 s hover completed (its flight AI picks random targets on arrival). Now it holds over the mouth as at a fruit
   (`perch`), and lets go if its escape neurons send it elsewhere or time runs out.
3. **Stale trap entries after R** (a new fly): the trap's per-fly books kept entries for flies that no longer exist, so the HUD said "1 in the trap" for nobody. Purged each frame.
4. **Reduced flashing cut its own swell short:** the lightning state was cleared when the normal flash ended (0.35 s), so the slow 1.2 s swell jumped to 0 (a visible jump, the one
   thing the setting exists to prevent). Found by a test that checks frame-to-frame steps.
5. **Rain drawing cost the brain 30-50%** (see above); found by the benchmark.
6. **Humidity under rain:** my first playthrough check used mean firing against calm and the humidity neurons (idle at ~18 Hz) failed it at 1.14x (needed 1.15x). That criterion was
   weaker than the bot's own method for tools, so I switched the weather check to the bot's standard windowed-peak test (peak over 100 ms vs calm mean + 4 sd). It passes (peak
   41.7 Hz vs 28.8 needed) and the check records both numbers. I did **not** raise the humidity poke strength; rain moves the humidity neurons' mean by only ~14%.
7. **Two UI bugs found only by looking at real HUD frames** (my first screenshots had HUD hidden by the game's own "Clean screenshots" setting): the red pills were drawn over the
   status panel at the top left and were partly hidden, and on the Mic and streamer page my status text ran into each toggle's own On/Off label ("Offoff", "Onlarena"). The pills now sit
   in the free strip under the health bar (the vote tally below them, a long channel name is shortened); the page has the spacing it needed. The draw tests and the bot's indicator check
   were updated to the new place.
8. Smaller: the 2D loadout tests hard-coded 15 tools (now they follow `TOOL_NAMES`); the editor drag test scrolled to the bottom to find a card that moved (now it scrolls until
   visible); `tests/test_bundle.py::test_verify_streams_and_never_loads_a_big_member` (a `tracemalloc` measure) failed once when run after game tests in the same process and passes alone,
   in its own file and in the full run: an existing order sensitivity, not touched.

## Sources, and what was and was not verified

- **Kamikouchi et al. 2009** (Nature 458:165, doi 10.1038/nature07810): abstract read through Europe PMC: JO neurons form clusters; gravity- and sound-sensitive neurons differ.
- **JO-B prefers < ~100 Hz, JO-A higher, ~10 Hz to ~1,000 Hz together:** from a 2019 review (Frontiers in Physiology 10:1552) **read through a summarising tool, not by a person**.
- **Song numbers (about 220 Hz pulses every ~35 ms; a ~150 Hz sine hum):** from **search-result summaries** of Current Biology 2024, eLife 2015;4:e08477 and Bennet-Clark & Ewing 1969;
  not read in the papers. Used only to shape the demo's sounds.
- **JO-A (50) and JO-B (88) neuron counts, P1 = 86 neurons, ps1 = 2, pIP10 = 2:** counted in the real pack.
- **Twitch IRC:** the official page (dev.twitch.tv/docs/chat/irc/) gives the hosts, ports (TLS 6697), `JOIN`, `PING`/`PONG`; it does **not** mention anonymous access. The anonymous
  `justinfan<digits>` read-only login is from Twitch developer-forum threads (found by search). It may stop working; the game reports a refused login and does nothing else.
- Everything else (every predator, weather, kitchen, mic and vote number) is a game rule with no source, and says so on screen, in the docs and in Model assumptions.

## What I did NOT verify

- **A real microphone was never opened** (tests fake the device; the hum test page was never seen with real sound; capture from a PipeWire/PulseAudio device is untested here).
- **The real Twitch network was never contacted.** The client is tested against an in-memory fake server (login, join, PING, a refused login, a hang-up); TLS and the real server's
  behaviour are not.
- **A visible window with real input and a gamepad:** not done. Frames were rendered offscreen and looked at: the kitchen (with the cook mid-swing), the frog, mantis and dragonfly
  (mid-dive), a storm flash with rain, the red Twitch pill with the vote tally on the real HUD, and the Mic and streamer page. The frog's tongue and the mantis's claw at full extension
  were not seen (a strike lasts 4 frames and my capture caught the bodies). The MIC ON pill and the 2D predators were only exercised by draw tests and the bot, not looked at.
  A rain frame in the orchard was not usable (my capture script put the camera inside a tree).
- GPU compute backends, exe/AppImage builds, `nwbinspector` (as before).
- Larger-text, colorblind palettes: the pills use the bold HUD font and a text-plus-dot design; not looked at under each setting.
- Weather and the kitchen are 3D only (gated in 2D with a message, like the outdoor arenas). Predators work in both.
- Rain has no shelter under trees; the kitchen is the 3D room with different furniture and the counter as its floor.

## Decisions I made that are yours to change

- The predators are **not** in the Chaos preset (the presets are unchanged; they are in All and Lab). The tool keys are unchanged: the new tools are appended after `fruit`, so
  existing number-key muscle memory is untouched; there are now 18 tools and the hotbar pages.
- "A cook NPC who periodically swats with looming **you** can dodge" was read as the **fly** dodging (the player is not hit in this mode).
- Streamer defaults: `!tool` is on; `!arena` and `!surgery` are off until switched on. Options viewers can pick are only the tools on your hotbar, the arenas the game can show, the
  surgery list. A vote winner changes the arena **without** saving it.
- The microphone and stream switches are never persisted, so neither can start by itself.
- Screenshots in the README were not remade (`tools/make_screenshots.py` was not run).
