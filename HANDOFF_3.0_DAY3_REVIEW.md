# Handoff: 3.0 day 3 review (Opus, branch `opus/3.0-day3-review` from `sonnet/3.0-day3` at 40205e2, merged into `release/3.0`)

Adversarial review of the five Day 3 features (predators, weather, kitchen, microphone, Streamer mode). Real adult pack (166,700
neurons), Python 3.14.7, this machine, CPU backend. Nothing installed, no driver or system change; the only GPU use was SDL's
offscreen GL for the frames and the benchmark, with no GPU error. **The microphone was never opened and Twitch was never contacted.**

## Read this first

1. **20 bugs found and fixed; all but the page layout (16) have a regression test that failed before its fix** (`tests/test_day3_review.py`, plus
   updated `tests/test_live_inputs.py`). The four that matter most: a click with any tool pulled a fly out of a frog's mouth and the frog
   ate it anyway; turning Streamer mode off and on while it was connecting left a **second, hidden Twitch connection** that the on-screen
   list no longer showed; a chat vote for `!arena`/`!surgery` could still win after the streamer had switched that command off; and
   Settings > Brain was a broken page whenever the kitchen was the arena.
2. **One reported science number changed:** the predator assay counted a giant-fiber crossing *in the capture frame itself* as an escape
   (lead 0.0 s). Both dragonfly "escapes" were that. By the module's own rule ("before the capture") and the game's (a held fly can't
   escape) they are captures: **dragonfly 2/30 -> 0/30** (CI 0.00-0.11). P1 and P2 are unchanged and still PASS (10/10); the criteria
   were not touched. Hum demo: Sonnet's numbers reproduced exactly (H1 PASS 8/10, H2 10/10, H3 10/10).
3. **A literature finding the docs now state:** the song numbers come from Zhou et al. 2015 (eLife 4:e08477), which I read in full. The
   same paper measured **pC1 neurons tuned to 35-65 ms pulse intervals, only above 80 dB**. The model's P1 is not tuned (steady, 35 ms and
   70 ms raise it alike), so on this point **the model disagrees with a measurement**; it was described as "no song-rhythm filter".
   The "2019 review" of JO-A/B is a research article (Ishikawa et al. 2019); its sentence was checked word for word.
4. **Weather slowdown:** today, relative to clear weather in the same session, rain costs the brain ~8-11% of its real-time speed and a
   storm ~15-17% (3 runs each). Clear weather itself is identical on e55adf8, 40205e2 and this branch; this machine is slower today than in
   Sonnet's session for all three. The outdoor budget in docs/performance.md is 1.00x real time with one fly: weather does not meet it
   here, and clear weather doesn't today either. I found nothing cheap to win (below). **Your decision.**
5. **The flaky `game2d:adult:flypaper:spider` leg is not load noise.** The playthrough rig is lockstep (it steps the brain itself; no
   brain thread runs; the loom measure is per frame, not wall clock), so CPU load cannot change what the brain receives. I reproduced the
   FAIL **in isolation on a quiet machine** (4.3 Hz vs 8.7 needed). The loom peak ranges from 4 to 48 Hz (most runs 42-48) and differs between fresh
   processes given identical inputs, so something in the 2D frame loop is nondeterministic. Ruled out: a brain thread, the brain-view
   thread, the asynchronous Neurodex build, Python hash ordering, the memory autosave, the fly's wall-clock timestamps. **Root cause not
   found.** It is very likely the same thing the Day 2 review saw on `flypaper:bomb` (4.574 Hz vs 8.725, "history-dependent"). The criterion
   was not touched. Recommendation: make the 2D game legs deterministic first (a per-leg trace of every `random`/`np.random` call and every
   poke would find the divergence), then judge the leg.
6. **The follow-ups you approved during the session are merged in** (section below), including the test-suite memory leak.

## What I ran

| run | result |
|---|---|
| `--validate` baseline e55adf8 vs the final code (follow-ups included), cpu, 6 workers | **identical**: 2,024 values compared; the only differences are the 3 new keys `weather.rain`, `weather.gust_hz`, `weather.storm` (all 0.0) in the recorded Lab parameters, plus the run time and date; 21 tests, **12 PASS / 9 FAIL in both** (the documented results). Also identical on the pre-follow-up code |
| full pytest, final code | **1,032 passed, 20 skipped, 0 failed.** The non-validation tests ran as 4 chunks, 2 at a time (chunk 1: 242 passed / 19 skipped, 8.6 min; 2: 287 / 1, 11.9 min; 3: 263 passed, 6.7 min; 4: 218 passed, 1.4 min; about **15 min of wall time instead of ~83**). The 22 validation tests passed earlier (17.5 min) and were not rerun on the final code: they run the same simulations as `--validate`, which is identical there. Earlier full run: half 1 555 passed / 19 skipped; half 2 stopped at 95% by the memory leak with 0 failures. The whole suite runs with the network guard in `tests/conftest.py` |
| `--playthrough adult` / `larva`, cpu, final code | **adult 334 passed, 0 failed**, 4 gated, 13 skipped; **larva 55 passed, 0 failed**, 100 gated (the new tools are hidden for the larva). Day 3 checks weather, mic, predators, kitchen, live-inputs (now also: Streamer mode refused in Lab mode) PASS; `game2d:adult:flypaper:spider` PASS this time. The 13 skips are all "inconclusive: the item never reached the fly" (hand on flypaper and in the escape room, sugar/alcohol/fruit in the pool or by the lamp, the dragonfly that only hunts a flying fly in flypaper and the pool); which legs land there varies between runs (9 on the earlier run), very likely the same nondeterminism as item 5 |
| `predator_escape` assay, seeds 1000-1009 (before / after the capture-frame fix) | frog 0/30, mantis 0/30 both times; dragonfly 2/30 (lead 0.0 s, both) -> 0/30; max loom before the strike 5.0 / 14.8 / 0.09 rad/s; **P1 PASS, P2 PASS (10/10)** |
| `hum_demo` assay, seeds 1000-1009 | P1 x calm: silence 0.94 [0.88, 1.00], steady 200 Hz 1.40 [1.32, 1.47], 35 ms pulses 1.39 [1.29, 1.48], 70 ms 1.32 [1.21, 1.42], 50 Hz 1.30, 600 Hz 1.20; ps1 0.87-1.09. **H1 PASS (8/10), H2 PASS (10/10), H3 PASS (10/10)**: identical to Sonnet's |
| streamer fuzz: 200,000 random lines (CRLF, NUL, huge, unicode, bidi, odd tags) through `parse_line`, `parse_command`, `slug`, `clean_channel` | 0 exceptions; `clean_channel` never returned anything outside `[a-z0-9_]`. Found: a PING payload with a CR was echoed into the PONG (fixed) |
| vote-board fuzz: 300 boards x 2,000 messages, random rules, clock jumps of 0 to 10^6 s and backwards, 10,000 viewers in one second | 0 violations (no disallowed winner, no winner under the minimum, bounded memory; 30 counted of 10,000 in a second) |
| protocol fuzz: 40,000 random protocols with hostile `weather:` / `audio:` / `predator:` blocks and `predator_escape` / `hum_demo` options | **13 TypeErrors before** (unhashable `kind` / `kinds` / `conditions`), **0 after**; no accepted `audio:` block ever carried a key other than the six synthetic-hum keys (no device can be named) |
| `--record-replay` with each new block, then `--replay` | `weather:` 76 events, identical; `predator:` (10 s) identical; **`audio:` spikes DIFFER** (its current is not a replay event): now refused, like the Day 2 current blocks |
| `--bundle` with all three blocks, then `--rerun-bundle` | MATCH (bit-exact), 2/2 |
| artifacts made by release/3.0 (e55adf8), loaded here: save state (orchard), config.toml, three share codes, a bundle, a replay | all load; the save gets the three `weather.*` keys at off; bundle MATCH bit-exact; replay identical; mic and stream off |
| frames rendered offscreen and looked at | below |
| outdoor benchmark, 25 s each, 3 repeats | below |
| the full 2D playthrough matrix, 3 times in one process (as `run()` builds it) | **0 FAIL in every 2D leg, all 3 times**; `flypaper:spider` loom peak 8.06 / 32.97 / 47.75 Hz (needed ~6.8) |
| `flypaper:spider` alone, 3 legs per process, 13 fresh processes (with and without the view thread, Neurodex built synchronously, `PYTHONHASHSEED=0`) | peaks 4.3-48.2 Hz, most at 42-48 but 4.3, 13.3, 13.4, 13.7, 17.5 and 28.3 too; **1 FAIL in isolation on a quiet machine: 4.315 Hz vs 8.725 needed** |

## Bugs found and fixed (each has a regression test in `tests/test_day3_review.py` unless noted)

**Privacy and network**
1. **A hidden second Twitch connection.** `TwitchChat` shared one stop flag across runs. Off-then-on while the first connect was still
   blocked cleared the flag the old reader waited on: two readers ran, and the old one's connection had already been taken off the on-screen
   list. Now each run has its own stop flag, socket and list entry; a run takes itself off the list only once its socket is closed.
2. `stop()` called `sendall`/`close` on the SSL socket from the game thread while the reader thread was inside `recv` (an SSL object must
   not be used from two threads; a blocked `sendall` could also freeze a frame for up to 5 s). Only the reader touches its socket now; it
   polls every 0.25 s, says QUIT and closes.
3. A failed TLS handshake left the TCP connection open (raw socket leaked).
4. Nothing may add a second IRC command: outgoing lines with CR, LF or NUL are refused, and a PING payload is stripped of them (the server
   could make the client send `PONG :x\rJOIN #other`).
5. **A vote for a command the streamer switched off mid-round still ran.** The allowlist is checked again when the winner executes, and
   votes for a command that is switched off are dropped from the open round.
6. The docstring and the page promised a viewer's hash is dropped when the round ends; the flood limit's book kept them for up to a
   minute. Only the last 2 s (what the limit needs) are kept now, and the page says so.
7. **The brain view's neuPrint skeleton fetch ignored the network switch.** `--validate`, `--playthrough` and protocols disable the network
   (netguard + `KTF_NO_NETWORK`), but `sim/morphology.py` only looked at `KICK_THE_FLY_OFFLINE`. It goes through netguard now. The netguard
   docstring, the self-test text, `docs/streamer.md` ("the only network feature") and README ("None of these uses the network or the
   microphone", printed right under the Microphone and Streamer bullets) said otherwise; all corrected.
8. A stale microphone: a device that stopped delivering (unplugged, or a capture error) left its last sound on JO-A/B forever. After 0.5 s
   without a new chunk the drive goes to zero; the switch and the pill stay as the player left them.
9. Test-suite guard (`tests/conftest.py`): any `connect()` to an IPv4/IPv6 address in the test process raises. The full suite ran with it
   (see the pytest row), so "nothing connects during pytest" is checked, not assumed.

**Game**
10. **Any click pulled a fly out of a predator's mouth.** Every left mouse-up sets `fly.grabbed = None` (both games), so the fly dropped free
    while the frog still "carried" it, and was eaten anyway when the frog left. The predator re-asserts its hold each frame, as the vinegar
    trap does. Also: a fly in one predator's mouth is no longer prey for a second one (a frog and a mantis could both catch it, double damage).
11. **Settings > Brain broke in the kitchen:** the arena setting had 10 options and 9 labels; the page showed an error instead of the
    settings. Label added; a test checks every choice setting.
12. **Wet air reached only the first fly:** the humidity poke was drawn inside fly 0's turn (none at all while fly 0 was dead). Drawn once
    a frame for every fly now; one fly sees exactly what it did before.
13. **Leaving the kitchen with a fly in the trap** left it `grabbed`, pinned to the player's hand point in the next arena.
14. **Loading a save** restored a fly's own `grabbed` part (and the trap's `perch = "vinegar"` and trip), but not what held it (hand,
    predator, trap are not saved): the fly stayed pinned to the hand point by nothing. Released on load (a spider-wrapped fly keeps its wrap).
15. The red pills: the "●" was a missing-glyph box (the HUD's fallback font on Linux has no U+25CF; now a drawn circle); a long channel
    name (25 characters, the Twitch maximum) ran over the status card; at y = 58 they sat under the video REC badge and the "saved" line.
    Now from y = 90, never left of the status card, text cut to fit. Found by looking at frames.
16. Mic and streamer page: with Larger text "Viewers may vote" ran into "!tool", the slider labels into their sliders, the hum-demo text out
    of its button; the streamer's status line ran off the panel's right edge **even at normal size**. Label column and controls are sized
    from the font, the status has its own line. Looked at in both sizes (no unit test: layout).
17. `render3d.segment(a, a)` (a tongue on its strike's first frame) built a NaN model matrix.

**Lab**
18. The predator assay's capture-frame off-by-one (above, "Read this first" 2). `docs/predators.md` updated with both numbers.
19. `protocols/mic_hum_demo.yaml` **crashed at export** (`KeyError: 'per_fly'`): the hum demo's summary had none. Added, and `headline()`
    no longer fails when the 35 ms condition isn't in the run. Both assays now write their pre-registered verdict (`criteria`, `verdict`)
    into `summary.json`; before, the P/H verdicts were only ever computed by hand or in tests.
20. `--record-replay` accepted an `audio:` block and wrote a replay that does not replay (spikes differ). Refused now.
21. Protocol validation raised TypeError for unhashable `predator.kind`, `predator_escape` `kinds`, `hum_demo` `conditions` or
    non-mapping options; `fly.hear(seconds=1e6)` tried to build 22 billion samples (now the protocol block's ranges), `fly.attack([])`
    raised TypeError.

## Sources (read by me, not through summaries, except where said)

| claim | source | what I found |
|---|---|---|
| JO-A/B are the sound-sensitive groups | Kamikouchi et al. 2009, Nature 458:165, abstract (Europe PMC) | quote verbatim |
| JO-B < ~100 Hz, JO-A higher, ~10-1,000 Hz together | Ishikawa, Fujiwara, Wong, Ura & Kamikouchi 2019, Front. Physiol. 10:1552, full text PMC6960095 | sentence verbatim in its introduction (citing Matsuo et al. 2014; Patella & Wilson 2018). **It is a research article (brief report), not a review**: fixed in `core/mic.py`, `docs/microphone.md`, Model Assumptions, the main docstring |
| ~35 ms IPI, 220 Hz pulse carrier, ~150 Hz sine | Zhou et al. 2015, eLife 4:e08477, full text PMC4575990 | ~35 ms IPI and ~160 Hz sine (introduction, citing Bennet-Clark & Ewing 1967 among others); 220 Hz carrier and 140 Hz sine are their synthetic stimuli (methods). **pC1 band-pass tuning to 35-65 ms, only above 80 dB** (their Fig. 6): now in the docs as a disagreement with the model |
| Current Biology 2024 | Lillvis et al. 2024, Curr Biol 34, doi 10.1016/j.cub.2024.01.015 | real; the abstract has no song numbers; full text not open, **not read**; no longer cited for numbers |
| Bennet-Clark & Ewing 1969 | not open access | **not read**; the numbers no longer rest on it |
| Twitch IRC host, port, PING | dev.twitch.tv/docs/chat/irc (fetched) | `irc.chat.twitch.tv:6697` TLS, PING/PONG; the documented login is `PASS oauth:...`; nothing about anonymous access |
| anonymous `justinfan` login | discuss.dev.twitch.com threads 2046 (2015) and 5921, and the Feb 2024 join-limits announcement (54997), read as JSON | both cited threads exist; in 2024 a forum moderator wrote justinfan "was never officially documented" and its fate under the limits was unknown: added to `docs/streamer.md` |
| JO-A 50, JO-B 88 neurons | the pack | Model Assumptions had them in the wrong order ("JO-B ... and JO-A ... (50 and 88)"): fixed |

## Frames looked at (offscreen, real adult pack; capture script in the session scratchpad, not committed)

- Frog: tongue mid-strike, at the catch (0.82 m of a 1.09 m aim), the fly held on the tongue tip (GRABBED, touch neurons firing,
  health 45). Mantis: claw at full reach, the fly held. Both 2D predators and the 2D dragonfly drawn holding the fly.
  At 0.9 m the frog's head sphere overlaps the fly (cosmetic; a player can put a frog that close).
- MIC ON + TWITCH CHAT pills and the vote tally: default, Larger text, Blue/yellow and High contrast (the palettes only change the brain
  view; the pills carry text, so they don't depend on red). The Mic and streamer page at both sizes (bugs 15, 16).
- Orchard storm: rain streaks, darker scene, the touch neurons firing on drops. Reduced flashing: a single slow brightening (play-area
  mean 69 -> 99 over 0.6 s and back), no strobe. The normal flash's pulses (4 frames) fell between my captures; the frame-to-frame rule
  is covered by `tests/test_weather.py`.
- Kitchen: burner, cook mid wind-up, vinegar jar, sink with water, bowl; the fly flew to the bowl and ate.
- My capture method sometimes returned an all-black frame when two captures were one frame apart (reading the screen right after a swap).
  It never happened to the game's own photo path, which re-renders; I treated it as an artifact of my script.

## Performance (`tools/bench_outdoor.py`, SDL offscreen, 25 s, one fly, sim/real; 3 runs each, same session)

| arena | clear | rain | rain, streaks not drawn | storm |
|---|---|---|---|---|
| field | 0.856 / 0.905 / 0.915 (mean 0.89) | 0.80 / 0.84 / 0.73 (0.79) | 0.93 / 0.83 / 0.82 (0.86) | 0.74 / 0.73 / 0.80 (0.76) |
| orchard | 0.76 / 0.74 / 0.75 (0.75) | 0.74 / 0.66 / 0.67 (0.69) | 0.74 / 0.84 / 0.70 (0.76) | 0.59 / 0.72 / 0.55 (0.62) |

fps 60.5-62.1 everywhere. Clear weather, interleaved across builds (3 runs each): field e55adf8 0.87, 40205e2 0.88, this branch 0.92;
orchard 0.81 / 0.74 / 0.80: **no regression from Day 3 or from this review**; this machine is slower today than in Sonnet's session for all
three. (One rain row may have overlapped 15 s of tests I ran by mistake; the spread is the same without it.)
Decision material: rain costs ~8-11% and a storm ~15-17% relative to clear. The streaks are already one instanced batch; their Python
cost per frame is well under a millisecond, and drawing them on vs off differs by less than the run-to-run spread. The rest is the pokes
and the real extra spiking they cause, which I may not change. So: **not inside a 1.00x budget on this machine, and nothing cheap left.**

## Not verified

- **Test-suite memory is still high after the leak fix:** per-process peaks of 3.6 / 5.5 / 6.9 / 9.9 GB (RAM + swap) for the four chunks.
  Something else keeps state between tests (likely real-pack games that tests don't stop, or module caches). Running two chunks at a
  time fits in 16 GB; three does not. Worth a follow-up with `tracemalloc` per test file.

- A real microphone, a real Twitch connection (TLS, the server's real behaviour), a visible window with real input and a gamepad, GPU
  compute backends, exe/AppImage builds, `nwbinspector` (as before).
- The normal (not reduced) lightning flash's pulses on screen (captures missed them; covered by tests).
- Lillvis et al. 2024 and Bennet-Clark & Ewing 1969 full texts.
- Pre-existing, not touched: the video/timelapse REC badges also use "●" and will show the same missing-glyph box on this machine.

## Follow-ups you approved in the session (branch `opus/3.0-day3-followups`, merged here)

| change | why | test |
|---|---|---|
| **Test-suite memory leak fixed.** Each `LiveInputs` (Day 3) called `atexit.register(self.stop_all)`, which kept every game ever created, brains included (~1 GB each on the real pack), alive until the process ended. One module-level hook over a `WeakSet` now. | The first full run's second half grew to **20 GB (4.8 GB RAM + 15.5 GB swap)** and was force-stopped at 95% (0 failures to that point); this is also why each half took ~40 min | `test_a_finished_game_is_freed` (fails without the fix) |
| **Frog, mantis and dragonfly redrawn in 3D**: ellipsoid bodies, jointed limbs, eyes, wings; the frog's head is built around the engine's mouth point so the tongue leaves between the lips. Each is still scaled from its own looming radius, so what the fly sees is unchanged. | you: "that does not look like a frog"; the tongue came out of the chin | draw tests; looked at from 4 angles, at rest and mid-strike |
| **Frog and mantis in the Chaos preset** (appended: keys 1-7 unchanged; the dragonfly stays in All and Lab: it only hunts a flying fly) | your decision | `test_loadout.py` |
| **Streamer mode is off in Lab mode** (refused, and stopped if Lab mode starts while streaming); the playthrough bot checks it | a chat vote could silence a neuron group in the middle of a recording | `test_streamer_mode_is_refused_in_lab_mode...` |
| **Recordings say whether the microphone was on** (`live_inputs` in the metadata) | its JO-A/B current is not a logged stimulus | `test_a_recording_says_whether_the_microphone_was_on` |
| **neuPrint shape download is opt-in**: Settings > Brain > "Download real neuron shapes" (default off) and a one-time question after the tutorial; shapes already cached still load; the brain view says how to turn it on | the project rule: network use is opt-in. Drawing only: every neuron is simulated as a point either way | `test_the_neuprint_download_is_opt_in_and_asked_once` |

Not done, flagged as separate tasks (they need more than a review's change): delivering **all** live-game stimuli in brain time (weather alone would
put rain on a different clock from every tool, wind and the sun; today every live stimulus is delivered per screen frame, so a brain below real time
gets more stimulation per simulated second; Lab protocols, assays and the API are lockstep and unaffected); comparing the predator assay with
published data (papers to be read, criteria pre-registered); and the root cause of the 2D flypaper nondeterminism.

## First full run (before the follow-ups), for the record
pytest half 1 555 passed / 19 skipped / 0 failed (41.5 min); half 2 light part 0 failures up to 95%, then stopped (the leak); validation tests 22
passed (17.5 min); playthrough adult 338 passed / 0 failed / 4 gated / 9 skipped, larva 55 / 0 / 100 / 0; `--validate` identical (below).

## Decisions (made by you in the session, after my recommendations)

| question | what was decided |
|---|---|
| Predators in the Chaos preset? | **Done:** frog and mantis appended to Chaos (keys 1-7 unchanged); the dragonfly stays in All and Lab, since it only hunts a flying fly. |
| 0/30 escapes for every predator (held-still fly) | **Kept** as a labelled MODEL PREDICTION; no strike slowed to create escapes. A comparison with published data is a separate task (papers to be read; criteria pre-registered). |
| "A cook you can dodge" = the fly dodges | **Kept:** the dodge runs through the real looming neurons; a cook that hits the player would be a pure game rule. |
| Streamer defaults | **Kept** (`!tool` on, `!arena`/`!surgery` off), and **Streamer mode is now off in Lab mode**. |
| ~10% (rain) to ~16% (storm) brain slowdown | **Accepted.** The accuracy issue underneath (every live stimulus is delivered per screen frame, so a slow brain gets more per simulated second) is a separate game-wide task; Lab protocols and assays are lockstep and exact. |
| The neuPrint skeleton download | **Done:** opt-in (default off), a one-time question after the tutorial, cached shapes still load. |
| Sonnet's other choices: tool keys appended, switches never persisted, README screenshots not remade | Agree; the README screenshots should be remade now that these are settled (`tools/make_screenshots.py`, not run in this review). |
