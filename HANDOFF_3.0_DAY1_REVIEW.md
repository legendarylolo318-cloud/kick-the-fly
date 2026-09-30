# Handoff: 3.0 day 1 review (Opus, branch `opus/3.0-day1-review`, merged into `release/3.0`)

Reviewed Sonnet's `sonnet/3.0-day1` (see HANDOFF_3.0_DAY1.md). Everything below was run on this machine; numbers are measured.

## The adult brain pack now exists here

Sonnet had no adult connectome. I built it with the README's documented steps (`python -m kickthefly.sim.connectome.loader
build` then `python -m kickthefly.sim.brainpack build`: Janelia's public MaleCNS v1.0 bucket, CC BY 4.0, 1.1 GB into the
git-ignored `data/`). Pack: 166,700 neurons, 10,272,125 synapse pairs, 38.6 MB. Nothing outside `data/` and the repo's `.venv`
was touched; no system packages, drivers or settings. The GPU was used only through the app's own paths (`--selftest`'s GL probe
and `tools/make_screenshots.py`-style offscreen 3D rendering): no GPU error occurred.

## Validation: identical

`--headless --validate --sim-backend cpu --workers 10` on `release/3.0` (2f003e1, own worktree) and on `sonnet/3.0-day1` (6b25b85):
**21 tests, 0 differences** in any value or per-seed number (ignoring timestamps, worker count, version), 12 PASS / 9 FAIL, the
documented results (looming -> GF x11.81, MDN FAIL x0.96, ...). The review's own commits touch nothing validation reads
(`lab/validation.py`, `assays.py`, `sim/`, `simcore.py`, `memory.py`, `labjobs.py`, `labstats.py` are unchanged; the only edit near
it is `protocol.run`'s output-folder naming and a `MAX_FLIES` check, which validation does not call).

**p-values.** Sonnet's handoff said the n = 10 floor is 0.00195 and "p < 0.001" would be wrong. That is wrong: validation uses a
**one-sided** exact Wilcoxon (`_wilcoxon_greater`), whose floor is 1/1024 = 0.00098, so "p < 0.001" is exactly right. The
summary line currently prints it as `p=0.0010`, which reads as >= 0.001. **Your call** whether to change that formatting (I didn't:
it changes validation output text, not results).

## Bugs found and fixed (each with a regression test; all 25 new tests fail on Sonnet's code, checked in a clean worktree)

### Science / data
1. **Neurodex discovery misfired on the real brain.** Seen in a real 3D frame: an untouched fly "discovered" types within
   seconds. Measured headless: **~190 types per calm minute** (seeds 0, 1), 89% of them types of 1-4 neurons at ~2 Hz whose noise
   triples a small mean. I fixed the pass criteria first, on exploration seeds 0-4, individuality off and subtle: (1) a calm fly
   discovers nothing in 60 s, (2) driving DNp01 / MDN / DNp09 at amp 0.5 (validation's activation current) discovers it within 3 s.
   New rule: counted spikes over 150 ms plus a one-sided Poisson test against the type's calm rate, alpha derived from a budget
   (one false discovery per 100 h of calm play if spiking were Poisson: 1/(11,751 x 72,000 x 100) ~ 1.2e-11); the Day 1 magnitude
   numbers are unchanged. Result: **criterion 2 PASS 30/30** (0.15-0.25 s); **criterion 1 still FAIL: 25-35 per calm minute**,
   now all sensory types (ORN_VM5v, ORN_VA2, SNta*, WG*) whose spontaneous firing in this model comes in correlated bursts, which
   breaks the Poisson assumption. I did not iterate the design against the same measurement. Instead calm discoveries are tagged
   **"at rest"**, so an entry never claims "in play" for something the player didn't do. **Decision for you:** should calm
   discoveries count at all (e.g. only count while the player interacts), or is "at rest" fine? Documented in docs/neurodex.md,
   the docstring, README, Settings tip and Lab > Model assumptions. Tests: `test_a_tiny_type_needs_more_evidence_than_a_big_one`,
   `test_alpha_comes_from_the_stated_budget`, `test_a_driven_single_neuron_can_still_be_discovered`, the playthrough check.
2. **Curated facts vs the real pack** (`test_every_curated_fact_matches_a_type_in_the_real_adult_pack`, now running, not skipped):
   the pack spells it **EPG** (E-PG removed; `EPGt` is another type and is not matched); **JO-C / JO-E match no type** (the pack has
   JO-CA1, JO-CM, JO-ED1, JO-EV3...), now prefixes. All other names match.
3. **MBON14 citation:** its "2-hour appetitive memory" statement is in Aso et al. 2014 **e04580** (PMC4273436), which Day 1 had read
   but cited as e04577. e04577 (PMC4273437) does hold the names, cell counts and transmitters (checked). Fixed; MBON01's check note
   updated. `test_a_fact_cites_the_paper_its_claim_was_checked_in`.
4. **Dorsal FB fact** attached to all 43 FB6*/FB7* types as if each were shown to induce sleep; reworded to say the dataset doesn't
   tell which of them the paper's neurons are.

### Code
5. **Double key bindings around D.** `Config.bind` moved only the first action on a key, so rebinding the Neurodex to J put Recall on
   D beside walk-right, and binding Arena to D left Arena and the Neurodex both on D. Day 1's own test hit this and was edited to
   dodge it (it moved the binding to F9). Now every displaced action moves (the allowed pair moves together), leftovers are unbound
   with a message; a hand-edited double on D is repaired on load (the old repair reset to the default, which for the Neurodex is D).
   Tests incl. 1,200 random rebinds with no conflict, and schema 1-3 configs with keys on D and `;`.
6. **Share code: path traversal.** A protocol code named `../../../../etc/passwd` was accepted; the saved file name was sanitized,
   but `protocol.run` built its output folder from the name, so running it wrote outside the exports folder. Refused at import now,
   and `protocol.folder_name` sanitizes run folders for any protocol file.
7. **Share code: memory blow-up while previewing.** `flies: 10**9` made `protocol.check` build a billion-seed list (10^8 cost 3.9 GB
   and 1.5 s) before Apply. `protocol.MAX_FLIES = 10,000`.
8. **Share code: the loadout preview lied** (showed duplicates and no hand; Apply dedupes, adds the hand and drops the laser outside
   Lab); control characters in a loadout name were accepted. Fixed; the context now knows the mode.
9. **Bundle: zip bomb.** `verify` read every member into memory: a 0.3 MB bundle cost 300 MB. Now streamed and cut at the size the
   crate declares; a member whose size disagrees is refused. Non-object metadata and a missing `results/summary.json` are refusals,
   not crashes.
10. **neurodex.json with one bad value** (`"x": "abc"`, `null`) crashed the load and so the Neurodex; now per-entry, unknown `how`
    becomes "play".
11. **Thread race:** the table worker created its own `Progress`, racing the game thread's; the tracker and the panel could hold two
    collections. Now made on the game thread first.
12. **Gamepad B didn't skip the kill cam** (the code compared button names with action names). 
13. **The tutorial drew over the kill cam and took its keys** (seen in a real 3D frame). "[recording]" ran off its button.

## What I ran (final state of the merged branch)
| run | result |
|---|---|
| `--validate` before vs after (real pack, cpu) | identical, see above |
| `pytest` (everything, real pack) | **NOT COMPLETED**: stopped at 54% after ~50 min (PC had to shut down). One failure appeared in the first 10%, name not captured (pytest prints names at the end). Every file the review touched ran instead: **208 passed** (neurodex, extras3, sharecode, bundle, compat_3_0, config, killcam, neuron_of_day, playthrough_extras, i18n, compat) |
| `--selftest` | not re-run on the final code (the earlier run's GL/GPU, audio, ffmpeg, folders and Neurodex checks passed; the adult-pack checks now have a pack) |
| `--headless --playthrough all --sim-backend cpu` | **NOT RUN on the real pack** (no time); its four 3.0 checks pass on the synthetic pack in tests. **Run it first next session**, with the full `pytest`, and find that early failure |
| real 3D frames (offscreen GPU, the screenshot tool's machinery) | kill cam, Neurodex (entry, JO-CM), launch card, toast, Share: looked at |
| kill cam MP4 on the real brain (2D) | h264 1280x760, 24 s for 6 s at 0.25x, 719 frames; frame inspected; DNp01 then GFC4 top the risers after driving DNp01 |
| Neurodex table on the real pack | 11,751 types, 0.8 s on a background thread, +193 MB transient, 17 MB kept |
| play session (touches every 0.5 s for 30 s, seeds 5, 6) | ~590 types discovered in play, 7-13 at rest before it: the collection fills fast (pacing, your call) |

## Findings that are not bugs (for you)
- Brain surgery's gentle current (Neuron of the Day's **Try it** in Play) discovers MDN, DNp09 and LPLC2 on the real brain but not
  DNp01, MBON01 or PAM08; the Lab laser and driven currents do.
- D as both walk-right and Neurodex (3D opens it only with the mouse free) and `;` for the kill cam: kept; now safe to rebind.

## Not verified
- The real clipboard (would need a visible window and would overwrite yours) and a real gamepad (none attached).
- GPU compute backends with any of this (no torch/numba here); bundle reruns across backends only via edited metadata.
- The exe / AppImage builds (`--collect-data kickthefly`).
- The 2D-only Neurodex/kill cam on the real pack beyond the headless 2D driver above.
