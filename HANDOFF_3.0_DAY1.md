# Handoff: 3.0 day 1 (written by Sonnet on branch `sonnet/3.0-day1`, for Opus to test and merge into `release/3.0`)

Nothing is pushed. `release/3.0` was created locally from `main` (2f003e1) and this branch from it, 8 commits, author `Claude
<noreply@anthropic.com>` set per command (no git config was changed). Version is **not** bumped (`2.13.1` still shows in the pause
menu); CHANGELOG has a "3.0.0 (unreleased)" entry.

## Read this first

1. **There is no adult brain pack or connectome on this machine** (`data/` has only `kick_larva_brain.npz`). So: `--validate` was
   **not run, before or after**, and `pytest -m validation` skipped all 25 tests. I can't give you the before/after diff. What I can
   say is structural: `git diff release/3.0..HEAD --name-only` touches none of `lab/validation.py`, `lab/assays.py`, `sim/*`,
   `core/simcore.py`, `core/memory.py`, `lab/labjobs.py`, `lab/labstats.py`, `sim/wiring.py`, `core/individuality.py` or
   `lab/recorder.py`. The only edit near the simulation is `protocol.run_seed`, which adds three metadata keys to the per-fly JSON
   (`sim_backend`, `sim_dtype`, `spike_sha256`). **Please run `--validate` on `release/3.0` before and on this branch and diff them.**
2. **Everything that needs a Brain ran on a SYNTHETIC pack** (`tests/synthetic_pack.py`: 2,346 neurons, random wiring, real type
   names, made from a fixed seed). It proves the plumbing (UI, saves, exports, reruns). It says nothing about biology and I did not
   read a single result off it. The one place real data was available, the **larva pack**, is used for real (see Larva below).
3. **Environment changes I made (all inside the repo's `.venv`, which is git-ignored):** created `.venv`, installed
   `requirements.txt` and `pytest`, then replaced `pygame` with **`pygame-ce`** in the venv, because pygame 2.6.1 built here without its
   font module on Python 3.14 (fixing that would have needed system SDL_ttf). `requirements.txt` is unchanged; CI is unaffected.
   No system packages, no sudo, no GPU/driver/env changes. `--selftest` exercised the app's own OpenGL/GPU probe: **no GPU failure**
   (GL 4.6 Mesa 26.2.3, AMD RX 9070 XT, both checks PASS).

## What was built (one section per task)

### 1. Neurodex (D, Esc > Neurodex, pad d-pad up) and the curated facts
- `kickthefly/core/neurodex.py` (pure logic): `TypeTable` built from the pack's own arrays (count, superclass, regions, transmitter +
  the dataset's confidence, top input/output partner types as summed synapse counts from the signed CSR, built in chunks so the real pack
  never needs several full-size index arrays); `Progress` (saved to `memory/neurodex.json`, per brain; corrupt file kept as `.bad`, a
  newer-version file never overwritten); `Tracker` (the discovery rule); facts loader/validator; `entry()`.
- **Discovery rule (GAME RULE), fixed before any play and not tuned:** a type's mean firing >= 6 spikes/s **and** >= 3x its own calm
  rate (floor 2 spikes/s) for 3 checks (150 ms), after 5 s of settling; per-type calm tracked while the fly is calm. Tagged
  "discovered by stimulation" when surgery/drive/laser was on. Only the fly the brain panel shows is tracked. Never runs in
  `--validate`/protocols/tests (only `Game.update` and the opt-in `Fly.collect()` call it).
- `kickthefly/ui/neurodex_ui.py`: the panel (regions with progress, searchable list, entry with CONNECTOME / GAME RULE / LITERATURE
  chips, EM skeleton from the **cache only**, never the network, Reset Neurodex with a second click). An undiscovered type shows `???`
  and nothing else; search only finds discovered names.
- **Facts: `kickthefly/data/neurodex_facts.yaml`, not `data/neurodex_facts.yaml`**, because the repo-root `data/` is git-ignored (the
  file would never have been committed). I added `--collect-data kickthefly` to `build_appimage.sh` and `build_exe.ps1` so it ships in
  the exe/AppImage; **I could not test either build**.
- 25 entries covering ~60 dataset type names (P1 is an explicit 25-type list; PAM, PPL1, KC, T4, T5, FB6, FB7, LC10 are prefixes):
  DNp01 (von Reyn 2014), LPLC2 (Klapoetke 2017), LC4 (von Reyn 2017), MDN (Bidaye 2014), DNp09/P9 (Bidaye 2020), MN9 (Schwarz 2017),
  aDN1/2 = DNg62/DNge078 (Hampel 2015), DNa01 + DNa02 (Rayshubskiy 2025), PAM (Liu 2012), PPL1 (Aso 2012), Kenyon cells (Honegger 2011),
  MBON01 and MBON14 (Aso 2014 Table 1; Owald 2015), pIP10 (von Philipsborn 2011), P1 (Hindmarsh Sten 2021), LC10 (Ribeiro 2018), E-PG
  (Green 2017), ORN_DA1 (Kurtovic 2007), ORN_V (Suh 2004), AVLP727m (Asahina 2014), T4/T5 (Maisak 2013), l-LNv/s-LNv (Grima 2004),
  dorsal FB FB6/FB7 (Donlea 2011), JO-C/JO-E (Yorozu 2009, Kamikouchi 2009).
- **How they were checked (this session, not from memory):** every DOI resolved through Crossref or PubMed and the title/first
  author/journal/year/volume/pages matched; each fact was then written from that paper's abstract (PubMed/Crossref) or full text (PMC),
  and each entry's `checked:` field says which. **Two DOIs I first remembered resolved to unrelated papers (one was a prefrontal-cortex
  paper) and were discarded**, which is why nothing was left to memory. Weaker spots, stated in the entries: LC4 leans on the abstract
  ("a visual projection neuron type") plus the dataset paper's reference list naming LC4 for that study; the MBON01/14 names and cell
  counts are from Aso 2014 Table 1; the JO-C/E mapping to wind/sound/gravity is the game's, which the fact says it doesn't claim.
- **Left out because I couldn't verify a source:** the leg motor neuron `ps1 MN`, DNg02 (flight), TRN_VP2/VP3, DNg28, anything with a
  number I couldn't find in the paper.
- **Not verified against the real pack:** the exact type-name spellings. They come from names the game's own code already uses
  (`SENSE`, `MOTOR`, `assays.groups`, `SURGERY`), except **E-PG**: the entry lists `EPG` and `E-PG`. `tests/test_neurodex.py::
  test_every_curated_fact_matches_a_type_in_the_real_adult_pack` skips here and **will check all of it when the pack exists**.
- Known limits, in the docs and the Model Assumptions page: the rule reads the type's *mean*, so Kenyon cells are mostly discovered by
  stimulation; I could not observe how many types a real play session discovers (the constants are a priori). Please look at a real
  session before anyone is tempted to "fix" it: if the numbers change, fix the criteria first.

### 2. Kill cam (`;`, pad d-pad down, autopsy card button)
- `core/killcam.py`: ring buffer of per-neuron rates (uint8, sqrt-compressed, ~25 MB for the real brain, **only the brain panel's fly**),
  `Replay.top_risers` (last 1.5 s minus first 1.5 s, top 12, min rise 5 spikes/s, deterministic ties), `Player` (slow motion 0.25x,
  0.2x with Reduced flashing). Time comes from the recorded **step numbers** (a first version timed by frame count and would have run
  ~25% fast; a test caught it), and the replay is trimmed to 6 s.
- `game/extras3.py`: samples in `update()`, freezes in `_die()` before the drive is cut, draws the overlay (timeline, riser list,
  rings on the brain panel and big view), replays the recorded **group rates** in the panel bars, skip by Esc/Space/Enter/click/pad,
  Save = replays once through the game's own `VideoRecorder` (MP4 if ffmpeg, else GIF).
- **Not done:** no body/room replay (the room is dimmed behind the card); no replay for non-focused flies.

### 3. Experiment bundles
- `lab/bundle.py`: zip with `protocol.yaml`, `results/`, `raw/`, `metadata.json`, `ro-crate-metadata.json` (RO-Crate 1.1, SHA-256 per
  file, the dataset named with its CC BY 4.0 license, **no license invented for the bundle**), `README.txt`; deterministic bytes.
  `--headless --rerun-bundle ZIP --out DIR` verifies hashes, refuses another pack/newer format/damage, reruns, judges **bit-exact** when
  both backends are cpu/numba/torch-cpu and the precision matches (spike SHA-256 per fly), else **statistical** by a rule fixed before
  any rerun (CI widened by 10% of the mean; single seed 25% + 0.5 Hz; these three numbers are a choice, not a measurement).
  Stimulus protocols and assays both round-trip bit-exact on the synthetic pack. Live recordings bundle as records that say why they
  can't be rerun. UI: Lab > Record and export > Bundle and a Bundle button on the Protocols page. CLI: `--protocol FILE --bundle ZIP`.
- **Not tested:** a GPU-backend rerun (none here); the real 166k-neuron pack end to end.

### 4. Share codes
- `core/sharecode.py`: `KTF1-KIND-…` Crockford base32 (no I L O U), raw-deflate JSON, 4-byte SHA-256 checksum, version byte, 1,200-char
  limit then a `.ktfshare` file. Kinds: surgery, loadout, protocol, challenge setup, Lab parameters. Every refusal has a reason
  (damaged, truncated, newer version, unknown kind/tool/cell type/parameter/arena, out of range, decompression cap). `preview` says
  what Apply will change; nothing changes before Apply. `ui/share_ui.py` = Esc > Share (make / import, Copy, Paste, Ctrl+V in text
  boxes, Save as a file); `--share-decode CODE` on the command line. 400 random single-character corruptions (seeded) never crash and are
  refused.
- **Gamepad:** the general menu has never been pad-navigable, and import needs paste, so Share is mouse/keyboard.

### 5. Neuron of the Day
- `core/neuron_of_day.py` + `extras3`: a launch card (after the tutorial, once per launch), curated facts only, picked by date (every
  entry once per cycle, reshuffled each cycle), own setting `brain.neuron_of_day` (default ON), own "Don't show again", separate from
  the real-science cards. **Try it** sets up surgery or the Lab laser (replaces your surgery / laser target, and says so first); it
  also lives in the Neurodex panel (mouse, and the pad trigger).

## Decisions you should look at
- **D conflict.** D is walk-right in 3D. I kept your D default and let both exist: in 3D the Neurodex opens on D only while the mouse is
  free (Tab); `Config.conflicts()` allows that one pair; `Config.actions_for()` is new. Old configs that already use D for something
  else keep it and start with the Neurodex unbound (tested). The alternative is a different default key.
- **Kill cam key is `;`:** every letter was taken and `c` is crouch in 3D.
- **`p < 0.001` instruction.** At n = 10 the smallest two-sided exact Wilcoxon p is 2/1024 = 0.00195, so printing "p < 0.001" there
  would be wrong, and the existing `labstats.fmt_p` prints `p = 0.002`. Day 1 adds no statistical test, so I changed nothing and
  report no such p; please decide what you want.
- Pause menu buttons got slightly smaller to fit two more entries.
- The playthrough bot has four new checks (`extra:neurodex`, `extra:killcam`, `extra:share-codes`, `extra:bundle-rerun`) that only run
  when the adult pack exists; I ran them through tests on the synthetic pack.

## Larva (the one real-data check)
Testing against the real larva pack found two things I fixed: its `nt` is the **game's sign rule** (`nt_source == "inferred"`, constant
0.8), so the larval Neurodex shows **no transmitter** instead of passing a rule off as data; and its 346 "unassigned" neurons aren't a
type, so they are skipped. The larval dex has 17 cell classes (Winding 2023's annotation classes); `test_larva_dex_discovers_by_
stimulation_on_the_real_larval_brain` drives KC on the real larval brain and discovers it, tagged as stimulated. The windowed game
still runs only the adult brain, so the larva list fills from code only (bot, API, tests).

## Test results
| run | result |
|---|---|
| `pytest -m "not validation"`, baseline on untouched `release/3.0` | 270 passed, 18 failed, 5 errors, 150 skipped (all the failures are missing-data `FileNotFoundError`s, plus `test_benchmark::test_run_benchmark_fast`, a `TypeError` that was already failing) |
| same, this branch | **434 passed**, 18 failed, 5 errors, 151 skipped: **the failing set is identical to the baseline** (diffed), +164 new tests |
| `pytest -m validation` | 25 skipped (needs the pack) |
| `--selftest` | 14 PASS, 3 FAIL: adult pack, backends and smoke, all "brain pack not found"; GPU, GL, audio, ffmpeg, folders and the new Neurodex check PASS |
| `--headless --playthrough all --sim-backend cpu` | 55 passed, 0 failed, 67 gated by design, 1 skipped (adult: no pack); larva runs on real data |
| bundle round trip | stimulus protocol and assay protocol both bit-exact on the synthetic pack (tests), tamper/mismatch/damage/wrong-pack/newer-format refusals tested |
| `tools/i18n_sync.py --check` | OK (new strings are English only; German falls back) |
| screenshots | 2D pause menu, Neurodex (list, entry, undiscovered), Share, kill cam, launch card, toast, Lab export and assumptions pages were rendered on the synthetic pack and looked at; overlaps found that way are fixed |

## What I did NOT verify
- Anything on the real adult connectome (validation, discovery behavior, the real kill cam, real rerun, real table build time and
  memory: built in chunks, but never run at 166k neurons x ~10M synapses).
- The 3D game visually (no GL render here): `Game3D` is constructed and its key/pad handlers and overlays run in tests, but nobody
  looked at a 3D frame.
- A real gamepad; a real clipboard (tests use an in-memory one; `pygame.scrap`/SDL clipboard paths are untested here); ffmpeg MP4 output
  of the kill cam (the recorder starts and stops cleanly; the GIF/MP4 encoding was not inspected).
- The exe/AppImage builds with `--collect-data kickthefly`.
- GPU backends (torch/gl) with any of this. No GPU test was run beyond `--selftest`'s own probe.
- The facts against the real pack's type names (test is written, skips here), and LC4/MBON/E-PG points noted above.

## For Opus
Please: run `--validate` before/after; run with the real pack the test above and a real play session (discoveries per session,
kill cam on a real death, the table build time on first Neurodex open); try to break share-code import (hand-edited payloads) and
bundle rerun (truncated/odd zips); look at 3D frames; review `extras3.tick` threading (the table builds on a daemon thread and sets
`tracker`/`table_state` that the game thread reads).
