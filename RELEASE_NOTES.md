# Kick the Fly 2.12.0: a larva brain, flies that differ, and a pet fly

Downloads: **KickTheFly.exe** (Windows) and **KickTheFly-x86_64.AppImage** (Linux). Check them against `SHA256SUMS`.
Your saves, settings and your fly's training memory carry over. Everything in
[2.11.0](https://github.com/legendarylolo318-cloud/kick-the-fly/releases/tag/v2.11.0) is in this build.

This is the 2.11 work (larval brain, fly individuality and pet mode, written by Gemini) after review. 2.11.0 was
already tagged, so it ships as 2.12.0.

## For players

**Pet mode** (`Esc > Mode`). Keep one fly across real days. Nothing runs while the game is closed: no background
process, service, autostart or timer. When you come back, the time you were away is caught up at launch (at most 7
days), and the fly gets hungry and sleepy. Feeding and rest act on its real taste, reward and sleep neurons. It can't
die unless you turn that on. The pet is saved at launch and on quit, with a backup, and a pet save is never deleted.

**Every fly is a little different** (Settings > Brain > Individuality: off / subtle / strong). Each fly gets its own
per-neuron gains on the same wiring, so two flies react differently to the same kick. **The default is subtle, so your
fly behaves a bit differently after the upgrade**; set it to off to keep the old fly.

**The larva brain is not playable yet.** Choosing the larva in the game still runs the adult fly (and logs a warning).

## For researchers

- **Larval brain, headless only.** The first-instar *Drosophila* larva connectome (Winding et al. 2023, *Science*
  379:eadd9330): 2,952 neurons, 110,677 connected pairs, 352,611 synapses. Use it with
  `--headless --validate --brain larva` or from Python. It is built on your machine from the paper's Data S1 on first
  use (SHA-256 checked) and is **not** shipped in the exe, the AppImage or git: no license is stated for it. Data S1
  has no transmitter identities, so the signs are a guess (LN and MBON inhibitory, the rest excitatory). Neither of the
  two larva validation tests passes (x0.66 vs x0.65 and x0.68 vs x0.64 of control). See docs/larva.md.
- **Individuality** is W_fly = D_post·W·D_pre with the shared matrix unchanged and signs kept; log-normal gains with
  σ 0.05 (subtle) or 0.15 (strong). NumPy and Numba are bit-exact with each other at every setting; torch-cpu only
  with it off; the gl backend does not implement it and runs with it off. It is always off in validation, so the adult
  validation results are the same as 2.11.0. Pass rates for four validated behaviors at off / subtle / strong are in
  docs/individuality.md.
- Pet needs, rates and couplings, and the individuality sigmas, are GAME RULE entries in Lab > Parameters.

## Known issues

- The windowed larva game isn't playable; it falls back to the adult brain.
- The larva "nociceptor" and "chordotonal" groups are ascending neurons, the "Basin" readout is 2nd-order PNs, and
  "_telegoro-1" is not verified as Goro.
- Personality cards show their thresholds, but the in-game metrics on them are seeded random draws, not measurements.
- In pet mode, falling asleep while you play is random (not deterministic); the catch-up at launch is deterministic.
- The individuality ICC/consistency results in the docs were run at σ 0.15 and not re-run.
- The larva fly caps (64 on NumPy, 128 on Numba/GPU) rest on a benchmark that did not reproduce; they can't be reached
  while the larva game falls back to the adult.
- The batched GPU individuality test uses a backend name that doesn't exist, so it never tests the GPU; torch-cuda and
  torch-rocm individuality are untested.
- `CHANGES_GEMINI_2.11.md` is kept as written and has claims that didn't hold (larva validation results, a license for
  the larva data, σ 0.30 for strong, torch-cpu bit-exactness, the larva benchmark); the docs and CHANGELOG have the
  measured versions.
