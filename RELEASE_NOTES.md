# Kick the Fly 2.13.0: tool loadouts, a self-test and a bug report that sends nothing

Downloads: **KickTheFly.exe** (Windows) and **KickTheFly-x86_64.AppImage** (Linux). Check them against `SHA256SUMS`.
Your saves, settings and your fly's training memory carry over. Everything in
[2.12.0](https://github.com/legendarylolo318-cloud/kick-the-fly/releases/tag/v2.12.0) is in this build.

Nothing in the simulation changed: the validation results are identical to 2.12.0's.

## For players

**Tool loadouts.** The hotbar is now a loadout: up to 10 tools on keys 1-9 and 0, with - and = turning the page of a
longer one. The mouse wheel and the gamepad bumpers step through it. Pick a preset (Base, Chaos, Chemist, Lab, All,
Pet) or build your own in the **loadout editor** (Q, or X on a gamepad). The editor lists every tool with a one-line
description and the real neurons it drives, tagged CONNECTOME or GAME RULE. Drag tools to reorder them, drag one off
to remove it, and save up to five loadouts. Hold ` (or Y on a gamepad) for the **tool wheel**, which reaches every
tool. The hand is always in slot 1, and the laser is Lab-only. Every new key can be rebound; only Esc is fixed.

**If you played before, your keys still work.** An existing install is set to the All preset, which puts every tool on
the hotbar in the order the number keys always had. You see a one-time popup pointing at the editor. A fresh install
starts on Base, with a short tutorial.

**The 2D spider drops on its thread** before it hunts, as it always did in 3D, so the fly sees it coming (its
looming detectors fire) even when it's stuck to flypaper or floating in the pool.

**A fruit tool.** Drop ripe fruit. The fly eats it exactly as it eats sugar: same taste and reward neurons, same rules.

**Settings > Help** has the tutorial, a **self-test** and **Report a bug**. The self-test checks this install (brain
pack, compute backends, OpenGL, audio, folders, memory) and says in plain English what to fix. Report a bug shows
exactly what it would include, lets you remove anything, and then copies it, saves it, or opens a prefilled GitHub
issue in your browser. Nothing is uploaded, and there is no telemetry. Your home folder and user name are replaced by
placeholders. After a crash, the crash screen offers the same thing.

## For researchers and developers

- `--selftest [--out FILE]` exits 0 (all pass), 3 (warnings only) or 1 (a failure). CPU backends are compared spike
  for spike with NumPy, GPU backends statistically.
- `--bugreport` prints or saves the report without opening anything.
- `--headless --playthrough [adult|larva|all] --out DIR` runs a bot that uses every tool in every arena on each brain.
  It checks that the documented neurons fire above baseline, plus death and the autopsy, save-then-load, replay
  determinism, multi-fly, surgery, training, the duel, pet catch-up, individuality and every loadout preset.
  See docs/playthrough.md.
- The Python API has `fly.loadout`, `fly.set_loadout(...)` and `fly.use_tool(...)`. Replays record `tool` events,
  which older versions skip.
- CI adds a Claude review of pull requests (skipped cleanly without an API key), a nightly validation and playthrough
  run with a history page, and self-test and playthrough gates on releases. See docs/ci.md.

## Known issues

- The default tool wheel key ` sits on a different physical key on non-US keyboard layouts. Rebind it in Settings >
  Controls.
- Accented letters typed through dead keys or an input method don't reach loadout names.
- The German translation doesn't have the new 2.13 strings yet; they show in English.
- Frozen builds (exe and AppImage), Windows, a real gamepad and a visible window on Wayland were not tested by hand for
  this release.
