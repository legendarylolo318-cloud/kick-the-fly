# Bugs and notes found while playing the Rust build

Fill this in as you play, then give it to Claude Opus together with `REVIEW_PROMPT.md`. Copy the block for each new item.

```
## N. short title
- Build: AppImage / cargo run   (date, branch rust-4.0, commit from `git log -1 --oneline`)
- Where: menu page / arena / tool / mode (Play, Lab, Pet)
- Did:        1) ...  2) ...
- Expected:   ...
- Happened:   ...
- Terminal / crash output: ...
- Python 3.1.1 does this differently?  yes / no / not checked
- Severity: crash / wrong result / looks wrong / minor
```

## Found so far

## 1. Big brain view: cannot zoom in and out any more
- Build: rust-4.0. Where: B (big brain view), mouse free (Tab / free_mouse).
- Expected: the mouse wheel over the big view zooms (Python: `view.zoom_by(1.15 if ev.y > 0 else 0.87)` in `game/kick3d.py` MOUSEWHEEL, only when the mouse is free and the pointer is inside `big_rect`); drag orbits, Shift+drag pans.
- Happened: no zoom (reported by the user, who could zoom before).
- Lead: `crates/game/src/brainview.rs` keeps `BrainView.zoom` (init 0.8, fed to the shader at `params.rot`) but I found nothing that reads `MouseWheel` and
  changes it (`grep -rn "\.zoom\|MouseWheel" crates/game/src` shows only reads); `tools.rs:282` uses the wheel to cycle tools. Check whether zoom/pan were ever wired
  or got lost, and whether the inspector's `project()` and `shapes_play` projection agree with the zoom and pan once they change.

## 2. Several flies: the brain panel does not auto-switch to the nearest fly
- Build: rust-4.0. Where: N spawns more flies, walk toward another one.
- Expected: the brain shown (panel, HUD, big view) follows the fly nearest to you unless you chose one with F (Python `_update_focus`).
- Happened: it stays on the first fly (reported by the user).
- Lead: `crates/game/src/flies_play.rs:112 follow_nearest` exists; it returns early when `arcade.open`, `now < flies.manual_until` (set 5 s after F),
  every fly dead, or the snapshot has a challenge/training in progress (`s.challenge.is_some()`, `memory.train.is_some()`): check those conditions are not
  stuck true; it measures the *player body* transform (`Player`, y 0.7) to each thorax. Check `Command::FocusFly` really switches the snapshot's focus brain
  and that `Flies.focus` and the sim thread's focus cannot disagree. Compare with Python `_update_focus` in `game/kick_the_fly.py`.

## 3. The held tool and its swing animation are not visible (swatter and the others)
- Expected: first person shows the held tool (Python `Game3D.draw_viewmodel`, `legacy-python/kickthefly/game/kick3d.py:3211`) and animates it on use (swat swing, etc.).
- Lead: `grep -rni viewmodel crates/game/src` finds nothing; `ToolState.swing_t` (tools.rs) is set on a swat but nothing draws it. Likely never ported: port
  `draw_viewmodel` for every tool (hand, flick, swatter, bomb, torch, ...) and its animation timing, matching the Python shapes/positions.

## 4. The fly looks like a generic fly, not a fruit fly (Drosophila)
- Expected: Python's 3D fly model (`kick3d.py` fly drawing: body parts, red eyes, wing shape, bristles/colouring). Rust: `crates/game/src/fly.rs:spawn_fly`
  and `fly_visuals.rs`: ellipsoid body parts with a PD ragdoll; legs/wings are visual only. Compare proportions, eye colour/size, wing outline and tint,
  stripes on the abdomen, leg count/segments with the Python draw code and fix the model (no new features: match).

## 5. FPS counter does nothing (Settings > Graphics > show fps)
- Lead: `graphics.show_fps` is never referenced in the Rust code; only F3 / `KTF_PROFILER` toggles the profiler overlay (`profiler.rs`,
  `Settings.show_profiler`). Python's plain FPS counter is separate from the profiler. Implement what the setting did in Python (find it in
  `legacy-python`: grep `show_fps`), including `graphics.fps_cap` and `graphics.vsync` if they are unwired too (also never referenced).

## 6. UI scale does nothing (Settings > Graphics > ui_scale)
- Lead: `graphics.ui_scale` is read only in `hud_canvas.rs:128` -> `ktf_ui::layout::compute_layout` (HUD layout). Check it takes effect live, that the
  menu pages scale too (Python scales both), and that `graphics.menu_size`, `access.larger_text`, `graphics.resolution_scale` do what Python did.

## 7. "Not running properly on the GPU" (reported by the user, no details yet)
- Unclear what exactly: wrong adapter (integrated instead of the dedicated GPU), low fps/stutter, rendering glitches, or crash. Ask the user for the
  symptom if it cannot be told from the logs. Cheap evidence first: the adapter line in the startup log (`bevy_render` ... `AdapterInfo`), whether
  `WgpuSettings.power_preference` is set to `HighPerformance` (grep found no `PowerPreference`/`WgpuSettings` in `crates/game/src`, so it is probably
  the default, which on hybrid laptops can pick the integrated GPU), `KTF_PROFILER=1` frame times, and `--smoke 5` steps/s. Also check a clear error
  (not a panic) when no adapter exists.

## 8. The fly does not react when sugar is put down
- Where: the SUGAR tool (hotbar 0 / sugar cube), any arena. Expected (Python `_sugar` in `game/kick_the_fly.py` ~5850 and the 3D equivalent): the nearest
  fly walks to the cube, eats it (taste + PAM reward pokes, "YUM!" popups, `eating_until`), health +0.15; the sugar smell source draws it.
- Lead: `crates/game/src/tools.rs:fn treats` (~970-1070). It picks the nearest `Treat`, walks only if `on_floor && now >= stun_until && landed`
  where `landed = velocity < 0.2`, and eats only when within 0.2 m of the head in xz AND `on_floor`. Check: the cube spawns with a velocity that never
  settles (physics jitter), `thx.y < STAND3 + 0.2` is never true, `now >= escape_until` is stuck, the fly is held by `f.wrapped/grabbed/frozen`, or
  `Treat` entities are not found because of a query filter. Python also pokes the `sweet` sugar-taste group (`poke("sweet", None, t_gain, recruit=0.6)`),
  which Rust does not (only `taste` + `reward`): that affects the brain's response, not the walking. Add a test with a spawned treat and a standing fly.

## Mechanical hint for the settings audit
Setting keys of `crates/ktf_data/data/schema.json` whose name is never mentioned in `crates/game`, `crates/ktf_ui`, `crates/ktf_data` code (apart from the
schema itself), as of commit 82a56ae. Some are false positives (read through a helper or `format!`: `brain.brain` via `cfg.larva()`, `stream.allow_*` via
`format!("stream.allow_{n}")`), all others have NO effect in the Rust game and need wiring (or a documented reason). 34 of 75:
graphics.resolution_scale, fps_cap, show_fps, vsync, backend, brain_view, panel_mode, photo_focus, timelapse_speedup, timelapse_format,
timelapse_target; audio.buzz, sfx, sonify, sonify_vol, sonify_song; stream.allow_sonify; brain.brain*, real_vs_rule, science_popups,
autopilot_orbit, song_buzz, lunge, day_night, dtype, backend, process; controls.gamepad, pad_look_speed, pad_deadzone, pad_invert_y.
Regenerate the list with the script in `REVIEW_PROMPT.md` (section SETTINGS AUDIT) after your fixes.

## Review of 2026-10-11 (laptop: i7-13620H, Intel UHD + RTX 4060 Laptop, Wayland): what was found and done for 1-8

1. **Big view zoom: fixed** (6f74fb2). The wheel never reached `BrainView.zoom`. The view now has Python's camera: wheel 1.15 / 0.87 clipped to
   0.2-5x, drag turns *and tilts* (0.4 degrees a HUD pixel), Shift+drag or middle-drag pans, 1 / 2 / 3 / 0 = front / side / top / reset. The
   inspector's picking and the skeleton overlay use the same `BrainView::turn` as the shader. The wheel cycles tools only while the mouse is
   captured (`tools.rs:select` returns early otherwise), so the two never fight. Changed on purpose: the cloud no longer turns by itself
   (Python's turns only with Autopilot + 'brain orbit', 12 degrees a second), and R no longer resets the view (R is respawn; 0 resets).
2. **Focus follows the nearest fly: it does** (measured with `KTF_DEBUG=1 KTF_FLIES=2`: game and sim focus went 2 -> 1 as fly 1 came nearest;
   the status card shows `flies 3/16 · #2`). No early return is stuck and `Flies.focus` and the sim thread agree. What was really wrong and
   probably what it looked like: the zapper, torch, cans, cVA, decoys, treats and predators poked the *focused* brain whichever fly they touched,
   so the panel lit up for the wrong fly. **Fixed** (7768e4d): they poke the touched fly's own brain (`Fly.slot`, `Command::PokeFly`).
   If the panel still seems not to follow, please write down which readout (health bar, state word, `#n`, the brain picture) and what you did.
3. **Held tool: fixed** (db54b74, e219da9). `Game3D.draw_viewmodel` was never ported; `viewmodel.rs` ports it shape for shape with its animations.
4. **Fruit fly: fixed** (7c285c0). `_draw_fly` ported shape for shape (tan body, banded abdomen with a dark tip, scutellum, big red eyes with
   glints, antennae, mouthparts, bristles, clear ellipsoid wings, two-tone legs with knee and foot beads), colours through a port of `k2.shade`.
5. **FPS counter, FPS cap, VSync: fixed** (6b2e502).
6. **UI scale: works exactly as in Python, and that is why it looks dead.** `compute_layout` is the same code in both. In 'crisp' menu size the
   scale snaps to a whole number and is then limited to window height / 760: at 1080p every value from 100 % to 145 % gives the same picture,
   at 1600p every value from 75 % to 145 %. Below 100 % at 1080p (and anywhere in 'large' menu size) it shrinks smoothly. Not changed (parity).
   The menu pages use the same scale as the HUD; `access.larger_text` and `graphics.menu_size` are read and live.
7. **GPU**: the adapter is right on this laptop (`AdapterInfo { name: "NVIDIA GeForce RTX 4060 Laptop GPU", backend: Vulkan }`; Bevy asks for
   `HighPerformance` by default). The GPU is nearly idle (main 3D pass 0.26 ms): the frame is CPU-bound, see the performance notes in
   `continue.md`. Before this review VSync was always on whatever the setting said. Still needed from you: the symptom (low fps? stutter?
   glitches? which machine?).
8. **Sugar: fixed** (7768e4d). The cube was a physics ball that rolled on for seconds, and the fly only walks to a treat that has landed; it is
   now pinned where it first touches the floor (`_sugar3d`). The `sweet` taste poke (recruit 0.6) was missing and is back.

## Found during the review (not reported by hand)

9. **Profiler overlay (F3)**: 'render 0.0 fps frame 0.00 ms' (the frame-time diagnostics plugin was never added) and the overlay sat *under* the
   HUD cards. Fixed; it also lists CPU sections now (HUD state / draw / copy, brain view).
10. **Settings with no effect** (found by the audit, fixed in 499d9c7): `audio.sfx`, `audio.buzz`, `audio.sonify`, `audio.sonify_vol`,
    `audio.sonify_song`, `stream.allow_sonify`, `brain.song_buzz`, `graphics.panel_mode`, `brain.day_night`, `brain.autopilot_orbit`. Still without
    effect, with the reason: see the table 'Settings audit' in PORTING.md.
11. **Twitch reader could freeze the game for up to 10 s** when Streamer mode was switched off (or the game closed) during a DNS lookup or a
    connection attempt: `stop()` joined the thread. Fixed (4a2b136).
12. **CI could not have run**: `ci.yml` only triggered on `main` and `bevy-port` (the branch is `rust-4.0`), and its test job would link about
    60 fat-LTO test binaries on a 4-core runner inside 90 minutes. Fixed (c55727b); still never run on GitHub.
13. Missing from the port and not written anywhere before: the **PROBOSCIS reaction** (MN9 over an eating bout, proboscis drawn out for 1.5 s),
    **aggression lunges** (`brain.lunge`), the **'Real flies do this too' cards** (`brain.science_popups`), the **spectator behaviour of Autopilot**
    (only the HUD hiding and the orbit exist), the cap of **3 sugar / 3 alcohol** items at a time. Not fixed (no new features in this pass).
14. **First CI runs (2026-10-12)**. `ci.yml` was invalid YAML (an unquoted `: ` in a step name): no job had ever started. After that:
    - Linux (glibc 2.35): the noise bank differed from NumPy's in a few values, so every spike golden failed. Cause: the float ziggurat's tail
      calls `log1pf`, which only recent glibc rounds correctly. Fixed in `fly_sim/src/rng.rs` (through double precision); the whole test step
      passes on Ubuntu 22.04 since.
    - Windows: a byte-compared fixture was checked out with CRLF (`.gitattributes` now forbids conversion); the duel engine differs from Python
      in the last bit (Windows' `sin`/`cos`/`atan2`/`hypot`; no portable library reproduces glibc, the `libm` crate was tried), so
      `duel_golden` is bit-exact on Linux and 1e-9 elsewhere; the wind-compass probe's circular correlation is numerically arbitrary (the mean
      of evenly spread wind directions is a zero vector) and comes out 0.396 on Windows against the published 0.470, so that one number is
      only compared on Linux; the Graphics settings tab has a Linux-only row, so its picture is only compared on Linux.
    **Meaning for players:** on Windows the brain's spikes equal Linux's (same noise, same sim), but a duel can differ in the last digit and so,
    rarely, in its outcome, and Lab > Validation shows a different 'direction tracking' number for the wind compass than the published one.
15. **Done on request, 2026-10-12** (the open findings of the review): the PROBOSCIS reaction and aggression lunges (01cad77), the 3 + 3 treat cap
    (85eed16), the 'Real flies do this too' cards (308ecdc), Autopilot as a spectator with Y and `--autopilot` (c28ea0d), the gamepad
    (ec9539d), the HUD's drawing cost (ecd11f8: 4.8 -> 2.8 ms a frame, no copy), the X11 pointer grab and Display backend = x11, the memory
    golden's MBON tolerance (5 % -> 1e-4). **Not done:** `graphics.resolution_scale` and `graphics.photo_scale` (both need the 3D view drawn
    into an off-screen picture of another size), gamepad rebinding by button press and menu navigation with the pad.
16. **Open: about 4 ms a frame appeared with the science cards / Autopilot / gamepad commits** (308ecdc..ec9539d). Measured back to back at
    1080p, uncapped: 11.5 ms a frame at 01cad77, 13.0 ms at the head although the HUD draws 3 ms faster there. Ruled out with switches: the
    gamepad reading, the gilrs backend, the science-card watcher, the HUD's skip and buffer swap. Not found; no effect at the 60 fps cap.

(the user may add more below)
