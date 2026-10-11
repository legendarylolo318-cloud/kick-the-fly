# Bugs and notes found while playing the Rust build

Fill this in as you play, then give it to Claude Opus together with `REVIEW_PROMPT.md`. Copy the block for each new item.

```
## N. short title
- Build: AppImage / cargo run   (date, branch rust-3.1.1, commit from `git log -1 --oneline`)
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
- Build: rust-3.1.1. Where: B (big brain view), mouse free (Tab / free_mouse).
- Expected: the mouse wheel over the big view zooms (Python: `view.zoom_by(1.15 if ev.y > 0 else 0.87)` in `game/kick3d.py` MOUSEWHEEL, only when the mouse is free and the pointer is inside `big_rect`); drag orbits, Shift+drag pans.
- Happened: no zoom (reported by the user, who could zoom before).
- Lead: `crates/game/src/brainview.rs` keeps `BrainView.zoom` (init 0.8, fed to the shader at `params.rot`) but I found nothing that reads `MouseWheel` and
  changes it (`grep -rn "\.zoom\|MouseWheel" crates/game/src` shows only reads); `tools.rs:282` uses the wheel to cycle tools. Check whether zoom/pan were ever wired
  or got lost, and whether the inspector's `project()` and `shapes_play` projection agree with the zoom and pan once they change.

## 2. Several flies: the brain panel does not auto-switch to the nearest fly
- Build: rust-3.1.1. Where: N spawns more flies, walk toward another one.
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

## Mechanical hint for the settings audit
Setting keys of `crates/ktf_data/data/schema.json` whose name is never mentioned in `crates/game`, `crates/ktf_ui`, `crates/ktf_data` code (apart from the
schema itself), as of commit 82a56ae. Some are false positives (read through a helper or `format!`: `brain.brain` via `cfg.larva()`, `stream.allow_*` via
`format!("stream.allow_{n}")`), all others have NO effect in the Rust game and need wiring (or a documented reason). 34 of 75:
graphics.resolution_scale, fps_cap, show_fps, vsync, backend, brain_view, panel_mode, photo_focus, timelapse_speedup, timelapse_format,
timelapse_target; audio.buzz, sfx, sonify, sonify_vol, sonify_song; stream.allow_sonify; brain.brain*, real_vs_rule, science_popups,
autopilot_orbit, song_buzz, lunge, day_night, dtype, backend, process; controls.gamepad, pad_look_speed, pad_deadzone, pad_invert_y.
Regenerate the list with the script in `REVIEW_PROMPT.md` (section SETTINGS AUDIT) after your fixes.

(the user may add more below)
