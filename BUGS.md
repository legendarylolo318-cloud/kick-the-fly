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

(the user said there are a few more, found by hand; add them below)
