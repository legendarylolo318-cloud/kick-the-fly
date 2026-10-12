//! Fly vs fly duel: the engine is `fly_sim::duel` (Python `game/flyduel.py`, verified frame for frame, see PORTING.md). This module keeps the
//! names the game's other modules use.

pub use fly_sim::duel::{DEFAULT_SECONDS, DuelResult, Side, TICK_STEPS, TIEBREAK_REMATCHES, run_duel};

/// Rad/s of angular expansion: below `LOOM_MIN` nothing, `LOOM_MIN + LOOM_FULL` = full drive (the 3D game's looming input uses these too).
pub const LOOM_MIN: f32 = fly_sim::duel::LOOM_MIN as f32;
pub const LOOM_FULL: f32 = fly_sim::duel::LOOM_FULL as f32;
