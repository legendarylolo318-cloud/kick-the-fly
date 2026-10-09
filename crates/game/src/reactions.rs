//! Reactions read from the descending neurons: which of the fly's moves its brain's firing triggers. Plain logic, no engine, no brain
//! thread: it takes a [`Levels`] lookup (the snapshot's `levels`) and a clock and says what to do. Python: `game/kick3d.py:update3d`
//! (the "reactions read from the descending neurons" block) and `game/kick_the_fly.py:{readouts,_sleep}`.
//!
//! REAL here means live descending-neuron firing crossed a threshold (CONNECTOME); how the body then moves is game physics, and the
//! thresholds and hold times are GAME RULES (the Python comments record how each was probed against calm baselines).

use crate::rng::Rng;

/// `THRESH`: x calm baseline.
pub const T_GROOM: f32 = 4.0;
pub const T_JUMP: f32 = 3.0;
pub const T_RUN: f32 = 2.4;
pub const T_KICK: f32 = 2.0;
pub const T_WALK: f32 = 3.0;
pub const T_BACK: f32 = 3.8;
pub const T_TURN: f32 = 2.1;
pub const T_FLY: f32 = 1.58;
pub const T_ESCAPE: f32 = 4.0;
pub const T_FIRE: f32 = 3.0;
pub const T_SONG: f32 = 1.8;
pub const T_SLEEP: f32 = 2.0;

/// A lookup of group levels (multiples of the calm baseline) by name.
pub struct Levels<'a> {
    pub names: &'a [String],
    pub values: &'a [f32],
}

impl Levels<'_> {
    pub fn get(&self, name: &str) -> f32 {
        self.names.iter().position(|n| n == name).and_then(|i| self.values.get(i)).copied().unwrap_or(0.0)
    }
}

/// What the body is able to do right now (the Python `can_fly`, `free`).
#[derive(Debug, Clone, Copy, Default)]
pub struct Body {
    pub can_fly: bool,
    /// Not held and not already in an escape.
    pub free: bool,
    pub grabbed: bool,
    pub asleep_ok: bool,
    pub eating: bool,
    pub duel_focus: bool,
}

/// The ready-times of one fly's reactions (seconds on the game clock).
#[derive(Debug, Clone, Default)]
pub struct ReactState {
    pub escape_ready: f32,
    pub stun_until: f32,
    pub escape_until: f32,
    pub walk_until: f32,
    pub back_until: f32,
    pub flail_until: f32,
    pub turn_ready: f32,
    pub groom_ready: f32,
    pub song_ready: f32,
    pub asleep_until: f32,
    /// The ps1 level read over about a second (two neurons are noisy).
    pub song_slow: f32,
    pub eating_until: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Reaction {
    /// Giant fiber: the dodge.
    Dodge(f32),
    /// Head-touch DNs: fly away.
    Yikes(f32),
    /// DNg02: take off and wander for `seconds`.
    TakeOff {
        level: f32,
        seconds: f32,
    },
    Run(f32),
    Kick(f32),
    BackUp(f32),
    Walk(f32),
    /// Turn by `radians` (positive right).
    Turn {
        right: bool,
        radians: f32,
        diff: f32,
    },
    Groom(f32),
    Song(f32),
    Sleep(f32),
}

impl Reaction {
    /// The first word of the on-screen note.
    pub fn word(&self) -> &'static str {
        match self {
            Reaction::Dodge(_) => "DODGE",
            Reaction::Yikes(_) => "FLY AWAY",
            Reaction::TakeOff { .. } => "TAKE OFF",
            Reaction::Run(_) => "RUN",
            Reaction::Kick(_) => "KICK",
            Reaction::BackUp(_) => "BACK UP",
            Reaction::Walk(_) => "WALK",
            Reaction::Turn { .. } => "TURN",
            Reaction::Groom(_) => "GROOM",
            Reaction::Song(_) => "SONG",
            Reaction::Sleep(_) => "SLEEP",
        }
    }
}

/// One tick of the reaction logic for one fly. Mutates the ready-times and returns what fired, in the order Python checks them.
pub fn decide(lv: &Levels, st: &mut ReactState, body: Body, now: f32, rng: &mut Rng) -> Vec<Reaction> {
    let mut out = Vec::new();
    let (escape, jump, fly) = (lv.get("escape"), lv.get("jump"), lv.get("fly"));
    if escape > T_ESCAPE && now >= st.escape_ready && !body.grabbed && body.can_fly {
        st.stun_until = 0.0;
        st.escape_until = now + 2.4;
        st.escape_ready = now + 2.4 + 1.0;
        out.push(Reaction::Dodge(escape));
    } else if jump > T_JUMP && now >= st.escape_ready && body.free && body.can_fly {
        st.stun_until = 0.0;
        st.escape_until = now + 2.4;
        st.escape_ready = now + 2.4 + 1.0;
        out.push(Reaction::Yikes(jump));
    } else if fly > T_FLY
        && now >= st.escape_ready
        && body.free
        && body.can_fly
        && now >= st.stun_until
        && now >= st.eating_until
        && now >= st.asleep_until
    {
        let seconds = rng.uniform(2.5, 4.0);
        st.escape_until = now + seconds;
        st.escape_ready = now + seconds + 1.0;
        out.push(Reaction::TakeOff { level: fly, seconds });
    }
    let run = lv.get("run");
    if run > T_RUN && now >= st.walk_until && body.free {
        st.stun_until = st.stun_until.min(now + 0.2);
        st.walk_until = now + 1.1;
        st.back_until = 0.0;
        out.push(Reaction::Run(run));
    }
    let kick = lv.get("kick");
    if kick > T_KICK && now >= st.flail_until {
        st.flail_until = now + 0.6;
        out.push(Reaction::Kick(kick));
    }
    let (back, walk) = (lv.get("back"), lv.get("walk"));
    if back > T_BACK && now >= st.back_until && now >= st.walk_until {
        st.back_until = now + 0.8;
        out.push(Reaction::BackUp(back));
    } else if walk > T_WALK && now >= st.walk_until && now >= st.back_until && now >= st.asleep_until {
        st.walk_until = now + 1.2;
        out.push(Reaction::Walk(walk));
    }
    let diff = lv.get("turn_r") - lv.get("turn_l");
    if diff.abs() > T_TURN && now >= st.turn_ready && body.free && now >= st.walk_until && !body.duel_focus {
        st.turn_ready = now + 1.5;
        out.push(Reaction::Turn { right: diff > 0.0, radians: rng.uniform(0.9, 1.6).copysign(diff), diff });
    }
    // readouts with no body animation of their own
    let groom = lv.get("groom");
    if lv.get("wind") > 2.0 && groom > T_GROOM && now >= st.groom_ready {
        st.groom_ready = now + 4.0;
        out.push(Reaction::Groom(groom));
    }
    st.song_slow += (lv.get("song") - st.song_slow) / 60.0;
    if st.song_slow > T_SONG && now >= st.song_ready {
        st.song_ready = now + 1.8;
        out.push(Reaction::Song(st.song_slow));
    }
    let sleep = lv.get("sleep");
    if sleep > T_SLEEP && body.free {
        if now >= st.asleep_until {
            out.push(Reaction::Sleep(sleep));
        }
        st.asleep_until = now + 2.0;
    }
    out
}

/// The tags the game keeps on screen: which reactions are triggered by the connectome sim's own neurons (REAL) and which by a game rule.
pub fn reaction_source(word: &str) -> Option<&'static str> {
    match word {
        "DODGE" | "FLY AWAY" | "TAKE OFF" | "RUN" | "KICK" | "BACK UP" | "WALK" | "TURN" | "SHOOT" | "GROOM" | "PROBOSCIS" | "SONG"
        | "CO2" | "HEAT" | "COLD" => Some("real"),
        "EATING" | "TO LIGHT" | "AVOID" | "APPROACH" | "FLEE" | "WRAPPED" | "BROKE FREE" | "DIED" | "HIT YOU" | "YOU DIED"
        | "AUTOPILOT" | "ROLL" | "CRAWL" | "SLEEP" | "LUNGE" => Some("rule"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(vals: &[(&str, f32)], st: &mut ReactState, body: Body, now: f32) -> Vec<Reaction> {
        let names: Vec<String> = vals.iter().map(|v| v.0.to_string()).collect();
        let values: Vec<f32> = vals.iter().map(|v| v.1).collect();
        decide(&Levels { names: &names, values: &values }, st, body, now, &mut Rng::new(1))
    }

    const FREE: Body = Body { can_fly: true, free: true, grabbed: false, asleep_ok: true, eating: false, duel_focus: false };

    #[test]
    fn the_giant_fiber_dodges_then_waits_for_its_cooldown() {
        let mut st = ReactState::default();
        assert_eq!(run(&[("escape", 6.0)], &mut st, FREE, 1.0), vec![Reaction::Dodge(6.0)]);
        assert!(run(&[("escape", 6.0)], &mut st, FREE, 2.0).is_empty(), "cooldown 3.4 s after the dodge started");
        assert_eq!(run(&[("escape", 6.0)], &mut st, FREE, 4.5).len(), 1);
        assert!(run(&[("escape", 3.9)], &mut ReactState::default(), FREE, 1.0).is_empty(), "below 4x calm nothing happens");
    }

    #[test]
    fn a_held_fly_cannot_dodge_and_a_walled_in_one_cannot_take_off() {
        let held = Body { grabbed: true, free: false, ..FREE };
        assert!(run(&[("escape", 9.0)], &mut ReactState::default(), held, 1.0).is_empty());
        let stuck = Body { can_fly: false, ..FREE };
        assert!(run(&[("escape", 9.0), ("jump", 9.0), ("fly", 3.0)], &mut ReactState::default(), stuck, 1.0).is_empty());
    }

    #[test]
    fn walking_backing_and_turning_follow_their_thresholds() {
        let mut st = ReactState::default();
        assert_eq!(run(&[("walk", 3.5)], &mut st, FREE, 0.0), vec![Reaction::Walk(3.5)]);
        assert!(run(&[("walk", 3.5)], &mut st, FREE, 0.5).is_empty(), "walks for 1.2 s");
        let mut st = ReactState::default();
        assert_eq!(run(&[("back", 4.0), ("walk", 5.0)], &mut st, FREE, 0.0), vec![Reaction::BackUp(4.0)], "backing wins");
        let mut st = ReactState::default();
        let r = run(&[("turn_r", 4.0), ("turn_l", 1.0)], &mut st, FREE, 0.0);
        assert!(matches!(r[0], Reaction::Turn { right: true, radians, .. } if (0.9..=1.6).contains(&radians)));
        let r = run(&[("turn_r", 1.0), ("turn_l", 4.0)], &mut ReactState::default(), FREE, 0.0);
        assert!(matches!(r[0], Reaction::Turn { right: false, radians, .. } if radians < 0.0));
    }

    #[test]
    fn sleep_holds_for_2_s_and_blocks_spontaneous_walking() {
        let mut st = ReactState::default();
        assert_eq!(run(&[("sleep", 3.0)], &mut st, FREE, 0.0), vec![Reaction::Sleep(3.0)]);
        assert!(run(&[("sleep", 3.0), ("walk", 5.0)], &mut st, FREE, 1.0).is_empty(), "still asleep; no walk");
        assert_eq!(st.asleep_until, 3.0);
    }

    #[test]
    fn groom_needs_the_antennae_and_song_is_read_over_a_second() {
        let mut st = ReactState::default();
        assert!(run(&[("groom", 6.0)], &mut st, FREE, 0.0).is_empty());
        assert_eq!(run(&[("groom", 6.0), ("wind", 3.0)], &mut st, FREE, 0.0), vec![Reaction::Groom(6.0)]);
        let mut st = ReactState { song_slow: 1.0, ..ReactState::default() };
        let mut fired = None;
        for i in 0..120 {
            if !run(&[("song", 3.0)], &mut st, FREE, i as f32 / 60.0).is_empty() {
                fired = Some(i);
                break;
            }
        }
        assert!(fired.unwrap() > 10, "the slow average takes a moment to cross 1.8x");
    }

    #[test]
    fn tags_match_the_python_table() {
        assert_eq!(reaction_source("DODGE"), Some("real"));
        assert_eq!(reaction_source("WRAPPED"), Some("rule"));
        assert_eq!(reaction_source("NOPE"), None);
    }
}
