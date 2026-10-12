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

/// The thresholds the reactions read (`kick_the_fly.THRESH`, `LOOM_MIN`, `LOOM_FULL`); the Lab's parameters change them live.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Thresholds {
    pub groom: f32,
    pub jump: f32,
    pub run: f32,
    pub kick: f32,
    pub walk: f32,
    pub back: f32,
    pub turn: f32,
    pub fly: f32,
    pub escape: f32,
    pub fire: f32,
    pub song: f32,
    pub aggression: f32,
    pub sleep: f32,
    pub co2: f32,
    pub hot_pn: f32,
    pub cold_pn: f32,
    pub loom_min: f32,
    pub loom_full: f32,
}

impl Thresholds {
    pub const DEFAULT: Thresholds = Thresholds {
        groom: T_GROOM,
        jump: T_JUMP,
        run: T_RUN,
        kick: T_KICK,
        walk: T_WALK,
        back: T_BACK,
        turn: T_TURN,
        fly: T_FLY,
        escape: T_ESCAPE,
        fire: T_FIRE,
        song: T_SONG,
        aggression: 2.0,
        sleep: T_SLEEP,
        co2: 2.0,
        hot_pn: 1.6,
        cold_pn: 1.7,
        loom_min: 1.5,
        loom_full: 8.0,
    };

    /// `Lab > Parameters`: the rule named `thresh.<x>`, `loom_min` or `loom_full`. False for a name that is not a threshold.
    pub fn set(&mut self, name: &str, v: f32) -> bool {
        let slot = match name.strip_prefix("thresh.").unwrap_or(name) {
            "groom" => &mut self.groom,
            "jump" => &mut self.jump,
            "run" => &mut self.run,
            "kick" => &mut self.kick,
            "walk" => &mut self.walk,
            "back" => &mut self.back,
            "turn" => &mut self.turn,
            "fly" => &mut self.fly,
            "escape" => &mut self.escape,
            "fire" => &mut self.fire,
            "song" => &mut self.song,
            "aggression" => &mut self.aggression,
            "sleep" => &mut self.sleep,
            "co2" => &mut self.co2,
            "hot_pn" => &mut self.hot_pn,
            "cold_pn" => &mut self.cold_pn,
            "loom_min" => &mut self.loom_min,
            "loom_full" => &mut self.loom_full,
            _ => return false,
        };
        *slot = v;
        true
    }
}

static LIVE: std::sync::RwLock<Thresholds> = std::sync::RwLock::new(Thresholds::DEFAULT);

/// The thresholds in force now.
pub fn thresholds() -> Thresholds {
    *LIVE.read().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Change one rule (the Lab's parameters page); returns whether the name was a threshold.
pub fn set_rule(name: &str, v: f32) -> bool {
    LIVE.write().unwrap_or_else(std::sync::PoisonError::into_inner).set(name, v)
}

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
    /// MN9's firing rate in Hz (`br.hz("proboscis")`).
    pub mn9_hz: f32,
    /// How far the nearest other living fly is, in metres (`None`: alone, or Settings > Brain > Aggression lunges is off).
    pub rival: Option<f32>,
}

/// The ready-times of one fly's reactions (seconds on the game clock).
#[derive(Debug, Clone)]
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
    /// MN9 over the current eating bout (one value a tick) and its slow average while not eating.
    pub mn9_bout: Vec<f32>,
    pub mn9_calm: Option<f32>,
    /// The proboscis is out until then (`fly.proboscis_until`).
    pub proboscis_until: f32,
    pub lunge_ready: f32,
    /// The logged-only readouts (`slot.co2_ready`, `hot_pn_ready`, `cold_pn_ready`): at most one note every 3 s each.
    pub co2_ready: f32,
    pub hot_ready: f32,
    pub cold_ready: f32,
}

impl Default for ReactState {
    fn default() -> Self {
        ReactState {
            escape_ready: 0.0,
            stun_until: 0.0,
            escape_until: 0.0,
            walk_until: 0.0,
            back_until: 0.0,
            flail_until: 0.0,
            turn_ready: 0.0,
            groom_ready: 0.0,
            song_ready: 0.0,
            asleep_until: 0.0,
            // `getattr(slot, "song_slow", 1.0)`: the slow average starts at the calm level
            song_slow: 1.0,
            eating_until: 0.0,
            mn9_bout: Vec::new(),
            mn9_calm: None,
            proboscis_until: 0.0,
            lunge_ready: 0.0,
            co2_ready: 0.0,
            hot_ready: 0.0,
            cold_ready: 0.0,
        }
    }
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
    /// MN9 fired at this multiple of its calm rate over the first second of eating: the proboscis comes out.
    Proboscis(f32),
    /// Aggression neurons (AVLP727m, pC1) above the threshold with another fly near: a lunge at it.
    Lunge(f32),
    /// Second-order CO2, hot and cold sensory neurons: logged only, the fly does not act on them (`Game.readouts`).
    Co2(f32),
    Heat(f32),
    Cold(f32),
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
            Reaction::Proboscis(_) => "PROBOSCIS",
            Reaction::Lunge(_) => "LUNGE",
            Reaction::Co2(_) => "CO2",
            Reaction::Heat(_) => "HEAT",
            Reaction::Cold(_) => "COLD",
        }
    }
}

/// One tick of the reaction logic for one fly. Mutates the ready-times and returns what fired, in the order Python checks them.
pub fn decide(lv: &Levels, st: &mut ReactState, body: Body, now: f32, rng: &mut Rng) -> Vec<Reaction> {
    let mut out = Vec::new();
    let th = thresholds();
    let (escape, jump, fly) = (lv.get("escape"), lv.get("jump"), lv.get("fly"));
    if escape > th.escape && now >= st.escape_ready && !body.grabbed && body.can_fly {
        st.stun_until = 0.0;
        st.escape_until = now + 2.4;
        st.escape_ready = now + 2.4 + 1.0;
        out.push(Reaction::Dodge(escape));
    } else if jump > th.jump && now >= st.escape_ready && body.free && body.can_fly {
        st.stun_until = 0.0;
        st.escape_until = now + 2.4;
        st.escape_ready = now + 2.4 + 1.0;
        out.push(Reaction::Yikes(jump));
    } else if fly > th.fly
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
    if run > th.run && now >= st.walk_until && body.free {
        st.stun_until = st.stun_until.min(now + 0.2);
        st.walk_until = now + 1.1;
        st.back_until = 0.0;
        out.push(Reaction::Run(run));
    }
    let kick = lv.get("kick");
    if kick > th.kick && now >= st.flail_until {
        st.flail_until = now + 0.6;
        out.push(Reaction::Kick(kick));
    }
    let (back, walk) = (lv.get("back"), lv.get("walk"));
    if back > th.back && now >= st.back_until && now >= st.walk_until {
        st.back_until = now + 0.8;
        out.push(Reaction::BackUp(back));
    } else if walk > th.walk && now >= st.walk_until && now >= st.back_until && now >= st.asleep_until {
        st.walk_until = now + 1.2;
        out.push(Reaction::Walk(walk));
    }
    let diff = lv.get("turn_r") - lv.get("turn_l");
    if diff.abs() > th.turn && now >= st.turn_ready && body.free && now >= st.walk_until && !body.duel_focus {
        st.turn_ready = now + 1.5;
        out.push(Reaction::Turn { right: diff > 0.0, radians: rng.uniform(0.9, 1.6).copysign(diff), diff });
    }
    // readouts with no body animation of their own
    let groom = lv.get("groom");
    if lv.get("wind") > 2.0 && groom > th.groom && now >= st.groom_ready {
        st.groom_ready = now + 4.0;
        out.push(Reaction::Groom(groom));
    }
    // PROBOSCIS (`assays.PER_RATIO`): one second into an eating bout, MN9 at 1.5 times its calm rate or more
    let mn9 = body.mn9_hz;
    if body.eating {
        // only the first second of a bout is looked at; 61 marks 'already judged'
        if st.mn9_bout.len() <= 60 {
            st.mn9_bout.push(mn9);
        }
        if st.mn9_bout.len() == 60 {
            let mean = st.mn9_bout.iter().sum::<f32>() / 60.0;
            let ratio = mean / st.mn9_calm.unwrap_or(mn9).max(1.0);
            if ratio >= PER_RATIO {
                st.proboscis_until = now + 1.5;
                out.push(Reaction::Proboscis(ratio));
            }
        }
    } else {
        st.mn9_bout.clear();
        let calm = st.mn9_calm.unwrap_or(mn9);
        st.mn9_calm = Some(calm + (mn9 - calm) * 0.02);
    }
    st.song_slow += (lv.get("song") - st.song_slow) / 60.0;
    if st.song_slow > th.song && now >= st.song_ready {
        st.song_ready = now + 1.8;
        out.push(Reaction::Song(st.song_slow));
    }
    // CO2, HEAT, COLD: second-order sensory neurons, logged only
    let co2 = lv.get("co2");
    if co2 > th.co2 && now >= st.co2_ready {
        st.co2_ready = now + 3.0;
        out.push(Reaction::Co2(co2));
    }
    let hot = lv.get("hot_pn");
    if hot > th.hot_pn && now >= st.hot_ready {
        st.hot_ready = now + 3.0;
        out.push(Reaction::Heat(hot));
    }
    let cold = lv.get("cold_pn");
    if cold > th.cold_pn && now >= st.cold_ready {
        st.cold_ready = now + 3.0;
        out.push(Reaction::Cold(cold));
    }
    let sleep = lv.get("sleep");
    if sleep > th.sleep && body.free {
        if now >= st.asleep_until {
            out.push(Reaction::Sleep(sleep));
        }
        st.asleep_until = now + 2.0;
    }
    // LUNGE (`Game._aggression`): another living fly within 180 fly-pixels
    let aggr = lv.get("aggression");
    if body.free && now >= st.lunge_ready && aggr > th.aggression && body.rival.is_some_and(|d| d <= LUNGE_REACH) {
        st.lunge_ready = now + 1.8;
        out.push(Reaction::Lunge(aggr));
    }
    out
}

/// `assays.PER_RATIO`.
pub const PER_RATIO: f32 = 1.5;
/// How near the other fly must be for a lunge: 180 of the 2D game's pixels, in metres.
pub const LUNGE_REACH: f32 = 180.0 * 0.006;

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

/// `POPUP_SOURCE`: which floating words carry a REAL / RULE chip.
pub fn popup_source(word: &str) -> Option<&'static str> {
    match word {
        "DODGE!" | "YIKES!" => Some("real"),
        "ROLL!"
        | "NOPE!"
        | "RUN AWAY!"
        | "YUM!"
        | "SWEET!"
        | "NOM NOM"
        | "K.O.!"
        | "BROKE FREE!"
        | "FLY WINS!"
        | "GOTCHA!"
        | "PEW PEW!"
        | "TAKE THAT!"
        | "AUTOPILOT"
        | "SPECTATOR"
        | "PHOTO MODE"
        | "*HIC*"
        | "SIP..."
        | "GLUG!"
        | "STUMBLE!"
        | "ALL GONE"
        | "♪ BUZZ ♪"
        | "LUNGE!"
        | "ZZZ"
        | "COURTSHIP"
        | "cVA PUFF"
        | "DECOY FEMALE"
        | "A FROG!"
        | "A DRAGONFLY!"
        | "A MANTIS!"
        | "SNAP!"
        | "ZAP!"
        | "GULP!"
        | "GRABBED!"
        | "SNATCH!"
        | "STRIKE!"
        | "SWAT!"
        | "STUCK IN THE TRAP!"
        | "THE COOK RAISES A SWATTER" => Some("rule"),
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

    const FREE: Body =
        Body { can_fly: true, free: true, grabbed: false, asleep_ok: true, eating: false, duel_focus: false, mn9_hz: 0.0, rival: None };

    #[test]
    fn co2_heat_and_cold_are_only_logged_every_three_seconds() {
        let mut st = ReactState::default();
        let lv = [("co2", 2.5), ("hot_pn", 1.7), ("cold_pn", 1.0)];
        assert_eq!(run(&lv, &mut st, FREE, 0.0), vec![Reaction::Co2(2.5), Reaction::Heat(1.7)], "cold is under 1.7");
        assert!(run(&lv, &mut st, FREE, 2.9).is_empty());
        assert_eq!(run(&lv, &mut st, FREE, 3.1).len(), 2);
        // the song's slow average starts at the calm level, as in Python (`getattr(slot, "song_slow", 1.0)`)
        assert_eq!(ReactState::default().song_slow, 1.0);
    }

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
    #[test]
    fn the_proboscis_comes_out_one_second_into_a_meal_when_mn9_fires() {
        let mut st = ReactState::default();
        // calm: MN9 at 4 Hz
        for i in 0..120 {
            run(&[], &mut st, Body { mn9_hz: 4.0, ..FREE }, i as f32 / 60.0);
        }
        // eating with MN9 at 8 Hz: twice calm, so on the 60th tick it fires, once
        let mut fired = Vec::new();
        for i in 0..90 {
            let now = 2.0 + i as f32 / 60.0;
            for r in run(&[], &mut st, Body { mn9_hz: 8.0, eating: true, ..FREE }, now) {
                fired.push((i, r));
            }
        }
        assert_eq!(fired.len(), 1);
        assert_eq!(fired[0].0, 59);
        assert!(matches!(fired[0].1, Reaction::Proboscis(x) if (x - 2.0).abs() < 0.05));
        assert!(st.proboscis_until > 4.4);
        // a meal with MN9 at its calm rate: nothing
        let mut st = ReactState { mn9_calm: Some(4.0), ..Default::default() };
        for i in 0..90 {
            assert!(run(&[], &mut st, Body { mn9_hz: 4.5, eating: true, ..FREE }, i as f32 / 60.0).is_empty());
        }
    }

    #[test]
    fn a_lunge_needs_aggression_a_near_rival_and_its_cooldown() {
        let mut st = ReactState::default();
        let near = Body { rival: Some(0.8), ..FREE };
        assert!(run(&[("aggression", 3.0)], &mut st, FREE, 1.0).is_empty(), "alone");
        assert!(run(&[("aggression", 3.0)], &mut st, Body { rival: Some(1.5), ..FREE }, 1.0).is_empty(), "too far");
        assert!(run(&[("aggression", 1.5)], &mut st, near, 1.0).is_empty(), "calm");
        assert_eq!(run(&[("aggression", 3.0)], &mut st, near, 1.0), vec![Reaction::Lunge(3.0)]);
        assert!(run(&[("aggression", 3.0)], &mut st, near, 2.0).is_empty(), "1.8 s cooldown");
        assert_eq!(run(&[("aggression", 3.0)], &mut st, near, 2.9).len(), 1);
    }
}
