//! Puppeteer mode: a puzzle mode in which you cannot hit the fly. You steer it only with the optogenetics laser and with holding neurons
//! on or off, on the fly's real cell types, to make it do ten things. Python: `game/puppeteer.py`.
//!
//! GAME RULE: nothing of yours touches the fly (every tool but the laser is locked); you pick a cell type and an effect (activate or
//! silence); a level is won when the fly does what it asks (a reaction the brain makes, or travelling some distance, or surviving some
//! looms). Score = actions (one per laser pulse that reached the fly, one per latch or surgery change, one per hint); stars: par or
//! better 3, up to par + 2 two, otherwise one. CONNECTOME: every level is solved by a real circuit (listed per level).

use crate::reactions::Reaction;

/// `(label, laser/latch target spec)`: the same real types the inspector shows, with no labels that give the answer away.
pub const PALETTE: [(&str, &str); 16] = [
    ("DNp09", "type:DNp09"),
    ("MDN", "type:MDN"),
    ("DNa01+DNa02 (right)", "side:R:DNa01,DNa02"),
    ("DNa01+DNa02 (left)", "side:L:DNa01,DNa02"),
    ("DNg02", "prefix:DNg02"),
    ("pIP10", "type:pIP10"),
    ("JO-C + JO-E", "prefix:JO-C,JO-E"),
    ("DNp01", "type:DNp01"),
    ("LPLC2 + LC4", "type:LPLC2,LC4"),
    ("FB6 + FB7", "prefix:FB6,FB7"),
    ("aDN1 + aDN2", "type:DNg62,DNge078"),
    ("MN9", "type:MN9"),
    ("PPL1", "prefix:PPL1"),
    ("PAM", "prefix:PAM"),
    ("Kenyon cells", "prefix:KC"),
    ("LC10", "prefix:LC10"),
];

/// The shadow the survive-looms level sends (GAME RULE).
pub const LOOM_EVERY_S: f32 = 3.5;
pub const LOOM_FOR_S: f32 = 0.6;
pub const LOOM_WINDOW_S: f32 = 1.6;
/// Two stars up to par + this.
pub const PAR_SLACK: u32 = 2;

#[derive(Debug, Clone, PartialEq)]
pub enum Goal {
    /// The brain makes this reaction (the first word of the on-screen note).
    Event(&'static str),
    /// Walk `m` metres along the way the fly faces (negative: backwards).
    Travel(f32),
    /// Let `count` looming shadows pass without the `forbid` reaction.
    SurviveLoom { count: u32, forbid: &'static str },
    Sequence(Vec<Goal>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum How {
    Laser,
    Latch,
}

#[derive(Debug, Clone)]
pub struct Step {
    pub how: How,
    pub target: &'static str,
    pub silence: bool,
}

#[derive(Debug, Clone)]
pub struct Level {
    pub id: &'static str,
    pub title: &'static str,
    pub brief: &'static str,
    pub goal: Goal,
    pub par: u32,
    pub solution: Vec<Step>,
    pub start_x: f32,
    pub facing: i8,
    pub hint: &'static str,
}

fn step(how: How, target: &'static str, silence: bool) -> Step {
    Step { how, target, silence }
}

pub fn levels() -> Vec<Level> {
    use How::*;
    let lv = |id, title, brief, goal, par, solution, start_x, facing, hint| Level { id, title, brief, goal, par, solution, start_x, facing, hint };
    vec![
        lv("first_steps", "First steps", "Make the fly walk.", Goal::Event("WALK"), 1, vec![step(Laser, "type:DNp09", false)], -0.9, 1, "A single descending neuron type starts walking."),
        lv("about_face", "About face", "Make the fly turn to its right.", Goal::Event("TURN R"), 1, vec![step(Laser, "side:R:DNa01,DNa02", false)], 0.0, -1, "Steering neurons come in left and right pairs."),
        lv("long_walk", "The long walk", "Get the fly 0.9 m across the floor.", Goal::Travel(0.9), 1, vec![step(Latch, "type:DNp09", false)], -1.4, 1, "One step is short. What keeps a command on?"),
        lv("reverse_gear", "Reverse gear", "Make the fly back up half a metre.", Goal::Travel(-0.5), 1, vec![step(Latch, "type:MDN", false)], 0.5, 1, "There is a neuron named for walking backward."),
        lv("lift_off", "Lift off", "Get the fly airborne.", Goal::Event("TAKE OFF"), 1, vec![step(Laser, "prefix:DNg02", false)], 0.0, 1, "Flight power comes down a family of descending neurons."),
        lv("serenade", "Serenade", "Make the fly sing.", Goal::Event("SONG"), 1, vec![step(Latch, "type:pIP10", false)], 0.0, 1, "A command neuron specific to males."),
        lv("clean_antennae", "Clean antennae", "Make the fly groom.", Goal::Event("GROOM"), 1, vec![step(Laser, "prefix:JO-C,JO-E", false)], 0.0, 1, "What does an antenna feel with?"),
        lv("sleep_tight", "Sleep tight", "Put the fly to sleep.", Goal::Event("SLEEP"), 1, vec![step(Laser, "prefix:FB6,FB7", false)], 0.0, 1, "Sleep is in the central complex."),
        lv("nerves_of_steel", "Nerves of steel", "Let three looming shadows pass without the fly dodging.", Goal::SurviveLoom { count: 3, forbid: "DODGE" }, 1, vec![step(Latch, "type:DNp01", true)], 0.0, 1, "The dodge comes from one neuron. Switch something OFF."),
        lv(
            "detour",
            "The detour",
            "Turn the fly around, then walk it half a metre the way it now faces.",
            Goal::Sequence(vec![Goal::Event("TURN R"), Goal::Travel(0.5)]),
            2,
            vec![step(Laser, "side:R:DNa01,DNa02", false), step(Latch, "type:DNp09", false)],
            -1.2,
            -1,
            "Steer first, then move.",
        ),
    ]
}

/// The first word(s) of a reaction as the goals name them (`TURN R`, `TAKE OFF`...).
pub fn event_name(r: &Reaction) -> String {
    match r {
        Reaction::Turn { right, .. } => format!("TURN {}", if *right { 'R' } else { 'L' }),
        other => other.word().to_string(),
    }
}

pub fn stars_for(actions: u32, par: u32) -> u32 {
    if actions <= par { 3 } else if actions <= par + PAR_SLACK { 2 } else { 1 }
}

pub fn score_key(level_id: &str) -> String {
    format!("puppet_{level_id}")
}

/// Tracks one level's goal from what the game reports: reactions by name, the fly's position and heading each frame, and the time.
#[derive(Debug, Clone)]
pub struct Tracker {
    steps: Vec<Goal>,
    pub stage: usize,
    pub done: bool,
    origin: Option<(f32, f32, f32, f32)>,
    last: (f32, f32),
    pub progress: f32,
    pub message: String,
    pub looms_survived: u32,
    last_loom: Option<f32>,
    next_loom: Option<f32>,
    dodged: bool,
}

impl Tracker {
    pub fn new(goal: &Goal) -> Self {
        let steps = match goal {
            Goal::Sequence(s) => s.clone(),
            g => vec![g.clone()],
        };
        Self { steps, stage: 0, done: false, origin: None, last: (0.0, 0.0), progress: 0.0, message: String::new(), looms_survived: 0, last_loom: None, next_loom: None, dodged: false }
    }

    pub fn cur(&self) -> &Goal {
        &self.steps[self.stage.min(self.steps.len() - 1)]
    }

    pub fn start(&mut self, t: f32) {
        self.next_loom = Some(t + 2.0);
    }

    fn advance(&mut self) {
        self.stage += 1;
        self.origin = None;
        if self.stage >= self.steps.len() {
            self.done = true;
        }
    }

    /// A reaction the brain made.
    pub fn event(&mut self, name: &str, t: f32) {
        if self.done {
            return;
        }
        match self.cur().clone() {
            Goal::Event(e) if e == name => self.advance(),
            Goal::SurviveLoom { forbid, .. } if forbid == name && self.last_loom.is_some_and(|l| t - l <= LOOM_WINDOW_S + 0.4) => self.dodged = true,
            _ => {}
        }
    }

    /// Where the fly is (metres) and the unit vector it faces. Distance is what it has walked along the way it faces: each frame's step
    /// counts forward if it moved the way it faces and backward if against it. Flying is not walking: nothing is counted while airborne.
    pub fn position(&mut self, x: f32, z: f32, hx: f32, hz: f32, airborne: bool) {
        if self.done {
            return;
        }
        let Goal::Travel(m) = *self.cur() else {
            self.origin = None;
            return;
        };
        let n = hx.hypot(hz);
        let (hx, hz) = if n > 0.0 { (hx / n, hz / n) } else { (hx, hz) };
        if self.origin.is_none() {
            self.origin = Some((x, z, hx, hz)); // where the stage began, and the way it faced
            self.last = (x, z);
            self.progress = 0.0;
            return;
        }
        let step = (x - self.last.0) * hx + (z - self.last.1) * hz;
        if step.abs() < 0.5 && !airborne {
            self.progress += step; // a jump of half a metre in one frame is a teleport, not a step
        }
        self.last = (x, z);
        if (m >= 0.0 && self.progress >= m) || (m < 0.0 && self.progress <= m) {
            self.advance();
        }
    }

    pub fn loom_due(&self, t: f32) -> bool {
        matches!(self.cur(), Goal::SurviveLoom { .. }) && !self.done && self.next_loom.is_some_and(|n| t >= n) && self.last_loom.is_none()
    }

    pub fn loom_fired(&mut self, t: f32) {
        self.last_loom = Some(t);
        self.dodged = false;
    }

    /// Settle a loom once its window has passed: survived, or the count starts again.
    pub fn tick(&mut self, t: f32) {
        let Goal::SurviveLoom { count, .. } = *self.cur() else { return };
        let Some(last) = self.last_loom else { return };
        if self.done {
            return;
        }
        if self.dodged {
            self.looms_survived = 0;
            self.last_loom = None;
            self.next_loom = Some(t + LOOM_EVERY_S);
            self.dodged = false;
            self.message = "It saw that one. The count starts again.".into();
        } else if t - last >= LOOM_WINDOW_S {
            self.looms_survived += 1;
            self.last_loom = None;
            self.next_loom = Some(t + LOOM_EVERY_S - LOOM_WINDOW_S);
            self.message = format!("It did not flinch ({}/{count}).", self.looms_survived);
            if self.looms_survived >= count {
                self.advance();
            }
        }
    }

    pub fn describe(&self) -> String {
        match self.cur() {
            Goal::Travel(m) => format!("{:.2} / {:.2} m{}", self.progress.abs(), m.abs(), if *m < 0.0 { " back" } else { "" }),
            Goal::SurviveLoom { count, .. } => format!("{} / {count} shadows passed", self.looms_survived),
            _ if self.steps.len() > 1 => format!("step {} of {}", (self.stage + 1).min(self.steps.len()), self.steps.len()),
            _ => String::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn there_are_ten_levels_with_solutions_in_the_palette() {
        let all = levels();
        assert_eq!(all.len(), 10);
        let specs: Vec<&str> = PALETTE.iter().map(|p| p.1).collect();
        for l in &all {
            assert_eq!(l.par as usize, l.solution.len(), "{}: par is the author's own action count", l.id);
            for s in &l.solution {
                assert!(specs.contains(&s.target), "{}: {} is not in the palette", l.id, s.target);
            }
        }
        let ids: std::collections::HashSet<_> = all.iter().map(|l| l.id).collect();
        assert_eq!(ids.len(), 10);
    }

    #[test]
    fn stars_follow_par() {
        assert_eq!((stars_for(1, 1), stars_for(3, 1), stars_for(4, 1)), (3, 2, 1));
        assert_eq!(score_key("lift_off"), "puppet_lift_off");
    }

    #[test]
    fn an_event_goal_is_won_by_that_reaction_only() {
        let mut t = Tracker::new(&Goal::Event("WALK"));
        t.event("TURN R", 1.0);
        assert!(!t.done);
        t.event("WALK", 2.0);
        assert!(t.done);
    }

    #[test]
    fn walking_counts_forward_backing_counts_negative_and_teleports_and_flight_do_not() {
        let mut t = Tracker::new(&Goal::Travel(0.5));
        t.position(0.0, 0.0, 1.0, 0.0, false);
        for i in 1..=30 {
            t.position(i as f32 * 0.02, 0.0, 1.0, 0.0, false);
        }
        assert!(t.progress >= 0.5 && t.progress < 0.56 && t.done);
        let mut b = Tracker::new(&Goal::Travel(-0.5));
        b.position(0.0, 0.0, 1.0, 0.0, false);
        for i in 1..=30 {
            b.position(-(i as f32) * 0.02, 0.0, 1.0, 0.0, false);
        }
        assert!(b.done, "moving against the heading is negative distance");
        let mut f = Tracker::new(&Goal::Travel(0.5));
        f.position(0.0, 0.0, 1.0, 0.0, false);
        f.position(0.4, 0.0, 1.0, 0.0, true);
        f.position(3.0, 0.0, 1.0, 0.0, false);
        assert_eq!(f.progress, 0.0, "airborne distance and a 2.6 m jump in one frame are not walking");
    }

    #[test]
    fn surviving_looms_resets_when_the_fly_dodges() {
        let mut t = Tracker::new(&Goal::SurviveLoom { count: 3, forbid: "DODGE" });
        t.start(0.0);
        assert!(!t.loom_due(1.0) && t.loom_due(2.0));
        t.loom_fired(2.0);
        t.tick(3.0);
        assert_eq!(t.looms_survived, 0, "not settled until the window has passed");
        t.tick(3.7);
        assert_eq!(t.looms_survived, 1);
        t.loom_fired(5.0);
        t.event("DODGE", 5.3);
        t.tick(5.4);
        assert_eq!(t.looms_survived, 0);
        assert!(t.message.contains("starts again"));
        for k in 0..3 {
            let at = 10.0 + k as f32 * 4.0;
            t.loom_fired(at);
            t.tick(at + LOOM_WINDOW_S);
        }
        assert!(t.done);
    }

    #[test]
    fn a_sequence_runs_its_stages_in_order() {
        let mut t = Tracker::new(&levels()[9].goal);
        t.event("TURN R", 1.0);
        assert!(!t.done && t.stage == 1);
        assert_eq!(t.describe(), "0.00 / 0.50 m");
    }
}
