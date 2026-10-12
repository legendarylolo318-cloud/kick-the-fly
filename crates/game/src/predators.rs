//! Predators: the frog, the dragonfly and the mantis. Plain logic, no engine, no brain. Python: `game/predators.py`.
//!
//! The 3D game, the Lab assay and the tests all use this one engine, so a predator behaves the same everywhere. A predator is a small
//! state machine that moves a body and, when it strikes, a striking part (the tongue, the claw). It reports where those parts are each
//! frame; the game turns them into looming exactly as it does for every other object.
//!
//! What comes from the connectome and what is a game rule (the Python docstring in full): *detection* is the fly's own looming
//! detectors (LPLC2/LC4 -> DNp01) seeing the predator grow, nothing here tells the fly a predator is near; a *capture* fires the fly's
//! touch neurons by body part; the *AI* (when it waits, hops, stalks, pursues or strikes, how fast, how far, how many strikes) is GAME
//! RULE, numbers chosen for play and not measurements of real animals.
//!
//! Units are metres and seconds in the 3D game's scale (the fly's thorax stands 0.38 m up). Time advances in fixed 1/60 s steps.
//!
//! The frog's locomotion is a series of ballistic leaps: crouch, leap, airborne, land, recover. The arc comes from the game's own gravity
//! and one fixed apex height, so a longer leap is a faster, flatter-looking one in the same time; nothing ever slides along the floor.

use std::f32::consts::{PI, TAU};

use bevy::math::Vec3;

use crate::anim::ease;
use crate::rng::Rng;

pub const DT: f32 = 1.0 / 60.0;
/// How high the game's standing fly's thorax is (`kick3d.STAND3`), m.
pub const THORAX_Y: f32 = 0.38;
pub const KINDS: [&str; 3] = ["frog", "dragonfly", "mantis"];

#[derive(Debug, Clone, Copy)]
pub struct Spec {
    pub kind: &'static str,
    pub label: &'static str,
    /// Looming radius of the body, m.
    pub body_r: f32,
    /// Looming radius of the striking part (0: none), m.
    pub tip_r: f32,
    /// How far the tongue / claw reaches from the body, m.
    pub reach: f32,
    /// How close a target must be (horizontally for the frog and mantis) before it aims, m.
    pub trigger: f32,
    /// A pause before the strike, during which the predator is still.
    pub aim_s: f32,
    pub strike_s: f32,
    pub retract_s: f32,
    /// After a miss, how long before it can strike again.
    pub recover_s: f32,
    pub max_strikes: u32,
    /// The striking part grabs a fly this close to it, m.
    pub capture_r: f32,
    /// m/s: hopping (frog), stalking (mantis), pursuit (dragonfly).
    pub speed: f32,
    pub note: &'static str,
}

pub const SPECS: [Spec; 3] = [
    Spec {
        kind: "frog",
        label: "FROG",
        body_r: 0.45,
        tip_r: 0.12,
        reach: 1.8,
        trigger: 1.5,
        aim_s: 0.30,
        strike_s: 0.07,
        retract_s: 0.12,
        recover_s: 1.8,
        max_strikes: 3,
        capture_r: 0.28,
        speed: 1.4,
        note: "sits still, hops closer, then a tongue that takes 0.07 s to reach where the fly was when it aimed",
    },
    Spec {
        kind: "dragonfly",
        label: "DRAGONFLY",
        body_r: 0.20,
        tip_r: 0.0,
        reach: 0.0,
        trigger: 5.0,
        aim_s: 0.0,
        strike_s: 0.0,
        retract_s: 0.0,
        recover_s: 0.0,
        max_strikes: 1,
        capture_r: 0.30,
        speed: 4.0,
        note: "patrols high up and chases a flying fly from above; a walking fly is not prey",
    },
    Spec {
        kind: "mantis",
        label: "MANTIS",
        body_r: 0.35,
        tip_r: 0.09,
        reach: 0.9,
        trigger: 0.8,
        aim_s: 0.20,
        strike_s: 0.06,
        retract_s: 0.15,
        recover_s: 2.0,
        max_strikes: 2,
        capture_r: 0.22,
        speed: 0.10,
        note: "creeps up at 10 cm/s, freezing now and then, then strikes with a forelimb in 0.06 s",
    },
];

pub fn spec(kind: &str) -> Option<&'static Spec> {
    SPECS.iter().find(|s| s.kind == kind)
}

/// The mantis's stalk: it freezes for this long every [`STALK_RUN_S`] of walking.
pub const TELL_FREEZE_S: (f32, f32) = (0.4, 0.9);
pub const STALK_RUN_S: f32 = 2.0;
/// m/s^2: `kick3d.GRAV` (0.9 px-scale metres per frame^2) at the 60 Hz tick.
pub const GRAVITY: f32 = 0.9 * 0.006 * 3600.0;
/// How high every leap goes, m.
pub const LEAP_APEX: f32 = 0.45;
/// Seconds in the air, the same for every leap (apex and gravity are fixed).
pub fn leap_t() -> f32 {
    2.0 * (2.0 * LEAP_APEX / GRAVITY).sqrt()
}
pub const LEAP_MAX: f32 = 1.8;
pub const LEAP_MIN: f32 = 0.35;
/// A hunting frog lands this far from the fly (inside its tongue's trigger range).
pub const STANDOFF: f32 = 1.2;
pub const IDLE_S: f32 = 0.8;
pub const TRACK_S: f32 = 0.30;
pub const CROUCH_S: f32 = 0.28;
pub const TAKEOFF_S: f32 = 0.07;
pub const LAND_S: f32 = 0.16;
pub const RECOVER_S: f32 = 0.30;
/// The tongue goes this far past its aim point before it snaps back.
pub const TONGUE_OVERSHOOT: f32 = 0.08;
pub const DRAGONFLY_GIVE_UP_S: f32 = 30.0;
/// A target that lands this long stops being prey.
pub const DRAGONFLY_GROUND_PATIENCE_S: f32 = 1.0;
pub const DRAGONFLY_TURN: f32 = 5.0;
/// How long a predator takes to get out of the arena after it gives up.
pub const LEAVE_S: f32 = 3.0;
/// How long a captured fly is held before it is eaten.
pub const CARRY_S: f32 = 1.2;

/// What a capture fires: `(region, side, strength)` on the fly's real touch groups. GAME RULE: which parts, and how hard.
pub fn capture_touch(kind: &str) -> &'static [(&'static str, Option<char>, f32)] {
    match kind {
        "frog" => &[("body", None, 1.0), ("head", None, 0.6), ("legs", Some('L'), 0.5), ("legs", Some('R'), 0.5)],
        "dragonfly" => &[("legs", Some('L'), 0.9), ("legs", Some('R'), 0.9), ("wing", None, 0.9), ("body", None, 0.8)],
        _ => &[("body", None, 1.0), ("legs", Some('L'), 0.9), ("legs", Some('R'), 0.9)],
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Bounds {
    pub xmin: f32,
    pub xmax: f32,
    /// The floor.
    pub ymin: f32,
    /// The ceiling (or how high the predator may go).
    pub ymax: f32,
    pub zmin: f32,
    pub zmax: f32,
}

impl Bounds {
    pub fn room() -> Self {
        Self { xmin: -4.2, xmax: 4.2, ymin: 0.0, ymax: 3.0, zmin: -3.6, zmax: 3.6 }
    }

    pub fn clip(&self, p: Vec3, margin: f32) -> Vec3 {
        let lo = Vec3::new(self.xmin + margin, self.ymin, self.zmin + margin);
        let hi = Vec3::new(self.xmax - margin, self.ymax - margin, self.zmax - margin);
        p.max(lo).min(hi)
    }
}

/// What a predator needs to know about a fly: where its thorax is, whether it is in the air, whether it is alive.
#[derive(Debug, Clone, Copy)]
pub struct Target {
    pub id: u32,
    pub p: Vec3,
    pub flying: bool,
    pub alive: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EventKind {
    Aim,
    Strike,
    Capture,
    Miss,
    Leave,
    Gone,
}

#[derive(Debug, Clone)]
pub struct Event {
    pub kind: EventKind,
    pub target: Option<u32>,
    pub at: Option<Vec3>,
}

impl Event {
    fn new(kind: EventKind) -> Self {
        Self { kind, target: None, at: None }
    }
}

/// The body pose for the drawing; a pure function of state and time, so the drawing never decides anything.
#[derive(Debug, Clone, Copy, Default)]
pub struct FrogPose {
    /// Body low and wide.
    pub squash: f32,
    /// Long along the velocity.
    pub stretch: f32,
    /// Hind legs extended (0 folded).
    pub hind: f32,
    /// Legs pulled in under the body.
    pub tuck: f32,
    /// Front legs reaching ahead for the ground.
    pub reach: f32,
    /// Nose up, rad.
    pub pitch: f32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct MantisPose {
    /// The rocking of a stalking mantis (0 while it freezes).
    pub sway: f32,
    /// The wind-up: the body rearing back with the forelegs cocked.
    pub rear: f32,
    pub cock: f32,
    pub walking: bool,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct DragonflyPose {
    pub pitch: f32,
    pub bank: f32,
    /// Legs thrown forward into a basket as it closes on a fly.
    pub reach: f32,
    pub tuck: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    Patrol,
    Idle,
    Lurk,
    Track,
    Crouch,
    Leap,
    Airborne,
    Land,
    Recover,
    Aim,
    Strike,
    Retract,
    Regroup,
    Eat,
    Stalk,
    Leave,
    Pursue,
}

#[derive(Debug, Clone)]
pub struct Predator {
    pub kind: &'static str,
    pub spec: &'static Spec,
    rng: Rng,
    pub bounds: Bounds,
    pub p: Vec3,
    pub state: State,
    /// Time in this state.
    pub t: f32,
    pub age: f32,
    pub strikes: u32,
    pub tip: Option<Vec3>,
    tip_full: Vec3,
    aim_at: Vec3,
    pub mouth: Vec3,
    pub target: Option<u32>,
    /// `(target id, time left)` while a captured fly is held.
    pub carry: Option<(u32, f32)>,
    pub heading: Vec3,
    /// Where it looks: toward the nearest fly.
    pub face: Vec3,
    anchor: Vec3,
    pub vel: Vec3,
    launch: Vec3,
    tgt: Option<Target>,
    reach: f32,
    bank: f32,
    pub leaving: bool,
    leave_age: f32,
    land_at: Vec3,
    v0: Vec3,
    leave_dir: Vec3,
    stalk_t: f32,
    freeze_left: f32,
    ground_t: f32,
    prev: Vec<(u32, Vec3)>,
    pub done: bool,
}

fn unit(v: Vec3) -> Vec3 {
    v.try_normalize().unwrap_or(Vec3::ZERO)
}

impl Predator {
    pub fn new(kind: &str, position: Vec3, mut rng: Rng, bounds: Bounds) -> Result<Self, String> {
        let spec = spec(kind).ok_or_else(|| format!("unknown predator {kind:?}; use one of {}", KINDS.join(", ")))?;
        let heading = unit(Vec3::new(rng.uniform(-1.0, 1.0), 0.0, rng.uniform(-1.0, 1.0)) + Vec3::splat(1e-6));
        let state = match spec.kind {
            "dragonfly" => State::Patrol,
            "frog" => State::Idle,
            _ => State::Lurk,
        };
        Ok(Self {
            kind: spec.kind,
            spec,
            rng,
            bounds,
            p: position,
            state,
            t: 0.0,
            age: 0.0,
            strikes: 0,
            tip: None,
            tip_full: Vec3::ZERO,
            aim_at: Vec3::ZERO,
            mouth: position,
            target: None,
            carry: None,
            heading,
            face: heading,
            anchor: position,
            vel: Vec3::ZERO,
            launch: position,
            tgt: None,
            reach: 0.0,
            bank: 0.0,
            leaving: false,
            leave_age: 0.0,
            land_at: position,
            v0: Vec3::ZERO,
            leave_dir: Vec3::ZERO,
            stalk_t: 0.0,
            freeze_left: 0.0,
            ground_t: 0.0,
            prev: Vec::new(),
            done: false,
        })
    }

    pub fn alive(&self) -> bool {
        !self.done
    }

    /// `(key, position, radius)` of what the fly could see grow: the body, and the striking part while it strikes.
    pub fn threats(&self) -> Vec<Threat> {
        if self.done {
            return Vec::new();
        }
        let body = if self.kind == "dragonfly" { self.p } else { self.p + Vec3::new(0.0, self.spec.body_r * 0.6, 0.0) };
        let mut out = vec![((self.kind, "body"), body, self.spec.body_r)];
        if let Some(tip) = self.tip
            && self.spec.tip_r > 0.0
            && matches!(self.state, State::Strike | State::Retract | State::Eat)
        {
            out.push(((self.kind, "tip"), tip, self.spec.tip_r));
        }
        out
    }

    /// Where a captured fly is held.
    pub fn grab_point(&self) -> Vec3 {
        self.tip.unwrap_or(self.p)
    }

    pub fn step(&mut self, dt: f32, targets: &[Target]) -> Vec<Event> {
        let mut ev = Vec::new();
        if self.done {
            return ev;
        }
        self.age += dt;
        self.t += dt;
        let live: Vec<Target> = targets.iter().copied().filter(|t| t.alive).collect();
        for t in &live {
            match self.prev.iter_mut().find(|(id, _)| *id == t.id) {
                Some(e) => e.1 = t.p,
                None => self.prev.push((t.id, t.p)),
            }
        }
        if !live.is_empty()
            && !(self.leaving || matches!(self.state, State::Leap | State::Airborne | State::Land))
            && let Some((near, _)) = self.nearest(&live, false)
        {
            let h = Vec3::new(near.p.x - self.p.x, 0.0, near.p.z - self.p.z);
            if h.length() > 1e-6 {
                self.face = unit(h);
            }
        }
        match self.kind {
            "frog" => self.frog(dt, &live, &mut ev),
            "mantis" => self.mantis(dt, &live, &mut ev),
            _ => self.dragonfly(dt, &live, &mut ev),
        }
        if self.state != State::Leave && !self.leaving {
            self.p = self.bounds.clip(self.p, 0.0);
        }
        ev
    }

    fn set(&mut self, s: State) {
        self.state = s;
        self.t = 0.0;
    }

    fn nearest(&self, live: &[Target], flying_only: bool) -> Option<(Target, f32)> {
        live.iter().filter(|t| !flying_only || t.flying).map(|t| (*t, (t.p - self.p).length())).min_by(|a, b| a.1.total_cmp(&b.1))
    }

    fn horizontal(&self, t: &Target) -> (f32, Vec3) {
        let h = Vec3::new(t.p.x - self.p.x, 0.0, t.p.z - self.p.z);
        (h.length(), h)
    }

    fn begin_aim(&mut self, t: &Target, ev: &mut Vec<Event>) {
        self.target = Some(t.id);
        self.set(State::Aim);
        ev.push(Event { kind: EventKind::Aim, target: Some(t.id), at: None });
    }

    /// aim -> strike -> retract -> recover, for the two predators that strike with a part. True while handled.
    fn strike_phase(&mut self, dt: f32, live: &[Target], ev: &mut Vec<Event>) -> bool {
        let sp = *self.spec;
        let after_miss = if self.kind == "frog" { State::Regroup } else { State::Recover };
        let calm = if self.kind == "frog" { State::Idle } else { State::Stalk };
        match self.state {
            State::Aim => {
                if self.t >= sp.aim_s {
                    let tgt = live.iter().find(|t| Some(t.id) == self.target).copied();
                    let Some(tgt) = tgt else {
                        self.set(calm);
                        return true;
                    };
                    self.aim_at = tgt.p; // it aims once, then commits: no tracking
                    self.mouth = self.p + Vec3::new(0.0, sp.body_r * 0.7, 0.0) + self.face * sp.body_r * 0.8;
                    self.tip = Some(self.mouth);
                    self.set(State::Strike);
                    ev.push(Event { kind: EventKind::Strike, target: self.target, at: Some(self.aim_at) });
                }
                true
            }
            State::Strike => {
                let f = (self.t / sp.strike_s).min(1.0);
                let aim_vec = self.aim_at - self.mouth;
                let reach_vec = unit(aim_vec) * sp.reach.min(aim_vec.length() + 0.05);
                // the frog's tongue overshoots its aim point a little and snaps back; the mantis's claw does not
                let over = if self.kind == "frog" && f > 0.5 { 1.0 + TONGUE_OVERSHOOT * (PI * f).sin().powi(2) } else { 1.0 };
                let tip = self.mouth + reach_vec * f * over;
                self.tip = Some(tip);
                self.tip_full = tip;
                if self.capture_check(live, ev) {
                    return true;
                }
                if f >= 1.0 {
                    self.set(State::Retract);
                }
                true
            }
            State::Retract => {
                let mut f = (1.0 - self.t / sp.retract_s).max(0.0);
                if self.kind == "frog" {
                    f *= f; // a tongue that snaps back fast
                }
                self.tip = Some(self.mouth + (self.tip_full - self.mouth) * f);
                if self.t >= sp.retract_s {
                    self.tip = None;
                    self.strikes += 1;
                    ev.push(Event { kind: EventKind::Miss, target: self.target, at: None });
                    if self.strikes >= sp.max_strikes {
                        self.go(ev);
                    } else {
                        self.set(after_miss);
                    }
                }
                true
            }
            State::Recover | State::Regroup => {
                if self.t >= sp.recover_s {
                    self.set(calm);
                }
                true
            }
            State::Eat => {
                if let Some(tip) = self.tip {
                    self.tip = Some(self.mouth + (tip - self.mouth) * (1.0 - dt / CARRY_S).max(0.0));
                }
                if self.t >= CARRY_S {
                    self.tip = None;
                    self.go(ev);
                }
                true
            }
            _ => false,
        }
    }

    /// The predator is done here: it leaves (the frog in hops).
    fn go(&mut self, ev: &mut Vec<Event>) {
        if self.kind == "frog" {
            self.begin_leave(ev);
        } else {
            self.set(State::Leave);
            ev.push(Event::new(EventKind::Leave));
        }
    }

    /// The striking part grabs any fly within `capture_r` of it. Returns true once something was caught.
    fn capture_check(&mut self, live: &[Target], ev: &mut Vec<Event>) -> bool {
        let Some(tip) = self.tip else { return false };
        for t in live {
            if self.kind == "dragonfly" && !t.flying {
                continue;
            }
            if (t.p - tip).length() <= self.spec.capture_r {
                self.tip_full = tip;
                self.target = Some(t.id);
                self.carry = Some((t.id, CARRY_S));
                self.set(State::Eat);
                ev.push(Event { kind: EventKind::Capture, target: Some(t.id), at: Some(tip) });
                return true;
            }
        }
        false
    }

    /// idle -> track -> crouch -> leap -> airborne -> land -> recover, and a tongue strike (aim, strike, retract, regroup) when a fly is in
    /// range. The frog only ever moves while it is airborne, and then along one ballistic arc. Leaving the arena is the same hops in one
    /// direction.
    fn frog(&mut self, dt: f32, live: &[Target], ev: &mut Vec<Event>) {
        let sp = *self.spec;
        if matches!(self.state, State::Aim | State::Strike | State::Retract | State::Regroup | State::Eat) {
            self.strike_phase(dt, live, ev);
            return;
        }
        if self.leaving {
            self.leave_age += dt;
        }
        if matches!(self.state, State::Crouch | State::Leap | State::Airborne | State::Land | State::Recover) {
            self.hop_step(ev); // a hop in progress is committed; its aim point was fixed at takeoff
            return;
        }
        if self.leaving {
            if self.leave_age >= LEAVE_S {
                self.done = true;
                ev.push(Event::new(EventKind::Gone));
            } else {
                self.set(State::Crouch);
            }
            return;
        }
        if live.is_empty() {
            if self.t > 8.0 {
                self.begin_leave(ev);
            }
            return;
        }
        let Some((t, _)) = self.nearest(live, false) else { return };
        self.tgt = Some(t);
        let (dh, _) = self.horizontal(&t);
        let height = t.p.y - self.p.y;
        if dh <= sp.trigger && height <= sp.reach * 0.9 {
            self.begin_aim(&t, ev);
            return;
        }
        if self.state == State::Idle && self.t >= IDLE_S && dh > sp.trigger * 0.6 {
            self.set(State::Track);
        } else if self.state == State::Track && self.t >= TRACK_S {
            self.set(State::Crouch);
        }
        if self.t > 25.0 && self.state == State::Idle {
            self.begin_leave(ev); // nothing came near for a long time
        }
    }

    fn begin_leave(&mut self, ev: &mut Vec<Event>) {
        self.leaving = true;
        self.leave_age = 0.0;
        self.leave_dir = unit(Vec3::new(self.heading.x, 0.0, self.heading.z));
        self.set(State::Crouch);
        ev.push(Event::new(EventKind::Leave));
    }

    /// crouch -> leap (a push-off on the ground) -> airborne (the ballistic arc) -> land -> recover -> idle.
    fn hop_step(&mut self, _ev: &mut Vec<Event>) {
        match self.state {
            State::Crouch => {
                if self.t >= CROUCH_S {
                    self.begin_leap();
                }
            }
            State::Leap => {
                if self.t >= TAKEOFF_S {
                    self.set(State::Airborne);
                }
            }
            State::Airborne => {
                let ta = self.t;
                if ta >= leap_t() {
                    self.p = self.land_at;
                    self.vel = Vec3::ZERO;
                    self.set(State::Land);
                } else {
                    self.p = self.launch + self.v0 * ta + Vec3::new(0.0, -0.5 * GRAVITY * ta * ta, 0.0);
                    self.vel = self.v0 + Vec3::new(0.0, -GRAVITY * ta, 0.0);
                }
            }
            State::Land => {
                if self.t >= LAND_S {
                    self.set(State::Recover);
                }
            }
            State::Recover if self.t >= RECOVER_S => self.set(State::Idle),
            _ => {}
        }
    }

    /// Fix the arc: from here to a landing point, in `leap_t()`, peaking `LEAP_APEX` above the ground. A hunting frog aims at the fly's
    /// position now, so a fly that moves away during the flight is not followed.
    fn begin_leap(&mut self) {
        let (d, dist) = if self.leaving {
            (self.leave_dir, LEAP_MAX)
        } else {
            let (dh, h) = self.tgt.map_or((0.0, self.face), |t| self.horizontal(&t));
            let d = if dh > 1e-6 { unit(h) } else { self.face };
            (d, (dh - STANDOFF).clamp(LEAP_MIN, LEAP_MAX))
        };
        let mut land = self.p + d * dist;
        land.y = 0.0;
        if !self.leaving {
            land = self.bounds.clip(land, 0.0);
        }
        let mut horiz = land - self.p;
        horiz.y = 0.0;
        self.launch = self.p;
        self.land_at = land;
        self.v0 = horiz / leap_t() + Vec3::new(0.0, GRAVITY * leap_t() / 2.0, 0.0);
        if horiz.length() > 1e-6 {
            self.face = unit(horiz);
        }
        self.set(State::Leap);
    }

    pub fn frog_pose(&self) -> FrogPose {
        let rest = FrogPose { pitch: 0.42, ..FrogPose::default() };
        let (f, e) = (self.t, |x: f32| x.clamp(0.0, 1.0));
        match self.state {
            State::Crouch => {
                let k = e(f / CROUCH_S);
                FrogPose { squash: k, pitch: 0.42 - 0.3 * k, ..rest }
            }
            State::Leap => {
                let k = e(f / TAKEOFF_S);
                FrogPose { squash: 1.0 - k, stretch: k, hind: k, pitch: 0.12 + 0.5 * k, ..rest }
            }
            State::Airborne => {
                let k = e(f / leap_t());
                let (vy, vh) = (self.vel.y, Vec3::new(self.vel.x, 0.0, self.vel.z).length());
                FrogPose {
                    stretch: 1.0 - e((k - 0.1) / 0.5) * 0.7,
                    hind: 1.0 - e((k - 0.35) / 0.4),
                    tuck: e((k - 0.35) / 0.4),
                    pitch: vy.atan2(vh.max(1e-6)),
                    reach: e((k - 0.7) / 0.3),
                    ..rest
                }
            }
            State::Land => {
                let k = e(f / LAND_S);
                FrogPose { squash: 0.7 * (PI * k).sin(), reach: 1.0 - k, pitch: 0.1 + 0.3 * k, ..rest }
            }
            State::Recover => {
                let k = e(f / RECOVER_S);
                FrogPose { squash: 0.25 * (1.0 - k), pitch: 0.3 + 0.12 * k, ..rest }
            }
            State::Strike | State::Retract | State::Eat => FrogPose { pitch: 0.55, ..rest },
            _ => rest,
        }
    }

    pub fn mantis_pose(&self) -> MantisPose {
        let sp = self.spec;
        let (st, f) = (self.state, self.t);
        match st {
            State::Aim => {
                let w = ease(f / sp.aim_s.max(1e-6));
                MantisPose { sway: 0.0, rear: w, cock: w, walking: false }
            }
            State::Strike => MantisPose { rear: 1.0, ..MantisPose::default() },
            State::Retract => MantisPose { rear: 1.0 - ease(f / sp.retract_s.max(1e-6)), ..MantisPose::default() },
            State::Recover => MantisPose { rear: 0.5 * (1.0 - ease(f / sp.recover_s.max(1e-6))), ..MantisPose::default() },
            State::Eat => MantisPose::default(),
            _ => {
                let frozen = self.freeze_left > 0.0;
                MantisPose { sway: if frozen { 0.0 } else { (self.age * 3.1).sin() }, rear: 0.0, cock: 0.0, walking: !frozen }
            }
        }
    }

    pub fn dragonfly_pose(&self) -> DragonflyPose {
        let speed = self.vel.length();
        let pitch = if speed > 1e-6 { (self.vel.y / speed).clamp(-1.0, 1.0).asin() } else { 0.0 };
        let (reach, bank) = if self.state == State::Pursue { (self.reach, self.bank) } else { (0.0, 0.0) };
        DragonflyPose { pitch, bank, reach, tuck: if matches!(self.state, State::Eat | State::Leave) { 1.0 } else { 1.0 - reach } }
    }

    fn mantis(&mut self, dt: f32, live: &[Target], ev: &mut Vec<Event>) {
        let sp = *self.spec;
        if matches!(self.state, State::Aim | State::Strike | State::Retract | State::Recover | State::Eat) {
            self.strike_phase(dt, live, ev);
            return;
        }
        if self.state == State::Leave {
            self.p += self.heading * sp.speed * 3.0 * dt;
            if self.t >= LEAVE_S * 2.0 {
                self.done = true;
                ev.push(Event::new(EventKind::Gone));
            }
            return;
        }
        if self.state == State::Lurk {
            self.set(State::Stalk);
        }
        if live.is_empty() {
            if self.t > 15.0 {
                self.set(State::Leave);
                ev.push(Event::new(EventKind::Leave));
            }
            return;
        }
        let Some((t, _)) = self.nearest(live, false) else { return };
        let (dh, h) = self.horizontal(&t);
        let height = (t.p.y - self.p.y).abs();
        if dh <= sp.trigger && height <= sp.reach {
            self.begin_aim(&t, ev);
            return;
        }
        if self.freeze_left > 0.0 {
            // frozen: no motion, so no looming
            self.freeze_left -= dt;
            return;
        }
        self.p += unit(h) * sp.speed * dt;
        self.stalk_t += dt;
        if self.stalk_t >= STALK_RUN_S {
            self.stalk_t = 0.0;
            self.freeze_left = self.rng.range(TELL_FREEZE_S);
        }
    }

    fn dragonfly(&mut self, dt: f32, live: &[Target], ev: &mut Vec<Event>) {
        let sp = *self.spec;
        if self.state == State::Leave {
            self.p += Vec3::Y * 2.5 * dt + self.heading * dt;
            if self.t >= LEAVE_S {
                self.done = true;
                ev.push(Event::new(EventKind::Gone));
            }
            return;
        }
        if self.state == State::Eat {
            self.tip = Some(self.p);
            if self.t >= CARRY_S {
                self.tip = None;
                self.set(State::Leave);
                ev.push(Event::new(EventKind::Leave));
            }
            return;
        }
        if self.age >= DRAGONFLY_GIVE_UP_S {
            self.set(State::Leave);
            ev.push(Event::new(EventKind::Leave));
            return;
        }
        let prey = self.nearest(live, true);
        if self.state == State::Patrol {
            let ang = self.age * 0.9;
            let goal = self.anchor + Vec3::new(ang.cos() * 1.6, (self.age * 1.7).sin() * 0.12, ang.sin() * 1.6);
            let to = goal - self.p;
            self.p += unit(to) * (2.0 * dt).min(to.length());
            if let Some((p, d)) = prey
                && d <= sp.trigger
            {
                self.target = Some(p.id);
                self.set(State::Pursue);
            }
            return;
        }
        // pursue: steer toward where the fly will be, at a limited turn rate, and grab it within capture_r
        let tgt = live.iter().find(|t| Some(t.id) == self.target && t.flying).copied();
        let Some(tgt) = tgt else {
            if live.iter().any(|t| Some(t.id) == self.target) {
                self.ground_t += dt;
                if self.ground_t >= DRAGONFLY_GROUND_PATIENCE_S {
                    self.ground_t = 0.0;
                    self.set(State::Patrol);
                }
            } else {
                self.set(State::Patrol);
            }
            return;
        };
        self.ground_t = 0.0;
        let tp = tgt.p;
        let prev = self.prev.iter().find(|(id, _)| *id == tgt.id).map_or(tp, |e| e.1);
        let fly_v = if dt > 0.0 && self.t > dt { (tp - prev) / dt } else { Vec3::ZERO };
        let lead = ((tp - self.p).length() / sp.speed.max(1e-6)).min(0.35);
        let want = unit(tp + fly_v * lead - self.p);
        let cur = if self.vel.length() > 1e-6 { unit(self.vel) } else { want };
        let ang = cur.dot(want).clamp(-1.0, 1.0).acos();
        let k = if ang < 1e-6 { 1.0 } else { (DRAGONFLY_TURN * dt / ang).min(1.0) };
        let dirn = unit(cur * (1.0 - k) + want * k);
        let turn = cur.cross(dirn).dot(Vec3::Y) / dt.max(1e-9); // rad/s, signed
        self.bank += ((turn * 0.12).clamp(-0.7, 0.7) - self.bank) * (8.0 * dt).min(1.0);
        self.reach = (1.0 - (tp - self.p).length() / (4.0 * sp.capture_r)).clamp(0.0, 1.0);
        self.vel = dirn * sp.speed;
        self.p += self.vel * dt;
        if (tp - self.p).length() <= sp.capture_r {
            self.tip = Some(self.p);
            self.carry = Some((tgt.id, CARRY_S));
            self.set(State::Eat);
            ev.push(Event { kind: EventKind::Capture, target: Some(tgt.id), at: Some(self.p) });
        }
    }
}

/// Where a predator starts. GAME RULE. `near` is a fly's thorax: the frog and mantis start about a metre from it, on the floor; the
/// dragonfly starts at altitude above it.
pub fn spawn_position(kind: &str, bounds: &Bounds, rng: &mut Rng, near: Option<Vec3>, altitude: Option<f32>) -> Vec3 {
    let n = near.unwrap_or(Vec3::new((bounds.xmin + bounds.xmax) / 2.0, bounds.ymin, (bounds.zmin + bounds.zmax) / 2.0));
    let p = if kind == "dragonfly" {
        let top = altitude.unwrap_or(bounds.ymax - 0.5);
        Vec3::new(n.x + rng.uniform(-1.5, 1.5), top, n.z + rng.uniform(-1.5, 1.5))
    } else {
        let r = if kind == "frog" { rng.uniform(2.0, 2.8) } else { rng.uniform(1.4, 2.0) };
        let a = rng.uniform(0.0, TAU);
        Vec3::new(n.x + a.cos() * r, bounds.ymin, n.z + a.sin() * r)
    };
    bounds.clip(p, 0.5)
}

/// `(key, position, radius)` of one threat.
pub type Threat = ((&'static str, &'static str), Vec3, f32);

/// The headless trace the Lab assay and the tests use: one predator against one fly that stays where it is.
#[derive(Debug, Clone)]
pub struct Trace {
    pub frames: Vec<Vec<Threat>>,
    pub capture_frame: Option<usize>,
    pub aim_frame: Option<usize>,
    pub strike_frame: Option<usize>,
    pub positions: Vec<Vec3>,
}

pub fn trace(kind: &str, seed: u64, max_s: f32) -> Result<Trace, String> {
    let bounds = Bounds::room();
    let mut rng = Rng::new(seed);
    let flying = kind == "dragonfly";
    let fly = if flying { Vec3::new(0.0, 1.5, 0.0) } else { Vec3::new(0.0, THORAX_Y, 0.0) };
    let start = spawn_position(kind, &bounds, &mut rng, Some(fly), None);
    let mut p = Predator::new(kind, start, rng, bounds)?;
    let target = Target { id: 0, p: fly, flying, alive: true };
    let mut out = Trace { frames: Vec::new(), capture_frame: None, aim_frame: None, strike_frame: None, positions: Vec::new() };
    for i in 0..(max_s / DT) as usize {
        let ev = p.step(DT, &[target]);
        out.frames.push(p.threats());
        out.positions.push(p.p);
        for e in ev {
            match e.kind {
                EventKind::Aim if out.aim_frame.is_none() => out.aim_frame = Some(i),
                EventKind::Strike if out.strike_frame.is_none() => out.strike_frame = Some(i),
                EventKind::Capture => out.capture_frame = Some(i),
                _ => {}
            }
        }
        if out.capture_frame.is_some() || p.done {
            break;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leap_time_comes_from_the_games_gravity() {
        assert!((GRAVITY - 19.44).abs() < 1e-3);
        assert!((leap_t() - 0.4303).abs() < 5e-3, "{}", leap_t());
    }

    #[test]
    fn the_frog_hops_on_a_ballistic_arc_and_never_slides() {
        let bounds = Bounds::room();
        let mut f = Predator::new("frog", Vec3::new(-3.0, 0.0, 0.0), Rng::new(1), bounds).unwrap();
        let target = Target { id: 1, p: Vec3::new(1.0, THORAX_Y, 0.0), flying: false, alive: true };
        let (mut peak, mut airborne_frames, mut slid) = (0.0f32, 0, 0);
        let mut prev = f.p;
        for _ in 0..(8.0 / DT) as usize {
            let _ = f.step(DT, &[target]);
            let moved = Vec3::new(f.p.x - prev.x, 0.0, f.p.z - prev.z).length();
            if f.state == State::Airborne {
                airborne_frames += 1;
                peak = peak.max(f.p.y);
                assert!(f.p.y >= 0.0, "in the air the frog is never below the floor");
            } else if moved > 0.2 && f.state != State::Land {
                slid += 1;
            }
            prev = f.p;
            if f.state == State::Aim {
                break;
            }
        }
        assert!(airborne_frames > 20, "it jumped");
        assert!((peak - LEAP_APEX).abs() < 0.04, "apex {peak}");
        assert_eq!(slid, 0, "all horizontal travel happens in the air");
        assert!(f.p.x > -3.0, "it got closer");
    }

    #[test]
    fn frog_crouches_before_it_leaps() {
        let mut f = Predator::new("frog", Vec3::new(-3.0, 0.0, 0.0), Rng::new(2), Bounds::room()).unwrap();
        let target = Target { id: 1, p: Vec3::new(1.0, THORAX_Y, 0.0), flying: false, alive: true };
        let mut order = Vec::new();
        for _ in 0..(6.0 / DT) as usize {
            f.step(DT, &[target]);
            if order.last() != Some(&f.state) {
                order.push(f.state);
            }
        }
        let i = order.iter().position(|s| *s == State::Crouch).expect("crouch");
        assert_eq!(order[i + 1], State::Leap);
        assert_eq!(order[i + 2], State::Airborne);
        assert_eq!(order[i + 3], State::Land);
        let mut c = f.clone();
        c.state = State::Crouch;
        c.t = CROUCH_S;
        assert!(c.frog_pose().squash > 0.99, "the anticipation pose is fully squashed at the end of the crouch");
    }

    #[test]
    fn the_frogs_tongue_strikes_in_0_07_s_and_a_still_fly_is_caught() {
        let t = trace("frog", 3, 40.0).unwrap();
        let (aim, strike) = (t.aim_frame.unwrap(), t.strike_frame.unwrap());
        assert!(((strike - aim) as f32 * DT - 0.30).abs() < 0.04, "aim pause is 0.30 s");
        assert!(t.capture_frame.is_some(), "a fly that does not move is caught");
        let cap = t.capture_frame.unwrap();
        assert!(((cap - strike) as f32 * DT) <= 0.08, "the tongue reaches in about 0.07 s");
    }

    #[test]
    fn mantis_creeps_freezes_and_strikes_dragonfly_only_takes_flyers() {
        let t = trace("mantis", 5, 120.0).unwrap();
        assert!(t.strike_frame.is_some());
        let speeds: Vec<f32> = t.positions.windows(2).map(|w| (w[1] - w[0]).length() / DT).collect();
        assert!(speeds.iter().cloned().fold(0.0, f32::max) < 0.4 || t.strike_frame.is_some());
        assert!(speeds.iter().take(300).any(|s| *s < 1e-6), "it freezes now and then");
        let d = trace("dragonfly", 4, 30.0).unwrap();
        assert!(d.capture_frame.is_some(), "a hovering fly is prey");
        // a walking fly is not prey to a dragonfly
        let bounds = Bounds::room();
        let mut r = Rng::new(9);
        let mut p = Predator::new("dragonfly", spawn_position("dragonfly", &bounds, &mut r, None, None), r, bounds).unwrap();
        let walker = Target { id: 1, p: Vec3::new(0.0, THORAX_Y, 0.0), flying: false, alive: true };
        let caught = (0..(25.0 / DT) as usize).any(|_| p.step(DT, &[walker]).iter().any(|e| e.kind == EventKind::Capture));
        assert!(!caught);
    }

    #[test]
    fn predators_leave_when_nothing_is_there() {
        for kind in KINDS {
            let bounds = Bounds::room();
            let mut r = Rng::new(1);
            let mut p = Predator::new(kind, spawn_position(kind, &bounds, &mut r, None, None), r, bounds).unwrap();
            for _ in 0..(120.0 / DT) as usize {
                p.step(DT, &[]);
                if p.done {
                    break;
                }
            }
            assert!(p.done, "{kind} gives up");
        }
    }

    #[test]
    fn unknown_kind_is_an_error() {
        assert!(Predator::new("newt", Vec3::ZERO, Rng::new(1), Bounds::room()).is_err());
    }
}
