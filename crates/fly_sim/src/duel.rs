//! Fly vs fly duel: the 1v1 duel with a second brain in place of you. Nothing about who wins is scripted. Python: `game/flyduel.py`.
//!
//! CONNECTOME: what the neurons do with what they are given (turning is DNa01/02 right minus left, shooting is DNp35/DNpe052 above its
//! threshold, the dodge the giant fiber, the run the body-touch group). GAME RULE: the arena, the blaster, how a turning command becomes a
//! heading change, what a hit does, the match length. MODEL PREDICTION: who wins; a prediction of this model, not of real flies.
//!
//! The engine is Python's line for line in f64 with NumPy's own generator (`rng::Generator`), so a scripted pair of brains gives the
//! same frames, events and driver tables (`tests/golden/duel_engine.json`). It takes anything that looks like a brain ([`Fighter`]).

use std::collections::VecDeque;
use std::f64::consts::{PI, TAU};

use crate::rng::Generator;

pub const HP: f64 = 100.0;
/// Brain steps (5 ms) per engine tick: 20 ms.
pub const TICK_STEPS: usize = 4;
pub const ARENA_HALF: f64 = 6.0;
pub const START_APART: f64 = 6.0;
pub const START_YAW_JITTER: f64 = 1.2;
pub const FLY_RADIUS: f64 = 0.32;
pub const PELLET_SPEED: f64 = 9.0;
pub const PELLET_SPREAD: f64 = 0.035;
pub const PELLET_DAMAGE: f64 = 9.0;
pub const PELLET_LIFE_S: f64 = 1.6;
pub const PELLET_RADIUS: f64 = 0.10;
pub const FIRE_COOLDOWN: f64 = 0.45;
pub const REWARD_PULSE_S: f64 = 0.45;
pub const PUNISH_PULSE_S: f64 = 0.6;
pub const STEER_DEADZONE_HZ: f64 = 1.5;
pub const STEER_GAIN: f64 = 0.004;
pub const STEER_MAX: f64 = 0.035;
pub const TRACK_SCALE: f64 = 0.45;
pub const WALK_SPEED: f64 = 0.5;
pub const DODGE_HOP: f64 = 0.9;
pub const DODGE_COOLDOWN: f64 = 1.2;
pub const RUN_HOP: f64 = 0.7;
pub const RUN_COOLDOWN: f64 = 1.5;
pub const SEE_RANGE: f64 = 7.5;
pub const DEFAULT_SECONDS: f64 = 20.0;
/// 200 ms of spikes before a shot are kept for "what drove the shots that landed".
pub const WINDOW_STEPS: usize = 40;
/// A replay frame every this many ticks (10 Hz).
pub const REPLAY_EVERY: usize = 5;
pub const TIEBREAK_REMATCHES: usize = 2;
/// Rad/s of angular expansion: below `LOOM_MIN` nothing, `LOOM_MIN + LOOM_FULL` = full drive.
pub const LOOM_MIN: f64 = 1.5;
pub const LOOM_FULL: f64 = 8.0;
/// `kick_the_fly.THRESH` for the four readouts the duel uses.
pub const T_FIRE: f64 = 3.0;
pub const T_ESCAPE: f64 = 4.0;
pub const T_RUN: f64 = 2.4;
pub const T_WALK: f64 = 3.0;

/// The part of a brain the duel needs (Python: poke, level, hz, `_step`, `sim.spikes`, `types`, `n`, `dt`).
pub trait Fighter {
    fn level(&self, name: &str) -> f64;
    fn hz(&self, name: &str) -> f64;
    fn poke(&mut self, region: &str, side: Option<char>, strength: f64, recruit: Option<f64>);
    /// One 5 ms step; the rows that spiked (ascending).
    fn step(&mut self) -> &[u32];
    fn n(&self) -> usize;
    fn dt(&self) -> f64 {
        0.005
    }
    /// The cell type of every neuron (only read when drivers are asked for).
    fn types(&self) -> &[String];
}

impl Fighter for crate::Brain {
    fn level(&self, name: &str) -> f64 {
        f64::from(crate::Brain::level(self, name))
    }
    fn hz(&self, name: &str) -> f64 {
        f64::from(crate::Brain::hz(self, name))
    }
    fn poke(&mut self, region: &str, side: Option<char>, strength: f64, recruit: Option<f64>) {
        self.poke_recruit(region, side, strength as f32, recruit.map(|r| r as f32));
    }
    fn step(&mut self) -> &[u32] {
        crate::Brain::step(self)
    }
    fn n(&self) -> usize {
        crate::Brain::n(self)
    }
    fn dt(&self) -> f64 {
        self.sim.p.dt_ms / 1000.0
    }
    fn types(&self) -> &[String] {
        &self.labels().types
    }
}

fn wrap(a: f64) -> f64 {
    (a + PI).rem_euclid(TAU) - PI
}

/// `round(x, digits)` the way Python does it: the correctly rounded decimal of the binary value.
pub fn round_to(x: f64, digits: usize) -> f64 {
    format!("{x:.digits$}").parse().unwrap_or(x)
}

type V2 = [f64; 2];

fn hypot(v: V2) -> f64 {
    v[0].hypot(v[1])
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    A,
    B,
}

impl Side {
    pub fn index(self) -> usize {
        match self {
            Side::A => 0,
            Side::B => 1,
        }
    }
}

struct Duelist {
    pos: V2,
    yaw: f64,
    hp: f64,
    steer: f64,
    fire_ready: f64,
    dodge_ready: f64,
    run_ready: f64,
    reward_until: f64,
    punish_until: f64,
    hop: V2,
    shots: u32,
    hits: u32,
    /// fire, escape, run, walk
    max_levels: [f64; 4],
    ring: VecDeque<Vec<u32>>,
    hit_windows: u32,
    hit_idx: Vec<Vec<u32>>,
}

impl Duelist {
    fn new(pos: V2, yaw: f64) -> Self {
        Self {
            pos,
            yaw,
            hp: HP,
            steer: 0.0,
            fire_ready: 0.0,
            dodge_ready: 0.0,
            run_ready: 0.0,
            reward_until: 0.0,
            punish_until: 0.0,
            hop: [0.0; 2],
            shots: 0,
            hits: 0,
            max_levels: [0.0; 4],
            ring: VecDeque::with_capacity(WINDOW_STEPS),
            hit_windows: 0,
            hit_idx: Vec::new(),
        }
    }

    fn fwd(&self) -> V2 {
        [self.yaw.cos(), self.yaw.sin()]
    }
}

struct Pellet {
    pos: V2,
    vel: V2,
    born: f64,
    owner: usize,
    window: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Event {
    pub kind: &'static str,
    pub t: f64,
    /// 0 for A, 1 for B.
    pub who: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    pub t: f64,
    pub pos: [V2; 2],
    pub yaw: [f64; 2],
    pub hp: [f64; 2],
    /// `(x, y, owner)`, x and y rounded to 2 places.
    pub pellets: Vec<(f64, f64, usize)>,
}

/// Per cell type: spikes in the 200 ms windows before the shots that landed, spikes over the whole match, neurons.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TypeSums {
    pub types: Vec<String>,
    pub hit: Vec<f64>,
    pub base: Vec<f64>,
    pub n: Vec<f64>,
    pub hit_windows: u64,
    pub steps: u64,
    pub dt: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DuelResult {
    pub winner: Option<Side>,
    pub hp: [f64; 2],
    pub shots: [u32; 2],
    pub hits: [u32; 2],
    pub seconds: f64,
    pub knockout: bool,
    /// fire, escape, run, walk, rounded to 3 places.
    pub max_levels: [[f64; 4]; 2],
    pub events: Vec<Event>,
    pub frames: Vec<Frame>,
    pub steps: u64,
    /// Present when drivers were asked for.
    pub drivers: Option<[TypeSums; 2]>,
}

impl Default for DuelResult {
    fn default() -> Self {
        Self {
            winner: None,
            hp: [HP; 2],
            shots: [0; 2],
            hits: [0; 2],
            seconds: 0.0,
            knockout: false,
            max_levels: [[0.0; 4]; 2],
            events: Vec::new(),
            frames: Vec::new(),
            steps: 0,
            drivers: None,
        }
    }
}

/// What `me` sees of `other`: LC10 target tracking on the side the other is on, small-object detectors when it is dead ahead, plus
/// looming from the other's pellets coming at it (game rule: a 0.1 m pellet).
fn senses<F: Fighter + ?Sized>(br: &mut F, me: &Duelist, other: &Duelist, pellets: &[Pellet], me_idx: usize) {
    let to = [other.pos[0] - me.pos[0], other.pos[1] - me.pos[1]];
    let dist = hypot(to);
    if dist <= SEE_RANGE && me.hp > 0.0 {
        let d = dist.max(1e-6);
        let u = [to[0] / d, to[1] / d];
        let f = me.fwd();
        let side_v = [-f[1], f[0]];
        let ahead = u[0] * f[0] + u[1] * f[1];
        let sideness = u[0] * side_v[0] + u[1] * side_v[1];
        let near = (1.3 - dist / 7.0).clamp(0.3, 1.0);
        let s_side = (sideness.abs() / TRACK_SCALE).clamp(0.0, 1.0) * near;
        // the 3D duel's +side is its right: this plane's +side is the left
        let key = if sideness > 0.0 { 'L' } else { 'R' };
        if ahead < 0.995 && s_side > 0.03 {
            br.poke("track", Some(key), 0.05, Some(0.15 + 0.45 * s_side));
        }
        if ahead > 0.985 && dist < 7.0 {
            let s = near * ((ahead - 0.985) / 0.012).clamp(0.0, 1.0);
            if s > 0.02 {
                let sides: &[char] = if sideness.abs() < 0.12 { &['L', 'R'] } else { std::slice::from_ref(&key) };
                for sd in sides {
                    br.poke("small", Some(*sd), 0.05, Some(0.2 + 0.4 * s));
                }
            }
        }
    }
    let mut best = 0.0f64;
    for p in pellets {
        if p.owner == me_idx {
            continue;
        }
        let rel = [p.pos[0] - me.pos[0], p.pos[1] - me.pos[1]];
        let d = hypot(rel);
        if !(0.3..=4.0).contains(&d) {
            continue;
        }
        let closing = -(rel[0] * p.vel[0] + rel[1] * p.vel[1]) / d;
        if closing <= 0.0 {
            continue;
        }
        best = best.max(2.0 * PELLET_RADIUS * closing / (d * d));
    }
    if best > 0.0 && me.hp > 0.0 {
        let strength = ((best - LOOM_MIN) / LOOM_FULL).clamp(0.0, 1.0);
        if strength > 0.0 {
            br.poke("loom", None, strength, Some(0.6 * strength));
        }
    }
}

/// Steering from DNa01/02 right minus left, shooting when DNp35/DNpe052 fire above threshold, dodging on the giant fiber, running on the
/// body-touch group: the 3D duel's `_duel_motor` and the game's reactions, with this plane's geometry.
#[allow(clippy::too_many_arguments)]
fn motor<F: Fighter + ?Sized>(
    br: &mut F,
    me: &mut Duelist,
    other_pos: V2,
    pellets: &mut Vec<Pellet>,
    me_idx: usize,
    t: f64,
    dt: f64,
    rng: &mut Generator,
    pid: &mut u64,
    events: &mut Vec<Event>,
) {
    if me.hp <= 0.0 {
        return;
    }
    if t < me.reward_until {
        br.poke("reward", None, 0.8, None);
    }
    if t < me.punish_until {
        br.poke("punish", None, 1.0, None);
    }
    let turn = br.hz("turn_r") - br.hz("turn_l");
    me.steer += (turn - me.steer) * 0.35;
    let mag = (me.steer.abs() - STEER_DEADZONE_HZ).max(0.0);
    if mag > 0.0 {
        let dyaw = (mag * STEER_GAIN).min(STEER_MAX).copysign(me.steer) * (dt * 60.0);
        me.yaw = wrap(me.yaw - dyaw); // heading is counter-clockwise in this plane, so "right" is a negative step
    }
    let lv = [br.level("fire"), br.level("escape"), br.level("run"), br.level("walk")];
    for (m, v) in me.max_levels.iter_mut().zip(lv) {
        *m = m.max(v);
    }
    if lv[0] > T_FIRE && t >= me.fire_ready {
        me.fire_ready = t + FIRE_COOLDOWN;
        let f = me.fwd();
        let ang = f[1].atan2(f[0]) + rng.normal(0.0, PELLET_SPREAD);
        let v = [ang.cos() * PELLET_SPEED, ang.sin() * PELLET_SPEED];
        *pid += 1;
        let window: Vec<u32> = me.ring.iter().flatten().copied().collect();
        pellets.push(Pellet { pos: [me.pos[0] + f[0] * 0.25, me.pos[1] + f[1] * 0.25], vel: v, born: t, owner: me_idx, window });
        me.shots += 1;
        events.push(Event { kind: "shot", t, who: me_idx });
    }
    let away = [me.pos[0] - other_pos[0], me.pos[1] - other_pos[1]];
    let al = hypot(away).max(1e-6);
    let away = [away[0] / al, away[1] / al];
    if lv[1] > T_ESCAPE && t >= me.dodge_ready && !pellets.is_empty() {
        // a hop sideways to the nearest incoming pellet's line
        let mut nearest: Option<(&Pellet, f64)> = None;
        for p in pellets.iter().filter(|p| p.owner != me_idx) {
            let d = hypot([p.pos[0] - me.pos[0], p.pos[1] - me.pos[1]]);
            if d < 4.0 && nearest.is_none_or(|(_, bd)| d < bd) {
                nearest = Some((p, d));
            }
        }
        if let Some((p, _)) = nearest {
            let dirp = [p.vel[0] / PELLET_SPEED, p.vel[1] / PELLET_SPEED];
            let mut perp = [-dirp[1], dirp[0]];
            if perp[0] * (me.pos[0] - p.pos[0]) + perp[1] * (me.pos[1] - p.pos[1]) < 0.0 {
                perp = [-perp[0], -perp[1]];
            }
            me.hop[0] += perp[0] * DODGE_HOP;
            me.hop[1] += perp[1] * DODGE_HOP;
            me.dodge_ready = t + DODGE_COOLDOWN;
            events.push(Event { kind: "dodge", t, who: me_idx });
        }
    }
    if lv[2] > T_RUN && t >= me.run_ready && t < me.punish_until + 0.5 {
        // hurt, and the body-touch group says run
        me.hop[0] += away[0] * RUN_HOP;
        me.hop[1] += away[1] * RUN_HOP;
        me.run_ready = t + RUN_COOLDOWN;
        events.push(Event { kind: "run", t, who: me_idx });
    }
    let mut step = [0.0f64; 2];
    if lv[3] > T_WALK {
        let f = me.fwd();
        step[0] += f[0] * WALK_SPEED * dt;
        step[1] += f[1] * WALK_SPEED * dt;
    }
    if hypot(me.hop) > 1e-6 {
        // hops play out over ~0.2 s
        let k = (dt / 0.2).min(1.0);
        let part = [me.hop[0] * k, me.hop[1] * k];
        step[0] += part[0];
        step[1] += part[1];
        me.hop = [me.hop[0] - part[0], me.hop[1] - part[1]];
    }
    me.pos = [(me.pos[0] + step[0]).clamp(-ARENA_HALF, ARENA_HALF), (me.pos[1] + step[1]).clamp(-ARENA_HALF, ARENA_HALF)];
}

fn move_pellets(
    pellets: &mut Vec<Pellet>,
    d: &mut [Duelist; 2],
    ba: &mut dyn Fighter,
    bb: &mut dyn Fighter,
    t: f64,
    dt: f64,
    events: &mut Vec<Event>,
) {
    let mut keep = Vec::with_capacity(pellets.len());
    for mut p in pellets.drain(..) {
        p.pos = [p.pos[0] + p.vel[0] * dt, p.pos[1] + p.vel[1] * dt];
        let mut hit = None;
        for (i, du) in d.iter().enumerate() {
            if i == p.owner || du.hp <= 0.0 {
                continue;
            }
            if hypot([p.pos[0] - du.pos[0], p.pos[1] - du.pos[1]]) < FLY_RADIUS {
                hit = Some(i);
                break;
            }
        }
        if let Some(h) = hit {
            d[h].hp = (d[h].hp - PELLET_DAMAGE).max(0.0);
            d[h].punish_until = t + PUNISH_PULSE_S;
            if h == 0 {
                ba.poke("body", None, 0.5, None); // the same touch as a hit from you: it feels the pellet
            } else {
                bb.poke("body", None, 0.5, None);
            }
            let s = &mut d[p.owner];
            s.hits += 1;
            s.reward_until = t + REWARD_PULSE_S;
            events.push(Event { kind: "hit", t, who: p.owner });
            s.hit_windows += 1;
            s.hit_idx.push(p.window);
            continue;
        }
        let outside = p.pos[0].abs() > ARENA_HALF || p.pos[1].abs() > ARENA_HALF;
        if outside || t - p.born > PELLET_LIFE_S {
            continue;
        }
        keep.push(p);
    }
    *pellets = keep;
}

fn step_one(br: &mut dyn Fighter, du: &mut Duelist, base: &mut [i32], drivers: bool) {
    let spikes = br.step();
    if drivers {
        if du.ring.len() == WINDOW_STEPS {
            du.ring.pop_front();
        }
        du.ring.push_back(spikes.to_vec());
        for &r in spikes {
            base[r as usize] += 1;
        }
    }
}

fn type_sums(d: &Duelist, types: &[String], base: &[i32], steps: u64, dt: f64) -> TypeSums {
    let mut names: Vec<&String> = types.iter().collect();
    names.sort_unstable();
    names.dedup();
    let idx = |t: &String| names.binary_search(&t).unwrap_or(0);
    let k = names.len();
    let (mut hit, mut b, mut n) = (vec![0.0; k], vec![0.0; k], vec![0.0; k]);
    let mut hitrow = vec![0i64; types.len()];
    for w in &d.hit_idx {
        for &r in w {
            hitrow[r as usize] += 1;
        }
    }
    for (i, t) in types.iter().enumerate() {
        let j = idx(t);
        hit[j] += hitrow[i] as f64;
        b[j] += f64::from(base[i]);
        n[j] += 1.0;
    }
    TypeSums { types: names.into_iter().cloned().collect(), hit, base: b, n, hit_windows: u64::from(d.hit_windows), steps, dt }
}

/// One match between two brains. Returns the winner, health, shots, hits, the replay frames and events, and (`drivers`) the neurons'
/// spike counts before the shots that landed. The match seed sets the starting headings and the pellets' scatter; the brains' own noise
/// is theirs. The match ends when one health reaches 0 or the time is up (more health wins; equal health is a draw).
pub fn run_duel(
    brain_a: &mut dyn Fighter,
    brain_b: &mut dyn Fighter,
    seed: u64,
    seconds: f64,
    record: bool,
    drivers: bool,
    mut progress: impl FnMut(f64) -> bool,
) -> DuelResult {
    let mut rng = Generator::new(seed & 0x7FFF_FFFF);
    let uniform = |rng: &mut Generator| -START_YAW_JITTER + (START_YAW_JITTER - -START_YAW_JITTER) * rng.random();
    let ya = uniform(&mut rng);
    let yb = uniform(&mut rng);
    let mut d = [Duelist::new([-START_APART / 2.0, 0.0], ya), Duelist::new([START_APART / 2.0, 0.0], PI + yb)];
    let mut pellets: Vec<Pellet> = Vec::new();
    let mut pid = 0u64;
    let mut events: Vec<Event> = Vec::new();
    let dt = TICK_STEPS as f64 * brain_a.dt();
    let ticks = (seconds / dt).round() as usize;
    let mut frames = Vec::new();
    let mut base: [Vec<i32>; 2] = [Vec::new(), Vec::new()];
    if drivers {
        base = [vec![0; brain_a.n()], vec![0; brain_b.n()]];
    }
    let mut steps = 0u64;
    let mut last_tick = 0usize;
    for tick in 0..ticks {
        last_tick = tick;
        if !progress(tick as f64 / ticks as f64) {
            break;
        }
        let t = tick as f64 * dt;
        {
            let (da, db) = (&d[0], &d[1]);
            senses(brain_a, da, db, &pellets, 0);
            senses(brain_b, db, da, &pellets, 1);
        }
        for _ in 0..TICK_STEPS {
            let [d0, d1] = &mut d;
            let [b0, b1] = &mut base;
            step_one(brain_a, d0, b0, drivers);
            step_one(brain_b, d1, b1, drivers);
            steps += 1;
        }
        // a then b, each seeing the other's position as it is by then (Python reads the live object)
        let pb = d[1].pos;
        motor(brain_a, &mut d[0], pb, &mut pellets, 0, t, dt, &mut rng, &mut pid, &mut events);
        let pa = d[0].pos;
        motor(brain_b, &mut d[1], pa, &mut pellets, 1, t, dt, &mut rng, &mut pid, &mut events);
        move_pellets(&mut pellets, &mut d, &mut *brain_a, &mut *brain_b, t, dt, &mut events);
        if record && tick % REPLAY_EVERY == 0 {
            frames.push(Frame {
                t: round_to(t, 3),
                pos: [d[0].pos, d[1].pos],
                yaw: [d[0].yaw, d[1].yaw],
                hp: [d[0].hp, d[1].hp],
                pellets: pellets.iter().map(|p| (round_to(p.pos[0], 2), round_to(p.pos[1], 2), p.owner)).collect(),
            });
        }
        if d[0].hp <= 0.0 || d[1].hp <= 0.0 {
            break;
        }
    }
    let winner = if d[0].hp > d[1].hp {
        Some(Side::A)
    } else if d[1].hp > d[0].hp {
        Some(Side::B)
    } else {
        None
    };
    let drivers = drivers.then(|| {
        [type_sums(&d[0], brain_a.types(), &base[0], steps, brain_a.dt()), type_sums(&d[1], brain_b.types(), &base[1], steps, brain_b.dt())]
    });
    let r3 = |v: [f64; 4]| v.map(|x| round_to(x, 3));
    DuelResult {
        winner,
        hp: [d[0].hp, d[1].hp],
        shots: [d[0].shots, d[1].shots],
        hits: [d[0].hits, d[1].hits],
        seconds: round_to(ticks.min(last_tick + 1) as f64 * dt, 3),
        knockout: d[0].hp <= 0.0 || d[1].hp <= 0.0,
        max_levels: [r3(d[0].max_levels), r3(d[1].max_levels)],
        events: events.into_iter().map(|e| Event { t: round_to(e.t, 3), ..e }).collect(),
        frames: if record { frames } else { Vec::new() },
        steps,
        drivers,
    }
}

/// Sum the type sums of several matches (one pack, so one type list) into one.
pub fn combine_drivers(parts: &[&TypeSums]) -> Option<TypeSums> {
    let first = parts.first()?;
    let mut out = TypeSums {
        types: first.types.clone(),
        hit: vec![0.0; first.hit.len()],
        base: vec![0.0; first.base.len()],
        n: first.n.clone(),
        hit_windows: 0,
        steps: 0,
        dt: first.dt,
    };
    for p in parts {
        for (o, v) in out.hit.iter_mut().zip(&p.hit) {
            *o += v;
        }
        for (o, v) in out.base.iter_mut().zip(&p.base) {
            *o += v;
        }
        out.hit_windows += p.hit_windows;
        out.steps += p.steps;
    }
    Some(out)
}

#[derive(Debug, Clone, PartialEq)]
pub struct DriverRow {
    pub kind: String,
    pub neurons: u32,
    pub hz_before_hits: f64,
    pub hz_overall: f64,
    pub ratio: f64,
}

fn row(s: &TypeSums, i: usize) -> DriverRow {
    let win_s = WINDOW_STEPS as f64 * s.dt;
    let n = s.n[i];
    let r_hit = s.hit[i] / (s.hit_windows as f64 * win_s * n);
    let r_base = s.base[i] / (s.steps as f64 * s.dt * n);
    DriverRow {
        kind: s.types[i].clone(),
        neurons: n as u32,
        hz_before_hits: r_hit,
        hz_overall: r_base,
        ratio: if r_base > 0.0 { r_hit / r_base } else { f64::NAN },
    }
}

/// The cell types whose firing in the 200 ms before landed shots most exceeds their firing over the whole match. A correlation of the
/// brain's own activity with its shots, not a causal test (nothing was silenced).
pub fn top_drivers(sums: Option<&TypeSums>, top: usize, min_base: f64, min_hit: f64, min_windows: u64) -> Vec<DriverRow> {
    let Some(s) = sums else { return Vec::new() };
    if s.hit_windows < min_windows {
        return Vec::new();
    }
    let mut rows: Vec<DriverRow> = (0..s.types.len())
        .filter(|&i| !(s.base[i] < min_base || s.hit[i] < min_hit || s.n[i] < 1.0))
        .map(|i| row(s, i))
        .filter(|r| !r.ratio.is_nan())
        .collect();
    rows.sort_by(|a, b| b.ratio.total_cmp(&a.ratio)); // stable, like Python's sort on -ratio
    rows.truncate(top);
    rows
}

/// The cell types the duel's own rules read or poke: how they fired before landed shots, whatever their rank in the full list.
pub const READOUT_TYPES: [&str; 11] = ["DNp35", "DNpe052", "DNa01", "DNa02", "LC10", "LC11", "LC18", "LC21", "LC26", "DNp01", "SNta13"];

pub fn named_drivers(sums: Option<&TypeSums>) -> Vec<DriverRow> {
    let Some(s) = sums else { return Vec::new() };
    if s.hit_windows < 1 {
        return Vec::new();
    }
    READOUT_TYPES
        .iter()
        .filter_map(|name| {
            let i = s.types.iter().position(|t| t == name)?;
            (s.n[i] >= 1.0).then(|| row(s, i))
        })
        .collect()
}
