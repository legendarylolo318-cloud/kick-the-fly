//! Classic behavior rigs (`game/rigs.py`, `lab/rigassay.py`): a tethered flight simulator, a fly on a ball, Buridan's paradigm and a
//! four-field olfactory arena, and the pre-registered assay over them.
//!
//! Each rig is a small piece of physics around a real brain. CONNECTOME: yaw is read from DNa01 + DNa02 (right minus left), forward speed
//! from DNp09 against the race's fixed reference, wide-field rotation reaches the steering neurons through T4/T5, an odor reaches its
//! receptor neurons. GAME RULE: how Hz becomes a yaw rate (the duel's gain), the EMD stage, LC10 tracking of a stripe or bar, the platform,
//! the arena, the measures. MODEL PREDICTION: whatever the fly then does.
//!
//! Conventions: angles in radians, clockwise from above positive, heading 0 along +y; one engine tick is 4 brain steps (20 ms).
//! Not ported: the Lab recorder files of `scene_run(folder=...)` (they arrive with the Lab export).

use std::sync::Arc;

use serde_json::{Value, json};

use crate::brain::Brain;
use crate::cards::Build;
use crate::csr::Weights;
use crate::pack::Labels;
use crate::validation::{self, np_sum};

pub const TICK_STEPS: usize = 4;
pub const YAW_DEADZONE_HZ: f64 = 1.5;
pub const STEER_SMOOTH: f64 = 0.35;
pub const TRACK_SCALE: f64 = 0.45;
pub const TRACK_AHEAD: f64 = 0.995;
pub const TRACK_RECRUIT: (f64, f64) = (0.15, 0.45);
pub const TRACK_POKE: f64 = 0.05;
pub const PLATFORM_RADIUS: f64 = 0.5;
pub const STRIPE_DISTANCE: f64 = 1.5;
pub const FIELD_HALF: f64 = 0.5;
pub const SCENT_POKE: f64 = 0.3;
pub const BALL_BAR_DISTANCE: f64 = 3.0;
pub const TRACE_EVERY: usize = 5;
pub const RULE_VERSION: i64 = 1;
pub const RIGS: [&str; 4] = ["tethered", "ball", "buridan", "fourfield"];
pub const COLS: [&str; 12] = ["t", "omega_ext", "slip", "steer_hz", "yaw_rate", "psi", "x", "y", "speed", "turn_r", "turn_l", "aux"];
pub const ODOR_QUADRANTS: [i64; 2] = [0, 2];

fn yaw_gain() -> f64 {
    0.004 * 60.0
}

fn yaw_max() -> f64 {
    0.035 * 60.0
}

fn fov_half() -> f64 {
    150.0f64.to_radians()
}

pub fn rig_title(rig: &str) -> &'static str {
    match rig {
        "tethered" => "Tethered flight simulator",
        "ball" => "Fly on a ball",
        "buridan" => "Buridan's paradigm",
        _ => "Four-field olfactory arena",
    }
}

/// `RIG_TAGS`: `(name, tag)` in the order Python's dict has.
pub fn rig_tags(rig: &str) -> Vec<(&'static str, &'static str)> {
    match rig {
        "tethered" => vec![
            ("yaw_from_DNa01_02", "CONNECTOME"),
            ("optomotor_input", "CONNECTOME (through a GAME RULE EMD stage)"),
            ("gain_loop_panorama", "GAME RULE"),
            ("response", "MODEL PREDICTION"),
        ],
        "ball" => vec![
            ("turning_and_walking_from_DNa01_02_and_DNp09", "CONNECTOME"),
            ("ball_and_VR", "GAME RULE"),
            ("response", "MODEL PREDICTION"),
        ],
        "buridan" => vec![
            ("steering_and_walking_from_DNs", "CONNECTOME"),
            ("stripes_edge_attention", "GAME RULE"),
            ("trajectory", "MODEL PREDICTION"),
        ],
        _ => vec![("odor_to_ORNs", "CONNECTOME"), ("arena_odor_quadrants_PI_definition", "GAME RULE"), ("preference", "MODEL PREDICTION")],
    }
}

pub fn wrap(a: f64) -> f64 {
    (a + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI) - std::f64::consts::PI
}

/// The optomotor transduction (`Transducer`): a world rotation relative to the fly becomes current on the T4/T5 subtypes it matches.
fn drive_slip(br: &mut Brain, g: &Groups, slip: f64) {
    for (rows, amp) in validation::emd_stage(g, slip) {
        br.drive(&rows, amp);
    }
}

fn release(br: &mut Brain, g: &Groups) {
    for k in ["t45_prog_r", "t45_prog_l", "t45_regr_r", "t45_regr_l"] {
        br.undrive(&g[k]);
    }
}

/// Yaw rate from the steering neurons: the duel's rule.
#[derive(Default)]
pub struct Steerer {
    pub s: f64,
}

impl Steerer {
    pub fn update(&mut self, br: &Brain) -> f64 {
        let turn = br.hz64("turn_r") - br.hz64("turn_l");
        self.s += (turn - self.s) * STEER_SMOOTH;
        let mag = (self.s.abs() - YAW_DEADZONE_HZ).max(0.0);
        if mag > 0.0 { (mag * yaw_gain()).min(yaw_max()).copysign(self.s) } else { 0.0 }
    }
}

/// `walk_speed`: forward speed from DNp09 through the race's speed rule (version 2), m/s.
pub fn walk_speed(br: &Brain) -> f64 {
    crate::race::speed_of(crate::race::walk_level(br.hz64("walk")), crate::race::T_WALK)
}

fn advance(br: &mut Brain) {
    for _ in 0..TICK_STEPS {
        br.step();
    }
}

/// `track_object`: LC10 tracking of one object at relative bearing `beta` (positive = on the right) and distance `dist`.
pub fn track_object(br: &mut Brain, beta: f64, dist: f64, see_range: f64) -> f64 {
    if dist > see_range || beta.abs() > fov_half() {
        return 0.0;
    }
    let (ahead, side) = (beta.cos(), beta.sin());
    let near = (1.3 - dist / 7.0).clamp(0.3, 1.0);
    let s_side = (side.abs() / TRACK_SCALE).clamp(0.0, 1.0) * near;
    if ahead < TRACK_AHEAD && s_side > 0.03 {
        br.poke_recruit64("track", Some(if side > 0.0 { 'R' } else { 'L' }), TRACK_POKE, Some(TRACK_RECRUIT.0 + TRACK_RECRUIT.1 * s_side));
        return s_side;
    }
    0.0
}

type Row = [f64; 12];
type Groups = std::collections::HashMap<&'static str, Vec<u32>>;

/// A finished run: the full-resolution rows and the dict the pages and exports use.
pub struct Run {
    pub full: Vec<Row>,
    pub dt: f64,
    pub meta: Value,
}

impl Run {
    fn new(rows: Vec<Row>, dt: f64, meta: Value) -> Self {
        Self { full: rows, dt, meta }
    }

    pub fn seconds(&self) -> f64 {
        (self.full.len() as f64 * self.dt * 1000.0).round_ties_even() / 1000.0
    }

    /// The decimated trace (`np.round(a[::TRACE_EVERY], 4)`).
    pub fn trace(&self) -> Vec<Vec<f64>> {
        self.full.iter().step_by(TRACE_EVERY).map(|r| r.iter().map(|v| round4(*v)).collect()).collect()
    }

    pub fn col(&self, name: &str) -> Vec<f64> {
        let i = COLS.iter().position(|c| *c == name).unwrap_or(0);
        self.full.iter().map(|r| r[i]).collect()
    }
}

fn round4(x: f64) -> f64 {
    format!("{x:.4}").parse().unwrap_or(x)
}

fn ticks_for(seconds: f64, dt: f64) -> usize {
    ((seconds / dt).round_ties_even() as usize).max(1)
}

fn hz_pair(br: &Brain) -> (f64, f64) {
    (br.hz64("turn_r"), br.hz64("turn_l"))
}

fn cancelled(cancel: Option<&std::sync::atomic::AtomicBool>) -> bool {
    cancel.is_some_and(|c| c.load(std::sync::atomic::Ordering::Relaxed))
}

// --- 1. tethered flight simulator -------------------------------------------------------------------------------------------------
/// `run_tethered`: the fly is fixed and the panorama rotates. `schedule` = `[(seconds, omega_ext)]`; closed loop subtracts the fly's own
/// yaw (times `gain`). aux = the phase index.
pub fn run_tethered(
    br: &mut Brain,
    g: &Groups,
    schedule: &[(f64, f64)],
    closed: bool,
    gain: f64,
    cancel: Option<&std::sync::atomic::AtomicBool>,
) -> Result<Run, String> {
    let mut st = Steerer::default();
    let dt = TICK_STEPS as f64 * 0.005;
    let plan: Vec<(usize, usize, f64)> = schedule.iter().enumerate().map(|(i, (sec, om))| (i, ticks_for(*sec, dt), *om)).collect();
    let (mut rows, mut psi, mut yaw, mut t) = (Vec::new(), 0.0f64, 0.0f64, 0.0f64);
    for (i, n, om) in plan {
        for _ in 0..n {
            if cancelled(cancel) {
                release(br, g);
                return Err("cancelled".into());
            }
            let slip = om - if closed { gain * yaw } else { 0.0 };
            drive_slip(br, g, slip);
            advance(br);
            yaw = st.update(br);
            psi = wrap(psi + yaw * dt);
            t += dt;
            let (r, l) = hz_pair(br);
            rows.push([t, om, slip, st.s, yaw, psi, 0.0, 0.0, 0.0, r, l, i as f64]);
        }
    }
    release(br, g);
    Ok(Run::new(rows, dt, json!({"rig": "tethered", "mode": if closed { "closed" } else { "open" }, "gain": gain})))
}

// --- 2. fly on a ball -------------------------------------------------------------------------------------------------------------
pub struct BallOptions<'a> {
    pub bar_scene: bool,
    pub closed: bool,
    pub seconds: f64,
    pub omega_ext: &'a dyn Fn(f64) -> f64,
    pub bar_offset: f64,
    pub bar_visible: bool,
}

/// `run_ball`: a spherical treadmill with a VR world driven by the ball. aux = the bar's relative bearing in the bar scene.
pub fn run_ball(br: &mut Brain, g: &Groups, o: &BallOptions, cancel: Option<&std::sync::atomic::AtomicBool>) -> Result<Run, String> {
    let mut st = Steerer::default();
    let dt = TICK_STEPS as f64 * 0.005;
    let ticks = ticks_for(o.seconds, dt);
    let (mut x, mut y, mut psi, mut yaw, mut t) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut rows = Vec::new();
    for _ in 0..ticks {
        if cancelled(cancel) {
            release(br, g);
            return Err("cancelled".into());
        }
        let om = if o.bar_scene { 0.0 } else { (o.omega_ext)(t) };
        let slip = om - if o.closed { yaw } else { 0.0 };
        drive_slip(br, g, slip);
        let mut beta = 0.0;
        if o.bar_scene {
            beta = if o.closed { wrap(o.bar_offset - psi) } else { wrap(o.bar_offset) };
            if o.bar_visible {
                track_object(br, beta, BALL_BAR_DISTANCE, 7.5);
            }
        }
        advance(br);
        yaw = st.update(br);
        let v = walk_speed(br);
        if o.closed {
            psi = wrap(psi + yaw * dt);
            x += v * psi.sin() * dt;
            y += v * psi.cos() * dt;
        } else {
            psi = wrap(psi + yaw * dt);
        }
        t += dt;
        let (r, l) = hz_pair(br);
        rows.push([t, om, slip, st.s, yaw, psi, x, y, v, r, l, beta]);
    }
    release(br, g);
    Ok(Run::new(rows, dt, json!({"rig": "ball"})))
}

// --- 3. Buridan's paradigm --------------------------------------------------------------------------------------------------------
fn stripe_points() -> [[f64; 2]; 2] {
    [90.0f64, 270.0].map(|d| {
        let a = d.to_radians();
        [STRIPE_DISTANCE * a.sin(), STRIPE_DISTANCE * a.cos()]
    })
}

fn hypot(a: f64, b: f64) -> f64 {
    a.hypot(b)
}

fn reflect_circle(pos: [f64; 2], psi: f64, radius: f64) -> ([f64; 2], f64, bool) {
    let r = hypot(pos[0], pos[1]);
    if r < radius {
        return (pos, psi, false);
    }
    let n = [pos[0] / r, pos[1] / r];
    let f = [psi.sin(), psi.cos()];
    let dot = f[0] * n[0] + f[1] * n[1];
    if dot <= 0.0 {
        return ([pos[0] / r * (radius - 1e-3), pos[1] / r * (radius - 1e-3)], psi, false);
    }
    let f2 = [f[0] - 2.0 * dot * n[0], f[1] - 2.0 * dot * n[1]];
    ([n[0] * (radius - 1e-3), n[1] * (radius - 1e-3)], f2[0].atan2(f2[1]), true)
}

/// `run_buridan`: a round platform with two opposite stripes beyond its edge. aux = the number of edge contacts so far.
pub fn run_buridan(
    br: &mut Brain,
    seconds: f64,
    stripes: bool,
    start_heading: f64,
    cancel: Option<&std::sync::atomic::AtomicBool>,
) -> Result<Run, String> {
    let mut st = Steerer::default();
    let dt = TICK_STEPS as f64 * 0.005;
    let ticks = ticks_for(seconds, dt);
    let pts = stripe_points();
    let (mut pos, mut psi, mut t) = ([0.0f64; 2], start_heading, 0.0f64);
    let (mut rows, mut edges) = (Vec::new(), 0i64);
    for _ in 0..ticks {
        if cancelled(cancel) {
            return Err("cancelled".into());
        }
        if stripes {
            let mut best: Option<(f64, [f64; 2])> = None;
            for p in &pts {
                let d = [p[0] - pos[0], p[1] - pos[1]];
                let ab = wrap(d[0].atan2(d[1]) - psi).abs();
                if best.is_none_or(|b| ab < b.0) {
                    best = Some((ab, d));
                }
            }
            let (_, d) = best.expect("two stripes");
            track_object(br, wrap(d[0].atan2(d[1]) - psi), hypot(d[0], d[1]), 7.5);
        }
        advance(br);
        let yaw = st.update(br);
        let v = walk_speed(br);
        psi = wrap(psi + yaw * dt);
        pos = [pos[0] + v * dt * psi.sin(), pos[1] + v * dt * psi.cos()];
        let (p2, ps2, hit) = reflect_circle(pos, psi, PLATFORM_RADIUS);
        pos = p2;
        psi = ps2;
        edges += i64::from(hit);
        t += dt;
        let (r, l) = hz_pair(br);
        rows.push([t, 0.0, 0.0, st.s, yaw, psi, pos[0], pos[1], v, r, l, edges as f64]);
    }
    Ok(Run::new(
        rows,
        dt,
        json!({"rig": "buridan", "stripes": stripes, "platform_radius": PLATFORM_RADIUS, "stripe_distance": STRIPE_DISTANCE,
               "stripe_bearings_deg": [90.0, 270.0]}),
    ))
}

// --- 4. olfactory four-field arena -------------------------------------------------------------------------------------------------
pub fn quadrant(x: f64, y: f64) -> i64 {
    if x >= 0.0 && y >= 0.0 {
        0
    } else if x >= 0.0 && y < 0.0 {
        1
    } else if x < 0.0 && y < 0.0 {
        2
    } else {
        3
    }
}

/// `run_fourfield`: the odor in two opposite quadrants, air in the other two; `deliver = false` is the sham. aux = the quadrant.
pub fn run_fourfield(
    br: &mut Brain,
    seconds: f64,
    odor: &str,
    deliver: bool,
    start_heading: f64,
    cancel: Option<&std::sync::atomic::AtomicBool>,
) -> Result<Run, String> {
    let mut st = Steerer::default();
    let dt = TICK_STEPS as f64 * 0.005;
    let ticks = ticks_for(seconds, dt);
    let (mut pos, mut psi, mut t) = ([1e-3f64; 2], start_heading, 0.0f64);
    let mut rows = Vec::new();
    let region = format!("scent:{odor}");
    for _ in 0..ticks {
        if cancelled(cancel) {
            return Err("cancelled".into());
        }
        let q = quadrant(pos[0], pos[1]);
        if deliver && ODOR_QUADRANTS.contains(&q) {
            br.poke64(&region, None, SCENT_POKE);
        }
        advance(br);
        let yaw = st.update(br);
        let v = walk_speed(br);
        psi = wrap(psi + yaw * dt);
        pos = [pos[0] + v * dt * psi.sin(), pos[1] + v * dt * psi.cos()];
        for ax in 0..2 {
            if pos[ax].abs() > FIELD_HALF {
                pos[ax] = (FIELD_HALF - 1e-3).copysign(pos[ax]);
                let mut f = [psi.sin(), psi.cos()];
                f[ax] = -f[ax];
                psi = f[0].atan2(f[1]);
            }
        }
        t += dt;
        let (r, l) = hz_pair(br);
        rows.push([t, 0.0, 0.0, st.s, yaw, psi, pos[0], pos[1], v, r, l, q as f64]);
    }
    Ok(Run::new(
        rows,
        dt,
        json!({"rig": "fourfield", "odor": odor, "delivered": deliver, "field_half": FIELD_HALF, "odor_quadrants": ODOR_QUADRANTS}),
    ))
}

// --- the measures ------------------------------------------------------------------------------------------------------------------
fn after(full: &[Row], settle: f64) -> Vec<&Row> {
    full.iter().filter(|r| r[0] > settle).collect()
}

/// `preference_index`: PI = (time in odor quadrants - time in air quadrants) / (their sum) after `settle_s`, and the mean speed in each.
pub fn preference_index(full: &[Row], dt: f64, settle_s: f64) -> Value {
    let a = after(full, settle_s);
    let odor: Vec<bool> = a.iter().map(|r| ODOR_QUADRANTS.contains(&(r[11] as i64))).collect();
    let n_o = odor.iter().filter(|&&b| b).count();
    let n_a = odor.len() - n_o;
    let pi = (n_o as f64 - n_a as f64) / (n_o + n_a).max(1) as f64;
    let mean = |sel: bool| {
        let v: Vec<f64> = a.iter().zip(&odor).filter(|(_, o)| **o == sel).map(|(r, _)| r[8]).collect();
        if v.is_empty() { f64::NAN } else { np_sum(&v) / v.len() as f64 }
    };
    json!({"pi": pi, "seconds_odor": ((n_o as f64 * dt) * 100.0).round() / 100.0, "seconds_air": ((n_a as f64 * dt) * 100.0).round() / 100.0,
           "speed_odor": fnum(mean(true)), "speed_air": fnum(mean(false))})
}

fn fnum(x: f64) -> Value {
    if x.is_finite() { json!(x) } else { Value::Null }
}

/// `stripe_deviation`: the median angle between the direction of travel and the east-west axis, folded to 0-90 degrees.
pub fn stripe_deviation(full: &[Row], settle_s: f64, min_step: f64) -> f64 {
    let a = after(full, settle_s);
    let mut angs = Vec::new();
    for w in a.windows(2) {
        let (dx, dy) = (w[1][6] - w[0][6], w[1][7] - w[0][7]);
        if hypot(dx, dy) > min_step {
            angs.push(dy.abs().atan2(dx.abs()).to_degrees());
        }
    }
    median(&mut angs)
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}

/// `transits`: the east-west position must reach the east edge zone and then the west one (or the reverse); each crossing counts once.
pub fn transits(full: &[Row], settle_s: f64, zone: f64) -> i64 {
    let (mut n, mut side) = (0, 0i32);
    for r in after(full, settle_s) {
        let v = r[6] / PLATFORM_RADIUS;
        let s = if v > zone {
            1
        } else if v < -zone {
            -1
        } else {
            0
        };
        if s != 0 && side != 0 && s != side {
            n += 1;
        }
        if s != 0 {
            side = s;
        }
    }
    n
}

// --- per-fly measurements (`rigassay`) -------------------------------------------------------------------------------------------------
pub const DEFAULT_MODE: &str = "subtle";
pub const CONTROL_MODE: &str = "off";
pub const TETHER_PRE_S: f64 = 2.0;
pub const TETHER_STIM_S: f64 = 4.0;
pub const TETHER_OMEGAS: [f64; 4] = [3.0, -3.0, 1.0, -1.0];
pub const TETHER_CLOSED_S: f64 = 10.0;
pub const TETHER_CLOSED_OMEGAS: [f64; 2] = [1.0, -1.0];
pub const SETTLE_S: f64 = 0.5;
pub const LAST_S: f64 = 5.0;
pub const BALL_DRIFT_S: f64 = 16.0;
pub const BALL_BAR_S: f64 = 20.0;
pub const BAR_OFFSETS_DEG: [f64; 2] = [90.0, -90.0];
pub const BURIDAN_S: f64 = 120.0;
pub const FOURFIELD_S: f64 = 120.0;
pub const ODOR: &str = "fruit";
pub const P_MAX: f64 = 0.01;

/// `_win(a, t0, t1, col).mean()`.
fn win_mean(run: &Run, t0: f64, t1: f64, col: &str) -> f64 {
    let i = COLS.iter().position(|c| *c == col).unwrap_or(0);
    let v: Vec<f64> = run.full.iter().filter(|r| r[0] > t0 && r[0] <= t1).map(|r| r[i]).collect();
    if v.is_empty() { f64::NAN } else { np_sum(&v) / v.len() as f64 }
}

fn win_abs_mean(run: &Run, t0: f64, t1: f64, col: &str) -> f64 {
    let i = COLS.iter().position(|c| *c == col).unwrap_or(0);
    let v: Vec<f64> = run.full.iter().filter(|r| r[0] > t0 && r[0] <= t1).map(|r| r[i].abs()).collect();
    if v.is_empty() { f64::NAN } else { np_sum(&v) / v.len() as f64 }
}

fn mean(v: &[f64]) -> f64 {
    np_sum(v) / v.len() as f64
}

/// A brain for one fly (`simcore.new_brain(seed, individuality=mode)`).
pub fn new_brain(weights: &Arc<Weights>, labels: &Arc<Labels>, seed: u64, mode: &str) -> Brain {
    let build = Build { weights, labels, mode, sigma: None, lif: crate::cards::LIF_DEFAULTS, wired: None };
    build.brain_with(seed, None, 600, true)
}

fn key(x: f64) -> String {
    crate::cards::py_float(x)
}

fn start_heading(seed: u64) -> f64 {
    crate::rng::Generator::new(seed + 770).uniform(-std::f64::consts::PI, std::f64::consts::PI)
}

pub struct Ctx<'a> {
    pub weights: &'a Arc<Weights>,
    pub labels: &'a Arc<Labels>,
    pub cancel: Option<&'a std::sync::atomic::AtomicBool>,
}

pub fn tethered_fly(c: &Ctx, seed: u64, mode: &str) -> Result<Value, String> {
    let mut br = new_brain(c.weights, c.labels, seed, mode);
    let g = validation::groups(c.labels, c.weights);
    let (mut ds, mut dr, mut dl) = (Vec::new(), Vec::new(), Vec::new());
    for om in TETHER_OMEGAS {
        let r = run_tethered(&mut br, &g, &[(TETHER_PRE_S, 0.0), (TETHER_STIM_S, om)], false, 1.0, c.cancel)?;
        let pre = win_mean(&r, 0.5, TETHER_PRE_S, "steer_hz");
        let stim = win_mean(&r, TETHER_PRE_S + SETTLE_S, TETHER_PRE_S + TETHER_STIM_S, "steer_hz");
        ds.push((om, stim - pre));
        dr.push((om, win_mean(&r, TETHER_PRE_S + SETTLE_S, 99.0, "turn_r") - win_mean(&r, 0.5, TETHER_PRE_S, "turn_r")));
        dl.push((om, win_mean(&r, TETHER_PRE_S + SETTLE_S, 99.0, "turn_l") - win_mean(&r, 0.5, TETHER_PRE_S, "turn_l")));
    }
    let (mut comp, mut slip) = (Vec::new(), Vec::new());
    for om in TETHER_CLOSED_OMEGAS {
        let r = run_tethered(&mut br, &g, &[(TETHER_PRE_S, 0.0), (TETHER_CLOSED_S, om)], true, 1.0, c.cancel)?;
        let end = TETHER_PRE_S + TETHER_CLOSED_S;
        let yaw_pre = win_mean(&r, 0.5, TETHER_PRE_S, "yaw_rate");
        let yaw_end = win_mean(&r, end - LAST_S, end, "yaw_rate");
        comp.push((om, (yaw_end - yaw_pre) / om));
        slip.push((om, win_abs_mean(&r, end - LAST_S, end, "slip")));
    }
    let get = |v: &[(f64, f64)], om: f64| v.iter().find(|x| x.0 == om).map_or(f64::NAN, |x| x.1);
    let obj = |v: &[(f64, f64)]| -> Value { v.iter().map(|(k, x)| (key(*k), json!(x))).collect::<serde_json::Map<_, _>>().into() };
    Ok(json!({
        "seed": seed, "individuality": mode, "delta_steer_hz": obj(&ds), "delta_R_hz": obj(&dr), "delta_L_hz": obj(&dl),
        "d_directional": get(&ds, 3.0) - get(&ds, -3.0), "compensation": obj(&comp),
        "compensation_mean": mean(&comp.iter().map(|x| x.1).collect::<Vec<_>>()), "closed_slip": obj(&slip),
    }))
}

pub fn ball_fly(c: &Ctx, seed: u64, mode: &str) -> Result<Value, String> {
    let mut br = new_brain(c.weights, c.labels, seed, mode);
    let g = validation::groups(c.labels, c.weights);
    let (mut comp, mut fwd) = (Vec::new(), Vec::new());
    for om in TETHER_CLOSED_OMEGAS {
        let f = move |t: f64| if t > TETHER_PRE_S { om } else { 0.0 };
        let o = BallOptions {
            bar_scene: false,
            closed: true,
            seconds: TETHER_PRE_S + BALL_DRIFT_S,
            omega_ext: &f,
            bar_offset: 90.0f64.to_radians(),
            bar_visible: true,
        };
        let r = run_ball(&mut br, &g, &o, c.cancel)?;
        let end = TETHER_PRE_S + BALL_DRIFT_S;
        comp.push((om, (win_mean(&r, end - LAST_S, end, "yaw_rate") - win_mean(&r, 0.5, TETHER_PRE_S, "yaw_rate")) / om));
        fwd.push(win_mean(&r, 0.5, end, "speed"));
    }
    let (mut vis, mut hid) = (Vec::new(), Vec::new());
    for visible in [true, false] {
        for off in BAR_OFFSETS_DEG {
            let f = |_t: f64| 0.0;
            let o = BallOptions {
                bar_scene: true,
                closed: true,
                seconds: BALL_BAR_S,
                omega_ext: &f,
                bar_offset: off.to_radians(),
                bar_visible: visible,
            };
            let r = run_ball(&mut br, &g, &o, c.cancel)?;
            let beta = win_abs_mean(&r, BALL_BAR_S - LAST_S, BALL_BAR_S, "aux").to_degrees();
            if visible {
                vis.push(beta)
            } else {
                hid.push(beta)
            }
            fwd.push(win_mean(&r, 0.5, BALL_BAR_S, "speed"));
        }
    }
    let cm: Vec<f64> = comp.iter().map(|x| x.1).collect();
    Ok(json!({
        "seed": seed, "individuality": mode,
        "compensation": comp.iter().map(|(k, x)| (key(*k), json!(x))).collect::<serde_json::Map<_, _>>(),
        "compensation_mean": mean(&cm), "bar_error_deg_visible": mean(&vis), "bar_error_deg_hidden": mean(&hid),
        "bar_error_deg_by_offset": {"visible": vis, "hidden": hid}, "forward_speed_mean": mean(&fwd),
    }))
}

pub fn buridan_fly(c: &Ctx, seed: u64, mode: &str, seconds: f64) -> Result<Value, String> {
    let mut out = serde_json::Map::new();
    for (k, on) in [("stripes", true), ("none", false)] {
        let mut br = new_brain(c.weights, c.labels, seed, mode);
        let r = run_buridan(&mut br, seconds, on, start_heading(seed), c.cancel)?;
        let f = &r.full;
        let path: Vec<f64> = f.windows(2).map(|w| hypot(w[1][6] - w[0][6], w[1][7] - w[0][7])).collect();
        out.insert(
            k.into(),
            json!({"deviation_deg": fnum(stripe_deviation(f, 10.0, 1e-4)), "transits": transits(f, 10.0, 0.8),
                   "edge_contacts": f.last().map_or(0, |l| l[11] as i64), "speed": mean(&f.iter().map(|r| r[8]).collect::<Vec<_>>()),
                   "path": np_sum(&path)}),
        );
    }
    let mut o = json!({"seed": seed, "individuality": mode});
    for (k, v) in out {
        o[k] = v;
    }
    Ok(o)
}

pub fn fourfield_fly(c: &Ctx, seed: u64, mode: &str, seconds: f64) -> Result<Value, String> {
    let mut o = json!({"seed": seed, "individuality": mode});
    for (k, dl) in [("odor", true), ("sham", false)] {
        let mut br = new_brain(c.weights, c.labels, seed, mode);
        let r = run_fourfield(&mut br, seconds, ODOR, dl, start_heading(seed), c.cancel)?;
        o[k] = preference_index(&r.full, r.dt, 10.0);
    }
    Ok(o)
}

/// `FLY[rig](seed, mode)` with the rig's own default duration.
pub fn fly(c: &Ctx, rig: &str, seed: u64, mode: &str) -> Result<Value, String> {
    match rig {
        "tethered" => tethered_fly(c, seed, mode),
        "ball" => ball_fly(c, seed, mode),
        "buridan" => buridan_fly(c, seed, mode, BURIDAN_S),
        "fourfield" => fourfield_fly(c, seed, mode, FOURFIELD_S),
        _ => Err(format!("unknown rig {rig:?} (choose from {})", RIGS.join(", "))),
    }
}

// --- one scene run (`scene_run`) ---------------------------------------------------------------------------------------------------
/// The defaults of `SCENES[rig]["default"]`.
pub fn scene_defaults(rig: &str) -> Value {
    match rig {
        "tethered" => json!({"mode": "open", "omega": 3.0, "seconds": 14.0, "gain": 1.0}),
        "ball" => json!({"mode": "closed", "scene": "bar", "seconds": 30.0, "omega": 0.0, "bar_deg": 90.0}),
        "buridan" => json!({"mode": "stripes", "seconds": 60.0}),
        _ => json!({"mode": "odor", "seconds": 60.0}),
    }
}

pub fn scene_modes(rig: &str) -> [&'static str; 2] {
    match rig {
        "tethered" => ["open", "closed"],
        "ball" => ["closed", "open"],
        "buridan" => ["stripes", "none"],
        _ => ["odor", "sham"],
    }
}

/// `tethered_schedule`.
pub fn tethered_schedule(omega: f64, seconds: f64) -> Vec<(f64, f64)> {
    let u = seconds.max(5.0) / 9.0;
    vec![(u, 0.0), (3.0 * u, omega), (u, 0.0), (3.0 * u, -omega), (u, 0.0)]
}

/// `scene_run` without the recorder: one fly in one rig. `opt` overrides the defaults (mode, seconds, omega, gain, scene, bar_deg).
pub fn scene_run(c: &Ctx, rig: &str, seed: u64, individuality: &str, opt: &Value) -> Result<Value, String> {
    if !RIGS.contains(&rig) {
        return Err(format!("unknown rig {rig:?} (choose from {})", RIGS.join(", ")));
    }
    let mut p = scene_defaults(rig);
    for (k, v) in opt.as_object().into_iter().flatten() {
        if !v.is_null() {
            p[k] = v.clone();
        }
    }
    let mode = p["mode"].as_str().unwrap_or("").to_string();
    if !scene_modes(rig).contains(&mode.as_str()) {
        return Err(format!("{rig}: mode must be one of {}", scene_modes(rig).join(", ")));
    }
    for (k, lo, hi) in [("seconds", 5.0, 600.0), ("omega", -10.0, 10.0), ("gain", 0.0, 3.0), ("bar_deg", -180.0, 180.0)] {
        if p.get(k).is_some_and(|v| !v.as_f64().is_some_and(|x| x.is_finite() && lo <= x && x <= hi)) {
            return Err(format!("{rig}: {k} must be a number from {lo} to {hi}"));
        }
    }
    let scene = p["scene"].as_str().unwrap_or("bar").to_string();
    if scene != "panorama" && scene != "bar" {
        return Err(format!("{rig}: scene must be panorama or bar"));
    }
    let mut br = new_brain(c.weights, c.labels, seed, individuality);
    let g = validation::groups(c.labels, c.weights);
    let f64p = |k: &str| p[k].as_f64().unwrap_or(0.0);
    let (run, summ) = match rig {
        "tethered" => {
            let r =
                run_tethered(&mut br, &g, &tethered_schedule(f64p("omega"), f64p("seconds")), mode == "closed", f64p("gain"), c.cancel)?;
            let sh = r.col("steer_hz");
            let om = r.col("omega_ext");
            let sel = |f: &dyn Fn(f64) -> bool| -> f64 {
                let v: Vec<f64> = sh.iter().zip(&om).filter(|(_, o)| f(**o)).map(|(s, _)| *s).collect();
                if v.is_empty() { f64::NAN } else { np_sum(&v) / v.len() as f64 }
            };
            let slip: Vec<f64> = r.col("slip").iter().map(|x| x.abs()).collect();
            let summ = json!({"response_right_hz": fnum(sel(&|o| o > 0.0)), "response_left_hz": fnum(sel(&|o| o < 0.0)),
                              "calm_hz": fnum(sel(&|o| o == 0.0)), "mean_abs_slip": mean(&slip)});
            (r, summ)
        }
        "ball" => {
            let om = f64p("omega");
            let fo = move |_t: f64| om;
            let o = BallOptions {
                bar_scene: scene == "bar",
                closed: mode == "closed",
                seconds: f64p("seconds"),
                omega_ext: &fo,
                bar_offset: f64p("bar_deg").to_radians(),
                bar_visible: true,
            };
            let r = run_ball(&mut br, &g, &o, c.cancel)?;
            let last = f64p("seconds") - LAST_S;
            let yaw: Vec<f64> = r.col("yaw_rate").iter().map(|x| x.abs()).collect();
            let end = r.full.last().copied().unwrap_or([0.0; 12]);
            let summ = json!({
                "forward_speed_mean": mean(&r.col("speed")), "end_x": end[6], "end_y": end[7], "mean_abs_yaw_rate": mean(&yaw),
                "final_bar_error_deg": if scene == "bar" { json!(win_abs_mean(&r, last, 1e9, "aux").to_degrees()) } else { Value::Null },
            });
            (r, summ)
        }
        "buridan" => {
            let r = run_buridan(&mut br, f64p("seconds"), mode == "stripes", start_heading(seed), c.cancel)?;
            let f = &r.full;
            let summ = json!({"stripe_deviation_deg": fnum(stripe_deviation(f, 10.0, 1e-4)), "transits": transits(f, 10.0, 0.8),
                              "edge_contacts": f.last().map_or(0, |l| l[11] as i64), "mean_speed": mean(&r.col("speed"))});
            (r, summ)
        }
        _ => {
            let r = run_fourfield(&mut br, f64p("seconds"), ODOR, mode == "odor", start_heading(seed), c.cancel)?;
            let summ = preference_index(&r.full, r.dt, 10.0);
            (r, summ)
        }
    };
    let mut geometry = serde_json::Map::new();
    for k in ["platform_radius", "stripe_distance", "stripe_bearings_deg", "field_half", "odor_quadrants"] {
        if let Some(v) = run.meta.get(k) {
            geometry.insert(k.into(), v.clone());
        }
    }
    if rig == "ball" {
        geometry.insert("bar_offset".into(), json!(f64p("bar_deg").to_radians()));
    }
    Ok(json!({
        "kind": format!("rig_{rig}"), "rig": rig, "seed": seed, "individuality": individuality, "params": p, "cols": COLS,
        "trace": run.trace(), "seconds": run.seconds(), "dt": run.dt, "summary": summ,
        "tags": rig_tags(rig).iter().map(|(k, v)| (k.to_string(), json!(v))).collect::<serde_json::Map<_, _>>(),
        "title": rig_title(rig), "rule_version": RULE_VERSION, "geometry": geometry,
    }))
}

// --- the pre-registered analysis -----------------------------------------------------------------------------------------------------
fn wilcoxon_greater_one(d: &[f64]) -> f64 {
    let d: Vec<f64> = d.iter().copied().filter(|x| x.is_finite()).collect();
    if d.is_empty() {
        return 1.0;
    }
    validation::wilcoxon_greater(&d, &vec![0.0; d.len()])
}

fn crit(id: &str, label: &str, diffs: &[f64], min_effect: f64, unit: &str) -> Value {
    let d: Vec<f64> = diffs.iter().copied().filter(|x| x.is_finite()).collect();
    let p = wilcoxon_greater_one(&d);
    let m = if d.is_empty() { f64::NAN } else { mean(&d) };
    json!({"id": id, "label": label, "n": d.len(), "mean_effect": fnum(m), "sd": if d.len() > 1 { validation::sd(&d) } else { 0.0 },
           "min_effect": min_effect, "unit": unit, "p": p, "passed": !d.is_empty() && m >= min_effect && p < P_MAX})
}

fn col_of(flies: &[Value], f: &dyn Fn(&Value) -> f64) -> Vec<f64> {
    flies.iter().map(f).collect()
}

/// `analyze`: the criteria of one rig on its per-fly measurements.
pub fn analyze(rig: &str, flies: &[Value]) -> Result<Value, String> {
    let n = |v: &Value| v.as_f64().unwrap_or(f64::NAN);
    let mut crits: Vec<Value> = match rig {
        "tethered" => vec![
            crit(
                "T1",
                "open loop: right-minus-left steering after a rightward minus a leftward panorama rotation (3 rad/s), Hz",
                &col_of(flies, &|x| n(&x["d_directional"])),
                3.0,
                "Hz",
            ),
            crit(
                "T2",
                "closed loop: share of an imposed 1 rad/s rotation the fly cancels (yaw rate / rotation, calm baseline removed)",
                &col_of(flies, &|x| n(&x["compensation_mean"])),
                0.10,
                "fraction",
            ),
        ],
        "ball" => {
            let mut b2 = crit(
                "BL2",
                "closed-loop bar: heading error (deg) with the bar hidden minus with it visible, last 5 s of 20 s",
                &col_of(flies, &|x| n(&x["bar_error_deg_hidden"]) - n(&x["bar_error_deg_visible"])),
                10.0,
                "deg",
            );
            let vis = mean(&col_of(flies, &|x| n(&x["bar_error_deg_visible"])));
            b2["visible_mean_deg"] = json!(vis);
            b2["visible_below_30"] = json!(vis < 30.0);
            b2["passed"] = json!(b2["passed"].as_bool().unwrap_or(false) && vis < 30.0);
            vec![
                crit(
                    "BL1",
                    "closed loop, walking: share of an imposed 1 rad/s drift of the VR panorama the fly cancels",
                    &col_of(flies, &|x| n(&x["compensation_mean"])),
                    0.10,
                    "fraction",
                ),
                b2,
            ]
        }
        "buridan" => vec![
            crit(
                "B1",
                "stripe deviation (deg) without stripes minus with stripes (median over the run, 0 = along the stripe axis)",
                &col_of(flies, &|x| n(&x["none"]["deviation_deg"]) - n(&x["stripes"]["deviation_deg"])),
                10.0,
                "deg",
            ),
            crit(
                "B2",
                "transits between the stripes with stripes minus without",
                &col_of(flies, &|x| n(&x["stripes"]["transits"]) - n(&x["none"]["transits"])),
                3.0,
                "transits",
            ),
        ],
        "fourfield" => vec![crit(
            "F1",
            "preference index with the odor minus the sham (same arena and start, no odor), positive = prefers the odor quadrants",
            &col_of(flies, &|x| n(&x["odor"]["pi"]) - n(&x["sham"]["pi"])),
            0.15,
            "PI",
        )],
        _ => return Err(rig.to_string()),
    };
    let mut extra = serde_json::Map::new();
    if rig == "fourfield" {
        let so: Vec<f64> = col_of(flies, &|x| n(&x["odor"]["speed_odor"]) - n(&x["odor"]["speed_air"]));
        let finite: Vec<f64> = so.iter().copied().filter(|x| x.is_finite()).collect();
        extra.insert(
            "speed_odor_minus_air_m_s".into(),
            json!({"mean": if finite.is_empty() { f64::NAN } else { mean(&finite) }, "p_greater": wilcoxon_greater_one(&so),
                   "note": "reported, not a criterion (a kinesis would show here)"}),
        );
    }
    let all = crits.iter().all(|c| c["passed"].as_bool().unwrap_or(false));
    for c in &mut crits {
        c["passed"] = json!(c["passed"].as_bool().unwrap_or(false));
    }
    Ok(json!({"rig": rig, "n": flies.len(), "criteria": crits, "all_passed": all, "extra": extra}))
}

/// `run_flies`: the per-fly measurements of one rig, on `workers` threads. `progress(done, total, label)` returns false to cancel.
pub fn run_flies(
    c: &Ctx,
    rig: &str,
    seeds: &[u64],
    mode: &str,
    workers: usize,
    mut progress: impl FnMut(usize, usize, &str) -> bool,
) -> Result<Vec<Value>, String> {
    if !RIGS.contains(&rig) {
        return Err(format!("unknown rig {rig:?} (choose from {})", RIGS.join(", ")));
    }
    let t0 = std::time::Instant::now();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let (tx, rx) = std::sync::mpsc::channel::<(usize, Result<Value, String>)>();
    let mut out: Vec<Option<Value>> = vec![None; seeds.len()];
    let mut err: Option<String> = None;
    std::thread::scope(|sc| {
        for _ in 0..workers.max(1).min(seeds.len().max(1)) {
            let tx = tx.clone();
            let next = &next;
            sc.spawn(move || {
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= seeds.len() || cancelled(c.cancel) {
                        return;
                    }
                    let r = fly(c, rig, seeds[i], mode);
                    if tx.send((i, r)).is_err() {
                        return;
                    }
                }
            });
        }
        drop(tx);
        let mut done = 0;
        while let Ok((i, r)) = rx.recv() {
            match r {
                Ok(v) => {
                    done += 1;
                    let label = format!("{rig}, {mode}: fly {} done ({:.0} s)", seeds[i], t0.elapsed().as_secs_f64());
                    out[i] = Some(v);
                    if !progress(done, seeds.len(), &label) {
                        err = Some("cancelled".into());
                        if let Some(cn) = c.cancel {
                            cn.store(true, std::sync::atomic::Ordering::Relaxed);
                        }
                    }
                }
                Err(e) => err = Some(e),
            }
        }
    });
    if let Some(e) = err {
        return Err(e);
    }
    out.into_iter().map(|o| o.ok_or_else(|| "cancelled".to_string())).collect()
}

/// `run_assay`: the rig's per-fly measurements and criteria for the default mode and for the off control.
pub fn run_assay(
    c: &Ctx,
    rig: &str,
    seeds: &[u64],
    modes: &[&str],
    workers: usize,
    created: &str,
    mut progress: impl FnMut(usize, usize, &str) -> bool,
) -> Result<Value, String> {
    let t0 = std::time::Instant::now();
    let mut results = serde_json::Map::new();
    for mode in modes {
        let flies = run_flies(c, rig, seeds, mode, workers, &mut progress)?;
        let an = analyze(rig, &flies)?;
        results.insert((*mode).to_string(), json!({"flies": flies, "analysis": an}));
    }
    Ok(json!({
        "kind": format!("rig_assay_{rig}"), "rig": rig, "created": created, "seconds": (t0.elapsed().as_secs_f64() * 10.0).round() / 10.0,
        "seeds": seeds, "modes": modes, "tags": rig_tags(rig).iter().map(|(k, v)| (k.to_string(), json!(v))).collect::<serde_json::Map<_, _>>(),
        "title": rig_title(rig), "rule_version": RULE_VERSION, "results": results,
    }))
}

fn fmt_p(p: f64) -> String {
    if p.is_nan() {
        "p = n/a".into()
    } else if p < 0.001 {
        "p < 0.001".into()
    } else {
        format!("p = {p:.3}")
    }
}

/// `summary`.
pub fn summary(res: &Value) -> String {
    let seeds: Vec<i64> = res["seeds"].as_array().into_iter().flatten().filter_map(Value::as_i64).collect();
    let mut lines = vec![format!(
        "{} ASSAY, {} flies (seeds {}-{})",
        res["title"].as_str().unwrap_or("").to_uppercase(),
        seeds.len(),
        seeds.first().copied().unwrap_or(0),
        seeds.last().copied().unwrap_or(0)
    )];
    let modes: Vec<String> = res["modes"].as_array().into_iter().flatten().filter_map(|m| m.as_str().map(String::from)).collect();
    for mode in modes {
        let r = &res["results"][&mode];
        lines.push(format!("  individuality {mode}{}:", if mode == CONTROL_MODE { "  (the control)" } else { "" }));
        for c in r["analysis"]["criteria"].as_array().into_iter().flatten() {
            lines.push(format!(
                "    {} {}: {}: {} {} (needs >= {}), {}, n = {}",
                c["id"].as_str().unwrap_or(""),
                if c["passed"].as_bool().unwrap_or(false) { "PASS" } else { "FAIL" },
                c["label"].as_str().unwrap_or(""),
                format_args!("{}", signed_g3(c["mean_effect"].as_f64().unwrap_or(f64::NAN))),
                c["unit"].as_str().unwrap_or(""),
                crate::sensitivity::fmt_gp(c["min_effect"].as_f64().unwrap_or(0.0), 6),
                fmt_p(c["p"].as_f64().unwrap_or(f64::NAN)),
                c["n"]
            ));
        }
    }
    lines.join("\n")
}

/// Python's `f"{x:+.3g}"`.
fn signed_g3(x: f64) -> String {
    if x.is_nan() {
        return "+nan".into();
    }
    let s = crate::sensitivity::fmt_gp(x.abs(), 3);
    format!("{}{s}", if x.is_sign_negative() && x != 0.0 { "-" } else { "+" })
}
