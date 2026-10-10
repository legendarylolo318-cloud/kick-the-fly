//! Play-mode challenges (`lab/challenges.py`): "Teach it to pick the right door", "How close can you sneak?", "Find its sweet tooth", "Mystery
//! defect" and "Predict the move". The door and sweet-tooth protocols run on the brain's own thread (`fly_sim::challenge`); this module is
//! their panels, the sneak/predict/reverse logic that needs the body, the scores, and the hit areas. Puppeteer and the contraption belong to
//! the arcade (O) and the C key; Start on their cards opens those.
//!
//! The panels are recorded as [`ktf_ui::hud::Op`] lists, the pygame drawing calls of each `Challenge.draw`, and replayed by the HUD.
//!
//! Differences from Python, each a defect there: Mystery defect set the silenced neurons' override to 0.0 (which silences nothing), here
//! they get the real silencing current and are released at the end; Predict's result card read a `desc` key its readouts do not have (a
//! KeyError in Python), here it shows the cue text.

use bevy::prelude::*;
use ktf_ui::Rect;
use ktf_ui::hud::{FontSel, Op};
use ktf_ui::pages::challenges::stars;

use crate::app::{LatestSnapshot, SimEvents, SimLink};
use crate::config::Arena;
use crate::feedback::{Notes, Sfx, Tag};
use crate::fly::{Fly, FlyPart, Part, ReactionFired};
use crate::hud_canvas::HudStats;
use crate::paths::Scores;
use crate::player::{Aim, MouseLook};
use crate::sim_thread::{Command, SimEvent};
use crate::tools::{ToolState, tool_tip};

const PLAY_W: i32 = 890;
const H: i32 = 760;
const INK: [u8; 3] = [240, 243, 248];
const TEXT: [u8; 3] = [205, 212, 224];
const LABEL: [u8; 3] = [130, 142, 160];
const DIM: [u8; 3] = [80, 88, 102];
const AMBER: [u8; 3] = [255, 176, 64];
const GOOD: [u8; 3] = [90, 200, 120];
const BAD: [u8; 3] = [230, 90, 80];

fn unix_now() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64())
}

fn rect(x: i32, y: i32, w: i32, h: i32) -> Rect {
    Rect::new(x, y, w, h)
}

// --- the circuits of the mystery defect --------------------------------------------------------------------------------------------
struct Circuit {
    id: &'static str,
    play_label: &'static str,
    target_spec: &'static str,
    bio_name: &'static str,
    citation: &'static str,
    missing: &'static str,
    hints: [&'static str; 2],
}

const CIRCUITS: [Circuit; 5] = [
    Circuit {
        id: "jump",
        play_label: "Emergency visual jump",
        target_spec: "type:DNp01",
        bio_name: "DNp01 (Giant Fiber escape command)",
        citation: "von Reyn et al. 2014, Nat Neurosci 17:962",
        missing: "It ignores rapid looming shadows and flyswatters without jumping or dodging.",
        hints: [
            "Try waving the flyswatter fast toward the fly or moving quickly near it.",
            "Normal flies dodge sudden approaching shadows; this one doesn't flinch.",
        ],
    },
    Circuit {
        id: "wings",
        play_label: "Wing take-off & flight",
        target_spec: "prefix:DNg02",
        bio_name: "DNg02 (Wing power & take-off command)",
        citation: "Shiu et al. 2024, Nature 634:210",
        missing: "It runs along the floor when startled, but its wings never lift it into flight.",
        hints: [
            "Try prodding the fly or giving it a puff of air to launch into flight.",
            "It can scramble along the ground, but its wings never produce flight lift.",
        ],
    },
    Circuit {
        id: "sweet",
        play_label: "Sweet taste & sugar reach",
        target_spec: "sweet",
        bio_name: "Sweet GRNs & SEL pathway (Proboscis extension to sugar)",
        citation: "Shiu et al. 2024, Nature 634:210; Yao & Scott 2022",
        missing: "It never unrolls or extends its proboscis when touching sugar droplets.",
        hints: [
            "Place a drop of sugar near its feet or mouthparts.",
            "It completely ignores sugar and won't extend its feeding tube to eat.",
        ],
    },
    Circuit {
        id: "fear",
        play_label: "Fear of zapped smells",
        target_spec: "type:PPL101,PPL103",
        bio_name: "PPL1 Dopaminergic neurons (Aversive shock reinforcement)",
        citation: "Claridge-Chang et al. 2009, Cell 139:405; Aso et al. 2014",
        missing: "It smells odors normally, but fails to learn which smell gave it an electric zap.",
        hints: [
            "Try pairing an odor scent with an electric zap in training.",
            "It can smell odors, but never forms an aversive memory of the shock.",
        ],
    },
    Circuit {
        id: "steering",
        play_label: "Steering & turning",
        target_spec: "type:DNa02",
        bio_name: "DNa02 (Steering descending command neurons)",
        citation: "Shiu et al. 2024, Nature 634:210; Rayshubskiy et al. 2020",
        missing: "It struggles to coordinate left vs right steering when encountering obstacles.",
        hints: [
            "Nudge the fly from the side or watch how it steers around objects.",
            "Its asymmetric steering is disabled; it struggles to turn cleanly away from obstacles.",
        ],
    },
];

/// (id, label, target readout, cue text, action description).
const READOUTS: [(&str, &str, &str, &str, &str); 5] = [
    ("jump", "Jump!", "jump", "HEAD-TOUCH DN SPIKE SURGE", "Jumped up!"),
    ("run", "Run!", "run", "BODY-TOUCH DN SPIKE SURGE", "Darted forward!"),
    ("kick", "Kick!", "kick", "LEG-TOUCH DN SPIKE SURGE", "Kicked its legs!"),
    ("back", "Back up!", "back", "MOONWALKER (MDN) SPIKE SURGE", "Walked backward!"),
    ("takeoff", "Take off!", "fly", "WING-POWER (DNg02) SPIKE SURGE", "Launched into flight!"),
];

fn motor_spec(target: &str) -> &'static str {
    match target {
        "jump" => "type:DNg85,DNg48,DNg37,DNge067,DNg29,DNge132",
        "run" => "type:DNge122,DNge104,DNge102,DNxl114,DNge182,DNge048",
        "kick" => "type:DNge074,DNge075,DNge096,DNg30,DNg34,DNp38",
        "back" => "type:MDN",
        _ => "type:DNg02_a,DNg02_b,DNg02_c,DNg02_d,DNg02_e,DNg02_f,DNg02_g",
    }
}

// --- state ---------------------------------------------------------------------------------------------------------------------------
#[derive(Default)]
struct TMazeUi {
    started: bool,
    best: bool,
    right_total: Option<usize>,
}

#[derive(Default)]
struct SneakUi {
    state: u8, // 0 armed, 1 sneaking, 2 result
    closest: f32,
    result: Option<(bool, f32)>, // (caught, distance)
    result_t: f32,
    hits0: u32,
    best: bool,
}

struct SweetUi {
    best: bool,
    track: Rect,
    /// The strength set with the bar.
    dose: u32,
}

struct ReverseUi {
    state: u8, // 0 testing, 1 guessing, 2 revealed
    hints: usize,
    circuit: usize,
    guess: Option<(bool, &'static str)>,
}

struct PredictUi {
    state: u8, // 0 ready, 1 cue, 2 result, 3 done
    round: usize,
    score: u32,
    readout: usize,
    timer: f32,
    history: Vec<bool>,
    rng: crate::rng::Rng,
    driven: Option<&'static str>,
}

enum Active {
    TMaze(TMazeUi),
    Sneak(SneakUi),
    Sweet(SweetUi),
    Reverse(ReverseUi),
    Predict(PredictUi),
}

#[derive(Resource, Default)]
pub struct Challenges {
    /// A key from the Challenges page (`tmaze`, `sneak`, ...), taken by `update`.
    pub request: Option<String>,
    active: Option<Active>,
    /// The panel as drawing steps, and the enabled buttons it had last frame.
    pub ops: Vec<Op>,
    hits: Vec<(Rect, String)>,
    /// Panels that block the tools (T-maze, sweet tooth).
    pub overlay: bool,
    /// Stars/message the game wants shown next frame: a one-shot note for the log.
    pub note: Option<String>,
}

impl Challenges {
    pub fn is_active(&self) -> bool {
        self.active.is_some()
    }
}

struct Panel;

impl Panel {
    /// `Challenge.panel`: the veil, the card, the title and subtitle. Returns the card.
    fn draw(ops: &mut Vec<Op>, title: &str, subtitle: &str, h: i32) -> Rect {
        let p = rect(40, 30, (PLAY_W - 80).min(820), (H - 60).min(h));
        ops.push(Op::Veil(140));
        ops.push(Op::Card { rect: p, fill: [18, 21, 28], alpha: 255, radius: 16 });
        ops.push(Op::Stroke { rect: p, color: [52, 60, 76], width: 1, radius: 16 });
        text(ops, &title.to_uppercase(), (p.x + 22, p.y + 14), INK, FontSel::Title, "topleft");
        text(ops, subtitle, (p.x + 24, p.y + 58), LABEL, FontSel::Text, "topleft");
        p
    }
}

fn text(ops: &mut Vec<Op>, s: &str, pos: (i32, i32), color: [u8; 3], font: FontSel, anchor: &'static str) {
    ops.push(Op::Text { s: s.into(), pos, color, font, anchor });
}

fn button(ops: &mut Vec<Op>, r: Rect, label: &str, id: &str, style: u8, enabled: bool) {
    ops.push(Op::Button { rect: r, label: label.into(), id: id.into(), style, enabled });
}

fn card(ops: &mut Vec<Op>, r: Rect, fill: [u8; 3], alpha: u32, border: Option<([u8; 3], i32)>) {
    ops.push(Op::Card { rect: r, fill, alpha, radius: 12 });
    if let Some((c, w)) = border {
        ops.push(Op::Stroke { rect: r, color: c, width: w, radius: 12 });
    }
}

fn py_g1(v: f32) -> String {
    format!("{v:.1}")
}

// --- the systems -------------------------------------------------------------------------------------------------------------------
#[derive(bevy::ecs::system::SystemParam)]
struct World3<'w, 's> {
    flies: Query<'w, 's, (Entity, &'static mut Fly)>,
    vel: Query<'w, 's, (&'static FlyPart, &'static mut avian3d::prelude::LinearVelocity)>,
    tf: Query<'w, 's, &'static Transform, With<FlyPart>>,
}

#[allow(clippy::too_many_arguments)]
fn update(
    mut ch: ResMut<Challenges>,
    virt: Res<Time<Virtual>>,
    events: Res<SimEvents>,
    snap: Res<LatestSnapshot>,
    link: Res<SimLink>,
    aim: Res<Aim>,
    tool: Res<ToolState>,
    hud: Res<HudStats>,
    mut look: ResMut<MouseLook>,
    mut notes: ResMut<Notes>,
    mut sfx: MessageWriter<Sfx>,
    mut fired: MessageReader<ReactionFired>,
    arena: Res<Arena>,
    mut w: World3,
) {
    let now = virt.elapsed_secs();
    let dodged = fired.read().any(|r| r.name == "DODGE");
    // starting (or ending) a challenge
    if let Some(key) = ch.request.take() {
        end(&mut ch, &link);
        if key == "end" {
            look.captured = true;
            return;
        }
        let rng_seed = (unix_now() * 1000.0) as u64;
        let active = match key.as_str() {
            "tmaze" => Active::TMaze(TMazeUi::default()),
            "sneak" => Active::Sneak(SneakUi { closest: f32::INFINITY, hits0: hud.hits, ..Default::default() }),
            "sweet" => {
                link.0.send(Command::StartSweet);
                Active::Sweet(SweetUi { best: false, track: rect(0, 0, 0, 0), dose: 50 })
            }
            "reverse_surgery" => {
                let mut rng = crate::rng::Rng::new(rng_seed);
                let i = rng.index(CIRCUITS.len());
                link.0.send(Command::Surgery { spec: CIRCUITS[i].target_spec.into(), mode: -1 });
                Active::Reverse(ReverseUi { state: 0, hints: 0, circuit: i, guess: None })
            }
            "predict" => Active::Predict(PredictUi {
                state: 0,
                round: 0,
                score: 0,
                readout: 0,
                timer: 0.0,
                history: vec![],
                rng: crate::rng::Rng::new(rng_seed ^ 999),
                driven: None,
            }),
            _ => return,
        };
        ch.overlay = matches!(active, Active::TMaze(_) | Active::Sweet(_));
        if ch.overlay {
            look.captured = false;
        }
        ch.active = Some(active);
    }
    let Some(active) = ch.active.take() else {
        ch.ops.clear();
        ch.hits.clear();
        ch.overlay = false;
        return;
    };
    let mut ops: Vec<Op> = vec![];
    let mut scores = Scores::default_file();
    let view = snap.0.as_deref().and_then(|s| s.challenge.clone());
    let sim_events: Vec<&SimEvent> = events.0.iter().collect();
    let mut active = active;
    match &mut active {
        Active::TMaze(t) => {
            for e in &sim_events {
                match e {
                    SimEvent::TMazeChose { right } => {
                        sfx.write(Sfx { name: if *right { "pop" } else { "zap" }, volume: 0.5 });
                    }
                    SimEvent::TMazeFinished { right } => {
                        t.right_total = Some(*right);
                        t.best = scores.record("tmaze", *right as f64, true);
                    }
                    _ => {}
                }
            }
            tmaze_ops(&mut ops, t, view.as_deref(), snap.0.as_deref().is_some_and(|s| s.memory.is_some()), &scores);
        }
        Active::Sneak(s) => {
            let Some((_, f)) = w.flies.iter().next() else { return };
            let head = w.tf.get(f.parts[Part::Head.index()]).map(|t| t.translation).unwrap_or(Vec3::ZERO);
            let mut d = Vec2::new(aim.eye.x - head.x, aim.eye.z - head.z).length();
            if !matches!(tool.name(), "hand" | "sugar" | "fruit" | "alcohol") {
                d = d.min((tool_tip(&aim) - head).length());
            }
            let d = d / 0.55;
            let airborne = now < f.react.escape_until;
            let dead = f.dead();
            if s.state == 2 {
                if now - s.result_t > 2.5 && d > 3.0 && !airborne && !dead {
                    *s = SneakUi { closest: f32::INFINITY, hits0: hud.hits, ..Default::default() };
                }
            } else if !dead {
                if s.state == 0 && d < 3.0 && !airborne {
                    s.state = 1;
                }
                if s.state == 1 {
                    s.closest = s.closest.min(d);
                    if hud.hits > s.hits0 || d < 0.35 {
                        let caught_dist = if d < 0.35 { 0.0 } else { s.closest };
                        finish_sneak(s, true, caught_dist, now, &mut scores, &mut sfx);
                    } else if dodged {
                        let c = s.closest;
                        finish_sneak(s, false, c, now, &mut scores, &mut sfx);
                    }
                }
            }
            sneak_ops(&mut ops, s, d, &scores);
        }
        Active::Sweet(sw) => {
            for e in &sim_events {
                match e {
                    SimEvent::SweetAnswer(dose, ext, ratio) => {
                        sfx.write(Sfx { name: if *ext { "yum" } else { "click" }, volume: 0.6 });
                        if *ext {
                            notes.push(now, format!("PROBOSCIS MN9 x{ratio:.1} to {dose}% sugar"), Tag::Connectome);
                        }
                    }
                    SimEvent::SweetFinished(score) => {
                        sw.best = score.is_some_and(|s| scores.record("sweet", f64::from(s), false));
                    }
                    _ => {}
                }
            }
            sweet_ops(&mut ops, sw, view.as_deref(), &scores);
        }
        Active::Reverse(r) => reverse_ops(&mut ops, r, &scores),
        Active::Predict(p) => {
            predict_update(p, now, &link, &mut sfx, &mut scores, &mut w, &arena);
            predict_ops(&mut ops, p, now);
        }
    }
    ch.hits = ops
        .iter()
        .filter_map(|o| if let Op::Button { rect, id, enabled: true, .. } = o { Some((*rect, id.clone())) } else { None })
        .collect();
    ch.ops = ops;
    ch.active = Some(active);
}

fn finish_sneak(s: &mut SneakUi, caught: bool, dist: f32, now: f32, scores: &mut Scores, sfx: &mut MessageWriter<Sfx>) {
    s.state = 2;
    s.result = Some((caught, dist));
    s.result_t = now;
    s.best = scores.record("sneak", f64::from((dist * 100.0).round() / 100.0), false);
    sfx.write(Sfx { name: if caught { "yum" } else { "dodge" }, volume: 0.6 });
}

/// Release everything a challenge holds: silenced circuits, a driven cue, the borrowed memory and speed.
fn end(ch: &mut Challenges, link: &SimLink) {
    if let Some(a) = ch.active.take() {
        match a {
            Active::Reverse(r) => {
                link.0.send(Command::Surgery { spec: CIRCUITS[r.circuit].target_spec.into(), mode: 0 });
            }
            Active::Predict(p) => {
                if let Some(t) = p.driven {
                    link.0.send(Command::DriveSpec { spec: motor_spec(t).into(), amp: 0.0 });
                }
            }
            Active::TMaze(_) | Active::Sweet(_) => {
                link.0.send(Command::EndChallenge);
            }
            Active::Sneak(_) => {}
        }
    }
    ch.ops.clear();
    ch.hits.clear();
    ch.overlay = false;
}

// --- the panels -------------------------------------------------------------------------------------------------------------------
fn tmaze_ops(ops: &mut Vec<Op>, t: &TMazeUi, view: Option<&crate::snapshot::ChallengeView>, mem_ok: bool, scores: &Scores) {
    let now = unix_now();
    let p = Panel::draw(ops, "Teach it to pick the right door", "One door zaps. Can the fly learn which one?", 580);
    let cx = p.centerx();
    let doors = [
        ("odor_a", rect(p.x + 120, p.y + 120, 150, 210), [255, 150, 60], "orange smell"),
        ("odor_b", rect(p.right() - 270, p.y + 120, 150, 210), [180, 120, 255], "purple smell"),
    ];
    let (phase, stage, cycle, zap, choices, anim_at) = match view.filter(|v| v.kind == "tmaze" && t.started) {
        Some(v) => (v.phase.as_str(), v.stage, v.cycle, v.zap.as_str(), v.choices.as_slice(), v.anim_at),
        None => ("choose", 0, 0, "", &[][..], 0.0),
    };
    let phase = if t.right_total.is_some() { "done" } else { phase };
    for (odor, r, col, name) in &doors {
        let dim = col.map(|c: u8| c / 3);
        card(ops, *r, dim, 255, Some((*col, 3)));
        for k in 0..3 {
            let pts: Vec<(i32, i32)> = (0..=60)
                .step_by(6)
                .map(|x| {
                    (
                        r.centerx() - 30 + x,
                        (r.y as f32 + 40.0 + k as f32 * 22.0 + 5.0 * (now as f32 * 3.0 + x as f32 / 8.0 + k as f32).sin()) as i32,
                    )
                })
                .collect();
            for seg in pts.windows(2) {
                ops.push(Op::Line { a: seg[0], b: seg[1], w: 2, color: *col });
            }
        }
        text(ops, name, (r.centerx(), r.bottom() + 8), *col, FontSel::Bold, "midtop");
        if zap == *odor {
            text(ops, "ZAP", (r.centerx(), r.bottom() - 40), AMBER, FontSel::Head, "center");
        }
    }
    let (mut fx, mut fy) = (cx as f32, (p.y + 330) as f32);
    if phase == "test" && stage == 3 && !choices.is_empty() {
        let door = doors.iter().find(|d| d.0 == choices[choices.len() - 1].0).map_or(doors[0].1, |d| d.1);
        let e = (((now - anim_at) / 0.7) as f32).min(1.0);
        fx += (door.centerx() as f32 - fx) * e;
        fy += ((door.bottom() - 20) as f32 - fy) * e;
    }
    ops.push(Op::Ellipse { rect: rect(fx as i32 - 14, fy as i32 - 9, 28, 18), color: [70, 60, 40] });
    ops.push(Op::Circle { c: (fx as i32 + 12, fy as i32 - 2), r: 5, color: [170, 40, 40] });
    let y = p.y + 390;
    match phase {
        "choose" => {
            text(ops, "Pick the door that zaps the fly:", (cx, y), TEXT, FontSel::Text, "midtop");
            button(ops, rect(p.x + 120, y + 34, 150, 44), "Orange zaps", "tmaze:odor_a", 0, mem_ok);
            button(ops, rect(p.right() - 270, y + 34, 150, 44), "Purple zaps", "tmaze:odor_b", 0, mem_ok);
        }
        "naive" | "train" => {
            let total = fly_sim::challenge::CYCLES as f32;
            let frac = if phase == "train" { (cycle as f32 + stage as f32 / 5.0) / total } else { 0.0 };
            let label = if phase == "naive" {
                "Letting it smell both doors...".to_string()
            } else {
                format!(
                    "Training {}/{}: {}",
                    cycle + 1,
                    total as u32,
                    if stage == 1 {
                        "smell + zap"
                    } else if stage == 0 || stage == 3 {
                        "smell"
                    } else {
                        "rest"
                    }
                )
            };
            text(ops, &label, (cx, y), TEXT, FontSel::Text, "midtop");
            ops.push(Op::Card { rect: rect(p.x + 60, y + 32, p.w - 120, 12), fill: [30, 36, 48], alpha: 255, radius: 6 });
            ops.push(Op::Card {
                rect: rect(p.x + 60, y + 32, 10.max(((p.w - 120) as f32 * frac) as i32), 12),
                fill: AMBER,
                alpha: 255,
                radius: 6,
            });
        }
        _ => {
            let right = choices.iter().filter(|c| c.1).count();
            let right = t.right_total.unwrap_or(right);
            text(
                ops,
                &format!("Choices: {}/{}    right door: {right}", choices.len(), fly_sim::challenge::TRIALS),
                (cx, y),
                INK,
                FontSel::Bold,
                "midtop",
            );
            for (i, (_, ok)) in choices.iter().enumerate() {
                ops.push(Op::Circle { c: (cx - 99 + i as i32 * 22, y + 40), r: 8, color: if *ok { GOOD } else { BAD } });
            }
            if phase == "done" {
                ops.push(Op::Stars { c: (cx as f32, (y + 84) as f32), n: stars("tmaze", right as f64), size: 14 });
                let msg = if t.best {
                    "New best!".to_string()
                } else {
                    format!("Best: {:.0}/10", scores.map.get("tmaze").copied().unwrap_or(right as f64))
                };
                text(ops, &msg, (cx, y + 106), AMBER, FontSel::Text, "midtop");
                button(ops, rect(cx - 200, p.bottom() - 64, 180, 44), "Play again", "again:tmaze", 1, true);
            }
        }
    }
    let close = if phase == "choose" || phase == "done" { "Close" } else { "Stop" };
    button(ops, rect(if phase != "done" { p.right() - 160 } else { cx + 20 }, p.bottom() - 64, 140, 44), close, "end", 0, true);
}

fn sneak_ops(ops: &mut Vec<Op>, s: &SneakUi, d: f32, scores: &Scores) {
    let (w, h) = (330, 120);
    let b = rect(PLAY_W / 2 - w / 2, 120, w, h);
    card(ops, b, [10, 12, 18], 200, None);
    text(ops, "HOW CLOSE CAN YOU SNEAK?", (b.x + 14, b.y + 8), AMBER, FontSel::Bold, "topleft");
    button(ops, rect(b.right() - 70, b.y + 6, 60, 26), "Quit", "end", 0, true);
    match s.state {
        0 => text(ops, "Back off, then creep toward the fly.", (b.x + 14, b.y + 38), TEXT, FontSel::Text, "topleft"),
        1 => {
            text(ops, &format!("{d:4.1} fly lengths away"), (b.x + 14, b.y + 36), INK, FontSel::Head, "topleft");
            text(ops, &format!("closest {}", py_g1(s.closest)), (b.x + 14, b.y + 66), TEXT, FontSel::Text, "topleft");
        }
        _ => {
            if let Some((caught, dist)) = s.result {
                let msg = if caught {
                    "You touched it before it saw you!".to_string()
                } else {
                    format!("It saw you at {} fly lengths", py_g1(dist))
                };
                text(ops, &msg, (b.x + 14, b.y + 36), if caught { GOOD } else { INK }, FontSel::Bold, "topleft");
                ops.push(Op::Stars { c: ((b.x + 70) as f32, (b.y + 76) as f32), n: stars("sneak", f64::from(dist)), size: 10 });
                if s.best {
                    text(ops, "New best!", (b.x + 140, b.y + 68), AMBER, FontSel::Text, "topleft");
                }
            }
        }
    }
    if let Some(best) = scores.map.get("sneak") {
        text(ops, &format!("best {best:.1}"), (b.right() - 14, b.bottom() - 24), LABEL, FontSel::Small, "topright");
    }
}

fn sweet_ops(ops: &mut Vec<Op>, sw: &mut SweetUi, view: Option<&crate::snapshot::ChallengeView>, scores: &Scores) {
    let p = Panel::draw(ops, "Find its sweet tooth", "Offer sugar. Find the weakest sugar it still reaches for.", 520);
    let cx = p.centerx();
    let v = view.filter(|v| v.kind == "sweet");
    let (phase, dose, offers, score) = match v {
        Some(v) => (v.phase.as_str(), v.dose, v.offers.as_slice(), v.score),
        None => ("ready", 50, &[][..], None),
    };
    let dose = if phase == "ready" { sw.dose } else { dose };
    text(ops, &format!("Sugar strength: {dose}%"), (p.x + 60, p.y + 110), INK, FontSel::Head, "topleft");
    let track = rect(p.x + 60, p.y + 160, p.w - 120, 8);
    sw.track = track;
    ops.push(Op::Card { rect: track, fill: [50, 56, 68], alpha: 255, radius: 4 });
    ops.push(Op::Card {
        rect: rect(track.x, track.y, (track.w as f32 * dose as f32 / 100.0) as i32, 8),
        fill: [255, 150, 190],
        alpha: 255,
        radius: 4,
    });
    ops.push(Op::Circle { c: (track.x + (track.w as f32 * dose as f32 / 100.0) as i32, track.centery()), r: 10, color: INK });
    text(ops, "click the bar to set the strength", (p.x + 60, p.y + 178), DIM, FontSel::Small, "topleft");
    let head = (cx, p.y + 270);
    ops.push(Op::Circle { c: head, r: 34, color: [120, 90, 50] });
    ops.push(Op::Circle { c: (head.0 - 18, head.1 - 10), r: 9, color: [190, 40, 40] });
    ops.push(Op::Circle { c: (head.0 + 18, head.1 - 10), r: 9, color: [190, 40, 40] });
    let ext = phase == "shown" && offers.last().is_some_and(|o| o.1);
    let length = if ext { 46 } else { 10 };
    ops.push(Op::Line { a: (head.0, head.1 + 26), b: (head.0, head.1 + 26 + length), w: 8, color: [150, 110, 70] });
    if phase == "before" || phase == "during" {
        text(ops, if phase == "during" { "tasting..." } else { "watching..." }, (cx, p.y + 350), TEXT, FontSel::Text, "midtop");
    } else if phase == "shown"
        && let Some((_, e, _)) = offers.last()
    {
        text(
            ops,
            if *e { "It reached for it!" } else { "Not interested." },
            (cx, p.y + 350),
            if *e { GOOD } else { LABEL },
            FontSel::Bold,
            "midtop",
        );
    }
    let row: Vec<String> = offers.iter().map(|(d, e, _)| format!("{d}%{}", if *e { "+" } else { "-" })).collect();
    text(
        ops,
        &format!("Tries {}/{}:  {}", offers.len(), fly_sim::challenge::SWEET_TRIES, row.join("  ")),
        (p.x + 30, p.bottom() - 110),
        TEXT,
        FontSel::Text,
        "topleft",
    );
    if phase == "done" {
        match score {
            None => text(ops, "It never reached for the sugar this time.", (cx, p.y + 384), INK, FontSel::Bold, "midtop"),
            Some(s) => {
                text(ops, &format!("Weakest sugar it reached for: {s}%"), (cx, p.y + 384), INK, FontSel::Bold, "midtop");
                ops.push(Op::Stars { c: (cx as f32, (p.y + 424) as f32), n: stars("sweet", f64::from(s)), size: 14 });
            }
        }
        button(ops, rect(cx - 200, p.bottom() - 64, 180, 44), "Play again", "again:sweet", 1, true);
        button(ops, rect(cx + 20, p.bottom() - 64, 140, 44), "Close", "end", 0, true);
    } else {
        button(ops, rect(cx - 90, p.bottom() - 64, 180, 44), "Offer sugar", &format!("offer:{dose}"), 1, phase == "ready");
        button(ops, rect(p.right() - 160, p.bottom() - 64, 140, 44), "Close", "end", 0, true);
    }
    if let Some(best) = scores.map.get("sweet") {
        text(ops, &format!("best: {best}%"), (p.right() - 30, p.y + 110), LABEL, FontSel::Text, "topright");
    }
}

fn reverse_ops(ops: &mut Vec<Op>, r: &ReverseUi, scores: &Scores) {
    let cx = (PLAY_W - 250).min(250.max(PLAY_W / 2));
    let c = &CIRCUITS[r.circuit];
    match r.state {
        0 => {
            let (w, h) = (480, 200);
            let pr = rect(cx - w / 2, 80, w, h);
            card(ops, pr, [14, 18, 26], 235, Some(([70, 90, 130], 1)));
            text(ops, "MYSTERY DEFECT (Reverse Brain Surgery)", (pr.x + 16, pr.y + 12), AMBER, FontSel::Bold, "topleft");
            text(
                ops,
                "One brain circuit is turned off! Test the fly with your tools.",
                (pr.x + 16, pr.y + 38),
                TEXT,
                FontSel::Small,
                "topleft",
            );
            let (txt, col) = match r.hints {
                0 => ("No hints used yet. Tap [Hint] if you need a clue.".to_string(), LABEL),
                1 => (format!("Hint 1: {}", c.hints[0]), [130, 200, 255]),
                _ => (format!("Hint 2: {}", c.hints[1]), [255, 210, 120]),
            };
            text(ops, &txt, (pr.x + 16, pr.y + 72), col, FontSel::Small, "topleft");
            if let Some(best) = scores.map.get("reverse_surgery") {
                text(ops, &format!("Best: {best} stars"), (pr.right() - 16, pr.y + 12), LABEL, FontSel::Small, "topright");
            }
            button(ops, rect(pr.x + 16, pr.bottom() - 52, 130, 38), &format!("Hint ({} left)", 2 - r.hints), "hint", 0, r.hints < 2);
            button(ops, rect(pr.x + 154, pr.bottom() - 52, 170, 38), "Make a Guess", "guessing", 1, true);
            button(ops, rect(pr.right() - 116, pr.bottom() - 52, 100, 38), "Quit", "end", 0, true);
        }
        1 => {
            let (w, h) = (480, 340);
            let pr = rect(cx - w / 2, 60, w, h);
            card(ops, pr, [14, 18, 26], 240, Some(([80, 110, 160], 1)));
            text(ops, "WHICH BEHAVIOR IS MISSING?", (pr.x + 16, pr.y + 12), INK, FontSel::Head, "topleft");
            text(ops, "Choose the circuit you believe was turned off:", (pr.x + 16, pr.y + 40), LABEL, FontSel::Small, "topleft");
            let mut by = pr.y + 68;
            for ci in &CIRCUITS {
                button(ops, rect(pr.x + 20, by, pr.w - 40, 38), ci.play_label, &format!("guess:{}", ci.id), 0, true);
                by += 44;
            }
            button(ops, rect(pr.centerx() - 60, pr.bottom() - 46, 120, 36), "Cancel", "testing", 0, true);
        }
        _ => {
            let (w, h) = (520, 340);
            let pr = rect(cx - w / 2, 60, w, h);
            let ok = r.guess.is_some_and(|g| g.0);
            let border = if ok { GOOD } else { BAD };
            card(ops, pr, [16, 20, 30], 245, Some((border, 2)));
            text(ops, if ok { "CORRECT DIAGNOSIS!" } else { "NOT QUITE!" }, (pr.x + 20, pr.y + 14), border, FontSel::Head, "topleft");
            if ok {
                let score = 1.max(3 - r.hints as i32) as usize;
                ops.push(Op::Stars { c: ((pr.right() - 80) as f32, (pr.y + 26) as f32), n: score, size: 11 });
                text(ops, &format!("Diagnosed with {} hints!", r.hints), (pr.x + 20, pr.y + 46), TEXT, FontSel::Small, "topleft");
            } else {
                let lbl = r.guess.and_then(|g| CIRCUITS.iter().find(|c| c.id == g.1)).map_or("", |c| c.play_label);
                text(ops, &format!("You guessed: '{lbl}'"), (pr.x + 20, pr.y + 46), [220, 160, 160], FontSel::Small, "topleft");
            }
            let rb = rect(pr.x + 16, pr.y + 74, pr.w - 32, 190);
            ops.push(Op::Card { rect: rb, fill: [24, 30, 44], alpha: 255, radius: 8 });
            ops.push(Op::Stroke { rect: rb, color: [50, 62, 88], width: 1, radius: 8 });
            text(ops, &format!("Missing Behavior: {}", c.play_label), (rb.x + 12, rb.y + 10), AMBER, FontSel::Bold, "topleft");
            text(ops, &format!("Biological Circuit: {}", c.bio_name), (rb.x + 12, rb.y + 36), [140, 200, 255], FontSel::Small, "topleft");
            text(ops, &format!("What happens: {}", c.missing), (rb.x + 12, rb.y + 64), TEXT, FontSel::Small, "topleft");
            text(ops, &format!("Citation: {}", c.citation), (rb.x + 12, rb.y + 140), LABEL, FontSel::Small, "topleft");
            button(ops, rect(pr.x + 30, pr.bottom() - 56, 200, 42), "Play Another Mystery", "again:reverse_surgery", 1, true);
            button(ops, rect(pr.right() - 170, pr.bottom() - 56, 140, 42), "Close", "end", 0, true);
        }
    }
}

const ROUNDS: usize = 10;
const CUE_TIME: f32 = 2.2;

fn predict_update(
    p: &mut PredictUi,
    now: f32,
    link: &SimLink,
    sfx: &mut MessageWriter<Sfx>,
    scores: &mut Scores,
    w: &mut World3,
    arena: &Arena,
) {
    let mut start_round = |p: &mut PredictUi| {
        if p.round >= ROUNDS {
            p.state = 3;
            scores.record("predict", f64::from(p.score), true);
            return;
        }
        p.readout = p.rng.index(READOUTS.len());
        p.state = 1;
        p.timer = now + CUE_TIME;
        let target = READOUTS[p.readout].2;
        link.0.send(Command::DriveSpec { spec: motor_spec(target).into(), amp: 0.6 });
        p.driven = Some(target);
    };
    match p.state {
        0 => start_round(p),
        1 if now >= p.timer => predict_answer(p, "timeout", now, link, sfx, w, arena),
        2 if now >= p.timer => {
            p.round += 1;
            if p.round >= ROUNDS {
                p.state = 3;
                scores.record("predict", f64::from(p.score), true);
            } else {
                start_round(p);
            }
        }
        _ => {}
    }
}

fn predict_answer(p: &mut PredictUi, choice: &str, now: f32, link: &SimLink, sfx: &mut MessageWriter<Sfx>, w: &mut World3, arena: &Arena) {
    if p.state != 1 {
        return;
    }
    let ok = choice == READOUTS[p.readout].0;
    if ok {
        p.score += 1;
    }
    sfx.write(Sfx::new(if ok { "yum" } else { "click" }));
    p.history.push(ok);
    // the readout's own motor action
    let target = READOUTS[p.readout].2;
    let outdoor = crate::world::is_outdoor(arena.name());
    for (e, mut f) in &mut w.flies {
        let Ok(thx) = w.tf.get(f.parts[Part::Thorax.index()]).map(|t| t.translation) else { continue };
        let launch = match target {
            "jump" => Some(f.escape(now, thx, 1.0, false, outdoor)),
            "fly" => Some(f.escape(now, thx, 2.5, true, outdoor)),
            "run" => {
                let away = f.away_from(thx, f.threat_x);
                f.yaw_target = away.z.atan2(away.x);
                f.react.walk_until = now + 1.2;
                f.react.back_until = 0.0;
                f.run = true;
                None
            }
            "kick" => {
                f.react.flail_until = now + 0.8;
                None
            }
            _ => {
                f.react.back_until = now + 1.0;
                None
            }
        };
        if let Some(l) = launch {
            for (part, mut v) in &mut w.vel {
                if part.fly == e {
                    v.0 = l;
                }
            }
        }
    }
    if let Some(t) = p.driven.take() {
        link.0.send(Command::DriveSpec { spec: motor_spec(t).into(), amp: 0.0 });
    }
    p.state = 2;
    p.timer = now + 1.2;
}

fn predict_ops(ops: &mut Vec<Op>, p: &PredictUi, now: f32) {
    let cx = (PLAY_W - 260).min(260.max(PLAY_W / 2));
    let r = &READOUTS[p.readout];
    match p.state {
        1 => {
            let (w, h) = (500, 230);
            let pr = rect(cx - w / 2, 70, w, h);
            card(ops, pr, [12, 16, 24], 235, Some(([70, 110, 170], 1)));
            text(
                ops,
                &format!("PREDICT THE MOVE  ·  Round {}/{ROUNDS}", p.round + 1),
                (pr.x + 16, pr.y + 12),
                AMBER,
                FontSel::Bold,
                "topleft",
            );
            text(ops, &format!("Score: {}", p.score), (pr.right() - 16, pr.y + 12), INK, FontSel::Bold, "topright");
            let pulse_r = (14.0 + 4.0 * ((now * 6.0) % (2.0 * std::f32::consts::PI)).sin()) as i32;
            let c = (pr.x + 36, pr.y + 54);
            ops.push(Op::Circle { c, r: pulse_r, color: [255, 180, 50] });
            ops.push(Op::Circle { c, r: 4.max(pulse_r - 6), color: [255, 235, 150] });
            text(ops, "DESCENDING MOTOR SURGE DETECTED!", (pr.x + 64, pr.y + 42), [255, 220, 120], FontSel::Bold, "topleft");
            text(ops, "What will the fly do before it moves?", (pr.x + 64, pr.y + 60), TEXT, FontSel::Small, "topleft");
            let frac = (p.timer - now).max(0.0) / CUE_TIME;
            let bar = rect(pr.x + 20, pr.y + 86, pr.w - 40, 8);
            ops.push(Op::Card { rect: bar, fill: [30, 36, 48], alpha: 255, radius: 4 });
            ops.push(Op::Card { rect: rect(bar.x, bar.y, (bar.w as f32 * frac) as i32, 8), fill: [80, 180, 240], alpha: 255, radius: 4 });
            let bw = (pr.w - 48) / 5;
            for (i, ro) in READOUTS.iter().enumerate() {
                button(ops, rect(pr.x + 18 + i as i32 * (bw + 3), pr.y + 110, bw, 52), ro.1, &format!("pick:{}", ro.0), 1, true);
            }
            button(ops, rect(pr.right() - 100, pr.bottom() - 46, 84, 32), "Quit", "end", 0, true);
        }
        2 => {
            let (w, h) = (480, 180);
            let pr = rect(cx - w / 2, 80, w, h);
            let ok = p.history.last().copied().unwrap_or(false);
            let border = if ok { GOOD } else { BAD };
            card(ops, pr, [14, 18, 28], 235, Some((border, 2)));
            text(ops, if ok { "CORRECT PREDICTION!" } else { "MISSED!" }, (pr.centerx(), pr.y + 16), border, FontSel::Head, "midtop");
            text(ops, &format!("The fly {}", r.4), (pr.centerx(), pr.y + 54), INK, FontSel::Bold, "midtop");
            text(ops, &format!("Circuit: {}", r.3), (pr.centerx(), pr.y + 82), LABEL, FontSel::Small, "midtop");
            text(ops, &format!("Score: {}/{}", p.score, p.round + 1), (pr.centerx(), pr.y + 110), AMBER, FontSel::Bold, "midtop");
        }
        3 => {
            let (w, h) = (480, 240);
            let pr = rect(cx - w / 2, 80, w, h);
            card(ops, pr, [16, 20, 32], 245, Some(([80, 120, 180], 2)));
            text(ops, "CHALLENGE COMPLETE!", (pr.centerx(), pr.y + 16), INK, FontSel::Head, "midtop");
            text(
                ops,
                &format!("Final Score: {} / {ROUNDS} correct predictions", p.score),
                (pr.centerx(), pr.y + 54),
                AMBER,
                FontSel::Bold,
                "midtop",
            );
            ops.push(Op::Stars { c: (pr.centerx() as f32, (pr.y + 96) as f32), n: stars("predict", f64::from(p.score)), size: 14 });
            button(ops, rect(pr.centerx() - 170, pr.bottom() - 56, 160, 42), "Play Again", "again:predict", 1, true);
            button(ops, rect(pr.centerx() + 20, pr.bottom() - 56, 140, 42), "Close", "end", 0, true);
        }
        _ => {}
    }
}

// --- clicks ----------------------------------------------------------------------------------------------------------------------
/// A click at HUD position `pos`: the panel's buttons, the sweet-tooth bar. True when the challenge took it (overlays take every click).
pub fn click(ch: &mut Challenges, pos: (i32, i32), link: &SimLink, now: f32) -> bool {
    if ch.active.is_none() {
        return false;
    }
    let hit = ch.hits.iter().find(|(r, _)| r.contains(pos)).map(|(_, id)| id.clone());
    if let Some(id) = hit {
        let mut again: Option<&str> = None;
        match (id.as_str(), ch.active.as_mut()) {
            ("end", _) => ch.request = Some("end".into()),
            (i, Some(Active::TMaze(t))) if i.starts_with("tmaze:") => {
                let zap = &i["tmaze:".len()..];
                link.0.send(Command::StartTMaze { zap: zap.into(), seed: (unix_now() * 1000.0) as u64 });
                t.started = true;
            }
            (i, _) if i.starts_with("again:") => {
                again = Some(if i.ends_with("tmaze") {
                    "tmaze"
                } else if i.ends_with("sweet") {
                    "sweet"
                } else if i.ends_with("predict") {
                    "predict"
                } else {
                    "reverse_surgery"
                })
            }
            (i, Some(Active::Sweet(_))) if i.starts_with("offer:") => {
                link.0.send(Command::SweetOffer(i["offer:".len()..].parse().unwrap_or(50)));
            }
            ("hint", Some(Active::Reverse(r))) => r.hints = (r.hints + 1).min(2),
            ("guessing", Some(Active::Reverse(r))) => r.state = 1,
            ("testing", Some(Active::Reverse(r))) => r.state = 0,
            (i, Some(Active::Reverse(r))) if i.starts_with("guess:") => {
                let id = &i["guess:".len()..];
                let ok = CIRCUITS[r.circuit].id == id;
                let cid = CIRCUITS.iter().find(|c| c.id == id).map_or("", |c| c.id);
                r.guess = Some((ok, cid));
                r.state = 2;
                if ok {
                    Scores::default_file().record("reverse_surgery", f64::from(1.max(3 - r.hints as i32)), true);
                }
            }
            (i, Some(Active::Predict(p))) if i.starts_with("pick:") => {
                let _ = p; // answered by `answer_pick`, where the fly's body is available
                ch.note = Some(i["pick:".len()..].to_string());
            }
            _ => {}
        }
        if let Some(k) = again {
            ch.request = Some(k.into());
        }
        let _ = now;
        return true;
    }
    if let Some(Active::Sweet(sw)) = ch.active.as_ref()
        && sw.track.h > 0
        && Rect::new(sw.track.x, sw.track.y - 12, sw.track.w, sw.track.h + 24).contains(pos)
    {
        let frac = ((pos.0 - sw.track.x) as f32 / sw.track.w as f32).clamp(0.0, 1.0);
        // (set below: the borrow above is shared)
        let dose = (frac * 20.0).round() as u32 * 5;
        if let Some(Active::Sweet(sw)) = ch.active.as_mut() {
            sw.dose = dose;
        }
        return true;
    }
    ch.overlay
}

pub struct ChallengesPlugin;

impl Plugin for ChallengesPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Challenges>().add_systems(Update, (update, answer_pick.after(update)));
    }
}

/// A prediction button was clicked: hand the choice to the round (needs the fly's body for the motor action).
fn answer_pick(
    mut ch: ResMut<Challenges>,
    virt: Res<Time<Virtual>>,
    link: Res<SimLink>,
    mut sfx: MessageWriter<Sfx>,
    arena: Res<Arena>,
    mut w: World3,
) {
    let Some(choice) = ch.note.take() else { return };
    if let Some(Active::Predict(p)) = ch.active.as_mut() {
        p.state = 1;
        predict_answer(p, &choice, virt.elapsed_secs(), &link, &mut sfx, &mut w, &arena);
    }
}

#[cfg(test)]
mod tests {
    //! Every panel state against `legacy-python/make_challenge_reference.py`'s drawing of the same state (`tests/ui_reference/challenge_*.png`):
    //! the same rectangles and text positions; only glyph edges may differ.
    use std::path::PathBuf;

    use ktf_ui::Canvas;
    use ktf_ui::hud::{HudFonts, draw_ops};

    use super::*;
    use crate::snapshot::ChallengeView;

    fn reference(name: &str) -> Vec<u8> {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/ui_reference").join(format!("challenge_{name}.png"));
        let dec = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(&p).unwrap_or_else(|e| panic!("{p:?}: {e}"))));
        let mut r = dec.read_info().unwrap();
        let mut buf = vec![0; r.output_buffer_size().unwrap()];
        let info = r.next_frame(&mut buf).unwrap();
        assert_eq!((info.width, info.height), (1280, 760));
        buf[..info.buffer_size()].to_vec()
    }

    fn check(name: &str, ops: &[Op]) {
        let mut cv = Canvas { linear_light: false, ..Canvas::new(1280, 760) };
        draw_ops(&mut cv, &HudFonts::new(false), ops, (-1, -1));
        let ours = cv.to_rgb_over([60, 70, 80]);
        let theirs = reference(name);
        let bad = ours
            .as_chunks::<3>()
            .0
            .iter()
            .zip(theirs.as_chunks::<3>().0.iter())
            .filter(|(a, b)| a.iter().zip(b.iter()).any(|(x, y)| (i32::from(*x) - i32::from(*y)).abs() > 60))
            .count();
        let share = bad as f64 / (1280.0 * 760.0);
        eprintln!("challenge_{name}: {:.3}% differ", share * 100.0);
        assert!(share < 0.012, "challenge_{name}: {:.3}% of pixels differ clearly", share * 100.0);
    }

    fn scores() -> Scores {
        let mut s = Scores::open(std::env::temp_dir().join("ktf-no-such-scores.json"));
        for (k, v) in [("tmaze", 8.0), ("sneak", 1.4), ("sweet", 35.0), ("reverse_surgery", 2.0)] {
            s.map.insert(k.into(), v);
        }
        s
    }

    fn view(phase: &str, stage: u32, cycle: u32, choices: Vec<(&str, bool)>) -> ChallengeView {
        ChallengeView {
            kind: "tmaze".into(),
            phase: phase.into(),
            stage,
            cycle,
            zap: "odor_a".into(),
            choices: choices.into_iter().map(|(d, ok)| (d.to_string(), ok)).collect(),
            ..Default::default()
        }
    }

    fn tmaze(name: &str, t: &TMazeUi, v: Option<&ChallengeView>) {
        let mut ops = vec![];
        tmaze_ops(&mut ops, t, v, true, &scores());
        check(name, &ops);
    }

    #[test]
    fn tmaze_panels() {
        let started = TMazeUi { started: true, ..Default::default() };
        tmaze("tmaze_choose", &TMazeUi::default(), None);
        tmaze("tmaze_train", &started, Some(&view("train", 1, 2, vec![])));
        tmaze("tmaze_naive", &started, Some(&view("naive", 1, 0, vec![])));
        let six = vec![("odor_a", false), ("odor_b", true), ("odor_b", true), ("odor_a", false), ("odor_b", true), ("odor_b", true)];
        tmaze("tmaze_test", &started, Some(&view("test", 0, 0, six)));
        let ten: Vec<(&str, bool)> = std::iter::repeat_n(("odor_b", true), 7).chain(std::iter::repeat_n(("odor_a", false), 3)).collect();
        let done = TMazeUi { started: true, best: false, right_total: Some(7) };
        tmaze("tmaze_done", &done, Some(&view("done", 3, 0, ten.clone())));
        let best = TMazeUi { started: true, best: true, right_total: Some(7) };
        tmaze("tmaze_done_best", &best, Some(&view("done", 3, 0, ten)));
    }

    #[test]
    fn sneak_panels() {
        let mk = |state: u8, result: Option<(bool, f32)>, best: bool| SneakUi { state, closest: 1.4, result, best, ..Default::default() };
        let mut ops = vec![];
        sneak_ops(&mut ops, &mk(0, None, false), 1.7, &scores());
        check("sneak_armed", &ops);
        let mut ops = vec![];
        sneak_ops(&mut ops, &mk(1, None, false), 1.7, &scores());
        check("sneak_sneaking", &ops);
        let mut ops = vec![];
        sneak_ops(&mut ops, &mk(2, Some((false, 1.3)), true), 1.7, &scores());
        check("sneak_seen", &ops);
        let mut ops = vec![];
        sneak_ops(&mut ops, &mk(2, Some((true, 0.0)), false), 1.7, &scores());
        check("sneak_caught", &ops);
    }

    #[test]
    fn sweet_panels() {
        let mut sw = SweetUi { best: false, track: rect(0, 0, 0, 0), dose: 35 };
        let sv = |phase: &str, offers: Vec<(u32, bool, f32)>, score: Option<u32>| ChallengeView {
            kind: "sweet".into(),
            phase: phase.into(),
            dose: 35,
            offers,
            score,
            ..Default::default()
        };
        let mut ops = vec![];
        sweet_ops(&mut ops, &mut sw, None, &scores());
        check("sweet_ready", &ops);
        let mut ops = vec![];
        sweet_ops(&mut ops, &mut sw, Some(&sv("shown", vec![(50, true, 2.1), (30, false, 1.1), (40, true, 1.8)], None)), &scores());
        check("sweet_shown", &ops);
        let offers: Vec<(u32, bool, f32)> =
            std::iter::repeat_n((50, true, 2.0), 3).chain(std::iter::repeat_n((30, false, 1.0), 5)).collect();
        let mut ops = vec![];
        sweet_ops(&mut ops, &mut sw, Some(&sv("done", offers, Some(25))), &scores());
        check("sweet_done", &ops);
    }

    #[test]
    fn reverse_panels() {
        let mk = |state: u8, guess: Option<(bool, &'static str)>| ReverseUi { state, hints: 1, circuit: 0, guess };
        for (name, r) in [
            ("reverse_testing", mk(0, None)),
            ("reverse_guessing", mk(1, None)),
            ("reverse_correct", mk(2, Some((true, "jump")))),
            ("reverse_wrong", mk(2, Some((false, "wings")))),
        ] {
            let mut ops = vec![];
            reverse_ops(&mut ops, &r, &scores());
            check(name, &ops);
        }
    }

    #[test]
    fn predict_panels() {
        let mk = |state: u8, score: u32, history: Vec<bool>| PredictUi {
            state,
            round: 2,
            score,
            readout: 1,
            timer: 101.1,
            history,
            rng: crate::rng::Rng::new(1),
            driven: None,
        };
        // the Python reference pulses at phase 1.0 (= now * 6 mod 2pi) at now 100.0; 100 * 6 = 600 = 95 * 2pi + 3.4, so give the time that makes 1.0
        let now = 1.0 / 6.0;
        let mut p = mk(1, 2, vec![]);
        p.timer = now + 1.1;
        let mut ops = vec![];
        predict_ops(&mut ops, &p, now);
        check("predict_cue", &ops);
        let mut ops = vec![];
        predict_ops(&mut ops, &mk(2, 2, vec![true, false, true]), now);
        check("predict_result", &ops);
        let mut ops = vec![];
        predict_ops(&mut ops, &mk(3, 7, vec![]), now);
        check("predict_done", &ops);
    }
}
