//! The first-launch tutorial (`ui/tutorial.py`, 2.13): about a minute, skippable at any step, replayable from Settings > Help. Five short
//! steps on top of the running game, none of which changes the simulation: move, use a tool, watch the brain, sugar, the loadout editor.
//! Enter or the Next button skips a step, Backspace or the Skip button ends it. "Shown once per install": `first_run.tutorial_done`.

use bevy::prelude::*;

use crate::first_run::{FirstRun, Launch};
use crate::menu_ui::{MenuOpen, UserConfig};
use crate::player::{MouseLook, PlayerCamera};
use crate::tools::{ActiveLoadout, TOOLS, ToolState};

pub const STEPS: [&str; 5] = ["move", "tool", "brain", "sugar", "loadout"];
const BRAIN_SECONDS: f32 = 7.0;
const MOVE_METERS: f32 = 1.0;

#[derive(Resource, Default)]
pub struct Tutorial {
    pub active: bool,
    pub replay: bool,
    pub i: usize,
    t_step: f32,
    uses0: u32,
    player0: Vec2,
    saw_editor: bool,
    started: bool,
    /// Requests from the HUD buttons and the menu: "next", "skip", "replay".
    pub request: Option<&'static str>,
}

impl Tutorial {
    pub fn step(&self) -> &'static str {
        STEPS[self.i.min(STEPS.len() - 1)]
    }
}

fn select(tools: &mut ToolState, lo: &mut ActiveLoadout, name: &str) {
    if let Some(i) = TOOLS.iter().position(|t| t.0 == name) {
        tools.selected = i;
        lo.0.show(name);
        tools.page = lo.0.page;
    }
}

fn enter_step(t: &mut Tutorial, now: f32, tools: &mut ToolState, lo: &mut ActiveLoadout, pos: Vec2) {
    t.t_step = now;
    t.uses0 = tools.uses;
    t.player0 = pos;
    t.saw_editor = false;
    match t.step() {
        "sugar" => select(tools, lo, "sugar"),
        "tool" if !["hand", "flick", "swatter"].contains(&TOOLS[tools.selected].0) => select(tools, lo, "hand"),
        _ => {}
    }
}

fn finish(t: &mut Tutorial, cfg: &mut UserConfig) {
    t.active = false;
    if !cfg.0.first_run.get("tutorial_done").copied().unwrap_or(false) {
        cfg.0.first_run.insert("tutorial_done".into(), true);
        cfg.0.dirty = true;
        cfg.0.save(env!("CARGO_PKG_VERSION"));
    }
}

#[allow(clippy::too_many_arguments)]
fn run(
    mut t: ResMut<Tutorial>,
    mut cfg: ResMut<UserConfig>,
    real: Res<Time<Real>>,
    keys: Res<ButtonInput<KeyCode>>,
    mut tools: ResMut<ToolState>,
    mut lo: ResMut<ActiveLoadout>,
    cam: Single<&Transform, With<PlayerCamera>>,
    menu: Res<MenuOpen>,
    screen: Res<crate::menu_ui::MenuScreen>,
    look: Res<MouseLook>,
    first: Res<FirstRun>,
) {
    let now = real.elapsed_secs();
    let pos = Vec2::new(cam.translation.x, cam.translation.z);
    // start: when the first-run decision for this launch is the tutorial, or on request (Settings > Help)
    if !t.started && first.0.is_some() {
        t.started = true;
        if first.0 == Some(Launch::Tutorial) {
            t.active = true;
            t.i = 0;
            t.replay = false;
            enter_step(&mut t, now, &mut tools, &mut lo, pos);
        }
    }
    if t.request == Some("replay") {
        t.request = None;
        t.active = true;
        t.replay = true;
        t.i = 0;
        enter_step(&mut t, now, &mut tools, &mut lo, pos);
    }
    if !t.active {
        t.request = None;
        return;
    }
    let mut next = t.request == Some("next") || (!menu.0 && keys.just_pressed(KeyCode::Enter));
    let skip = t.request == Some("skip") || (!menu.0 && keys.just_pressed(KeyCode::Backspace));
    t.request = None;
    if skip {
        finish(&mut t, &mut cfg);
        return;
    }
    if !next {
        next = match t.step() {
            "move" => pos.distance(t.player0) > MOVE_METERS,
            "tool" => tools.uses > t.uses0,
            "brain" => now - t.t_step > BRAIN_SECONDS,
            "sugar" => tools.uses > t.uses0 && ["sugar", "fruit"].contains(&tools.last_used),
            _ => {
                // the loadout step is done when the editor was opened and then closed
                if menu.0 && screen.0 == "loadout" {
                    t.saw_editor = true;
                }
                t.saw_editor && !menu.0
            }
        };
    }
    let _ = &look;
    if next {
        t.i += 1;
        if t.i >= STEPS.len() {
            finish(&mut t, &mut cfg);
        } else {
            enter_step(&mut t, now, &mut tools, &mut lo, pos);
        }
    }
}

pub struct TutorialPlugin;

impl Plugin for TutorialPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Tutorial>().add_systems(Update, run);
    }
}
