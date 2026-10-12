//! Learning in play (`Game._scents`, `_learn`, `_memory_behavior`, the T training screen): the smell of the tool in your hand near the fly's
//! head pokes its olfactory neurons and is observed (its Kenyon-cell pattern is stored); hurting the fly then teaches that pattern through the
//! PPL1 dopamine neurons (the sim thread's plasticity); a smell it fears makes it AVOID the tool, a sugar smell it likes makes it APPROACH.

use bevy::prelude::*;

use crate::actions::Actions;
use crate::app::{LatestSnapshot, SimLink};
use crate::arcade::Arcade;
use crate::brainview::BrainView;
use crate::controls::TimeControl;
use crate::feedback::{Notes, Popup, Tag};
use crate::fly::{Fly, FlyPart, Part, ReactionFired};
use crate::hud_canvas::HudStats;
use crate::menu_ui::{MenuInputSet, MenuOpen};
use crate::player::{Aim, MouseLook};
use crate::senses::Scents;
use crate::sim_thread::Command;
use crate::tools::{ToolState, tool_tip};

pub const FEAR_ACT: f32 = 0.35;
pub const LIKE_ACT: f32 = 0.35;

#[derive(Resource)]
pub struct Learned {
    /// The smell of the tool in hand while it is near the fly's head (fruit counts as sugar).
    pub scent_now: Option<&'static str>,
    pub sugar_scent: bool,
    pub fear_now: f32,
    pub like_now: f32,
    pub avoid_ready: f32,
    tick: u64,
    pub train_open: bool,
    pub train_scent: &'static str,
    pub train_speed: f32,
    /// Real time of the first click on wipe (a second click within 3 s wipes).
    pub wipe_armed_at: Option<f32>,
    speed_sent: f32,
}

impl Default for Learned {
    fn default() -> Self {
        Learned {
            scent_now: None,
            sugar_scent: false,
            fear_now: 0.0,
            like_now: 0.0,
            avoid_ready: 0.0,
            tick: 0,
            train_open: false,
            train_scent: "swatter",
            train_speed: 1.0,
            wipe_armed_at: None,
            speed_sent: 1.0,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn scents_and_learn(
    mut l: ResMut<Learned>,
    tool: Res<ToolState>,
    aim: Res<Aim>,
    sources: Res<Scents>,
    snap: Res<LatestSnapshot>,
    menu: Res<MenuOpen>,
    arcade: Res<Arcade>,
    hud: Res<HudStats>,
    view: Res<BrainView>,
    link: Res<SimLink>,
    flies: Query<(&Fly, &crate::fly::Slot)>,
    which: Res<crate::flies_play::Flies>,
    parts: Query<&Transform, With<FlyPart>>,
) {
    l.scent_now = None;
    l.sugar_scent = false;
    let overlay = menu.0 || arcade.open || hud.help || view.big || l.train_open;
    let name = if tool.name() == "fruit" { "sugar" } else { tool.name() };
    // `_scents` runs for every fly: the tool in your hand smells to each fly whose head it is near, and that fly's own brain feels it
    if !overlay && !tool.puppet {
        for (f, slot) in &flies {
            if f.dead() {
                continue;
            }
            let Ok(head) = parts.get(f.parts[Part::Head.index()]).map(|t| t.translation) else { continue };
            if (tool_tip(&aim) - head).length() < 2.5 {
                link.0.send(Command::PokeFly { slot: slot.0, region: format!("scent:{name}"), side: None, strength: 0.3 });
            }
        }
    }
    // what it has learned is read for the fly in front (the memory follows the focus)
    let Some((f, _)) = flies.iter().find(|(_, s)| s.0 == which.focus) else { return };
    if f.dead() || tool.puppet {
        l.fear_now = 0.0;
        l.like_now = 0.0;
        return;
    }
    let Ok(head) = parts.get(f.parts[Part::Head.index()]).map(|t| t.translation) else { return };
    if !overlay && (tool_tip(&aim) - head).length() < 2.5 {
        l.scent_now = Some(name);
    }
    l.sugar_scent = sources.0.iter().any(|s| s.scent == "sugar" && Vec2::new(s.pos.x - head.x, s.pos.z - head.z).length() < 2.4);
    l.tick += 1;
    if l.tick.is_multiple_of(3) {
        for key in l.scent_now.into_iter().chain(l.sugar_scent.then_some("sugar")) {
            link.0.send(Command::Observe(key.to_string()));
        }
        let key = l.scent_now.or(l.sugar_scent.then_some("sugar"));
        let (fear, like) = key
            .and_then(|k| snap.0.as_deref()?.memory.as_deref()?.scores.iter().find(|s| s.0 == k).map(|s| (s.1, s.2)))
            .unwrap_or((0.0, 0.0));
        l.fear_now = fear;
        l.like_now = like;
    }
}

/// `_memory_behavior`: the fly acts on what it learned about the smell of your tool.
#[allow(clippy::too_many_arguments)]
fn behavior(
    mut l: ResMut<Learned>,
    virt: Res<Time<Virtual>>,
    aim: Res<Aim>,
    arena: Res<crate::config::Arena>,
    which: Res<crate::flies_play::Flies>,
    mut flies: Query<(Entity, &mut Fly, &crate::fly::Slot)>,
    mut vel: Query<(&FlyPart, &mut avian3d::prelude::LinearVelocity)>,
    tf: Query<&Transform, With<FlyPart>>,
    mut notes: ResMut<Notes>,
    mut popups: MessageWriter<Popup>,
    mut fired: MessageWriter<ReactionFired>,
) {
    let now = virt.elapsed_secs();
    let Some(scent) = l.scent_now else { return };
    for (e, mut f, slot) in &mut flies {
        if slot.0 != which.focus {
            continue; // only the focused fly's brain carries the learned memory here
        }
        let free = f.grabbed.is_none() && now >= f.react.escape_until;
        if f.dead() || !free || now < l.avoid_ready || f.frozen_at.is_some() || now < f.react.stun_until {
            continue;
        }
        let Ok(thx) = tf.get(f.parts[Part::Thorax.index()]).map(|t| t.translation) else { continue };
        let eye = aim.eye;
        let above = thx + Vec3::Y * 0.4;
        if scent != "sugar" && l.fear_now > FEAR_ACT {
            l.avoid_ready = now + 2.5;
            f.last_hit = eye;
            if f.can_fly() && l.fear_now > 0.6 {
                let outdoor = crate::world::is_outdoor(arena.name());
                let launch = f.escape(now, thx, 2.4, false, outdoor);
                for (p, mut v) in &mut vel {
                    if p.fly == e {
                        v.0 = launch;
                    }
                }
            } else {
                let away = f.away_from(thx, eye);
                f.yaw_target = away.z.atan2(away.x);
                f.react.walk_until = now + 1.2;
                f.react.back_until = 0.0;
                f.run = true;
            }
            notes.push(now, format!("AVOID    remembers the {scent} ({:.2})", l.fear_now), Tag::of_reaction("AVOID"));
            fired.write(ReactionFired { fly: e, name: "AVOID".into() });
            popups.write(Popup::new(above, "NOPE!", (255, 220, 120)));
        } else if scent == "sugar" && l.like_now > LIKE_ACT && now >= f.react.walk_until {
            l.avoid_ready = now + 1.5;
            let toward = -f.away_from(thx, eye);
            f.yaw_target = toward.z.atan2(toward.x);
            f.react.walk_until = now + 1.0;
            f.run = false;
            notes.push(now, format!("APPROACH remembers sugar ({:.2})", l.like_now), Tag::of_reaction("APPROACH"));
        }
    }
}

/// T opens and closes the training screen; Esc closes it first (before the pause menu would open).
fn keys(
    act: Res<Actions>,
    mut keycodes: ResMut<ButtonInput<KeyCode>>,
    mut l: ResMut<Learned>,
    menu: Res<MenuOpen>,
    mut look: ResMut<MouseLook>,
) {
    if keycodes.just_pressed(KeyCode::Escape) && l.train_open {
        l.train_open = false;
        keycodes.clear_just_pressed(KeyCode::Escape);
    } else if act.just("training") && !menu.0 {
        l.train_open = !l.train_open;
        if l.train_open {
            look.captured = false;
        }
    }
}

/// Training runs the brain at the screen's SPEED (x1, x2, x3) on top of the slow-motion setting, and restores it afterwards.
fn train_speed(mut l: ResMut<Learned>, snap: Res<LatestSnapshot>, tc: Res<TimeControl>, link: Res<SimLink>) {
    let active = snap.0.as_deref().and_then(|s| s.memory.as_deref()).is_some_and(|m| m.train.is_some());
    let challenge = snap.0.as_deref().map_or(1.0, |s| if s.challenge.is_some() { s.challenge_speed.max(1.0) } else { 1.0 });
    let want = tc.speed * if active { l.train_speed } else { challenge };
    if (want - l.speed_sent).abs() > 1e-4 {
        l.speed_sent = want;
        link.0.send(Command::SetSpeed(want));
    }
}

pub struct MemoryPlayPlugin;

impl Plugin for MemoryPlayPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Learned>()
            .add_systems(PreUpdate, keys.before(MenuInputSet).after(bevy::input::InputSystems))
            .add_systems(Update, train_speed)
            .add_systems(FixedLast, (scents_and_learn, behavior).chain());
    }
}
