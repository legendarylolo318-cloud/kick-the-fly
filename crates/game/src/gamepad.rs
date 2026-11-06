//! Gamepad support (Python: `game/gamepad.py`, `Game3D._pad_input`): on top of keyboard and mouse, never instead of them.
//!
//! Left stick walks, right stick looks, the right trigger uses the tool in your hand, the bumpers cycle tools, holding the tool-wheel
//! button opens the wheel (point with the right stick), Start is the menu. The bindings are the config's `[gamepad]` table in SDL's names
//! (`leftx`, `righttrigger`, `rightshoulder`, `start`, ... or a raw `axisN` / `buttonN`), as the Python game wrote them; Bevy's gamepad
//! backend (gilrs) reads the controller and uses the same SDL controller database for the layout.
//!
//! Nothing new is taught to the rest of the game: the pad's input is turned into what the keyboard and mouse already produce (held
//! movement actions, mouse motion for the look, a left click for the trigger, a wheel step for the bumpers, Esc for Start).
//!
//! Not ported: rebinding by pressing a button in Settings > Controls (the table shows and keeps the bindings; edit `config.toml` to
//! change one), and moving through menu pages with the pad.

use std::collections::{HashMap, HashSet};

use bevy::input::InputSystems;
use bevy::input::gamepad::{Gamepad, GamepadAxis, GamepadButton};
use bevy::input::mouse::{AccumulatedMouseMotion, MouseScrollUnit, MouseWheel};
use bevy::input::touch::TouchPhase;
use bevy::prelude::*;
use bevy::window::PrimaryWindow;

use crate::actions::Actions;
use crate::menu_ui::{MenuInputSet, UserConfig};

/// `STICKS`: the four analog actions; every other pad action is a button.
pub const STICKS: [&str; 4] = ["move_x", "move_y", "look_x", "look_y"];

/// One input of a controller, in SDL's sign convention (sticks: right and down are positive; triggers and buttons 0..1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Input {
    /// A Bevy axis, and whether Bevy's sign is the other way round from SDL's (the Y axes: up is positive in Bevy).
    Axis(GamepadAxis, bool),
    Button(GamepadButton),
}

/// `gamepad.parse`: a binding from the config to an input. `None` for "" and for anything that is not a binding.
pub fn parse(binding: &str) -> Option<Input> {
    use GamepadAxis as A;
    use GamepadButton as B;
    Some(match binding {
        "leftx" => Input::Axis(A::LeftStickX, false),
        "lefty" => Input::Axis(A::LeftStickY, true),
        "rightx" => Input::Axis(A::RightStickX, false),
        "righty" => Input::Axis(A::RightStickY, true),
        "lefttrigger" => Input::Button(B::LeftTrigger2),
        "righttrigger" => Input::Button(B::RightTrigger2),
        "a" => Input::Button(B::South),
        "b" => Input::Button(B::East),
        "x" => Input::Button(B::West),
        "y" => Input::Button(B::North),
        "back" => Input::Button(B::Select),
        "guide" => Input::Button(B::Mode),
        "start" => Input::Button(B::Start),
        "leftstick" => Input::Button(B::LeftThumb),
        "rightstick" => Input::Button(B::RightThumb),
        "leftshoulder" => Input::Button(B::LeftTrigger),
        "rightshoulder" => Input::Button(B::RightTrigger),
        "dpup" => Input::Button(B::DPadUp),
        "dpdown" => Input::Button(B::DPadDown),
        "dpleft" => Input::Button(B::DPadLeft),
        "dpright" => Input::Button(B::DPadRight),
        raw => {
            if let Some(n) = raw.strip_prefix("axis").and_then(|n| n.parse::<u8>().ok()) {
                Input::Axis(A::Other(n), false)
            } else {
                Input::Button(B::Other(raw.strip_prefix("button").and_then(|n| n.parse::<u8>().ok())?))
            }
        }
    })
}

/// `Gamepad.stick`: a stick with the dead zone taken out, each axis -1..1.
pub fn stick(x: f32, y: f32, deadzone: f32) -> (f32, f32) {
    let m = x.hypot(y);
    if m < deadzone || m <= 0.0 {
        return (0.0, 0.0);
    }
    let k = ((m - deadzone) / (1.0 - deadzone)).min(1.0) / m;
    (x * k, y * k)
}

/// `Game3D._pad_input`'s look: stick deflection to mouse pixels this frame.
pub fn look_pixels(lx: f32, ly: f32, look_speed: f32, dt: f32, mouse_sensitivity: f32, pad_invert_y: bool, invert_y: bool) -> Vec2 {
    let speed = look_speed * dt / 0.0024 / mouse_sensitivity.max(0.1);
    let flip = if pad_invert_y { -1.0 } else { 1.0 } * if invert_y { -1.0 } else { 1.0 };
    Vec2::new(lx * speed, ly * speed * flip)
}

/// What the pad did this frame.
#[derive(Resource, Default)]
pub struct Pad {
    pub connected: bool,
    /// Movement and modifier actions held (the keyboard's names: forward, back, left, right, sprint, crouch, tool_wheel).
    held: HashSet<&'static str>,
    /// Button actions that went down this frame (pad action names).
    down: HashSet<String>,
    was: HashMap<String, bool>,
    /// The trigger is acting as the left mouse button, Start as Esc.
    using: bool,
    esc: bool,
    /// The right stick is pushed.
    looking: bool,
}

impl Pad {
    /// Someone is playing on the pad right now: a stick is pushed or the trigger is down.
    pub fn in_use(&self) -> bool {
        self.connected && (self.looking || self.using || ["forward", "back", "left", "right"].iter().any(|a| self.held.contains(a)))
    }
}

fn value(pads: &Query<&Gamepad>, binding: &str) -> f32 {
    let Some(input) = parse(binding) else { return 0.0 };
    let mut best = 0.0f32;
    for p in pads {
        let v = match input {
            Input::Axis(a, flip) => p.get(a).unwrap_or(0.0) * if flip { -1.0 } else { 1.0 },
            Input::Button(b) => p.get(b).unwrap_or(0.0).max(f32::from(u8::from(p.pressed(b)))),
        };
        if v.abs() > best.abs() {
            best = v;
        }
    }
    best
}

/// Read the pads and turn their input into keyboard and mouse input, before anything reads those.
#[allow(clippy::too_many_arguments)]
fn read(
    pads: Query<&Gamepad>,
    cfg: Res<UserConfig>,
    time: Res<Time<Real>>,
    mut pad: ResMut<Pad>,
    mut keys: ResMut<ButtonInput<KeyCode>>,
    mut mouse: ResMut<ButtonInput<MouseButton>>,
    mut motion: ResMut<AccumulatedMouseMotion>,
    mut wheel: MessageWriter<MouseWheel>,
    window: Single<Entity, With<PrimaryWindow>>,
) {
    let c = &cfg.0;
    let on = c.get_bool("controls.gamepad") && !pads.is_empty();
    pad.connected = on;
    pad.held.clear();
    pad.down.clear();
    let binding = |action: &str| c.pad.iter().find(|(a, _)| a == action).map_or("", |(_, b)| b.as_str());
    let val = |action: &str| if on { value(&pads, binding(action)) } else { 0.0 };
    let dz = c.get_f32("controls.pad_deadzone");
    // buttons: what went down this frame
    let names: Vec<String> = c.pad.iter().map(|(a, _)| a.clone()).filter(|a| !STICKS.contains(&a.as_str())).collect();
    for a in names {
        let now = val(&a) > 0.5;
        if now && !pad.was.get(&a).copied().unwrap_or(false) {
            pad.down.insert(a.clone());
        }
        pad.was.insert(a, now);
    }
    let pressed = |pad: &Pad, a: &str| pad.was.get(a).copied().unwrap_or(false);
    // the left stick walks (past 0.35), the buttons sprint and crouch, the wheel button holds the wheel open
    let (mx, my) = stick(val("move_x"), val("move_y"), dz);
    for (held, name) in [
        (my < -0.35, "forward"),
        (my > 0.35, "back"),
        (mx < -0.35, "left"),
        (mx > 0.35, "right"),
        (pressed(&pad, "sprint"), "sprint"),
        (pressed(&pad, "crouch"), "crouch"),
        (pressed(&pad, "tool_wheel"), "tool_wheel"),
    ] {
        if held {
            pad.held.insert(name);
        }
    }
    // the right stick is the mouse (it also points at the tool wheel while that is open)
    let (lx, ly) = stick(val("look_x"), val("look_y"), dz);
    pad.looking = lx != 0.0 || ly != 0.0;
    if pad.looking {
        motion.delta += look_pixels(
            lx,
            ly,
            c.get_f32("controls.pad_look_speed"),
            time.delta_secs().min(0.1),
            c.get_f32("controls.mouse_sensitivity"),
            c.get_bool("controls.pad_invert_y"),
            c.get_bool("controls.invert_y"),
        );
    }
    // the trigger is the left mouse button (use the tool; with the mouse free, 'click the room'); letting go is the mouse-up
    let using = pressed(&pad, "use");
    if using && !pad.using {
        mouse.press(MouseButton::Left);
    } else if !using && pad.using {
        mouse.release(MouseButton::Left);
    }
    pad.using = using;
    // Start is Esc
    let esc = pressed(&pad, "menu");
    if esc && !pad.esc {
        keys.press(KeyCode::Escape);
    } else if !esc && pad.esc {
        keys.release(KeyCode::Escape);
    }
    pad.esc = esc;
    // the bumpers step through the loadout as the mouse wheel does (`cycle_tool(1)` is a wheel step down)
    for (action, y) in [("tool_next", -1.0), ("tool_prev", 1.0)] {
        if pad.down.contains(action) {
            wheel.write(MouseWheel { unit: MouseScrollUnit::Line, x: 0.0, y, window: *window, phase: TouchPhase::Moved });
        }
    }
}

/// After the keyboard's actions are collected: add the pad's.
fn merge(pad: Res<Pad>, mut act: ResMut<Actions>) {
    if !pad.connected {
        return;
    }
    for h in &pad.held {
        act.hold(h);
    }
    for (pad_action, action) in
        [("big_view", "big_view"), ("neurodex", "neurodex"), ("killcam", "killcam"), ("loadout", "loadout"), ("tool_wheel", "tool_wheel")]
    {
        if pad.down.contains(pad_action) {
            act.fire(action);
        }
    }
}

pub struct GamepadPlugin;

impl Plugin for GamepadPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Pad>()
            .add_systems(PreUpdate, read.after(InputSystems).before(MenuInputSet))
            .add_systems(PreUpdate, merge.after(crate::actions::collect));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_default_binding_of_the_schema_is_an_input() {
        let cfg = ktf_data::config::Config::default();
        assert_eq!(cfg.pad.len(), 16);
        for (action, binding) in &cfg.pad {
            let input = parse(binding).unwrap_or_else(|| panic!("{action}: '{binding}'"));
            assert_eq!(matches!(input, Input::Axis(..)), STICKS.contains(&action.as_str()), "{action}");
        }
        // SDL's Y axes point down, Bevy's up
        assert_eq!(parse("lefty"), Some(Input::Axis(GamepadAxis::LeftStickY, true)));
        assert_eq!(parse("rightx"), Some(Input::Axis(GamepadAxis::RightStickX, false)));
        assert_eq!(parse("button12"), Some(Input::Button(GamepadButton::Other(12))));
        assert_eq!(parse("axis3"), Some(Input::Axis(GamepadAxis::Other(3), false)));
        assert_eq!(parse(""), None);
        assert_eq!(parse("nonsense"), None);
    }

    /// `Gamepad.stick` and the look speed of `_pad_input`, at Python's numbers.
    #[test]
    fn the_dead_zone_and_the_look_speed_are_pythons() {
        assert_eq!(stick(0.1, 0.1, 0.2), (0.0, 0.0));
        let (x, y) = stick(0.6, 0.0, 0.2);
        assert!((x - 0.5).abs() < 1e-6 && y == 0.0);
        let (x, y) = stick(1.0, 1.0, 0.2);
        assert!((x.hypot(y) - 1.0).abs() < 1e-6, "clamped to the unit circle");
        // 2.5 rad/s at full deflection: 2.5 * dt / 0.0024 pixels at mouse sensitivity 1
        let v = look_pixels(1.0, 0.5, 2.5, 1.0 / 60.0, 1.0, false, false);
        assert!((v.x - 2.5 / 60.0 / 0.0024).abs() < 1e-3 && (v.y - v.x / 2.0).abs() < 1e-3);
        // 'gamepad invert Y' flips it; the mouse's own invert is undone here because the walk code applies it to all mouse motion
        assert!(look_pixels(0.0, 1.0, 2.5, 0.016, 1.0, true, false).y < 0.0);
        assert!(look_pixels(0.0, 1.0, 2.5, 0.016, 1.0, false, true).y < 0.0);
        assert!(look_pixels(0.0, 1.0, 2.5, 0.016, 1.0, true, true).y > 0.0);
    }
}
