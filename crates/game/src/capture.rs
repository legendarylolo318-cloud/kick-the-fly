//! Developer/CI capture: `--shot FILE [--shot-frames N]` saves a screenshot of the primary window after N frames and exits. It is how the
//! rendering is checked on a machine with a display (and what the screenshots in the docs are made with).

use bevy::app::AppExit;
use bevy::prelude::*;
use bevy::render::view::screenshot::{Screenshot, save_to_disk};

#[derive(Resource, Debug, Clone)]
pub struct CaptureRequest {
    pub path: std::path::PathBuf,
    pub after_frames: u32,
    pub wait_after: u32,
    /// `--shot-seq COUNT,EVERY`: instead of one picture, COUNT of them, EVERY frames apart, named `<stem>_000.png`, ... (one run makes the
    /// frames of a GIF; launching the game once per frame showed the same opening again and again).
    pub seq: Option<(u32, u32)>,
}

#[derive(Resource, Default)]
struct CaptureState {
    frame: u32,
    taken_at: Option<u32>,
}

fn capture(req: Res<CaptureRequest>, mut st: ResMut<CaptureState>, mut commands: Commands, mut exit: MessageWriter<AppExit>) {
    st.frame += 1;
    if let Some((count, every)) = req.seq {
        let every = every.max(1);
        if st.frame >= req.after_frames && (st.frame - req.after_frames).is_multiple_of(every) {
            let k = (st.frame - req.after_frames) / every;
            if k < count {
                let stem = req.path.file_stem().and_then(|x| x.to_str()).unwrap_or("frame");
                commands.spawn(Screenshot::primary_window()).observe(save_to_disk(req.path.with_file_name(format!("{stem}_{k:03}.png"))));
            } else if st.taken_at.is_none() {
                st.taken_at = Some(st.frame);
            }
        }
    } else if st.taken_at.is_none() && st.frame >= req.after_frames {
        commands.spawn(Screenshot::primary_window()).observe(save_to_disk(req.path.clone()));
        st.taken_at = Some(st.frame);
    }
    if let Some(t) = st.taken_at
        && st.frame >= t + req.wait_after
    {
        exit.write(AppExit::Success);
    }
}

pub struct CapturePlugin(pub CaptureRequest);

impl Plugin for CapturePlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(self.0.clone()).init_resource::<CaptureState>().add_systems(Last, capture);
    }
}

/// `--tool NAME --fire SECONDS [--hold SECONDS]`: select a tool and press the left button at a time, so a screenshot (or a test) can show
/// what the tool does without a person at the mouse.
#[derive(Resource, Debug, Clone)]
pub struct Script {
    pub tool: usize,
    pub fire_at: f32,
    pub hold_s: f32,
}

fn run_script(
    script: Res<Script>,
    time: Res<Time<Real>>,
    mut mouse: ResMut<ButtonInput<MouseButton>>,
    mut tools: ResMut<crate::tools::ToolState>,
    mut look: ResMut<crate::player::MouseLook>,
) {
    tools.selected = script.tool;
    look.captured = true;
    let t = time.elapsed_secs();
    if t >= script.fire_at && t < script.fire_at + script.hold_s {
        mouse.press(MouseButton::Left);
    } else if t >= script.fire_at + script.hold_s {
        mouse.release(MouseButton::Left);
    }
}

pub struct ScriptPlugin(pub Script);

impl Plugin for ScriptPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(self.0.clone()).add_systems(PreUpdate, run_script.after(bevy::input::InputSystems));
    }
}

/// `--kill-at SECONDS`: hurt the fly to death at a time, for a screenshot of the kill cam.
#[derive(Resource, Clone, Copy)]
pub struct KillAt(pub f32);

fn kill_at(
    at: Res<KillAt>,
    time: Res<Time<Real>>,
    flies: Query<Entity, With<crate::fly::Fly>>,
    mut dmg: MessageWriter<crate::fly::FlyDamage>,
    mut done: Local<bool>,
) {
    if !*done && time.elapsed_secs() >= at.0 {
        *done = true;
        for e in &flies {
            dmg.write(crate::fly::FlyDamage { fly: e, amount: 1000.0, source: "the test harness" });
        }
    }
}

pub struct KillAtPlugin(pub f32);

impl Plugin for KillAtPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(KillAt(self.0)).add_systems(Update, kill_at);
    }
}

/// `--demo`: the scene of the README's GIF, played by the game itself. You walk up to the fly and keep facing it, swat twice, then hold the
/// blowtorch on it. Used with `--arena field --shot-seq ...`.
#[derive(Resource, Default)]
pub struct Demo;

#[allow(clippy::too_many_arguments)]
fn run_demo(
    time: Res<Time<Real>>,
    mut mouse: ResMut<ButtonInput<MouseButton>>,
    mut tools: ResMut<crate::tools::ToolState>,
    mut look: ResMut<crate::player::MouseLook>,
    mut player: Single<(&mut crate::player::Player, &mut Transform), Without<crate::fly::FlyPart>>,
    flies: Query<&crate::fly::Fly>,
    parts: Query<&Transform, With<crate::fly::FlyPart>>,
) {
    let t = time.elapsed_secs();
    let dt = time.delta_secs().min(0.05);
    look.captured = true;
    let tool = |name: &str| crate::tools::TOOLS.iter().position(|x| x.0 == name).unwrap_or(0);
    // the timeline: (from, to, tool, button down)
    let steps = [
        (0.0, 3.4, "swatter", false),
        (3.4, 3.5, "swatter", true),
        (3.5, 4.3, "swatter", false),
        (4.3, 4.4, "swatter", true),
        (4.4, 5.2, "swatter", false),
        (5.2, 5.6, "torch", false),
        (5.6, 8.0, "torch", true),
        (8.0, 99.0, "torch", false),
    ];
    let (_, _, name, down) = steps.iter().find(|s| t >= s.0 && t < s.1).copied().unwrap_or((0.0, 0.0, "swatter", false));
    tools.selected = tool(name);
    if down {
        mouse.press(MouseButton::Left);
    } else {
        mouse.release(MouseButton::Left);
    }
    // stay about a metre and a half from the fly, looking at it
    let Some(target) = flies.iter().next().and_then(|f| parts.get(f.parts[crate::fly::Part::Thorax.index()]).ok()).map(|x| x.translation)
    else {
        return;
    };
    let (p, tf) = &mut *player;
    let flat = Vec2::new(target.x - tf.translation.x, target.z - tf.translation.z);
    let dist = flat.length();
    if dist > 1.6 && t > 1.0 {
        let step = flat / dist * (dist - 1.6).min(3.5 * dt);
        tf.translation.x += step.x;
        tf.translation.z += step.y;
    }
    let d = target - Vec3::new(tf.translation.x, p.eye_h, tf.translation.z);
    if d.length() > 1e-3 {
        let (yaw, pitch) = (d.z.atan2(d.x), (d.y / d.length()).asin().clamp(-1.4, 1.4));
        // turn smoothly, so the picture does not jump
        let k = (dt * 8.0).min(1.0);
        let dyaw = (yaw - p.yaw + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI;
        p.yaw += dyaw * k;
        p.pitch += (pitch - p.pitch) * k;
    }
}

pub struct DemoPlugin;

impl Plugin for DemoPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Demo>().add_systems(PreUpdate, run_demo.after(bevy::input::InputSystems));
    }
}
