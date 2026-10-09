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
}

#[derive(Resource, Default)]
struct CaptureState {
    frame: u32,
    taken_at: Option<u32>,
}

fn capture(req: Res<CaptureRequest>, mut st: ResMut<CaptureState>, mut commands: Commands, mut exit: MessageWriter<AppExit>) {
    st.frame += 1;
    if st.taken_at.is_none() && st.frame >= req.after_frames {
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
