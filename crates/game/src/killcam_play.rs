//! The kill cam in the game: on death, a slow-motion replay of the brain's last six seconds with the neurons that rose most highlighted.
//! Python: `core/killcam.py` (the logic: `killcam.rs`) and `game/extras3.py:Extras`.
//!
//! The recording is the snapshot's per-neuron rates, one sample per 8 brain steps, in a ring of the last 6 s; nothing is fed back to the
//! simulation. On [`FlyDied`] the ring is frozen into a replay; the big brain view shows it at a quarter of real speed (a fifth with
//! reduced flashing), the highlighted neurons pulse (a steady glow with reduced flashing), and Enter or Space skips it.

use bevy::prelude::*;

use crate::app::LatestSnapshot;
use crate::brainview::BrainView;
use crate::config::Settings;
use crate::feedback::{Notes, Tag};
use crate::fly::FlyDied;
use crate::killcam::{self, Buffer, MIN_RISE_HZ, Player, Replay, TOP_N};

#[derive(Resource, Default)]
pub struct KillCam {
    buffer: Option<Buffer>,
    /// The fly whose brain the buffer holds: a focus change starts a new recording, so a replay never splices two brains together.
    slot: u32,
    pub player: Option<Player>,
    pub summary: Vec<String>,
    was_big: bool,
    hold: f32,
}

#[derive(Component)]
struct KillCamText;

fn setup(mut commands: Commands) {
    commands.spawn((
        KillCamText,
        Text::new(""),
        TextFont { font_size: FontSize::Px(17.0), ..default() },
        TextColor(Color::srgb(1.0, 0.85, 0.8)),
        Node {
            position_type: PositionType::Absolute,
            left: px(24),
            top: px(90),
            display: Display::None,
            padding: UiRect::all(px(10)),
            ..default()
        },
        BackgroundColor(Color::srgba(0.05, 0.02, 0.03, 0.8)),
    ));
}

/// Keep the last six seconds of rates while the fly lives.
fn record(snap: Res<LatestSnapshot>, mut kc: ResMut<KillCam>) {
    if kc.player.is_some() {
        return;
    }
    let Some(s) = snap.0.as_deref() else { return };
    if s.rates.is_empty() || s.dead {
        return;
    }
    let n = s.rates.len();
    if kc.buffer.as_ref().is_none_or(|b| b.n() != n) || kc.slot != s.focus {
        kc.buffer = Some(Buffer::new(n));
        kc.slot = s.focus;
    }
    if let Some(b) = kc.buffer.as_mut()
        && b.due(s.step)
    {
        b.push(s.step, &s.rates);
    }
}

fn on_death(
    mut died: MessageReader<FlyDied>,
    bodies: Query<(&crate::fly::Fly, &crate::fly::Slot)>,
    flies: Res<crate::flies_play::Flies>,
    mut kc: ResMut<KillCam>,
    snap: Res<LatestSnapshot>,
    settings: Res<Settings>,
    mut view: ResMut<BrainView>,
    mut notes: ResMut<Notes>,
    time: Res<Time<Real>>,
) {
    // the replay is the focused brain's last seconds, and like the death screen it waits for the last fly: while another fly lives the
    // focus moves on to it (`flies_play::follow_nearest`) and play goes on
    let focused_died = died.read().fold(false, |acc, d| acc | bodies.get(d.fly).is_ok_and(|(_, s)| s.0 == flies.focus));
    let all_dead = bodies.iter().all(|(f, _)| f.dead());
    if !focused_died || !all_dead || !settings.killcam || kc.player.is_some() {
        return;
    }
    let Some(replay) = kc.buffer.as_ref().and_then(Buffer::freeze) else { return };
    let info = snap.0.as_deref().and_then(|s| s.info.clone());
    let risers = replay.top_risers(TOP_N, MIN_RISE_HZ);
    let mut lines = vec!["KILL CAM  the brain's last 6 seconds, slowed down  (Enter skips)".to_string()];
    if risers.is_empty() {
        lines.push("no neuron's firing rose clearly before the end".into());
    } else {
        lines.push("neurons whose firing rose most (CONNECTOME: the simulator's own rates):".into());
        for r in risers.iter().take(8) {
            let ty = info.as_ref().and_then(|i| i.types.get(r.index)).map_or("?", String::as_str);
            lines.push(format!("  {ty:<14} {:>5.1} -> {:>5.1} Hz", r.before_hz, r.after_hz));
        }
    }
    kc.summary = lines;
    view.highlight = risers.iter().map(|r| r.index as u32).collect();
    kc.was_big = view.big;
    view.big = true;
    let speed = if settings.reduced_flashing { killcam::SPEED_REDUCED } else { killcam::SPEED };
    kc.player = Some(Player::new(replay, speed));
    kc.hold = 0.0;
    notes.push(time.elapsed_secs(), "KILL CAM replaying the last 6 s of the brain", Tag::Connectome);
}

fn play(
    time: Res<Time<Real>>,
    keys: Res<ButtonInput<KeyCode>>,
    mut kc: ResMut<KillCam>,
    mut view: ResMut<BrainView>,
    mut text: Single<(&mut Text, &mut Node), With<KillCamText>>,
) {
    let kc = &mut *kc;
    let Some(p) = kc.player.as_mut() else {
        text.1.display = Display::None;
        return;
    };
    if keys.just_pressed(KeyCode::Enter) || keys.just_pressed(KeyCode::Space) {
        p.skip();
    }
    p.advance(time.delta_secs());
    let mut rates = Vec::new();
    let frame = p.frame();
    Replay::rates_per_step(&p.replay, frame, &mut rates);
    view.replay = Some(rates);
    text.1.display = Display::Flex;
    text.0.0 = format!("{}\n[{:>3.0}%]", kc.summary.join("\n"), p.progress() * 100.0);
    if p.done {
        kc.hold += time.delta_secs();
        if kc.hold > 3.0 || p.skipped {
            kc.player = None;
            view.replay = None;
            view.highlight.clear();
            view.big = kc.was_big;
            if let Some(b) = kc.buffer.as_mut() {
                b.reset();
            }
        }
    }
}

pub struct KillCamPlugin;

impl Plugin for KillCamPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<KillCam>().add_systems(Startup, setup).add_systems(Update, (record, on_death, play).chain());
    }
}
