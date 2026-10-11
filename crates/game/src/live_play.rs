//! Microphone and Streamer mode in the running game (`game/live_inputs.py`): the switches, the per-frame tick, the vote tally, the on-screen
//! indicators and the state of the menu page (Esc > Mic and streamer).
//!
//! Both are OPT-IN and OFF AT EVERY LAUNCH: the switches live in memory only, never in config.toml. What is stored (Settings > Brain) is
//! only tuning that does nothing by itself: the microphone's sensitivity, which chat commands count, the vote window, the cooldown and the
//! minimum votes. A red pill says so on screen while either is on. The microphone itself is `mic.rs`; the chat reader is `twitch.rs`; the
//! vote rules are `ktf_data::streamer` (a scripted chat equals Python's).
//!
//! Not ported: the hum test's "N of 50 neurons firing" counts (the page shows the drive).

use bevy::prelude::*;
use ktf_data::streamer::{self, Rules, State, VoteBoard};
use serde_json::json;

use crate::app::{LatestSnapshot, SimLink};
use crate::config::SwitchArena;
use crate::feedback::{Notes, Tag};
use crate::labtools::LabTools;
use crate::menu_ui::{MenuFlash, MenuScreen, UserConfig};
use crate::mic::{Mic, MicStatus};
use crate::sim_thread::Command;
use crate::twitch::{self, ChatState, TwitchChat};

const LAB_LOCK: &str = "Streamer mode is off in Lab mode, so chat can't change an experiment (Play or Pet mode allows it)";
const STALE_S: f32 = 0.5;

#[derive(Resource)]
pub struct Live {
    chat: Option<TwitchChat>,
    pub stream_on: bool,
    pub stream_error: String,
    pub channel: String,
    board: VoteBoard,
    pub history: Vec<String>,
    opts_at: f32,
    pub mic_error: String,
    mic_ok: (bool, String),
    mic_checked: f32,
    fresh_at: f32,
    last_t: f32,
    sent: (f32, f32),
}

impl Default for Live {
    fn default() -> Self {
        // the salt hashing viewer names: std's per-process random state, hashed over the clock, is plenty (it only stops double votes)
        use std::hash::{BuildHasher, Hasher};
        let mut salt = [0u8; 16];
        for chunk in salt.chunks_mut(8) {
            let mut h = std::collections::hash_map::RandomState::new().build_hasher();
            h.write_u128(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos()));
            chunk.copy_from_slice(&h.finish().to_le_bytes());
        }
        Self {
            chat: None,
            stream_on: false,
            stream_error: String::new(),
            channel: String::new(),
            board: VoteBoard::new(salt),
            history: vec![],
            opts_at: -10.0,
            mic_error: String::new(),
            mic_ok: (true, String::new()),
            mic_checked: -100.0,
            fresh_at: 0.0,
            last_t: -1.0,
            sent: (0.0, 0.0),
        }
    }
}

impl Live {
    fn stop_stream(&mut self) {
        if let Some(mut c) = self.chat.take() {
            c.stop();
        }
        self.stream_on = false;
        self.board.reset();
    }

    /// The red pills and, while streaming, the tally (for the HUD).
    pub fn hud(&self, mic_on: bool, now: f32) -> Option<ktf_ui::hud::LiveView> {
        let mut pills = vec![];
        if mic_on {
            pills.push("MIC ON - listening (nothing is saved or sent)".to_string());
        }
        if let (true, Some(c)) = (self.stream_on, &self.chat) {
            let st = match c.state() {
                ChatState::Connecting => "connecting to".to_string(),
                ChatState::Live => "reading".to_string(),
                other => other.name().to_string(),
            };
            pills.push(format!("TWITCH CHAT - {st} {}:{} #{} (read-only)", c.host, c.port, c.channel));
        }
        if pills.is_empty() {
            return None;
        }
        let tally = self.stream_on.then(|| ktf_ui::hud::TallyView {
            title: match self.board.state {
                State::Idle => "CHAT VOTE: !tool NAME",
                State::Open => "CHAT VOTE",
                State::Cooldown => "VOTE COOLDOWN",
            }
            .into(),
            remaining: (self.board.state != State::Idle).then(|| self.board.remaining(f64::from(now)) as f32),
            rows: self.board.tally().into_iter().take(5).map(|(c, _, l, n)| (format!("{c} {l}"), n as u32)).collect(),
            empty: if self.board.state == State::Cooldown { "no votes counted" } else { "no votes yet" }.into(),
            history: self.history.iter().take(2).cloned().collect(),
        });
        Some(ktf_ui::hud::LiveView { pills, tally })
    }
}

fn rules(cfg: &UserConfig) -> Rules {
    let allow = streamer::COMMANDS.iter().filter(|n| cfg.0.get_bool(&format!("stream.allow_{n}"))).map(|n| n.to_string()).collect();
    Rules {
        window_s: f64::from(cfg.0.get_f32("stream.window_s")),
        cooldown_s: f64::from(cfg.0.get_f32("stream.cooldown_s")),
        min_votes: cfg.0.get_i64("stream.min_votes").max(1) as usize,
        allow,
    }
}

fn lab_locked(cfg: &UserConfig) -> bool {
    cfg.0.get_str("brain.mode") == "lab"
}

/// The names a viewer can pick: only what the game offers now.
fn options(loadout: &crate::tools::ActiveLoadout) -> std::collections::HashMap<String, Vec<(String, String)>> {
    let mut o = std::collections::HashMap::new();
    o.insert("tool".to_string(), loadout.0.tools.iter().map(|t| (streamer::slug(t), t.to_uppercase())).collect::<Vec<_>>());
    o.insert("arena".to_string(), crate::world::ARENAS.iter().map(|a| (a.to_string(), a.to_string())).collect());
    o.insert(
        "surgery".to_string(),
        fly_sim::surgery::ENTRIES
            .iter()
            .map(|(label, _)| (streamer::slug(label), label.split(" (").next().unwrap_or("").to_string()))
            .collect(),
    );
    o
}

#[allow(clippy::too_many_arguments)]
fn tick(
    mut live: ResMut<Live>,
    mut mic: ResMut<Mic>,
    cfg: Res<UserConfig>,
    loadout: Res<crate::tools::ActiveLoadout>,
    snap: Res<LatestSnapshot>,
    link: Res<SimLink>,
    mut notes: ResMut<Notes>,
    mut flash: MessageWriter<MenuFlash>,
    mut tool_state: ResMut<crate::tools::ToolState>,
    mut arena: MessageWriter<SwitchArena>,
    time: Res<Time<Real>>,
    mut tools: ResMut<LabTools>,
    screen: Res<MenuScreen>,
) {
    let now = time.elapsed_secs();
    // --- the page's commands
    for (cmd, a) in std::mem::take(&mut tools.live_cmds) {
        match cmd.as_str() {
            "lab.live_mic" => {
                let on = a["on"].as_bool().unwrap_or(false);
                if on {
                    let (ok, why) = crate::mic::availability();
                    if !ok {
                        live.mic_error = why.clone();
                        flash.write(MenuFlash(why, false));
                        continue;
                    }
                }
                crate::mic::set_mic(&mut mic, on, cfg.0.get_f32("brain.mic_sensitivity"), &link, &mut notes, now);
                live.mic_error.clear();
                live.sent = (0.0, 0.0);
            }
            "lab.live_stream" => {
                let on = a["on"].as_bool().unwrap_or(false);
                if !on {
                    if live.stream_on {
                        live.stop_stream();
                        live.stream_error.clear();
                        notes.push(now, "STREAM   off", Tag::GameRule);
                    }
                    continue;
                }
                if live.stream_on {
                    continue;
                }
                if lab_locked(&cfg) {
                    live.stream_error = LAB_LOCK.into();
                    flash.write(MenuFlash(LAB_LOCK.into(), false));
                    continue;
                }
                let started = streamer::clean_channel(&live.channel).and_then(|ch| {
                    let mut c = TwitchChat::new(&ch)?;
                    c.start(10_000 + (now * 1000.0) as u32 % 90_000);
                    Ok((ch, c))
                });
                match started {
                    Ok((ch, c)) => {
                        live.channel = ch.clone();
                        live.chat = Some(c);
                        live.stream_on = true;
                        live.stream_error.clear();
                        live.board.reset();
                        notes.push(
                            now,
                            format!(
                                "STREAM   reading #{ch} on {} (read-only, anonymous): viewers vote with !tool, !arena, !surgery",
                                streamer::HOST
                            ),
                            Tag::GameRule,
                        );
                    }
                    Err(e) => {
                        live.stream_error = e.clone();
                        flash.write(MenuFlash(e, false));
                    }
                }
            }
            "lab.live_channel" => live.channel = a["text"].as_str().unwrap_or("").trim().to_string(),
            _ => {}
        }
    }
    if let Some(h) = mic.handle.as_ref() {
        h.set_sensitivity(cfg.0.get_f32("brain.mic_sensitivity"));
    }
    // --- the microphone: a device that stops delivering gets its drive zeroed after STALE_S (the pill and switch stay)
    if let Some(h) = mic.handle.as_ref() {
        match h.status() {
            MicStatus::Unavailable(why) => {
                live.mic_error = why;
                mic.handle = None;
            }
            MicStatus::On { .. } => {
                let r = h.latest();
                if r.t != live.last_t {
                    live.last_t = r.t;
                    live.fresh_at = now;
                } else if now - live.fresh_at > STALE_S && (r.drive_a != 0.0 || r.drive_b != 0.0) {
                    mic.last = Default::default();
                }
            }
            _ => {}
        }
    }
    if screen.0 == "live_inputs" && now - live.mic_checked > 4.0 {
        live.mic_checked = now;
        live.mic_ok = crate::mic::availability();
    }
    // --- streamer mode
    if !live.stream_on {
        return;
    }
    if lab_locked(&cfg) {
        live.stop_stream();
        live.stream_error = LAB_LOCK.into();
        flash.write(MenuFlash(LAB_LOCK.into(), false));
        return;
    }
    let Some(chat) = live.chat.as_ref() else {
        live.stop_stream();
        return;
    };
    if let ChatState::Error(e) = chat.state() {
        live.stop_stream();
        live.stream_error = e.clone();
        flash.write(MenuFlash(e, false));
        return;
    }
    let incoming = chat.drain();
    let t = f64::from(now);
    if now - live.opts_at > 1.0 {
        live.opts_at = now;
        live.board.set_rules(rules(&cfg));
        live.board.options = options(&loadout);
    }
    for (user, text) in incoming {
        live.board.submit(&user, &text, t);
    }
    let Some(res) = live.board.tick(t) else { return };
    if !live.board.rules.allow.contains(&res.command) {
        return;
    }
    let label = format!("CHAT VOTED  {} {} ({} of {})", res.command, res.label, res.votes, res.total);
    match res.command.as_str() {
        "tool" => {
            let Some(name) = loadout.0.tools.iter().find(|t| streamer::slug(t) == res.slug) else { return };
            let Some(i) = crate::tools::TOOLS.iter().position(|t| t.0 == name) else { return };
            tool_state.selected = i;
        }
        "arena" => {
            let Some(i) = crate::world::ARENAS.iter().position(|a| *a == res.slug) else { return };
            arena.write(SwitchArena(i));
        }
        "surgery" => {
            let Some(k) = fly_sim::surgery::ENTRIES.iter().position(|(l, _)| streamer::slug(l) == res.slug) else { return };
            let cur = snap.0.as_deref().and_then(|s| s.surgery_modes.get(k).copied()).unwrap_or(0);
            link.0.send(Command::SurgeryEntry { index: k, mode: if cur != 0 { 0 } else { -1 } });
        }
        _ => return,
    }
    notes.push(now, label.clone(), Tag::GameRule);
    live.history.insert(0, format!("{} {}: {} of {} votes", res.command, res.label, res.votes, res.total));
    live.history.truncate(4);
}

/// The page's state (`lab.live_state`) and the pills for the HUD.
fn publish(
    live: Res<Live>,
    mic: Res<Mic>,
    cfg: Res<UserConfig>,
    time: Res<Time<Real>>,
    mut tools: ResMut<LabTools>,
    screen: Res<MenuScreen>,
) {
    if screen.0 != "live_inputs" {
        return;
    }
    let r = mic.last;
    let (net_ok, net_why) = match twitch::allowed() {
        Ok(()) => (true, String::new()),
        Err(e) => (false, e),
    };
    let chat = live
        .chat
        .as_ref()
        .filter(|_| live.stream_on)
        .map(|c| json!({"state": c.state().name(), "desc": c.describe(), "commands": c.commands_read()}));
    let _ = time;
    tools.live_state = json!({
        "mic_on": mic.handle.is_some(),
        "mic_error": live.mic_error,
        "mic_ok": live.mic_ok.0,
        "mic_why": live.mic_ok.1,
        "peak_hz": r.peak_hz,
        "bins": r.bins,
        "drive_a": r.drive_a,
        "drive_b": r.drive_b,
        "max_current": crate::mic_analysis::MAX_CURRENT,
        "sens": cfg.0.get_f32("brain.mic_sensitivity"),
        "stream_on": live.stream_on,
        "stream_error": live.stream_error,
        "channel": live.channel,
        "chat": chat,
        "lab_locked": lab_locked(&cfg),
        "net_ok": net_ok,
        "net_why": net_why,
        "lab_lock_text": LAB_LOCK,
    });
}

pub struct LivePlugin;

impl Plugin for LivePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Live>().add_systems(Update, (tick, publish).chain());
    }
}
