//! Save State and Load State (`Game.save_state`, `Game.load_state`): the `.ktfsave` container of `ktf_data::savestate`. A save holds the
//! whole simulator and the group rate trackers of the focused fly's brain, plus the room (arena, tool, kills, immortal, pain level, sim
//! speed). A save made by the Python game loads its brain state here (the simulator continues bit-exactly); its fly body, timers and
//! things in flight are not restored (the Rust fly is made fresh in the saved arena). Saves made here carry the same header, so the
//! Python game lists them, but its fly-body restore does not understand a Rust save: it refuses by the `backend` field.

use std::path::{Path, PathBuf};

use bevy::prelude::*;
use fly_sim::brain::BrainState;
use fly_sim::npy::{Array, Data};
use ktf_data::savestate::{PackInfo, SaveFile, compatible, read_meta};
use serde_json::{Value, json};

use crate::app::{LatestSnapshot, SimEvents, SimLink};
use crate::config::{Settings, SwitchArena};
use crate::fly::ResetFly;
use crate::hud_canvas::HudStats;
use crate::menu_ui::{MenuFlash, UserConfig};
use crate::sim_thread::{Command, SimEvent};
use crate::tools::{ActiveLoadout, TOOLS, ToolState};
use crate::world::ARENAS;

#[derive(Resource, Default)]
pub struct SaveLoad {
    /// `Some` while a save is waiting for the sim thread's answer.
    pub saving: bool,
    pub load: Option<PathBuf>,
    pub entries: Vec<Value>,
    pub scanned_at: f32,
}

fn saves_dir() -> PathBuf {
    ktf_data::paths::get().saves_dir
}

fn utc_stamp(fmt_dash: bool) -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs()) as i64;
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    let (h, mi, s) = (rem / 3600, rem % 3600 / 60, rem % 60);
    if fmt_dash { format!("{y:04}{m:02}{d:02}-{h:02}{mi:02}{s:02}") } else { format!("{y:04}-{m:02}-{d:02} {h:02}:{mi:02}:{s:02}") }
}

fn f64_array(v: &[f32]) -> Array {
    Array { shape: vec![v.len()], data: Data::F64(v.iter().map(|&x| f64::from(x)).collect()) }
}

fn pack_info(snap: &LatestSnapshot) -> Option<PackInfo> {
    let s = snap.0.as_deref()?;
    Some(PackInfo { n_neurons: s.neurons, synapses: s.info.as_deref()?.synapses, brain: "adult".into(), mode: "3d".into() })
}

#[allow(clippy::too_many_arguments)]
fn save_and_load(
    mut sl: ResMut<SaveLoad>,
    events: Res<SimEvents>,
    link: Res<SimLink>,
    snap: Res<LatestSnapshot>,
    cfg: Res<UserConfig>,
    real: Res<Time<Real>>,
    arena: Res<crate::config::Arena>,
    mut tools: ResMut<ToolState>,
    mut lo: ResMut<ActiveLoadout>,
    mut stats: ResMut<HudStats>,
    mut settings: ResMut<Settings>,
    mut switch: MessageWriter<SwitchArena>,
    mut reset: MessageWriter<ResetFly>,
    mut flash: MessageWriter<MenuFlash>,
    screen: Res<crate::menu_ui::MenuScreen>,
) {
    // --- save: ask the sim thread, then write when the state arrives
    for ev in &events.0 {
        let SimEvent::State(bs) = ev else { continue };
        if !std::mem::take(&mut sl.saving) {
            continue;
        }
        let Some(pack) = pack_info(&snap) else { continue };
        let mut meta = SaveFile::new_header(env!("CARGO_PKG_VERSION"), "3d", "adult", &pack, arena.name(), arena.0);
        meta["created"] = utc_stamp(false).into();
        meta["platform"] = "Linux".into();
        meta["seed"] = cfg.0.get_i64("brain.seed").into();
        meta["tool"] = tools.selected.into();
        meta["kills"] = stats.kills.into();
        meta["immortal"] = settings.immortal.into();
        meta["pain_level"] = snap.0.as_deref().map_or(0, |s| s.pain_level).into();
        meta["sim_speed"] = cfg.0.get_f32("brain.sim_speed").into();
        meta["individuality"] = cfg.0.get_str("brain.individuality").into();
        meta["flies"] = json!([{"seed": cfg.0.get_i64("brain.seed"), "primary": true, "brain": {"steps": bs.steps}, "fly": {}, "slot": {}, "fly_class": "RustFly"}]);
        let mut file = SaveFile { meta, arrays: Default::default() };
        file.put_sim_state(0, &bs.sim);
        file.arrays.insert("f0_b_fast".into(), f64_array(&bs.fast));
        file.arrays.insert("f0_b_base".into(), f64_array(&bs.base));
        file.meta["flies"][0]["brain"]["steps"] = bs.steps.into();
        let dir = saves_dir();
        let _ = std::fs::create_dir_all(&dir);
        let stem = format!("save-{}", utc_stamp(true));
        let (mut path, mut n) = (dir.join(format!("{stem}{}", ktf_data::savestate::SUFFIX)), 2);
        while path.exists() {
            path = dir.join(format!("{stem}-{n}{}", ktf_data::savestate::SUFFIX));
            n += 1;
        }
        match file.write(&path) {
            Ok(()) => {
                flash.write(MenuFlash(format!("Saved {}", path.file_name().unwrap_or_default().to_string_lossy()), true));
            }
            Err(e) => {
                flash.write(MenuFlash(format!("Couldn't save: {e}"), false));
            }
        }
    }
    // --- load
    if let Some(path) = sl.load.take() {
        let result = (|| -> Result<(), String> {
            let meta = read_meta(&path).map_err(|e| e.to_string())?;
            let pack = pack_info(&snap).ok_or("the brain is not ready yet")?;
            if let Some(why) = compatible(&meta, &pack) {
                return Err(format!("can't load this save: {why}"));
            }
            let file = SaveFile::read(&path).map_err(|e| e.to_string())?;
            let sim = file.sim_state(0).map_err(|e| e.to_string())?;
            let groups = |name: &str| -> Vec<f32> {
                match file.arrays.get(name).map(|a| &a.data) {
                    Some(Data::F64(v)) => v.iter().map(|&x| x as f32).collect(),
                    Some(Data::F32(v)) => v.clone(),
                    _ => vec![],
                }
            };
            let steps = meta["flies"][0]["brain"]["steps"].as_u64().unwrap_or(0);
            link.0.send(Command::ImportState(Box::new(BrainState { sim, fast: groups("f0_b_fast"), base: groups("f0_b_base"), steps })));
            // the room
            let name = meta["arena"].as_str().unwrap_or("room");
            let arena_i = ARENAS.iter().position(|a| *a == name).unwrap_or(0);
            switch.write(SwitchArena(arena_i));
            reset.write(ResetFly::default());
            if let Some(t) = meta["tool"].as_u64().filter(|&t| (t as usize) < TOOLS.len()) {
                tools.selected = t as usize;
                lo.0.show(TOOLS[t as usize].0);
            }
            stats.kills = meta["kills"].as_u64().unwrap_or(0) as u32;
            settings.immortal = meta["immortal"].as_bool().unwrap_or(false);
            link.0.send(Command::PainLevel(meta["pain_level"].as_u64().unwrap_or(0).min(2) as usize));
            Ok(())
        })();
        match result {
            Ok(()) => {
                flash.write(MenuFlash(format!("Loaded {}", path.file_name().unwrap_or_default().to_string_lossy()), true));
            }
            Err(e) => {
                flash.write(MenuFlash(e, false));
            }
        }
    }
    // --- the Load State list, rescanned every 2 s while that page is open
    let now = real.elapsed_secs();
    if screen.0 == "load_state" && now - sl.scanned_at > 2.0 {
        sl.scanned_at = now;
        let dir = saves_dir();
        let mut files: Vec<(std::time::SystemTime, PathBuf)> = std::fs::read_dir(&dir)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| format!(".{}", e.to_string_lossy()) == ktf_data::savestate::SUFFIX))
            .filter_map(|p| Some((p.metadata().ok()?.modified().ok()?, p)))
            .collect();
        files.sort_by_key(|f| std::cmp::Reverse(f.0));
        let pack = pack_info(&snap);
        sl.entries = files
            .into_iter()
            .take(40)
            .map(|(_, f)| {
                let name = f.file_name().unwrap_or_default().to_string_lossy().to_string();
                match read_meta(&f) {
                    Ok(meta) => {
                        let why = pack.as_ref().and_then(|p| compatible(&meta, p));
                        json!({"file": f.to_string_lossy(), "name": name, "meta": meta, "why": why})
                    }
                    Err(e) => json!({"file": f.to_string_lossy(), "name": name, "meta": null, "why": e.to_string()}),
                }
            })
            .collect();
    }
}

/// `save_state` and `load_state:<file>` come in as menu actions.
pub fn request_save(sl: &mut SaveLoad, link: &SimLink) {
    sl.saving = true;
    link.0.send(Command::ExportState);
}

pub fn request_load(sl: &mut SaveLoad, file: &str) {
    sl.load = Some(Path::new(file).to_path_buf());
}

pub struct SaveStatePlugin;

impl Plugin for SaveStatePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SaveLoad>().add_systems(Update, save_and_load);
    }
}
