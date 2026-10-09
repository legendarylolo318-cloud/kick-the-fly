//! The Neurodex in the running game (`game/extras3.py`'s Neurodex part): the type table is built on a worker thread, the sim thread checks
//! for discoveries every 10 steps and reports them, and this module marks the progress file, shows the toasts and the reactions-log line,
//! and runs the Neuron of the Day card. What is CONNECTOME, GAME RULE or LITERATURE is in `fly_sim::neurodex` and `ktf_data::notd`.

use std::sync::{Arc, Mutex, PoisonError};

use bevy::prelude::*;
use fly_sim::neurodex::TypeTable;
use ktf_data::notd::{self, Card};
use ktf_data::stores::Progress;
use ktf_ui::pages::neurodex::{NeurodexShared, SharedDex};

use crate::app::SimLink;
use crate::feedback::{Notes, Sfx, Tag};
use crate::menu_ui::UserConfig;
use crate::sim_thread::{Command, PackPath, SimEvent};

pub const TOAST_S: f32 = 4.0;
pub const TOAST_MAX: usize = 3;
pub const NOTD_S: f32 = 14.0;

#[derive(Resource)]
pub struct Dex {
    pub shared: SharedDex,
    /// (text, born) in real seconds
    pub toasts: Vec<(String, f32)>,
    pub notd: Option<(Card, f32)>,
    started: bool,
    /// The Neuron of the Day was decided for this launch (shown, or skipped because the tutorial is on).
    pub notd_tried: bool,
}

/// UTC date as `date.toordinal()` (the Python game used the local date).
pub fn today_ordinal() -> i64 {
    let days = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs()) as i64 / 86_400;
    days + 719_163
}

fn progress_path() -> std::path::PathBuf {
    ktf_data::paths::get().memory_dir.join(ktf_data::stores::NEURODEX_FILE)
}

/// Build the table once the sim thread has loaded the brain (a few seconds on the real pack), off the game thread.
fn start_build(mut dex: ResMut<Dex>, link: Res<SimLink>, pack: Res<PackPath>, cfg: Res<UserConfig>) {
    if dex.started {
        return;
    }
    let Some(shared) = link.0.shared.lock().unwrap_or_else(PoisonError::into_inner).clone() else { return };
    dex.started = true;
    link.0.send(Command::DexEnabled(cfg.0.get_bool("brain.neurodex")));
    let state = dex.shared.clone();
    let tx = link.0.sender();
    let path = pack.0.clone();
    state.lock().unwrap_or_else(PoisonError::into_inner).state = "building".into();
    std::thread::Builder::new()
        .name("neurodex-table".into())
        .spawn(move || {
            let build = || -> Result<TypeTable, String> {
                let read = |mut npz: fly_sim::npy::Npz| -> Result<TypeTable, String> {
                    let indptr = npz.take("indptr").and_then(|a| a.into_i32()).map_err(|e| e.to_string())?;
                    let indices = npz.take("indices").and_then(|a| a.into_i32()).map_err(|e| e.to_string())?;
                    let data = npz.take("data").and_then(|a| a.into_i16()).map_err(|e| e.to_string())?;
                    Ok(TypeTable::build(
                        &shared.labels,
                        "adult",
                        Some(fly_sim::neurodex::RawCsr { indptr: &indptr, indices: &indices, data: &data }),
                    ))
                };
                match (&path, crate::sim_thread::embedded_pack()) {
                    (Some(p), _) => read(
                        fly_sim::npy::Npz::read(std::io::BufReader::new(std::fs::File::open(p).map_err(|e| e.to_string())?))
                            .map_err(|e| e.to_string())?,
                    ),
                    (None, Some(bytes)) => read(fly_sim::npy::Npz::read(std::io::Cursor::new(bytes)).map_err(|e| e.to_string())?),
                    (None, None) => Ok(TypeTable::build(&shared.labels, "adult", None)),
                }
            };
            match build() {
                Ok(table) => {
                    let table = Arc::new(table);
                    let known: Vec<String> = {
                        let mut g = state.lock().unwrap_or_else(PoisonError::into_inner);
                        g.table = Some(table.clone());
                        g.state = "ready".into();
                        g.progress.brains.get("adult").map(|b| b.keys().cloned().collect()).unwrap_or_default()
                    };
                    let _ = tx.send(Command::DexTable { table, known });
                }
                Err(e) => {
                    let mut g = state.lock().unwrap_or_else(PoisonError::into_inner);
                    g.state = "error".into();
                    g.error = e;
                }
            }
        })
        .expect("could not start the Neurodex thread");
}

fn poll(
    mut dex: ResMut<Dex>,
    link: Res<SimLink>,
    events: Res<crate::app::SimEvents>,
    real: Res<Time<Real>>,
    mut notes: ResMut<Notes>,
    mut sfx: MessageWriter<Sfx>,
    cfg: Res<UserConfig>,
    menu: Res<crate::menu_ui::MenuOpen>,
    first: Res<crate::first_run::FirstRun>,
) {
    let now = real.elapsed_secs();
    for ev in &events.0 {
        let SimEvent::Discovered(f) = ev else { continue };
        let mut g = dex.shared.lock().unwrap_or_else(PoisonError::into_inner);
        let when = format!("{}T00:00:00", date_string());
        g.progress.mark("adult", &f.name, f.ratio, f.how, &when);
        let how = match f.how {
            "stimulated" => " (by stimulation)",
            "rest" => " (at rest)",
            _ => "",
        };
        let x = g.progress.record("adult", &f.name).map_or(0.0, |d| d.x);
        g.progress.save();
        drop(g);
        dex.toasts.push((format!("NEURODEX  {} discovered{how}", f.name), now));
        notes.push(now, format!("NEURODEX  {} discovered{how}  x{x:.1} calm", f.name), Tag::GameRule);
        sfx.write(Sfx::vol("click", 0.6));
    }
    dex.toasts.retain(|(_, t)| now - t < TOAST_S);
    // a wipe in the panel: tell the tracker and save
    {
        let mut g = dex.shared.lock().unwrap_or_else(PoisonError::into_inner);
        if std::mem::take(&mut g.wiped) {
            g.progress.save();
            let known: Vec<String> = g.progress.brains.get("adult").map(|b| b.keys().cloned().collect()).unwrap_or_default();
            link.0.send(Command::DexKnown(known));
        }
    }
    link.0.send(Command::DexEnabled(cfg.0.get_bool("brain.neurodex")));
    // the Neuron of the Day, once per launch, when the table is ready (never in a menu)
    if !dex.notd_tried && first.0 == Some(crate::first_run::Launch::NeuronOfDay) && cfg.0.get_bool("brain.neuron_of_day") && !menu.0 {
        let table = dex.shared.lock().unwrap_or_else(PoisonError::into_inner).table.clone();
        if let Some(t) = table {
            dex.notd_tried = true;
            if let Some(c) = notd::card(today_ordinal(), &t, cfg.0.lab()) {
                dex.notd = Some((c, now));
            }
        }
    }
    if dex.notd.as_ref().is_some_and(|(_, t0)| now - t0 > NOTD_S) {
        dex.notd = None;
    }
    // "Try it" from the panel
    let card = dex.shared.lock().unwrap_or_else(PoisonError::into_inner).try_card.take();
    if let Some(c) = card {
        let note = apply(&c, &link);
        notes.push(now, note, Tag::GameRule);
    }
}

fn date_string() -> String {
    let d = today_ordinal() - 719_163;
    let z = d + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let dd = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{dd:02}")
}

/// `neuron_of_day.apply`: set the card's experiment up. It never runs anything by itself; the brain surgery switches are on and the fly and
/// the brain panel do the rest. (The Lab's aimable laser is not part of this build yet, so a "laser" plan uses brain surgery too.)
pub fn apply(c: &Card, link: &SimLink) -> String {
    match c.plan.action {
        "none" => "TRY IT  nothing to set up for this one".into(),
        action => {
            let spec = format!("type:{}", c.plan.types.join(","));
            link.0.send(Command::Surgery { spec, mode: 1 });
            if action == "laser" {
                format!(
                    "TRY IT  stimulating {} (the laser is Lab-only and not in this build yet: brain surgery instead)",
                    c.plan.target.as_deref().unwrap_or("?")
                )
            } else {
                format!("TRY IT  stimulating {} type(s) in brain surgery", c.plan.types.len())
            }
        }
    }
}

pub struct DexPlugin;

impl Plugin for DexPlugin {
    fn build(&self, app: &mut App) {
        let shared: SharedDex = Arc::new(Mutex::new(NeurodexShared::new(Progress::open(progress_path()))));
        app.insert_resource(Dex { shared, toasts: vec![], notd: None, started: false, notd_tried: false })
            .add_systems(Update, (start_build, poll).chain());
    }
}
