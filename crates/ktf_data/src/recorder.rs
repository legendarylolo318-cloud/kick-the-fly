//! Recording and export (`lab/recorder.py`): spike times and firing rates to CSV and npz, always with a metadata JSON.
//!
//! A [`Recorder`] watches a set of neurons (row indices into the connectome) and keeps every spike of them, cheaply, plus the run's context:
//! the stimuli the brain was given, the notes and tool uses the game logged, where the fly was and how it was doing. Saving writes, next to
//! each other: `<name>-spikes.csv`, `-rates.csv`, `-group-rates.csv`, `-events.csv`, `-kinematics.csv`, `<name>.npz` and
//! `<name>-metadata.json`. The NWB file (HDF5) is not written by this build.

use std::io::Write as _;
use std::path::{Path, PathBuf};

use fly_sim::npy::{Array, Data};
use fly_sim::pack::Labels;
use serde_json::{Value, json};

pub const BIN_STEPS: i64 = 10;
pub const CONNECTOME: &str = "Janelia FlyEM MaleCNS v1.0 (minconf 0.5, synapse count >= 3, signed by predicted neurotransmitter)";
/// `lab.RECORD_GROUPS`: the offered groups, label and spec.
pub const RECORD_GROUPS: [(&str, &str); 15] = [
    ("Giant fiber DNp01", "dnp01"),
    ("Looming detectors LPLC2, LC4", "loom"),
    ("Touch neurons", "touch"),
    ("Reaction descending neurons", "reaction_dns"),
    ("All descending neurons", "superclass:descending_neuron"),
    ("Proboscis motor neuron MN9", "mn9"),
    ("Grooming command aDN1/aDN2", "adn"),
    ("Antennal JO-C/E", "jo_ce"),
    ("Sugar-pathway taste neurons", "sweet"),
    ("Kenyon cells", "prefix:KC"),
    ("Mushroom body output neurons", "prefix:MBON"),
    ("Reward dopamine PAM", "reward"),
    ("Punishment dopamine PPL1", "punish"),
    ("Leg motor neurons", "mn_legs"),
    ("Whole brain (large files)", "whole brain"),
];

#[derive(Debug, Clone)]
pub struct Event {
    pub step: i64,
    pub kind: String,
    pub label: String,
    pub detail: String,
    pub value: f64,
}

#[derive(Debug, Clone)]
pub struct Kin {
    pub step: i64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub speed: f64,
    pub health: f64,
    pub action: String,
    pub arena: String,
}

pub struct Recorder {
    pub groups: Vec<(String, Vec<u32>)>,
    pub rows: Vec<u32>,
    group_of: Vec<String>,
    /// row -> index into `rows` (-1: not watched).
    lookup: Vec<i32>,
    steps: Vec<i32>,
    idx: Vec<i32>,
    pub start_step: u64,
    pub end_step: u64,
    pub events: Vec<Event>,
    pub kinematics: Vec<Kin>,
    pub active: bool,
}

impl Recorder {
    /// `Recorder(brain, groups).start()`: groups with no neurons are left out.
    pub fn new(n_neurons: usize, groups: Vec<(String, Vec<u32>)>, brain_steps: u64) -> Self {
        let groups: Vec<(String, Vec<u32>)> = groups.into_iter().filter(|g| !g.1.is_empty()).collect();
        let mut rows: Vec<u32> = groups.iter().flat_map(|g| g.1.iter().copied()).collect();
        rows.sort_unstable();
        rows.dedup();
        let mut group_of = vec![String::new(); rows.len()];
        let mut lookup = vec![-1i32; n_neurons];
        for (i, &r) in rows.iter().enumerate() {
            lookup[r as usize] = i as i32;
        }
        for (name, grows) in &groups {
            for &r in grows {
                let i = lookup[r as usize] as usize;
                if group_of[i].is_empty() {
                    group_of[i] = name.clone();
                } else {
                    group_of[i] = format!("{}|{name}", group_of[i]);
                }
            }
        }
        Self {
            groups,
            rows,
            group_of,
            lookup,
            steps: Vec::new(),
            idx: Vec::new(),
            start_step: brain_steps,
            end_step: brain_steps,
            events: Vec::new(),
            kinematics: Vec::new(),
            active: true,
        }
    }

    /// One brain step: `step` is the step index before the step ran (`Brain.steps`), `spikes` the neurons that fired.
    pub fn push(&mut self, step: u64, spikes: &[u32]) {
        self.end_step = step + 1;
        for &s in spikes {
            let i = self.lookup[s as usize];
            if i >= 0 {
                self.idx.push(i);
                self.steps.push((step - self.start_step) as i32);
            }
        }
    }

    pub fn log_event(&mut self, step: u64, kind: &str, label: &str, detail: &str, value: f64) {
        if self.active && self.events.len() < 200_000 {
            self.events.push(Event { step: step as i64, kind: kind.into(), label: label.into(), detail: detail.into(), value });
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn log_kinematics(&mut self, step: u64, x: f64, y: f64, z: f64, speed: f64, health: f64, action: &str, arena: &str) {
        if self.active && self.kinematics.len() < 200_000 {
            self.kinematics.push(Kin { step: step as i64, x, y, z, speed, health, action: action.into(), arena: arena.into() });
        }
    }

    pub fn seconds(&self) -> f64 {
        (self.end_step - self.start_step) as f64 * 0.005
    }

    pub fn n_spikes(&self) -> usize {
        self.idx.len()
    }
}

fn q(c: &str) -> String {
    if c.contains([',', '"', '\n', '\r']) { format!("\"{}\"", c.replace('"', "\"\"")) } else { c.to_string() }
}

fn csv_line(fields: &[String]) -> String {
    format!("{}\r\n", fields.iter().map(|f| q(f)).collect::<Vec<_>>().join(","))
}

fn npz(path: &Path, arrays: &[(&str, Array)]) -> std::io::Result<()> {
    let mut zw = zip::ZipWriter::new(std::fs::File::create(path)?);
    let opts = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
    for (name, a) in arrays {
        zw.start_file(format!("{name}.npy"), opts).map_err(std::io::Error::other)?;
        zw.write_all(&crate::savestate::to_npy(a).map_err(|e| std::io::Error::other(e.0))?)?;
    }
    zw.finish().map_err(std::io::Error::other)?;
    Ok(())
}

fn arr<T>(shape: Vec<usize>, data: Data) -> Array {
    let _ = std::marker::PhantomData::<T>;
    Array { shape, data }
}

impl Recorder {
    /// `save`: the CSVs, the npz and the metadata JSON next to `stem`. `meta` is `recorder.metadata(...)`'s dict (the caller knows the app
    /// and game state); the recording's own block is added to it. Returns the files in the order Python lists them.
    pub fn save(&self, stem: &Path, labels: &Labels, meta: Value) -> std::io::Result<Vec<PathBuf>> {
        let n_steps = ((self.end_step - self.start_step) as i64).max(1);
        let rows = &self.rows;
        let body: Vec<i64> = rows.iter().map(|&r| labels.body_id.get(r as usize).copied().unwrap_or(-1)).collect();
        let types: Vec<String> = rows.iter().map(|&r| labels.types[r as usize].clone()).collect();
        let inst: Vec<String> = rows.iter().map(|&r| labels.instance.get(r as usize).cloned().unwrap_or_default()).collect();
        let mut counts = vec![0i64; rows.len()];
        for &i in &self.idx {
            counts[i as usize] += 1;
        }
        if let Some(d) = stem.parent() {
            std::fs::create_dir_all(d)?;
        }
        let name = stem.file_name().unwrap_or_default().to_string_lossy().to_string();
        let sib = |suffix: &str| stem.with_file_name(format!("{name}{suffix}"));
        let mut out: Vec<PathBuf> = Vec::new();
        // spikes: time_ms, row, body_id, type, instance, group (ordered by step, then neuron index)
        let p = sib("-spikes.csv");
        let mut order: Vec<usize> = (0..self.idx.len()).collect();
        order.sort_by_key(|&k| (self.steps[k], self.idx[k]));
        let mut text = csv_line(&["time_ms", "row", "body_id", "type", "instance", "group"].map(String::from));
        for k in order {
            let i = self.idx[k] as usize;
            text.push_str(&csv_line(&[
                format!("{:.1}", f64::from(self.steps[k]) * 5.0),
                rows[i].to_string(),
                body[i].to_string(),
                types[i].clone(),
                inst[i].clone(),
                self.group_of[i].clone(),
            ]));
        }
        std::fs::write(&p, text)?;
        out.push(p);
        // rates
        let p = sib("-rates.csv");
        let mut text = csv_line(&["row", "body_id", "type", "instance", "group", "spikes", "rate_hz"].map(String::from));
        for i in 0..rows.len() {
            text.push_str(&csv_line(&[
                rows[i].to_string(),
                body[i].to_string(),
                types[i].clone(),
                inst[i].clone(),
                self.group_of[i].clone(),
                counts[i].to_string(),
                format!("{:.3}", counts[i] as f64 / (n_steps as f64 * 0.005)),
            ]));
        }
        std::fs::write(&p, text)?;
        out.push(p);
        // group rates in 50 ms bins
        let n_bins = ((n_steps as f64) / BIN_STEPS as f64).ceil() as usize;
        let mut binned: Vec<(String, Vec<f64>)> = Vec::new();
        for (gname, grows) in &self.groups {
            let member: std::collections::HashSet<i32> = grows.iter().map(|&r| self.lookup[r as usize]).collect();
            let mut c = vec![0i64; n_bins];
            for (k, &i) in self.idx.iter().enumerate() {
                if member.contains(&i) {
                    let b = (self.steps[k] as i64 / BIN_STEPS) as usize;
                    if b < n_bins {
                        c[b] += 1;
                    }
                }
            }
            binned.push((gname.clone(), c.iter().map(|&x| x as f64 / grows.len() as f64 / (BIN_STEPS as f64 * 0.005)).collect()));
        }
        let p = sib("-group-rates.csv");
        let mut head = vec!["time_ms".to_string()];
        head.extend(binned.iter().map(|b| b.0.clone()));
        let mut text = csv_line(&head);
        for b in 0..n_bins {
            let mut row = vec![format!("{:.0}", (b as i64 * BIN_STEPS) as f64 * 5.0)];
            row.extend(binned.iter().map(|g| format!("{:.3}", g.1[b])));
            text.push_str(&csv_line(&row));
        }
        std::fs::write(&p, text)?;
        out.push(p);
        if !self.events.is_empty() {
            let p = sib("-events.csv");
            let mut text = csv_line(&["time_ms", "kind", "label", "detail", "value"].map(String::from));
            for e in &self.events {
                text.push_str(&csv_line(&[
                    format!("{:.1}", (e.step - self.start_step as i64) as f64 * 5.0),
                    e.kind.clone(),
                    e.label.clone(),
                    e.detail.clone(),
                    format!("{:.4}", e.value),
                ]));
            }
            std::fs::write(&p, text)?;
            out.push(p);
        }
        if !self.kinematics.is_empty() {
            let p = sib("-kinematics.csv");
            let mut text = csv_line(&["time_ms", "x", "y", "z", "speed", "health", "action", "arena"].map(String::from));
            for k in &self.kinematics {
                text.push_str(&csv_line(&[
                    format!("{:.1}", (k.step - self.start_step as i64) as f64 * 5.0),
                    format!("{:.3}", k.x),
                    format!("{:.3}", k.y),
                    format!("{:.3}", k.z),
                    format!("{:.3}", k.speed),
                    format!("{:.1}", k.health),
                    k.action.clone(),
                    k.arena.clone(),
                ]));
            }
            std::fs::write(&p, text)?;
            out.push(p);
        }
        let p = sib(".npz");
        let spike_rows: Vec<i64> = self.idx.iter().map(|&i| i64::from(rows[i as usize])).collect();
        let group_rates: Vec<f64> = binned.iter().flat_map(|g| g.1.iter().copied()).collect();
        let gshape = if binned.is_empty() { vec![0, 0] } else { vec![binned.len(), n_bins] };
        npz(
            &p,
            &[
                ("spike_steps", arr::<i32>(vec![self.steps.len()], Data::I32(self.steps.clone()))),
                ("spike_rows", arr::<i64>(vec![spike_rows.len()], Data::I64(spike_rows))),
                ("rows", arr::<i64>(vec![rows.len()], Data::I64(rows.iter().map(|&r| i64::from(r)).collect()))),
                ("body_ids", arr::<i64>(vec![body.len()], Data::I64(body.clone()))),
                ("types", arr::<String>(vec![types.len()], Data::Str(types.clone()))),
                ("instances", arr::<String>(vec![inst.len()], Data::Str(inst.clone()))),
                ("groups", arr::<String>(vec![self.group_of.len()], Data::Str(self.group_of.clone()))),
                ("spike_counts", arr::<i64>(vec![counts.len()], Data::I64(counts.clone()))),
                ("dt_ms", arr::<f64>(vec![], Data::F64(vec![5.0]))),
                ("n_steps", arr::<i64>(vec![], Data::I64(vec![n_steps]))),
                ("group_names", arr::<String>(vec![binned.len()], Data::Str(binned.iter().map(|b| b.0.clone()).collect()))),
                ("group_rates", arr::<f64>(gshape, Data::F64(group_rates))),
            ],
        )?;
        out.push(p);
        let mut meta = meta;
        meta["recording"] = json!({
            "start_brain_step": self.start_step, "steps": n_steps, "seconds": n_steps as f64 * 0.005, "neurons": rows.len(),
            "spikes": self.idx.len(), "groups": self.groups.iter().map(|(g, r)| (g.clone(), json!(r.len()))).collect::<serde_json::Map<_, _>>(),
            "events": self.events.len(), "kinematics_samples": self.kinematics.len(),
            "files": out.iter().filter_map(|p| p.file_name().and_then(|f| f.to_str())).collect::<Vec<_>>(),
        });
        let p = sib("-metadata.json");
        std::fs::write(&p, serde_json::to_string_pretty(&meta).unwrap_or_default())?;
        out.push(p);
        Ok(out)
    }
}

/// `recorder.metadata` without the game: everything needed to know how a recording or result was made.
pub fn base_metadata(app_version: &str, created: &str, brain_pack: Value, extra: Value) -> Value {
    let mut m = json!({
        "app": "Kick the Fly", "app_version": app_version, "created": created,
        "platform": format!("{} {}", std::env::consts::OS, std::env::consts::ARCH), "connectome": CONNECTOME, "brain_pack": brain_pack, "dt_ms": 5.0,
    });
    if let (Some(o), Some(e)) = (m.as_object_mut(), extra.as_object()) {
        for (k, v) in e {
            o.insert(k.clone(), v.clone());
        }
    }
    m
}
