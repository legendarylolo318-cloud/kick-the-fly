//! The small files next to the saves: the Neurodex (`neurodex.json`), contraption slots (`contraptions.json`), saved lesion loadouts
//! (`lesion_loadouts.json`) and the fly's learned memory (`fly-memory.npz` + `training-log.json`). Python: `core/neurodex.py`,
//! `game/contraption.py`, `game/lesion_battle.py`, `core/memory.py`. Every reader is as forgiving as Python's: a damaged file or entry
//! is ignored (the Neurodex keeps the unreadable file as `.bad`), a file from a newer version is left alone.

use std::collections::BTreeMap;
use std::io::Cursor;
use std::path::{Path, PathBuf};

use fly_sim::npy::{Array, Data, Npz};
use serde_json::{Map, Value};

use crate::savestate::to_npy;

fn write_json_atomic(path: &Path, v: &Value, indent_one: bool, tmp_suffix: &str) -> std::io::Result<()> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    let tmp = path.with_file_name(format!("{}{tmp_suffix}", path.file_name().unwrap_or_default().to_string_lossy()));
    let text = if indent_one {
        // Python's indent=1
        use serde::Serialize;
        let mut buf = Vec::new();
        let fmt = serde_json::ser::PrettyFormatter::with_indent(b" ");
        v.serialize(&mut serde_json::Serializer::with_formatter(&mut buf, fmt)).map_err(std::io::Error::other)?;
        String::from_utf8(buf).map_err(std::io::Error::other)?
    } else {
        serde_json::to_string(v).map_err(std::io::Error::other)?
    };
    std::fs::write(&tmp, text)?;
    std::fs::rename(&tmp, path)
}

// --- Neurodex ---------------------------------------------------------------------------------------------------------------------

pub const NEURODEX_FILE: &str = "neurodex.json";
pub const NEURODEX_VERSION: i64 = 1;
const HOW: [&str; 3] = ["play", "stimulated", "rest"];

/// Which cell types this player has discovered, per brain (`adult`, `larva`).
#[derive(Debug, Clone)]
pub struct Progress {
    pub path: PathBuf,
    /// brain → type → (when, ratio, how)
    pub brains: BTreeMap<String, BTreeMap<String, Discovery>>,
    pub warnings: Vec<String>,
    pub read_only: bool,
    pub dirty: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Discovery {
    pub t: String,
    pub x: f64,
    pub how: String,
}

impl Progress {
    /// The Neurodex lives in the memory folder (`ktf_data::paths::get().memory_dir`).
    pub fn open(path: PathBuf) -> Self {
        let mut p = Progress { path, brains: BTreeMap::new(), warnings: vec![], read_only: false, dirty: false };
        p.load();
        p
    }

    fn load(&mut self) {
        if !self.path.exists() {
            return;
        }
        let name = self.path.file_name().unwrap_or_default().to_string_lossy().to_string();
        let raw = std::fs::read_to_string(&self.path)
            .map_err(|e| e.to_string())
            .and_then(|t| serde_json::from_str::<Value>(&t).map_err(|e| e.to_string()))
            .and_then(|v| {
                if v.is_object() && v.get("brains").is_none_or(|b| b.is_object()) { Ok(v) } else { Err("unexpected layout".to_string()) }
            });
        let raw = match raw {
            Ok(v) => v,
            Err(e) => {
                self.warnings.push(format!("{name} could not be read ({e}); starting an empty Neurodex"));
                if std::fs::rename(&self.path, self.path.with_file_name(format!("{name}.bad"))).is_ok() {
                    self.warnings.push(format!("the unreadable file was kept as {name}.bad"));
                }
                return;
            }
        };
        match raw.get("version").map_or(Some(1), |v| v.as_i64()) {
            Some(v) if v <= NEURODEX_VERSION => {}
            _ => {
                self.read_only = true;
                self.warnings.push(format!("{name} is from a newer version; this one won't change it"));
                return;
            }
        }
        for (brain, rec) in raw.get("brains").and_then(|b| b.as_object()).into_iter().flatten() {
            let Some(types) = rec.get("types").and_then(|t| t.as_object()) else {
                continue;
            };
            let mut kept = BTreeMap::new();
            for (t, v) in types {
                let Some(v) = v.as_object() else { continue };
                let x = v.get("x").and_then(|x| x.as_f64()).filter(|x| x.is_finite()).unwrap_or(0.0);
                let how = v.get("how").and_then(|h| h.as_str()).filter(|h| HOW.contains(h)).unwrap_or("play");
                kept.insert(t.clone(), Discovery { t: v.get("t").and_then(|t| t.as_str()).unwrap_or("").into(), x, how: how.into() });
            }
            self.brains.insert(brain.clone(), kept);
        }
    }

    pub fn types(&mut self, brain: &str) -> &mut BTreeMap<String, Discovery> {
        self.brains.entry(brain.into()).or_default()
    }
    pub fn discovered(&mut self, brain: &str, name: &str) -> bool {
        self.types(brain).contains_key(name)
    }
    pub fn n_discovered(&mut self, brain: &str) -> usize {
        self.types(brain).len()
    }

    /// Record a discovery. False if it was already known (the first record stays).
    pub fn mark(&mut self, brain: &str, name: &str, ratio: f64, how: &str, when: &str) -> bool {
        let t = self.types(brain);
        if t.contains_key(name) {
            return false;
        }
        t.insert(name.into(), Discovery { t: when.into(), x: (ratio * 100.0).round() / 100.0, how: how.into() });
        self.dirty = true;
        true
    }

    pub fn wipe(&mut self, brain: Option<&str>) {
        match brain {
            Some(b) => {
                self.brains.remove(b);
            }
            None => self.brains.clear(),
        }
        self.dirty = true;
    }

    pub fn save(&mut self) -> bool {
        if self.read_only {
            return false;
        }
        let mut brains = Map::new();
        for (b, types) in &self.brains {
            let mut t = Map::new();
            for (name, d) in types {
                t.insert(name.clone(), serde_json::json!({"t": d.t, "x": d.x, "how": d.how}));
            }
            brains.insert(b.clone(), serde_json::json!({ "types": t }));
        }
        let v = serde_json::json!({"version": NEURODEX_VERSION, "brains": brains});
        let ok = write_json_atomic(&self.path, &v, true, ".tmp").is_ok();
        self.dirty &= !ok;
        ok
    }
}

// --- contraption slots ------------------------------------------------------------------------------------------------------------

pub const CONTRAPTION_SLOTS: usize = 8;

/// The eight save slots (`contraptions.json`), each a build or None. Entries that are not objects are ignored; per-part checking of a
/// build is the contraption module's (`crate::sharecode::check_payload("contraption", ..)` does the container check).
pub fn load_contraption_slots(path: &Path) -> Vec<Option<Value>> {
    let mut out = vec![None; CONTRAPTION_SLOTS];
    let Some(Value::Array(raw)) = std::fs::read_to_string(path).ok().and_then(|t| serde_json::from_str(&t).ok()) else {
        return out;
    };
    for (i, d) in raw.into_iter().take(CONTRAPTION_SLOTS).enumerate() {
        if d.is_object() && crate::sharecode::check_payload("contraption", &d).is_none() {
            out[i] = Some(d);
        }
    }
    out
}

pub fn save_contraption_slot(path: &Path, i: usize, build: &Value) -> bool {
    if i >= CONTRAPTION_SLOTS || build.get("parts").and_then(|p| p.as_array()).is_none_or(|p| p.is_empty()) {
        return false;
    }
    let mut slots = load_contraption_slots(path);
    slots[i] = Some(build.clone());
    let v = Value::Array(slots.into_iter().map(|s| s.unwrap_or(Value::Null)).collect());
    write_json_atomic(path, &v, true, ".tmp").is_ok()
}

// --- saved lesion loadouts --------------------------------------------------------------------------------------------------------

pub const LESION_MAX_SAVED: usize = 5;

/// `lesion_loadouts.json`: name (at most 24 characters) → {cell type: -1 (silence) or 1 (stimulate)}.
pub fn load_lesion_loadouts(path: &Path) -> BTreeMap<String, BTreeMap<String, i32>> {
    let mut out = BTreeMap::new();
    let Some(Value::Object(raw)) = std::fs::read_to_string(path).ok().and_then(|t| serde_json::from_str(&t).ok()) else {
        return out;
    };
    for (name, picks) in raw {
        let Some(picks) = picks.as_object() else {
            continue;
        };
        let pm1 = |v: &Value| v.as_f64().filter(|x| *x == 1.0 || *x == -1.0).map(|x| x as i32);
        if picks.values().all(|v| pm1(v).is_some()) {
            out.insert(name.chars().take(24).collect(), picks.iter().map(|(t, v)| (t.clone(), pm1(v).unwrap())).collect());
        }
    }
    out
}

fn lesion_value(m: &BTreeMap<String, BTreeMap<String, i32>>) -> Value {
    serde_json::to_value(m).expect("plain maps serialize")
}

/// Save under `name` (overwriting that name); false if the folder is full of other names.
pub fn save_lesion_loadout(path: &Path, name: &str, picks: &BTreeMap<String, i32>) -> bool {
    let mut saved = load_lesion_loadouts(path);
    let name: String = name.trim().chars().take(24).collect();
    let name = if name.is_empty() { "my fly".to_string() } else { name };
    if !saved.contains_key(&name) && saved.len() >= LESION_MAX_SAVED {
        return false;
    }
    saved.insert(name, picks.clone());
    write_json_atomic(path, &lesion_value(&saved), true, ".tmp").is_ok()
}

pub fn delete_lesion_loadout(path: &Path, name: &str) {
    let mut saved = load_lesion_loadouts(path);
    if saved.remove(name).is_some() {
        let _ = write_json_atomic(path, &lesion_value(&saved), true, "");
    }
}

// --- the fly's memory -------------------------------------------------------------------------------------------------------------

pub const MEMORY_VERSION: i64 = 1;

/// What training taught the mushroom body (`core/memory.py`): the Kenyon-cell → MBON weights, per-smell templates, calm baselines.
#[derive(Debug, Clone, PartialEq)]
pub struct FlyMemory {
    pub signature: Vec<f64>,
    pub w: Vec<f32>,
    /// smell → Kenyon-cell pattern
    pub templates: BTreeMap<String, Vec<f32>>,
    pub kc_calm: Vec<f32>,
    pub dan_calm: Vec<f32>,
    /// from `training-log.json`
    pub naive_mbon: Value,
    pub trials: Value,
}

impl FlyMemory {
    /// `None` when the file is missing, unreadable, or made for a different brain pack (the signature differs): the caller starts fresh.
    pub fn load(npz_path: &Path, signature: &[f64]) -> Option<Self> {
        let bytes = std::fs::read(npz_path).ok()?;
        let mut z = Npz::read(Cursor::new(bytes)).ok()?;
        let version = z.take("version").ok()?.into_i64().ok()?.first().copied()?;
        let sig = match z.take("signature").ok()?.data {
            Data::F64(v) => v,
            _ => return None,
        };
        if version != MEMORY_VERSION
            || sig.len() != signature.len()
            || sig.iter().zip(signature).any(|(a, b)| (a - b).abs() > 1e-8 + 1e-5 * b.abs())
        {
            return None;
        }
        let w = z.take("w").ok()?.into_f32().ok()?;
        let names = z.take("names").ok()?.into_strings().ok()?;
        let t = z.take("templates").ok()?;
        let width = if names.is_empty() { 0 } else { t.len() / names.len() };
        let t = t.into_f32().ok()?;
        let templates = names.into_iter().enumerate().map(|(i, n)| (n, t[i * width..(i + 1) * width].to_vec())).collect();
        let kc_calm = z.take("kc_calm").ok()?.into_f32().ok()?;
        let dan_calm = z.take("dan_calm").ok()?.into_f32().ok()?;
        let log: Value = std::fs::read_to_string(npz_path.with_file_name("training-log.json"))
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or(Value::Null);
        Some(FlyMemory {
            signature: sig,
            w,
            templates,
            kc_calm,
            dan_calm,
            naive_mbon: log.get("naive_mbon").cloned().unwrap_or_else(|| serde_json::json!({})),
            trials: log.get("trials").cloned().unwrap_or_else(|| serde_json::json!({})),
        })
    }

    /// Write `fly-memory.npz` (as `np.savez_compressed` does) and `training-log.json` next to it.
    pub fn save(&self, npz_path: &Path) -> std::io::Result<()> {
        use std::io::Write;
        if let Some(d) = npz_path.parent() {
            std::fs::create_dir_all(d)?;
        }
        let width = self.templates.values().next().map_or(self.kc_calm.len(), |t| t.len());
        let names: Vec<String> = self.templates.keys().cloned().collect();
        let templates: Vec<f32> = self.templates.values().flatten().copied().collect();
        let arrays: Vec<(&str, Array)> = vec![
            ("version", Array { shape: vec![], data: Data::I64(vec![MEMORY_VERSION]) }),
            ("signature", Array { shape: vec![self.signature.len()], data: Data::F64(self.signature.clone()) }),
            ("w", Array { shape: vec![self.w.len()], data: Data::F32(self.w.clone()) }),
            ("names", Array { shape: vec![names.len()], data: Data::Str(names.clone()) }),
            ("templates", Array { shape: vec![names.len(), width], data: Data::F32(templates) }),
            ("kc_calm", Array { shape: vec![self.kc_calm.len()], data: Data::F32(self.kc_calm.clone()) }),
            ("dan_calm", Array { shape: vec![self.dan_calm.len()], data: Data::F32(self.dan_calm.clone()) }),
        ];
        let mut zw = zip::ZipWriter::new(Cursor::new(Vec::new()));
        let opts = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
        for (n, a) in &arrays {
            zw.start_file(format!("{n}.npy"), opts).map_err(std::io::Error::other)?;
            zw.write_all(&to_npy(a).map_err(|e| std::io::Error::other(e.0))?)?;
        }
        let bytes = zw.finish().map_err(std::io::Error::other)?.into_inner();
        let tmp = npz_path.with_extension("tmp.npz");
        std::fs::write(&tmp, bytes)?;
        std::fs::rename(&tmp, npz_path)?;
        let log = serde_json::json!({"naive_mbon": self.naive_mbon, "trials": self.trials});
        write_json_atomic(&npz_path.with_file_name("training-log.json"), &log, true, ".tmp")
    }
}
