//! Save states: `.ktfsave` is a zip with `meta.json` and `arrays.npz` (little-endian NumPy arrays), format `kick-the-fly-save` version 1.
//! Python: `core/savestate.py`. A save made by the Python game loads here and the other way round for the parts both games have:
//! the simulator (membrane potentials, refractory counters, last spikes, gain, the exact random-generator state, activity), the brain
//! bookkeeping (rate trackers, history, stimuli still being delivered, death, pain), the mushroom body, and the header (arena, tool,
//! kills, Lab parameters, surgery). Bodies, timers and the rest of `flies[i].fly/slot` are kept as JSON and arrays as they are
//! (`extra`), so a file read and written again loses nothing.

use std::collections::BTreeMap;
use std::io::{Cursor, Read, Write};
use std::path::Path;

use fly_sim::lif::SimState;
use fly_sim::npy::{Array, Data, Npz};
use fly_sim::rng::PcgState;
use serde_json::Value;

pub const FORMAT: &str = "kick-the-fly-save";
pub const FORMAT_VERSION: i64 = 1;
pub const SUFFIX: &str = ".ktfsave";

#[derive(Debug)]
pub struct SaveError(pub String);
impl std::fmt::Display for SaveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for SaveError {}
fn err<T>(s: impl Into<String>) -> Result<T, SaveError> {
    Err(SaveError(s.into()))
}

/// A save file in memory: the header and all arrays.
#[derive(Debug, Clone)]
pub struct SaveFile {
    pub meta: Value,
    pub arrays: BTreeMap<String, Array>,
}

/// What this game has loaded, for the compatibility check ("pack_signature" in Python).
#[derive(Debug, Clone, PartialEq)]
pub struct PackInfo {
    pub n_neurons: usize,
    pub synapses: usize,
    pub brain: String,
    /// "3d" or "2d"
    pub mode: String,
}

/// Read only the header (fast; used by the Load menu to list saves).
pub fn read_meta(path: &Path) -> Result<Value, SaveError> {
    let unreadable = |e: &dyn std::fmt::Display| SaveError(format!("not a readable save file ({e})"));
    let f = std::fs::File::open(path).map_err(|e| unreadable(&e))?;
    let mut zip = zip::ZipArchive::new(f).map_err(|e| unreadable(&e))?;
    let mut text = String::new();
    zip.by_name("meta.json").map_err(|e| unreadable(&e))?.read_to_string(&mut text).map_err(|e| unreadable(&e))?;
    let meta: Value = serde_json::from_str(&quote_big_numbers(&text)).map_err(|e| unreadable(&e))?;
    if meta.get("format").and_then(|f| f.as_str()) != Some(FORMAT) {
        return err("not a Kick the Fly save file");
    }
    Ok(meta)
}

impl SaveFile {
    pub fn read(path: &Path) -> Result<Self, SaveError> {
        let meta = read_meta(path)?;
        let f = std::fs::File::open(path).map_err(|e| SaveError(e.to_string()))?;
        let mut zip = zip::ZipArchive::new(f).map_err(|e| SaveError(e.to_string()))?;
        let mut bytes = Vec::new();
        zip.by_name("arrays.npz")
            .map_err(|e| SaveError(format!("not a readable save file ({e})")))?
            .read_to_end(&mut bytes)
            .map_err(|e| SaveError(e.to_string()))?;
        let npz = Npz::read(Cursor::new(bytes)).map_err(|e| SaveError(format!("the save's arrays are damaged ({e})")))?;
        let names: Vec<String> = npz.names().cloned().collect();
        let arrays = names.into_iter().map(|n| (n.clone(), npz.get(&n).expect("listed").clone())).collect();
        Ok(Self { meta, arrays })
    }

    /// Write atomically (temp file, then rename).
    pub fn write(&self, path: &Path) -> Result<(), SaveError> {
        let io = |e: std::io::Error| SaveError(e.to_string());
        if let Some(d) = path.parent() {
            std::fs::create_dir_all(d).map_err(io)?;
        }
        let tmp = path.with_file_name(format!("{}.tmp", path.file_name().unwrap_or_default().to_string_lossy()));
        let mut inner = zip::ZipWriter::new(Cursor::new(Vec::new()));
        let opts = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
        for (name, a) in &self.arrays {
            inner.start_file(format!("{name}.npy"), opts).map_err(|e| SaveError(e.to_string()))?;
            inner.write_all(&to_npy(a)?).map_err(io)?;
        }
        let npz = inner.finish().map_err(|e| SaveError(e.to_string()))?.into_inner();
        let f = std::fs::File::create(&tmp).map_err(io)?;
        let mut zw = zip::ZipWriter::new(f);
        zw.start_file("meta.json", opts).map_err(|e| SaveError(e.to_string()))?;
        let text = serde_json::to_string(&self.meta).map_err(|e| SaveError(e.to_string()))?;
        zw.write_all(unquote_big_numbers(&text).as_bytes()).map_err(io)?;
        zw.start_file("arrays.npz", opts).map_err(|e| SaveError(e.to_string()))?;
        zw.write_all(&npz).map_err(io)?;
        zw.finish().map_err(|e| SaveError(e.to_string()))?;
        std::fs::rename(&tmp, path).map_err(io)
    }

    fn arr(&self, name: &str) -> Result<&Array, SaveError> {
        self.arrays.get(name).ok_or_else(|| SaveError(format!("the save lacks '{name}'")))
    }

    pub fn flies(&self) -> usize {
        self.meta["flies"].as_array().map_or(0, |f| f.len())
    }

    /// The simulator part of fly `i` (`f{i}_b_*` arrays and `flies[i].brain.sim`).
    pub fn sim_state(&self, i: usize) -> Result<SimState, SaveError> {
        let b = &self.meta["flies"][i]["brain"];
        let sim = &b["sim"];
        let p = format!("f{i}_b_");
        let v = self.arr(&format!("{p}v"))?.clone().into_f32().map_err(conv)?;
        let refr = self.arr(&format!("{p}refr"))?.clone().into_i16().map_err(conv)?;
        let mask = self.arr(&format!("{p}spikes"))?.clone().into_bool().map_err(conv)?;
        let spikes = mask.iter().enumerate().filter(|(_, s)| **s).map(|(i, _)| i as u32).collect();
        Ok(SimState {
            seed: b["seed"].as_u64().ok_or_else(|| SaveError("the save lacks the brain seed".into()))?,
            v,
            refr,
            spikes,
            gain: sim["gain"].as_f64().ok_or_else(|| SaveError("the save lacks the gain".into()))?,
            target_p: sim["target_p"].as_f64().ok_or_else(|| SaveError("the save lacks target_p".into()))?,
            rng: pcg_state(&sim["rng"])?,
            activity_rate: self.arr(&format!("{p}act_rate"))?.clone().into_f32().map_err(conv)?,
            activity_last: self.arr(&format!("{p}act_last"))?.clone().into_i32().map_err(conv)?,
            activity_steps: sim["act_steps"].as_u64().unwrap_or(0),
            activity_count: sim["act_count"].as_u64().unwrap_or(0) as usize,
        })
    }

    /// Store a simulator state as fly `i` of a new save (Rust-made saves carry the simulator and the header; the body belongs to the game).
    pub fn put_sim_state(&mut self, i: usize, st: &SimState) {
        let p = format!("f{i}_b_");
        let n = st.v.len();
        let mut mask = vec![false; n];
        for &s in &st.spikes {
            mask[s as usize] = true;
        }
        let ins = |a: &mut BTreeMap<String, Array>, k: String, d: Data| {
            let len = match &d {
                Data::F32(v) => v.len(),
                Data::I16(v) => v.len(),
                Data::I32(v) => v.len(),
                Data::Bool(v) => v.len(),
                _ => 0,
            };
            a.insert(k, Array { shape: vec![len], data: d });
        };
        ins(&mut self.arrays, format!("{p}v"), Data::F32(st.v.clone()));
        ins(&mut self.arrays, format!("{p}refr"), Data::I16(st.refr.clone()));
        ins(&mut self.arrays, format!("{p}spikes"), Data::Bool(mask));
        ins(&mut self.arrays, format!("{p}act_rate"), Data::F32(st.activity_rate.clone()));
        ins(&mut self.arrays, format!("{p}act_last"), Data::I32(st.activity_last.clone()));
        let flies = self.meta["flies"].as_array_mut().expect("flies list");
        while flies.len() <= i {
            flies.push(serde_json::json!({"seed": st.seed, "primary": flies.is_empty(), "brain": {}, "fly": {}, "slot": {}}));
        }
        flies[i]["brain"]["seed"] = st.seed.into();
        flies[i]["brain"]["sim"] = serde_json::json!({
            "gain": st.gain, "target_p": st.target_p, "rng": pcg_json(&st.rng),
            "act_steps": st.activity_steps, "act_count": st.activity_count,
        });
    }

    /// A header for a save made here (the fields Python's `compatible` reads, plus what the Load menu shows).
    pub fn new_header(app_version: &str, mode: &str, brain: &str, pack: &PackInfo, arena: &str, arena_i: usize) -> Value {
        serde_json::json!({
            "format": FORMAT, "format_version": FORMAT_VERSION, "app_version": app_version,
            "created": "", "platform": std::env::consts::OS, "mode": mode, "seed": 0, "brain": brain, "individuality": "subtle",
            "backend": "rust", "device": "CPU (rayon)",
            "signature": {"n_neurons": pack.n_neurons, "synapses": pack.synapses, "memory_signature": null},
            "arena_i": arena_i, "arena": arena, "tool": 0, "focus": 0, "kills": 0, "immortal": false, "pain_level": 0,
            "sim_speed": 1.0, "lab_params": {}, "surgery_modes": [], "wiring": null, "type_ops": {}, "flies": [], "extra": {},
        })
    }
}

fn conv(e: fly_sim::Error) -> SaveError {
    SaveError(format!("the save's arrays are damaged ({e})"))
}

/// numpy's `bit_generator.state` dict → [`PcgState`]. The 128-bit integers are JSON numbers; `serde_json` needs `arbitrary_precision`
/// for those, so the file text is rewritten into strings first (see [`quote_big_numbers`]).
fn pcg_state(v: &Value) -> Result<PcgState, SaveError> {
    let big = |x: &Value| -> Result<u128, SaveError> {
        match x {
            Value::String(s) => s.parse().map_err(|_| SaveError("bad random-generator state".into())),
            Value::Number(n) => n.as_u64().map(u128::from).ok_or_else(|| SaveError("bad random-generator state".into())),
            _ => err("bad random-generator state"),
        }
    };
    if v["bit_generator"] != "PCG64" {
        return err("the save uses a random generator this version doesn't have");
    }
    Ok(PcgState {
        state: big(&v["state"]["state"])?,
        inc: big(&v["state"]["inc"])?,
        has_uint32: v["has_uint32"].as_i64().unwrap_or(0) != 0,
        uinteger: v["uinteger"].as_u64().unwrap_or(0) as u32,
    })
}

fn pcg_json(s: &PcgState) -> Value {
    serde_json::json!({
        "bit_generator": "PCG64",
        "state": {"state": s.state.to_string(), "inc": s.inc.to_string()},
        "has_uint32": s.has_uint32 as i32, "uinteger": s.uinteger,
    })
}

/// `None` if this save can load into a game with `mine`, else a plain-language reason (Python: `compatible`).
pub fn compatible(meta: &Value, mine: &PackInfo) -> Option<String> {
    let v = meta.get("format_version").and_then(|v| v.as_i64());
    match v {
        Some(v) if v > FORMAT_VERSION => {
            return Some(format!(
                "made by a newer version ({}); update the game to load it",
                meta.get("app_version").and_then(|a| a.as_str()).unwrap_or("?")
            ));
        }
        Some(v) if v < 1 => return Some("unsupported save format".into()),
        None => {
            return Some(format!(
                "made by a newer version ({}); update the game to load it",
                meta.get("app_version").and_then(|a| a.as_str()).unwrap_or("?")
            ));
        }
        _ => {}
    }
    let mode = meta.get("mode").and_then(|m| m.as_str()).unwrap_or("?");
    if mode != mine.mode {
        return Some(format!("made in the {} game", mode.to_uppercase()));
    }
    let brain = meta.get("brain").and_then(|b| b.as_str()).unwrap_or("adult");
    if brain != mine.brain {
        return Some(format!("made for the {brain} brain (currently using {})", mine.brain));
    }
    let sig = &meta["signature"];
    if sig["n_neurons"].as_u64() != Some(mine.n_neurons as u64) || sig["synapses"].as_u64() != Some(mine.synapses as u64) {
        return Some("made with a different brain pack".into());
    }
    None
}

pub(crate) fn to_npy(a: &Array) -> Result<Vec<u8>, SaveError> {
    let (descr, body): (&str, Vec<u8>) = match &a.data {
        Data::Bool(v) => ("|b1", v.iter().map(|&b| b as u8).collect()),
        Data::I16(v) => ("<i2", v.iter().flat_map(|x| x.to_le_bytes()).collect()),
        Data::I32(v) => ("<i4", v.iter().flat_map(|x| x.to_le_bytes()).collect()),
        Data::I64(v) => ("<i8", v.iter().flat_map(|x| x.to_le_bytes()).collect()),
        Data::F32(v) => ("<f4", v.iter().flat_map(|x| x.to_le_bytes()).collect()),
        Data::F64(v) => ("<f8", v.iter().flat_map(|x| x.to_le_bytes()).collect()),
        Data::Str(v) => {
            // `<U<n>`: UCS-4, NUL padded to the longest string
            let w = v.iter().map(|s| s.chars().count()).max().unwrap_or(1).max(1);
            let mut body = Vec::with_capacity(v.len() * w * 4);
            for s in v {
                let mut n = 0;
                for c in s.chars() {
                    body.extend((c as u32).to_le_bytes());
                    n += 1;
                }
                body.resize(body.len() + (w - n) * 4, 0);
            }
            return Ok(npy_bytes(&format!("<U{w}"), &a.shape, body));
        }
    };
    Ok(npy_bytes(descr, &a.shape, body))
}

fn npy_bytes(descr: &str, shape_dims: &[usize], body: Vec<u8>) -> Vec<u8> {
    let shape = match shape_dims.len() {
        0 => "()".to_string(),
        1 => format!("({},)", shape_dims[0]),
        _ => format!("({})", shape_dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")),
    };
    let mut header = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}");
    // version 1.0: magic(6) + version(2) + len(2) + header, padded with spaces to a multiple of 64 and ended by a newline
    let total = 10 + header.len() + 1;
    header.push_str(&" ".repeat((64 - total % 64) % 64));
    header.push('\n');
    let mut out = b"\x93NUMPY\x01\x00".to_vec();
    out.extend((header.len() as u16).to_le_bytes());
    out.extend(header.as_bytes());
    out.extend(body);
    out
}

/// numpy's generator state holds integers wider than 64 bits; JSON parsers without big numbers lose them. Wrap any integer with more than
/// 15 digits that is not already inside a string into a string, outside strings only, so `serde_json` keeps every digit.
pub fn quote_big_numbers(text: &str) -> String {
    let b = text.as_bytes();
    let mut out = Vec::<u8>::with_capacity(text.len() + 16);
    let (mut i, mut in_str) = (0, false);
    while i < b.len() {
        let c = b[i];
        if in_str {
            out.push(c);
            if c == b'\\' && i + 1 < b.len() {
                i += 1;
                out.push(b[i]);
            } else if c == b'"' {
                in_str = false;
            }
            i += 1;
        } else if c == b'"' {
            in_str = true;
            out.push(b'"');
            i += 1;
        } else if c.is_ascii_digit() {
            let s = i;
            while i < b.len() && (b[i].is_ascii_digit() || matches!(b[i], b'.' | b'e' | b'E' | b'+' | b'-')) {
                i += 1;
            }
            let tok = &text[s..i];
            if tok.len() > 15 && tok.bytes().all(|d| d.is_ascii_digit()) {
                out.push(b'"');
                out.extend_from_slice(tok.as_bytes());
                out.push(b'"');
            } else {
                out.extend_from_slice(tok.as_bytes());
            }
        } else {
            out.push(c);
            i += 1;
        }
    }
    String::from_utf8(out).expect("only whole UTF-8 sequences are copied")
}

/// The reverse of [`quote_big_numbers`], for writing: a JSON string of 16 or more digits becomes a number again, so NumPy reads its
/// generator state back as the integers it expects.
pub fn unquote_big_numbers(text: &str) -> String {
    let b = text.as_bytes();
    let mut out = Vec::<u8>::with_capacity(text.len());
    let (mut i, mut in_str) = (0, false);
    while i < b.len() {
        if b[i] == b'"' && !in_str {
            // a whole string made of digits?
            let mut j = i + 1;
            while j < b.len() && b[j].is_ascii_digit() {
                j += 1;
            }
            if j < b.len() && b[j] == b'"' && j - i - 1 > 15 {
                out.extend_from_slice(&b[i + 1..j]);
                i = j + 1;
                continue;
            }
            in_str = true;
        } else if in_str {
            if b[i] == b'\\' {
                out.push(b[i]);
                i += 1;
            } else if b[i] == b'"' {
                in_str = false;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8(out).expect("only whole UTF-8 sequences are copied")
}
