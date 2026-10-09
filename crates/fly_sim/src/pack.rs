//! The brain pack: the connectome as the Python game ships it (`kickthefly/sim/brainpack.py`, `data/kick_brain.npz`).
//!
//! The file holds the signed synapse counts as CSR `[post, pre]` (int16), each neuron's `1 / total input synapses`, and the labels
//! the game reads (type, superclass, instance, subclass, region, transmitter, soma position, FlyEM body id, the dopamine → mushroom
//! body output matrix). Loading rate-normalises exactly like `brainpack.load`: `W[post, pre] = count * inv[post]` in `f32`.

use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use crate::csr::{Csr, Weights};
use crate::npy::Npz;
use crate::{Error, util};

/// Neuron labels and the extra tables of the pack (everything except the weights).
#[derive(Debug, Clone, Default)]
pub struct Labels {
    pub types: Vec<String>,
    pub superclass: Vec<String>,
    pub instance: Vec<String>,
    pub subclass: Vec<String>,
    pub region: Vec<String>,
    pub nt: Vec<String>,
    pub nt_source: Vec<String>,
    pub nt_conf: Vec<f32>,
    /// Cell-body position per neuron, NaN where the dataset has none.
    pub soma: Vec<[f32; 3]>,
    pub body_id: Vec<i64>,
    /// Dopamine neuron rows (PAM, PPL1) and mushroom body output neuron rows, and the `[dan, mbon]` synapse counts.
    pub dan: Vec<u32>,
    pub mbon: Vec<u32>,
    pub dan_mbon: Vec<i16>,
    pub brain_type: String,
}

/// A loaded pack.
#[derive(Debug, Clone)]
pub struct Pack {
    pub n: usize,
    pub weights: Weights,
    pub labels: Labels,
    pub path: Option<PathBuf>,
}

/// Where the pack is looked for: `$KICK_THE_FLY_PACK`, then next to the program (`kick_brain.npz`, `data/kick_brain.npz`, and the AppImage's
/// `../share/kick-the-fly/kick_brain.npz`), then `data/kick_brain.npz` in the current directory and its parents.
pub fn find_default() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("KICK_THE_FLY_PACK") {
        let p = PathBuf::from(p);
        return p.exists().then_some(p);
    }
    if let Some(dir) = std::env::current_exe().ok().and_then(|e| e.parent().map(Path::to_path_buf)) {
        for cand in [dir.join("kick_brain.npz"), dir.join("data").join("kick_brain.npz"), dir.join("../share/kick-the-fly/kick_brain.npz")]
        {
            if cand.exists() {
                return Some(cand);
            }
        }
    }
    let mut dir = std::env::current_dir().ok()?;
    loop {
        let cand = dir.join("data").join("kick_brain.npz");
        if cand.exists() {
            return Some(cand);
        }
        if !dir.pop() {
            return None;
        }
    }
}

impl Pack {
    /// Load the pack with all labels.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Error> {
        Self::load_inner(path.as_ref(), true, Weights::default_blocks())
    }

    /// Load only the weights (skips the ~100 MB of string tables); what the benchmark and headless sweeps need.
    pub fn load_weights_only(path: impl AsRef<Path>) -> Result<Self, Error> {
        Self::load_inner(path.as_ref(), false, Weights::default_blocks())
    }

    /// Load a pack held in memory (the Windows build embeds it, so the program is one file).
    pub fn load_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let mut npz = Npz::read(std::io::Cursor::new(bytes))?;
        Self::from_npz(&mut npz, None, true, Weights::default_blocks())
    }

    fn load_inner(path: &Path, with_labels: bool, blocks: usize) -> Result<Self, Error> {
        let mut npz = Npz::read(BufReader::new(File::open(path)?))?;
        Self::from_npz(&mut npz, Some(path.to_path_buf()), with_labels, blocks)
    }

    fn from_npz(npz: &mut Npz, path: Option<PathBuf>, with_labels: bool, blocks: usize) -> Result<Self, Error> {
        let indptr = npz.take("indptr")?.into_i32()?;
        let indices = npz.take("indices")?.into_i32()?;
        let counts = npz.take("data")?.into_i16()?;
        let inv = npz.take("inv")?.into_f32()?;
        let n = inv.len();
        if indptr.len() != n + 1 || indices.len() != counts.len() || indptr[n] as usize != indices.len() {
            return Err(Error::Format("indptr / indices / data lengths disagree".into()));
        }
        let mut data = Vec::with_capacity(counts.len());
        for i in 0..n {
            let scale = inv[i];
            for &c in &counts[indptr[i] as usize..indptr[i + 1] as usize] {
                data.push(f32::from(c) * scale);
            }
        }
        let csr = Csr {
            n_rows: n,
            n_cols: n,
            indptr: indptr.into_iter().map(|v| v as u32).collect(),
            indices: indices.into_iter().map(|v| v as u32).collect(),
            data,
        };
        let weights = Weights::new(csr, blocks);
        let labels = if with_labels { read_labels(npz, n)? } else { Labels::default() };
        Ok(Self { n, weights, labels, path })
    }

    /// SHA-256 of the pack file (replays and bundles record it).
    pub fn file_sha256(path: impl AsRef<Path>) -> Result<String, Error> {
        Ok(util::sha256_hex(&std::fs::read(path)?))
    }

    pub fn nnz(&self) -> usize {
        self.weights.csr.nnz()
    }
}

fn read_labels(npz: &mut Npz, n: usize) -> Result<Labels, Error> {
    let strings = |npz: &mut Npz, name: &str| -> Result<Vec<String>, Error> { npz.take(name)?.into_strings() };
    let optional_strings = |npz: &mut Npz, name: &str| -> Result<Vec<String>, Error> {
        match npz.take_opt(name) {
            Some(a) => a.into_strings(),
            None => Ok(vec![String::new(); n]),
        }
    };
    let types = strings(npz, "type")?;
    let superclass = strings(npz, "superclass")?;
    let instance = strings(npz, "instance")?;
    if types.len() != n || superclass.len() != n || instance.len() != n {
        return Err(Error::Format("label arrays do not match the neuron count".into()));
    }
    let soma = match npz.take_opt("soma") {
        Some(a) => a.into_f32()?.as_chunks::<3>().0.to_vec(),
        None => vec![[f32::NAN; 3]; n],
    };
    let dan_mbon = match npz.take_opt("dan_mbon") {
        Some(a) => a.into_i16()?,
        None => Vec::new(),
    };
    let dan = match npz.take_opt("dan") {
        Some(a) => a.into_i32()?.into_iter().map(|v| v as u32).collect(),
        None => Vec::new(),
    };
    let mbon = match npz.take_opt("mbon") {
        Some(a) => a.into_i32()?.into_iter().map(|v| v as u32).collect(),
        None => Vec::new(),
    };
    let brain_type = match npz.take_opt("brain_type") {
        Some(a) => a.into_strings()?.into_iter().next().unwrap_or_else(|| "adult".into()),
        None => "adult".into(),
    };
    Ok(Labels {
        subclass: optional_strings(npz, "subclass")?,
        region: optional_strings(npz, "region")?,
        nt: optional_strings(npz, "nt")?,
        nt_source: optional_strings(npz, "nt_source")?,
        nt_conf: match npz.take_opt("nt_conf") {
            Some(a) => a.into_f32()?,
            None => vec![f32::NAN; n],
        },
        body_id: match npz.take_opt("body_id") {
            Some(a) => a.into_i64()?,
            None => Vec::new(),
        },
        types,
        superclass,
        instance,
        soma,
        dan,
        mbon,
        dan_mbon,
        brain_type,
    })
}
