//! Real neuron shapes (`sim/realshapes.py`, `sim/morphology.py`): SWC skeletons from the public MaleCNS v1.0 release, the cache folder with
//! its sha256 manifest, and the samples the Neurodex draws. No network here: fetching is the game's (`kick_the_fly::shapes_play`), which
//! hands a verified download to [`ShapeStore::put`].

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

pub const BASE_URL: &str = "https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-swc";
pub const CREDIT: &str = "Neuron shapes: male CNS connectome v1.0 (FlyEM, HHMI Janelia, with the University of Cambridge, the MRC LMB and Google Research), CC BY 4.0, male-cns.janelia.org";
pub const CREDIT_SHORT: &str = "Shape: MaleCNS v1.0, FlyEM / Janelia, CC BY 4.0";
pub const MAX_BYTES: usize = 5_000_000;
pub const MAX_NODES: usize = 400_000;
pub const MANIFEST: &str = "manifest.json";

/// A reconstructed skeleton in EM space (8 nm units, the space of the cell-body positions).
#[derive(Debug, Clone)]
pub struct Skeleton {
    pub body_id: i64,
    pub xyz: Vec<[f32; 3]>,
    pub radius: Vec<f32>,
    /// Index of the parent node or -1.
    pub parent: Vec<i32>,
}

impl Skeleton {
    pub fn n(&self) -> usize {
        self.xyz.len()
    }

    /// One line segment per node that has a parent.
    pub fn segments(&self) -> impl Iterator<Item = ([f32; 3], [f32; 3])> + '_ {
        self.parent.iter().enumerate().filter(|&(_, &p)| p >= 0).map(|(i, &p)| (self.xyz[i], self.xyz[p as usize]))
    }

    pub fn bounds(&self) -> ([f32; 3], [f32; 3]) {
        let mut lo = [f32::INFINITY; 3];
        let mut hi = [f32::NEG_INFINITY; 3];
        for p in &self.xyz {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        (lo, hi)
    }

    pub fn cable_length_um(&self) -> f64 {
        self.segments().map(|(a, b)| f64::from(((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt())).sum::<f64>()
            * 0.008
    }
}

/// The whole SWC (id, type, x, y, z, radius, parent per line).
pub fn parse_swc(text: &str, body_id: i64) -> Result<Skeleton, String> {
    let (mut ids, mut xyz, mut rad, mut par) = (vec![], vec![], vec![], vec![]);
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let p: Vec<&str> = line.split_whitespace().collect();
        if p.len() < 7 {
            continue;
        }
        let parsed = (|| -> Option<(i64, [f32; 3], f32, i64)> {
            Some((p[0].parse().ok()?, [p[2].parse().ok()?, p[3].parse().ok()?, p[4].parse().ok()?], p[5].parse().ok()?, p[6].parse().ok()?))
        })();
        let Some((id, c, r, pa)) = parsed else { continue };
        ids.push(id);
        xyz.push(c);
        rad.push(r);
        par.push(pa);
        if ids.len() > MAX_NODES {
            return Err(format!("more than {MAX_NODES} nodes"));
        }
    }
    if ids.len() < 2 {
        return Err("fewer than two nodes".into());
    }
    if xyz.iter().any(|c| c.iter().any(|v| !v.is_finite())) {
        return Err("non-finite coordinates".into());
    }
    let index: std::collections::HashMap<i64, i32> = ids.iter().enumerate().map(|(k, &i)| (i, k as i32)).collect();
    let parent = par.iter().map(|p| index.get(p).copied().unwrap_or(-1)).collect();
    Ok(Skeleton { body_id, xyz, radius: rad, parent })
}

/// `morphology.parse_swc(text, n)`: `n` evenly spaced node positions in file order (the Neurodex's small drawing).
pub fn sample_points(text: &str, n: usize) -> Option<Vec<[f32; 3]>> {
    let mut nodes: Vec<[f32; 3]> = vec![];
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let p: Vec<&str> = line.split_whitespace().collect();
        if p.len() >= 7
            && let (Ok(x), Ok(y), Ok(z)) = (p[2].parse::<f32>(), p[3].parse::<f32>(), p[4].parse::<f32>())
        {
            nodes.push([x, y, z]);
        }
    }
    if nodes.len() < 2 {
        return None;
    }
    if nodes.len() == n {
        return Some(nodes);
    }
    // np.linspace(0, len - 1, n).astype(int64)
    let last = (nodes.len() - 1) as f64;
    Some((0..n).map(|k| nodes[(last * k as f64 / (n - 1).max(1) as f64) as usize]).collect())
}

/// The first cached skeleton among these body ids, sampled to `n` points (`core/neurodex.skeleton_points`).
pub fn skeleton_points(body_ids: &[i64], folders: &[PathBuf], n: usize) -> Option<Vec<[f32; 3]>> {
    for &id in body_ids {
        for d in folders {
            if let Ok(text) = std::fs::read(d.join(format!("{id}.swc")))
                && let Some(p) = sample_points(&String::from_utf8_lossy(&text), n)
            {
                return Some(p);
            }
        }
    }
    None
}

/// The cache folder (`<id>.swc` and `manifest.json`): every file is checked against its recorded sha256 and size on load.
pub struct ShapeStore {
    pub folder: PathBuf,
    manifest: BTreeMap<String, Value>,
}

impl ShapeStore {
    pub fn open(folder: &Path) -> Self {
        let _ = std::fs::create_dir_all(folder);
        let manifest = std::fs::read_to_string(folder.join(MANIFEST)).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
        ShapeStore { folder: folder.to_path_buf(), manifest }
    }

    fn path(&self, body_id: i64) -> PathBuf {
        self.folder.join(format!("{body_id}.swc"))
    }

    fn save_manifest(&self) {
        let tmp = self.folder.join(format!("{MANIFEST}.tmp"));
        if serde_json::to_string_pretty(&self.manifest).ok().and_then(|t| std::fs::write(&tmp, t).ok()).is_some() {
            let _ = std::fs::rename(tmp, self.folder.join(MANIFEST));
        }
    }

    pub fn discard(&mut self, body_id: i64) {
        let _ = std::fs::remove_file(self.path(body_id));
        if self.manifest.remove(&body_id.to_string()).is_some() {
            self.save_manifest();
        }
    }

    fn verified_bytes(&mut self, body_id: i64) -> Option<Vec<u8>> {
        let rec = self.manifest.get(&body_id.to_string())?.clone();
        let data = std::fs::read(self.path(body_id)).ok()?;
        let sha: String = Sha256::digest(&data).iter().map(|b| format!("{b:02x}")).collect();
        if rec["sha256"].as_str() != Some(sha.as_str()) || rec["size"].as_u64() != Some(data.len() as u64) {
            self.discard(body_id);
            return None;
        }
        Some(data)
    }

    /// From the cache only, checked against its sha256. None if absent or bad (a bad file is thrown away).
    pub fn load(&mut self, body_id: i64) -> Option<Skeleton> {
        let data = self.verified_bytes(body_id)?;
        match parse_swc(&String::from_utf8_lossy(&data), body_id) {
            Ok(s) => Some(s),
            Err(_) => {
                self.discard(body_id);
                None
            }
        }
    }

    pub fn count(&self) -> usize {
        self.manifest.len()
    }

    pub fn size_bytes(&self) -> u64 {
        std::fs::read_dir(&self.folder)
            .map(|d| {
                d.filter_map(Result::ok)
                    .filter(|e| e.path().extension().is_some_and(|x| x == "swc"))
                    .filter_map(|e| e.metadata().ok())
                    .map(|m| m.len())
                    .sum()
            })
            .unwrap_or(0)
    }

    pub fn clear(&mut self) -> usize {
        let mut n = 0;
        if let Ok(d) = std::fs::read_dir(&self.folder) {
            for e in d.filter_map(Result::ok) {
                if e.path().extension().is_some_and(|x| x == "swc") && std::fs::remove_file(e.path()).is_ok() {
                    n += 1;
                }
            }
        }
        self.manifest.clear();
        let _ = std::fs::remove_file(self.folder.join(MANIFEST));
        n
    }

    /// Cache bytes the caller has verified against the server's checksum (`md5_b64`), and keep the record.
    pub fn put(&mut self, body_id: i64, data: &[u8], md5_b64: &str, url: &str, fetched: &str) -> std::io::Result<Skeleton> {
        let skel = parse_swc(&String::from_utf8_lossy(data), body_id).map_err(std::io::Error::other)?;
        let tmp = self.path(body_id).with_extension("tmp");
        std::fs::write(&tmp, data)?;
        std::fs::rename(&tmp, self.path(body_id))?;
        let sha: String = Sha256::digest(data).iter().map(|b| format!("{b:02x}")).collect();
        self.manifest.insert(
            body_id.to_string(),
            json!({"size": data.len(), "sha256": sha, "md5": md5_b64, "url": url, "fetched": fetched, "license": "CC BY 4.0", "source": "MaleCNS v1.0 (FlyEM, Janelia)"}),
        );
        self.save_manifest();
        Ok(skel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SWC: &str = "# a tiny neuron\n1 1 0 0 0 1 -1\n2 3 10 0 0 1 1\n3 3 10 10 0 1 2\n4 3 10 0 5 1 2\n";

    #[test]
    fn parses_and_measures() {
        let s = parse_swc(SWC, 7).unwrap();
        assert_eq!((s.n(), s.parent.clone()), (4, vec![-1, 0, 1, 1]));
        assert_eq!(s.segments().count(), 3);
        assert!((s.cable_length_um() - (10.0 + 10.0 + 5.0) * 0.008).abs() < 1e-6);
        assert!(parse_swc("1 1 0 0 0 1 -1\n", 1).is_err());
        assert!(parse_swc("1 1 NaN 0 0 1 -1\n2 3 1 0 0 1 1\n", 1).is_err());
    }

    #[test]
    fn samples_like_linspace() {
        let p = sample_points(SWC, 3).unwrap(); // linspace(0, 3, 3) = 0, 1.5, 3 -> 0, 1, 3
        assert_eq!(p, vec![[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 0.0, 5.0]]);
        assert_eq!(sample_points(SWC, 4).unwrap().len(), 4);
    }

    #[test]
    fn the_cache_checks_what_it_loads() {
        let d = std::env::temp_dir().join(format!("ktf-shapes-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        let mut st = ShapeStore::open(&d);
        assert!(st.load(7).is_none());
        st.put(7, SWC.as_bytes(), "AAAA", "u", "now").unwrap();
        assert_eq!(st.load(7).unwrap().n(), 4);
        assert_eq!(st.count(), 1);
        std::fs::write(d.join("7.swc"), "# edited\n1 1 0 0 0 1 -1\n2 3 1 0 0 1 1\n").unwrap();
        assert!(st.load(7).is_none(), "an edited file is thrown away");
        assert!(!d.join("7.swc").exists());
        assert_eq!(st.clear(), 0);
        let _ = std::fs::remove_dir_all(&d);
    }
}
