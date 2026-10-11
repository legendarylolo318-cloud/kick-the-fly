//! Experiment bundles (`lab/bundle.py`): one zip that holds an experiment and everything needed to check it, and to run it again.
//!
//! `protocol.yaml`, `results/`, `raw/`, `metadata.json`, `ro-crate-metadata.json` (RO-Crate 1.1, each file with its SHA-256) and a
//! `README.txt`. A bundle is a container and a provenance record: it adds nothing to the simulation and claims no biological result.

use std::io::{Read, Write as _};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde_json::{Value, json};

pub const FORMAT: &str = "kick-the-fly-bundle";
pub const FORMAT_VERSION: i64 = 1;
const CRATE_CONTEXT: &str = "https://w3id.org/ro/crate/1.1/context";
const CRATE_SPEC: &str = "https://w3id.org/ro/crate/1.1";
const CC_BY_4: &str = "https://creativecommons.org/licenses/by/4.0/";

#[derive(Debug)]
pub struct BundleError(pub String);

impl std::fmt::Display for BundleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for BundleError {}

fn be(e: impl std::fmt::Display) -> BundleError {
    BundleError(e.to_string())
}

/// `replay.pack_sha256`: the brain pack's SHA-256 (hashed once).
pub fn pack_sha256() -> Option<String> {
    static C: OnceLock<Option<String>> = OnceLock::new();
    C.get_or_init(|| {
        let p = fly_sim::pack::find_default()?;
        let b = std::fs::read(p).ok()?;
        Some(fly_sim::util::sha256_hex(&b))
    })
    .clone()
}

fn safe_name(name: &str) -> bool {
    let p = Path::new(name);
    !(p.is_absolute()
        || p.components().any(|c| matches!(c, std::path::Component::ParentDir))
        || name.starts_with(['/', '\\'])
        || name.contains('\\')
        || name.contains(':'))
}

fn readme(meta: &Value) -> String {
    let s = |k: &str| meta[k].as_str().unwrap_or("").to_string();
    let mut lines = vec![
        format!("Kick the Fly experiment bundle ({}), format {FORMAT_VERSION}", s("kind")),
        format!("made by Kick the Fly {} on {}, {}", s("app_version"), s("platform"), s("created")),
        String::new(),
        format!(
            "backend {}, precision {}, brain pack SHA-256 {}",
            meta["backend"].as_str().unwrap_or("None"),
            meta["dtype"].as_str().unwrap_or("None"),
            meta["brain_pack_sha256"].as_str().unwrap_or("None")
        ),
        String::new(),
    ];
    if meta["rerunnable"].as_bool().unwrap_or(false) {
        lines.push("Run it again:".into());
        lines.push("    KickTheFly --headless --rerun-bundle THIS.zip --out DIR".into());
        lines.push(
            "CPU backends must reproduce every fly's spikes exactly; GPU backends are compared statistically (see kickthefly/lab/bundle.py)."
                .into(),
        );
        lines.push(String::new());
    } else {
        lines.push(format!("This bundle can't be rerun: {}", meta["not_rerunnable_reason"].as_str().unwrap_or("None")));
        lines.push(String::new());
    }
    lines.push("metadata.json lists every parameter; ro-crate-metadata.json (RO-Crate 1.1) describes each file with its SHA-256.".into());
    lines.push("A bundle is a container. It is not a claim that the result is biologically true.".into());
    lines.join("\n") + "\n"
}

fn crate_json(meta: &Value, entries: &std::collections::BTreeMap<String, (usize, String)>, has_protocol: bool) -> Value {
    let parts: Vec<Value> = entries.keys().filter(|n| *n != "ro-crate-metadata.json").map(|n| json!({"@id": n})).collect();
    let mut graph = vec![
        json!({"@type": "CreativeWork", "@id": "ro-crate-metadata.json", "conformsTo": {"@id": CRATE_SPEC}, "about": {"@id": "./"}}),
        json!({"@id": "./", "@type": "Dataset",
               "name": format!("Kick the Fly experiment: {}", meta["protocol_name"].as_str().or(meta["kind"].as_str()).unwrap_or("None")),
               "description": "An experiment run with the Kick the Fly connectome simulation, with its protocol, results, raw exports and the provenance needed to run it again. A container: it makes no biological claim.",
               "datePublished": meta["created"], "hasPart": parts, "isBasedOn": {"@id": "#malecns-v1.0"}, "mentions": {"@id": "#kick-the-fly"}}),
        json!({"@id": "#kick-the-fly", "@type": "SoftwareApplication", "name": "Kick the Fly", "version": meta["app_version"],
               "description": format!("simulation backend {}, state precision {}", meta["backend"].as_str().unwrap_or("None"), meta["dtype"].as_str().unwrap_or("None"))}),
        json!({"@id": "#malecns-v1.0", "@type": "Dataset", "name": "MaleCNS v1.0 connectome (Janelia FlyEM)",
               "description": crate::recorder::CONNECTOME, "license": CC_BY_4,
               "identifier": {"@type": "PropertyValue", "name": "brain pack SHA-256", "value": meta["brain_pack_sha256"]}}),
    ];
    for (name, (size, sha)) in entries {
        if name == "ro-crate-metadata.json" {
            continue;
        }
        let fmt = match Path::new(name).extension().and_then(|e| e.to_str()) {
            Some("yaml") => "application/yaml",
            Some("json") => "application/json",
            Some("csv") => "text/csv",
            Some("npz") => "application/zip",
            Some("txt") => "text/plain",
            Some("nwb") => "application/x-hdf5",
            _ => "application/octet-stream",
        };
        graph.push(json!({"@id": name, "@type": "File", "name": name, "encodingFormat": fmt, "contentSize": size, "sha256": sha}));
    }
    if has_protocol {
        let results: Vec<Value> =
            entries.keys().filter(|n| n.starts_with("results/") || n.starts_with("raw/")).map(|n| json!({"@id": n})).collect();
        graph.push(json!({"@id": "#run", "@type": "CreateAction", "name": "Run the protocol", "instrument": {"@id": "#kick-the-fly"},
                          "object": {"@id": "protocol.yaml"}, "endTime": meta["created"], "result": results}));
    }
    json!({"@context": CRATE_CONTEXT, "@graph": graph})
}

/// `_write`: the zip with fixed timestamps, so the same inputs make the same bytes.
pub fn write(out_zip: &Path, meta: &Value, files: Vec<(String, Vec<u8>)>, has_protocol: bool) -> Result<PathBuf, BundleError> {
    if let Some(d) = out_zip.parent() {
        std::fs::create_dir_all(d).map_err(be)?;
    }
    let mut data: std::collections::BTreeMap<String, Vec<u8>> = files.into_iter().collect();
    data.insert("README.txt".into(), readme(meta).into_bytes());
    data.insert("metadata.json".into(), serde_json::to_string_pretty(meta).map_err(be)?.into_bytes());
    let mut entries: std::collections::BTreeMap<String, (usize, String)> =
        data.iter().map(|(n, b)| (n.clone(), (b.len(), fly_sim::util::sha256_hex(b)))).collect();
    entries.insert("ro-crate-metadata.json".into(), (0, String::new()));
    data.insert(
        "ro-crate-metadata.json".into(),
        serde_json::to_string_pretty(&crate_json(meta, &entries, has_protocol)).map_err(be)?.into_bytes(),
    );
    let tmp = out_zip.with_file_name(format!("{}.tmp", out_zip.file_name().unwrap_or_default().to_string_lossy()));
    {
        let mut z = zip::ZipWriter::new(std::fs::File::create(&tmp).map_err(be)?);
        let time = zip::DateTime::from_date_and_time(2000, 1, 1, 0, 0, 0).map_err(be)?;
        let opts = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated)
            .last_modified_time(time)
            .unix_permissions(0o644);
        for (name, bytes) in &data {
            z.start_file(name, opts).map_err(be)?;
            z.write_all(bytes).map_err(be)?;
        }
        z.finish().map_err(be)?;
    }
    std::fs::rename(&tmp, out_zip).map_err(be)?;
    Ok(out_zip.to_path_buf())
}

fn base_meta(kind: &str, app_version: &str, created: &str, extra: &Value) -> Value {
    let mut m = json!({
        "format": FORMAT, "format_version": FORMAT_VERSION, "kind": kind, "app": "Kick the Fly", "app_version": app_version,
        "created": created, "platform": format!("{} {}", std::env::consts::OS, std::env::consts::ARCH),
        "brain_pack_sha256": pack_sha256(), "individuality": "off",
        "tags": {"container": "GAME RULE", "numbers": "CONNECTOME (the simulation's own output)"},
    });
    if let (Some(o), Some(e)) = (m.as_object_mut(), extra.as_object()) {
        for (k, v) in e {
            o.insert(k.clone(), v.clone());
        }
    }
    m
}

/// `create_live`: bundle a live Lab recording (a record, not a rerunnable experiment).
pub fn create_live(
    files: &[PathBuf],
    meta_extra: &Value,
    out_zip: &Path,
    description: &Value,
    app_version: &str,
    created: &str,
) -> Result<PathBuf, BundleError> {
    let mut extra = json!({
        "rerunnable": false,
        "not_rerunnable_reason": "recorded live: the stimuli were delivered by hand (see the events file in raw/), so there is no script to run again",
        "flies": {},
    });
    if let (Some(o), Some(e)) = (extra.as_object_mut(), meta_extra.as_object()) {
        for (k, v) in e {
            o.insert(k.clone(), v.clone());
        }
    }
    let meta = base_meta("live-recording", app_version, created, &extra);
    let mut blobs: Vec<(String, Vec<u8>)> = Vec::new();
    for f in files {
        if f.is_file() {
            blobs.push((format!("raw/{}", f.file_name().unwrap_or_default().to_string_lossy()), std::fs::read(f).map_err(be)?));
        }
    }
    let _ = description;
    write(out_zip, &meta, blobs, false)
}

/// A bundle opened for reading and checking.
pub struct Bundle {
    zip: zip::ZipArchive<std::fs::File>,
    pub meta: Value,
    pub crate_: Value,
}

impl Bundle {
    pub fn open(path: &Path) -> Result<Self, BundleError> {
        let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
        let f = std::fs::File::open(path).map_err(|e| BundleError(format!("{name} is not a readable zip file ({e})")))?;
        let mut z = zip::ZipArchive::new(f).map_err(|e| BundleError(format!("{name} is not a readable zip file ({e})")))?;
        let names: Vec<String> = z.file_names().map(String::from).collect();
        let bad: Vec<&String> = names.iter().filter(|n| !safe_name(n)).collect();
        if !bad.is_empty() {
            return Err(BundleError(format!("refusing a bundle with unsafe file names: {:?}", bad.iter().take(3).collect::<Vec<_>>())));
        }
        for need in ["metadata.json", "ro-crate-metadata.json"] {
            if !names.iter().any(|n| n == need) {
                return Err(BundleError(format!("not a Kick the Fly bundle: {need} is missing")));
            }
        }
        let mut read = |n: &str| -> Result<Value, BundleError> {
            let mut s = String::new();
            let e = z.by_name(n).map_err(be)?;
            if e.size() > 16 << 20 {
                return Err(BundleError(format!("{n} is implausibly large")));
            }
            e.take(16 << 20).read_to_string(&mut s).map_err(be)?;
            serde_json::from_str(&s).map_err(|e| BundleError(format!("the bundle's metadata is not readable ({e})")))
        };
        let meta = read("metadata.json")?;
        let crate_ = read("ro-crate-metadata.json")?;
        if meta["format"].as_str() != Some(FORMAT) {
            return Err(BundleError("not a Kick the Fly bundle (metadata.json has another format)".into()));
        }
        match meta["format_version"].as_i64() {
            Some(v) if v <= FORMAT_VERSION => {}
            v => {
                return Err(BundleError(format!(
                    "the bundle is a newer format ({}; this version reads up to {FORMAT_VERSION}): update Kick the Fly",
                    v.map_or("None".to_string(), |v| v.to_string())
                )));
            }
        }
        Ok(Self { zip: z, meta, crate_ })
    }

    pub fn names(&self) -> Vec<String> {
        self.zip.file_names().map(String::from).collect()
    }

    pub fn read(&mut self, name: &str) -> Result<Vec<u8>, BundleError> {
        let mut out = Vec::new();
        self.zip.by_name(name).map_err(be)?.read_to_end(&mut out).map_err(be)?;
        Ok(out)
    }

    /// Problems found checking every file against the crate's SHA-256 (empty = intact).
    pub fn verify(&mut self) -> Vec<String> {
        let mut problems = Vec::new();
        let listed: std::collections::HashMap<String, Value> = self.crate_["@graph"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|e| e["@type"] == "File")
            .filter_map(|e| Some((e["@id"].as_str()?.to_string(), e.clone())))
            .collect();
        let names = self.names();
        for n in &names {
            if n == "ro-crate-metadata.json" || n.ends_with('/') {
                continue;
            }
            let Some(e) = listed.get(n) else {
                problems.push(format!("{n} is in the zip but not in the crate"));
                continue;
            };
            let Ok(bytes) = self.read(n) else {
                problems.push(format!("{n} can't be read"));
                continue;
            };
            if e["contentSize"].as_u64() != Some(bytes.len() as u64) {
                problems.push(format!("{n} doesn't have the size the crate records for it"));
            } else if e["sha256"].as_str() != Some(fly_sim::util::sha256_hex(&bytes).as_str()) {
                problems.push(format!("{n} has changed since the bundle was made (its SHA-256 doesn't match)"));
            }
        }
        for n in listed.keys() {
            if !names.contains(n) {
                problems.push(format!("{n} is listed in the crate but missing from the zip"));
            }
        }
        problems
    }
}

/// `_run_manifest`: per-fly facts a run wrote (spike hashes, backend, precision) from its metadata files.
pub fn run_manifest(run_folder: &Path) -> serde_json::Map<String, Value> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(run_folder)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            let n = p.file_name().unwrap_or_default().to_string_lossy();
            n.ends_with("-metadata.json") && n.contains("-seed")
        })
        .collect();
    files.sort();
    let mut flies = serde_json::Map::new();
    for f in files {
        let Some(m) = std::fs::read_to_string(&f).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok()) else { continue };
        let name = f.file_name().unwrap_or_default().to_string_lossy().to_string();
        let key = name.trim_end_matches("-metadata.json").to_string();
        let mut o = serde_json::Map::new();
        for k in [
            "seed",
            "sim_backend",
            "sim_dtype",
            "spike_sha256",
            "n_neurons",
            "synapses",
            "lab_params_modified",
            "surgery",
            "condition",
            "lif_params",
        ] {
            o.insert(k.into(), m.get(k).cloned().unwrap_or(Value::Null));
        }
        flies.insert(key, Value::Object(o));
    }
    flies
}

/// `create`: bundle a protocol run (the folder `protocol_run::run` made).
pub fn create(run_folder: &Path, out_zip: &Path, extra: &Value, app_version: &str, created: &str) -> Result<PathBuf, BundleError> {
    let read = |n: &str| -> Result<Value, BundleError> {
        let t = std::fs::read_to_string(run_folder.join(n)).map_err(|_| {
            BundleError(format!("{} is not a protocol run folder (no protocol.json and summary.json)", run_folder.display()))
        })?;
        serde_json::from_str(&t).map_err(be)
    };
    let p = read("protocol.json")?;
    let _summary = read("summary.json")?;
    let flies = run_manifest(run_folder);
    let backends: std::collections::BTreeSet<String> = flies.values().filter_map(|f| f["sim_backend"].as_str().map(String::from)).collect();
    let dtypes: std::collections::BTreeSet<String> = flies.values().filter_map(|f| f["sim_dtype"].as_str().map(String::from)).collect();
    let is_assay = p.get("assay").is_some();
    let (backend, dtype, rerunnable, why): (String, String, bool, Option<&str>) = if is_assay {
        ("rust-cpu".into(), "float32".into(), true, None)
    } else {
        if backends.len() > 1 || dtypes.len() > 1 {
            return Err(BundleError(format!("the run mixed backends {backends:?} / precisions {dtypes:?}; it can't be bundled")));
        }
        let ok = !flies.is_empty() && flies.values().all(|f| f["spike_sha256"].as_str().is_some_and(|s| !s.is_empty()));
        (
            backends.iter().next().cloned().unwrap_or_else(|| "rust-cpu".into()),
            dtypes.iter().next().cloned().unwrap_or_else(|| "float32".into()),
            ok,
            if ok { None } else { Some("the run predates spike fingerprints (made by a version before 3.0)") },
        )
    };
    let mut ex = json!({
        "protocol_name": p["name"], "seeds": p["seeds"], "backend": backend, "dtype": dtype,
        "parameters": p.get("params").cloned().unwrap_or(json!({})), "surgery": p.get("surgery").cloned().unwrap_or(json!({})),
        "arena": "headless", "rerunnable": rerunnable, "not_rerunnable_reason": why, "flies": flies,
    });
    if let (Some(o), Some(e)) = (ex.as_object_mut(), extra.as_object()) {
        for (k, v) in e {
            o.insert(k.clone(), v.clone());
        }
    }
    let meta = base_meta(if is_assay { "assay" } else { "protocol" }, app_version, created, &ex);
    let mut blobs: Vec<(String, Vec<u8>)> = vec![("protocol.yaml".into(), serde_yaml_ng::to_string(&p).map_err(be)?.into_bytes())];
    blobs.push(("results/summary.json".into(), std::fs::read(run_folder.join("summary.json")).map_err(be)?));
    let mut names: Vec<PathBuf> = std::fs::read_dir(run_folder).map_err(be)?.flatten().map(|e| e.path()).filter(|p| p.is_file()).collect();
    names.sort();
    for f in names {
        let n = f.file_name().unwrap_or_default().to_string_lossy().to_string();
        if n == "protocol.json" || n == "summary.json" {
            continue;
        }
        let dest =
            if matches!(n.as_str(), "result.json" | "per_fly.csv" | "metadata.json") { format!("results/{n}") } else { format!("raw/{n}") };
        blobs.push((dest, std::fs::read(&f).map_err(be)?));
    }
    write(out_zip, &meta, blobs, true)
}
