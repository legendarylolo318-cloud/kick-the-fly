//! `kick_the_fly --headless ...`: runs with no window, no sound and no display (SSH, CI), the Rust side of `lab/headless.py`.
//!
//! ```text
//! kick_the_fly --headless --validate [--out results.json] [--workers N] [--seeds 1000-1009] [--strict]
//! kick_the_fly --headless --protocol experiment.yaml [--out folder] [--bundle experiment.zip]
//! kick_the_fly --headless --rerun-bundle experiment.zip --out folder
//! kick_the_fly --headless --rig NAME | --rig-assay NAME | --minipaper ID | --activation-screen | --knockout-screen [BEHAVIOR ...]
//! ```
//!
//! Exit codes as Python: 0 success; 1 a validation result differs from the expected pass/fail (only with `--strict`) or a rerun does not
//! match; 2 bad input. Not ported (they need the Python-only engines named in PORTING.md): `--replay`, `--record-replay`, `--playthrough`,
//! `--benchmark`, `--audit-asymmetry`, `--threshold-sweep`, `--signflip-test`, `--critical-path`, `--sensitivity`, `--tournament`, `--race`,
//! `--netsci`, `--sleep-deprivation`, `--share-decode`.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use fly_sim::pack::Pack;
use fly_sim::{minipapers, rigs, screens, validation};
use serde_json::{Value, json};

/// The flags only the headless runs take (everything else is the windowed game's).
const FLAGS: [&str; 7] = ["--headless", "--validate", "--protocol", "--rerun-bundle", "--rig", "--rig-assay", "--minipaper"];
const FLAGS_MORE: [&str; 2] = ["--activation-screen", "--knockout-screen"];
/// Python flags that are recognised and refused with a pointer instead of silently opening the game.
const NOT_PORTED: [&str; 14] = [
    "--replay",
    "--record-replay",
    "--playthrough",
    "--benchmark",
    "--audit-asymmetry",
    "--threshold-sweep",
    "--signflip-test",
    "--critical-path",
    "--sensitivity",
    "--tournament",
    "--race",
    "--netsci",
    "--sleep-deprivation",
    "--share-decode",
];

/// Is this a headless invocation?
pub fn wanted(argv: &[String]) -> bool {
    argv.iter().skip(1).any(|a| FLAGS.contains(&a.as_str()) || FLAGS_MORE.contains(&a.as_str()) || NOT_PORTED.contains(&a.as_str()))
}

#[derive(Default)]
struct Cli {
    pack: Option<PathBuf>,
    out: Option<PathBuf>,
    bundle: Option<PathBuf>,
    seeds: Option<String>,
    workers: Option<usize>,
    individuality: Option<String>,
    strict: bool,
    validate: bool,
    protocol: Option<String>,
    rerun: Option<PathBuf>,
    rig: Option<String>,
    rig_assay: Option<String>,
    minipaper: Option<String>,
    activation: bool,
    knockout: Option<Vec<String>>,
    types: Option<Vec<String>>,
    controls: Option<usize>,
    min_neurons: Option<usize>,
    max_types: Option<usize>,
    top: Option<usize>,
    restart: bool,
    refused: Option<String>,
}

fn parse(argv: &[String]) -> Result<Cli, String> {
    let mut c = Cli::default();
    let mut it = argv.iter().skip(1).peekable();
    while let Some(a) = it.next() {
        let mut val = |what: &str| it.next().cloned().ok_or_else(|| format!("{a} needs {what}"));
        let num = |s: String| s.parse::<usize>().map_err(|_| format!("{a}: {s:?} is not a number"));
        match a.as_str() {
            "--headless" | "--strict" if a == "--headless" => {}
            "--strict" => c.strict = true,
            "--validate" => c.validate = true,
            "--pack" => c.pack = Some(PathBuf::from(val("a file")?)),
            "--out" => c.out = Some(PathBuf::from(val("a folder")?)),
            "--bundle" => c.bundle = Some(PathBuf::from(val("a file")?)),
            "--seeds" => c.seeds = Some(val("seeds, e.g. 1000-1009")?),
            "--workers" => c.workers = Some(num(val("a number")?)?),
            "--individuality" => c.individuality = Some(val("off, subtle or strong")?),
            "--protocol" => c.protocol = Some(val("a protocol file")?),
            "--rerun-bundle" => c.rerun = Some(PathBuf::from(val("a bundle .zip")?)),
            "--rig" => c.rig = Some(val("a rig name")?),
            "--rig-assay" => c.rig_assay = Some(val("a rig name")?),
            "--minipaper" => c.minipaper = Some(val("a mini-paper id or 'list'")?),
            "--activation-screen" => c.activation = true,
            "--knockout-screen" => {
                let mut b = Vec::new();
                while let Some(n) = it.peek() {
                    if n.starts_with("--") {
                        break;
                    }
                    b.push(it.next().cloned().unwrap_or_default());
                }
                c.knockout = Some(b);
            }
            "--types" => {
                let mut b = Vec::new();
                while let Some(n) = it.peek() {
                    if n.starts_with("--") {
                        break;
                    }
                    b.push(it.next().cloned().unwrap_or_default());
                }
                c.types = Some(b);
            }
            "--screen-controls" => c.controls = Some(num(val("a number")?)?),
            "--min-neurons" => c.min_neurons = Some(num(val("a number")?)?),
            "--max-types" => c.max_types = Some(num(val("a number")?)?),
            "--top" => c.top = Some(num(val("a number")?)?),
            "--restart" => c.restart = true,
            "--seed" => {
                val("a number")?;
            }
            other if NOT_PORTED.contains(&other) => c.refused = Some(other.to_string()),
            other if other.starts_with("--") => return Err(format!("unknown argument {other} for a headless run")),
            _ => {}
        }
    }
    Ok(c)
}

/// `headless.parse_seeds`: `1000-1009`, `1,5,7` or a mix.
pub fn parse_seeds(text: Option<&str>, default: &[u64]) -> Result<Vec<u64>, String> {
    let Some(text) = text.filter(|t| !t.is_empty()) else { return Ok(default.to_vec()) };
    let mut out = Vec::new();
    for part in text.split(',') {
        let bad = |_| format!("bad seeds {text:?}: use A-B or A,B,C");
        if let Some((a, b)) = part.split_once('-') {
            let (a, b): (u64, u64) = (a.trim().parse().map_err(bad)?, b.trim().parse().map_err(bad)?);
            out.extend(a..=b);
        } else if !part.trim().is_empty() {
            out.push(part.trim().parse().map_err(bad)?);
        }
    }
    Ok(out)
}

fn exports_dir() -> PathBuf {
    ktf_data::paths::get().data_dir.join("exports")
}

fn stamp() -> String {
    crate::savestate_play::utc_stamp_compact()
}

fn created() -> String {
    crate::savestate_play::utc_stamp_pub().replace(' ', "T")
}

fn load_pack(path: Option<&Path>) -> Result<(Arc<fly_sim::csr::Weights>, Arc<fly_sim::pack::Labels>), String> {
    let pack = match path.map(Path::to_path_buf).or_else(fly_sim::pack::find_default) {
        Some(p) => Pack::load(&p).map_err(|e| format!("{}: {e}", p.display()))?,
        None => match crate::sim_thread::embedded_pack() {
            Some(b) => Pack::load_bytes(b).map_err(|e| e.to_string())?,
            None => return Err("no brain pack found (data/kick_brain.npz); use --pack FILE".into()),
        },
    };
    Ok((Arc::new(pack.weights), Arc::new(pack.labels)))
}

fn workers(c: &Cli) -> usize {
    c.workers.filter(|&w| w > 0).unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()).min(4))
}

/// `protocol.find`: a path, or a name (with or without `.yaml`) among the bundled protocols.
fn find_protocol(name: &str) -> Result<(PathBuf, Value), String> {
    let path = Path::new(name);
    if path.exists() {
        return ktf_data::protocol::load(path).map(|p| (path.to_path_buf(), p)).map_err(|e| e.to_string());
    }
    let want = path.file_name().and_then(|f| f.to_str()).unwrap_or(name);
    for (file, text) in ktf_data::protocol::bundled() {
        let stem = file.strip_suffix(".yaml").unwrap_or(file);
        if file == want || stem == want {
            return ktf_data::protocol::load_text(text, file).map(|p| (PathBuf::from(file), p)).map_err(|e| e.to_string());
        }
    }
    let have: Vec<&str> = ktf_data::protocol::bundled().iter().map(|(f, _)| f.as_str()).collect();
    Err(format!("protocol file {name} not found (bundled: {})", have.join(", ")))
}

fn print_progress(t0: Instant) -> impl FnMut(usize, usize, &str) -> bool {
    move |d, n, label| {
        println!("  {d}/{n} {label} ({:.0}s)", t0.elapsed().as_secs_f64());
        true
    }
}

fn run_validate(c: &Cli, w: &Arc<fly_sim::csr::Weights>, l: &Arc<fly_sim::pack::Labels>) -> Result<i32, String> {
    let seeds = parse_seeds(c.seeds.as_deref(), &validation::seeds())?;
    let t0 = Instant::now();
    let mut res = validation::run(w, l, &seeds, None, workers(c), print_progress(t0))?;
    res["app_version"] = json!(env!("CARGO_PKG_VERSION"));
    res["created"] = json!(crate::savestate_play::utc_stamp_pub());
    res["seconds"] = json!((t0.elapsed().as_secs_f64() * 10.0).round() / 10.0);
    res["backend"] = json!("Rust CPU (rayon)");
    res["device"] = json!("CPU");
    res["synapses"] = json!(w.csr.nnz());
    let name = crate::lab_play::RESULTS_NAME;
    let mut out = c.out.clone().unwrap_or_else(|| ktf_data::paths::get().data_dir.join(name));
    if out.extension().is_none_or(|e| !e.eq_ignore_ascii_case("json")) {
        out = out.join(name);
    }
    if let Some(d) = out.parent() {
        std::fs::create_dir_all(d).map_err(|e| e.to_string())?;
    }
    std::fs::write(&out, serde_json::to_string_pretty(&res).unwrap_or_default()).map_err(|e| e.to_string())?;
    println!("{}", validation::summary(&res));
    println!("results written to {}", out.display());
    if c.strict {
        let expected = validation::table()["expected"].as_object().cloned().unwrap_or_default();
        let wrong: Vec<String> = res["tests"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|t| {
                let id = t["id"].as_str().unwrap_or("");
                expected.get(id).and_then(Value::as_bool) != t["passed"].as_bool()
            })
            .map(|t| t["id"].as_str().unwrap_or("").to_string())
            .collect();
        if !wrong.is_empty() {
            println!("differs from the expected results: {}", wrong.join(", "));
            return Ok(1);
        }
    }
    Ok(0)
}

fn run_protocol(c: &Cli, w: &Arc<fly_sim::csr::Weights>, l: &Arc<fly_sim::pack::Labels>, file: &str, p: Option<Value>) -> Result<i32, String> {
    let (name, p) = match p {
        Some(p) => (file.to_string(), p),
        None => match find_protocol(file) {
            Ok((path, p)) => (path.display().to_string(), p),
            Err(e) if e.starts_with("protocol file") => return Err(e),
            Err(e) => {
                println!("error: {e}");
                return Ok(2);
            }
        },
    };
    if p.get("classroom").is_some_and(|v| v.as_bool().unwrap_or(!v.is_null())) {
        println!("error: {name} is a classroom lecture, not a runnable protocol: step through it in Lab > Classroom");
        return Ok(2);
    }
    let pname = p["name"].as_str().unwrap_or("protocol");
    let n = p["seeds"].as_array().map_or(0, Vec::len);
    let mut head = format!("protocol {pname}: {n} fly(s)");
    if let Some(a) = p.get("assay") {
        head += &format!(", assay {}", a.as_str().unwrap_or(""));
    }
    if p.get("patch").is_some() {
        head += ", patch clamp (MODEL)";
    }
    if let Some(r) = p.get("rig") {
        head += &format!(", rig {}", r["name"].as_str().unwrap_or(""));
    }
    if let Some(s) = p.get("screen") {
        head += &format!(", {} screen (MODEL PREDICTION)", s["kind"].as_str().unwrap_or(""));
    }
    println!("{head}");
    let t0 = Instant::now();
    let counts = None;
    let env = crate::protocol_run::Env { weights: w, labels: l, counts, cancel: None };
    let out = c.out.clone().unwrap_or_else(exports_dir);
    let mut prog = print_progress(t0);
    let folder = crate::protocol_run::run(&env, &p, &out, workers(c), |d, n, _| prog(d, n, ""), &created())?;
    let summary: Value =
        serde_json::from_str(&std::fs::read_to_string(folder.join("summary.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    for (tag, groups) in summary["mean_rate_hz"].as_object().into_iter().flatten() {
        for (g, cell) in groups.as_object().into_iter().flatten() {
            println!(
                "  {tag:8} {g:20} {:8.2} Hz  (n={})",
                cell["mean"].as_f64().unwrap_or(f64::NAN),
                cell["n"].as_i64().unwrap_or(0)
            );
        }
    }
    if let Some(t) = summary["text"].as_str().filter(|t| !t.is_empty()) {
        println!("{t}");
    }
    for (sd, sm) in summary["rig_summary"].as_object().into_iter().flatten() {
        let parts: Vec<String> = sm
            .as_object()
            .into_iter()
            .flatten()
            .filter(|(_, v)| !v.is_null())
            .map(|(k, v)| match v.as_f64() {
                Some(f) if v.is_f64() => format!("{k} {}", fly_sim::cards::py_float(f)),
                _ => format!("{k} {}", v.as_str().map_or_else(|| v.to_string(), String::from)),
            })
            .collect();
        println!("  seed {sd}: {}", parts.join(", "));
    }
    for (a, cell) in summary["rate_hz_by_current"].as_object().into_iter().flatten() {
        println!(
            "  current {a:>6}  {:8.2} Hz  (n={})   [{}]",
            cell["mean"].as_f64().unwrap_or(f64::NAN),
            cell["n"].as_i64().unwrap_or(0),
            summary["tag"].as_str().unwrap_or("")
        );
    }
    println!("done in {:.0}s; results in {}", t0.elapsed().as_secs_f64(), folder.display());
    if let Some(zip) = &c.bundle {
        match ktf_data::bundle::create(&folder, zip, &json!({}), env!("CARGO_PKG_VERSION"), &created()) {
            Ok(z) => println!("bundle written to {} (rerun it with --headless --rerun-bundle {} --out DIR)", z.display(), z.display()),
            Err(e) => {
                println!("error: can't bundle this run: {e}");
                return Ok(2);
            }
        }
    }
    Ok(0)
}

fn run_rerun(c: &Cli, w: &Arc<fly_sim::csr::Weights>, l: &Arc<fly_sim::pack::Labels>, zip: &Path) -> Result<i32, String> {
    let Some(out) = &c.out else {
        eprintln!("error: --rerun-bundle needs --out DIR for the rerun's files");
        return Ok(2);
    };
    let t0 = Instant::now();
    let env = crate::protocol_run::Env { weights: w, labels: l, counts: None, cancel: None };
    let mut prog = print_progress(t0);
    match crate::protocol_run::rerun_bundle(&env, zip, out, workers(c), |d, n, _| prog(d, n, ""), &created()) {
        Ok(rep) => {
            println!("{}", rerun_report_text(&rep));
            println!("report: {}", out.join("rerun_report.json").display());
            Ok(i32::from(!rep["match"].as_bool().unwrap_or(false)))
        }
        Err(e) => {
            eprintln!("error: {e}");
            Ok(2)
        }
    }
}

/// `bundle.format_report`.
fn rerun_report_text(rep: &Value) -> String {
    let s = |k: &str| rep[k].as_str().unwrap_or("").to_string();
    let mut lines = vec![
        format!("Rerun of {} ({})", s("bundle"), s("protocol")),
        format!("  recorded on {} {}, rerun on {} {} ({})", s("recorded_backend"), s("recorded_dtype"), s("rerun_backend"), s("rerun_dtype"), s("mode")),
    ];
    for d in rep["details"].as_array().into_iter().flatten() {
        let ok = d["identical"].as_bool().or_else(|| d["ok"].as_bool()).unwrap_or(false);
        let more = d["detail"].as_str().map(|t| format!("  {t}")).unwrap_or_default();
        lines.push(format!("  {} {}{}", if ok { "ok  " } else { "FAIL" }, d["what"].as_str().unwrap_or(""), more));
    }
    lines.push(format!("  {}", s("verdict")));
    lines.join("\n")
}

fn run_rig(c: &Cli, w: &Arc<fly_sim::csr::Weights>, l: &Arc<fly_sim::pack::Labels>) -> Result<i32, String> {
    let assay = c.rig_assay.clone();
    let name = assay.clone().or_else(|| c.rig.clone()).unwrap_or_default();
    if !rigs::RIGS.contains(&name.as_str()) {
        println!("error: unknown rig {name:?}; choose from {}", rigs::RIGS.join(", "));
        return Ok(2);
    }
    let mode = c.individuality.clone().unwrap_or_else(|| "subtle".into());
    let folder = c.out.clone().unwrap_or_else(|| exports_dir().join(format!("{}-rig-{name}", stamp())));
    let t0 = Instant::now();
    let ctx = rigs::Ctx { weights: w, labels: l, cancel: None };
    if assay.is_some() {
        let seeds = parse_seeds(c.seeds.as_deref(), &validation::seeds())?;
        let modes = ["subtle", "off"];
        let res = rigs::run_assay(&ctx, &name, &seeds, &modes, workers(c).max(1), &created(), print_progress(t0))?;
        std::fs::create_dir_all(&folder).map_err(|e| e.to_string())?;
        std::fs::write(folder.join(format!("rig_assay_{name}.json")), serde_json::to_string_pretty(&res).unwrap_or_default())
            .map_err(|e| e.to_string())?;
        let mut csv = String::from("individuality,id,label,n,mean_effect,unit,min_effect,p,passed\r\n");
        for (m, r) in res["results"].as_object().into_iter().flatten() {
            for cr in r["analysis"]["criteria"].as_array().into_iter().flatten() {
                let f = |k: &str| match &cr[k] {
                    Value::Null => String::new(),
                    Value::String(s) => s.clone(),
                    Value::Number(n) => n.as_f64().map_or_else(|| n.to_string(), fly_sim::cards::py_float),
                    v => v.to_string(),
                };
                csv += &format!(
                    "{m},{},{},{},{},{},{},{},{}\r\n",
                    f("id"),
                    csv_field(&f("label")),
                    f("n"),
                    f("mean_effect"),
                    f("unit"),
                    f("min_effect"),
                    f("p"),
                    i32::from(cr["passed"].as_bool().unwrap_or(false))
                );
            }
        }
        std::fs::write(folder.join(format!("rig_assay_{name}_criteria.csv")), csv).map_err(|e| e.to_string())?;
        println!("{}", rigs::summary(&res));
        println!("results written to {} ({:.0} s)", folder.display(), t0.elapsed().as_secs_f64());
        return Ok(0);
    }
    // one fly in the rig, recorded in the Lab's format: the protocol runner with the rig's default scene
    let seeds = parse_seeds(c.seeds.as_deref(), &[0])?;
    let sc = rigs::scene_defaults(&name);
    let mut rig = sc.as_object().cloned().unwrap_or_default();
    rig.insert("name".into(), json!(name));
    rig.insert("individuality".into(), json!(mode));
    let p = ktf_data::protocol::check(
        &json!({"name": format!("rig-{name}"), "seeds": [seeds[0]], "rig": Value::Object(rig)}),
        "--rig",
    )
    .map_err(|e| e.to_string())?;
    let sub = Cli { out: Some(folder.clone()), workers: Some(1), bundle: None, ..Cli::default() };
    run_protocol(&sub, w, l, &format!("rig-{name}"), Some(p))
}

fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n', '\r']) { format!("\"{}\"", s.replace('"', "\"\"")) } else { s.to_string() }
}

fn run_minipaper(c: &Cli, w: &Arc<fly_sim::csr::Weights>, l: &Arc<fly_sim::pack::Labels>) -> Result<i32, String> {
    let pid = c.minipaper.clone().unwrap_or_default();
    if pid == "list" {
        for p in minipapers::papers() {
            println!("{:18} {:28} {} ({})", p.id, p.short, p.citation.split('(').next().unwrap_or("").trim(), p.read.split(" (").next().unwrap_or(""));
        }
        return Ok(0);
    }
    if minipapers::paper(&pid).is_none() {
        println!("error: unknown mini-paper {pid:?}; use --minipaper list");
        return Ok(2);
    }
    let seeds = parse_seeds(c.seeds.as_deref(), &minipapers::QUICK_SEEDS)?;
    let t0 = Instant::now();
    let res = minipapers::run_paper(&pid, &seeds, Some((w, l)), workers(c), print_progress(t0), &created())?;
    let folder = c.out.clone().unwrap_or_else(|| exports_dir().join(format!("{}-minipaper-{pid}", stamp())));
    let none = Default::default();
    minipapers::save(&res, &folder, &none).map_err(|e| e.to_string())?;
    println!("{}", minipapers::render(&res, &none));
    println!("\nwritten to {} ({:.0} s)", folder.display(), t0.elapsed().as_secs_f64());
    Ok(0)
}

fn run_screen(c: &Cli, w: &Arc<fly_sim::csr::Weights>, l: &Arc<fly_sim::pack::Labels>, kind: &str) -> Result<i32, String> {
    let folder = c.out.clone().unwrap_or_else(|| exports_dir().join(format!("{kind}-screen")));
    let seeds = parse_seeds(c.seeds.as_deref(), &screens::SCREEN_SEEDS)?;
    let t0 = Instant::now();
    let r = if kind == "activation" {
        let o = screens::ActivityOptions {
            seeds,
            types: c.types.clone(),
            min_neurons: c.min_neurons.unwrap_or(1),
            max_types: c.max_types,
            controls: c.controls.unwrap_or(2),
            workers: workers(c),
            restart: c.restart,
            ..Default::default()
        };
        screens::run_activation(w, l, &folder, &o, &created(), print_progress(t0))
    } else {
        let mut o = screens::KnockoutOptions {
            seeds,
            top: c.top.unwrap_or(25),
            controls: c.controls.unwrap_or(2),
            workers: workers(c),
            types: c.types.clone(),
            restart: c.restart,
            ..Default::default()
        };
        if let Some(b) = c.knockout.as_ref().filter(|b| !b.is_empty()) {
            o.behaviors = b.clone();
        }
        screens::run_knockout(w, l, &folder, &o, &created(), print_progress(t0))
    };
    match r {
        Ok(res) => {
            println!("{}", screens::summary(&res, c.top.unwrap_or(25)));
            println!("results written to {}", folder.display());
            Ok(0)
        }
        Err(e) => {
            eprintln!("error: {e}");
            Ok(2)
        }
    }
}

/// Run a headless invocation and return the process exit code.
pub fn main(argv: &[String]) -> i32 {
    let c = match parse(argv) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {e}");
            return 2;
        }
    };
    if let Some(flag) = &c.refused {
        eprintln!("error: {flag} is not part of the Rust build (the Python tools for it are in legacy-python/; see PORTING.md)");
        return 2;
    }
    let nothing = !(c.validate
        || c.protocol.is_some()
        || c.rerun.is_some()
        || c.rig.is_some()
        || c.rig_assay.is_some()
        || c.minipaper.is_some()
        || c.activation
        || c.knockout.is_some());
    if nothing {
        eprintln!(
            "nothing to do: use --validate, --protocol FILE, --rerun-bundle ZIP, --rig NAME, --rig-assay NAME, --minipaper ID, --activation-screen or --knockout-screen"
        );
        return 2;
    }
    let (w, l) = match load_pack(c.pack.as_deref()) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: {e}");
            return 2;
        }
    };
    let r = if let Some(z) = &c.rerun {
        run_rerun(&c, &w, &l, z)
    } else if c.rig.is_some() || c.rig_assay.is_some() {
        run_rig(&c, &w, &l)
    } else if let Some(pid) = &c.minipaper {
        let _ = pid;
        run_minipaper(&c, &w, &l)
    } else if c.activation {
        run_screen(&c, &w, &l, "activation")
    } else if c.knockout.is_some() {
        run_screen(&c, &w, &l, "knockout")
    } else if c.validate {
        run_validate(&c, &w, &l)
    } else if let Some(f) = &c.protocol {
        run_protocol(&c, &w, &l, f, None)
    } else {
        Ok(2)
    };
    match r {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e}");
            2
        }
    }
}
