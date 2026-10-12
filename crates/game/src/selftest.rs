//! The self-test (`core/selftest.py`): `--selftest [--out FILE.json]` and Settings > Help > Self-test.
//!
//! Each check ends PASS, WARN or FAIL with a plain-English fix for anything that is not a PASS. Nothing here installs, downloads, removes or
//! configures anything: it only looks. Exit codes: 0 all pass, 1 any FAIL, 3 warnings only.
//!
//! What differs from Python (the Rust build has no Python, NumPy, ModernGL or pygame): one compute backend (rayon on the CPU) instead of
//! NumPy/Numba/torch/GL, compared spike for spike against a hash of the NumPy reference run; graphics = the Vulkan/DX12/Metal adapters
//! wgpu finds instead of OpenGL 3.3; audio and microphone through cpal. The 3.0 module checks (toolkit, day 3/4/5) become "game rules",
//! which runs the small pure parts of the modules this build has.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use fly_sim::brain::Brain;
use fly_sim::lif::{LifParams, LifSim};
use fly_sim::pack::{self, Pack};
use ktf_data::diag::{self, Check, Report, Status};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const BACKEND_STEPS: usize = 200;
const MIN_DISK_MB: f64 = 200.0;
const WARN_DISK_MB: f64 = 2000.0;
pub const ADULT_NEURONS: usize = 166_700;
/// SHA-256 of the 200-step run (seed 42, a pulse on neurons 100..150 every 20 steps) over the packed spike vectors, from NumPy's
/// `selftest.run_backend("cpu")` on the shipped pack: 1,162,816 spikes. The Rust sim must produce exactly this.
const REFERENCE_SHA: &str = "1d906708c4a16b5c717c9cde0b524467f556689ead3fc6dbff06069ad608e953";
const REFERENCE_SPIKES: u64 = 1_162_816;
/// Resident memory the game needs besides the brain (engine, GPU buffers, 3D scene) and per extra fly, in MB (measured, see PORTING.md).
const BASE_MB: f64 = 1500.0;
const PER_FLY_MB: f64 = 60.0;

static PACK_PATH: OnceLock<Option<PathBuf>> = OnceLock::new();

/// The brain pack the game was started with (`--pack`), so the checks look at the same file.
pub fn set_pack_path(p: Option<PathBuf>) {
    let _ = PACK_PATH.set(p);
}

fn pack_path() -> Option<PathBuf> {
    PACK_PATH.get().cloned().flatten().or_else(pack::find_default)
}

pub fn app_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

pub struct Options {
    pub smoke: bool,
    pub graphics: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options { smoke: true, graphics: true }
    }
}

/// What the checks share: the pack, loaded once.
struct Ctx {
    pack: Option<Loaded>,
}

/// The pack's parts the checks run on (shared, never cloned: the weights are over a hundred MB).
pub struct Loaded {
    pub n: usize,
    pub weights: Arc<fly_sim::csr::Weights>,
    pub labels: Arc<fly_sim::pack::Labels>,
}

impl From<Pack> for Loaded {
    fn from(p: Pack) -> Self {
        Loaded { n: p.n, weights: Arc::new(p.weights), labels: Arc::new(p.labels) }
    }
}

fn timed(f: impl FnOnce() -> Vec<Check>) -> Vec<Check> {
    let t0 = Instant::now();
    let mut out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).unwrap_or_else(|e| {
        let msg = e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default();
        vec![
            Check::new("selftest", "Self-test", Status::Fail, format!("the check itself failed: {msg}"))
                .fix("This is a bug in the self-test. Report it with Settings > Help > Report a bug."),
        ]
    });
    let secs = (t0.elapsed().as_secs_f64() * 100.0).round() / 100.0;
    for c in &mut out {
        c.seconds = secs;
    }
    out
}

fn one(f: impl FnOnce() -> Check) -> Vec<Check> {
    timed(|| vec![f()])
}

// --- build -----------------------------------------------------------------------------------------------------------------------
fn check_build() -> Check {
    let kind = if diag::is_appimage() {
        "AppImage"
    } else if cfg!(debug_assertions) {
        "debug build"
    } else {
        "release build"
    };
    Check::new(
        "build",
        "App and engine",
        Status::Pass,
        format!("Kick the Fly {}, Rust / Bevy 0.19, {kind}, {}", app_version(), diag::os_description()),
    )
    .data(diag::build_info(app_version()))
}

// --- brain packs -----------------------------------------------------------------------------------------------------------------
/// A checksum published next to the pack: `<pack>.sha256`, or a line for it in a `SHA256SUMS` file beside it.
fn published_checksum(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_string_lossy().into_owned();
    if let Ok(t) = std::fs::read_to_string(path.with_file_name(format!("{name}.sha256"))) {
        return t.split_whitespace().next().map(str::to_lowercase);
    }
    let sums = std::fs::read_to_string(path.with_file_name("SHA256SUMS")).ok()?;
    sums.lines().find_map(|l| {
        let parts: Vec<&str> = l.split_whitespace().collect();
        (parts.len() >= 2 && parts[parts.len() - 1].trim_start_matches('*') == name).then(|| parts[0].to_lowercase())
    })
}

fn check_pack(brain: &str, ctx: &mut Ctx) -> Check {
    let (cid, name) = (format!("brainpack_{brain}"), format!("{} brain pack", if brain == "adult" { "Adult" } else { "Larva" }));
    let path =
        if brain == "adult" { pack_path() } else { pack_path().map(|p| p.with_file_name("kick_larva_brain.npz")).filter(|p| p.exists()) };
    let Some(path) = path else {
        if brain == "adult" {
            if crate::sim_thread::embedded_pack().is_some() {
                return match Pack::load_bytes(crate::sim_thread::embedded_pack().unwrap_or(&[])) {
                    Ok(p) => {
                        let n = p.n;
                        ctx.pack = Some(p.into());
                        Check::new(&cid, &name, Status::Pass, format!("the pack inside the program: {} neurons", diag::group3(n)))
                    }
                    Err(e) => Check::new(&cid, &name, Status::Fail, format!("the pack inside the program can't be read: {e}"))
                        .fix("This build is damaged. Download it again."),
                };
            }
            return Check::new(&cid, &name, Status::Fail, "the adult brain pack (kick_brain.npz) was not found").fix(
                "Build it with the Python game's dev tool (the python-legacy branch; README, 'Build from source') and put it in data/, or use a release build (it has the pack inside).",
            );
        }
        return Check::new(&cid, &name, Status::Pass, "not present (optional: the larva brain is built locally, see docs/larva.md)");
    };
    let fname = path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    // reading every array also runs the zip CRC-32 of each member: a damaged file fails here
    let loaded = if brain == "adult" { Pack::load(&path) } else { Pack::load_weights_only(&path) };
    let pk = match loaded {
        Ok(p) => p,
        Err(e) => {
            return Check::new(&cid, &name, Status::Fail, format!("{fname} can't be read as a pack: {e}"))
                .fix("The file is damaged. Delete it and rebuild or re-download it; nothing else can fix it.")
                .data(json!({"path": path.to_string_lossy()}));
        }
    };
    let digest = Pack::file_sha256(&path).unwrap_or_default();
    let size_mb = (std::fs::metadata(&path).map_or(0.0, |m| m.len() as f64) / 1_048_576.0 * 10.0).round() / 10.0;
    let data = json!({"path": path.to_string_lossy(), "sha256": digest, "size_mb": size_mb, "neurons": pk.n});
    let expected = published_checksum(&path);
    let n = pk.n;
    if brain == "adult" {
        ctx.pack = Some(pk.into());
    }
    if let Some(exp) = &expected
        && *exp != digest
    {
        return Check::new(
            &cid,
            &name,
            Status::Fail,
            format!("{fname}: SHA-256 {}... does not match the published {}...", &digest[..16], &exp[..16.min(exp.len())]),
        )
        .fix("Re-download or rebuild the pack: this file is not the one that was published.")
        .data(data);
    }
    if brain == "adult" && n != ADULT_NEURONS {
        return Check::new(
            &cid,
            &name,
            Status::Warn,
            format!("{fname} has {} neurons; the game expects {}", diag::group3(n), diag::group3(ADULT_NEURONS)),
        )
        .fix("Rebuild the pack with the current version. Results will not match the published validation.")
        .data(data);
    }
    let how =
        if expected.is_some() { "matches the published checksum" } else { "no published checksum next to it; file integrity checked" };
    Check::new(&cid, &name, Status::Pass, format!("{fname}: SHA-256 {}..., {size_mb} MB, {how}", &digest[..16])).data(data)
}

// --- compute backend -------------------------------------------------------------------------------------------------------------
/// The quick fixed run of `selftest.run_backend`: (spikes, SHA-256 over the packed spike vectors, steps per second).
pub fn run_backend(pack: &Loaded, steps: usize, seed: u64) -> (u64, String, f64) {
    let n = pack.n;
    let mut sim = LifSim::new(pack.weights.clone(), LifParams::default(), seed);
    let mut hasher = Sha256::new();
    let mut total = 0u64;
    let mut sens = vec![0.0f32; n];
    sens[100..150].fill(2.0);
    let mut bits = vec![0u8; n.div_ceil(8)];
    let t0 = Instant::now();
    for step in 0..steps {
        let sp = sim.step(if step % 20 == 10 { Some(&sens) } else { None });
        bits.fill(0);
        for &i in sp {
            bits[i as usize / 8] |= 0x80 >> (i % 8);
        }
        hasher.update(&bits);
        total += sp.len() as u64;
    }
    let dt = t0.elapsed().as_secs_f64().max(1e-9);
    let sha: String = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect();
    (total, sha, steps as f64 / dt)
}

fn check_backend(ctx: &Ctx) -> Check {
    let Some(pack) = &ctx.pack else {
        return Check::new("backend_cpu", "Compute backend (CPU, rayon)", Status::Fail, "no brain pack to run")
            .fix("See the brain pack check above.");
    };
    let (spikes, sha, speed) = run_backend(pack, BACKEND_STEPS, 42);
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let data =
        json!({"spikes": spikes, "sha": sha, "steps": BACKEND_STEPS, "steps_per_s": (speed * 10.0).round() / 10.0, "threads": threads});
    diag::set_info("sim_backend", "cpu (rayon)");
    diag::set_info("sim_device", format!("CPU, {threads} threads"));
    let label = "Compute backend (CPU, rayon)";
    if pack.n != ADULT_NEURONS {
        // another pack: no published reference, so check only that two runs agree with each other
        let (s2, sha2, _) = run_backend(pack, BACKEND_STEPS, 42);
        let same = sha2 == sha && s2 == spikes;
        return Check::new(
            "backend_cpu",
            label,
            if same { Status::Pass } else { Status::Fail },
            format!("{BACKEND_STEPS} steps ran, {} spikes, {speed:.0} steps/s; two runs with one seed {} (no reference run exists for this pack)", diag::group3(spikes as usize), if same { "are identical" } else { "DIFFER" }),
        )
        .data(data);
    }
    if sha == REFERENCE_SHA && spikes == REFERENCE_SPIKES {
        Check::new(
            "backend_cpu",
            label,
            Status::Pass,
            format!("identical to the NumPy reference spike for spike ({BACKEND_STEPS} steps, {} spikes), {speed:.0} steps/s, {threads} threads", diag::group3(spikes as usize)),
        )
        .data(data)
    } else {
        Check::new(
            "backend_cpu",
            label,
            Status::Fail,
            format!(
                "DIFFERENT from the NumPy reference: {} spikes, SHA-256 {}... (expected {} spikes)",
                diag::group3(spikes as usize),
                &sha[..16],
                diag::group3(REFERENCE_SPIKES as usize)
            ),
        )
        .fix("The simulation must match NumPy bit for bit. Report this with Settings > Help > Report a bug.")
        .data(data)
    }
}

// --- graphics --------------------------------------------------------------------------------------------------------------------
fn check_graphics() -> Vec<Check> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapters: Vec<wgpu::AdapterInfo> =
        bevy::tasks::block_on(instance.enumerate_adapters(wgpu::Backends::all())).iter().map(|a| a.get_info()).collect();
    let desc = |i: &wgpu::AdapterInfo| {
        format!(
            "{} ({:?}, {:?}{})",
            i.name,
            i.backend,
            i.device_type,
            if i.driver.is_empty() { String::new() } else { format!(", driver {} {}", i.driver, i.driver_info) }
        )
    };
    let hw: Vec<&wgpu::AdapterInfo> = adapters.iter().filter(|i| !matches!(i.device_type, wgpu::DeviceType::Cpu)).collect();
    let data = json!({"adapters": adapters.iter().map(desc).collect::<Vec<_>>()});
    if adapters.is_empty() {
        return vec![Check::new("gpu", "Graphics adapter", Status::Fail, "no Vulkan, DirectX 12 or Metal adapter was found")
            .fix("Kick the Fly draws with wgpu and needs a GPU driver for Vulkan (Linux), DirectX 12 (Windows) or Metal (macOS). Install your GPU's driver through your OS (on Linux: vulkan-icd-loader and the driver's Vulkan package); this program never does it.")
            .data(data)];
    }
    if hw.is_empty() {
        return vec![Check::new("gpu", "Graphics adapter", Status::Warn, format!("only a software renderer was found: {}", desc(&adapters[0])))
            .fix("There is no hardware GPU driver in use, so 3D will be slow. Installing your GPU's driver is up to you; the game never does it.")
            .data(data)];
    }
    let used = diag::info("gpu_name").map(|n| format!("; the game window uses {n}")).unwrap_or_default();
    vec![
        Check::new("gpu", "Graphics adapter", Status::Pass, format!("{}{used}", hw.iter().map(|i| desc(i)).collect::<Vec<_>>().join("; ")))
            .data(data),
    ]
}

// --- environment -----------------------------------------------------------------------------------------------------------------
fn check_audio() -> Check {
    use cpal::traits::{DeviceTrait, HostTrait};
    for id in crate::mic::host_order() {
        let Ok(host) = cpal::host_from_id(id) else { continue };
        if let Some(dev) = host.default_output_device()
            && let Ok(cfg) = dev.default_output_config()
        {
            let name = dev.description().map_or_else(|_| "unnamed device".into(), |d| d.to_string());
            return Check::new(
                "audio",
                "Audio device",
                Status::Pass,
                format!("found an output device ({id:?}: {name}, {} Hz); the self-test did not play anything", cfg.sample_rate()),
            );
        }
    }
    Check::new("audio", "Audio device", Status::Warn, "no audio output device was found")
        .fix("The game plays silently. Check your system's sound output or that another program isn't holding it.")
}

pub fn find_in_path(exe: &str) -> Option<PathBuf> {
    let paths = std::env::var_os("PATH")?;
    std::env::split_paths(&paths).flat_map(|d| [d.join(exe), d.join(format!("{exe}.exe"))]).find(|p| p.is_file())
}

fn check_ffmpeg() -> Check {
    match find_in_path("ffmpeg") {
        Some(p) => Check::new("ffmpeg", "ffmpeg on PATH", Status::Pass, format!("found {}", p.display())),
        None => Check::new("ffmpeg", "ffmpeg on PATH", Status::Warn, "ffmpeg was not found").fix(
            "Video recording (Shift+R) falls back to a short GIF. For MP4, install ffmpeg with your package manager yourself; the game never installs anything.",
        ),
    }
}

fn check_neurodex() -> Check {
    let facts = ktf_data::neurodex_facts::facts();
    if facts.is_empty() || facts.iter().any(|f| f.cite.is_empty() || !f.doi.starts_with("10.")) {
        return Check::new("neurodex", "Neurodex facts", Status::Fail, "the curated facts can't be used (a fact has no citation or DOI)")
            .fix("This build is damaged. Download it again.");
    }
    let prog = ktf_data::stores::Progress::open(ktf_data::paths::get().memory_dir.join("neurodex.json"));
    match prog.warnings.first() {
        Some(w) => {
            Check::new("neurodex", "Neurodex facts", Status::Warn, format!("{} curated facts, all with a citation; {w}", facts.len()))
                .fix("The unreadable progress file was kept aside as neurodex.json.bad; the Neurodex starts empty.")
        }
        None => Check::new("neurodex", "Neurodex facts", Status::Pass, format!("{} curated facts, all with a citation", facts.len())),
    }
}

/// The small pure parts of the game modules this build has, as `check_day3/4/5` do for Python's: a constant out of range or a broken rule
/// shows up here before it shows up in a match.
fn check_rules() -> Check {
    use crate::{points, tournament, weather};
    let mut bad: Vec<&str> = vec![];
    if (weather::PART_WEIGHTS.iter().map(|p| p.1).sum::<f32>() - 1.0).abs() > 1e-6 {
        bad.push("rain hit weights do not sum to 1");
    }
    if tournament::SIZES != [4, 8, 16] {
        bad.push("tournament sizes");
    }
    if crate::duel::TICK_STEPS != 4 {
        bad.push("duel tick");
    }
    if points::decimal_odds(0.5) < points::MIN_ODDS {
        bad.push("odds below the minimum");
    }
    if crate::race::speed_of(99.0, 3.0) != crate::race::V_MAX {
        bad.push("race speed rule");
    }
    if crate::world::ARENAS.len() != 10 || crate::tools::TOOLS.len() != 18 {
        bad.push("arena or tool list");
    }
    if bad.is_empty() {
        Check::new(
            "rules",
            "Game rules",
            Status::Pass,
            "10 arenas, 18 tools, the rain hit weights, the tournament sizes, the odds and the race speed rule all check out",
        )
    } else {
        Check::new("rules", "Game rules", Status::Warn, format!("a game module is inconsistent: {}", bad.join(", ")))
            .fix("Reinstall the game.")
    }
}

fn check_microphone() -> Check {
    let (ok, why) = crate::mic::availability();
    Check::new(
        "microphone",
        "Microphone (optional)",
        Status::Pass,
        if ok {
            format!("{why}; the self-test did not open it, and the feature is off until you turn it on")
        } else {
            format!("not available: {why} (optional: only the microphone feature needs it)")
        },
    )
    .data(json!({"available": ok}))
}

/// `netguard.allowed()`: false when `KTF_NO_NETWORK` is set (the test suite does) or `KICK_THE_FLY_OFFLINE=1`.
pub fn network_allowed() -> (bool, String) {
    // the one switch (`netguard.allowed`): off in headless runs and under KTF_NO_NETWORK
    if let Err(why) = crate::twitch::allowed() {
        return (false, why);
    }
    if std::env::var("KICK_THE_FLY_OFFLINE").is_ok_and(|v| v == "1") {
        return (false, "the network is off here (KICK_THE_FLY_OFFLINE=1)".into());
    }
    (true, String::new())
}

fn check_network() -> Check {
    let (ok, why) = network_allowed();
    let head = if ok { "the self-test did not connect to anything".to_string() } else { format!("off here: {why}") };
    Check::new(
        "network",
        "Network features (optional)",
        Status::Pass,
        format!("{head}; the only network use this build has is the real neuron shapes download, off until you opt in (Settings > Brain): a skeleton is fetched from Janelia's public MaleCNS bucket (storage.googleapis.com) when you inspect that neuron; KICK_THE_FLY_OFFLINE=1 turns it off; Streamer mode (Twitch chat) is not in this build; no telemetry"),
    )
    .data(json!({"allowed": ok}))
}

fn writable(d: &Path) -> Option<String> {
    if let Err(e) = std::fs::create_dir_all(d) {
        return Some(e.to_string());
    }
    let probe = d.join(format!(".ktf-selftest-{}", std::process::id()));
    match std::fs::write(&probe, "x").and_then(|()| std::fs::remove_file(&probe)) {
        Ok(()) => None,
        Err(e) => Some(e.to_string()),
    }
}

fn check_folders() -> Vec<Check> {
    let p = ktf_data::paths::get();
    let mut out = vec![];
    for (cid, label, dir, level, why) in [
        ("dir_config", "Config folder", &p.config_dir, Status::Fail, "Your settings can't be saved"),
        ("dir_data", "Data folder", &p.data_dir, Status::Fail, "Saves, replays and training memory can't be written"),
        ("dir_state", "Log folder", &p.state_dir, Status::Warn, "Logs and crash reports can't be written"),
        ("dir_screenshots", "Screenshot folder", &p.pictures_dir, Status::Warn, "Screenshots and GIFs can't be saved"),
    ] {
        out.push(match writable(dir) {
            None => Check::new(cid, label, Status::Pass, format!("{} is writable", dir.display())),
            Some(err) => Check::new(cid, label, level, format!("{} is not writable ({err})", dir.display()))
                .fix(format!("{why}. Make the folder writable, or point KICK_THE_FLY_HOME at a folder you own.")),
        });
    }
    out
}

fn free_disk_mb(d: &Path) -> Option<f64> {
    let mut probe = d.to_path_buf();
    while !probe.exists() && probe.pop() {}
    fs4::available_space(&probe).ok().map(|b| b as f64 / 1_048_576.0)
}

/// Available memory in MB (`platform_env.available_memory_mb`); None where it can't be read.
pub fn available_memory_mb() -> Option<f64> {
    #[cfg(target_os = "linux")]
    {
        let t = std::fs::read_to_string("/proc/meminfo").ok()?;
        let kb: f64 = t.lines().find_map(|l| l.strip_prefix("MemAvailable:"))?.split_whitespace().next()?.parse().ok()?;
        Some(kb / 1024.0)
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
        // SAFETY: the struct is zeroed and its length field set, as the API requires
        unsafe {
            let mut m: MEMORYSTATUSEX = std::mem::zeroed();
            m.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
            (GlobalMemoryStatusEx(&mut m) != 0).then(|| m.ullAvailPhys as f64 / 1_048_576.0)
        }
    }
    #[cfg(not(any(target_os = "linux", windows)))]
    {
        None
    }
}

fn check_resources() -> Vec<Check> {
    let mut out = vec![];
    match free_disk_mb(&ktf_data::paths::get().data_dir) {
        None => out.push(Check::new("disk", "Free disk space", Status::Warn, "could not read the free disk space")),
        Some(f) if f < MIN_DISK_MB => out.push(
            Check::new("disk", "Free disk space", Status::Fail, format!("only {f:.0} MB free where your data is kept"))
                .fix("Free some space: saves, replays and recordings can't be written."),
        ),
        Some(f) if f < WARN_DISK_MB => out.push(
            Check::new("disk", "Free disk space", Status::Warn, format!("{f:.0} MB free where your data is kept"))
                .fix("Video and replays can be large; free some space if recording fails."),
        ),
        Some(f) => {
            out.push(Check::new("disk", "Free disk space", Status::Pass, format!("{:.1} GB free where your data is kept", f / 1024.0)))
        }
    }
    match available_memory_mb() {
        None => out.push(Check::new(
            "memory",
            "Free memory",
            Status::Warn,
            format!("could not read free memory (each extra fly needs about {PER_FLY_MB:.0} MB)"),
        )),
        Some(avail) => {
            let flies = ((avail - BASE_MB) / PER_FLY_MB).floor().max(0.0) as i64;
            let st = if avail >= BASE_MB + 2.0 * PER_FLY_MB {
                Status::Pass
            } else if avail >= BASE_MB + PER_FLY_MB {
                Status::Warn
            } else {
                Status::Fail
            };
            let fix = match st {
                Status::Pass => "",
                Status::Warn => "Close other programs. With this little memory a second fly may not start.",
                Status::Fail => "There is not enough free memory to run even one fly comfortably. Close other programs.",
            };
            out.push(
                Check::new("memory", "Free memory vs per-fly estimate", st, format!("{avail:.0} MB available; the game needs about {BASE_MB:.0} MB and each extra fly about {PER_FLY_MB:.0} MB (the brain's weights are shared), so about {flies} flies fit"))
                    .fix(fix)
                    .data(json!({"available_mb": avail.round(), "per_fly_mb": PER_FLY_MB, "flies_fit": flies})),
            );
        }
    }
    out
}

fn check_display() -> Check {
    if !cfg!(target_os = "linux") {
        return Check::new(
            "display",
            "Display backend",
            Status::Pass,
            format!("{}: the native window system is used", std::env::consts::OS),
        );
    }
    let e = |k: &str| std::env::var(k).unwrap_or_default();
    let session = match e("XDG_SESSION_TYPE") {
        s if !s.is_empty() => s,
        _ if !e("WAYLAND_DISPLAY").is_empty() => "wayland".into(),
        _ if !e("DISPLAY").is_empty() => "x11".into(),
        _ => "none".into(),
    };
    let want = if !e("WAYLAND_DISPLAY").is_empty() {
        "Wayland"
    } else if !e("DISPLAY").is_empty() {
        "X11"
    } else {
        "no window system"
    };
    let detail =
        format!("session {session} (WAYLAND_DISPLAY={:?}, DISPLAY={:?}); the game would use {want}", e("WAYLAND_DISPLAY"), e("DISPLAY"));
    if session == "none" {
        return Check::new("display", "Display backend", Status::Warn, format!("no display is available: {detail}"))
            .fix("Fine for headless runs. To open the game you need a graphical session (or xvfb).")
            .data(json!({"session": session}));
    }
    Check::new("display", "Display backend", Status::Pass, detail).data(json!({"session": session, "choice": want}))
}

/// `protocols/selftest_smoke.yaml`: 10 s of simulated time, 0.5 s warm-up, seed 7, the looming detectors driven at 0.5 from 3 s to 5 s;
/// the mean rates of the giant fiber (DNp01) and the looming detectors (LPLC2 + LC4) over the run.
pub fn smoke_rates(pack: &Loaded, seconds: f64) -> Result<(f64, f64), String> {
    let mut brain = Brain::warmed_up(pack.weights.clone(), pack.labels.clone(), LifParams::default(), 7, 100);
    let dt = 0.005;
    let rows = |spec: &str| brain.rows_of(spec).map_err(|e| e.to_string());
    let gf = rows("type:DNp01")?;
    let mut loom = rows("type:LPLC2")?;
    loom.extend(rows("type:LC4")?);
    if gf.is_empty() || loom.is_empty() {
        return Err("the giant fiber or the looming detectors are not in this pack".into());
    }
    let mut mask = vec![0u8; brain.n()];
    for &r in &gf {
        mask[r as usize] |= 1;
    }
    for &r in &loom {
        mask[r as usize] |= 2;
    }
    let steps = (seconds / dt).round() as usize;
    let (mut gf_spikes, mut loom_spikes) = (0u64, 0u64);
    for step in 0..steps {
        let t = step as f64 * dt;
        if (t - 3.0).abs() < dt / 2.0 {
            brain.drive(&loom, 0.5);
        }
        if (t - 5.0).abs() < dt / 2.0 {
            brain.undrive(&loom);
        }
        for &s in brain.step() {
            let m = mask[s as usize];
            gf_spikes += u64::from(m & 1);
            loom_spikes += u64::from(m >> 1 & 1);
        }
    }
    Ok((loom_spikes as f64 / (loom.len() as f64 * seconds), gf_spikes as f64 / (gf.len() as f64 * seconds)))
}

fn check_smoke(ctx: &Ctx) -> Check {
    let Some(pack) = &ctx.pack else {
        return Check::new("smoke", "10-second smoke protocol", Status::Fail, "no brain pack to run")
            .fix("See the brain pack check above.");
    };
    match smoke_rates(pack, 10.0) {
        Err(e) => {
            Check::new("smoke", "10-second smoke protocol", Status::Fail, e).fix("Rebuild the brain pack and report it if it persists.")
        }
        Ok((loom, gf)) if loom > 0.0 => Check::new(
            "smoke",
            "10-second smoke protocol",
            Status::Pass,
            format!("10 s of simulated time ran; looming detectors {loom:.1} Hz, giant fiber {gf:.1} Hz"),
        )
        .data(json!({"looming_detectors": loom, "giant_fiber": gf})),
        Ok((loom, gf)) => Check::new(
            "smoke",
            "10-second smoke protocol",
            Status::Fail,
            format!("the driven looming detectors did not fire (looming {loom:.1} Hz, giant fiber {gf:.1} Hz)"),
        )
        .fix("The simulation ran but did not respond to input. Rebuild the brain pack and report it if it persists."),
    }
}

// --- running it all --------------------------------------------------------------------------------------------------------------
/// All checks; `progress` is called as each finishes. The report is also kept for the bug report.
pub fn run(opts: &Options, progress: &mut dyn FnMut(&Check)) -> Report {
    let mut ctx = Ctx { pack: None };
    let mut checks: Vec<Check> = vec![];
    let mut push = |cs: Vec<Check>, checks: &mut Vec<Check>| {
        for c in cs {
            progress(&c);
            checks.push(c);
        }
    };
    push(one(check_build), &mut checks);
    let c = one(|| check_pack("adult", &mut ctx));
    push(c, &mut checks);
    let c = one(|| check_pack("larva", &mut ctx));
    push(c, &mut checks);
    let c = one(|| check_backend(&ctx));
    push(c, &mut checks);
    if opts.graphics {
        push(timed(check_graphics), &mut checks);
    }
    push(one(check_audio), &mut checks);
    push(one(check_ffmpeg), &mut checks);
    push(timed(check_folders), &mut checks);
    push(one(check_neurodex), &mut checks);
    push(one(check_rules), &mut checks);
    push(one(check_microphone), &mut checks);
    push(one(check_network), &mut checks);
    push(timed(check_resources), &mut checks);
    push(one(check_display), &mut checks);
    if opts.smoke {
        let c = one(|| check_smoke(&ctx));
        push(c, &mut checks);
    }
    let report = diag::summarize(checks, app_version(), diag::build_info(app_version()));
    diag::set_last_report(report.clone());
    report
}

/// `--selftest [--out FILE.json]`: headless, no window.
pub fn main(out: Option<&Path>) -> i32 {
    let rep = run(&Options::default(), &mut |c| println!("[{}] {}: {}", c.status.as_str(), c.name, c.detail));
    println!();
    println!("{}", rep.to_text());
    if let Some(p) = out {
        let p = if p.extension().is_some_and(|e| e.eq_ignore_ascii_case("json")) { p.to_path_buf() } else { p.join("selftest.json") };
        if let Some(d) = p.parent() {
            let _ = std::fs::create_dir_all(d);
        }
        match std::fs::write(&p, rep.to_json()) {
            Ok(()) => println!("JSON written to {}", p.display()),
            Err(e) => eprintln!("could not write {}: {e}", p.display()),
        }
    }
    rep.exit_code
}

// --- the page's worker -----------------------------------------------------------------------------------------------------------
#[derive(Default)]
struct PageState {
    running: bool,
    started: bool,
    checks: Vec<Check>,
    counts: Option<(usize, usize, usize)>,
}

fn page_state() -> &'static Mutex<PageState> {
    static S: OnceLock<Mutex<PageState>> = OnceLock::new();
    S.get_or_init(Default::default)
}

/// Settings > Help > Self-test: run on a worker thread, checks appear as they finish. Does nothing while a run is going.
pub fn start_page_run() {
    {
        let mut st = page_state().lock().unwrap();
        if st.running {
            return;
        }
        *st = PageState { running: true, started: true, ..Default::default() };
    }
    std::thread::Builder::new()
        .name("selftest".into())
        .spawn(|| {
            let rep = std::panic::catch_unwind(|| run(&Options::default(), &mut |c| page_state().lock().unwrap().checks.push(c.clone())));
            let mut st = page_state().lock().unwrap();
            match rep {
                Ok(r) => st.counts = Some((r.counts.pass, r.counts.warn, r.counts.fail)),
                Err(_) => st.checks.push(
                    Check::new("selftest", "Self-test", Status::Fail, "the self-test stopped unexpectedly")
                        .fix("Report this with Report a bug."),
                ),
            }
            st.running = false;
        })
        .ok();
}

/// What the Self-test page draws (query `selftest`); starts the first run itself, like Python's `page_selftest`.
pub fn page_json() -> Value {
    let started = page_state().lock().unwrap().started;
    if !started {
        start_page_run();
    }
    let st = page_state().lock().unwrap();
    let c = st.counts;
    json!({
        "running": st.running,
        "checks": st.checks.iter().map(|c| json!({"name": c.name, "status": c.status.as_str(), "detail": c.detail, "fix": c.fix})).collect::<Vec<_>>(),
        "counts": c.map(|(p, w, f)| json!({"PASS": p, "WARN": w, "FAIL": f})),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn network_switch() {
        // not set in the test environment unless the CI sets it; both answers must carry a reason when off
        let (ok, why) = network_allowed();
        assert!(ok || !why.is_empty());
    }

    #[test]
    fn folder_and_display_checks_run_anywhere() {
        let f = check_folders();
        assert_eq!(f.len(), 4);
        let _ = check_display();
        let _ = check_ffmpeg();
        assert!(check_rules().status == Status::Pass, "{:?}", check_rules().detail);
    }

    #[test]
    fn published_checksum_is_read_from_a_sums_file() {
        let d = std::env::temp_dir().join(format!("ktf-st-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("SHA256SUMS"), "ABCDEF  other.npz\n123456 *kick_brain.npz\n").unwrap();
        assert_eq!(published_checksum(&d.join("kick_brain.npz")).as_deref(), Some("123456"));
        std::fs::write(d.join("kick_brain.npz.sha256"), "FEED  kick_brain.npz\n").unwrap();
        assert_eq!(published_checksum(&d.join("kick_brain.npz")).as_deref(), Some("feed"));
        let _ = std::fs::remove_dir_all(&d);
    }

    /// The reference run on the real pack (skipped without one): the Rust simulator gives NumPy's spikes.
    #[test]
    fn backend_run_matches_numpy() {
        let Some(path) = pack::find_default() else {
            eprintln!("SKIPPED: no brain pack");
            return;
        };
        let pack: Loaded = Pack::load(path).unwrap().into();
        if pack.n != ADULT_NEURONS {
            return;
        }
        let (spikes, sha, _) = run_backend(&pack, BACKEND_STEPS, 42);
        assert_eq!((spikes, sha.as_str()), (REFERENCE_SPIKES, REFERENCE_SHA));
    }
}
