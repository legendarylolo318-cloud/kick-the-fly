//! Diagnostics: the self-test report, the bug report and the crash report (`core/selftest.py`, `core/bugreport.py`, `core/crash.py`).
//!
//! Pure data and text, no Bevy and no network: the game runs the checks (`kick_the_fly::selftest`) and shows the pages; everything that
//! leaves the machine is text the player has seen first (copied, saved to a file, or put in a prefilled GitHub issue URL they open
//! themselves). Times are UTC (the Python game used local time).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use serde::Serialize;
use serde_json::{Value, json};

pub const REPO: &str = "legendarylolo318-cloud/kick-the-fly";
/// Characters of the whole issue URL we allow ourselves (GitHub refuses somewhere near 8,000).
pub const URL_LIMIT: usize = 6000;
pub const LOG_LINES: usize = 500;
pub const CRASH_NAME: &str = "KickTheFly-crash.txt";
pub const LOG_NAME: &str = "kickthefly.log";
pub const PREVIOUS_LOG: &str = "kickthefly.previous.log";

// --- facts the game records as it starts (crash.info) ----------------------------------------------------------------------------
fn info_map() -> &'static Mutex<BTreeMap<String, String>> {
    static M: OnceLock<Mutex<BTreeMap<String, String>>> = OnceLock::new();
    M.get_or_init(Default::default)
}

/// Remember a fact for crash reports and bug reports (`seed`, `mode`, `sim_backend`, `sim_device`, `video_driver`, `gl_renderer`, ...).
pub fn set_info(key: &str, value: impl Into<String>) {
    info_map().lock().unwrap().insert(key.into(), value.into());
}

pub fn info(key: &str) -> Option<String> {
    info_map().lock().unwrap().get(key).cloned()
}

fn info_or(key: &str, default: &str) -> String {
    info(key).unwrap_or_else(|| default.into())
}

// --- time ------------------------------------------------------------------------------------------------------------------------
/// (year, month, day, hour, minute, second) in UTC for a Unix time.
pub fn utc(secs: i64) -> (i64, i64, i64, i64, i64, i64) {
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
    (y, m, d, rem / 3600, rem % 3600 / 60, rem % 60)
}

pub fn now_secs() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs() as i64)
}

/// `2026-10-09 17:03:11` (Python: `time.strftime("%Y-%m-%d %H:%M:%S")`).
pub fn timestamp(secs: i64) -> String {
    let (y, mo, d, h, mi, s) = utc(secs);
    format!("{y:04}-{mo:02}-{d:02} {h:02}:{mi:02}:{s:02}")
}

/// `20261009-170311` (the bug-report file name).
pub fn file_stamp(secs: i64) -> String {
    let (y, mo, d, h, mi, s) = utc(secs);
    format!("{y:04}{mo:02}{d:02}-{h:02}{mi:02}{s:02}")
}

// --- the self-test report --------------------------------------------------------------------------------------------------------
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum Status {
    #[serde(rename = "PASS")]
    Pass,
    #[serde(rename = "WARN")]
    Warn,
    #[serde(rename = "FAIL")]
    Fail,
}

impl Status {
    pub fn as_str(self) -> &'static str {
        match self {
            Status::Pass => "PASS",
            Status::Warn => "WARN",
            Status::Fail => "FAIL",
        }
    }
}

/// One line of the self-test: a verdict with a plain-English fix for anything that is not a PASS.
#[derive(Clone, Debug, Serialize)]
pub struct Check {
    pub id: String,
    pub name: String,
    pub status: Status,
    pub detail: String,
    pub fix: String,
    pub seconds: f64,
    pub data: Value,
}

impl Check {
    pub fn new(id: &str, name: &str, status: Status, detail: impl Into<String>) -> Self {
        Check { id: id.into(), name: name.into(), status, detail: detail.into(), fix: String::new(), seconds: 0.0, data: json!({}) }
    }
    pub fn fix(mut self, fix: impl Into<String>) -> Self {
        self.fix = fix.into();
        self
    }
    pub fn data(mut self, data: Value) -> Self {
        self.data = data;
        self
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Counts {
    #[serde(rename = "PASS")]
    pub pass: usize,
    #[serde(rename = "WARN")]
    pub warn: usize,
    #[serde(rename = "FAIL")]
    pub fail: usize,
}

/// The report (`selftest.summarize`): same keys as Python's JSON, in the same order.
#[derive(Clone, Debug, Serialize)]
pub struct Report {
    pub format: &'static str,
    pub created: String,
    pub app_version: String,
    pub build: Value,
    pub counts: Counts,
    pub exit_code: i32,
    pub checks: Vec<Check>,
}

/// Exit codes: 0 all pass, 1 any FAIL, 3 warnings only (2 is left for bad usage).
pub fn summarize(checks: Vec<Check>, app_version: &str, build: Value) -> Report {
    let n = |s: Status| checks.iter().filter(|c| c.status == s).count();
    let counts = Counts { pass: n(Status::Pass), warn: n(Status::Warn), fail: n(Status::Fail) };
    let exit_code = if counts.fail > 0 {
        1
    } else if counts.warn > 0 {
        3
    } else {
        0
    };
    Report {
        format: "kick-the-fly-selftest",
        created: timestamp(now_secs()),
        app_version: app_version.into(),
        build,
        counts,
        exit_code,
        checks,
    }
}

impl Report {
    /// The plain-text form `--selftest` prints (`selftest.to_text`).
    pub fn to_text(&self) -> String {
        let mut lines = vec![format!("Kick the Fly {} self-test, {}", self.app_version, self.created), String::new()];
        for c in &self.checks {
            lines.push(format!("[{}] {}: {}", c.status.as_str(), c.name, c.detail));
            if !c.fix.is_empty() {
                lines.push(format!("       fix: {}", c.fix));
            }
        }
        let n = &self.counts;
        lines.push(String::new());
        lines.push(format!(
            "{} passed, {} warnings, {} failed. Exit code {} (0 all pass, 1 any FAIL, 3 warnings only).",
            n.pass, n.warn, n.fail, self.exit_code
        ));
        lines.join("\n")
    }

    /// `json.dumps(report, indent=1)`.
    pub fn to_json(&self) -> String {
        json_indent1(&serde_json::to_value(self).unwrap_or(Value::Null))
    }
}

/// Python's `json.dumps(v, indent=1)` (serde's pretty printer indents by two).
pub fn json_indent1(v: &Value) -> String {
    use serde::Serialize as _;
    let mut out = Vec::new();
    let fmt = serde_json::ser::PrettyFormatter::with_indent(b" ");
    let mut ser = serde_json::Serializer::with_formatter(&mut out, fmt);
    if v.serialize(&mut ser).is_err() {
        return String::new();
    }
    String::from_utf8(out).unwrap_or_default()
}

fn last_report() -> &'static Mutex<Option<Report>> {
    static M: OnceLock<Mutex<Option<Report>>> = OnceLock::new();
    M.get_or_init(Default::default)
}

/// The newest self-test report of this process (`selftest.LAST`); the bug report attaches it.
pub fn set_last_report(r: Report) {
    *last_report().lock().unwrap() = Some(r);
}

pub fn last_report_value() -> Option<Value> {
    last_report().lock().unwrap().as_ref().and_then(|r| serde_json::to_value(r).ok())
}

// --- this machine ----------------------------------------------------------------------------------------------------------------
/// `Ubuntu 22.04.4 LTS, kernel 6.8.0` / `Windows ...` / `macOS ...` (`crash.os_description`).
pub fn os_description() -> String {
    if cfg!(target_os = "linux") {
        let mut name = "Linux".to_string();
        if let Ok(text) = std::fs::read_to_string("/etc/os-release") {
            for line in text.lines() {
                if let Some(v) = line.strip_prefix("PRETTY_NAME=") {
                    name = v.trim().trim_matches('"').to_string();
                }
            }
        }
        let kernel = std::fs::read_to_string("/proc/sys/kernel/osrelease").unwrap_or_default();
        return format!("{name}, kernel {}", kernel.trim());
    }
    if cfg!(windows) {
        let mut cmd = std::process::Command::new("cmd");
        cmd.args(["/C", "ver"]);
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
        }
        if let Ok(out) = cmd.output() {
            let v = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !v.is_empty() {
                return v;
            }
        }
        return "Windows".into();
    }
    std::env::consts::OS.to_string()
}

pub fn session_description() -> String {
    if !cfg!(target_os = "linux") {
        return String::new();
    }
    let e = |k: &str| std::env::var(k).unwrap_or_default();
    format!(
        "XDG_SESSION_TYPE={} WAYLAND_DISPLAY={} DISPLAY={} SDL_VIDEODRIVER={}",
        e("XDG_SESSION_TYPE"),
        e("WAYLAND_DISPLAY"),
        e("DISPLAY"),
        e("SDL_VIDEODRIVER")
    )
}

/// The folder of the exe or AppImage the player launched (not the AppImage's mount).
pub fn beside_executable() -> Option<PathBuf> {
    if let Some(a) = std::env::var_os("APPIMAGE").filter(|s| !s.is_empty()) {
        return Path::new(&a).canonicalize().ok().and_then(|p| p.parent().map(Path::to_path_buf));
    }
    // a `cargo run` binary is in target/: not "beside the exe" in the sense of a shipped build
    let exe = std::env::current_exe().ok()?;
    if exe.components().any(|c| c.as_os_str() == "target") {
        return None;
    }
    exe.parent().map(Path::to_path_buf)
}

pub fn is_appimage() -> bool {
    std::env::var_os("APPIMAGE").is_some_and(|s| !s.is_empty())
}

/// The `build` block of the self-test JSON.
pub fn build_info(app_version: &str) -> Value {
    json!({
        "app_version": app_version,
        "engine": format!("Rust {} / Bevy 0.19", option_env!("CARGO_PKG_RUST_VERSION").unwrap_or("")),
        "appimage": is_appimage(),
        "os": os_description(),
        "machine": std::env::consts::ARCH,
        "executable": std::env::current_exe().map(|p| p.to_string_lossy().into_owned()).unwrap_or_default(),
    })
}

// --- crash reports ---------------------------------------------------------------------------------------------------------------
/// The crash report text: what helps to reproduce it first (`crash.report_text`).
pub fn crash_report_text(app_version: &str, what: &str) -> String {
    let mut lines = vec![
        format!("Kick the Fly {app_version} crash report, {}", timestamp(now_secs())),
        format!("seed: {}", info_or("seed", "unknown")),
        format!("mode: {}", info_or("mode", "unknown")),
        format!("sim backend: {}", info_or("sim_backend", "unknown")),
        format!("sim device: {}", info_or("sim_device", "unknown")),
        format!("OS: {}", os_description()),
        format!("engine: Rust, appimage: {}", is_appimage()),
    ];
    if cfg!(target_os = "linux") {
        lines.push(format!("session: {}", session_description()));
    }
    lines.push(format!("window system: {}", info_or("video_driver", "unknown")));
    lines.push(format!("GPU: {}", info_or("gpu_name", "not created (failed before the window)")));
    lines.push(format!("GPU backend: {}", info_or("gpu_backend", "")));
    lines.push(format!("GPU driver: {}", info_or("gpu_driver", "unknown")));
    let known = ["seed", "mode", "sim_backend", "sim_device", "video_driver", "gpu_name", "gpu_backend", "gpu_driver"];
    for (k, v) in info_map().lock().unwrap().iter() {
        if !known.contains(&k.as_str()) {
            lines.push(format!("{k}: {v}"));
        }
    }
    format!("{}\n\n{what}", lines.join("\n"))
}

/// Write the crash report next to the exe (when that folder is writable) and into the state folder. Returns the files written.
pub fn write_crash_report(app_version: &str, what: &str, state_dir: Option<&Path>) -> Vec<PathBuf> {
    let text = crash_report_text(app_version, what);
    let mut written = vec![];
    let state = state_dir.map(Path::to_path_buf).unwrap_or_else(|| crate::paths::get().state_dir);
    for d in [beside_executable(), Some(state)].into_iter().flatten() {
        if std::fs::create_dir_all(&d).is_ok() && std::fs::write(d.join(CRASH_NAME), &text).is_ok() {
            written.push(d.join(CRASH_NAME));
        }
    }
    written
}

// --- the log ---------------------------------------------------------------------------------------------------------------------
/// Start this session's log: the last one (which may be the one that crashed) is kept as `kickthefly.previous.log`.
pub fn open_log(state_dir: &Path) -> Option<std::fs::File> {
    std::fs::create_dir_all(state_dir).ok()?;
    let cur = state_dir.join(LOG_NAME);
    if std::fs::metadata(&cur).is_ok_and(|m| m.len() > 0) {
        let _ = std::fs::rename(&cur, state_dir.join(PREVIOUS_LOG));
    }
    std::fs::File::create(cur).ok()
}

fn read_lines(p: &Path) -> Vec<String> {
    std::fs::read(p).map(|b| String::from_utf8_lossy(&b).split_inclusive('\n').map(String::from).collect()).unwrap_or_default()
}

/// The last `n` lines of the log, reaching back into the previous session's log (`bugreport.tail_log`).
pub fn tail_log(state_dir: &Path, n: usize) -> String {
    let cur = read_lines(&state_dir.join(LOG_NAME));
    let prev = read_lines(&state_dir.join(PREVIOUS_LOG));
    if cur.len() >= n || prev.is_empty() {
        return cur[cur.len().saturating_sub(n)..].concat();
    }
    let room = n - cur.len() - 1;
    if room == 0 {
        return cur[cur.len().saturating_sub(n)..].concat();
    }
    let mut out = prev[prev.len().saturating_sub(room)..].concat();
    out.push_str("----- this session -----\n");
    out.push_str(&cur.concat());
    out
}

/// The newest crash report a crash left, in the state folder or beside the exe.
pub fn find_crash_report(state_dir: &Path) -> Option<PathBuf> {
    [beside_executable(), Some(state_dir.to_path_buf())]
        .into_iter()
        .flatten()
        .map(|d| d.join(CRASH_NAME))
        .filter(|p| p.exists())
        .max_by_key(|p| std::fs::metadata(p).and_then(|m| m.modified()).ok())
}

// --- the bug report --------------------------------------------------------------------------------------------------------------
#[derive(Clone, Debug)]
pub struct Item {
    pub id: &'static str,
    pub title: String,
    pub text: String,
    /// Too long for a URL: goes to the attached file.
    pub long: bool,
    pub include: bool,
}

impl Item {
    /// `1,234 characters, 12 lines` / `empty`.
    pub fn size(&self) -> String {
        let n = self.text.chars().count();
        if n == 0 { "empty".into() } else { format!("{} characters, {} lines", group3(n), group3(self.text.matches('\n').count() + 1)) }
    }
}

/// `1234567` -> `1,234,567` (Python's `{:,}`).
pub fn group3(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// Replace the home folder and the user name with placeholders, so the preview is what leaves (`bugreport.scrub`).
pub fn scrub(text: &str) -> String {
    let mut text = text.to_string();
    let home = crate::paths::Bases::from_system().home.to_string_lossy().into_owned();
    for needle in [home.clone(), home.replace('\\', "/")] {
        if needle.len() > 3 {
            text = text.replace(&needle, "~");
        }
    }
    let user = std::env::var("USER").or_else(|_| std::env::var("USERNAME")).unwrap_or_default();
    if user.len() > 2 {
        text = text.replace(&format!("/{user}/"), "/<user>/").replace(&format!("\\{user}\\"), "\\<user>\\");
    }
    text
}

/// The summary item: app version, OS, backend, display session.
pub fn summary_text(app_version: &str, report: Option<&Value>) -> String {
    let mut lines = vec![
        format!("Kick the Fly {app_version}"),
        format!("OS: {}", os_description()),
        format!("Engine: Rust / Bevy  AppImage: {}", is_appimage()),
        format!("Compute backend: {} ({})", info_or("sim_backend", "not started in this session"), info_or("sim_device", "unknown device")),
        format!("Mode: {}", info_or("mode", "unknown")),
        format!("Display: {} {}", info_or("video_driver", "unknown"), session_description()).trim().to_string(),
        format!(
            "GPU: {}  Backend: {}  Driver: {}",
            info_or("gpu_name", "unknown"),
            info_or("gpu_backend", "unknown"),
            info_or("gpu_driver", "unknown")
        ),
    ];
    if let Some(r) = report {
        lines.push(format!("Self-test verdicts: {}", r["counts"]));
    }
    lines.join("\n")
}

/// The items a bug report could contain, scrubbed (`bugreport.collect`). `report`: the last self-test result, if there is one.
pub fn collect(app_version: &str, report: Option<&Value>, state_dir: &Path) -> Vec<Item> {
    let mut items =
        vec![Item { id: "summary", title: "Summary".into(), text: scrub(&summary_text(app_version, report)), long: false, include: true }];
    match report {
        Some(r) => {
            items.push(Item { id: "selftest", title: "Self-test (JSON)".into(), text: scrub(&json_indent1(r)), long: true, include: true })
        }
        None => items.push(Item { id: "selftest", title: "Self-test (JSON)".into(), text: String::new(), long: true, include: false }),
    }
    let log = scrub(&tail_log(state_dir, LOG_LINES));
    let has_log = !log.trim().is_empty();
    items.push(Item { id: "log", title: format!("Last {LOG_LINES} log lines"), text: log, long: true, include: has_log });
    let crash =
        find_crash_report(state_dir).and_then(|p| std::fs::read(p).ok()).map(|b| scrub(&String::from_utf8_lossy(&b))).unwrap_or_default();
    let has_crash = !crash.trim().is_empty();
    items.push(Item { id: "crash", title: "Crash report".into(), text: crash, long: true, include: has_crash });
    items
}

fn included(items: &[Item]) -> impl Iterator<Item = &Item> {
    items.iter().filter(|i| i.include && !i.text.trim().is_empty())
}

/// Everything included, as one Markdown text (also what "Save to file" writes).
pub fn clipboard_text(items: &[Item], note: &str) -> String {
    let mut parts = vec![];
    if !note.trim().is_empty() {
        parts.push(format!("**What happened**\n\n{}\n", note.trim()));
    }
    for it in included(items) {
        if it.id == "summary" {
            parts.push(format!("### {}\n\n{}\n", it.title, it.text.trim_end()));
        } else {
            parts.push(format!("### {}\n\n```\n{}\n```\n", it.title, it.text.trim_end()));
        }
    }
    format!("{}\n", parts.join("\n").trim_end())
}

/// One line per check that was not a PASS, short enough for a URL.
pub fn selftest_verdicts(items: &[Item]) -> String {
    for it in included(items) {
        if it.id == "selftest" {
            let Ok(rep) = serde_json::from_str::<Value>(&it.text) else { return String::new() };
            let lines: Vec<String> = rep["checks"]
                .as_array()
                .map(|cs| {
                    cs.iter()
                        .filter(|c| c["status"] != "PASS")
                        .map(|c| format!("- {}: {}", c["status"].as_str().unwrap_or(""), c["name"].as_str().unwrap_or("")))
                        .collect()
                })
                .unwrap_or_default();
            return if lines.is_empty() { "- all checks passed".into() } else { lines.join("\n") };
        }
    }
    String::new()
}

/// (title, body for the URL, text for the attachment or None).
pub fn compose_issue(items: &[Item], note: &str, attach_name: &str) -> (String, String, Option<String>) {
    let title = "Bug report: ".to_string();
    let what = if note.trim().is_empty() { "(please describe what you were doing)" } else { note.trim() };
    let mut body = vec![format!("**What happened**\n\n{what}\n")];
    let mut long_items = vec![];
    for it in included(items) {
        if it.long {
            long_items.push(it);
        } else {
            body.push(format!("### {}\n\n{}\n", it.title, it.text.trim_end()));
        }
    }
    let verdicts = selftest_verdicts(items);
    if !verdicts.is_empty() {
        body.push(format!("### Self-test (checks that were not PASS)\n\n{verdicts}\n"));
    }
    let mut attach = None;
    if !long_items.is_empty() {
        attach =
            Some(long_items.iter().map(|it| format!("===== {} =====\n{}\n", it.title, it.text.trim_end())).collect::<Vec<_>>().join("\n"));
        let names = long_items.iter().map(|i| i.title.as_str()).collect::<Vec<_>>().join(", ");
        let file = if attach_name.is_empty() { "the bug report file" } else { attach_name };
        body.push(format!("_Attached by hand: `{file}` ({names})._\n"));
    }
    (title, body.join("\n"), attach)
}

/// `urllib.parse.quote(s)`: everything but letters, digits and `_.-~/` is percent-encoded as UTF-8.
pub fn quote(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || b"_.-~/".contains(&b) {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

pub fn unquote(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%'
            && i + 2 < b.len()
            && let Ok(v) = u8::from_str_radix(&s[i + 1..i + 3], 16)
        {
            out.push(v);
            i += 3;
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// The prefilled GitHub issue URL, with the body cut (and marked) to fit `limit` characters.
pub fn issue_url(title: &str, body: &str, limit: usize) -> String {
    let base = format!("https://github.com/{REPO}/issues/new?title={}&body=", quote(title));
    let room = limit.saturating_sub(base.len()).max(200);
    let enc = quote(body);
    if enc.len() <= room {
        return base + &enc;
    }
    let mark = "\n\n_(cut to fit the URL: the rest is in the attached file)_";
    let mut cut: Vec<char> = body.chars().collect();
    while !cut.is_empty() && quote(&(cut.iter().collect::<String>() + mark)).len() > room {
        cut.truncate(cut.len() * 9 / 10);
    }
    base + &quote(&(cut.iter().collect::<String>() + mark))
}

/// Save text next to the logs as `bugreports/kickthefly-bugreport-<stamp>.txt`.
pub fn save_attachment(text: &str, folder: Option<&Path>) -> Option<PathBuf> {
    let folder = folder.map(Path::to_path_buf).unwrap_or_else(|| crate::paths::get().state_dir.join("bugreports"));
    std::fs::create_dir_all(&folder).ok()?;
    let p = folder.join(format!("kickthefly-bugreport-{}.txt", file_stamp(now_secs())));
    std::fs::write(&p, text).ok()?;
    Some(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quote_matches_python() {
        // urllib.parse.quote("a b/c?d&é~_.-") == 'a%20b/c%3Fd%26%C3%A9~_.-'
        assert_eq!(quote("a b/c?d&é~_.-"), "a%20b/c%3Fd%26%C3%A9~_.-");
        assert_eq!(unquote("a%20b/c%3Fd%26%C3%A9"), "a b/c?d&é");
    }

    #[test]
    fn utc_is_right() {
        assert_eq!(timestamp(0), "1970-01-01 00:00:00");
        assert_eq!(timestamp(1_760_000_000), "2025-10-09 08:53:20");
        assert_eq!(group3(1_234_567), "1,234,567");
        assert_eq!(group3(12), "12");
    }

    #[test]
    fn summary_exit_codes() {
        let c = |s| Check::new("x", "x", s, "d");
        assert_eq!(summarize(vec![c(Status::Pass)], "1", json!({})).exit_code, 0);
        assert_eq!(summarize(vec![c(Status::Pass), c(Status::Warn)], "1", json!({})).exit_code, 3);
        assert_eq!(summarize(vec![c(Status::Warn), c(Status::Fail)], "1", json!({})).exit_code, 1);
        let r = summarize(vec![c(Status::Warn).fix("do it")], "1.2", json!({}));
        assert!(r.to_text().contains("[WARN] x: d\n       fix: do it"));
        assert!(r.to_json().contains("\n \"format\": \"kick-the-fly-selftest\""));
    }

    #[test]
    fn report_text_and_issue() {
        let items = vec![
            Item { id: "summary", title: "Summary".into(), text: "Kick the Fly 4".into(), long: false, include: true },
            Item { id: "log", title: "Last 500 log lines".into(), text: "line\n".into(), long: true, include: true },
            Item { id: "crash", title: "Crash report".into(), text: "".into(), long: true, include: false },
        ];
        let t = clipboard_text(&items, " it broke ");
        assert_eq!(t, "**What happened**\n\nit broke\n\n### Summary\n\nKick the Fly 4\n\n### Last 500 log lines\n\n```\nline\n```\n");
        let (title, body, attach) = compose_issue(&items, "", "f.txt");
        assert_eq!(title, "Bug report: ");
        assert!(
            body.contains("(please describe what you were doing)") && body.contains("_Attached by hand: `f.txt` (Last 500 log lines)._")
        );
        assert_eq!(attach.unwrap(), "===== Last 500 log lines =====\nline\n");
        let big = "x".repeat(20_000);
        let url = issue_url("T", &big, URL_LIMIT);
        assert!(url.len() <= URL_LIMIT && url.contains("cut%20to%20fit"));
    }

    #[test]
    fn tail_log_reaches_into_the_previous_session() {
        let d = std::env::temp_dir().join(format!("ktf-diag-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join(PREVIOUS_LOG), "p1\np2\np3\n").unwrap();
        std::fs::write(d.join(LOG_NAME), "c1\nc2\n").unwrap();
        assert_eq!(tail_log(&d, 4), "p3\n----- this session -----\nc1\nc2\n");
        assert_eq!(tail_log(&d, 2), "c1\nc2\n");
        let _ = std::fs::remove_dir_all(&d);
    }
}
