//! Share codes: short copyable text for a surgery, a loadout, a protocol, a challenge setup, Lab parameters or a contraption.
//! Python: `core/sharecode.py`. The format is version 1 and must stay readable forever: `KTF1-<KIND>-<body>`, the body is Crockford
//! base32 in groups of five of `version | kind | raw-deflate(canonical JSON) | first 4 bytes of SHA-256(all before it)`.
//!
//! Differences from Python: the deflate stream is whatever `flate2` produces (valid for Python's reader, not byte-identical to zlib
//! level 9), and canonical JSON prints floats the way `serde_json` does (Python's `repr` differs only for exponents like `1e-05`).
//! Both read each other's codes; the checksum is over the compressed bytes, so it travels with them.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

pub const VERSION: u8 = 1;
pub const MAX_CODE_CHARS: usize = 1200;
pub const MAX_PAYLOAD_BYTES: usize = 256 * 1024;
pub const FILE_SUFFIX: &str = ".ktfshare";
const GROUP: usize = 5;
const CROCKFORD: &[u8; 32] = b"0123456789ABCDEFGHJKMNPQRSTVWXYZ";
const KINDS: [(&str, &str); 6] = [
    ("surgery", "SRG"),
    ("loadout", "LDT"),
    ("protocol", "PRT"),
    ("challenge", "CHL"),
    ("lab", "LAB"),
    ("contraption", "CTN"),
];
/// The part kinds a contraption may hold (`game/contraption.py:KINDS`).
const CONTRAPTION_KINDS: [&str; 11] = [
    "marble", "ramp", "spring", "domino", "fan", "lamp", "sugar", "tool", "button", "timer", "fly",
];
const MAX_PARTS: usize = 24;

/// A code that can't be used; `Display` is the reason shown to the player (the same words as Python's).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShareError {
    Refused(String),
    /// Too long for a chat message: export it as a file. Holds the kind, and the character count.
    TooLarge {
        kind: String,
        chars: usize,
    },
}

impl std::fmt::Display for ShareError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShareError::Refused(s) => f.write_str(s),
            ShareError::TooLarge { kind, chars } => write!(
                f,
                "this {} is too big for a code ({} characters; the limit is {}); it can be exported as a file instead",
                kind_label(kind).to_lowercase(),
                group_digits(*chars),
                group_digits(MAX_CODE_CHARS)
            ),
        }
    }
}
impl std::error::Error for ShareError {}

fn refuse<T>(s: impl Into<String>) -> Result<T, ShareError> {
    Err(ShareError::Refused(s.into()))
}

fn group_digits(n: usize) -> String {
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

pub fn kind_label(kind: &str) -> &'static str {
    match kind {
        "surgery" => "Brain surgery",
        "loadout" => "Tool loadout",
        "protocol" => "Protocol",
        "challenge" => "Challenge setup",
        "lab" => "Lab parameters",
        _ => "Contraption build",
    }
}

fn kind_tag(kind: &str) -> Option<&'static str> {
    KINDS.iter().find(|k| k.0 == kind).map(|k| k.1)
}
fn kind_of_tag(tag: &str) -> Option<&'static str> {
    KINDS.iter().find(|k| k.1 == tag).map(|k| k.0)
}
fn kind_byte(kind: &str) -> Option<u8> {
    KINDS.iter().position(|k| k.0 == kind).map(|i| i as u8 + 1)
}
fn byte_kind(b: u8) -> Option<&'static str> {
    (b as usize)
        .checked_sub(1)
        .and_then(|i| KINDS.get(i))
        .map(|k| k.0)
}

#[derive(Debug, Clone, PartialEq)]
pub struct Code {
    pub version: u8,
    pub kind: String,
    pub payload: Value,
}

// --- encoding ---------------------------------------------------------------------------------------------------------------

/// Sorted keys, no spaces, ASCII only (`json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=True)`). `serde_json`'s map
/// is a `BTreeMap` (the `preserve_order` feature is off), so keys come out sorted.
pub fn canonical(payload: &Value) -> Vec<u8> {
    let s = serde_json::to_string(payload).expect("a Value always serializes");
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if c.is_ascii() {
            out.push(c);
        } else {
            let mut buf = [0u16; 2];
            for u in c.encode_utf16(&mut buf) {
                out.push_str(&format!("\\u{:04x}", u));
            }
        }
    }
    out.into_bytes()
}

fn deflate(raw: &[u8]) -> Vec<u8> {
    let mut e = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::best());
    e.write_all(raw).expect("writing to a Vec");
    e.finish().expect("finishing a Vec")
}

fn inflate(blob: &[u8]) -> Result<Vec<u8>, ShareError> {
    let mut out = Vec::new();
    let mut d = flate2::read::DeflateDecoder::new(blob).take(MAX_PAYLOAD_BYTES as u64 + 1);
    if d.read_to_end(&mut out).is_err() {
        return refuse("the code's contents are damaged (they don't decompress)");
    }
    if out.len() > MAX_PAYLOAD_BYTES {
        return refuse("the code decompresses to more than this version accepts");
    }
    Ok(out)
}

fn to_b32(data: &[u8]) -> String {
    let mut out = String::new();
    let (mut acc, mut bits) = (0u32, 0u32);
    for &b in data {
        acc = (acc << 8) | b as u32;
        bits += 8;
        while bits >= 5 {
            bits -= 5;
            out.push(CROCKFORD[((acc >> bits) & 31) as usize] as char);
        }
        acc &= (1 << bits) - 1;
    }
    if bits > 0 {
        out.push(CROCKFORD[((acc << (5 - bits)) & 31) as usize] as char);
    }
    out
}

fn from_b32(text: &str) -> Result<Vec<u8>, ShareError> {
    let bad =
        || ShareError::Refused("the code has characters that aren't part of a share code".into());
    // RFC 4648 base32 (as Python's b32decode, padded) only exists for lengths that are 0, 2, 4, 5 or 7 mod 8
    if matches!(text.len() % 8, 1 | 3 | 6) {
        return Err(bad());
    }
    let mut out = Vec::new();
    let (mut acc, mut bits) = (0u32, 0u32);
    for c in text.bytes() {
        let v = CROCKFORD.iter().position(|&a| a == c).ok_or_else(bad)? as u32;
        acc = (acc << 5) | v;
        bits += 5;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    Ok(out)
}

fn checksum(data: &[u8]) -> [u8; 4] {
    let h = Sha256::digest(data);
    [h[0], h[1], h[2], h[3]]
}

/// The share code for a payload. `limit` None writes any length (for files).
pub fn encode(kind: &str, payload: &Value, limit: Option<usize>) -> Result<String, ShareError> {
    let (Some(tag), Some(kb)) = (kind_tag(kind), kind_byte(kind)) else {
        return refuse(format!("unknown kind '{kind}'"));
    };
    if let Some(problem) = check_payload(kind, payload) {
        return refuse(problem);
    }
    let mut data = vec![VERSION, kb];
    data.extend(deflate(&canonical(payload)));
    let sum = checksum(&data);
    data.extend(sum);
    let body = to_b32(&data);
    let groups: Vec<&str> = (0..body.len())
        .step_by(GROUP)
        .map(|i| &body[i..(i + GROUP).min(body.len())])
        .collect();
    let text = format!("KTF{VERSION}-{tag}-{}", groups.join("-"));
    if let Some(l) = limit
        && text.len() > l
    {
        return Err(ShareError::TooLarge {
            kind: kind.into(),
            chars: text.len(),
        });
    }
    Ok(text)
}

/// Parse and verify a code (or the text of a `.ktfshare` file). Does not apply it and does not check it against this install:
/// call [`validate`] for that.
pub fn decode(text: &str) -> Result<Code, ShareError> {
    let s: String = text.split_whitespace().collect::<String>().to_uppercase();
    if s.is_empty() {
        return refuse("nothing to import: paste a share code");
    }
    let not_code = || {
        ShareError::Refused("that doesn't look like a share code (they start with KTF1-)".into())
    };
    let rest = s.strip_prefix("KTF").ok_or_else(not_code)?;
    let digits = rest.chars().take_while(|c| c.is_ascii_digit()).count();
    if !(1..=3).contains(&digits) {
        return Err(not_code());
    }
    let version: u32 = rest[..digits].parse().map_err(|_| not_code())?;
    let rest = rest[digits..].strip_prefix('-').ok_or_else(not_code)?;
    let (kind3, body) = (
        rest.get(..3).ok_or_else(not_code)?,
        rest.get(3..)
            .and_then(|b| b.strip_prefix('-'))
            .ok_or_else(not_code)?,
    );
    if !kind3.bytes().all(|b| b.is_ascii_uppercase()) || body.is_empty() {
        return Err(not_code());
    }
    let body: String = body
        .replace('-', "")
        .chars()
        .map(|c| match c {
            'O' => '0',
            'I' | 'L' => '1',
            c => c,
        })
        .collect();
    if version > VERSION as u32 {
        return refuse(format!(
            "this code was made by a newer version (code format {version}; this one reads up to {VERSION}): update Kick the Fly to import it"
        ));
    }
    if version < 1 {
        return refuse("unsupported code format");
    }
    let Some(kind) = kind_of_tag(kind3) else {
        return refuse(format!("unknown kind '{kind3}' in the code"));
    };
    let raw = from_b32(&body)?;
    if raw.len() < 7 {
        return refuse("the code is too short: part of it is missing");
    }
    let (data, check) = raw.split_at(raw.len() - 4);
    if checksum(data) != check {
        return refuse(
            "the code is damaged (its checksum doesn't match): a character was mistyped or part is missing",
        );
    }
    if data[0] as u32 != version || byte_kind(data[1]) != Some(kind) {
        return refuse("the code's header doesn't match its contents");
    }
    let blob = inflate(&data[2..])?;
    let payload: Value = match serde_json::from_slice(&blob) {
        Ok(v) if blob.is_ascii() => v,
        _ => return refuse("the code's contents are damaged (not readable data)"),
    };
    if !payload.is_object() {
        return refuse("the code's contents are not what a share code carries");
    }
    Ok(Code {
        version: version as u8,
        kind: kind.into(),
        payload,
    })
}

/// Write a code of any length to `folder/<stem>.ktfshare` (never overwriting: a number is added).
pub fn export_file(code_text: &str, folder: &Path, stem: &str) -> std::io::Result<PathBuf> {
    std::fs::create_dir_all(folder)?;
    let mut clean = String::new();
    for c in stem.chars() {
        let ok = c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-');
        if ok {
            clean.push(c);
        } else if !clean.ends_with('-') {
            clean.push('-');
        }
    }
    let clean: String = clean
        .trim_matches(|c| c == '-' || c == '.')
        .chars()
        .take(40)
        .collect();
    let clean = if clean.is_empty() {
        "share".to_string()
    } else {
        clean
    };
    let mut p = folder.join(format!("{clean}{FILE_SUFFIX}"));
    let mut k = 2;
    while p.exists() {
        p = folder.join(format!("{clean}-{k}{FILE_SUFFIX}"));
        k += 1;
    }
    std::fs::write(&p, format!("{code_text}\n"))?;
    Ok(p)
}

/// `(Some(code), None)` when it fits, else `(None, Some(path of the exported file))`.
pub fn encode_or_file(
    kind: &str,
    payload: &Value,
    folder: &Path,
    stem: &str,
) -> Result<(Option<String>, Option<PathBuf>), ShareError> {
    match encode(kind, payload, Some(MAX_CODE_CHARS)) {
        Ok(t) => Ok((Some(t), None)),
        Err(ShareError::TooLarge { .. }) => {
            let text = encode(kind, payload, None)?;
            let p = export_file(&text, folder, stem)
                .map_err(|e| ShareError::Refused(format!("can't write the file: {e}")))?;
            Ok((None, Some(p)))
        }
        Err(e) => Err(e),
    }
}

/// What Esc > Share > Import does with its text box: a pasted code, or the path of a `.ktfshare` file.
pub fn read_any(text_or_path: &str) -> Result<Code, ShareError> {
    let s = text_or_path.trim().trim_matches('"');
    if !s.is_empty() && !s.contains('\n') && s.len() < 1024 && !s.to_uppercase().starts_with("KTF")
    {
        let p = PathBuf::from(s);
        if p.extension()
            .is_some_and(|e| format!(".{}", e.to_string_lossy().to_lowercase()) == FILE_SUFFIX)
        {
            return match std::fs::read(&p) {
                Ok(b) => decode(&String::from_utf8_lossy(&b)),
                Err(e) => refuse(format!(
                    "can't read {}: {e}",
                    p.file_name()
                        .map(|n| n.to_string_lossy())
                        .unwrap_or_default()
                )),
            };
        }
    }
    decode(s)
}

// --- checking a payload (this version's schema) ------------------------------------------------------------------------------

/// One line of printable text, as the loadout editor's text box allows.
pub fn clean_name(name: &str, limit: usize) -> String {
    let s: String = name
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect();
    s.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .take(limit)
        .collect()
}

fn is_pm1(v: &Value) -> bool {
    v.as_f64().is_some_and(|x| x == 1.0 || x == -1.0)
}
fn is_mode_map(v: &Value) -> bool {
    v.as_object()
        .is_some_and(|m| m.iter().all(|(k, v)| !k.is_empty() && is_pm1(v)))
}
fn extra_keys(m: &Map<String, Value>, allowed: &[&str]) -> Vec<String> {
    m.keys()
        .filter(|k| !allowed.contains(&k.as_str()))
        .cloned()
        .collect()
}
fn py_list(v: &[String]) -> String {
    format!(
        "[{}]",
        v.iter()
            .map(|s| format!("'{s}'"))
            .collect::<Vec<_>>()
            .join(", ")
    )
}
fn len_of(v: Option<&Value>) -> usize {
    v.and_then(|v| v.as_object()).map_or(0, |m| m.len())
}

/// A reason this payload is malformed for its kind, or None. Needs nothing from the running game. For protocols and contraptions
/// this checks the container only; the per-part ranges and the protocol schema are checked where those are ported
/// (`contraption.rs`, the Lab's protocol loader) when the code is validated for import.
pub fn check_payload(kind: &str, p: &Value) -> Option<String> {
    let Some(m) = p.as_object() else {
        return Some("the contents are not a mapping".into());
    };
    match kind {
        "surgery" => {
            let extra = extra_keys(m, &["groups", "types"]);
            if !extra.is_empty() {
                return Some(format!(
                    "unknown fields in a surgery code: {}",
                    py_list(&extra)
                ));
            }
            let empty = Value::Object(Map::new());
            if !(is_mode_map(m.get("groups").unwrap_or(&empty))
                && is_mode_map(m.get("types").unwrap_or(&empty)))
            {
                return Some(
                    "a surgery code's switches must each be -1 (silence) or 1 (stimulate)".into(),
                );
            }
            let (g, t) = (len_of(m.get("groups")), len_of(m.get("types")));
            if g == 0 && t == 0 {
                return Some("this surgery code switches nothing".into());
            }
            if g + t > 200 {
                return Some("this surgery code switches more than 200 things".into());
            }
        }
        "loadout" => {
            let extra = extra_keys(m, &["name", "tools"]);
            if !extra.is_empty() {
                return Some(format!(
                    "unknown fields in a loadout code: {}",
                    py_list(&extra)
                ));
            }
            let tools_ok = m
                .get("tools")
                .and_then(|t| t.as_array())
                .is_some_and(|t| !t.is_empty() && t.len() <= 40 && t.iter().all(|x| x.is_string()));
            if !tools_ok {
                return Some("a loadout code needs a list of tool names".into());
            }
            let name = match m.get("name") {
                None => "",
                Some(Value::String(s)) => s,
                Some(_) => return Some("a loadout's name is at most 24 characters".into()),
            };
            if name.chars().count() > 24 {
                return Some("a loadout's name is at most 24 characters".into());
            }
            if name != clean_name(name, 24) {
                return Some("a loadout's name is one line of printable text".into());
            }
        }
        "protocol" => {
            if !extra_keys(m, &["protocol"]).is_empty()
                || !m.get("protocol").is_some_and(|p| p.is_object())
            {
                return Some("a protocol code carries one protocol".into());
            }
        }
        "challenge" => {
            let extra = extra_keys(m, &["challenge", "seed", "arena", "surgery", "params"]);
            if !extra.is_empty() {
                return Some(format!(
                    "unknown fields in a challenge code: {}",
                    py_list(&extra)
                ));
            }
            if !m.get("challenge").is_some_and(|c| c.is_string()) {
                return Some("a challenge code names its challenge".into());
            }
            if let Some(s) = m.get("seed")
                && !s.as_i64().is_some_and(|n| (0..1 << 31).contains(&n))
            {
                return Some("the seed must be a whole number from 0 to 2147483647".into());
            }
            if m.get("arena").is_some_and(|a| !a.is_string()) {
                return Some("the arena must be a name".into());
            }
            if let Some(s) = m.get("surgery")
                && let Some(r) = check_payload("surgery", s)
            {
                return Some(r);
            }
            if let Some(pr) = m.get("params") {
                let mut o = Map::new();
                o.insert("params".into(), pr.clone());
                if let Some(r) = check_payload("lab", &Value::Object(o)) {
                    return Some(r);
                }
            }
        }
        "contraption" => {
            let extra = extra_keys(m, &["name", "parts"]);
            if !extra.is_empty() {
                return Some(format!(
                    "unknown fields in a contraption: {}",
                    py_list(&extra)
                ));
            }
            let Some(parts) = m
                .get("parts")
                .and_then(|p| p.as_array())
                .filter(|p| !p.is_empty())
            else {
                return Some("a contraption needs at least one part".into());
            };
            match m.get("name") {
                None => {}
                Some(Value::String(s)) if s.chars().count() <= 24 => {}
                _ => return Some("a contraption's name is at most 24 characters".into()),
            }
            if parts.len() > MAX_PARTS + 1 {
                return Some(format!("a contraption has at most {MAX_PARTS} parts"));
            }
            for (i, r) in parts.iter().enumerate() {
                if !r
                    .get("k")
                    .and_then(|k| k.as_str())
                    .is_some_and(|k| CONTRAPTION_KINDS.contains(&k))
                {
                    return Some(format!("part {}: not a part this version has", i + 1));
                }
            }
        }
        "lab" => {
            let params = m
                .get("params")
                .and_then(|p| p.as_object())
                .filter(|p| !p.is_empty());
            if !extra_keys(m, &["params"]).is_empty() || params.is_none() {
                return Some("a Lab code carries a list of parameter values".into());
            }
            for (k, v) in params.into_iter().flatten() {
                let ok =
                    v.as_f64().is_some_and(|x| x.is_finite() && x.abs() <= 1e9) && v.is_number();
                if !ok {
                    return Some(format!("parameter '{k}' must be a number"));
                }
            }
        }
        _ => return Some(format!("unknown kind '{kind}'")),
    }
    None
}

// --- validating against this install, and previewing -------------------------------------------------------------------------

/// What this install knows. A field left `None` means "can't check that here".
#[derive(Debug, Default, Clone)]
pub struct Context {
    pub tools: Option<Vec<String>>,
    /// name → (lo, hi)
    pub lab_params: Option<std::collections::BTreeMap<String, (f64, f64)>>,
    pub current_params: std::collections::BTreeMap<String, f64>,
    pub surgery_labels: Option<Vec<String>>,
    pub type_names: Option<Vec<String>>,
    pub challenges: Option<Vec<String>>,
    pub arenas: Option<Vec<String>>,
    pub current_surgery_switches: usize,
    pub current_loadout_len: usize,
    pub saved_loadouts: Vec<String>,
    pub contraption_slots_used: usize,
    pub max_saved_loadouts: usize,
    pub protocol_names: Vec<String>,
    pub larva: bool,
    pub lab: bool,
    pub brain: String,
}

fn unknown_of(map: Option<&Value>, known: &Option<Vec<String>>) -> Vec<String> {
    match (map.and_then(|m| m.as_object()), known) {
        (Some(m), Some(k)) => m
            .keys()
            .filter(|x| !k.contains(x))
            .take(5)
            .cloned()
            .collect(),
        _ => vec![],
    }
}

fn validate_surgery(p: &Value, ctx: &Context) -> Option<String> {
    let u = unknown_of(p.get("groups"), &ctx.surgery_labels);
    if !u.is_empty() {
        return Some(format!(
            "this surgery switches things this version doesn't have: {}",
            u[..u.len().min(3)].join(", ")
        ));
    }
    let u = unknown_of(p.get("types"), &ctx.type_names);
    if !u.is_empty() {
        let brain = if ctx.brain.is_empty() {
            "adult"
        } else {
            &ctx.brain
        };
        return Some(format!(
            "this surgery names cell types that aren't in the {brain} brain you are running: {}",
            u.join(", ")
        ));
    }
    None
}

fn validate_params(params: &Value, ctx: &Context) -> Option<String> {
    let bounds = ctx.lab_params.as_ref()?;
    for (name, v) in params.as_object()? {
        let Some(&(lo, hi)) = bounds.get(name) else {
            return Some(format!("unknown Lab parameter '{name}'"));
        };
        let x = v.as_f64().unwrap_or(f64::NAN);
        if !(lo <= x && x <= hi) {
            return Some(format!(
                "Lab parameter '{name}' = {} is outside {} to {}",
                g(x),
                g(lo),
                g(hi)
            ));
        }
    }
    None
}

/// Python's `{:g}` for the numbers shown to the player (6 significant digits, trailing zeros dropped).
pub fn g(x: f64) -> String {
    if x == 0.0 || !x.is_finite() {
        return format!("{x}");
    }
    let exp = x.abs().log10().floor() as i32;
    if !(-5..6).contains(&exp) {
        let s = format!("{:.5e}", x);
        let (m, e) = s.split_once('e').unwrap();
        let m = if m.contains('.') {
            m.trim_end_matches('0').trim_end_matches('.')
        } else {
            m
        };
        let e: i32 = e.parse().unwrap();
        return format!("{m}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs());
    }
    let decimals = (5 - exp).max(0) as usize;
    let s = format!("{:.*}", decimals, x);
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s
    }
}

/// A reason this code can't be imported here (unknown tool, cell type, parameter, out-of-range value...), or None.
pub fn validate(code: &Code, ctx: &Context) -> Option<String> {
    let p = &code.payload;
    if let Some(bad) = check_payload(&code.kind, p) {
        return Some(bad);
    }
    match code.kind.as_str() {
        "surgery" => return validate_surgery(p, ctx),
        "loadout" => {
            if let Some(tools) = &ctx.tools {
                let unknown: Vec<&str> = p["tools"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|t| t.as_str())
                    .filter(|t| !tools.iter().any(|x| x == t))
                    .take(5)
                    .collect();
                if !unknown.is_empty() {
                    return Some(format!(
                        "this loadout uses tools this version doesn't have: {}",
                        unknown.join(", ")
                    ));
                }
            }
        }
        "lab" => return validate_params(&p["params"], ctx),
        "protocol" => {
            let name = p["protocol"]
                .get("name")
                .cloned()
                .unwrap_or(Value::String("shared-protocol".into()));
            let ok = name.as_str().is_some_and(|n| {
                n == clean_name(n, 60)
                    && !n.contains('/')
                    && !n.contains('\\')
                    && !n.starts_with('.')
            });
            if !ok {
                return Some(
                    "the protocol's name must be plain text without path separators".into(),
                );
            }
        }
        "challenge" => {
            let name = p["challenge"].as_str().unwrap_or("");
            if ctx
                .challenges
                .as_ref()
                .is_some_and(|c| !c.iter().any(|x| x == name))
            {
                return Some(format!("unknown challenge '{name}'"));
            }
            if let (Some(arenas), Some(a)) = (&ctx.arenas, p.get("arena").and_then(|a| a.as_str()))
                && !arenas.iter().any(|x| x == a)
            {
                return Some(format!("unknown arena '{a}'"));
            }
            if let Some(s) = p.get("surgery")
                && let Some(r) = validate_surgery(s, ctx)
            {
                return Some(r);
            }
            if let Some(pr) = p.get("params") {
                return validate_params(pr, ctx);
            }
        }
        _ => {}
    }
    None
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Preview {
    pub title: String,
    pub lines: Vec<String>,
    pub warnings: Vec<String>,
}

fn mode(v: &Value) -> &'static str {
    if v.as_f64().unwrap_or(0.0) < 0.0 {
        "silence"
    } else {
        "stimulate"
    }
}
fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    c.next()
        .map(|f| f.to_uppercase().collect::<String>() + c.as_str())
        .unwrap_or_default()
}

fn surgery_lines(p: &Value, ctx: &Context) -> Vec<String> {
    let mut lines = vec![];
    for (label, m) in p
        .get("groups")
        .and_then(|g| g.as_object())
        .into_iter()
        .flatten()
    {
        lines.push(format!("{}: {label}", capitalize(mode(m))));
    }
    for (t, m) in p
        .get("types")
        .and_then(|g| g.as_object())
        .into_iter()
        .flatten()
    {
        lines.push(format!("{} every {t} neuron", capitalize(mode(m))));
    }
    let now = ctx.current_surgery_switches;
    if now > 0 {
        lines.push(format!(
            "Replaces the surgery now active ({now} switch{})",
            if now != 1 { "es" } else { "" }
        ));
    }
    lines
}

fn param_lines(params: &Value, ctx: &Context) -> Vec<String> {
    params
        .as_object()
        .into_iter()
        .flatten()
        .map(|(name, v)| {
            let x = v.as_f64().unwrap_or(0.0);
            match ctx.current_params.get(name) {
                Some(old) => format!("{name}: {} -> {}", g(*old), g(x)),
                None => format!("{name} = {}", g(x)),
            }
        })
        .collect()
}

/// Plain-language lines saying exactly what importing will change. Call [`validate`] first. (Python's loadout line asks the loadout
/// class which tools really end up on the hotbar; here duplicates are dropped and the hand goes first, and `ctx.tools` filtering is
/// the caller's.)
pub fn preview(code: &Code, ctx: &Context) -> Preview {
    let p = &code.payload;
    let (mut lines, mut warns) = (vec![], vec![]);
    let title = format!("{} (code format {})", kind_label(&code.kind), code.version);
    match code.kind.as_str() {
        "surgery" => lines.extend(surgery_lines(p, ctx)),
        "loadout" => {
            let name = p
                .get("name")
                .and_then(|n| n.as_str())
                .filter(|n| !n.is_empty())
                .unwrap_or("Shared loadout");
            let mut tools: Vec<String> = vec!["hand".into()];
            for t in p["tools"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|t| t.as_str())
            {
                if !tools.iter().any(|x| x == t) {
                    tools.push(t.into());
                }
            }
            lines.push(format!("Hotbar becomes: {}", tools.join(", ")));
            if ctx.current_loadout_len > 0 {
                lines.push(format!(
                    "Replaces the custom loadout ({} tools) and selects the Custom preset",
                    ctx.current_loadout_len
                ));
            }
            let max = if ctx.max_saved_loadouts == 0 {
                5
            } else {
                ctx.max_saved_loadouts
            };
            if ctx.saved_loadouts.iter().any(|s| s == name) {
                lines.push(format!("Replaces your saved loadout '{name}'"));
            } else if ctx.saved_loadouts.len() < max {
                lines.push(format!("Saves it as '{name}'"));
            } else {
                warns.push(format!(
                    "All {max} loadout slots are full, so it won't be saved by name; it still becomes your custom loadout"
                ));
            }
            if ctx.larva {
                warns.push("Tools with no larval sensory mapping stay hidden in larva mode".into());
            }
        }
        "lab" => lines.extend(param_lines(&p["params"], ctx)),
        "contraption" => {
            let mut kinds: Vec<(String, usize)> = vec![];
            for r in p["parts"].as_array().into_iter().flatten() {
                let k = r["k"].as_str().unwrap_or("").to_string();
                match kinds.iter_mut().find(|x| x.0 == k) {
                    Some(x) => x.1 += 1,
                    None => kinds.push((k, 1)),
                }
            }
            let name = p
                .get("name")
                .and_then(|n| n.as_str())
                .filter(|n| !n.is_empty())
                .unwrap_or("shared");
            let list: Vec<String> = kinds.iter().map(|(k, n)| format!("{n} x {k}")).collect();
            lines.push(format!("Contraption '{name}': {}", list.join(", ")));
            let mut tools: Vec<&str> = p["parts"]
                .as_array()
                .into_iter()
                .flatten()
                .filter(|r| r["k"] == "tool")
                .map(|r| r.get("tool").and_then(|t| t.as_str()).unwrap_or("swatter"))
                .collect();
            tools.sort();
            tools.dedup();
            if !tools.is_empty() {
                lines.push(format!(
                    "It fires: {} (at the fly's own risk: these are the game's tools)",
                    tools.join(", ")
                ));
            }
            if ctx.contraption_slots_used < 8 {
                lines.push(
                    "Saves it into the first free contraption slot (it does not run it)".into(),
                );
            } else {
                warns.push("All 8 contraption slots are full, so it won't be saved; paste the code in the builder to open it without saving".into());
            }
        }
        "protocol" => {
            let pr = &p["protocol"];
            let nm = pr
                .get("name")
                .and_then(|n| n.as_str())
                .unwrap_or("shared-protocol");
            lines.push(format!(
                "Adds the protocol '{nm}' to your protocols folder (it does not run it)"
            ));
            let what = match pr.get("assay") {
                Some(a) => format!(
                    "Assay {}",
                    a.as_str()
                        .map(String::from)
                        .unwrap_or_else(|| a.to_string())
                ),
                None => "Stimulus protocol".into(),
            };
            let seeds = pr
                .get("seeds")
                .or_else(|| pr.get("seed"))
                .map(|s| s.to_string())
                .unwrap_or_else(|| "default".into());
            lines.push(format!("{what}; seeds {seeds}"));
            if let Some(s) = pr
                .get("surgery")
                .and_then(|s| s.as_object())
                .filter(|s| !s.is_empty())
            {
                let items: Vec<String> = s
                    .iter()
                    .take(6)
                    .map(|(k, m)| format!("{k} {}", mode(m)))
                    .collect();
                lines.push(format!("Its surgery: {}", items.join(", ")));
            }
            if ctx.protocol_names.iter().any(|n| n == nm) {
                lines.push("A protocol with that name exists: this one is saved under a new name, yours is untouched".into());
            }
        }
        "challenge" => {
            lines.push(format!(
                "Starts the challenge '{}' now",
                p["challenge"].as_str().unwrap_or("")
            ));
            if let Some(s) = p.get("seed") {
                lines.push(format!("Sets the random seed to {s}"));
            }
            if let Some(a) = p.get("arena").and_then(|a| a.as_str()) {
                lines.push(format!("Switches the arena to {a}"));
            }
            if let Some(s) = p.get("surgery") {
                lines.extend(surgery_lines(s, ctx));
            }
            if let Some(pr) = p.get("params") {
                lines.extend(param_lines(pr, ctx));
            }
        }
        _ => {}
    }
    if matches!(code.kind.as_str(), "surgery" | "challenge") && ctx.larva {
        warns.push(
            "Cell types are matched by name; larval brains share few of the adult's names".into(),
        );
    }
    Preview {
        title,
        lines,
        warnings: warns,
    }
}
