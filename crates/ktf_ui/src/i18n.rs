//! `tr()` as in `core/i18n.py`: JSON catalogs (English and German), English fallback, then the string itself. Cell types, gene names and
//! citations are never passed through it.

use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

pub const AVAILABLE: [(&str, &str); 2] = [("en", "English"), ("de", "Deutsch (Machine-translated)")];

type Catalog = HashMap<String, String>;
struct State {
    lang: String,
    catalog: Catalog,
    fallback: Catalog,
}

fn state() -> &'static RwLock<State> {
    static S: OnceLock<RwLock<State>> = OnceLock::new();
    S.get_or_init(|| RwLock::new(State { lang: "en".into(), catalog: HashMap::new(), fallback: HashMap::new() }))
}

fn load(lang: &str) -> Catalog {
    let text = match lang {
        "en" => include_str!("../data/locales/en.json"),
        "de" => include_str!("../data/locales/de.json"),
        _ => return HashMap::new(),
    };
    let v: serde_json::Value = serde_json::from_str(text).unwrap_or_default();
    v.as_object()
        .map(|m| {
            m.iter().filter(|(k, _)| !k.starts_with('_')).filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string()))).collect()
        })
        .unwrap_or_default()
}

/// The 2-letter system language if the game has it, else "en".
pub fn detect_system_language() -> String {
    for var in ["LC_ALL", "LC_MESSAGES", "LANG"] {
        if let Ok(v) = std::env::var(var)
            && !v.is_empty()
        {
            let code = v.split('.').next().unwrap_or("").split('_').next().unwrap_or("").to_lowercase();
            if AVAILABLE.iter().any(|(c, _)| *c == code) {
                return code;
            }
        }
    }
    "en".into()
}

/// 'auto', 'en', 'de': returns the resolved code.
pub fn set_language(lang: &str) -> String {
    let resolved = if lang == "auto" { detect_system_language() } else { lang.to_lowercase() };
    let resolved = if AVAILABLE.iter().any(|(c, _)| *c == resolved) { resolved } else { "en".into() };
    let fallback = load("en");
    let catalog = if resolved == "en" { fallback.clone() } else { load(&resolved) };
    *state().write().unwrap() = State { lang: resolved.clone(), catalog, fallback };
    resolved
}

pub fn language() -> String {
    state().read().unwrap().lang.clone()
}

pub fn tr(text: &str) -> String {
    let s = state().read().unwrap();
    s.catalog.get(text).or_else(|| s.fallback.get(text)).cloned().unwrap_or_else(|| text.to_string())
}

/// `tr(text, **kwargs)`: translate, then fill the `{name}` placeholders.
pub fn tr_args(text: &str, args: &[(&str, &str)]) -> String {
    let mut s = tr(text);
    for (k, v) in args {
        s = s.replace(&format!("{{{k}}}"), v);
    }
    s
}
