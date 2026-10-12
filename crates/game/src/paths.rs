//! Where the game keeps its files. Python: `core/paths.py`: `KICK_THE_FLY_HOME` first, else the platform's data folder (XDG on Linux,
//! `%APPDATA%` on Windows, Application Support on macOS).

use std::path::PathBuf;

/// The folder of `scores.json`, `arcade_points.json` and the rest of the player's data: the same one the Python game used
/// (`ktf_data::paths`, which follows the platform's conventions through the `directories` crate).
pub fn data_dir() -> PathBuf {
    ktf_data::paths::get().data_dir
}

/// The best scores (puppeteer stars and the like), `scores.json`: `{ "<key>": <number> }`.
pub struct Scores {
    path: PathBuf,
    pub map: std::collections::BTreeMap<String, f64>,
}

impl Scores {
    pub fn open(path: PathBuf) -> Self {
        let map = std::fs::read_to_string(&path).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
        Self { path, map }
    }

    pub fn default_file() -> Self {
        Self::open(data_dir().join("scores.json"))
    }

    /// Keep the better of the old and new value (`high`: bigger is better). Returns whether it improved.
    pub fn record(&mut self, key: &str, value: f64, high: bool) -> bool {
        let better = self.map.get(key).is_none_or(|old| if high { value > *old } else { value < *old });
        if better {
            self.map.insert(key.to_string(), value);
            if let Some(dir) = self.path.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            let tmp = self.path.with_extension("tmp");
            if serde_json::to_string_pretty(&self.map).ok().and_then(|t| std::fs::write(&tmp, t).ok()).is_some() {
                let _ = std::fs::rename(&tmp, &self.path);
            }
        }
        better
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scores_keep_the_best_and_survive_a_reload() {
        let dir = std::env::temp_dir().join(format!("ktf-scores-{}", std::process::id()));
        let path = dir.join("scores.json");
        let mut s = Scores::open(path.clone());
        assert!(s.record("puppet_first_steps", 2.0, true));
        assert!(!s.record("puppet_first_steps", 1.0, true));
        assert!(s.record("puppet_first_steps", 3.0, true));
        assert!(s.record("escaperoom_speedrun", 40.0, false));
        assert!(s.record("escaperoom_speedrun", 31.5, false));
        let again = Scores::open(path);
        assert_eq!(again.map["puppet_first_steps"], 3.0);
        assert_eq!(again.map["escaperoom_speedrun"], 31.5);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn the_home_variable_wins() {
        // not set here in CI; when it is, its data folder is used
        if let Some(p) = std::env::var_os("KICK_THE_FLY_HOME") {
            assert_eq!(data_dir(), PathBuf::from(p).join("data"));
        } else {
            assert!(data_dir().ends_with("kickthefly") || data_dir().ends_with("Kick the Fly"));
        }
    }
}
