//! Where Kick the Fly keeps its files on each OS. Python: `core/paths.py`; the layout is the same so a player's existing config, saves,
//! training memory and scores are found by the Rust game.
//!
//! ```text
//! Linux    config  $XDG_CONFIG_HOME/kickthefly/config.toml     data  $XDG_DATA_HOME/kickthefly (memory/, saves/)
//!          pictures <xdg PICTURES>/Kick the Fly                state $XDG_STATE_HOME/kickthefly
//! Windows  config  %APPDATA%\Kick the Fly\config.toml          data  Documents\Kick the Fly (memory\, saves\)
//!          pictures Pictures\Kick the Fly                      state %LOCALAPPDATA%\Kick the Fly
//! macOS    everything under ~/Library/Application Support/Kick the Fly
//! ```
//! `KICK_THE_FLY_HOME` puts everything under one folder (portable installs, tests, CI); `KICK_THE_FLY_MEMORY` overrides the memory
//! folder. The base folders come from the `directories` crate (XDG variables, Windows Known Folders), the layout above from Python.

use std::path::{Path, PathBuf};

pub const XDG_NAME: &str = "kickthefly";
pub const WIN_NAME: &str = "Kick the Fly";
pub const MEMORY_FILES: [&str; 2] = ["fly-memory.npz", "training-log.json"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    Linux,
    Windows,
    MacOs,
}

impl Platform {
    pub fn current() -> Self {
        if cfg!(target_os = "windows") {
            Platform::Windows
        } else if cfg!(target_os = "macos") {
            Platform::MacOs
        } else {
            Platform::Linux
        }
    }
}

/// The OS's own base folders (what `directories` reports), so [`resolve`] stays a pure function that tests can feed other OSes' values.
#[derive(Debug, Clone, Default)]
pub struct Bases {
    pub home: PathBuf,
    /// XDG_CONFIG_HOME / `%APPDATA%` (Roaming)
    pub config: PathBuf,
    /// XDG_DATA_HOME
    pub data: PathBuf,
    /// XDG_STATE_HOME (Linux)
    pub state: PathBuf,
    /// `%LOCALAPPDATA%`
    pub local: PathBuf,
    pub documents: PathBuf,
    pub pictures: PathBuf,
}

impl Bases {
    pub fn from_system() -> Self {
        let b = directories::BaseDirs::new();
        let u = directories::UserDirs::new();
        let home = b.as_ref().map(|b| b.home_dir().to_path_buf()).unwrap_or_else(|| PathBuf::from("."));
        Bases {
            config: b.as_ref().map(|b| b.config_dir().to_path_buf()).unwrap_or_else(|| home.join(".config")),
            data: b.as_ref().map(|b| b.data_dir().to_path_buf()).unwrap_or_else(|| home.join(".local/share")),
            state: b.as_ref().and_then(|b| b.state_dir().map(Path::to_path_buf)).unwrap_or_else(|| home.join(".local/state")),
            local: b.as_ref().map(|b| b.data_local_dir().to_path_buf()).unwrap_or_else(|| home.join("AppData/Local")),
            documents: u.as_ref().and_then(|u| u.document_dir().map(Path::to_path_buf)).unwrap_or_else(|| home.join("Documents")),
            pictures: u.as_ref().and_then(|u| u.picture_dir().map(Path::to_path_buf)).unwrap_or_else(|| home.join("Pictures")),
            home,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppPaths {
    pub config_dir: PathBuf,
    pub data_dir: PathBuf,
    pub memory_dir: PathBuf,
    pub saves_dir: PathBuf,
    pub pictures_dir: PathBuf,
    pub state_dir: PathBuf,
    pub notes: Vec<String>,
}

impl AppPaths {
    pub fn config_file(&self) -> PathBuf {
        self.config_dir.join("config.toml")
    }
}

/// Pure path resolution (no directories created).
pub fn resolve(platform: Platform, kick_home: Option<&Path>, memory_override: Option<&Path>, b: &Bases) -> AppPaths {
    let mut p = if let Some(r) = kick_home {
        AppPaths {
            config_dir: r.join("config"),
            data_dir: r.join("data"),
            memory_dir: r.join("data/memory"),
            saves_dir: r.join("data/saves"),
            pictures_dir: r.join("pictures"),
            state_dir: r.join("state"),
            notes: vec![],
        }
    } else {
        match platform {
            Platform::Windows => {
                let base = b.documents.join(WIN_NAME);
                let mut p = AppPaths {
                    config_dir: b.config.join(WIN_NAME),
                    data_dir: base.clone(),
                    memory_dir: base.join("memory"),
                    saves_dir: base.join("saves"),
                    pictures_dir: b.pictures.join(WIN_NAME),
                    state_dir: b.local.join(WIN_NAME),
                    notes: vec![],
                };
                // never move Windows data: if the memory only exists where older versions put it, keep using that folder
                if !p.memory_dir.join(MEMORY_FILES[0]).exists() {
                    let old = b.home.join("Documents").join(WIN_NAME).join("memory");
                    if old != p.memory_dir && old.join(MEMORY_FILES[0]).exists() {
                        p.notes.push(format!("using existing training memory in {}", old.display()));
                        p.memory_dir = old;
                    }
                }
                p
            }
            Platform::MacOs => {
                let sup = b.home.join("Library/Application Support").join(WIN_NAME);
                AppPaths {
                    config_dir: sup.clone(),
                    data_dir: sup.clone(),
                    memory_dir: sup.join("memory"),
                    saves_dir: sup.join("saves"),
                    pictures_dir: b.home.join("Pictures").join(WIN_NAME),
                    state_dir: b.home.join("Library/Logs").join(WIN_NAME),
                    notes: vec![],
                }
            }
            Platform::Linux => {
                let data = b.data.join(XDG_NAME);
                AppPaths {
                    config_dir: b.config.join(XDG_NAME),
                    memory_dir: data.join("memory"),
                    saves_dir: data.join("saves"),
                    data_dir: data,
                    pictures_dir: b.pictures.join(WIN_NAME),
                    state_dir: b.state.join(XDG_NAME),
                    notes: vec![],
                }
            }
        }
    };
    if let Some(m) = memory_override {
        p.memory_dir = m.to_path_buf();
    }
    p
}

/// The folders for this machine, from the environment.
pub fn get() -> AppPaths {
    let home = std::env::var_os("KICK_THE_FLY_HOME").filter(|s| !s.is_empty()).map(PathBuf::from);
    let mem = std::env::var_os("KICK_THE_FLY_MEMORY").filter(|s| !s.is_empty()).map(PathBuf::from);
    resolve(Platform::current(), home.as_deref(), mem.as_deref(), &Bases::from_system())
}

/// The first of these folders that exists or can be created, else the next (read-only homes, full disks).
pub fn ensure_dir(preferred: &Path, fallbacks: &[&Path]) -> PathBuf {
    for d in std::iter::once(preferred).chain(fallbacks.iter().copied()) {
        if std::fs::create_dir_all(d).is_ok() {
            return d.to_path_buf();
        }
    }
    preferred.to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bases() -> Bases {
        Bases {
            home: "/h".into(),
            config: "/h/.config".into(),
            data: "/h/.local/share".into(),
            state: "/h/.local/state".into(),
            local: "/h/AppData/Local".into(),
            documents: "/h/Documents".into(),
            pictures: "/h/Pictures".into(),
        }
    }

    #[test]
    fn linux_layout_matches_python() {
        let p = resolve(Platform::Linux, None, None, &bases());
        assert_eq!(p.config_file(), Path::new("/h/.config/kickthefly/config.toml"));
        assert_eq!(p.memory_dir, Path::new("/h/.local/share/kickthefly/memory"));
        assert_eq!(p.saves_dir, Path::new("/h/.local/share/kickthefly/saves"));
        assert_eq!(p.state_dir, Path::new("/h/.local/state/kickthefly"));
        assert_eq!(p.pictures_dir, Path::new("/h/Pictures/Kick the Fly"));
    }

    #[test]
    fn windows_layout_matches_python() {
        let mut b = bases();
        b.config = "C:/Users/me/AppData/Roaming".into();
        b.local = "C:/Users/me/AppData/Local".into();
        b.documents = "C:/Users/me/OneDrive/Documents".into();
        let p = resolve(Platform::Windows, None, None, &b);
        assert_eq!(p.config_file(), Path::new("C:/Users/me/AppData/Roaming/Kick the Fly/config.toml"));
        assert_eq!(p.saves_dir, Path::new("C:/Users/me/OneDrive/Documents/Kick the Fly/saves"));
        assert_eq!(p.state_dir, Path::new("C:/Users/me/AppData/Local/Kick the Fly"));
    }

    #[test]
    fn home_override_and_memory_override() {
        let p = resolve(Platform::Linux, Some(Path::new("/portable")), Some(Path::new("/mem")), &bases());
        assert_eq!(p.config_file(), Path::new("/portable/config/config.toml"));
        assert_eq!(p.data_dir, Path::new("/portable/data"));
        assert_eq!(p.memory_dir, Path::new("/mem"));
    }
}
