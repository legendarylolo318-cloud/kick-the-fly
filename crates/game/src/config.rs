//! Game-wide settings and the shared clock/arena state the plugins read. Python: `core/config.py` (the subset the ported features use).
//! `Settings::from_config` reads the player's `config.toml` (schema v3, shared with the Python game, see `ktf_data::config`).

use bevy::prelude::*;

use crate::world::ARENAS;

/// Which arena is loaded (index into [`ARENAS`]).
#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Arena(pub usize);

impl Arena {
    pub fn name(&self) -> &'static str {
        ARENAS[self.0 % ARENAS.len()]
    }
}

/// Written when the player asks for another arena (`E`); the arena plugin rebuilds the scenery.
#[derive(Message, Debug, Clone, Copy)]
pub struct SwitchArena(pub usize);

#[derive(Resource, Debug, Clone)]
pub struct Settings {
    pub master_volume: f32,
    /// Settings > Audio: the wing buzz and every other effect have their own volume under the master (`Sound.configure`).
    pub sfx_volume: f32,
    pub buzz_volume: f32,
    /// The one-off buzz and the popup when the fly sings (`brain.song_buzz`).
    pub song_buzz: bool,
    /// Settings > Brain > Aggression lunges.
    pub lunge: bool,
    /// `--day SECONDS` on the command line wins over Settings > Day/night cycle.
    pub day_cli: Option<f32>,
    pub mute: bool,
    /// Accessibility: no strobing lightning, no flashes.
    pub reduced_flashing: bool,
    /// Length of the day/night cycle in seconds (0 = off, the Lab's fixed sun).
    pub day_night_s: f32,
    pub mouse_sensitivity: f32,
    pub invert_y: bool,
    pub fov_deg: f32,
    pub show_profiler: bool,
    pub immortal: bool,
    /// Rain 0..1, gusts per second, storm on: the Lab's `weather.*` parameters (outdoor arenas).
    pub rain: f32,
    pub gust_hz: f32,
    pub storm: bool,
    pub wind_speed: f32,
    pub wind_dir_deg: f32,
    pub killcam: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            master_volume: 0.7,
            sfx_volume: 1.0,
            buzz_volume: 1.0,
            song_buzz: true,
            lunge: true,
            day_cli: None,
            mute: false,
            reduced_flashing: false,
            day_night_s: 0.0,
            mouse_sensitivity: 0.0024,
            invert_y: false,
            fov_deg: 70.0,
            show_profiler: std::env::var_os("KTF_PROFILER").is_some(),
            immortal: false,
            rain: 0.0,
            gust_hz: 0.0,
            storm: false,
            wind_speed: 0.0,
            wind_dir_deg: 180.0,
            killcam: true,
        }
    }
}

impl Settings {
    /// Radians per pixel at sensitivity 1.0 (Python: 0.0024).
    pub const BASE_SENSITIVITY: f32 = 0.0024;

    /// The values the ported features read, taken from the player's config (the rest of the file is kept as is by `ktf_data`).
    pub fn from_config(c: &ktf_data::config::Config) -> Self {
        let mut s = Self::default();
        s.apply_config(c);
        s
    }

    /// Copy the settings the game reads live out of the config (after the menu changed one, or at start).
    pub fn apply_config(&mut self, c: &ktf_data::config::Config) {
        self.master_volume = c.get_f32("audio.master");
        self.sfx_volume = c.get_f32("audio.sfx");
        self.buzz_volume = c.get_f32("audio.buzz");
        self.song_buzz = c.get_bool("brain.song_buzz");
        self.lunge = c.get_bool("brain.lunge");
        self.day_night_s = self.day_cli.unwrap_or(c.get_i64("brain.day_night") as f32);
        self.mute = c.get_bool("audio.mute");
        self.reduced_flashing = c.get_bool("access.reduced_flashing");
        self.mouse_sensitivity = Self::BASE_SENSITIVITY * c.get_f32("controls.mouse_sensitivity");
        self.invert_y = c.get_bool("controls.invert_y");
        self.fov_deg = c.get_f32("controls.fov");
        self.immortal = c.get_bool("brain.immortal");
        self.killcam = c.get_bool("brain.killcam");
    }
}

/// Settings > Graphics > Display backend (`platform_env.choose_backend`; Linux only): should this run be forced onto X11 / XWayland?
/// winit takes Wayland whenever `WAYLAND_DISPLAY` is set, so 'auto' and 'wayland' need nothing; 'x11' needs that variable gone, and
/// only makes sense when an X display exists.
pub fn force_x11(configured: &str, wayland_display: bool, x_display: bool) -> bool {
    cfg!(target_os = "linux") && configured == "x11" && wayland_display && x_display
}

#[cfg(test)]
mod backend_tests {
    #[test]
    #[cfg(target_os = "linux")]
    fn x11_is_only_forced_when_asked_for_and_possible() {
        use super::force_x11;
        assert!(force_x11("x11", true, true));
        assert!(!force_x11("auto", true, true));
        assert!(!force_x11("wayland", true, true));
        assert!(!force_x11("x11", false, true), "already X11");
        assert!(!force_x11("x11", true, false), "no X display to fall back to");
    }
}
