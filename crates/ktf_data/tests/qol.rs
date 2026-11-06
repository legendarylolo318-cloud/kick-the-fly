//! Small quality-of-life settings: window geometry, name tags, the last-seen version, and that "Real flies do this too" stays off.

use ktf_data::config::{Config, WindowGeom};

fn reload(c: &Config) -> Config {
    Config::from_toml(&c.to_toml("test")).unwrap()
}

#[test]
fn window_geometry_round_trips_and_bad_values_are_ignored() {
    let mut c = Config::new(None);
    assert!(!c.to_toml("t").contains("[window]"), "nothing saved, nothing written");
    c.window = Some(WindowGeom { width: 1280, height: 720, pos: Some((-20, 40)) });
    assert_eq!(reload(&c).window, c.window);
    c.window = Some(WindowGeom { width: 900, height: 600, pos: None });
    assert_eq!(reload(&c).window, c.window);
    let bad = Config::from_toml("schema_version = 3\n[window]\nwidth = 5\nheight = 700\n").unwrap();
    assert_eq!(bad.window, None);
}

#[test]
fn name_tags_default_on_and_persist_off() {
    let mut c = Config::new(None);
    assert!(c.name_tags && reload(&c).name_tags);
    c.name_tags = false;
    assert!(!reload(&c).name_tags);
}

#[test]
fn last_version_is_kept() {
    let mut c = Config::new(None);
    assert_eq!(reload(&c).last_version, None);
    c.last_version = Some("4.0.0".into());
    assert_eq!(reload(&c).last_version.as_deref(), Some("4.0.0"));
}

#[test]
fn real_flies_cards_default_off_and_migrations_never_turn_them_on() {
    assert!(!Config::new(None).get_bool("brain.science_popups"));
    for v in [1, 2, 3] {
        for on in [false, true] {
            let text = format!("schema_version = {v}\n[brain]\nscience_popups = {on}\n");
            let c = Config::from_toml(&text).unwrap();
            // a v3 file keeps what the player chose; older files are migrated to off; none is ever forced on
            assert_eq!(c.get_bool("brain.science_popups"), on && v >= 2, "schema {v}, saved {on}");
        }
    }
    assert!(!Config::from_toml("schema_version = 1\n").unwrap().get_bool("brain.science_popups"));
    assert!(!Config::from_toml("").unwrap().get_bool("brain.science_popups"));
}

#[test]
fn the_new_palettes_are_valid_choices() {
    let mut c = Config::new(None);
    for p in ["viridis", "cividis"] {
        assert!(c.set("access.palette", &serde_json::json!(p)).is_ok());
    }
}
