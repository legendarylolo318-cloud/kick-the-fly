//! The Rust menus against screenshots of the pygame menus (`tests/ui_reference/*.png`, made by `legacy-python/make_ui_reference.py`).
//! Shapes and text positions must agree; the only differences allowed are anti-aliasing of glyph edges (pygame hints text, we do not) and
//! the host-supplied ENGINE lines of Settings > Brain.

use std::path::PathBuf;

use ktf_data::config::Config;
use ktf_ui::{Canvas, Menu};

const BG: [u8; 3] = [60, 70, 80];

fn reference(name: &str) -> (Vec<u8>, u32, u32) {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/ui_reference").join(name);
    let dec = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(&p).unwrap_or_else(|e| panic!("{p:?}: {e}"))));
    let mut r = dec.read_info().unwrap();
    let mut buf = vec![0; r.output_buffer_size().unwrap()];
    let info = r.next_frame(&mut buf).unwrap();
    assert_eq!(info.color_type, png::ColorType::Rgb);
    (buf[..info.buffer_size()].to_vec(), info.width, info.height)
}

fn draw(screen: &str, tab: &str, three_d: bool, mutate: impl Fn(&mut Config)) -> Vec<u8> {
    let mut cfg = Config::new(None);
    mutate(&mut cfg);
    let mut m = Menu::new("3.1.1");
    m.show(screen);
    m.tab = tab.into();
    let mut cv = Canvas::new(1280, 720);
    m.draw(&mut cv, &cfg, three_d, (-1, -1), 0.0);
    cv.to_rgb_over(BG)
}

/// Share of pixels that differ clearly (any channel by more than 60) and the largest such share in any 80-px band of rows.
fn compare(name: &str, ours: &[u8], max_share: f64) {
    let (theirs, w, h) = reference(name);
    assert_eq!((w, h), (1280, 720));
    let mut bad = 0usize;
    let mut bands = vec![0usize; (h as usize).div_ceil(80)];
    for (i, (a, b)) in ours.as_chunks::<3>().0.iter().zip(theirs.as_chunks::<3>().0.iter()).enumerate() {
        if a.iter().zip(b).any(|(x, y)| (*x as i32 - *y as i32).abs() > 60) {
            bad += 1;
            bands[i / 1280 / 80] += 1;
        }
    }
    let share = bad as f64 / (w * h) as f64;
    assert!(
        share < max_share,
        "{name}: {:.3}% of pixels differ clearly (limit {:.3}%); per 80-row band: {bands:?}",
        share * 100.0,
        max_share * 100.0
    );
}

#[test]
fn pause_and_quit_dialog() {
    compare("pause.png", &draw("pause", "Graphics", true, |_| {}), 0.003);
    compare("confirm_quit.png", &draw("confirm_quit", "Graphics", true, |_| {}), 0.003);
    compare(
        "pause_lab.png",
        &draw("pause", "Graphics", true, |c| {
            c.set("brain.mode", &serde_json::json!("lab")).unwrap();
        }),
        0.003,
    );
}

#[test]
fn every_settings_tab() {
    for tab in ["Graphics", "Audio", "Brain", "Controls", "Accessibility"] {
        compare(&format!("settings_{}.png", tab.to_lowercase()), &draw("settings", tab, true, |_| {}), 0.009);
    }
    compare("settings_controls_2d.png", &draw("settings", "Controls", false, |_| {}), 0.009);
    compare(
        "settings_graphics_big.png",
        &draw("settings", "Graphics", true, |c| {
            c.set("access.larger_text", &serde_json::json!(true)).unwrap();
        }),
        0.009,
    );
}
