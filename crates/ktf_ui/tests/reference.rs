//! The Rust menus against screenshots of the pygame menus (`tests/ui_reference/*.png`, made by `legacy-python/make_ui_reference.py`).
//! Shapes and text positions must agree; the only differences allowed are anti-aliasing of glyph edges (pygame hints text, we do not) and
//! the host-supplied ENGINE lines of Settings > Brain.

use std::path::PathBuf;

use ktf_data::config::Config;
use ktf_ui::UiEvent;
use ktf_ui::menu::PageHost;
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

/// The game state the reference screenshots were made with (`make_ui_reference.py`): nothing changed, a three-tool loadout.
struct Mock;
impl PageHost for Mock {
    fn query(&self, key: &str) -> serde_json::Value {
        match key {
            "surgery_payload" => serde_json::json!({"groups": {}, "types": {}}),
            "loadout_tools" => serde_json::json!(["hand", "swatter", "torch", "freeze", "sugar"]),
            "selected_tool" => serde_json::json!("hand"),
            "lab_modified" => serde_json::json!({}),
            _ => serde_json::Value::Null,
        }
    }
    fn command(&mut self, _: &str, _: serde_json::Value) -> serde_json::Value {
        serde_json::Value::Null
    }
}

fn draw(screen: &str, tab: &str, three_d: bool, mutate: impl Fn(&mut Config)) -> Vec<u8> {
    draw_with(screen, tab, three_d, mutate, |_, _| {})
}

/// `drive` clicks through the menu before the frame that is compared (it gets the menu, its config and a way to draw).
fn draw_with(
    screen: &str,
    tab: &str,
    three_d: bool,
    mutate: impl Fn(&mut Config),
    drive: impl Fn(&mut Menu, &mut dyn FnMut(&mut Menu, Option<UiEvent>)),
) -> Vec<u8> {
    let mut cfg = Config::new(None);
    mutate(&mut cfg);
    let mut m = Menu::new("3.1.1");
    ktf_ui::pages::install(&mut m);
    m.show(screen);
    m.tab = tab.into();
    let mut cv = Canvas::new(1280, 720);
    let mut host = Mock;
    let mut frame = |m: &mut Menu, ev: Option<UiEvent>| {
        if let Some(ev) = ev {
            m.handle(&ev, &mut cfg, three_d, &mut host);
        }
        cv.clear();
        m.draw(&mut cv, &mut cfg, three_d, (-1, -1), 0.0, &mut host);
    };
    frame(&mut m, None);
    drive(&mut m, &mut frame);
    cv.to_rgb_over(BG)
}

fn click(m: &mut Menu, frame: &mut dyn FnMut(&mut Menu, Option<UiEvent>), id: &str) {
    let pos = m.center_of(id).unwrap_or_else(|| panic!("no widget {id}"));
    frame(m, Some(UiEvent::MouseDown { pos }));
    frame(m, Some(UiEvent::MouseUp { pos }));
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
    for tab in ["Graphics", "Audio", "Brain", "Controls", "Accessibility", "Help"] {
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

#[test]
fn whatsnew_page() {
    compare("page_whatsnew.png", &draw("whatsnew", "Graphics", true, |_| {}), 0.009);
}

#[test]
fn share_page_tabs() {
    compare("page_share_surgery.png", &draw("share", "Graphics", true, |_| {}), 0.009);
    compare("page_share_loadout.png", &draw_with("share", "Graphics", true, |_| {}, |m, f| click(m, f, "share-kind:loadout")), 0.009);
    compare("page_share_import.png", &draw_with("share", "Graphics", true, |_| {}, |m, f| click(m, f, "share-tab:import")), 0.009);
}

#[test]
fn loadout_editor_page() {
    compare("page_loadout_base.png", &draw("loadout", "Graphics", true, |_| {}), 0.009);
    compare(
        "page_loadout_all.png",
        &draw("loadout", "Graphics", true, |c| {
            c.set("controls.loadout_preset", &serde_json::json!("all")).unwrap();
        }),
        0.009,
    );
}
