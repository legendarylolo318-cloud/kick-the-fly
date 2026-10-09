//! `cargo run -p ktf_ui --example render_menu -- [OUT_DIR]`: draws the same screens as `legacy-python/make_ui_reference.py`.
use std::path::{Path, PathBuf};

use ktf_data::config::Config;
use ktf_ui::{Canvas, Menu};

pub const BG: [u8; 3] = [60, 70, 80];

pub fn render(screen: &str, tab: &str, out: &Path, three_d: bool, mutate: impl Fn(&mut Config)) {
    let mut cfg = Config::new(None);
    mutate(&mut cfg);
    let mut m = Menu::new("3.1.1");
    m.show(screen);
    m.tab = tab.into();
    let mut cv = Canvas::new(1280, 720);
    m.draw(&mut cv, &cfg, three_d, (-1, -1), 0.0);
    let file = std::fs::File::create(out).unwrap();
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), 1280, 720);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header().unwrap().write_image_data(&cv.to_rgb_over(BG)).unwrap();
}

fn main() {
    let out: PathBuf = std::env::args().nth(1).map(PathBuf::from).unwrap_or_else(|| PathBuf::from("tests/ui_reference/rust"));
    std::fs::create_dir_all(&out).unwrap();
    render("pause", "Graphics", &out.join("pause.png"), true, |_| {});
    render("confirm_quit", "Graphics", &out.join("confirm_quit.png"), true, |_| {});
    for tab in ["Graphics", "Audio", "Brain", "Controls", "Accessibility"] {
        render("settings", tab, &out.join(format!("settings_{}.png", tab.to_lowercase())), true, |_| {});
    }
    render("settings", "Controls", &out.join("settings_controls_2d.png"), false, |_| {});
    render("pause", "Graphics", &out.join("pause_lab.png"), true, |c| {
        c.set("brain.mode", &serde_json::json!("lab")).unwrap();
    });
    render("settings", "Graphics", &out.join("settings_graphics_big.png"), true, |c| {
        c.set("access.larger_text", &serde_json::json!(true)).unwrap();
    });
}
