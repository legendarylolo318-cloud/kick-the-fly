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
    compare_size(name, ours, max_share, (1280, 720));
}

fn compare_size(name: &str, ours: &[u8], max_share: f64, size: (u32, u32)) {
    let (theirs, w, h) = reference(name);
    assert_eq!((w, h), size);
    let mut bad = 0usize;
    let mut bands = vec![0usize; (h as usize).div_ceil(80)];
    for (i, (a, b)) in ours.as_chunks::<3>().0.iter().zip(theirs.as_chunks::<3>().0.iter()).enumerate() {
        if a.iter().zip(b).any(|(x, y)| (*x as i32 - *y as i32).abs() > 60) {
            bad += 1;
            bands[i / w as usize / 80] += 1;
        }
    }
    let share = bad as f64 / (w * h) as f64;
    eprintln!("{name}: {:.3}% differ", share * 100.0);
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

/// The Neurodex panel on the real pack: the type table and a deterministic progress, as `make_ui_reference.py:dex_host_parts` builds them.
fn dex_shot(selected: &str, only_found: bool) -> Option<Vec<u8>> {
    use fly_sim::neurodex::{RawCsr, TypeTable};
    use ktf_ui::pages::neurodex::{NeurodexPage, NeurodexShared};
    let path = fly_sim::pack::find_default()?;
    let pack = fly_sim::Pack::load(&path).unwrap();
    let mut npz = fly_sim::npy::Npz::read(std::io::BufReader::new(std::fs::File::open(&path).unwrap())).unwrap();
    let (indptr, indices, data) = (
        npz.take("indptr").unwrap().into_i32().unwrap(),
        npz.take("indices").unwrap().into_i32().unwrap(),
        npz.take("data").unwrap().into_i16().unwrap(),
    );
    let tab = TypeTable::build(&pack.labels, "adult", Some(RawCsr { indptr: &indptr, indices: &indices, data: &data }));
    let mut shared = NeurodexShared::new(Default::default());
    let names: Vec<String> = tab.names.iter().take(400).cloned().chain(["DNp01".to_string(), "LPLC2".to_string()]).collect();
    for (i, n) in names.iter().enumerate() {
        shared.progress.mark("adult", n, 2.5 + (i % 7) as f64 * 0.5, if i % 3 == 0 { "stimulated" } else { "play" }, "2026-01-02T03:04:05");
    }
    shared.table = Some(std::sync::Arc::new(tab));
    shared.state = "ready".into();
    shared.today = ktf_data::notd::toordinal(2026, 10, 9);
    let shared = std::sync::Arc::new(std::sync::Mutex::new(shared));
    let mut cfg = Config::new(None);
    let mut m = Menu::new("3.1.1");
    ktf_ui::pages::install_with(&mut m, shared);
    m.page_mut::<NeurodexPage>("neurodex").unwrap().select(selected, only_found);
    m.show("neurodex");
    let mut cv = Canvas::new(1280, 720);
    m.draw(&mut cv, &mut cfg, true, (-1, -1), 0.0, &mut Mock);
    Some(cv.to_rgb_over(BG))
}

#[test]
fn neurodex_panel() {
    let Some(a) = dex_shot("DNp01", true) else { return };
    compare("page_neurodex.png", &a, 0.012);
    compare("page_neurodex_unknown.png", &dex_shot("DNp02", false).unwrap(), 0.012);
}

/// The self-test, bug report and crash pages (`make_ui_reference.py` writes the fixture both sides draw).
struct DiagMock(serde_json::Value);
impl PageHost for DiagMock {
    fn query(&self, key: &str) -> serde_json::Value {
        match key {
            "selftest" => self.0["selftest"].clone(),
            "crash_paths" => {
                serde_json::json!(["/home/user/.local/state/kick-the-fly/KickTheFly-crash.txt", "/opt/kick/KickTheFly-crash.txt"])
            }
            _ => serde_json::Value::Null,
        }
    }
    fn command(&mut self, _: &str, _: serde_json::Value) -> serde_json::Value {
        serde_json::Value::Null
    }
}

fn diag_fixture() -> serde_json::Value {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/ui_reference/diag_fixture.json");
    serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap()
}

fn diag_items(fx: &serde_json::Value) -> Vec<ktf_data::diag::Item> {
    fx["items"]
        .as_array()
        .unwrap()
        .iter()
        .map(|i| ktf_data::diag::Item {
            id: ["summary", "selftest", "log", "crash"].into_iter().find(|n| *n == i["id"].as_str().unwrap()).unwrap(),
            title: i["title"].as_str().unwrap().into(),
            text: i["text"].as_str().unwrap().into(),
            long: i["long"].as_bool().unwrap(),
            include: i["include"].as_bool().unwrap(),
        })
        .collect()
}

fn diag_shot(screen: &str, size: (i32, i32), prep: impl Fn(&mut Menu)) -> Vec<u8> {
    let fx = diag_fixture();
    let mut host = DiagMock(fx);
    let mut cfg = Config::new(None);
    let mut m = Menu::new("3.1.1");
    ktf_ui::pages::install(&mut m);
    m.register_page("crash", Box::new(ktf_ui::pages::diag::CrashPage));
    prep(&mut m);
    m.show(screen);
    let mut cv = Canvas::new(size.0, size.1);
    m.draw(&mut cv, &mut cfg, true, (-1, -1), 0.0, &mut host);
    cv.to_rgb_over(BG)
}

#[test]
fn selftest_page() {
    compare("page_selftest.png", &diag_shot("selftest", (1280, 720), |_| {}), 0.009);
}

#[test]
fn bugreport_page() {
    let fx = diag_fixture();
    let note = "The fly froze when I used the bomb";
    for (name, sel, note, saved) in [
        ("page_bugreport_summary.png", 0, note, None),
        ("page_bugreport_log.png", 2, note, None),
        ("page_bugreport_copied.png", 4, note, None),
        (
            "page_bugreport_saved.png",
            3,
            "",
            Some("/home/user/.local/state/kick-the-fly/bugreports/kickthefly-bugreport-20261009-120000.txt"),
        ),
    ] {
        let shot = diag_shot("bugreport", (1280, 720), |m| {
            m.page_mut::<ktf_ui::pages::diag::BugReportPage>("bugreport").unwrap().preset(diag_items(&fx), sel, note, saved);
        });
        compare(name, &shot, 0.012);
    }
}

#[test]
fn crash_page() {
    compare_size("page_crash.png", &diag_shot("crash", (900, 620), |_| {}), 0.009, (900, 620));
}

#[test]
fn challenges_page() {
    struct H(serde_json::Value);
    impl PageHost for H {
        fn query(&self, key: &str) -> serde_json::Value {
            if key == "challenge_scores" { self.0.clone() } else { serde_json::Value::Null }
        }
        fn command(&mut self, _: &str, _: serde_json::Value) -> serde_json::Value {
            serde_json::Value::Null
        }
    }
    for (name, scores) in [
        (
            "page_challenges.png",
            serde_json::json!({"tmaze": 8, "sneak": 1.4, "sweet": 35, "reverse_surgery": 2, "puppeteer": 4, "predict": 7}),
        ),
        ("page_challenges_new.png", serde_json::json!({})),
    ] {
        let mut cfg = Config::new(None);
        let mut m = Menu::new("3.1.1");
        ktf_ui::pages::install(&mut m);
        m.show("challenges");
        let mut cv = Canvas::new(1280, 720);
        m.draw(&mut cv, &mut cfg, true, (-1, -1), 0.0, &mut H(scores));
        compare(name, &cv.to_rgb_over(BG), 0.012);
    }
}

#[test]
fn arcade_page() {
    use ktf_ui::pages::arcade::ArcadePage;
    struct H(serde_json::Value);
    impl PageHost for H {
        fn query(&self, key: &str) -> serde_json::Value {
            if key == "arcade" { self.0.clone() } else { serde_json::Value::Null }
        }
        fn command(&mut self, _: &str, _: serde_json::Value) -> serde_json::Value {
            serde_json::Value::Null
        }
    }
    let fx: serde_json::Value = serde_json::from_slice(
        &std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/ui_reference/arcade_fixture.json")).unwrap(),
    )
    .unwrap();
    for (name, tab, scroll) in [
        ("page_arcade_tournament_top.png", 0u8, 0),
        ("page_arcade_tournament_replay.png", 0, 330),
        ("page_arcade_tournament_end.png", 0, 560),
        ("page_arcade_race_top.png", 1, 0),
        ("page_arcade_race_lanes.png", 1, 100),
    ] {
        let mut cfg = Config::new(None);
        let mut m = Menu::new("3.1.1");
        ktf_ui::pages::install(&mut m);
        m.show("arcade");
        m.page_mut::<ArcadePage>("arcade").unwrap().preset(tab, Some(2003), Some((2, 0)), Some(3002));
        let mut cv = Canvas::new(1280, 720);
        m.draw(&mut cv, &mut cfg, true, (-1, -1), 0.0, &mut H(fx.clone())); // the first frame learns the content height the scroll clamps to
        m.set_scroll(if tab == 0 { "arcade_tournament" } else { "arcade_race" }, scroll);
        let mut cv = Canvas::new(1280, 720);
        m.draw(&mut cv, &mut cfg, true, (-1, -1), 0.0, &mut H(fx.clone()));
        if let Ok(dir) = std::env::var("KTF_DUMP") {
            let rgb = cv.to_rgb_over(BG);
            let file = std::fs::File::create(format!("{dir}/rust_{name}")).unwrap();
            let mut enc = png::Encoder::new(std::io::BufWriter::new(file), 1280, 720);
            enc.set_color(png::ColorType::Rgb);
            enc.set_depth(png::BitDepth::Eight);
            enc.write_header().unwrap().write_image_data(&rgb).unwrap();
        }
        compare(name, &cv.to_rgb_over(BG), 0.012);
    }
}

#[test]
fn arcade_battle_tab() {
    use ktf_ui::pages::arcade::ArcadePage;
    struct H(serde_json::Value, serde_json::Value);
    impl PageHost for H {
        fn query(&self, key: &str) -> serde_json::Value {
            match key {
                "arcade" => self.0.clone(),
                "arcade_sizes" => self.1["sizes"].clone(),
                "lesion_saved" => self.1["saved"].clone(),
                _ => serde_json::Value::Null,
            }
        }
        fn command(&mut self, _: &str, _: serde_json::Value) -> serde_json::Value {
            serde_json::Value::Null
        }
    }
    let read = |f: &str| -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(format!("{}/../../tests/ui_reference/{f}", env!("CARGO_MANIFEST_DIR"))).unwrap()).unwrap()
    };
    let mut arcade = read("arcade_fixture.json");
    let bt = read("arcade_battle_fixture.json");
    arcade["battle"] = bt["result"].clone();
    let a: &[(&str, i8)] = &[("DNp35", -1), ("MDN", 1)];
    let b: &[(&str, i8)] = &[("LC10a", 1)];
    let none: &[(&str, i8)] = &[];
    #[allow(clippy::type_complexity)]
    let cases: [(&str, u8, [&[(&str, i8)]; 2], &str, u32, f64, Option<(&str, bool)>); 5] = [
        ("setup1", 0, [a, none], "DN", 1, 20.0, None),
        (
            "setup2",
            2,
            [&[("DNp35", -1)], &[("LC10a", 1), ("LC10b", -1), ("LPLC2", 1)]],
            "LC",
            1,
            20.0,
            Some(("the budget is 3 picks: take one out first", true)),
        ),
        ("pass", 1, [a, none], "", 1, 20.0, None),
        ("ready", 3, [a, b], "", 3, 25.0, None),
        ("result", 5, [a, b], "", 1, 20.0, None),
    ];
    for (name, phase, picks, search, rounds, secs, msg) in cases {
        let mut cfg = Config::new(None);
        let mut m = Menu::new("3.1.1");
        ktf_ui::pages::install(&mut m);
        m.show("arcade");
        m.page_mut::<ArcadePage>("arcade").unwrap().preset_battle(phase, picks, search, rounds, secs, msg);
        let mut cv = Canvas::new(1280, 720);
        m.draw(&mut cv, &mut cfg, true, (-1, -1), 0.0, &mut H(arcade.clone(), bt.clone()));
        if let Ok(dir) = std::env::var("KTF_DUMP") {
            let rgb = cv.to_rgb_over(BG);
            let file = std::fs::File::create(format!("{dir}/rust_battle_{name}.png")).unwrap();
            let mut enc = png::Encoder::new(std::io::BufWriter::new(file), 1280, 720);
            enc.set_color(png::ColorType::Rgb);
            enc.set_depth(png::BitDepth::Eight);
            enc.write_header().unwrap().write_image_data(&rgb).unwrap();
        }
        compare(&format!("page_arcade_battle_{name}.png"), &cv.to_rgb_over(BG), 0.012);
    }
}

#[test]
fn lab_pages() {
    use ktf_ui::canvas::Rect;
    use ktf_ui::menu::{Ctx, Page};
    struct Stub;
    impl Page for Stub {
        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }
        fn draw(&mut self, _: &mut Menu, _: &mut Canvas, _: Rect, _: &mut Ctx) {}
        fn custom(&mut self, _: &mut Menu, _: &str, _: Option<&str>, _: &mut Ctx) {}
    }
    struct H(serde_json::Value, bool);
    impl PageHost for H {
        fn query(&self, key: &str) -> serde_json::Value {
            match key {
                "lab_pages" => self.0["pages"].clone(),
                "lab_params" => {
                    if self.1 {
                        self.0["params"].clone()
                    } else {
                        serde_json::json!({})
                    }
                }
                "lab_engine" => serde_json::json!("CPU (NumPy) · CPU"),
                _ => serde_json::Value::Null,
            }
        }
        fn command(&mut self, _: &str, _: serde_json::Value) -> serde_json::Value {
            serde_json::Value::Null
        }
    }
    let fx: serde_json::Value =
        serde_json::from_slice(&std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/ui_reference/lab_fixture.json")).unwrap())
            .unwrap();
    for (name, screen, modified, scroll) in [
        ("page_lab_hub.png", "lab", false, 0),
        ("page_lab_hub_modified.png", "lab", true, 0),
        ("page_lab_params.png", "lab_params", true, 0),
        ("page_lab_params_scrolled.png", "lab_params", true, 600),
        ("page_lab_assumptions.png", "lab_assumptions", false, 0),
        ("page_lab_assumptions_scrolled.png", "lab_assumptions", false, 700),
    ] {
        let mut cfg = Config::new(None);
        let mut m = Menu::new("3.1.1");
        ktf_ui::pages::install(&mut m);
        for p in fx["pages"].as_array().unwrap() {
            let n = p[1].as_str().unwrap();
            if !m.has_page(n) {
                m.register_page(n, Box::new(Stub));
            }
        }
        m.show(screen);
        let mut cv = Canvas::new(1280, 720);
        m.draw(&mut cv, &mut cfg, true, (-1, -1), 0.0, &mut H(fx.clone(), modified));
        m.set_scroll(screen, scroll);
        let mut cv = Canvas::new(1280, 720);
        m.draw(&mut cv, &mut cfg, true, (-1, -1), 0.0, &mut H(fx.clone(), modified));
        if let Ok(dir) = std::env::var("KTF_DUMP") {
            let rgb = cv.to_rgb_over(BG);
            let file = std::fs::File::create(format!("{dir}/rust_{name}")).unwrap();
            let mut enc = png::Encoder::new(std::io::BufWriter::new(file), 1280, 720);
            enc.set_color(png::ColorType::Rgb);
            enc.set_depth(png::BitDepth::Eight);
            enc.write_header().unwrap().write_image_data(&rgb).unwrap();
        }
        // a page of small text on dark cards: edges differ by glyph anti-aliasing (PORTING.md, 'Cannot be matched exactly')
        compare(name, &cv.to_rgb_over(BG), if screen == "lab_assumptions" { 0.02 } else { 0.012 });
    }
}

#[test]
fn lab_validation_page() {
    struct H(serde_json::Value);
    impl PageHost for H {
        fn query(&self, key: &str) -> serde_json::Value {
            if key == "validation_results" {
                serde_json::json!({"results": self.0, "source": "from this build", "job": null})
            } else if key == "lab_params" {
                serde_json::json!({})
            } else {
                serde_json::Value::Null
            }
        }
        fn command(&mut self, _: &str, _: serde_json::Value) -> serde_json::Value {
            serde_json::Value::Null
        }
    }
    let published: serde_json::Value = serde_json::from_slice(
        &std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/results/3.0-release/validate_after.json")).unwrap(),
    )
    .unwrap();
    for (name, scroll) in [("page_lab_validation.png", 0), ("page_lab_validation_scrolled.png", 700)] {
        let mut cfg = Config::new(None);
        let mut m = Menu::new("3.1.1");
        ktf_ui::pages::install(&mut m);
        m.show("lab_validation");
        let mut cv = Canvas::new(1280, 720);
        m.draw(&mut cv, &mut cfg, true, (-1, -1), 0.0, &mut H(published.clone()));
        m.set_scroll("lab_validation", scroll);
        let mut cv = Canvas::new(1280, 720);
        m.draw(&mut cv, &mut cfg, true, (-1, -1), 0.0, &mut H(published.clone()));
        compare(name, &cv.to_rgb_over(BG), 0.02);
    }
}

#[test]
fn lab_assays_page() {
    use ktf_ui::pages::lab::LabAssays;
    struct H(serde_json::Value, Option<String>);
    impl PageHost for H {
        fn query(&self, key: &str) -> serde_json::Value {
            match key {
                "assay_state" => {
                    let result = self.1.as_ref().map_or(serde_json::Value::Null, |k| {
                        let mut r = self.0["results"][k].clone();
                        r["label"] = serde_json::json!(match r["kind"].as_str().unwrap_or("") {
                            "tmaze" => "T-maze conditioning",
                            "looming" => "Looming escape",
                            _ => "Sugar response",
                        });
                        r
                    });
                    serde_json::json!({"job": null, "result": result, "options": self.0["options"], "kinds": self.0["kinds"], "base_seed": 1000})
                }
                _ => serde_json::Value::Null,
            }
        }
        fn command(&mut self, _: &str, _: serde_json::Value) -> serde_json::Value {
            serde_json::Value::Null
        }
    }
    let fx: serde_json::Value = serde_json::from_slice(
        &std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/ui_reference/lab_assays_fixture.json")).unwrap(),
    )
    .unwrap();
    for (name, kind, result, flies, surgery_i) in [
        ("page_lab_assays_empty.png", "tmaze", None, 6, 0),
        ("page_lab_assays_tmaze.png", "tmaze", Some("tmaze"), 6, 0),
        ("page_lab_assays_looming_surgery.png", "looming", Some("looming_surgery"), 6, 7),
        ("page_lab_assays_sugar.png", "sugar", Some("sugar"), 2, 0),
    ] {
        let mut cfg = Config::new(None);
        let mut m = Menu::new("3.1.1");
        ktf_ui::pages::install(&mut m);
        m.show("lab_assays");
        m.page_mut::<LabAssays>("lab_assays").unwrap().set_view(kind, flies, surgery_i, false);
        let mut cv = Canvas::new(1280, 720);
        m.draw(&mut cv, &mut cfg, true, (-1, -1), 0.0, &mut H(fx.clone(), result.map(String::from)));
        if let Ok(dir) = std::env::var("KTF_DUMP") {
            let rgb = cv.to_rgb_over(BG);
            let file = std::fs::File::create(format!("{dir}/rust_{name}")).unwrap();
            let mut enc = png::Encoder::new(std::io::BufWriter::new(file), 1280, 720);
            enc.set_color(png::ColorType::Rgb);
            enc.set_depth(png::BitDepth::Eight);
            enc.write_header().unwrap().write_image_data(&rgb).unwrap();
        }
        compare(name, &cv.to_rgb_over(BG), 0.02);
    }
}

#[test]
fn lab_asymmetry_and_benchmark_pages() {
    struct H(serde_json::Value);
    impl PageHost for H {
        fn query(&self, key: &str) -> serde_json::Value {
            if key == "benchmark_state" { serde_json::json!({"results": self.0, "status": null}) } else { serde_json::Value::Null }
        }
        fn command(&mut self, _: &str, _: serde_json::Value) -> serde_json::Value {
            serde_json::Value::Null
        }
    }
    let bench: serde_json::Value = serde_json::from_slice(
        &std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/ui_reference/lab_benchmark_fixture.json")).unwrap(),
    )
    .unwrap();
    for (name, screen, mirror) in [
        ("page_lab_asymmetry.png", "lab_asymmetry", false),
        ("page_lab_asymmetry_mirror.png", "lab_asymmetry", true),
        ("page_lab_benchmark.png", "lab_benchmark", false),
    ] {
        let mut cfg = Config::new(None);
        if mirror {
            cfg.set("brain.mirror_weights", &serde_json::json!(true)).unwrap();
        }
        let mut m = Menu::new("3.1.1");
        ktf_ui::pages::install(&mut m);
        m.show(screen);
        let mut cv = Canvas::new(1280, 720);
        m.draw(&mut cv, &mut cfg, true, (-1, -1), 0.0, &mut H(bench.clone()));
        compare(name, &cv.to_rgb_over(BG), 0.02);
    }
}

#[test]
fn lab_day4_pages() {
    struct H(serde_json::Value);
    impl PageHost for H {
        fn query(&self, key: &str) -> serde_json::Value {
            if key == "lab.day4_state" { self.0.clone() } else { serde_json::Value::Null }
        }
        fn command(&mut self, _: &str, _: serde_json::Value) -> serde_json::Value {
            serde_json::Value::Null
        }
    }
    let load = |n: &str| -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(format!("{}/../../tests/golden/{n}.json", env!("CARGO_MANIFEST_DIR"))).unwrap()).unwrap()
    };
    let (net, sleep, sens) = (load("netsci"), load("sleepdep"), load("sensitivity"));
    let empty =
        serde_json::json!({"job": null, "error": "", "msg": "", "net": {}, "cached": {"adult": false}, "sleep": null, "sens": null});
    for (name, screen, state) in [
        ("page_lab_netsci_empty.png", "lab_netsci", empty.clone()),
        (
            "page_lab_netsci.png",
            "lab_netsci",
            serde_json::json!({"job": null, "error": "", "msg": "", "net": {"adult": net}, "cached": {"adult": true}, "sleep": null, "sens": null}),
        ),
        ("page_lab_sleepdep_empty.png", "lab_sleepdep", empty.clone()),
        (
            "page_lab_sleepdep.png",
            "lab_sleepdep",
            serde_json::json!({"job": null, "error": "", "msg": "", "net": {}, "cached": {}, "sleep": sleep, "sens": null}),
        ),
        ("page_lab_sensitivity_empty.png", "lab_sensitivity", empty.clone()),
        (
            "page_lab_sensitivity.png",
            "lab_sensitivity",
            serde_json::json!({"job": null, "error": "", "msg": "", "net": {}, "cached": {}, "sleep": null, "sens": sens}),
        ),
    ] {
        let mut cfg = Config::new(None);
        let mut m = Menu::new("3.1.1");
        ktf_ui::pages::install(&mut m);
        m.show(screen);
        let mut cv = Canvas::new(1280, 720);
        m.draw(&mut cv, &mut cfg, true, (-1, -1), 0.0, &mut H(state));
        if let Ok(dir) = std::env::var("KTF_DUMP") {
            let rgb = cv.to_rgb_over(BG);
            let file = std::fs::File::create(format!("{dir}/rust_{name}")).unwrap();
            let mut enc = png::Encoder::new(std::io::BufWriter::new(file), 1280, 720);
            enc.set_color(png::ColorType::Rgb);
            enc.set_depth(png::BitDepth::Eight);
            enc.write_header().unwrap().write_image_data(&rgb).unwrap();
        }
        compare(name, &cv.to_rgb_over(BG), 0.02);
    }
}
