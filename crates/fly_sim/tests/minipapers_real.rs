//! The mini-papers equal `lab/minipapers.py` (`tests/golden/make_minipapers.py`): the quick run of each paper on the real pack, the comparison
//! table, the text and the SVG.

use std::collections::BTreeMap;
use std::sync::Arc;

use fly_sim::minipapers as mp;
use fly_sim::pack;
use serde_json::Value;

#[test]
fn papers_equal_python() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/golden");
    let g: Value = serde_json::from_slice(&std::fs::read(format!("{root}/minipapers.json")).unwrap()).unwrap();
    assert_eq!(mp::papers().len(), 6);
    let answers: BTreeMap<String, String> = [
        ("looming_escape", "up"),
        ("mb_conditioning", "none"),
        ("sugar_feeding", "up"),
        ("antenna_grooming_circuit", "up"),
        ("adn_grooming_motor", "small"),
    ]
    .iter()
    .map(|(a, b)| (a.to_string(), b.to_string()))
    .collect();
    let pk = pack::find_default().map(|p| pack::Pack::load(&p).unwrap());
    let shared = pk.map(|p| (Arc::new(p.weights), Arc::new(p.labels)));
    for pid in ["von_reyn_2014", "tully_quinn_1985", "shiu_2024", "hampel_2015", "ohyama_2015"] {
        if shared.is_none() && pid != "ohyama_2015" {
            continue;
        }
        let mut res =
            mp::run_paper(pid, &mp::QUICK_SEEDS, shared.as_ref().map(|s| (&s.0, &s.1)), 1, |_, _, _| true, "2026-01-01 00:00:00").unwrap();
        res["seconds"] = serde_json::json!(0.0);
        let want = &g[pid]["result"];
        assert_eq!(res["full"], want["full"], "{pid}");
        assert_eq!(res["source"], want["source"], "{pid}");
        for (a, b) in res["questions"].as_array().unwrap().iter().zip(want["questions"].as_array().unwrap()) {
            assert_eq!(a["test"], b["test"]);
            assert_eq!(a["verdict"]["reproduced"], b["verdict"]["reproduced"], "{pid}");
            assert_eq!(a["verdict"]["basis"], b["verdict"]["basis"], "{pid}");
            assert_eq!(a["note"], b["note"], "{pid}");
            let (ma, mb) = (a["verdict"]["measured"].as_object().unwrap(), b["verdict"]["measured"].as_object().unwrap());
            for (k, v) in mb {
                if let (Some(x), Some(y)) = (ma[k].as_f64(), v.as_f64()) {
                    assert!((x - y).abs() <= 1e-6 * y.abs().max(1e-9), "{pid} {k}: {x} vs {y}");
                }
            }
            assert_eq!(a["pairs"].as_array().unwrap().len(), b["pairs"].as_array().unwrap().len());
        }
        let text = mp::render(&res, &answers);
        let want_text = g[pid]["text"].as_str().unwrap();
        assert_eq!(text, want_text, "{pid}");
        let svg_path = format!("{root}/minipapers_files/{pid}/minipaper_{pid}.svg");
        if let Ok(want_svg) = std::fs::read_to_string(&svg_path) {
            assert_eq!(mp::svg_pairs(&res, 560, 160), want_svg, "{pid}");
        }
        for (a, b) in mp::compare(&res, &answers).iter().zip(g[pid]["compare"].as_array().unwrap()) {
            for k in [
                "test",
                "your_hypothesis",
                "paper_direction",
                "model_reproduces_paper",
                "you_matched_the_paper",
                "you_matched_the_model",
                "basis",
                "note",
            ] {
                assert_eq!(a[k], b[k], "{pid} {k}");
            }
        }
    }
}

/// The Buridan paper runs the rig (individuality `subtle` cannot equal Python spike for spike; the rig itself is checked in `rigs_real`).
#[test]
fn buridan_paper_runs_the_rig() {
    let Some(path) = pack::find_default() else { return };
    let pk = pack::Pack::load(&path).unwrap();
    let (w, l) = (Arc::new(pk.weights), Arc::new(pk.labels));
    let res = mp::run_paper("colomb_2012", &[0, 1], Some((&w, &l)), 2, |_, _, _| true, "2026-01-01 00:00:00").unwrap();
    assert_eq!(res["source"], "live");
    let q = &res["questions"][0];
    assert_eq!(q["test"], "rig:buridan");
    assert_eq!(q["pairs"].as_array().unwrap().len(), 2);
    assert!(q["verdict"]["measured"]["deviation_stripes_deg"].as_f64().unwrap() < 90.0);
    assert!(mp::render(&res, &Default::default()).contains("deviation_stripes_deg"));
}
