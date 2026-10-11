//! `lab/neurosearch.py` on the real pack (tests/golden/make_neurosearch.py): search results and the strongest paths equal Python's.
use fly_sim::neurosearch::{search, top_paths};
use fly_sim::pack::Pack;
use serde_json::Value;

#[test]
fn search_and_paths_equal_python() {
    let Some(path) = fly_sim::pack::find_default() else {
        eprintln!("no brain pack: skipped");
        return;
    };
    let pack = Pack::load(path).unwrap();
    let g: Value =
        serde_json::from_str(&std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/golden/neurosearch.json")).unwrap()).unwrap();
    for q in g["queries"].as_array().unwrap() {
        let text = q[0].as_str().unwrap();
        let want: Vec<u32> = q[1].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        assert_eq!(search(&pack.labels, text, 12), want, "search {text:?}");
        let want3: Vec<u32> = q[2].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        assert_eq!(search(&pack.labels, text, 3), want3, "search {text:?} limit 3");
    }
    let mut found = 0;
    for p in g["paths"].as_array().unwrap() {
        let (s, d) = (p["src"].as_u64().unwrap() as u32, p["dst"].as_u64().unwrap() as u32);
        let got = top_paths(&pack.weights, s, d, 5, 3);
        let want = p["paths"].as_array().unwrap();
        assert_eq!(got.len(), want.len(), "{s} -> {d}");
        for (g, w) in got.iter().zip(want) {
            let nodes: Vec<u32> = w["nodes"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
            assert_eq!(g.nodes, nodes, "{s} -> {d}");
            assert!((g.strength - w["strength"].as_f64().unwrap()).abs() <= 1e-12 * g.strength.max(1e-30), "{s} -> {d}: {} vs {}", g.strength, w["strength"]);
            assert_eq!(i64::from(g.sign), w["sign"].as_i64().unwrap());
            found += 1;
        }
    }
    assert!(found > 20, "only {found} paths compared");
}
