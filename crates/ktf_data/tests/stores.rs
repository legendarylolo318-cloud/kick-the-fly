//! The small stores, against files the Python modules wrote (`tests/python_files/stores/`).

use std::path::PathBuf;

use ktf_data::stores::*;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/python_files/stores")
}
fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ktf-stores-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn neurodex_from_python() {
    let d = tmp("dex");
    std::fs::copy(dir().join("neurodex.json"), d.join("neurodex.json")).unwrap();
    let mut p = Progress::open(d.join("neurodex.json"));
    assert!(p.warnings.is_empty());
    assert_eq!(p.n_discovered("adult"), 2);
    assert_eq!(p.brains["adult"]["LC4"], Discovery { t: "2026-01-02T03:04:05".into(), x: 3.5, how: "stimulated".into() });
    assert!(p.discovered("larva", "Ka"));
    assert!(!p.mark("adult", "LC4", 9.0, "play", "later"), "the first record stays");
    assert!(p.mark("adult", "T4", 0.126, "play", "2026-03-01T00:00:00"));
    assert!(p.save());
    // the file is still plain JSON with sorted keys and indent 1, as Python writes it
    let text = std::fs::read_to_string(d.join("neurodex.json")).unwrap();
    assert!(text.starts_with("{\n \"brains\": {"), "{text}");
    let back = Progress::open(d.join("neurodex.json"));
    assert_eq!(back.brains["adult"]["T4"].x, 0.13);
    std::fs::remove_dir_all(d).ok();
}

#[test]
fn neurodex_is_forgiving_like_python() {
    let d = tmp("dexhand");
    std::fs::copy(dir().join("neurodex_hand.json"), d.join("neurodex.json")).unwrap();
    let p = Progress::open(d.join("neurodex.json"));
    let a = &p.brains["adult"];
    assert_eq!(a["A"], Discovery { t: "x".into(), x: 0.0, how: "play".into() });
    assert!(!a.contains_key("B"));
    assert_eq!(a["C"].x, 2.0);
    assert!(!p.brains.contains_key("odd"));

    std::fs::write(d.join("neurodex.json"), "{broken").unwrap();
    let p = Progress::open(d.join("neurodex.json"));
    assert_eq!(p.warnings.len(), 2);
    assert!(d.join("neurodex.json.bad").exists());

    std::fs::copy(dir().join("neurodex_newer.json"), d.join("neurodex.json")).unwrap();
    let mut p = Progress::open(d.join("neurodex.json"));
    assert!(p.read_only);
    assert!(!p.save(), "a newer file is never overwritten");
    std::fs::remove_dir_all(d).ok();
}

#[test]
fn contraption_slots_from_python() {
    let slots = load_contraption_slots(&dir().join("contraptions.json"));
    assert_eq!(slots.len(), 8);
    assert_eq!(slots.iter().filter(|s| s.is_some()).count(), 2);
    assert_eq!(slots[2].as_ref().unwrap()["name"], "Domino run");
    let d = tmp("ctn");
    let p = d.join("contraptions.json");
    std::fs::copy(dir().join("contraptions.json"), &p).unwrap();
    assert!(save_contraption_slot(&p, 7, slots[2].as_ref().unwrap()));
    assert!(!save_contraption_slot(&p, 8, slots[2].as_ref().unwrap()));
    assert!(!save_contraption_slot(&p, 0, &serde_json::json!({"name": "x", "parts": []})));
    let again = load_contraption_slots(&p);
    assert_eq!(again.iter().filter(|s| s.is_some()).count(), 3);
    assert_eq!(again[7], slots[2]);
    std::fs::remove_dir_all(d).ok();
}

#[test]
fn lesion_loadouts_from_python() {
    let saved = load_lesion_loadouts(&dir().join("lesion_loadouts.json"));
    assert_eq!(saved.len(), 2);
    assert_eq!(saved["quiet"]["LC4"], -1);
    assert_eq!(saved["loud"]["DNp01"], 1);
    let d = tmp("lesion");
    let p = d.join("lesion_loadouts.json");
    std::fs::copy(dir().join("lesion_loadouts.json"), &p).unwrap();
    for i in 0..3 {
        assert!(save_lesion_loadout(&p, &format!("n{i}"), &[("A".to_string(), 1)].into()));
    }
    assert!(!save_lesion_loadout(&p, "one too many", &[("A".to_string(), 1)].into()), "5 slots");
    assert!(save_lesion_loadout(&p, "quiet", &[("B".to_string(), -1)].into()), "an existing name is overwritten");
    delete_lesion_loadout(&p, "n0");
    let now = load_lesion_loadouts(&p);
    assert_eq!(now.len(), 4);
    assert_eq!(now["quiet"]["B"], -1);
    std::fs::remove_dir_all(d).ok();
}

#[test]
fn fly_memory_from_python_and_back() {
    let sig = [41495.0, 50.26125336];
    let m = FlyMemory::load(&dir().join("fly-memory.npz"), &sig).expect("the v2.5.0 file loads");
    assert_eq!(m.w.len(), 41495);
    assert_eq!(m.templates.keys().collect::<Vec<_>>(), ["swatter"]);
    assert_eq!(m.templates["swatter"].len(), 4064);
    assert_eq!((m.kc_calm.len(), m.dan_calm.len()), (4064, 332));
    assert_eq!(m.naive_mbon["swatter"], 12.5);
    assert_eq!(m.trials["swatter"][0][1], "punish");
    // a different brain pack: start fresh
    assert!(FlyMemory::load(&dir().join("fly-memory.npz"), &[41495.0, 51.0]).is_none());
    // write, read again, identical; and NumPy-format valid (the reader of fly_sim is the independent check of names and dtypes)
    let d = tmp("mem");
    m.save(&d.join("fly-memory.npz")).unwrap();
    let back = FlyMemory::load(&d.join("fly-memory.npz"), &sig).unwrap();
    assert_eq!(back, m);
    std::fs::remove_dir_all(d).ok();
}
