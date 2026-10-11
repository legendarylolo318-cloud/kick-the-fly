//! `core/pet.py` scripted life (tests/golden/make_pet.py): catch-up, feeding, live updates equal Python's.
use ktf_data::pet::Pet;
use serde_json::Value;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

#[test]
fn pet_life_equals_python() {
    let g: Value = serde_json::from_str(&std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/golden/pet.json")).unwrap()).unwrap();
    let dir = std::env::temp_dir().join(format!("ktf-pet-golden-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let mut now = 1_700_000_000.0;
    let mut p = Pet::new(dir.clone(), now);
    let steps = g["steps"].as_array().unwrap();
    let check = |p: &Pet, s: &Value| {
        let tag = s["tag"].as_str().unwrap();
        assert!(close(p.hunger, s["hunger"].as_f64().unwrap()), "{tag} hunger {} vs {}", p.hunger, s["hunger"]);
        assert!(close(p.sleep_pressure, s["sleep"].as_f64().unwrap()), "{tag} sleep");
        assert!(close(p.mood, s["mood"].as_f64().unwrap()), "{tag} mood");
        assert_eq!(p.is_sleeping, s["sleeping"].as_bool().unwrap(), "{tag}");
        assert_eq!(p.is_dead, s["dead"].as_bool().unwrap(), "{tag}");
        assert_eq!(p.welcome_message, s["welcome"].as_str().unwrap(), "{tag}");
        let ev: Vec<String> = p
            .timeline
            .iter()
            .map(|e| format!("{}:{}", e["event_type"].as_str().unwrap(), e["description"].as_str().unwrap()))
            .collect();
        let want: Vec<String> = s["events"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
        assert_eq!(ev, want, "{tag}");
    };
    p.create_new(Some(42), "adult", now);
    check(&p, &steps[0]);
    let reload = |p: &mut Pet, now: f64| {
        // a fresh manager reading the file, as the game does on launch
        let mut q = Pet::new(p.dir.clone(), now);
        q.load_or_create("adult", now);
        *p = q;
    };
    now += 3.0 * 3600.0;
    reload(&mut p, now);
    check(&p, &steps[1]);
    p.feed(0.2, "sugar", true, now);
    check(&p, &steps[2]);
    now += 30.0 * 60.0;
    reload(&mut p, now);
    check(&p, &steps[3]);
    now += 10.0 * 3600.0;
    reload(&mut p, now);
    check(&p, &steps[4]);
    p.real_stakes = true;
    p.hunger = 1.0;
    now += 3.0 * 86400.0;
    // Python re-loads the file saved by `load_or_create`; real_stakes/hunger were set in memory only, so save first as its next load would
    // not see them either: it calls load_or_create on the same object, which re-reads the file
    reload(&mut p, now);
    check(&p, &steps[5]);

    // live updates
    let rolls: Vec<f64> = g["live_rolls"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let mut it = rolls.into_iter();
    let mut m = Pet::new(dir.join("m2"), now);
    m.create_new(Some(1), "adult", now);
    m.hunger = 0.4;
    m.sleep_pressure = 0.88;
    m.mood = 0.5;
    for (i, want) in g["live"].as_array().unwrap().iter().enumerate() {
        m.update_live(0.5 + i as f64, i % 5 == 4, 2.0 + i as f64 * 0.1, 1.0, &mut || it.next().unwrap(), now);
        assert!(close(m.hunger, want[0].as_f64().unwrap()), "live {i} hunger");
        assert!(close(m.sleep_pressure, want[1].as_f64().unwrap()), "live {i} sleep");
        assert!(close(m.mood, want[2].as_f64().unwrap()), "live {i} mood");
        assert_eq!(m.is_sleeping, want[3].as_bool().unwrap(), "live {i}");
    }
    m.record_injury("a test", now);
    m.record_training("tmaze", now);
    assert!(close(m.mood, g["after"]["mood"].as_f64().unwrap()));
    let _ = std::fs::remove_dir_all(dir);
}
