//! Python's `random.shuffle`, the curated facts and the Neuron of the Day against `core/neuron_of_day.py` on the real pack
//! (`tests/golden/neurodex.json`, made by `tests/golden/make_neurodex.py`).
use fly_sim::neurodex::TypeTable;
use ktf_data::{neurodex_facts::fact_for, notd, pyrandom::Mt};

fn gold() -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/golden/neurodex.json")).unwrap())
        .unwrap()
}

#[test]
fn toordinal_and_shuffle_match_python() {
    assert_eq!(notd::toordinal(1, 1, 1), 1);
    assert_eq!(notd::toordinal(1970, 1, 1), 719_163);
    assert_eq!(notd::toordinal(2026, 10, 9), 739_898);
    // computed with CPython: random.Random(123456789012).shuffle(list(range(10))) and random.Random(7).shuffle(list(range(20)))
    let mut a: Vec<usize> = (0..10).collect();
    Mt::seeded(123_456_789_012).shuffle(&mut a);
    assert_eq!(a, [2, 8, 5, 7, 9, 3, 1, 4, 0, 6]);
    let mut b: Vec<usize> = (0..20).collect();
    Mt::seeded(7).shuffle(&mut b);
    assert_eq!(b, [17, 15, 11, 18, 7, 6, 19, 3, 14, 0, 9, 5, 16, 8, 13, 2, 1, 12, 4, 10]);
}

#[test]
fn neuron_of_the_day_matches_python() {
    let Some(path) = fly_sim::pack::find_default() else {
        eprintln!("no pack: skipping");
        return;
    };
    let pack = fly_sim::Pack::load(&path).unwrap();
    let tab = TypeTable::build(&pack.labels, "adult", None);
    let g = gold();
    // which curated fact each sampled type has
    for (name, want) in g["facts_for"].as_object().unwrap() {
        assert_eq!(fact_for(name, "adult").map(|f| f.id.as_str()), want.as_str(), "{name}");
    }
    let cands: Vec<String> = notd::candidates(&tab).iter().map(|f| f.id.clone()).collect();
    assert_eq!(cands, g["notd_candidates"].as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect::<Vec<_>>());
    for c in g["notd"].as_array().unwrap() {
        let day = c["day"].as_str().unwrap();
        let (y, m, d): (i32, u32, u32) = (day[..4].parse().unwrap(), day[5..7].parse().unwrap(), day[8..10].parse().unwrap());
        let lab = c["lab"].as_bool().unwrap();
        let card = notd::card(notd::toordinal(y, m, d), &tab, lab).unwrap();
        assert_eq!(card.fact.id, c["fact_id"].as_str().unwrap(), "{day} fact");
        assert_eq!(
            card.types,
            c["types"].as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect::<Vec<_>>(),
            "{day} types"
        );
        assert_eq!(card.plan.action, c["plan"]["action"].as_str().unwrap(), "{day} plan");
        assert_eq!(card.plan.text, c["plan"]["text"].as_str().unwrap(), "{day} plan text");
    }
}
