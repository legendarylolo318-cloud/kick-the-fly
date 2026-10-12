//! The laser's cell-type lookup equals `lab/laser.py` on the real pack (`tests/golden/make_laser.py`).

use fly_sim::{laser, pack};
use serde_json::Value;

#[test]
fn rows_equal_python() {
    let Some(path) = pack::find_default() else { return };
    let golden: Value =
        serde_json::from_slice(&std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/golden/laser.json")).unwrap()).unwrap();
    let pk = pack::Pack::load(&path).unwrap();
    for (target, want) in golden.as_object().unwrap() {
        let rows = laser::resolve_rows(&pk.labels, target); // the constructor keeps the text as given (set_target would strip it)
        let sum: u64 = rows.iter().map(|&r| u64::from(r)).sum();
        assert_eq!((rows.len() as u64, sum), (want["n"].as_u64().unwrap(), want["sum"].as_u64().unwrap()), "{target:?}");
    }
}
