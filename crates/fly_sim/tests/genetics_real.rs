//! The driver-line toolkit on the real pack equals `lab/genetics.py` (`tests/golden/make_genetics.py`).

use fly_sim::{genetics, pack};
use serde_json::Value;

#[test]
fn toolkit_equals_python() {
    let Some(path) = pack::find_default() else { return };
    let golden: Value =
        serde_json::from_slice(&std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/golden/genetics.json")).unwrap()).unwrap();
    let pk = pack::Pack::load(&path).unwrap();
    assert_eq!(genetics::coverage(&pk.labels), golden["coverage"]);
    for (n, want) in golden["describe"].as_object().unwrap() {
        assert_eq!(&genetics::describe(n, &pk.labels).unwrap(), want, "{n}");
    }
    for (q, want) in golden["search"].as_object().unwrap() {
        let got: Vec<&str> = genetics::search(q, 50).iter().map(|l| l.name.as_str()).collect();
        let want: Vec<&str> = want.as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
        assert_eq!(got, want, "search {q}");
    }
    let rows = genetics::rows_of_lines(&pk.labels, "line:SS00727").unwrap();
    assert_eq!(rows.len() as u64, golden["rows"]["n"].as_u64().unwrap());
    assert_eq!(rows.iter().map(|&r| u64::from(r)).sum::<u64>(), golden["rows"]["sum"].as_u64().unwrap());
}
