//! `protocol::check` equals `lab/protocol.py` on every bundled protocol file and on crafted bad ones (`tests/golden/make_protocol.py`).

use ktf_data::protocol::{check, load};
use serde_json::Value;

#[test]
fn bundled_files_and_bad_cases_equal_python() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    let g: Value = serde_json::from_str(&std::fs::read_to_string(format!("{root}/tests/golden/protocol.json")).unwrap()).unwrap();
    for (name, want) in g["files"].as_object().unwrap() {
        let got = load(std::path::Path::new(&format!("{root}/tests/python_files/protocols/{name}")));
        match (got, want.get("ok")) {
            (Ok(v), Some(w)) => {
                // compare structure ignoring float/int spelling
                assert_eq!(v, *w, "{name}");
            }
            (Err(e), None) => assert_eq!(e.0, want["error"].as_str().unwrap(), "{name}"),
            (Ok(_), None) => panic!("{name}: Rust accepts, Python rejects: {}", want["error"]),
            (Err(e), Some(_)) => panic!("{name}: Rust rejects ({e}), Python accepts"),
        }
    }
    for (i, c) in g["cases"].as_array().unwrap().iter().enumerate() {
        let got = check(&c["input"], "case");
        match (got, c.get("ok")) {
            (Ok(v), Some(w)) => assert_eq!(v, *w, "case {i}: {}", c["input"]),
            (Err(e), None) => assert_eq!(e.0, c["error"].as_str().unwrap(), "case {i}: {}", c["input"]),
            (Ok(_), None) => panic!("case {i} {}: Rust accepts, Python says {}", c["input"], c["error"]),
            (Err(e), Some(_)) => panic!("case {i} {}: Rust rejects ({e})", c["input"]),
        }
    }
}
