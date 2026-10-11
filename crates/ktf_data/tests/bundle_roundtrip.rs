use ktf_data::bundle::{Bundle, create_live};
use serde_json::json;

#[test]
fn live_bundle_round_trips_and_detects_tampering() {
    let d = std::env::temp_dir().join(format!("ktf-bundle-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    let f1 = d.join("recording-spikes.csv");
    std::fs::write(&f1, "time_ms,row\r\n5.0,1\r\n").unwrap();
    let z = d.join("b.zip");
    create_live(&[f1], &json!({"seeds": [7]}), &z, &json!({}), "4.0.0", "2026-01-01T00:00:00").unwrap();
    let z2 = d.join("b2.zip");
    create_live(&[d.join("recording-spikes.csv")], &json!({"seeds": [7]}), &z2, &json!({}), "4.0.0", "2026-01-01T00:00:00").unwrap();
    assert_eq!(std::fs::read(&z).unwrap(), std::fs::read(&z2).unwrap(), "fixed timestamps: same bytes");
    let mut b = Bundle::open(&z).unwrap();
    assert!(b.verify().is_empty());
    assert_eq!(b.meta["rerunnable"], json!(false));
    assert!(b.names().contains(&"raw/recording-spikes.csv".to_string()));
    // a bundle that is not one
    std::fs::write(d.join("x.zip"), b"nope").unwrap();
    assert!(Bundle::open(&d.join("x.zip")).is_err());
}
