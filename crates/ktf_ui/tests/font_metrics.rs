//! Text widths and line heights against what pygame measured for the same fonts (`tests/ui_reference/font_metrics.json`).
use ktf_ui::font::Font;

#[test]
fn widths_match_pygame() {
    let d: serde_json::Value = serde_json::from_str(include_str!("../../../tests/ui_reference/font_metrics.json")).unwrap();
    let mut bad = vec![];
    let mut total = 0;
    for (key, m) in d.as_object().unwrap() {
        let f = Font::new(m["size"].as_i64().unwrap() as i32, m["bold"].as_bool().unwrap());
        assert_eq!(f.linesize as i64, m["linesize"].as_i64().unwrap(), "{key} linesize");
        assert_eq!(f.ascent as i64, m["ascent"].as_i64().unwrap(), "{key} ascent");
        for (s, w) in m["widths"].as_object().unwrap() {
            total += 1;
            let got = f.size(s) as i64;
            if got != w.as_i64().unwrap() {
                bad.push(format!("{key} {s:?}: rust {got}, pygame {w}"));
            }
        }
    }
    assert!(bad.is_empty(), "{} of {total} widths differ:\n{}", bad.len(), bad.join("\n"));
}
