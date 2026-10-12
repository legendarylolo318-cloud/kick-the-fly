//! `core/streamer.py` scripted chat (tests/golden/make_streamer.py).
use ktf_data::streamer::*;
use serde_json::Value;

#[test]
fn votes_equal_python() {
    let g: Value =
        serde_json::from_str(&std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/golden/streamer.json")).unwrap())
            .unwrap();
    let mut b = VoteBoard::new([1; 16]);
    b.set_rules(Rules { window_s: 10.0, cooldown_s: 15.0, min_votes: 2, allow: vec!["tool".into(), "arena".into()] });
    let opt = |v: &[(&str, &str)]| v.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect::<Vec<_>>();
    b.options.insert("tool".into(), opt(&[("swatter", "SWATTER"), ("bomb", "BOMB"), ("frog_food", "FROG FOOD")]));
    b.options.insert("arena".into(), opt(&[("pool", "pool"), ("lamp", "lamp")]));
    b.options.insert("surgery".into(), opt(&[("x", "X")]));
    let rows = g["submits"].as_array().unwrap();
    let ticks = g["ticks"].as_array().unwrap();
    let check = |b: &VoteBoard, r: &Value| {
        let (u, t, now) = (r[0].as_str().unwrap(), r[1].as_str().unwrap(), r[2].as_f64().unwrap());
        let _ = (u, t);
        let tally: Vec<(String, String, String, usize)> = r[4]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| {
                (
                    x[0].as_str().unwrap().into(),
                    x[1].as_str().unwrap().into(),
                    x[2].as_str().unwrap().into(),
                    x[3].as_u64().unwrap() as usize,
                )
            })
            .collect();
        assert_eq!(b.tally(), tally, "tally after {t}");
        assert_eq!(b.state.name(), r[5].as_str().unwrap(), "state after {t}");
        assert!((b.remaining(now) - r[6].as_f64().unwrap()).abs() < 1e-9);
    };
    for r in &rows[..11] {
        let why = b.submit(r[0].as_str().unwrap(), r[1].as_str().unwrap(), r[2].as_f64().unwrap());
        assert_eq!(why, r[3].as_str().unwrap(), "{}", r[1]);
        check(&b, r);
    }
    let t = &ticks[0];
    let res = b.tick(t[0].as_f64().unwrap());
    let want = &t[1];
    assert_eq!(
        res.map(|o| (o.command, o.slug, o.label, o.votes as u64, o.total as u64)),
        want.as_array().map(|a| (
            a[0].as_str().unwrap().into(),
            a[1].as_str().unwrap().into(),
            a[2].as_str().unwrap().into(),
            a[3].as_u64().unwrap(),
            a[4].as_u64().unwrap()
        ))
    );
    assert_eq!(b.state.name(), t[2].as_str().unwrap());
    for r in &rows[11..] {
        let why = b.submit(r[0].as_str().unwrap(), r[1].as_str().unwrap(), r[2].as_f64().unwrap());
        assert_eq!(why, r[3].as_str().unwrap(), "{}", r[1]);
        check(&b, r);
    }
    let t = &ticks[1];
    let res = b.tick(t[0].as_f64().unwrap());
    assert_eq!(res.is_some(), !t[1].is_null());
    for (l, want) in g["lines"].as_array().unwrap().iter().map(|x| (x[0].as_str().unwrap(), &x[1])) {
        let got = match parse_line(l) {
            None => Value::Null,
            Some(Line::Ping(p)) => serde_json::json!(["ping", p]),
            Some(Line::Msg { user, channel, text }) => serde_json::json!(["msg", user, channel, text]),
            Some(Line::Numeric(c, r)) => serde_json::json!(["numeric", c, r]),
            Some(Line::Notice(t)) => serde_json::json!(["notice", t]),
        };
        assert_eq!(&got, want, "{l}");
    }
    for s in g["slugs"].as_array().unwrap() {
        assert_eq!(slug(s[0].as_str().unwrap()), s[1].as_str().unwrap());
    }
}
