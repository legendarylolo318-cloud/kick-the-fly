//! The recorder writes what `lab/recorder.py` writes (`tests/golden/make_recorder.py`): CSVs byte for byte, the npz arrays.

use std::sync::Arc;

use fly_sim::cards::{Build, LIF_DEFAULTS};
use fly_sim::{npy::Npz, pack, validation};
use ktf_data::recorder::Recorder;

#[test]
fn recording_equals_python() {
    let Some(path) = pack::find_default() else { return };
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/golden");
    let pk = pack::Pack::load(&path).unwrap();
    let (w, l) = (Arc::new(pk.weights), Arc::new(pk.labels));
    let build = Build { weights: &w, labels: &l, mode: "off", sigma: None, lif: LIF_DEFAULTS, wired: None };
    let mut br = build.brain_with(7, None, 600, true);
    let g = validation::groups(&l, &w);
    let groups = vec![
        ("dn".to_string(), g["dn"].clone()),
        ("dnp01".to_string(), g["dnp01"].clone()),
        ("loom".to_string(), g["loom"].clone()),
        ("empty".to_string(), vec![]),
    ];
    let mut rec = Recorder::new(br.n(), groups, br.steps);
    for k in 0..300 {
        if k == 50 {
            br.poke64("head", Some('R'), 0.6);
            rec.log_event(br.steps, "stimulus", "head", "R", 0.6);
        }
        if k == 100 {
            rec.log_kinematics(br.steps, 0.5, 0.25, 0.0, 0.1, 100.0, "walk", "kitchen");
        }
        if k == 200 {
            rec.log_event(br.steps, "note", "JUMP", "JUMP  head touch", 0.0);
        }
        let step = br.steps;
        let spikes = br.step().to_vec();
        rec.push(step, &spikes);
    }
    let tmp = std::env::temp_dir().join(format!("ktf-rec-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    let files = rec.save(&tmp.join("recording"), &l, serde_json::json!({"recorded_live": true})).unwrap();
    assert_eq!(files.len(), 7);
    for name in ["spikes.csv", "rates.csv", "group-rates.csv", "events.csv", "kinematics.csv"] {
        let a = std::fs::read_to_string(tmp.join(format!("recording-{name}"))).unwrap();
        let b = std::fs::read_to_string(format!("{root}/recorder_files/recording-{name}")).unwrap();
        assert_eq!(a, b, "{name}");
    }
    let ours = Npz::read(std::fs::File::open(tmp.join("recording.npz")).unwrap()).unwrap();
    let theirs = Npz::read(std::fs::File::open(format!("{root}/recorder_files/recording.npz")).unwrap()).unwrap();
    let mut a: Vec<&String> = ours.names().collect();
    let mut b: Vec<&String> = theirs.names().collect();
    a.sort();
    b.sort();
    assert_eq!(a, b);
    for n in a {
        let (x, y) = (ours.get(n).unwrap(), theirs.get(n).unwrap());
        assert_eq!(x.shape, y.shape, "{n}");
        assert_eq!(format!("{:?}", x.data), format!("{:?}", y.data), "{n}");
    }
}
