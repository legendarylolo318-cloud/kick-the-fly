//! `.ktfsave` files written by the Python game (`tests/python_files/savestate/`, made by the recipe in
//! `legacy-python/make_persistence_fixtures.py`): header, compatibility, a lossless rewrite, and the strong one: importing the saved
//! simulator state into the Rust sim continues the exact spike raster Python produced after saving (`continue.json`).

use std::path::PathBuf;
use std::sync::Arc;

use fly_sim::Pack;
use fly_sim::lif::{Individuality, LifParams, LifSim};
use fly_sim::util::sha16_u32;
use ktf_data::savestate::{PackInfo, SaveFile, compatible, read_meta};

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/python_files/savestate")
}

#[test]
fn header_and_compatibility_follow_python() {
    let meta = read_meta(&dir().join("real.ktfsave")).unwrap();
    assert_eq!(meta["format"], "kick-the-fly-save");
    assert_eq!(meta["arena"], "pool");
    assert_eq!(meta["flies"].as_array().unwrap().len(), 1);
    let mine = PackInfo { n_neurons: 166_700, synapses: 10_272_125, brain: "adult".into(), mode: "2d".into() };
    assert_eq!(compatible(&meta, &mine), None);
    let other = PackInfo { mode: "3d".into(), ..mine.clone() };
    assert_eq!(compatible(&meta, &other).unwrap(), "made in the 2D game");
    let other = PackInfo { brain: "larva".into(), ..mine.clone() };
    assert_eq!(compatible(&meta, &other).unwrap(), "made for the adult brain (currently using larva)");
    let other = PackInfo { synapses: 5, ..mine };
    assert_eq!(compatible(&meta, &other).unwrap(), "made with a different brain pack");
    let mut newer = meta.clone();
    newer["format_version"] = 2.into();
    assert!(compatible(&newer, &other).unwrap().starts_with("made by a newer version (3.1.1)"));
}

#[test]
fn a_garbage_file_is_refused_with_a_reason() {
    let p = std::env::temp_dir().join(format!("ktf-garbage-{}.ktfsave", std::process::id()));
    std::fs::write(&p, b"not a zip").unwrap();
    assert!(read_meta(&p).unwrap_err().0.starts_with("not a readable save file"));
    std::fs::remove_file(&p).ok();
}

#[test]
fn rewriting_a_save_keeps_everything_including_128_bit_generator_state() {
    let a = SaveFile::read(&dir().join("real.ktfsave")).unwrap();
    let p = std::env::temp_dir().join(format!("ktf-rewrite-{}.ktfsave", std::process::id()));
    a.write(&p).unwrap();
    let b = SaveFile::read(&p).unwrap();
    std::fs::remove_file(&p).ok();
    assert_eq!(a.meta, b.meta);
    assert_eq!(a.arrays.len(), b.arrays.len());
    for (k, x) in &a.arrays {
        assert_eq!(x.shape, b.arrays[k].shape, "{k}");
        assert_eq!(format!("{:?}", x.data), format!("{:?}", b.arrays[k].data), "{k}");
    }
    assert_eq!(a.sim_state(0).unwrap().rng, b.sim_state(0).unwrap().rng);
    // the generator state keeps its full 128 bits
    assert!(a.sim_state(0).unwrap().rng.state > u128::from(u64::MAX));
}

#[test]
fn imported_state_continues_the_python_raster_exactly() {
    let Some(pack_path) = fly_sim::pack::find_default() else {
        eprintln!("no data/kick_brain.npz: skipping");
        return;
    };
    let pack = Pack::load_weights_only(&pack_path).unwrap();
    let save = SaveFile::read(&dir().join("real.ktfsave")).unwrap();
    let st = save.sim_state(0).unwrap();
    assert_eq!(st.v.len(), pack.weights.n());
    let mut sim = LifSim::new(Arc::new(pack.weights.clone()), LifParams::default(), st.seed);
    // the save says "individuality": "subtle" (sigma 0.05): the gains are rebuilt from the brain seed, as the game does on load
    assert_eq!(save.meta["individuality"], "subtle");
    sim.set_individuality(Individuality::compute(st.seed, pack.weights.n(), 0.05));
    sim.import_state(&st);
    let want: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir().join("continue.json")).unwrap()).unwrap();
    for (t, row) in want.as_array().unwrap().iter().enumerate() {
        let idx = sim.step(None).to_vec();
        assert_eq!(idx.len() as u64, row["n"].as_u64().unwrap(), "step {t}: spike count");
        assert_eq!(sha16_u32(&idx), row["sha"].as_str().unwrap(), "step {t}: spike set");
        let g = row["gain"].as_f64().unwrap();
        assert!((sim.gain - g).abs() < 1e-9 * g.abs().max(1.0), "step {t}: gain {} vs {g}", sim.gain);
    }
}
