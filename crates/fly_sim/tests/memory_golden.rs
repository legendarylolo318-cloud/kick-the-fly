//! The mushroom body's plasticity against NumPy's (`core/memory.py`): the scripted training run of `tests/golden/make_memory.py` on the
//! real pack. The simulator is spike-exact with Python's and the plasticity does the same arithmetic, so the memory after every trial
//! agrees to rounding (the sums that feed back into the spikes differ in the last bits).

use std::sync::Arc;

use fly_sim::Brain;
use fly_sim::lif::{LifParams, LifSim};
use fly_sim::pack::{self, Pack};

fn golden() -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/golden/memory.json")).unwrap()).unwrap()
}

/// `run()` of the Python script: hold the drives for `steps`, observing the smell every 3rd step; returns the approach MBONs' mean Hz over
/// the last 40 steps.
fn run(brain: &mut Brain, steps: usize, drives: &[(&[u32], f32)], observe: Option<&str>) -> f64 {
    for (rows, amp) in drives {
        brain.drive(rows, *amp);
    }
    let mut mbon = vec![];
    for i in 0..steps {
        brain.step();
        if let (Some(s), 0) = (observe, i % 3) {
            let rates = brain.sim.activity.rates().to_vec();
            brain.memory.as_mut().unwrap().observe(s, &rates);
        }
        if i >= steps - 40 {
            mbon.push(f64::from(brain.memory.as_ref().unwrap().mbon_response(brain.sim.activity.rates())));
        }
    }
    for (rows, _) in drives {
        brain.undrive(rows);
    }
    mbon.iter().sum::<f64>() / mbon.len() as f64
}

#[test]
fn training_run_matches_numpy() {
    let Some(path) = pack::find_default() else {
        eprintln!("SKIPPED: no brain pack");
        return;
    };
    let g = golden();
    let pk = Pack::load(path).unwrap();
    let labels = Arc::new(pk.labels);
    let mut brain = Brain::new(LifSim::new(Arc::new(pk.weights), LifParams::default(), 5), labels.clone());
    assert!(brain.enable_memory());
    {
        let m = brain.memory.as_ref().unwrap();
        assert_eq!(m.kc.len() as u64, g["n_kc"].as_u64().unwrap());
        assert_eq!(m.mbon.len() as u64, g["n_mbon"].as_u64().unwrap());
        assert_eq!(m.w0.len() as u64, g["n_syn"].as_u64().unwrap());
        assert_eq!(m.punish_comp.iter().filter(|&&p| p).count() as u64, g["punish_comps"].as_u64().unwrap());
        let sig = g["signature"].as_array().unwrap();
        assert_eq!(m.signature[0], sig[0].as_f64().unwrap());
        assert!((m.signature[1] / sig[1].as_f64().unwrap() - 1.0).abs() < 1e-6);
    }
    brain.warmup(600);
    let m = brain.memory.as_ref().unwrap();
    let (a, b): (Vec<u32>, Vec<u32>) = (m.kc[..700].to_vec(), m.kc[1500..2200].to_vec());
    let ppl1: Vec<u32> = m.dan.iter().copied().filter(|&r| labels.types[r as usize].starts_with("PPL1")).collect();
    let pam: Vec<u32> = m.dan.iter().copied().filter(|&r| !labels.types[r as usize].starts_with("PPL1")).collect();
    run(&mut brain, 1200, &[], None);

    // (label, smell read, mbon Hz) per checkpoint
    let mut got: Vec<(String, &str, f64)> = vec![];
    for t in 0..6 {
        run(&mut brain, 120, &[(&a, 0.35)], Some("A"));
        let hz = run(&mut brain, 100, &[(&a, 0.35), (&ppl1, 0.8)], Some("A"));
        run(&mut brain, 130, &[], None);
        got.push((format!("fear{}", t + 1), "A", hz));
        check_last(&brain, &g, &got);
    }
    let hz = run(&mut brain, 120, &[(&b, 0.35)], Some("B"));
    got.push(("test_B".into(), "B", hz));
    check_last(&brain, &g, &got);
    for t in 0..4 {
        run(&mut brain, 120, &[(&a, 0.35)], Some("A"));
        let hz = run(&mut brain, 100, &[(&a, 0.35), (&pam, 0.8)], Some("A"));
        run(&mut brain, 130, &[], None);
        got.push((format!("like{}", t + 1), "A", hz));
        check_last(&brain, &g, &got);
    }
    let hz = run(&mut brain, 120, &[(&a, 0.35)], Some("A"));
    got.push(("test_A".into(), "A", hz));
    check_last(&brain, &g, &got);
    assert_eq!(got.len(), g["checks"].as_array().unwrap().len());

    // the weights the simulator runs on are the learned ones: the first synapse that changed shows in the matrix
    let mem = brain.memory.as_ref().unwrap();
    let i = (0..mem.w.len()).find(|&i| mem.w[i] < mem.w0[i] * 0.5).expect("some synapse is at half strength");
    assert_ne!(mem.w[i], mem.w0[i]);
    let samples = g["w_sample"].as_array().unwrap();
    for (k, v) in samples.iter().enumerate() {
        let (got, want) = (f64::from(mem.w[k * 997]), v.as_f64().unwrap());
        assert!((got - want).abs() <= 1e-3 * want.abs() + 1e-6, "w[{}]: {got} vs {want}", k * 997);
    }
    // the calm baselines and the smell's template
    for (k, v) in g["kc_calm_sample"].as_array().unwrap().iter().enumerate() {
        let (got, want) = (f64::from(mem.kc_calm[k * 97]), v.as_f64().unwrap());
        assert!((got - want).abs() <= 1e-3 * want.abs() + 1e-6, "kc_calm[{}]: {got} vs {want}", k * 97);
    }
    for (k, v) in g["template_A_sample"].as_array().unwrap().iter().enumerate() {
        let (got, want) = (f64::from(mem.templates["A"][k * 97]), v.as_f64().unwrap());
        assert!((got - want).abs() <= 2e-2 * want.abs() + 1e-5, "template[{}]: {got} vs {want}", k * 97);
    }
}

fn check_last(brain: &Brain, g: &serde_json::Value, got: &[(String, &str, f64)]) {
    let (label, scent, hz) = got.last().unwrap();
    let want = &g["checks"][got.len() - 1];
    assert_eq!(want["label"].as_str().unwrap(), label);
    let mem = brain.memory.as_ref().unwrap();
    let (fear, like) = mem.memory_of(scent);
    let wsum: f64 = mem.w.iter().map(|&x| f64::from(x)).sum();
    let f = |k: &str| want[k].as_f64().unwrap();
    eprintln!(
        "{label}: fear {fear:.4}/{:.4} like {like:.4}/{:.4} wsum {wsum:.4}/{:.4} weakened {:.4}/{:.4} mbon {hz:.3}/{:.3}",
        f("fear"),
        f("like"),
        f("w_sum"),
        mem.weakened_share(),
        f("weakened"),
        f("mbon_hz")
    );
    assert_eq!(mem.updates, want["updates"].as_u64().unwrap(), "{label}: updates");
    assert!((f64::from(fear) - f("fear")).abs() < 2e-3, "{label}: fear");
    assert!((f64::from(like) - f("like")).abs() < 2e-3, "{label}: like");
    assert!((wsum / f("w_sum") - 1.0).abs() < 1e-4, "{label}: weight sum");
    assert!((f64::from(mem.weakened_share()) - f("weakened")).abs() < 2e-3, "{label}: weakened share");
    assert!((hz - f("mbon_hz")).abs() < 0.05 * f("mbon_hz").abs(), "{label}: MBON response");
}
