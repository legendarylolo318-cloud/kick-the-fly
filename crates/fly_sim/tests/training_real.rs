//! The conditioning protocol on the real connectome: fear training teaches a smell, a test afterwards does not change what was learned,
//! and the untrained response is measured first.

use std::sync::Arc;

use fly_sim::Brain;
use fly_sim::lif::{LifParams, LifSim};
use fly_sim::pack::{self, Pack};
use fly_sim::training::{Kind, Outcome, Training};

fn brain() -> Option<Brain> {
    let p = Pack::load(pack::find_default()?).ok()?;
    let mut b = Brain::new(LifSim::new(Arc::new(p.weights), LifParams::default(), 3), Arc::new(p.labels));
    assert!(b.enable_memory());
    b.warmup(1400); // past the memory's 100 settling updates
    Some(b)
}

fn run(b: &mut Brain, t: &mut Training) -> u64 {
    let start = b.steps;
    loop {
        b.step();
        if t.tick(b, 1.0e9) == Outcome::Done {
            return b.steps - start;
        }
        assert!(b.steps - start < 20_000, "the protocol never ended");
    }
}

#[test]
fn fear_training_teaches_the_smell_and_a_test_only_reads_it() {
    let Some(mut b) = brain() else {
        eprintln!("SKIPPED: no brain pack");
        return;
    };
    let mut t = Training::start(&b, "torch", Kind::Fear, 4).unwrap();
    let steps = run(&mut b, &mut t);
    // naive 400 + rest 260, then 4 x (odor 240 + pair 200 + rest 260)
    assert_eq!(steps, 400 + 260 + 4 * 700);
    let m = b.memory.as_ref().unwrap();
    let kinds: Vec<&str> = m.log["torch"].iter().map(|r| r.kind.as_str()).collect();
    assert_eq!(kinds, ["naive", "fear", "fear", "fear", "fear"]);
    assert!(m.naive_mbon.contains_key("torch"));
    let (fear, like) = m.memory_of("torch");
    assert!(fear > 0.05, "four pairings should leave a fear memory: {fear}");
    assert!(fear > like, "fear {fear} vs liking {like}");
    // the second run on the same smell skips the naive measurement
    let before = (fear, like);
    let mut test = Training::start(&b, "torch", Kind::Test, 10).unwrap();
    assert_eq!(test.trials, 1);
    run(&mut b, &mut test);
    let m = b.memory.as_ref().unwrap();
    assert_eq!(m.log["torch"].last().unwrap().kind, "test");
    let (f2, l2) = m.memory_of("torch");
    assert!((f2 - before.0).abs() < 0.08 && l2 < before.1 + 0.08, "a test does not train: {before:?} -> {:?}", (f2, l2));
}
