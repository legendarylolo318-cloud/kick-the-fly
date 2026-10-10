//! The T-maze and sweet-tooth challenges on the real connectome.

use std::sync::Arc;

use fly_sim::Brain;
use fly_sim::challenge::{SEvent, SPhase, Sweet, TEvent, TMaze, TPhase};
use fly_sim::lif::{LifParams, LifSim};
use fly_sim::pack::{self, Pack};

fn brain(seed: u64) -> Option<Brain> {
    let p = Pack::load(pack::find_default()?).ok()?;
    let mut b = Brain::new(LifSim::new(Arc::new(p.weights), LifParams::default(), seed), Arc::new(p.labels));
    assert!(b.enable_memory());
    b.warmup(1400);
    Some(b)
}

#[test]
fn tmaze_fly_learns_which_door_zaps() {
    let Some(mut b) = brain(11) else { return };
    let mut t = TMaze::start(&b, "odor_a", 11).unwrap();
    let mut now = 0.0;
    let right = loop {
        b.step();
        now += 0.005 / 3.0; // brain speed 3: a step takes a third of 5 ms of wall time
        match t.tick(&mut b, now) {
            TEvent::Finished { right } => break right,
            _ => assert!(b.steps < 200_000, "never finished"),
        }
    };
    assert_eq!(t.phase, TPhase::Done);
    assert_eq!(t.choices.len(), 10);
    assert!(right >= 6, "a trained fly picks the safe door most of the time: {right}/10");
}

#[test]
fn a_strong_sugar_dose_makes_it_reach_and_none_does_not() {
    let Some(mut b) = brain(5) else { return };
    let mut s = Sweet::new();
    let mut now = 0.0;
    for dose in [100u32, 0] {
        s.offer(&b, dose);
        assert_eq!(s.phase, SPhase::Before);
        loop {
            b.step();
            now += 0.005;
            if let SEvent::Answer(d, ext, ratio) = s.tick(&mut b, now) {
                eprintln!("dose {d}: extended {ext} ratio {ratio:.2}");
                assert_eq!(ext, dose == 100, "dose {d}");
                break;
            }
            assert!(b.steps < 100_000);
        }
        while s.phase != SPhase::Ready {
            b.step();
            now += 0.005;
            s.tick(&mut b, now);
        }
    }
}
