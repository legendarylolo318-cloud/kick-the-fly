//! The arcade's brains on the real connectome (skipped without `data/kick_brain.npz`): two fighters duel for a moment, a fly runs a few
//! seconds of the track, a lesion loadout silences what it names. Short on purpose; the full matches take minutes.

use std::sync::Arc;

use fly_sim::pack::{self, Pack};
use kick_the_fly::arcade::{apply_loadout, build_brain};
use kick_the_fly::duel;
use kick_the_fly::lesion::{Loadout, SILENCE};
use kick_the_fly::race;
use kick_the_fly::sim_thread::Shared;

fn shared() -> Option<Shared> {
    let path = pack::find_default()?;
    let p = Pack::load(path).expect("the brain pack loads");
    Some(Shared { weights: Arc::new(p.weights), labels: Arc::new(p.labels) })
}

#[test]
fn two_real_brains_duel_and_one_runs_the_track() {
    let Some(sh) = shared() else {
        eprintln!("SKIPPED: no brain pack");
        return;
    };
    let mut a = build_brain(&sh, 1, 11, 0.05);
    let mut b = build_brain(&sh, 2, 12, 0.05);
    let r = duel::run_duel(&mut a, &mut b, 3, 2.0);
    assert!(r.seconds > 1.9, "the match ran its 2 s: {}", r.seconds);
    assert!(r.hp.0 <= 100.0 && r.hp.1 <= 100.0);
    let mut c = build_brain(&sh, 3, 13, 0.05);
    let lane = race::run_lane(&mut c, 3.0, race::TRACK_LENGTH);
    assert!(lane.distance >= 0.0 && lane.seconds > 2.9 || lane.finish_s.is_some());
    assert!(lane.trace.len() > 3);
}

#[test]
fn a_loadout_silences_the_cell_type_it_names() {
    let Some(sh) = shared() else {
        return;
    };
    let mut brain = build_brain(&sh, 1, 21, 0.0);
    let mut l = Loadout::new("test");
    l.add("DNp09", SILENCE, None).unwrap();
    apply_loadout(&mut brain, &l);
    for _ in 0..400 {
        brain.step();
    }
    let rows = brain.rows_of("type:DNp09").unwrap();
    let rate: f32 = rows.iter().map(|&r| brain.sim.activity.rates()[r as usize]).sum();
    assert!(rate < 0.001, "a silenced DNp09 does not fire: {rate}");
}
