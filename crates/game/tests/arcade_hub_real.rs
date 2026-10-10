//! The arcade page's jobs on the real connectome: a 4-fly tournament of 2-second duels and a look at a 3-lane field, through the same
//! `arcade:*` actions the page sends. Skipped without `data/kick_brain.npz`.

use std::sync::Arc;
use std::time::{Duration, Instant};

use fly_sim::pack::{self, Pack};
use kick_the_fly::arcade_hub::ArcadeHub;
use kick_the_fly::sim_thread::Shared;

fn wait(hub: &mut ArcadeHub) {
    let t = Instant::now();
    while hub.busy() {
        hub.poll();
        assert!(t.elapsed() < Duration::from_secs(600), "the job did not finish");
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[test]
fn tournament_and_field_jobs_fill_the_pages_data() {
    let Some(path) = pack::find_default() else {
        eprintln!("SKIPPED: no brain pack");
        return;
    };
    let p = Pack::load(path).unwrap();
    let shared = Shared { weights: Arc::new(p.weights), labels: Arc::new(p.labels) };
    let mut hub = ArcadeHub::with_wallet(kick_the_fly::points::Wallet::in_memory());
    hub.command("arcade:tournament:4:2000:2:2001", Some(shared.clone()), "subtle");
    assert!(hub.busy());
    assert!(hub.snapshot("default")["job"]["label"].as_str().is_some());
    wait(&mut hub);
    let q = hub.snapshot("default");
    assert_eq!(q["error"], "", "{}", q["error"]);
    let b = &q["bracket"];
    assert_eq!(b["size"], 4);
    assert_eq!(b["rounds"].as_array().unwrap().len(), 2);
    assert_eq!(q["bracket_serial"], 1);
    let champ = b["champion"].as_u64().unwrap();
    assert!((2000..2004).contains(&champ));
    let m = &b["rounds"][0][0];
    assert!(m["frames"].as_array().unwrap().len() >= 5, "a replay is recorded");
    assert!(b["cards"]["2000"]["title"].as_str().is_some());
    assert_eq!(b["favorite"], 2001);

    hub.command("arcade:look:3:3000", Some(shared), "subtle");
    wait(&mut hub);
    let q = hub.snapshot("default");
    assert_eq!(q["field"].as_array().unwrap().len(), 3);
    let odds = q["odds"].as_array().unwrap();
    let total: f64 = odds.iter().map(|o| o["p_win"].as_f64().unwrap()).sum();
    assert!((total - 1.0).abs() < 1e-9);
    assert!(odds.iter().all(|o| o["decimal_odds"].as_f64().unwrap() >= 1.1));
}
