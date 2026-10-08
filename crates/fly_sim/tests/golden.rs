//! Golden tests: the Rust simulator against the Python reference outputs in `tests/golden/` (written by
//! `legacy-python/make_golden.py` from `kickthefly/sim/connectome/sim.py`).
//!
//! # Tolerances
//!
//! | quantity | tolerance |
//! |---|---|
//! | pack weights, labels | exact (SHA-256 of the f32 / i32 arrays) |
//! | NumPy random stream (PCG64, float32 ziggurat bank, bounded integers, float64 normals) | exact bits |
//! | spikes, every step | exact: the SHA-256 of the whole spike set must match, so the full 166,700-neuron raster is compared, not a sample |
//! | monitored neurons' spikes and final membrane potentials | exact (f32 bit patterns) |
//! | gain controller | relative 1e-12 (it is f64; `exp` may differ from NumPy's by one ulp) |
//! | sums of floats (`rate_ema_sum`) | relative 1e-6 (summation order differs; the per-neuron arrays are compared exactly) |
//! | individuality gains | relative 1e-6 (NumPy's float32 `exp` is a SIMD routine whose last bit varies by CPU; the normals are exact) |
//!
//! The spike comparison is exact because the whole step is reproduced operation for operation in `f32`: the same noise bank and
//! offsets (`rng`), the same summation order (`csr`), the same update order (`lif`). If a run stops matching, `rng_matches_numpy`
//! and `pack_matches_python` say whether the generator or the loader is to blame before the step is.
//!
//! The brain pack (`data/kick_brain.npz`, 39 MB) is not in the repository. Without it these tests print `SKIPPED` and pass, unless
//! `KTF_REQUIRE_PACK=1` is set (CI for releases should set it).

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use fly_sim::brain::{rows_of, INHIB};
use fly_sim::lif::{Individuality, LifParams, LifSim};
use fly_sim::pack::{self, Pack};
use fly_sim::rng::Generator;
use fly_sim::util::{sha16, sha16_f32, sha16_u32, sha256_hex};
use serde_json::Value;

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden")
}

fn load_json(name: &str) -> Value {
    let path = golden_dir().join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("bad JSON in {}: {e}", path.display()))
}

/// The pack, loaded once for all tests; `None` (after printing why) when it is not installed.
fn pack() -> Option<&'static Pack> {
    static PACK: OnceLock<Option<Pack>> = OnceLock::new();
    PACK.get_or_init(|| match pack::find_default() {
        Some(p) => Some(Pack::load(&p).expect("the brain pack failed to load")),
        None => {
            assert!(
                std::env::var("KTF_REQUIRE_PACK").is_err(),
                "KTF_REQUIRE_PACK is set but data/kick_brain.npz was not found (set KICK_THE_FLY_PACK)"
            );
            eprintln!("SKIPPED: data/kick_brain.npz not found; golden tests need the brain pack (set KICK_THE_FLY_PACK)");
            None
        }
    })
    .as_ref()
}

fn bits32(v: &Value) -> Vec<u32> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()
}

fn rel_close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * a.abs().max(b.abs()).max(1e-300)
}

#[test]
fn pack_matches_python() {
    let Some(pack) = pack() else { return };
    let m = load_json("manifest.json");
    let p = &m["pack"];
    assert_eq!(pack.n as u64, p["n"].as_u64().unwrap());
    assert_eq!(pack.nnz() as u64, p["nnz"].as_u64().unwrap());
    let csr = &pack.weights.csr;
    assert_eq!(sha16_f32(&csr.data), p["w_data_sha"].as_str().unwrap(), "rate-normalised weights differ from scipy's");
    let indptr: Vec<u8> = csr.indptr.iter().flat_map(|v| (*v as i32).to_le_bytes()).collect();
    assert_eq!(sha16(&indptr), p["indptr_sha"].as_str().unwrap());
    let indices: Vec<u8> = csr.indices.iter().flat_map(|v| (*v as i32).to_le_bytes()).collect();
    assert_eq!(sha16(&indices), p["indices_sha"].as_str().unwrap());
    let types = pack.labels.types.join("\n");
    assert_eq!(sha256_hex(types.as_bytes())[..16], *p["type_sha"].as_str().unwrap());
    let file = Pack::file_sha256(pack.path.as_ref().unwrap()).unwrap();
    assert_eq!(file, p["sha256"].as_str().unwrap(), "the golden files were made from a different brain pack file");
    assert_eq!(pack.labels.brain_type, "adult");
}

#[test]
fn rng_matches_numpy() {
    let r = load_json("rng.json");
    let n = 166_700usize;
    for (seed, expect) in r["seeds"].as_object().unwrap() {
        let seed: u64 = seed.parse().unwrap();
        let mut g = Generator::new(seed);
        let mut bank = vec![0f32; n * 16];
        g.fill_standard_normal_f32(&mut bank);
        for x in &mut bank {
            *x *= 0.05f32;
        }
        let first: Vec<u32> = bank[..16].iter().map(|x| x.to_bits()).collect();
        assert_eq!(first, bits32(&expect["bank_first_bits"]), "seed {seed}: first noise values (SeedSequence / PCG64 / ziggurat)");
        let last: Vec<u32> = bank[bank.len() - 16..].iter().map(|x| x.to_bits()).collect();
        assert_eq!(last, bits32(&expect["bank_last_bits"]), "seed {seed}: last noise values");
        assert_eq!(sha16_f32(&bank), expect["bank_sha"].as_str().unwrap(), "seed {seed}: the whole noise bank");
        let offs: Vec<u64> = (0..32).map(|_| g.integers(0, (bank.len() - n) as u64)).collect();
        let want: Vec<u64> = expect["offsets"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect();
        assert_eq!(offs, want, "seed {seed}: bounded integers (Lemire, buffered 32-bit)");
    }
    let mut g = Generator::new(1000);
    let raw: Vec<u64> = (0..8).map(|_| g.random_raw()).collect();
    let want: Vec<u64> = r["u64_first"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect();
    assert_eq!(raw, want, "PCG64 raw output");
    let mut g = Generator::new(1000);
    let normals: Vec<u64> = (0..16).map(|_| g.standard_normal_f64().to_bits()).collect();
    let want: Vec<u64> = r["normal_f64_first_bits"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect();
    assert_eq!(normals, want, "float64 ziggurat normals");
}

#[test]
fn individuality_gains_match_numpy() {
    let r = load_json("rng.json");
    for (mode, sigma) in [("subtle", 0.05), ("strong", 0.15)] {
        let want = &r["individuality"][mode];
        let g = Individuality::compute(1000, 166_700, sigma).unwrap();
        for (got, key) in [(&g.d_pre, "pre_first_bits"), (&g.d_post, "post_first_bits")] {
            for (a, b) in got.iter().zip(bits32(&want[key])) {
                let b = f32::from_bits(b);
                assert!(rel_close(f64::from(*a), f64::from(b), 1e-6), "{mode} {key}: {a} vs {b}");
            }
        }
        if sha16_f32(&g.d_pre) != want["pre_sha"].as_str().unwrap() {
            eprintln!("note: {mode} d_pre is not bit-identical to this machine's NumPy float32 exp (allowed, see the tolerance table)");
        }
    }
}

fn f32_slice_bytes_i16(v: &[i16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn run_scenario(name: &str) {
    let Some(pack) = pack() else { return };
    let manifest = load_json("manifest.json");
    let sc = load_json(&format!("{name}.json"));
    let n = pack.n;
    let mon: Vec<u32> = manifest["monitor"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
    let params = LifParams::default();
    // the golden manifest records the parameters it was made with; the Rust defaults must be the same numbers
    let mp = &manifest["params"];
    assert_eq!(mp["dt_ms"].as_f64().unwrap(), params.dt_ms);
    assert_eq!(mp["bias"].as_f64().unwrap(), params.bias);
    assert_eq!(mp["ext_gain"].as_f64().unwrap(), params.ext_gain);
    assert_eq!(mp["noise_std"].as_f64().unwrap(), params.noise_std);
    assert_eq!(mp["syn_gain"].as_f64().unwrap(), params.syn_gain);
    assert_eq!(mp["refractory_steps"].as_i64().unwrap(), i64::from(params.refractory_steps));

    let seed = sc["seed"].as_u64().unwrap();
    let steps = sc["steps"].as_u64().unwrap() as usize;
    let mut sim = LifSim::new(Arc::new(pack.weights.clone()), params, seed);

    struct Input {
        start: usize,
        end: usize,
        op: String,
        amp: f32,
        rows: Vec<u32>,
        ramp_steps: f64,
    }
    let mut inputs = Vec::new();
    for inp in sc["inputs"].as_array().unwrap() {
        let op = inp["op"].as_str().unwrap().to_string();
        let rows: Vec<u32> = if op == "ramp_all" {
            Vec::new()
        } else if inp["spec"] == "rows" {
            inp["rows"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()
        } else {
            let spec = inp["spec"].as_str().unwrap();
            let rows = rows_of(&pack.labels, n, spec).unwrap();
            assert_eq!(rows.len() as u64, inp["n_rows"].as_u64().unwrap(), "{name}: rows_of('{spec}') count");
            let bytes: Vec<u8> = rows.iter().flat_map(|&r| i64::from(r).to_le_bytes()).collect();
            assert_eq!(sha16(&bytes), inp["rows_sha"].as_str().unwrap(), "{name}: rows_of('{spec}') rows");
            rows
        };
        inputs.push(Input {
            start: inp["start"].as_u64().unwrap() as usize,
            end: inp["end"].as_u64().unwrap() as usize,
            op,
            amp: inp["amp"].as_f64().unwrap() as f32,
            rows,
            ramp_steps: inp.get("ramp_steps").and_then(Value::as_f64).unwrap_or(0.0),
        });
    }
    let params_at = sc["params_at"].as_object().unwrap();

    let per_step = sc["per_step"].as_array().unwrap();
    assert_eq!(per_step.len(), steps);
    let mut cur = vec![0f32; n];
    for (t, want) in per_step.iter().enumerate() {
        if let Some(changes) = params_at.get(&t.to_string()) {
            for (k, v) in changes.as_object().unwrap() {
                match k.as_str() {
                    "gain_adapt" => sim.p.gain_adapt = v.as_f64().unwrap(),
                    other => panic!("unknown param {other}"),
                }
            }
        }
        let mut any = false;
        for inp in inputs.iter().filter(|i| i.start <= t && t < i.end) {
            if !any {
                cur.fill(0.0);
                any = true;
            }
            match inp.op.as_str() {
                "set" => inp.rows.iter().for_each(|&r| cur[r as usize] = inp.amp),
                "add" => inp.rows.iter().for_each(|&r| cur[r as usize] += inp.amp),
                "ramp_all" => {
                    let ramp = ((t - inp.start) as f64 / inp.ramp_steps).clamp(0.0, 1.0) as f32;
                    cur.fill(INHIB * ramp);
                }
                other => panic!("unknown op {other}"),
            }
        }
        let spikes = sim.step(if any { Some(&cur[..]) } else { None }).to_vec();
        let wn = want["n"].as_u64().unwrap() as usize;
        assert_eq!(spikes.len(), wn, "{name}: step {t}: spike count");
        assert_eq!(sha16_u32(&spikes), want["sha"].as_str().unwrap(), "{name}: step {t}: the spike set differs (counts agree)");
        let got_mon: Vec<u64> = spikes.iter().filter(|s| mon.binary_search(s).is_ok()).map(|&s| u64::from(s)).collect();
        let want_mon: Vec<u64> = want["mon"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect();
        assert_eq!(got_mon, want_mon, "{name}: step {t}: monitored spikes");
        let wg = want["gain"].as_f64().unwrap();
        assert!(rel_close(sim.gain, wg, 1e-12), "{name}: step {t}: gain {} vs {wg}", sim.gain);
    }

    let end = &sc["end"];
    let v_mon: Vec<u32> = mon.iter().map(|&i| sim.v[i as usize].to_bits()).collect();
    assert_eq!(v_mon, bits32(&end["v_mon_bits"]), "{name}: monitored membrane potentials");
    assert_eq!(sha16_f32(&sim.v), end["v_sha"].as_str().unwrap(), "{name}: all membrane potentials");
    assert_eq!(sha16(&f32_slice_bytes_i16(&sim.refr)), end["refr_sha"].as_str().unwrap(), "{name}: refractory counters");
    assert_eq!(sim.spike_total, end["spike_total"].as_u64().unwrap(), "{name}: total spikes");
    assert_eq!(sha16_f32(sim.activity.rates()), end["rate_ema_sha"].as_str().unwrap(), "{name}: EMA rates");
    let sum: f64 = sim.activity.rates().iter().map(|&x| f64::from(x)).sum();
    assert!(rel_close(sum, end["rate_ema_sum"].as_f64().unwrap(), 1e-6), "{name}: EMA rate sum");
    assert!(rel_close(sim.gain, end["gain"].as_f64().unwrap(), 1e-12));
}

#[test]
fn scenario_calm_seed1000() {
    run_scenario("calm_seed1000");
}

#[test]
fn scenario_calm_seed7() {
    run_scenario("calm_seed7");
}

#[test]
fn scenario_loom_drive() {
    run_scenario("loom_drive");
}

#[test]
fn scenario_touch_pulses() {
    run_scenario("touch_pulses");
}

#[test]
fn scenario_reward_and_surgery() {
    run_scenario("reward_and_surgery");
}

#[test]
fn scenario_death_ramp() {
    run_scenario("death_ramp");
}
