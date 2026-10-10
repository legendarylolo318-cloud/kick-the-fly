//! Simulation performance benchmark for 1, 8 and 16 flies (`lab/benchmark.py`): paced steps/s (real-time keeping), uncapped throughput,
//! neurons/second, synapse events/second, sim-time against real-time, and the process's memory.

use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::Brain;
use crate::csr::Weights;
use crate::lif::{LifParams, LifSim};
use crate::pack::Labels;

pub const FILE: &str = "benchmark_results.json";

/// The process's resident memory in MB (Linux: `/proc/self/statm`; elsewhere 0).
pub fn memory_mb() -> f64 {
    std::fs::read_to_string("/proc/self/statm")
        .ok()
        .and_then(|t| t.split_whitespace().nth(1).and_then(|p| p.parse::<f64>().ok()))
        .map_or(0.0, |pages| pages * 4096.0 / (1024.0 * 1024.0))
}

pub fn system_info() -> Value {
    let cpu = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|t| t.lines().find(|l| l.starts_with("model name")).and_then(|l| l.split_once(':').map(|x| x.1.trim().to_string())))
        .unwrap_or_else(|| "Unknown CPU".into());
    json!({
        "os": std::env::consts::OS, "cpu_count": std::thread::available_parallelism().map_or(1, |n| n.get()), "cpu_model": cpu,
        "python": format!("Rust {}", env!("CARGO_PKG_VERSION")),
    })
}

/// Steps per second of each brain over `seconds`, and the spikes per step; `speed` 1.0 keeps the 200 steps/s pace, anything larger runs flat out.
fn measure(brains: &mut [Brain], seconds: f64, paced: bool) -> (Vec<f64>, f64) {
    let dur = Duration::from_secs_f64(seconds);
    let spikes = std::sync::atomic::AtomicU64::new(0);
    let steps: Vec<u64> = std::thread::scope(|s| {
        let handles: Vec<_> = brains
            .iter_mut()
            .map(|b| {
                let spikes = &spikes;
                s.spawn(move || {
                    let t0 = Instant::now();
                    let mut n = 0u64;
                    while t0.elapsed() < dur {
                        if paced {
                            // the next step is due at (n + 1) * 5 ms
                            let due = Duration::from_millis(5) * (n as u32 + 1);
                            if let Some(wait) = due.checked_sub(t0.elapsed()) {
                                std::thread::sleep(wait);
                            }
                        }
                        let k = b.step().len() as u64;
                        spikes.fetch_add(k, std::sync::atomic::Ordering::Relaxed);
                        n += 1;
                    }
                    (n, t0.elapsed().as_secs_f64())
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().map_or(0, |(n, _)| n)).collect()
    });
    let total: u64 = steps.iter().sum();
    let rates = steps.iter().map(|&n| n as f64 / seconds).collect();
    (rates, spikes.load(std::sync::atomic::Ordering::Relaxed) as f64 / total.max(1) as f64)
}

/// `benchmark.run_benchmark`; `progress(done, total, label)`.
pub fn run(
    weights: &Arc<Weights>,
    labels: &Arc<Labels>,
    fly_counts: &[usize],
    seconds: f64,
    mut progress: impl FnMut(usize, usize, &str),
) -> Value {
    let (n_neurons, n_synapses) = (weights.n(), weights.csr.nnz());
    let mut records = Vec::new();
    for (idx, &n) in fly_counts.iter().enumerate() {
        progress(idx, fly_counts.len(), &format!("benchmarking {n} flies"));
        let mut brains: Vec<Brain> =
            (0..n as u64).map(|s| Brain::warmed_up(weights.clone(), labels.clone(), LifParams::default(), s, 50)).collect();
        let (paced_rates, _) = measure(&mut brains, seconds, true);
        let mean_paced = crate::validation::mean(&paced_rates);
        let min_paced = paced_rates.iter().copied().fold(f64::INFINITY, f64::min);
        let rt_target = 1000.0 / LifParams::default().dt_ms;
        let (unc, spikes_per_step) = measure(&mut brains, seconds, false);
        let mean_unc = crate::validation::mean(&unc);
        let agg: f64 = unc.iter().sum();
        let r1 = |x: f64| (x * 10.0).round() / 10.0;
        records.push(json!({
            "flies": n, "paced_steps_per_s": r1(mean_paced), "paced_min_steps_per_s": r1(min_paced),
            "paced_realtime_ratio": (mean_paced / rt_target * 1000.0).round() / 1000.0,
            "uncapped_steps_per_s": r1(mean_unc), "uncapped_latency_ms": if mean_unc > 0.0 { (1000.0 / mean_unc * 1000.0).round() / 1000.0 } else { 0.0 },
            "uncapped_realtime_ratio": (mean_unc / rt_target * 100.0).round() / 100.0, "agg_uncapped_steps_per_s": r1(agg),
            "neurons_per_sec": (agg * n_neurons as f64).round(), "active_fraction": (spikes_per_step / n_neurons as f64 * 10_000.0).round() / 10_000.0,
            "synapses_per_sec": (agg * spikes_per_step * (n_synapses as f64 / n_neurons.max(1) as f64)).round(), "memory_mb": r1(memory_mb()),
        }));
        drop(brains);
    }
    progress(fly_counts.len(), fly_counts.len(), "complete");
    json!({
        "system": system_info(), "backend": "Rust CPU (rayon)", "device": "CPU",
        "connectome": {"neurons": n_neurons, "synapses": n_synapses}, "seconds_per_run": seconds, "records": records,
    })
}

/// The sim of one fly, for tests of the pacing.
pub fn one_brain(weights: &Arc<Weights>, labels: &Arc<Labels>) -> Brain {
    let sim = LifSim::new(weights.clone(), LifParams::default(), 0);
    Brain::new(sim, labels.clone())
}
