//! `cargo bench -p fly_sim`: simulation steps per second.
//!
//! Criterion reports time per step and, through `Throughput::Elements(1)`, steps per second (the game needs 200/s per fly).
//! Uses the real brain pack when `data/kick_brain.npz` (or `$KICK_THE_FLY_PACK`) exists, otherwise a synthetic network of the
//! same size and in-degree so the number is still meaningful on a machine without the data.

use std::sync::Arc;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use fly_sim::brain::STIM_AMP;
use fly_sim::csr::Weights;
use fly_sim::lif::{LifParams, LifSim};
use fly_sim::pack::{self, Pack};
use fly_sim::synthetic;

struct Net {
    label: &'static str,
    weights: Arc<Weights>,
    drive_rows: Vec<usize>,
}

fn nets() -> Vec<Net> {
    let mut nets = Vec::new();
    match pack::find_default().map(Pack::load) {
        Some(Ok(p)) => {
            let rows = p.labels.types.iter().enumerate().filter(|(_, t)| *t == "LPLC2" || *t == "LC4").map(|(i, _)| i).collect();
            nets.push(Net { label: "malecns-166700", weights: Arc::new(p.weights), drive_rows: rows });
        }
        other => {
            if let Some(Err(e)) = other {
                eprintln!("brain pack failed to load ({e}); benchmarking synthetic networks only");
            } else {
                eprintln!("no brain pack found; benchmarking synthetic networks only");
            }
            let w = synthetic::random_weights(166_700, 62, 1, Weights::default_blocks());
            nets.push(Net { label: "synthetic-166700", weights: Arc::new(w), drive_rows: (0..6000).collect() });
        }
    }
    let small = synthetic::random_weights(20_000, 62, 2, Weights::default_blocks());
    nets.push(Net { label: "synthetic-20000", weights: Arc::new(small), drive_rows: (0..1000).collect() });
    nets
}

fn bench_steps(c: &mut Criterion) {
    let mut group = c.benchmark_group("sim_step");
    group.throughput(Throughput::Elements(1));
    group.sample_size(30);
    group.measurement_time(Duration::from_secs(6));
    for net in nets() {
        let n = net.weights.n();
        let mut sim = LifSim::new(net.weights.clone(), LifParams::default(), 1000);
        for _ in 0..400 {
            sim.step(None); // settle the gain controller before measuring
        }
        group.bench_with_input(BenchmarkId::new("calm", net.label), &(), |b, _| {
            b.iter(|| sim.step(None).len());
        });
        let mut cur = vec![0f32; n];
        for &r in &net.drive_rows {
            cur[r] = STIM_AMP;
        }
        group.bench_with_input(BenchmarkId::new("driven", net.label), &(), |b, _| {
            b.iter(|| sim.step(Some(&cur)).len());
        });
    }
    group.finish();
}

criterion_group!(benches, bench_steps);
criterion_main!(benches);
