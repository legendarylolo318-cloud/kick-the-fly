# Performance

How fast the brain simulation runs on each compute backend, and how many flies each one keeps in real time. The
backend is picked in Settings > Brain > Compute backend or with `--backend NAME` (README: Run from source).

A brain steps every 5 ms of brain time, so real time is 200 steps/s. "Paced" is whether a brain held to real time
keeps up; "uncapped" is how fast it steps when it isn't held back, as a multiple of real time.

```bash
python kick_the_fly.py --headless --benchmark --backend NAME --flies 1 8 16 --seconds 5
```

Lab > Simulation benchmark runs the same thing in the game and reports the backend that actually ran.

## 2.9

AMD Radeon RX 9070 XT (radeonsi, Mesa, OpenGL 4.6) / Intel Core Ultra 7 270K Plus (24 cores), 32 GB, Linux 7.2,
Python 3.14.7, NumPy 2.5.3, Numba 0.67.0, ModernGL 5.12.0.

| brains | `cpu` (NumPy) paced / uncapped | `numba` paced / uncapped | `gl` (OpenGL compute) paced / uncapped |
|---|---|---|---|
| 1 | 1.00x / 4.09x (819 steps/s, 1.22 ms) | 1.00x / 4.72x (944 steps/s, 1.06 ms) | 1.00x / 2.11x (422 steps/s, 2.37 ms) |
| 8 | 1.00x / 2.32x | 1.00x / 3.20x | 0.22x / 0.22x |
| 16 | 0.91x / 0.91x | 1.00x / 1.89x | 0.11x / 0.11x |
| synaptic events/s, best | 963 M (8 brains) | 1,571 M (16 brains) | 117 M |
| memory, 16 brains | 6.7 GB | 6.7 GB | 6.2 GB |

- **Numba** gives identical spikes to NumPy and scales best on the CPU, because its kernels release Python's
  interpreter lock so each brain's thread runs on its own core. `auto` picks it when it's installed.
- **gl** is slower than NumPy for one brain here and doesn't run brains in parallel: the brains share one GPU queue
  and each step waits for its readback. So `auto` never picks it (the order is PyTorch GPU, then Numba, then NumPy);
  choose it yourself if you want to try it, on any vendor's GPU with OpenGL 4.3.
- **torch-cuda / torch-rocm** weren't re-measured for 2.9 (no CUDA or ROCm build of PyTorch on the test machine); the
  2.8 numbers are below.

### gl: learning no longer re-uploads the whole matrix (issue #2)

The mushroom body's learning changes KC -> MBON synapses every 50 ms of brain time. Until 2.9 the gl backend then
re-sent the whole 41 MB weight buffer. It now sends only the synapses that changed, as (index, value) pairs that a
small compute shader writes into place. `tools/bench_gl_plastic.py` replays the updates of a real 10-pairing
conditioning session (median 8,500 changed synapses per update) through each path:

| upload path | bytes per update | time per update (median) | the session's 400 updates |
|---|---|---|---|
| whole matrix (before) | 41,088,500 | 2.103 ms | 16,435 MB |
| contiguous runs, `buffer.write(offset=...)` | 88,356 | 0.151 ms | 34 MB |
| **scatter shader (2.9)** | **68,000** | **0.032 ms** | **26 MB** |

The learned weights on the GPU stay bit-exact with NumPy's: `tests/test_backends.py` runs 10 shock pairings on gl,
replays the learning rule's inputs on NumPy and compares every KC -> MBON weight read back from the GPU.

## Fly cap

How many flies N can spawn, from `get_max_flies()` in `kickthefly/game/kick_the_fly.py`:

| backend | cap |
|---|---|
| `cpu` (NumPy), `torch-cpu` | 16 |
| `numba` | one per CPU core, at least 16 and at most 32 |
| `gl`, `torch-cuda`, `torch-rocm` | 32, or 64 with `KICK_THE_FLY_EXPANDED_SWARM=1` |

Each fly also needs about 350 MB of free memory, checked when you press N. The cap is how many you may spawn, not how
many keep real time: see the tables (gl in particular falls behind with more than one).

## 2.8: simulation backends and GPU acceleration

Measured for 2.8 on an AMD Radeon RX 9070 XT / Intel Core Ultra 7 270K Plus (24 cores), 32 GB, Linux, Python 3.14,
NumPy 2.5, Numba 0.67, PyTorch 2.14 (ROCm build), ModernGL 5.12.

| brains | `cpu` paced / uncapped | `numba` paced / uncapped | `torch-rocm` / `torch-cuda` paced / uncapped |
|---|---|---|---|
| 1 | 1.00x / 4.17x (834 steps/s) | 1.00x / 4.76x (951 steps/s) | **1.00x / 7.25x** (0.69 ms/fly) |
| 8 | 1.00x / 2.21x | 1.00x / 3.11x | **1.00x / 15.62x** (0.32 ms/fly batched) |
| 16 | 0.87x / 0.91x | 1.00x / 1.83x | **1.00x / 15.80x** (0.32 ms/fly batched) |
| 32 | 0.35x / 0.38x | 0.71x / 0.79x | **1.00x / 15.15x** (0.33 ms/fly batched) |
| synaptic events/s, best | 915 M (8 brains) | 1,517 M (16 brains) | **7,850 M** (32 brains) |
| memory, 32 brains | 12.8 GB | 10.5 GB | 4.8 GB (VRAM) |

- **Batched multi-fly SpMM (`torch-rocm`, `torch-cuda`)** combines per-fly spike vectors into a single
  `(166,700 x N)` tensor, streaming the ~129 MB connectome matrix from VRAM once per step instead of N times.
- **Device-resident state:** membrane potentials, refractory counters, spike buffers and pre-scaled noise stay in
  VRAM across steps.
- The 2.8 README also listed `gl` at 11.57x uncapped for one brain and 1.95x for 32. Those numbers don't reproduce
  on the same hardware (2.9's measurement above), so they were dropped.

## Earlier releases

AMD Radeon RX 9070 XT / 24-thread CPU, Python 3.11 (`tools/bench_sim.py`, and the 3D game with flies spawned):

| | before (2.5.0) | after (2.6.0) | after restructure (2.7) |
|---|---|---|---|
| 1 brain, paced / uncapped | 1.00x real time / 4.17x | 1.00x / 4.21x | 1.00x / 4.04x (807.4 steps/s, 134.6 M neurons/s, 1079 MB) |
| 8 brains, no game loop, paced / uncapped | 1.00x / 2.16x | 1.00x / 2.21x | 1.00x / 2.13x (425.4 steps/s/fly, 567.4 M neurons/s, 3178 MB) |
| 3D game, 1 fly | 200 steps/s (1.00x), 62 fps | 200 steps/s (1.00x), 62 fps | 200 steps/s (1.00x), 62 fps |
| 3D game, 8 flies | 0.41x real time, 60 fps | 0.43x real time, 59 fps | 0.43x real time, 59 fps |

With several flies in the game, the brain threads, the renderer and the brain view share Python's interpreter lock,
so the brains fall behind real time. That was already true before 2.6 and isn't changed by it.

The 3D game in each arena, 2.7 (`python kick_the_fly.py --arena NAME --flies N --smoke 60`; fps averaged over the
second half, sim/real is each brain's steps per second over the 200 of real time, mean over flies and the slowest
fly):

| | 1 fly | 8 flies |
|---|---|---|
| Room | 1.00x real time, 62 fps | 0.39x (slowest 0.36x), 58 fps |
| Open field | 1.00x, 62 fps | 0.34x (slowest 0.32x), 58 fps |
| Orchard | 1.00x, 62 fps | 0.32x (slowest 0.31x), 55 fps |

Outdoors, scenery further than 38 m (grass beyond 16 m) or well behind the camera isn't drawn, static scenery is built
once per arena, distant trees are skipped by the fly's collision checks, and distant ground fades into haze. Before
those, the orchard starved the brain to 0.07x real time with a single fly.
