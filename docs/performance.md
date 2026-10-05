# Performance

How fast the brain simulation runs on each compute backend, and how many flies each one keeps in real time. The
backend is picked in Settings > Brain > Compute backend or with `--backend NAME` (README: Run from source).

**Auto (3.1.0).** In the interactive game, `auto` tries `gl` first (only on a real GPU: a software rasterizer such as llvmpipe has the compute shaders but
runs them on the CPU, so it does not count), then a PyTorch GPU, then Numba, then NumPy, moving on at any error; Settings > Brain shows the engine in use and
the chain. Headless runs (`--validate`, protocols, bundles, replays, the selftest) keep the exact chain (PyTorch GPU, Numba, NumPy) so their numbers do not
move; `KICK_THE_FLY_AUTO=exact` gives the game the same chain. Pick `cpu` or `numba` for spike-for-spike runs. gl is fastest for one brain; Numba is faster for
8 to 16 brains on a CPU (table below).

A brain steps every 5 ms of brain time, so real time is 200 steps/s. "Paced" is whether a brain held to real time
keeps up; "uncapped" is how fast it steps when it isn't held back, as a multiple of real time.

```bash
python kick_the_fly.py --headless --benchmark --backend NAME --flies 1 8 16 32 --seconds 5
```

Lab > Simulation benchmark runs the same thing in the game and reports the backend that actually ran.

## The profiler and the frame profile (3.1.0)

**F3** (rebindable in Settings > Controls; off at every launch) puts a card over the game: FPS, the frame time with its 95th percentile and worst, a
strip of the last frames (green and amber lines at 60 and 30 fps), the milliseconds of each part of a frame, the compute engine and the brain view's
renderer. Tag: **GAME RULE** (it reads clocks and counters; nothing the simulation does). The rows are milliseconds of the game's own thread, per frame,
over the last 180 frames:

| row | what it is |
|---|---|
| `sim` | the brains' own step time summed over the steps they took this frame. The brains run on threads of their own, so this is compute used while the frame ran, **not** time the frame waited (flies batched on one GPU overlap, so it can exceed the frame) |
| `physics` | the fly's physics, the tools, predators, the rest of the game update, and the player's movement |
| `render` | 3D: building and submitting the scene; 2D: drawing the arena |
| `ui` | the HUD, the brain panel, menus and the science card |
| `present` | swapping buffers: where vsync, the frame cap and a busy GPU show |

`render`, `ui` and `physics` are CPU time. Sections nest and the inner time is taken out of the outer, so the rows add up to the frame. With the profiler off
each section costs one method call returning a shared no-op context (under 2 microseconds).

`python kick_the_fly.py --benchmark` runs the simulation table above and then the **frame profile**: the game's own smoke run, seed 1, the room, 4 flies in
3D and one in 2D, offscreen (EGL; no display needed, and the monitor cannot cap it), ten seconds of which the first five are warm-up, one row each for
3D at the 60 fps cap (what a player gets), 3D uncapped (how fast the frame can go; flat out it competes with a GPU brain for the GPU, so its steps/s drop),
3D with the brain view on the CPU, and 2D. `--no-render-bench` skips it. Each child's numbers also land in `benchmark_results-frames.json` next to the
simulation results. On this machine (an RX 9070 XT; every number is in `HANDOFF.md`) the GPU brain view takes `ui` from 6.6 to 4.8 ms and
`physics` from 3.5 to 2.5 ms in the capped 3D scene, and the brains keep 198 steps/s against 174 with the CPU view (the CPU view's sparse products compete
with the brains for the interpreter).

## Where the simulation runs (3.1.0)

Each fly's brain steps on a thread of its own at a fixed 200 steps per second of brain time (times the game speed: slow motion scales it, pause stops
it, single-step runs one), paced by the wall clock, while the game's frame loop draws and moves the fly at 60 Hz and only reads the brain's latest group
rates. On the gl backend the GPU work runs on a further thread per group of flies. That was already so in 3.0; 3.1.0 checks it and fixes one way it
failed: CPython gives a waiting thread the GIL only every 5 ms while another thread is running Python, so a busy frame left the brain **27 of its 200
steps/s** (measured with a main thread that never yields; 0.001 s: 78, 0.0005 s: 163, 0.0002 s: 200). The brain thread now lowers the interpreter's switch
interval to 0.2 ms when it starts. The spikes are unchanged: the same seed gives the same spikes whether a thread or a caller steps the brain
(`tests/test_brain_thread.py`). Rendering does not interpolate brain state: nothing the player sees moves with the brain's 5 ms steps (the fly's body
moves at the game's 60 Hz tick from the rates read that frame), and the brain panel reads its own 20 Hz view. A separate *process* would avoid the GIL
altogether and was not built.

**In the game, the GIL is the limit, not the GPU or the CPU cores (3.1.0 review).** The headless table above has no render thread; in the game the
render thread and every brain thread share one interpreter lock, so with many flies the brains fall behind real time while the GPU and most
cores sit idle. The review removed the render thread's longest lock-holding calls (the HUD was converted to RGBA bytes every frame, ~1 ms at
1280x760 and ~4 ms at 1440p; every limb segment ran numpy on 3-vectors; the memory card rescored every smell against 41k synapses every frame).
Measured in the 3D game, RX 9070 XT, 60 fps cap unless noted, steady second half of a 60-75 s run, quiet machine (steps/s per fly; 200 is real time):

| scene | 3.0.0-rc.2 (NumPy) | 3.1.0 before the review fixes (gl) | 3.1.0 (gl) |
|---|---|---|---|
| 1 fly | 200, GPU 9% busy | 200 | 200, GPU 31% busy, main thread 28% (rc.2 43%) |
| 4 flies | 200 | 200 | 200 |
| 8 flies | 70 (0.35x) | 78 (0.39x), 52 fps | 132 (0.66x), 58 fps |
| 1 fly, no fps cap | 59 (0.29x) | 133 (0.67x), 170 fps | 187 (0.94x), 302 fps |

So up to about 4-5 flies keep real time in the game on this machine; beyond that the brains slow down (the game shows steps/s in the profiler, F3)
even though the headless benchmark runs 16 gl flies in real time. The remaining main-thread cost is mostly the flies' physics and drawing.

**The brain process (3.1.0 release review).** The fix for that limit: the brains now run in a process of their own (`kickthefly/core/brainproc.py`; Settings > Brain > Run the brains in their own process). Same machine, 16 flies: brains 89 -> 195 steps/s at a 240 fps cap and 83 -> 197 at 60, 19 -> 50 fps, GPU 24% -> 48% busy; one fly uncapped 200 steps/s at 311 fps. With 16 flies the game's own thread (physics and drawing for every fly) is now the limit on fps.

## 2.10

Same machine as 2.9: AMD Radeon RX 9070 XT (radeonsi, Mesa, OpenGL 4.6) / Intel Core Ultra 7 270K Plus (24 cores),
32 GB, Linux 7.2, Python 3.14.7, NumPy 2.5.3, Numba 0.67.0, ModernGL 5.12.0. All four backends measured the same day,
`--flies 1 8 16 32 --seconds 5`.

| brains | `cpu` (NumPy) paced / uncapped | `numba` paced / uncapped | `gl` paced / uncapped | `gl`, flies per dispatch |
|---|---|---|---|---|
| 1 | 1.00x / 4.14x (829 steps/s, 1.21 ms) | 1.00x / 4.71x (942 steps/s, 1.06 ms) | **1.00x / 5.20x** (1,040 steps/s, 0.96 ms) | 1.0 |
| 8 | 1.00x / 2.28x | **1.00x / 3.32x** | 1.00x / 2.48x | 8.0 |
| 16 | 0.94x / 0.94x | **1.00x / 1.91x** | 1.00x / 1.49x | 12.8 |
| 32 | 0.37x / 0.38x | 0.78x / 0.82x | **0.86x** / 0.78x | 8.7 |
| synaptic events/s, best | 944 M (8 brains) | 1,588 M (16 brains) | 1,313 M (32 brains) | |
| memory, 32 brains | 9.9 GB | 9.8 GB | 9.8 GB | |

### gl: batched multi-fly

Until 2.9 each gl brain had a context of its own and read the whole connectome (41 MB of weights plus 41 MB of
column indices) every step, so eight brains read it eight times and queued behind each other on the GPU. In 2.10 the
brains of a process join a batch group (`_GLGroup` in `kickthefly/sim/connectome/backends.py`), the same idea as the
PyTorch backends' batched SpMM:

- One GL context on a thread of its own holds the connectome once and the state of up to 32 flies. Each brain still
  steps on its own thread and hands its step to the group; the group waits up to 2 ms for the others that are running
  and dispatches all of them together. One SpMM pass streams the weights once for every fly in the dispatch (a fly's
  spikes are one bit of a 32-bit word per neuron), then one LIF pass updates them all.
- Flies learn separately. A row whose weights differ between flies (the KC -> MBON rows, once learning touches them)
  becomes per-fly: the shader reads that row's weights from a per-fly table and every other row from the shared copy.
  Learning still uploads only the synapses it changed. A fly whose weights differ in too many rows (a lesion, a
  synapse threshold) moves to a group of its own.
- Each fly's input is summed over its row's synapses in CSR order, whatever the batch, so batched and unbatched gl are
  bit-exact with each other: `tests/test_backends.py::test_gl_batched_multi_fly_plastic_weights_bit_exact` conditions
  two flies on different odors on their own threads, batched and not, and compares every learned KC -> MBON weight,
  the weights read back from the GPU and the spikes.
- `KICK_THE_FLY_GL_BATCH=0` gives every brain a group of its own (the unbatched path, same shaders).

What batching is worth, same kernels, `--backend gl`:

| brains | unbatched (`KICK_THE_FLY_GL_BATCH=0`) paced / uncapped | batched paced / uncapped | memory, unbatched / batched |
|---|---|---|---|
| 1 | 1.00x / 5.20x | 1.00x / 5.20x | 0.9 / 0.9 GB |
| 8 | 0.91x / 0.90x | 1.00x / 2.48x | 3.5 / 3.0 GB |
| 16 | 0.42x / 0.42x | 1.00x / 1.49x | 7.8 / 5.4 GB |
| 32 | 0.21x / 0.21x | 0.86x / 0.78x | 16.2 / 9.8 GB |

Three other changes are behind the single-brain speed (2.37 ms per step in 2.9, 0.96 ms now):

- **A workgroup per row.** The old SpMM gave each neuron's row to one thread. The longest row has 9,184 synapses
  (median 35), and that one thread's chain of dependent loads set the time of the whole pass. Now 64 lanes look up
  64 synapses' presynaptic spikes at once and lane f adds fly f's weights in order.
- **No Python lock while the GPU works.** ModernGL holds Python's GIL through every GL call, `ctx.finish()` and
  buffer reads included, so every brain thread (and the game) stood still while the GPU ran. The group waits on a GL
  fence through ctypes instead, which lets them go (`_GLFence`; falls back to `ctx.finish()` where the entry points
  can't be found).
- **The LIF step in NumPy's order.** The shader does `CPUBackend.step`'s float operations in its order, with
  `precise` so they aren't fused into multiply-adds. On this GPU gl now gives NumPy's spikes exactly (0 of 166.7 M
  differ over 1,000 steps, sparse and dense paths). That is an observation, not a promise: another driver may round
  or flush denormals differently, so gl stays under the statistical tolerance in `tests/test_backends.py`.

The limit at 32 brains is Python, not the GPU. Bare `sim.step()` loops on 32 threads keep 1.18x real time with 27.6
flies per dispatch (8: 2.98x, 16: 1.88x). With the benchmark's full `Brain` step around each, the brains' own Python
work shares one interpreter lock and they reach the group spread out, 8.7 flies per dispatch.

VRAM: the connectome once per group (about 85 MB) plus about 45 MB per fly (its 16-step noise bank is most of it), so
about 1.5 GB for 32.

Learning uploads (`tools/bench_gl_plastic.py`, the same 10-pairing session as below): scatter 68 KB and 0.074 ms per
update, runs 88 KB and 0.353 ms, full re-upload 41 MB and 2.016 ms. The bytes are 2.9's; the times now include the
group keeping its host copy of the shared weights, and are measured on the group's thread.

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
| `gl` | 32 |
| `torch-cuda`, `torch-rocm` | 32, or 64 with `KICK_THE_FLY_EXPANDED_SWARM=1` |

Each fly also needs about 350 MB of free memory, checked when you press N. The cap is how many you may spawn, not how
many keep real time: see the tables.

gl's cap, revisited in 2.10 from the numbers above: 16 flies keep real time with room to spare (1.49x uncapped) and
32 run at 0.86x, better than Numba (0.78x) and NumPy (0.37x) at 32, so 32 stays. `KICK_THE_FLY_EXPANDED_SWARM` no
longer raises it to 64 on gl: one group holds 32 flies, a 33rd starts a second group that reads the weights again
every step, and that hasn't been measured. Until 2.10 the gl cap of 32 had no measurement behind it (0.11x at 16).

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

## 2.11: larva and individuality (short run, cloud box)

**Not the user's PC.** Measured during the 2.11 review on a 4-core cloud VM (Intel Xeon @ 2.80 GHz, Linux 6.18,
Python 3.11.15, NumPy 2.4.6, Numba 0.67.0), single process, 5 s of wall-clock time per row. This box is much slower
than the RX 9070 XT / Core Ultra 7 machine above (its adult NumPy brain runs 275 steps/s here vs 829 there), so compare
rows with each other, not with the tables above. Each brain is a full game `Brain` built by `simcore.new_brain`
(50-step warm-up), stepped uncapped; "larvae in turn" steps N larval brains one after another in one thread and gives
steps/s per brain. The adult is 166,700 neurons (10,272,125 connections, the nonzero entries of the weight matrix); the larva is 2,952 neurons (110,677
connections, 352,611 synapses).

| brain | backend | individuality | steps/s per brain | x real time |
|---|---|---|---|---|
| larva | `cpu` | off | 3,137 | 15.7x |
| larva | `cpu` | subtle | 3,132 | 15.7x |
| larva | `numba` | off | 3,986 | 19.9x |
| larva | `numba` | subtle | 3,806 | 19.0x |
| adult | `cpu` | off | 275 | 1.37x |
| adult | `cpu` | subtle | 245 | 1.23x |
| adult | `numba` | off | 325 | 1.63x |
| adult | `numba` | subtle | 296 | 1.48x |
| 16 larvae in turn | `cpu` | off | 167 | 0.83x |
| 64 larvae in turn | `cpu` | off | 35 | 0.18x |
| 16 larvae in turn | `numba` | off | 226 | 1.13x |
| 64 larvae in turn | `numba` | off | 46 | 0.23x |

- A larva step costs about 0.25-0.32 ms, most of it the `Brain` step's Python work, not the 2,952-neuron sparse
  product: 64 larvae stepped in turn are 64 times that and do not keep real time here.
- Individuality `subtle` cost 0-11% in these single 5 s runs (larva cpu 0%, numba 5%; adult cpu 11%, numba 9%): the
  D_pre scaling of the spike vector and D_post of the input are two extra O(n) multiplies per step. Not "< 0.4%".
- Gemini's 2.11 numbers (larva 161.6x / 213.6x real time for one larva, 2.40x / 3.39x for 64, "adult 139,255
  neurons") were not reproduced and are removed. The larva fly caps in `get_max_flies()` (64 NumPy / 128 Numba and GPU)
  rest on those numbers and are unmeasured; the windowed larva game is not playable yet, so they are not reached.
