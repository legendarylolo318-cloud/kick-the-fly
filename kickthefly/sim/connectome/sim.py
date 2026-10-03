"""Leaky integrate-and-fire over the signed connectome.

Synaptic input is rate-normalized per neuron: W_in[post, pre] is the signed
synapse count divided by the post neuron's total input synapse count
(zero-sign contacts included), so I_syn is the signed fraction of a neuron's
input synapses whose presynaptic partner spiked last step. A slow global gain
controller holds the population rate near a target.

Propagation is a scipy sparse matmul. When few neurons are active it
multiplies only the active columns of the CSC matrix; otherwise the full CSR
matvec.

    python -m kickthefly.sim.connectome.sim bench
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

if TYPE_CHECKING:  # the loader needs pyarrow; packaged builds that pass W_in never import it
    from kickthefly.sim.connectome.loader import Graph


@dataclass
class LIFParams:
    dt_ms: float = 5.0
    tau_ms: float = 20.0
    v_thresh: float = 1.0
    v_reset: float = 0.0
    refractory_steps: int = 2
    syn_gain: float = 6.0            # initial; adapted toward target_rate_hz
    ext_gain: float = 4.0            # tuned (tools/tune_brain.py round 2, G8): stronger eye drive lets vision reach the DNs
    bias: float = 0.20               # tonic drive per step; resting v = bias / leak = 0.8 of threshold
    noise_std: float = 0.05          # tuned: 0.10 drowned vision (gain control cut coupling to 0.05); 0.03 too quiet
    target_rate_hz: float = 5.0
    gain_adapt: float = 0.002        # per-step log-gain learning rate
    gain_bounds: tuple[float, float] = (1.5, 40.0)
    sparse_path_max_active: float = 0.10  # active fraction below which the column-gather path is used
    # state precision, "float32" or "float64"; KICK_THE_FLY_SIM_DTYPE sets the default (headless --dtype)
    dtype: str = field(default_factory=lambda: os.environ.get("KICK_THE_FLY_SIM_DTYPE", "float32"))
    # compute backend: auto | cpu | numba | torch-cpu | torch-cuda | torch-rocm. KICK_THE_FLY_SIM_BACKEND sets the
    # default for the whole process and its worker processes (headless --backend does that), so validation, assays and
    # protocols run on the backend you asked for.
    backend: str = field(default_factory=lambda: os.environ.get("KICK_THE_FLY_SIM_BACKEND", "auto"))
    # batched multi-fly SpMM on GPU backend
    batched_spmm: bool = field(default_factory=lambda: os.environ.get("KICK_THE_FLY_BATCHED_SPMM", "0").lower() in ("1", "true", "yes"))
    # fused LIF kernel execution on GPU backend (torch.compile / fused kernels)
    fuse_lif: bool = field(default_factory=lambda: os.environ.get("KICK_THE_FLY_FUSE_LIF", "0").lower() in ("1", "true", "yes"))
    # individuality variation: off | subtle | strong
    individuality: str = field(default_factory=lambda: os.environ.get("KICK_THE_FLY_INDIVIDUALITY", "off"))
    individuality_sigma: float | None = None
    # 3.0 day 4 review: the seed the per-neuron gains are drawn from, when it should differ from the noise/warm-up seed (the race
    # assay's R4 holds the brain state fixed and varies only the individual). None, the default, is the sim's own seed, as always.
    individuality_seed: int | None = None


class ActivityBuffer:
    """Thread-safe rolling activity for the UI: per-neuron EMA rate and recent spike raster."""

    def __init__(self, n: int, raster_steps: int = 400, rate_tau_steps: float = 40.0):
        self.n = n
        self._lock = threading.Lock()
        self._rate = np.zeros(n, dtype=np.float32)       # spikes/step, EMA
        self._decay = np.float32(np.exp(-1.0 / rate_tau_steps))
        self._raster: deque[np.ndarray] = deque(maxlen=raster_steps)
        self._last_spike = np.full(n, -(1 << 30), dtype=np.int32)   # step index of each neuron's latest spike
        self.active_count = 0
        self.steps = 0

    def push(self, spikes: np.ndarray) -> None:
        idx = np.flatnonzero(spikes)
        with self._lock:
            self._rate *= self._decay
            self._rate[idx] += 1 - self._decay
            self._raster.append(idx.astype(np.int32))
            self._last_spike[idx] = self.steps
            self.active_count = len(idx)
            self.steps += 1

    def active_within(self, steps: int) -> int:
        """Distinct neurons that spiked in the last `steps` steps. Per-step counts (~2.5% at the 5 Hz target) hide
        how many take part: with game footage ~128k of 166.7k neurons fire within any 1 s."""
        return int(np.count_nonzero(self.active_mask(steps)))

    def active_mask(self, steps: int) -> np.ndarray:
        with self._lock:
            return self._last_spike > self.steps - 1 - steps

    def rates(self) -> np.ndarray:
        with self._lock:
            return self._rate.copy()

    def raster(self) -> list[np.ndarray]:
        with self._lock:
            return list(self._raster)


class LIFSim:
    def __init__(self, graph: Graph | None, params: LIFParams | None = None, seed: int = 0,
                 W_in: "sp.csr_array | None" = None):
        """Build W_in from the graph, or pass a prebuilt rate-normalized W_in [post, pre] (graph may then be None)."""
        self.p = params or LIFParams()
        t = time.perf_counter()
        if W_in is None:
            signed = graph.adjacency                        # [pre, post]
            counts_in = np.asarray(graph.weights.sum(axis=0)).ravel().astype(np.float32)  # per post
            inv = np.where(counts_in > 0, 1.0 / np.maximum(counts_in, 1), 0).astype(np.float32)
            W_in = signed.T.tocsr()                         # [post, pre]
            W_in = sp.diags_array(inv) @ W_in
            W_in.eliminate_zeros()                          # zero-sign contacts only matter for the normalization
        self.n = W_in.shape[0]
        self.W_csr = W_in.astype(np.float32).tocsr()
        self.W_csc = self.W_csr.tocsc()
        self.build_s = time.perf_counter() - t

        self.dtype = np.float64 if getattr(self.p, "dtype", "float32") == "float64" else np.float32
        self.v = np.zeros(self.n, dtype=self.dtype)
        self.refr = np.zeros(self.n, dtype=np.int16)
        self.spikes = np.zeros(self.n, dtype=bool)
        self._sfloat = np.zeros(self.n, dtype=self.dtype)
        self._drive = np.zeros(self.n, dtype=self.dtype)
        self._mask = np.zeros(self.n, dtype=bool)
        self._zeros = np.zeros(self.n, dtype=self.dtype)
        self.gain = float(self.p.syn_gain)
        self.rng = np.random.default_rng(seed)
        # Pre-scaled noise bank read at a random offset each step: ~0.85 ms saved vs drawing 166k normals per step.
        self._noise = (self.rng.standard_normal(self.n * 16, dtype=self.dtype) * self.dtype(self.p.noise_std))
        self.leak = self.dtype(self.p.dt_ms / self.p.tau_ms)
        self.target_p = self.p.target_rate_hz * self.p.dt_ms / 1000.0
        self.activity = ActivityBuffer(self.n)
        self.last_step_ms = 0.0
        self.path_counts = {"columns": 0, "full": 0}
        self.spike_total = 0
        self.individuality = getattr(self.p, "individuality", "off")
        self.individuality_sigma = getattr(self.p, "individuality_sigma", None)
        from kickthefly.core import individuality as indiv_mod
        iseed = getattr(self.p, "individuality_seed", None)
        self.d_pre, self.d_post = indiv_mod.compute_fly_gains(
            seed if iseed is None else int(iseed), self.n, self.individuality, self.individuality_sigma
        )
        from kickthefly.sim.connectome import backends
        self.backend_choice = getattr(self.p, "backend", "auto")
        self.backend = backends.create_backend(self, self.backend_choice)
        if self.d_pre is not None and self.backend.name == "gl":
            # the gl shaders propagate with the shared W only; say so rather than report gains that are not applied
            import logging
            logging.getLogger("kickthefly").warning(
                "individuality '%s' is not implemented on the gl backend; this fly runs the shared connectome",
                self.individuality)
            self.d_pre = self.d_post = None

    def _propagate(self) -> np.ndarray:
        active = np.flatnonzero(self.spikes)
        k = len(active)
        if k == 0:
            return self._zeros
        if self.d_pre is not None:
            if k <= self.p.sparse_path_max_active * self.n:
                self.path_counts["columns"] += 1
                sub = self.W_csc[:, active]
                out = sub @ (self.d_pre[active].astype(self.dtype))
            else:
                self.path_counts["full"] += 1
                self._sfloat[:] = self.spikes * self.d_pre
                out = self.W_csr @ self._sfloat
            if self.d_post is not None:
                out = out * self.d_post
            return out
        if k <= self.p.sparse_path_max_active * self.n:
            self.path_counts["columns"] += 1
            sub = self.W_csc[:, active]
            return sub @ np.ones(k, dtype=np.float32)
        self.path_counts["full"] += 1
        self._sfloat[:] = self.spikes
        return self.W_csr @ self._sfloat

    def step(self, sensory_input: np.ndarray | None = None) -> np.ndarray:
        """Advance one dt. sensory_input: dense float32 current per neuron, or None. Returns bool spike vector."""
        t0 = time.perf_counter()
        p = self.p
        spikes = self.backend.step(sensory_input)

        fired = int(np.count_nonzero(spikes))
        self.spike_total += fired                    # for the benchmark's synaptic events/s
        frac = fired / self.n
        err = (self.target_p - frac) / max(self.target_p, 1e-9)
        self.gain = float(np.clip(self.gain * np.exp(p.gain_adapt * np.clip(err, -1, 1)), *p.gain_bounds))

        self.activity.push(spikes)
        self.last_step_ms = (time.perf_counter() - t0) * 1000
        return spikes


def _bench(steps: int) -> None:
    from kickthefly.sim.connectome.loader import load_graph
    from kickthefly.sim.connectome.retina import CH_LUM, RetinaEncoder, RetinaMap

    t = time.perf_counter()
    g = load_graph()
    print(f"[sim] graph loaded {time.perf_counter() - t:.1f}s  neurons {g.n:,}  edges {g.adjacency.nnz:,}")
    sim = LIFSim(g)
    print(f"[sim] W_in built in {sim.build_s:.1f}s  nnz after dropping zero-sign {sim.W_csr.nnz:,}")
    rm = RetinaMap.load()
    enc = RetinaEncoder(rm, g.n)

    # raw matmul costs
    s = np.zeros(g.n, dtype=np.float32)
    s[np.random.default_rng(1).choice(g.n, int(0.025 * g.n), replace=False)] = 1
    for label, fn in (("full CSR matvec", lambda: sim.W_csr @ s),
                      ("CSC active-column matmul @2.5% active", lambda: sim.W_csc[:, np.flatnonzero(s)] @ np.ones(int(s.sum()), np.float32))):
        ts = []
        for _ in range(30):
            t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1000)
        print(f"[sim] {label:38s} median {np.median(ts):6.2f} ms")

    pops = {
        "photoreceptors R1-R6": rm.rows[rm.channel == CH_LUM],
        "lamina L1-L3": np.flatnonzero(np.isin(g.type.astype(str), ["L1", "L2", "L3"])),
        "Mi1/Tm1/Tm2/Tm9": np.flatnonzero(np.isin(g.type.astype(str), ["Mi1", "Tm1", "Tm2", "Tm9"])),
        "T4/T5": np.flatnonzero(np.char.startswith(g.type.astype(str), "T4") | np.char.startswith(g.type.astype(str), "T5")),
        "LC/LPLC": np.flatnonzero(np.char.startswith(g.type.astype(str), "LC") | np.char.startswith(g.type.astype(str), "LPLC")),
        "descending neurons": np.flatnonzero(g.superclass == "descending_neuron"),
        "all": np.arange(g.n),
    }
    dark = np.zeros((360, 640, 3), np.uint8)
    rng = np.random.default_rng(2)

    def run(label, frame_fn, n):
        times, frac = [], []
        pop_spikes = {k: 0 for k in pops}
        for i in range(n):
            cur = enc.encode(frame_fn(i), [], (0, 0, 640, 360))
            spk = sim.step(cur)
            times.append(sim.last_step_ms)
            frac.append(spk.mean())
            for k, rows in pops.items():
                pop_spikes[k] += int(spk[rows].sum())
        t = np.array(times[10:])
        hz = {k: pop_spikes[k] / (len(rows) * n * sim.p.dt_ms / 1000) for k, rows in pops.items()}
        print(f"[sim] {label:26s} step ms: mean {t.mean():5.2f}  p50 {np.percentile(t, 50):5.2f}  p95 {np.percentile(t, 95):5.2f}"
              f"  | active/step {np.mean(frac) * 100:4.2f}%  gain {sim.gain:5.2f}")
        print("      rates (Hz): " + "  ".join(f"{k} {v:6.2f}" for k, v in hz.items()))
        return hz

    run("warm-up (dark)", lambda i: dark, steps // 2)
    base = run("dark", lambda i: dark, steps)

    def moving_box(i):
        f = dark.copy()
        x = int((i * 7) % 560)
        f[120:240, x:x + 80] = 255
        return f

    lit = run("moving bright box", moving_box, steps)
    noise = run("full-field flicker", lambda i: rng.integers(0, 256, (360, 640, 3), dtype=np.uint8), steps)
    print("[sim] response vs dark (Hz delta):")
    for k in pops:
        print(f"      {k:22s} box {lit[k] - base[k]:+6.2f}   flicker {noise[k] - base[k]:+6.2f}")
    print(f"[sim] propagation path usage: {sim.path_counts}")


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="python -m kickthefly.sim.connectome.sim")
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench", help="time sim.step and check that visual input propagates")
    b.add_argument("--steps", type=int, default=600)
    args = parser.parse_args(argv)
    if args.cmd == "bench":
        _bench(args.steps)
    return 0


if __name__ == "__main__":
    from kickthefly.sim.connectome.sim import main as _main
    raise SystemExit(_main())
