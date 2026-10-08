"""Profiling script for Kick the Fly simulation loop.

Measures:
- Breakdown of time inside LIFSim.step, Brain._step, and rendering:
  - sparse matmul
  - synapse gather / CSC column slicing
  - state update (leak, drive, noise, refractory, threshold/reset)
  - plasticity (Memory.step)
  - readout (spike gathering, binning, EMA rates, group rates)
  - rendering (BrainView)
- Sim-time vs real-time for 1, 8, 16 flies
- Neurons/second and synapse-evaluations/second
- Current data layout (format, dtype, sharing across flies)
"""
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import pygame

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))     # run as `python tools/profile_sim.py`

from kickthefly.core import simcore  # noqa: E402
from kickthefly.game import kick_the_fly as k  # noqa: E402
from kickthefly.sim.connectome.sim import LIFParams, LIFSim  # noqa: E402
from kickthefly.core import memory as mem_mod  # noqa: E402


def inspect_data_layout():
    g, W, soma = simcore.pack()
    print("=== DATA LAYOUT ===")
    print(f"Total neurons: {g.n:,}")
    print(f"W matrix type: {type(W)} (format: {W.format if hasattr(W, 'format') else 'unknown'})")
    print(f"W data dtype: {W.data.dtype}, shape: {W.shape}, nnz: {W.nnz:,}")
    print(f"W indptr dtype: {W.indptr.dtype}, indices dtype: {W.indices.dtype}")
    print(f"Soma array shape: {soma.shape}, dtype: {soma.dtype}")
    
    # Test Brain instantiation
    b1 = simcore.new_brain(seed=1, warmup=0)
    b2 = simcore.new_brain(seed=2, warmup=0)
    w_shared_csr = (b1.sim.W_csr is b2.sim.W_csr) or (b1.sim.W_csr.data.base is b2.sim.W_csr.data.base)
    w_shared_pack = (b1.sim.W_csr is W)
    print(f"W_csr in LIFSim: format={b1.sim.W_csr.format}, dtype={b1.sim.W_csr.data.dtype}, nnz={b1.sim.W_csr.nnz:,}")
    print(f"W_csc in LIFSim: format={b1.sim.W_csc.format}, dtype={b1.sim.W_csc.data.dtype}, nnz={b1.sim.W_csc.nnz:,}")
    print(f"Is W matrix shared between separate flies in memory: {w_shared_csr}")
    print(f"Is W_csr identical object to brainpack W: {w_shared_pack}")
    print(f"Does Brain allocate independent voltage/state vectors: {b1.sim.v is not b2.sim.v}")
    print()


def profile_sim_components(steps=200):
    print(f"=== PROFILING SIM COMPONENTS ({steps} steps) ===")
    brain = simcore.new_brain(seed=42, warmup=50)
    sim = brain.sim
    mem = brain.memory
    
    t_propagate_matmul = 0.0
    t_propagate_gather = 0.0
    t_state_update = 0.0
    t_activity_push = 0.0
    t_readout_brain = 0.0
    t_plasticity = 0.0
    
    total_spikes = 0
    
    for s in range(steps):
        spk = sim.spikes
        active = np.flatnonzero(spk)
        k_active = len(active)
        total_spikes += k_active
        
        # 1. Sparse propagation
        if k_active <= sim.p.sparse_path_max_active * sim.n:
            t_sub0 = time.perf_counter()
            sub = sim.W_csc[:, active]
            t_sub1 = time.perf_counter()
            i_syn = sub @ np.ones(k_active, dtype=np.float32)
            t_sub2 = time.perf_counter()
            t_propagate_gather += (t_sub1 - t_sub0)
            t_propagate_matmul += (t_sub2 - t_sub1)
        else:
            t_full0 = time.perf_counter()
            sim._sfloat[:] = spk
            i_syn = sim.W_csr @ sim._sfloat
            t_full1 = time.perf_counter()
            t_propagate_matmul += (t_full1 - t_full0)
            
        # 2. State update
        t_state0 = time.perf_counter()
        p = sim.p
        drive = sim._drive
        np.multiply(i_syn, np.float32(sim.gain), out=drive)
        drive += np.float32(p.bias)
        off = int(sim.rng.integers(0, sim._noise.size - sim.n))
        drive += sim._noise[off:off + sim.n]
        v = sim.v
        v *= np.float32(1.0 - sim.leak)
        if p.v_reset:
            v += sim.leak * np.float32(p.v_reset)
        v += drive
        mask = sim._mask
        np.greater(sim.refr, 0, out=mask)
        np.copyto(v, np.float32(p.v_reset), where=mask)
        np.subtract(sim.refr, 1, out=sim.refr, where=mask)
        spikes = v >= p.v_thresh
        np.copyto(v, np.float32(p.v_reset), where=spikes)
        np.copyto(sim.refr, np.int16(p.refractory_steps), where=spikes)
        sim.spikes = spikes
        frac = np.count_nonzero(spikes) / sim.n
        err = (sim.target_p - frac) / max(sim.target_p, 1e-9)
        sim.gain = float(np.clip(sim.gain * np.exp(p.gain_adapt * np.clip(err, -1, 1)), *p.gain_bounds))
        t_state1 = time.perf_counter()
        t_state_update += (t_state1 - t_state0)
        
        # 3. Activity push
        t_act0 = time.perf_counter()
        sim.activity.push(spikes)
        t_act1 = time.perf_counter()
        t_activity_push += (t_act1 - t_act0)
        
        # 4. Brain readout
        t_rd0 = time.perf_counter()
        on = np.flatnonzero(spikes)
        G = len(brain.names)
        counts = np.zeros(G)
        d = brain.det_id[on]
        counts[:brain.n_det] = np.bincount(d[d >= 0], minlength=brain.n_det)
        p_ids = brain.pop_id[on]
        counts[brain.n_det:G - 1] = np.bincount(p_ids[p_ids >= 0], minlength=len(k.POPS))
        counts[-1] = len(on)
        inst = counts / brain.g_size / brain.dt
        brain.fast += (inst - brain.fast) * brain.k_fast
        brain.steps += 1
        t_rd1 = time.perf_counter()
        t_readout_brain += (t_rd1 - t_rd0)
        
        # 5. Plasticity (runs every 10 steps)
        if mem is not None and brain.steps % k.MEMORY_STEPS == 0:
            t_mem0 = time.perf_counter()
            calm = True
            mem.step(sim.activity.rates(), calm, brain.steps)
            t_mem1 = time.perf_counter()
            t_plasticity += (t_mem1 - t_mem0)
            
    total_time = (t_propagate_gather + t_propagate_matmul + t_state_update +
                  t_activity_push + t_readout_brain + t_plasticity)
    print(f"Total measured sim step time: {total_time * 1000:.1f} ms for {steps} steps ({total_time / steps * 1000:.2f} ms/step)")
    print(f"  - CSC column gather / slice:   {t_propagate_gather * 1000:6.1f} ms ({t_propagate_gather / total_time * 100:4.1f}%)")
    print(f"  - Sparse matmul / matvec:       {t_propagate_matmul * 1000:6.1f} ms ({t_propagate_matmul / total_time * 100:4.1f}%)")
    print(f"  - LIF state & noise update:     {t_state_update * 1000:6.1f} ms ({t_state_update / total_time * 100:4.1f}%)")
    print(f"  - Activity buffer push:         {t_activity_push * 1000:6.1f} ms ({t_activity_push / total_time * 100:4.1f}%)")
    print(f"  - Brain readout & group rates:  {t_readout_brain * 1000:6.1f} ms ({t_readout_brain / total_time * 100:4.1f}%)")
    print(f"  - Plasticity (KC->MBON update): {t_plasticity * 1000:6.1f} ms ({t_plasticity / total_time * 100:4.1f}%)")
    print(f"Avg active spikes/step: {total_spikes / steps:.0f} ({total_spikes / steps / sim.n * 100:.2f}%)")
    
    # Measure BrainView rendering
    g, W, soma = simcore.pack()
    view = k.BrainView(soma, W, np.zeros(g.n, bool), seed=0)
    t_render0 = time.perf_counter()
    n_frames = 60
    for _ in range(n_frames):
        view.render("panel", sim.activity.rates(), np.flatnonzero(sim.spikes), 0.0, False)
    t_render_panel = (time.perf_counter() - t_render0) / n_frames * 1000.0
    
    t_render0 = time.perf_counter()
    for _ in range(n_frames):
        view.render("big", sim.activity.rates(), np.flatnonzero(sim.spikes), 0.0, False)
    t_render_big = (time.perf_counter() - t_render0) / n_frames * 1000.0
    print(f"Rendering (BrainView): panel = {t_render_panel:.2f} ms/frame, big = {t_render_big:.2f} ms/frame")
    print()


def benchmark_fly_counts(fly_counts=(1, 8, 16), steps_per_fly=100):
    print("=== BENCHMARK MULTI-FLY SIM-TIME VS REAL-TIME ===")
    g, W, _ = simcore.pack()
    nnz = W.nnz
    n_neurons = g.n
    dt_s = 0.005 # 5 ms sim time per step
    
    for n_flies in fly_counts:
        brains = [simcore.new_brain(seed=i, warmup=20) for i in range(n_flies)]
        t0 = time.perf_counter()
        for step_idx in range(steps_per_fly):
            for b in brains:
                b._step()
        wall_s = time.perf_counter() - t0
        
        sim_time_s = steps_per_fly * dt_s
        speed_ratio = sim_time_s / wall_s
        total_neuron_steps = n_flies * n_neurons * steps_per_fly
        neurons_per_sec = total_neuron_steps / wall_s
        synapses_per_sec = (n_flies * nnz * steps_per_fly) / wall_s
        
        print(f"{n_flies:2d} flies: wall-time {wall_s:6.2f}s for {sim_time_s:.2f}s sim-time -> "
              f"speed: {speed_ratio:5.2f}x real-time ({'FASTER' if speed_ratio >= 1.0 else 'SLOWER'} than real-time) | "
              f"{neurons_per_sec / 1e6:6.2f} M neurons/s | "
              f"{synapses_per_sec / 1e9:6.2f} G syn-evals/s")
    print()


if __name__ == "__main__":
    inspect_data_layout()
    profile_sim_components(steps=200)
    benchmark_fly_counts((1, 8, 16), steps_per_fly=100)
