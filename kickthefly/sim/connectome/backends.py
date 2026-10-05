"""Pluggable simulation backends for Kick the Fly connectome simulation.

Provides backend interface and implementations:
- CPUBackend: Reference NumPy CPU implementation using CSC column gathering / CSR matvec
- NumbaBackend: JIT-compiled hot loops for CPU execution
- TorchBackend: PyTorch device-resident sparse ops for CUDA / ROCm / CPU
- Backend discovery, selection, and probing with automatic CPU fallback
"""
from __future__ import annotations

import logging
import os
import threading
import time
import warnings
import weakref
from collections import deque
from concurrent.futures import Future
from typing import Any
import numpy as np
import scipy.sparse as sp

log = logging.getLogger("kickthefly")

BACKEND_NAMES = ("auto", "cpu", "numba", "torch-cpu", "torch-cuda", "torch-rocm", "gl")


class SimBackend:
    """Interface for pluggable LIF connectome simulator execution."""

    name: str = "base"
    device_name: str = "cpu"

    @property
    def device(self) -> str:
        """Human-readable device, for the Lab header, benchmarks, save states, crash reports and NWB metadata."""
        return self.device_name

    def __init__(self, sim: Any) -> None:
        self.sim = sim
        self.n = sim.n

    def setup(self) -> None:
        """Initialize backend-specific data structures or device allocations."""
        pass

    def on_weights_changed(self, positions: np.ndarray | None = None) -> None:
        """Notify backend that W_csr/W_csc matrix weights have been modified. positions: the W_csr.data indices that
        changed, sorted (the mushroom body's learning passes these), or None when anything may have changed."""
        pass

    def sync_to_host(self) -> None:
        """Sync device-resident state (v, refr, spikes) back to host numpy arrays."""
        pass

    def sync_from_host(self) -> None:
        """Sync host numpy arrays (v, refr, spikes) up to device tensors."""
        pass

    def step(self, sensory_input: np.ndarray | None = None) -> np.ndarray:
        """Advance one LIF step. Returns boolean spike vector of length n."""
        raise NotImplementedError


class CPUBackend(SimBackend):
    """Reference CPU / NumPy implementation (preserves exact single-vector math)."""

    name = "cpu"
    device_name = "CPU (NumPy)"

    def step(self, sensory_input: np.ndarray | None = None) -> np.ndarray:
        sim = self.sim
        p = sim.p
        dt = sim.dtype
        i_syn = sim._propagate()
        drive = sim._drive
        np.multiply(i_syn, dt(sim.gain), out=drive)
        drive += dt(p.bias)
        off = int(sim.rng.integers(0, sim._noise.size - self.n))
        drive += sim._noise[off:off + self.n]
        if sensory_input is not None:
            if p.ext_gain == 1.0:
                drive += sensory_input
            else:
                drive += dt(p.ext_gain) * sensory_input

        v = sim.v
        v *= dt(1.0 - sim.leak)
        if p.v_reset:
            v += sim.leak * dt(p.v_reset)
        v += drive
        mask = sim._mask
        np.greater(sim.refr, 0, out=mask)
        np.copyto(v, dt(p.v_reset), where=mask)
        np.subtract(sim.refr, 1, out=sim.refr, where=mask)
        spikes = v >= p.v_thresh
        np.copyto(v, dt(p.v_reset), where=spikes)
        np.copyto(sim.refr, np.int16(p.refractory_steps), where=spikes)
        sim.spikes = spikes
        return spikes


# --- Numba Backend -------------------------------------------------------------
_numba_available = False
_jit_step_core = None
_jit_csc_propagate = None

try:
    import numba

    # nogil: each fly's brain thread runs its kernels in parallel with the others. No fastmath, and every scalar arrives
    # already cast to the state dtype: the kernels must round exactly like the NumPy reference (same operations, same
    # order, same precision), or a chaotic network drifts apart in ~300 steps.
    @numba.njit(cache=True, nogil=True)
    def _jit_csc_propagate(indices, indptr, data, active_cols, n_out):
        out = np.zeros(n_out, dtype=data.dtype)
        for c in active_cols:                     # scipy's CSC matvec order: column by column, rows within a column
            for idx in range(indptr[c], indptr[c + 1]):
                out[indices[idx]] += data[idx]
        return out

    @numba.njit(cache=True, nogil=True)
    def _jit_csc_propagate_indiv(indices, indptr, data, active_cols, n_out, d_pre, d_post):
        out = np.zeros(n_out, dtype=data.dtype)
        for c in active_cols:
            scale = d_pre[c]
            for idx in range(indptr[c], indptr[c + 1]):
                out[indices[idx]] += data[idx] * scale
        for i in range(n_out):
            out[i] *= d_post[i]
        return out

    @numba.njit(cache=True, nogil=True)
    def _jit_step_core(v, refr, i_syn, noise_slice, sensory, has_sensory, gain, bias, ext_gain, scale_sensory,
                       keep, leak_reset, has_reset, v_reset, v_thresh, refractory_steps):
        n = len(v)
        spikes = np.zeros(n, dtype=np.bool_)
        for i in range(n):
            drive = i_syn[i] * gain
            drive += bias
            drive += noise_slice[i]
            if has_sensory:
                if scale_sensory:
                    drive += ext_gain * sensory[i]
                else:
                    drive += sensory[i]
            v_val = v[i] * keep
            if has_reset:
                v_val += leak_reset
            v_val += drive
            if refr[i] > 0:
                v_val = v_reset
                refr[i] -= 1
            if v_val >= v_thresh:
                spikes[i] = True
                v[i] = v_reset
                refr[i] = refractory_steps
            else:
                v[i] = v_val
        return spikes

    _numba_available = True
except ImportError:
    pass


class NumbaBackend(SimBackend):
    """JIT-compiled CPU backend. Bit-exact with CPUBackend: same float operations in the same order."""

    name = "numba"
    device_name = "CPU (Numba JIT)"

    def setup(self) -> None:
        if not _numba_available:
            raise RuntimeError("Numba is not installed")

    def step(self, sensory_input: np.ndarray | None = None) -> np.ndarray:
        sim = self.sim
        p = sim.p
        dt = sim.dtype
        active = np.flatnonzero(sim.spikes)
        k_active = len(active)
        if k_active == 0:
            i_syn = sim._zeros
        elif sim.d_pre is not None:
            if k_active <= p.sparse_path_max_active * self.n:
                sim.path_counts["columns"] += 1
                csc = sim.W_csc
                i_syn = _jit_csc_propagate_indiv(csc.indices, csc.indptr, csc.data, active.astype(np.int32), self.n, sim.d_pre, sim.d_post)
            else:
                sim.path_counts["full"] += 1
                sim._sfloat[:] = sim.spikes * sim.d_pre
                i_syn = (sim.W_csr @ sim._sfloat) * sim.d_post
        elif k_active <= p.sparse_path_max_active * self.n:
            sim.path_counts["columns"] += 1
            csc = sim.W_csc
            i_syn = _jit_csc_propagate(csc.indices, csc.indptr, csc.data, active, self.n)
        else:
            sim.path_counts["full"] += 1
            sim._sfloat[:] = sim.spikes
            i_syn = sim.W_csr @ sim._sfloat

        off = int(sim.rng.integers(0, sim._noise.size - self.n))
        noise_slice = sim._noise[off:off + self.n]
        has_sensory = sensory_input is not None
        sensory = sensory_input if has_sensory else sim._zeros
        spikes = _jit_step_core(
            sim.v, sim.refr, i_syn, noise_slice, sensory, has_sensory,
            dt(sim.gain), dt(p.bias), dt(p.ext_gain), p.ext_gain != 1.0,
            dt(1.0 - sim.leak), sim.leak * dt(p.v_reset), bool(p.v_reset), dt(p.v_reset), dt(p.v_thresh),
            np.int16(p.refractory_steps))
        sim.spikes = spikes
        return spikes


# --- PyTorch Backend (CUDA, ROCm, CPU) ------------------------------------------
_torch_available = False
try:
    import torch
    _torch_available = True
except ImportError:
    torch = None


def _torch_gpu_kind() -> str | None:
    """'torch-rocm' or 'torch-cuda' for the GPU this torch build can use, or None."""
    if not _torch_available or not torch.cuda.is_available():
        return None
    return "torch-rocm" if getattr(torch.version, "hip", None) else "torch-cuda"


_compiled_lif_step = None


def _get_compiled_lif_step():
    global _compiled_lif_step
    if _compiled_lif_step is None and _torch_available and torch is not None:
        def _core(v, refr, i_syn, gain, bias, noise_slice, sens, ext_gain, leak_decay, leak_reset, v_reset, v_thresh, refr_steps):
            drive = i_syn * gain + bias + noise_slice
            if sens is not None:
                drive = drive + (sens if ext_gain == 1.0 else sens * ext_gain)
            v_new = v * leak_decay + leak_reset + drive
            refr_mask = refr > 0
            v_new = torch.where(refr_mask, v_reset, v_new)
            refr_new = torch.where(refr_mask, refr - 1, refr)
            spikes = v_new >= v_thresh
            v_out = torch.where(spikes, v_reset, v_new)
            refr_out = torch.where(spikes, refr_steps, refr_new)
            return v_out, refr_out, spikes

        try:
            _compiled_lif_step = torch.compile(_core)
        except Exception as e:
            log.warning("torch.compile unavailable for fused LIF: %s", e)
            _compiled_lif_step = _core
    return _compiled_lif_step



class TorchBackend(SimBackend):
    """PyTorch backend: state tensors (v, refr, spikes, noise bank) and weight matrix live on device
    (CUDA, ROCm, or CPU) across steps for zero-copy simulation. Host arrays (sim.v, sim.refr, sim.spikes)
    are synchronized on demand via sync_to_host() / sync_from_host()."""

    name = "torch"

    def __init__(self, sim: Any, device: str = "cpu") -> None:
        super().__init__(sim)
        self.device_str = device
        self.tdev = None
        self.tdt = None
        self.W_torch = None
        self._W64 = None
        self.v_dev = None
        self.refr_dev = None
        self.spikes_dev = None
        self.s_float_dev = None
        self.noise_dev = None
        self._noise_id = None
        # Hoisted parameter tensors
        self._bias_tensor = None
        self._leak_decay_tensor = None
        self._leak_reset_tensor = None
        self._v_reset_tensor = None
        self._v_thresh_tensor = None
        self._refr_steps_tensor = None
        self._ext_gain_tensor = None

    def setup(self) -> None:
        if not _torch_available:
            raise RuntimeError("PyTorch is not installed")
        self.tdev = torch.device(self.device_str)
        if self.tdev.type == "cuda":
            kind = _torch_gpu_kind()
            if kind is None:
                raise RuntimeError("a GPU was requested but torch.cuda.is_available() is False")
            self.name = kind
            self.device_name = f"{'ROCm' if kind == 'torch-rocm' else 'CUDA'}: {torch.cuda.get_device_name(self.tdev)}"
        else:
            self.name = f"torch-{self.tdev.type}"
            self.device_name = f"PyTorch ({self.tdev.type})"
        dt = self.sim.dtype
        self.tdt = torch.float64 if dt is np.float64 else torch.float32
        self._upload_weights()
        self._init_params()
        self.sync_from_host()

    def _init_params(self) -> None:
        sim = self.sim
        p = sim.p
        dt = sim.dtype
        dev = self.tdev
        tdt = self.tdt
        self._bias_tensor = torch.tensor(dt(p.bias), dtype=tdt, device=dev)
        self._leak_decay_tensor = torch.tensor(dt(1.0 - sim.leak), dtype=tdt, device=dev)
        self._leak_reset_tensor = torch.tensor(sim.leak * dt(p.v_reset), dtype=tdt, device=dev)
        self._v_reset_tensor = torch.tensor(dt(p.v_reset), dtype=tdt, device=dev)
        self._v_thresh_tensor = torch.tensor(dt(p.v_thresh), dtype=tdt, device=dev)
        self._refr_steps_tensor = torch.tensor(int(p.refractory_steps), dtype=torch.int16, device=dev)
        self._ext_gain_tensor = torch.tensor(dt(p.ext_gain), dtype=tdt, device=dev)

    def _upload_weights(self) -> None:
        csr = self.sim.W_csr
        with warnings.catch_warnings():             # "sparse CSR support is in beta", on every learning step otherwise
            warnings.simplefilter("ignore", UserWarning)
            self.W_torch = self._csr_tensor(csr)
        self._W64 = None                            # rebuilt from W_torch on the next busy float64 step

    def _csr_tensor(self, csr):
        return torch.sparse_csr_tensor(
            torch.from_numpy(csr.indptr.astype(np.int64)), torch.from_numpy(csr.indices.astype(np.int64)),
            torch.from_numpy(np.ascontiguousarray(csr.data, dtype=np.float32)), size=(self.n, self.n),
            device=self.tdev, check_invariants=False)

    def on_weights_changed(self, positions: np.ndarray | None = None) -> None:
        if self.W_torch is not None:
            self._upload_weights()

    def sync_to_host(self) -> None:
        if self.v_dev is not None:
            self.sim.v[:] = self.v_dev.cpu().numpy()
            self.sim.refr[:] = self.refr_dev.cpu().numpy()
            self.sim.spikes[:] = self.spikes_dev.cpu().numpy()

    def sync_from_host(self) -> None:
        if self.tdev is not None:
            sim = self.sim
            dev = self.tdev
            tdt = self.tdt or (torch.float64 if sim.dtype is np.float64 else torch.float32)
            self.v_dev = torch.from_numpy(sim.v).to(dev, dtype=tdt)
            self.refr_dev = torch.from_numpy(sim.refr).to(dev, dtype=torch.int16)
            self.spikes_dev = torch.from_numpy(sim.spikes).to(dev, dtype=torch.bool)
            self.s_float_dev = self.spikes_dev.to(self.W_torch.dtype if self.W_torch is not None else torch.float32).unsqueeze(1)
            self.noise_dev = torch.from_numpy(sim._noise).to(dev, dtype=tdt)
            self._noise_id = id(sim._noise)
            if getattr(sim, "d_pre", None) is not None:
                self.d_pre_dev = torch.from_numpy(sim.d_pre).to(dev, dtype=self.W_torch.dtype if self.W_torch is not None else tdt)
                self.d_post_dev = torch.from_numpy(sim.d_post).to(dev, dtype=tdt)
            else:
                self.d_pre_dev = None
                self.d_post_dev = None

    def step(self, sensory_input: np.ndarray | None = None) -> np.ndarray:
        sim = self.sim
        p = sim.p
        dt = sim.dtype
        tdt = self.tdt
        dev = self.tdev
        if self.W_torch is None or self.W_torch.values().numel() != sim.W_csr.nnz:
            self._upload_weights()
        if self.v_dev is None:
            self.sync_from_host()
        if self._noise_id != id(sim._noise):
            self.noise_dev = torch.from_numpy(sim._noise).to(dev, dtype=tdt)
            self._noise_id = id(sim._noise)

        dense64 = (tdt == torch.float64) and (self.spikes_dev.sum().item() > p.sparse_path_max_active * self.n)
        if dense64 and getattr(self, "_W64", None) is None:
            self._W64 = self.W_torch.to(torch.float64)
        W = self._W64 if dense64 else self.W_torch

        if getattr(self, "d_pre_dev", None) is not None:
            s_float = (self.spikes_dev.to(W.dtype) * self.d_pre_dev).unsqueeze(1)
            i_syn = torch.sparse.mm(W, s_float).squeeze(1)
            if getattr(self, "d_post_dev", None) is not None:
                i_syn = i_syn * self.d_post_dev
        else:
            s_float = self.spikes_dev.to(W.dtype).unsqueeze(1)
            i_syn = torch.sparse.mm(W, s_float).squeeze(1)
        off = int(sim.rng.integers(0, sim._noise.size - self.n))

        if getattr(p, "fuse_lif", False):
            fused_fn = _get_compiled_lif_step()
            sens = None
            if sensory_input is not None:
                sens = torch.from_numpy(np.ascontiguousarray(sensory_input)).to(dev, dtype=tdt)
            v, refr, spikes = fused_fn(
                self.v_dev, self.refr_dev, i_syn.to(tdt), sim.gain, self._bias_tensor,
                self.noise_dev[off:off + self.n], sens, p.ext_gain, self._leak_decay_tensor,
                self._leak_reset_tensor if p.v_reset else torch.zeros(1, dtype=tdt, device=dev),
                self._v_reset_tensor, self._v_thresh_tensor, self._refr_steps_tensor
            )
        else:
            drive = i_syn.to(tdt) * sim.gain
            drive += self._bias_tensor
            drive += self.noise_dev[off:off + self.n]

            if sensory_input is not None:
                sens = torch.from_numpy(np.ascontiguousarray(sensory_input)).to(dev, dtype=tdt)
                if p.ext_gain == 1.0:
                    drive += sens
                else:
                    drive += self._ext_gain_tensor * sens

            v = self.v_dev * self._leak_decay_tensor
            if p.v_reset:
                v += self._leak_reset_tensor
            v += drive

            refr_mask = self.refr_dev > 0
            v = torch.where(refr_mask, self._v_reset_tensor, v)
            refr = torch.where(refr_mask, self.refr_dev - 1, self.refr_dev)

            spikes = v >= self._v_thresh_tensor
            v = torch.where(spikes, self._v_reset_tensor, v)
            refr = torch.where(spikes, self._refr_steps_tensor, refr)

        self.v_dev = v
        self.refr_dev = refr
        self.spikes_dev = spikes

        spikes_np = spikes.cpu().numpy()
        sim.spikes = spikes_np
        return spikes_np

    @classmethod
    def step_batch(cls, sims: list[Any], sensory_inputs: list[np.ndarray | None] | None = None) -> list[np.ndarray]:
        """Advance a batch of LIF simulations on GPU with a single batched SpMM multiplication."""
        if not sims:
            return []
        if len(sims) == 1:
            sens = sensory_inputs[0] if sensory_inputs else None
            return [sims[0].step(sens)]

        backends = [s.backend for s in sims]
        for b in backends:
            if b.v_dev is None:
                b.sync_from_host()
            if b._noise_id != id(b.sim._noise):
                b.noise_dev = torch.from_numpy(b.sim._noise).to(b.tdev, dtype=b.tdt)
                b._noise_id = id(b.sim._noise)

        # Group simulations by weight tensor for batched SpMM
        groups: dict[Any, list[int]] = {}
        for idx, b in enumerate(backends):
            w = b.W_torch
            groups.setdefault(w, []).append(idx)

        i_syn_list = [None] * len(sims)
        for w, indices in groups.items():
            if len(indices) == 1:
                idx = indices[0]
                b = backends[idx]
                if getattr(b, "d_pre_dev", None) is not None:
                    s_float = (b.spikes_dev.to(w.dtype) * b.d_pre_dev).unsqueeze(1)
                    i_syn = torch.sparse.mm(w, s_float).squeeze(1)
                    if getattr(b, "d_post_dev", None) is not None:
                        i_syn = i_syn * b.d_post_dev
                    i_syn_list[idx] = i_syn
                else:
                    s_float = b.spikes_dev.to(w.dtype).unsqueeze(1)
                    i_syn_list[idx] = torch.sparse.mm(w, s_float).squeeze(1)
            else:
                stacked = []
                for idx in indices:
                    b = backends[idx]
                    spk = b.spikes_dev.to(w.dtype)
                    if getattr(b, "d_pre_dev", None) is not None:
                        spk = spk * b.d_pre_dev
                    stacked.append(spk)
                stacked_spikes = torch.stack(stacked, dim=1)
                batched_out = torch.sparse.mm(w, stacked_spikes)
                for col, idx in enumerate(indices):
                    b = backends[idx]
                    col_out = batched_out[:, col]
                    if getattr(b, "d_post_dev", None) is not None:
                        col_out = col_out * b.d_post_dev
                    i_syn_list[idx] = col_out

        results = []
        for idx, (sim, b) in enumerate(zip(sims, backends)):
            p = sim.p
            tdt = b.tdt
            dev = b.tdev
            i_syn = i_syn_list[idx]
            drive = i_syn.to(tdt) * sim.gain
            drive += b._bias_tensor
            off = int(sim.rng.integers(0, sim._noise.size - sim.n))
            drive += b.noise_dev[off:off + sim.n]

            sens = sensory_inputs[idx] if sensory_inputs else None
            if sens is not None:
                sens_t = torch.from_numpy(np.ascontiguousarray(sens)).to(dev, dtype=tdt)
                if p.ext_gain == 1.0:
                    drive += sens_t
                else:
                    drive += b._ext_gain_tensor * sens_t

            v = b.v_dev * b._leak_decay_tensor
            if p.v_reset:
                v += b._leak_reset_tensor
            v += drive

            refr_mask = b.refr_dev > 0
            v = torch.where(refr_mask, b._v_reset_tensor, v)
            refr = torch.where(refr_mask, b.refr_dev - 1, b.refr_dev)

            spikes = v >= b._v_thresh_tensor
            v = torch.where(spikes, b._v_reset_tensor, v)
            refr = torch.where(spikes, b._refr_steps_tensor, refr)

            b.v_dev = v
            b.refr_dev = refr
            b.spikes_dev = spikes

            spikes_np = spikes.cpu().numpy()
            sim.spikes = spikes_np

            fired = int(np.count_nonzero(spikes_np))
            sim.spike_total += fired
            frac = fired / sim.n
            err = (sim.target_p - frac) / max(sim.target_p, 1e-9)
            sim.gain = float(np.clip(sim.gain * np.exp(p.gain_adapt * np.clip(err, -1, 1)), *p.gain_bounds))
            sim.activity.push(spikes_np)
            results.append(spikes_np)

        return results


# --- ModernGL Compute Shader Backend (Vendor-Neutral GPU) --------------------
_moderngl_available = False
try:
    import moderngl
    _moderngl_available = True
except ImportError:
    moderngl = None


class _GLContextBinder:
    """Helper to bind ModernGL standalone contexts across worker threads (EGL, GLX, WGL)."""

    def __init__(self) -> None:
        self._libegl = None
        self._libgl = None
        self._libwgl = None
        import ctypes
        import sys

        try:
            libegl = ctypes.CDLL("libEGL.so.1")
            libegl.eglGetCurrentContext.restype = ctypes.c_void_p
            libegl.eglGetCurrentDisplay.restype = ctypes.c_void_p
            libegl.eglMakeCurrent.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            libegl.eglMakeCurrent.restype = ctypes.c_bool
            libegl.eglGetCurrentSurface.restype = ctypes.c_void_p
            libegl.eglGetCurrentSurface.argtypes = [ctypes.c_int]
            self._libegl = libegl
        except Exception:
            pass

        try:
            libgl = ctypes.CDLL("libGL.so.1")
            libgl.glXGetCurrentContext.restype = ctypes.c_void_p
            libgl.glXGetCurrentDisplay.restype = ctypes.c_void_p
            libgl.glXMakeCurrent.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            libgl.glXMakeCurrent.restype = ctypes.c_bool
            libgl.glXGetCurrentDrawable.restype = ctypes.c_ulong
            self._libgl = libgl
        except Exception:
            pass

        if sys.platform == "win32":
            try:
                opengl32 = ctypes.windll.opengl32
                opengl32.wglGetCurrentContext.restype = ctypes.c_void_p
                opengl32.wglGetCurrentDC.restype = ctypes.c_void_p
                opengl32.wglMakeCurrent.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
                opengl32.wglMakeCurrent.restype = ctypes.c_bool
                self._libwgl = opengl32
            except Exception:
                pass

    def capture_current(self):
        if self._libegl:
            ctx = self._libegl.eglGetCurrentContext()
            if ctx:
                # 3.1.0 review: the draw and read surfaces too. Restoring the context without them left the game window with no surface
                # on Wayland (EGL): every frame went nowhere and the window kept showing the loading screen while the game ran.
                return ("egl", self._libegl.eglGetCurrentDisplay(), ctx, self._libegl.eglGetCurrentSurface(0x3059),
                        self._libegl.eglGetCurrentSurface(0x305A))
        if self._libgl:
            ctx = self._libgl.glXGetCurrentContext()
            if ctx:
                return ("glx", self._libgl.glXGetCurrentDisplay(), ctx, self._libgl.glXGetCurrentDrawable())
        if self._libwgl:
            ctx = self._libwgl.wglGetCurrentContext()
            if ctx:
                return ("wgl", self._libwgl.wglGetCurrentDC(), ctx)
        return None

    def make_current(self, handle) -> None:
        if not handle:
            return
        kind = handle[0]
        if kind == "egl":
            draw, read = (handle[3], handle[4]) if len(handle) > 4 else (None, None)
            self._libegl.eglMakeCurrent(handle[1], draw, read, handle[2])
        elif kind == "glx":
            self._libgl.glXMakeCurrent(handle[1], handle[3] if len(handle) > 3 else 0, handle[2])
        elif kind == "wgl":
            self._libwgl.wglMakeCurrent(handle[1], handle[2])

    def release_current(self, handle) -> None:
        if not handle:
            return
        kind = handle[0]
        if kind == "egl":
            self._libegl.eglMakeCurrent(handle[1], None, None, None)
        elif kind == "glx":
            self._libgl.glXMakeCurrent(handle[1], 0, None)
        elif kind == "wgl":
            self._libwgl.wglMakeCurrent(handle[1], None)


_context_binder = _GLContextBinder()


def _create_gl_context():
    """A standalone compute context, EGL for preference.

    The game's window already holds a GLX context on the main thread, and a brain steps on a thread of its own.
    A second GLX context goes through the same Xlib display connection, and Mesa answers the worker thread's
    glXMakeCurrent with BadAccess, which takes the process down. An EGL context carries its own connection and
    is not affected. Where EGL is missing (older Mesa, and Windows, which uses WGL) the plain context is fine:
    there the second context does not share a display connection to begin with.
    """
    last: Exception | None = None
    for kwargs in ({"standalone": True, "backend": "egl"}, {"standalone": True}):
        try:
            return moderngl.create_context(**kwargs)
        except Exception as e:
            last = e
    raise RuntimeError(f"Failed to create ModernGL context: {last}")


def _gl_compute_available() -> tuple[bool, str]:
    """Whether OpenGL 4.3+ compute shaders are supported here.

    Creating a context makes it current, and releasing it leaves the calling thread with none, so the caller's
    own context is captured and put back around the probe. get_max_flies() asks this from the main thread,
    where the 3D renderer's context is the one that must survive.
    """
    if not _moderngl_available:
        return False, "ModernGL is not installed"
    binder = _GLContextBinder()
    saved = binder.capture_current()
    try:
        ctx = _create_gl_context()
        ver = ctx.version_code / 100.0
        if ctx.version_code < 430:
            ctx.release()
            return False, f"OpenGL {ver:.1f} < 4.3 (Compute shaders not supported)"
        renderer = ctx.info.get("GL_RENDERER", "OpenGL GPU")
        ctx.release()
        return True, f"OpenGL {ver:.1f}: {renderer}"
    except Exception as e:
        return False, f"OpenGL context creation failed: {e}"
    finally:
        binder.make_current(saved)


# Batched multi-fly on gl (2.10). A group of flies shares one GL context on a thread of its own, with the connectome's
# weights on the GPU once. Each step the flies that are due are dispatched together: one SpMM pass streams the weights
# once for all of them (the torch backends' batched SpMM, done in a compute shader), then one LIF pass updates every
# fly. Per-fly state is fly-major (fly f's neuron i at f * n + i); spikes are one word per neuron with a bit per fly,
# so a group holds up to 32 flies. A fly on its own (KICK_THE_FLY_GL_BATCH=0, or the only gl brain) is a group of one
# and runs the same shaders, which is what keeps batched and unbatched bit-exact with each other.
#
# Flies learn separately, so their weights are not all the same. Rows where they differ (the KC -> MBON rows, once
# learning touches them) become private: row_off[row] points into row_vals, which holds that row's weights once per
# fly. Every other row reads the shared values. A fly whose weights differ from the group's in too many rows (a lesion,
# a synapse threshold) moves to a group of its own.

_NO_ROW = 0xFFFFFFFF


class _GLFence:
    """Waits for the GPU without holding Python's GIL.

    ModernGL keeps the GIL through every GL call, ctx.finish() and buffer reads included, so while one group's GPU
    work runs every brain thread in the process stands still. glClientWaitSync called through ctypes releases it:
    the brains do their Python work while the GPU steps them. Resolved with the group's context current; if any
    entry point is missing, wait() is ctx.finish()."""

    _GPU_COMMANDS_COMPLETE, _FLUSH, _TIMEOUT, _FAILED = 0x9117, 0x1, 0x911B, 0x911D

    def __init__(self, ctx) -> None:
        import ctypes
        import sys
        self.ctx = ctx
        self._fence = self._wait = self._delete = None
        functype = getattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE) if sys.platform == "win32" else ctypes.CFUNCTYPE
        protos = {"glFenceSync": functype(ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint),
                  "glClientWaitSync": functype(ctypes.c_uint, ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint64),
                  "glDeleteSync": functype(None, ctypes.c_void_p)}
        found = {}
        for name, proto in protos.items():
            addr = self._proc_address(ctypes, sys, name)
            if not addr:
                return
            found[name] = proto(addr)
        self._fence, self._wait, self._delete = found["glFenceSync"], found["glClientWaitSync"], found["glDeleteSync"]

    @staticmethod
    def _proc_address(ctypes, sys, name: str) -> int:
        b = name.encode()
        loaders = []
        if sys.platform == "win32":
            loaders.append(("opengl32", "wglGetProcAddress"))
        else:
            loaders += [("libEGL.so.1", "eglGetProcAddress"), ("libGL.so.1", "glXGetProcAddressARB")]
        for lib, fn in loaders:
            try:
                handle = ctypes.windll.opengl32 if lib == "opengl32" else ctypes.CDLL(lib)
                get = getattr(handle, fn)
                get.restype = ctypes.c_void_p
                get.argtypes = [ctypes.c_char_p]
                addr = get(b)
                if addr and addr not in (1, 2, 3, -1):         # wglGetProcAddress's error values
                    return addr
            except Exception:
                continue
        return 0

    @property
    def available(self) -> bool:
        return self._fence is not None

    def wait(self) -> None:
        if self._fence is None:
            self.ctx.finish()
            return
        sync = self._fence(self._GPU_COMMANDS_COMPLETE, 0)
        try:
            while True:
                r = self._wait(sync, self._FLUSH, 1_000_000_000)
                if r == self._FAILED:
                    self.ctx.finish()
                    return
                if r != self._TIMEOUT:
                    return
        finally:
            self._delete(sync)

# SSBO binding points, shared by the three shaders of one context
_B_ROWPTR, _B_COLIND, _B_VALUES, _B_V, _B_REFR, _B_SPIKES, _B_NOISE, _B_SENS, _B_ISYN, _B_SCATTER, _B_ROWOFF, \
    _B_ROWVALS, _B_PARAMS, _B_TARGET, _B_DPRE, _B_DPOST = range(16)

_SPMM_COMPUTE_SHADER = """
#version 430
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer BRowPtr { uint row_ptr[]; };
layout(std430, binding = 1) readonly buffer BColInd { uint col_ind[]; };
layout(std430, binding = 2) readonly buffer BValues { float values[]; };
layout(std430, binding = 5) readonly buffer BSpikes { uint spikes[]; };
layout(std430, binding = 8) writeonly buffer BISyn { float i_syn[]; };
layout(std430, binding = 10) readonly buffer BRowOff { uint row_off[]; };
layout(std430, binding = 11) readonly buffer BRowVals { float row_vals[]; };
layout(std430, binding = 14) readonly buffer BDPre { float d_pre[]; };      // individuality (3.1.0): per fly, per neuron
layout(std430, binding = 15) readonly buffer BDPost { float d_post[]; };
uniform uint num_neurons;
uniform uint num_slots;
uniform uint active_mask;
uniform uint gain_mask;                 // the flies whose brains carry individuality gains; the others run the unchanged path

shared float s_w[64];
shared uint s_col[64];
shared uint s_mask[32][2];              // per fly: which of the chunk's 64 synapses had a spiking presynaptic neuron

// One workgroup per row (a neuron's inputs). The 64 lanes look up 64 synapses' presynaptic spikes at once, which is
// what a thread walking a 9,184-synapse row alone waits on; then lane f adds fly f's weights for the spiking ones in
// CSR order. Each fly's input is the same additions in the same order as CPUBackend's, whatever the batch.
void main() {
    uint row = gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x;
    if (row >= num_neurons) return;                     // the whole workgroup, so the barriers below stay uniform
    uint lane = gl_LocalInvocationID.x;
    uint start = row_ptr[row];
    uint end = row_ptr[row + 1];
    uint off = row_off[row];
    bool mine = lane < num_slots && ((active_mask >> lane) & 1u) != 0u;
    bool gaining = lane < 32u && ((gain_mask >> lane) & 1u) != 0u;
    precise float acc = 0.0;
    for (uint base = start; base < end; base += 64u) {
        if (lane < 32u) { s_mask[lane][0] = 0u; s_mask[lane][1] = 0u; }
        barrier();
        uint idx = base + lane;
        if (idx < end) {
            uint bits = spikes[col_ind[idx]] & active_mask;
            if (bits != 0u) {
                if (off == 0xFFFFFFFFu) s_w[lane] = values[idx];
                s_col[lane] = col_ind[idx];
                uint word = lane >> 5;
                uint bit = 1u << (lane & 31u);
                while (bits != 0u) { int f = findLSB(bits); atomicOr(s_mask[f][word], bit); bits &= bits - 1u; }
            }
        }
        memoryBarrierShared();
        barrier();
        if (mine) {
            for (uint word = 0u; word < 2u; ++word) {
                uint m = s_mask[lane][word];
                while (m != 0u) {
                    uint j = word * 32u + uint(findLSB(m));
                    m &= m - 1u;
                    float w = (off == 0xFFFFFFFFu) ? s_w[j] : row_vals[(off + base - start + j) * num_slots + lane];
                    if (gaining) w = w * d_pre[lane * num_neurons + s_col[j]];
                    acc += w;
                }
            }
        }
        barrier();
    }
    if (mine) {
        if (gaining) acc = acc * d_post[lane * num_neurons + row];
        i_syn[lane * num_neurons + row] = acc;
    }
}
"""

_LIF_COMPUTE_SHADER = """
#version 430
layout(local_size_x = 64) in;
struct FlyParams {
    float gain; float bias; float ext_gain; float leak_decay; float leak_reset; float v_reset; float v_thresh;
    int refr_steps; uint noise_off; uint has_sens; uint scale_sens; uint has_reset;
};
layout(std430, binding = 3) buffer BV { float v[]; };
layout(std430, binding = 4) buffer BRefr { int refr[]; };
layout(std430, binding = 5) buffer BSpikes { uint spikes[]; };
layout(std430, binding = 6) readonly buffer BNoise { float noise[]; };
layout(std430, binding = 7) readonly buffer BSens { float sensory[]; };
layout(std430, binding = 8) readonly buffer BISyn { float i_syn[]; };
layout(std430, binding = 12) readonly buffer BParams { FlyParams fp[]; };
uniform uint num_neurons;
uniform uint noise_len;
uniform uint active_mask;

// CPUBackend.step's operations in its order; precise keeps the compiler from fusing them into multiply-adds.
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= num_neurons) return;
    uint bits = spikes[i] & ~active_mask;
    uint a = active_mask;
    while (a != 0u) {
        uint f = uint(findLSB(a));
        a &= a - 1u;
        uint k = f * num_neurons + i;
        precise float drive = i_syn[k] * fp[f].gain;
        drive += fp[f].bias;
        drive += noise[f * noise_len + fp[f].noise_off + i];
        if (fp[f].has_sens != 0u) {
            float s = sensory[k];
            drive += (fp[f].scale_sens != 0u) ? fp[f].ext_gain * s : s;
        }
        precise float vv = v[k] * fp[f].leak_decay;
        if (fp[f].has_reset != 0u) vv += fp[f].leak_reset;
        vv += drive;
        int r = refr[k];
        if (r > 0) {
            vv = fp[f].v_reset;
            r -= 1;
        }
        if (vv >= fp[f].v_thresh) {
            bits |= 1u << f;
            v[k] = fp[f].v_reset;
            refr[k] = fp[f].refr_steps;
        } else {
            v[k] = vv;
            refr[k] = r;
        }
    }
    spikes[i] = bits;
}
"""

_SCATTER_COMPUTE_SHADER = """
#version 430
layout(local_size_x = 64) in;
layout(std430, binding = 13) buffer BTarget { float target[]; };
layout(std430, binding = 9) readonly buffer BScatter { uvec2 items[]; };   // (index into target, float bits)
uniform uint count;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= count) return;
    uvec2 it = items[i];
    target[it.x] = uintBitsToFloat(it.y);
}
"""

_FLY_PARAMS = np.dtype([("gain", "<f4"), ("bias", "<f4"), ("ext_gain", "<f4"), ("leak_decay", "<f4"),
                        ("leak_reset", "<f4"), ("v_reset", "<f4"), ("v_thresh", "<f4"), ("refr_steps", "<i4"),
                        ("noise_off", "<u4"), ("has_sens", "<u4"), ("scale_sens", "<u4"), ("has_reset", "<u4")])


class _GLGroup:
    """Up to MAX_SLOTS flies' brains on one GL context, stepped together. Everything GL runs on the group's own
    thread: brains hand it their steps (submit) and anything else (call) and wait for the answer."""

    MAX_SLOTS = 32                  # one bit per fly in a neuron's spike word
    GATHER_S = 0.002                # how long a dispatch waits for the group's other running flies
    LIVE_S = 0.05                   # a fly that stepped this recently is running, and worth waiting for
    MAX_PRIVATE = 0.125             # at most this share of the synapses may be per-fly before a fly gets its own group

    _registry: list["_GLGroup"] = []
    _registry_lock = threading.Lock()

    @classmethod
    def join(cls, backend: "GLBackend", shared: bool) -> tuple["_GLGroup", int]:
        """A slot for this brain: in an open shared group that takes it, else in a new group."""
        with cls._registry_lock:
            cls._registry[:] = [g for g in cls._registry if not g.closed]
            if shared:
                for g in list(cls._registry):
                    try:
                        slot = g.call(g._add, backend)
                    except Exception:                       # closed since, or broken: try the next one
                        continue
                    if slot is not None:
                        return g, slot
            g = cls(backend.sim, shared)
            slot = g.call(g._add, backend)
            if slot is None:
                g.close()
                raise RuntimeError("a new OpenGL brain group refused its first brain")
            if shared:
                cls._registry.append(g)
            return g, slot

    def __init__(self, sim: Any, shared: bool) -> None:
        W = sim.W_csr
        self.shared = shared
        self.n = int(sim.n)
        self.nnz = int(W.nnz)
        self.indptr = W.indptr                       # structure the members share (compared by value on joining)
        self.indices = W.indices
        self.base = np.ascontiguousarray(W.data, dtype=np.float32).copy()   # the shared weights, as on the GPU
        self.noise_len = int(sim._noise.size)
        self.members: list[Any] = []                 # weakref to each slot's backend, or None
        self.serials: dict[int, int] = {}            # which admission owns each slot (see _remove)
        self._serial = 0
        self.row_off = np.full(self.n, _NO_ROW, np.uint32)
        self.priv_pos = np.zeros(0, np.int64)        # W_csr.data index of each private entry, in row_vals order
        self.slots = 0
        self.max_slots = self.MAX_SLOTS
        self.dispatches = 0                          # how many batched steps ran, and how many fly-steps they carried
        self.fly_steps = 0
        self.closed = False
        self._cond = threading.Condition()
        self._tasks: deque = deque()
        self._pending: dict[int, tuple] = {}
        self._first_pending = 0.0
        self._last_seen: dict[int, float] = {}
        self._ready = threading.Event()
        self._error: Exception | None = None
        self.ctx = None
        self.device_name = ""
        self._thread = threading.Thread(target=self._run, name="gl-brains", daemon=True)
        self._thread.start()
        self._ready.wait()
        if self._error is not None:
            raise self._error

    # --- called from brain threads --------------------------------------------------------------------------
    def call(self, fn, *args):
        """Run fn(*args) on the group's thread and return its result (or raise its exception)."""
        fut: Future = Future()
        with self._cond:
            if self.closed:
                raise RuntimeError("this OpenGL brain group is closed")
            self._tasks.append((fn, args, fut))
            self._cond.notify_all()
        return fut.result()

    def post(self, fn, *args) -> None:
        """call() without waiting (a garbage-collected brain giving its slot back)."""
        with self._cond:
            if not self.closed:
                self._tasks.append((fn, args, Future()))
                self._cond.notify_all()

    def submit(self, slot: int, params: np.ndarray, sens: np.ndarray | None) -> np.ndarray:
        """One step for this slot's fly, batched with whichever of the others are due. Returns its spikes."""
        fut: Future = Future()
        with self._cond:
            if self.closed:
                raise RuntimeError("this OpenGL brain group is closed")
            now = time.perf_counter()
            if not self._pending:
                self._first_pending = now
            self._pending[slot] = (params, sens, fut)
            self._last_seen[slot] = now
            self._cond.notify_all()
        return fut.result()

    def close(self) -> None:
        # join() drops closed groups from the registry. Taking its lock here could deadlock: this runs on the
        # group's thread (the last brain leaving) while join() may hold the lock waiting for this very thread.
        with self._cond:
            self.closed = True
            self._cond.notify_all()

    @property
    def member_count(self) -> int:
        return sum(1 for m in self.members if m is not None and m() is not None)

    # --- the group's thread ---------------------------------------------------------------------------------
    def _run(self) -> None:
        try:
            self._init_context()
        except Exception as e:
            self._error = e
            self.closed = True
            self._ready.set()
            return
        self._ready.set()
        while True:
            with self._cond:
                while not self._tasks and not self._pending and not self.closed:
                    self._cond.wait()
                if self._tasks:
                    task = self._tasks.popleft()
                    batch = None
                elif self._pending:
                    task = None
                    deadline = self._first_pending + self.GATHER_S
                    while not self._tasks:
                        now = time.perf_counter()
                        live = {s for s, t in self._last_seen.items() if now - t < self.LIVE_S}
                        if live.issubset(self._pending) or now >= deadline:
                            break
                        self._cond.wait(deadline - now)
                    batch, self._pending = self._pending, {}
                else:                                         # closed, nothing left to do
                    break
            if task is not None:
                fn, args, fut = task
                try:
                    fut.set_result(fn(*args))
                except Exception as e:
                    fut.set_exception(e)
            elif batch:
                try:
                    self._dispatch(batch)
                except Exception as e:
                    for _, _, fut in batch.values():
                        if not fut.done():
                            fut.set_exception(e)
                    self.close()                              # its brains fall back to the CPU; no one joins it again
        for _, _, fut in list(self._pending.values()) + [(None, None, t[2]) for t in self._tasks]:
            if not fut.done():
                fut.set_exception(RuntimeError("this OpenGL brain group is closed"))
        try:
            self.ctx.release()
        except Exception:
            pass

    def _init_context(self) -> None:
        self.ctx = ctx = _create_gl_context()
        if ctx.version_code < 430:
            ver = ctx.version_code / 100.0
            ctx.release()
            raise RuntimeError(f"OpenGL {ver:.1f} does not support compute shaders (OpenGL 4.3+ required)")
        self.device_name = f"OpenGL {ctx.version_code / 100.0:.1f}: {ctx.info.get('GL_RENDERER', 'OpenGL GPU')}"
        bindings = int(ctx.info.get("GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS", 0) or 0)
        if bindings and bindings < 16:
            raise RuntimeError(f"OpenGL offers {bindings} storage buffer bindings, the brain needs 16")
        block = int(ctx.info.get("GL_MAX_SHADER_STORAGE_BLOCK_SIZE", 0) or 0) & 0xFFFFFFFF
        if block:
            # every fly's noise bank is one buffer, and it has to fit in one shader storage block
            self.max_slots = max(1, min(self.MAX_SLOTS, block // (self.noise_len * 4)))
        self.fence = _GLFence(ctx)
        self.cs_spmm = ctx.compute_shader(_SPMM_COMPUTE_SHADER)
        self.cs_lif = ctx.compute_shader(_LIF_COMPUTE_SHADER)
        self.cs_scatter = ctx.compute_shader(_SCATTER_COMPUTE_SHADER)
        self.num_groups = (self.n + 63) // 64
        self.row_groups = (min(self.n, 65535), (self.n + 65534) // 65535)
        self.buf_rowptr = ctx.buffer(self.indptr.astype(np.uint32).tobytes())
        self.buf_colind = ctx.buffer(self.indices.astype(np.uint32).tobytes())
        self.buf_values = ctx.buffer(self.base.tobytes())
        self.buf_rowoff = ctx.buffer(self.row_off.tobytes())
        self.buf_spikes = ctx.buffer(reserve=self.n * 4)
        self.buf_rowvals = ctx.buffer(reserve=64 * 1024)
        self.buf_scatter = ctx.buffer(reserve=64 * 1024)
        self.buf_v = self.buf_refr = self.buf_isyn = self.buf_sens = self.buf_noise = self.buf_params = None
        self.buf_dpre = self.buf_dpost = None
        self.gain_mask = 0
        self.cs_spmm["num_neurons"] = self.n
        self.cs_lif["num_neurons"] = self.n
        self.cs_lif["noise_len"] = self.noise_len
        self._alloc_slots(1)

    def _bind(self) -> None:
        for buf, b in ((self.buf_rowptr, _B_ROWPTR), (self.buf_colind, _B_COLIND), (self.buf_values, _B_VALUES),
                       (self.buf_v, _B_V), (self.buf_refr, _B_REFR), (self.buf_spikes, _B_SPIKES),
                       (self.buf_noise, _B_NOISE), (self.buf_sens, _B_SENS), (self.buf_isyn, _B_ISYN),
                       (self.buf_scatter, _B_SCATTER), (self.buf_rowoff, _B_ROWOFF), (self.buf_rowvals, _B_ROWVALS),
                       (self.buf_params, _B_PARAMS), (self.buf_dpre, _B_DPRE), (self.buf_dpost, _B_DPOST)):
            buf.bind_to_storage_buffer(b)

    def _alloc_slots(self, slots: int) -> None:
        """(Re)allocate the per-fly buffers for `slots` flies. Callers re-upload every member afterwards."""
        for name in ("buf_v", "buf_refr", "buf_isyn", "buf_sens", "buf_noise", "buf_params", "buf_dpre", "buf_dpost"):
            buf = getattr(self, name)
            if buf is not None:
                buf.release()
        n, ctx = self.n, self.ctx
        self.buf_v = ctx.buffer(reserve=n * 4 * slots)
        self.buf_refr = ctx.buffer(reserve=n * 4 * slots)
        self.buf_isyn = ctx.buffer(reserve=n * 4 * slots)
        self.buf_sens = ctx.buffer(reserve=n * 4 * slots)
        self.buf_noise = ctx.buffer(reserve=self.noise_len * 4 * slots)
        self.buf_params = ctx.buffer(reserve=_FLY_PARAMS.itemsize * slots)
        self.buf_dpre = ctx.buffer(reserve=n * 4 * slots)
        self.buf_dpost = ctx.buffer(reserve=n * 4 * slots)
        self.gain_mask = 0                                    # re-set by _push for every member after a resize
        old = self.slots
        self.slots = slots
        self.members += [None] * (slots - len(self.members))
        self.cs_spmm["num_slots"] = slots
        if len(self.priv_pos):
            self._write_rowvals(0, self.priv_pos)             # row_vals' layout depends on the slot count
        self._bind()
        if old and old != slots:
            log.info("OpenGL brain group resized from %d to %d flies", old, slots)

    def _live_members(self) -> list[tuple[int, Any]]:
        out = []
        for slot, ref in enumerate(self.members):
            be = ref() if ref is not None else None
            if be is not None:
                out.append((slot, be))
            elif ref is not None:
                self.members[slot] = None
        return out

    def _data_of(self, slot: int) -> np.ndarray:
        ref = self.members[slot] if slot < len(self.members) else None
        be = ref() if ref is not None else None
        return be.sim.W_csr.data if be is not None else self.base

    def _rows_of(self, pos: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.indptr, pos, side="right") - 1

    def _add(self, backend: "GLBackend") -> int | None:
        """Take this brain into a free slot, or None if it doesn't fit here (other wiring, or no room, or this group has closed)."""
        if self.closed:
            # 3.1.0: a join queued behind the departure of the group's last brain ran after _remove had closed the group, and the
            # brain it admitted then failed on its first step and fell back to the CPU (seen in the 3D game's first seconds on gl)
            return None
        sim = backend.sim
        W = sim.W_csr
        live = self._live_members()
        if live:
            if (sim.n != self.n or W.nnz != self.nnz or sim._noise.size != self.noise_len
                    or not (W.indptr is self.indptr or np.array_equal(W.indptr, self.indptr))
                    or not (W.indices is self.indices or np.array_equal(W.indices, self.indices))):
                return None
        free = [s for s in range(self.slots) if self.members[s] is None]
        if not free and self.slots >= self.max_slots:
            return None
        data = np.ascontiguousarray(W.data, dtype=np.float32)
        if live:
            rows = np.unique(self._rows_of(np.flatnonzero(data != self.base)))
            rows = rows[self.row_off[rows] == _NO_ROW]
            if self._private_len() + int(np.sum(np.diff(self.indptr)[rows])) > self.MAX_PRIVATE * self.nnz:
                return None
        else:                                                  # an empty group adopts this brain's weights
            self.base[:] = data
            self.buf_values.write(self.base.tobytes())
            rows = np.zeros(0, np.int64)
        if not free:
            for s, be in live:
                self._pull(s, be.sim)
            self._alloc_slots(min(self.max_slots, max(1, self.slots * 2)))
            for s, be in live:
                self._push(s, be.sim)
                self._push_noise(s, be.sim)
            free = [s for s in range(self.slots) if self.members[s] is None]
        slot = free[0]
        self.members[slot] = weakref.ref(backend)
        self._serial += 1
        self.serials[slot] = self._serial
        self._last_seen.pop(slot, None)
        if len(rows):
            self._privatize(rows)
        if len(self.priv_pos):
            self._write_rowvals(slot, self.priv_pos, only_slot=True)
        self._push(slot, sim)
        self._push_noise(slot, sim)
        return slot

    def _remove(self, slot: int, serial: int | None = None) -> None:
        # 3.1.0: a garbage-collected brain's departure is posted to this thread and can run after a NEW brain took its slot (a slot whose
        # owner has died is free to reuse). It must remove only the brain that posted it, or it empties the new brain's slot, the group
        # closes under it, and the brain fails on its next step and falls back to the CPU (seen in the 3D game's first seconds on gl).
        if serial is not None and self.serials.get(slot) != serial:
            return
        if slot < len(self.members):
            self.members[slot] = None
        self.gain_mask &= ~(1 << slot)
        self._last_seen.pop(slot, None)
        if not self._live_members():
            self.close()

    def _private_len(self) -> int:
        return len(self.priv_pos)

    def _privatize(self, rows: np.ndarray) -> int:
        """Give these rows per-fly weights: each slot's copy comes from its own brain's W_csr.data."""
        rows = np.asarray(rows, np.int64)
        lens = np.diff(self.indptr)[rows].astype(np.int64)
        starts = self.indptr[rows].astype(np.int64)
        first = len(self.priv_pos)
        offs = first + np.concatenate(([0], np.cumsum(lens)[:-1])) if len(rows) else np.zeros(0, np.int64)
        pos = np.repeat(starts - offs, lens) + np.arange(first, first + int(lens.sum()))
        self.row_off[rows] = offs.astype(np.uint32)
        self.priv_pos = np.concatenate((self.priv_pos, pos))
        self.buf_rowoff.write(self.row_off.tobytes())
        return self._write_rowvals(None, pos, first=first) + self.row_off.nbytes

    def _write_rowvals(self, slot: int | None, pos: np.ndarray, first: int = 0, only_slot: bool = False) -> int:
        """Upload private entries. slot None: every slot's values for entries first.. (appended rows); only_slot: one
        slot's values for all entries (a brain joining or a full weight change); slot 0 with first 0 and not
        only_slot: rebuild the whole table (after a resize)."""
        F = self.slots
        if only_slot:
            e = np.arange(len(self.priv_pos), dtype=np.int64)
            return self._scatter_to(self.buf_rowvals, e * F + slot, self._data_of(slot)[self.priv_pos])
        need = (first + len(pos)) * F * 4
        if self.buf_rowvals.size < need:
            grown = self.ctx.buffer(reserve=max(need, 2 * self.buf_rowvals.size))
            if first:
                self.ctx.copy_buffer(grown, self.buf_rowvals, size=first * F * 4)
            self.buf_rowvals.release()
            self.buf_rowvals = grown
            self.buf_rowvals.bind_to_storage_buffer(_B_ROWVALS)
        block = np.empty((len(pos), F), np.float32)
        for s in range(F):
            block[:, s] = self._data_of(s)[pos]
        self.buf_rowvals.write(block.tobytes(), offset=first * F * 4)
        return block.nbytes

    def _scatter_to(self, target, index: np.ndarray, values: np.ndarray) -> int:
        """(index, value) pairs written into `target` by a compute shader (issue #2: 8 bytes per changed synapse)."""
        if not len(index):
            return 0
        items = np.empty((len(index), 2), np.uint32)
        items[:, 0] = index
        items[:, 1] = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
        if self.buf_scatter.size < items.nbytes:
            size = max(items.nbytes, 2 * self.buf_scatter.size)
            self.buf_scatter.release()
            self.buf_scatter = self.ctx.buffer(reserve=size)
            self.buf_scatter.bind_to_storage_buffer(_B_SCATTER)
        self.buf_scatter.write(items.tobytes())
        target.bind_to_storage_buffer(_B_TARGET)
        self.cs_scatter["count"] = len(index)
        self.cs_scatter.run((len(index) + 63) // 64)
        self.ctx.memory_barrier(moderngl.SHADER_STORAGE_BARRIER_BIT)
        return items.nbytes

    def _write_runs(self, pos: np.ndarray, data: np.ndarray, gap: int) -> int:
        """Changed shared weights as a few contiguous writes. pos must be sorted."""
        brk = np.flatnonzero(np.diff(pos) > gap)
        starts = np.concatenate(([pos[0]], pos[brk + 1]))
        ends = np.concatenate((pos[brk], [pos[-1]])) + 1
        total = 0
        for a, b in zip(starts.tolist(), ends.tolist()):
            chunk = np.ascontiguousarray(data[a:b], dtype=np.float32)
            self.buf_values.write(chunk, offset=a * 4)
            total += chunk.nbytes
        return total

    def _weights(self, slot: int, positions: np.ndarray | None, mode: str, gap: int):
        """A brain's W_csr.data changed at `positions` (None: anywhere). Returns (kind, bytes, seconds), or None when
        the brain's weights now differ from the group's in too many rows and it should move to a group of its own."""
        t0 = time.perf_counter()
        data = self._data_of(slot)
        sole = len(self._live_members()) == 1
        if positions is None:
            kind = "full"
            if sole:                                          # its weights are the shared ones
                self.base[:] = data
                self.buf_values.write(self.base.tobytes())
                nbytes = self.base.nbytes
            else:
                rows = np.unique(self._rows_of(np.flatnonzero(np.asarray(data, np.float32) != self.base)))
                rows = rows[self.row_off[rows] == _NO_ROW]
                if self._private_len() + int(np.sum(np.diff(self.indptr)[rows])) > self.MAX_PRIVATE * self.nnz:
                    return None
                nbytes = self._privatize(rows) if len(rows) else 0
            if len(self.priv_pos):
                nbytes += self._write_rowvals(slot, self.priv_pos, only_slot=True)
            return kind, nbytes, time.perf_counter() - t0
        pos = np.asarray(positions, np.int64)
        if not len(pos):
            return "part", 0, 0.0
        if sole and not len(self.priv_pos):                   # a fly on its own: every row is shared, skip the lookup
            rows, shared = None, np.ones(len(pos), bool)
        else:
            rows = self._rows_of(pos)
            shared = self.row_off[rows] == _NO_ROW
        nbytes = 0
        if shared.any():
            if sole:
                sp_pos = pos[shared]
                self.base[sp_pos] = data[sp_pos]
                if mode == "scatter":
                    nbytes += self._scatter_to(self.buf_values, sp_pos, self.base[sp_pos])
                else:
                    nbytes += self._write_runs(sp_pos, self.base, gap)
                pos = pos[~shared]
                rows = rows[~shared] if rows is not None else None
            else:
                new = np.unique(rows[shared])
                if self._private_len() + int(np.sum(np.diff(self.indptr)[new])) > self.MAX_PRIVATE * self.nnz:
                    return None
                nbytes += self._privatize(new)
        if len(pos):
            e = self.row_off[rows].astype(np.int64) + (pos - self.indptr[rows])
            nbytes += self._scatter_to(self.buf_rowvals, e * self.slots + slot, data[pos])
        return "part", nbytes, time.perf_counter() - t0

    def _read_weights(self, slot: int) -> np.ndarray:
        """The weights this slot's fly is using, read back from the GPU."""
        self.ctx.memory_barrier(moderngl.BUFFER_UPDATE_BARRIER_BIT)
        self.ctx.finish()
        out = np.frombuffer(self.buf_values.read(), dtype=np.float32).copy()
        if len(self.priv_pos):
            F = self.slots
            rv = np.frombuffer(self.buf_rowvals.read(size=len(self.priv_pos) * F * 4), dtype=np.float32)
            out[self.priv_pos] = rv.reshape(-1, F)[:, slot]
        return out

    def _push(self, slot: int, sim: Any) -> None:
        n = self.n
        self.buf_v.write(np.ascontiguousarray(sim.v, dtype=np.float32).tobytes(), offset=slot * n * 4)
        self.buf_refr.write(np.ascontiguousarray(sim.refr, dtype=np.int32).tobytes(), offset=slot * n * 4)
        self.ctx.finish()
        bits = np.frombuffer(self.buf_spikes.read(), dtype=np.uint32).copy()
        bit = np.uint32(1 << slot)
        bits &= ~bit
        bits[np.asarray(sim.spikes, bool)] |= bit
        self.buf_spikes.write(bits.tobytes())
        self._push_gains(slot, sim)

    def _push_gains(self, slot: int, sim: Any) -> None:
        """This fly's individuality gains (d_pre scales each presynaptic neuron's spike, d_post each postsynaptic neuron's input, as
        LIFSim._propagate does), or nothing: a fly without them runs the unchanged shader path."""
        n = self.n
        pre, post = getattr(sim, "d_pre", None), getattr(sim, "d_post", None)
        if pre is None and post is None:
            self.gain_mask &= ~(1 << slot)
            return
        pre = np.ones(n, np.float32) if pre is None else np.ascontiguousarray(pre, dtype=np.float32)
        post = np.ones(n, np.float32) if post is None else np.ascontiguousarray(post, dtype=np.float32)
        self.buf_dpre.write(pre.tobytes(), offset=slot * n * 4)
        self.buf_dpost.write(post.tobytes(), offset=slot * n * 4)
        self.gain_mask |= 1 << slot

    def _push_noise(self, slot: int, sim: Any) -> None:
        if sim._noise.size != self.noise_len:
            raise RuntimeError("the noise bank changed size")
        self.buf_noise.write(np.ascontiguousarray(sim._noise, dtype=np.float32).tobytes(),
                             offset=slot * self.noise_len * 4)

    def _pull(self, slot: int, sim: Any) -> None:
        n = self.n
        self.ctx.memory_barrier(moderngl.BUFFER_UPDATE_BARRIER_BIT)
        self.ctx.finish()
        sim.v[:] = np.frombuffer(self.buf_v.read(size=n * 4, offset=slot * n * 4), dtype=np.float32)
        sim.refr[:] = np.frombuffer(self.buf_refr.read(size=n * 4, offset=slot * n * 4), dtype=np.int32)
        bits = np.frombuffer(self.buf_spikes.read(), dtype=np.uint32)
        sim.spikes[:] = (bits >> np.uint32(slot)) & np.uint32(1)

    def _dispatch(self, batch: dict[int, tuple]) -> None:
        n = self.n
        params = np.zeros(self.slots, _FLY_PARAMS)
        active = 0
        for slot, (p, sens, _) in batch.items():
            params[slot] = p
            active |= 1 << slot
            if sens is not None:
                self.buf_sens.write(sens.tobytes(), offset=slot * n * 4)
        self.buf_params.write(params.tobytes())
        self.cs_spmm["active_mask"] = active
        self.cs_spmm["gain_mask"] = self.gain_mask
        self.cs_lif["active_mask"] = active
        self.cs_spmm.run(*self.row_groups)
        self.ctx.memory_barrier(moderngl.SHADER_STORAGE_BARRIER_BIT)
        self.cs_lif.run(self.num_groups)
        # SHADER_STORAGE orders the next shader's view of these buffers. Reading one back on the host is a
        # different hazard and needs BUFFER_UPDATE too; without it the map is undefined and Mesa refuses it
        # outright ("cannot map the buffer"). The fence then waits for the write actually to land.
        self.ctx.memory_barrier(moderngl.SHADER_STORAGE_BARRIER_BIT | moderngl.BUFFER_UPDATE_BARRIER_BIT)
        self.fence.wait()
        bits = np.frombuffer(self.buf_spikes.read(), dtype=np.uint32)
        self.dispatches += 1
        self.fly_steps += len(batch)
        for slot, (_, _, fut) in batch.items():
            fut.set_result(((bits >> np.uint32(slot)) & np.uint32(1)).astype(bool))


class GLBackend(SimBackend):
    """ModernGL compute shader backend: vendor-neutral GPU acceleration using OpenGL 4.3+ compute shaders and SSBOs.
    Runs on AMD, NVIDIA, and Intel GPUs across Linux and Windows without requiring PyTorch.

    Every gl brain in a process joins a shared _GLGroup (up to 32 flies on one context, the weights streamed once per
    step for all of them); KICK_THE_FLY_GL_BATCH=0 gives each brain a group of its own instead. The two are
    bit-exact with each other (tests/test_backends.py)."""

    name = "gl"
    # How learning's changed synapses reach the GPU (issue #2). The weight SSBO is W_csr.data in order, so a plastic
    # KC -> MBON synapse at W_csr.data[i] lives at byte offset 4 * i (memory.Memory.csr_pos maps all 41,495 of them).
    #   "scatter"  one upload of (index, value) pairs and a compute shader that writes them into place (default)
    #   "runs"     merge the sorted indices into runs less than RUN_GAP entries apart (the gap is re-sent from the
    #              host copy, the source of truth) and write each with buffer.write(offset=...); 130 runs cover all
    #              41,495 plastic synapses
    # tools/bench_gl_plastic.py, RX 9070 XT, 10 shock pairings (median 8,500 changed synapses per update): full
    # re-upload 41.1 MB and 2.10 ms per update, runs 88 KB and 0.151 ms, scatter 68 KB and 0.032 ms.
    # In a group of several flies a fly's learned rows are per-fly (see _GLGroup) and always go by scatter.
    PLASTIC_UPLOAD = "scatter"
    RUN_GAP = 16
    BATCH = os.environ.get("KICK_THE_FLY_GL_BATCH", "1").strip().lower() not in ("0", "false", "no", "off")

    def __init__(self, sim: Any) -> None:
        super().__init__(sim)
        self.batched = self.BATCH                   # fixed per brain, so batched and unbatched brains can coexist
        self._group: _GLGroup | None = None
        self._slot = -1
        self._serial: int | None = None
        self._finalizer = None
        self._noise_id = None
        self._W = None
        self._fallback: SimBackend | None = None
        # what weight uploads cost: full re-uploads and partial (learning) ones, bytes and seconds (benchmarks, tests)
        self.upload_stats = dict(full_n=0, full_bytes=0, full_s=0.0, part_n=0, part_bytes=0, part_s=0.0)

    def setup(self) -> None:
        if not _moderngl_available:
            raise RuntimeError("ModernGL is not installed")
        ok, dev = _gl_compute_available()
        if not ok:
            raise RuntimeError(dev)
        self.device_name = dev
        # The brain joins its group on first use, from the host state it has then (save states and the Lab's
        # parameters are applied between construction and the first step).

    @property
    def group_size(self) -> int:
        """How many brains share this one's GPU group (1 when unbatched)."""
        return self._group.member_count if self._group is not None else 0

    def _attach(self) -> _GLGroup:
        sim = self.sim
        if self._group is not None and sim.W_csr is not self._W:
            # the matrix itself was replaced (the game's mirror-weights setting does that): leave and rejoin with it
            self._group.call(self._group._pull, self._slot, sim)
            self._leave()
        if self._group is None:
            self._group, self._slot = _GLGroup.join(self, shared=self.batched)
            self._serial = self._group.serials.get(self._slot)
            self._finalizer = weakref.finalize(self, self._group.post, self._group._remove, self._slot, self._serial)
            self._noise_id = id(sim._noise)
            self._W = sim.W_csr
        return self._group

    def _leave(self) -> None:
        g, self._group = self._group, None
        if self._finalizer is not None:
            self._finalizer.detach()
            self._finalizer = None
        if g is not None:
            try:
                g.call(g._remove, self._slot, self._serial)
            except Exception:
                pass

    def close(self) -> None:
        """Give this brain's slot back (it otherwise goes when the brain is garbage-collected)."""
        self._leave()

    def _degrade(self, exc: Exception) -> SimBackend:
        """Hand this brain to the CPU backend after a GL failure, rather than killing the thread it steps on.

        setup() cannot catch these: the context comes up on the group's thread, and the failures land later, where
        create_backend's fallback is long gone. v/refr/spikes keep whatever the last sync left on the host, so the
        fly carries on from there.
        """
        log.warning("the OpenGL compute backend failed (%s: %s); this brain falls back to the CPU backend",
                    type(exc).__name__, exc)
        if self._group is not None:
            try:
                self._group.call(self._group._pull, self._slot, self.sim)
            except Exception:
                pass
            self._leave()
        self._fallback = CPUBackend(self.sim)
        self._fallback.setup()
        self.name = CPUBackend.name
        self.device_name = CPUBackend.device_name
        return self._fallback

    def on_weights_changed(self, positions: np.ndarray | None = None) -> None:
        if self._fallback is not None:
            return self._fallback.on_weights_changed(positions)
        g = self._attach()
        res = g.call(g._weights, self._slot, positions, self.PLASTIC_UPLOAD, self.RUN_GAP)
        if res is None:                              # too different from the others now: a group of its own
            g.call(g._pull, self._slot, self.sim)
            self._leave()
            self._attach()
            res = ("full", 0, 0.0)
        kind, nbytes, secs = res
        if positions is not None and len(positions) == 0:
            return
        st = self.upload_stats
        st[f"{kind}_n"] += 1
        st[f"{kind}_bytes"] += int(nbytes)
        st[f"{kind}_s"] += secs

    def read_weights(self) -> np.ndarray:
        """The weights as they are on the GPU, for tests: this brain's copy of the whole W_csr.data."""
        if self._fallback is not None:
            return np.ascontiguousarray(self.sim.W_csr.data, dtype=np.float32).copy()
        g = self._attach()
        return g.call(g._read_weights, self._slot)

    def sync_to_host(self) -> None:
        if self._fallback is not None:
            return self._fallback.sync_to_host()
        if self._group is not None:
            self._group.call(self._group._pull, self._slot, self.sim)

    def sync_from_host(self) -> None:
        if self._fallback is not None:
            return self._fallback.sync_from_host()
        if self._group is None:
            return                                   # joining uploads the host state anyway
        g = self._group
        g.call(g._push, self._slot, self.sim)
        if self._noise_id != id(self.sim._noise):
            g.call(g._push_noise, self._slot, self.sim)
            self._noise_id = id(self.sim._noise)

    def _params(self, noise_off: int, has_sens: bool) -> np.ndarray:
        sim = self.sim
        p = sim.p
        dt = sim.dtype
        out = np.zeros((), _FLY_PARAMS)
        out["gain"] = dt(sim.gain)
        out["bias"] = dt(p.bias)
        out["ext_gain"] = dt(p.ext_gain)
        out["leak_decay"] = dt(1.0 - sim.leak)
        out["leak_reset"] = sim.leak * dt(p.v_reset)
        out["v_reset"] = dt(p.v_reset)
        out["v_thresh"] = dt(p.v_thresh)
        out["refr_steps"] = int(p.refractory_steps)
        out["noise_off"] = noise_off
        out["has_sens"] = has_sens
        out["scale_sens"] = p.ext_gain != 1.0
        out["has_reset"] = bool(p.v_reset)
        return out

    def step(self, sensory_input: np.ndarray | None = None) -> np.ndarray:
        if self._fallback is not None:
            return self._fallback.step(sensory_input)
        sim = self.sim
        try:
            g = self._attach()
            if self._noise_id != id(sim._noise):
                g.call(g._push_noise, self._slot, sim)
                self._noise_id = id(sim._noise)
            off = int(sim.rng.integers(0, sim._noise.size - self.n))
            sens = None if sensory_input is None else np.ascontiguousarray(sensory_input, dtype=np.float32)
            spikes = g.submit(self._slot, self._params(off, sens is not None), sens)
        except Exception as e:
            return self._degrade(e).step(sensory_input)
        sim.spikes = spikes
        return spikes


def detect_available_backends() -> dict[str, str]:
    """The backends that can run here, mapped to human-readable device names."""
    avail = {"cpu": CPUBackend.device_name}
    if _numba_available:
        avail["numba"] = NumbaBackend.device_name
    ok, gl_dev = _gl_compute_available()
    if ok:
        avail["gl"] = gl_dev
    if _torch_available:
        avail["torch-cpu"] = "PyTorch (cpu)"
        kind = _torch_gpu_kind()
        if kind:
            try:
                name = torch.cuda.get_device_name(0)
            except Exception:
                name = "AMD GPU" if kind == "torch-rocm" else "NVIDIA GPU"
            avail[kind] = f"{'ROCm' if kind == 'torch-rocm' else 'CUDA'}: {name}"
    return avail


# --- 'auto' (3.1.0 task 2) ------------------------------------------------------------------------------------------------------
# Two policies for what 'auto' means. "exact" (the default, and what every headless path uses: --validate, protocols, bundles, replays,
# the selftest) keeps the old chain: a PyTorch GPU, then Numba, then NumPy. "fastest" is what the interactive game switches on at launch
# (set_auto_policy): a capable GPU first through OpenGL compute (any vendor), then a PyTorch GPU, then Numba, then NumPy, each with a
# fallback to the next on any error. GPU backends agree with NumPy statistically, not spike for spike (docs/performance.md); the engine
# in use is recorded in replays, save states and bundles as before, and Settings > Brain shows it with a manual override.
AUTO_POLICIES = ("exact", "fastest")
_auto_policy = "exact"
_SOFTWARE_GL = ("llvmpipe", "softpipe", "swrast", "software", "lavapipe", "microsoft basic render")


def set_auto_policy(policy: str) -> None:
    global _auto_policy
    _auto_policy = policy if policy in AUTO_POLICIES else "exact"


def auto_policy() -> str:
    return _auto_policy


_gpu_cache: tuple[bool, str] | None = None


def gpu_capable(refresh: bool = False) -> tuple[bool, str]:
    """Whether OpenGL compute (4.3+) runs on a real GPU here, and the device string. A software rasterizer (llvmpipe and friends) has the
    compute shaders but runs them on the CPU, slower than NumPy, so it does not count. Cached: the probe makes a context."""
    global _gpu_cache
    if _gpu_cache is None or refresh:
        ok, dev = _gl_compute_available()
        if ok and any(w in dev.lower() for w in _SOFTWARE_GL):
            ok, dev = False, f"{dev} (a software renderer, not a GPU)"
        _gpu_cache = (ok, dev)
    return _gpu_cache


def auto_chain(avail: dict[str, str] | None = None) -> list[str]:
    """The names 'auto' would try, in order, under the current policy. `avail` is detect_available_backends() (probed if omitted)."""
    avail = detect_available_backends() if avail is None else avail
    chain: list[str] = []
    if _auto_policy == "fastest" and "gl" in avail and gpu_capable()[0]:
        chain.append("gl")
    chain += [b for b in ("torch-cuda", "torch-rocm") if b in avail]
    if "numba" in avail:
        chain.append("numba")
    return chain + ["cpu"]


def _try(make, label: str) -> SimBackend | None:
    try:
        b = make()
        b.setup()
        return b
    except Exception as e:
        log.warning("%s backend unavailable (%s)", label, e)
        return None


def create_backend(sim: Any, backend_choice: str = "auto") -> SimBackend:
    """The requested backend, or the best one available for 'auto', falling back to the NumPy CPU backend.
    The returned backend's .name is what actually runs (e.g. 'torch-rocm' when 'torch-cuda' was asked for on a ROCm
    build of PyTorch, 'gl' for OpenGL Compute, or 'cpu' after a fallback), and that is what gets recorded everywhere."""
    choice = (backend_choice or "auto").lower()
    b: SimBackend | None = None
    if choice == "auto":
        # 'gl' is in the chain only under the "fastest" policy (the interactive game). Since 2.10 it is the fastest backend here for one
        # brain (0.96 ms/step on a 9070 XT, Numba 1.06, NumPy 1.21) and batches a process's brains (16 in real time,
        # docs/performance.md), but it is held only to a statistical tolerance of NumPy on other drivers, where Numba is bit-exact; so
        # headless runs (validation, protocols, bundles, replays) keep the exact chain.
        if _auto_policy == "fastest" and _moderngl_available and gpu_capable()[0]:
            b = _try(lambda: GLBackend(sim), "OpenGL Compute")
        if b is None and _torch_gpu_kind():
            b = _try(lambda: TorchBackend(sim, "cuda:0"), "PyTorch GPU")
        if b is None and _numba_available:
            b = _try(lambda: NumbaBackend(sim), "Numba")
    elif choice in ("torch-cuda", "torch-rocm"):
        kind = _torch_gpu_kind()
        if not _torch_available:
            log.warning("%s requested but PyTorch is not installed; using the CPU backend", choice)
        elif kind is None:
            log.warning("%s requested but this PyTorch build sees no GPU (torch %s); using the CPU backend",
                        choice, torch.__version__)
        else:
            if kind != choice:
                log.warning("%s requested; this PyTorch build is %s, using that", choice, kind)
            b = _try(lambda: TorchBackend(sim, "cuda:0"), choice)
    elif choice == "torch-cpu":
        if not _torch_available:
            log.warning("torch-cpu requested but PyTorch is not installed; using the CPU backend")
        else:
            b = _try(lambda: TorchBackend(sim, "cpu"), "PyTorch CPU")
    elif choice == "gl":
        if not _moderngl_available:
            log.warning("gl requested but ModernGL is not installed; using the CPU backend")
        else:
            b = _try(lambda: GLBackend(sim), "OpenGL Compute")
    elif choice == "numba":
        if not _numba_available:
            log.warning("numba requested but Numba is not installed; using the CPU backend")
        else:
            b = _try(lambda: NumbaBackend(sim), "Numba")
    elif choice != "cpu":
        log.warning("unknown simulation backend %r; using the CPU backend", choice)
    if b is None:
        b = CPUBackend(sim)
    try:
        from kickthefly.core import crash
        crash.record_backend(b.name, b.device)
    except Exception:
        pass
    return b
