"""Real training for Kick the Fly: dopamine-gated plasticity on the connectome's own mushroom body synapses.

How flies learn which smells mean trouble (Aso et al. 2014; Hige et al. 2015; Cohn et al. 2015): a smell switches on
a sparse set of Kenyon cells (KCs). Those KCs synapse onto mushroom body output neurons (MBONs). Dopamine neurons
(DANs) that fire during the smell weaken the synapses from the active KCs onto the MBONs in their compartment:
  - PPL1 dopamine signals punishment. Its MBONs (MBON11-20, 30-35 here) promote approach, so weakening them makes
    the fly avoid that smell.
  - PAM dopamine signals reward. Its MBONs (MBON01-10, 21, 24, 26-29 here) promote avoidance, so weakening them
    makes the fly approach that smell.

This module does exactly that on the simulator's synapse matrix:
  - The plastic synapses are all 44k KC -> MBON connections in the connectome. Which dopamine neurons gate each
    MBON comes from the connectome's DAN -> MBON synapse counts (37,909 synapses).
  - Every 50 ms of brain time: eligibility = how far each KC fires above its calm rate, past the resting noise
    (only the most active ~7% count, as KC coding is sparse); dopamine = how far each MBON's DANs fire above calm,
    weighted by their synapse counts, past the resting noise. Each synapse is weakened by  rate * dopamine * eligibility * weight,  down to 10% of its connectome
    value, and slowly recovers toward that value (a forgetting half-life of about 30 minutes of play).
  - The changed weights are the real ones the simulation runs on, so the MBONs genuinely fire less to a trained
    smell.
  - Memory for a smell is read from those synapses: how much input the smell's KC pattern still delivers to the
    punishment-compartment MBONs (fear) and to the reward-compartment MBONs (liking), relative to the untrained
    connectome.
  - Reversal: dopamine in one compartment also restores the same smell's weakened synapses in the opposite
    compartment (Felsenberg et al. 2018 describe opposing memories in separate compartments competing), so being
    hurt by something that used to be rewarding turns liking into fear. Its rate is a game choice.
  - Everything learned is saved to the memory folder (Documents\\Kick the Fly\\memory on Windows,
    ~/.local/share/kickthefly/memory on Linux; see paths.py) and loaded next time, so training carries over
    between flies and sessions until you wipe it.

What is a game rule: pain driving the PPL1 punishment neurons and sugar driving the PAM reward neurons (in real flies
those links go through sensory pathways this sim doesn't reach reliably), each tool having a smell, and the
learning rate and forgetting speed.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

import numpy as np

KC_SPARSE = 280                 # Kenyon cells counted per pattern (of 4,064), like the ~5-10% a real odor activates
LEARN_RATE = 0.012              # per 50 ms update at full dopamine and eligibility: ~6-10 pairings to full memory
FLOOR = 0.1                     # a synapse can be weakened to 10% of its connectome weight
REVERSAL_RATE = 0.03            # opposite dopamine restores a smell's weakened synapses (reversal learning)
FORGET_HALF_LIFE_S = 1800.0     # brain-time seconds for a weakened synapse to recover half-way
DA_MIN_HZ, DA_FULL_HZ = 12.0, 28.0   # measured: calm dopamine flicker peaks at 11 Hz above calm, punishment/reward
                                     # drive the right compartment to ~30 Hz and the other to ~1 Hz
KC_MIN_HZ, KC_FULL_HZ = 8.0, 18.0    # measured: at rest ~4 KCs exceed calm by 10 Hz, a smell ~220 of them
FULL_DEPRESSION = 0.55          # a smell whose KC input is weakened this much reads as memory 1.0
UPDATE_STEPS = 10               # sim steps (5 ms each) between plasticity updates
SETTLE_UPDATES = 100            # 5 s of brain time to learn resting activity before any plasticity
VERSION = 1


def memory_dir() -> Path:
    """Per-OS memory folder from paths.py (Documents\\Kick the Fly\\memory on Windows, XDG data on Linux).
    KICK_THE_FLY_MEMORY still overrides it for tests and portable installs."""
    from kickthefly.core import paths

    return paths.ensure_dir(paths.get().memory_dir, Path.cwd() / "Kick the Fly memory")


class Memory:
    def __init__(self, g, sim, load: bool = True):
        self.sim = sim
        types = np.asarray(g.type).astype(str)
        self.kc = np.flatnonzero(np.char.startswith(types, "KC"))
        dan, mbon, counts = g.dan, g.mbon, g.dan_mbon.astype(np.float32)       # [DAN, MBON]
        dan_types = types[dan]
        is_ppl1 = np.char.startswith(dan_types, "PPL1")
        ppl, pam = counts[is_ppl1].sum(0), counts[~is_ppl1].sum(0)
        plastic_mbon = (ppl + pam) >= 20
        self.mbon = mbon[plastic_mbon]
        self.punish_comp = (ppl >= pam)[plastic_mbon]            # True: PPL1 compartment (approach MBON)
        self.dan = dan
        gate = counts[:, plastic_mbon]
        self.gate = gate / np.maximum(gate.sum(0, keepdims=True), 1)            # DAN share of each MBON's dopamine

        # the KC -> MBON entries of the simulator's matrices (same weights in the CSR and CSC copies)
        W = sim.W_csr
        kc_local = np.full(W.shape[1], -1, np.int32)
        kc_local[self.kc] = np.arange(len(self.kc))
        mb_local = np.full(W.shape[0], -1, np.int32)
        mb_local[self.mbon] = np.arange(len(self.mbon))
        rows = np.repeat(np.arange(W.shape[0]), np.diff(W.indptr))
        pick = np.flatnonzero((mb_local[rows] >= 0) & (kc_local[W.indices] >= 0) & (W.data > 0))
        self.csr_pos = pick
        self.syn_mbon = mb_local[rows[pick]]
        self.syn_kc = kc_local[W.indices[pick]]
        tag = W.copy()
        tag.data = np.arange(W.nnz, dtype=np.float64)
        csc_tag = tag.tocsc()
        inverse = np.empty(W.nnz, np.int64)
        inverse[csc_tag.data.astype(np.int64)] = np.arange(W.nnz)
        self.csc_pos = inverse[pick]
        self.w0 = W.data[pick].astype(np.float32).copy()
        self.w = self.w0.copy()
        self.punish_syn = self.punish_comp[self.syn_mbon]

        n_all = W.shape[0]
        self.kc_calm = np.zeros(len(self.kc), np.float32)
        self.dan_calm = np.zeros(len(dan), np.float32)
        self.calm_ready = False
        self.updates = 0
        self.templates: dict[str, np.ndarray] = {}
        self.log: dict[str, list] = {}           # scent -> [[time, trial_kind, fear, like, mbon_hz], ...]
        self.naive_mbon: dict[str, float] = {}   # approach-MBON response to a smell before any training
        self.last_elig = np.zeros(len(self.kc), np.float32)
        self.last_da = np.zeros(len(self.mbon), np.float32)
        self.enabled = True
        self.dirty = False
        self.lock = threading.Lock()
        self.path = memory_dir() / "fly-memory.npz"
        self.log_path = self.path.with_name("training-log.json")
        self.signature = np.array([len(self.w0), float(self.w0.sum())])
        self.n_neurons = n_all
        if load:                                  # False: an untrained fly (assays, validation, tests)
            self.load()

    # --- the plasticity rule (brain thread) -------------------------------------------------------------------------
    def step(self, rates: np.ndarray, calm: bool, steps: int) -> None:
        """rates: per-neuron spikes/step EMA from the simulator. Runs every UPDATE_STEPS sim steps."""
        kr = rates[self.kc]
        dr = rates[self.dan]
        self.updates += 1
        if self.updates == 1 and not self.calm_ready:
            self.kc_calm[:], self.dan_calm[:] = kr, dr
        # calm tracks resting activity: quickly while nothing is happening, very slowly (~100 s) otherwise, so it never
        # stays stuck on the start-up transient; learning waits until it has settled
        k = 0.02 if calm or self.updates < SETTLE_UPDATES else 0.0005
        self.kc_calm += (kr - self.kc_calm) * k
        self.dan_calm += (dr - self.dan_calm) * k
        if self.updates < SETTLE_UPDATES:
            return
        elig = self.pattern(kr)
        self.last_elig = elig
        da_hz = np.maximum(dr - self.dan_calm, 0) / 0.005                     # spikes/step -> Hz above calm
        gated = da_hz @ self.gate                                             # [DAN] @ [DAN, MBON] -> per MBON
        da = np.clip((gated - DA_MIN_HZ) / (DA_FULL_HZ - DA_MIN_HZ), 0, 1).astype(np.float32)
        self.last_da = da
        w = self.w
        changed = False
        if self.enabled and elig.any() and da.max() > 0.02:
            e = elig[self.syn_kc]
            dw = LEARN_RATE * da[self.syn_mbon] * e * w
            dw = np.minimum(dw, w - FLOOR * self.w0)
            if dw.max() > 1e-7:
                w -= dw
                changed = True
            # reversal: dopamine in one compartment restores the smell's synapses in the opposite one, so a new
            # experience (hurt after being rewarded, or the reverse) overturns the old memory instead of cancelling it
            pun_da = float(da[self.punish_comp].mean()) if self.punish_comp.any() else 0.0
            rew_da = float(da[~self.punish_comp].mean()) if (~self.punish_comp).any() else 0.0
            opp = np.where(self.punish_syn, rew_da, pun_da)
            if opp.max() > 0.02:
                w += REVERSAL_RATE * opp * e * (self.w0 - w)
        k = 1 - 0.5 ** (UPDATE_STEPS * 0.005 / FORGET_HALF_LIFE_S)
        w += (self.w0 - w) * k                                               # slow forgetting
        if changed or k > 0:
            self._write_back()
            self.dirty = self.dirty or changed

    def _write_back(self, full: bool = False) -> None:
        """Copy the plastic weights into the simulator. Only the synapses whose value changed are written and handed
        to the backend (a GPU backend then uploads just those, not the 41 MB matrix: issue #2); full=True writes all
        of them and has the backend re-upload everything (loading or wiping memory)."""
        W = self.sim.W_csr
        if full:
            dirty = np.arange(len(self.w))
        else:
            dirty = np.flatnonzero(W.data[self.csr_pos] != self.w)
            if not len(dirty):
                return
        w = self.w[dirty]
        W.data[self.csr_pos[dirty]] = w
        self.sim.W_csc.data[self.csc_pos[dirty]] = w
        if hasattr(self.sim, "backend") and hasattr(self.sim.backend, "on_weights_changed"):
            self.sim.backend.on_weights_changed(None if full else self.csr_pos[dirty])

    def pattern(self, kc_rates: np.ndarray) -> np.ndarray:
        """KC eligibility in 0..1: firing above calm past the resting noise, sparsified to the most active KCs."""
        excess = (kc_rates - self.kc_calm) / 0.005
        act = np.clip((excess - KC_MIN_HZ) / (KC_FULL_HZ - KC_MIN_HZ), 0, 1).astype(np.float32)
        if np.count_nonzero(act) > KC_SPARSE:
            act[act < np.partition(act, -KC_SPARSE)[-KC_SPARSE]] = 0
        return act

    # --- reading memory -----------------------------------------------------------------------------------------------
    def score(self, pattern: np.ndarray | None) -> tuple[float, float]:
        """(fear, liking) in 0..1 for a KC pattern: how much its input to each compartment's MBONs is weakened."""
        if pattern is None or not pattern.any():
            return 0.0, 0.0
        if np.count_nonzero(pattern) > KC_SPARSE:            # only a smell's core Kenyon cells, like the real code
            pattern = np.where(pattern >= np.partition(pattern, -KC_SPARSE)[-KC_SPARSE], pattern, 0)
        e = pattern[self.syn_kc]
        out = []
        for mask in (self.punish_syn, ~self.punish_syn):
            base = float((self.w0[mask] * e[mask]).sum())
            now = float((self.w[mask] * e[mask]).sum())
            out.append(float(np.clip((1 - now / base) / FULL_DEPRESSION, 0, 1)) if base > 0 else 0.0)
        return out[0], out[1]

    def observe(self, scent: str, rates: np.ndarray) -> None:
        """Keep a running KC pattern for a smell while it is present (what memory is read against)."""
        p = self.pattern(rates[self.kc])
        if not p.any():
            return
        p = p / p.sum()
        t = self.templates.get(scent)
        self.templates[scent] = p if t is None else t + (p - t) * 0.05
        self.dirty = True

    def memory_of(self, scent: str) -> tuple[float, float]:
        return self.score(self.templates.get(scent))

    def mbon_response(self, rates: np.ndarray) -> float:
        """Mean firing (Hz) of the approach-promoting (PPL1-compartment) MBONs: what fear training lowers."""
        return float(rates[self.mbon[self.punish_comp]].mean() / 0.005)

    def record(self, scent: str, kind: str, mbon_hz: float | None = None) -> None:
        fear, like = self.memory_of(scent)
        self.log.setdefault(scent, []).append([time.time(), kind, round(fear, 4), round(like, 4),
                                               None if mbon_hz is None else round(mbon_hz, 3)])
        self.dirty = True

    def weakened_share(self) -> float:
        return float(np.mean(self.w < self.w0 * 0.9))

    # --- persistence ------------------------------------------------------------------------------------------------------
    def save(self) -> None:
        with self.lock:
            tmp = self.path.with_suffix(".tmp.npz")
            names = sorted(self.templates)
            try:
                np.savez_compressed(tmp, version=VERSION, signature=self.signature, w=self.w,
                                    names=np.array(names), templates=np.array([self.templates[n] for n in names]) if names
                                    else np.zeros((0, len(self.kc)), np.float32), kc_calm=self.kc_calm, dan_calm=self.dan_calm)
                tmp.replace(self.path)
                self.log_path.write_text(json.dumps({"naive_mbon": self.naive_mbon, "trials": self.log}, indent=1))
            except OSError as e:
                # 3.1.0 review: a read-only or full data folder raised here at quit, and the game hung instead of closing (rc.2 too)
                logging.getLogger("kickthefly").warning("could not save the fly's memory to %s: %s", self.path, e)
                return
            self.dirty = False

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            z = np.load(self.path)
            if int(z["version"]) != VERSION or not np.allclose(z["signature"], self.signature):
                return                                   # a different brain pack: start fresh
            self.w = z["w"].astype(np.float32)
            self.templates = {str(n): t for n, t in zip(z["names"], z["templates"])}
            self.kc_calm, self.dan_calm = z["kc_calm"], z["dan_calm"]
            self.calm_ready = True
            if self.log_path.exists():
                data = json.loads(self.log_path.read_text())
                self.log, self.naive_mbon = data.get("trials", {}), data.get("naive_mbon", {})
            self._write_back(full=True)
        except Exception:
            self.w = self.w0.copy()                      # unreadable file: start fresh

    def wipe(self) -> None:
        with self.lock:
            self.w = self.w0.copy()
            self.templates, self.log, self.naive_mbon = {}, {}, {}
            self._write_back(full=True)
        self.save()
