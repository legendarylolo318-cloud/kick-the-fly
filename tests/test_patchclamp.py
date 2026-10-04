"""3.0 day 2: the virtual patch clamp. A point-neuron model: these tests check the arithmetic against the simulator's own update
and the plumbing (steps, I-F, exports). They say nothing about real cells."""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from kickthefly.lab import patchclamp as pc
from kickthefly.sim.connectome.sim import LIFParams, LIFSim


def test_the_model_says_what_it_is():
    assert "NOT real electrophysiology" in pc.__doc__ or "NOT ELECTROPHYSIOLOGY" in pc.__doc__
    assert "MODEL" in pc.TAG_TEXT and "Not millivolts" in pc.UNITS_NOTE


# --- the protocol -----------------------------------------------------------------------------------------------------------------------
def test_protocol_schedule_and_windows():
    p = pc.ClampProtocol([0.0, 0.1], duration_ms=100, pre_ms=50, post_ms=50, repeats=2, gap_ms=0, holding=0.01)
    assert p.sweeps() == [0.0, 0.0, 0.1, 0.1]
    sched = p.schedule()
    assert len(sched) == p.total_steps() == 4 * 40                   # 200 ms per sweep at 5 ms
    wins = p.sweep_windows()
    assert wins[2] == (0.1, 2 * 40 + 10, 2 * 40 + 30)
    assert np.allclose(sched[wins[2][1]:wins[2][2]], 0.1) and np.allclose(sched[:10], 0.01)


@pytest.mark.parametrize("kw,msg", [(dict(amplitudes=[]), "amplitudes"), (dict(amplitudes=[9.0]), "limited"),
                                    (dict(duration_ms=1), "duration_ms"), (dict(repeats=0), "repeats"),
                                    (dict(holding=7.0), "limited")])
def test_protocol_refuses_nonsense(kw, msg):
    with pytest.raises(pc.PatchError, match=msg):
        pc.ClampProtocol(**kw)


# --- the isolated unit is the simulator's own LIF update ------------------------------------------------------------------------------
def _sim_one_neuron(cur, params):
    sim = LIFSim(None, params, W_in=sp.csr_array((1, 1), dtype=np.float32), seed=0)
    vs, sp_ = [], []
    for c in cur:
        sim.step(np.array([c], np.float32) if c else None)
        vs.append(float(sim.v[0]))
        sp_.append(bool(sim.spikes[0]))
    return np.array(vs), np.array(sp_)


def test_isolated_unit_matches_the_reference_backend_exactly_without_noise():
    p = LIFParams(noise_std=0.0, backend="cpu")
    cur = np.concatenate([np.zeros(40), np.full(120, 0.03), np.zeros(40), np.full(100, 0.4), np.zeros(40)]).astype(np.float32)
    ref_v, ref_s = _sim_one_neuron(np.concatenate([np.zeros(400), cur]), p)          # 400 settle steps, like isolated()
    rec = pc.isolated(cur, p, seed=0, settle_steps=400)
    assert np.array_equal(rec.spikes, ref_s[400:])
    assert np.allclose(rec.v, ref_v[400:], atol=1e-6)


def test_rheobase_follows_from_the_lif_equation_not_from_tuning():
    """Without noise the unit rests at bias/leak = 0.8 and spikes iff bias + ext_gain * I >= leak * threshold, i.e. I >= 0.0125."""
    p = LIFParams(noise_std=0.0)
    leak = p.dt_ms / p.tau_ms
    rheo = (leak * p.v_thresh - p.bias) / p.ext_gain
    assert rheo == pytest.approx(0.0125)
    below = pc.isolated(np.full(200, rheo * 0.9, np.float32), p).spikes.sum()
    above = pc.isolated(np.full(200, rheo * 1.5, np.float32), p).spikes.sum()
    assert below == 0 and above > 0
    rest = pc.isolated(np.zeros(100, np.float32), p).v
    assert rest[-1] == pytest.approx(p.bias / leak, abs=1e-4) and rest[-1] < p.v_thresh


def test_isolated_if_curve_is_monotone_and_saturates_at_the_refractory_limit():
    p = LIFParams(noise_std=0.0)
    c = pc.if_curve(None, None, [0.0, 0.01, 0.02, 0.05, 0.2, 1.0, 4.0], 500.0, 1, "isolated", 0, p)
    r = c["rate_hz"]
    assert all(b >= a for a, b in zip(r, r[1:])) and r[0] == 0.0
    assert r[-1] == pytest.approx(1000.0 / (5.0 * (p.refractory_steps + 1)), abs=2.0)        # one spike per refractory cycle (+-1 spike at the window edge)
    assert c["rheobase"] == 0.02 and "MODEL" in c["tag"]


def test_every_isolated_unit_is_the_same_unit():
    """The honest point of the page: in isolation all neurons share one curve (only noise differs)."""
    p = LIFParams(noise_std=0.0)
    a = pc.if_curve(None, 1, [0.0, 0.05, 0.2], 300.0, 1, "isolated", 1, p)["rate_hz"]
    b = pc.if_curve(None, 99999, [0.0, 0.05, 0.2], 300.0, 1, "isolated", 2, p)["rate_hz"]
    assert a == b


def test_isolated_is_seeded():
    cur = np.full(200, 0.0125, np.float32)
    a, b, c = pc.isolated(cur, seed=3), pc.isolated(cur, seed=3), pc.isolated(cur, seed=4)
    assert np.array_equal(a.v, b.v) and not np.array_equal(a.v, c.v)


# --- embedded in a wired brain ----------------------------------------------------------------------------------------------------------
@pytest.fixture
def br(synthetic_pack):
    from kickthefly.core import simcore

    return simcore.new_brain(seed=4, warmup=100)


def test_pick_neuron_by_type_index_and_line(br):
    r = pc.pick_neuron(br, "type:DNp01", 1)
    assert br.types[r] == "DNp01" and r == int(np.sort(np.flatnonzero(br.types == "DNp01"))[1])
    assert pc.pick_neuron(br, "line:SS00727", 0) == int(np.flatnonzero(br.types == "DNp01")[0])
    with pytest.raises(pc.PatchError, match="out of range"):
        pc.pick_neuron(br, "type:DNp01", 5)
    with pytest.raises(pc.PatchError, match="unknown neuron spec|selects no"):
        pc.pick_neuron(br, "type:Nope", 0)


def test_embedded_current_changes_firing_both_ways_and_is_removed(br):
    row = pc.pick_neuron(br, "type:MDN", 0)
    proto = pc.ClampProtocol([0.0, -0.3, 0.0, 0.3], duration_ms=500, repeats=1, pre_ms=0, post_ms=0, gap_ms=0)
    rec = pc.run_current_clamp(br, row, proto, "embedded")
    w = proto.sweep_windows()
    n = [int(rec.spikes[lo:hi].sum()) for _, lo, hi in w]
    assert n[1] < min(n[0], n[2]) - 10                                   # hyperpolarizing current silences it (it fires a lot at rest here)
    assert n[3] >= max(n[0], n[2])                                       # depolarizing current never lowers it
    assert not br.injecting and "patch" not in br.currents              # the electrode leaves nothing behind
    assert rec.meta["neuron"]["type"] == "MDN" and rec.meta["protocol"]["amplitudes"] == [0.0, -0.3, 0.0, 0.3]
    assert rec.v.shape == rec.spikes.shape == rec.current.shape


def test_embedded_is_deterministic(synthetic_pack):
    from kickthefly.core import simcore

    def run():
        b = simcore.new_brain(seed=5, warmup=60)
        return pc.run_current_clamp(b, pc.pick_neuron(b, "type:DNp01"), pc.ClampProtocol([0.1], duration_ms=200))

    a, b = run(), run()
    assert np.array_equal(a.v, b.v) and np.array_equal(a.spikes, b.spikes)


def test_the_electrode_only_reads_when_no_current_is_set(synthetic_pack):
    from kickthefly.core import simcore

    a, b = simcore.new_brain(seed=6, warmup=50), simcore.new_brain(seed=6, warmup=50)
    ele = pc.LiveElectrode(b, 0, 100).attach()
    for _ in range(200):
        a._step()
        b._step()
    assert np.array_equal(a.sim.v, b.sim.v) and ele.count == 200
    v, sp_, cur = ele.window()
    assert len(v) == 100 and v[-1] == b.sim.v[0]
    ele.detach()
    assert b.probe is None


def test_live_electrode_plays_a_protocol_and_keeps_the_recording(synthetic_pack):
    from kickthefly.core import simcore

    b = simcore.new_brain(seed=7, warmup=50)
    row = pc.pick_neuron(b, "type:MDN", 0)
    ele = pc.LiveElectrode(b, row, 200).attach()
    proto = pc.ClampProtocol([0.3], duration_ms=200, pre_ms=50, post_ms=50, gap_ms=0)
    ele.run(proto)
    for _ in range(proto.total_steps() + 5):
        b._step()
    assert ele.done and len(ele.recording.v) == proto.total_steps()
    assert ele.recording.spikes.sum() > 3 and not b.injecting
    ele.detach()


# --- exports ------------------------------------------------------------------------------------------------------------------------------
def test_csv_carries_the_tag_units_and_every_step(tmp_path):
    rec = pc.run_current_clamp(None, 3, pc.ClampProtocol([0.1], duration_ms=50, pre_ms=10, post_ms=10, gap_ms=0), "isolated", 1)
    f = pc.export_csv(rec, tmp_path / "x" / "t.csv")
    lines = f.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("# MODEL") and "Not millivolts" in lines[1] and lines[3].startswith("time_ms,membrane_potential_model_units")
    assert len(lines) == 4 + len(rec.v)
    curve = pc.if_curve(None, 3, [0.0, 0.1], 100.0, 2, "isolated", 1)
    g = pc.export_if_csv(curve, tmp_path / "if.csv")
    assert g.read_text(encoding="utf-8").count("\n") == 2 + 1 + 2


def test_nwb_keeps_model_units_and_spike_times(tmp_path):
    pynwb = pytest.importorskip("pynwb")
    rec = pc.run_current_clamp(None, 3, pc.ClampProtocol([0.3], duration_ms=200, pre_ms=20, post_ms=20, gap_ms=0), "isolated", 1)
    f = pc.export_nwb(rec, tmp_path / "p.nwb")
    with pynwb.NWBHDF5IO(str(f), "r") as io:
        nwb = io.read()
        ts = nwb.acquisition["membrane_potential_model_units"]
        assert ts.unit == "a.u." and len(ts.data) == len(rec.v) and ts.rate == 200.0
        assert "NOT" in nwb.session_description.upper() or "not" in nwb.session_description
        assert not any(type(x).__name__ == "CurrentClampSeries" for x in nwb.acquisition.values())   # not volts, so not a volts series
        assert np.allclose(nwb.units["spike_times"][0], rec.spike_times_ms() / 1000.0)
        assert nwb.stimulus["injected_current_model_units"].unit == "a.u."
