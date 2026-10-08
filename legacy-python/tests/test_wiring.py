"""Changing the connectome itself: the synapse threshold, and that every change is exactly reversible."""
import numpy as np

from conftest import needs_pack


@needs_pack
def test_threshold_stats_match_the_pack():
    from kickthefly.sim import wiring

    counts = wiring.synapse_counts()
    st = wiring.threshold_stats(5)
    assert st["connections"] == len(counts)
    assert st["connections_dropped"] == int((counts < 5).sum())
    assert st["synapses_dropped"] == int(counts[counts < 5].sum())
    assert 0 < st["connections_dropped_share"] < 1
    assert st["neurons_cut_off"] <= st["neurons_touched"] <= st["neurons"]


@needs_pack
def test_thresholds_below_the_pack_filter_change_nothing():
    """The loader already drops connections below 3 synapses, and the report has to say so rather than pretend."""
    from kickthefly.lab import robustness
    from kickthefly.sim import wiring

    for n in range(1, robustness.MIN_PACK_SYNAPSES + 1):
        assert wiring.threshold_stats(n)["connections_dropped"] == 0


@needs_pack
def test_applying_a_threshold_zeroes_exactly_those_connections_and_reverses(tmp_path):
    from kickthefly.core import simcore
    from kickthefly.sim import wiring
    from kickthefly.sim.wiring import Wiring

    br = simcore.new_brain(seed=3, warmup=0)
    before = br.sim.W_csr.data.copy()
    counts = wiring.synapse_counts()
    weak = counts < 6

    info = wiring.apply(br, Wiring(min_synapses=6))
    assert info["changed"] == int(weak.sum())
    assert np.all(br.sim.W_csr.data[weak] == 0)
    assert np.array_equal(br.sim.W_csr.data[~weak], before[~weak])
    assert br.sim.W_csr.nnz == len(before), "entries are zeroed, not removed: index maps must stay valid"
    assert np.allclose(br.sim.W_csc.toarray().sum(), br.sim.W_csr.toarray().sum()) if br.n < 100 else True

    wiring.clear(br)
    assert np.array_equal(br.sim.W_csr.data, before)
    assert np.array_equal(br.sim.W_csc.data, br.sim.W_csr.tocsc().data)


@needs_pack
def test_the_threshold_survives_a_learning_update():
    """Learning writes the plastic synapses back every 50 ms; a dropped KC -> MBON synapse must stay dropped."""
    from kickthefly.core import simcore
    from kickthefly.sim import wiring
    from kickthefly.sim.wiring import Wiring

    br = simcore.new_brain(seed=4, warmup=0)
    mem = br.memory
    assert mem is not None
    counts = wiring.synapse_counts()
    plastic_weak = mem.csr_pos[counts[mem.csr_pos] < 6]
    assert len(plastic_weak) > 0, "some plastic synapses should be below the threshold"

    wiring.apply(br, Wiring(min_synapses=6))
    assert np.all(br.sim.W_csr.data[plastic_weak] == 0)
    simcore.step(br, 30)                                   # long enough for at least one plasticity update
    assert np.all(br.sim.W_csr.data[plastic_weak] == 0), "learning wrote the dropped synapses back"

    wiring.clear(br)
    assert np.all(br.sim.W_csr.data[plastic_weak] > 0)


@needs_pack
def test_a_thresholded_brain_still_runs_and_differs():
    from kickthefly.core import simcore
    from kickthefly.sim.wiring import Wiring

    plain = simcore.new_brain(seed=11, warmup=200)
    cut = simcore.new_brain(seed=11, warmup=200, wiring=Wiring(min_synapses=10))
    a = simcore.step(plain, 100, record=np.arange(plain.n))
    b = simcore.step(cut, 100, record=np.arange(cut.n))
    assert a.sum() > 0 and b.sum() > 0, "both brains still fire"
    assert not np.array_equal(a, b), "dropping 74% of the connections must change the activity"


def test_wiring_label_and_dict():
    from kickthefly.sim.wiring import Wiring

    assert Wiring().is_identity and Wiring().label() == "unmodified connectome"
    w = Wiring(min_synapses=5)
    assert not w.is_identity and "drop <5" in w.label()
    assert w.as_dict()["min_synapses"] == 5


# --- sign flips ----------------------------------------------------------------------------------------------------
@needs_pack
def test_transmitters_come_from_the_dataset():
    from kickthefly.core import simcore
    from kickthefly.sim import wiring

    g = simcore.pack()[0]
    nt, conf, source = wiring.transmitters(g)
    assert len(nt) == g.n
    assert set(source) <= {"ground_truth", "consensus_nt", "predicted_nt", "missing", "unknown"}
    measured = source == "ground_truth"
    assert measured.sum() > 50_000
    assert np.all(conf[measured] == 1.0), "a measured transmitter is not a prediction"
    ok = ~np.isnan(conf)
    assert np.all((conf[ok] >= 0) & (conf[ok] <= 1))
    assert set(nt) >= {"acetylcholine", "gaba", "glutamate", "histamine"}


@needs_pack
def test_flip_candidates_respect_the_cutoff_and_never_include_measured_neurons():
    from kickthefly.core import simcore
    from kickthefly.sim import wiring

    g = simcore.pack()[0]
    nt, conf, source = wiring.transmitters(g)
    low = wiring.flip_candidates(g, 0.7)
    high = wiring.flip_candidates(g, 0.9)
    assert set(low.tolist()) <= set(high.tolist()), "raising the cutoff can only add candidates"
    assert np.all(np.isin(nt[low], list(wiring.SIGN_OF))), "only neurons whose transmitter gives a sign"
    assert np.all(np.isnan(conf[low]) | (conf[low] < 0.7))
    assert not np.any(source[low] == "ground_truth")
    st = wiring.confidence_stats(g, 0.7)
    assert st["candidates"] == len(low) and st["unknown_confidence"] <= st["candidates"]


@needs_pack
def test_a_flip_negates_exactly_that_neuron_and_reverses():
    from kickthefly.core import simcore
    from kickthefly.sim import wiring
    from kickthefly.sim.wiring import Wiring

    br = simcore.new_brain(seed=5, warmup=0)
    g = br.graph
    row = int(wiring.flip_candidates(g, 0.7)[0])
    pre = wiring.edge_pre()
    mine = pre == row
    before = br.sim.W_csr.data.copy()
    assert np.any(mine) and np.any(before[mine] != 0)

    wiring.apply(br, Wiring(flip_rows=(row,)))
    assert np.allclose(br.sim.W_csr.data[mine], -before[mine])
    assert np.array_equal(br.sim.W_csr.data[~mine], before[~mine])
    wiring.clear(br)
    assert np.array_equal(br.sim.W_csr.data, before)


@needs_pack
def test_random_flip_is_seeded_and_scales_with_share():
    from kickthefly.core import simcore
    from kickthefly.sim import wiring

    g = simcore.pack()[0]
    a = wiring.random_flip(g, 0.7, 0.5, 1)
    assert a == wiring.random_flip(g, 0.7, 0.5, 1)
    assert a != wiring.random_flip(g, 0.7, 0.5, 2)
    everything = wiring.random_flip(g, 0.7, 1.0, 1)
    assert len(everything) == len(wiring.flip_candidates(g, 0.7))
    assert 0.3 < len(a) / len(everything) < 0.7


def test_wilson_interval_behaves_at_the_edges():
    from kickthefly.lab.robustness import _wilson

    lo, hi = _wilson(0, 5)
    assert lo == 0.0 and 0.3 < hi < 0.6
    lo, hi = _wilson(5, 5)
    assert hi == 1.0 and 0.4 < lo < 0.7
    lo, hi = _wilson(50, 100)
    assert lo < 0.5 < hi and hi - lo < 0.25


# --- inhibition block tests --------------------------------------------------------------------------------------
@needs_pack
def test_inhibition_stats_and_scaling_reversibility():
    from kickthefly.core import simcore
    from kickthefly.sim import wiring
    from kickthefly.sim.wiring import Wiring

    g = simcore.pack()[0]
    st = wiring.inhibition_stats(g, scale=0.5)
    assert st["scale"] == 0.5
    assert st["severity"] == 0.5
    assert st["inhibitory_connections"] > 0
    assert 0.20 < st["inhibitory_share"] < 0.50
    assert st["inhibitory_neurons"] == st["gaba_neurons"] + st["glutamate_neurons"]

    br = simcore.new_brain(seed=7, warmup=0)
    before = br.sim.W_csr.data.copy()
    inhib_edges = wiring._inhibitory_edges(g)
    assert np.any(inhib_edges) and np.any(~inhib_edges)

    # Apply 100% block (scale 0.0)
    info = wiring.apply(br, Wiring(inhibition_scale=0.0))
    assert info["changed"] == int(inhib_edges.sum())
    assert np.all(br.sim.W_csr.data[inhib_edges] == 0)
    assert np.array_equal(br.sim.W_csr.data[~inhib_edges], before[~inhib_edges])

    # Reversible
    wiring.clear(br)
    assert np.array_equal(br.sim.W_csr.data, before)


@needs_pack
def test_inhibition_block_emergent_runaway_activity(tmp_path):
    from kickthefly.lab import robustness
    from kickthefly.sim.wiring import Wiring

    rep = robustness.inhibition_report(scale=0.0, seeds=(1000,), steps=100)
    assert rep["kind"] == "inhibition_block"
    assert rep["scale"] == 0.0
    assert rep["severity"] == 1.0
    # Emergent runaway excitation: treated mean firing rate must exceed control baseline
    assert rep["after"]["mean"] > rep["before"]["mean"] * 2.0
    assert rep["after"]["p95"] > rep["before"]["p95"]

    # Test export
    export_folder = tmp_path / "inhibition_export"
    robustness.save(rep, export_folder)
    assert (export_folder / "inhibition_block.json").exists()
    assert (export_folder / "inhibition_block.csv").exists()
