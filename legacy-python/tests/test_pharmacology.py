"""3.0 day 2: pharmacology (MODEL PREDICTION): drugs as synaptic scaling by predicted transmitter. Weight arithmetic and plumbing on
the synthetic pack; equivalence with the original inhibition block; compatibility of the Wiring format."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from kickthefly.lab import pharmacology as ph
from kickthefly.sim import wiring as W
from kickthefly.sim.wiring import Wiring


@pytest.fixture
def g(synthetic_pack):
    """The synthetic pack's graph, with the wiring module's per-process caches cleared before and after."""
    from kickthefly.core import simcore

    for f in (W.synapse_counts, W.edge_pre, W.edge_post):
        f.cache_clear()
    graph = simcore.pack()[0]
    yield graph
    for f in (W.synapse_counts, W.edge_pre, W.edge_post):
        f.cache_clear()


# --- the rules of the drugs --------------------------------------------------------------------------------------------------------------
def test_dose_to_scale_mapping():
    assert ph.scale_for("picrotoxin", 0.0) == 1.0 and ph.scale_for("picrotoxin", 1.0) == 0.0 and ph.scale_for("cholinergic", 0.25) == 0.75
    assert ph.scale_for("gabaa_agonist", 0.0) == 1.0 and ph.scale_for("gabaa_agonist", 1.0) == ph.AGONIST_MAX == 2.0
    assert ph.scale_for("glucl", 0.5) == 0.5
    with pytest.raises(ph.PharmError, match="between 0 and 1"):
        ph.scale_for("picrotoxin", 1.5)


def test_the_drug_list_and_its_targets():
    assert set(ph.DRUGS) == {"picrotoxin", "cholinergic", "glucl", "gabaa_agonist"}
    assert ph.DRUGS["picrotoxin"].targets == ("gaba", "glutamate") == tuple(W.INHIBITORY)
    assert ph.DRUGS["cholinergic"].targets == ("acetylcholine",) and ph.DRUGS["glucl"].targets == ("glutamate",)
    assert ph.DRUGS["gabaa_agonist"].targets == ("gaba",) and ph.DRUGS["gabaa_agonist"].kind == "agonist"
    assert ph.drug("PTX").key == "picrotoxin" and ph.drug("cholinergic block".replace(" block", "_block")).key == "cholinergic"


def test_octopamine_and_dopamine_are_refused_not_invented():
    for name in ("octopamine", "dopamine", "serotonin"):
        with pytest.raises(ph.PharmError, match="unknown drug"):
            ph.drug(name)
    assert "octopamine and dopamine modulation" in ph.__doc__.lower().replace("\n", " ")
    # the reason is checkable: those neurons' synapses carry no sign, so they are not in the simulated matrix at all
    assert "octopamine" not in W.SIGN_OF and "dopamine" not in W.SIGN_OF


def test_the_tag_says_model_prediction():
    assert ph.TAG_TEXT.startswith("MODEL PREDICTION")


def test_wiring_for_merges_drugs_on_the_same_transmitter():
    w = ph.wiring_for({"picrotoxin": 0.5, "glucl": 0.5})
    assert dict(w.nt_scales) == {"gaba": 0.5, "glutamate": 0.25}
    assert ph.wiring_for({"picrotoxin": 0.0}).is_identity
    assert ph.wiring_for({}).is_identity
    ex = ph.wiring_for({"cholinergic": 0.4}, include_low_confidence=False, cut=0.8)
    assert ex.nt_min_conf == 0.8 and ph.wiring_for({"cholinergic": 0.4}).nt_min_conf == 0.0
    base = Wiring(min_synapses=4, flip_rows=(3, 5), inhibition_scale=0.9)
    w2 = ph.wiring_for({"cholinergic": 0.4}, base=base)
    assert (w2.min_synapses, w2.flip_rows, w2.inhibition_scale) == (4, (3, 5), 0.9)
    with pytest.raises(ph.PharmError):
        ph.wiring_for({"picrotoxin": 0.5}, cut=0.0)


# --- compatibility of the Wiring format --------------------------------------------------------------------------------------------------
def test_default_wiring_dict_is_unchanged_so_validation_output_is_identical():
    assert Wiring().as_dict() == dict(min_synapses=1, flipped_neurons=0, inhibition_scale=1.0, label="unmodified connectome")
    assert Wiring(inhibition_scale=0.5).as_dict() == dict(min_synapses=1, flipped_neurons=0, inhibition_scale=0.5,
                                                          label="inhibition x0.5")


def test_old_wiring_dicts_still_load_and_new_ones_roundtrip():
    old = {"min_synapses": 5, "flipped_neurons": 2, "inhibition_scale": 0.3, "label": "x"}
    w = Wiring.from_dict(old, rows=(1, 2))
    assert w == Wiring(5, (1, 2), 0.3) and w.nt_scales == () and w.nt_min_conf == 0.0
    new = Wiring(nt_scales=(("acetylcholine", 0.5), ("gaba", 1.5)), nt_min_conf=0.7)
    assert Wiring.from_dict(new.as_dict()) == new
    assert "acetylcholine synapses x0.5" in new.label() and "confidence >= 0.7" in new.label()
    assert Wiring(nt_scales=(("gaba", 1.0),)).is_identity                               # a factor of 1 changes nothing
    assert new.with_rows([4]).nt_scales == new.nt_scales


# --- the arithmetic on the synapse matrix ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("dose", [0.25, 0.5, 1.0])
def test_picrotoxin_equals_the_original_inhibition_block_exactly(g, dose):
    """The drug must be the same change as Wiring.inhibition_scale (same synapses, same multipliers)."""
    i_old, m_old = W.modified_entries(g, Wiring(inhibition_scale=1.0 - dose))
    i_new, m_new = W.modified_entries(g, ph.wiring_for({"picrotoxin": dose}))
    assert np.array_equal(i_old, i_new) and np.allclose(m_old, m_new)


def test_each_drug_scales_exactly_the_synapses_of_its_transmitters(g):
    nt, _, _ = W.transmitters(g)
    pre = W.edge_pre()
    for name, targets in (("cholinergic", ("acetylcholine",)), ("glucl", ("glutamate",)), ("gabaa_agonist", ("gaba",))):
        w = ph.wiring_for({name: 0.5})
        idx, mult = W.modified_entries(g, w)
        assert set(nt[pre[idx]]) == set(targets)
        assert len(idx) == int(np.isin(nt[pre], targets).sum())
        assert np.allclose(mult, ph.scale_for(name, 0.5))


def test_excluding_low_confidence_leaves_those_neurons_alone(g):
    nt, conf, source = W.transmitters(g)
    pre = W.edge_pre()
    cut = 0.85
    idx_all, _ = W.modified_entries(g, ph.wiring_for({"cholinergic": 1.0}, True))
    idx_hi, _ = W.modified_entries(g, ph.wiring_for({"cholinergic": 1.0}, False, cut))
    assert len(idx_hi) < len(idx_all) and set(idx_hi) <= set(idx_all)
    kept = conf[pre[idx_hi]]
    assert np.all((kept >= cut) | (source[pre[idx_hi]] == "ground_truth"))
    dropped = np.setdiff1d(idx_all, idx_hi)
    assert np.all(conf[pre[dropped]] < cut)


def test_measured_transmitters_always_count_when_low_confidence_is_excluded():
    nt = np.array(["acetylcholine"] * 4)
    conf = np.array([0.2, 0.2, 0.95, np.nan], np.float32)
    src = np.array(["ground_truth", "predicted_nt", "predicted_nt", "predicted_nt"])
    g = SimpleNamespace(nt=nt, nt_conf=conf, nt_source=src)
    assert list(W.confidence_ok(g, 0.7)) == [True, False, True, False]                 # measured, low, high, no confidence
    assert W.confidence_ok(g, 0.0).all()


def test_levels_partition_the_neurons():
    conf = np.array([0.3, 0.5, 0.69, 0.7, 0.89, 0.9, 1.0, np.nan, 0.1], np.float32)
    src = np.array(["predicted_nt"] * 8 + ["ground_truth"])
    lv = ph._level(conf, src)
    assert list(lv) == [4, 3, 3, 2, 2, 1, 1, 5, 0]
    assert len(ph.LEVELS) == 6


def test_affected_counts_partition_and_follow_the_setting(g):
    a = ph.affected(g, "cholinergic", 0.5, include_low_confidence=True)
    assert sum(r["connections"] for r in a["rows"]) == a["connections"] and a["connections_applied"] == a["connections"]
    nt, _, _ = W.transmitters(g)
    pre = W.edge_pre()
    assert a["connections"] == int((nt[pre] == "acetylcholine").sum())
    assert a["synapses"] == int(W.synapse_counts()[nt[pre] == "acetylcholine"].sum())
    b = ph.affected(g, "cholinergic", 0.5, include_low_confidence=False, cut=0.9)
    assert b["connections_applied"] < b["connections"] == a["connections"]
    idx, _ = W.modified_entries(g, ph.wiring_for({"cholinergic": 0.5}, False, 0.9))
    assert b["connections_applied"] == len(idx)                                         # the panel counts what the wiring will change
    assert b["synapses_applied"] == int(W.synapse_counts()[idx].sum())
    assert b["tag"].startswith("MODEL PREDICTION") and b["scale"] == 0.5
    assert {r["level"] for r in b["rows"]} == set(ph.LEVELS)


# --- on a live brain ---------------------------------------------------------------------------------------------------------------------------
def test_a_drug_changes_the_brains_weights_and_washout_restores_them_exactly(synthetic_pack):
    from kickthefly.core import simcore

    for f in (W.synapse_counts, W.edge_pre, W.edge_post):
        f.cache_clear()
    br = simcore.new_brain(seed=2, warmup=30)
    before = br.sim.W_csr.data.copy()
    info = W.apply(br, ph.wiring_for({"cholinergic": 1.0}))
    assert info["changed"] > 0 and not np.array_equal(br.sim.W_csr.data, before)
    W.apply(br, Wiring())
    assert np.array_equal(br.sim.W_csr.data, before)
    W.apply(br, ph.wiring_for({"picrotoxin": 0.5, "gabaa_agonist": 0.5}))
    W.clear(br)
    assert np.array_equal(br.sim.W_csr.data, before)
    for f in (W.synapse_counts, W.edge_pre, W.edge_post):
        f.cache_clear()


def test_a_drug_works_together_with_the_other_wiring_changes(g):
    both = Wiring(min_synapses=6, nt_scales=(("acetylcholine", 0.5),))
    idx, mult = W.modified_entries(g, both)
    i_thr, m_thr = W.modified_entries(g, Wiring(min_synapses=6))
    assert set(i_thr) <= set(idx) and len(idx) > len(i_thr)
    assert np.all(mult[np.isin(idx, i_thr)] == 0.0)                                     # a dropped synapse stays dropped


def test_a_protocol_can_carry_a_drug(synthetic_pack, tmp_path):
    from kickthefly.lab import protocol

    def run(drug):
        p = dict(name="d", seed=0, flies=1, warmup_s=0.5, duration_s=1.0, stimuli=[], recordings=[dict(name="all", neurons="superclass:cb_intrinsic")])
        if drug:
            p["drug"] = drug
        folder = tmp_path / f"r{bool(drug)}"
        folder.mkdir()
        return protocol.run_seed(protocol.check(p), 0, None, folder, "run")["all"]

    for f in (W.synapse_counts, W.edge_pre, W.edge_post):
        f.cache_clear()
    base = run(None)
    blocked = run(dict(doses=dict(cholinergic=1.0)))
    assert blocked != base                                                              # the drug reached the brain the protocol ran
    with pytest.raises(protocol.ProtocolError, match="unknown drug"):
        protocol.check(dict(name="d", drug=dict(doses=dict(octopamine=0.5))))
    with pytest.raises(protocol.ProtocolError, match="drug needs doses"):
        protocol.check(dict(name="d", drug=dict(dose=0.5)))
    with pytest.raises(protocol.ProtocolError, match="stimulus protocols only"):
        protocol.check(dict(name="d", assay="sugar", drug=dict(doses=dict(picrotoxin=0.5))))
    for f in (W.synapse_counts, W.edge_pre, W.edge_post):
        f.cache_clear()


from conftest import needs_pack  # noqa: E402


@needs_pack
def test_real_pack_picrotoxin_equals_the_original_block_and_counts_are_the_datasets():
    """On the real MaleCNS pack: the same synapses and multipliers as the original inhibition block, and the confidence table adds up."""
    from kickthefly.core import simcore

    for f in (W.synapse_counts, W.edge_pre, W.edge_post):
        f.cache_clear()
    g = simcore.pack()[0]
    i_old, m_old = W.modified_entries(g, Wiring(inhibition_scale=0.5))
    i_new, m_new = W.modified_entries(g, ph.wiring_for({"picrotoxin": 0.5}))
    assert np.array_equal(i_old, i_new) and np.allclose(m_old, m_new)
    a = ph.affected(g, "cholinergic", 1.0, True)
    nt, _, source = W.transmitters(g)
    assert a["rows"][0]["neurons"] == int(((nt == "acetylcholine") & (source == "ground_truth")).sum())
    assert sum(r["neurons"] for r in a["rows"]) == int((nt == "acetylcholine").sum())
    assert ph.affected(g, "cholinergic", 1.0, False, 0.9)["synapses_applied"] < a["synapses"]
    for f in (W.synapse_counts, W.edge_pre, W.edge_post):
        f.cache_clear()
