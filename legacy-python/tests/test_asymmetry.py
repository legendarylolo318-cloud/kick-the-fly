"""Tests for bilateral asymmetry audit and weight mirror-averaging."""

import numpy as np
import pytest
from kickthefly.lab import headless
from kickthefly.core import simcore


def test_symmetrize_weights_structure():
    g, W, _ = simcore.pack()
    W_sym = simcore.symmetrize_weights(g, W)
    assert W_sym.shape == W.shape
    assert W_sym.dtype == np.float32

    inst = g.instance.astype(str)
    types = g.type.astype(str)

    # Verify that bilateral pairs have symmetric synaptic totals
    W_csr = W_sym.tocsr()
    W_csc = W_sym.tocsc()
    for t in ["DNa01", "DNa02", "DNp01"]:
        l_idx = np.flatnonzero((types == t) & np.char.endswith(inst, "_L"))[0]
        r_idx = np.flatnonzero((types == t) & np.char.endswith(inst, "_R"))[0]

        # In-degree nnz and out-degree nnz in symmetrized weights must be equal
        assert W_csr[l_idx].nnz == W_csr[r_idx].nnz
        assert W_csc[:, l_idx].nnz == W_csc[:, r_idx].nnz


def test_audit_asymmetry_fast():
    # Run a short 0.05s audit for speed in test suite
    res = headless.audit_asymmetry(seconds=0.05, seed=42, mirror=False)
    assert "records" in res
    assert "turning_bias_hz" in res
    assert "turning_direction" in res
    assert len(res["records"]) == 6

    types = [r["type"] for r in res["records"]]
    for expected in ["DNa01", "DNa02", "LC10", "LPLC2", "LC4", "DNp01"]:
        assert expected in types

    report = headless.format_asymmetry_report(res)
    assert "LEFT/RIGHT ASYMMETRY AUDIT" in report
    assert "Raw connectome weights" in report


def test_audit_asymmetry_mirrored():
    res_m = headless.audit_asymmetry(seconds=0.05, seed=42, mirror=True)
    report_m = headless.format_asymmetry_report(res_m)
    assert "GAME RULE: data modification" in report_m

    for r in res_m["records"]:
        if r["type"] in ["DNa01", "DNa02", "DNp01"]:
            assert r["l_in_syn"] == r["r_in_syn"]
            assert r["l_out_syn"] == r["r_out_syn"]
