"""Tests for real neuron morphology SWC parsing, local caching, and offline fallback."""
from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from kickthefly.sim import morphology
from kickthefly.core.simcore import pack
from kickthefly.game import kick_the_fly as k2


def test_parse_swc():
    sample_swc = """
    # Sample SWC skeleton header
    # id type x y z radius parent_id
    1 1 37000.0 22000.0 36000.0 500.0 -1
    2 2 37100.0 22100.0 36100.0 400.0 1
    3 3 37200.0 22200.0 36200.0 300.0 2
    4 3 37300.0 22300.0 36300.0 200.0 3
    """
    coords = morphology.parse_swc(sample_swc, n_samples=4)
    assert coords is not None
    assert coords.shape == (4, 3)
    assert np.allclose(coords[0], [37000.0, 22000.0, 36000.0])
    assert np.allclose(coords[-1], [37300.0, 22300.0, 36300.0])


def test_cached_skeleton_loading():
    cache_dir = morphology.default_cache_dir()
    # Check if cached files exist from our earlier fetch
    p = cache_dir / "10001.swc"
    if p.exists():
        coords = morphology.fetch_or_load_skeleton(10001, cache_dir=cache_dir, allow_network=False)
        assert coords is not None
        assert coords.shape == (21, 3)
        assert coords.dtype == np.float32


def test_offline_fallback():
    # Non-existent body ID with allow_network=False should return None gracefully
    coords = morphology.fetch_or_load_skeleton(999999999, allow_network=False)
    assert coords is None


def test_brainview_with_skeletons():
    g, W, soma = pack()
    pain_mask = np.zeros(g.n, bool)
    view = k2.BrainView(soma, W, pain_mask, graph=g)
    cached = [bid for bid in g.body_id[np.char.startswith(g.type.astype(str), "DNp01")][:2]
              if (morphology.default_cache_dir() / f"{int(bid)}.swc").exists()]
    if cached:          # a cached skeleton must really be drawn (a signature mismatch once hid this behind the fallback)
        assert view.skeleton_status.startswith("Real morphology"), view.skeleton_status
    else:
        # since the 3.0 day 3 follow-up the download is opt-in, and the view says so; a machine with no cached skeleton (a fresh clone) sees this text
        assert "Skeletons offline" in view.skeleton_status or "real shapes are off" in view.skeleton_status, view.skeleton_status


def test_load_key_skeletons_takes_the_view_sample_count(tmp_path):
    g, _, _ = pack()
    swc = "\n".join(f"{i} 1 {i}.0 {2 * i}.0 {3 * i}.0 1.0 {i - 1 if i > 1 else -1}" for i in range(1, 200))
    bid = int(g.body_id[np.flatnonzero(g.type.astype(str) == "DNp01")[0]])
    (tmp_path / f"{bid}.swc").write_text(swc)
    skels, status = morphology.load_key_skeletons(g, cache_dir=tmp_path, allow_network=False,
                                                   n_samples=k2.BrainView.FIBER + k2.BrainView.ARBOR)
    assert status.startswith("Real morphology")
    assert all(c.shape == (k2.BrainView.FIBER + k2.BrainView.ARBOR, 3) for c in skels.values())
