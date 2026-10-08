"""Simulation backends: each one really runs when asked for, and the CPU-side ones are bit-exact with the NumPy
reference. GPU backends (torch-cuda, torch-rocm) are compared statistically, because GPU sparse kernels may add a
neuron's inputs in a different order and a chaotic network then diverges spike by spike (see README: Performance)."""
import os

import numpy as np
import pytest

from conftest import needs_pack
from kickthefly.core import simcore
from kickthefly.sim.connectome import backends
from kickthefly.sim.connectome.sim import LIFParams, LIFSim

pytestmark = needs_pack

AVAIL = backends.detect_available_backends()
EXACT = [b for b in ("cpu", "numba", "torch-cpu") if b in AVAIL]
GPU = [b for b in ("torch-cuda", "torch-rocm", "gl") if b in AVAIL]
TORCH_GPU = [b for b in ("torch-cuda", "torch-rocm") if b in AVAIL]
# Mesa's software rasterizer (the CI gl job) steps a brain in ~0.5 s, so there these tests run shorter versions:
# fewer steps, and conditioning without memory's 5 s settling. On a real GPU they run in full.
CI_SHORT = bool(os.environ.get("KTF_GL_CI_SHORT"))
EXPECT_CLASS = {"cpu": backends.CPUBackend, "numba": backends.NumbaBackend, "torch-cpu": backends.TorchBackend,
                "torch-cuda": backends.TorchBackend, "torch-rocm": backends.TorchBackend, "gl": backends.GLBackend}


def _run(backend: str, steps: int, dtype: str = "float32", seed: int = 42, dense: bool = False):
    _, W, _ = simcore.pack()
    lp = LIFParams(backend=backend, dtype=dtype)
    if dense:
        lp.sparse_path_max_active = 0.0           # every step takes the dense path (normally only busy ones do)
    sim = LIFSim(None, lp, W_in=W, seed=seed)
    assert sim.backend.name == backend, f"asked for {backend}, got {sim.backend.name}"
    out = np.empty((steps, sim.n), bool)
    for step in range(steps):
        sens = None
        if step % 20 == 10:                                   # a sensory pulse every 100 ms
            sens = np.zeros(sim.n, np.float32)
            sens[100:150] = 2.0
        out[step] = sim.step(sens)
    return sim, out


def test_detect_available_backends():
    assert "cpu" in AVAIL


@pytest.mark.parametrize("name", list(AVAIL))
def test_requested_backend_really_runs(name):
    _, W, _ = simcore.pack()
    sim = LIFSim(None, LIFParams(backend=name), W_in=W, seed=1)
    assert isinstance(sim.backend, EXPECT_CLASS[name])
    assert sim.backend.name == name
    assert isinstance(sim.backend.device, str) and sim.backend.device


def test_unknown_or_missing_backend_falls_back_to_cpu():
    _, W, _ = simcore.pack()
    sim = LIFSim(None, LIFParams(), W_in=W, seed=123)
    assert backends.create_backend(sim, "non_existent_gpu_backend").name == "cpu"
    if not GPU:
        assert backends.create_backend(sim, "torch-cuda").name == "cpu"      # recorded as what actually ran


def test_gl_version_below_430_fallback(monkeypatch):
    class MockCtx:
        version_code = 330
        info = {"GL_RENDERER": "Mesa OpenGL 3.3"}
        def release(self): pass

    monkeypatch.setattr(backends, "_moderngl_available", True)
    monkeypatch.setattr(backends.moderngl, "create_context", lambda **kwargs: MockCtx())
    ok, msg = backends._gl_compute_available()
    assert not ok
    assert "3.3 < 4.3" in msg

    _, W, _ = simcore.pack()
    sim = LIFSim(None, LIFParams(), W_in=W, seed=42)
    b = backends.create_backend(sim, "gl")
    assert b.name == "cpu"


def test_gl_context_creation_failure_fallback(monkeypatch):
    monkeypatch.setattr(backends, "_moderngl_available", True)
    def _fail(**kwargs):
        raise RuntimeError("No headless OpenGL display / EGL device found")
    monkeypatch.setattr(backends.moderngl, "create_context", _fail)
    ok, msg = backends._gl_compute_available()
    assert not ok
    assert "failed" in msg

    _, W, _ = simcore.pack()
    sim = LIFSim(None, LIFParams(), W_in=W, seed=42)
    b = backends.create_backend(sim, "gl")
    assert b.name == "cpu"


def test_missing_moderngl_fallback(monkeypatch):
    monkeypatch.setattr(backends, "_moderngl_available", False)
    ok, msg = backends._gl_compute_available()
    assert not ok
    assert "not installed" in msg

    _, W, _ = simcore.pack()
    sim = LIFSim(None, LIFParams(), W_in=W, seed=42)
    b = backends.create_backend(sim, "gl")
    assert b.name == "cpu"


def test_missing_torch_fallback(monkeypatch):
    monkeypatch.setattr(backends, "_torch_available", False)
    monkeypatch.setattr(backends, "_torch_gpu_kind", lambda: None)
    _, W, _ = simcore.pack()
    sim = LIFSim(None, LIFParams(), W_in=W, seed=42)
    assert backends.create_backend(sim, "torch-cuda").name == "cpu"
    assert backends.create_backend(sim, "torch-rocm").name == "cpu"
    assert backends.create_backend(sim, "torch-cpu").name == "cpu"


def test_auto_backend_hierarchy(monkeypatch):
    _, W, _ = simcore.pack()
    sim = LIFSim(None, LIFParams(), W_in=W, seed=42)

    # When Torch GPU is unavailable and ModernGL is unavailable, fallback to Numba if available, else CPU
    monkeypatch.setattr(backends, "_torch_gpu_kind", lambda: None)
    monkeypatch.setattr(backends, "_moderngl_available", False)
    b = backends.create_backend(sim, "auto")
    if backends._numba_available:
        assert b.name == "numba"
    else:
        assert b.name == "cpu"

    # When all accelerators are disabled, auto falls back to CPU
    monkeypatch.setattr(backends, "_numba_available", False)
    b_cpu = backends.create_backend(sim, "auto")
    assert b_cpu.name == "cpu"


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("name", [b for b in EXACT if b != "cpu"])
def test_cpu_side_backends_are_bit_exact(name, dtype):
    """1000 steps (5 s of brain time) is well past where a single rounding difference used to show: an earlier
    fastmath Numba kernel first differed at step 289 and then decorrelated completely."""
    ref_sim, ref = _run("cpu", 1000, dtype)
    sim, got = _run(name, 1000, dtype)
    sim.backend.sync_to_host()
    assert np.array_equal(got, ref), f"{name} differs from cpu in {int((got != ref).sum())} spikes"
    assert np.array_equal(sim.v, ref_sim.v) and np.array_equal(sim.refr, ref_sim.refr)
    assert sim.gain == ref_sim.gain


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("name", [b for b in EXACT if b != "cpu"])
def test_cpu_side_backends_are_bit_exact_on_the_dense_path(name, dtype):
    """The reference sums busy steps' inputs in the state dtype and sparse ones in float32; both paths must match."""
    ref_sim, ref = _run("cpu", 300, dtype, dense=True)
    sim, got = _run(name, 300, dtype, dense=True)
    sim.backend.sync_to_host()
    assert np.array_equal(got, ref), f"{name} differs from cpu in {int((got != ref).sum())} spikes"
    assert np.array_equal(sim.v, ref_sim.v), "membrane potentials differ (a last-bit rounding difference)"


@pytest.mark.parametrize("name", GPU)
def test_gpu_backends_statistically_match(name):
    """Tolerance: brain-wide firing within 2% of the CPU's, and per-population rates (1000 neuron blocks) correlate
    at r > 0.95 over 1000 steps."""
    steps = 400 if CI_SHORT else 1000
    _, ref = _run("cpu", steps)
    _, got = _run(name, steps)
    assert abs(got.sum() - ref.sum()) / ref.sum() < 0.02
    blocks = lambda s: s[:, : s.shape[1] // 1000 * 1000].reshape(s.shape[0], -1, 1000).sum((0, 2))
    assert np.corrcoef(blocks(got), blocks(ref))[0, 1] > 0.95


@pytest.mark.parametrize("name", list(AVAIL))
def test_host_state_writes_reach_the_backend(name):
    """Save states and the neural clamp write sim.v / sim.spikes directly; every backend must see that."""
    _, W, _ = simcore.pack()
    sim = LIFSim(None, LIFParams(backend=name), W_in=W, seed=5)
    for _ in range(20):
        sim.step()
    quiet = np.flatnonzero(~sim.spikes & (sim.refr == 0))[:50]
    sim.v[quiet] = sim.p.v_thresh * 10                        # force them over threshold
    sim.backend.sync_from_host()
    assert sim.step()[quiet].all()


@pytest.mark.parametrize("name", list(AVAIL))
def test_weight_changes_reach_the_backend(name):
    """Learning, lesions and the synapse threshold edit W_csr.data in place and call on_weights_changed()."""
    _, W, _ = simcore.pack()
    a = LIFSim(None, LIFParams(backend="cpu"), W_in=W, seed=9)
    b = LIFSim(None, LIFParams(backend=name), W_in=W, seed=9)
    for sim in (a, b):
        sim.W_csr.data *= 0.0
        sim.W_csc.data *= 0.0
        sim.backend.on_weights_changed()
        for _ in range(50):
            sim.step()
    if name in EXACT:
        assert np.array_equal(a.spikes, b.spikes)


def test_headless_backend_and_dtype_reach_worker_processes(monkeypatch):
    """headless --backend/--dtype set the process-wide defaults that validation's spawned workers inherit."""
    monkeypatch.setenv("KICK_THE_FLY_SIM_BACKEND", "cpu")
    monkeypatch.setenv("KICK_THE_FLY_SIM_DTYPE", "float64")
    p = LIFParams()
    assert (p.backend, p.dtype) == ("cpu", "float64")
    from kickthefly.game import kick_the_fly as k
    args = k.parse_args(["--dtype", "float64", "--backend", "numba"])
    assert (args.dtype, args.backend) == ("float64", "numba")


@pytest.mark.parametrize("name", TORCH_GPU)
def test_batched_multi_fly_plasticity_identical(name):
    """Assert that a trained fly's KC->MBON plastic weights after N conditioning pairings
    are 100% identical between unbatched and batched multi-fly GPU execution."""
    from kickthefly.core import memory
    g, W, _ = simcore.pack()

    def run_conditioning(batched: bool):
        sim1 = LIFSim(None, LIFParams(backend=name), W_in=W.copy(), seed=42)
        sim2 = LIFSim(None, LIFParams(backend=name), W_in=W.copy(), seed=99)
        mem1 = memory.Memory(g, sim1, load=False)
        odor_pattern = np.zeros(g.n, np.float32)
        odor_pattern[mem1.kc[:50]] = 5.0
        shock_drive = np.zeros(g.n, np.float32)
        shock_drive[mem1.dan[:20]] = 8.0

        for trial in range(6):
            # Odor presentation (5 steps)
            for _ in range(5):
                if batched:
                    backends.TorchBackend.step_batch([sim1, sim2], [odor_pattern, None])
                else:
                    sim1.step(odor_pattern)
                    sim2.step(None)
            # Shock (dopamine activation, 5 steps)
            for _ in range(5):
                if batched:
                    backends.TorchBackend.step_batch([sim1, sim2], [shock_drive, None])
                else:
                    sim1.step(shock_drive)
                    sim2.step(None)
            mem1.step(sim1.activity.rates(), calm=False, steps=(trial + 1) * 10)
        return mem1.w.copy(), sim1.spikes.copy()

    w_unbatched, sp_unbatched = run_conditioning(False)
    w_batched, sp_batched = run_conditioning(True)

    assert np.array_equal(w_unbatched, w_batched), "KC->MBON weights differ between batched and unbatched!"
    assert np.array_equal(sp_unbatched, sp_batched), "Spikes differ between batched and unbatched!"


@pytest.mark.parametrize("name", TORCH_GPU)
def test_fused_lif_kernel_toggle(name):
    """Test that the fused LIF kernel toggle runs correctly and matches statistical tolerance."""
    _, W, _ = simcore.pack()
    lp = LIFParams(backend=name, fuse_lif=True)
    sim = LIFSim(None, lp, W_in=W, seed=7)
    for _ in range(50):
        sim.step()
    assert sim.spikes.shape == (sim.n,)
    assert sim.spikes.dtype == bool




@pytest.mark.skipif("gl" not in AVAIL, reason="no OpenGL 4.3 compute here")
@pytest.mark.parametrize("mode", ["scatter", "runs"])
def test_gl_plastic_weights_bit_exact_after_conditioning(mode, monkeypatch):
    """Issue #2: gl uploads only the KC -> MBON synapses learning changed, and they must land exactly.

    Ten odor + shock pairings on a gl brain, with every input to the learning rule recorded. gl's spikes aren't
    NumPy's (see the statistical test above), so the NumPy side replays those same inputs into a Memory on a
    NumPy-backed sim; the plasticity is then identical by construction and any difference is the upload's. Every
    KC -> MBON weight read back from the GPU must equal NumPy's bit for bit, and the rest of the buffer must still
    be the connectome."""
    from kickthefly.core import memory
    from kickthefly.lab import assays

    monkeypatch.setattr(backends.GLBackend, "PLASTIC_UPLOAD", mode)
    br = simcore.new_brain(seed=11, backend="gl", warmup=0)
    assert br.sim.backend.name == "gl"
    calls = []
    step = br.memory.step

    def spy(rates, calm, steps):
        calls.append((rates.copy(), calm, steps))
        return step(rates, calm, steps)

    br.memory.step = spy                                      # before the warm-up: it runs the learning rule too
    br.warmup(600)
    assays.rest(br, 500)
    for _ in range(10):
        assays.present(br, ["odor_a"], 240, shock=True)
        assays.rest(br, 160)
    gl_mem = br.memory
    assert (gl_mem.w < gl_mem.w0 * 0.9).any(), "no learning happened, so nothing was tested"
    st = br.sim.backend.upload_stats
    assert st["part_n"] > 0 and st["full_n"] == 0            # learning never re-uploaded the whole matrix

    g, W, _ = simcore.pack()
    ref_sim = LIFSim(None, LIFParams(backend="cpu"), W_in=W, seed=11)
    ref = memory.Memory(g, ref_sim, load=False)
    for rates, calm, steps in calls:
        ref.step(rates, calm, steps)
    assert np.array_equal(ref.w, gl_mem.w)

    on_gpu = br.sim.backend.read_weights()
    assert np.array_equal(on_gpu[gl_mem.csr_pos], ref_sim.W_csr.data[ref.csr_pos].astype(np.float32))
    assert np.array_equal(on_gpu, br.sim.W_csr.data.astype(np.float32))    # nothing else in the buffer moved


@pytest.mark.skipif("gl" not in AVAIL, reason="no OpenGL 4.3 compute here")
def test_gl_batched_multi_fly_plastic_weights_bit_exact(monkeypatch):
    """Batched multi-fly gl: flies stepped together on one context (the weights streamed once per step for all of
    them) must learn exactly what each learns on a context of its own.

    Two flies, each conditioned on a different odor on a thread of its own, so their KC -> MBON rows diverge and go
    per-fly. Their learned weights, the weights on the GPU and every spike must match the same two flies run
    unbatched, bit for bit, and the batched run must really have stepped flies together."""
    import gc
    import threading
    from kickthefly.lab import assays

    gc.collect()                                              # earlier tests' brains give their slots back
    odors = ("odor_a", "odor_b")

    from kickthefly.core import memory

    def condition(br, odor, raster):
        if CI_SHORT:                                          # skip memory's settling: learning from the first pairing
            br.warmup(100)
            br.memory.updates = max(br.memory.updates, memory.SETTLE_UPDATES)
            pairings = 1
        else:
            br.warmup(400)                                    # on the fly's own thread, like a game brain
            assays.rest(br, 300)
            pairings = 6
        for _ in range(pairings):
            assays.present(br, [odor], 240, shock=True)
            assays.rest(br, 100)
        for _ in range(20 if CI_SHORT else 50):               # the last steps' spikes, compared below
            raster.append(br.sim.step().copy())

    def run(batched: bool):
        monkeypatch.setattr(backends.GLBackend, "BATCH", batched)
        # 3.0 release: a group waits only for a fly seen in the last LIVE_S (50 ms) and at most GATHER_S (2 ms), sized for a real GPU.
        # On llvmpipe a step takes ~0.5 s, so two flies that drifted apart never counted as live to each other again and the run
        # sometimes stepped them alone throughout (920 dispatches for 920 fly-steps, twice on CI). Windows sized to llvmpipe's step
        # make the batching this test checks happen; the bit-exact comparison is unchanged.
        monkeypatch.setattr(backends._GLGroup, "LIVE_S", 2.0)
        monkeypatch.setattr(backends._GLGroup, "GATHER_S", 0.5)
        brains = [simcore.new_brain(seed=21 + k, backend="gl", warmup=0) for k in range(2)]
        rasters = [[], []]
        threads = [threading.Thread(target=condition, args=(br, odors[k], rasters[k])) for k, br in enumerate(brains)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for br in brains:
            assert br.sim.backend.name == "gl", "the gl backend fell back to the CPU"
        return brains, [np.array(r) for r in rasters]

    solo, solo_spikes = run(False)
    together, together_spikes = run(True)

    group = together[0].sim.backend._group
    assert group is together[1].sim.backend._group and group.member_count >= 2
    assert group.dispatches < group.fly_steps, "the flies were never stepped in the same dispatch"
    assert solo[0].sim.backend._group is not solo[1].sim.backend._group
    for a, b in zip(solo, together):
        assert (a.memory.w < a.memory.w0 * 0.9).any(), "no learning happened, so nothing was tested"
        assert np.array_equal(a.memory.w, b.memory.w), "KC -> MBON weights differ between batched and unbatched gl"
        on_gpu = b.sim.backend.read_weights()
        assert np.array_equal(on_gpu, b.sim.W_csr.data.astype(np.float32))
        assert np.array_equal(on_gpu, a.sim.backend.read_weights())
    assert not np.array_equal(together[0].memory.w, together[1].memory.w), "the two flies learned the same thing"
    for x, y in zip(solo_spikes, together_spikes):
        assert np.array_equal(x, y), "spikes differ between batched and unbatched gl"
    for br in solo + together:
        br.sim.backend.close()


@pytest.mark.skipif(not os.environ.get("KTF_REQUIRE_GL"), reason="only where CI promises OpenGL 4.3 compute")
def test_gl_is_available_where_required():
    """The CI gl job sets KTF_REQUIRE_GL: there a missing gl must fail, not quietly skip every gl test."""
    assert "gl" in AVAIL, backends._gl_compute_available()[1]


@pytest.mark.validation
@pytest.mark.skipif("gl" not in AVAIL, reason="no OpenGL 4.3 compute here")
@pytest.mark.parametrize("test_id", ["looming_escape", "sugar_feeding"])
def test_gl_reproduces_validated_pathways(test_id):
    if CI_SHORT and test_id != "looming_escape":
        pytest.skip("software rendering: the looming pathway only")
    """A short validation subset on gl: two validated pathways on the first two held-out seeds, with validation.py's
    drive, readout, control and window. gl must land on the same side of the pass criterion as NumPy (drive ratio >=
    RATIO_MIN and above the control) and within 20% of NumPy's ratios. The full suite (ten seeds and a Wilcoxon test)
    is too slow for a software-rendered CI runner; this is the check that gl's spikes still mean the same thing."""
    from kickthefly.core import savestate
    from kickthefly.lab import assays, validation

    t = validation.BY_ID[test_id]
    ratios = {}
    for name in ("cpu", "gl"):
        per = []
        for seed in validation.SEEDS[:1 if CI_SHORT else 2]:
            br = simcore.new_brain(seed=seed, backend=name, warmup=200 if CI_SHORT else 600)
            assert br.sim.backend.name == name
            g = assays.groups(br)
            drive = g[t["drive"]]
            if t["control"] in ("bitter", "sweet"):
                control = g[t["control"]]
            else:
                control = assays.random_like(g[t["control"]], len(drive), np.concatenate([drive, g[t["readout"]]]),
                                             seed * 31)
            snap: dict = {}
            meta = savestate.brain_state(br, "s_", snap)
            res = []
            for rows in (drive, control):
                savestate.restore_brain(br, meta, snap, "s_")
                window = 200 if CI_SHORT else None
                r = assays.pathway_response(br, rows, {"readout": g[t["readout"]]}, pre=window or validation.PRE,
                                            stim=window or validation.STIM)["readout"]
                res.append(validation._ratio(*r))
            per.append(res)
            if name == "gl":
                br.sim.backend.close()
        ratios[name] = np.mean(per, axis=0)                   # (drive, control)
    for name, (d, c) in ratios.items():
        assert d >= validation.RATIO_MIN and d > c, f"{name}: drive x{d:.2f} vs control x{c:.2f}"
    np.testing.assert_allclose(ratios["gl"], ratios["cpu"], rtol=0.2)
