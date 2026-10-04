"""3.1.0 task 9, real neuron shapes (sim/realshapes.py, game/shape_draw.py): the download is opt-in, checksummed, cached, bounded and falls back; the
drawing matches between the GPU and the CPU. The server is a local fake that speaks Google Cloud Storage's x-goog-hash dialect; nothing here touches the
network and no real shape was downloaded to write this (the live path is documented as unverified in docs/real-shapes.md)."""
from __future__ import annotations

import base64
import hashlib
import http.server
import threading

import numpy as np
import pygame
import pytest

from kickthefly.sim import realshapes as rs

SWC = "\n".join(["# test skeleton"] + [f"{i} 3 {1000 + 40 * i} {2000 + (i % 7) * 30} {3000 + i * 5} 2.0 {i - 1}" for i in range(1, 60)]) + "\n"


def _md5(b):
    return base64.b64encode(hashlib.md5(b).digest()).decode()


class Server:
    def __init__(self, files=None, hash_header=True, lie=False, big=False):
        self.files = files or {}
        self.hits: list[str] = []
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                outer.hits.append(self.path)
                name = self.path.rsplit("/", 1)[-1]
                if name not in outer.files:
                    self.send_response(404)
                    self.end_headers()
                    return
                data = outer.files[name]
                self.send_response(200)
                self.send_header("Content-Length", str(len(data) if not big else rs.MAX_BYTES + 10))
                if hash_header:
                    self.send_header("x-goog-hash", f"crc32c=AAAAAA==, md5={_md5(b'something else' if lie else data)}")
                self.end_headers()
                self.wfile.write(data)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1.0/segmentation/skeletons-malecns/skeletons-swc"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def online(monkeypatch):
    """Opted in, the netguard open, the pause between requests gone."""
    from kickthefly.core import netguard
    from kickthefly.sim import morphology

    monkeypatch.delenv("KTF_NO_NETWORK", raising=False)
    monkeypatch.delenv("KICK_THE_FLY_OFFLINE", raising=False)
    monkeypatch.setattr(netguard, "_disabled", None)
    monkeypatch.setattr(rs, "RATE_LIMIT_S", 0.0)
    old = morphology._OPT_IN
    morphology.set_opt_in(True)
    yield
    morphology.set_opt_in(old)


@pytest.fixture
def server():
    made = []

    def make(**kw):
        s = Server(**kw)
        made.append(s)
        return s

    yield make
    for s in made:
        s.close()


# --- the file --------------------------------------------------------------------------------------------------------------------------
def test_the_swc_parses_into_nodes_parents_and_segments():
    sk = rs.parse_swc(SWC, 77)
    assert sk.body_id == 77 and sk.n == 59 and sk.parent[0] == -1 and sk.parent[5] == 4
    seg = sk.segments()
    assert seg.shape == (58, 2, 3) and np.allclose(seg[0, 0], sk.xyz[1]) and np.allclose(seg[0, 1], sk.xyz[0])
    lo, hi = sk.bounds()
    assert lo[0] == 1040 and hi[0] == 1000 + 40 * 59 and sk.cable_length_um() > 0


@pytest.mark.parametrize("bad", ["", "# only a comment\n", "1 3 0 0 0 1 -1\n", "1 3 nan 0 0 1 -1\n2 3 0 0 0 1 1\n"])
def test_a_bad_swc_is_refused(bad):
    with pytest.raises(rs.ShapeUnavailable):
        rs.parse_swc(bad)


def test_the_goog_hash_header_is_read():
    h = rs.goog_hashes({"x-goog-hash": f"crc32c=AAAAAA==, md5={_md5(b'abc')}"})
    assert h["md5"] == hashlib.md5(b"abc").digest() and "crc32c" in h and rs.goog_hashes({}) == {}


# --- the store -------------------------------------------------------------------------------------------------------------------------
def test_a_download_is_verified_cached_and_recorded(online, server, tmp_path):
    srv = server(files={"123.swc": SWC.encode()})
    st = rs.ShapeStore(tmp_path, srv.url)
    sk = st.fetch(123)
    assert sk.n == 59 and (tmp_path / "123.swc").read_bytes() == SWC.encode()
    rec = st.manifest()["123"]
    assert rec["sha256"] == hashlib.sha256(SWC.encode()).hexdigest() and rec["license"] == "CC BY 4.0" and rec["size"] == len(SWC) and rec["md5"] == _md5(SWC.encode())
    assert st.cached(123) and st.count() == 1 and st.size_bytes() == len(SWC)
    assert rs.ShapeStore(tmp_path, srv.url).load(123).n == 59, "a new store finds it in the cache"
    n = len(srv.hits)
    assert st.get(123).n == 59 and len(srv.hits) == n, "a cached shape is not fetched again"


def test_nothing_is_fetched_unless_opted_in_and_allowed(server, tmp_path, monkeypatch):
    from kickthefly.core import netguard
    from kickthefly.sim import morphology

    srv = server(files={"5.swc": SWC.encode()})
    st = rs.ShapeStore(tmp_path, srv.url)
    monkeypatch.delenv("KICK_THE_FLY_OFFLINE", raising=False)
    monkeypatch.setenv("KTF_NO_NETWORK", "1")                      # the test suite's own default
    morphology.set_opt_in(True)
    with pytest.raises(rs.ShapeUnavailable, match="off here"):
        st.fetch(5)
    monkeypatch.delenv("KTF_NO_NETWORK")
    morphology.set_opt_in(False)
    with pytest.raises(rs.ShapeUnavailable, match="off"):
        st.fetch(5)
    morphology.set_opt_in(True)
    monkeypatch.setenv("KICK_THE_FLY_OFFLINE", "1")
    with pytest.raises(rs.ShapeUnavailable, match="OFFLINE"):
        st.fetch(5)
    monkeypatch.delenv("KICK_THE_FLY_OFFLINE")
    monkeypatch.setattr(netguard, "_disabled", "a headless run")
    with pytest.raises(rs.ShapeUnavailable, match="headless"):
        st.fetch(5)
    monkeypatch.setattr(netguard, "_disabled", None)
    assert srv.hits == [], "not one request was made"
    monkeypatch.setenv("KTF_NO_NETWORK", "1")
    assert st.get(5) is None and "off here" in st.last_error
    assert srv.hits == []
    morphology.set_opt_in(False)


def test_a_cached_shape_loads_even_when_the_download_is_off(online, server, tmp_path):
    from kickthefly.sim import morphology

    srv = server(files={"9.swc": SWC.encode()})
    rs.ShapeStore(tmp_path, srv.url).fetch(9)
    morphology.set_opt_in(False)
    assert rs.ShapeStore(tmp_path, srv.url).get(9, allow_network=False).n == 59


def test_a_wrong_checksum_a_missing_checksum_and_a_404_are_refused_and_not_cached(online, server, tmp_path):
    for kw, what in ((dict(lie=True), "md5"), (dict(hash_header=False), "no checksum")):
        srv = server(files={"7.swc": SWC.encode()}, **kw)
        st = rs.ShapeStore(tmp_path / what.replace(" ", "_"), srv.url)
        with pytest.raises(rs.ShapeUnavailable, match=what):
            st.fetch(7)
        assert not st.cached(7) and not list(st.folder.glob("*.swc"))
        n = len(srv.hits)
        with pytest.raises(rs.ShapeUnavailable):
            st.fetch(7)
        assert len(srv.hits) == n, "a refused neuron is not asked for again this session"
    srv = server(files={})
    st = rs.ShapeStore(tmp_path / "x", srv.url)
    with pytest.raises(rs.ShapeUnavailable, match="no published skeleton"):
        st.fetch(1)
    with pytest.raises(rs.ShapeUnavailable, match="no body id"):
        st.fetch(0)


def test_an_oversize_or_unusable_file_is_refused(online, server, tmp_path):
    srv = server(files={"3.swc": SWC.encode()}, big=True)
    with pytest.raises(rs.ShapeUnavailable, match="limit"):
        rs.ShapeStore(tmp_path / "big", srv.url).fetch(3)
    junk = b"this is not an swc file\n"
    srv2 = server(files={"4.swc": junk})
    st = rs.ShapeStore(tmp_path / "junk", srv2.url)
    with pytest.raises(rs.ShapeUnavailable, match="not a usable skeleton"):
        st.fetch(4)
    assert not st.cached(4)


def test_a_dead_server_falls_back_quietly(online, tmp_path):
    st = rs.ShapeStore(tmp_path, "http://127.0.0.1:9")            # nothing listens on the discard port
    assert st.get(1) is None and "could not reach" in st.last_error


def test_a_tampered_cache_file_is_thrown_away_and_fetched_again(online, server, tmp_path):
    srv = server(files={"11.swc": SWC.encode()})
    st = rs.ShapeStore(tmp_path, srv.url)
    st.fetch(11)
    (tmp_path / "11.swc").write_text(SWC.replace("1040", "9999"), encoding="utf-8")
    assert st.load(11) is None and not (tmp_path / "11.swc").exists() and "11" not in st.manifest()
    assert st.get(11).n == 59 and (tmp_path / "11.swc").read_bytes() == SWC.encode()


def test_clear_empties_the_cache(online, server, tmp_path):
    srv = server(files={"1.swc": SWC.encode(), "2.swc": SWC.encode()})
    st = rs.ShapeStore(tmp_path, srv.url)
    st.fetch(1)
    st.fetch(2)
    assert st.clear() == 2 and st.count() == 0 and st.manifest() == {} and not (tmp_path / "manifest.json").exists()


def test_the_loader_reports_states_for_the_ui(online, server, tmp_path, monkeypatch):
    import time

    from kickthefly.sim import morphology

    srv = server(files={"21.swc": SWC.encode()})
    ld = rs.ShapeLoader(rs.ShapeStore(tmp_path, srv.url))
    assert ld.status(None) == ("none", "") and ld.skeleton(21) is None
    ld.request(21)
    for _ in range(100):
        if ld.status(21)[0] != "loading":
            break
        time.sleep(0.05)
    assert ld.status(21) == ("ready", "downloaded") and ld.skeleton(21).n == 59
    ld.request(22)                                                  # the server has no such file
    for _ in range(100):
        if ld.status(22)[0] != "loading":
            break
        time.sleep(0.05)
    assert ld.status(22)[0] == "failed" and "no published skeleton" in ld.status(22)[1]
    ld2 = rs.ShapeLoader(rs.ShapeStore(tmp_path, srv.url))
    ld2.request(21)
    assert ld2.status(21) == ("ready", "cached")
    morphology.set_opt_in(False)
    ld3 = rs.ShapeLoader(rs.ShapeStore(tmp_path / "other", srv.url))
    ld3.request(21)
    assert ld3.status(21)[0] == "off" and "off" in ld3.status(21)[1]
    morphology.set_opt_in(True)
    ld3.reset_offline()
    assert ld3.status(21) == ("none", "")
    ld3.request(0)
    ld3.request(None)


def test_the_credit_is_on_every_surface():
    for needle in ("CC BY 4.0", "male-cns.janelia.org", "FlyEM"):
        assert needle in rs.CREDIT
    assert "CC BY 4.0" in rs.CREDIT_SHORT and "Janelia" in rs.CREDIT_SHORT
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    for f in ("README.md", "docs/real-shapes.md"):
        text = (root / f).read_text(encoding="utf-8")
        assert "MaleCNS" in text and "CC BY 4.0" in text and "Janelia" in text, f


def test_the_ten_key_neurons_use_the_public_release_first(online, server, tmp_path):
    from kickthefly.sim import morphology

    srv = server(files={"31.swc": SWC.encode()})
    store_dir = tmp_path / "cns-v1.0"
    rs.ShapeStore(store_dir, srv.url).fetch(31)
    pts = morphology.fetch_or_load_skeleton(31, cache_dir=tmp_path, allow_network=False, n_samples=21)
    assert pts is not None and pts.shape == (21, 3), "the verified cached copy is used, resampled for the view"


# --- drawing -----------------------------------------------------------------------------------------------------------------------------
@pytest.fixture
def views(synthetic_pack):
    from kickthefly.core import simcore
    from kickthefly.game import gpu_brainview as gv, kick_the_fly as k2

    pygame.init()
    g, W, soma = simcore.pack()
    hot = np.zeros(g.n, bool)
    cpu = k2.BrainView(soma, W, hot)
    gpu = gv.GPUBrainView(soma, W, hot)
    gpu.pref = "gpu"
    try:
        gpu._gpu = gv._Worker()
        gpu._gpu.call(gpu._upload_static)
    except Exception as e:
        pytest.skip(f"no offscreen OpenGL 4.3 here ({e})")
    yield cpu, gpu
    if gpu._gpu is not None:
        gpu._gpu.close()


def _skel():
    sk = rs.parse_swc(SWC)
    sk.xyz[:, 0] = 20000 + np.linspace(0, 30000, sk.n)
    sk.xyz[:, 1] = 25000 + 8000 * np.sin(np.linspace(0, 9, sk.n))
    sk.xyz[:, 2] = 30000
    return sk


def test_the_camera_and_projection_match_the_views_own_pixel_mapping(views):
    from kickthefly.game import kick_the_fly as k2, shape_draw as sd

    cpu, _ = views
    cam = sd.camera_for(cpu, "big")
    w, h = k2.VIEW_SIZES["big"]
    px, py, depth = sd.project(np.array([cpu.center]), cam, (w, h))
    assert px[0] == pytest.approx(w / 2, abs=0.5) and py[0] == pytest.approx(h / 2, abs=0.5)
    cpu.set_camera(30.0, -10.0, 100.0, 0.0, 1.5)
    cam2 = sd.camera_for(cpu, "big")
    soma0 = cpu.pts[:, 0, :]
    px2, py2, _ = sd.project(soma0, cam2, (w, h))
    ok = cpu.spark_pix["big"] >= 0
    assert np.all(np.abs(py2[ok].astype(np.int32) * w + px2[ok].astype(np.int32) - cpu.spark_pix["big"][ok]) <= 1) or \
        np.mean(cpu.spark_pix["big"][ok] == (py2[ok].astype(np.int32) * w + px2[ok].astype(np.int32))) > 0.99, "the same pixel as the picking table"


def test_the_gpu_and_the_cpu_draw_the_same_shape(views):
    from kickthefly.game import shape_draw as sd

    cpu, gpu = views
    sk = _skel()
    a = sd.overlay(cpu, sk)
    b = sd.overlay(gpu, sk)
    assert gpu.on_gpu and a.get_size() == b.get_size()
    ia = np.frombuffer(pygame.image.tobytes(a, "RGBA"), np.uint8).reshape(a.get_height(), a.get_width(), 4).astype(int)
    ib = np.frombuffer(pygame.image.tobytes(b, "RGBA"), np.uint8).reshape(b.get_height(), b.get_width(), 4).astype(int)
    assert ia[..., 3].max() > 100 and ib[..., 3].max() > 100, "something was drawn"
    both = (ia[..., 3] > 30) | (ib[..., 3] > 30)
    inter = ((ia[..., 3] > 30) & (ib[..., 3] > 30)).sum() / max(both.sum(), 1)
    assert inter > 0.6, f"the two pictures cover {inter:.0%} of the same pixels"


def test_the_thumbnail_fits_its_box_and_a_gpu_failure_falls_back(views, monkeypatch):
    from kickthefly.game import shape_draw as sd

    cpu, gpu = views
    sk = _skel()
    for v in (cpu, gpu):
        t = sd.thumbnail(v, sk, (236, 118))
        assert t.get_size() == (236, 118)
        r = t.get_bounding_rect()
        assert r.w > 60 and r.left >= 0 and r.right <= 236 and r.bottom <= 118
    monkeypatch.setattr(gpu, "_draw_lines", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("driver lost")))
    t = sd.thumbnail(gpu, sk, (236, 118))
    assert t.get_bounding_rect().w > 60 and gpu._gpu_failed, "the shape still drew, on the CPU"


def test_fit_camera_centres_and_scales_the_skeleton():
    from kickthefly.game import shape_draw as sd

    sk = _skel()
    cam = sd.fit_camera(sk, (200, 100))
    x, y, _ = sd.project(sk.xyz, cam, (200, 100))
    assert x.min() >= -1 and x.max() <= 201 and y.min() >= -1 and y.max() <= 101
    assert (x.max() - x.min() > 150) or (y.max() - y.min() > 70)


def test_the_inspector_draws_a_real_shape_and_says_what_it_is_showing(synthetic_pack, monkeypatch):
    import test_extras3 as t3
    from kickthefly.game import kick_the_fly as k2

    g = t3.make_game(False)
    try:
        i = int(np.flatnonzero(g.brain.types == "LPLC2")[0])
        g.inspect = g._neuron_info(i)
        g.view.spark_pix["big"][i] = 100 * 860 + 400
        bid = int(g.brain.body_id[i]) if getattr(g.brain, "body_id", None) is not None else 1234
        if getattr(g.brain, "body_id", None) is None:
            g.brain.body_id = np.arange(g.brain.n) + 1
            bid = int(g.brain.body_id[i])
        surf = pygame.Surface((860, 466), pygame.SRCALPHA)
        rect = pygame.Rect(0, 0, 860, 466)
        monkeypatch.setattr(rs, "network_allowed", lambda: (False, "real neuron shapes are off (Settings > Brain > Download real neuron shapes)"))
        g._draw_inspect(surf, rect, 1.0)
        assert g.shapes.status(bid)[0] == "off"
        assert g._real_shape(i, rect, surf).startswith("estimated fiber (real shapes are off")
        g.shapes._state[bid] = ("ready", "cached", _skel())
        line = g._real_shape(i, rect, surf)
        assert line.startswith("real shape: 59 nodes")
        g._draw_inspect(surf, rect, 1.0)
        assert surf.get_bounding_rect().w > 100
        assert g._shape_cache["overlay"] is not None and g._shape_cache["thumb"] is not None
        sig = g._shape_cache["sig"]
        g._real_shape(i, rect, surf)
        assert g._shape_cache["sig"] == sig, "the picture is kept while the camera does not move"
        g.view.set_camera(20.0, 5.0, 0.0, 0.0, 1.2)
        g._real_shape(i, rect, surf)
        assert g._shape_cache["sig"] != sig, "and redrawn when it moves"
    finally:
        for slot in g.flies:
            slot.brain.stop()


def test_the_setting_text_says_what_is_downloaded_from_where():
    from kickthefly.core import config

    tip = config.BY_KEY["brain.neuron_shapes"].tip
    for needle in ("MaleCNS", "CC BY 4.0", "checksummed", "never bundled", "Off by default", "estimated fiber"):
        assert needle in tip
    assert config.Config(None)["brain.neuron_shapes"] is False
