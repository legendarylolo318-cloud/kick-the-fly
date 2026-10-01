"""3.0 day 3, Microphone -> Johnston's organ: the analysis and the sound-to-current rules (core/mic.py), the hum demo's plumbing
(lab/audio.py), and the privacy promises. No test opens a capture device."""
from __future__ import annotations

import ast
import socket
from pathlib import Path

import numpy as np
import pytest

from conftest import needs_pack
from kickthefly.core import mic


def _reading(hz, amp=0.1, seconds=1.0, **kw):
    an = mic.Analyzer(**kw)
    r = None
    for ch in mic.chunks(mic.hum(hz, seconds, amp=amp)):
        r = an.push(ch)
    return r


@pytest.mark.parametrize("hz", [60.0, 120.0, 200.0, 440.0, 900.0])
def test_the_analyzer_reads_the_frequency_of_a_hum(hz):
    r = _reading(hz)
    assert r.peak_hz == pytest.approx(hz, rel=0.02)
    if hz >= 200:                                  # a chunk holds whole cycles only above ~90 Hz; lower hums swing chunk to chunk
        assert r.rms == pytest.approx(0.1 / np.sqrt(2), rel=0.07)


def test_silence_and_hiss_drive_nothing_and_have_no_peak():
    r = _reading(200.0, amp=0.0)
    assert r.peak_hz == 0.0 and r.drive_a == r.drive_b == 0.0
    rng = np.random.default_rng(0)
    an = mic.Analyzer()
    for _ in range(200):
        r = an.push((rng.standard_normal(mic.CHUNK) * 0.0005).astype(np.float32))
    assert r.drive_a == r.drive_b == 0.0, "the noise gate: a quiet room is not a sound"


def test_low_hums_drive_jo_b_and_higher_hums_drive_jo_a_as_the_review_says():
    lo, hi = _reading(40.0), _reading(400.0)
    assert lo.drive_b > 0.4 and lo.drive_a < 0.1, "JO-B prefers below ~100 Hz"
    assert hi.drive_a > 0.4 and hi.drive_b < 0.05, "JO-A prefers higher frequencies"


def test_the_drive_rises_with_loudness_to_the_activation_current_and_stops_there():
    d = [_reading(300.0, amp=a).drive_a for a in (0.002, 0.02, 0.05, 0.2, 0.9)]
    assert d[0] == 0.0 and all(x <= y + 1e-9 for x, y in zip(d, d[1:])) and d[-1] == pytest.approx(mic.MAX_CURRENT)
    assert mic.band_drive(0.025) == pytest.approx(0.25) and mic.MAX_CURRENT == 0.5
    assert _reading(300.0, amp=0.01, sensitivity=5.0).drive_a > _reading(300.0, amp=0.01, sensitivity=1.0).drive_a


def test_sensitivity_is_limited_and_the_analysis_is_deterministic():
    assert mic.Analyzer(sensitivity=1e9).sensitivity == mic.SENSITIVITY[1] and mic.Analyzer(sensitivity=0).sensitivity == mic.SENSITIVITY[0]
    assert _reading(200.0).bins == _reading(200.0).bins and len(_reading(200.0).bins) == 24
    r = _reading(200.0)
    assert max(range(24), key=lambda i: r.bins[i]) == next(i for i in range(24) if mic.bin_edges()[i] <= 200 < mic.bin_edges()[i + 1])


def test_pulses_follow_the_interpulse_interval_in_the_drive():
    an = mic.Analyzer()
    d = [an.push(ch).drive_a for ch in mic.chunks(mic.hum(200.0, 2.0, amp=0.1, ipi_ms=35.0))]
    d = np.array(d[20:])
    assert d.max() > 0.3 and d.min() < 0.1, "the drive swings with the pulses"
    spec = np.abs(np.fft.rfft(d - d.mean()))
    f = np.fft.rfftfreq(len(d), mic.CHUNK / mic.RATE)
    assert f[int(np.argmax(spec[1:])) + 1] == pytest.approx(1000 / 35.0, rel=0.25), "its rhythm is about one pulse per 35 ms"


def test_apply_drives_only_the_jo_a_and_jo_b_neurons_and_release_removes_it():
    class Br:
        types = np.array(["JO-A1", "JO-B2", "JO-C1", "JO-E3", "DNp01", "JO-A-unclear", "JO-B1_a"])

        def __init__(self):
            self.cur = {}

        def set_current(self, name, rows, values):
            self.cur[name] = (np.asarray(rows), np.asarray(values))

        def clear_current(self, name):
            self.cur.pop(name, None)

    br = Br()
    a, b = mic.jo_rows(br)
    assert a.tolist() == [0, 5] and b.tolist() == [1, 6]
    mic.apply(br, mic.Reading(drive_a=0.4, drive_b=0.1))
    rows, vals = br.cur["mic"]
    assert sorted(rows.tolist()) == [0, 1, 5, 6], "C/E (wind, gravity) and DNp01 are left alone"
    assert dict(zip(rows.tolist(), vals.tolist())) == pytest.approx({0: 0.4, 5: 0.4, 1: 0.1, 6: 0.1})
    mic.release(br)
    assert "mic" not in br.cur


def test_availability_does_not_open_a_device_and_says_why(monkeypatch):
    monkeypatch.setattr(mic, "devices", lambda: [])
    ok, why = mic.availability()
    assert not ok and "microphone" in why
    monkeypatch.setattr(mic, "devices", lambda: ["Built-in Audio"])
    assert mic.availability()[0]


def test_starting_with_no_device_is_a_clear_error_not_a_crash(monkeypatch):
    m = mic.Mic()

    class Boom:
        def AudioDevice(self, *a, **k):
            raise RuntimeError("no such device")

    import pygame._sdl2.audio as audio
    monkeypatch.setattr(audio, "AudioDevice", Boom().AudioDevice)
    with pytest.raises(mic.MicError, match="couldn't open the microphone"):
        m.start()
    assert not m.active


# --- privacy ------------------------------------------------------------------------------------------------------------------------
def test_the_microphone_code_never_opens_a_file_or_a_socket():
    src = Path(mic.__file__).read_text()
    tree = ast.parse(src)
    banned_calls = {"open", "write", "writelines", "save", "savez", "savez_compressed", "tofile", "dump", "send", "sendall"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
            assert name not in banned_calls, f"core/mic.py calls {name}()"
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] + ([node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            assert not any(m.split(".")[0] in {"socket", "urllib", "http", "requests", "ssl", "wave", "soundfile", "ftplib", "smtplib"}
                           for m in mods), mods


def test_analysing_a_sound_writes_nothing_and_opens_no_socket(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def no_socket(*a, **k):
        raise AssertionError("the microphone code opened a socket")

    monkeypatch.setattr(socket, "socket", no_socket)
    monkeypatch.setattr(socket, "create_connection", no_socket)
    _reading(200.0, seconds=2.0)
    m = mic.Mic()
    m._latest = mic.Reading(peak_hz=200.0)
    assert m.latest().peak_hz == 200.0
    assert list(tmp_path.iterdir()) == []


def test_a_reading_holds_only_a_few_numbers_never_samples():
    r = _reading(200.0)
    assert all(isinstance(v, (int, float, tuple)) for v in vars(r).values())
    assert len(r.bins) == 24 and not any(isinstance(v, np.ndarray) for v in vars(r).values())


# --- the hum demo (headless, synthetic: no device) ------------------------------------------------------------------------------------
def test_the_demo_is_registered_and_names_its_conditions_and_criteria():
    from kickthefly.lab import audio, labjobs

    assert "hum_demo" in labjobs.ASSAYS and labjobs.ASSAY_LABEL["hum_demo"]
    assert "pulses_200_ipi35" in audio.CONDITIONS and "silence" in audio.CONDITIONS and len(audio.CRITERIA) == 3
    with pytest.raises(ValueError, match="condition"):
        audio.demo_fly(0, conditions=("humming_bird",))
    x = audio.sound("pulses_200_ipi35", 0.5)
    assert len(x) == int(0.5 * mic.RATE) and float(np.abs(x).max()) <= audio.HUM_AMP + 1e-6
    assert not audio.sound("silence", 0.5).any()


def test_the_verdict_scores_the_pre_registered_criteria_on_fake_flies():
    from kickthefly.lab import audio

    def fly(p1_pulse, p1_sil, ps1):
        cell = lambda r: dict(calm_hz=1.0, hz=r, ratio=r)  # noqa: E731
        return dict(conditions={"pulses_200_ipi35": dict(p1=cell(p1_pulse), ps1=cell(ps1)),
                                "silence": dict(p1=cell(p1_sil), ps1=cell(1.0))})

    good = [fly(1.5, 1.0, 1.0) for _ in range(10)]
    assert audio.verdict(good)["passed"]
    no_p1 = [fly(1.1, 1.0, 1.0) for _ in range(10)]
    assert not audio.verdict(no_p1)["H1"]
    song = [fly(1.5, 1.0, 2.5) for _ in range(10)]
    v = audio.verdict(song)
    assert v["H1"] and not v["H3"] and not v["passed"], "if the song motor neurons fire, H3 fails and says so"
    noisy = [fly(1.5, 1.5, 1.0) for _ in range(10)]
    assert not audio.verdict(noisy)["H2"]


@needs_pack
def test_a_hum_drives_the_real_jo_neurons_and_silence_does_not():
    from kickthefly.lab import audio

    f = audio.demo_fly(0, conditions=("silence", "steady_200"), seconds=1.0)
    s, h = f["conditions"]["silence"], f["conditions"]["steady_200"]
    assert f["n_jo_a"] == 50 and f["n_jo_b"] == 88, "the dataset's JO-A and JO-B counts"
    assert h["jo_a"]["hz"] > 20 * max(s["jo_a"]["hz"], 0.1), "a loud 200 Hz hum fires JO-A"
    assert h["analysis"]["peak_hz"] == pytest.approx(200.0, rel=0.02)
    assert s["jo_a"]["ratio"] < 2.0
