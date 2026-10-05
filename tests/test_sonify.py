"""3.1.0 task 14, brain sonification (core/sonify.py, game/sonify_play.py): the mapping from activity to notes, the synthesis (no clicks, bounded, on a scale, the song's pulses),
the silence policy (mute, volume, microphone, Streamer mode, pause), and in the game (2D and 3D) that it listens to the brain's own rates and never touches it."""
from __future__ import annotations

import numpy as np
import pygame
import pytest

from conftest import needs_pack
from kickthefly.core import sonify as so


def run(syn, secs, targets, song=False, loud=0.6):
    out = []
    for _ in range(int(round(secs / so.CHUNK_S))):
        syn.set_targets(targets)
        syn.set_song(song, loud)
        out.append(syn.render(int(so.RATE * so.CHUNK_S)))
    return np.concatenate(out)


def peaks(x, k=6):
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1 / so.RATE)
    return f[np.argsort(spec)[-k:]]


CALM = {n: (0, so.PRESENCE) for n in so.VOICE_NAMES}


# --- the mapping ------------------------------------------------------------------------------------------------------------------------
def test_the_scale_is_pentatonic_and_notes_are_in_tune():
    assert [so.degree_semitones(i) for i in range(11)] == [0, 3, 5, 7, 10, 12, 15, 17, 19, 22, 24]
    assert so.midi_hz(69) == pytest.approx(440.0) and so.midi_hz(57) == pytest.approx(220.0)
    assert len(so.VOICES) == 8 and len({v[1] for v in so.VOICES}) == 8 and [v[1] for v in so.VOICES] == sorted(v[1] for v in so.VOICES)
    assert all((root - 38) % 12 in so.SCALE for _, root, _ in so.VOICES), "every voice's root is a degree of the same scale"


def test_louder_and_higher_as_a_region_is_stirred_up_and_silent_when_it_is_not_firing():
    steps = [so.voice_target(r, 10.0) for r in (10, 14, 20, 40, 160)]
    assert [s for s, _ in steps] == sorted(s for s, _ in steps) and steps[0][0] == 0 and steps[-1][0] == so.MAX_STEPS
    louds = [l for _, l in steps]
    assert louds == sorted(louds) and louds[0] == pytest.approx(so.PRESENCE) and louds[-1] > 0.9
    assert so.voice_target(0.0, 0.0) == (0, 0.0) and so.voice_target(0.01, 0.01) == (0, 0.0)
    assert so.voice_target(5.0, 10.0)[1] == 0.0, "at half its calm it has faded out"
    assert 0.0 < so.voice_target(8.0, 10.0)[1] < so.PRESENCE
    assert so.voice_target(10.0, 0.01)[0] <= so.MAX_STEPS, "a region with no baseline cannot run away"


def test_voice_masks_find_the_neuropils_and_the_descending_neurons():
    names = ["Antennal Lobe", "Mushroom Body", "Central Complex", "Optic Lobe", "Central Brain", "Gnathal (GNG)", "VNC (T1)", "VNC (T2)", "unassigned"]
    rid = np.repeat(np.arange(9), 30)
    sc = np.array(["descending_neuron"] * 30 + ["ol_intrinsic"] * 240)
    m = so.voice_masks(names, rid, sc)
    assert list(m) == list(so.VOICE_NAMES)
    assert len(m["Nerve cord (VNC)"]) == 60 and len(m["Mushroom body"]) == 30 and len(m["Descending neurons"]) == 30
    assert set(m["Antennal lobe"]) == set(range(30))
    small = so.voice_masks(names, rid[:60], None)
    assert "Mushroom body" not in small or len(small["Mushroom body"]) >= 20


def test_a_brain_with_other_regions_still_gets_voices():
    names = [f"region {i}" for i in range(5)]
    rid = np.repeat(np.arange(5), [200, 100, 60, 10, 80])
    m = so.voice_masks(names, rid, None)
    assert 3 <= len(m) <= 4 and all(len(v) >= 50 for v in m.values()), "regions under 50 neurons are not voices"


def test_the_meter_keeps_a_slow_baseline():
    masks = {"Mushroom body": np.arange(0, 10), "Optic lobe": np.arange(10, 20)}
    mt = so.RegionMeter(masks)
    r = np.full(20, 5.0)
    mt.update(r, 0.1)
    assert list(mt.base) == [5.0, 5.0]
    r2 = r.copy()
    r2[:10] = 20.0
    mt.update(r2, 0.1)
    assert mt.rate[0] == 20.0 and 5.0 < mt.base[0] < 5.1, "the baseline barely moves in 0.1 s"
    for _ in range(3000):
        mt.update(r2, 0.1)
    assert mt.base[0] == pytest.approx(20.0, rel=0.01), "and settles on a lasting level"
    t = mt.targets()
    assert t["Optic lobe"][0] == 0


# --- the song readout -----------------------------------------------------------------------------------------------------------------------
def test_the_song_gate_has_hysteresis_at_the_games_threshold():
    g = so.SongGate(1.8)
    assert not g.update(1.0) and not g.update(1.79) and g.update(1.8)
    assert g.update(1.5) and g.update(1.3), "stays on a little below the threshold"
    assert not g.update(1.2)
    assert g.loudness() == pytest.approx(0.4)
    g.update(3.6)
    assert g.loudness() == pytest.approx(0.9) and 0.4 <= so.SongGate().loudness() <= 0.9


def test_the_song_is_pulses_35_ms_apart_at_220_hz_and_stops_cleanly():
    syn = so.Synth()
    quiet = {n: (0, 0.0) for n in so.VOICE_NAMES}
    x = run(syn, 1.0, quiet, True, 0.9)
    env = np.convolve(np.abs(x), np.ones(110) / 110, mode="same")          # 5 ms: smooth over the carrier's cycles
    on = env > 0.03
    starts = np.flatnonzero(on[1:] & ~on[:-1])
    gaps = np.diff(starts) / so.RATE
    assert len(starts) > 20 and np.median(gaps) == pytest.approx(so.SONG_IPI_S, abs=0.002), "one pulse every 35 ms, across chunk boundaries too"
    assert any(abs(peaks(x, 12) - so.SONG_CARRIER_HZ) < 40), "its energy is around the 220 Hz carrier"
    tail = run(syn, 0.3, quiet, False)
    assert np.abs(tail[-so.RATE // 10:]).max() < 1e-3, "after the song stops it is silent"
    assert np.abs(np.diff(x)).max() < 0.6


# --- the mix -------------------------------------------------------------------------------------------------------------------------------------------
def test_the_mix_is_bounded_smooth_and_on_the_scale():
    syn = so.Synth()
    calm = run(syn, 1.0, CALM)
    stir = dict(CALM)
    stir["Mushroom body"] = so.voice_target(40.0, 10.0)
    loud = {n: so.voice_target(80.0, 10.0) for n in so.VOICE_NAMES}
    mid = run(syn, 1.0, stir)
    allv = run(syn, 1.0, loud)
    x = np.concatenate([calm, mid, allv])
    assert np.abs(x).max() <= so.OUT_PEAK + 1e-9, "the soft limiter holds even with every voice at full"
    d = np.abs(np.diff(x))
    n = int(so.RATE * so.CHUNK_S)
    for k in range(n, len(d) - n, n):
        around = max(d[k - n:k - 1].max(), d[k + 1:k + n].max())
        assert d[k - 1] <= 1.5 * around, f"a click at the join at {k / so.RATE:.1f} s: {d[k - 1]:.3f} against {around:.3f} around it"
    assert abs(float(x.mean())) < 0.02
    root = {v[0]: so.midi_hz(v[1]) for v in so.VOICES}
    f = peaks(calm, 3)
    assert any(abs(f - root["Mushroom body"]) < 3), "a calm voice sits on its root"
    seg = mid[-so.RATE // 2:]
    assert any(abs(peaks(seg, 8) - so.midi_hz(so.voice_midi(57, so.voice_target(40.0, 10.0)[0]))) < 4), "a stirred voice has moved up the scale"


def test_every_pitch_the_mix_can_make_is_on_the_scale():
    for name, root, _ in so.VOICES:
        assert so.voice_midi(root, 0) == root, "each voice's calm note is its root"
        notes = [so.voice_midi(root, step) for step in range(so.MAX_STEPS + 1)]
        assert notes == sorted(notes) and len(set(notes)) == len(notes)
        assert all((n - so.SCALE_ROOT) % 12 in so.SCALE for n in notes), name


def test_a_loud_region_is_louder_in_the_mix_and_more_voices_do_not_make_it_unbounded():
    one = dict(CALM)
    one["Optic lobe"] = so.voice_target(80.0, 10.0)
    a = run(so.Synth(), 1.0, CALM)
    b = run(so.Synth(), 1.0, one)
    assert np.sqrt((b[-5000:] ** 2).mean()) > np.sqrt((a[-5000:] ** 2).mean())
    allv = run(so.Synth(), 1.0, {n: so.voice_target(80.0, 10.0) for n in so.VOICE_NAMES})
    assert np.sqrt((allv[-5000:] ** 2).mean()) < 0.6


def test_pcm_is_int16_and_stereo_repeats():
    pcm = so.to_pcm(np.array([0.0, 0.5, -2.0, 2.0]))
    assert pcm.dtype == np.int16 and list(pcm) == [0, 16383, -32767, 32767]
    st = so.to_pcm(np.array([0.5, -0.5]), 2)
    assert st.shape == (2, 2) and st[0, 0] == st[0, 1]


# --- the policy ------------------------------------------------------------------------------------------------------------------------------------
def test_the_policy_says_why_it_is_silent():
    base = dict(enabled=True, audio_ok=True, muted=False, master=1.0, volume=0.4)
    assert so.silent_reason(**base) is None
    assert so.silent_reason(**{**base, "enabled": False}) == "off"
    assert so.silent_reason(**{**base, "audio_ok": False}) == "no audio device"
    assert so.silent_reason(**{**base, "muted": True}) == "muted"
    assert so.silent_reason(**{**base, "master": 0.0}) == "volume is 0" and so.silent_reason(**{**base, "volume": 0.0}) == "volume is 0"
    assert "microphone" in so.silent_reason(**base, mic_on=True)
    assert "Streamer" in so.silent_reason(**base, stream_on=True)
    assert so.silent_reason(**base, stream_on=True, allow_in_stream=True) is None
    assert so.silent_reason(**base, paused=True) == "time is paused"
    assert so.silent_reason(**{**base, "muted": True}, mic_on=True) == "muted", "the more basic reason comes first"


# --- in the game ------------------------------------------------------------------------------------------------------------------------------------------
def _rig(three_d):
    from kickthefly.lab.playthrough import Rig

    return Rig(three_d, "cpu", seed=5)


class _Live:
    def __init__(self, mic=False, stream=False):
        self.mic_on, self.stream_on = mic, stream

    def tick(self):
        pass


@needs_pack
def test_settings_and_the_hotkey_exist_and_it_is_off_by_default():
    from kickthefly.core import config

    cfg = config.Config(None)
    assert cfg["audio.sonify"] is False and cfg["stream.allow_sonify"] is False and cfg["audio.sonify_song"] is True and 0 < cfg["audio.sonify_vol"] <= 1
    assert dict((a, d) for a, _, d in config.ACTIONS)["sonify"] == "f5"
    keys = [d for _, _, d in config.ACTIONS]
    assert keys.count("f5") == 1


@needs_pack
@pytest.mark.parametrize("three_d", [False, True], ids=["2d", "3d"])
def test_it_is_silent_until_asked_then_follows_the_brain_and_obeys_mute_mic_and_stream(three_d):
    r = _rig(three_d)
    try:
        g = r.game
        sn = g.sonify
        r.seconds(1.0)
        assert sn.reason == "off" and sn.chunks == 0 and not sn._playing
        g.cfg.set("audio.sonify", True)
        r.seconds(3.0)
        assert sn.reason == "" and sn.chunks > 5 and set(sn.levels) == set(so.VOICE_NAMES) and pygame.mixer.Channel(0).get_busy()
        # it reads the brain's own rates: a strong smell stirs the antennal lobe voice up the scale
        br = g.flies[0].brain
        calm = sn.levels["Antennal lobe"]
        for k in range(120):
            if k % 6 == 0:
                br.poke("smell", None, 1.0)
            r.frames(1)
        assert sn.levels["Antennal lobe"][0] > calm[0] and sn.levels["Antennal lobe"][1] > calm[1] + 0.2
        g.cfg.set("audio.mute", True)
        g.sound.configure(g.cfg)
        r.frames(2)
        assert sn.reason == "muted" and not sn._playing and not pygame.mixer.Channel(0).get_busy()
        g.cfg.set("audio.mute", False)
        g.sound.configure(g.cfg)
        r.frames(2)
        assert sn.reason == ""
        real = g.live
        g.live = _Live(mic=True)
        r.frames(2)
        assert "microphone" in sn.reason and not pygame.mixer.Channel(0).get_busy()
        g.live = _Live(stream=True)
        r.frames(2)
        assert "Streamer" in sn.reason
        g.cfg.set("stream.allow_sonify", True)
        r.frames(2)
        assert sn.reason == ""
        g.live = real
        g.cfg.set("audio.sonify_vol", 0.0)
        r.frames(2)
        assert sn.reason == "volume is 0"
        g.cfg.set("audio.sonify_vol", 0.25)
        g.cfg.set("audio.master", 0.5)
        r.frames(2)
        assert pygame.mixer.Channel(0).get_volume() == pytest.approx(0.125, abs=0.01), "master x its own volume"
        g.cfg.set("audio.sonify", False)
        r.frames(2)
        assert sn.reason == "off" and not pygame.mixer.Channel(0).get_busy()
    finally:
        r.close()


@needs_pack
@pytest.mark.parametrize("three_d", [False, True], ids=["2d", "3d"])
def test_the_song_voice_follows_the_song_neurons_and_replaces_the_one_off_buzz(three_d):
    r = _rig(three_d)
    try:
        g = r.game
        sn = g.sonify
        g.cfg.set("audio.sonify", True)
        r.seconds(3.0)
        assert not sn.gate.on and sn.voicing_song()
        played = []
        real = g.sound.play
        g.sound.play = lambda name, vol=1.0: (played.append(name), real(name, vol))[1]
        br = g.flies[0].brain
        rows = np.flatnonzero(np.asarray(br.types).astype(str) == "pIP10")
        assert len(rows) >= 1
        br.set_current("t", rows, 0.48)
        for _ in range(60 * 12):
            r.frames(1)
            if sn.gate.on:
                break
        assert sn.gate.on and sn.synth.song_on, "pIP10 driven: the ps1 song readout crosses the game's SONG threshold and the song sounds"
        for _ in range(120):
            r.frames(1)
        assert "pulse_song" not in played, "the sonification's song is that buzz, so the game's own is not played on top"
        g.cfg.set("audio.sonify_song", False)
        r.frames(15)
        assert not sn.synth.song_on and not sn.voicing_song()
    finally:
        r.close()


@needs_pack
def test_it_never_changes_the_brain():
    on, off = _rig(False), _rig(False)
    try:
        on.game.cfg.set("audio.sonify", True)
        for r in (on, off):
            r.seconds(4.0)
        a, b = on.game.flies[0].brain, off.game.flies[0].brain
        assert a.steps == b.steps and np.array_equal(a.sim.activity.rates(), b.sim.activity.rates()), "with it on, every neuron's rate is what it is with it off"
    finally:
        on.close()
        off.close()


@needs_pack
def test_the_legend_draws_and_says_why_it_is_silent():
    r = _rig(False)
    try:
        g = r.game
        g.cfg.set("audio.sonify", True)
        r.seconds(1.0)
        s1 = pygame.Surface((1280, 760))
        g.x3.draw(s1, g.clock.now)
        g.cfg.set("audio.mute", True)
        g.sound.configure(g.cfg)
        r.frames(3)
        s2 = pygame.Surface((1280, 760))
        g.x3.draw(s2, g.clock.now)
        assert g.sonify.reason == "muted"
        assert pygame.image.tobytes(s1, "RGB") != pygame.image.tobytes(s2, "RGB") and s1.get_bounding_rect().w > 0
    finally:
        r.close()
