"""Brain sonification in the game (3.1.0 task 14): plays kickthefly/core/sonify.py through a mixer channel of its own, in the 3D game and in --2d.

Opt-in (Settings > Audio > Brain sonification, or the F5 key; saved like the stethoscope), and silent whenever the game's audio rules say so: muted (M), master volume or its own volume
at 0, no audio device, the microphone on (the speakers would feed the fly's own hearing neurons: a loop), Streamer mode on (unless Settings > Audio allows it), time paused.
Its volume is `audio.master x audio.sonify_vol`. It only reads the brain (`sim.activity.rates()` and `Brain.level("song")`), so it cannot change a spike.

The audio is streamed in 0.1 s chunks, one queued ahead, on a reserved mixer channel, so it is about 0.1-0.2 s behind the brain; the voices need no more than that.
Everything here is GAME RULE (the rendering); what it reads is CONNECTOME.
"""
from __future__ import annotations

import math
import time

import numpy as np

from kickthefly.core import sonify as so

CHANNEL = 0                                       # the mixer channel reserved for it (everything else plays on the others)
METER_EVERY_S = 0.1
SONG_SMOOTH_S = 0.4


class Sonifier:
    def __init__(self, game):
        self.g = game
        self.reason = "off"                       # why it is silent right now ("" while it sounds)
        self.meter: so.RegionMeter | None = None
        self.synth: so.Synth | None = None
        self.gate: so.SongGate | None = None
        self.song_slow = 1.0
        self._brain = None
        self._last_meter = 0.0
        self._chan = None
        self._playing = False
        self.chunks = 0
        self.levels: dict[str, tuple[int, float]] = {}

    # --- the switch ----------------------------------------------------------------------------------------------------------------------------
    @property
    def wanted(self) -> bool:
        return bool(self.g.cfg["audio.sonify"])

    def toggle(self) -> bool:
        g = self.g
        val = not self.wanted
        g.cfg.set("audio.sonify", val)
        g.cfg.save()
        self.tick(time.perf_counter())
        if val:
            g.note("SONIFY   on" + (f" but silent: {self.reason}" if self.reason else "") + " [GAME RULE: the sound; CONNECTOME: the activity it follows]", source="rule")
        else:
            g.note("SONIFY   off", source="rule")
        return val

    def voicing_song(self) -> bool:
        """Whether the song voice is sounding the SONG reaction, so the game's own one-off buzz should not double it."""
        return self.wanted and not self.reason and bool(self.g.cfg["audio.sonify_song"])

    # --- policy --------------------------------------------------------------------------------------------------------------------------------
    def silent_reason(self) -> str | None:
        g = self.g
        live = getattr(g, "live", None)
        clock = getattr(g, "clock", None)
        return so.silent_reason(
            enabled=self.wanted, audio_ok=bool(g.sound.ok), muted=bool(g.sound.muted or g.cfg["audio.mute"]), master=float(g.cfg["audio.master"]),
            volume=float(g.cfg["audio.sonify_vol"]), mic_on=bool(getattr(live, "mic_on", False)), stream_on=bool(getattr(live, "stream_on", False)),
            allow_in_stream=bool(g.cfg["stream.allow_sonify"]), paused=bool(getattr(clock, "user_paused", False)))

    # --- per frame -----------------------------------------------------------------------------------------------------------------------------
    def tick(self, now: float) -> None:
        try:
            self._tick()
        except Exception:
            from kickthefly.core.crash import log

            log.exception("sonification failed; turned off")
            self.g.cfg.set("audio.sonify", False)
            self.reason = "off"
            self._stop_audio()

    def _tick(self) -> None:
        why = self.silent_reason()
        self.reason = why or ""
        if why is not None:
            self._stop_audio()
            if why == "off":
                self.meter = self.synth = self.gate = None
            return
        br = self.g.flies[self.g.focus].brain
        if self.meter is None or br is not self._brain:
            self._build(br)
        if self.meter is None:
            self.reason = "this brain has no regions to listen to"
            return
        wall = float(self.g.clock.now)                  # game time: the same in a test as in play, and it follows slow motion
        if wall - self._last_meter >= METER_EVERY_S:
            dt = min(1.0, wall - self._last_meter) if self._last_meter else METER_EVERY_S
            self._last_meter = wall
            self.meter.update(br.sim.activity.rates() * 200.0, dt)
            self.levels = self.meter.targets()
            self.synth.set_targets(self.levels)
            if self.g.cfg["audio.sonify_song"] and "song" in br.col:
                self.song_slow += (br.level("song") - self.song_slow) * (1.0 - math.exp(-dt / SONG_SMOOTH_S))
                self.synth.set_song(self.gate.update(self.song_slow), self.gate.loudness())
            else:
                self.gate.update(0.0)
                self.synth.set_song(False)
        self._feed()

    def _build(self, br) -> None:
        from kickthefly.game import kick_the_fly as k2

        v = self.g.view
        masks = so.voice_masks(v.region_names, v.region_id, getattr(br, "superclass", None))
        self._brain = br
        self.song_slow = 1.0
        if not masks:
            self.meter = None
            return
        self.meter = so.RegionMeter(masks)
        self.synth = so.Synth([voice for voice in so.VOICES if voice[0] in masks])
        self.gate = so.SongGate(float(k2.THRESH["song"]))
        self._last_meter = 0.0

    # --- the mixer ----------------------------------------------------------------------------------------------------------------------------
    def _channel(self):
        import pygame

        if pygame.mixer.get_num_channels() < 21:                # (the game's Sound sets 20 again when a new game starts)
            pygame.mixer.set_num_channels(21)
        pygame.mixer.set_reserved(1)                             # so no other sound is put on this channel
        if self._chan is None:
            self._chan = pygame.mixer.Channel(CHANNEL)
        return self._chan

    def _chunk(self):
        import pygame

        n = int(so.RATE * so.CHUNK_S)
        pcm = so.to_pcm(self.synth.render(n), int(getattr(self.g.sound, "channels", 1)))
        self.chunks += 1
        return pygame.sndarray.make_sound(pcm)

    def volume(self) -> float:
        return float(min(1.0, self.g.cfg["audio.master"] * self.g.cfg["audio.sonify_vol"]))

    def _feed(self) -> None:
        ch = self._channel()
        ch.set_volume(self.volume())
        if not ch.get_busy():
            ch.play(self._chunk())
            ch.queue(self._chunk())
            self._playing = True
        elif ch.get_queue() is None:
            ch.queue(self._chunk())

    def _stop_audio(self) -> None:
        if self._playing and self._chan is not None:
            self._chan.stop()
            self._chan.stop()                      # twice: the first stop lets the queued chunk start, the second ends that too
        self._playing = False
        if self.synth is not None:                 # when it starts again it fades in from nothing rather than from where it was
            self.synth.amp[:] = 0.0
            self.synth.set_song(False)

    def stop(self) -> None:
        self._stop_audio()
