"""A background job for the menu pages (3.0 day 4): a thread with a progress fraction, a label, a cancel flag and a result, so a
heavy analysis (network science, the tournament, the sensitivity grid) never runs on the game thread. The page reads it each frame
and draws a progress bar; nothing here touches pygame."""
from __future__ import annotations

import threading
import time


class BgJob:
    def __init__(self, label: str, fn, **kw):
        self.label = label
        self.fn, self.kw = fn, kw
        self.frac = 0.0
        self.note = ""
        self.result = None
        self.error: str | None = None
        self.cancelled = False
        self.cancel = threading.Event()
        self.t0 = time.time()
        self.thread = threading.Thread(target=self._run, name=f"bg-{label}", daemon=True)

    def start(self) -> "BgJob":
        self.thread.start()
        return self

    def update(self, frac: float | None = None, note: str | None = None) -> None:
        if frac is not None:
            self.frac = max(0.0, min(1.0, float(frac)))
        if note is not None:
            self.note = str(note)

    def _run(self) -> None:
        try:
            self.result = self.fn(self, **self.kw)
        except Exception as e:                                   # shown on the page; a failed job never reaches the game thread
            if self.cancel.is_set():
                self.cancelled = True
            else:
                self.error = f"{type(e).__name__}: {e}"
        finally:
            self.frac = 1.0 if self.error is None and not self.cancelled else self.frac

    @property
    def running(self) -> bool:
        return self.thread.is_alive()

    @property
    def elapsed(self) -> float:
        return time.time() - self.t0

    def stop(self) -> None:
        self.cancel.set()


def draw_progress(m, surf, rect, job: BgJob, ui) -> None:
    """A progress bar with the job's note and a Cancel button, in the page's own style."""
    import pygame

    pygame.draw.rect(surf, (30, 36, 48), (rect.x, rect.y, rect.w - 110, 12), border_radius=6)
    pygame.draw.rect(surf, ui.AMBER, (rect.x, rect.y, max(8, int((rect.w - 110) * job.frac)), 12), border_radius=6)
    m.text(surf, f"{job.label}: {job.note}  ({job.elapsed:.0f} s)", (rect.x, rect.y + 18), ui.TEXT, m.f_small)
    m.button(surf, (rect.right - 100, rect.y - 8, 100, 30), "Cancel", job.stop, id=("bgjob_cancel", job.label), style="danger",
             tip="Stops after the current step. Nothing already written is lost.")
