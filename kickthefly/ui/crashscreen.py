"""The crash screen (2.13): after a crash in the windowed game, a small window says where the crash report went and
offers "Report a bug" (the same review screen as Settings > Help, ui/help_ui.py) before the process ends.

It opens its own window (the game's may be gone with its OpenGL context), draws nothing of the game, and does not send
anything anywhere: reporting is the player copying the text or opening a prefilled issue in their own browser.
KICK_THE_FLY_NO_CRASH_SCREEN=1 skips it (CI, wrappers).
"""
from __future__ import annotations

from pathlib import Path

import pygame

from kickthefly.core import config
from kickthefly.core.i18n import tr
from kickthefly.ui import help_ui
from kickthefly.ui import menu as mu

SIZE = (900, 620)


class _Host:
    """The little of a Game the Menu asks for."""
    three_d = False

    def __init__(self):
        try:
            from kickthefly.core import paths
            self.cfg = config.Config.load(paths.get().config_file)
        except Exception:
            self.cfg = config.Config(None)
        self.want_quit = False
        self.paths: list[Path] = []

    def set_setting(self, *a, **k):
        pass

    def menu_action(self, name: str) -> None:
        pass


def _page_crash(menu, surf, rect, mouse) -> None:
    host = menu.host
    cx = rect.centerx
    menu.text(surf, tr("Kick the Fly stopped"), (cx, rect.y + 34), mu.INK, menu.f_head, "midtop")
    menu.wrapped(surf, tr("Something went wrong and the game had to close. Your settings, saves and the fly's training "
                          "memory are not affected. A crash report was written:"),
                 (rect.x + 40, rect.y + 90), rect.w - 80, mu.TEXT, menu.f_text, max_lines=3)
    y = rect.y + 170
    for p in host.paths or []:
        menu.text(surf, str(p), (rect.x + 40, y), mu.ACCENT, menu.f_small)
        y += 22
    if not host.paths:
        menu.text(surf, tr("(no folder was writable, so no file could be written)"), (rect.x + 40, y), mu.AMBER, menu.f_small)
    menu.wrapped(surf, tr("You can report it: you will see exactly what would be included before anything leaves this "
                          "computer, and you choose whether to copy it or open a prefilled GitHub issue. Nothing is sent "
                          "automatically."),
                 (rect.x + 40, y + 24), rect.w - 80, mu.LABEL, menu.f_small, max_lines=4)
    menu.button(surf, (cx - 270, rect.bottom - 90, 260, 50), tr("Report a bug"), lambda: menu.show("bugreport"),
                style="primary", id=("crash", "report"))
    menu.button(surf, (cx + 10, rect.bottom - 90, 260, 50), tr("Quit"), lambda: setattr(host, "want_quit", True),
                id=("crash", "quit"))


def show(written: list[Path]) -> None:
    """Block on the crash screen until the player closes it. Returns at once if no window can be opened."""
    try:
        pygame.display.quit()
        pygame.display.init()
        pygame.font.init()
        screen = pygame.display.set_mode(SIZE)
        pygame.display.set_caption("Kick the Fly")
    except pygame.error:
        return
    host = _Host()
    host.paths = list(written)
    menu = mu.Menu(host)
    menu.pages["crash"] = _page_crash
    help_ui.install(menu)
    menu.show("crash")
    clock = pygame.time.Clock()
    while not host.want_quit:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                host.want_quit = True
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE and menu.screen == "crash":
                host.want_quit = True
            else:
                menu.handle(ev, getattr(ev, "pos", pygame.mouse.get_pos()))
        if not menu.open:
            menu.show("crash")
        screen.fill(mu.BG)
        menu.draw(screen, pygame.mouse.get_pos())
        pygame.display.flip()
        clock.tick(30)
    pygame.display.quit()
