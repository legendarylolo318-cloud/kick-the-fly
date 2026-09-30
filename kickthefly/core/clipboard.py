"""Copy and paste text (share codes). Uses the window system's clipboard through pygame when it works, and says so when it
doesn't, so the code can still be selected from the screen or exported to a file.

Nothing is read from the clipboard except when the player presses Paste or Ctrl+V in a text box, and nothing is sent
anywhere: it is the operating system's own clipboard, local to this machine.
"""
from __future__ import annotations

_memory: str | None = None          # tests (and a window system without a clipboard) use this instead
use_memory = False


def put_text(text: str) -> bool:
    global _memory
    if use_memory:
        _memory = text
        return True
    try:
        import pygame

        if hasattr(pygame, "scrap"):
            pygame.scrap.init()
            pygame.scrap.put_text(text)
            return True
    except Exception:
        pass
    try:
        import pygame

        pygame.display.get_surface()
        pygame.system.set_clipboard_text(text) if hasattr(pygame, "system") and hasattr(pygame.system, "set_clipboard_text") else None
        return hasattr(pygame, "system") and hasattr(pygame.system, "set_clipboard_text")
    except Exception:
        return False


def get_text() -> str | None:
    if use_memory:
        return _memory
    try:
        import pygame

        if hasattr(pygame, "scrap"):
            pygame.scrap.init()
            t = pygame.scrap.get_text()
            if isinstance(t, bytes):
                t = t.decode("utf-8", "replace")
            return t or None
    except Exception:
        pass
    try:
        import pygame

        if hasattr(pygame, "system") and hasattr(pygame.system, "get_clipboard_text"):
            return pygame.system.get_clipboard_text() or None
    except Exception:
        pass
    return None
