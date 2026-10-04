"""The clickable pause menu and Settings screen, drawn onto the HUD surface by both the 2D and the 3D game.

Immediate mode: each frame the current screen draws its widgets and registers where they are; the next mouse or
keyboard event is matched against those regions. Screens are small functions, so the game can add its own (save
states, challenges, Lab tools) with `Menu.pages[name] = fn(menu, surf, rect, mouse)`.

The host (the game) provides:
  host.cfg                         config.Config
  host.three_d                     bool
  host.set_setting(key, value)     validate, store, apply live, save
  host.menu_action(name)           "resume", "quit", "save_state", "load_state", ...
"""
from __future__ import annotations

import random
import time

import pygame

from kickthefly.core import config
from kickthefly.core.config import ACTIONS, SETTINGS, TABS
from kickthefly.core.i18n import tr
from kickthefly.core.version import __version__

BG = (14, 17, 23)
PANEL = (20, 24, 32)
ROW_HOVER = (30, 35, 46)
BTN = (44, 50, 64)
BTN_HOVER = (62, 70, 90)
BTN_DOWN = (80, 90, 116)
BTN_PRIMARY = (40, 110, 150)
BTN_PRIMARY_HOVER = (52, 136, 182)
BTN_DANGER = (130, 44, 44)
BTN_DANGER_HOVER = (160, 56, 56)
BORDER = (52, 60, 76)
ACCENT = (86, 214, 255)
AMBER = (255, 176, 64)
INK = (240, 243, 248)
TEXT = (205, 212, 224)
LABEL = (130, 142, 160)
DIM = (80, 88, 102)
GOOD = (90, 200, 120)
BAD = (230, 90, 80)
TAG_COLORS = {config.CONNECTOME: (60, 170, 220), config.GAME_RULE: (220, 150, 50), "CONNECTOME": (60, 170, 220),
              "GAME RULE": (220, 150, 50), "RESTART": (110, 118, 136),
              "3D ONLY": (110, 118, 136), "REAL": (60, 170, 220), "RULE": (220, 150, 50),
              "LITERATURE": (110, 190, 130), "REFUSED": (200, 70, 60)}


class Menu:
    def __init__(self, host):
        self.host = host
        self.screen: str | None = None
        self.stack: list[str] = []
        self.tab = TABS[0]
        self.scroll: dict[str, float] = {}
        self.hits: list[tuple[pygame.Rect, str, object]] = []
        self.pressed: object = None
        self.drag: dict | None = None
        self.edit: dict | None = None             # typed value: {id, text, commit}; "alpha": True takes any printable text
        self.item_drag: dict | None = None        # a card being dragged (kind "dragsrc" regions): {payload, start, pos, moved}
        self.capture: str | None = None           # action waiting for a key press
        self.message: tuple[str, float, tuple] | None = None
        self.hover_id: object = None
        self.hover_since = 0.0
        self.tip: tuple[pygame.Rect, str] | None = None
        self.clip: pygame.Rect | None = None
        self.pages: dict[str, object] = {}
        self._font_key = None
        self.content_h: dict[str, int] = {}

    # --- fonts ---------------------------------------------------------------------------------------------------------
    def fonts(self):
        big = bool(self.host.cfg["access.larger_text"])
        if self._font_key != big:
            k = 1.25 if big else 1.0
            fam = "segoeui,dejavusans,liberationsans,arial"
            self.f_small = pygame.font.SysFont(fam, round(14 * k))
            self.f_text = pygame.font.SysFont(fam, round(16 * k))
            self.f_bold = pygame.font.SysFont("segoeuisemibold," + fam, round(16 * k), bold=True)
            self.f_head = pygame.font.SysFont("segoeuiblack," + fam, round(22 * k), bold=True)
            self.f_title = pygame.font.SysFont("segoeuiblack," + fam, round(38 * k), bold=True)
            self._font_key = big
        return self

    # --- navigation ----------------------------------------------------------------------------------------------------
    @property
    def open(self) -> bool:
        return self.screen is not None

    def show(self, screen: str = "pause") -> None:
        if self.screen is None:
            self.stack = []
        elif self.screen != screen:
            self.stack.append(self.screen)
        self.screen = screen
        self.edit = self.capture = None
        self.drag = None

    def back(self) -> None:
        self.edit = self.capture = None
        self.drag = None
        if self.stack:
            self.screen = self.stack.pop()
        else:
            self.close()

    def close(self) -> None:
        self.screen = None
        self.stack = []
        self.edit = self.capture = None
        self.host.menu_action("closed")

    def flash(self, text: str, color=TEXT, seconds: float = 4.0) -> None:
        self.message = (text, time.perf_counter() + seconds, color)

    # --- events --------------------------------------------------------------------------------------------------------
    def _hit(self, pos):
        for rect, kind, data in reversed(self.hits):
            if rect.collidepoint(pos) and (data.get("clip") is None or data["clip"].collidepoint(pos)):
                return rect, kind, data
        return None

    def handle(self, ev, pos) -> bool:
        """Consume an event while the menu is open. pos: the mouse in HUD coordinates (for mouse events)."""
        if not self.open:
            return False
        if self.capture and self.capture.startswith("pad:") and ev.type in (
                pygame.JOYBUTTONDOWN, pygame.JOYAXISMOTION, pygame.CONTROLLERBUTTONDOWN, pygame.CONTROLLERAXISMOTION):
            from kickthefly.game import gamepad

            b = gamepad.capture_binding(ev, getattr(self.host, "pad", None))
            if b is not None:
                self._bind_pad(self.capture[4:], b)
            return True
        if ev.type == pygame.KEYDOWN:
            return self._key(ev)
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            if self.edit is not None:
                self._commit_edit()
            h = self._hit(pos)
            if h is None:
                return True
            rect, kind, data = h
            if kind == "slider":
                self.drag = dict(data, rect=rect)
                self._drag_to(pos[0])
            elif kind == "edit":
                self.edit = dict(id=data["id"], text=data["text"], commit=data["commit"], fresh=True,
                                 alpha=bool(data.get("alpha")), limit=int(data.get("limit", 12)),
                                 compact=bool(data.get("compact")))
            elif kind == "dragsrc":
                self.item_drag = dict(data, start=tuple(pos), pos=tuple(pos), moved=False)
                self.pressed = data["id"]
            else:
                self.pressed = data["id"]
            return True
        if ev.type == pygame.MOUSEMOTION and self.drag is not None:
            self._drag_to(pos[0])
            return True
        if ev.type == pygame.MOUSEMOTION and self.item_drag is not None:
            d = self.item_drag
            d["pos"] = tuple(pos)
            if abs(pos[0] - d["start"][0]) + abs(pos[1] - d["start"][1]) > 8:
                d["moved"] = True
            return True
        if ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            if self.item_drag is not None:
                d, self.item_drag = self.item_drag, None
                self.pressed = None
                if not d["moved"]:
                    if d.get("click") and d.get("enabled", True):
                        self.host.menu_action("click_sound")
                        d["click"]()
                    return True
                target = next((data for rect, kind, data in reversed(self.hits)
                               if kind == "drop" and rect.collidepoint(pos)
                               and (data.get("accept") is None or data["accept"](d["payload"]))), None)
                if target is not None:
                    target["drop"](d["payload"], pos)
                elif d.get("release_outside"):
                    d["release_outside"](d["payload"], pos)
                return True
            if self.drag is not None:
                d, self.drag = self.drag, None
                if d.get("release"):
                    d["release"]()
                return True
            h = self._hit(pos)
            if h is not None and h[2]["id"] == self.pressed and h[2].get("enabled", True) and "click" in h[2]:
                self.pressed = None
                self.host.menu_action("click_sound")
                h[2]["click"]()
            self.pressed = None
            return True
        if ev.type == pygame.MOUSEWHEEL:
            key = self._scroll_key()
            limit = max(0, self.content_h.get(key, 0))
            self.scroll[key] = min(max(0.0, self.scroll.get(key, 0.0) - ev.y * 40), limit)
            return True
        return ev.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION, pygame.KEYUP,
                           pygame.TEXTINPUT)

    def _scroll_key(self) -> str:
        return f"{self.screen}:{self.tab}" if self.screen == "settings" else str(self.screen)

    def _bind_pad(self, action: str, binding: str) -> None:
        self.capture = None
        ok, msg = self.host.cfg.bind_pad(action, binding)
        if ok:
            self.host.set_setting("keys", None)
        from kickthefly.game import gamepad

        self.flash(msg or f"{config.PAD_LABEL[action]}: {gamepad.pretty(binding)}", AMBER if msg else GOOD)

    def _key(self, ev) -> bool:
        if self.capture is not None and self.capture.startswith("pad:"):
            if ev.key == pygame.K_ESCAPE:
                self.capture = None
            elif ev.key in (pygame.K_BACKSPACE, pygame.K_DELETE):
                self._bind_pad(self.capture[4:], "")
            return True
        if self.capture is not None:
            action, self.capture = self.capture, None
            if ev.key == pygame.K_ESCAPE:
                return True
            ok, msg = self.host.cfg.bind(action, pygame.key.name(ev.key))
            if ok:
                self.host.set_setting("keys", None)
            self.flash(msg or f"{config.ACTION_LABEL[action]}: {pygame.key.name(ev.key)}", AMBER if msg else GOOD)
            return True
        if self.edit is not None:
            if ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self._commit_edit()
            elif ev.key == pygame.K_ESCAPE:
                self.edit = None
            elif ev.key == pygame.K_BACKSPACE:
                self.edit["text"] = "" if self.edit.pop("fresh", False) else self.edit["text"][:-1]
            elif ev.key == pygame.K_v and (ev.mod & (pygame.KMOD_CTRL | pygame.KMOD_META)) and self.edit.get("alpha"):
                from kickthefly.core import clipboard

                pasted = clipboard.get_text()                 # only when the player presses Ctrl+V in a text box
                if pasted:
                    text = "".join(pasted.split()) if self.edit.get("compact") else " ".join(pasted.split())
                    self.edit["text"] = ("" if self.edit.pop("fresh", False) else self.edit["text"]) + text
                    self.edit["text"] = self.edit["text"][:self.edit.get("limit", 12)]
            elif ev.unicode and (ev.unicode.isprintable() if self.edit.get("alpha") else ev.unicode in "0123456789.-") \
                    and len(self.edit["text"]) < self.edit.get("limit", 12):
                if self.edit.pop("fresh", False):        # typing replaces the old value
                    self.edit["text"] = ""
                self.edit["text"] += ev.unicode
            return True
        if ev.key == pygame.K_ESCAPE:
            self.back()
        return True

    def _commit_edit(self) -> None:
        e, self.edit = self.edit, None
        if e.get("alpha"):
            e["commit"](e["text"])
            return
        try:
            e["commit"](float(e["text"]))
        except ValueError:
            self.flash("not a number", BAD)

    def _drag_to(self, x: float) -> None:
        d = self.drag
        r = d["track"]
        frac = min(1.0, max(0.0, (x - r.x) / max(1, r.w)))
        v = d["lo"] + frac * (d["hi"] - d["lo"])
        if d["step"]:
            v = round(v / d["step"]) * d["step"]
        d["change"](v)

    # --- widgets (call during draw) -------------------------------------------------------------------------------------
    def _register(self, rect: pygame.Rect, kind: str, **data) -> bool:
        """Adds a hit region. Returns True if the mouse is over it (and inside the current clip)."""
        data["clip"] = self.clip
        self.hits.append((rect, kind, data))
        m = self.mouse
        over = rect.collidepoint(m) and (self.clip is None or self.clip.collidepoint(m))
        if over and data.get("tip"):
            if self.hover_id != data["id"]:
                self.hover_id, self.hover_since = data["id"], time.perf_counter()
            elif time.perf_counter() - self.hover_since > 0.35:
                self.tip = (rect, data["tip"])
        return over

    def text(self, surf, s, pos, color=TEXT, font=None, anchor="topleft") -> pygame.Rect:
        img = (font or self.f_text).render(str(s), True, color)
        r = img.get_rect(**{anchor: pos})
        surf.blit(img, r)
        return r

    def wrapped(self, surf, s, pos, width: int, color=TEXT, font=None, max_lines: int = 3) -> int:
        """Word-wrapped text; returns the y below it."""
        font = font or self.f_small
        lines, line = [], ""
        for word in str(s).split():
            trial = (line + " " + word).strip()
            if font.size(trial)[0] > width and line:
                lines.append(line)
                line = word
            else:
                line = trial
        lines.append(line)
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            lines[-1] = lines[-1][:-1] + "…"
        y = pos[1]
        for ln in lines:
            self.text(surf, ln, (pos[0], y), color, font)
            y += font.get_linesize()
        return y

    @staticmethod
    def fit_lines(font, s: str, width: int, max_lines: int) -> tuple[list[str], bool]:
        """A label in at most max_lines lines of at most `width` px: word-wrapped, and the last line cut with an ellipsis when it still
        does not fit (never a smaller font). Returns (lines, cut)."""
        s = str(s)
        if font.size(s)[0] <= width or not s.strip():
            return [s], False
        lines, line = [], ""
        for word in s.split():
            trial = (line + " " + word).strip()
            if font.size(trial)[0] > width and line:
                lines.append(line)
                line = word
            else:
                line = trial
        lines.append(line)
        cut = len(lines) > max_lines
        lines = lines[:max(1, max_lines)]
        if cut or font.size(lines[-1])[0] > width:
            last = lines[-1]
            while last and font.size(last + "…")[0] > width:
                last = last[:-1]
            lines[-1] = last.rstrip() + "…"
            cut = True
        return lines, cut

    def _label(self, surf, rect: pygame.Rect, label, color, font) -> bool:
        """A button's label centred in its rect, fitted by fit_lines (3.0 release review: labels used to run past their buttons, onto
        the next one, at larger text, in German and in a narrow window). Returns True when it had to be cut."""
        lh = font.get_linesize()
        lines, cut = self.fit_lines(font, label, rect.w - 12, max(1, (rect.h - 4) // lh))
        y = rect.centery - lh * len(lines) / 2
        for ln in lines:
            self.text(surf, ln, (rect.centerx, int(y + lh / 2)), color, font, "center")
            y += lh
        return cut

    def label_fits(self, label, width: int, height: int, font=None) -> bool:
        font = font or self.f_bold
        return not self.fit_lines(font, label, width - 12, max(1, (height - 4) // font.get_linesize()))[1]

    def subtitle(self, surf, rect, text: str, y: int | None = None, max_lines: int = 3) -> int:
        """A page's description line under its heading, wrapped to the panel (3.0 release review: most pages drew it as one line, which ran
        off a 860 px menu and at larger text). Returns the y below it, with a gap."""
        return self.wrapped(surf, text, (rect.x + 24, max(self.under_heading(rect), rect.y + 48 if y is None else y)), rect.w - 48, LABEL,
                            self.f_small, max_lines) + 8

    def under_heading(self, rect) -> int:
        """The first y below a page heading drawn at rect.y + 16, at any font (Segoe UI's heading is taller than DejaVu's)."""
        return rect.y + 16 + self.f_head.get_linesize() + 2

    def flow_buttons(self, surf, x: int, y: int, right: int, items, *, h: int | None = None, font=None, gap: int = 8,
                     label: str | None = None) -> int:
        """A row of choice buttons, each as wide as its label, wrapping onto more rows before `right`. items: (text, click, id, active,
        tip). With `label`, the label comes first and the buttons follow it on the same line when they fit there. Returns the bottom y."""
        font = font or self.f_small
        h = h or max(28, font.get_linesize() + 10)
        x0 = x
        if label:
            r = self.text(surf, label, (x, y + h // 2), TEXT, self.f_text, "midleft")
            widest = max((font.size(str(it[0]))[0] + 24 for it in items), default=0)
            if r.right + 12 + widest <= right:
                x0 = x = r.right + 12
            else:
                y += h + 4
        for text, click, ident, active, tip in items:
            w = min(font.size(str(text))[0] + 24, right - x0)
            if x + w > right and x > x0:
                x, y = x0, y + h + 6
            self.button(surf, (x, y, w, h), text, click, id=ident, active=active, tip=tip, font=font)
            x += w + gap
        return y + h

    def bw(self, label, min_w: int = 0, font=None, pad: int = 28) -> int:
        """A button width that fits its label on one line in the current font (3.0 release review: fixed widths cut labels at larger
        text with wider fonts, e.g. DejaVu Sans on Ubuntu)."""
        return max(int(min_w), (font or self.f_bold).size(str(label))[0] + pad)

    def button_row(self, surf, x: int, y: int, h: int, items, *, gap: int = 10, right: int | None = None) -> tuple[int, int]:
        """Buttons side by side, each at least its `w` and never narrower than its label; wraps before `right`. items: dicts with label,
        click and the button() keywords (id, style, enabled, tip, font, active) plus an optional minimum width `w`. Returns (x after the last,
        y below the last row)."""
        x0 = x
        for it in items:
            it = dict(it)
            label, click, w = it.pop("label"), it.pop("click"), it.pop("w", 0)
            bw = self.bw(label, w, it.get("font"))
            if right is not None and x + bw > right and x > x0:
                x, y = x0, y + h + 8
            self.button(surf, (x, y, bw, h), label, click, **it)
            x += bw + gap
        return x, y + h

    def button(self, surf, rect, label, click, *, id=None, style="normal", enabled=True, tip=None, active=False,
               font=None) -> bool:
        rect = pygame.Rect(rect)
        id = id or ("btn", label, rect.topleft)
        if tip is None and not self.label_fits(label, rect.w, rect.h, font):
            tip = str(label)                         # a cut label can still be read in full
        over = self._register(rect, "button", id=id, click=click, enabled=enabled, tip=tip)
        down = over and self.pressed == id
        base, hover = {"primary": (BTN_PRIMARY, BTN_PRIMARY_HOVER), "danger": (BTN_DANGER, BTN_DANGER_HOVER)}.get(
            style, (BTN, BTN_HOVER))
        fill = (32, 36, 46) if not enabled else BTN_DOWN if down else hover if over else base
        if active:
            fill = BTN_PRIMARY_HOVER if over else BTN_PRIMARY
        pygame.draw.rect(surf, fill, rect, border_radius=8)
        if (over and enabled) or active:
            pygame.draw.rect(surf, ACCENT if active else (120, 132, 156), rect, 1, border_radius=8)
        self._label(surf, rect, label, INK if enabled else DIM, font or self.f_bold)
        return over

    def toggle(self, surf, rect, value: bool, change, *, id, enabled=True, tip=None) -> None:
        rect = pygame.Rect(rect)
        over = self._register(rect, "button", id=id, click=lambda: change(not value), enabled=enabled, tip=tip)
        track = pygame.Rect(rect.x, rect.centery - 11, 44, 22)
        on_col = (40, 150, 200) if enabled else (40, 70, 90)
        pygame.draw.rect(surf, on_col if value else ((70, 76, 90) if over else (50, 56, 68)), track, border_radius=11)
        knob = (track.right - 11 if value else track.x + 11, track.centery)
        pygame.draw.circle(surf, INK if enabled else DIM, knob, 8)
        self.text(surf, tr("On") if value else tr("Off"), (track.right + 10, track.centery), TEXT if enabled else DIM,
                  self.f_small, "midleft")

    def segmented(self, surf, rect, labels, current: int, select, *, id, enabled=True, tip=None) -> None:
        rect = pygame.Rect(rect)
        n = len(labels)
        w = rect.w // n
        for i, lab in enumerate(labels):
            r = pygame.Rect(rect.x + i * w, rect.y, w - 4, rect.h)
            t = tip if tip is not None or self.label_fits(lab, r.w, r.h, self.f_small) else str(lab)
            over = self._register(r, "button", id=(id, i), click=(lambda i=i: select(i)), enabled=enabled, tip=t)
            on = i == current
            fill = (BTN_PRIMARY_HOVER if over else BTN_PRIMARY) if on else (BTN_HOVER if over and enabled else BTN)
            if not enabled:
                fill = (40, 60, 74) if on else (32, 36, 46)
            pygame.draw.rect(surf, fill, r, border_radius=6)
            if on:
                pygame.draw.rect(surf, ACCENT if enabled else DIM, r, 1, border_radius=6)
            self._label(surf, r, lab, INK if enabled else DIM, self.f_small)

    def slider(self, surf, rect, value: float, lo: float, hi: float, step: float, fmt: str, change, release, *, id,
               enabled=True, tip=None) -> None:
        rect = pygame.Rect(rect)
        box = pygame.Rect(rect.right - 76, rect.y, 76, rect.h)
        track = pygame.Rect(rect.x, rect.centery - 3, rect.w - 92, 6)
        grab = pygame.Rect(track.x - 8, rect.y, track.w + 16, rect.h)
        over = False
        if enabled:
            over = self._register(grab, "slider", id=id, track=track, lo=lo, hi=hi, step=step, change=change,
                                  release=release, tip=tip)
        frac = 0.0 if hi == lo else (value - lo) / (hi - lo)
        pygame.draw.rect(surf, (50, 56, 68), track, border_radius=3)
        pygame.draw.rect(surf, (40, 150, 200) if enabled else (40, 70, 90),
                         (track.x, track.y, max(3, int(track.w * frac)), track.h), border_radius=3)
        dragging = self.drag is not None and self.drag.get("id") == id
        pygame.draw.circle(surf, ACCENT if (over or dragging) else (INK if enabled else DIM),
                           (track.x + int(track.w * frac), track.centery), 9 if (over or dragging) else 7)
        pct = "%" in fmt

        def commit(v):
            change(v / 100.0 if pct else v)
            release()

        editing = self.edit is not None and self.edit["id"] == id
        shown = self.edit["text"] + ("|" if int(time.perf_counter() * 2) % 2 else "") if editing else fmt.format(value)
        if enabled:
            self._register(box, "edit", id=id, text=f"{value * 100:g}" if pct else f"{value:g}", commit=commit,
                           tip="Click to type a value, Enter to apply.")
        pygame.draw.rect(surf, (10, 12, 18) if editing else (30, 34, 44), box, border_radius=6)
        pygame.draw.rect(surf, ACCENT if editing else BORDER, box, 1, border_radius=6)
        self.text(surf, shown, box.center, INK if enabled else DIM, self.f_small, "center")

    def number_field(self, surf, rect, value: int, commit, *, id, tip=None) -> None:
        rect = pygame.Rect(rect)
        editing = self.edit is not None and self.edit["id"] == id
        self._register(rect, "edit", id=id, text=str(value), commit=lambda v: commit(int(v)), tip=tip)
        pygame.draw.rect(surf, (10, 12, 18) if editing else (30, 34, 44), rect, border_radius=6)
        pygame.draw.rect(surf, ACCENT if editing else BORDER, rect, 1, border_radius=6)
        shown = self.edit["text"] + ("|" if int(time.perf_counter() * 2) % 2 else "") if editing else str(value)
        self.text(surf, shown, (rect.x + 10, rect.centery), INK, self.f_small, "midleft")

    def text_field(self, surf, rect, value: str, commit, *, id, tip=None, limit: int = 24, compact: bool = False) -> None:
        """A one-line text box (names). Click, type, Enter commits, Esc cancels. Ctrl+V pastes (compact: drops all
        whitespace, for share codes that arrive wrapped over several lines)."""
        rect = pygame.Rect(rect)
        editing = self.edit is not None and self.edit["id"] == id
        self._register(rect, "edit", id=id, text=value, commit=commit, tip=tip, alpha=True, limit=limit, compact=compact)
        pygame.draw.rect(surf, (10, 12, 18) if editing else (30, 34, 44), rect, border_radius=6)
        pygame.draw.rect(surf, ACCENT if editing else BORDER, rect, 1, border_radius=6)
        shown = self.edit["text"] + ("|" if int(time.perf_counter() * 2) % 2 else "") if editing else value
        self.text(surf, shown, (rect.x + 10, rect.centery), INK, self.f_small, "midleft")

    def chip(self, surf, pos, label: str) -> pygame.Rect:
        img = self.f_small.render(label.upper(), True, (12, 14, 18))
        r = pygame.Rect(pos[0], pos[1], img.get_width() + 12, img.get_height() + 2)
        pygame.draw.rect(surf, TAG_COLORS.get(label, LABEL), r, border_radius=5)
        surf.blit(img, img.get_rect(center=r.center))
        return r

    # --- drawing -------------------------------------------------------------------------------------------------------
    def draw(self, surf: pygame.Surface, mouse, now: float | None = None) -> None:
        if not self.open:
            return
        self.fonts()
        self.hits = []
        self.tip = None
        self.mouse = mouse
        W, H = surf.get_size()
        veil = pygame.Surface((W, H), pygame.SRCALPHA)
        veil.fill((4, 5, 8, 196))
        surf.blit(veil, (0, 0))
        page = self.pages.get(self.screen) or getattr(self, f"_page_{self.screen}", None)
        if page is None:
            self.screen = "pause"
            page = self._page_pause
        # 3.0 release review: a narrow window gives a tall HUD (1280x2588 or 900x1820 logical); pages that do not scroll may use more of that height
        pw, ph = (440, min(650, H - 20)) if self.screen in ("pause", "confirm_quit") else (min(980, W - 40), min(680 if H < 1100 else 860, H - 30))
        if self.screen == "confirm_quit":
            ph = 250
        rect = pygame.Rect((W - pw) // 2, (H - ph) // 2, pw, ph)
        pygame.draw.rect(surf, PANEL, rect, border_radius=16)
        pygame.draw.rect(surf, BORDER, rect, 1, border_radius=16)
        try:
            if self.screen in self.pages:
                page(self, surf, rect, mouse)
            else:
                page(surf, rect)
        except Exception as e:                      # a broken page shows an error instead of crashing the game
            from kickthefly.core.crash import log
            if getattr(self, "_page_error", None) != (self.screen, str(e)):
                self._page_error = (self.screen, str(e))
                log.exception("menu page %s failed", self.screen)
            self.clip = None
            surf.set_clip(None)
            self.text(surf, f"This screen hit an error: {type(e).__name__}: {e}", (rect.centerx, rect.centery),
                      BAD, self.f_small, "center")
            self.button(surf, (rect.right - 164, rect.bottom - 58, 140, 42), tr("Back"), self.back, style="primary",
                        id=("error", "back"))
        if self.message and time.perf_counter() < self.message[1]:
            self.text(surf, self.message[0], (rect.centerx, rect.bottom + 10), self.message[2], self.f_small, "midtop")
        if self.tip is not None:
            self._draw_tip(surf, *self.tip)

    def _draw_tip(self, surf, anchor: pygame.Rect, text: str) -> None:
        W, H = surf.get_size()
        width = 360
        lines, line = [], ""
        for word in text.split():
            trial = (line + " " + word).strip()
            if self.f_small.size(trial)[0] > width - 20 and line:
                lines.append(line)
                line = word
            else:
                line = trial
        lines.append(line)
        lh = self.f_small.get_linesize()
        box = pygame.Rect(0, 0, width, 14 + lh * len(lines))
        box.topleft = (min(self.mouse[0] + 16, W - width - 8), self.mouse[1] + 20)
        if box.bottom > H - 8:
            box.bottom = self.mouse[1] - 10
        pygame.draw.rect(surf, (8, 10, 14), box, border_radius=8)
        pygame.draw.rect(surf, ACCENT, box, 1, border_radius=8)
        for i, ln in enumerate(lines):
            self.text(surf, ln, (box.x + 10, box.y + 7 + i * lh), TEXT, self.f_small)

    def _page_pause(self, surf, rect) -> None:
        cfg = self.host.cfg
        self.text(surf, tr("PAUSED"), (rect.centerx, rect.y + 22), INK, self.f_title, "midtop")
        lab = cfg.lab
        mode_title = "Pet" if getattr(cfg, "pet", False) else "Lab" if lab else "Play"
        self.text(surf, f"Kick the Fly {__version__}   ·   {mode_title} mode", (rect.centerx, rect.y + 76),
                  LABEL, self.f_small, "midtop")
        items = [("Resume", "resume", "primary", True, "Back to the fly (Esc)."),
                 ("Challenges" if not lab else "Lab tools", "challenges" if not lab else "lab", "normal", True,
                  "Games with a goal and a score." if not lab else
                  "Validation results, assays, repeated trials, data export and protocol files."),
                 ("Neurodex", "neurodex", "normal", True,
                  "The cell types you have discovered, with what the dataset says about each. Default key D."),
                 ("Settings", "settings", "normal", True, "Graphics, audio, brain, controls and accessibility."),
                 ("Mic and streamer", "live_inputs", "normal", True,
                  "Opt-in: sound from your microphone drives the fly's Johnston's organ, and Twitch viewers vote on what happens. "
                  "Both are off every time the game starts and show a red indicator while on."),
                 ("Fly arcade", "arcade", "normal", True,
                  "A tournament of flies (each brain steers and shoots in 1v1 duels) and fly racing with in-game points. Points are not money."),
                 ("Mini-papers", "minipapers", "normal", True,
                  "Short guided experiments that reproduce a classic fly paper: state a hypothesis, run it on the model and see your result next to what "
                  "the paper found."),
                 ("Share", "share", "normal", True,
                  "Make a short code for your surgery, loadout, protocol, challenge setup or Lab parameters, or import one "
                  "and see what it would change first."),
                 ("Save State", "save_state", "normal", True,
                  "Save the whole simulation: every neuron's voltage, the learned synapses, surgery, the room and the "
                  "flies."),
                 ("Load State", "load_state", "normal", True, "Go back to a saved moment."),
                 (f"Mode: {mode_title}", "toggle_mode", "normal", True,
                  "Switch between Play (the game), Lab (research tools), and Pet (one persistent fly). Saved in your settings."),
                 ("Quit", "quit", "danger", True, "Asks first. Training memory is saved.")]
        lh = self.f_text.get_linesize() if hasattr(self, "f_text") else 22                    # (fonts are made on the first draw)
        bw, bh, gap = 300, max(34, min(40, lh + 10)), 5                                          # 12 buttons since 3.0 day 5 (Mini-papers)
        y = rect.y + 108
        for label, action, style, enabled, tip in items:
            self.button(surf, (rect.centerx - bw // 2, y, bw, bh), tr(label),
                        (lambda a=action: self.host.menu_action(a)), style=style, enabled=enabled, tip=tr(tip),
                        id=("pause", action))
            y += bh + gap

    def _page_confirm_quit(self, surf, rect) -> None:
        self.text(surf, tr("Quit Kick the Fly?"), (rect.centerx, rect.y + 30), INK, self.f_head, "midtop")
        msg = tr("The fly's training memory and your settings are saved.")         # wrapped: the German line is wider than the dialog
        lines = Menu.fit_lines(self.f_small, msg, rect.w - 48, 3)[0]
        for i, ln in enumerate(lines):
            self.text(surf, ln, (rect.centerx, rect.y + 76 + i * self.f_small.get_linesize()), LABEL, self.f_small, "midtop")
        self.button(surf, (rect.centerx - 170, rect.bottom - 90, 160, 50), tr("Quit"),
                    lambda: self.host.menu_action("quit_now"), style="danger", id=("cq", "quit"))
        self.button(surf, (rect.centerx + 10, rect.bottom - 90, 160, 50), tr("Cancel"), self.back, id=("cq", "cancel"))

    def _page_settings(self, surf, rect) -> None:
        cfg, host = self.host.cfg, self.host
        self.text(surf, "SETTINGS", (rect.x + 24, rect.y + 16), INK, self.f_head)
        # 3.0 release review: equal-width tabs cut "Accessibility" (and "Barrierefreiheit") at larger text in a 860 px menu; each tab is now
        # as wide as its label needs, with the spare room shared out
        need = [self.f_bold.size(tr(t))[0] + 24 for t in TABS]
        spare = max(0, (rect.w - 48) - sum(need) - 8 * len(TABS)) // len(TABS)
        tx = rect.x + 24
        for tab, w in zip(TABS, need):
            self.button(surf, (tx, rect.y + 54, w + spare, 38), tr(tab),
                        (lambda t=tab: setattr(self, "tab", t)), active=tab == self.tab, id=("tab", tab))
            tx += w + spare + 8
        body = pygame.Rect(rect.x + 16, rect.y + 104, rect.w - 32, rect.h - 104 - 70)
        key = self._scroll_key()
        off = int(self.scroll.get(key, 0))
        self.clip = body
        prev_clip = surf.get_clip()
        surf.set_clip(body)
        y = body.y + 6 - off
        label_w = 250
        ctl_x = body.x + label_w + 150
        ctl_w = body.right - ctl_x - 16
        for s in SETTINGS:
            if s.tab != self.tab or not config.visible(s, host.three_d):
                continue
            on = config.enabled(s, host.three_d)
            # a choice whose labels do not fit their segments on one line gets a taller row, so they wrap instead of being cut
            extra, own_line = 0, False
            chips = ([s.tag] if s.tag else []) + (["RESTART"] if s.restart else []) + ([] if on else ["3D ONLY"])
            chips_end = max(body.x + 12 + self.f_text.size(tr(s.label))[0] + 10, body.x + label_w) + \
                sum(self.f_small.size(c.upper())[0] + 18 for c in chips)
            cx0 = max(ctl_x, chips_end + 8)              # 3.0 release review: two chips (RESTART, 3D ONLY) ran into the control
            if s.kind == "choice" and len(s.labels or s.options) <= 4:
                labs = s.labels or tuple(map(str, s.options))
                seg = (ctl_x + ctl_w - cx0) // len(labs) - 4
                if any(self.f_small.size(str(lb))[0] > seg - 12 for lb in labs):
                    extra = self.f_small.get_linesize()
                    if any(Menu.fit_lines(self.f_small, str(lb), seg - 12, 2)[1] for lb in labs):
                        own_line = True                 # not even two lines fit: the choice gets the row's full width, under its label
                        extra = 38 + self.f_small.get_linesize()
            row = pygame.Rect(body.x, y, body.w, 44 + extra)
            if row.collidepoint(self.mouse) and body.collidepoint(self.mouse):
                pygame.draw.rect(surf, ROW_HOVER, row, border_radius=8)
            tip = tr(s.tip) + ("" if on else "  (3D game only.)")
            ly = row.y + 22 if own_line else row.centery
            lr = self.text(surf, tr(s.label), (row.x + 12, ly), TEXT if on else DIM, self.f_text, "midleft")
            self._register(pygame.Rect(row.x, row.y, label_w + 150, row.h), "label", id=("label", s.key), tip=tip)
            cx = max(lr.right + 10, row.x + label_w)
            for chip in ([s.tag] if s.tag else []) + (["RESTART"] if s.restart else []) + ([] if on else ["3D ONLY"]):
                cx = self.chip(surf, (cx, ly - 10), chip).right + 6
            ctl = pygame.Rect(body.x + 24, row.y + 44, body.w - 40, 30 + self.f_small.get_linesize()) if own_line else pygame.Rect(cx0, row.y + 7, ctl_x + ctl_w - cx0, 30 + extra)
            v = cfg[s.key]
            if s.kind == "bool":
                self.toggle(surf, ctl, bool(v), lambda nv, k=s.key: host.set_setting(k, nv), id=s.key, enabled=on,
                            tip=tip)
            elif s.kind == "choice":
                labels = s.labels or tuple(map(str, s.options))
                if len(labels) <= 4:
                    self.segmented(surf, ctl, labels, s.options.index(v),
                                   lambda i, k=s.key, o=s.options: host.set_setting(k, o[i]), id=s.key, enabled=on,
                                   tip=tip)
                else:
                    i = s.options.index(v)
                    mid = max(152, max(self.f_bold.size(str(lb))[0] for lb in labels) + 16)   # the widest label fits between the arrows
                    self.button(surf, (ctl.x, ctl.y, 44, ctl.h), "<",
                                lambda k=s.key, o=s.options, i=i: host.set_setting(k, o[(i - 1) % len(o)]),
                                id=(s.key, "<"), enabled=on, tip=tip)
                    self.text(surf, labels[i], (ctl.x + 44 + mid // 2, ctl.centery), INK, self.f_bold, "center")
                    self.button(surf, (ctl.x + 44 + mid, ctl.y, 44, ctl.h), ">",
                                lambda k=s.key, o=s.options, i=i: host.set_setting(k, o[(i + 1) % len(o)]),
                                id=(s.key, ">"), enabled=on, tip=tip)
            elif s.kind == "float":
                self.slider(surf, ctl, float(v), s.lo, s.hi, s.step, s.fmt,
                            lambda nv, k=s.key: host.set_setting(k, nv, save=False),
                            lambda: host.set_setting(None, None), id=s.key, enabled=on, tip=tip)
            elif s.kind == "int":
                self.number_field(surf, (ctl.x, ctl.y, 180, ctl.h), int(v), lambda nv, k=s.key: host.set_setting(k, nv),
                                  id=s.key, tip=tip)
                self.button(surf, (ctl.x + 192, ctl.y, 110, ctl.h), "Random",
                            lambda k=s.key: host.set_setting(k, random.randrange(1, 2**31 - 1)), id=(s.key, "rand"),
                            tip="Pick a new random seed.", font=self.f_small)
            y += 48 + extra
        if self.tab == "Brain":
            y = self._engine_info(surf, body, y + 4, host)
        if self.tab == "Controls":
            y = self._keybinds(surf, body, y + 8)
        if self.tab == "Help":
            from kickthefly.ui import help_ui

            y = help_ui.help_tab(self, surf, body, y + 4)
        self.content_h[key] = max(0, y + off - body.bottom + 12)
        surf.set_clip(prev_clip)
        self.clip = None
        if self.content_h[key] > 0:                        # scroll bar
            frac = body.h / (body.h + self.content_h[key])
            bar_h = max(30, int(body.h * frac))
            by = body.y + int((body.h - bar_h) * (off / max(1, self.content_h[key])))
            pygame.draw.rect(surf, (60, 66, 80), (body.right - 6, by, 4, bar_h), border_radius=2)
        fy = rect.bottom - 58
        if self.tab != "Help":
            self.button(surf, (rect.x + 24, fy, self.bw("Reset to defaults", 200), 42), "Reset to defaults", lambda: self._reset_tab(),
                        id=("reset", self.tab), tip=f"Put every {self.tab} setting back to how the game ships.")
        self.button(surf, (rect.right - 164, fy, 140, 42), tr("Back"), self.back, style="primary", id=("settings", "back"))
        note = "Changes apply right away and are saved."
        left = rect.x + 24 + (self.bw("Reset to defaults", 200) + 16 if self.tab != "Help" else 0)       # clear of the Reset button
        nx = max(rect.centerx, left + self.f_small.size(note)[0] // 2)
        if nx + self.f_small.size(note)[0] // 2 < rect.right - 176:
            self.text(surf, note, (nx, fy + 21), LABEL, self.f_small, "center")

    def _engine_info(self, surf, body, y: int, host) -> int:
        """Which engine is running the brain now (3.1.0 task 2), what 'Auto' would try, and a note when a fallback happened."""
        from kickthefly.sim.connectome import backends

        lines = []
        try:
            flies = getattr(host, "flies", None) or []
            be = flies[0].brain.sim.backend if flies else None
            asked = str(host.cfg["brain.backend"])
            if be is not None:
                lines.append(f"Engine in use: {be.name}  ({be.device})")
                if asked not in ("auto", be.name) and not (asked in ("torch-cuda", "torch-rocm") and be.name.startswith("torch")):
                    lines.append(f"You chose {asked}, but it is not available here or failed; {be.name} runs instead.")
            else:
                lines.append("Engine in use: none yet (a fly has not been created)")
            vw = getattr(host, "view", None)
            if vw is not None and hasattr(vw, "engine_note"):
                lines.append(f"Brain view: {vw.engine_note}")
            chain = backends.auto_chain()
            lines.append(f"Auto tries: {' > '.join(chain)}   ({backends.auto_policy()} policy; pick an engine above to override)")
        except Exception as e:                       # the panel is information only: never let it break Settings
            lines.append(f"Engine information unavailable ({type(e).__name__})")
        self.text(surf, "ENGINE", (body.x + 12, y + 8), LABEL, self.f_small, "midleft")
        y += 22
        for ln in lines:
            self.text(surf, ln, (body.x + 12, y + 8), TEXT, self.f_small, "midleft")
            y += 22
        return y + 4

    def _reset_tab(self) -> None:
        changed = self.host.cfg.reset_tab(self.tab)
        for k in changed:
            self.host.set_setting(k, self.host.cfg[k] if k != "keys" else None, force=True)
        self.host.set_setting(None, None)
        self.flash(f"{self.tab} reset to defaults" if changed else f"{self.tab} already at defaults", GOOD)

    def _keybinds(self, surf, body, y) -> int:
        cfg = self.host.cfg
        k = self.text(surf, "KEYS", (body.x + 12, y), LABEL, self.f_small)
        y = max(y + 26, self.wrapped(surf, "Click a key, then press the new one (Esc cancels). A key that's taken swaps with that action.",
                                     (k.right + 16, y), body.right - 16 - k.right - 16, DIM, self.f_small, 2) + 8)
        # 3.0 release review: two columns only when the longest label and its key button fit in half the width (they overlapped at larger
        # text with wider fonts); otherwise one action per row
        need = max(self.f_small.size(label)[0] for _, label, _ in list(ACTIONS) + list(config.PAD_ACTIONS)) + 200
        ncol = 2 if need <= (body.w - 24) // 2 else 1
        col_w = (body.w - 24) // ncol
        conflicts = {a for acts in cfg.conflicts().values() for a in acts}
        for i, (action, label, _) in enumerate(ACTIONS):
            cx = body.x + 12 + (i % ncol) * col_w
            ry = y + (i // ncol) * 38
            active = self.capture == action
            dim = action in config.MOVEMENT_3D_ONLY and not self.host.three_d
            self.text(surf, label, (cx, ry + 16), DIM if dim else TEXT, self.f_small, "midleft")
            name = "press a key..." if active else (cfg.keys[action] or tr("unbound"))
            self.button(surf, (cx + col_w - 190, ry + 2, 170, 30), name, (lambda a=action: setattr(self, "capture", a)),
                        id=("key", action), active=active, style="danger" if action in conflicts else "normal",
                        font=self.f_small, tip=f"{label}: click, then press a key.")
        y += ((len(ACTIONS) + ncol - 1) // ncol) * 38 + 10
        if not self.host.three_d:
            return y
        pad = getattr(self.host, "pad", None)
        status = f"connected: {pad.name()[:32]}" if pad is not None and pad.pads else "none connected"
        self.text(surf, f"GAMEPAD ({status})", (body.x + 12, y), LABEL, self.f_small)
        y = self.wrapped(surf, "Click a binding, then press the button or push the stick. Backspace unbinds.",
                         (body.x + 12, y + 18), body.w - 24, DIM, self.f_small, 2) + 8
        for i, (action, label, _) in enumerate(config.PAD_ACTIONS):
            cx = body.x + 12 + (i % ncol) * col_w
            ry = y + (i // ncol) * 38
            active = self.capture == "pad:" + action
            self.text(surf, label, (cx, ry + 16), TEXT, self.f_small, "midleft")
            from kickthefly.game import gamepad

            name = "press or push..." if active else gamepad.pretty(cfg.pad[action])
            self.button(surf, (cx + col_w - 190, ry + 2, 170, 30), name,
                        (lambda a=action: setattr(self, "capture", "pad:" + a)), id=("pad", action), active=active,
                        font=self.f_small, tip=f"{label}: click, then press a gamepad button or push a stick.")
        return y + ((len(config.PAD_ACTIONS) + ncol - 1) // ncol) * 38


def draw_check(surf, center, size: int, color) -> None:
    """A check mark drawn with lines (fonts don't all have the ✓ glyph)."""
    x, y = center
    s = size
    pygame.draw.lines(surf, color, False, [(x - s * 0.45, y), (x - s * 0.1, y + s * 0.35), (x + s * 0.5, y - s * 0.4)],
                      max(2, s // 6))
