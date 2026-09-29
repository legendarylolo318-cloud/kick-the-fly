"""The first-launch tutorial (2.13): about a minute, skippable at any step, replayable from Settings > Help.

Five short steps on top of the running game, none of which changes the simulation:
  1 move       walk (3D: WASD or the left stick) or sweep the mouse across the room (2D)
  2 tool       use the tool in your hand
  3 brain      look at the brain panel: neurons that fire glow, and the touch you just made lit the touch neurons
  4 sugar      sugar is a reward: it fires the taste and PAM reward neurons and the fly walks over to eat
  5 loadout    the loadout editor (default key Q) chooses which tools are on the hotbar

Keyboard and mouse: Enter or the Next button skips a step, Backspace or the Skip button ends the tutorial.
Gamepad: A skips a step, Back/Select ends it. The card is a still panel: with Accessibility > Reduced flashing on, its
highlight around the brain panel does not pulse either way (it never flashes, it only breathes slowly when allowed).
"Shown once per install": Config.first_run["tutorial_done"] is set when it ends or is skipped; a config migrated from
before 2.13 counts as already onboarded. The game starts it from its run loop (Game.show_first_run_notices), never from
Game.__init__, so tests, headless runs and the playthrough bot never see it.
"""
from __future__ import annotations

import math
import time

import pygame

from kickthefly.core.i18n import tr
from kickthefly.ui import loadout_ui
from kickthefly.ui import menu as mu

STEPS = ("move", "tool", "brain", "sugar", "loadout")
BRAIN_SECONDS = 7.0            # how long the "watch the brain" step waits before it moves on by itself
MOVE_METERS = 1.0              # 3D: how far you walk
MOVE_PIXELS = 500.0            # 2D: how far the mouse travels


class Tutorial:
    def __init__(self, game, replay: bool = False):
        self.game = game
        self.replay = replay
        self.i = 0
        self.active = True
        self.skipped = False
        self.t_step = time.perf_counter()
        self.buttons: list[tuple[pygame.Rect, str]] = []
        self._enter_step()
        game.tutorial = self

    # --- the steps ----------------------------------------------------------------------------------------------------
    @property
    def step(self) -> str:
        return STEPS[min(self.i, len(STEPS) - 1)]

    def _enter_step(self) -> None:
        g = self.game
        self.t_step = time.perf_counter()
        self.tool_uses0 = getattr(g, "tool_uses", 0)
        self.travel = 0.0
        self.last_mouse = pygame.mouse.get_pos()
        self.player0 = tuple(g.player.pos) if getattr(g, "three_d", False) and hasattr(g, "player") else None
        self.saw_editor = False
        if self.step == "sugar":
            g.select_tool("sugar")                 # in the hand whatever the loadout: the wheel or the editor can swap it
        elif self.step == "tool" and g.tool_name() not in ("hand", "flick", "swatter"):
            g.select_tool("hand")

    def next(self) -> None:
        self.i += 1
        if self.i >= len(STEPS):
            self.finish()
        else:
            self._enter_step()

    def finish(self, skipped: bool = False) -> None:
        self.skipped = skipped
        self.active = False
        self.game.tutorial = None
        cfg = self.game.cfg
        if not cfg.first_run.get("tutorial_done"):
            cfg.first_run["tutorial_done"] = True
            cfg.dirty = True
            cfg.save()

    def poll(self) -> bool:
        """Has this step been done? Advances when it has. Called every frame from draw(); tests call it directly."""
        if not self.active:
            return False
        g = self.game
        s = self.step
        done = False
        if s == "move":
            if self.player0 is not None:
                p = g.player.pos
                done = math.dist((p[0], p[1]), (self.player0[0], self.player0[1])) > MOVE_METERS    # pos is (x, z)
            else:
                m = pygame.mouse.get_pos()
                self.travel += math.dist(m, self.last_mouse)
                self.last_mouse = m
                done = self.travel > MOVE_PIXELS
        elif s == "tool":
            done = getattr(g, "tool_uses", 0) > self.tool_uses0
        elif s == "brain":
            done = time.perf_counter() - self.t_step > BRAIN_SECONDS
        elif s == "sugar":
            done = getattr(g, "tool_uses", 0) > self.tool_uses0 and getattr(g, "last_tool_used", "") in ("sugar", "fruit")
        elif s == "loadout":
            if g.menu.open and g.menu.screen == "loadout":
                self.saw_editor = True
            elif self.saw_editor and not g.menu.open:
                done = True
        if done:
            self.next()
        return done

    # --- input --------------------------------------------------------------------------------------------------------
    def handle(self, ev) -> bool:
        """True if the tutorial used the event (its two keys and its two buttons); everything else reaches the game."""
        if ev.type == pygame.KEYDOWN and ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self.next()
            return True
        if ev.type == pygame.KEYDOWN and ev.key == pygame.K_BACKSPACE:
            self.finish(skipped=True)
            return True
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and not getattr(self.game, "look", False):
            pos = getattr(ev, "pos", (0, 0))           # (while the 3D mouse is captured for looking, a click uses the tool)
            for r, what in self.buttons:
                if r.collidepoint(pos):
                    self.next() if what == "next" else self.finish(skipped=True)
                    return True
        return False

    def pad_event(self, down) -> bool:
        """Gamepad: A skips the step, Back/Select ends the tutorial. True if one of them was pressed."""
        if "up" in down:
            self.next()
            return True
        if "big_view" in down:
            self.finish(skipped=True)
            return True
        return False

    # --- text ---------------------------------------------------------------------------------------------------------
    def _text(self) -> tuple[str, str]:
        g = self.game
        keys = g.cfg.keys
        pad = getattr(getattr(g, "pad", None), "connected", False)
        s = self.step
        if s == "move":
            if getattr(g, "three_d", False):
                pad_hint = " " + tr("Or push the left stick.") if pad else ""
                return (tr("Move around"), tr("Walk with {f}{l}{b}{r}.", f=keys["forward"].upper(), l=keys["left"].upper(),
                                              b=keys["back"].upper(), r=keys["right"].upper()) + pad_hint + " "
                        + tr("Click the room first to look around."))
            return tr("Move around"), tr("Sweep the mouse across the room. The fly lives at the bottom; everything you "
                                         "do happens under your cursor.")
        if s == "tool":
            pad_hint = " " + tr("Or pull the right trigger.") if pad else ""
            return (tr("Use a tool"), tr("Click on the fly with the tool in your hand. The hand grabs and throws it, a "
                                         "gentle way to start.") + pad_hint)
        if s == "brain":
            return (tr("Watch its brain"), tr("The panel is the fly's real brain, a simulation of the MaleCNS connectome. "
                                              "Neurons that fire glow. What you just did fired its touch neurons, and "
                                              "the neurons that react to them light up a moment later."))
        if s == "sugar":
            return (tr("Sugar is a reward"), tr("Sugar is in your hand. Click near the fly to drop some: it fires the "
                                                "taste neurons and the PAM reward neurons, then the fly walks over "
                                                "and eats."))
        return (tr("Choose your tools"), tr("Press {key} to open the loadout editor: pick which tools are on the hotbar "
                                            "and in what order. Esc closes it. Hold {wheel} for the tool wheel, which "
                                            "reaches every tool.", key=keys["loadout"].upper(),
                                            wheel=keys["tool_wheel"].upper()))

    # --- drawing ------------------------------------------------------------------------------------------------------
    def draw(self, surf) -> None:
        if not self.active:
            return
        self.poll()
        if not self.active:
            return
        g = self.game
        cfg = g.cfg
        pal = loadout_ui.palette(cfg)
        from kickthefly.game import kick_the_fly as k2
        f_b, f_s, f_t = g.f_bold, g.f_small, g.f_text
        title, body = self._text()
        w = 560
        pad = 14
        lines = self._wrap(body, f_t, w - 2 * pad)
        lh = f_t.get_linesize()
        h = 44 + lh * len(lines) + 56
        card = pygame.Rect(0, 0, w, h)
        card.midtop = (k2.PLAY_W // 2, 128)
        veil = pygame.Surface(card.size, pygame.SRCALPHA)
        pygame.draw.rect(veil, (8, 10, 16, 236), veil.get_rect(), border_radius=12)
        surf.blit(veil, card)
        pygame.draw.rect(surf, pal["select"], card, 2, border_radius=12)
        surf.blit(f_s.render(tr("Step {i} of {n}", i=self.i + 1, n=len(STEPS)) + ("  ·  " + tr("replay") if self.replay else ""),
                             True, k2.LABEL), (card.x + pad, card.y + 8))
        surf.blit(f_b.render(title, True, k2.INK), (card.x + pad, card.y + 26))
        y = card.y + 50
        for ln in lines:
            surf.blit(f_t.render(ln, True, k2.TEXT), (card.x + pad, y))
            y += lh
        self.buttons = []
        by = card.bottom - 40
        for i, (label, what) in enumerate(((tr("Next") + "  (Enter)", "next"), (tr("Skip tutorial") + "  (Backspace)", "skip"))):
            img = f_s.render(label, True, k2.INK)
            r = pygame.Rect(card.x + pad + i * 190, by, img.get_width() + 24, 30)
            pygame.draw.rect(surf, (44, 50, 64) if what == "skip" else (40, 110, 150), r, border_radius=8)
            surf.blit(img, img.get_rect(center=r.center))
            self.buttons.append((r, what))
        # what to look at: an outline around the brain panel while the brain step is on. A slow swell of the line
        # width, none at all with reduced flashing.
        if self.step == "brain" and getattr(g, "view_rect", None) is not None and g.view_rect.w > 0:
            grow = 0 if cfg["access.reduced_flashing"] else int(1 + 1.5 * (1 + math.sin(time.perf_counter() * 2)) / 2)
            pygame.draw.rect(surf, pal["select"], g.view_rect.inflate(8, 8), 3 + grow, border_radius=6)

    @staticmethod
    def _wrap(text: str, font, width: int) -> list[str]:
        lines, line = [], ""
        for word in text.split():
            trial = (line + " " + word).strip()
            if font.size(trial)[0] > width and line:
                lines.append(line)
                line = word
            else:
                line = trial
        lines.append(line)
        return lines
