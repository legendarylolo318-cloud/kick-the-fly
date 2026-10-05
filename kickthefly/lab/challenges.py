"""Play mode challenges: three lab assays turned into games with a goal and a score.

  Teach it to pick the right door   T-maze olfactory conditioning (assays.tmaze_fly, the same steps and choice rule)
  How close can you sneak?          looming escape: the fly's giant fiber decides when it has seen you
  Find its sweet tooth              sugar response: the weakest sugar that still makes its proboscis motor neuron fire

The T-maze challenge trains the fly's real mushroom body synapses, then puts them back as they were when it ends, so
a practice round never changes how the fly treats your tools. Best scores are saved in scores.json in the data folder.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pygame

from kickthefly.lab import assays
from kickthefly.core import paths

INK, TEXT, LABEL, DIM = (240, 243, 248), (205, 212, 224), (130, 142, 160), (80, 88, 102)
AMBER, ACCENT, GOOD, BAD = (255, 176, 64), (86, 214, 255), (90, 200, 120), (230, 90, 80)
ODOR_COLORS = {"odor_a": (255, 150, 60), "odor_b": (180, 120, 255)}
ODOR_NAMES = {"odor_a": "orange smell", "odor_b": "purple smell"}

INFO = (
    ("tmaze", "Teach it to pick the right door",
     "Choose which door gives a zap. Train the fly, then watch it choose a door 10 times.", "right choices", "high"),
    ("sneak", "How close can you sneak?",
     "Creep up on the fly. Move fast and its escape neuron fires and it dodges. How close can you get?",
     "fly lengths away", "low"),
    ("sweet", "Find its sweet tooth",
     "Offer sugar at different strengths. Find the weakest sugar it still reaches for, in 8 tries.", "% sugar", "low"),
    ("reverse_surgery", "Mystery defect (Reverse surgery)",
     "One brain circuit is turned off! Test the fly with tools, ask for hints, and deduce what's missing.",
     "stars", "high"),
    ("puppeteer", "Puppeteer",
     "You cannot touch the fly. Steer it only with the optogenetics laser and by switching its real neurons on and off: ten puzzles, each with a par and the circuit that solves it.",
     "levels", "high"),
    ("contraption", "Contraption builder",
     "A sandbox: place ramps, dominoes, springs, fans, lamps, sugar, tool triggers, buttons and timers, then run the machine against the fly.",
     "runs", "high"),
    ("predict", "Predict the move (Motor readouts)",
     "A descending motor spike surge flashes on the monitor. Can you predict the fly's move before it triggers?",
     "correct predictions", "high"),
)


def scores_path():
    return paths.get().data_dir / "scores.json"


def load_scores() -> dict:
    try:
        return json.loads(scores_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def record_score(key: str, value: float, better: str) -> bool:
    """Save a score if it beats the best. Returns True for a new best."""
    s = load_scores()
    old = s.get(key)
    new_best = old is None or (value > old if better == "high" else value < old)
    if new_best:
        s[key] = value
        try:
            p = scores_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(s, indent=1), encoding="utf-8")
        except OSError:
            pass
    return new_best


def stars(key: str, value: float) -> int:
    if key == "tmaze":
        return 3 if value >= 9 else 2 if value >= 7 else 1 if value >= 5 else 0
    if key == "sneak":
        return 3 if value <= 1.0 else 2 if value <= 2.0 else 1 if value <= 3.5 else 0
    if key == "puppeteer":                                   # levels finished, of ten
        return 3 if value >= 10 else 2 if value >= 7 else 1 if value >= 3 else 0
    if key == "contraption":                                 # a sandbox: no stars, the count of runs is the whole score
        return 0
    if key in ("reverse_surgery", "mystery"):
        return 3 if value >= 3 else 2 if value >= 2 else 1 if value >= 1 else 0
    if key == "predict":
        return 3 if value >= 8 else 2 if value >= 6 else 1 if value >= 4 else 0
    return 3 if value <= 10 else 2 if value <= 25 else 1 if value <= 50 else 0


def draw_stars(surf, center, n: int, size: int = 14) -> None:
    for i in range(3):
        cx = center[0] + (i - 1) * size * 2.4
        pts = []
        for k in range(10):
            r = size if k % 2 == 0 else size * 0.45
            a = -math.pi / 2 + k * math.pi / 5
            pts.append((cx + r * math.cos(a), center[1] + r * math.sin(a)))
        pygame.draw.polygon(surf, AMBER if i < n else (60, 64, 76), pts)


class Button:
    def __init__(self, rect, label, action, enabled=True, style="normal"):
        self.rect, self.label, self.action, self.enabled, self.style = pygame.Rect(rect), label, action, enabled, style

    def draw(self, game, surf, mouse):
        over = self.enabled and self.rect.collidepoint(mouse)
        base = {"primary": (40, 110, 150), "danger": (130, 44, 44)}.get(self.style, (44, 50, 64))
        fill = tuple(min(255, c + 22) for c in base) if over else base
        if not self.enabled:
            fill = (32, 36, 46)
        pygame.draw.rect(surf, fill, self.rect, border_radius=8)
        if over:
            pygame.draw.rect(surf, (120, 132, 156), self.rect, 1, border_radius=8)
        game._text(surf, self.label, self.rect.center, INK if self.enabled else DIM, game.f_bold, "center")


class Challenge:
    key = ""
    overlay = True                    # a panel that blocks tool use

    def __init__(self, game):
        self.game = game
        self.buttons: list[Button] = []
        self.done = False

    @property
    def slot(self):
        return self.game.flies[self.game.focus]

    def update(self, now: float) -> None:
        pass

    def on_reaction(self, kind: str, slot) -> None:
        pass

    def click(self, pos) -> bool:
        for b in self.buttons:
            if b.enabled and b.rect.collidepoint(pos):
                self.game.sound.play("click")
                b.action()
                return True
        return self.overlay

    def end(self) -> None:
        self.game.challenge = None

    def panel(self, surf, title: str, subtitle: str, h: int = 560) -> pygame.Rect:
        from kickthefly.game.kick_the_fly import PLAY_W, H
        g = self.game
        panel = pygame.Rect(40, 30, min(PLAY_W - 80, 820), min(H - 60, h))
        veil = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        veil.fill((4, 5, 8, 140))
        surf.blit(veil, (0, 0))
        pygame.draw.rect(surf, (18, 21, 28), panel, border_radius=16)
        pygame.draw.rect(surf, (52, 60, 76), panel, 1, border_radius=16)
        g._text(surf, title.upper(), (panel.x + 22, panel.y + 14), INK, g.f_title)
        g._text(surf, subtitle, (panel.x + 24, panel.y + 58), LABEL, g.f_text)
        return panel

    def draw_buttons(self, surf, mouse) -> None:
        for b in self.buttons:
            b.draw(self.game, surf, mouse)


# --- Teach it to pick the right door ----------------------------------------------------------------------------------
class TMaze(Challenge):
    key = "tmaze"
    SPEED = 3.0
    CYCLES = 6
    TRIALS = 10

    @property
    def borrows_memory(self) -> bool:
        """While training, the fly's memory holds practice learning that must never be saved to disk."""
        return self.saved_memory is not None

    def __init__(self, game):
        super().__init__(game)
        self.phase = "choose"
        self.zap: str | None = None
        self.choices: list[tuple[str, bool]] = []
        self.anim = 0.0
        self.saved_memory = None
        self.cycle = 0
        self.until = 0
        self.drive: dict[str, float] = {}
        self.rng = np.random.default_rng(int(game.brain.seed) + 424242)

    @property
    def speed(self) -> float:
        return self.SPEED if self.phase in ("naive", "train", "test") else 1.0

    def _snapshot_memory(self) -> None:
        br = self.game.brain
        self.saved_memory = (br.run(__name__, "memory_snapshot") if getattr(br, "remote", False) else memory_snapshot(br))

    def _restore_memory(self) -> None:
        if self.saved_memory is None:
            return
        br = self.game.brain
        if getattr(br, "remote", False):                # 3.1.0 review: the memory lives in the brain process
            br.run(__name__, "memory_restore", self.saved_memory)
        else:
            memory_restore(br, self.saved_memory)
        self.saved_memory = None

    def pick(self, door: str) -> None:
        br = self.game.brain
        if br.memory is None:
            return
        self.zap = door
        self._snapshot_memory()
        self.phase, self.stage = "naive", 0
        self.until = br.steps + 240

    def end(self) -> None:
        self._restore_memory()
        super().end()

    def update(self, now: float) -> None:
        br = self.game.brain
        if self.phase in ("choose", "done") or br.memory is None:
            return
        plus = self.zap
        minus = "odor_b" if plus == "odor_a" else "odor_a"
        step = br.steps
        mem = br.memory
        if self.phase == "naive":                       # the fly smells each door once so it knows the smells
            odor = plus if self.stage == 0 else minus
            br.poke("scent", odor, 0.5)
            if self.game.frame % 3 == 0:
                mem.observe(odor, br.sim.activity.rates())
            if step >= self.until:
                self.stage += 1
                self.until = step + 240
                if self.stage == 2:
                    self.phase, self.stage, self.cycle = "train", 0, 0
                    self.until = step + 240
            return
        if self.phase == "train":                       # the same schedule as assays.tmaze_fly
            schedule = ((plus, False, 240), (plus, True, 200), (None, False, 260), (minus, False, 440), (None, False, 260))
            odor, shock, _ = schedule[self.stage]
            if odor:
                br.poke("scent", odor, 0.5)
                if self.game.frame % 3 == 0:
                    mem.observe(odor, br.sim.activity.rates())
            if shock:
                br.poke("punish", None, 1.0)
                br.poke("legs", "L", 0.6)
                br.poke("legs", "R", 0.6)
            if step >= self.until:
                self.stage = (self.stage + 1) % len(schedule)
                if self.stage == 0:
                    self.cycle += 1
                    if self.cycle >= self.CYCLES:
                        self.phase, self.stage, self.choices = "test", 0, []
                self.until = step + schedule[self.stage][2]
            return
        if self.phase == "test":
            order = (plus, minus) if len(self.choices) % 2 == 0 else (minus, plus)
            if self.stage < 2:                          # smell each door
                odor = order[self.stage]
                br.poke("scent", odor, 0.5)
                if self.game.frame % 3 == 0:
                    mem.observe(odor, br.sim.activity.rates())
                if step >= self.until:
                    fear, like = mem.memory_of(odor)
                    self.drive[odor] = like - fear
                    self.stage += 1
                    self.until = step + (60 if self.stage < 2 else 1)
            elif self.stage == 2:                       # choose (assays.tmaze_fly's rule)
                noise = 0.08
                pick_plus = (self.drive[plus] + self.rng.normal(0, noise) > self.drive[minus] + self.rng.normal(0, noise))
                door = plus if pick_plus else minus
                self.choices.append((door, not pick_plus))
                self.anim, self.stage = now, 3
                self.game.sound.play("pop" if not pick_plus else "zap", 0.5)
            elif now - self.anim > 0.9:
                if len(self.choices) >= self.TRIALS:
                    self.phase = "done"
                    right = sum(ok for _, ok in self.choices)
                    self.best = record_score("tmaze", right, "high")
                    self._restore_memory()
                else:
                    self.stage, self.until = 0, step + 60

    def draw(self, surf, now, mouse) -> None:
        g = self.game
        p = self.panel(surf, "Teach it to pick the right door", "One door zaps. Can the fly learn which one?", 580)
        self.buttons = []
        cx = p.centerx
        doors = {"odor_a": pygame.Rect(p.x + 120, p.y + 120, 150, 210), "odor_b": pygame.Rect(p.right - 270, p.y + 120, 150, 210)}
        for odor, r in doors.items():
            col = ODOR_COLORS[odor]
            pygame.draw.rect(surf, tuple(c // 3 for c in col), r, border_radius=12)
            pygame.draw.rect(surf, col, r, 3, border_radius=12)
            for k in range(3):                               # wavy smell lines
                pts = [(r.centerx - 30 + x, r.y + 40 + k * 22 + 5 * math.sin(now * 3 + x / 8 + k)) for x in range(0, 61, 6)]
                pygame.draw.lines(surf, col, False, pts, 2)
            g._text(surf, ODOR_NAMES[odor], (r.centerx, r.bottom + 8), col, g.f_bold, "midtop")
            if self.zap == odor:
                g._text(surf, "ZAP", (r.centerx, r.bottom - 40), AMBER, g.f_head, "center")
        # the fly icon at the choice point, walking to its latest choice
        fx, fy = cx, p.y + 330
        if self.phase == "test" and self.stage == 3 and self.choices:
            door = doors[self.choices[-1][0]]
            e = min(1.0, (now - self.anim) / 0.7)
            fx, fy = fx + (door.centerx - fx) * e, fy + (door.bottom - 20 - fy) * e
        pygame.draw.ellipse(surf, (70, 60, 40), (fx - 14, fy - 9, 28, 18))
        pygame.draw.circle(surf, (170, 40, 40), (int(fx + 12), int(fy - 2)), 5)
        y = p.y + 390
        if self.phase == "choose":
            g._text(surf, "Pick the door that zaps the fly:", (cx, y), TEXT, g.f_text, "midtop")
            mem_ok = g.brain.memory is not None
            self.buttons = [Button((p.x + 120, y + 34, 150, 44), "Orange zaps", lambda: self.pick("odor_a"), mem_ok),
                            Button((p.right - 270, y + 34, 150, 44), "Purple zaps", lambda: self.pick("odor_b"), mem_ok)]
        elif self.phase in ("naive", "train"):
            total = self.CYCLES
            frac = (self.cycle + self.stage / 5) / total if self.phase == "train" else 0.0
            label = "Letting it smell both doors..." if self.phase == "naive" else \
                f"Training {self.cycle + 1}/{total}: " + ("smell + zap" if self.stage == 1 else "smell" if self.stage in (0, 3) else "rest")
            g._text(surf, label, (cx, y), TEXT, g.f_text, "midtop")
            pygame.draw.rect(surf, (30, 36, 48), (p.x + 60, y + 32, p.w - 120, 12), border_radius=6)
            pygame.draw.rect(surf, AMBER, (p.x + 60, y + 32, max(10, int((p.w - 120) * frac)), 12), border_radius=6)
        elif self.phase in ("test", "done"):
            right = sum(ok for _, ok in self.choices)
            g._text(surf, f"Choices: {len(self.choices)}/{self.TRIALS}    right door: {right}", (cx, y), INK, g.f_bold, "midtop")
            for i, (_, ok) in enumerate(self.choices):
                pygame.draw.circle(surf, GOOD if ok else BAD, (cx - 9 * 22 // 2 + i * 22, y + 40), 8)
            if self.phase == "done":
                draw_stars(surf, (cx, y + 84), stars("tmaze", right))
                msg = "New best!" if getattr(self, "best", False) else f"Best: {load_scores().get('tmaze', right):.0f}/10"
                g._text(surf, msg, (cx, y + 106), AMBER, g.f_text, "midtop")
                self.buttons.append(Button((cx - 200, p.bottom - 64, 180, 44), "Play again", lambda: self.game.start_challenge("tmaze"),
                                           style="primary"))
        close = "Close" if self.phase in ("choose", "done") else "Stop"
        self.buttons.append(Button((p.right - 160 if self.phase != "done" else cx + 20, p.bottom - 64, 140, 44), close, self.end))
        self.draw_buttons(surf, mouse)


# --- How close can you sneak? -----------------------------------------------------------------------------------------
class Sneak(Challenge):
    key = "sneak"
    overlay = False

    def __init__(self, game):
        super().__init__(game)
        self.closest = math.inf
        self.state = "armed"               # armed -> sneaking -> result
        self.result: tuple[str, float] | None = None
        self.result_t = 0.0
        self.hits0 = game.flies[game.focus].hits

    def update(self, now: float) -> None:
        g = self.game
        slot = self.slot
        d = g.sneak_distance(slot)
        airborne = now < slot.fly.escape_until              # (fly.flying stays true after its first flight)
        if self.state == "result":
            if now - self.result_t > 2.5 and d > 3.0 and not airborne and not slot.fly.dead:
                self.state, self.closest, self.hits0 = "armed", math.inf, slot.hits
            return
        if slot.fly.dead:
            return
        if self.state == "armed" and d < 3.0 and not airborne:
            self.state = "sneaking"
        if self.state == "sneaking":
            self.closest = min(self.closest, d)
            if slot.hits > self.hits0 or d < 0.35:
                self._finish("caught", 0.0 if d < 0.35 else self.closest, now)

    def on_reaction(self, kind: str, slot) -> None:
        if kind == "DODGE" and slot is self.slot and self.state == "sneaking":
            self._finish("seen", self.closest, self.game.clock.now)

    def _finish(self, how: str, dist: float, now: float) -> None:
        self.state, self.result, self.result_t = "result", (how, dist), now
        self.best = record_score("sneak", round(dist, 2), "low")
        self.game.sound.play("dodge" if how == "seen" else "yum", 0.6)

    def draw(self, surf, now, mouse) -> None:
        from kickthefly.game.kick_the_fly import PLAY_W
        g = self.game
        w, h = 330, 120
        box = pygame.Rect(PLAY_W // 2 - w // 2, 120, w, h)
        card = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(card, (10, 12, 18, 200), card.get_rect(), border_radius=12)
        surf.blit(card, box)
        g._text(surf, "HOW CLOSE CAN YOU SNEAK?", (box.x + 14, box.y + 8), AMBER, g.f_bold)
        best = load_scores().get("sneak")
        self.buttons = [Button((box.right - 70, box.y + 6, 60, 26), "Quit", self.end)]
        if self.state == "armed":
            g._text(surf, "Back off, then creep toward the fly.", (box.x + 14, box.y + 38), TEXT, g.f_text)
        elif self.state == "sneaking":
            d = g.sneak_distance(self.slot)
            g._text(surf, f"{d:4.1f} fly lengths away", (box.x + 14, box.y + 36), INK, g.f_head)
            g._text(surf, f"closest {self.closest:.1f}", (box.x + 14, box.y + 66), TEXT, g.f_text)
        elif self.result:
            how, dist = self.result
            msg = "You touched it before it saw you!" if how == "caught" else f"It saw you at {dist:.1f} fly lengths"
            g._text(surf, msg, (box.x + 14, box.y + 36), GOOD if how == "caught" else INK, g.f_bold)
            draw_stars(surf, (box.x + 70, box.y + 76), stars("sneak", dist), 10)
            if getattr(self, "best", False):
                g._text(surf, "New best!", (box.x + 140, box.y + 68), AMBER, g.f_text)
        if best is not None:
            g._text(surf, f"best {best:.1f}", (box.right - 14, box.bottom - 24), LABEL, g.f_small, "topright")
        self.draw_buttons(surf, mouse)


# --- Find its sweet tooth ---------------------------------------------------------------------------------------------
class Sweet(Challenge):
    key = "sweet"
    TRIES = 8

    def __init__(self, game):
        super().__init__(game)
        self.dose = 50
        self.phase = "ready"
        self.offers: list[tuple[int, bool, float]] = []
        self.samples: dict[str, list] = {"before": [], "during": []}
        self.until = 0
        self.dragging = False

    def offer(self) -> None:
        br = self.game.brain
        self.phase, self.samples = "before", {"before": [], "during": []}
        self.until = br.steps + 200

    def update(self, now: float) -> None:
        br = self.game.brain
        if self.phase == "before":
            self.samples["before"].append(br.hz("proboscis"))
            if br.steps >= self.until:
                self.phase, self.until = "during", br.steps + 200
        elif self.phase == "during":
            if self.dose > 0:
                br.poke("sweet", None, 0.5, recruit=self.dose / 100)
            self.samples["during"].append(br.hz("proboscis"))
            if br.steps >= self.until:
                b = float(np.mean(self.samples["before"])) if self.samples["before"] else 0.0
                d = float(np.mean(self.samples["during"])) if self.samples["during"] else 0.0
                ratio = d / max(b, 1.0)
                ext = ratio >= assays.PER_RATIO
                self.offers.append((self.dose, ext, ratio))
                self.phase, self.shown = "shown", now
                self.game.sound.play("yum" if ext else "click", 0.6)
                if ext:
                    self.game.on_reaction("PROBOSCIS", self.slot)
                    self.game.note(f"PROBOSCIS MN9 x{ratio:.1f} to {self.dose}% sugar")
        elif self.phase == "shown" and now - self.shown > 1.4:
            if len(self.offers) >= self.TRIES:
                self.phase = "done"
                good = [d for d, ext, _ in self.offers if ext]
                self.score = min(good) if good else None
                self.best = self.score is not None and record_score("sweet", self.score, "low")
            else:
                self.phase = "ready"

    def click(self, pos) -> bool:
        if self.phase == "ready" and self.track.inflate(0, 24).collidepoint(pos):
            frac = (pos[0] - self.track.x) / self.track.w
            self.dose = int(round(np.clip(frac, 0, 1) * 20)) * 5
            return True
        return super().click(pos)

    def draw(self, surf, now, mouse) -> None:
        g = self.game
        p = self.panel(surf, "Find its sweet tooth", "Offer sugar. Find the weakest sugar it still reaches for.", 520)
        cx = p.centerx
        self.buttons = []
        g._text(surf, f"Sugar strength: {self.dose}%", (p.x + 60, p.y + 110), INK, g.f_head)
        self.track = pygame.Rect(p.x + 60, p.y + 160, p.w - 120, 8)
        pygame.draw.rect(surf, (50, 56, 68), self.track, border_radius=4)
        pygame.draw.rect(surf, (255, 150, 190), (self.track.x, self.track.y, int(self.track.w * self.dose / 100), 8), border_radius=4)
        pygame.draw.circle(surf, INK, (self.track.x + int(self.track.w * self.dose / 100), self.track.centery), 10)
        g._text(surf, "click the bar to set the strength", (p.x + 60, p.y + 178), DIM, g.f_small)
        # proboscis icon
        head = (cx, p.y + 270)
        pygame.draw.circle(surf, (120, 90, 50), head, 34)
        pygame.draw.circle(surf, (190, 40, 40), (head[0] - 18, head[1] - 10), 9)
        pygame.draw.circle(surf, (190, 40, 40), (head[0] + 18, head[1] - 10), 9)
        ext = self.phase == "shown" and self.offers and self.offers[-1][1]
        length = 46 if ext else 10
        pygame.draw.line(surf, (150, 110, 70), (head[0], head[1] + 26), (head[0], head[1] + 26 + length), 8)
        if self.phase in ("before", "during"):
            g._text(surf, "tasting..." if self.phase == "during" else "watching...", (cx, p.y + 350), TEXT, g.f_text, "midtop")
        elif self.phase == "shown":
            d, e, r = self.offers[-1]
            g._text(surf, "It reached for it!" if e else "Not interested.", (cx, p.y + 350), GOOD if e else LABEL, g.f_bold, "midtop")
        row = "  ".join(f"{d}%{'+' if e else '-'}" for d, e, _ in self.offers)
        g._text(surf, f"Tries {len(self.offers)}/{self.TRIES}:  {row}", (p.x + 30, p.bottom - 110), TEXT, g.f_text)
        if self.phase == "done":
            if self.score is None:
                g._text(surf, "It never reached for the sugar this time.", (cx, p.y + 384), INK, g.f_bold, "midtop")
            else:
                g._text(surf, f"Weakest sugar it reached for: {self.score}%", (cx, p.y + 384), INK, g.f_bold, "midtop")
                draw_stars(surf, (cx, p.y + 424), stars("sweet", self.score))
            self.buttons.append(Button((cx - 200, p.bottom - 64, 180, 44), "Play again",
                                       lambda: self.game.start_challenge("sweet"), style="primary"))
            self.buttons.append(Button((cx + 20, p.bottom - 64, 140, 44), "Close", self.end))
        else:
            self.buttons.append(Button((cx - 90, p.bottom - 64, 180, 44), "Offer sugar", self.offer,
                                       enabled=self.phase == "ready", style="primary"))
            self.buttons.append(Button((p.right - 160, p.bottom - 64, 140, 44), "Close", self.end))
        best = load_scores().get("sweet")
        if best is not None:
            g._text(surf, f"best: {best}%", (p.right - 30, p.y + 110), LABEL, g.f_text, "topright")
        self.draw_buttons(surf, mouse)


# --- Reverse brain surgery (Mystery defect) ---------------------------------------------------------------------------
class ReverseSurgery(Challenge):
    key = "reverse_surgery"
    overlay = False

    CIRCUITS = [
        dict(
            id="jump",
            play_label="Emergency visual jump",
            target_spec="dnp01",
            bio_name="DNp01 (Giant Fiber escape command)",
            citation="von Reyn et al. 2014, Nat Neurosci 17:962",
            missing_behavior="It ignores rapid looming shadows and flyswatters without jumping or dodging.",
            hints=[
                "Try waving the flyswatter fast toward the fly or moving quickly near it.",
                "Normal flies dodge sudden approaching shadows; this one doesn't flinch.",
            ],
        ),
        dict(
            id="wings",
            play_label="Wing take-off & flight",
            target_spec="prefix:DNg02",
            bio_name="DNg02 (Wing power & take-off command)",
            citation="Shiu et al. 2024, Nature 634:210",
            missing_behavior="It runs along the floor when startled, but its wings never lift it into flight.",
            hints=[
                "Try prodding the fly or giving it a puff of air to launch into flight.",
                "It can scramble along the ground, but its wings never produce flight lift.",
            ],
        ),
        dict(
            id="sweet",
            play_label="Sweet taste & sugar reach",
            target_spec="sweet",
            bio_name="Sweet GRNs & SEL pathway (Proboscis extension to sugar)",
            citation="Shiu et al. 2024, Nature 634:210; Yao & Scott 2022",
            missing_behavior="It never unrolls or extends its proboscis when touching sugar droplets.",
            hints=[
                "Place a drop of sugar near its feet or mouthparts.",
                "It completely ignores sugar and won't extend its feeding tube to eat.",
            ],
        ),
        dict(
            id="fear",
            play_label="Fear of zapped smells",
            target_spec="type:PPL101,PPL103",
            bio_name="PPL1 Dopaminergic neurons (Aversive shock reinforcement)",
            citation="Claridge-Chang et al. 2009, Cell 139:405; Aso et al. 2014",
            missing_behavior="It smells odors normally, but fails to learn which smell gave it an electric zap.",
            hints=[
                "Try pairing an odor scent with an electric zap in training.",
                "It can smell odors, but never forms an aversive memory of the shock.",
            ],
        ),
        dict(
            id="steering",
            play_label="Steering & turning",
            target_spec="type:DNa02",
            bio_name="DNa02 (Steering descending command neurons)",
            citation="Shiu et al. 2024, Nature 634:210; Rayshubskiy et al. 2020",
            missing_behavior="It struggles to coordinate left vs right steering when encountering obstacles.",
            hints=[
                "Nudge the fly from the side or watch how it steers around objects.",
                "Its asymmetric steering is disabled; it struggles to turn cleanly away from obstacles.",
            ],
        ),
    ]

    def __init__(self, game, choice_idx: int | None = None):
        super().__init__(game)
        self.state = "testing"  # testing | guessing | revealed
        self.hints_revealed = 0
        self.guess_result: bool | None = None
        self.guessed_id: str | None = None
        self.silenced_rows = np.array([], dtype=int)
        self.old_overrides: dict[int, float] = {}
        self.panel_rect = pygame.Rect(40, 40, 460, 200)

        if choice_idx is not None and 0 <= choice_idx < len(self.CIRCUITS):
            self.circuit_idx = choice_idx
        else:
            seed = int(getattr(getattr(game, "brain", None), "seed", 0)) + int(getattr(getattr(game, "clock", None), "now", 0) * 1000) % 10000
            rng = np.random.default_rng(seed)
            self.circuit_idx = int(rng.integers(0, len(self.CIRCUITS)))

        self.target_circuit = self.CIRCUITS[self.circuit_idx]
        self._apply_silence()

    def _apply_silence(self) -> None:
        br = getattr(self.game, "brain", None)
        if br is None:
            return
        from kickthefly.lab import assays
        from kickthefly.core import simcore
        spec = self.target_circuit["target_spec"]
        g = assays.groups(br)
        if spec in g:
            rows = g[spec]
        else:
            try:
                rows = simcore.rows_of(br, spec)
            except Exception:
                rows = np.array([], dtype=int)
        self.silenced_rows = np.asarray(rows, dtype=int)
        if len(self.silenced_rows) > 0:
            self.old_overrides = {int(r): float(br.override[r]) for r in self.silenced_rows}
            br.override[self.silenced_rows] = 0.0

    def _restore_silence(self) -> None:
        br = getattr(self.game, "brain", None)
        if br is not None and len(self.silenced_rows) > 0:
            for r, v in self.old_overrides.items():
                br.override[r] = v
            self.silenced_rows = np.array([], dtype=int)
            self.old_overrides = {}

    def end(self) -> None:
        self._restore_silence()
        super().end()

    def request_hint(self) -> str:
        hints = self.target_circuit["hints"]
        if self.hints_revealed < len(hints):
            self.hints_revealed += 1
            return hints[self.hints_revealed - 1]
        return hints[-1]

    def make_guess(self, circuit_id: str) -> bool:
        self.guessed_id = circuit_id
        self.guess_result = (circuit_id == self.target_circuit["id"])
        self.state = "revealed"
        if self.guess_result:
            score = max(1, 3 - self.hints_revealed)
            record_score("reverse_surgery", score, "high")
        return self.guess_result

    def click(self, pos) -> bool:
        for b in self.buttons:
            if b.enabled and b.rect.collidepoint(pos):
                self.game.sound.play("click")
                b.action()
                return True
        if hasattr(self, "panel_rect") and self.panel_rect.collidepoint(pos):
            return True
        return False

    def draw(self, surf, now, mouse) -> None:
        from kickthefly.game.kick_the_fly import PLAY_W
        g = self.game
        cx = min(PLAY_W - 250, max(250, PLAY_W // 2))
        self.buttons = []

        if self.state == "testing":
            w, h = 480, 200
            self.panel_rect = pygame.Rect(cx - w // 2, 80, w, h)
            card = pygame.Surface((w, h), pygame.SRCALPHA)
            pygame.draw.rect(card, (14, 18, 26, 235), card.get_rect(), border_radius=12)
            pygame.draw.rect(card, (70, 90, 130), card.get_rect(), 1, border_radius=12)
            surf.blit(card, self.panel_rect)

            g._text(surf, "MYSTERY DEFECT (Reverse Brain Surgery)", (self.panel_rect.x + 16, self.panel_rect.y + 12), AMBER, g.f_bold)
            g._text(surf, "One brain circuit is turned off! Test the fly with your tools.", (self.panel_rect.x + 16, self.panel_rect.y + 38), TEXT, g.f_small)

            hints = self.target_circuit["hints"]
            if self.hints_revealed == 0:
                hint_txt = "No hints used yet. Tap [Hint] if you need a clue."
                col = LABEL
            elif self.hints_revealed == 1:
                hint_txt = f"Hint 1: {hints[0]}"
                col = (130, 200, 255)
            else:
                hint_txt = f"Hint 2: {hints[1]}"
                col = (255, 210, 120)
            g._text(surf, hint_txt, (self.panel_rect.x + 16, self.panel_rect.y + 72), col, g.f_small)

            best = load_scores().get("reverse_surgery")
            if best is not None:
                g._text(surf, f"Best: {best} stars", (self.panel_rect.right - 16, self.panel_rect.y + 12), LABEL, g.f_small, "topright")

            can_hint = (self.hints_revealed < len(hints))
            self.buttons.append(Button(
                (self.panel_rect.x + 16, self.panel_rect.bottom - 52, 130, 38),
                f"Hint ({2 - self.hints_revealed} left)",
                self.request_hint,
                enabled=can_hint,
            ))
            self.buttons.append(Button(
                (self.panel_rect.x + 154, self.panel_rect.bottom - 52, 170, 38),
                "Make a Guess",
                lambda: setattr(self, "state", "guessing"),
                style="primary",
            ))
            self.buttons.append(Button(
                (self.panel_rect.right - 116, self.panel_rect.bottom - 52, 100, 38),
                "Quit",
                self.end,
            ))

        elif self.state == "guessing":
            w, h = 480, 340
            self.panel_rect = pygame.Rect(cx - w // 2, 60, w, h)
            card = pygame.Surface((w, h), pygame.SRCALPHA)
            pygame.draw.rect(card, (14, 18, 26, 240), card.get_rect(), border_radius=12)
            pygame.draw.rect(card, (80, 110, 160), card.get_rect(), 1, border_radius=12)
            surf.blit(card, self.panel_rect)

            g._text(surf, "WHICH BEHAVIOR IS MISSING?", (self.panel_rect.x + 16, self.panel_rect.y + 12), INK, g.f_head)
            g._text(surf, "Choose the circuit you believe was turned off:", (self.panel_rect.x + 16, self.panel_rect.y + 40), LABEL, g.f_small)

            by = self.panel_rect.y + 68
            for c in self.CIRCUITS:
                cid = c["id"]
                clabel = c["play_label"]
                self.buttons.append(Button(
                    (self.panel_rect.x + 20, by, self.panel_rect.w - 40, 38),
                    clabel,
                    lambda cid=cid: self.make_guess(cid),
                ))
                by += 44

            self.buttons.append(Button(
                (self.panel_rect.centerx - 60, self.panel_rect.bottom - 46, 120, 36),
                "Cancel",
                lambda: setattr(self, "state", "testing"),
            ))

        elif self.state == "revealed":
            w, h = 520, 340
            self.panel_rect = pygame.Rect(cx - w // 2, 60, w, h)
            card = pygame.Surface((w, h), pygame.SRCALPHA)
            pygame.draw.rect(card, (16, 20, 30, 245), card.get_rect(), border_radius=12)
            border_col = GOOD if self.guess_result else BAD
            pygame.draw.rect(card, border_col, card.get_rect(), 2, border_radius=12)
            surf.blit(card, self.panel_rect)

            header_text = "CORRECT DIAGNOSIS!" if self.guess_result else "NOT QUITE!"
            g._text(surf, header_text, (self.panel_rect.x + 20, self.panel_rect.y + 14), border_col, g.f_head)

            if self.guess_result:
                score = max(1, 3 - self.hints_revealed)
                draw_stars(surf, (self.panel_rect.right - 80, self.panel_rect.y + 26), score, 11)
                g._text(surf, f"Diagnosed with {self.hints_revealed} hints!", (self.panel_rect.x + 20, self.panel_rect.y + 46), TEXT, g.f_small)
            else:
                guessed_lbl = next((c["play_label"] for c in self.CIRCUITS if c["id"] == self.guessed_id), self.guessed_id)
                g._text(surf, f"You guessed: '{guessed_lbl}'", (self.panel_rect.x + 20, self.panel_rect.y + 46), (220, 160, 160), g.f_small)

            # Circuit reveal box
            rev_box = pygame.Rect(self.panel_rect.x + 16, self.panel_rect.y + 74, self.panel_rect.w - 32, 190)
            pygame.draw.rect(surf, (24, 30, 44), rev_box, border_radius=8)
            pygame.draw.rect(surf, (50, 62, 88), rev_box, 1, border_radius=8)

            g._text(surf, f"Missing Behavior: {self.target_circuit['play_label']}", (rev_box.x + 12, rev_box.y + 10), AMBER, g.f_bold)
            g._text(surf, f"Biological Circuit: {self.target_circuit['bio_name']}", (rev_box.x + 12, rev_box.y + 36), (140, 200, 255), g.f_small)
            g._text(surf, f"What happens: {self.target_circuit['missing_behavior']}", (rev_box.x + 12, rev_box.y + 64), TEXT, g.f_small)
            g._text(surf, f"Citation: {self.target_circuit['citation']}", (rev_box.x + 12, rev_box.y + 140), LABEL, g.f_small)

            self.buttons.append(Button(
                (self.panel_rect.x + 30, self.panel_rect.bottom - 56, 200, 42),
                "Play Another Mystery",
                lambda: self.game.start_challenge("reverse_surgery"),
                style="primary",
            ))
            self.buttons.append(Button(
                (self.panel_rect.right - 170, self.panel_rect.bottom - 56, 140, 42),
                "Close",
                self.end,
            ))

        self.draw_buttons(surf, mouse)


# --- Predict the move (Descending motor readouts) ---------------------------------------------------------------------
class PredictNeuron(Challenge):
    key = "predict"
    overlay = False
    ROUNDS = 10
    CUE_TIME = 2.2

    READOUTS = [
        dict(
            id="jump",
            label="Jump!",
            target="jump",
            cue_text="HEAD-TOUCH DN SPIKE SURGE",
            action_desc="Jumped up!",
        ),
        dict(
            id="run",
            label="Run!",
            target="run",
            cue_text="BODY-TOUCH DN SPIKE SURGE",
            action_desc="Darted forward!",
        ),
        dict(
            id="kick",
            label="Kick!",
            target="kick",
            cue_text="LEG-TOUCH DN SPIKE SURGE",
            action_desc="Kicked its legs!",
        ),
        dict(
            id="back",
            label="Back up!",
            target="back",
            cue_text="MOONWALKER (MDN) SPIKE SURGE",
            action_desc="Walked backward!",
        ),
        dict(
            id="takeoff",
            label="Take off!",
            target="fly",
            cue_text="WING-POWER (DNg02) SPIKE SURGE",
            action_desc="Launched into flight!",
        ),
    ]

    def __init__(self, game):
        super().__init__(game)
        self.round_idx = 0
        self.score = 0
        self.state = "ready"  # ready | cue | result | done
        self.current_readout = self.READOUTS[0]
        self.timer = 0.0
        self.guess: str | None = None
        self.panel_rect = pygame.Rect(40, 40, 480, 240)
        self.history: list[tuple[str, str, bool]] = []
        seed = int(getattr(getattr(game, "brain", None), "seed", 0)) + 999
        self.rng = np.random.default_rng(seed)
        self.driven_rows = np.array([], dtype=int)
        self.pulse_phase = 0.0

    def start_round(self, now: float) -> None:
        if self.round_idx >= self.ROUNDS:
            self.state = "done"
            record_score("predict", self.score, "high")
            return
        idx = int(self.rng.integers(0, len(self.READOUTS)))
        self.current_readout = self.READOUTS[idx]
        self.state = "cue"
        self.timer = now + self.CUE_TIME
        self.guess = None
        self._drive_cue(self.current_readout["target"])

    def _drive_cue(self, target_name: str) -> None:
        br = getattr(self.game, "brain", None)
        if br is None:
            return
        from kickthefly.core import simcore
        from kickthefly.game.kick_the_fly import MOTOR
        tys = ()
        for name, tys_, _, _ in MOTOR:
            if name == target_name:
                tys = tys_
                break
        if tys:
            rows = np.flatnonzero(np.isin(br.types, tys))
            if len(rows) > 0:
                self.driven_rows = rows
                simcore.drive(br, rows, 0.6)

    def _clear_cue(self) -> None:
        br = getattr(self.game, "brain", None)
        if br is not None and len(self.driven_rows) > 0:
            from kickthefly.core import simcore
            simcore.undrive(br, self.driven_rows)
            self.driven_rows = np.array([], dtype=int)

    def predict(self, choice_id: str, now: float) -> bool:
        if self.state != "cue":
            return False
        self.guess = choice_id
        is_correct = (choice_id == self.current_readout["id"])
        if is_correct:
            self.score += 1
            self.game.sound.play("yum")
        else:
            self.game.sound.play("click")
        self.history.append((self.current_readout["id"], choice_id, is_correct))
        self._trigger_motor_action(self.current_readout["target"], now)
        self._clear_cue()
        self.state = "result"
        self.timer = now + 1.2
        return is_correct

    def _trigger_motor_action(self, target_name: str, now: float) -> None:
        slot = getattr(self, "slot", None)
        if slot is None or not hasattr(slot, "fly"):
            return
        fly = slot.fly
        away = 1.0 if fly.p[0, 0] >= getattr(slot, "threat_x", fly.p[0, 0]) else -1.0
        if target_name == "jump":
            fly.escape(now, seconds=1.0)
        elif target_name == "fly":
            fly.escape(now, seconds=2.5, wander=True)
        elif target_name == "run":
            fly.facing, fly.walk_until, fly.back_until, fly.run = away, now + 1.2, 0.0, True
        elif target_name == "kick":
            fly.flail_until = now + 0.8
        elif target_name == "back":
            fly.back_until = now + 1.0

    def update(self, now: float) -> None:
        self.pulse_phase = (now * 6.0) % (2 * math.pi)
        if self.state == "ready":
            self.start_round(now)
        elif self.state == "cue":
            if now >= self.timer:
                self.predict("timeout", now)
        elif self.state == "result":
            if now >= self.timer:
                self.round_idx += 1
                if self.round_idx >= self.ROUNDS:
                    self.state = "done"
                    record_score("predict", self.score, "high")
                else:
                    self.start_round(now)

    def end(self) -> None:
        self._clear_cue()
        super().end()

    def click(self, pos) -> bool:
        for b in self.buttons:
            if b.enabled and b.rect.collidepoint(pos):
                self.game.sound.play("click")
                b.action()
                return True
        if hasattr(self, "panel_rect") and self.panel_rect.collidepoint(pos):
            return True
        return False

    def draw(self, surf, now, mouse) -> None:
        from kickthefly.game.kick_the_fly import PLAY_W
        g = self.game
        cx = min(PLAY_W - 260, max(260, PLAY_W // 2))
        self.buttons = []

        if self.state == "cue":
            w, h = 500, 230
            self.panel_rect = pygame.Rect(cx - w // 2, 70, w, h)
            card = pygame.Surface((w, h), pygame.SRCALPHA)
            pygame.draw.rect(card, (12, 16, 24, 235), card.get_rect(), border_radius=12)
            pygame.draw.rect(card, (70, 110, 170), card.get_rect(), 1, border_radius=12)
            surf.blit(card, self.panel_rect)

            g._text(surf, f"PREDICT THE MOVE  ·  Round {self.round_idx + 1}/{self.ROUNDS}",
                    (self.panel_rect.x + 16, self.panel_rect.y + 12), AMBER, g.f_bold)
            g._text(surf, f"Score: {self.score}", (self.panel_rect.right - 16, self.panel_rect.y + 12), INK, g.f_bold, "topright")

            # Pulsing neural flash indicator
            pulse_r = int(14 + 4 * math.sin(self.pulse_phase))
            p_center = (self.panel_rect.x + 36, self.panel_rect.y + 54)
            pygame.draw.circle(surf, (255, 180, 50), p_center, pulse_r)
            pygame.draw.circle(surf, (255, 235, 150), p_center, max(4, pulse_r - 6))

            g._text(surf, "DESCENDING MOTOR SURGE DETECTED!", (self.panel_rect.x + 64, self.panel_rect.y + 42), (255, 220, 120), g.f_bold)
            g._text(surf, "What will the fly do before it moves?", (self.panel_rect.x + 64, self.panel_rect.y + 60), TEXT, g.f_small)

            # Time bar
            rem = max(0.0, self.timer - now)
            frac = rem / self.CUE_TIME
            bar_rect = pygame.Rect(self.panel_rect.x + 20, self.panel_rect.y + 86, self.panel_rect.w - 40, 8)
            pygame.draw.rect(surf, (30, 36, 48), bar_rect, border_radius=4)
            pygame.draw.rect(surf, (80, 180, 240), (bar_rect.x, bar_rect.y, int(bar_rect.w * frac), 8), border_radius=4)

            # 5 descending motor action buttons
            bw = (self.panel_rect.w - 48) // 5
            for i, r in enumerate(self.READOUTS):
                bx = self.panel_rect.x + 18 + i * (bw + 3)
                rid = r["id"]
                self.buttons.append(Button(
                    (bx, self.panel_rect.y + 110, bw, 52),
                    r["label"],
                    lambda rid=rid: self.predict(rid, now),
                    style="primary",
                ))

            self.buttons.append(Button(
                (self.panel_rect.right - 100, self.panel_rect.bottom - 46, 84, 32),
                "Quit",
                self.end,
            ))

        elif self.state == "result":
            w, h = 480, 180
            self.panel_rect = pygame.Rect(cx - w // 2, 80, w, h)
            card = pygame.Surface((w, h), pygame.SRCALPHA)
            pygame.draw.rect(card, (14, 18, 28, 235), card.get_rect(), border_radius=12)
            last_ok = bool(self.history and self.history[-1][2])
            border_col = GOOD if last_ok else BAD
            pygame.draw.rect(card, border_col, card.get_rect(), 2, border_radius=12)
            surf.blit(card, self.panel_rect)

            res_hdr = "CORRECT PREDICTION!" if last_ok else "MISSED!"
            g._text(surf, res_hdr, (self.panel_rect.centerx, self.panel_rect.y + 16), border_col, g.f_head, "midtop")
            g._text(surf, f"The fly {self.current_readout['action_desc']}", (self.panel_rect.centerx, self.panel_rect.y + 54), INK, g.f_bold, "midtop")
            g._text(surf, f"Circuit: {self.current_readout['desc']}", (self.panel_rect.centerx, self.panel_rect.y + 82), LABEL, g.f_small, "midtop")
            g._text(surf, f"Score: {self.score}/{self.round_idx + 1}", (self.panel_rect.centerx, self.panel_rect.y + 110), AMBER, g.f_bold, "midtop")

        elif self.state == "done":
            w, h = 480, 240
            self.panel_rect = pygame.Rect(cx - w // 2, 80, w, h)
            card = pygame.Surface((w, h), pygame.SRCALPHA)
            pygame.draw.rect(card, (16, 20, 32, 245), card.get_rect(), border_radius=12)
            pygame.draw.rect(card, (80, 120, 180), card.get_rect(), 2, border_radius=12)
            surf.blit(card, self.panel_rect)

            g._text(surf, "CHALLENGE COMPLETE!", (self.panel_rect.centerx, self.panel_rect.y + 16), INK, g.f_head, "midtop")
            g._text(surf, f"Final Score: {self.score} / {self.ROUNDS} correct predictions", (self.panel_rect.centerx, self.panel_rect.y + 54), AMBER, g.f_bold, "midtop")
            draw_stars(surf, (self.panel_rect.centerx, self.panel_rect.y + 96), stars("predict", self.score), 14)

            self.buttons.append(Button(
                (self.panel_rect.centerx - 170, self.panel_rect.bottom - 56, 160, 42),
                "Play Again",
                lambda: self.game.start_challenge("predict"),
                style="primary",
            ))
            self.buttons.append(Button(
                (self.panel_rect.centerx + 20, self.panel_rect.bottom - 56, 140, 42),
                "Close",
                self.end,
            ))

        self.draw_buttons(surf, mouse)


class _Classes(dict):
    """The challenge classes; Puppeteer and the contraption builder (3.1.0) are loaded when asked for, since their modules import this one."""

    LAZY = {"puppeteer": ("kickthefly.lab.puppet_challenge", "Puppeteer"), "contraption": ("kickthefly.lab.contraption_challenge", "Contraption")}

    def __missing__(self, key):
        if key in self.LAZY:
            import importlib

            mod, name = self.LAZY[key]
            self[key] = getattr(importlib.import_module(mod), name)
            return self[key]
        raise KeyError(key)

    def __contains__(self, key):
        return key in self.LAZY or super().__contains__(key)


CLASSES = _Classes({"tmaze": TMaze, "sneak": Sneak, "sweet": Sweet, "reverse_surgery": ReverseSurgery, "predict": PredictNeuron})


def page_challenges(m, surf, rect, mouse) -> None:
    from kickthefly.ui import menu as ui

    game = m.host
    m.text(surf, "CHALLENGES", (rect.x + 24, rect.y + 16), ui.INK, m.f_head)
    top = max(rect.y + 80, m.subtitle(surf, rect, "Games built on real fly experiments. The fly's brain decides how it does.", rect.y + 50))
    body = pygame.Rect(rect.x + 16, top, rect.w - 32, rect.bottom - 70 - top)
    key = "challenges"
    off = int(m.scroll.get(key, 0))
    m.clip = body
    prev = surf.get_clip()
    surf.set_clip(body)
    scores = load_scores()
    y = body.y + 4 - off
    for c_key, title, desc, unit, better in INFO:
        # 3.0 release review: the description was one line (cut at larger text or a 860 px menu); it wraps now and the card grows with it
        n_desc = len(ui.Menu.fit_lines(m.f_text, desc, body.w - 56, 3)[0])
        extra = (n_desc - 1) * m.f_text.get_linesize()
        card = pygame.Rect(body.x + 8, y, body.w - 16, 150 + extra)
        pygame.draw.rect(surf, (28, 32, 42), card, border_radius=12)
        m.text(surf, title, (card.x + 20, card.y + 16), ui.INK, m.f_head)
        m.wrapped(surf, desc, (card.x + 20, card.y + 54), card.w - 40, ui.TEXT, m.f_text, max_lines=3)
        card = card.move(0, extra)                      # the score row and Start button sit under the description
        best = scores.get(c_key)
        if best is not None:
            shown = f"{best:.0f}/10" if c_key == "tmaze" else f"{best:.1f} fly lengths" if c_key == "sneak" else f"{best:.0f} stars" if c_key == "reverse_surgery" else f"{best:.0f}/10 levels" if c_key == "puppeteer" else f"{best:.0f} runs" if c_key == "contraption" else f"{best:.0f}%"
            m.text(surf, f"Best: {shown}", (card.x + 20, card.y + 96), ui.AMBER, m.f_bold)
            if c_key != "contraption":
                draw_stars(surf, (card.x + 250, card.y + 106), stars(c_key, best), 11)
        else:
            m.text(surf, "Not played yet", (card.x + 20, card.y + 96), ui.LABEL, m.f_text)
        m.button(surf, (card.right - 170, card.y + 86, 150, 46), "Start", (lambda k=c_key: game.start_challenge(k)),
                 style="primary", id=("challenge", c_key))
        y += 166 + extra
    m.content_h[key] = max(0, y + off - body.bottom + 8)
    surf.set_clip(prev)
    m.clip = None
    m.button(surf, (rect.right - 164, rect.bottom - 58, 140, 42), "Back", m.back, style="primary", id=("ch", "back"))


def memory_snapshot(br) -> tuple:
    """A copy of a brain's mushroom body memory (the challenges that borrow it put it back with memory_restore)."""
    mem = br.memory
    with mem.lock:
        return (mem.w.copy(), {k: v.copy() for k, v in mem.templates.items()}, dict(mem.naive_mbon),
                {k: list(v) for k, v in mem.log.items()}, mem.dirty)


def memory_restore(br, saved: tuple) -> None:
    mem = br.memory
    w, templates, naive, log, dirty = saved
    with br.step_lock, mem.lock:
        mem.w[:] = w
        mem.templates, mem.naive_mbon, mem.log, mem.dirty = templates, naive, log, dirty
        mem._write_back()
