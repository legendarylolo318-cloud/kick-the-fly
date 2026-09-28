"""Pet mode: one long-lived, persistent fly across real days.

GAME RULE physics & state machine:
- Persists across sessions in the XDG data dir / Documents\\Kick the Fly\\pet:
    pet.ktfsave (primary) and pet.ktfsave.bak (backup of previous save).
- The simulation ONLY runs while the game is open. NO background process, daemon,
  service, autostart entry, or background timer ever runs.
- When the game is opened, time is caught up via a deterministic catch-up rule
  based on wall-clock elapsed time since the last save.
- Clock guard:
    - If elapsed time is negative (clock moved back), clamp to 0 and log warning.
    - If elapsed time > 7 days, clamp catch-up to 7 days and log warning.

Needs & couplings to real neurons (all GAME RULE: the need variables, their rates and how strongly they scale neuron
input are chosen for play, not measured; the neurons they drive are the real connectome groups):
- Hunger (0.0 to 1.0):
    - Rises over time (1.0 per 24 hours of elapsed time).
    - Scales sugar taste input gain: (1.0 + 2.0 * hunger).
    - Scales PAM reward dopamine drive: (1.0 + 1.5 * hunger) (starved flies find sugar more rewarding).
    - Decreased by feeding (eating sugar pile or orchard fruit).
- Sleep pressure (0.0 to 1.0):
    - Rises while awake (1.0 per 16 hours awake).
    - Drives dorsal fan-shaped body (dFB) sleep neurons (FB6/FB7 in adult, premotor resting DNs in larva).
    - Decays during sleep (recovers in 8 hours of sleep).
    - The fly sleeps when sleep neurons fire above thresh.sleep or during night in the day/night cycle.
- Mood (0.0 to 1.0):
    - Readout from PAM (reward/contentment) vs PPL1 (punishment/stress) dopamine activity.
- Immortality & Death:
    - Starts immortal by default (death: off).
    - Optional "real stakes" toggle enables mortal health & starvation risks, triggering autopsy upon death.
    - Saves are never deleted without explicit confirmation.
"""
from __future__ import annotations

import json
import logging
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from kickthefly.core import paths
from kickthefly.core.individuality import compute_personality_card

logger = logging.getLogger("kickthefly.pet")

MAX_CATCHUP_SECONDS = 7.0 * 86400.0  # 7 days max catch-up
HUNGER_RATE_PER_SEC = 1.0 / 86400.0   # Empty in 24 hours without food
SLEEP_RATE_PER_SEC = 1.0 / (16.0 * 3600.0)  # Tired after 16 hours awake
SLEEP_RECOVERY_RATE = 1.0 / (8.0 * 3600.0)   # Fully rested after 8 hours sleep
HUNGER_SUGAR_GAIN = 2.0     # sugar taste input x (1 + this * hunger)
HUNGER_PAM_GAIN = 1.5       # PAM reward drive x (1 + this * hunger)
SLEEP_DFB_DRIVE = 0.5       # dFB (FB6/FB7) drive = this * sleep pressure
# All of the above are GAME RULE and shown in Lab > Parameters (lab.PARAMS "pet.*").


@dataclass
class PetTimelineEvent:
    timestamp: float
    event_type: str  # "born", "meal", "sleep", "wake", "training", "injury", "catchup"
    description: str


class PetManager:
    """Manages the lifecycle, persistence, and neural couplings of the pet fly."""

    def __init__(self, data_dir: Path | None = None):
        self.paths = paths.get()
        self.pet_dir = data_dir or (self.paths.data_dir / "pet")
        self.pet_dir.mkdir(parents=True, exist_ok=True)
        self.save_file = self.pet_dir / "pet.ktfsave"
        self.backup_file = self.pet_dir / "pet.ktfsave.bak"

        # Pet State
        self.seed: int = 12345
        self.brain_type: str = "adult"  # "adult" or "larva"
        self.born_time: float = time.time()
        self.last_saved_time: float = time.time()
        self.hunger: float = 0.2  # 0.0 (full) to 1.0 (starving)
        self.sleep_pressure: float = 0.1  # 0.0 (fully rested) to 1.0 (exhausted)
        self.is_sleeping: bool = False
        self.mood: float = 0.6  # 0.0 (stressed) to 1.0 (happy/content)
        self.real_stakes: bool = False  # Immortal by default
        self.is_dead: bool = False
        self.timeline: list[dict[str, Any]] = []
        self.personality_card: dict[str, Any] = {}
        self.welcome_message: str = ""

    def exists(self) -> bool:
        return self.save_file.exists() or self.backup_file.exists()

    def create_new_pet(self, seed: int | None = None, brain_type: str = "adult") -> None:
        """Initialize a fresh persistent pet fly."""
        now = time.time()
        self.seed = seed if seed is not None else int(now) % 100000
        self.brain_type = brain_type
        self.born_time = now
        self.last_saved_time = now
        self.hunger = 0.15
        self.sleep_pressure = 0.10
        self.is_sleeping = False
        self.mood = 0.70
        self.real_stakes = False
        self.is_dead = False
        self.timeline = [
            asdict(PetTimelineEvent(timestamp=now, event_type="born", description=f"Adopted {self.brain_type} fly!"))
        ]
        self.personality_card = compute_personality_card(self.seed, mode="subtle")
        self.welcome_message = f"Welcome to Pet Mode! Meet your new {self.brain_type}."
        self.save()

    def load_or_create(self, brain_type: str = "adult") -> None:
        """Load existing pet with deterministic catch-up, or create a new one."""
        if not self.exists():
            self.create_new_pet(brain_type=brain_type)
            return

        target_file = self.save_file if self.save_file.exists() else self.backup_file
        try:
            with open(target_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._unpack_state(data)
            self._apply_catchup()
        except Exception as e:
            logger.error("Failed to load pet save %s: %s", target_file, e)
            if target_file == self.save_file and self.backup_file.exists():
                logger.info("Attempting fallback to backup file %s", self.backup_file)
                try:
                    with open(self.backup_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self._unpack_state(data)
                    self._apply_catchup()
                    return
                except Exception as be:
                    logger.error("Failed to load backup save: %s", be)
            self._quarantine_unreadable()
            self.create_new_pet(brain_type=brain_type)

    def _quarantine_unreadable(self) -> None:
        """Keep unreadable saves under a new name instead of letting the fresh pet's save() overwrite them."""
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for f in (self.save_file, self.backup_file):
            if f.exists():
                dest = f.with_name(f"{f.name}.unreadable-{stamp}")
                try:
                    f.replace(dest)
                    logger.warning("Kept unreadable pet save as %s", dest)
                except OSError as e:
                    logger.error("Could not move unreadable pet save %s: %s", f, e)

    def _unpack_state(self, data: dict[str, Any]) -> None:
        self.seed = int(data.get("seed", 12345))
        self.brain_type = str(data.get("brain_type", "adult"))
        self.born_time = float(data.get("born_time", time.time()))
        self.last_saved_time = float(data.get("last_saved_time", time.time()))
        self.hunger = float(data.get("hunger", 0.2))
        self.sleep_pressure = float(data.get("sleep_pressure", 0.1))
        self.is_sleeping = bool(data.get("is_sleeping", False))
        self.mood = float(data.get("mood", 0.6))
        self.real_stakes = bool(data.get("real_stakes", False))
        self.is_dead = bool(data.get("is_dead", False))
        self.timeline = list(data.get("timeline", []))
        self.personality_card = data.get("personality_card") or compute_personality_card(self.seed, mode="subtle")

    def _apply_catchup(self) -> None:
        """Deterministic wall-clock catch-up rule on game launch."""
        now = time.time()
        elapsed = now - self.last_saved_time

        # Clock-tampering guards
        if elapsed < 0.0:
            logger.warning("System clock moved backwards by %.1f seconds; clamping catch-up to 0.", abs(elapsed))
            elapsed = 0.0
        elif elapsed > MAX_CATCHUP_SECONDS:
            logger.warning("Large elapsed time of %.1f days; clamping catch-up to 7 days.", elapsed / 86400.0)
            elapsed = MAX_CATCHUP_SECONDS

        hours = elapsed / 3600.0
        # Hunger advances continuously while offline
        added_hunger = elapsed * HUNGER_RATE_PER_SEC
        self.hunger = min(1.0, self.hunger + added_hunger)

        # Sleep cycle catch-up: if away for > 4 hours, pet took a sleep
        if hours >= 4.0:
            self.sleep_pressure = max(0.05, self.sleep_pressure - 0.7)
            self.is_sleeping = False
            self.timeline.append(
                asdict(PetTimelineEvent(timestamp=now, event_type="sleep", description=f"Slept while away ({hours:.1f}h)"))
            )
        else:
            self.sleep_pressure = min(1.0, self.sleep_pressure + elapsed * SLEEP_RATE_PER_SEC)

        # Check death from starvation if real stakes is on
        if self.real_stakes and self.hunger >= 1.0 and hours >= 48.0:
            self.is_dead = True
            self.timeline.append(
                asdict(PetTimelineEvent(timestamp=now, event_type="injury", description="Starved from prolonged neglect."))
            )

        # Compose casual-friendly welcome summary
        if hours < 0.1:
            self.welcome_message = "Welcome back! Your fly is buzzing cheerfully."
        elif hours < 2.0:
            self.welcome_message = f"Welcome back! You were away for {int(hours * 60)} minutes."
        elif hours < 24.0:
            self.welcome_message = f"Welcome back! You were away for {hours:.1f} hours. Your fly is getting hungry."
        else:
            days = hours / 24.0
            self.welcome_message = f"Welcome back! You were away for {days:.1f} days. Your fly missed you!"

        self.last_saved_time = now
        self.save()

    def feed(self, amount: float = 0.35, source: str = "sugar", save: bool = True) -> None:
        """Feed the pet to reduce hunger. save=False for the per-frame feeding in the game (saved on quit)."""
        self.hunger = max(0.0, self.hunger - amount)
        self.mood = min(1.0, self.mood + 0.15 * min(1.0, amount / 0.35))   # a full meal (0.35) lifts mood by 0.15
        if save:
            self.timeline.append(
                asdict(PetTimelineEvent(timestamp=time.time(), event_type="meal", description=f"Fed on {source}"))
            )
            self.save()

    def record_training(self, task_name: str) -> None:
        self.timeline.append(
            asdict(PetTimelineEvent(timestamp=time.time(), event_type="training", description=f"Trained in {task_name}"))
        )
        self.save()

    def record_injury(self, source: str) -> None:
        self.mood = max(0.0, self.mood - 0.25)
        self.timeline.append(
            asdict(PetTimelineEvent(timestamp=time.time(), event_type="injury", description=f"Injured by {source}"))
        )
        self.save()

    def get_sugar_gain_multiplier(self) -> float:
        """Coupling: hunger scales sugar taste neuron gain."""
        return float(1.0 + HUNGER_SUGAR_GAIN * self.hunger)

    def get_pam_reward_multiplier(self) -> float:
        """Coupling: hunger scales PAM reward dopamine drive."""
        return float(1.0 + HUNGER_PAM_GAIN * self.hunger)

    def get_dfb_sleep_drive(self) -> float:
        """Coupling: sleep pressure drives dFB sleep neurons."""
        return float(SLEEP_DFB_DRIVE * self.sleep_pressure)

    def update_live(self, dt: float, is_night: bool = False, pam_rate: float = 0.0, ppl1_rate: float = 0.0) -> None:
        """Step live pet state while the game is running."""
        # Hunger increases slowly during game
        self.hunger = min(1.0, self.hunger + dt * HUNGER_RATE_PER_SEC * 1.5)

        # Sleep pressure
        if self.is_sleeping:
            self.sleep_pressure = max(0.0, self.sleep_pressure - dt * SLEEP_RECOVERY_RATE)
            if self.sleep_pressure < 0.1 and not is_night:
                self.is_sleeping = False
                self.timeline.append(
                    asdict(PetTimelineEvent(timestamp=time.time(), event_type="wake", description="Woke up refreshed."))
                )
        else:
            self.sleep_pressure = min(1.0, self.sleep_pressure + dt * SLEEP_RATE_PER_SEC)
            if (self.sleep_pressure > 0.85 or is_night) and random.random() < 0.005:
                self.is_sleeping = True
                self.timeline.append(
                    asdict(PetTimelineEvent(timestamp=time.time(), event_type="sleep", description="Fell asleep."))
                )

        # Mood tracks PAM vs PPL1
        tot = pam_rate + ppl1_rate
        if tot > 0.01:
            target_mood = 0.5 + 0.5 * (pam_rate - ppl1_rate) / tot
            self.mood += (target_mood - self.mood) * 0.05
            self.mood = float(np.clip(self.mood, 0.0, 1.0))

    def save(self) -> None:
        """Atomic save with backup copy."""
        self.last_saved_time = time.time()
        data = {
            "seed": self.seed,
            "brain_type": self.brain_type,
            "born_time": self.born_time,
            "last_saved_time": self.last_saved_time,
            "hunger": self.hunger,
            "sleep_pressure": self.sleep_pressure,
            "is_sleeping": self.is_sleeping,
            "mood": self.mood,
            "real_stakes": self.real_stakes,
            "is_dead": self.is_dead,
            "timeline": self.timeline[-50:],  # keep last 50 events
            "personality_card": self.personality_card,
        }

        # Backup existing file before overwrite
        if self.save_file.exists():
            try:
                shutil.copy2(self.save_file, self.backup_file)
            except OSError as e:
                logger.warning("Could not create pet backup: %s", e)

        tmp_file = self.save_file.with_name(self.save_file.name + ".tmp")
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp_file.replace(self.save_file)
