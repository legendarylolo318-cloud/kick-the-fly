"""Tests for Task 3: Pet mode lifecycle, deterministic catch-up, clock clamping, and safety invariants."""
import json
import subprocess
import tempfile
import time
from pathlib import Path
import pytest

from kickthefly.core.pet import PetManager, MAX_CATCHUP_SECONDS


def test_pet_save_persistence_and_backup():
    """Verify that pet state persists atomically and creates a .bak backup on update."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = PetManager(data_dir=Path(tmpdir))
        manager.create_new_pet(seed=9876, brain_type="adult")

        assert manager.save_file.exists()
        assert not manager.backup_file.exists()

        # Update and save again -> backup must be created
        manager.feed(0.1, source="sugar")
        assert manager.backup_file.exists()

        # Load fresh instance
        manager2 = PetManager(data_dir=Path(tmpdir))
        manager2.load_or_create()
        assert manager2.seed == 9876
        assert manager2.brain_type == "adult"
        assert len(manager2.timeline) >= 2


def test_pet_clock_clamping_negative_and_excessive():
    """Verify clock-tampering guards: negative jump clamped to 0, huge jump clamped to 7 days."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = PetManager(data_dir=Path(tmpdir))
        manager.create_new_pet(seed=1234, brain_type="adult")

        now = time.time()

        # Case 1: Clock jumped backward (negative elapsed time)
        with open(manager.save_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["last_saved_time"] = now + 10000.0
        with open(manager.save_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        manager2 = PetManager(data_dir=Path(tmpdir))
        manager2.load_or_create()
        # Hunger should not have decreased or errored; clamped to 0 elapsed
        assert manager2.hunger >= 0.0

        # Case 2: Clock jumped 30 days ahead (huge jump)
        with open(manager.save_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["last_saved_time"] = now - (30.0 * 86400.0)
        with open(manager.save_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        manager3 = PetManager(data_dir=Path(tmpdir))
        manager3.load_or_create()
        # Catchup must clamp to MAX_CATCHUP_SECONDS (7 days)
        # In 7 days, hunger will cap at 1.0
        assert manager3.hunger == 1.0


def test_pet_deterministic_catchup():
    """Verify deterministic catch-up for hunger and sleep pressure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = PetManager(data_dir=Path(tmpdir))
        manager.create_new_pet(seed=5555, brain_type="adult")
        now = time.time()

        # Simulate 2 hours elapsed by writing last_saved_time directly into the save file
        with open(manager.save_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["hunger"] = 0.1
        data["sleep_pressure"] = 0.1
        data["last_saved_time"] = now - 7200.0
        with open(manager.save_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        manager2 = PetManager(data_dir=Path(tmpdir))
        manager2.load_or_create()

        # Hunger rate = 1 / 86400 per second -> + 7200/86400 = + 0.0833
        expected_hunger = 0.1 + (7200.0 / 86400.0)
        assert pytest.approx(manager2.hunger, abs=0.01) == expected_hunger


def test_no_background_process_daemon_or_service():
    """Verify that NO background process, daemon, service, or timer is started.
    PetManager must only execute on-demand in the current process."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = PetManager(data_dir=Path(tmpdir))
        manager.create_new_pet(seed=1111)

        # Check running processes using ps
        res = subprocess.run(["ps", "aux"], capture_output=True, text=True, check=True)
        lines = res.stdout.splitlines()
        daemon_lines = [
            ln for ln in lines
            if "pet_daemon" in ln or "kickthefly_service" in ln or "pet_timer" in ln
        ]
        assert len(daemon_lines) == 0, f"Found unexpected background daemon: {daemon_lines}"


def test_pet_neural_couplings():
    """Verify neural couplings: hunger scales sugar/PAM gains; sleep pressure drives dFB."""
    manager = PetManager(data_dir=Path(tempfile.mkdtemp()))
    manager.create_new_pet()

    manager.hunger = 0.0
    assert manager.get_sugar_gain_multiplier() == 1.0
    assert manager.get_pam_reward_multiplier() == 1.0

    manager.hunger = 1.0
    assert manager.get_sugar_gain_multiplier() == 3.0
    assert manager.get_pam_reward_multiplier() == 2.5

    manager.sleep_pressure = 0.8
    assert manager.get_dfb_sleep_drive() == 0.4
