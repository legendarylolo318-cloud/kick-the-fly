"""Speedrun timer and verification codes for Escape-Room arena.

Generates and verifies cryptographic verification codes for player speedruns
embedding the simulation seed, elapsed time, and integrity checksum.
"""
from __future__ import annotations

import hashlib
import time

PREFIX = "KTF"
SECRET_SALT = "ktf_escape_room_v1"


def make_speedrun_code(seed: int, elapsed_seconds: float) -> str:
    """Generate a shareable verification code: KTF-<SEED>-<CENTISECONDS>-<SIG>."""
    centis = int(round(max(0.0, elapsed_seconds) * 100))
    payload = f"{seed}:{centis}:{SECRET_SALT}"
    sig = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:6].upper()
    return f"{PREFIX}-{seed}-{centis:05d}-{sig}"


def verify_speedrun_code(code: str) -> dict:
    """Verify and decode a speedrun verification string.

    Returns:
        dict with valid: bool, seed: int, time_seconds: float, formatted: str.
    """
    code = code.strip().upper()
    parts = code.split("-")
    if len(parts) != 4 or parts[0] != PREFIX:
        return {"valid": False, "error": "Invalid format. Expected KTF-<SEED>-<TIME>-<SIG>."}

    try:
        seed = int(parts[1])
        centis = int(parts[2])
        sig = parts[3]
    except ValueError:
        return {"valid": False, "error": "Seed or time component is not an integer."}

    expected_payload = f"{seed}:{centis}:{SECRET_SALT}"
    expected_sig = hashlib.sha256(expected_payload.encode("utf-8")).hexdigest()[:6].upper()

    if sig != expected_sig:
        return {"valid": False, "error": "Checksum signature mismatch; code is forged or invalid."}

    time_sec = centis / 100.0
    return {
        "valid": True,
        "seed": seed,
        "time_seconds": time_sec,
        "formatted_time": f"{time_sec:.2f}s",
        "code": code,
    }
