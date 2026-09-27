"""Localization framework for Kick the Fly.

Uses lightweight JSON catalogs stored in kickthefly/data/locales/.
Provides tr() and _() for UI string translations, with automatic system locale
detection and an English fallback.
Cell type names, gene names, and scientific citations are never translated.
"""
from __future__ import annotations

import json
import locale
import os
from pathlib import Path
from typing import Any

from kickthefly.core.crash import log
from kickthefly import data

_CURRENT_LANG: str = "en"
_CATALOG: dict[str, str] = {}
_FALLBACK_CATALOG: dict[str, str] = {}

AVAILABLE_LANGUAGES: dict[str, str] = {
    "en": "English",
    "de": "Deutsch (Machine-translated)",
}


def _locales_dir() -> Path:
    """Path to the locales directory."""
    return data.path("locales")


def detect_system_language() -> str:
    """Detect the system's 2-letter language code, defaulting to 'en'."""
    try:
        env_lang = os.environ.get("LC_ALL") or os.environ.get("LC_MESSAGES") or os.environ.get("LANG")
        if env_lang:
            code = env_lang.split(".")[0].split("_")[0].lower()
            if code in AVAILABLE_LANGUAGES:
                return code
        loc = locale.getlocale(locale.LC_MESSAGES)[0] if hasattr(locale, "LC_MESSAGES") else locale.getlocale()[0]
        if loc:
            code = loc.split("_")[0].lower()
            if code in AVAILABLE_LANGUAGES:
                return code
    except Exception as e:
        log.debug("Failed to detect system locale: %s", e)
    return "en"


def _load_catalog(lang: str) -> dict[str, str]:
    """Load JSON translation catalog for a given language code."""
    path = _locales_dir() / f"{lang}.json"
    if not path.exists():
        return {}
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(content, dict):
            # Only keep string-to-string mappings, filter comments/metadata
            return {k: v for k, v in content.items() if isinstance(k, str) and isinstance(v, str) and not k.startswith("_")}
    except Exception as e:
        log.warning("Could not load translation file %s: %s", path, e)
    return {}


def set_language(lang: str) -> str:
    """Set the active language ('auto', 'en', 'de', etc.). Returns resolved language code."""
    global _CURRENT_LANG, _CATALOG, _FALLBACK_CATALOG
    resolved = detect_system_language() if lang == "auto" else lang.lower()
    if resolved not in AVAILABLE_LANGUAGES:
        resolved = "en"

    _FALLBACK_CATALOG = _load_catalog("en")
    if resolved == "en":
        _CATALOG = _FALLBACK_CATALOG
    else:
        _CATALOG = _load_catalog(resolved)

    _CURRENT_LANG = resolved
    return _CURRENT_LANG


def get_language() -> str:
    """Get the currently active language code."""
    return _CURRENT_LANG


def tr(text: str, **kwargs: Any) -> str:
    """Translate UI text, falling back to English catalog, then the original string.
    Never translate cell types, gene names, or scientific citations.
    """
    translated = _CATALOG.get(text)
    if translated is None:
        translated = _FALLBACK_CATALOG.get(text, text)
    if kwargs:
        try:
            return translated.format(**kwargs)
        except Exception:
            return translated
    return translated


# Standard alias
_ = tr
