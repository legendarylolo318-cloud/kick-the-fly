"""Tests for i18n localization framework."""
import json
from pathlib import Path
from kickthefly.core import i18n


def test_i18n_default_and_fallback():
    i18n.set_language("en")
    assert i18n.get_language() == "en"
    assert i18n.tr("PAUSED") == "PAUSED"
    assert i18n.tr("Resume") == "Resume"
    # Unknown string returns unchanged
    assert i18n.tr("Unkown String 1234") == "Unkown String 1234"


def test_i18n_german():
    i18n.set_language("de")
    assert i18n.get_language() == "de"
    assert i18n.tr("PAUSED") == "PAUSE"
    assert i18n.tr("Resume") == "Fortsetzen"
    assert i18n.tr("Settings") == "Einstellungen"
    # Fallback to English/original for missing keys
    assert i18n.tr("Unkown String 1234") == "Unkown String 1234"
    # Switch back to English
    i18n.set_language("en")


def test_template_and_catalogs_structure():
    locales_dir = i18n._locales_dir()
    en = json.loads((locales_dir / "en.json").read_text(encoding="utf-8"))
    template = json.loads((locales_dir / "template.json").read_text(encoding="utf-8"))
    de = json.loads((locales_dir / "de.json").read_text(encoding="utf-8"))

    # Template must contain all keys in en (ignoring _comment)
    en_keys = {k for k in en if not k.startswith("_")}
    tmpl_keys = {k for k in template if not k.startswith("_")}
    de_keys = {k for k in de if not k.startswith("_")}

    assert en_keys == tmpl_keys
    # German catalog should have keys from en
    for k in de_keys:
        assert k in en_keys


def test_scientific_names_untouched():
    # Verify cell types and gene names are never altered
    i18n.set_language("de")
    for term in ("ORN_DA1", "DA1_lPN", "pC1", "pIP10", "ps1", "Or67d", "ppk23", "ppk25"):
        assert i18n.tr(term) == term
    i18n.set_language("en")


def test_unknown_language_and_bad_placeholders_fall_back_instead_of_crashing():
    assert i18n.set_language("xx") == "en"                   # no catalog: English
    assert i18n.tr("Resume") == "Resume"
    assert i18n.tr("{n} flies", m=3) == "{n} flies"           # a placeholder the caller didn't fill: text, no KeyError
    assert i18n.tr("never translated {x}") == "never translated {x}"
    i18n.set_language("en")


def test_translations_keep_the_english_placeholders():
    """A translated string with other {placeholders} than the English one would raise when formatted."""
    import re
    en = json.loads((i18n._locales_dir() / "en.json").read_text(encoding="utf-8"))
    for lang in ("de", "template"):
        cat = json.loads((i18n._locales_dir() / f"{lang}.json").read_text(encoding="utf-8"))
        for k, v in cat.items():
            if k.startswith("_") or not v:
                continue
            assert set(re.findall(r"{(\w*)}", v)) == set(re.findall(r"{(\w*)}", en[k])), (lang, k, v)


def test_the_pause_menu_is_translated(monkeypatch):
    """The language setting reaches the screen: the pause menu draws its German labels, and English again after."""
    import pygame
    from kickthefly.ui import menu as menu_mod

    drawn = []
    pygame.init()
    surf = pygame.Surface((1280, 760))

    class Host:
        cfg = type("Cfg", (), {"lab": False, "get": lambda self, key, default=None: default})()

        def menu_action(self, a):
            pass

    m = menu_mod.Menu.__new__(menu_mod.Menu)
    monkeypatch.setattr(menu_mod.Menu, "text", lambda self, s, text, *a, **k: drawn.append(text) or pygame.Rect(0, 0, 1, 1))
    monkeypatch.setattr(menu_mod.Menu, "button", lambda self, s, r, label, *a, **k: drawn.append(label))
    m.host = Host()
    m.f_title = m.f_small = None
    try:
        i18n.set_language("de")
        m._page_pause(surf, pygame.Rect(0, 0, 440, 560))
        assert "PAUSE" in drawn and "Fortsetzen" in drawn and "Einstellungen" in drawn
        drawn.clear()
        i18n.set_language("en")
        m._page_pause(surf, pygame.Rect(0, 0, 440, 560))
        assert "PAUSED" in drawn and "Resume" in drawn
    finally:
        i18n.set_language("en")
