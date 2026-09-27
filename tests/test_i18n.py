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
    locales_dir = Path("kickthefly/data/locales")
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
