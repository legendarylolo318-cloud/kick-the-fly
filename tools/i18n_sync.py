"""Add the new UI strings to the localization catalogs (kickthefly/data/locales/en.json and template.json).

    python tools/i18n_sync.py            # add what is missing (en.json: text -> text, template.json: text -> "")
    python tools/i18n_sync.py --check    # exit 1 if anything is missing (what tests/test_i18n_coverage.py does)

Collected from: every tr("...") call with a literal in the 2.12 UI modules, the labels and tips of the settings and
keybinding actions, the tool catalog (kickthefly/core/loadout.py) and the fixed strings the menus pass to tr() as
variables. Existing entries are never changed, so translations made in de.json stay valid.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
LOCALES = ROOT / "kickthefly" / "data" / "locales"
MODULES = ("kickthefly/ui/loadout_ui.py", "kickthefly/ui/tutorial.py", "kickthefly/ui/help_ui.py",
           "kickthefly/ui/crashscreen.py", "kickthefly/game/kick_the_fly.py", "kickthefly/game/kick3d.py")
NEW_SETTINGS = ("controls.loadout_preset",)


def literals(path: Path) -> set[str]:
    out = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) in ("tr", "_") and node.args:
            a = node.args[0]
            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                out.add(a.value)
    return out


def wanted() -> set[str]:
    from kickthefly.core import config, loadout as lo

    out: set[str] = set()
    for m in MODULES:
        out |= literals(ROOT / m)
    for key in NEW_SETTINGS:
        s = config.BY_KEY[key]
        out |= {s.label, s.tip, *s.labels}
    for a, label, _ in config.ACTIONS:
        if a in ("loadout", "tool_wheel", "page_prev", "page_next") or a.startswith("slot"):
            out.add(label)
    for t in lo.CATALOG:
        out |= {t.label, t.category, t.desc, t.neurons, t.tag, t.larva_note} - {""}
    out |= set(lo.PRESET_LABELS.values()) | set(lo.PRESET_TIPS.values()) | {"Play", "Lab", "Pet", "Help"}
    return {x for x in out if x.strip()}


def main(argv) -> int:
    en_p, tm_p = LOCALES / "en.json", LOCALES / "template.json"
    en, tm = json.loads(en_p.read_text(encoding="utf-8")), json.loads(tm_p.read_text(encoding="utf-8"))
    missing = sorted(w for w in wanted() if w not in en)
    if "--check" in argv:
        for w in missing:
            print("missing:", w)
        gaps = [k for k in en if not k.startswith("_") and k not in tm]
        for k in gaps:
            print("missing from template.json:", k)
        return 1 if (missing or gaps) else 0
    for w in missing:
        en[w] = w
    for k in en:                                    # the template lists every key of en.json (tests/test_i18n.py)
        if not k.startswith("_"):
            tm.setdefault(k, "")
    en_p.write_text(json.dumps(en, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tm_p.write_text(json.dumps(tm, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"added {len(missing)} strings")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
