#!/usr/bin/env python3
"""Font metrics of the pygame menu (kickthefly/ui/menu.py fonts) for the Rust text renderer: python make_ui_metrics.py OUT.json"""
import json, sys, os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame
pygame.init()
fam = "segoeui,dejavusans,liberationsans,arial"
corpus = ["Resume", "Accessibility", "Settings", "PAUSED", "Kick the Fly 3.1.1   ·   Play mode", "Mic and streamer", "Reset to defaults",
          "Changes apply right away and are saved.", "On", "Off", "100%", "0.35", "<", ">", "press a key...", "left shift", "unbound",
          "Walk forward", "Brain surgery", "Time-lapse record", "Fullscreen", "GAME RULE", "CONNECTOME", "RESTART", "3D ONLY",
          "Click a key, then press the new one (Esc cancels). A key that's taken swaps with that action.", "…", "|", " ", "AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz0123456789",
          "WAVE To AV Ty fi fl", "Kenyon cells → MBON", "Ä ö ü ß é"]
out = {}
for k in (1.0, 1.25):
    for name, base, bold, pre in [("small", 14, False, ""), ("text", 16, False, ""), ("bold", 16, True, "segoeuisemibold,"),
                                  ("head", 22, True, "segoeuiblack,"), ("title", 38, True, "segoeuiblack,")]:
        size = round(base * k)
        f = pygame.font.SysFont(pre + fam, size, bold=bold)
        key = f"{name}@{size}{'b' if bold else ''}"
        img = f.render("Hg", True, (255, 255, 255))
        out[key] = dict(size=size, bold=bold, linesize=f.get_linesize(), height=f.get_height(), ascent=f.get_ascent(),
                        descent=f.get_descent(), render_h=img.get_height(), widths={s: f.size(s)[0] for s in corpus})
json.dump(out, open(sys.argv[1], "w"), indent=0, ensure_ascii=False)
# per-glyph advances (pygame's hinted ones; the Rust text renderer cannot hint, so it uses this table) for the printable Latin range and the
# symbols the menus use, per font; sizes are the ones above
chars = [chr(c) for c in range(32, 127)] + [chr(c) for c in range(160, 256)] + list("·→←↑↓…–—‘’“”•×±°≈≤≥µ✓✗▶◀●○■□▲▼½¼¾€")
adv = {}
for key, m in out.items():
    f = pygame.font.SysFont(("segoeuisemibold," if key.startswith("bold") else "segoeuiblack," if key.startswith(("head", "title")) else "") + fam, m["size"], bold=m["bold"])
    adv[key] = {c: ([f.metrics(c)[0][4], f.metrics(c)[0][0], f.metrics(c)[0][1]] if f.metrics(c)[0] else [0, 0, 0]) for c in chars}
if len(sys.argv) > 2:
    json.dump(adv, open(sys.argv[2], "w"), ensure_ascii=False, separators=(",", ":"))
