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
# per-glyph advances (pygame's hinted ones; the Rust text renderer cannot hint, so it uses this table), keyed "<face><size>" with face
# sans (FreeSans), bold (FreeSansBold) or mono (FreeMono, what "consolas" resolves to), for every size the menus and the HUD use
chars = [chr(c) for c in range(32, 127)] + [chr(c) for c in range(160, 256)] + list("·→←↑↓…–—‘’“”•×±°≈≤≥µ✓✗▶◀●○■□▲▼½¼¾€")
adv = {}
faces = {"sans": ("", False, (14, 15, 16, 18, 19, 20, 40, 50)), "bold": ("segoeuisemibold,", True, (16, 20, 22, 25, 28, 34, 38, 42, 48)),
         "mono": ("consolas", False, (13,))}
for face, (pre, bold, sizes) in faces.items():
    for size in sizes:
        spec = "consolas" if face == "mono" else pre + fam
        f = pygame.font.SysFont(spec, size, bold=bold)
        if face == "bold" and size in (20, 22, 25, 28, 34, 38, 42, 48):      # headings: segoeuiblack first, same file
            f = pygame.font.SysFont("segoeuiblack," + fam, size, bold=True)
        row = {c: ([f.metrics(c)[0][4], f.metrics(c)[0][0], f.metrics(c)[0][1]] if f.metrics(c)[0] else [0, 0, 0]) for c in chars}
        adv[f"{face}{size}"] = dict(row, _metrics=[f.get_linesize(), f.get_ascent()])
if len(sys.argv) > 2:
    json.dump(adv, open(sys.argv[2], "w"), ensure_ascii=False, separators=(",", ":"))
