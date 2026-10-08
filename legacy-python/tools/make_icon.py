"""Render the game's fly into a 256 px PNG for the exe icon.  python tools/make_icon.py build/icon.png"""
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pygame

from kickthefly.game import kick_the_fly as k

out = Path(sys.argv[1] if len(sys.argv) > 1 else "build/icon.png")
out.parent.mkdir(parents=True, exist_ok=True)
pygame.init()
pygame.display.set_mode((1, 1))
big = pygame.Surface((300, 300), pygame.SRCALPHA)
fly = k.Fly(0)
fly.p = fly.p - fly.p[k.THX] + (158, 150)      # center the standing fly
k.draw_fly(big, fly, 0.0)
surf = pygame.transform.smoothscale(big, (256, 256))
try:
    pygame.image.save(surf, str(out))
except (NotImplementedError, pygame.error):
    # pygame wheels built without SDL_image (the cp314 one) cannot write PNG; Pillow can.
    from PIL import Image
    Image.frombytes("RGBA", surf.get_size(), pygame.image.tostring(surf, "RGBA")).save(out)
print(f"wrote {out}")
