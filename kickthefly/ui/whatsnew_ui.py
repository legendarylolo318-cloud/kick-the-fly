"""What's New in 3.0 (3.0 day 5): one screen, shown once on the first launch after an upgrade, skippable (Got it, Esc, or the gamepad's B).

"Once" is a migrated config key: `[first_run] whatsnew_3_0_seen` (core/config.py). An older config.toml lacks it, so the screen shows on its
first launch, the key is written when it is shown, and it never shows again. A fresh install starts with the key set (its welcome is the
tutorial). Settings > Help > What's new in 3.0 opens it again at any time without touching the key.

Every line says what it is and where it lives; nothing here is a claim about the fly that the Lab does not back up."""
from __future__ import annotations

import pygame

from kickthefly.core.i18n import tr
from kickthefly.ui import menu as mu

# (heading, text, where). Short on purpose: the full list is docs/changelog.md.
ITEMS = (
    ("Neurodex, Kill cam, Neuron of the Day", "Collect the cell types your fly fires, watch the last seconds of its brain in slow motion, and meet one curated "
     "neuron a day.", "D, ;, a card at launch"),
    ("Lab toolkit", "Split-GAL4 driver lines, thermogenetics, a virtual patch clamp, simulated calcium imaging and pharmacology, all tagged "
     "CONNECTOME, GAME RULE or MODEL PREDICTION.", "Esc > Lab tools"),
    ("Live inputs, predators, weather, kitchen", "Opt-in microphone and streamer mode (off at every launch), frog, dragonfly and mantis, "
     "rain and storms, and a kitchen arena.", "Esc > Mic and streamer; tools; the arenas"),
    ("Fly arcade", "A tournament of flies and fly racing, each brain steering and walking for itself, with in-game points only "
     "(no money, nothing to buy). Cards are measured, never drawn.", "Esc > Fly arcade"),
    ("Network science, sleep deprivation, sensitivity analysis", "Motifs, rich club and communities of the connectome; a sleep-"
     "deprivation assay; and how fragile each validated behavior is to each model parameter.", "Lab"),
    ("Behavior rigs", "A tethered flight simulator, a fly on a ball, Buridan's paradigm and a four-field olfactory arena. Yaw and walking are "
     "read from real descending neurons; each rig has a headless protocol and a pre-registered assay, and a result the model misses "
     "is reported as a miss.", "Lab > Behavior rigs"),
    ("Mini-papers", "Short guided experiments that reproduce a classic paper step by step: you state a hypothesis, run it on the model, and "
     "see your result next to what the paper found, cited. Where the model fails the paper, it says so and why.", "Esc > Mini-papers; Lab"),
    ("Honest by construction", "Every new behavior is tagged. Validation, assays and rigs keep their pass criteria written down before the run, "
     "and failed results stay in the table.", "README, Lab > Model assumptions"),
)


def page(menu, surf, rect, mouse) -> None:
    key = "whatsnew"
    menu.text(surf, tr("What's new in Kick the Fly 3.0"), (rect.x + 24, rect.y + 16), mu.INK, menu.f_head)
    menu.wrapped(surf, tr("Everything below is optional, and nothing changed what your fly does by itself. This screen shows once."),
                 (rect.x + 24, rect.y + 52), rect.w - 48, mu.LABEL, menu.f_small, max_lines=3)
    body = pygame.Rect(rect.x + 16, rect.y + 96, rect.w - 32, rect.h - 96 - 80)
    off = int(menu.scroll.get(key, 0))
    prev = surf.get_clip()
    surf.set_clip(body)
    menu.clip = body
    y = body.y + 4 - off
    width = body.w - 40
    for head, text, where in ITEMS:
        menu.text(surf, tr(head), (body.x + 8, y), mu.AMBER, menu.f_bold)
        y += menu.f_bold.get_linesize() + 2
        y = menu.wrapped(surf, tr(text), (body.x + 8, y), width, mu.TEXT, menu.f_text, max_lines=6)
        y = menu.wrapped(surf, tr("Where: {where}", where=tr(where)), (body.x + 8, y + 2), width, mu.LABEL, menu.f_small, max_lines=3) + 12
    surf.set_clip(prev)
    menu.clip = None
    menu.content_h[key] = max(0, y + off - body.bottom + 8)
    if menu.content_h[key] > 0:
        frac = body.h / (body.h + menu.content_h[key])
        bar_h = max(30, int(body.h * frac))
        by = body.y + int((body.h - bar_h) * (off / max(1, menu.content_h[key])))
        pygame.draw.rect(surf, (60, 66, 80), (body.right - 6, by, 4, bar_h), border_radius=2)
    menu.button(surf, (rect.centerx - 130, rect.bottom - 62, 260, 46), tr("Got it"), menu.back, style="primary", id=("whatsnew", "ok"),
                tip=tr("Close this. It will not show again; Settings > Help brings it back."))


def install(menu) -> None:
    menu.pages["whatsnew"] = page
