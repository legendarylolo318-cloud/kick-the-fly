"""The loadout editor (default key Q), the tool wheel and the one-time notice for migrated configs (2.13).

The editor is a menu page (`Menu.pages["loadout"]`) so it pauses the game like the Settings do and Esc closes it. It shows
every tool grouped by category with its icon, name, a one-line description and which real neurons it drives (tagged
CONNECTOME or GAME RULE), the hotbar, the presets and up to five saved custom loadouts. Click a card to equip or
unequip it, drag a card onto the hotbar to put it in a slot, drag hotbar slots to reorder, drag one off the hotbar to
remove it. The hand can't be removed. Every edit turns the preset into "Custom" and is saved at once.

All colors come from `palette()`, which follows Settings > Accessibility > Brain view colors, so the selection and tag
colors stay distinguishable with the blue/yellow and high-contrast palettes; nothing here flashes.
"""
from __future__ import annotations

import math

import pygame

from kickthefly.core import loadout as lo
from kickthefly.core.i18n import tr
from kickthefly.ui import menu as mu

PALETTES = {
    "default": dict(select=(255, 176, 64), real=(60, 170, 220), rule=(220, 150, 50), ok=(90, 200, 120)),
    "blue-yellow": dict(select=(255, 224, 70), real=(70, 130, 240), rule=(255, 224, 70), ok=(70, 130, 240)),
    "high-contrast": dict(select=(255, 255, 255), real=(255, 255, 255), rule=(255, 60, 220), ok=(255, 255, 255)),
}


def palette(cfg) -> dict:
    return PALETTES.get(str(cfg.get("access.palette", "default")), PALETTES["default"])


def tag_color(cfg, tag: str):
    p = palette(cfg)
    return p["real"] if tag == lo.CONNECTOME else p["rule"]


def draw_tag(surf, font, pos, tag: str, cfg) -> pygame.Rect:
    img = font.render(tr(tag), True, (12, 14, 18))
    r = pygame.Rect(pos[0], pos[1], img.get_width() + 10, img.get_height() + 2)
    pygame.draw.rect(surf, tag_color(cfg, tag), r, border_radius=5)
    surf.blit(img, img.get_rect(center=r.center))
    return r


def key_label(cfg, action: str) -> str:
    """The key an action is on, as the player sees it, or where to bind it when it has none (an old config can leave
    a newer action unbound: core/config.py)."""
    k = cfg.keys.get(action, "")
    return k.upper() if k else tr("(unbound: Settings > Controls)")


def _k2():
    from kickthefly.game import kick_the_fly as k2
    return k2


# --- the wheel ---------------------------------------------------------------------------------------------------------
WHEEL_R = 150
WHEEL_DEAD = 38


def wheel_center(game) -> tuple[int, int]:
    k2 = _k2()
    return (k2.PLAY_W // 2, (game.hud_h if getattr(game, "three_d", False) else k2.FLOOR) // 2)


def wheel_pick(vec, n: int) -> int | None:
    """The tool a pointer offset (dx, dy from the center) points at, 0 at the top and clockwise, or None in the middle."""
    x, y = vec
    if n <= 0 or math.hypot(x, y) < WHEEL_DEAD:
        return None
    ang = math.atan2(x, -y) % (2 * math.pi)
    return int(round(ang / (2 * math.pi) * n)) % n


def draw_wheel(game, surf) -> None:
    """Every tool the mode allows in a ring. Hold the wheel key (or the pad's wheel button) and point; let go to pick."""
    k2 = _k2()
    cfg = game.cfg
    pal = palette(cfg)
    tools = game._wheel_tools()
    n = len(tools)
    cx, cy = wheel_center(game)
    r = WHEEL_R
    disc = pygame.Surface((2 * r + 130, 2 * r + 130), pygame.SRCALPHA)
    pygame.draw.circle(disc, (8, 10, 16, 200), (r + 65, r + 65), r + 58)
    surf.blit(disc, (cx - r - 65, cy - r - 65))
    pad = getattr(game, "pad", None)
    if game.kwheel_open:
        pick = wheel_pick(game.kwheel_vec, n)
    else:
        pick = getattr(pad, "wheel_pick", None)
    cur = game.tool_name()
    for i, name in enumerate(tools):
        a = i / n * 2 * math.pi
        x, y = cx + r * math.sin(a), cy - r * math.cos(a)
        on = i == pick
        equipped = name in game.loadout
        pygame.draw.circle(surf, (60, 66, 84) if on else (26, 30, 40), (int(x), int(y)), 31)
        pygame.draw.circle(surf, pal["select"] if (on or name == cur) else (mu.BORDER if equipped else (44, 50, 62)),
                           (int(x), int(y)), 31, 3 if (on or name == cur) else 1)
        k2.draw_icon(surf, name, (int(x), int(y) - 4), pal["select"] if on else k2.TEXT)
        slot = game.loadout.slot_of(name)
        if slot is not None:
            game._text(surf, game.cfg.keys.get(f"slot{slot + 1}", "").upper(), (int(x) - 26, int(y) - 26), k2.LABEL,
                       game.f_small)
    shown = tools[pick] if pick is not None else cur
    info = lo.BY_NAME[shown]
    game._text(surf, tr(info.label), (cx, cy - 14), k2.INK, game.f_bold, "center")
    game._text(surf, tr(info.category), (cx, cy + 8), k2.LABEL, game.f_small, "center")
    game._text(surf, tr("let go to pick, Esc to cancel"), (cx, cy + r + 62), k2.LABEL, game.f_small, "center")
    if pick is not None:
        game._text(surf, tr(info.desc), (cx, cy - r - 62), k2.TEXT, game.f_small, "center")


# --- editing -----------------------------------------------------------------------------------------------------------
def commit(menu, tools) -> None:
    """Save this list as the custom loadout, switch to it, and apply it right away."""
    host = menu.host
    cfg = host.cfg
    tools = lo.clean(tools, lab=cfg.lab, larva=host.is_larva)
    hidden = [n for n in cfg.loadout.get("custom", []) if n in lo.BY_NAME and n not in tools
              and not lo.available(n, lab=cfg.lab, larva=host.is_larva)]     # a Lab-only tool chosen in Lab stays chosen
    cfg.custom_loadout([*tools, *hidden])
    host.set_setting("controls.loadout_preset", "custom", save=False, force=True)
    host.refresh_loadout()
    cfg.save()


def _current(menu) -> list[str]:
    return list(menu.host.loadout.tools)


def toggle(menu, name: str) -> None:
    tools = _current(menu)
    if name == lo.HAND:
        menu.flash(tr("The hand is in every loadout."), mu.AMBER)
    elif name in tools:
        tools.remove(name)
        commit(menu, tools)
    else:
        commit(menu, [*tools, name])


def drop_on_slot(menu, payload, index: int) -> None:
    """A dragged card lands on hotbar position `index` (a card from the grid equips, a hotbar slot reorders)."""
    _, name, _ = payload
    tools = _current(menu)
    if name in tools:
        if name == lo.HAND:
            return
        tools.remove(name)
        if tools.index(lo.HAND) >= index:
            index = max(1, index)
        tools.insert(max(1, min(index, len(tools))), name)
    else:
        tools.insert(max(1, min(index, len(tools))), name)
    commit(menu, tools)


def drop_off(menu, payload) -> None:
    _, name, from_slot = payload
    if from_slot is not None and name != lo.HAND:
        tools = _current(menu)
        if name in tools:
            tools.remove(name)
            commit(menu, tools)


def choose_preset(menu, name: str) -> None:
    menu.lo_from = name
    menu.host.set_setting("controls.loadout_preset", name, force=True)


def reset_to_preset(menu) -> None:
    name = getattr(menu, "lo_from", None) or "auto"
    if name == "custom":
        name = "auto"
    menu.lo_from = name
    menu.host.set_setting("controls.loadout_preset", name, force=True)
    menu.flash(tr("Back to the {name} preset.", name=tr(lo.PRESET_LABELS[name])), mu.GOOD)


def save_current(menu, name: str) -> None:
    cfg = menu.host.cfg
    name = name.strip() or f"Custom {len(cfg.loadout['saved']) + 1}"
    if cfg.save_loadout(name, menu.host.loadout.tools):
        cfg.save()
        menu.flash(tr("Saved loadout '{name}'.", name=name), mu.GOOD)
    else:
        menu.flash(tr("You can save up to {n} loadouts. Delete one first.", n=lo.MAX_SAVED), mu.AMBER)


def load_saved(menu, item) -> None:
    commit(menu, item["tools"])
    menu.flash(tr("Loaded '{name}'.", name=item["name"]), mu.GOOD)


# --- the page ----------------------------------------------------------------------------------------------------------
def _why_unavailable(host, t: lo.ToolInfo) -> str:
    if t.lab_only and not host.cfg.lab:
        return tr("Lab mode only")
    if host.is_larva and not t.larva:
        return tr("No larval sensory mapping") + (f": {tr(t.larva_note)}" if t.larva_note else "")
    return ""


def page_editor(menu, surf, rect, mouse) -> None:
    host = menu.host
    cfg = host.cfg
    pal = palette(cfg)
    if not hasattr(menu, "lo_from"):
        menu.lo_from = cfg.get("controls.loadout_preset", "auto")
    lop = host.loadout
    menu.text(surf, tr("TOOL LOADOUT"), (rect.x + 24, rect.y + 14), mu.INK, menu.f_head)
    mode = cfg.get("brain.mode", "play")
    sub = tr("{mode} mode").format(mode=tr(mode.capitalize())) + ("  ·  " + tr("larva: tools without a larval mapping are hidden")
                                                                if host.is_larva else "")
    menu.text(surf, sub, (rect.right - 24, rect.y + 20), mu.LABEL, menu.f_small, "topright")

    # presets
    y = rect.y + 52
    cur_preset = cfg.get("controls.loadout_preset", "auto")
    n = len(lo.CHOICES)
    w = (rect.w - 48) // n
    for i, name in enumerate(lo.CHOICES):
        menu.button(surf, (rect.x + 24 + i * w, y, w - 6, 34), tr(lo.PRESET_LABELS[name]),
                    (lambda nm=name: choose_preset(menu, nm)), active=name == cur_preset, id=("lo-preset", name),
                    tip=tr(lo.PRESET_TIPS[name]), font=menu.f_small)

    # saved loadouts
    y += 44
    saved = cfg.loadout["saved"]
    menu.text(surf, tr("SAVED"), (rect.x + 24, y + 8), mu.LABEL, menu.f_small)
    sw = (rect.w - 48 - 70) // lo.MAX_SAVED
    for i in range(lo.MAX_SAVED):
        x = rect.x + 24 + 70 + i * sw
        if i < len(saved):
            item = saved[i]
            menu.button(surf, (x, y, sw - 36, 30), item["name"], (lambda it=item: load_saved(menu, it)),
                        id=("lo-saved", i), font=menu.f_small,
                        tip=tr("Load this loadout: {tools}", tools=", ".join(item["tools"])))
            menu.button(surf, (x + sw - 34, y, 28, 30), "x", (lambda nm=item["name"]: cfg.delete_loadout(nm) or cfg.save()),
                        id=("lo-del", i), style="danger", font=menu.f_small, tip=tr("Delete this saved loadout."))
        else:
            pygame.draw.rect(surf, mu.BORDER, (x, y, sw - 6, 30), 1, border_radius=8)
            menu.text(surf, tr("empty"), (x + (sw - 6) // 2, y + 15), mu.DIM, menu.f_small, "center")

    # the hotbar
    y += 42
    hb = menu.text(surf, tr("HOTBAR"), (rect.x + 24, y), mu.LABEL, menu.f_small)
    y = max(y + 22, menu.wrapped(surf, tr("Drag to reorder, drag off to remove, drop a card here to add. The hand always stays in slot 1."),
                                 (hb.right + 16, y), rect.right - 24 - hb.right - 16, mu.DIM, menu.f_small, max_lines=2) + 4)
    k2 = _k2()
    tools = lop.tools
    rows = max(1, lop.n_pages)
    # 3.0 release review: ten 84 px slots plus the page labels were wider than the 860 px menu a narrow window (brain panel hidden) gets
    page_w = (max(menu.f_small.size(tr("page {n}", n=r + 1))[0] for r in range(rows)) + 12) if rows > 1 else 0
    slot_w, slot_h = min(84, (rect.w - 48 - page_w) // lo.PAGE_SIZE), 58
    x0 = rect.x + 24
    strip = pygame.Rect(x0, y, 10 * slot_w, rows * (slot_h + 6))
    menu._register(strip, "drop", id=("lo-strip",), drop=lambda p, pos: drop_on_slot(menu, p, len(tools)),
                   accept=lambda p: True)
    for i in range(rows * lo.PAGE_SIZE):
        col, row = i % lo.PAGE_SIZE, i // lo.PAGE_SIZE
        r = pygame.Rect(x0 + col * slot_w, y + row * (slot_h + 6), slot_w - 6, slot_h)
        key = cfg.keys.get(f"slot{col + 1}", "").upper()
        if i < len(tools):
            name = tools[i]
            on = name == host.tool_name()
            fill = (60, 46, 22) if on else (24, 28, 38)
            pygame.draw.rect(surf, fill, r, border_radius=8)
            pygame.draw.rect(surf, pal["select"] if on else mu.BORDER, r, 2 if on else 1, border_radius=8)
            k2.draw_icon(surf, name, (r.centerx, r.y + 24), pal["select"] if on else mu.TEXT)
            short_name = mu.Menu.fit_lines(menu.f_small, k2.TOOLS[k2.TOOL_NAMES.index(name)][1], r.w - 6, 1)[0][0]   # never onto the next slot
            menu.text(surf, short_name, (r.centerx, r.y + 40), mu.TEXT, menu.f_small, "midtop")
            menu.text(surf, key, (r.x + 5, r.y + 3), mu.LABEL, menu.f_small)
            menu._register(r, "drop", id=("lo-drop", i), drop=lambda p, pos, ix=i: drop_on_slot(menu, p, ix),
                           accept=lambda p: True)
            menu._register(r, "dragsrc", id=("lo-slot", i), payload=("tool", name, i),     # after the drop: it wins a press
                           click=(lambda nm=name: host.select_tool(nm)), tip=tr(lo.BY_NAME[name].desc),
                           release_outside=lambda p, pos: drop_off(menu, p))
        else:
            pygame.draw.rect(surf, mu.BORDER, r, 1, border_radius=8)
            menu.text(surf, key, (r.x + 5, r.y + 3), mu.DIM, menu.f_small)
    if rows > 1:                                       # which row is which page (the - and = keys turn it)
        for row in range(rows):
            menu.text(surf, tr("page {n}", n=row + 1), (x0 + 10 * slot_w + 4, y + row * (slot_h + 6) + 4), mu.DIM,
                      menu.f_small)
            menu.text(surf, cfg.keys["page_prev"] + " " + cfg.keys["page_next"], (x0 + 10 * slot_w + 4,
                      y + row * (slot_h + 6) + 24), mu.DIM, menu.f_small)
    y += rows * (slot_h + 6) + 6

    # the grid of every tool, by category (scrolls)
    foot = rect.bottom - 62
    body = pygame.Rect(rect.x + 16, y, rect.w - 32, foot - y - 6)
    key = "loadout"
    off = int(menu.scroll.get(key, 0))
    prev_clip = surf.get_clip()
    menu.clip = body
    surf.set_clip(body)
    yy = body.y - off
    colw = (body.w - 24) // 2
    fsm, fbo = menu.f_small, menu.f_bold
    lh_s = fsm.get_linesize()

    def card_h(t) -> int:
        # 3.0 release review: cards were a fixed 82 px with fixed rows, so at larger text the description and "Drives:" lines overlapped
        # and a long description ran past the card. Measured from the font now; both cards of a row share the taller height.
        tw = colw - 80
        n_desc = len(mu.Menu.fit_lines(fsm, tr(t.desc), tw, 3)[0])
        n_drv = len(mu.Menu.fit_lines(fsm, tr("Drives: {neurons}", neurons=tr(t.neurons)), tw, 3)[0])
        return max(82, 6 + fbo.get_linesize() + 4 + (n_desc + n_drv) * lh_s + 10)

    for cat in lo.CATEGORIES:
        members = [t for t in lo.CATALOG if t.category == cat]
        if not members:
            continue
        menu.text(surf, tr(cat).upper(), (body.x + 8, yy + 4), mu.ACCENT, menu.f_small)
        pygame.draw.line(surf, mu.BORDER, (body.x + 8 + 90, yy + 12), (body.right - 12, yy + 12))
        yy += 24
        row_h = [max(card_h(t) for t in members[j:j + 2]) for j in range(0, len(members), 2)]
        for i, t in enumerate(members):
            cx = body.x + 8 + (i % 2) * (colw + 8)
            cy = yy + sum(h + 6 for h in row_h[:i // 2])
            card = pygame.Rect(cx, cy, colw, row_h[i // 2])
            why = _why_unavailable(host, t)
            equipped = t.name in lop
            hover = card.collidepoint(mouse) and body.collidepoint(mouse)
            fill = (36, 42, 56) if hover and not why else (24, 28, 38)
            pygame.draw.rect(surf, fill, card, border_radius=10)
            pygame.draw.rect(surf, pal["select"] if equipped else mu.BORDER, card, 2 if equipped else 1, border_radius=10)
            k2.draw_icon(surf, t.name, (card.x + 34, card.centery), (mu.DIM if why else (pal["select"] if equipped else mu.TEXT)))
            tx = card.x + 70
            nm = menu.text(surf, tr(t.label), (tx, card.y + 6), mu.DIM if why else mu.INK, menu.f_bold)
            draw_tag(surf, menu.f_small, (nm.right + 8, card.y + 8), t.tag, cfg)
            badge = why or (tr("slot {n}", n=(lop.tools.index(t.name) % lo.PAGE_SIZE) + 1) if equipped else tr("click to equip"))
            menu.text(surf, badge, (card.right - 10, card.y + 8), mu.DIM if (why or not equipped) else pal["ok"],
                      menu.f_small, "topright")
            ty = menu.wrapped(surf, tr(t.desc), (tx, card.y + 6 + fbo.get_linesize() + 4), card.w - 80, mu.TEXT if not why else mu.DIM, fsm,
                              max_lines=3)
            menu.wrapped(surf, tr("Drives: {neurons}", neurons=tr(t.neurons)), (tx, ty), card.w - 80,
                         mu.LABEL, fsm, max_lines=3)
            tip = f"{tr(t.label)}: {tr(t.desc)} " + tr("Drives: {neurons}", neurons=tr(t.neurons)) + f" [{tr(t.tag)}]"
            if why:
                tip = why + ". " + tip
                menu._register(card, "label", id=("lo-card", t.name), tip=tip)
            else:
                menu._register(card, "dragsrc", id=("lo-card", t.name), payload=("tool", t.name, None),
                               click=(lambda nm=t.name: toggle(menu, nm)), tip=tip)
        yy += sum(h + 6 for h in row_h) + 6
    menu.content_h[key] = max(0, yy + off - body.bottom + 8)
    surf.set_clip(prev_clip)
    menu.clip = None
    if menu.content_h[key] > 0:
        frac = body.h / (body.h + menu.content_h[key])
        bar_h = max(30, int(body.h * frac))
        by = body.y + int((body.h - bar_h) * (off / max(1, menu.content_h[key])))
        pygame.draw.rect(surf, (60, 66, 80), (body.right - 6, by, 4, bar_h), border_radius=2)

    # the card being dragged follows the pointer
    d = menu.item_drag
    if d is not None and d["moved"]:
        name = d["payload"][1]
        pygame.draw.circle(surf, (40, 46, 60), d["pos"], 30)
        pygame.draw.circle(surf, pal["select"], d["pos"], 30, 2)
        k2.draw_icon(surf, name, d["pos"], pal["select"])

    # footer
    fy = rect.bottom - 54
    menu.button(surf, (rect.x + 24, fy, 190, 42), tr("Reset to preset"), lambda: reset_to_preset(menu),
                id=("lo", "reset"), tip=tr("Go back to the preset you started from, dropping your changes."))
    default_name = f"Custom {len(saved) + 1}"
    menu.text_field(surf, (rect.x + 232, fy + 6, 200, 30), getattr(menu, "lo_name", "") or default_name,
                    lambda v: setattr(menu, "lo_name", v), id=("lo", "name"), tip=tr("Name for a new saved loadout."))
    menu.button(surf, (rect.x + 442, fy, 170, 42), tr("Save as new"),
                lambda: save_current(menu, getattr(menu, "lo_name", "") or default_name), id=("lo", "save"),
                enabled=True, tip=tr("Save the hotbar as one of your {n} custom loadouts.", n=lo.MAX_SAVED))
    menu.button(surf, (rect.right - 164, fy, 140, 42), tr("Back"), menu.back, style="primary", id=("lo", "back"),
                tip=tr("Close the editor (Esc)."))


def page_notice(menu, surf, rect, mouse) -> None:
    """The one-time popup for a config from before 2.13: keys 1-9, 0, - and = still reach every tool."""
    cx = rect.centerx
    menu.text(surf, tr("Tool loadouts are new"), (cx, rect.y + 40), mu.INK, menu.f_head, "midtop")
    menu.wrapped(surf, tr("The hotbar is now a loadout: a short list of tools on the number keys, with every other tool "
                          "one gesture away on the tool wheel. So your keys keep working, this install was set to the "
                          "All preset, which puts every tool on the hotbar (- and = turn the page). Press {key} any time "
                          "to choose your own tools, or pick a smaller preset.", key=key_label(menu.host.cfg, "loadout")),
                 (rect.x + 60, rect.y + 96), rect.w - 120, mu.TEXT, menu.f_text, max_lines=8)
    menu.button(surf, (cx - 250, rect.bottom - 90, 240, 50), tr("Open the editor"),
                lambda: (menu.back(), menu.host.open_loadout_editor()), style="primary", id=("ln", "open"))
    menu.button(surf, (cx + 10, rect.bottom - 90, 240, 50), tr("Got it"), menu.back, id=("ln", "ok"))


def install(menu) -> None:
    menu.pages["loadout"] = page_editor
    menu.pages["loadout_notice"] = page_notice
