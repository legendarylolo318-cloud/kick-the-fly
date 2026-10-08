"""Lab configuration screen for Targeted Optogenetics Laser.

Target cell types directly in-world with activating or silencing beams.
Scientific note: This models targeted cellular stimulation and silencing.
It does NOT claim Gal4/UAS driver line or opsin-specific kinetics.
"""
from __future__ import annotations

import pygame

from kickthefly.ui import menu as ui
from kickthefly.lab.laser import DEFAULT_TARGETS, LaserState


def _get_laser(host) -> LaserState:
    if not hasattr(host, "laser_state"):
        host.laser_state = LaserState()
    return host.laser_state


def page(m: ui.Menu, surf, rect, mouse) -> None:
    host = m.host
    ls = _get_laser(host)

    m.text(surf, "TARGETED OPTOGENETICS LASER", (rect.x + 24, rect.y + 16), ui.INK, m.f_head)
    y = m.subtitle(surf, rect, "In-world aimable laser for real-time cellular stimulation and silencing without menus.")
    # 3.0 release review: the banner was one fixed line that ran past the panel, the label column and controls were fixed pixels (cut at
    # larger text), and the quick-target chips used button styles that do not exist ("good", "quiet"), so the chosen one never showed.
    note = ("TARGETED CELLULAR STIMULATION / SILENCING: Injects current directly into connectome rows. "
            "Does not model Gal4/UAS drivers or opsin kinetics.")
    n = len(ui.Menu.fit_lines(m.f_small, note, rect.w - 68, 4)[0])
    banner = pygame.Rect(rect.x + 24, y, rect.w - 48, n * m.f_small.get_linesize() + 10)
    pygame.draw.rect(surf, (20, 32, 45), banner, border_radius=6)
    pygame.draw.rect(surf, (60, 140, 220), banner, 1, border_radius=6)
    m.wrapped(surf, note, (banner.x + 10, banner.y + 5), banner.w - 20, (160, 210, 255), m.f_small, max_lines=4)
    y = banner.bottom + 10
    lw = max(m.f_text.size(t)[0] for t in ("Target cell type:", "Laser effect:", "Intensity:", "Trigger mode:", "Pulse duration:")) + 20
    cx = rect.x + 24 + lw
    ch = max(32, m.f_small.get_linesize() + 12)

    # Target cell type
    m.text(surf, "Target cell type:", (rect.x + 24, y + 4), ui.TEXT, m.f_text)
    m.text(surf, f"Active: {ls.target_type}", (cx, y + 4), ui.INK, m.f_bold)

    insp = getattr(host, "inspect", None)
    if insp and insp.get("type"):
        it = insp["type"]
        m.button(surf, (rect.right - 240, y, 216, 28), f"Use Inspected ({it[:10]})",
                 lambda: ls.set_target(it), id="laser_use_insp", font=m.f_small)
    y += max(36, m.f_bold.get_linesize() + 12)

    # Quick target chips
    y = m.flow_buttons(surf, rect.x + 24, y, rect.right - 24,
                       [(tgt, (lambda t=tgt: ls.set_target(t)), ("chip", tgt), ls.target_type.lower() == tgt.lower(), None) for tgt in DEFAULT_TARGETS],
                       h=max(24, m.f_small.get_linesize() + 6)) + 14

    # Mode: Activate vs Silence
    seg_w = max(280, 2 * max(m.f_small.size(t)[0] for t in ("Activate (+current)", "Silence (-current)", "Hold (continuous)", "Pulse (timed)")) + 40)
    m.text(surf, "Laser effect:", (rect.x + 24, y + ch // 2), ui.TEXT, m.f_text, "midleft")
    mode_idx = 0 if ls.mode == "activate" else 1
    m.segmented(surf, (cx, y, seg_w, ch), ["Activate (+current)", "Silence (-current)"],
                mode_idx, lambda i: ls.set_mode("activate" if i == 0 else "silence"), id="laser_mode")
    col = (255, 160, 60) if ls.mode == "activate" else (80, 180, 255)
    cur_val = ls.current_value()
    m.text(surf, f"{cur_val:+.2f} pA/step", (cx + seg_w + 20, y + ch // 2), col, m.f_bold, "midleft")
    y += ch + 12

    # Intensity slider
    m.text(surf, "Intensity:", (rect.x + 24, y + ch // 2), ui.TEXT, m.f_text, "midleft")
    m.slider(surf, (cx, y, seg_w, ch), ls.intensity, 0.1, 3.0, 0.1, "{:.1f}x",
             lambda v: ls.set_intensity(v), lambda: None, id="laser_intensity",
             tip="Current intensity scale multiplier (0.1x to 3.0x)")
    y += ch + 12

    # Trigger mode: Hold vs Pulse
    m.text(surf, "Trigger mode:", (rect.x + 24, y + ch // 2), ui.TEXT, m.f_text, "midleft")
    trig_idx = 0 if ls.trigger_mode == "hold" else 1
    m.segmented(surf, (cx, y, seg_w, ch), ["Hold (continuous)", "Pulse (timed)"],
                trig_idx, lambda i: ls.set_trigger_mode("hold" if i == 0 else "pulse"), id="laser_trigger")
    y += ch + 12

    if ls.trigger_mode == "pulse":
        m.text(surf, "Pulse duration:", (rect.x + 24, y + ch // 2), ui.TEXT, m.f_text, "midleft")
        m.slider(surf, (cx, y, seg_w, ch), ls.pulse_duration * 1000, 50, 1000, 50, "{:.0f} ms",
                 lambda v: setattr(ls, "pulse_duration", v / 1000.0), lambda: None, id="laser_pulse_dur",
                 tip="Duration of single laser pulse upon trigger")
        y += ch + 12

    # Status / In-World Equip Button
    y += 10
    has_laser_tool = False
    from kickthefly.game.kick_the_fly import TOOLS
    for idx, (tname, _, _) in enumerate(TOOLS):
        if tname == "laser":
            has_laser_tool = True
            is_equipped = getattr(host, "tool", None) == idx
            btn_txt = "Equipped in Hand" if is_equipped else "Equip Laser Tool in World"

            def equip(i=idx):
                host.tool = i
                m.back()

            m.button(surf, (rect.x + 24, y, max(240, m.f_bold.size(btn_txt)[0] + 32), 42), btn_txt, equip,
                     style="primary", active=is_equipped, id="laser_equip",
                     tip="Equips the laser tool in hand for immediate 2D or 3D in-world use.")
            break

    y += 56
    m.wrapped(surf, "In-World Usage: Aim crosshair/cursor at the fly's body and click/hold. Beam shines cyan for "
                    "silencing and orange for activation.", (rect.x + 24, y), rect.w - 220, ui.LABEL, m.f_small, max_lines=3)

    m.button(surf, (rect.right - 164, rect.bottom - 58, 140, 42), "Back", m.back, style="primary", id=("laser", "back"))
