"""Classroom mode and lecture presentation UI screen.

Sequential step-by-step walkthroughs of verified Drosophila neural circuits
with Next/Back controls, step explanations, participating neuron breakdowns,
scientific citations, and interactive live demonstration actions.
"""
from __future__ import annotations

import pygame

from kickthefly.ui import menu as ui
from kickthefly.lab.classroom import CURATED_LECTURES, ClassroomSession


def _get_session(host) -> ClassroomSession:
    if not hasattr(host, "classroom_session") or host.classroom_session is None:
        br = getattr(host, "brain", None)
        host.classroom_session = ClassroomSession(lecture_id="looming", brain=br)
    elif getattr(host, "brain", None) is not None:
        host.classroom_session.brain = host.brain
    return host.classroom_session


def page(m: ui.Menu, surf, rect: pygame.Rect, mouse) -> None:
    """3.0 release review: the lecture tabs drew their title and then an empty button on top of it (blank tabs); the arrows and the
    play sign (U+25B6, U+21BA, U+27F5, U+27F6) are not in the game's font (empty boxes); and every box had a fixed height, so at larger text the neuron cards,
    the takeaway and the explanation overlapped. The step card now flows and scrolls, and every size comes from the font."""
    host = m.host
    sess = _get_session(host)
    proto = sess.protocol
    step = sess.current_step
    fs, ft, fb = m.f_small, m.f_text, m.f_bold
    lh = fs.get_linesize()

    # Header
    m.text(surf, "CLASSROOM MODE & LECTURE PROTOCOLS", (rect.x + 24, rect.y + 16), ui.INK, m.f_head)
    y = m.wrapped(surf, "Curated sequential demonstrations for lectures and teaching. "
                        "Every behavior is either grounded in connectome wiring or labelled a game rule.",
                  (rect.x + 24, max(m.under_heading(rect), rect.y + 48)), rect.w - 48, ui.LABEL, fs, max_lines=2) + 8

    # Lecture selection tabs across top (they wrap onto a second row when they do not fit)
    lab = m.text(surf, "Select Lecture:", (rect.x + 24, y + 4), ui.TEXT, fs)
    lectures_meta = [
        ("looming", "Looming escape"),
        ("tmaze", "T-maze conditioning"),
        ("moonwalker", "Moonwalker (MDN)"),
        ("sugar", "Sugar feeding"),
        ("gf_lesion", "GF lesion"),
    ]
    th = max(26, lh + 8)
    x0 = lab.right + 12
    tx = x0
    for lid, ltitle in lectures_meta:
        tw = fs.size(ltitle)[0] + 24
        if tx + tw > rect.right - 24 and tx > x0:
            tx, y = x0, y + th + 6
        m.button(surf, (tx, y, tw, th), ltitle, lambda id_=lid: sess.set_lecture(id_), id=("class_tab", lid), font=fs,
                 active=sess.lecture_id == lid)
        tx += tw + 8
    y += th + 10

    # Protocol title & Step indicator banner
    step_str = f"Step {sess.step_idx + 1} of {sess.total_steps}"
    sw = fb.size(step_str)[0]
    title_lines = ui.Menu.fit_lines(fb, f"Lecture: {proto.title}", rect.w - 48 - sw - 40, 2)[0]
    banner = pygame.Rect(rect.x + 24, y, rect.w - 48, max(32, len(title_lines) * fb.get_linesize() + 10))
    pygame.draw.rect(surf, (20, 24, 34), banner, border_radius=6)
    pygame.draw.rect(surf, (60, 70, 95), banner, 1, border_radius=6)
    for i, ln in enumerate(title_lines):
        m.text(surf, ln, (banner.x + 12, banner.y + 5 + i * fb.get_linesize()), (140, 190, 255), fb)
    m.text(surf, step_str, (banner.right - 12, banner.centery), ui.AMBER, fb, "midright")
    y = banner.bottom + 10

    # Step Card (scrolls when its content is taller than the room left)
    card = pygame.Rect(rect.x + 24, y, rect.w - 48, rect.bottom - 72 - y)
    pygame.draw.rect(surf, (18, 22, 32), card, border_radius=8)
    pygame.draw.rect(surf, (40, 48, 66), card, 1, border_radius=8)
    key = "lab_classroom"
    off = int(m.scroll.get(key, 0))
    inner = card.inflate(-4, -4)
    prev = surf.get_clip()
    surf.set_clip(inner)
    m.clip = inner
    top = card.y + 16 - off

    # Step title and grounding tag (the tag goes under the title when both do not fit on one line)
    gw = fs.size(step.grounding)[0]
    tr_ = m.wrapped(surf, step.title, (card.x + 18, top), card.w - 36 - (gw + 20 if card.w > 900 else 0), ui.INK, m.f_head, max_lines=3)
    if card.w > 900:
        m.text(surf, step.grounding, (card.right - 18, top + 4), (100, 200, 140), fs, "topright")
        content_y = tr_ + 10
    else:
        content_y = m.wrapped(surf, step.grounding, (card.x + 18, tr_ + 2), card.w - 36, (100, 200, 140), fs, max_lines=3) + 10
    col_top = content_y

    # Left Column: Narrative Explanation & Key Takeaway & Live Action (w = ~60% card)
    left_w = int(card.w * 0.58)
    m.text(surf, "CIRCUIT MECHANISM & EXPLANATION", (card.x + 18, content_y), ui.LABEL, fs)
    content_y += lh + 4
    content_y = m.wrapped(surf, step.explanation, (card.x + 18, content_y), left_w - 24, ui.TEXT, ft, max_lines=99) + 12

    # Key Takeaway Box
    kt_lines = ui.Menu.fit_lines(fs, step.key_takeaway, left_w - 44, 99)[0]
    takeaway_rect = pygame.Rect(card.x + 18, content_y, left_w - 24, 8 + lh + 2 + len(kt_lines) * lh + 8)
    pygame.draw.rect(surf, (28, 38, 52), takeaway_rect, border_radius=6)
    pygame.draw.rect(surf, (70, 120, 180), takeaway_rect, 1, border_radius=6)
    m.text(surf, "Key Takeaway:", (takeaway_rect.x + 10, takeaway_rect.y + 8), (140, 200, 255), fs)
    m.wrapped(surf, step.key_takeaway, (takeaway_rect.x + 10, takeaway_rect.y + 8 + lh + 2), takeaway_rect.w - 20, ui.INK, fs, max_lines=99)
    content_y = takeaway_rect.bottom + 10

    # Action demonstration button
    if step.action:
        act_desc = step.action.get("kind", "demo").lower()
        target_name = step.action.get("target", "")
        btn_label = f"Demonstrate step: {act_desc} {target_name}".strip()
        bh = max(36, fb.get_linesize() + 12)
        bw = min(left_w - 24, fb.size(btn_label)[0] + 32)
        m.button(surf, (card.x + 18, content_y, bw, bh), btn_label, lambda: sess.execute_step_action(getattr(host, "brain", None), host),
                 id="class_run_demo", style="primary")
        content_y += bh + 6
        if sess.last_action_applied:
            content_y = m.wrapped(surf, f"Status: {sess.last_action_applied}", (card.x + 18, content_y), left_w - 24, (120, 220, 150), fs, max_lines=3)
    else:
        content_y = m.wrapped(surf, "No active stimulus required for this step.", (card.x + 18, content_y + 4), left_w - 24, ui.LABEL, fs, max_lines=2)

    # Right Column: Neurons Involved & Scientific Citation (w = ~40% card)
    right_x = card.x + left_w + 12
    right_w = card.w - left_w - 30
    m.text(surf, "NEURONS INVOLVED", (right_x, col_top), ui.LABEL, fs)
    ny = col_top + lh + 6
    for nr in step.neurons[:4]:
        cnt = f"n={nr.count}" if nr.count > 0 else ""
        name_lines = ui.Menu.fit_lines(fb, nr.label, right_w - 16 - (fs.size(cnt)[0] + 10 if cnt else 0), 2)[0]
        role_lines = ui.Menu.fit_lines(fs, nr.role, right_w - 16, 99)[0]
        n_box = pygame.Rect(right_x, ny, right_w, 6 + len(name_lines) * fb.get_linesize() + 2 + len(role_lines) * lh + 6)
        pygame.draw.rect(surf, (24, 28, 40), n_box, border_radius=5)
        pygame.draw.rect(surf, (45, 54, 76), n_box, 1, border_radius=5)
        yy = n_box.y + 6
        for ln in name_lines:
            m.text(surf, ln, (n_box.x + 8, yy), (255, 200, 110), fb)
            yy += fb.get_linesize()
        if cnt:
            m.text(surf, cnt, (n_box.right - 8, n_box.y + 6), ui.LABEL, fs, "topright")
        m.wrapped(surf, nr.role, (n_box.x + 8, yy + 2), n_box.w - 16, ui.TEXT, fs, max_lines=99)
        ny = n_box.bottom + 6

    ny += 8
    # Scientific Citation
    m.text(surf, "SCIENTIFIC CITATION", (right_x, ny), ui.LABEL, fs)
    ny += lh + 4
    c_lines = ui.Menu.fit_lines(fs, step.citation, right_w - 20, 99)[0]
    cite_box = pygame.Rect(right_x, ny, right_w, 16 + len(c_lines) * lh)
    pygame.draw.rect(surf, (15, 20, 30), cite_box, border_radius=6)
    pygame.draw.rect(surf, (55, 75, 105), cite_box, 1, border_radius=6)
    m.wrapped(surf, step.citation, (cite_box.x + 10, cite_box.y + 8), cite_box.w - 20, (180, 210, 255), fs, max_lines=99)

    bottom = max(content_y, cite_box.bottom) + 12
    m.content_h[key] = max(0, bottom + off - card.bottom)
    surf.set_clip(prev)
    m.clip = None

    # Bottom Control Bar: buttons sized from their labels
    ctrl_y = rect.bottom - 58
    x = rect.x + 24
    can_prev = sess.step_idx > 0
    can_next = sess.step_idx < sess.total_steps - 1
    for label, fn, ident, style, en in (("Reset protocol", sess.reset, "class_reset", "normal", True),
                                        ("< Previous step", (sess.prev_step if can_prev else lambda: None), "class_prev", "normal", can_prev),
                                        ("Next step >", (sess.next_step if can_next else lambda: None), "class_next", ("primary" if can_next else "normal"), can_next)):
        bw = max(140, fb.size(label)[0] + 28)
        m.button(surf, (x, ctrl_y, bw, 42), label, fn, id=ident, style=style, enabled=en)
        x += bw + 12

    # Back to Lab Hub button
    m.button(surf, (rect.right - 164, ctrl_y, 140, 42), "Back", m.back, style="primary", id="class_back")
