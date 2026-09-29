"""Report a bug (2.12): shows exactly what is included, lets you remove items, copies or opens a prefilled issue, never sends."""
from __future__ import annotations

import ast
import json
import socket
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, unquote, urlparse

import pygame
import pytest

from conftest import ROOT
from kickthefly.core import bugreport, selftest
from kickthefly.core.selftest import Check


@pytest.fixture
def state(tmp_path, monkeypatch):
    """A state folder with a 700-line log and a crash report, and no network at all."""
    from kickthefly.core import paths

    d = paths.get().state_dir
    d.mkdir(parents=True, exist_ok=True)
    (d / "kickthefly.log").write_text("".join(f"12:00:{i % 60:02d} INFO line {i} from {Path.home()}/game\n" for i in range(700)))
    (d / "KickTheFly-crash.txt").write_text(f"Kick the Fly crash report\nTraceback...\n  File \"{Path.home()}/x.py\"\nValueError: boom\n")
    monkeypatch.setattr(socket.socket, "connect", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network!")))
    selftest.LAST = selftest.summarize([Check("a", "Adult brain pack", "PASS", "ok"), Check("b", "ffmpeg", "WARN", "missing", "install it")],
                                       app_version="2.12.0")
    yield d
    selftest.LAST = None


def test_the_module_has_no_way_to_reach_the_network():
    """No telemetry, ever: the module imports nothing that opens a connection (webbrowser hands a URL to the player's
    own browser; urllib.parse only builds strings)."""
    tree = ast.parse((ROOT / "kickthefly" / "core" / "bugreport.py").read_text())
    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imported |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            imported.add(n.module)
    banned = {"socket", "requests", "http", "http.client", "urllib.request", "urllib3", "ssl", "smtplib", "ftplib",
              "xmlrpc", "aiohttp", "httpx", "asyncio"}
    assert not imported & banned, imported & banned
    for path in list((ROOT / "kickthefly" / "core").glob("*.py")) + list((ROOT / "kickthefly" / "ui").glob("*.py")):
        if path.name in ("selftest.py", "bugreport.py", "crashscreen.py", "help_ui.py"):
            src = path.read_text()
            assert "urlopen" not in src and "requests." not in src, path.name


def test_collect_scrubs_the_home_folder_and_tails_500_lines(state):
    items = {i.id: i for i in bugreport.collect()}
    assert set(items) == {"summary", "selftest", "log", "crash"}
    log = items["log"].text.splitlines()
    assert len(log) == 500 and "line 699" in log[-1] and "line 200" in log[0]
    assert str(Path.home()) not in items["log"].text and "~/game" in items["log"].text
    assert str(Path.home()) not in items["crash"].text and "ValueError: boom" in items["crash"].text
    assert json.loads(items["selftest"].text)["counts"]["WARN"] == 1
    assert "Kick the Fly" in items["summary"].text and "OS:" in items["summary"].text and "Compute backend" in items["summary"].text
    assert all(i.include for i in items.values())


def test_nothing_that_does_not_exist_is_included(tmp_path, monkeypatch):
    selftest.LAST = None
    items = {i.id: i for i in bugreport.collect()}
    assert not items["crash"].include and not items["selftest"].include
    assert "empty" == items["crash"].size


def test_what_is_shown_is_what_is_copied_and_removed_items_are_gone(state):
    items = bugreport.collect()
    full = bugreport.clipboard_text(items, "it froze when I clicked")
    assert "it froze when I clicked" in full and "ValueError: boom" in full and "line 699" in full
    for it in items:
        if it.id in ("log", "crash"):
            it.include = False
    trimmed = bugreport.clipboard_text(items, "")
    assert "line 699" not in trimmed and "ValueError" not in trimmed and "Adult brain pack" in trimmed
    for it in items:
        it.include = False
    assert bugreport.clipboard_text(items, "").strip() == ""


def test_the_issue_url_is_prefilled_within_the_limit_and_long_parts_go_to_a_file(state):
    items = bugreport.collect()
    title, body, attach = bugreport.compose_issue(items, "it froze", "bugreport.txt")
    url = bugreport.issue_url(title, body)
    assert url.startswith("https://github.com/legendarylolo318-cloud/kick-the-fly/issues/new?") and len(url) <= bugreport.URL_LIMIT
    q = parse_qs(urlparse(url).query)
    assert "it froze" in q["body"][0] and "ffmpeg" in q["body"][0], "the self-test verdicts that are not PASS are in the link"
    assert "line 699" not in q["body"][0] and "ValueError" not in q["body"][0], "long items are not in the URL"
    assert attach and "line 699" in attach and "ValueError: boom" in attach and "Self-test (JSON)" in attach
    assert "bugreport.txt" in q["body"][0] and "attached" in q["body"][0].lower()
    huge = "word " * 5000
    url = bugreport.issue_url("t", huge)
    assert len(url) <= bugreport.URL_LIMIT and "cut to fit the URL" in unquote(url)
    for it in items:
        if it.long:
            it.include = False
    _, body2, attach2 = bugreport.compose_issue(items, "")
    assert attach2 is None and "attached" not in body2.lower()


def test_open_in_browser_hands_over_only_the_url(state, monkeypatch):
    opened = []
    monkeypatch.setattr(bugreport.webbrowser, "open", lambda u: opened.append(u) or True)
    assert bugreport.open_in_browser("https://example.invalid/x")
    assert opened == ["https://example.invalid/x"]
    monkeypatch.setattr(bugreport.webbrowser, "open", lambda u: (_ for _ in ()).throw(RuntimeError("no browser")))
    assert bugreport.open_in_browser("https://x") is False


def test_clipboard_falls_back_to_os_tools_and_says_when_there_is_none(monkeypatch):
    monkeypatch.setattr(bugreport.shutil, "which", lambda n: None)
    ok, msg = bugreport.copy_to_clipboard("hello")
    assert not ok and "no clipboard" in msg
    calls = []
    monkeypatch.setattr(bugreport.shutil, "which", lambda n: "/usr/bin/" + n if n == "wl-copy" else None)
    monkeypatch.setattr(bugreport.subprocess, "run", lambda cmd, **k: calls.append((cmd, k["input"])))
    ok, msg = bugreport.copy_to_clipboard("hello")
    assert ok and calls == [(["wl-copy"], b"hello")]


def test_save_attachment_writes_a_file_next_to_the_logs(state):
    p = bugreport.save_attachment("the long parts")
    assert p is not None and p.read_text() == "the long parts" and p.parent.name == "bugreports"


def test_the_review_page_shows_items_lets_you_remove_them_and_draws_on_the_crash_screen(state, monkeypatch):
    from kickthefly.ui import crashscreen, help_ui, menu as mu

    pygame.init()
    host = crashscreen._Host()
    host.paths = [state / "KickTheFly-crash.txt"]
    m = mu.Menu(host)
    m.pages["crash"] = crashscreen._page_crash
    help_ui.install(m)
    surf = pygame.Surface((1280, 760))
    m.show("crash")
    m.draw(surf, (0, 0))
    assert any(d["id"] == ("crash", "report") for _, _, d in m.hits), "the crash screen offers Report a bug"
    dict(next((d for _, _, d in m.hits if d["id"] == ("crash", "report"))))["click"]()
    assert m.screen == "bugreport"
    m.draw(surf, (0, 0))
    toggles = [d for _, _, d in m.hits if isinstance(d["id"], tuple) and d["id"][0] == "br-inc"]
    assert len(toggles) == 4
    br = m.br
    assert all(it.include for it in br["items"])
    toggles[2]["click"]()                                            # switch the log off
    assert not br["items"][2].include
    m.draw(surf, (0, 0))
    opened = []
    monkeypatch.setattr(bugreport.webbrowser, "open", lambda u: opened.append(u) or True)
    next(d for _, _, d in m.hits if d["id"] == ("br", "open"))["click"]()
    assert opened and "line 699" not in unquote(opened[0]) and br["saved"] and Path(br["saved"]).exists()
    assert "line 699" not in Path(br["saved"]).read_text(), "the log was switched off: it is in neither the URL nor the file"
    assert "ValueError: boom" in Path(br["saved"]).read_text()


def test_crash_handler_writes_the_report_and_only_opens_a_window_in_windowed_runs(state, monkeypatch):
    from kickthefly.core import crash
    from kickthefly.ui import crashscreen

    shown = []
    monkeypatch.setattr(crashscreen, "show", lambda written: shown.append(written))
    monkeypatch.setitem(crash.info, "mode", "")
    written = crash.handle_crash("Traceback: x")
    assert written and not shown, "headless runs (validation, CI) only get the report"
    monkeypatch.setitem(crash.info, "mode", "3d")
    crash.handle_crash("Traceback: y")
    assert len(shown) == 1
    monkeypatch.setenv("KICK_THE_FLY_NO_CRASH_SCREEN", "1")
    crash.handle_crash("Traceback: z")
    assert len(shown) == 1
    monkeypatch.delenv("KICK_THE_FLY_NO_CRASH_SCREEN")
    monkeypatch.setattr(crashscreen, "show", lambda w: (_ for _ in ()).throw(RuntimeError("screen broke")))
    assert crash.handle_crash("Traceback: w"), "a broken crash screen never hides the crash"


def test_bugreport_command_prints_and_saves_and_sends_nothing(state, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bugreport, "collect", lambda selftest_report=None, run_selftest=False: [
        bugreport.Item("summary", "Summary", "Kick the Fly 2.12.0"), bugreport.Item("log", "Log", "a\nb", long=True)])
    assert bugreport.main(SimpleNamespace(out=str(tmp_path / "o"))) == 0
    out = capsys.readouterr().out
    assert "Kick the Fly 2.12.0" in out and "nothing was sent" in out
    assert (tmp_path / "o" / "bugreport.md").read_text().startswith("### Summary")
