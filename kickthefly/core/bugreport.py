"""Bug report (2.12): Settings > Help > Report a bug, the crash screen, and `--bugreport`.

What it does: collects a few text items about this install, shows the player EXACTLY what would be included (the text
on screen is the text that is copied, saved or put in the URL, after the scrubbing below), lets them remove any item,
and then either copies it to the clipboard, saves it to a file, or opens a prefilled GitHub issue in their browser.

What it never does: upload anything, phone home, open a network connection, or run in the background. The only network
thing is the browser the player asks for, which is handed a URL and nothing else. There is no telemetry, ever.
Tests (tests/test_bugreport.py) fail if this module imports a networking library.

Items (each can be removed):
  summary    app version, OS, compute backend, Python, frozen/source, display session
  selftest   the self-test report as JSON (the last one from this session, else a quick one is run when you ask)
  log        the last 500 lines of kickthefly.log
  crash      the crash report, if a crash left one

Scrubbing: your home folder and user name are replaced by ~ and <user> in every item, so the preview is what leaves.

GitHub issue URLs are limited (about 8,000 characters). The URL carries the summary and the self-test verdicts; anything
long (the log, the crash trace, the full JSON) goes to a text file next to your logs, whose path is shown, and which you
attach to the issue by hand. The clipboard and file always get everything that is included.
"""
from __future__ import annotations

import getpass
import json
import os
import shutil
import subprocess
import sys
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

REPO = "legendarylolo318-cloud/kick-the-fly"
NEW_ISSUE = f"https://github.com/{REPO}/issues/new"
URL_LIMIT = 6000                 # characters of the whole URL we allow ourselves (GitHub refuses somewhere near 8,000)
LOG_LINES = 500
CRASH_NAME = "KickTheFly-crash.txt"


@dataclass
class Item:
    id: str
    title: str
    text: str
    long: bool = False           # too long for a URL: goes to the attached file
    include: bool = True

    @property
    def size(self) -> str:
        n = len(self.text)
        return f"{n:,} characters, {self.text.count(chr(10)) + 1:,} lines" if n else "empty"


def scrub(text: str) -> str:
    """Replace the home folder and the user name with placeholders."""
    home = str(Path.home())
    for needle, repl in ((home, "~"), (home.replace("\\", "/"), "~")):
        if needle and len(needle) > 3:
            text = text.replace(needle, repl)
    try:
        user = getpass.getuser()
    except Exception:
        user = ""
    if user and len(user) > 2:
        text = text.replace(f"/{user}/", "/<user>/").replace(f"\\{user}\\", "\\<user>\\")
    return text


def _log_path() -> Path | None:
    try:
        from kickthefly.core import paths
        return paths.get().state_dir / "kickthefly.log"
    except Exception:
        return None


def tail_log(n: int = LOG_LINES) -> str:
    p = _log_path()
    if p is None or not p.exists():
        return ""
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return ""
    return "".join(lines[-n:])


def find_crash_report() -> Path | None:
    """The newest crash report the last crash left, in the state folder or beside the exe."""
    from kickthefly.core import crash

    found = []
    cands = [crash.beside_executable()]
    try:
        from kickthefly.core import paths
        cands.append(paths.get().state_dir)
    except Exception:
        pass
    for d in cands:
        if d is not None and (d / CRASH_NAME).exists():
            found.append(d / CRASH_NAME)
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


def summary_text(selftest_report: dict | None = None) -> str:
    from kickthefly.core import crash
    from kickthefly.core.version import __version__

    b = (selftest_report or {}).get("build") or {}
    gl = crash.info
    lines = [
        f"Kick the Fly {__version__}",
        f"OS: {crash.os_description()}",
        f"Python: {sys.version.split()[0]}  frozen: {bool(getattr(sys, 'frozen', False))}  "
        f"AppImage: {bool(os.environ.get('APPIMAGE'))}",
        f"Compute backend: {gl.get('sim_backend', 'not started in this session')} ({gl.get('sim_device', 'unknown device')})",
        f"Mode: {gl.get('mode', 'unknown')}",
        f"Display: {gl.get('video_driver', 'unknown')} {crash.session_description()}".strip(),
        f"GPU: {gl.get('gl_renderer', 'unknown')}  GL: {gl.get('gl_version', 'unknown')}",
    ]
    if b:
        lines.append(f"Self-test verdicts: {selftest_report.get('counts')}")
    return "\n".join(lines)


def collect(selftest_report: dict | None = None, run_selftest: bool = False) -> list[Item]:
    """The items a bug report could contain, scrubbed. selftest_report: the last self-test result, if there is one.
    run_selftest: when there is none, run a quick self-test now (no GPU probing, no smoke run) instead of leaving it out."""
    from kickthefly.core import selftest

    rep = selftest_report or getattr(selftest, "LAST", None)
    if rep is None and run_selftest:
        rep = selftest.run(smoke=False, graphics=False, backends_only=["cpu"])
    items = [Item("summary", "Summary", scrub(summary_text(rep)))]
    if rep is not None:
        items.append(Item("selftest", "Self-test (JSON)", scrub(json.dumps(rep, indent=1)), long=True))
    else:
        items.append(Item("selftest", "Self-test (JSON)", "", long=True, include=False))
    log_text = scrub(tail_log())
    items.append(Item("log", f"Last {LOG_LINES} log lines", log_text, long=True, include=bool(log_text.strip())))
    crash_path = find_crash_report()
    crash_text = ""
    if crash_path is not None:
        try:
            crash_text = scrub(crash_path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            crash_text = ""
    items.append(Item("crash", "Crash report", crash_text, long=True, include=bool(crash_text)))
    return items


def _included(items) -> list[Item]:
    return [i for i in items if i.include and i.text.strip()]


def clipboard_text(items, note: str = "") -> str:
    """Everything included, as one Markdown text (also what "save to file" writes)."""
    parts = []
    if note.strip():
        parts.append(f"**What happened**\n\n{note.strip()}\n")
    for it in _included(items):
        fence = "\n" if it.id == "summary" else "\n```\n"
        parts.append(f"### {it.title}\n{fence}{it.text.rstrip()}{'' if it.id == 'summary' else chr(10) + '```'}\n")
    return "\n".join(parts).rstrip() + "\n"


def selftest_verdicts(items) -> str:
    """The self-test's one-line-per-check verdicts, short enough for a URL."""
    for it in _included(items):
        if it.id == "selftest":
            try:
                rep = json.loads(it.text)
            except ValueError:
                return ""
            return "\n".join(f"- {c['status']}: {c['name']}" for c in rep.get("checks", []) if c["status"] != "PASS") or \
                "- all checks passed"
    return ""


def compose_issue(items, note: str = "", attach_name: str = "") -> tuple[str, str, str | None]:
    """(title, body for the URL, text for the attachment or None). The body carries the summary and the self-test
    verdicts; every long item is left for the attachment, and the body says so."""
    title = "Bug report: "
    body = [f"**What happened**\n\n{note.strip() or '(please describe what you were doing)'}\n"]
    long_items = []
    for it in _included(items):
        if it.long:
            long_items.append(it)
        else:
            body.append(f"### {it.title}\n\n{it.text.rstrip()}\n")
    verdicts = selftest_verdicts(items)
    if verdicts:
        body.append(f"### Self-test (checks that were not PASS)\n\n{verdicts}\n")
    attach = None
    if long_items:
        attach = "\n".join(f"===== {it.title} =====\n{it.text.rstrip()}\n" for it in long_items)
        names = ", ".join(it.title for it in long_items)
        body.append(f"_Attached by hand: `{attach_name or 'the bug report file'}` ({names})._\n")
    return title, "\n".join(body), attach


def issue_url(title: str, body: str, limit: int = URL_LIMIT) -> str:
    """The prefilled GitHub issue URL, with the body cut (and marked) to fit `limit` characters."""
    base = f"{NEW_ISSUE}?title={quote(title)}&body="
    room = max(200, limit - len(base))
    enc = quote(body)
    if len(enc) <= room:
        return base + enc
    cut = body
    mark = "\n\n_(cut to fit the URL: the rest is in the attached file)_"
    while len(quote(cut + mark)) > room and cut:
        cut = cut[: int(len(cut) * 0.9)]
    return base + quote(cut + mark)


def save_attachment(text: str, folder: Path | None = None) -> Path | None:
    if folder is None:
        try:
            from kickthefly.core import paths
            folder = paths.get().state_dir / "bugreports"
        except Exception:
            return None
    try:
        folder.mkdir(parents=True, exist_ok=True)
        p = folder / time.strftime("kickthefly-bugreport-%Y%m%d-%H%M%S.txt")
        p.write_text(text, encoding="utf-8")
        return p
    except OSError:
        return None


def copy_to_clipboard(text: str) -> tuple[bool, str]:
    """Copy through pygame's clipboard, else the OS's own clipboard tool if one is installed (never installs anything)."""
    try:
        import pygame

        if pygame.display.get_init():
            import pygame.scrap

            pygame.scrap.init()
            pygame.scrap.put_text(text)
            if pygame.scrap.get_text() == text:
                return True, "copied to the clipboard"
    except Exception:
        pass
    tools = (["wl-copy"], ["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"], ["clip"], ["pbcopy"])
    for cmd in tools:
        if shutil.which(cmd[0]):
            try:
                subprocess.run(cmd, input=text.encode("utf-8", "replace"), timeout=5, check=True)
                return True, f"copied to the clipboard (with {cmd[0]})"
            except (OSError, subprocess.SubprocessError):
                continue
    return False, "no clipboard is available here"


def open_in_browser(url: str) -> bool:
    """Hand the URL to the player's browser. The browser does the talking; this program sends nothing."""
    try:
        return bool(webbrowser.open(url))
    except Exception:
        return False


def main(args) -> int:
    """--bugreport [--out DIR]: print exactly what the report would include and write it to DIR. Sends nothing."""
    items = collect(run_selftest=True)
    text = clipboard_text(items)
    print(text)
    out = getattr(args, "out", None)
    if out:
        p = Path(out)
        p.mkdir(parents=True, exist_ok=True)
        (p / "bugreport.md").write_text(text, encoding="utf-8")
        print(f"written to {p / 'bugreport.md'} (nothing was sent anywhere)")
    else:
        print("(nothing was sent anywhere; use --out DIR to save this, and attach it to an issue by hand)")
    return 0
