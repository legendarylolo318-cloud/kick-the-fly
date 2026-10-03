"""Self-test (2.13): `--selftest [--out FILE.json]` and Settings > Help > Run self-test.

Each check ends PASS, WARN or FAIL, with a plain-English "fix" for anything that isn't a PASS. Nothing here installs,
downloads, removes or configures anything: it only looks, and says what you could do. It runs from source, from the
exe and from the AppImage (a check that needs a feature the build doesn't have says so instead of failing).

  build          app version, Python, frozen or source, OS
  brain packs    adult (and larva if present): readable, zip CRCs, neuron count, and the SHA-256 (compared with a
                 published checksum when one sits next to it)
  backends       every compute backend that can run here does a quick 200-step run. CPU-side backends (cpu, numba,
                 torch-cpu) must match the NumPy reference spike for spike; GPU backends are compared statistically
                 (brain-wide firing and per-population rates), like tests/test_backends.py. A GPU backend runs in a
                 child process with a time limit, so a driver that hangs or crashes can't take the self-test with it
  graphics       OpenGL 3.3 for the 3D game, and the GPU vendor, renderer and GL version the app itself sees
  audio, ffmpeg, folders, disk, memory (against the per-fly estimate), the display backend (Wayland or X11)
  smoke          a 10-second headless protocol (protocols/selftest_smoke.yaml)

Exit codes: 0 all pass, 1 any FAIL, 3 warnings only (2 is left for bad usage, as elsewhere).
The JSON has the same content as the text and is what Settings > Help > Report a bug attaches.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
ORDER = {PASS: 0, WARN: 1, FAIL: 2}
BACKEND_STEPS = 200
CHILD_TIMEOUT_S = 240
CPU_EXACT = ("cpu", "numba", "torch-cpu")
GPU_BACKENDS = ("torch-cuda", "torch-rocm", "gl")
MIN_DISK_MB, WARN_DISK_MB = 200, 2000
ADULT_NEURONS = 166_700
LAST: dict | None = None          # the newest report of this process (the bug report attaches it)


@dataclass
class Check:
    id: str
    name: str
    status: str
    detail: str
    fix: str = ""
    seconds: float = 0.0
    data: dict = field(default_factory=dict)


def _timed(fn):
    def wrapper(*a, **k):
        t0 = time.perf_counter()
        try:
            out = fn(*a, **k)
        except Exception as e:                                   # a broken check is a FAIL of that check, not a crash
            out = Check(fn.__name__.removeprefix("check_"), fn.__name__.removeprefix("check_").replace("_", " "), FAIL,
                        f"the check itself failed: {type(e).__name__}: {e}",
                        "This is a bug in the self-test. Report it with Settings > Help > Report a bug.")
        for c in (out if isinstance(out, list) else [out]):
            c.seconds = round(time.perf_counter() - t0, 2)
        return out
    wrapper.__name__ = fn.__name__
    return wrapper


# --- build ---------------------------------------------------------------------------------------------------------
def build_info() -> dict:
    from kickthefly.core import crash
    from kickthefly.core.version import __version__

    return dict(app_version=__version__, python=sys.version.split()[0], frozen=bool(getattr(sys, "frozen", False)),
                appimage=bool(os.environ.get("APPIMAGE")), os=crash.os_description(), machine=platform.machine(),
                executable=sys.executable)


@_timed
def check_build() -> Check:
    b = build_info()
    kind = "AppImage" if b["appimage"] else "frozen build (exe)" if b["frozen"] else "source"
    return Check("build", "App and Python", PASS, f"Kick the Fly {b['app_version']}, Python {b['python']}, {kind}, {b['os']}",
                 data=b)


# --- brain packs ---------------------------------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _published_checksum(path: Path) -> str | None:
    """A checksum published next to the pack: <pack>.sha256, or a line for it in a SHA256SUMS file beside it."""
    side = path.with_name(path.name + ".sha256")
    try:
        if side.exists():
            return side.read_text(encoding="utf-8").split()[0].lower()
        sums = path.with_name("SHA256SUMS")
        if sums.exists():
            for line in sums.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) >= 2 and parts[-1].lstrip("*") == path.name:
                    return parts[0].lower()
    except OSError:
        pass
    return None


def _check_pack(brain: str) -> Check:
    from kickthefly.sim import brainpack

    cid = f"brainpack_{brain}"
    name = f"{brain.capitalize()} brain pack"
    path = brainpack.find(brain=brain)
    if path is None:
        if brain == "adult":
            return Check(cid, name, FAIL, "the adult brain pack (kick_brain.npz) was not found",
                         "Rebuild it with:  python -m kickthefly.sim.connectome.loader build && "
                         "python -m kickthefly.sim.brainpack build   (a release build already has it inside).")
        return Check(cid, name, PASS, "not present (optional: the larva brain is built locally, see docs/larva.md)")
    try:
        with zipfile.ZipFile(path) as z:
            bad = z.testzip()
    except (zipfile.BadZipFile, OSError) as e:
        return Check(cid, name, FAIL, f"{path.name} can't be read as a pack: {e}",
                     "The file is damaged. Delete it and rebuild or re-download it; nothing else can fix it.",
                     data=dict(path=str(path)))
    if bad:
        return Check(cid, name, FAIL, f"{path.name} is corrupt (checksum error inside the file: {bad})",
                     "Delete it and rebuild or re-download the pack.", data=dict(path=str(path)))
    digest = _sha256(path)
    n_neurons = None
    try:
        import numpy as np

        with np.load(path, allow_pickle=True) as z:
            for key in ("n", "n_neurons"):
                if key in z.files:
                    n_neurons = int(z[key])
            if n_neurons is None and "type" in z.files:
                n_neurons = int(len(z["type"]))
    except Exception:
        pass
    data = dict(path=str(path), sha256=digest, size_mb=round(path.stat().st_size / 2**20, 1), neurons=n_neurons)
    expected = _published_checksum(path)
    if expected and expected != digest:
        return Check(cid, name, FAIL, f"{path.name}: SHA-256 {digest[:16]}... does not match the published {expected[:16]}...",
                     "Re-download or rebuild the pack: this file is not the one that was published.", data=data)
    if brain == "adult" and n_neurons is not None and n_neurons != ADULT_NEURONS:
        return Check(cid, name, WARN, f"{path.name} has {n_neurons:,} neurons; the game expects {ADULT_NEURONS:,}",
                     "Rebuild the pack with the current version. Results will not match the published validation.", data=data)
    how = "matches the published checksum" if expected else "no published checksum next to it; file integrity checked"
    return Check(cid, name, PASS, f"{path.name}: SHA-256 {digest[:16]}..., {data['size_mb']} MB, {how}", data=data)


@_timed
def check_brainpack_adult() -> Check:
    return _check_pack("adult")


@_timed
def check_brainpack_larva() -> Check:
    return _check_pack("larva")


# --- backends ------------------------------------------------------------------------------------------------------
def run_backend(name: str, steps: int = BACKEND_STEPS, seed: int = 42) -> dict:
    """A quick fixed run on one backend. Returns what the comparison needs (a spike hash, totals, per-block counts)."""
    import numpy as np

    from kickthefly.core import simcore
    from kickthefly.sim.connectome.sim import LIFParams, LIFSim

    _, W, _ = simcore.pack()
    sim = LIFSim(None, LIFParams(backend=name, dtype="float32"), W_in=W, seed=seed)
    if sim.backend.name != name:
        return dict(error=f"asked for {name}, but {sim.backend.name} ran instead")
    h = hashlib.sha256()
    total = 0
    nb = sim.n // 1000
    blocks = np.zeros(nb, np.int64)
    t0 = time.perf_counter()
    for step in range(steps):
        sens = None
        if step % 20 == 10:                                       # a sensory pulse every 100 ms, as in tests/test_backends
            sens = np.zeros(sim.n, np.float32)
            sens[100:150] = 2.0
        sp = np.asarray(sim.step(sens), bool)
        h.update(np.packbits(sp).tobytes())
        total += int(sp.sum())
        blocks += sp[:nb * 1000].reshape(nb, 1000).sum(1)
    dt = time.perf_counter() - t0
    return dict(backend=sim.backend.name, device=str(sim.backend.device), spikes=total, sha=h.hexdigest(),
                blocks=blocks.tolist(), steps=steps, seconds=round(dt, 3), steps_per_s=round(steps / max(dt, 1e-9), 1))


def _child_env(**extra: str) -> dict:
    """The child's environment. From source, the package's folder goes on PYTHONPATH (3.0 release review: run from any other directory,
    `python -m kickthefly` in the child could not import the package, so the gl backend FAILED and the OpenGL check warned for no reason)."""
    env = dict(os.environ, **extra)
    if not getattr(sys, "frozen", False):
        root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def _child_command(*args: str) -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, *args]
    return [sys.executable, "-m", "kickthefly", *args]


def _in_child(name: str, steps: int) -> dict:
    """Run one backend in a child process with a time limit. Used for GPU backends: a hung or crashed driver ends the
    child, not the self-test."""
    env = _child_env(SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy")
    try:
        p = subprocess.run(_child_command("--selftest-child", name, str(steps)), capture_output=True, text=True,
                           timeout=CHILD_TIMEOUT_S, env=env)
    except subprocess.TimeoutExpired:
        return dict(error=f"no answer after {CHILD_TIMEOUT_S} s (the driver may be hung)")
    except OSError as e:
        return dict(error=f"could not start the test process: {e}")
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
    if p.returncode != 0 or not lines:
        tail = (p.stderr.strip().splitlines() or ["no output"])[-1][:200]
        return dict(error=f"the test process ended with exit code {p.returncode}: {tail}")
    try:
        return json.loads(lines[-1])
    except ValueError:
        return dict(error="the test process gave an unreadable answer")


def _statistical(ref: dict, got: dict) -> tuple[bool, str]:
    import numpy as np

    rel = abs(got["spikes"] - ref["spikes"]) / max(ref["spikes"], 1)
    a, b = np.asarray(ref["blocks"], float), np.asarray(got["blocks"], float)
    corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0
    ok = rel < 0.05 and corr > 0.9
    return ok, f"total firing differs by {rel:.1%} from the CPU reference, population rates correlate at r={corr:.3f}"


@_timed
def check_backends(only: list[str] | None = None, steps: int = BACKEND_STEPS) -> list[Check]:
    from kickthefly.sim.connectome import backends

    avail = backends.detect_available_backends()
    names = [n for n in avail if only is None or n in only]
    out: list[Check] = []
    if "cpu" not in names:
        names.insert(0, "cpu")
    results: dict[str, dict] = {}
    for n in names:
        r = _in_child(n, steps) if n in GPU_BACKENDS else run_backend(n, steps)
        results[n] = r
    ref = results.get("cpu", {})
    if "error" in ref:
        return [Check("backend_cpu", "CPU (NumPy) backend", FAIL, ref["error"],
                      "Without the reference backend nothing else can be compared. Check the brain pack check above.")]
    for n in names:
        r = results[n]
        cid, label = f"backend_{n}", f"Backend {n}"
        if "error" in r:
            out.append(Check(cid, label, FAIL, r["error"],
                             f"Pick another backend in Settings > Brain > Compute backend ({'CPU (NumPy)' if n != 'cpu' else 'none is left'}). "
                             "Nothing was changed on your system."))
            continue
        speed = f"{r['steps_per_s']:.0f} steps/s, device: {r['device']}"
        data = dict(r, blocks=None)
        if n == "cpu":
            out.append(Check(cid, label, PASS, f"{BACKEND_STEPS if steps == BACKEND_STEPS else steps} steps ran, {r['spikes']:,} spikes, {speed} (the reference)", data=data))
        elif n in CPU_EXACT:
            same = r["sha"] == ref["sha"]
            out.append(Check(cid, label, PASS if same else FAIL,
                             f"{'identical to' if same else 'DIFFERENT from'} the NumPy reference spike for spike, {speed}",
                             "" if same else "This backend must match NumPy bit for bit. Use Settings > Brain > Compute backend > CPU "
                             "(NumPy) and report this with Settings > Help > Report a bug.", data=data))
        else:
            ok, msg = _statistical(ref, r)
            out.append(Check(cid, label, PASS if ok else WARN, f"{msg}; {speed}",
                             "" if ok else "GPU kernels add inputs in another order, so small differences are normal, but this is "
                             "larger than expected. Results may not be comparable with CPU runs; use CPU (NumPy) for science.",
                             data=data))
    return out


# --- graphics ------------------------------------------------------------------------------------------------------
def gl_info_child() -> dict:
    """What the app's own OpenGL sees (run in a child; a bad driver can't take the self-test down)."""
    try:
        import moderngl
    except Exception as e:
        return dict(error=f"ModernGL is not available in this build ({e})")
    last = None
    for kwargs in ({"standalone": True, "backend": "egl"}, {"standalone": True}):
        try:
            ctx = moderngl.create_context(**kwargs)
            i = ctx.info
            res = dict(vendor=str(i.get("GL_VENDOR", "")), renderer=str(i.get("GL_RENDERER", "")),
                       version=str(i.get("GL_VERSION", "")), version_code=int(ctx.version_code),
                       backend=kwargs.get("backend", "default"))
            ctx.release()
            return res
        except Exception as e:
            last = e
    return dict(error=f"could not create an OpenGL context: {last}")


@_timed
def check_graphics() -> list[Check]:
    env = _child_env(SDL_AUDIODRIVER="dummy")
    try:
        p = subprocess.run(_child_command("--selftest-child", "gl-info", "0"), capture_output=True, text=True, timeout=60,
                           env=env)
        lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
        r = json.loads(lines[-1]) if lines else dict(error="no answer from the test process")
    except subprocess.TimeoutExpired:
        r = dict(error="no answer after 60 s (the graphics driver may be hung)")
    except (OSError, ValueError) as e:
        r = dict(error=str(e))
    if "error" in r:
        return [Check("opengl", "OpenGL 3.3 (3D game)", WARN, r["error"],
                      "The 3D game needs OpenGL 3.3. Without it Kick the Fly starts the 2D game instead, which is fine. "
                      "A newer graphics driver from your OS's own updater would help; this program will not change it.")]
    v = r["version_code"]
    out = [Check("opengl", "OpenGL 3.3 (3D game)", PASS if v >= 330 else WARN,
                 f"GL {r['version']} (context {v / 100:.1f}), vendor: {r['vendor']}, renderer: {r['renderer']}",
                 "" if v >= 330 else "The 3D game needs OpenGL 3.3; it will start the 2D game instead. Update your graphics driver "
                 "through your OS, if you want 3D.", data=r)]
    software = any(s in r["renderer"].lower() for s in ("llvmpipe", "softpipe", "swrast", "software"))
    if software:
        out.append(Check("gpu", "GPU", WARN, f"OpenGL is rendered in software ({r['renderer']})",
                         "There is no hardware GPU driver in use, so 3D will be slow. Installing your GPU's driver is up to "
                         "you; the game never does it."))
    else:
        out.append(Check("gpu", "GPU", PASS, f"{r['vendor']} / {r['renderer']} / GL {r['version']}", data=r))
    return out


# --- environment ---------------------------------------------------------------------------------------------------
@_timed
def check_audio() -> Check:
    if os.environ.get("SDL_AUDIODRIVER") == "dummy":
        return Check("audio", "Audio device", WARN, "the dummy audio driver is set (SDL_AUDIODRIVER=dummy): sound is off",
                     "Unset SDL_AUDIODRIVER to hear the game. Fine on a server or in CI.")
    try:
        import pygame

        pygame.mixer.pre_init(22050, -16, 1, 512)
        pygame.mixer.init()
        cfg = pygame.mixer.get_init()
        pygame.mixer.quit()
        return Check("audio", "Audio device", PASS, f"opened an output device ({cfg[0]} Hz)")
    except Exception as e:
        return Check("audio", "Audio device", WARN, f"no audio device could be opened ({e})",
                     "The game plays silently. Check your system's sound output or that another program isn't holding it.")


@_timed
def check_ffmpeg() -> Check:
    path = shutil.which("ffmpeg")
    if path:
        return Check("ffmpeg", "ffmpeg on PATH", PASS, f"found {path}")
    return Check("ffmpeg", "ffmpeg on PATH", WARN, "ffmpeg was not found",
                 "Video recording (Shift+R) falls back to a short GIF. For MP4, install ffmpeg with your package manager "
                 "yourself; the game never installs anything.")


@_timed
def check_neurodex() -> Check:
    """3.0: the Neurodex's curated facts load and are valid (PyYAML and the data file), and the progress file, if there is
    one, is readable. The game runs without either; only the curated facts and your collection would be missing."""
    from kickthefly.core import neurodex as nd

    try:
        facts = nd.load_facts()
    except ImportError as e:
        return Check("neurodex", "Neurodex facts", WARN, f"PyYAML is missing ({e})",
                     "The Neurodex still works but shows data only. Install the requirements into the game's own venv: "
                     "pip install -r requirements.txt (the game never installs anything itself).")
    except (nd.FactsError, OSError) as e:
        return Check("neurodex", "Neurodex facts", WARN, f"the curated facts can't be used: {e}",
                     "Entries show data only until kickthefly/data/neurodex_facts.yaml is restored (reinstall the game).")
    prog = nd.Progress()
    note = f"; {prog.warnings[0]}" if prog.warnings else ""
    return Check("neurodex", "Neurodex facts", WARN if prog.warnings else PASS,
                 f"{len(facts)} curated facts, all with a citation{note}",
                 "The unreadable progress file was kept aside as neurodex.json.bad; the Neurodex starts empty."
                 if prog.warnings else "")


@_timed
def check_toolkit() -> Check:
    """3.0 day 2: the Lab toolkit's data and optional dependencies. The driver-line table must load (PyYAML and the data file); pynwb
    is optional (NWB export of patch clamp and imaging; CSV and TIFF need nothing extra); the pharmacology and imaging tables must be
    internally consistent. The game runs without any of it; only those Lab screens would be missing."""
    try:
        from kickthefly.lab import genetics, imaging, pharmacology, thermogenetics
    except ImportError as e:
        return Check("toolkit", "Lab toolkit", WARN, f"a toolkit module can't be imported ({e})",
                     "Install the requirements into the game's own venv: pip install -r requirements.txt.")
    try:
        n = len(genetics.table())
        src = genetics.source()
        if n < 1000 or src.get("license") != "CC BY 4.0":
            raise genetics.GeneticsError(f"the driver-line table looks wrong ({n} lines, license {src.get('license')!r})")
        for ind in imaging.INDICATORS.values():
            imaging.time_constants(ind)
        assert set(pharmacology.DRUGS) and set(thermogenetics.EFFECTORS)
    except Exception as e:
        return Check("toolkit", "Lab toolkit", WARN, f"the toolkit's data can't be used: {type(e).__name__}: {e}",
                     "Lab > Genetic toolkit and the other day-2 screens may not work until kickthefly/data/driver_lines.yaml is "
                     "restored (reinstall the game).")
    from kickthefly.lab import nwbexport

    nwb = nwbexport.available()
    return Check("toolkit", "Lab toolkit", PASS if nwb is None else WARN,
                 f"{n:,} driver lines ({src.get('license')}), {len(pharmacology.DRUGS)} drugs, {len(thermogenetics.EFFECTORS)} "
                 f"effectors, {len(imaging.INDICATORS)} indicators" + ("" if nwb is None else "; NWB export is off (pynwb missing)"),
                 "" if nwb is None else "Patch clamp and imaging still export CSV (and imaging TIFF). For NWB: pip install pynwb.")


def check_day3() -> Check:
    """3.0 day 3: the predators, weather, kitchen, microphone and streamer modules load and their tables are consistent. None of them
    needs anything outside the game's own requirements; this check opens no device and no connection."""
    try:
        from kickthefly.core import mic, streamer
        from kickthefly.game import kitchen, predators, weather
        from kickthefly.lab import audio, predators as lab_predators

        assert set(predators.SPECS) == set(predators.KINDS) == set(lab_predators.pr.KINDS)
        assert all(0 <= v <= 1 for _, v in [(n, w) for n, w in weather.PART_WEIGHTS]) and abs(sum(w for _, w in weather.PART_WEIGHTS) - 1) < 1e-9
        assert kitchen.colliders() and len(kitchen.bowl_orchard(0).fruit) == 6
        mic.Analyzer().push(mic.hum(200.0, 0.1))
        assert set(audio.CONDITIONS) and streamer.clean_channel("selftest") == "selftest"
    except Exception as e:
        return Check("day3", "Predators, weather, kitchen, mic, streamer", WARN, f"a 3.0 day 3 module can't be used: {type(e).__name__}: {e}",
                     "Reinstall the game; those features may not work until its files are restored.")
    return Check("day3", "Predators, weather, kitchen, mic, streamer", PASS,
                 f"{len(predators.KINDS)} predators, rain/gusts/storm, the kitchen, the microphone analysis and the vote counter all load")


def check_day4() -> Check:
    """3.0 day 4: network science, tournament, racing, sleep deprivation and sensitivity analysis load, and their small pure parts run
    (a planted-community graph through the community finder, the 13 motif classes, the odds, the points rule, the grids against the
    Lab parameter ranges). SciPy's exact tests are their one dependency. No brain is built, no network used, nothing written."""
    try:
        import numpy as np
        from scipy import stats

        from kickthefly.core import points
        from kickthefly.game import flyduel, flyrace
        from kickthefly.lab import lab, netsci, racing, sensitivity, sleepdep, tournament

        assert stats.binomtest(8, 10, 0.5).pvalue > 0 and stats.wilcoxon([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], alternative="greater").pvalue < 0.01
        assert sorted(set(netsci._TABLE[netsci._TABLE >= 0].tolist())) == list(range(13)) and len(netsci.MOTIFS) == 13
        rng = np.random.default_rng(0)
        n, k = 120, 4                                          # a planted partition: dense inside, sparse between
        lab_true = np.repeat(np.arange(k), n // k)
        src, dst = np.where((rng.random((n, n)) < np.where(lab_true[:, None] == lab_true[None, :], 0.30, 0.01)) & ~np.eye(n, dtype=bool))
        labels, q = netsci.louvain(src, dst, np.ones(len(src)), n, np.random.default_rng(1))
        assert q > 0.4 and netsci.nmi(labels, lab_true) > 0.8
        cards = [dict(sugar_ratio=1.5 + 0.2 * i, walk_level_calm=1.0 + 0.1 * i) for i in range(5)]
        p = racing.win_probabilities(cards)
        assert abs(p.sum() - 1) < 1e-9 and np.all(np.diff(p) > 0) and points.decimal_odds(0.5) >= points.MIN_ODDS
        assert flyrace.speed_of(99.0, 3.0) == flyrace.V_MAX and tournament.SIZES == (4, 8, 16) and flyduel.TICK_STEPS == 4
        for prm in sensitivity.PARAMETERS:
            if prm["kind"] == "lif":
                spec = lab.BY_NAME[prm["id"]]
                assert all(spec[4] <= v <= spec[5] for v in prm["values"]) and abs(spec[3] - prm["default"]) < 1e-12, prm["id"]
        assert sleepdep.DFB_CURRENT_AT_FULL > 0 and len(sensitivity.validated_behaviors()) == 12
    except Exception as e:
        return Check("day4", "Network science, tournament, racing, sleep, sensitivity", WARN,
                     f"a 3.0 day 4 module can't be used: {type(e).__name__}: {e}",
                     "Reinstall the game (and SciPy, which its statistics need); those features may not work until then.")
    return Check("day4", "Network science, tournament, racing, sleep, sensitivity", PASS,
                 "the community finder, the 13 motif classes, the odds, the points rule and the parameter grids all check out")


def check_day5() -> Check:
    """3.0 day 5: the behavior rigs and the mini-papers load and their small pure parts run (the steering rule's dead zone and cap, the platform's
    specular reflection, the preference index, the stripe deviation, six mini-papers each with a citation and a DOI, the What's New flag). No brain is
    built, no network used, nothing written."""
    try:
        import math

        import numpy as np

        from kickthefly.core import config
        from kickthefly.game import rigs
        from kickthefly.lab import classroom, minipapers, rigassay

        class B:
            dt = 0.005

            def hz(self, name):
                return {"turn_r": 9.0, "turn_l": 3.0}.get(name, 4.0)

        st = rigs.Steerer()
        for _ in range(40):
            y = st.update(B())
        assert abs(y - (6.0 - rigs.YAW_DEADZONE_HZ) * rigs.YAW_GAIN) < 1e-6 and rigs.YAW_MAX > y > 0
        _, psi, hit = rigs._reflect_circle(np.array([0.6, 0.0]), math.radians(90), rigs.PLATFORM_RADIUS)
        assert hit and abs(rigs.wrap(psi - math.radians(270))) < 1e-6 and set(rigassay.SCENES) == set(rigs.RIGS)
        a = np.zeros((100, len(rigs.COLS)))
        a[:, 0] = np.arange(1, 101) * 0.2
        assert rigs.preference_index(a, 0.02, settle_s=0.0)["pi"] == 1.0
        assert len(minipapers.ORDER) == 6 and all(minipapers.PAPERS[p].citation and minipapers.PAPERS[p].doi.startswith("10.") for p in minipapers.ORDER)
        assert all(classroom.lecture(f"paper_{p}") is not None for p in minipapers.ORDER) and len(classroom.CURATED_LECTURES) == 5
        assert config.FIRST_RUN_DEFAULTS["whatsnew_3_0_seen"] is False
    except Exception as e:
        return Check("day5", "Behavior rigs and mini-papers", WARN, f"a 3.0 day 5 module can't be used: {type(e).__name__}: {e}",
                     "Reinstall the game; the Lab's Behavior rigs and Mini-papers may not work until then.")
    return Check("day5", "Behavior rigs and mini-papers", PASS,
                 "the steering rule, the platform's reflection, the preference index and the six mini-papers' citations all check out")


def check_microphone() -> Check:
    """Optional. Looks for a capture device WITHOUT opening it: the self-test never listens. The microphone feature is off at every
    launch and is turned on by the player (Esc > Mic and streamer); a machine without a microphone just can't use it."""
    from kickthefly.core import mic

    ok, why = mic.availability()
    return Check("microphone", "Microphone (optional)", PASS,
                 (f"{why}; the self-test did not open it, and the feature is off until you turn it on" if ok else
                  f"not available: {why} (optional: only the microphone feature needs it)"), data=dict(available=ok))


def check_network() -> Check:
    """Optional. Reports whether the network features (Streamer mode: read-only Twitch chat; the brain view's one-time neuPrint
    skeleton download) are allowed here. The self-test never connects to anything, and there is no telemetry."""
    from kickthefly.core import netguard

    ok, why = netguard.allowed()
    return Check("network", "Network features (optional)", PASS,
                 ("Streamer mode could connect to irc.chat.twitch.tv (port 6697) if you turn it on; it is off at every launch and the "
                  "self-test did not connect" if ok else f"off here: {why}") + "; the only other network use is the brain view's one-time "
                 "download of ten neuron skeletons from neuPrint (KICK_THE_FLY_OFFLINE=1 turns it off); no telemetry", data=dict(allowed=ok))


def _writable(d: Path) -> str | None:
    try:
        d.mkdir(parents=True, exist_ok=True)
        probe = d / f".ktf-selftest-{os.getpid()}"
        probe.write_text("x")
        probe.unlink()
        return None
    except OSError as e:
        return str(e)


@_timed
def check_folders() -> list[Check]:
    from kickthefly.core import paths

    p = paths.get()
    out = []
    for cid, label, d, level, why in (
            ("dir_config", "Config folder", p.config_dir, FAIL, "Your settings can't be saved"),
            ("dir_data", "Data folder", p.data_dir, FAIL, "Saves, replays and training memory can't be written"),
            ("dir_state", "Log folder", p.state_dir, WARN, "Logs and crash reports can't be written"),
            ("dir_screenshots", "Screenshot folder", p.pictures_dir, WARN, "Screenshots and GIFs can't be saved")):
        err = _writable(d)
        out.append(Check(cid, label, PASS, f"{d} is writable") if err is None else
                   Check(cid, label, level, f"{d} is not writable ({err})",
                         f"{why}. Make the folder writable, or point KICK_THE_FLY_HOME at a folder you own."))
    return out


def _free_disk_mb(d: Path) -> float | None:
    try:
        probe = d
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        return shutil.disk_usage(probe).free / 2**20
    except OSError:
        return None


@_timed
def check_resources() -> list[Check]:
    from kickthefly.core import paths, platform_env
    from kickthefly.game import kick_the_fly as k2

    out = []
    free = _free_disk_mb(paths.get().data_dir)
    if free is None:
        out.append(Check("disk", "Free disk space", WARN, "could not read the free disk space"))
    elif free < MIN_DISK_MB:
        out.append(Check("disk", "Free disk space", FAIL, f"only {free:,.0f} MB free where your data is kept",
                         "Free some space: saves, replays and recordings can't be written."))
    elif free < WARN_DISK_MB:
        out.append(Check("disk", "Free disk space", WARN, f"{free:,.0f} MB free where your data is kept",
                         "Video and replays can be large; free some space if recording fails."))
    else:
        out.append(Check("disk", "Free disk space", PASS, f"{free / 1024:.1f} GB free where your data is kept"))
    avail = platform_env.available_memory_mb()
    per = k2.BRAIN_MB
    if avail is None:
        out.append(Check("memory", "Free memory", WARN, f"could not read free memory (each fly needs about {per} MB)"))
    else:
        flies = max(0, int((avail - 1500) // per))
        st = PASS if avail >= 1500 + 2 * per else WARN if avail >= 1500 + per else FAIL
        detail = f"{avail:,.0f} MB available; one fly needs about {per} MB on top of ~1.5 GB for the game, so about {flies} flies fit"
        fix = "" if st == PASS else ("Close other programs. With this little memory a second fly may not start."
                                     if st == WARN else "There is not enough free memory to run even one fly comfortably. Close other programs.")
        out.append(Check("memory", "Free memory vs per-fly estimate", st, detail, fix, data=dict(available_mb=round(avail), per_fly_mb=per, flies_fit=flies)))
    return out


@_timed
def check_display() -> Check:
    if not sys.platform.startswith("linux"):
        return Check("display", "Display backend", PASS, f"{platform.system()}: the native window system is used")
    e = os.environ
    session = e.get("XDG_SESSION_TYPE", "") or ("wayland" if e.get("WAYLAND_DISPLAY") else "x11" if e.get("DISPLAY") else "none")
    from kickthefly.core import platform_env

    env = {k: v for k, v in e.items() if not (k == "SDL_VIDEODRIVER" and v == "dummy")}   # a headless run sets it
    want = platform_env.choose_backend(None, None, env=env) or "SDL's choice"
    detail = (f"session {session} (WAYLAND_DISPLAY={e.get('WAYLAND_DISPLAY', '')!r}, DISPLAY={e.get('DISPLAY', '')!r}); "
              f"the game would use {want}")
    if session == "none":
        return Check("display", "Display backend", WARN, "no display is available: " + detail,
                     "Fine for headless runs. To open the game you need a graphical session (or xvfb).", data=dict(session=session))
    return Check("display", "Display backend", PASS, detail, data=dict(session=session, choice=want))


@_timed
def check_smoke(seconds: float | None = None) -> Check:
    from kickthefly.lab import protocol

    root = Path(__file__).resolve().parents[2] / "protocols"
    path = root / "selftest_smoke.yaml"
    if not path.exists():
        path = protocol.find(Path("selftest_smoke.yaml"))
    import tempfile

    with tempfile.TemporaryDirectory(prefix="ktf-selftest-") as tmp:
        p = protocol.load(path)
        if seconds is not None:
            p["duration_s"] = seconds
        folder = protocol.run(p, Path(tmp), 1)
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    rates = {g: c["mean"] for grp in summary.get("mean_rate_hz", {}).values() for g, c in grp.items()}
    if not rates:
        return Check("smoke", "10-second smoke protocol", FAIL, "the protocol ran but recorded nothing",
                     "Report this with Settings > Help > Report a bug.")
    driven = rates.get("looming_detectors", 0.0)
    calm = rates.get("giant_fiber", 0.0)
    if not driven > 0:
        return Check("smoke", "10-second smoke protocol", FAIL, f"the driven looming detectors did not fire ({rates})",
                     "The simulation ran but did not respond to input. Rebuild the brain pack and report it if it persists.",
                     data=rates)
    return Check("smoke", "10-second smoke protocol", PASS,
                 f"{p['duration_s']:g} s of simulated time ran; looming detectors {driven:.1f} Hz, giant fiber {calm:.1f} Hz", data=rates)


# --- running it all ------------------------------------------------------------------------------------------------
def run(*, backends_only: list[str] | None = None, smoke: bool = True, graphics: bool = True,
        progress=None) -> dict:
    """All checks. Returns the report dict (see summarize()). progress(check) is called as each finishes."""
    from kickthefly.core.version import __version__

    steps = [check_build, check_brainpack_adult, check_brainpack_larva, lambda: check_backends(backends_only)]
    if graphics:
        steps.append(check_graphics)
    steps += [check_audio, check_ffmpeg, check_folders, check_neurodex, check_toolkit, check_day3, check_day4, check_day5, check_microphone, check_network,
              check_resources, check_display]
    if smoke:
        steps.append(check_smoke)
    checks: list[Check] = []
    for fn in steps:
        res = fn()
        for c in (res if isinstance(res, list) else [res]):
            checks.append(c)
            if progress:
                progress(c)
    global LAST
    LAST = summarize(checks, app_version=__version__)
    return LAST


def summarize(checks: list[Check], **extra) -> dict:
    counts = {s: sum(1 for c in checks if c.status == s) for s in (PASS, WARN, FAIL)}
    code = 1 if counts[FAIL] else 3 if counts[WARN] else 0
    return dict(format="kick-the-fly-selftest", created=time.strftime("%Y-%m-%d %H:%M:%S"), **extra,
                build=build_info(), counts=counts, exit_code=code, checks=[asdict(c) for c in checks])


def to_text(report: dict) -> str:
    lines = [f"Kick the Fly {report.get('app_version', '')} self-test, {report['created']}", ""]
    for c in report["checks"]:
        lines.append(f"[{c['status']}] {c['name']}: {c['detail']}")
        if c["fix"]:
            lines.append(f"       fix: {c['fix']}")
    n = report["counts"]
    lines += ["", f"{n[PASS]} passed, {n[WARN]} warnings, {n[FAIL]} failed. "
                  f"Exit code {report['exit_code']} (0 all pass, 1 any FAIL, 3 warnings only)."]
    return "\n".join(lines)


def main(args) -> int:
    """--selftest [--out FILE.json] [--backend NAME ...]. Headless: no window is opened."""
    from kickthefly.core import platform_env

    platform_env.attach_console()              # (not headless.prepare(): that forces the dummy audio driver, which the
    only = None                                #  audio check is here to look past)
    b = getattr(args, "sim_backend", None)
    if b and b != "auto":
        only = [b]
    rep = run(backends_only=only, progress=lambda c: print(f"[{c.status}] {c.name}: {c.detail}", flush=True))
    print()
    print(to_text(rep))
    out = getattr(args, "out", None)
    if out:
        p = Path(out)
        if p.suffix.lower() != ".json":
            p = p / "selftest.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rep, indent=1), encoding="utf-8")
        print(f"JSON written to {p}")
    return rep["exit_code"]


def child_main(what: str, steps: int) -> int:
    """--selftest-child NAME STEPS: one backend's quick run, or the GL info, as one JSON line on stdout."""
    if what != "gl-info":
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    try:
        res = gl_info_child() if what == "gl-info" else run_backend(what, steps)
    except Exception as e:
        res = dict(error=f"{type(e).__name__}: {e}")
    print(json.dumps(res), flush=True)
    return 0 if "error" not in res else 1
