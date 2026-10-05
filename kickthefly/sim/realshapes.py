"""Real neuron shapes from the public MaleCNS v1.0 release (3.1.0 task 9). Opt-in download, checksummed, cached, never bundled.

What is public (checked 2026-10-04 on https://male-cns.janelia.org/download/, a metadata-only look; nothing was downloaded to write this):
the release is CC BY 4.0, and Janelia publishes every neuron's reconstructed **skeleton** as one SWC file in a public Google Cloud Storage bucket,

    https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-swc/<bodyId>.swc      (EM space, 8 nm units)

(also Neuroglancer-precomputed, mirrored and template-space copies). Google Cloud Storage serves an `x-goog-hash: crc32c=..., md5=...` header with
every object, which is the checksum this module verifies. **Meshes** are not offered as a download on that page (they live inside the Neuroglancer
precomputed segmentation volume, which is far too large to take pieces of here), so this module draws skeletons, not meshes. No account or token is
needed. The whole set of skeletons is hundreds of gigabytes of files; nothing here downloads it. A neuron's shape (10-100 kB) is fetched the first time
you inspect it, if you opted in.

Rules this module keeps:
  - OFF until the player opts in (Settings > Brain > Download real neuron shapes: the same switch the ten-neuron download has had since 3.0). Cached
    shapes load whether or not it is on. KICK_THE_FLY_OFFLINE=1, headless runs and the test suite (core/netguard.py) never fetch.
  - Checksummed. A download is accepted only if the server's md5 matches the bytes; a file with no checksum header is refused. Each cached file's
    sha256 is kept in manifest.json and checked on every load, so a corrupt or edited cache file is thrown away and refetched, not drawn.
  - Bounded. At most MAX_BYTES a file, at most MAX_NODES nodes; a polite pause between requests; a timeout; one attempt a session per neuron that fails.
  - Cached under the player's data folder (shapes/), never bundled. `clear()` empties it.
  - Falls back. Anything that goes wrong returns None and the game draws the estimated fiber it always drew.
  - Credit. CREDIT is shown wherever a real shape is drawn and is in the README credits and docs/real-shapes.md.

The data: FlyEM at HHMI Janelia with the University of Cambridge, the MRC Laboratory of Molecular Biology and Google Research. The male CNS connectome,
version 1.0, CC BY 4.0. The simulation does not use a shape: every neuron is still a single point; shapes are for looking.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger("kickthefly")

BASE_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-swc"
CREDIT = ("Neuron shapes: male CNS connectome v1.0 (FlyEM, HHMI Janelia, with the University of Cambridge, the MRC LMB and Google Research), "
          "CC BY 4.0, male-cns.janelia.org")
CREDIT_SHORT = "Shape: MaleCNS v1.0, FlyEM / Janelia, CC BY 4.0"
MAX_BYTES = 5_000_000
MAX_NODES = 400_000
RATE_LIMIT_S = 0.25
TIMEOUT_S = 8.0
MANIFEST = "manifest.json"
_last_request = 0.0
_lock = threading.Lock()


class ShapeUnavailable(RuntimeError):
    """No shape: not opted in, offline, a refused or failed download, a bad checksum, or a bad file."""


@dataclass
class Skeleton:
    body_id: int
    xyz: np.ndarray          # (n, 3) float32, EM space (8 nm units, the same space as the cell-body positions)
    radius: np.ndarray       # (n,) float32
    parent: np.ndarray       # (n,) int32, index of the parent node or -1

    @property
    def n(self) -> int:
        return len(self.xyz)

    def segments(self) -> np.ndarray:
        """(m, 2, 3): one line segment per node with a parent."""
        ok = np.flatnonzero(self.parent >= 0)
        return np.stack([self.xyz[ok], self.xyz[self.parent[ok]]], axis=1)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.xyz.min(axis=0), self.xyz.max(axis=0)

    def cable_length_um(self) -> float:
        s = self.segments()
        return float(np.linalg.norm(s[:, 0] - s[:, 1], axis=1).sum() * 0.008) if len(s) else 0.0


def parse_swc(text: str, body_id: int = 0) -> Skeleton:
    """The whole SWC (id, type, x, y, z, radius, parent per line), not the 21-point sample morphology.parse_swc keeps."""
    ids, xyz, rad, par = [], [], [], []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        if len(p) < 7:
            continue
        try:
            ids.append(int(p[0]))
            xyz.append((float(p[2]), float(p[3]), float(p[4])))
            rad.append(float(p[5]))
            par.append(int(p[6]))
        except ValueError:
            continue
        if len(ids) > MAX_NODES:
            raise ShapeUnavailable(f"more than {MAX_NODES:,} nodes")
    if len(ids) < 2:
        raise ShapeUnavailable("fewer than two nodes")
    index = {i: k for k, i in enumerate(ids)}
    parent = np.array([index.get(p, -1) for p in par], np.int32)
    a = np.array(xyz, np.float32)
    if not np.all(np.isfinite(a)):
        raise ShapeUnavailable("non-finite coordinates")
    return Skeleton(int(body_id), a, np.array(rad, np.float32), parent)


def goog_hashes(headers) -> dict[str, bytes]:
    """The checksums in an x-goog-hash header: 'crc32c=AAAA==, md5=BBBB==' -> {'md5': raw bytes, 'crc32c': raw bytes}."""
    out = {}
    raw = headers.get("x-goog-hash") or ""
    for part in (raw if isinstance(raw, str) else str(raw)).split(","):
        k, _, v = part.strip().partition("=")
        if k and v:
            try:
                out[k] = base64.b64decode(v)
            except Exception:
                pass
    return out


def default_dir() -> Path:
    from kickthefly.core import paths

    return paths.ensure_dir(paths.get().data_dir / "shapes")


def opted_in() -> bool:
    from kickthefly.sim import morphology

    return bool(morphology._OPT_IN)


def network_allowed() -> tuple[bool, str]:
    """Opted in, not KICK_THE_FLY_OFFLINE, and the netguard allows it (headless runs and the test suite do not)."""
    if not opted_in():
        return False, "real neuron shapes are off (Settings > Brain > Download real neuron shapes)"
    if os.environ.get("KICK_THE_FLY_OFFLINE", "").strip() in ("1", "true", "yes"):
        return False, "KICK_THE_FLY_OFFLINE is set"
    from kickthefly.core import netguard

    return netguard.allowed()


class ShapeStore:
    """The cache folder (<id>.swc and manifest.json) and the one place a shape is fetched."""

    def __init__(self, folder: Path | None = None, base_url: str | None = None):
        self.folder = Path(folder) if folder else default_dir()
        self.folder.mkdir(parents=True, exist_ok=True)
        self.base_url = (base_url or os.environ.get("KICK_THE_FLY_SHAPE_URL") or BASE_URL).rstrip("/")
        self._mf: dict | None = None
        self._failed: dict[int, str] = {}

    # --- the manifest ------------------------------------------------------------------------------------------------------------
    def manifest(self) -> dict:
        if self._mf is None:
            try:
                self._mf = json.loads((self.folder / MANIFEST).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self._mf = {}
        return self._mf

    def _save_manifest(self) -> None:
        tmp = self.folder / (MANIFEST + ".tmp")
        tmp.write_text(json.dumps(self.manifest(), indent=1), encoding="utf-8")
        tmp.replace(self.folder / MANIFEST)

    def path(self, body_id: int) -> Path:
        return self.folder / f"{int(body_id)}.swc"

    # --- the cache ---------------------------------------------------------------------------------------------------------------
    def cached(self, body_id: int) -> bool:
        return self._verified_bytes(body_id) is not None

    def _verified_bytes(self, body_id: int) -> bytes | None:
        p = self.path(body_id)
        rec = self.manifest().get(str(int(body_id)))
        if not p.exists() or not rec:
            return None
        try:
            data = p.read_bytes()
        except OSError:
            return None
        if hashlib.sha256(data).hexdigest() != rec.get("sha256") or len(data) != rec.get("size"):
            log.warning("cached shape %s does not match its checksum; discarding it", p.name)
            self.discard(body_id)
            return None
        return data

    def discard(self, body_id: int) -> None:
        self.path(body_id).unlink(missing_ok=True)
        if self.manifest().pop(str(int(body_id)), None) is not None:
            self._save_manifest()

    def load(self, body_id: int) -> Skeleton | None:
        """From the cache only (checked against its sha256). None if absent or bad."""
        data = self._verified_bytes(body_id)
        if data is None:
            return None
        try:
            return parse_swc(data.decode("utf-8", errors="ignore"), body_id)
        except ShapeUnavailable:
            self.discard(body_id)
            return None

    def size_bytes(self) -> int:
        return sum(f.stat().st_size for f in self.folder.glob("*.swc"))

    def count(self) -> int:
        return sum(1 for _ in self.folder.glob("*.swc"))

    def clear(self) -> int:
        n = 0
        for f in self.folder.glob("*.swc"):
            f.unlink(missing_ok=True)
            n += 1
        self._mf = {}
        (self.folder / MANIFEST).unlink(missing_ok=True)
        return n

    # --- fetching -----------------------------------------------------------------------------------------------------------------
    def fetch(self, body_id: int) -> Skeleton:
        """Download, verify and cache one neuron's skeleton. Raises ShapeUnavailable with the reason on anything wrong."""
        global _last_request
        body_id = int(body_id)
        if body_id <= 0:
            raise ShapeUnavailable("this neuron has no body id")
        ok, why = network_allowed()
        if not ok:
            raise ShapeUnavailable(why)
        if body_id in self._failed:
            raise ShapeUnavailable(self._failed[body_id])
        with _lock:                                               # one request at a time, with a pause between them
            wait = RATE_LIMIT_S - (time.perf_counter() - _last_request)
            if wait > 0:
                time.sleep(wait)
            _last_request = time.perf_counter()
        url = f"{self.base_url}/{body_id}.swc"
        from kickthefly.core import netguard
        from kickthefly.core.version import __version__

        req = urllib.request.Request(url, headers={"User-Agent": f"KickTheFly/{__version__} (+https://github.com/legendarylolo318-cloud/kick-the-fly)"})
        cid = netguard.register("real neuron shape", urllib.parse.urlparse(url).hostname or "", 443, f"MaleCNS v1.0 skeleton {body_id}")
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                if resp.status != 200:
                    raise ShapeUnavailable(f"HTTP {resp.status}")
                size = int(resp.headers.get("Content-Length") or 0)
                if size > MAX_BYTES:
                    raise ShapeUnavailable(f"{size:,} bytes is over the {MAX_BYTES:,} byte limit")
                data = resp.read(MAX_BYTES + 1)
                hashes = goog_hashes(resp.headers)
        except urllib.error.HTTPError as e:
            self._failed[body_id] = f"no published skeleton (HTTP {e.code})" if e.code == 404 else f"HTTP {e.code}"
            raise ShapeUnavailable(self._failed[body_id]) from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise ShapeUnavailable(f"could not reach the server ({type(e).__name__})") from None
        finally:
            netguard.unregister(cid)
        if len(data) > MAX_BYTES:
            raise ShapeUnavailable(f"over the {MAX_BYTES:,} byte limit")
        want = hashes.get("md5")
        if want is None:
            self._failed[body_id] = "the server offered no checksum, so the file was refused"
            raise ShapeUnavailable(self._failed[body_id])
        if hashlib.md5(data).digest() != want:
            self._failed[body_id] = "the download did not match the server's md5 checksum, so it was thrown away"
            raise ShapeUnavailable(self._failed[body_id])
        try:
            skel = parse_swc(data.decode("utf-8", errors="ignore"), body_id)
        except ShapeUnavailable as e:
            self._failed[body_id] = f"the file is not a usable skeleton ({e})"
            raise ShapeUnavailable(self._failed[body_id]) from None
        tmp = self.path(body_id).with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(self.path(body_id))
        self.manifest()[str(body_id)] = dict(size=len(data), sha256=hashlib.sha256(data).hexdigest(), md5=base64.b64encode(want).decode(),
                                             url=url, fetched=time.strftime("%Y-%m-%d %H:%M:%S"), license="CC BY 4.0", source="MaleCNS v1.0 (FlyEM, Janelia)")
        self._save_manifest()
        return skel

    def get(self, body_id: int, allow_network: bool = True) -> Skeleton | None:
        """The cached shape, or (opted in, allowed) the downloaded one; None on any failure, which sets `last_error`."""
        s = self.load(body_id)
        if s is not None or not allow_network:
            return s
        try:
            return self.fetch(body_id)
        except ShapeUnavailable as e:
            self.last_error = str(e)
            return None

    last_error = ""


class ShapeLoader:
    """What the game holds: asks for shapes on a background thread and says, for the UI, what state each is in. Thread-safe."""

    def __init__(self, store: ShapeStore | None = None):
        self.store = store
        self._state: dict[int, tuple[str, str, Skeleton | None]] = {}
        self._lock = threading.Lock()

    def _store(self) -> ShapeStore:
        if self.store is None:
            self.store = ShapeStore()
        return self.store

    def request(self, body_id: int | None) -> None:
        """Start getting a shape (from the cache at once if it is there, else a download if allowed)."""
        if body_id is None or int(body_id) <= 0:
            return
        body_id = int(body_id)
        with self._lock:
            if body_id in self._state:
                return
            self._state[body_id] = ("loading", "", None)
        try:
            s = self._store().load(body_id)
        except Exception:
            s = None
        if s is not None:
            self._set(body_id, "ready", "cached", s)
            return
        ok, why = network_allowed()
        if not ok:
            self._set(body_id, "off", why, None)
            return
        threading.Thread(target=self._run, args=(body_id,), name="real-shape", daemon=True).start()

    def _run(self, body_id: int) -> None:
        try:
            s = self._store().fetch(body_id)
            self._set(body_id, "ready", "downloaded", s)
        except ShapeUnavailable as e:
            self._set(body_id, "failed", str(e), None)
        except Exception as e:                        # never let a drawing helper kill the game
            self._set(body_id, "failed", f"{type(e).__name__}: {e}", None)

    def _set(self, body_id: int, state: str, msg: str, s: Skeleton | None) -> None:
        with self._lock:
            self._state[body_id] = (state, msg, s)

    def status(self, body_id: int | None) -> tuple[str, str]:
        """(state, message): none | loading | ready | off | failed."""
        if body_id is None:
            return "none", ""
        with self._lock:
            st = self._state.get(int(body_id))
        return ("none", "") if st is None else (st[0], st[1])

    def skeleton(self, body_id: int | None) -> Skeleton | None:
        if body_id is None:
            return None
        with self._lock:
            st = self._state.get(int(body_id))
        return None if st is None else st[2]

    def forget(self, body_id: int) -> None:
        """Ask again next time (after the player turns the download on)."""
        with self._lock:
            self._state.pop(int(body_id), None)

    def reset_offline(self) -> None:
        with self._lock:
            for k in [k for k, v in self._state.items() if v[0] in ("off", "failed")]:
                self._state.pop(k)
