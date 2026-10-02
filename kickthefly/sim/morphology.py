"""Real neuron morphology loader with neuPrint skeleton fetching and local caching.

Fetches real EM reconstruction skeletons (SWC format) from Janelia neuPrint (MaleCNS v1.0)
for key neuron classes:
  - Giant Fiber escape command neurons (DNp01)
  - Descending steering neurons (DNa02)
  - Mushroom body output neurons (MBONs)
  - Kenyon cells (KCs)

Features:
  - Local caching in data/skeletons/ so skeletons are only downloaded once.
  - Graceful fallback to synthetic fiber approximations if neuPrint is offline or unreachable.
  - Strict rate-limiting to respect server quotas and prevent 429/bans.
  - SWC skeleton parsing and resampling into 3D point arrays matching BrainView render buffers.
"""
from __future__ import annotations

import logging
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
import numpy as np

from kickthefly.core.version import __version__

log = logging.getLogger("kickthefly")

NEUPRINT_BASE_URL = "https://neuprint.janelia.org/api/skeletons/skeleton"
DATASET = "male-cns:v1.0"
KEY_TYPES = ("DNp01", "DNa02", "MBON01", "MBON14", "KCg")
RATE_LIMIT_S = 0.25
_last_request_time = 0.0


def default_cache_dir() -> Path:
    """Where downloaded skeletons are kept. From a source checkout: data/skeletons next to the brain pack. In the exe
    and the AppImage (whose own folders are temporary or read-only): a skeletons folder in the user's data folder
    (~/.local/share/kickthefly/skeletons, Documents\\Kick the Fly\\skeletons)."""
    from kickthefly.core import paths
    user = paths.get().data_dir / "skeletons"
    if getattr(sys, "frozen", False) or os.environ.get("APPIMAGE"):
        return paths.ensure_dir(user)
    return paths.ensure_dir(Path(__file__).resolve().parent.parent.parent / "data" / "skeletons", user)


_OPT_IN = False                     # 3.0 day 3 review: the download is opt-in (Settings > Brain > Download real neuron shapes)


def set_opt_in(on: bool) -> None:
    """The player's choice (config brain.neuron_shapes). Cached skeletons load either way; only fetching needs it."""
    global _OPT_IN
    _OPT_IN = bool(on)


def network_allowed() -> bool:
    """KICK_THE_FLY_OFFLINE=1 keeps the game (and the test suite) from contacting neuPrint; cached skeletons still load. 3.0 day 3
    review: the fetch also goes through core/netguard, so a headless run (--validate, --playthrough, protocols, which call
    netguard.disable and set KTF_NO_NETWORK) and the tests never contact neuPrint either."""
    if not _OPT_IN or os.environ.get("KICK_THE_FLY_OFFLINE", "").strip() in ("1", "true", "yes"):
        return False
    from kickthefly.core import netguard
    return netguard.allowed()[0]


def parse_swc(text: str, n_samples: int = 21) -> np.ndarray | None:
    """Parses SWC format text into an (n_samples, 3) float32 array of coordinates."""
    nodes = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 7:
            try:
                nodes.append([float(parts[2]), float(parts[3]), float(parts[4])])
            except ValueError:
                continue

    if len(nodes) < 2:
        return None

    coords = np.array(nodes, dtype=np.float32)
    if len(coords) == n_samples:
        return coords

    # Evenly sample indices along the reconstruction graph / trajectory
    indices = np.linspace(0, len(coords) - 1, n_samples).astype(np.int64)
    return coords[indices]


def fetch_or_load_skeleton(body_id: int, cache_dir: Path | None = None,
                           allow_network: bool = True, n_samples: int = 21) -> np.ndarray | None:
    """Returns (n_samples, 3) coordinates for a body_id, checking cache first then neuPrint."""
    global _last_request_time
    if cache_dir is None:
        cache_dir = default_cache_dir()

    cache_file = cache_dir / f"{body_id}.swc"
    if cache_file.exists():
        try:
            return parse_swc(cache_file.read_text(encoding="utf-8", errors="ignore"), n_samples)
        except Exception as e:
            log.warning("Failed to read cached skeleton %s: %s", cache_file, e)

    if not allow_network or not network_allowed():
        return None

    # Rate limiting
    now = time.perf_counter()
    elapsed = now - _last_request_time
    if elapsed < RATE_LIMIT_S:
        time.sleep(RATE_LIMIT_S - elapsed)

    url = f"{NEUPRINT_BASE_URL}/{DATASET}/{body_id}?format=swc"
    req = urllib.request.Request(url, headers={"User-Agent": f"KickTheFly/{__version__} (+https://github.com/legendarylolo318-cloud/kick-the-fly)"})
    try:
        _last_request_time = time.perf_counter()
        with urllib.request.urlopen(req, timeout=4.0) as resp:
            content = resp.read()
            try:
                cache_file.write_bytes(content)
            except OSError:
                pass
            return parse_swc(content.decode("utf-8", errors="ignore"), n_samples)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
        log.info("neuPrint skeleton fetch failed for body %d (%s); using synthetic fibers", body_id, e)
        return None


def load_key_skeletons(graph, cache_dir: Path | None = None, allow_network: bool = True,
                       n_samples: int = 21) -> tuple[dict[int, np.ndarray], str]:
    """Loads skeletons for key neurons (DNp01, DNa02, MBONs, KCs).

    Returns:
        (skeletons_by_idx, status_message)
    """
    if graph is None or getattr(graph, "body_id", None) is None:
        return {}, "Skeletons offline - using synthetic fibers"

    types = graph.type.astype(str)
    bodies = graph.body_id
    skeletons: dict[int, np.ndarray] = {}
    network_failed = False

    for t_name in KEY_TYPES:
        idxs = np.flatnonzero(np.char.startswith(types, t_name))
        # Pick up to 2 instances per type for crisp representation
        for idx in idxs[:2]:
            bid = int(bodies[idx])
            if bid <= 0:
                continue
            coords = fetch_or_load_skeleton(bid, cache_dir=cache_dir, n_samples=n_samples,
                                            allow_network=allow_network and not network_failed)
            if coords is not None:
                skeletons[int(idx)] = coords
            else:
                if allow_network:
                    network_failed = True

    if skeletons:
        status = f"Real morphology: {len(skeletons)} neurons drawn from neuPrint skeletons"
    else:
        status = ("Skeletons offline - using synthetic fibers" if _OPT_IN else
                  "Estimated fibers: real shapes are off (Settings > Brain > Download real neuron shapes)")

    return skeletons, status
