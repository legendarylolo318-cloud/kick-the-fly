# Flatpak (local builds)

A manifest for building Kick the Fly as a Flatpak from this checkout. It isn't on Flathub; build it yourself.

## Build and run

Needs `flatpak` and `flatpak-builder` (Arch: `sudo pacman -S flatpak-builder`; Debian/Ubuntu:
`sudo apt install flatpak-builder`) and the brain pack, `data/kick_brain.npz` (README: Run from source; the Flatpak
bundles it like the AppImage does). Add `data/validation_results.json` from `--validate` for the real-science popups.

```bash
flatpak remote-add --user --if-not-exists flathub https://dl.flathub.org/repo/flathub.flatpakrepo
flatpak-builder --user --install-deps-from=flathub --install --force-clean build-flatpak \
    packaging/flatpak/io.github.legendarylolo318_cloud.KickTheFly.yml
flatpak run io.github.legendarylolo318_cloud.KickTheFly
flatpak run io.github.legendarylolo318_cloud.KickTheFly --headless --protocol smoke.yaml --out /tmp/smoke
```

`--install-deps-from=flathub` fetches the Freedesktop 24.08 runtime and SDK into your user installation (a few hundred
MB the first time). Run it from the repo root.

## What's in it

- Runtime `org.freedesktop.Platform` 24.08 (Python 3.12), with numpy, scipy, pygame, pillow, moderngl and pyyaml from
  pinned wheels (`python3-deps.json`). Numba and PyTorch aren't included, as in the AppImage: the simulation runs on
  NumPy (or `--backend gl`).
- Files: the launcher (`kickthefly.sh`) points `XDG_CONFIG_HOME`, `XDG_DATA_HOME` and `XDG_STATE_HOME` back at the
  host's folders, and the manifest grants exactly `~/.config/kickthefly`, `~/.local/share/kickthefly`,
  `~/.local/state/kickthefly` and your Pictures folder. So settings, training memory, saves, exports and crash
  reports are the same files the AppImage and a source install use (README: File locations). If you moved your XDG
  folders somewhere else, add `--filesystem=` overrides for them (`flatpak override --user ...`).
- `--device=all` for the GPU (OpenGL) and gamepads, Wayland with an X11 fallback, PulseAudio, and network only for
  the optional neuPrint skeleton download.

## Updating the pinned wheels

```bash
.venv/bin/pip install --dry-run --quiet --report /tmp/report.json --ignore-installed --only-binary=:all: \
    --python-version 3.12 --implementation cp --platform manylinux_2_28_x86_64 --platform manylinux_2_17_x86_64 \
    --platform manylinux2014_x86_64 --target /tmp/fp "numpy>=2.0" "scipy>=1.13" "pygame>=2.6" "pillow>=10" \
    "moderngl>=5.10" "pyyaml>=6"
```

then rewrite `python3-deps.json` from the report's `download_info` URLs and sha256 hashes (one `file` source each).
