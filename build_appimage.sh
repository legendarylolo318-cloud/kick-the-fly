#!/usr/bin/env bash
# Builds dist/KickTheFly-x86_64.AppImage: one file with the game, Python, numpy/scipy/pygame and the brain pack inside.
# Needs data/kick_brain.npz first:  python -m kickthefly.sim.brainpack build   (after the connectome build in README.md)
# Build on an old glibc (the release uses ubuntu-22.04) so the AppImage runs on most distros.
# data/validation_results.json, if present, is bundled so the real-science popups work (python kick_the_fly.py
# --headless --validate --out data/validation_results.json).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
APP_ID=io.github.legendarylolo318_cloud.KickTheFly
VERSION=$(sed -n 's/^__version__ = "\(.*\)"/\1/p' kickthefly/core/version.py)

if [ ! -f data/kick_brain.npz ]; then
    echo "data/kick_brain.npz missing: run 'python -m kickthefly.sim.brainpack build' first" >&2
    exit 1
fi

PY=${PYTHON:-$(command -v python3.11 || command -v python3)}

if [ ! -d .venv-build ]; then
    "$PY" -m venv .venv-build
    .venv-build/bin/pip install --upgrade pip
fi
# PYGAME=pygame-ce builds with pygame-ce instead (same `import pygame`). pygame 2.6.1's cp314 wheel ships without its
# mixer, so on Python 3.14 the game would run silently: use pygame-ce there. The check below refuses such a build.
PYGAME=${PYGAME:-pygame}
if [ "$PYGAME" != pygame ]; then .venv-build/bin/pip uninstall -y pygame >/dev/null 2>&1 || true; fi
if [ "$PYGAME" != pygame-ce ]; then .venv-build/bin/pip uninstall -y pygame-ce >/dev/null 2>&1 || true; fi
.venv-build/bin/pip install numpy scipy "$PYGAME" pyinstaller pillow moderngl pyyaml
if ! .venv-build/bin/python -c "import pygame.mixer" 2>/dev/null; then
    echo "error: this pygame has no mixer, so the AppImage would have no sound. Rebuild with PYGAME=pygame-ce." >&2
    exit 1
fi

.venv-build/bin/python tools/make_icon.py build/icon.png

# pygame 2.6.1's cp314 wheel ships a font.py whose `from pygame.sysfont import ...` sits above the Font class it
# defines, so sysfont's `from pygame.font import Font` hits a half-built module and every font call dies with a
# misleading "circular import". Harmless where the wheel is already correct (the 3.11 CI build): the patch is a
# no-op unless that exact bad ordering is present. Without it the frozen app crashes the moment it draws text.
.venv-build/bin/python - .venv-build/lib/python*/site-packages/pygame/font.py <<'FONTFIX'
import glob, sys
for path in glob.glob(sys.argv[1]):
    src = open(path).read()
    imp = "from pygame.sysfont import match_font, get_fonts, SysFont as _SysFont\n"
    head, sep, tail = src.partition(imp)
    if not sep or "class Font(" not in tail:
        continue                      # correct wheel, or already patched
    open(path, "w").write(head + tail.rstrip("\n") + "\n\n" + imp)
    print(f"patched {path}: moved sysfont import below Font")
FONTFIX


EXTRA_DATA=(--add-data "data/kick_brain.npz:." --add-data "protocols:protocols")
if [ -f data/validation_results.json ]; then
    EXTRA_DATA+=(--add-data "data/validation_results.json:.")
else
    echo "note: data/validation_results.json not found; the build won't show real-science popups" >&2
fi

# glcontext.x11/egl are moderngl's Linux GL backends (glcontext.wgl is the Windows-only one build_exe.ps1 uses).
.venv-build/bin/pyi-makespec --onefile --name KickTheFly \
    "${EXTRA_DATA[@]}" \
    --exclude-module kickthefly.sim.connectome.loader --exclude-module pyarrow \
    --exclude-module tkinter --exclude-module matplotlib --exclude-module pynwb --exclude-module h5py \
    --exclude-module pandas \
    --hidden-import glcontext.x11 --hidden-import glcontext.egl --hidden-import glcontext.empty \
    --hidden-import yaml --collect-submodules kickthefly --collect-data kickthefly \
    kick_the_fly.py

# Never bundle the libraries the host's graphics driver loads next to us. Mesa (radeonsi, iris, llvmpipe...) is loaded
# into this process and needs the host's own C++ runtime and X11 client libraries; the copies from the old-glibc build
# machine are too old for a current distro's driver, OpenGL then fails to load and the game drops to 2D (2.6.0 and
# 2.7.0 did this on Arch). Every desktop Linux ships all of these (Wayland desktops too, through XWayland), and newer
# versions are backward compatible, so the host's copies always work for the bundled code. This is the same list the
# AppImage project's own excludelist keeps out.
HOST_LIBS="libstdc++.so.6 libgcc_s.so.1 libX11.so.6 libX11-xcb.so.1 libxcb.so.1 libXau.so.6 libXdmcp.so.6 libGL.so.1 libEGL.so.1 libGLX.so.0 libdrm.so.2 libgbm.so.1"
export HOST_LIBS
python3 - KickTheFly.spec <<'SPEC'
import sys
path = sys.argv[1]
s = open(path).read()
import os
host = tuple(os.environ["HOST_LIBS"].split())
drop = f"a.binaries = [b for b in a.binaries if b[0].split('/')[-1] not in {host!r}]\n"
s = s.replace("pyz = PYZ(", drop + "pyz = PYZ(", 1)
open(path, "w").write(s)
SPEC
.venv-build/bin/pyinstaller --noconfirm --clean KickTheFly.spec
bundled=$(.venv-build/bin/pyi-archive_viewer -l -r dist/KickTheFly 2>/dev/null)
for lib in $HOST_LIBS; do
    if grep -qE "['/]${lib//+/\\+}'" <<<"$bundled"; then
        echo "error: $lib got bundled; it must come from the host or OpenGL breaks on newer distros" >&2
        exit 1
    fi
done

# build check without a display: load the brain, run the smoke protocol, exit 0
rm -rf build/smoke
dist/KickTheFly --headless --protocol protocols/smoke.yaml --out build/smoke
ls build/smoke/*/summary.json >/dev/null

# --- assemble the AppDir ---
APPDIR=build/KickTheFly.AppDir
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/share/applications" "$APPDIR/usr/share/metainfo" \
         "$APPDIR/usr/share/icons/hicolor/256x256/apps"
cp dist/KickTheFly "$APPDIR/usr/bin/KickTheFly"
cp build/icon.png "$APPDIR/$APP_ID.png"
cp build/icon.png "$APPDIR/usr/share/icons/hicolor/256x256/apps/$APP_ID.png"
cp "packaging/linux/$APP_ID.desktop" "$APPDIR/$APP_ID.desktop"
cp "packaging/linux/$APP_ID.desktop" "$APPDIR/usr/share/applications/$APP_ID.desktop"
cp "packaging/linux/$APP_ID.appdata.xml" "$APPDIR/usr/share/metainfo/$APP_ID.appdata.xml"

cat > "$APPDIR/AppRun" <<'APPRUN'
#!/bin/sh
HERE="$(dirname "$(readlink -f "$0")")"
exec "$HERE/usr/bin/KickTheFly" "$@"
APPRUN
chmod +x "$APPDIR/AppRun"

APPIMAGETOOL=build/appimagetool-x86_64.AppImage
if [ ! -f "$APPIMAGETOOL" ]; then
    curl -fL -o "$APPIMAGETOOL" https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage
    chmod +x "$APPIMAGETOOL"
fi

mkdir -p dist
rm -f dist/KickTheFly-x86_64.AppImage
# APPIMAGE_EXTRACT_AND_RUN: appimagetool is itself an AppImage; this runs it without FUSE (containers, CI)
APPIMAGE_EXTRACT_AND_RUN=1 ARCH=x86_64 VERSION="$VERSION" "$APPIMAGETOOL" "$APPDIR" dist/KickTheFly-x86_64.AppImage
ls -lh dist/KickTheFly-x86_64.AppImage
