#!/usr/bin/env bash
# Builds dist/KickTheFly-x86_64.AppImage from the Rust release build: the game, the brain pack and the desktop files in one file.
# Needs data/kick_brain.npz (the pack is built from the MaleCNS download with the Python dev tool in legacy-python/, see README.md).
# Build on an old glibc (CI uses ubuntu-22.04) so the AppImage runs on most distros. The game needs a Vulkan driver at run time
# (vulkan-icd-loader plus the GPU's driver); like every AppImage it uses the host's libasound, libudev, libwayland and libxkbcommon.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
APP_ID=io.github.legendarylolo318_cloud.KickTheFly
VERSION=$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)

if [ ! -f data/kick_brain.npz ]; then
    echo "data/kick_brain.npz missing: build it first (README.md, 'Build from source')" >&2
    exit 1
fi

cargo build --release -p kick_the_fly

APPDIR=build/AppDir
rm -rf "$APPDIR" dist/KickTheFly-x86_64.AppImage
mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/share/kick-the-fly" "$APPDIR/usr/share/applications" "$APPDIR/usr/share/icons/hicolor/256x256/apps" "$APPDIR/usr/share/metainfo" dist
install -m755 target/release/kick_the_fly "$APPDIR/usr/bin/kick_the_fly"
install -m644 data/kick_brain.npz "$APPDIR/usr/share/kick-the-fly/kick_brain.npz"
install -m644 packaging/linux/$APP_ID.desktop "$APPDIR/usr/share/applications/$APP_ID.desktop"
install -m644 packaging/linux/$APP_ID.png "$APPDIR/usr/share/icons/hicolor/256x256/apps/$APP_ID.png"
install -m644 packaging/linux/$APP_ID.appdata.xml "$APPDIR/usr/share/metainfo/$APP_ID.appdata.xml"
cp "$APPDIR/usr/share/applications/$APP_ID.desktop" "$APPDIR/$APP_ID.desktop"
cp "$APPDIR/usr/share/icons/hicolor/256x256/apps/$APP_ID.png" "$APPDIR/$APP_ID.png"
ln -s "usr/bin/kick_the_fly" "$APPDIR/AppRun.bin" 2>/dev/null || true
cat > "$APPDIR/AppRun" <<'RUN'
#!/bin/sh
HERE="$(dirname "$(readlink -f "$0")")"
export KICK_THE_FLY_PACK="${KICK_THE_FLY_PACK:-$HERE/usr/share/kick-the-fly/kick_brain.npz}"
exec "$HERE/usr/bin/kick_the_fly" "$@"
RUN
chmod +x "$APPDIR/AppRun"
rm -f "$APPDIR/AppRun.bin"
desktop-file-validate "$APPDIR/$APP_ID.desktop" || true

TOOL=build/appimagetool
if [ ! -x "$TOOL" ]; then
    mkdir -p build
    curl -fsSL -o "$TOOL" https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage
    chmod +x "$TOOL"
fi
ARCH=x86_64 VERSION="$VERSION" "$TOOL" --appimage-extract-and-run "$APPDIR" dist/KickTheFly-x86_64.AppImage
echo "built dist/KickTheFly-x86_64.AppImage ($(du -h dist/KickTheFly-x86_64.AppImage | cut -f1))"
