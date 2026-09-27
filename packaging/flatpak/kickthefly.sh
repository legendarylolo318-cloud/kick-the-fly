#!/bin/sh
# Flatpak launcher. Inside the sandbox XDG_CONFIG_HOME and friends point at ~/.var/app/<id>; point them back at the
# host's folders (the manifest grants exactly the kickthefly ones) so the game shares settings, training memory, saves
# and crash reports with the AppImage and source installs. Flatpak passes the host's own values on as HOST_XDG_*.
export XDG_CONFIG_HOME="${HOST_XDG_CONFIG_HOME:-$HOME/.config}"
export XDG_DATA_HOME="${HOST_XDG_DATA_HOME:-$HOME/.local/share}"
export XDG_STATE_HOME="${HOST_XDG_STATE_HOME:-$HOME/.local/state}"
exec python3 /app/share/kickthefly/kick_the_fly.py "$@"
