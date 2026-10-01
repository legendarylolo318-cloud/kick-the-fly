"""Settings: the schema behind the Settings menu, stored as config.toml in the per-OS config folder (paths.py).

Every setting has a tab, a default, a type with its allowed values, a plain-language tooltip and, for Brain settings,
a tag saying whether it changes the connectome simulation ("Connectome") or a rule the game adds on top ("Game rule").
A missing, unreadable or corrupt config.toml falls back to defaults with a warning (the bad file is kept as
config.toml.bad), never a crash. Unknown keys are ignored and out-of-range values are clamped, so an older or newer
config still loads.
"""
from __future__ import annotations

import math
import os
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from kickthefly.core.crash import log

TABS = ("Graphics", "Audio", "Brain", "Controls", "Accessibility", "Help")
CONNECTOME, GAME_RULE = "Connectome", "Game rule"


@dataclass(frozen=True)
class Setting:
    key: str                     # "section.name" in config.toml
    tab: str
    label: str
    kind: str                    # bool | choice | float | int
    default: object
    tip: str
    options: tuple = ()          # choice: allowed values
    labels: tuple = ()           # choice: display names
    lo: float = 0.0
    hi: float = 1.0
    step: float = 0.0
    tag: str = ""                # Connectome | Game rule (Brain tab)
    restart: bool = False        # applies on the next launch
    only: str = ""               # "linux" | "3d": hidden or disabled elsewhere
    fmt: str = "{:g}"


S = Setting
SETTINGS: tuple[Setting, ...] = (
    # --- Graphics
    S("graphics.fullscreen", "Graphics", "Fullscreen", "bool", False,
      "Fill the whole screen. F11 or Alt+Enter also toggles it."),
    S("graphics.resolution_scale", "Graphics", "Resolution scale", "float", 1.0,
      "Draw the 3D room at a fraction of the window's resolution and stretch it. Lower is faster on weak GPUs. "
      "Menus and text stay sharp.", lo=0.5, hi=1.0, step=0.05, only="3d", fmt="{:.0%}"),
    S("graphics.fps_cap", "Graphics", "FPS cap", "choice", 60,
      "The most frames per second the game draws. The fly's brain always runs at its own real-time pace.",
      options=(30, 60, 90, 120, 144, 165, 240, 0), labels=("30", "60", "90", "120", "144", "165", "240", "unlimited")),
    S("graphics.vsync", "Graphics", "VSync", "bool", False,
      "Sync frames to the monitor to stop tearing. Applies on restart.", restart=True, only="3d"),
    S("graphics.backend", "Graphics", "Display backend", "choice", "auto",
      "Linux only. Auto uses native Wayland when available and falls back to X11 (XWayland). Try X11 if the window "
      "or mouse misbehaves. Applies on restart.", options=("auto", "wayland", "x11"), labels=("Auto", "Wayland", "X11"),
      restart=True, only="linux"),
    S("graphics.panel_mode", "Graphics", "Brain panel", "choice", "solid",
      "How the live brain panel on the right is drawn: solid, see-through, faint, or hidden to give the room the "
      "whole screen. Hotkey V.", options=("solid", "see-through", "faint", "hidden"),
      labels=("Solid", "See-through", "Faint", "Hidden"), only="3d"),
    S("graphics.menu_size", "Graphics", "Menu size", "choice", "crisp",
      "Crisp scales menus by whole pixels so text stays sharp. Large keeps them the same size on every screen. "
      "Hotkey U.", options=("crisp", "large"), labels=("Crisp", "Large"), only="3d"),
    S("graphics.ui_scale", "Graphics", "UI scale", "float", 1.0,
      "Makes every menu, meter and label bigger or smaller.", lo=0.75, hi=1.5, step=0.05, only="3d", fmt="{:.0%}"),
    S("graphics.photo_scale", "Graphics", "Screenshot resolution", "choice", 2,
      "Render scale multiplier for screenshots (1x, 2x, 4x) saved to pictures.", options=(1, 2, 4),
      labels=("1x", "2x (Hi-Res)", "4x (Ultra)")),
    S("graphics.clean_capture", "Graphics", "Clean screenshots", "bool", True,
      "Hide HUD, crosshairs, toolbars, and debug overlays in saved screenshots."),
    S("graphics.photo_dof", "Graphics", "Depth of field blur", "float", 0.0,
      "Camera depth of field blur amount in photo mode.", lo=0.0, hi=1.0, step=0.05, only="3d", fmt="{:.2f}"),
    S("graphics.photo_focus", "Graphics", "Photo focus distance", "float", 1.8,
      "Focus distance in meters for depth of field.", lo=0.1, hi=15.0, step=0.1, only="3d", fmt="{:.1f}m"),
    S("graphics.timelapse_speedup", "Graphics", "Time-lapse speed-up", "choice", 5,
      "Playback speed-up factor for time-lapse recordings (2x, 5x, 10x, 20x). Hotkey L.",
      options=(2, 5, 10, 20), labels=("2x", "5x", "10x", "20x"), tag=GAME_RULE),
    S("graphics.timelapse_format", "Graphics", "Time-lapse format", "choice", "mp4",
      "File format for exported time-lapse recordings (MP4 with ffmpeg, or GIF with Pillow).",
      options=("mp4", "gif"), labels=("MP4 (Video)", "GIF (Animated)")),
    S("graphics.timelapse_target", "Graphics", "Time-lapse target", "choice", "brain",
      "Camera subject for time-lapse recording: the big brain view or the full room.",
      options=("brain", "room"), labels=("Brain view", "Room view")),
    # --- Audio
    S("audio.master", "Audio", "Master volume", "float", 1.0, "Volume of everything.", lo=0, hi=1, step=0.05, fmt="{:.0%}"),
    S("audio.buzz", "Audio", "Wing buzz volume", "float", 1.0,
      "The buzz while it flies. Its pitch follows the fly's wing-power neurons.", lo=0, hi=1, step=0.05, fmt="{:.0%}"),
    S("audio.sfx", "Audio", "Sound effects volume", "float", 1.0,
      "Hits, tools, splashes, menus and every other sound.", lo=0, hi=1, step=0.05, fmt="{:.0%}"),
    S("audio.mute", "Audio", "Mute", "bool", False, "Silence everything. Hotkey M."),
    S("audio.stethoscope_enabled", "Audio", "Brain stethoscope", "bool", False,
      "Spike sonification: play subtle audio clicks for spikes in the probed region or group. Hotkey K.", tag=GAME_RULE),
    S("audio.stethoscope_vol", "Audio", "Stethoscope volume", "float", 0.5,
      "Volume of the brain stethoscope spike sonification. Synthetic sonification, not an extracellular LFP recording.",
      lo=0.0, hi=1.0, step=0.05, fmt="{:.0%}", tag=GAME_RULE),
    S("audio.stethoscope_target", "Audio", "Stethoscope target", "choice", "mushroom_body",
      "Neuropil region or neuron group to monitor with the stethoscope probe.",
      options=("mushroom_body", "antennal_lobe", "central_complex", "optic_lobes", "motor", "whole_brain"),
      labels=("Mushroom body", "Antennal lobe", "Central complex", "Optic lobes", "Motor neurons", "Whole brain"),
      tag=GAME_RULE),
    # --- Brain
    S("brain.mode", "Brain", "Mode", "choice", "play",
      "Play is the game with challenges and scores. Lab adds the research tools: validation results, repeated "
      "trials with statistics, data export and protocol files. Pet mode provides a persistent companion fly.",
      options=("play", "lab", "pet"), labels=("Play", "Lab", "Pet")),
    S("brain.brain", "Brain", "Brain", "choice", "adult",
      "Which connectome to simulate: adult Drosophila melanogaster (FlyWire) or Drosophila larva (Winding et al. 2023).",
      options=("adult", "larva"), labels=("Adult (FlyWire)", "Larva (Winding 2023)"), tag=CONNECTOME),
    S("brain.individuality", "Brain", "Individuality", "choice", "subtle",
      "Deterministic per-fly gain variation across neurons. Off gives identical brains across flies; subtle adds natural variation; strong accentuates behavioural differences.",
      options=("off", "subtle", "strong"), labels=("Off", "Subtle", "Strong"), tag=GAME_RULE),
    S("brain.pet_real_stakes", "Brain", "Pet real stakes", "bool", False,
      "In Pet mode, lets injury or prolonged starvation kill the fly, triggering an autopsy.", tag=GAME_RULE),
    S("brain.pain_level", "Brain", "Pain neurons", "choice", 0,
      "How many of the fly's real sensory neurons the pain meter listens to, and how many each hit fires. Normal: "
      "touch, heat, cold, smell and taste. More: plus the rest of the body's sensors. Max: plus the neurons that relay "
      "body signals to the brain. The pain score itself is an estimate, not a measurement. Hotkey P.",
      options=(0, 1, 2), labels=("Normal", "More", "Max"), tag=GAME_RULE),
    S("brain.immortal", "Brain", "Immortal", "bool", False,
      "It feels everything but can't die, and heals when you stop. Hotkey I.", tag=GAME_RULE),
    S("brain.sim_speed", "Brain", "Sim speed", "choice", 1.0,
      "Slow motion for the whole simulation: the brain and the room slow down together, so every spike still lines "
      "up with what happens. Hotkeys [ and ].", options=(0.1, 0.25, 0.5, 1.0),
      labels=("0.1x", "0.25x", "0.5x", "1x"), tag=CONNECTOME),
    S("brain.seed", "Brain", "Random seed", "int", 0,
      "Seeds the brain's noise and the game's randomness. The same seed and the same inputs give the same run in "
      "headless and protocol runs. Applies when you reset the fly (R).", lo=0, hi=2**31 - 1, tag=CONNECTOME),
    S("brain.real_vs_rule", "Brain", "Real vs rule tags", "choice", "auto",
      "Tags reactions on screen as coming from the connectome (REAL) or from a game rule (RULE). Auto: on in Lab, off "
      "in Play.", options=("auto", "on", "off"), labels=("Auto", "On", "Off")),
    S("brain.science_popups", "Brain", "Real-science popups", "bool", False,
      "Show a short card when the fly does something real flies were shown to do."),
    S("brain.neurodex", "Brain", "Neurodex discoveries", "bool", True,
      "Collect cell types in the Neurodex (default key D): a type is discovered the first time its neurons fire well "
      "above their own calm rate while you play or stimulate it (tagged 'in play' or 'by stimulation'; a calm, untouched fly discovers nothing, and 'at rest' entries from older builds are kept). The numbers in an entry come from the dataset (Connectome); what counts "
      "as discovered, and the collection, are game rules. Off stops collecting; what you found stays.",
      tag=GAME_RULE),
    S("brain.killcam", "Brain", "Kill cam offer", "bool", True,
      "When the fly dies, offer a slow-motion replay of the last 6 seconds of its brain and highlight the neurons whose "
      "firing rose most. The firing is the real simulated rate of each neuron (Connectome); the offer, the window and the "
      "slow motion are game rules. Skippable; can be saved as a video or GIF.", tag=GAME_RULE),
    S("brain.neuron_of_day", "Brain", "Neuron of the day", "bool", True,
      "A small card at launch with one curated cell type, a fact with its citation (from the literature, not measured "
      "here) and a Try it button that sets up a one-click experiment. Separate from the real-science popups. Which type, "
      "the card and Try it are game rules.", tag=GAME_RULE),
    S("brain.imaging_indicator", "Brain", "Imaging indicator", "choice", "gcamp6s",
      "Lab > Calcium imaging: which GCaMP the simulated imaging uses. MODEL: spikes from the simulation convolved with the "
      "indicator's kernel plus photon shot noise, not a measurement. The rise and decay come from Chen 2013 (GCaMP6s/6f, its "
      "Supplementary Table 3: mouse cortex, single spikes) and Zhang 2023 (jGCaMP8m, fly visual responses). Off until you turn "
      "Imaging mode on.", options=("gcamp6s", "gcamp6f", "jgcamp8m"), labels=("GCaMP6s", "GCaMP6f", "GCaMP8m"), tag="MODEL"),
    S("brain.imaging_fps", "Brain", "Imaging frame rate", "choice", 20,
      "Lab > Calcium imaging: frames per second of the simulated imaging (the 5 ms simulation step quantizes it). MODEL, not a "
      "measurement.", options=(5, 10, 20, 30, 40), labels=("5 Hz", "10 Hz", "20 Hz", "30 Hz", "40 Hz"), tag="MODEL"),
    S("brain.imaging_f0_tau_s", "Brain", "Imaging baseline (F0) time constant", "choice", 30,
      "Lab > Calcium imaging: dF/F is measured against a running mean of each neuron's own fluorescence with this time constant. "
      "A steady firing rate reads as 0 and only changes show; the first seconds of a session are inflated or deflated while the "
      "mean settles. Shorter hides slow changes, longer takes longer to settle. GAME RULE (a choice made for this game, not a "
      "measurement); Off until you turn Imaging mode on.", options=(10, 30, 60, 120),
      labels=("10 s", "30 s (default)", "60 s", "120 s"), tag=GAME_RULE),
    S("brain.autopilot", "Brain", "Autopilot / spectator", "bool", False,
      "Hands-off mode where only environmental inputs reach the fly. Hotkey Y.", tag=GAME_RULE),
    S("brain.autopilot_orbit", "Brain", "Autopilot brain orbit", "bool", True,
      "Slowly orbit the brain view in spectator mode.", tag=GAME_RULE),
    S("brain.autopilot_hide_hud", "Brain", "Autopilot hide HUD", "bool", True,
      "Hide HUD in spectator mode for demo or screensaver use.", tag=GAME_RULE),
    S("brain.arena", "Brain", "Arena", "choice", "room",
      "Where the fly lives. Room is the default. Thermo has a floor running from cold (left) to hot (right) that "
      "drives its real cold- and hot-sensing antennal neurons. Open field and Orchard are large outdoor 3D worlds "
      "(the 2D game stays indoors). The sensory neurons each arena drives are real; the places themselves are game "
      "rules. Hotkey E cycles them.",
      options=("room", "fan", "flypaper", "pool", "lamp", "thermo", "escaperoom", "field", "orchard"),
      labels=("Room", "Fan", "Flypaper", "Pool", "Lamp", "Thermo", "Escape room", "Open field", "Orchard"),
      tag=GAME_RULE),
    S("brain.song_buzz", "Brain", "Courtship song buzz", "bool", True,
      "Plays a synthesized pulse-song buzz when its ps1 wing motor neurons fire (SONG). The neurons are real "
      "(pIP10 -> ps1 is validated); the sound is the game's.", tag=GAME_RULE),
    S("brain.lunge", "Brain", "Aggression lunges", "bool", True,
      "With two or more flies, a fly whose aggression neurons (AVLP727m, pC1/P1) fire strongly lunges at the "
      "nearest one. The firing is real; the lunge is scripted. They rarely fire that hard unless P1 is stimulated in "
      "brain surgery.",
      tag=GAME_RULE),
    S("brain.day_night", "Brain", "Day/night cycle", "choice", 0,
      "Outdoors, the sun circles once a day and night falls. Daylight drives the real photoreceptors and morning "
      "clock neurons (l-LNv, s-LNv); the cycle and that link are game rules. Lab > Parameters can set any day length.",
      options=(0, 300, 600, 1200), labels=("Off", "5 min day", "10 min day", "20 min day"), only="3d", tag=GAME_RULE),
    S("brain.mirror_weights", "Brain", "Mirror-average weights", "bool", False,
      "Average left and right synaptic weights to enforce bilateral symmetry. Clearly a data modification game rule.",
      tag=GAME_RULE),
    S("brain.dtype", "Brain", "State precision", "choice", "float32",
      "Precision of each neuron's membrane voltage. float32 (the default) is what validation used; float64 changes "
      "the numbers slightly; validation hasn't been run in it.", options=("float32", "float64"), labels=("float32 (Fast)", "float64 (Double)"),
      tag=CONNECTOME),
    S("brain.backend", "Brain", "Compute backend", "choice", "auto",
      "What runs the brain simulation. Auto picks a PyTorch GPU, then Numba, then NumPy. OpenGL compute (4.3+) "
      "runs only when you pick it: it works on any vendor's GPU but is slower than NumPy here and doesn't scale to "
      "many flies. Numba and PyTorch are optional (from source only); anything missing falls back to NumPy. "
      "Applies to every fly right away.",
      options=("auto", "cpu", "numba", "gl", "torch-cpu", "torch-cuda", "torch-rocm"),
      labels=("Auto", "CPU (NumPy)", "Numba (JIT)", "OpenGL Compute", "PyTorch (CPU)", "PyTorch (CUDA)",
              "PyTorch (ROCm)"),
      tag=CONNECTOME),
    # --- Controls
    S("controls.mouse_sensitivity", "Controls", "Mouse sensitivity", "float", 1.0,
      "How far the view turns when you move the mouse.", lo=0.1, hi=5.0, step=0.1, only="3d", fmt="{:.1f}"),
    S("controls.invert_y", "Controls", "Invert Y", "bool", False, "Moving the mouse up looks down.", only="3d"),
    S("controls.fov", "Controls", "Field of view", "float", 70.0,
      "How wide your view of the room is, in degrees.", lo=50, hi=110, step=1, only="3d", fmt="{:.0f}°"),
    S("controls.loadout_preset", "Controls", "Tool loadout", "choice", "auto",
      "Which tools are on the hotbar (keys 1-9, 0). Automatic is Base in Play, Lab in Lab mode and Pet in Pet mode. "
      "Hotbar order and your own loadouts are made in the editor (default key Q). The tool wheel still reaches every "
      "tool. A loadout is a convenience: it never changes what a tool does to the fly.",
      options=("auto", "base", "chaos", "chemist", "lab", "all", "pet", "custom"),
      labels=("Automatic", "Base", "Chaos", "Chemist", "Lab", "All", "Pet", "Custom")),
    S("controls.gamepad", "Controls", "Gamepad", "bool", True,
      "Use a connected gamepad in the 3D game alongside keyboard and mouse: left stick walks, right stick looks, "
      "right trigger uses the tool, bumpers cycle your loadout, hold Y for the tool wheel. Rebind below.", only="3d"),
    S("controls.pad_look_speed", "Controls", "Gamepad look speed", "float", 2.5,
      "How fast the right stick turns your view, in radians per second at full tilt.", lo=0.5, hi=6.0, step=0.1,
      only="3d", fmt="{:.1f}"),
    S("controls.pad_deadzone", "Controls", "Gamepad dead zone", "float", 0.2,
      "How far a stick must move before it counts, so a worn stick doesn't drift.", lo=0.05, hi=0.5, step=0.01,
      only="3d", fmt="{:.2f}"),
    S("controls.pad_invert_y", "Controls", "Gamepad invert Y", "bool", False, "Pushing the right stick up looks down.",
      only="3d"),
    # --- Accessibility
    S("access.palette", "Accessibility", "Brain view colors", "choice", "default",
      "Colors for firing neurons in the brain view. Blue/yellow is safe for red-green color blindness. High contrast "
      "uses magenta and white.", options=("default", "blue-yellow", "high-contrast"),
      labels=("Orange/cyan", "Blue/yellow", "High contrast")),
    S("access.reduced_flashing", "Accessibility", "Reduced flashing", "bool", False,
      "Turns off screen shake, explosion and hit flashes, sparkles, the scanning band and blinking lights."),
    S("access.larger_text", "Accessibility", "Larger text", "bool", False,
      "Bigger text in menus, popups and the HUD."),
    S("access.language", "Accessibility", "Language", "choice", "auto",
      "Display language for menus and text. Auto uses your system language, falling back to English.",
      options=("auto", "en", "de"),
      labels=("Auto", "English", "Deutsch (Machine-translated)")),
)
BY_KEY = {s.key: s for s in SETTINGS}

# rebindable actions: (action, label, default key name as pygame.key.name() spells it)
ACTIONS: tuple[tuple[str, str, str], ...] = (
    ("forward", "Walk forward", "w"), ("back", "Walk back", "s"), ("left", "Walk left", "a"), ("right", "Walk right", "d"),
    ("sprint", "Sprint", "left shift"), ("crouch", "Crouch", "left ctrl"),
    ("free_mouse", "Free the mouse", "tab"), ("big_view", "Big brain view", "b"), ("surgery", "Brain surgery", "o"),
    ("training", "Training", "t"), ("duel", "1v1 duel", "x"), ("arena", "Next arena", "e"),
    ("pain", "Pain neurons", "p"), ("immortal", "Immortal", "i"), ("mute", "Mute", "m"),
    ("screenshot", "Screenshot", "f12"), ("gif", "Save GIF", "g"), ("panel", "Brain panel style", "v"),
    ("menu_size", "Menu size", "u"), ("fullscreen", "Fullscreen", "f11"), ("spawn", "Spawn a fly", "n"),
    ("reset", "Reset / respawn", "r"), ("help", "Controls help", "h"),
    ("time_pause", "Pause / resume time", "z"), ("time_slower", "Slower", "["), ("time_faster", "Faster", "]"),
    ("time_step", "Single step (paused)", "."),
    ("autopilot", "Autopilot / spectator", "y"),
    ("photo_mode", "Photo mode / free camera", "f10"),
    ("stethoscope", "Brain stethoscope", "k"),
    ("timelapse", "Time-lapse record", "l"),
    ("recall", "Recall a lost fly (outdoors)", "j"),
    ("cycle_fly", "Cycle focused fly", "f"),
    ("loadout", "Loadout editor", "q"),
    ("neurodex", "Neurodex", "d"),
    ("killcam", "Kill cam (after the fly dies)", ";"),
    ("tool_wheel", "Tool wheel (hold)", "`"),
    *((f"slot{i + 1}", f"Hotbar slot {i + 1}", str((i + 1) % 10)) for i in range(10)),
    ("page_prev", "Hotbar previous page", "-"),
    ("page_next", "Hotbar next page", "="),
)
ACTION_LABEL = {a: label for a, label, _ in ACTIONS}
RESERVED_KEYS = {"escape"}                                # the pause menu can't be rebound (2.13: the tool keys can)
SLOT_ACTIONS = tuple(f"slot{i + 1}" for i in range(10))     # hotbar slots: keys 1-9, 0 by default
MOVEMENT_3D_ONLY = {"forward", "back", "left", "right", "sprint", "crouch", "free_mouse", "duel", "panel", "menu_size"}
# gamepad bindings (game/gamepad.py): action, label, default. A binding is one of SDL's standard controller names
# (the same on Xbox, PlayStation, Switch Pro...; button names follow the pad's printed labels), a raw "axisN" or
# "buttonN" for a pad SDL doesn't know, or "" (unbound). PAD_AXES and PAD_BUTTONS are in SDL's own enum order.
PAD_AXES = ("leftx", "lefty", "rightx", "righty", "lefttrigger", "righttrigger")
PAD_BUTTONS = ("a", "b", "x", "y", "back", "guide", "start", "leftstick", "rightstick", "leftshoulder", "rightshoulder",
               "dpup", "dpdown", "dpleft", "dpright")
PAD_ACTIONS: tuple[tuple[str, str, str], ...] = (
    ("move_x", "Walk left / right (stick)", "leftx"), ("move_y", "Walk forward / back (stick)", "lefty"),
    ("look_x", "Look left / right (stick)", "rightx"), ("look_y", "Look up / down (stick)", "righty"),
    ("use", "Use the tool", "righttrigger"), ("tool_next", "Next tool", "rightshoulder"),
    ("tool_prev", "Previous tool", "leftshoulder"), ("tool_wheel", "Tool wheel (hold)", "y"), ("loadout", "Loadout editor", "x"),
    ("sprint", "Sprint", "leftstick"), ("crouch", "Crouch / fly down", "b"), ("up", "Fly up (photo mode)", "a"),
    ("menu", "Menu (Esc)", "start"), ("big_view", "Big brain view", "back"),
    ("neurodex", "Neurodex", "dpup"), ("killcam", "Kill cam / skip it", "dpdown"),
)
PAD_LABEL = {a: label for a, label, _ in PAD_ACTIONS}


def _pad_binding(v) -> str | None:
    if not isinstance(v, str):
        return None
    v = v.strip().lower()
    if v in PAD_AXES or v in PAD_BUTTONS:
        return v
    for kind in ("axis", "button"):
        if v.startswith(kind) and v[len(kind):].isdigit() and int(v[len(kind):]) < 64:
            return v
    return "" if v == "" else None


def _coerce(s: Setting, v):
    """A valid value for this setting, or raise ValueError."""
    if s.kind == "bool":
        if isinstance(v, bool):
            return v
        raise ValueError(f"{s.key}: expected true/false")
    if s.kind == "choice":
        for o in s.options:
            if v == o or (isinstance(o, float) and isinstance(v, (int, float)) and not isinstance(v, bool)
                          and math.isclose(float(v), o)):
                return o
        raise ValueError(f"{s.key}: {v!r} is not one of {s.options}")
    if s.kind in ("float", "int"):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"{s.key}: expected a number")
        v = min(max(v, s.lo), s.hi)
        return int(round(v)) if s.kind == "int" else float(v)
    raise ValueError(s.kind)


# Pairs that may share a key on purpose (3.0): the Neurodex key is D, which is also walk right in the 3D game. The 3D game
# opens the Neurodex on D only while the mouse is free, so walking is never taken away. Rebind either and it is gone.
OVERLAP_OK = ({"right", "neurodex"},)


def _allowed_overlap(actions: list[str]) -> bool:
    return any(set(actions) <= pair for pair in OVERLAP_OK)


SCHEMA_VERSION = 3
# 2.13 (schema 3): [loadout] custom slots and up to five saved loadouts, and [first_run] flags. A config from before
# 2.13 (schema < 3) had players on keys 1-9, 0, - and =, which then reached every tool, so it migrates to the All
# preset (nobody's muscle memory breaks) with a one-time notice pointing at the loadout editor, and it counts as
# already onboarded (no first-launch tutorial; Settings > Help replays it). A fresh install gets Base and the tutorial.
FIRST_RUN_DEFAULTS = {"tutorial_done": False, "loadout_notice": False}


class Config:
    def __init__(self, path: Path | None = None):
        self.path = path
        self.values: dict[str, object] = {s.key: s.default for s in SETTINGS}
        self.keys: dict[str, str] = {a: k for a, _, k in ACTIONS}
        self.pad: dict[str, str] = {a: b for a, _, b in PAD_ACTIONS}
        self.warnings: list[str] = []
        self.dirty = False
        self.loadout: dict = {"custom": [], "saved": []}     # custom: tool names; saved: [{"name", "tools"}] (max 5)
        self.first_run: dict = dict(FIRST_RUN_DEFAULTS)
        self.migrated_from: int | None = None                # the schema version this file was migrated from

    # --- values -----------------------------------------------------------------------------------------------------
    def __getitem__(self, key: str):
        return self.values[key]

    def get(self, key: str, default=None):
        return self.values.get(key, default)

    def set(self, key: str, value) -> bool:
        """Validate and store. Returns True if the stored value changed."""
        v = _coerce(BY_KEY[key], value)
        if self.values.get(key) == v:
            return False
        self.values[key] = v
        self.dirty = True
        return True

    def reset_tab(self, tab: str) -> list[str]:
        changed = []
        for s in SETTINGS:
            if s.tab == tab and self.values[s.key] != s.default:
                self.values[s.key] = s.default
                changed.append(s.key)
        if tab == "Controls":
            default = {a: k for a, _, k in ACTIONS}
            if self.keys != default:
                self.keys = default
                changed.append("keys")
            pad = {a: b for a, _, b in PAD_ACTIONS}
            if self.pad != pad:
                self.pad = pad
                changed.append("keys")
        self.dirty = self.dirty or bool(changed)
        return changed

    @property
    def lab(self) -> bool:
        return self.values["brain.mode"] == "lab"

    @property
    def pet(self) -> bool:
        return self.values["brain.mode"] == "pet"

    @property
    def larva(self) -> bool:
        return self.values.get("brain.brain", "adult") == "larva"

    def tags_on(self) -> bool:
        v = self.values["brain.real_vs_rule"]
        return v == "on" or (v == "auto" and self.lab)

    # --- keys -------------------------------------------------------------------------------------------------------
    def action_for(self, key_name: str) -> str | None:
        if not key_name:                                 # pygame names some keys ""; "" is also "unbound"
            return None
        for a, k in self.keys.items():
            if k == key_name:
                return a
        return None

    def bind(self, action: str, key_name: str) -> tuple[bool, str]:
        """Rebind an action. A key another action already uses swaps the two bindings. Returns (ok, message)."""
        key_name = key_name.lower()
        if key_name in RESERVED_KEYS:
            return False, f"'{key_name}' is reserved (Esc opens the menu)"
        old = self.keys[action]
        # Every action on that key that may not share it with this one moves: the first to this action's old key (a swap,
        # as before), unless that would itself clash (3.0: D is shared by walk-right and the Neurodex, so the old key can
        # still be taken); anything left over is unbound and says so. Before this, only the first action on the key was
        # moved, which left two actions on D (review, Day 1).
        displaced = [a for a in self.actions_for(key_name) if a != action and not _allowed_overlap([a, action])]
        self.keys[action] = key_name
        self.dirty = True
        notes = []
        for a in displaced:
            holders = [b for b in self.actions_for(old) if b != a] if old else []
            if old and all(_allowed_overlap([a, b]) for b in holders):
                self.keys[a] = old
                notes.append(f"'{key_name}' was used by {ACTION_LABEL[a]}; swapped, that is now '{old}'")
            else:
                self.keys[a] = ""
                notes.append(f"'{key_name}' was used by {ACTION_LABEL[a]}; that is now unbound (Settings > Controls)")
        return True, "; ".join(notes)

    def bind_pad(self, action: str, binding: str) -> tuple[bool, str]:
        """Rebind a gamepad action. A binding another action uses swaps the two. Returns (ok, message)."""
        b = _pad_binding(binding)
        if b is None:
            return False, f"'{binding}' isn't a gamepad button or axis"
        other = next((a for a, x in self.pad.items() if x == b and a != action and b), None)
        old = self.pad[action]
        self.pad[action] = b
        self.dirty = True
        if other:
            self.pad[other] = old
            return True, f"{b} was used by {other}; swapped, that is now {old or 'unbound'}"
        return True, ""

    def actions_for(self, key_name: str) -> list[str]:
        """Every action bound to a key (action_for returns only the first). D is walk-right and the Neurodex key at once
        by design (3.0): in the 3D game it opens the Neurodex only when the mouse is free (Tab)."""
        if not key_name:
            return []
        return [a for a, k in self.keys.items() if k == key_name]

    def conflicts(self) -> dict[str, list[str]]:
        seen: dict[str, list[str]] = {}
        for a, k in self.keys.items():
            if k:                                            # "" is unbound, and any number of actions can be
                seen.setdefault(k, []).append(a)
        return {k: v for k, v in seen.items() if len(v) > 1 and not _allowed_overlap(v)}

    # --- files ------------------------------------------------------------------------------------------------------
    @classmethod
    def load(cls, path: Path) -> "Config":
        cfg = cls(path)
        if not path.exists():
            return cfg
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as e:
            cfg.warnings.append(f"config.toml could not be read ({e}); using default settings")
            try:
                path.replace(path.with_name(path.name + ".bad"))
                cfg.warnings.append(f"the unreadable file was kept as {path.name}.bad")
            except OSError:
                pass
            for w in cfg.warnings:
                log.warning(w)
            return cfg
        for s in SETTINGS:
            section, name = s.key.split(".")
            sec = data.get(section)
            if isinstance(sec, dict) and name in sec:
                try:
                    cfg.values[s.key] = _coerce(s, sec[name])
                except ValueError as e:
                    cfg.warnings.append(f"{e}; using the default")
        pad = data.get("gamepad")
        if isinstance(pad, dict):
            for a, _, _ in PAD_ACTIONS:
                b = _pad_binding(pad.get(a))
                if b is not None:
                    cfg.pad[a] = b
            # An action newer than this file (2.13: loadout on X) whose default is a button the player already gave to
            # another action: the player's binding wins, the new action starts unbound (Settings > Controls binds it).
            mine = {cfg.pad[a] for a, _, _ in PAD_ACTIONS if a in pad and cfg.pad[a]}
            for a, _, _ in PAD_ACTIONS:
                if a not in pad and cfg.pad[a] in mine:
                    cfg.warnings.append(f"gamepad '{cfg.pad[a]}' is yours for another action; {a} is unbound")
                    cfg.pad[a] = ""
        keys = data.get("keys")
        if isinstance(keys, dict):
            for a, _, _ in ACTIONS:
                k = keys.get(a)
                if isinstance(k, str) and k.lower() not in RESERVED_KEYS:      # "" = unbound
                    cfg.keys[a] = k.lower()
            for k, acts in cfg.conflicts().items():
                saved = [a for a in acts if a in keys]
                if saved and len(saved) < len(acts):
                    # A default of an action newer than this file (2.13: the hotbar, the editor on Q, the wheel on `)
                    # on a key the player chose for something else: the player's binding wins, the new one is unbound.
                    for a in acts:
                        if a not in keys:
                            cfg.keys[a] = ""
                            cfg.warnings.append(f"key '{k}' is yours for {', '.join(saved)}; "
                                                f"{ACTION_LABEL[a]} is unbound (Settings > Controls)")
            for k, acts in cfg.conflicts().items():          # a hand-edited file bound one key twice: keep the first
                for a in acts[1:]:
                    default = next(d for x, _, d in ACTIONS if x == a)
                    # review (3.0): the default can be the contested key itself (the Neurodex's is D), or taken by another
                    # action; then the action is unbound instead of staying doubled up
                    taken = [b for b in cfg.actions_for(default) if b != a and not _allowed_overlap([a, b])]
                    cfg.keys[a] = "" if (default == k or taken) else default
                    cfg.warnings.append(f"key '{k}' was bound twice; {a} "
                                        + ("is unbound (Settings > Controls)" if not cfg.keys[a] else "reset to its default"))
        schema = data.get("schema_version", 1)
        if not isinstance(schema, int) or isinstance(schema, bool):
            schema = 1
        cfg._load_loadout(data, schema)
        if schema < 2:
            if cfg.values.get("brain.science_popups") is True:
                cfg.values["brain.science_popups"] = False
                cfg.dirty = True
                cfg.warnings.append("migrated brain.science_popups to off for schema 2")
        for w in cfg.warnings:
            log.warning(w)
        return cfg

    def _load_loadout(self, data: dict, schema: int) -> None:
        from kickthefly.core import loadout as lo

        def names(v):
            return [n for n in v if isinstance(n, str) and n in lo.BY_NAME] if isinstance(v, list) else []

        sec = data.get("loadout")
        if isinstance(sec, dict):
            self.loadout["custom"] = names(sec.get("custom"))
            for item in (sec.get("saved") if isinstance(sec.get("saved"), list) else [])[:lo.MAX_SAVED]:
                name = item.get("name") if isinstance(item, dict) else None
                name = " ".join("".join(ch if ch.isprintable() else " " for ch in name).split()) if isinstance(name, str) else ""
                if name:                                     # one line of printable text, as the editor's text box allows
                    self.loadout["saved"].append({"name": name[:24], "tools": names(item.get("tools"))})
        fr = data.get("first_run")
        if isinstance(fr, dict):
            for k in FIRST_RUN_DEFAULTS:
                if isinstance(fr.get(k), bool):
                    self.first_run[k] = fr[k]
        if schema < 3:
            self.migrated_from = schema
            if self.values["controls.loadout_preset"] == "auto" and "loadout_preset" not in (data.get("controls") or {}):
                self.values["controls.loadout_preset"] = "all"
            self.first_run["loadout_notice"] = True         # shown once, then cleared (Game.show_loadout_notice)
            self.first_run["tutorial_done"] = True
            self.dirty = True
            self.warnings.append("migrated to schema 3: hotbar loadout set to All so keys 1-9, 0, - and = still reach "
                                 "every tool (Settings > Controls > Tool loadout, or the editor on Q)")

    def custom_loadout(self, tools) -> None:
        self.loadout["custom"] = list(tools)
        self.dirty = True

    def save_loadout(self, name: str, tools) -> bool:
        """Store a named loadout (replacing one of the same name). False if all five slots are taken."""
        from kickthefly.core import loadout as lo

        name = name.strip()[:24] or "Loadout"
        saved = self.loadout["saved"]
        for item in saved:
            if item["name"] == name:
                item["tools"] = list(tools)
                self.dirty = True
                return True
        if len(saved) >= lo.MAX_SAVED:
            return False
        saved.append({"name": name, "tools": list(tools)})
        self.dirty = True
        return True

    def delete_loadout(self, name: str) -> None:
        self.loadout["saved"] = [i for i in self.loadout["saved"] if i["name"] != name]
        self.dirty = True

    def to_toml(self) -> str:
        from kickthefly.core.version import __version__

        out = [
            f"# Kick the Fly {__version__} settings. Edit in the game (Esc > Settings) or by hand.",
            f"schema_version = {SCHEMA_VERSION}",
            "",
        ]
        sections: dict[str, list[str]] = {}
        for s in SETTINGS:
            section, name = s.key.split(".")
            sections.setdefault(section, []).append(f"{name} = {_toml_value(self.values[s.key])}")
        for section, lines in sections.items():
            out += [f"[{section}]", *lines, ""]
        out.append("[keys]")
        out += [f"{a} = {_toml_value(k)}" for a, k in self.keys.items()]
        out.append("")
        out.append("[gamepad]")
        out += [f"{a} = {_toml_value(b)}" for a, b in self.pad.items()]
        out += ["", "[loadout]", f"custom = {_toml_value(self.loadout['custom'])}"]
        for item in self.loadout["saved"]:
            out += ["", "[[loadout.saved]]", f"name = {_toml_value(item['name'])}", f"tools = {_toml_value(item['tools'])}"]
        out += ["", "[first_run]"] + [f"{k} = {_toml_value(bool(self.first_run[k]))}" for k in FIRST_RUN_DEFAULTS]
        return "\n".join(out) + "\n"

    def save(self) -> bool:
        if self.path is None:
            return False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(self.to_toml(), encoding="utf-8")
            os.replace(tmp, self.path)
            self.dirty = False
            return True
        except OSError as e:
            log.warning("could not save settings to %s: %s", self.path, e)
            return False


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    if isinstance(v, float):
        return repr(v)
    s = str(v).replace("\\", "\\\\").replace('"', '\\"')
    # control characters are not allowed raw in a TOML string: escape them, or the next launch can't read the file
    s = "".join(f"\\u{ord(ch):04x}" if ord(ch) < 0x20 or ord(ch) == 0x7F else ch for ch in s)
    return f'"{s}"'


def visible(s: Setting, three_d: bool, platform: str | None = None) -> bool:
    """Linux-only settings are hidden elsewhere (the display backend on Windows)."""
    if s.only == "linux":
        return (platform or sys.platform).startswith("linux")
    return True


def enabled(s: Setting, three_d: bool) -> bool:
    return not (s.only == "3d" and not three_d)
