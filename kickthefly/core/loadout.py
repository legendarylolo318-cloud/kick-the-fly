"""Tool loadouts (2.12): which of the game's tools are on the hotbar, in which slots.

Pure data and logic, no pygame: the game, the loadout editor, the tool wheel, the Python API (`Fly.loadout`), the
playthrough bot and the tests all read this one module.

The catalog says, for every tool, its category, a one-line description, which real neurons it drives and whether that
is CONNECTOME (sensory neurons the connectome really has, driven the way the tool's physical stimulus would) or
GAME RULE (something the game adds around them). Nothing here changes what a tool does to the simulation: a loadout
only decides which tools you can reach with one key. `kickthefly/game/kick_the_fly.py:TOOLS` remains the list of tools
the game knows; `tests/test_loadout.py` keeps the two in step.

Rules (GAME RULE, all of them: UI conveniences, no neuron reads them):
  - the hotbar has 10 slots on keys 1-9 and 0; a loadout longer than 10 pages, and the previous/next page keys
    (default - and =) turn the page; with 10 tools or fewer they do nothing
  - the hand is in every loadout, in slot 1, and can't be removed
  - Lab-only tools (the laser) never appear outside Lab mode; larva mode hides tools with no larval sensory mapping
  - presets: base, chaos, chemist, lab, all, pet, and "custom" (the player's own). The preset "auto" picks per mode:
    Base in Play, Lab in Lab, Pet in Pet mode
"""
from __future__ import annotations

from dataclasses import dataclass

CONNECTOME, GAME_RULE = "CONNECTOME", "GAME RULE"
CATEGORIES = ("Touch", "Thermal", "Chemical", "Reward", "Creatures", "Lab")
PAGE_SIZE = 10
MAX_SAVED = 5
HAND = "hand"


@dataclass(frozen=True)
class ToolInfo:
    name: str
    label: str
    category: str
    desc: str                    # one line, for the editor and the wheel
    neurons: str                 # which real neurons it drives, in plain words
    tag: str                     # CONNECTOME | GAME RULE: whether what it drives is real or a rule
    lab_only: bool = False
    larva: bool = True           # has a larval sensory mapping (brain sense groups in the larval pack)
    larva_note: str = ""         # why it is hidden in larva mode when larva is False
    # The same claim as `neurons`, machine-readable, for the playthrough bot (lab/playthrough.py): a tuple of probes,
    # each a tuple of Brain.sense keys ((region, side), e.g. ("heat", None)) whose neurons are pooled. Every probe with
    # neurons in the brain under test must fire above its calm baseline when the tool is used. An empty tuple means
    # the tool is checked another way (the laser aims at cell types you pick).
    probes: tuple = ()


TOUCH_KEYS = (("head", None), ("body", None), ("legs", None), ("wing", None))
LOOM = (("loom", None),)
TASTE, REWARD = (("taste", None),), (("reward", None),)

# In the editor's category order (tests/test_loadout.py keeps the set of tools in step with the game's).
CATALOG: tuple[ToolInfo, ...] = (
    ToolInfo("hand", "HAND", "Touch", "Grab the fly and throw it.",
             "Touch neurons where you grab it: head BM_* + JO-*, body SNta*, legs SNpp*, wings WG*.", CONNECTOME,
             probes=(TOUCH_KEYS,)),
    ToolInfo("flick", "FLICK", "Touch", "Flick it away with a fingertip.",
             "Touch neurons near the flick (head, body, legs, wings). The flight that follows is game physics.",
             CONNECTOME,
             probes=(TOUCH_KEYS,)),
    ToolInfo("swatter", "SWATTER", "Touch", "Swat it. Hard, and it remembers.",
             "Touch neurons of the body parts hit (a strong hit fires more of them), plus looming detectors "
             "LPLC2 and LC4 as it swings in.", CONNECTOME,
             probes=(TOUCH_KEYS, LOOM)),
    ToolInfo("bomb", "BOMB", "Touch", "Drop a bomb with a 1.5 s fuse.",
             "Looming detectors LPLC2 and LC4 as it falls; touch neurons from the blast.", CONNECTOME,
             larva=False, larva_note="the larva has no looming detectors in this model",
             probes=(LOOM,)),
    ToolInfo("zapper", "ZAPPER", "Touch", "An electric shock through its body.",
             "Every touch neuron (real) plus current into a random 30% of all neurons for 40 ms (game rule).",
             GAME_RULE,
             probes=(TOUCH_KEYS,)),
    ToolInfo("torch", "BLOWTORCH", "Thermal", "Hold to burn it; maxes out pain.",
             "Every touch neuron and the heat cells TRN_VP2 at full strength; pain is a game estimate.", CONNECTOME,
             probes=((("heat", None),), TOUCH_KEYS)),
    ToolInfo("freeze", "FREEZE SPRAY", "Thermal", "Hold to freeze it solid, then smash the ice.",
             "Cold cells TRN_VP3 and a little touch. Cooling damps the brain and the shatter are game rules.",
             CONNECTOME,
             probes=((("cold", None),),)),
    ToolInfo("cleaner", "BRAKE CLEANER", "Chemical", "Hold: the solvent dissolves it.",
             "Smell neurons (ORN_*, 2,635) and taste neurons (leg, labellar, pharyngeal GRNs) at full strength; the "
             "growing inhibition is a game rule.", GAME_RULE,
             probes=((("smell", None),), TASTE)),
    ToolInfo("alcohol", "ALCOHOL", "Chemical", "Drop fermented fruit: sweet reward, escalating drunkenness.",
             "Sugar-pathway taste neurons and PAM reward neurons, and the fermentation glomeruli DM1, DM2, DP1m. The "
             "drunkenness is a game rule.", GAME_RULE, larva=False,
             larva_note="the larval brain has no DM1/DM2/DP1m glomeruli",
             probes=(TASTE, REWARD)),
    ToolInfo("cva", "cVA", "Chemical", "Puff cVA pheromone.",
             "Or67d ORNs of the DA1 glomerulus, then DA1 projection neurons and the lateral horn (validated). The "
             "cloud is a game rule.", CONNECTOME, larva=False, larva_note="no DA1 glomerulus in the larval brain",
             probes=((("scent", "cva"),),)),
    ToolInfo("sugar", "SUGAR", "Reward", "Drop sugar to reward it.",
             "Sugar-pathway taste neurons (30) and the PAM reward neurons (316), which the fly then walks over and "
             "eats. The walking and eating are game rules.", CONNECTOME,
             probes=(TASTE, REWARD)),
    ToolInfo("fruit", "FRUIT", "Reward", "Drop a piece of ripe fruit, like the orchard's.",
             "Exactly as sugar does: sugar-pathway taste neurons and PAM reward neurons. The fruit is a game rule.",
             CONNECTOME,
             probes=(TASTE, REWARD)),
    ToolInfo("spider", "SPIDER", "Creatures", "Drop a spider that hunts it.",
             "Bites hit body and leg touch neurons; looming detectors as it closes in. The hunting and the venom "
             "are game rules.", GAME_RULE, larva=False, larva_note="the larva has no looming detectors in this model",
             probes=(LOOM,)),
    ToolInfo("decoy", "DECOY", "Creatures", "Drop a decoy female to evoke courtship.",
             "Foreleg contact drives the 64 foreleg gustatory neurons LgLG5-8 (putative ppk23/ppk25). COURTSHIP is "
             "a game rule.", GAME_RULE, larva=False, larva_note="no LgLG5-8 neurons in the larval brain",
             probes=((("pheromone", "foreleg"),),)),
    ToolInfo("laser", "LASER", "Lab", "Stimulate or silence chosen cell types in real time.",
             "Whatever cell types you aim it at (optogenetics-style current), so it is Lab mode only.",
             CONNECTOME, lab_only=True, larva=False, larva_note="its cell types are the adult's"),
)
BY_NAME = {t.name: t for t in CATALOG}
# The order of kickthefly/game/kick_the_fly.py:TOOL_NAMES, which is the order the number keys always had (1 hand ... 0
# sugar, - alcohol, = laser). The "All" preset keeps it, so a migrated config's muscle memory still lands on the same
# tools. CATALOG above is in the editor's category order instead.
TOOL_NAMES = ("hand", "flick", "swatter", "bomb", "torch", "cleaner", "zapper", "freeze", "spider", "sugar", "alcohol",
              "laser", "cva", "decoy", "fruit")
assert set(TOOL_NAMES) == set(BY_NAME) and len(TOOL_NAMES) == len(CATALOG)

# name -> tuple of tool names (the hand is added by build() wherever a preset forgets it)
PRESETS: dict[str, tuple[str, ...]] = {
    "base": ("hand", "swatter", "torch", "freeze", "sugar"),
    "chaos": ("hand", "bomb", "torch", "cleaner", "zapper", "spider", "alcohol"),
    "chemist": ("hand", "cleaner", "alcohol", "cva", "sugar", "freeze"),
    "lab": TOOL_NAMES,                                     # every tool, the laser included
    "all": TOOL_NAMES,                                     # every tool the mode allows (no laser outside Lab)
    # Base minus the bomb, brake cleaner and zapper (it has none of them) plus sugar and the orchard's fruit
    "pet": ("hand", "swatter", "torch", "freeze", "sugar", "fruit"),
}
PRESET_LABELS = {"auto": "Auto", "base": "Base", "chaos": "Chaos", "chemist": "Chemist", "lab": "Lab",
                 "all": "All", "pet": "Pet", "custom": "Custom"}
PRESET_TIPS = {
    "auto": "Base in Play, Lab in Lab mode, Pet in Pet mode.",
    "base": "Hand, swatter, blowtorch, freeze spray, sugar.",
    "chaos": "Bomb, blowtorch, brake cleaner, zapper, spider, alcohol.",
    "chemist": "Brake cleaner, alcohol, cVA, sugar, freeze spray.",
    "lab": "Every tool, including the laser.",
    "all": "Every tool the mode allows (the laser only in Lab mode).",
    "pet": "Gentle tools for a pet: hand, swatter, blowtorch, freeze spray, sugar and fruit.",
    "custom": "Your own loadout, made in the editor.",
}
CHOICES = ("auto", "base", "chaos", "chemist", "lab", "all", "pet", "custom")   # values of controls.loadout_preset
MODE_DEFAULT = {"play": "base", "lab": "lab", "pet": "pet"}


def mode_of(cfg) -> str:
    return str(cfg.get("brain.mode", "play"))


def available(name: str, *, lab: bool, larva: bool) -> bool:
    """Whether a tool may be on the hotbar in this mode and brain."""
    t = BY_NAME.get(name)
    if t is None:
        return False
    if t.lab_only and not lab:
        return False
    if larva and not t.larva:
        return False
    return True


def available_tools(*, lab: bool, larva: bool) -> list[str]:
    return [n for n in TOOL_NAMES if available(n, lab=lab, larva=larva)]


def clean(names, *, lab: bool, larva: bool) -> list[str]:
    """A loadout's tools with unknown, duplicate and unavailable ones dropped and the hand first."""
    seen, out = set(), []
    for n in names:
        if isinstance(n, str) and n not in seen and available(n, lab=lab, larva=larva):
            seen.add(n)
            out.append(n)
    if HAND in out:
        out.remove(HAND)
    return [HAND, *out]


class Loadout:
    """The tools on the hotbar, in order, and which page of ten is showing."""

    def __init__(self, tools, *, preset: str = "custom", lab: bool = False, larva: bool = False):
        self.preset = preset
        self.lab, self.larva = lab, larva
        self.tools: list[str] = clean(tools, lab=lab, larva=larva)
        self.page = 0

    # --- reading ------------------------------------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.tools)

    def __iter__(self):
        return iter(self.tools)

    def __contains__(self, name) -> bool:
        return name in self.tools

    @property
    def n_pages(self) -> int:
        return max(1, -(-len(self.tools) // PAGE_SIZE))

    def page_tools(self, page: int | None = None) -> list[str]:
        p = self.page if page is None else page
        return self.tools[p * PAGE_SIZE:(p + 1) * PAGE_SIZE]

    def slot_tool(self, slot: int) -> str | None:
        """The tool in hotbar slot 0-9 (keys 1-9, 0) on the page showing, or None."""
        t = self.page_tools()
        return t[slot] if 0 <= slot < len(t) else None

    def slot_of(self, name: str) -> int | None:
        """The slot (0-9) of this tool on the page showing, or None."""
        t = self.page_tools()
        return t.index(name) if name in t else None

    def as_list(self) -> list[str]:
        return list(self.tools)

    # --- moving -------------------------------------------------------------------------------------------------------
    def turn_page(self, d: int) -> bool:
        """Previous (-1) / next (+1) page. False, and nothing happens, when everything fits on one page."""
        if self.n_pages <= 1:
            return False
        self.page = (self.page + d) % self.n_pages
        return True

    def show(self, name: str) -> None:
        """Turn to the page that has this tool."""
        if name in self.tools:
            self.page = self.tools.index(name) // PAGE_SIZE

    def neighbor(self, name: str, d: int) -> str:
        """The next (+1) or previous (-1) tool in the loadout, wrapping (mouse wheel and gamepad bumpers). A tool that
        isn't in it (picked from the wheel) starts from the front."""
        if name not in self.tools:
            return self.tools[0 if d > 0 else -1]
        return self.tools[(self.tools.index(name) + d) % len(self.tools)]

    # --- editing (the editor) -----------------------------------------------------------------------------------------
    def equip(self, name: str, index: int | None = None) -> bool:
        if not available(name, lab=self.lab, larva=self.larva) or name in self.tools:
            return False
        i = len(self.tools) if index is None else max(1, min(index, len(self.tools)))
        self.tools.insert(i, name)
        return True

    def unequip(self, name: str) -> bool:
        if name == HAND or name not in self.tools:
            return False
        self.tools.remove(name)
        self.page = min(self.page, self.n_pages - 1)
        return True

    def move(self, name: str, index: int) -> bool:
        """Reorder. The hand stays in slot 1, and nothing goes ahead of it."""
        if name == HAND or name not in self.tools:
            return False
        self.tools.remove(name)
        self.tools.insert(max(1, min(index, len(self.tools))), name)
        return True


def build(preset: str, custom, *, lab: bool, larva: bool, mode: str = "play") -> Loadout:
    """The loadout for a preset name (or "auto") in this mode. "custom" uses the saved custom slots."""
    name = MODE_DEFAULT.get(mode, "base") if preset == "auto" else preset
    if name == "custom":
        tools = list(custom) if custom else PRESETS["base"]
    else:
        tools = PRESETS.get(name, PRESETS["base"])
    return Loadout(tools, preset=name, lab=lab, larva=larva)


def resolve(cfg, *, larva: bool | None = None) -> Loadout:
    """The loadout a Config asks for, in its current mode (Config.lab / .pet / .larva)."""
    lab = bool(getattr(cfg, "lab", False))
    if larva is None:
        larva = bool(getattr(cfg, "larva", False))
    st = getattr(cfg, "loadout", None) or {}
    return build(str(cfg.get("controls.loadout_preset", "auto")), st.get("custom"), lab=lab, larva=larva,
                 mode=mode_of(cfg))


def describe(name: str) -> dict:
    t = BY_NAME[name]
    return dict(name=t.name, label=t.label, category=t.category, description=t.desc, neurons=t.neurons, tag=t.tag,
                lab_only=t.lab_only, larva=t.larva)


def stimulus_keys(brain, name: str) -> list[tuple]:
    """The Brain.sense keys a tool's documented neurons are driven through (the ones that have neurons in this brain).
    The same call the game's tools make (Brain.poke); the playthrough bot and Fly.use_tool use it."""
    return [k for probe in BY_NAME[name].probes for k in probe if len(brain.sense.get(k, ()))]


def use(brain, name: str, strength: float = 0.8) -> list[tuple]:
    """Drive a tool's documented neurons once, like one hit of the tool. Returns the keys driven."""
    keys = stimulus_keys(brain, name)
    for region, side in keys:
        brain.poke(region, side, strength)
    return keys
