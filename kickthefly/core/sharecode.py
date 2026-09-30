"""Share codes (3.0): short, copyable text for a surgery, a loadout, a protocol, a challenge setup or a set of Lab parameters.

    KTF1-SRG-0A3F9-...      what you paste into Esc > Share > Import

GAME RULE, all of it: a share code is a container. It carries settings the game already has (the surgery switches, a tool
loadout, a protocol file, Lab parameters, a challenge and the world to play it in); it adds nothing to the simulation and
claims nothing about biology. Importing one previews what it will change and changes nothing until you confirm.

Format (version 1). Text is `KTF<version>-<KIND>-<body>`; the body is Crockford base32 (no I, L, O or U, so a code read
aloud or copied from a screenshot survives; O reads as 0, I and L as 1; case, spaces, line breaks and dashes are ignored)
in groups of five. The decoded bytes are `version | kind | raw-deflate(canonical JSON) | first 4 bytes of SHA-256(all
before it)`. A code over MAX_CODE_CHARS is refused by `encode` and `encode_or_file` writes it to a `.ktfshare` file instead
(the same text, any length); `decode` reads either. Nothing is ever executed from a code: the payload is data, every
field is checked against what this version knows, and the decompressed size is capped.

Refusals carry a reason the player can read: not a code, damaged (checksum), made by a newer version, unknown kind,
unknown tool / cell type / parameter, value out of range, too big.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
import zlib
from dataclasses import dataclass, field
from pathlib import Path

VERSION = 1
MAX_CODE_CHARS = 1200               # about what fits in a chat message; bigger goes to a file
MAX_PAYLOAD_BYTES = 256 * 1024      # decompressed size cap (a deflate bomb is refused, not expanded)
FILE_SUFFIX = ".ktfshare"
GROUP = 5
ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"          # Crockford base32
_B32 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"
_TO_CROCKFORD = str.maketrans(_B32, ALPHABET)
_FROM_CROCKFORD = str.maketrans(ALPHABET, _B32)
_FIX = str.maketrans("OIL", "011")

KINDS = {"surgery": "SRG", "loadout": "LDT", "protocol": "PRT", "challenge": "CHL", "lab": "LAB"}
KIND_OF = {v: k for k, v in KINDS.items()}
KIND_BYTE = {k: i + 1 for i, k in enumerate(KINDS)}
BYTE_KIND = {v: k for k, v in KIND_BYTE.items()}
KIND_LABEL = {"surgery": "Brain surgery", "loadout": "Tool loadout", "protocol": "Protocol",
              "challenge": "Challenge setup", "lab": "Lab parameters"}


class ShareError(ValueError):
    """A code that can't be used; str(e) is the reason shown to the player."""


class CodeTooLarge(ShareError):
    def __init__(self, kind: str, payload: dict, chars: int):
        super().__init__(f"this {KIND_LABEL[kind].lower()} is too big for a code ({chars:,} characters; the limit is "
                         f"{MAX_CODE_CHARS:,}); it can be exported as a file instead")
        self.kind, self.payload, self.chars = kind, payload, chars


@dataclass
class Code:
    version: int
    kind: str
    payload: dict


# --- encoding -----------------------------------------------------------------------------------------------------------------
def canonical(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _deflate(raw: bytes) -> bytes:
    c = zlib.compressobj(9, zlib.DEFLATED, -15)
    return c.compress(raw) + c.flush()


def _inflate(blob: bytes) -> bytes:
    d = zlib.decompressobj(-15)
    try:
        out = d.decompress(blob, MAX_PAYLOAD_BYTES + 1)
    except zlib.error:
        raise ShareError("the code's contents are damaged (they don't decompress)") from None
    if len(out) > MAX_PAYLOAD_BYTES or d.unconsumed_tail:
        raise ShareError("the code decompresses to more than this version accepts")
    return out


def _b32(data: bytes) -> str:
    return base64.b32encode(data).decode("ascii").rstrip("=").translate(_TO_CROCKFORD)


def _unb32(text: str) -> bytes:
    s = text.translate(_FROM_CROCKFORD)
    s += "=" * (-len(s) % 8)
    try:
        return base64.b32decode(s)
    except Exception:
        raise ShareError("the code has characters that aren't part of a share code") from None


def _body_bytes(version: int, kind: str, payload: dict) -> bytes:
    head = bytes([version, KIND_BYTE[kind]]) + _deflate(canonical(payload))
    return head + hashlib.sha256(head).digest()[:4]


def encode(kind: str, payload: dict, limit: int | None = MAX_CODE_CHARS) -> str:
    """The share code for a payload. `limit` None writes any length (for files). Raises ShareError for an unknown kind or
    a payload this version wouldn't accept back, CodeTooLarge when over the limit."""
    if kind not in KINDS:
        raise ShareError(f"unknown kind {kind!r}")
    problem = check_payload(kind, payload)
    if problem:
        raise ShareError(problem)
    body = _b32(_body_bytes(VERSION, kind, payload))
    text = f"KTF{VERSION}-{KINDS[kind]}-" + "-".join(body[i:i + GROUP] for i in range(0, len(body), GROUP))
    if limit is not None and len(text) > limit:
        raise CodeTooLarge(kind, payload, len(text))
    return text


_PREFIX = re.compile(r"^KTF(\d{1,3})-([A-Z]{3})-(.+)$")


def decode(text: str) -> Code:
    """Parse and verify a code (or the text of a .ktfshare file). Does not apply it, and does not check it against this
    install's tools or brain: call `validate` for that."""
    if not isinstance(text, str):
        raise ShareError("that isn't text")
    s = "".join(text.split()).upper()
    if not s:
        raise ShareError("nothing to import: paste a share code")
    m = _PREFIX.match(s)
    if not m:
        raise ShareError("that doesn't look like a share code (they start with KTF1-)")
    version, kind3, body = int(m.group(1)), m.group(2), m.group(3).replace("-", "").translate(_FIX)
    if version > VERSION:
        raise ShareError(f"this code was made by a newer version (code format {version}; this one reads up to {VERSION}): "
                         "update Kick the Fly to import it")
    if version < 1:
        raise ShareError("unsupported code format")
    if kind3 not in KIND_OF:
        raise ShareError(f"unknown kind '{kind3}' in the code")
    raw = _unb32(body)
    if len(raw) < 7:
        raise ShareError("the code is too short: part of it is missing")
    data, check = raw[:-4], raw[-4:]
    if hashlib.sha256(data).digest()[:4] != check:
        raise ShareError("the code is damaged (its checksum doesn't match): a character was mistyped or part is missing")
    if data[0] != version or BYTE_KIND.get(data[1]) != KIND_OF[kind3]:
        raise ShareError("the code's header doesn't match its contents")
    blob = _inflate(data[2:])
    try:
        payload = json.loads(blob.decode("ascii"))
    except (ValueError, UnicodeDecodeError):
        raise ShareError("the code's contents are damaged (not readable data)") from None
    if not isinstance(payload, dict):
        raise ShareError("the code's contents are not what a share code carries")
    return Code(version, KIND_OF[kind3], payload)


def export_file(code_text: str, folder: Path, stem: str) -> Path:
    """Write a code of any length to folder/<stem>.ktfshare (never overwriting: a number is added)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-.")[:40] or "share"
    p = folder / (stem + FILE_SUFFIX)
    k = 2
    while p.exists():
        p = folder / f"{stem}-{k}{FILE_SUFFIX}"
        k += 1
    p.write_text(code_text + "\n", encoding="ascii")
    return p


def encode_or_file(kind: str, payload: dict, folder: Path, stem: str) -> tuple[str | None, Path | None]:
    """(code, None) when it fits, else (None, path of the exported file)."""
    try:
        return encode(kind, payload), None
    except CodeTooLarge:
        return None, export_file(encode(kind, payload, limit=None), folder, stem)


def read_any(text_or_path: str) -> Code:
    """What Esc > Share > Import does with its text box: a pasted code, or the path of a .ktfshare file."""
    s = (text_or_path or "").strip().strip('"')
    if s and "\n" not in s and len(s) < 1024 and not s.upper().startswith("KTF"):
        p = Path(s).expanduser()
        if p.suffix.lower() == FILE_SUFFIX:
            try:
                return decode(p.read_text(encoding="ascii", errors="replace"))
            except OSError as e:
                raise ShareError(f"can't read {p.name}: {e.strerror or e}") from None
    return decode(s)


# --- checking a payload (this version's schema) -------------------------------------------------------------------------------
def _is_mode_map(d) -> bool:
    return isinstance(d, dict) and all(isinstance(k, str) and k and v in (-1, 1) and not isinstance(v, bool)
                                       for k, v in d.items())


def check_payload(kind: str, p) -> str | None:
    """A reason this payload is malformed for its kind, or None. Needs nothing from the running game."""
    if not isinstance(p, dict):
        return "the contents are not a mapping"
    if kind == "surgery":
        if set(p) - {"groups", "types"}:
            return f"unknown fields in a surgery code: {sorted(set(p) - {'groups', 'types'})}"
        if not (_is_mode_map(p.get("groups", {})) and _is_mode_map(p.get("types", {}))):
            return "a surgery code's switches must each be -1 (silence) or 1 (stimulate)"
        if not p.get("groups") and not p.get("types"):
            return "this surgery code switches nothing"
        if len(p.get("groups", {})) + len(p.get("types", {})) > 200:
            return "this surgery code switches more than 200 things"
    elif kind == "loadout":
        if set(p) - {"name", "tools"}:
            return f"unknown fields in a loadout code: {sorted(set(p) - {'name', 'tools'})}"
        tools = p.get("tools")
        if not (isinstance(tools, list) and tools and all(isinstance(t, str) for t in tools)) or len(tools) > 40:
            return "a loadout code needs a list of tool names"
        if not isinstance(p.get("name", ""), str) or len(p.get("name", "")) > 24:
            return "a loadout's name is at most 24 characters"
    elif kind == "protocol":
        if set(p) - {"protocol"} or not isinstance(p.get("protocol"), dict):
            return "a protocol code carries one protocol"
    elif kind == "challenge":
        allowed = {"challenge", "seed", "arena", "surgery", "params"}
        if set(p) - allowed:
            return f"unknown fields in a challenge code: {sorted(set(p) - allowed)}"
        if not isinstance(p.get("challenge"), str):
            return "a challenge code names its challenge"
        if "seed" in p and (isinstance(p["seed"], bool) or not isinstance(p["seed"], int) or not 0 <= p["seed"] < 2**31):
            return "the seed must be a whole number from 0 to 2147483647"
        if "arena" in p and not isinstance(p["arena"], str):
            return "the arena must be a name"
        if "surgery" in p:
            r = check_payload("surgery", p["surgery"])
            if r:
                return r
        if "params" in p:
            r = check_payload("lab", {"params": p["params"]})
            if r:
                return r
    elif kind == "lab":
        if set(p) - {"params"} or not isinstance(p.get("params"), dict) or not p["params"]:
            return "a Lab code carries a list of parameter values"
        for k, v in p["params"].items():
            if not isinstance(k, str) or isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or abs(v) > 1e9:
                return f"parameter {k!r} must be a number"
    else:
        return f"unknown kind {kind!r}"
    return None


@dataclass
class Context:
    """What this install knows, for `validate` and `preview`. Any field left None means "can't check that here"."""
    tools: set | None = None                  # loadout tool names available (loadout.BY_NAME)
    lab_params: dict | None = None            # name -> (lo, hi) bounds; and current values in `current_params`
    current_params: dict = field(default_factory=dict)
    surgery_labels: set | None = None         # the game's surgery switch labels
    type_names: set | None = None             # cell types in the loaded brain
    challenges: set | None = None
    arenas: set | None = None
    current_surgery: dict = field(default_factory=dict)    # {"groups": {...}, "types": {...}} now active
    current_loadout: list = field(default_factory=list)
    saved_loadouts: list = field(default_factory=list)      # names
    max_saved_loadouts: int = 5
    protocol_names: set = field(default_factory=set)
    larva: bool = False
    brain: str = "adult"


def validate(code: Code, ctx: Context) -> str | None:
    """A reason this code can't be imported here (unknown tool, cell type, parameter, out-of-range value...), or None."""
    p, k = code.payload, code.kind
    bad = check_payload(k, p)
    if bad:
        return bad
    if k == "surgery":
        return _validate_surgery(p, ctx)
    if k == "loadout":
        if ctx.tools is not None:
            unknown = [t for t in p["tools"] if t not in ctx.tools]
            if unknown:
                return f"this loadout uses tools this version doesn't have: {', '.join(unknown[:5])}"
    elif k == "lab":
        return _validate_params(p["params"], ctx)
    elif k == "protocol":
        try:
            from kickthefly.lab import protocol

            protocol.check(p["protocol"], "the shared protocol")
        except Exception as e:                        # ProtocolError, and anything odd in a hand-made payload
            return f"the protocol can't be used here: {e}"
    elif k == "challenge":
        if ctx.challenges is not None and p["challenge"] not in ctx.challenges:
            return f"unknown challenge '{p['challenge']}'"
        if ctx.arenas is not None and "arena" in p and p["arena"] not in ctx.arenas:
            return f"unknown arena '{p['arena']}'"
        if "surgery" in p:
            r = _validate_surgery(p["surgery"], ctx)
            if r:
                return r
        if "params" in p:
            return _validate_params(p["params"], ctx)
    return None


def _validate_surgery(p: dict, ctx: Context) -> str | None:
    if ctx.surgery_labels is not None:
        unknown = [g for g in p.get("groups", {}) if g not in ctx.surgery_labels]
        if unknown:
            return f"this surgery switches things this version doesn't have: {', '.join(unknown[:3])}"
    if ctx.type_names is not None:
        unknown = [t for t in p.get("types", {}) if t not in ctx.type_names]
        if unknown:
            return (f"this surgery names cell types that aren't in the {ctx.brain} brain you are running: "
                    f"{', '.join(unknown[:5])}")
    return None


def _validate_params(params: dict, ctx: Context) -> str | None:
    if ctx.lab_params is None:
        return None
    for name, v in params.items():
        if name not in ctx.lab_params:
            return f"unknown Lab parameter '{name}'"
        lo, hi = ctx.lab_params[name]
        if not lo <= float(v) <= hi:
            return f"Lab parameter '{name}' = {v:g} is outside {lo:g} to {hi:g}"
    return None


# --- what importing will change ---------------------------------------------------------------------------------------------
@dataclass
class Preview:
    title: str
    lines: list
    warnings: list = field(default_factory=list)


def _mode(v: int) -> str:
    return "silence" if v < 0 else "stimulate"


def preview(code: Code, ctx: Context) -> Preview:
    """Plain-language lines saying exactly what `apply` will change. Call `validate` first."""
    p, k = code.payload, code.kind
    title = f"{KIND_LABEL[k]} (code format {code.version})"
    lines, warns = [], []
    if k == "surgery":
        lines += _surgery_lines(p, ctx)
    elif k == "loadout":
        name = p.get("name") or "Shared loadout"
        lines.append(f"Hotbar becomes: {', '.join(p['tools'])}")
        if ctx.current_loadout:
            lines.append(f"Replaces the custom loadout ({len(ctx.current_loadout)} tools) and selects the Custom preset")
        if name in ctx.saved_loadouts:
            lines.append(f"Replaces your saved loadout '{name}'")
        elif len(ctx.saved_loadouts) < ctx.max_saved_loadouts:
            lines.append(f"Saves it as '{name}'")
        else:
            warns.append(f"All {ctx.max_saved_loadouts} loadout slots are full, so it won't be saved by name; it still "
                         "becomes your custom loadout")
        if ctx.larva:
            warns.append("Tools with no larval sensory mapping stay hidden in larva mode")
    elif k == "lab":
        lines += _param_lines(p["params"], ctx)
    elif k == "protocol":
        pr = p["protocol"]
        nm = pr.get("name", "shared-protocol")
        lines.append(f"Adds the protocol '{nm}' to your protocols folder (it does not run it)")
        lines.append(f"{'Assay ' + str(pr['assay']) if 'assay' in pr else 'Stimulus protocol'}; seeds "
                     f"{pr.get('seeds', pr.get('seed', 'default'))}")
        if pr.get("surgery"):
            lines.append(f"Its surgery: {', '.join(f'{s} {_mode(m)}' for s, m in list(pr['surgery'].items())[:6])}")
        if nm in ctx.protocol_names:
            lines.append("A protocol with that name exists: this one is saved under a new name, yours is untouched")
    elif k == "challenge":
        lines.append(f"Starts the challenge '{p['challenge']}' now")
        if "seed" in p:
            lines.append(f"Sets the random seed to {p['seed']}")
        if "arena" in p:
            lines.append(f"Switches the arena to {p['arena']}")
        if "surgery" in p:
            lines += _surgery_lines(p["surgery"], ctx)
        if "params" in p:
            lines += _param_lines(p["params"], ctx)
    if k in ("surgery", "challenge") and ctx.larva:
        warns.append("Cell types are matched by name; larval brains share few of the adult's names")
    return Preview(title, lines, warns)


def _surgery_lines(p: dict, ctx: Context) -> list:
    lines = []
    for label, m in p.get("groups", {}).items():
        lines.append(f"{_mode(m).capitalize()}: {label}")
    for t, m in p.get("types", {}).items():
        lines.append(f"{_mode(m).capitalize()} every {t} neuron")
    now = len(ctx.current_surgery.get("groups", {})) + len(ctx.current_surgery.get("types", {}))
    if now:
        lines.append(f"Replaces the surgery now active ({now} switch{'es' if now != 1 else ''})")
    return lines


def _param_lines(params: dict, ctx: Context) -> list:
    out = []
    for name, v in params.items():
        old = ctx.current_params.get(name)
        out.append(f"{name}: {old:g} -> {v:g}" if isinstance(old, (int, float)) else f"{name} = {v:g}")
    return out


# --- applying (the windowed game; any object with the same attributes works, which is how the tests run it) ------------------------
def payload_for_surgery(game) -> dict:
    """The surgery currently switched on in the game, as a share payload."""
    from kickthefly.game import kick_the_fly as k

    groups = {label: int(m) for (label, _), m in zip(k.SURGERY, game.surgery_modes) if m}
    types = {t: int(m) for t, m in game.type_ops.items() if m}
    return {"groups": groups, "types": types}


def apply(code: Code, game) -> list[str]:
    """Do what `preview` said. Returns short result lines. Validate first; this trusts a validated payload."""
    p, k = code.payload, code.kind
    if k == "surgery":
        return _apply_surgery(p, game)
    if k == "loadout":
        return _apply_loadout(p, game)
    if k == "lab":
        return _apply_params(p["params"], game)
    if k == "protocol":
        return _apply_protocol(p["protocol"])
    if k == "challenge":
        res = []
        if "seed" in p:
            game.set_setting("brain.seed", p["seed"])
            res.append(f"seed {p['seed']}")
        if "arena" in p:
            game.set_setting("brain.arena", p["arena"])
            res.append(f"arena {p['arena']}")
        if "surgery" in p:
            res += _apply_surgery(p["surgery"], game)
        if "params" in p:
            res += _apply_params(p["params"], game)
        game.start_challenge(p["challenge"])
        res.append(f"started {p['challenge']}")
        return res
    raise ShareError(f"unknown kind {k!r}")


def _apply_surgery(p: dict, game) -> list[str]:
    from kickthefly.game import kick_the_fly as k

    modes = [0] * len(k.SURGERY)
    labels = [label for label, _ in k.SURGERY]
    for label, m in p.get("groups", {}).items():
        modes[labels.index(label)] = int(m)
    game.surgery_modes = modes
    game.type_ops = {t: int(m) for t, m in p.get("types", {}).items()}
    game._apply_surgery()
    return [f"surgery set ({sum(1 for m in modes if m) + len(game.type_ops)} switches)"]


def _apply_loadout(p: dict, game) -> list[str]:
    cfg = game.cfg
    tools = list(p["tools"])
    cfg.custom_loadout(tools)
    res = []
    name = p.get("name") or "Shared loadout"
    if cfg.save_loadout(name, tools):
        res.append(f"saved loadout '{name}'")
    game.set_setting("controls.loadout_preset", "custom")
    game.apply_loadout_setting()
    cfg.save()
    res.append("loadout set")
    return res


def _apply_params(params: dict, game) -> list[str]:
    for name, v in params.items():
        game.set_lab_param(name, float(v))
    return [f"{len(params)} Lab parameter(s) set"]


def _apply_protocol(pr: dict) -> list[str]:
    import yaml

    from kickthefly.core import paths

    folder = paths.get().data_dir / "protocols"
    folder.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", str(pr.get("name", "shared-protocol"))).strip("-.")[:40] or "shared-protocol"
    p, k = folder / f"{stem}.yaml", 2
    while p.exists():
        p = folder / f"{stem}-{k}.yaml"
        k += 1
    data = {kk: vv for kk, vv in pr.items() if kk != "seeds" or "seed" not in pr}
    p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return [f"protocol saved as {p.name}"]
