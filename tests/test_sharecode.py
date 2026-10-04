"""Share codes (3.0): round trips, damage detection, version gating, size fallback, validation, preview and apply."""
from __future__ import annotations

import random
import zlib
from types import SimpleNamespace

import pytest

from kickthefly.core import config
from kickthefly.core import loadout as lo
from kickthefly.core import sharecode as sc
from kickthefly.lab import lab


def payloads():
    return {
        "surgery": {"groups": {"Moonwalker neurons (MDN)": -1}, "types": {"DNp01": 1, "LC4": -1}},
        "loadout": {"name": "my kit", "tools": ["hand", "swatter", "sugar"]},
        "protocol": {"protocol": {"name": "tiny", "seed": 3, "flies": 2, "duration_s": 1.5, "warmup_s": 0.5,
                                  "stimuli": [{"at_s": 0.2, "target": "loom", "strength": 0.5}],
                                  "recordings": [{"name": "gf", "neurons": "dnp01"}]}},
        "challenge": {"challenge": "tmaze", "seed": 1004, "arena": "room",
                      "surgery": {"types": {"PPL101": -1}}, "params": {"noise_std": 0.06}},
        "lab": {"params": {"noise_std": 0.06, "thresh.escape": 5.0}},
    }


def ctx(**kw):
    base = dict(tools=set(lo.BY_NAME), lab_params={n: (p[4], p[5]) for n, p in lab.BY_NAME.items()},
                current_params={"noise_std": 0.05}, surgery_labels={"Moonwalker neurons (MDN)", "Touch neurons"},
                type_names={"DNp01", "LC4", "PPL101"}, challenges={"tmaze", "sneak"}, arenas={"room", "fan"})
    base.update(kw)
    return sc.Context(**base)


@pytest.mark.parametrize("kind", list(sc.KINDS))
def test_every_kind_round_trips(kind):
    p = payloads()[kind]
    text = sc.encode(kind, p)
    assert text.startswith(f"KTF{sc.VERSION}-{sc.KINDS[kind]}-") and len(text) <= sc.MAX_CODE_CHARS
    got = sc.decode(text)
    assert (got.kind, got.payload, got.version) == (kind, p, sc.VERSION)
    assert sc.validate(got, ctx()) is None


def test_encoding_is_deterministic_and_key_order_free():
    a = sc.encode("lab", {"params": {"noise_std": 0.06, "bias": 0.1}})
    b = sc.encode("lab", {"params": {"bias": 0.1, "noise_std": 0.06}})
    assert a == b


def test_codes_survive_how_people_copy_them():
    text = sc.encode("loadout", payloads()["loadout"])
    messy = "  " + text.lower().replace("-", " - ") + "\n"
    assert sc.decode(messy).payload == payloads()["loadout"]
    body = text.split("-", 2)[2]
    swapped = text.replace(body, body.replace("0", "O").replace("1", "l"))
    assert sc.decode(swapped).payload == payloads()["loadout"]          # O reads as 0, l as 1


def test_alphabet_has_no_confusables():
    assert not set("ILOU") & set(sc.ALPHABET) and len(set(sc.ALPHABET)) == 32


def test_any_single_changed_character_is_refused_with_a_reason():
    text = sc.encode("surgery", payloads()["surgery"])
    body_at = text.index("-", text.index("-") + 1) + 1
    rng = random.Random(1)
    for _ in range(400):
        i = rng.randrange(body_at, len(text))
        if text[i] == "-":
            continue
        c = rng.choice([x for x in sc.ALPHABET if x != text[i]])
        bad = text[:i] + c + text[i + 1:]
        # O/I/L-equivalent substitutions decode to the same code on purpose; everything else must be refused
        try:
            got = sc.decode(bad)
        except sc.ShareError as e:
            assert str(e)
        else:
            assert got.payload == payloads()["surgery"]


def test_truncated_and_padded_codes_are_refused():
    text = sc.encode("lab", payloads()["lab"])
    for cut in (len(text) - 1, len(text) - 7, len(text) // 2, 12):
        with pytest.raises(sc.ShareError):
            sc.decode(text[:cut])
    with pytest.raises(sc.ShareError):
        sc.decode(text + "AAAAA")


@pytest.mark.parametrize("junk", ["", "   ", "hello", "KTF", "KTF1-", "KTF1-XXX-AAAA", "KTF1-SRG-????", "KTFx-SRG-AAAA", None, 5])
def test_junk_is_refused_not_crashed(junk):
    with pytest.raises(sc.ShareError):
        sc.decode(junk)


def _forge(version, kind_byte, payload: bytes, kind3):
    import base64
    import hashlib

    c = zlib.compressobj(9, zlib.DEFLATED, -15)
    head = bytes([version, kind_byte]) + c.compress(payload) + c.flush()
    raw = head + hashlib.sha256(head).digest()[:4]
    body = sc._b32(raw)
    return f"KTF{version}-{kind3}-" + "-".join(body[i:i + 5] for i in range(0, len(body), 5))


def test_a_newer_version_is_refused_with_advice():
    code = _forge(2, 1, b"{}", "SRG")
    with pytest.raises(sc.ShareError, match="newer version"):
        sc.decode(code)


def test_unknown_kind_and_header_mismatch_are_refused():
    with pytest.raises(sc.ShareError, match="unknown kind"):
        sc.decode(_forge(1, 1, b"{}", "ZZZ"))
    with pytest.raises(sc.ShareError, match="header"):
        sc.decode(_forge(1, 2, b'{"groups":{}}', "SRG"))                # says surgery, header says loadout


def test_a_decompression_bomb_is_refused():
    big = b'{"a":"' + b"A" * (sc.MAX_PAYLOAD_BYTES * 4) + b'"}'
    with pytest.raises(sc.ShareError, match="more than this version accepts"):
        sc.decode(_forge(1, 1, big, "SRG"))


def test_non_json_contents_are_refused():
    with pytest.raises(sc.ShareError, match="not readable|damaged"):
        sc.decode(_forge(1, 1, b"\xff\xfe not json", "SRG"))
    with pytest.raises(sc.ShareError, match="not what a share code carries"):
        sc.decode(_forge(1, 1, b"[1,2,3]", "SRG"))


def test_encode_refuses_payloads_it_would_not_accept_back():
    for kind, bad in (("surgery", {"groups": {"x": 2}}), ("surgery", {}), ("loadout", {"tools": []}),
                      ("loadout", {"tools": ["hand"], "name": "x" * 30}), ("lab", {"params": {"a": "b"}}),
                      ("challenge", {"challenge": "t", "seed": -1}), ("protocol", {"protocol": 3}),
                      ("surgery", {"groups": {"a": 1}, "evil": 1}), ("bogus", {})):
        with pytest.raises(sc.ShareError):
            sc.encode(kind, bad)


def test_over_the_limit_falls_back_to_a_file(tmp_path):
    rng = random.Random(5)
    many = {f"type{rng.randrange(10**9)}": rng.choice((-1, 1)) for _ in range(190)}      # incompressible-ish
    payload = {"types": many}
    with pytest.raises(sc.CodeTooLarge, match="exported as a file"):
        sc.encode("surgery", payload)
    code, path = sc.encode_or_file("surgery", payload, tmp_path, "big one")
    assert code is None and path.suffix == sc.FILE_SUFFIX and path.exists()
    assert sc.read_any(str(path)).payload == payload                     # the import box accepts the file's path
    assert sc.read_any(path.read_text()).payload == payload               # or its text
    code2, path2 = sc.encode_or_file("surgery", payload, tmp_path, "big one")
    assert path2 != path and path.exists()                                # never overwrites
    small = sc.encode_or_file("lab", payloads()["lab"], tmp_path, "x")
    assert small[0].startswith("KTF1-LAB") and small[1] is None


def test_read_any_reports_a_missing_file(tmp_path):
    with pytest.raises(sc.ShareError, match="can't read"):
        sc.read_any(str(tmp_path / "nope.ktfshare"))


# --- validation: incompatible codes are refused with a reason ----------------------------------------------------------------
def test_validation_refusals():
    def v(kind, payload, **kw):
        return sc.validate(sc.Code(1, kind, payload), ctx(**kw))

    assert "tools this version doesn't have" in v("loadout", {"tools": ["hand", "plasma_cannon"]})
    assert "cell types that aren't in the adult brain" in v("surgery", {"types": {"NopeType": 1}})
    assert "doesn't have" in v("surgery", {"groups": {"Antennae of doom": -1}})
    assert "unknown Lab parameter" in v("lab", {"params": {"flux": 1.0}})
    assert "outside" in v("lab", {"params": {"noise_std": 50.0}})
    assert "unknown challenge" in v("challenge", {"challenge": "chess"})
    assert "unknown arena" in v("challenge", {"challenge": "tmaze", "arena": "moon"})
    assert v("protocol", {"protocol": {"name": "p", "surgery": {"x": 5}}}).startswith("the protocol can't be used")
    assert v("surgery", {"types": {"DNp01": 1}}, type_names=None) is None          # no brain to check against: accepted
    assert "larva" in v("surgery", {"types": {"DNp01": 1}}, type_names={"X"}, brain="larva")


def test_a_protocol_payload_is_checked_like_a_protocol_file():
    assert sc.validate(sc.Code(1, "protocol", payloads()["protocol"]), ctx()) is None


# --- preview and apply ---------------------------------------------------------------------------------------------------------
def test_preview_says_what_will_change():
    c = sc.Code(1, "surgery", payloads()["surgery"])
    pv = sc.preview(c, ctx(current_surgery={"groups": {"Touch neurons": -1}, "types": {}}))
    text = "\n".join(pv.lines)
    assert "Silence: Moonwalker neurons (MDN)" in text and "Stimulate every DNp01 neuron" in text
    assert "Replaces the surgery now active (1 switch)" in text
    pv = sc.preview(sc.Code(1, "lab", payloads()["lab"]), ctx())
    assert "noise_std: 0.05 -> 0.06" in "\n".join(pv.lines)
    pv = sc.preview(sc.Code(1, "loadout", payloads()["loadout"]), ctx(saved_loadouts=["a"] * 5))
    assert pv.warnings and "slots are full" in pv.warnings[0]
    pv = sc.preview(sc.Code(1, "protocol", payloads()["protocol"]), ctx(protocol_names={"tiny"}))
    assert "does not run it" in pv.lines[0] and "new name" in "\n".join(pv.lines)
    pv = sc.preview(sc.Code(1, "challenge", payloads()["challenge"]), ctx())
    assert pv.lines[0].startswith("Starts the challenge 'tmaze'")


class FakeGame:
    def __init__(self):
        from kickthefly.game import kick_the_fly as k

        self.k = k
        self.surgery_modes = [0] * len(k.SURGERY)
        self.type_ops = {"OLD": 1}
        self.cfg = config.Config(None)
        self.applied = 0
        self.params, self.settings, self.started = {}, {}, None

    def _apply_surgery(self):
        self.applied += 1

    def set_setting(self, key, value, save=True, force=False):
        self.settings[key] = value
        self.cfg.set(key, value)

    def apply_loadout_setting(self):
        self.loadout_applied = True

    def set_lab_param(self, name, value):
        self.params[name] = value

    def start_challenge(self, key):
        self.started = key


def test_apply_surgery_replaces_what_was_there():
    g = FakeGame()
    labels = [lab_ for lab_, _ in g.k.SURGERY]
    payload = {"groups": {labels[9]: -1}, "types": {"DNp01": 1}}
    sc.apply(sc.Code(1, "surgery", payload), g)
    assert g.surgery_modes[9] == -1 and sum(1 for m in g.surgery_modes if m) == 1
    assert g.type_ops == {"DNp01": 1} and g.applied == 1                # "OLD" is gone


def test_surgery_payload_round_trips_through_the_game_state():
    g = FakeGame()
    labels = [lab_ for lab_, _ in g.k.SURGERY]
    g.surgery_modes[2], g.surgery_modes[7] = 1, -1
    g.type_ops = {"MDN": -1, "KCg-m": 0}
    p = sc.payload_for_surgery(g)
    assert p == {"groups": {labels[2]: 1, labels[7]: -1}, "types": {"MDN": -1}}
    g2 = FakeGame()
    sc.apply(sc.decode(sc.encode("surgery", p)), g2)
    assert g2.surgery_modes == g.surgery_modes and g2.type_ops == {"MDN": -1}


def test_every_surgery_label_the_game_has_is_encodable():
    from kickthefly.game import kick_the_fly as k

    sc.encode("surgery", {"groups": {label: -1 for label, _ in k.SURGERY}})


def test_apply_loadout_and_params_and_challenge():
    g = FakeGame()
    sc.apply(sc.Code(1, "loadout", payloads()["loadout"]), g)
    assert g.cfg.loadout["custom"] == ["hand", "swatter", "sugar"]
    assert g.cfg.loadout["saved"][0]["name"] == "my kit" and g.settings["controls.loadout_preset"] == "custom"
    sc.apply(sc.Code(1, "lab", payloads()["lab"]), g)
    assert g.params == {"noise_std": 0.06, "thresh.escape": 5.0}
    g = FakeGame()
    sc.apply(sc.Code(1, "challenge", payloads()["challenge"]), g)
    assert g.started == "tmaze" and g.settings["brain.seed"] == 1004 and g.settings["brain.arena"] == "room"
    assert g.type_ops == {"PPL101": -1} and g.params == {"noise_std": 0.06}


def test_apply_protocol_writes_a_loadable_file_and_never_overwrites(isolated_home):
    from kickthefly.core import paths
    from kickthefly.lab import protocol

    c = sc.Code(1, "protocol", payloads()["protocol"])
    sc.apply(c, SimpleNamespace())
    sc.apply(c, SimpleNamespace())
    folder = paths.get().data_dir / "protocols"
    files = sorted(f.name for f in folder.glob("*.yaml"))
    assert files == ["tiny-2.yaml", "tiny.yaml"]
    p = protocol.load(folder / "tiny.yaml")
    assert p["seeds"] == [3, 4] and p["stimuli"][0]["target"] == "loom"


def test_a_code_contains_no_executable_content(monkeypatch):
    """Whatever a payload holds, decode/validate/preview only read it: a hostile string is just a string."""
    evil = {"groups": {"__import__('os').system('x')": 1}}
    text = sc.encode("surgery", evil)
    got = sc.decode(text)
    assert sc.validate(got, ctx()) is not None          # refused: no such switch, and nothing was executed


def test_share_decode_command_line(capsys):
    import subprocess
    import sys

    from kickthefly.game import kick_the_fly as k

    code = sc.encode("lab", payloads()["lab"])
    args = k.parse_args(["--share-decode", code])
    assert args.share_decode == code
    assert sc.main_decode(code) == 0
    out = capsys.readouterr().out
    assert "Lab parameters" in out and "accepted" in out and "noise_std = 0.06" in out
    assert sc.main_decode(code[:-3] + "AAA") == 2
    assert "refused" in capsys.readouterr().err
    assert sc.main_decode("KTF1-LDT-" + sc.encode("loadout", {"tools": ["hand", "no_such_tool"]}).split("-", 2)[2]) == 2
    assert "tools this version doesn't have" in capsys.readouterr().out


def test_new_command_line_flags_parse():
    from kickthefly.game import kick_the_fly as k

    a = k.parse_args(["--headless", "--rerun-bundle", "x.zip", "--out", "d"])
    assert a.rerun_bundle == "x.zip" and a.out == "d"
    a = k.parse_args(["--protocol", "p.yaml", "--bundle", "b.zip"])
    assert a.bundle == "b.zip"


# --- review (Day 1): hostile codes ----------------------------------------------------------------------------------------------
def test_a_protocol_name_that_is_a_path_is_refused():
    for name in ("../../../../etc/passwd", "a/b", "a\\b", ".hidden", "x\ny"):
        c = sc.Code(1, "protocol", {"protocol": {"name": name, "seed": 0, "duration_s": 1, "stimuli": [], "recordings": []}})
        assert "path separators" in sc.validate(c, ctx()), name


def test_protocol_runs_never_write_outside_their_folder(tmp_path):
    from kickthefly.lab import protocol

    assert protocol.folder_name("../../../../etc/passwd") == "etc-passwd"
    assert protocol.folder_name("") == "protocol" and "/" not in protocol.folder_name("a/b\\c")


def test_an_enormous_fly_count_is_refused_before_anything_is_built():
    """flies: 10**9 made protocol.check build a billion-seed list (tens of GB) while only previewing the code."""
    import time

    for payload in ({"name": "x", "seed": 0, "flies": 10**9}, {"name": "x", "seeds": list(range(20_000))}):
        c = sc.Code(1, "protocol", {"protocol": dict(payload, duration_s=1, stimuli=[], recordings=[])})
        t = time.time()
        why = sc.validate(c, ctx())
        assert why and "at most" in why and time.time() - t < 1.0


def test_the_loadout_preview_shows_what_apply_really_gives():
    pv = sc.preview(sc.Code(1, "loadout", {"tools": ["sugar", "sugar", "laser"]}), ctx())
    assert "Hotbar becomes: hand, sugar" == pv.lines[0]                  # duplicates dropped, hand added, no laser in Play
    pv = sc.preview(sc.Code(1, "loadout", {"tools": ["sugar", "laser"]}), ctx(lab=True))
    assert pv.lines[0] == "Hotbar becomes: hand, sugar, laser"


def test_a_loadout_name_with_control_characters_is_refused():
    why = sc.validate(sc.Code(1, "loadout", {"name": "a\x00\x1b[31mb", "tools": ["hand"]}), ctx())
    assert why and "printable" in why
