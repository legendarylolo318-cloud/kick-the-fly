# Share codes (3.0)

A share code is short copyable text for a **brain surgery**, a **tool loadout**, a **protocol**, a **challenge setup** (a challenge plus
the seed, arena, surgery and Lab parameters it is played with) a set of **Lab parameters** or (3.1.0) a **contraption build** (CTN, see [contraption.md](contraption.md)):

```
KTF1-SRG-040TP-NJA5Z-52YB9-8ATSA-NBPNA-4NAJB-280MR-SAQ7W-1860S-5DC1H-NPP0C-DP3M1-M
```

**Make one:** Esc > Share > Make a code. **Use one:** Esc > Share > Import a code: paste it (Ctrl+V or the Paste button), read what it
would change, press Apply. Nothing changes until Apply. `--share-decode CODE` prints the same preview on the command line.

GAME RULE, all of it: a code is a container for settings the game already has. It adds nothing to the simulation and makes no claim
about biology. Nothing is executed from a code: the contents are data, every field is checked, and the decompressed size is capped.

## Format (version 1)

`KTF<version>-<KIND>-<body>`. KIND is SRG, LDT, PRT, CHL, LAB or (3.1.0, appended, so every earlier kind keeps its byte) CTN. The body is Crockford base32 (no I, L, O or U; O reads as 0 and I
and L as 1; case, spaces, line breaks and dashes are ignored) in groups of five, of `version | kind | raw-deflate(canonical JSON) |
first 4 bytes of SHA-256 of everything before`. A code is refused, with the reason, when it is not a code, is damaged (checksum), was
made by a newer version, names an unknown kind, tool, cell type, parameter or arena, or has a value out of range.

**Too big:** a code over 1,200 characters is not made; the game writes a `.ktfshare` file (same text, any length) to the share folder
in your data folder instead, and Import accepts the file's path in the same box.

**Preview and Apply** (what each kind does):

| kind | apply |
|---|---|
| surgery | replaces the surgery switches and per-type switches now on |
| loadout | sets the custom loadout, selects the Custom preset, saves it under its name if a slot is free |
| Lab parameters | sets each listed parameter (they must be in range) |
| protocol | adds the protocol to your protocols folder under its name (a new name if one exists); it does not run it |
| challenge | sets the seed and arena, the surgery and parameters, then starts the challenge |
| contraption | saves the build in the first free of the eight contraption slots (refused with the reason if all are full, and nothing is overwritten); it does not run it |

Cell types are matched by name against the brain you are running, so a surgery made on the adult brain is refused on the larva if it
names types the larva doesn't have.

## Privacy

Nothing is sent anywhere. The clipboard is read only when you press Paste or Ctrl+V in the box, and written only when you press Copy.
