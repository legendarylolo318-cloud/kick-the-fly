# Translating Kick the Fly

Kick the Fly uses a lightweight, dependency-free JSON catalog localization system located in `kickthefly/data/locales/`. This design avoids compile-step dependencies (such as GNU gettext `msgfmt`), supports direct UTF-8 editing, and packages portably in AppImage and PyInstaller distributions across Linux, macOS, and Windows.

---

## Language Selection & Fallback

Language preferences are managed in **Settings > Accessibility > Language** (`access.language` in `config.toml`):
- **Auto (System)**: Automatically detects system language via environment variables (`LC_ALL`, `LC_MESSAGES`, `LANG`) or Python's `locale` module.
- Explicit language choices (e.g., `English`, `Deutsch`).
- Any missing or untranslated keys fall back automatically to the English translation.

What goes through the catalog so far (2.10): the pause menu, the quit confirmation, the Settings tab names and the
setting labels and tips that have catalog entries. Everything else (the HUD, the Lab, popups) is still English only.

---

## Critical Translation Rules

> [!IMPORTANT]
> **Never translate scientific nomenclature**:
> - **Cell type names** (e.g., `ORN_DA1`, `DA1_lPN`, `pC1`, `pIP10`, `ps1`, `DNa01`, `FB6/FB7`, `EPG`, `KC`, `MBON`) must **never** be translated.
> - **Gene and protein names** (e.g., `Or67d`, `fruitless` / `fru`, `ppk23`, `ppk25`) must **never** be translated.
> - **Citations and paper references** (e.g., `Felsenberg et al. 2018`, `Seeds et al. 2014`, `Cachero 2010`) must **never** be translated.
>
> All anatomical, synaptic, and genetic nomenclature reflects canonical FlyWire / MaleCNS v1.0 connectome datasets and must remain untouched across all locales.

---

## How to Add a New Language

1. **Copy the template**:
   ```bash
   cp kickthefly/data/locales/template.json kickthefly/data/locales/<language_code>.json
   ```
   Use the ISO 639-1 two-letter lowercase language code (e.g., `fr.json`, `es.json`, `ja.json`).

2. **Translate string values**:
   Open `<language_code>.json` in any UTF-8 text editor. Keys are English identifiers; fill in the corresponding values in your target language:
   ```json
   {
     "PAUSED": "PAUSE",
     "Resume": "Reprendre",
     "Settings": "Paramètres"
   }
   ```

3. **Register the language**:
   - In [`kickthefly/core/i18n.py`](../kickthefly/core/i18n.py), add your language code and display name to `AVAILABLE_LANGUAGES`.
   - In [`kickthefly/core/config.py`](../kickthefly/core/config.py), add your language code to the `access.language` setting `options` and `labels`.

4. **Verify and test**:
   Launch the game or run pytest:
   ```bash
   pytest tests/test_i18n.py
   ```

---

## Machine-Translated Catalogs

If contributing a machine-translated locale, you **must** clearly document that it is machine-translated and incomplete in the catalog metadata header (`_comment` field) and UI label (e.g., `Deutsch (Machine-translated)`).
