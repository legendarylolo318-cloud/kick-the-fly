# Pet Mode (Long-Lived Companion Fly)

Kick the Fly 2.11 introduces **Pet Mode** in `Esc > Mode`, allowing a single fly to persist across real days as a virtual companion. In the windowed game the pet is always an adult (the larva brain is headless-only in 2.11, see [larva.md](larva.md)).

---

## 1. Safety & System Architecture

### Absolute Process & Resource Invariants
- **No Background Execution**: The simulation **ONLY runs while the game is open**.
- **No Background Services**: There are **NO** background processes, daemons, cron jobs, systemd services, Windows services, startup/autostart entries, or background wake timers. When the game window is closed, Kick the Fly uses zero CPU, GPU, memory, or battery.
- **Save Location & Durability**:
  - Saved to the user's data directory:
    - Linux: `$XDG_DATA_HOME/kickthefly/pet/pet.ktfsave` (default `~/.local/share/kickthefly/pet/`, backup `pet.ktfsave.bak`).
    - Windows: `Documents\Kick the Fly\pet\pet.ktfsave` (and `pet.ktfsave.bak`).
  - Saves are written atomically via tempfiles, after copying the previous save to `pet.ktfsave.bak`. The game never deletes a pet save. If both files are unreadable, they are kept as `*.unreadable-<date>` before a new pet is created.
  - The pet is saved at launch (after the catch-up) and when the game quits.

---

## 2. Deterministic Wall-Clock Catch-Up Rule

When the game launches, the time elapsed since the fly was last saved is calculated:

$$\Delta t = t_{\text{launch}} - t_{\text{saved}}$$

### Clock-Tampering & Jump Guards
- **Negative Jump ($\Delta t < 0$)**: If system time moves backwards (e.g., NTP adjustment, timezone shift, or manual clock change), $\Delta t$ is clamped to 0.0 seconds and a notice is logged.
- **Excessive Jump ($\Delta t > 7 \text{ days}$)**: If the user returns after weeks or months, catch-up is clamped to a maximum of 7 days (604,800 s) to prevent overflow or immediate fatal decay.

### Catch-Up Progression
1. **Hunger**: Advances at a constant rate of $1.0$ per 24 hours of elapsed time ($+1.157 \times 10^{-5} \text{ s}^{-1}$).
2. **Sleep Cycle**:
   - If offline for $\ge 4.0$ hours, the fly is assumed to have slept naturally: sleep pressure is reset to a rested state ($0.05$) and a `"Slept while away"` event is logged in the timeline.
   - If offline for $< 4.0$ hours, sleep pressure advances by elapsed time ($1.0$ per 16 hours).
3. **Launch Greeting ("How's my fly")**: A casual summary banner greets the player on launch:
   - *e.g., "Welcome back! You were away for 3.5 hours. Your fly is getting hungry."*

---

## 3. Needs & Couplings to Real Neurons

The need states, their rates and the coupling strengths are all **GAME RULE** (chosen for play, not measured) and shown in `Lab > Parameters` (`Pet: ...`). What they drive are real connectome groups:

| Need State | Dynamics & Range | Coupling (GAME RULE) onto real neurons |
|---|---|---|
| **Hunger** ($0.0 \dots 1.0$) | Increases over time; decreased by feeding (sugar droplet, orchard fruit). | **Sugar Taste Gain**: Scales sensory input from taste receptor neurons by $(1.0 + 2.0 \cdot \text{hunger})$. Starved flies taste sugar more intensely.<br>**Reward Sensitivity**: Scales PAM dopaminergic reward excitation by $(1.0 + 1.5 \cdot \text{hunger})$. Starved flies find eating significantly more rewarding. |
| **Sleep Pressure** ($0.0 \dots 1.0$) | Accumulates while awake; dissipates during sleep. | **dFB Sleep Drive**: Drives the dorsal fan-shaped body sleep neurons (**FB6/FB7**) with $0.5 \cdot \text{pressure}$. When they cross the SLEEP threshold the existing SLEEP readout rests the fly. |
| **Mood** ($0.0 \dots 1.0$) | Evaluated dynamically from live dopamine firing rates. | A readout only: moves toward $0.5 + 0.5\,(\text{PAM} - \text{PPL1}) / (\text{PAM} + \text{PPL1})$ of the live group rates. It does not feed back into behaviour. |

---

## 4. Timeline & Memory Persistence

The pet retains all learned behavioral memories and its unique individuality profile:
- **Mushroom Body Synaptic Weights**: Kenyon cell $\to$ MBON plastic weights persist across sessions. A pet trained to avoid an odor remembers that conditioning across days.
- **Personality card**: the MEASURED card of the pet fly's own brain (built from Settings > Brain > Random seed), stored in the pet file once
  it has been measured (Esc > Fly arcade > Measure the flies in play); until then the widget says "card not measured". A pet file from before
  the 3.0 day 4 review carries a card drawn from the pet file's own seed (not measured, and not the brain's seed): it is kept as
  `legacy_personality_card` and never shown. See [individuality.md](individuality.md).
- **Life Timeline**: Logs major life events: adoption, meals, sleeps, conditioning sessions, and injuries.

---

## 5. Mortality & Death Rules

- **Default State: Immortal**: by default the pet save is never marked dead.
- **Optional "Real Stakes" Toggle**:
  `Settings > Brain > Pet Real Stakes`
  When enabled, a catch-up of $\ge 48$ hours that ends at $100\%$ hunger marks the pet dead and the fly in the game dies (the normal death and autopsy). Even in real-stakes mode, the previous save backup (`pet.ktfsave.bak`) is preserved.

---

## 6. Casual-Friendly HUD Widget

A clean status card renders in the bottom-left HUD during Pet Mode:
- **Title & Personality**: Displays the pet's individuality profile (*e.g., `PET: Bold Right-turner`*).
- **Hunger Bar**: Visual meter showing feeding need.
- **Sleep Bar**: Visual meter showing rest pressure.
- **Mood Indicator**: Content, Happy, Grumpy, or Sleeping (ZZZ).
- **Age Counter**: Real-world age in days since adoption.
