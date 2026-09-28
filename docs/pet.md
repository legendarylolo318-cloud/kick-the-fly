# Pet Mode (Long-Lived Companion Fly)

Kick the Fly 2.11 introduces **Pet Mode** in `Esc > Mode`, allowing a single fly (adult or larva) to persist across real days as a virtual companion.

---

## 1. Safety & System Architecture

### Absolute Process & Resource Invariants
- **No Background Execution**: The simulation **ONLY runs while the game is open**.
- **No Background Services**: There are **NO** background processes, daemons, cron jobs, systemd services, Windows services, startup/autostart entries, or background wake timers. When the game window is closed, Kick the Fly uses zero CPU, GPU, memory, or battery.
- **Save Location & Durability**:
  - Saved to the user's data directory:
    - Linux: `~/.local/share/Kick the Fly/pet/pet.ktfsave` (and backup `pet.ktfsave.bak`).
    - Windows: `Documents\Kick the Fly\pet\pet.ktfsave` (and `pet.ktfsave.bak`).
  - Saves are written atomically via tempfiles. Existing saves are never deleted without explicit confirmation.

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

Pet mode introduces biological need states (GAME RULE) that couple bidirectionally to live connectome circuits:

| Need State | Dynamics & Range | Circuit Couplings (CONNECTOME) |
|---|---|---|
| **Hunger** ($0.0 \dots 1.0$) | Increases over time; decreased by feeding (sugar droplet, orchard fruit). | **Sugar Taste Gain**: Scales sensory input from taste receptor neurons by $(1.0 + 2.0 \cdot \text{hunger})$. Starved flies taste sugar more intensely.<br>**Reward Sensitivity**: Scales PAM dopaminergic reward excitation by $(1.0 + 1.5 \cdot \text{hunger})$. Starved flies find eating significantly more rewarding. |
| **Sleep Pressure** ($0.0 \dots 1.0$) | Accumulates while awake; dissipates during sleep. | **dFB Sleep Drive**: Drives the dorsal fan-shaped body sleep neurons (**FB6/FB7** in adult; premotor resting DNs in larva). High sleep pressure triggers spontaneous resting and suppresses takeoff reflexes. |
| **Mood** ($0.0 \dots 1.0$) | Evaluated dynamically from live dopamine firing rates. | Driven by the ratio of PAM (reward / contentment) to PPL1 (punishment / stress) activity. Content flies show relaxed movement; stressed flies show heightened grooming and skittish reflexes. |

---

## 4. Timeline & Memory Persistence

The pet retains all learned behavioral memories and its unique individuality profile:
- **Mushroom Body Synaptic Weights**: Kenyon cell $\to$ MBON plastic weights persist across sessions. A pet trained to avoid an odor remembers that conditioning across days.
- **Individuality Seed & Personality**: Retains its deterministic individuality gains and personality profile (*e.g., "Alert Straight-walker"*).
- **Life Timeline**: Logs major life events: adoption, meals, sleeps, conditioning sessions, and injuries.

---

## 5. Mortality & Death Rules

- **Default State: Immortal**: By default, pet flies cannot die from hunger, neglect, or injury. If health drops to zero, the fly enters a stunned resting state and recovers.
- **Optional "Real Stakes" Toggle**:
  `Settings > Brain > Pet Real Stakes`
  When enabled, prolonged starvation ($\ge 48$ hours with $100\%$ hunger) or fatal physical trauma results in death, triggering the standard scientific autopsy screen. Even in real-stakes mode, the previous save backup (`pet.ktfsave.bak`) is preserved.

---

## 6. Casual-Friendly HUD Widget

A clean status card renders in the bottom-left HUD during Pet Mode:
- **Title & Personality**: Displays the pet's individuality profile (*e.g., `PET: Bold Right-turner`*).
- **Hunger Bar**: Visual meter showing feeding need.
- **Sleep Bar**: Visual meter showing rest pressure.
- **Mood Indicator**: Content, Happy, Grumpy, or Sleeping (ZZZ).
- **Age Counter**: Real-world age in days since adoption.
