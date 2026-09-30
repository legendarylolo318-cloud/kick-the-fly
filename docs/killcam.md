# Kill cam (3.0)

When the fly dies the game offers a slow-motion replay of the last ~6 seconds of its brain, with the neurons whose firing rose most
highlighted. Offered on the autopsy card (KILL CAM), by a short prompt at the moment of death, and by the **;** key (rebindable;
d-pad down on a gamepad). **Skip** with Esc, Space, Enter, a click or B. **Save video** replays it once through the game's own video
recorder (MP4 with ffmpeg, else a GIF), so it lands in your screenshots folder like Shift+R.

## What is what

- **CONNECTOME**: every neuron's own firing rate, exactly as the brain panel showed it (the simulator's 200 ms average, sampled every
  ~40 ms, stored in one byte per neuron on a square-root scale: about 0.2 spikes/s resolution at 10 spikes/s, enough to replay and
  rank, not a recording to analyse). The brain panel's group bars replay with it.
- **GAME RULE**: the offer (Settings > Brain > **Kill cam offer**, on by default), the 6 s window, the 0.25x speed (0.2x with
  Reduced flashing), 12 highlighted neurons, the minimum rise that counts (5 spikes/s) and the skip keys. Death itself is a game rule (a
  simulation can't die on its own: the drive is cancelled over 1.5 s), so the replay stops at the moment of death and the cut-off is not
  replayed.

"Rose most" is each neuron's mean rate over the last 1.5 s before death minus its mean over the first 1.5 s of the window. It says what
was active, not what caused the death. If nothing rose by 5 spikes/s it says so instead of listing noise.

## What it does not do

- It never touches the live brain: the replay is frames copied into memory, and nothing it does feeds back.
- It replays the brain, not the body or the room: the room is dimmed behind the card. The fly being watched (the brain panel's) is the
  one recorded; if another of several flies dies, there is no replay for it.
- Time comes from the recorded brain steps, so a slowed, paused or stepped game replays at the right speed.

## Accessibility

Reduced flashing: slower default speed, no pulsing rings (steady outlines), no sparkles. Larger text and the colorblind palettes apply.
Keyboard, mouse and gamepad all skip it.
