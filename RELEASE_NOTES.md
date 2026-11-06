# Kick the Fly 4.1.0: many flies, fixed death screen, a smoother picture

**The simulation is unchanged**: the same brains fire the same spikes. Stepping the flies' brains side by side gives exactly the same
spikes as stepping them one after another (checked on the real brain pack, on 1, 2, 5 and 16 threads).

**Several flies, and when one dies**

- Up to **64 flies** (was 16). Past about ten flies the brains run slower than real time, but the picture keeps its frame rate.
- When a fly dies while others live, the brain panel, the stats and the focus move to a living fly at once. The death screen (the brain
  autopsy) and the kill cam now come only with the **last** death, and they show the fly you were following. Before, the first death
  could put up the restart screen while other flies were still alive.
- F skips dead flies, and spawning many flies at once no longer stops every brain for several seconds per new fly (the new brains warm up
  side by side).

**Clicks and screens**

- Clicking **Resume** in the pause menu could leave the game frozen behind the closed menu until Esc was pressed twice. Fixed.
- Brain surgery: **CLEAR ALL** switched on the "Sleep neurons" row hidden under it instead of clearing. The button now clears; the
  last row is still clickable just above it.
- **Esc** closes the help card and the big brain view first, as in 3.x, before the pause menu opens.
- Leaving Pet mode puts the pet away (its panel and its needs kept going in Play mode).
- The window can no longer be made smaller than 640 x 400 (the HUD became unreadable), and on Linux it now carries its app id, so the
  dock shows the right icon and window rules can match it.

**Performance**

The brains now run on their own pool of threads (about two fifths of your cores, at a lower priority on Linux), so they no longer fight
the picture for the CPU. Measured on a 24-thread Core Ultra 7 with an RX 9070 XT, FPS cap off, on a test build (not the shipping build;
first-pass numbers, one machine):

| flies | frames/s before -> after | brain steps/s (200 = real time) | CPU used (of 2400%) |
|---|---|---|---|
| 1 | 162 -> 173 | 200 -> 200 | 512% -> 371% |
| 10 | 77 -> 112 | 143 -> 200 | 1715% -> 983% |
| 50 | 69 -> 101 | 29 -> 51 | 1744% -> 1096% |

With 64 flies (same test build) the game held 60 fps (FPS cap 60) with the brains at about 37 steps/s, in about 4 GB of memory.

The FPS cap stays at 60 by default; **Settings > Graphics > FPS cap = 0 uncaps it**. `KTF_SIM_THREADS=n` sets the brain pool's size.

**Known**

- The numbers above were not re-measured on the shipping build or on a second machine.
- The menus and click fixes were play-tested on Linux through X11 (on Wayland only launching and keyboard focus were checked); the Windows build passes CI's tests and a launch test but was
  not played by hand.

---

# Kick the Fly 4.0.0: a new engine

Kick the Fly was rewritten from Python to Rust on the Bevy engine. It is the same game with the same fly brain: all 166,700 neurons of the
MaleCNS v1.0 connectome, and the new simulator fires the same neurons at the same moments as the old one (checked spike for spike on Linux).

**What you get**

- **One native program.** `KickTheFly.exe` for Windows and an AppImage for Linux, nothing to install, the brain inside. It draws through
  Vulkan (DirectX 12 on Windows), and runs natively on Wayland and on X11.
- **A faster brain.** One step of the whole brain takes about 0.6 ms instead of roughly 2 ms, and the brain runs beside the picture on its
  own thread, so more flies cost the picture less.
- **Everything you had keeps working.** Settings, save states, share codes, your pet and your fly's memories from 3.x are read as they are.
- The whole Lab, the arcade, challenges, Pet mode, photo mode, the microphone and Streamer mode, the gamepad, the Neurodex and the kill cam.

**Added in the last review pass**

- **The 1v1 duel is back** (press X): the fly gets a blaster, aims with its real steering neurons and shoots when its DNp35 neurons fire.
  It can kill you; R respawns you. It can also learn to fear your smell.
- The fly's reactions were checked against 3.1.1 neuron by neuron; several stimuli were weaker than they should have been and some flies
  were reading each other's brains. Fixed.
- Clicking the room gives the mouse back to the game, the hotbar can be clicked, screen shake works, and a few small things the Python
  game did and this build had forgotten.

**Being honest about it**

- The picture is **not faster** than 3.1.1's yet. The on-screen panels are drawn by the processor and take about 3 ms of each frame at
  1080p; the game holds the default 60 frames a second, but an uncapped run is slower than the Python game was. This is the next thing to fix.
- **Not in 4.0:** the 2D game, the larva brain, GPU brain engines (Numba, PyTorch, OpenGL), the Python API, NWB / TIFF export, GIF, video and
  time-lapse recording (the L key does nothing), replays. If you need one of them, 3.1.1 stays available on the `python-legacy` branch.
- Two Graphics settings do nothing yet (resolution scale, screenshot resolution), gamepad buttons cannot be rebound in the menu yet (the
  defaults work, and `config.toml` can be edited), and the gamepad has not been tried on real hardware by us.
- On Windows a duel between two flies can differ from Linux in the last digit (the two systems round sines differently), and Lab >
  Validation shows a different 'direction tracking' number for the wind compass. The brain's spikes are the same on both.
- Built and tested by CI on Ubuntu 22.04 and Windows; played on Linux (NVIDIA). Not tried on AMD or Intel graphics. Please report what
  you find (Esc > Settings > Help > Report a bug writes the report for you; nothing is sent by itself).

The Python game 3.1.1 is unchanged on the branch `python-legacy` (tag `python-3.1.1`).

---

# Kick the Fly 3.1.1: the GPU engine on Windows, and an FPS counter

**The simulation is unchanged**: no simulation code changed; NumPy and Numba give the same spikes as 3.1.0.

- **Windows: the gl compute engine now runs.** On Windows, ModernGL's WGL context is exactly the OpenGL version it asks for (3.3 by default, no compute shaders), so `gl` reported "OpenGL 3.3 < 4.3" and quietly fell back to NumPy, and the game's `auto` engine never used the GPU on Windows. 3.1.1 asks for OpenGL 4.6, then 4.3, on Windows only; Linux keeps its context order. On an RX 9070 XT under Windows 11 (headless benchmark, x real time): one brain 618 -> 858 steps/s uncapped, 8 brains 0.61x -> 0.99x, 16 brains 0.16x -> 0.66x, 32 brains 0.04x -> 0.27x. `--selftest`'s gl check matches the CPU (0.0% firing difference).
- **Still slower than Linux with many brains** on the same PC (16 brains: 0.66x vs 1.00x), on every engine, CPU ones included, so it is not the GPU driver. The open hypotheses (Python 3.11 vs 3.14, NumPy/BLAS build, Windows thread scheduling on hybrid cores) and the optional Windows settings you can try yourself are in [docs/performance.md](docs/performance.md#windows-311). The game never changes system settings.
- **Settings > Graphics > Show FPS counter** (GAME RULE, interface; off by default): a one-second FPS average in the top-right corner of both games. F3's profiler takes precedence.
