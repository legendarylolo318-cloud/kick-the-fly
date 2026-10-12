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
