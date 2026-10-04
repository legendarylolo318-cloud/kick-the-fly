# Running from source, optional backends and building

Needs Python 3.11, a GPU with OpenGL 3.3 for 3D, and about 1.5 GB of disk for the connectome.

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m kickthefly.sim.connectome.loader build   # downloads the connectome (~1.1 GB) and builds data/graph.pkl
.venv\Scripts\python kick_the_fly.py                # the first run packs data/kick_brain.npz (~30 s)
```

On Linux/macOS, use `python3` and forward slashes instead:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m kickthefly.sim.connectome.loader build   # downloads the connectome (~1.1 GB) and builds data/graph.pkl
.venv/bin/python kick_the_fly.py                    # the first run packs data/kick_brain.npz (~30 s)
```

## Optional: faster simulation with Numba, PyTorch or OpenGL

The brain simulation runs on plain NumPy by default. Optional backends can run it instead; install one and the
game uses it by itself (`auto`), or pick one in Settings > Brain > Compute backend or with `--backend NAME`:

| backend | install | what it does |
|---|---|---|
| `numba` | `pip install numba` | JIT-compiled CPU kernels that release Python's interpreter lock, so several flies' brains run in parallel ([numbers](performance.md)) |
| `torch-cuda` | PyTorch with CUDA, from the selector on [pytorch.org](https://pytorch.org/get-started/locally/) | NVIDIA GPU |
| `torch-rocm` | PyTorch with ROCm (Linux), from the same selector | AMD GPU |
| `torch-cpu` | any PyTorch | PyTorch on the CPU; mainly for checking the torch code path |
| `gl` | nothing extra (ModernGL is already a requirement); needs OpenGL 4.3 | OpenGL compute shaders on any vendor's GPU (AMD, NVIDIA, Intel), with a process's flies stepped together on one context. Only when you pick it ([numbers](performance.md)) |

`auto` picks a PyTorch GPU, then Numba, then NumPy (never `gl`). A backend that can't start (library missing, no GPU visible to that
PyTorch build) falls back to NumPy and logs why; the backend that actually ran is what the Lab header, benchmarks,
validation results, exports, save states and crash reports record. Numba and `torch-cpu` give **exactly** the same
spikes as NumPy, so every result in this README is the same on them; GPU backends (`torch-cuda`, `torch-rocm`, `gl`) agree statistically but not spike for
spike (see [Deterministic runs](playing.md#time-controls-and-save-states)). The number of flies you can spawn depends on the
backend (see Controls: N).

**The exe and the AppImage include neither.** They always run the NumPy backend and don't pick up a Numba or PyTorch
you've installed on your system (a frozen app can't safely load another Python's packages); choosing another backend
there falls back to NumPy with a note in the log. For Numba or a GPU, run from source.

Tests (the validation suite takes a few minutes; `-m "not validation"` skips it):

```bash
.venv/bin/pip install pytest
.venv/bin/python -m pytest
```

To build the exe yourself (after one run from source, so the brain pack exists; add `data/validation_results.json` from `--validate` for the real-science popups):

```powershell
powershell -ExecutionPolicy Bypass -File build_exe.ps1
```

To build the AppImage yourself (same prerequisite):

```bash
./build_appimage.sh
```

Releases are built by `.github/workflows/release.yml` on a tag push: the brain pack is built from the public connectome, all tests (including validation) run on Ubuntu 22.04 and Windows, the AppImage is built on Ubuntu 22.04 and the exe on Windows, and the release is published with `SHA256SUMS` only if everything passes.
