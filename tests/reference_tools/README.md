# Reference tools

These scripts made the reference data the Rust tests compare against (`tests/golden/`, `tests/ui_reference/`, `tests/python_files/`), by
running the real Python game 3.1.1. That game is not in this branch: it is the branch `python-legacy` (tag `python-3.1.1`).

To regenerate something: check out `python-legacy` next to this checkout, copy the script (and the `tests/golden/make_*.py` one you need)
into its root, put `data/kick_brain.npz` in its `data/`, and run it with `SDL_VIDEODRIVER=dummy` in a venv with
`numpy scipy pygame-ce pyyaml pillow` (pygame-ce, not pygame). The outputs go back into `tests/` here.
