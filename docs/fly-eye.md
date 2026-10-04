# Fly's-eye view (3.1.0)

**F4** (rebindable in Settings > Controls; off at every launch) shows the scene as a fly's eyes would sample it, in the 3D game and in `--2d`. Tag: **GAME RULE**, and the screen says why:
**this picture is visual only and is not fed to the brain.** The README records that streaming pixels through the simulation's own photoreceptors failed (a looming image stayed inside the
brain's random flicker), so the game drives LPLC2/LC4 directly from how fast an object grows in the fly's view. Nothing here changes that. The view is drawn from the same scene
for you to look at; the panel beside it shows the activity the brain really has; neither feeds the other (`tests/test_fly_eye.py` checks the code has no way to poke, drive or step the
brain, and that a brain stepped with the view on and drawn ends in exactly the same state as one stepped with it off).

## What it does to the picture
- **Wide field.** In the 3D game the scene is rendered three times from the focused fly's head (straight ahead, and 90 degrees to each side, 100 degrees each) and the eyes sample the union:
  about 280 degrees across against the 70 you normally look through. Each eye covers about 165 degrees and the two overlap about 30 degrees in front. Your own tool and hand are not drawn: it is the
  fly's view.
- **Ommatidia.** Each eye is a hexagonal lattice, 5 degrees between neighbours (a fly's are about 4.6) and about 770 ommatidia an eye (a fly has about 700), each seeing one color over a 5 degree
  blur, drawn as a hexagon, the two eyes side by side with azimuth across and elevation up. Detail finer than 5 degrees is gone. The window can be any shape: the hexagons stay regular and the
  rest is black margin. It redraws about 20 times a second and the three renders cost a few milliseconds.
- **Spectral sensitivity, approximately.** Flies see UV, blue and green, and hardly red. The display's red, green and blue are approximated by Gaussians (peaks 620, 535, 455 nm) and each photoreceptor
  class by a Gaussian at its textbook peak: **R1-R6** 480 nm (the broad luminance and motion cells; their UV peak at 360 nm is dark on a screen), **R7** 345 nm (UV, which a monitor never emits: **always
  dark here**, whatever is shown), **R8** 437 nm (blue) and 520 nm (green). The responses are drawn in false color (R1-R6 warm, R8 green and blue, R7 violet), after a light adaptation (a slow gain that
  brings the bright parts of the scene up, never more than 12 times). Result: red things look nearly black (red gives R1-R6 0.04, R8 0.10), blue and green look bright, white is white.
  These are peak wavelengths and widths from the literature, not measured spectra: a sketch of the effect, not a measurement of a fly's vision. A real fly in a real room would also see the UV a screen cannot show.
- **2D game.** The 2D arena has no scene to render from the fly's head, so the 2D view is a 520 x 260 pixel crop of the arena ahead of the fly (the way it faces), sampled through a flat hexagonal lattice
  (14 px spacing) with the same spectral model and adaptation. It is a picture of what is ahead, not a wide field.

## Beside it: the brain's own visual neurons
Eight bars of the mean firing (Hz) of the real visual cell types in the live brain: R1-R6, R7/R8, L1/L2 (the lamina's ON and OFF motion inputs), T4/T5 (direction-selective motion), LC4 and LPLC2 (looming, to the
giant fiber), LC10 (target tracking) and LC11 (small objects), the bar turning orange when a group runs well above its own slow average. This is the sim's activity, read, never changed. In the
model most of the early visual cells are driven only by the game's own transductions (looming, motion), not by this picture, which is why they are not seen "responding to the image".

## Not verified
The look in a window of every size and the exact appearance on every GPU (the render is tested against fakes, and looked at by eye in the 3D game and in 2D here); how a human fly neuroscientist would judge the false-color mapping.
