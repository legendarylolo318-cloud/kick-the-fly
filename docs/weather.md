# Rain, gusts and storms (3.0 day 3)

The open field and the orchard (3D) can have weather. It is off by default, so every existing outdoor result is unchanged. Three Lab parameters (Lab > Parameters, or `lab_params`, or a protocol `weather:` block, or `fly.weather(...)`): `weather.rain` (0-1), `weather.gust_hz` (gusts a second, 0-0.5) and `weather.storm` (0 off, 1 on).

| what happens | what it drives | tag |
|---|---|---|
| raindrop hits | the fly's real **touch neurons by body part**: wings 35%, body 40%, head 15%, legs 10% of hits (the share of the body seen from above), as a tool hit does (`Brain.poke`) | CONNECTOME (the neurons) / **GAME RULE** (rate, part table, strength) |
| wet air | the humidity receptor neurons (HRN), as the pool does | CONNECTOME / GAME RULE |
| wet wings | rain fills the wings (20 s at full rain, faster with wing hits) and then the fly cannot take off, as after the pool; they dry in 30 s out of the rain | **GAME RULE** (the pool's rule) |
| gusts | extra wind speed (2-6 m/s for 1-3 s, direction wandering up to 25 degrees) fed through the **existing wind -> JO-C/E transduction** | CONNECTOME (JO-C/E) / GAME RULE |
| lightning | the photoreceptors R1-R8, both eyes, a 3-pulse flash every 5-14 s in a storm; thunder follows 1-3 s later | CONNECTOME / GAME RULE |
| a storm | at least 70% rain, 0.2 gusts a second and +3 m/s of wind, a darker scene, lightning | **GAME RULE** (the preset) |

**Reduced flashing** (Settings > Accessibility): the scene never flashes. The screen's lightning becomes one slow swell (0.6 s up, 0.6 s down, no brighter than 25%). The fly's photoreceptors still get the flash (they are the fly's eyes, not yours). Larger text and palettes affect the HUD line (`STORM 70%, gusts ...`) as everywhere else.

What it is not: raindrops are not simulated (a hit is a touch pulse of a set strength); no shelter under trees, no evaporation, no cooling, no wind shear. The humidity neurons idle at about 18 Hz in this model, so rain moves their **mean** rate only about 14%, although their 100 ms peaks rise clearly (the playthrough check reports both).

Visuals are cheap by design: up to 60 opaque streaks (0.7 m long) that fall in a 9 m box around the camera (opaque on purpose: translucent items go through the renderer's sorted layer, and the first version, 90 translucent streaks built with `segment()`, cost the brain a third of its real-time speed), a darker sky, a swelling flash; no particle state. `tests/test_weather.py` checks the budget (at most 60 streaks) and the performance numbers are in the day-3 handoff.

```python
fly.weather(rain=0.6, gust_hz=0.2, storm=True); fly.step(10.0)
```

Protocol: `protocols/weather_storm.yaml`. Playthrough: `extra:weather` (rain fires the wing, body, head, leg and humidity groups, a gust the wind neurons, lightning the photoreceptors; the screen's flash under reduced flashing is one slow swell).
