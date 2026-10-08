"""Machinery every other subpackage rests on, none of it specific to the game or the Lab.

  config        settings and their defaults, read from and written to config.toml
  crash         logging, the crash report, and where it goes
  memory        the fly's mushroom body training memory on disk
  paths         user folders (settings, memory, saves, exports, screenshots) per platform
  platform_env  display backend and DPI choices made before pygame starts
  savestate     the versioned .ktfsave format: every neuron, body and timer at one moment
  simclock      the virtual clock: pause, slow motion, single step
  simcore       brains without a window: build, drive and step LIF brains in lockstep
  version       the app version string
"""
