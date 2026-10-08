"""The game itself: the room, the fly's body, the tools and the arenas.

  kick_the_fly  the 2D game, the Brain wrapper around the simulator, the brain panel, and the shared game logic
                (its module docstring is the canonical connectome-vs-game-rule map)
  kick3d        the first-person 3D room built on top of it
  render3d      the OpenGL renderer for that room

Movement, damage and physics in here are game rules; what triggers them is read from live neuron firing.
"""
