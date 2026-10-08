"""The connectome and the spiking model that runs on it.

  brainpack           the compact ~40 MB pack of everything the game needs from the connectome
  connectome.loader   downloads MaleCNS v1.0, filters and signs it, pickles data/graph.pkl
  connectome.sim      the leaky integrate-and-fire simulator over the signed synapse matrix

Nothing here knows about the game, the UI or Lab mode; it is the layer they all read from.
"""
