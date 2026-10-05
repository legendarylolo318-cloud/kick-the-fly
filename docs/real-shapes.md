# Real neuron shapes (3.1.0)

**Credit.** Neuron shapes come from the male CNS connectome (MaleCNS), version 1.0: FlyEM at HHMI Janelia with the University of Cambridge, the MRC Laboratory of
Molecular Biology and Google Research. CC BY 4.0. <https://male-cns.janelia.org/>. The credit is on the inspector's shape inset, in the README credits.

## What is publicly downloadable (checked 2026-10-04, from the release's own download page and the bucket's public listing; metadata only)
The release is CC BY 4.0. Skeletons and meshes are not the same thing there:

| item | where | offered how |
|---|---|---|
| **skeletons**, one SWC per neuron, EM space (8 nm units) | `https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-swc/<bodyId>.swc` | public bucket, plain HTTPS, no account or token |
| the same, Neuroglancer precomputed (1 nm), mirrored copies, JRC2018 template space | siblings of that path | public bucket |
| **meshes** | inside the Neuroglancer-precomputed segmentation volume (`.../v1.0/segmentation`) | **not offered as per-neuron downloads**; a volume is far too large to take pieces of here |
| everything else (the EM, segmentation, synapses, Neo4j database) | the same bucket, up to terabytes | not used |

So this feature draws **skeletons, not meshes**. Each object in the bucket carries a checksum (`x-goog-hash: crc32c=..., md5=...` on the download; the bucket's JSON
listing also gives each object's `size` and `md5Hash`); the page itself says no checksums are published, so the object checksum is what is verified. A skeleton is 0.4-50 kB
for the files looked at in the listing; the whole set is hundreds of gigabytes and nothing here downloads it.

## What the game does
- **Opt-in.** Settings > Brain > *Download real neuron shapes* (the same switch the ten-neuron download has had since 3.0; one question after the tutorial). Off by default.
  It is the game's only network use outside Streamer mode, and is never used in headless runs, validation, protocols or tests (`core/netguard.py`; `KICK_THE_FLY_OFFLINE=1` too).
- **On demand.** The first time you inspect a neuron (B for the big view, then click a neuron or search for one) its skeleton is fetched, one file, in the background,
  with a pause between requests, an 8 s timeout, a 5 MB / 400,000-node limit and one attempt a session per neuron.
- **Checksummed.** The bytes must match the server's md5; a file without a checksum header is refused. Each cached file's sha256 is kept in `shapes/manifest.json` and checked at every load:
  a file that no longer matches is thrown away and fetched again, never drawn.
- **Cached, not bundled.** `Documents\Kick the Fly\shapes` / `~/.local/share/kickthefly/shapes` (from source: your data folder too). Cached shapes load with the download off.
- **Drawn on the GPU.** In the big view the inspected neuron's full skeleton is laid over the brain picture, and a fitted three-quarter-view inset with the credit sits in the corner; both are
  drawn by the brain view's GPU worker (one buffer of line segments, one draw call, supersampled), in the same camera the brain view uses, so they follow orbit, pan and zoom. With no GPU or
  after any GPU error the same projection is drawn with pygame lines (`game/shape_draw.py`); the two agree (`tests/test_real_shapes.py`).
- **Falls back.** Offline, opted out, refused, a 404 (a neuron the release has no skeleton for) or any error: the card says "estimated fiber (why)" and the view draws the fiber it always drew.
- The ten neurons the brain view always drew from neuPrint (DNp01, DNa02, MBON01, MBON14, KCg) now come from this public, verified release first, neuPrint's API second.

The simulation never sees a shape: every neuron is still a single point, so no result depends on this. **GAME RULE** (what is drawn and how; the data itself is the CONNECTOME release's).

## Not verified here
The live download from Google Cloud Storage was **not** run: downloading a file needed your say-so, and none of the tests contact the network (they use a local fake that serves the same
`x-goog-hash` header). What is verified is everything around it, and the bucket path, the licence and the existence of per-neuron SWCs and their md5s from the public listing. Two things to check by
hand on first use: that a real SWC's coordinates land on the neuron's cell body in the brain view (the 8 nm EM-space coordinates are the same space as the pack's soma positions, as the old
neuPrint skeletons were), and that a real file stays under the limits. To try it: turn the setting on, press B, click a neuron.
