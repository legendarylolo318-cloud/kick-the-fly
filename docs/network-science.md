# Network science (Lab > Network science, `--netsci`) — 3.0 day 4

Degree distributions, reciprocity, 3-node motifs against a null model, the rich-club coefficient, modularity and communities, and
per-region summaries of the brain pack, for the adult (MaleCNS v1.0) and the larva (Winding et al. 2023). Everything is computed from
the pack's synapse counts: **nothing is simulated**.

| | what it is |
|---|---|
| **CONNECTOME** | every number: it describes the wiring diagram as this game ships it |
| **GAME RULE** | only the analysis choices: how many null graphs and wedge samples, the seed, the rich-club cut-offs, the community method |
| **MODEL PREDICTION** | none: nothing here predicts behavior |

## Using it

- **Lab > Network science:** pick Adult or Larva and press **Compute**. The adult pack has 10.3 million connections, so the first run
  takes about five minutes on one core (2.4 GB of memory); the larva takes seconds. It runs in a background job with a progress bar and
  a Cancel button, never on the game thread. **Export CSV** writes one file per table into your exports folder.
- **Headless:** `python kick_the_fly.py --headless --netsci [adult|larva|both] [--out DIR]` prints the summary and writes the CSVs.
- **Python:** `from kickthefly.lab.api import network_science; res = network_science("larva")`.
- **Cache:** the result is kept next to your data (`cache/netsci_<brain>_<pack hash>_<settings hash>.json`) with a SHA-256 of its own
  content. A different pack, different settings or a file that fails its checksum is recomputed, never trusted.

## Definitions

- A **connection** is an ordered pair (pre, post) with a nonzero synapse count in the pack; self-connections are dropped. Degrees
  count connections (binary), and the synapse-weighted versions are reported separately.
- **Reciprocity:** the share of connections whose reverse exists; and sum(min(w_ij, w_ji)) / sum(w) for the weighted version.
- **Motifs:** the 13 connected 3-node directed subgraphs. Counted by **sampling** connected triples (a centre and two of its neighbours
  in the undirected skeleton, each triple weighted by 1 / the number of wedges it contains) and scaling by the exact wedge count: they
  are estimates with a sampling error (tested against an exact count on a small graph). The names are descriptive, with the standard
  M-A-N code first.
- **Null model:** degree-preserving directed edge swaps (every neuron keeps its in- and out-degree; no self-loops or repeats), by
  default 3 independent samples with the same estimator. Enrichment = real / null mean; z uses the spread of the nulls, which is rough
  with so few.
- **Rich club:** phi(k) = 2 E(>k) / (N(>k) (N(>k) - 1)) on the undirected skeleton, normalised by the same quantity on its own null
  graphs: degree-preserving **undirected** swaps of the skeleton, so every neuron keeps the skeleton degree phi(k) is defined on
  (rho = phi / phi_null; rho > 1 is more than degrees alone give), at the 50th-99th percentile degrees. (3.0 day 4 review, analysis
  version 2: version 1 normalised by the skeleton of the directed null, which splits reciprocal pairs and changed the skeleton degree of
  2,632 of the larva's 2,952 neurons.)
- **Cross-check against networkx** (`tools/netsci_crosscheck.py`, `docs/results/day4/netsci_crosscheck/`): on the whole larva graph the
  sampled motif counts match networkx's exact triad census (largest |estimate - exact| = 2.8 sampling SE over 13 classes; our `111a`/`111b`
  are networkx's `111U`/`111D`), the estimator also matches the exact census of a null graph, phi(k) is identical, the modularity of our
  partition is identical to networkx's for the same partition (0.533384), and our NMI equals a from-the-definition implementation
  (geometric-mean normalisation). networkx's own Louvain finds a slightly better partition (Q 0.548, 13 communities; NMI with ours 0.72).
- **Communities:** a Louvain-style modularity maximisation on the symmetrised synapse-weighted graph, with synchronous moves on a random
  half of the neurons each sweep. It is a heuristic: the partition is not unique, and **Q is the exact modularity of the partition
  found, not the maximum possible**. NMI compares the communities with the pack's own region labels.
- **Regions:** per region label, the neuron count, mean in/out degree, how much of its output synapses stay inside the region and how much
  of its input comes from inside, and the inhibitory share of its output (by the pack's sign).

## What it found (this release, default settings: 3 null graphs, 300,000 wedges, seed 0)

| | adult (166,700 neurons) | larva (2,952 neurons) |
|---|---|---|
| connections / synapses | 10,272,080 / 102,356,281 | 110,140 / 351,828 |
| density | 3.7e-4 | 1.3e-2 |
| in-degree mean (median, max) | 61.6 (35, 9,184) | 37.3 (31, 209) |
| out-degree mean (median, max) | 61.6 (44, 7,749) | 37.3 (32, 159) |
| reciprocity (null) | 0.183 (0.002) x74 | 0.257 (0.025) x10 |
| communities / modularity Q / NMI with regions | 653 / 0.700 / 0.53 | 12 / 0.533 / 0.24 |
| computing time | about 5 minutes, 2.4 GB | 2 seconds |

Motifs, enrichment over the degree-preserving null (adult): the one-way motifs are **depleted** (out-star x0.58, in-star x0.62, chain x0.54)
and everything with a reciprocal pair is strongly **enriched** (feed-forward loop x8.1, 3-cycle x2.6, two mutual pairs sharing a neuron x174,
a neuron sending to both of a mutual pair x107). The null never produced an all-mutual triad in the sampled wedges (the real graph has
about 1.5 million), so its enrichment and z are **undefined** (shown as such, with that reason; version 1 printed `nan`, and the z = 21.6
quoted before was not a valid z). Rich club (analysis version 2, its own undirected null): rho of 1.02, 1.08, 1.06, 0.97, 0.86 and 0.70 from
the 50th to the 99th percentile (version 1: 1.04, 1.15, 1.08, 1.02, 0.92, 0.73): **no rich club beyond what degrees alone give; the
highest-degree neurons are, if anything, less interconnected than the null predicts.** The larva shows the same reciprocity-and-motif
signature (reciprocity x10, two mutual pairs x52) and a mild rich club (version 2: rho 1.07, 1.26, 1.36, 1.26, 1.11, 1.02). Motifs,
reciprocity and communities are unchanged by version 2 (the motif null's random stream was kept). Tables: `docs/results/day4/netsci_v2/`.

Read these with the limits in mind: the pack keeps connections of 3 or more synapses, EM coverage is uneven between regions, the null
is one choice of null (degree-preserving; others, such as ones that also preserve region structure, would give different enrichments),
and no distribution is fitted (no power-law claim is made).

Verified: the motif table against an exact count, the null's invariants, modularity against the textbook formula, the community finder on a
planted partition, the cache's checksum (`tests/test_day4_netsci.py`), and the whole analysis on both real packs (this table).
**Not verified:** the adult numbers have not been compared with another implementation (networkx and igraph are not dependencies and were
not installed).
