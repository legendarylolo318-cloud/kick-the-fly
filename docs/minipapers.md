# Guided mini-papers (3.0 day 5)

Short guided experiments that reproduce a classic paper step by step: **read it, state a hypothesis, run it on the model, plot it, and see your
result next to what the paper found.** Open them from **Esc > Mini-papers** (Play, Lab and Pet) or **Lab > Mini-papers**; headless:

```bash
python kick_the_fly.py --headless --minipaper list
python kick_the_fly.py --headless --minipaper hampel_2015 --workers 4 --out DIR      # the quick run: 4 flies, exploration seeds 0-3
python kick_the_fly.py --headless --minipaper hampel_2015 --seeds 1000-1009 --workers 6   # the full run: the validation seeds
```

Python: `kickthefly.lab.api.minipaper(paper, seeds=None)`, then `kickthefly.lab.minipapers.render(result)` or `.compare(result, answers)`.
Code: `kickthefly/lab/minipapers.py` (the papers, the run, the comparison), `kickthefly/ui/minipaper_ui.py` (the page), tests
`tests/test_day5_pages.py`. Each paper is also a lecture protocol (`lab/classroom.py`: `LectureProtocol`, `ClassroomSession`, registered in
`classroom.EXTRA_LECTURES` so the five curated lectures stay five).

## How it works

1. **The paper.** Its citation, DOI, **what was read**, and **what the paper itself reports**, in its own terms.
2. **Your hypothesis.** One multiple-choice question per experiment; you answer before the Run is enabled and can change it until you run.
3. **Run.** *Quick* (4 flies, exploration seeds 0-3) is a first look: with n = 4 the validation's p < 0.01 cannot be reached (the smallest
   one-sided Wilcoxon p is 1/16), so a quick verdict rests on effect size and direction only and says so. *Full* (10 flies, the validation seeds
   1000-1009) uses the validation's own criteria. The run is a background job in worker processes with a progress bar and a Cancel button.
4. **Plot.** One dot per fly for the drive and one for its matched control, joined by a line, with the validation's bars drawn.
5. **Next to the paper.** Your hypothesis, the paper's direction, whether the model reproduces it, the numbers, and what this model cannot check.

Nothing is re-implemented: every adult experiment is `validation.run(seeds, include=[test id])`, so the numbers are the validation suite's. The
T-maze is `mb_conditioning`, the larva pair are the larva tests, and the Buridan paper runs `lab/rigassay.py`'s rig.

## What the comparison may say

- *What the paper found* is written from what the paper states, nothing more. **Where a paper states no number in the part read, none is shown**
  ("none stated"), and a claim that rests on another paper that was not read is marked as such. Quotes are kept to a few words.
- *The model's result* is a MODEL PREDICTION. *The paper* is LITERATURE.
- **Where the model fails the paper, the mini-paper says so and why**, from `docs/validation.md`.
- A result that differs from the validation's recorded outcome (a miss where the test passes, a pass where it fails) is flagged "Unexpected" and
  points back to the Full run.
- **The larva paper** needs the larva brain pack, which is built locally from a download (docs/larva.md). Without it the page shows the recorded
  validation numbers, marked *recorded, not run now*; with it, it runs live.

## The six papers, and how much of each was read

| id | paper | what was read | experiment | the model |
|---|---|---|---|---|
| `von_reyn_2014` | von Reyn et al. 2014, *Nat Neurosci* 17:962-970, doi:10.1038/nn.3741 | **abstract only** (Europe PMC); the full text is paywalled | drive LPLC2 + LC4, read the giant fiber DNp01 | validation `looming_escape`: **PASS** (x11.81 versus x0.80). The abstract states no effect size and does not name the detectors, so only the premise (detectors excite the giant fiber) is checked; spike timing and the two takeoff modes are not |
| `tully_quinn_1985` | Tully & Quinn 1985, *J Comp Physiol A* 157:263-277, doi:10.1007/BF01350033 (PubMed 3939242) | **abstract only** | odor + shock, six cycles, T-maze choice | validation `mb_conditioning`: **PASS** (PI 1.00 versus -0.03 unpaired). The paper's stated figure is 95% of trained flies avoiding the shock-associated odor; the model's per-fly PI converts to the share of choices that avoid it as (1 + PI) / 2, which is derived here, not stated by the paper. Retention, intensity, delay and the mutants are not modeled |
| `shiu_2024` | Shiu et al. 2024, *Nature* 634:210-219, doi:10.1038/s41586-024-07763-9 | **abstract only**; paywalled | drive 30 sugar-pathway taste neurons, read MN9, against 30 bitter-pathway | validation `sugar_feeding`: **PASS** (x2.11 versus x1.25). The abstract names no motor neuron and no effect size; the paper's model is on another connectome (a central brain of more than 125,000 neurons), this game's is the MaleCNS |
| `hampel_2015` | Hampel et al. 2015, *eLife* 4:e08758, doi:10.7554/eLife.08758 | **full text** (open access, PMC4599031), through a page summary, and the abstract | JO-C/E to aDN1/aDN2, then aDN to front-leg motor neurons | `antenna_grooming_circuit` **PASS** (x4.87 versus x0.85), `adn_grooming_motor` **FAIL** (x1.13, too weak; the model has no legs). The paper reports a strong aDN1 and only a weak aDN2 response to JO activation; the model pools them |
| `ohyama_2015` | Ohyama et al. 2015, *Nature* 520:633-639, doi:10.1038/nature14297 | **abstract only**; paywalled | larva: nociceptive then chordotonal ascending neurons to their targets and the `_telegoro-1` pair | both larva tests **FAIL** (the readouts drop for drive and control alike: x0.66 versus x0.65, x0.68 versus x0.64) because the pack's transmitter signs are a guess and the network idles at about 60 Hz; the dataset lacks the class IV neurons, the Basins and Goro itself |
| `colomb_2012` | Colomb et al. 2012, *PLoS ONE* 7(8):e42247, doi:10.1371/journal.pone.0042247 | **full text** (open access), through a page summary | Buridan's paradigm: stripe deviation with and without stripes | the Buridan rig's criteria B1 and B2: **PASS** (docs/rigs.md); the model's deviation is about 3 degrees against the paper's chance level of 45 and flies' much looser fixation |

"Through a page summary" means the text was read by fetching the page and having a small model extract the passages asked for, not line by line; the
abstracts are the Europe PMC / PubMed records. **Nobody has read the paywalled papers' full texts for this**, so every statement about them is limited
to their abstracts, and the Neurodex and validation citations that rest on other papers (Ache et al. 2019, von Reyn et al. 2017, Winding et al. 2023,
Jovanic et al. 2016) were not re-read here.

## A correction found while writing this

`docs/validation.md` cited Ohyama et al. 2015 as doi:10.1038/nature14424, which is a different paper (Carmi et al. 2015, *Nature* 521:99-104, on
allogeneic IgG and dendritic cells). Ohyama et al. is doi:10.1038/nature14297. Fixed in `docs/validation.md`; the citation strings in
`kickthefly/lab/validation.py` carry no DOI.
