# NSF Future Manufacturing Data Challenge — resume notes

Read `PROGRESS.md` next, then get to work. This file explains where things
actually stand as of the last local session (July 16, 2026) — it was
reconstructed from git history and file timestamps after the user lost
track of the project, so trust it over any assumptions.

## Where this repo came from

Cloned from https://github.com/abhishekhanchate/nsf-fmrg-data-challenge on
2026-07-16. Everything in `git ls-tree HEAD` (README, notebooks, `paper/`,
`src/nsf_fmrg_data.py`, `requirements.txt`, etc.) is organizer-provided
starter material — none of it is the user's own work. All of the real
work is untracked local files: `build_pairs_local.py`, `train_local.py`,
`diagnose_widths.py`, `diagnose_v3.py`, and everything under
`processed_data/`. **None of this has ever been committed to git.**
Consider committing it now (to a local branch at least) so it isn't lost
again.

## The problem being solved

Same challenge as usual: predict **local width variation** of a laser
track from thermal + SEM + height-map data, across 4 tracks/laser powers:

| track_id | laser power |
|---|---|
| 8  | 200 W |
| 10 | 300 W |
| 14 | 350 W |
| 21 | 400 W |

(Note: the dataset numbers tracks 8/10/14/21, not by power — don't assume
"power level" naming elsewhere corresponds directly; check
`TRACK_LASER_POWER_W` in `build_pairs_local.py` for the mapping.)

## What actually happened last session (reconstructed from timestamps)

1. **~12:38pm** — repo cloned.
2. **~2:08pm** — `diagnose_widths.py` written. Finding: many local-width
   ground-truth values extracted from the Wyko height maps were landing
   at exactly `0.0` — the extraction logic was unreliable.
3. **~2:12pm** — a model was trained anyway on the buggy (v2) data. Its
   outputs are saved at `processed_data/model_outputs/model.pt`,
   `training_curve.png`, `track21_prediction.png`. **These are broken.**
   The prediction plot has a y-axis in the millions of mm and the
   training curve flatlines after epoch 2 — the model's predicted
   log-variance collapsed to a degenerate huge value instead of learning
   anything. Do not use this model or treat its metrics as meaningful.
4. **~2:29–2:30pm** — `train_local.py` and `build_pairs_local.py` were
   rewritten as "v3" with real fixes, documented in their own docstrings:
   - `build_pairs_local.py` v3: SEM tile index now uses `floor()` instead
     of `round()` (the old version misassigned ~half the samples to the
     wrong tile); width extraction now does a robust per-profile linear
     detrend with a noise-adaptive threshold and takes the largest
     contiguous elevated run as the width, instead of whatever the v2
     logic did.
   - `train_local.py` v3: log-variance is now bounded *architecturally*
     in the model's `forward()` via `6*tanh(raw/6)`, not just clamped in
     the loss (the v2 approach), which is what allowed the collapse.
     Also added gradient clipping, best-validation-NLL checkpointing
     (not last-epoch), and a constant-baseline MAE comparison.
   - Track split: train on tracks [8, 10], validate on [14], held-out
     test on [21]. Explicitly **not** a random coordinate split, because
     spatially-adjacent points leak — see the comment in `train_local.py`.
5. **~2:30pm** — `build_pairs_local.py` v3 was run to completion.
   Confirmed: `processed_data/cache_v3/track_{8,10,14,21}_v3.npz` and
   `processed_data/qa_v3/{mask,width}_qa_track_*.png` all exist.
6. **~2:37pm** — `diagnose_v3.py` written (sanity-checks the detrend
   logic on a few sample x-positions of track 10). This is the last file
   touched. The session ended here.

**`train_local.py` v3 — the fixed training script — was never actually
run.** There is no `processed_data/model_outputs_v3/` directory. This is
the single most important fact: the next action is not "start over," it's
"run the script that's already sitting there, on data that's already
correctly rebuilt."

## Immediate next step

```bash
python -m pip install -r requirements.txt   # if the env is stale/fresh
python train_local.py
```

This reads `processed_data/cache_v3/*.npz` (already built, don't rebuild
unless you have a specific reason to) and writes to
`processed_data/model_outputs_v3/`: `model_best.pt`, `metrics.json`,
`track21_prediction.png`, `training_curve.png`. The script itself prints
a warning if the model fails to beat the constant-mean baseline — take
that warning seriously if it fires; it means there's still a real problem
before this is presentable.

Before fully trusting it, sanity check:
- Open `processed_data/qa_v3/mask_qa_track_*.png` — verify the red lines
  in the SEM tile QA actually bracket the track, not the substrate.
- Open `processed_data/qa_v3/width_qa_track_*.png` — verify width values
  look physically plausible (no giant spikes, no all-zero stretches)
  before trusting the training run built on top of them.

## After training works

Once `train_local.py` produces a sane `model_outputs_v3/` (beats the
baseline, coverage stats reasonably close to 0.68/0.95, plots look like
real predictions and not noise):

1. **Interpretability.** The rules explicitly reward separating
   process-driven (thermal) vs. substrate-driven (SEM) contributions to
   variation. Use SHAP or permutation importance on the two encoder
   branches (`thermal_enc`, `sem_enc` in `FusionModel`) to say something
   concrete about this — don't skip it, it's graded.
2. **Robustness across power.** Consider also checking performance
   per-power-level on the held-out track/segments, not just the single
   leave-one-out split, to speak to the "robustness across laser powers"
   judging criterion.
3. **Report** — PDF, max 3 pages, ≥10pt Arial, 1" margins. Sections:
   Executive Summary; Problem Formulation and Methodology (**must
   explicitly disclose Generative AI use** — be specific and honest,
   this session used Claude Code substantially); Modeling and Outcomes
   (predicted variation, width, boundary positions, descriptors,
   quantitative comparison to ground truth, uncertainty estimates);
   Conclusion (how thermal behavior relates to track variation).
4. **Executable notebook** — a clean end-to-end notebook, not the messy
   working scripts.
5. **Slide deck** — single self-contained PPT or PDF.
6. **Package** — zip report + notebook (or repo link) + slides for
   submission via the Qualtrics link.

Full submission/rules detail (scraped from the official site) is in the
project's Claude.ai project docs if you have access to that session, or
re-fetch from https://sites.google.com/tamu.edu/nsf-future-data-challenge/competition-rules
and .../final-report-materials.

**Deadline note:** the official site listed submission deadline July 27,
2026 and the finalist event July 31, 2026 for this cycle. Confirm current
status before assuming there's still time to submit.

## Working style

Update `PROGRESS.md` as you go. If you hit a real modeling decision (e.g.
the v3 model still doesn't beat baseline after training), stop and
surface it rather than silently trying more fixes indefinitely.
