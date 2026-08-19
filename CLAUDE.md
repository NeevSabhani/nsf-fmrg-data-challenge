# NSF Future Manufacturing Data Challenge — resume notes

Read `PROGRESS.md` next for the full evidence trail. This file is the
orientation: what the project is, what state it's in, and — importantly —
**which approaches are already ruled out, so you don't burn a session
re-deriving a wall that's already been hit.**

Last updated: 2026-08-18 (end of the v8/v9 session).

## Status in one paragraph

The pipeline had a root-cause bug in the **ground-truth labels** that
invalidated all modeling through v7. It's fixed (v8), and a reframed
model (v9) now genuinely works for one target. All work is committed on
the local branch `local-work`. Nothing is pending or half-finished.

## Where this repo came from

Cloned from https://github.com/abhishekhanchate/nsf-fmrg-data-challenge on
2026-07-16. Everything in the original commit (README, notebooks, `paper/`,
`src/nsf_fmrg_data.py`, `requirements.txt`) is organizer-provided starter
material — none of it is the user's own work. All real work is the
`*_local*.py` / `train_v*.py` / `diagnose_*.py` / `v9_*.py` scripts, now
committed on branch `local-work` (`main` is untouched, nothing pushed).
Generated data under `processed_data/` is gitignored and exists on disk
only — it can be rebuilt by rerunning the build scripts.

## The problem

Predict **local width variation** of a laser track from thermal + SEM +
height-map data, across 4 tracks / laser powers:

| track_id | laser power |
|---|---|
| 8  | 200 W |
| 10 | 300 W |
| 14 | 350 W |
| 21 | 400 W |

(`TRACK_LASER_POWER_W` in `build_pairs_local.py` is the authoritative map.)

## THE CRITICAL FINDING — read before touching anything

`extract_local_width_v3` thresholded the detrended height profile at a
**noise floor** (`max(2um, 3*sigma)`) instead of a track edge. The labels
therefore largely measured noise, not width. At track 10 / x=85mm the
threshold (7.5um) exceeded the track's own peak (7.0um) and reported
width `0.000mm` for a plainly-present track; ~25% of labels were spurious
exact zeros. Independently confirmed: SEM-measured band width correlated
**+0.045** with those labels and disagreed **3x in scale**.

**Consequence: every model through v7 was trained to predict noise.** That
is why v3/v4/v5/v6/v7 all plateaued at corr ~0.00 on held-out data
regardless of architecture, loss, ensembling, or augmentation.

**Do not use** `cache_v3` / `cache_v5` / `cache_v6`, or the
`model_outputs_v3..v7` results, for anything but historical comparison.

## Current good state

- **Data: `build_pairs_local_v8.py` → `processed_data/cache_v8/`.**
  Half-maximum width extraction (returns `NaN`, never a silent `0.0`,
  when no track is detectable) + per-column SEM band detection with row
  crops centred on the *local* band. Labels are now physically coherent
  and ordered by power: 0.569 / 0.470 / 0.390 / 0.195 mm at 200/300/350/400W.
- **Model: `v9_segment_model.py` → `processed_data/model_outputs_v9/`.**
  4mm segments (64 total), ridge regression on interpretable scalar
  features, leave-one-TRACK-out CV (each fold tests an unseen laser power).

### What v9 achieves (all leave-one-power-out)

- **Segment mean width: R2 +0.800, corr +0.902, MAE 0.0495** from
  **thermal features alone** (baseline 0.1469).
- **Interpretability:** thermal (R2 +0.800) decisively beats SEM
  (R2 -0.228) → width is **process-driven, not substrate-driven**. Also
  beats a power-only model (R2 +0.607), so thermal carries real
  information beyond identifying the power setpoint (this was checked
  explicitly — it was the obvious way the result could have been hollow).
- **Uncertainty: 68%/95% coverage = 0.625/0.922** vs 0.68/0.95 targets.
  Requires **out-of-fold** residuals for sigma; in-fold gives
  0.219/0.516 (wildly overconfident).

## ALREADY RULED OUT — do not retry without new data

These are established with evidence in `PROGRESS.md`, not guesses:

1. **Pointwise (per-0.2mm) width variation is not predictable.** Held-out
   corr ~0.00 across every version, *including after the label fix* (v8:
   +0.002). Two *independent* good measurements of it (Wyko half-max, SEM
   band thickness) correlate only +0.03..+0.20 **with each other** — the
   signal is below the noise floor of the available metrology.
2. **Within-track segment variation is not predictable.** v9 within-track
   corr **+0.079**. The R2 0.800 headline is a *between-power* effect.
3. **Segment width variability (std) is not predictable.** corr -0.192,
   fails to beat baseline under every feature set.
4. **More model capacity / ensembling / augmentation does not help.** v7
   (5-seed ensemble + calibration + augmentation) scored *worse* than a
   lucky single seed, and member-vs-truth correlations were ~0 with
   members disagreeing wildly with each other (-0.62..+0.96).
5. **Registration is not the problem.** All four SEM tile-order /
   orientation mappings were tested; the current one is best.
6. **Tighter SEM row crops hurt** (v6: MAE 0.118 vs v5's 0.063) — cropping
   around a *tile-average* band position cuts off the track where it drifts.

**If asked to "make it predict local variation," the honest answer is that
this dataset cannot support it.** What would change that: more tracks /
power levels, or lower-noise width metrology. Not another model.

## Remaining deliverables (nothing modeling-related is pending)

- [ ] Report — PDF, max 3 pages, >=10pt Arial, 1" margins. Sections:
      Executive Summary; Problem Formulation and Methodology (**must
      explicitly disclose Generative AI use** — Claude Code was used
      substantially); Modeling and Outcomes; Conclusion.
      **Recommended claim:** *"Thermal signature predicts track width
      across laser powers (R2 0.80 leave-one-power-out, calibrated
      uncertainty); width is process- not substrate-driven; and we show
      with independent metrology that pointwise width variation lies
      below the noise floor of the available measurements."* That is a
      real positive result plus a rigorously-evidenced negative one.
      Claiming pointwise local-variation prediction is **not** supportable.
- [ ] Clean executable notebook (end-to-end: `build_pairs_local_v8.py` →
      `v9_segment_model.py`), not the messy iteration scripts.
- [ ] Slide deck (single self-contained PPT or PDF).
- [ ] Package report + notebook + slides for submission.

## Deadline / context

The official deadline (July 27, 2026) and finalist event (July 31, 2026)
have **passed**. As of 2026-08-18 the user is continuing this purely as a
personal exercise to see how it would have played out — so there is no
submission clock, but the deliverables above are still the target shape.

## Working style

Update `PROGRESS.md` as you go. Surface real modeling findings rather than
silently trying more fixes — that norm is what turned this project around:
chasing *why* the SEM branch had zero permutation importance is what
uncovered the label bug.
